"""Step 8: the emission-line baseline's four line fluxes, fitted on our own spectra.

    python -m aionflow_data.line_features [--config CONFIG] [--nproc N] [--limit N]

For every sample row with a positive redshift, each of [O III] 5007, [Ne V] 3426,
H-alpha and H-beta whose rest-frame window falls inside the spectrograph
coverage at that redshift is fitted with `linefit.fit_one`. The integrated flux
is reported in the observed frame (rest-frame flux times 1 + z), the convention
of the FastSpecFit catalogue the baseline was first built on. A line outside
the coverage, or a failed fit, is written as 0 with its status recorded: a line
baseline genuinely has nothing to say about such a source, and writing NaN
would silently drop it from the comparison. Every sample row gets a row so the
baseline is scored on the same objects as the model.

Outputs: <work>/line_features.csv (one row per sample target) and
<work>/line_fits.csv (one row per fitted line, every parameter); the ledger
data/provenance/line_features.json records coverage, status counts and the
Balmer decrement by class as the extraction check.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .fetch_spectra import SOURCE as SPECTRA_SOURCE
from .linefit import COMPLEXES, fit_one, in_window
from .manifest_split import MANIFEST

STEP = "line_features"
FEATURES = "line_features.csv"
FITS = "line_fits.csv"
# fitter complex -> feature column stem, in the baseline's order
LINES = [("oiii", "oiii_5007"), ("nev", "nev_3426"), ("halpha", "halpha"), ("hbeta", "hbeta")]
BLOCK = 512
AN_MIN_FOR_BALMER = 5.0
CASE_B_DECREMENT = 2.86


class LineFeaturesError(RuntimeError):
    pass


def _fit_task(args):
    name, lam_rest, flux, ivar = args
    return fit_one(name, lam_rest, flux, ivar)


def fit_sample(sample: pd.DataFrame, source_path: Path, grid_lo: float, grid_hi: float, *,
               nproc: int, log=print) -> pd.DataFrame:
    """One row per (target, line) for every line inside the coverage at the row's z."""
    z = sample["z"].to_numpy(np.float64)
    masks = {name: in_window(name, z, grid_lo, grid_hi) for name, _ in LINES}
    for name, _ in LINES:
        log(f"[lines] {name:7s} in window for {int(masks[name].sum()):,} of {len(sample):,}")
    any_line = np.zeros(len(sample), bool)
    for m in masks.values():
        any_line |= m
    work = sample[any_line].sort_values("source_row").reset_index(drop=True)
    masks = {k: v[any_line][np.argsort(sample[any_line]["source_row"].to_numpy(),
                                       kind="stable")] for k, v in masks.items()}
    rows = work["source_row"].to_numpy(np.int64)
    log(f"[lines] fitting {len(rows):,} spectra with at least one line in window")
    # forkserver, not fork: numpy's BLAS threads are already running in this process
    pool = get_context("forkserver").Pool(nproc) if nproc > 1 else None
    results: list[dict] = []
    t0 = time.time()
    try:
        with h5py.File(source_path, "r") as h:
            lam_obs = h["spectra_lambda"][:].astype(np.float64)
            for start in range(0, len(rows), BLOCK):
                idx = rows[start:start + BLOCK]
                flux = h["spectra"][idx, :]
                ivar = h["spectra_ivar"][idx, :]
                tasks, meta = [], []
                for k in range(len(idx)):
                    row = work.iloc[start + k]
                    zz = float(row["z"])
                    lam_rest = lam_obs / (1.0 + zz)
                    for name, _ in LINES:
                        if not masks[name][start + k]:
                            continue
                        lo_w, hi_w = COMPLEXES[name]["window"]
                        a = max(int(np.searchsorted(lam_rest, lo_w)) - 2, 0)
                        b = min(int(np.searchsorted(lam_rest, hi_w)) + 2, lam_rest.size)
                        tasks.append((name, lam_rest[a:b].copy(), flux[k, a:b].astype(float),
                                      ivar[k, a:b].astype(float)))
                        meta.append({"targetid": int(row["targetid"]), "z": zz})
                fits = pool.map(_fit_task, tasks, chunksize=8) if pool else \
                    [_fit_task(t) for t in tasks]
                for fit, m in zip(fits, meta):
                    fit.update(m)
                    results.append(fit)
                log(f"[lines]   {min(start + BLOCK, len(rows)):,}/{len(rows):,} spectra, "
                    f"{time.time() - t0:.0f}s")
    finally:
        if pool:
            pool.close()
            pool.join()
    fits = pd.DataFrame(results)
    if len(fits):
        fits["flux"] = fits["flux_rest"] * (1.0 + fits["z"])
        fits["flux_err"] = fits["flux_rest_err"] * (1.0 + fits["z"])
    return fits


def features_from_fits(sample: pd.DataFrame, fits: pd.DataFrame, grid_lo: float,
                       grid_hi: float) -> pd.DataFrame:
    """One row per sample target, zeros where a line is missing, statuses kept."""
    base = sample[["targetid", "split", "spectype", "z"]].copy().reset_index(drop=True)
    z = base["z"].to_numpy(np.float64)
    for name, stem in LINES:
        base[f"{stem}_flux"] = 0.0
        base[f"{stem}_flux_err"] = np.nan
        base[f"{stem}_an"] = np.nan
        status = np.where(np.isfinite(z) & (z > 0), "not_in_window", "no_redshift")
        base[f"{stem}_status"] = np.where(in_window(name, z, grid_lo, grid_hi), "unfitted",
                                          status)
        if not len(fits):
            continue
        sub = fits[fits["line"] == name].drop_duplicates("targetid").set_index("targetid")
        aligned = sub.reindex(base["targetid"])
        fitted = aligned["status"].notna().to_numpy()
        ok = (aligned["status"] == "ok").to_numpy()
        base.loc[fitted, f"{stem}_status"] = aligned.loc[fitted, "status"].to_numpy()
        base.loc[ok, f"{stem}_flux"] = aligned.loc[ok, "flux"].to_numpy()
        base.loc[ok, f"{stem}_flux_err"] = aligned.loc[ok, "flux_err"].to_numpy()
        base.loc[ok, f"{stem}_an"] = aligned.loc[ok, "an"].to_numpy()
    return base


def balmer_decrement(features: pd.DataFrame) -> dict:
    """Median H-alpha / H-beta by class where both lines are confidently measured."""
    m = ((features["halpha_status"] == "ok") & (features["hbeta_status"] == "ok")
         & (features["halpha_an"] > AN_MIN_FOR_BALMER)
         & (features["hbeta_an"] > AN_MIN_FOR_BALMER) & (features["hbeta_flux"] > 0))
    out = {"case_b": CASE_B_DECREMENT, "an_min": AN_MIN_FOR_BALMER, "by_class": {}}
    for cls, grp in features[m].groupby("spectype"):
        ratio = (grp["halpha_flux"] / grp["hbeta_flux"]).to_numpy()
        if ratio.size:
            q = np.percentile(ratio, [16, 50, 84])
            out["by_class"][str(cls)] = {"n": int(ratio.size), "p16": float(q[0]),
                                         "median": float(q[1]), "p84": float(q[2])}
    return out


def run(cfg: dict, *, nproc: int | None = None, limit: int | None = None,
        log=print) -> pd.DataFrame:
    ensure_dirs(cfg)
    work = Path(cfg["paths"]["work"])
    manifest_path, source_path = work / MANIFEST, work / SPECTRA_SOURCE
    for p in (manifest_path, source_path):
        if not p.is_file():
            raise LineFeaturesError(f"missing input {p}")
    s = cfg["spectra"]
    grid_lo = float(s["lam0_angstrom"])
    grid_hi = grid_lo + float(s["dlam_angstrom"]) * (int(s["nbin"]) - 1)
    manifest = pd.read_csv(manifest_path)
    sample = manifest[manifest["in_sample"].astype(bool)].reset_index(drop=True)
    if limit:
        sample = sample.head(limit)
    log(f"[lines] {len(sample):,} sample rows; coverage {grid_lo:.1f}-{grid_hi:.2f} A")
    nproc = nproc or max(1, (os.cpu_count() or 2) - 1)
    fits = fit_sample(sample, source_path, grid_lo, grid_hi, nproc=nproc, log=log)
    features = features_from_fits(sample, fits, grid_lo, grid_hi)
    balmer = balmer_decrement(features)
    for cls, rec in balmer["by_class"].items():
        log(f"[lines] Balmer decrement {cls:7s} n={rec['n']:,} Ha/Hb = {rec['median']:.2f} "
            f"[{rec['p16']:.2f}, {rec['p84']:.2f}] (case B {CASE_B_DECREMENT})")

    features_path, fits_path = work / FEATURES, work / FITS
    features.to_csv(features_path, index=False)
    fits.to_csv(fits_path, index=False)
    counts: dict = {"sample_rows": int(len(sample)),
                    "spectra_fitted": int(fits["targetid"].nunique()) if len(fits) else 0,
                    "fits": int(len(fits))}
    status_counts = {}
    for name, stem in LINES:
        col = features[f"{stem}_status"]
        status_counts[stem] = col.value_counts().to_dict()
        counts[f"{stem}_measured"] = int((col == "ok").sum())
        counts[f"{stem}_in_window"] = int((col != "not_in_window").sum()
                                          - (col == "no_redshift").sum())
    log(f"[out] {features_path}: {len(features):,} rows; measured "
        + ", ".join(f"{stem} {counts[f'{stem}_measured']:,}" for _, stem in LINES))
    write_ledger(STEP, cfg, inputs={"manifest": manifest_path, "spectra_source": source_path},
                 counts=counts,
                 extra={"lines": {name: {"stem": stem, "window": list(COMPLEXES[name]["window"]),
                                         "primary": list(COMPLEXES[name]["primary"])}
                                  for name, stem in LINES},
                        "coverage_angstrom": [grid_lo, grid_hi], "flux_frame": "observed",
                        "status_counts": status_counts, "balmer_decrement": balmer,
                        "nproc": nproc, "limit": limit,
                        "outputs": {"features": describe_file(features_path),
                                    "fits": describe_file(fits_path)}})
    return features


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--nproc", type=int, default=None, help="worker processes")
    parser.add_argument("--limit", type=int, default=None, help="first N sample rows")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg, nproc=args.nproc, limit=args.limit)
    except LineFeaturesError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
