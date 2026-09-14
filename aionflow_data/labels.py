"""Step 2: X-ray labels from the eRASS:3 Main catalogue, host labels from the CIGALE VAC.

    python -m aionflow_data.labels [--config CONFIG]

Joins every crossmatch row to the Main catalogue on DETUID (an exact lookup; a
miss is an error) and to the CIGALE VAC on TARGETID (a left join; a miss is a
missing label). Per X-ray band (1 = 0.2-2.3, P2 = 0.5-1.0, P3 = 1.0-2.0 keV):
log10 flux with split-normal errors in dex, the detection likelihood, and the
aperture triple (N counts, B background, t exposure) behind the Poisson heads.
The broad-band luminosity uses Planck18 at z >= `labels.z_floor`; below that the
redshift is not cosmological (a Galactic star has a good redshift of order 1e-5
and ZWARN == 0) and the luminosity would be meaningless.

Counts integrity: APE_CTS is int16 in the published catalogue and wraps on the
brightest sources; a negative value is not invertible, so the whole triple for
that band is written as missing and counted. A negative APE_BKG is a
background-map residual within noise of zero; it is clipped to zero, flagged in
`ape_bkg_negative_<band>`, and counted.

CIGALE: one fit per target, preferring the fit of the same survey and program
as the DESI observation, then a main-survey fit, then the lowest chi2. A label
is missing where the fit failed (both values exactly zero), carries a -99
sentinel, has a best/Bayesian PDF flag outside the configured window for that
quantity, has a non-positive or non-finite error, or has an error above the
configured maximum.

Output: <work>/labels.csv, one row per crossmatch row, and the ledger
data/provenance/labels.json with every gate's cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    describe_file,
    ensure_dirs,
    load_config,
    read_fits_columns,
    step_parser,
    write_ledger,
)
from .crossmatch import OUTPUT as CROSSMATCH_OUTPUT

STEP = "labels"
OUTPUT = "labels.csv"

BANDS = [("1", "1"), ("P2", "p2"), ("P3", "p3")]          # (catalogue suffix, ours)
CIGALE_COLUMNS = ["SURVEY", "PROGRAM", "CHI2", "LOGM", "LOGM_ERR", "LOGSFR", "LOGSFR_ERR",
                  "FLAG_MASSPDF", "FLAG_SFRPDF"]
SENTINEL = -90.0
LABEL_COLUMNS = (
    ["log_flux_1", "log_flux_1_sig_lo", "log_flux_1_sig_hi", "log_lx", "det_like_0"]
    + [f"log_flux_{b}{s}" for _, b in BANDS[1:] for s in ("", "_sig_lo", "_sig_hi")]
    + [f"det_like_{b}" for _, b in BANDS[1:]]
    + [f"ape_{q}_{b}" for _, b in BANDS for q in ("cts", "bkg", "exp")]
    + ["log_sfr", "log_sfr_sig_lo", "log_sfr_sig_hi",
       "logmstar_cigale", "logmstar_cigale_sig_lo", "logmstar_cigale_sig_hi"]
)


class LabelsError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- formulas

def log_with_asym_errors(flux, lowerr, uperr, cap_dex: float):
    """log10 flux with split-normal sigmas in dex; NaN where the flux is not a measurement.

    sig_lo = -log10(1 - LOWERR / F), sig_hi = log10(1 + UPERR / F). A lower error that
    swallows the flux, or either error beyond `cap_dex`, makes the value NaN rather
    than an infinite or meaningless bar.
    """
    f = np.asarray(flux, np.float64)
    lo = np.asarray(lowerr, np.float64)
    hi = np.asarray(uperr, np.float64)
    good = np.isfinite(f) & (f > 0)
    out = np.full(f.shape, np.nan)
    slo = np.full(f.shape, np.nan)
    shi = np.full(f.shape, np.nan)
    out[good] = np.log10(f[good])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(good & np.isfinite(lo), lo / f, np.nan)
        slo_all = np.where(ratio < 1.0, -np.log10(np.clip(1.0 - ratio, 1e-12, None)), np.inf)
        shi_all = np.where(good & np.isfinite(hi), np.log10(1.0 + hi / f), np.nan)
    slo[good] = slo_all[good]
    shi[good] = shi_all[good]
    bad = ~np.isfinite(slo) | ~np.isfinite(shi) | (slo > cap_dex) | (shi > cap_dex)
    out[bad] = np.nan
    slo[bad] = np.nan
    shi[bad] = np.nan
    return out, slo, shi


def log_luminosity(log_flux, z, cosmology: str = "Planck18",
                   z_floor: float = 0.001) -> tuple[np.ndarray, int]:
    """log10 L from log10 flux (erg s-1 cm-2) and redshift; NaN below `z_floor`.

    The floor is not a guard against division by zero, it is the statement that the
    redshift is cosmological. A Galactic star sits at z of order 1e-5 to 1e-4 from
    its peculiar velocity alone, with a perfectly good redshift and ZWARN == 0, so
    no redshift-quality flag rejects it; pushed through a luminosity distance it
    becomes an object many sigma below the sample and drags the training
    standardizer with it. z_floor of 0.001 is about 4 Mpc, which separates the
    Galaxy from anything this survey calls a source.
    """
    import astropy.units as u
    from astropy import cosmology as cosmo

    cosmology_obj = getattr(cosmo, cosmology)
    z = np.asarray(z, np.float64)
    log_flux = np.asarray(log_flux, np.float64)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z >= z_floor) & np.isfinite(log_flux)
    if ok.any():
        dl = cosmology_obj.luminosity_distance(z[ok]).to(u.cm).value
        out[ok] = log_flux[ok] + np.log10(4.0 * np.pi * dl ** 2)
    return out, int((np.isfinite(z) & (z < z_floor)).sum())


# ----------------------------------------------------------------------------- X-ray

def main_row_index(main_path: Path, detuids: np.ndarray) -> np.ndarray:
    """Row of each DETUID in the Main catalogue; every DETUID must be present."""
    catalogue = np.char.strip(read_fits_columns(main_path, ["DETUID"])["DETUID"].astype("S32"))
    if np.unique(catalogue).size != catalogue.size:
        raise LabelsError(f"{main_path.name}: DETUID is not unique; the join would be ambiguous")
    order = np.argsort(catalogue, kind="stable")
    want = np.char.strip(np.asarray(detuids).astype("S32"))
    slot = np.clip(np.searchsorted(catalogue, want, sorter=order), 0, max(order.size - 1, 0))
    row = order[slot]
    hit = catalogue[row] == want
    if not hit.all():
        examples = [w.decode() for w in want[~hit][:5]]
        raise LabelsError(f"{int((~hit).sum()):,} of {want.size:,} DETUIDs are absent from "
                          f"{main_path.name} (first: {examples}); wrong catalogue?")
    return row


def xray_labels(main_path: Path, frame: pd.DataFrame, cfg_labels: dict,
                log=print) -> tuple[pd.DataFrame, dict]:
    rows = main_row_index(main_path, frame["ero_detuid"].to_numpy())
    columns = ["DET_LIKE_0"] + [f"DET_LIKE_{b}" for b, _ in BANDS[1:]]
    for fits_b, _ in BANDS:
        columns += [f"ML_FLUX_{fits_b}", f"ML_FLUX_LOWERR_{fits_b}", f"ML_FLUX_UPERR_{fits_b}",
                    f"APE_CTS_{fits_b}", f"APE_BKG_{fits_b}", f"APE_EXP_{fits_b}"]
    data = read_fits_columns(main_path, columns, rows=rows)
    cap = float(cfg_labels["sig_cap_dex"])
    det_min = float(cfg_labels["det_like_min"])

    out: dict[str, object] = {"det_like_0": data["DET_LIKE_0"].astype(np.float64)}
    stats: dict[str, dict] = {}
    for fits_b, ours in BANDS:
        lf, lo, hi = log_with_asym_errors(data[f"ML_FLUX_{fits_b}"],
                                          data[f"ML_FLUX_LOWERR_{fits_b}"],
                                          data[f"ML_FLUX_UPERR_{fits_b}"], cap)
        out[f"log_flux_{ours}"] = lf
        out[f"log_flux_{ours}_sig_lo"] = lo
        out[f"log_flux_{ours}_sig_hi"] = hi
        if ours != "1":
            out[f"det_like_{ours}"] = data[f"DET_LIKE_{fits_b}"].astype(np.float64)
        cts = data[f"APE_CTS_{fits_b}"].astype(np.int32)
        bkg = data[f"APE_BKG_{fits_b}"].astype(np.float64)
        exp = data[f"APE_EXP_{fits_b}"].astype(np.float64)
        wrapped = cts < 0
        negative_bkg = np.isfinite(bkg) & (bkg < 0) & ~wrapped
        bkg = np.where(negative_bkg, 0.0, bkg)
        cts_out = pd.array(cts.astype(np.int64), dtype="Int64")
        cts_out[wrapped] = pd.NA
        bkg[wrapped] = np.nan
        exp[wrapped] = np.nan
        out[f"ape_cts_{ours}"] = cts_out
        out[f"ape_bkg_{ours}"] = bkg
        out[f"ape_exp_{ours}"] = exp
        out[f"ape_bkg_negative_{ours}"] = negative_bkg
        det = out["det_like_0"] if ours == "1" else out[f"det_like_{ours}"]
        stats[ours] = {
            "flux_measured": int(np.isfinite(lf).sum()),
            "flux_not_a_measurement": int((~np.isfinite(lf)).sum()),
            f"det_like_gt_{det_min:g}": int((det > det_min).sum()),
            "ape_cts_wrapped": int(wrapped.sum()),
            "ape_bkg_negative_clipped": int(negative_bkg.sum()),
            "ape_cts_zero": int(((cts == 0) & ~wrapped).sum()),
            "ape_triple_complete": int((~wrapped & np.isfinite(exp) & np.isfinite(bkg)).sum()),
        }
        log(f"[xray] band {ours:>2}: {stats[ours]['flux_measured']:,} fluxes measured, "
            f"{stats[ours]['ape_cts_wrapped']} wrapped counts, "
            f"{stats[ours]['ape_bkg_negative_clipped']} negative backgrounds clipped")
    z_floor = float(cfg_labels["z_floor"])
    out["log_lx"], n_bad_z = log_luminosity(out["log_flux_1"], frame["z"].to_numpy(),
                                            cfg_labels.get("cosmology", "Planck18"), z_floor)
    stats["log_lx"] = {"finite": int(np.isfinite(out["log_lx"]).sum()),
                       f"z_below_{z_floor:g}": n_bad_z}
    log(f"[xray] log_lx defined for {stats['log_lx']['finite']:,} rows "
        f"({n_bad_z} below z = {z_floor:g}, not cosmological)")
    return pd.DataFrame(out), stats


# ----------------------------------------------------------------------------- CIGALE

def cigale_labels(cigale_path: Path, frame: pd.DataFrame, cfg_labels: dict,
                  log=print) -> tuple[pd.DataFrame, dict]:
    targetids = frame["targetid"].to_numpy(np.int64)
    all_tid = read_fits_columns(cigale_path, ["TARGETID"])["TARGETID"].astype(np.int64)
    mask = np.isin(all_tid, targetids)
    data = read_fits_columns(cigale_path, CIGALE_COLUMNS, rows=mask)
    cat = pd.DataFrame({k: (np.char.strip(v.astype(str)) if v.dtype.kind in "SU" else v)
                        for k, v in data.items()})
    cat.insert(0, "TARGETID", all_tid[mask])
    del all_tid, data
    stats = {"vac_rows_for_our_targets": int(len(cat)),
             "targets_in_vac": int(cat["TARGETID"].nunique())}

    # one fit per target: same observation, then main survey, then best chi2
    obs = frame[["targetid", "survey", "program"]].drop_duplicates("targetid")
    cat = cat.merge(obs, left_on="TARGETID", right_on="targetid", how="left")
    same_obs = (cat["SURVEY"] == cat["survey"]) & (cat["PROGRAM"] == cat["program"])
    cat["exact"] = same_obs.astype(int)
    cat["is_main"] = (cat["SURVEY"] == "main").astype(int)
    cat = (cat.sort_values(["TARGETID", "exact", "is_main", "CHI2"],
                           ascending=[True, False, False, True])
              .drop_duplicates("TARGETID", keep="first").reset_index(drop=True))
    stats["duplicate_fits_resolved"] = stats["vac_rows_for_our_targets"] - len(cat)

    lo_flag, hi_flag = (float(v) for v in cfg_labels["cigale_flag_window"])
    max_sigma = float(cfg_labels["cigale_max_sigma_dex"])
    logm = cat["LOGM"].to_numpy(np.float64)
    sfr = cat["LOGSFR"].to_numpy(np.float64)
    e_m = np.abs(cat["LOGM_ERR"].to_numpy(np.float64))
    e_s = np.abs(cat["LOGSFR_ERR"].to_numpy(np.float64))
    fm = cat["FLAG_MASSPDF"].to_numpy(np.float64)
    fs = cat["FLAG_SFRPDF"].to_numpy(np.float64)
    failed = (logm == 0) & (sfr == 0)
    sentinel = np.zeros(len(cat), bool)
    for col in ("LOGM", "LOGSFR", "LOGM_ERR", "LOGSFR_ERR", "FLAG_MASSPDF", "FLAG_SFRPDF"):
        sentinel |= cat[col].to_numpy(np.float64) <= SENTINEL
    broad_m = ~((fm > lo_flag) & (fm < hi_flag))
    broad_s = ~((fs > lo_flag) & (fs < hi_flag))
    stats.update({"failed_fit": int(failed.sum()), "sentinel": int(sentinel.sum()),
                  "broad_mass_pdf": int(broad_m.sum()), "broad_sfr_pdf": int(broad_s.sum())})

    out = pd.DataFrame({"targetid": cat["TARGETID"].to_numpy(np.int64)})

    def add(name: str, value, err, bad) -> None:
        unusable = bad | ~np.isfinite(value) | ~np.isfinite(err) | (err <= 0)
        wide = np.isfinite(err) & (err > max_sigma) & ~unusable
        gated = unusable | wide
        out[name] = np.where(gated, np.nan, value)
        out[f"{name}_sig_lo"] = np.where(gated, np.nan, err)
        out[f"{name}_sig_hi"] = np.where(gated, np.nan, err)
        stats[name] = {"usable": int((~gated).sum()), "unusable": int(unusable.sum()),
                       "removed_by_max_sigma": int(wide.sum())}

    add("logmstar_cigale", logm, e_m, failed | sentinel | broad_m)
    add("log_sfr", sfr, e_s, failed | sentinel | broad_s)

    aligned = out.set_index("targetid").reindex(targetids).reset_index(drop=True)
    for name in ("logmstar_cigale", "log_sfr"):
        log(f"[cigale] {name:16s} usable for {int(np.isfinite(aligned[name]).sum()):,} "
            f"of {len(aligned):,} rows "
            f"(max-sigma alone removed {stats[name]['removed_by_max_sigma']})")
    return aligned, stats


# ----------------------------------------------------------------------------- run

def run(cfg: dict, log=print) -> pd.DataFrame:
    ensure_dirs(cfg)
    raw, work = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["work"])
    xm_path = work / CROSSMATCH_OUTPUT
    main_path = raw / cfg["inputs"]["main"]["file"]
    cigale_path = raw / cfg["inputs"]["cigale"]["file"]
    for p in (xm_path, main_path, cigale_path):
        if not p.is_file():
            raise LabelsError(f"missing input {p}")
    frame = pd.read_parquet(xm_path)
    log(f"[labels] {len(frame):,} crossmatch rows")

    xray, xray_stats = xray_labels(main_path, frame, cfg["labels"], log=log)
    cigale, cigale_stats = cigale_labels(cigale_path, frame, cfg["labels"], log=log)
    labels = pd.concat([frame.reset_index(drop=True), xray, cigale], axis=1)

    out = work / OUTPUT
    labels.to_csv(out, index=False)
    counts = {"rows_out": int(len(labels))}
    for name in ("log_flux_1", "log_lx", "log_flux_p2", "log_flux_p3", "log_sfr",
                 "logmstar_cigale"):
        counts[f"{name}_finite"] = int(np.isfinite(labels[name]).sum())
    for _, ours in BANDS:
        counts[f"ape_triple_{ours}_complete"] = xray_stats[ours]["ape_triple_complete"]
    write_ledger(STEP, cfg,
                 inputs={"crossmatch": xm_path, "main": main_path, "cigale": cigale_path},
                 counts=counts,
                 extra={"xray": xray_stats, "cigale": cigale_stats,
                        "gates": {k: cfg["labels"][k] for k in
                                  ("sig_cap_dex", "det_like_min", "z_floor", "cosmology",
                                   "cigale_flag_window", "cigale_max_sigma_dex")},
                        "output": describe_file(out)})
    log(f"[out] {out}: {len(labels):,} rows, {len(labels.columns)} columns")
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg)
    except LabelsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
