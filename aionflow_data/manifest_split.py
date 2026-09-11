"""Step 5: presence flags, the sample, and the split.

    python -m aionflow_data.manifest_split [--config CONFIG]

Per crossmatch row, the presence flags `has_spectrum` (a row in
spectra/source.h5), `has_image` (a cutout file), `has_z` (finite, z > 0,
ZWARN == 0) and `has_wise` (an LS10 band with flux > 0 and ivar > 0).

The sample is the crossmatch rows with a spectrum and an image, minus the
split-source pairs; every sample target is unique. A missing redshift or WISE
photometry is masked at training time, never a reason to drop the row.

The split is a seeded random permutation of the sample, sorted by targetid,
cut at the cumulative fractions (seed and fractions in the config), so it
depends only on the sample, the seed and the fractions.

Output: <work>/manifest.csv (every crossmatch row, with `in_sample` and a blank
`split` outside the sample), <work>/split.csv (`targetid,split` for the sample)
and the ledger data/provenance/manifest_split.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import FilterLedger, describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .crossmatch import OUTPUT as CROSSMATCH_OUTPUT
from .fetch_cutouts import CUTOUT_DIR, cutout_path
from .fetch_spectra import SOURCE as SPECTRA_SOURCE

STEP = "manifest_split"
MANIFEST = "manifest.csv"
SPLIT = "split.csv"
SPLITS = ("train", "val", "test")
PRESENCE_FLAGS = ("has_spectrum", "has_z", "has_wise", "has_image")
WISE_BANDS = ("w1", "w2", "w3")
MANIFEST_COLUMNS = [
    "targetid", "ero_detuid", "in_sample", "split", *PRESENCE_FLAGS, "spectype", "z", "zwarn",
    "target_ra", "target_dec", "ls10_flux_w1", "ls10_flux_w2", "ls10_flux_w3",
    "survey", "program", "healpix", "split_source",
]


class ManifestError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- split

def assign(targetids, seed: int, fractions) -> np.ndarray:
    """Split name per target: the targets sorted, permuted with `seed`, and the
    permutation cut at the cumulative fractions (rounded to whole rows)."""
    tids = np.asarray(targetids, np.int64)
    fractions = np.asarray(fractions, np.float64)
    if abs(fractions.sum() - 1.0) > 1e-9 or (fractions <= 0).any():
        raise ManifestError(f"split fractions must be positive and sum to 1, got {fractions}")
    if np.unique(tids).size != tids.size:
        raise ManifestError("split targets must be unique")
    n = tids.size
    order = np.argsort(tids)
    perm = np.random.RandomState(int(seed)).permutation(n)
    rank = np.empty(n, dtype=np.int64)
    rank[perm] = np.arange(n)                # position of each sorted target in the draw
    edges = np.round(np.cumsum(fractions)[:-1] * n).astype(int)
    split_sorted = np.asarray(SPLITS)[np.searchsorted(edges, rank, side="right")]
    out = np.empty(n, dtype=object)
    out[order] = split_sorted
    return out


# ----------------------------------------------------------------------------- presence

def presence(frame: pd.DataFrame, source_path: Path, cutout_dir: Path) -> pd.DataFrame:
    tids = frame["targetid"].to_numpy(np.int64)
    with h5py.File(source_path, "r") as h:
        with_spectrum = h["targetid"][:].astype(np.int64)
    out = pd.DataFrame({"targetid": tids})
    out["has_spectrum"] = np.isin(tids, with_spectrum)
    out["has_image"] = np.array([cutout_path(cutout_dir, t).is_file() for t in tids], bool)
    z = frame["z"].to_numpy(np.float64)
    out["has_z"] = np.isfinite(z) & (z > 0) & (frame["zwarn"].to_numpy(np.float64) == 0)
    has_wise = np.zeros(len(out), bool)
    for band in WISE_BANDS:
        flux = frame[f"ls10_flux_{band}"].to_numpy(np.float64)
        ivar = frame[f"ls10_flux_ivar_{band}"].to_numpy(np.float64)
        has_wise |= np.isfinite(flux) & (flux > 0) & np.isfinite(ivar) & (ivar > 0)
    out["has_wise"] = has_wise
    return out


# ----------------------------------------------------------------------------- run

def run(cfg: dict, log=print) -> pd.DataFrame:
    ensure_dirs(cfg)
    work = Path(cfg["paths"]["work"])
    xm_path, source_path = work / CROSSMATCH_OUTPUT, work / SPECTRA_SOURCE
    cutout_dir = work / CUTOUT_DIR
    for p in (xm_path, source_path):
        if not p.is_file():
            raise ManifestError(f"missing input {p}")
    if not cutout_dir.is_dir():
        raise ManifestError(f"missing cutout directory {cutout_dir}; run fetch_cutouts first")
    s = cfg["split"]

    frame = pd.read_parquet(xm_path)
    flags = presence(frame, source_path, cutout_dir)
    manifest = pd.concat([frame, flags.drop(columns=["targetid"])], axis=1)
    for flag in PRESENCE_FLAGS:
        log(f"[presence] {flag:13s} {int(manifest[flag].sum()):,} of {len(manifest):,}")

    led = FilterLedger(len(manifest))
    in_sample = np.ones(len(manifest), bool)
    for name, keep in (("split_source_pairs_excluded", ~manifest["split_source"].to_numpy()),
                       ("has_spectrum", manifest["has_spectrum"].to_numpy()),
                       ("has_image", manifest["has_image"].to_numpy())):
        kept = led.apply(name, keep[in_sample])
        in_sample[in_sample] = kept
    manifest["in_sample"] = in_sample
    sample = manifest[in_sample]
    if sample["targetid"].duplicated().any():
        raise ManifestError("a target appears twice in the sample; the crossmatch should "
                            "have resolved every shared target")
    if len(sample) == 0:
        raise ManifestError("every row was filtered out")

    split = assign(sample["targetid"].to_numpy(), int(s["seed"]), s["fractions"])
    manifest["split"] = pd.Series(split, index=sample.index).reindex(manifest.index)
    n = len(sample)
    row_counts = {name: int((split == name).sum()) for name in SPLITS}
    for name in SPLITS:
        log(f"[split] {name:5s} {row_counts[name]:7,} rows ({row_counts[name] / n:.4f})")

    manifest = manifest[MANIFEST_COLUMNS].sort_values("ero_detuid").reset_index(drop=True)
    manifest_path, split_path = work / MANIFEST, work / SPLIT
    manifest.to_csv(manifest_path, index=False)
    split_frame = (manifest.loc[manifest["in_sample"], ["targetid", "split"]]
                   .sort_values("targetid").reset_index(drop=True))
    split_frame.to_csv(split_path, index=False)

    sample = manifest[manifest["in_sample"]]
    census = {name: sample.loc[sample["split"] == name, "spectype"].value_counts().to_dict()
              for name in SPLITS}
    coverage = {flag: int(sample[flag].sum()) for flag in PRESENCE_FLAGS}
    log(f"[out] sample {n:,} rows; census {sample['spectype'].value_counts().to_dict()}")
    write_ledger(STEP, cfg,
                 inputs={"crossmatch": xm_path, "spectra_source": source_path},
                 counts={"crossmatch_rows": int(len(manifest)), "sample_rows": n,
                         "split_source_rows_excluded": int(manifest["split_source"].sum()),
                         **{f"rows_{k}": v for k, v in row_counts.items()}},
                 filters=led.rows,
                 extra={"seed": int(s["seed"]), "fractions": list(s["fractions"]),
                        "presence_in_sample": coverage,
                        "presence_fraction_in_sample": {k: v / n for k, v in coverage.items()},
                        "census_by_split": census,
                        "sample_census": sample["spectype"].value_counts().to_dict(),
                        "cutout_dir": str(cutout_dir),
                        "outputs": {"manifest": describe_file(manifest_path),
                                    "split": describe_file(split_path)}})
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg)
    except ManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
