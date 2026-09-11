"""Step 5: presence flags, the sample, and the split.

    python -m aionflow_data.manifest_split [--config CONFIG]

Per crossmatch row: `source_row` into spectra/source.h5 and the presence flags
`has_spectrum` (a row exists), `has_image` (a cutout exists and reads back as the
expected image), `has_z` (finite, z > 0, ZWARN == 0), `has_w1..w3` (LS10 flux > 0
and ivar > 0) and `has_wise` (any band). A flag is a statement about
availability; a missing input is masked, never a reason to drop the row.

The sample is the crossmatch rows that have a spectrum, minus the split-source
pairs. The split groups the sample on connected components of the
detection-target graph, so a detection carrying several targets or a target
under several detections lands on one side, and assigns each component by a
keyed blake2b hash of its key (the smallest DETUID it contains) mapped to
[0, 1) and cut at the cumulative fractions. The assignment is a pure function
of (key, salt, fractions): reproducible from the ledger alone and stable under
row reordering. There is no seed.

Output: <work>/manifest.csv (every crossmatch row, with `in_sample` and a blank
`split` outside the sample), <work>/split.csv (`targetid,split` for the sample)
and the ledger data/provenance/manifest_split.json.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import FilterLedger, describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .crossmatch import OUTPUT as CROSSMATCH_OUTPUT
from .fetch_cutouts import CUTOUT_DIR, BadCutout, cutout_path, read_cutout
from .fetch_spectra import SOURCE as SPECTRA_SOURCE

STEP = "manifest_split"
MANIFEST = "manifest.csv"
SPLIT = "split.csv"
SPLITS = ("train", "val", "test")
PRESENCE_FLAGS = ("has_spectrum", "has_z", "has_wise", "has_image")
WISE_BANDS = ("w1", "w2", "w3")
MANIFEST_COLUMNS = [
    "targetid", "ero_detuid", "source_row", "in_sample", "split", "component",
    "has_spectrum", "has_z", "has_w1", "has_w2", "has_w3", "has_wise", "has_image",
    "spectype", "z", "zwarn", "target_ra", "target_dec", "flux_w1", "flux_w2", "flux_w3",
    "survey", "program", "healpix", "split_source",
]


class ManifestError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- split

def hash_unit(keys, salt: str) -> np.ndarray:
    """Map each key to a deterministic u in [0, 1) by keyed blake2b (8-byte digest)."""
    key = salt.encode()
    if len(key) > 64:
        raise ManifestError("the hash salt must encode to at most 64 bytes")
    digests = [hashlib.blake2b(str(k).encode(), key=key, digest_size=8).digest() for k in keys]
    return np.array([int.from_bytes(d, "big") for d in digests], dtype=np.uint64) / 2.0 ** 64


def assign(keys, salt: str, fractions) -> np.ndarray:
    """Split name per key: cumulative fractions cut the unit interval."""
    fractions = np.asarray(fractions, np.float64)
    if abs(fractions.sum() - 1.0) > 1e-9 or (fractions <= 0).any():
        raise ManifestError(f"split fractions must be positive and sum to 1, got {fractions}")
    edges = np.cumsum(fractions)[:-1]
    return np.asarray(SPLITS)[np.searchsorted(edges, hash_unit(keys, salt), side="right")]


def components(detuids, targetids) -> np.ndarray:
    """Component key per row: the smallest DETUID in the row's connected component.

    Rows are edges of a bipartite graph between detections and targets.
    """
    det = np.asarray(detuids).astype(str)
    tid = np.asarray(targetids).astype(np.int64)
    det_codes, det_names = pd.factorize(det, sort=True)
    tid_codes, _ = pd.factorize(tid, sort=True)
    n_det, n_tid = det_names.size, int(tid_codes.max()) + 1 if tid.size else 0
    n = n_det + n_tid
    graph = coo_matrix((np.ones(det.size), (det_codes, n_det + tid_codes)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    row_label = labels[det_codes]
    # the key of a component is the alphabetically smallest DETUID among its rows
    key_of = pd.Series(det).groupby(row_label).min()
    return key_of.reindex(row_label).to_numpy()


# ----------------------------------------------------------------------------- presence

def presence(frame: pd.DataFrame, source_path: Path, cutout_dir: Path, size: int,
             bands: str, min_bytes: int, log=print) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"targetid": frame["targetid"].to_numpy(np.int64)})
    stats: dict = {}

    with h5py.File(source_path, "r") as h:
        staged = h["desi_targetid"][:].astype(np.int64)
    row_of = pd.Series(np.arange(staged.size), index=staged)
    row_of = row_of[~row_of.index.duplicated()]
    out["source_row"] = row_of.reindex(out["targetid"]).fillna(-1).to_numpy(np.int64)
    out["has_spectrum"] = out["source_row"].to_numpy() >= 0

    unreadable = []
    has_image = np.zeros(len(out), bool)
    for i, tid in enumerate(out["targetid"]):
        path = cutout_path(cutout_dir, tid)
        if not path.is_file() or path.stat().st_size < min_bytes:
            continue
        try:
            read_cutout(path, size, bands)
        except (BadCutout, OSError, ValueError) as exc:
            unreadable.append([int(tid), str(exc)[:120]])
            continue
        has_image[i] = True
    out["has_image"] = has_image
    stats["cutouts_unreadable"] = len(unreadable)
    stats["cutouts_unreadable_list"] = unreadable[:50]

    z = frame["z"].to_numpy(np.float64)
    zwarn = frame["zwarn"].to_numpy(np.float64)
    out["has_z"] = np.isfinite(z) & (z > 0) & (zwarn == 0)
    stats["z_flagged_by_zwarn"] = int((np.isfinite(z) & (z > 0) & (zwarn != 0)).sum())
    stats["z_nonpositive_or_missing"] = int((~np.isfinite(z) | (z <= 0)).sum())

    for band in WISE_BANDS:
        flux = frame[f"ls10_flux_{band}"].to_numpy(np.float64)
        ivar = frame[f"ls10_flux_ivar_{band}"].to_numpy(np.float64)
        out[f"has_{band}"] = np.isfinite(flux) & (flux > 0) & np.isfinite(ivar) & (ivar > 0)
    out["has_wise"] = out[[f"has_{b}" for b in WISE_BANDS]].to_numpy().any(axis=1)
    for flag in PRESENCE_FLAGS + tuple(f"has_{b}" for b in WISE_BANDS):
        log(f"[presence] {flag:13s} {int(out[flag].sum()):,} of {len(out):,}")
    return out, stats


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
    c = cfg["cutouts"]
    s = cfg["split"]

    frame = pd.read_parquet(xm_path)
    flags, presence_stats = presence(frame, source_path, cutout_dir, int(c["size"]),
                                     str(c["bands"]), int(c["min_bytes"]), log=log)
    manifest = pd.concat([frame, flags.drop(columns=["targetid"])], axis=1)
    for band in WISE_BANDS:
        manifest[f"flux_{band}"] = manifest[f"ls10_flux_{band}"]

    # the sample: no split-source rows, and a spectrum
    led = FilterLedger(len(manifest))
    keep = led.apply("split_source_pairs_excluded", ~manifest["split_source"].to_numpy())
    kept = led.apply("has_spectrum", manifest["has_spectrum"].to_numpy()[keep])
    in_sample = keep.copy()
    in_sample[keep] = kept
    manifest["in_sample"] = in_sample
    sample = manifest[in_sample]
    if sample["targetid"].duplicated().any():
        raise ManifestError("a target appears twice in the sample; the crossmatch should "
                            "have resolved every shared target")
    if len(sample) == 0:
        raise ManifestError("every row was filtered out")

    # the split
    comp = components(sample["ero_detuid"].to_numpy(), sample["targetid"].to_numpy())
    split = assign(comp, str(s["hash"]["salt"]), s["fractions"])
    manifest["component"] = pd.Series(comp, index=sample.index).reindex(manifest.index)
    manifest["split"] = pd.Series(split, index=sample.index).reindex(manifest.index)
    n = len(sample)
    row_counts = {name: int((split == name).sum()) for name in SPLITS}
    comp_counts = {name: int(pd.unique(comp[split == name]).size) for name in SPLITS}
    drift = {name: row_counts[name] / n - f for name, f in zip(SPLITS, s["fractions"])}
    for name in SPLITS:
        log(f"[split] {name:5s} {row_counts[name]:7,} rows  {comp_counts[name]:7,} components "
            f"({row_counts[name] / n:.4f}, drift {drift[name]:+.4f})")
    bad = [name for name in SPLITS if abs(drift[name]) > float(s["tolerance"])]
    if bad:
        raise ManifestError(f"row fractions for {bad} drift more than the tolerance "
                            f"{s['tolerance']}; re-check the fractions or raise the tolerance")

    manifest = manifest[MANIFEST_COLUMNS].sort_values("ero_detuid").reset_index(drop=True)
    manifest_path, split_path = work / MANIFEST, work / SPLIT
    manifest.to_csv(manifest_path, index=False)
    split_frame = (manifest.loc[manifest["in_sample"], ["targetid", "split"]]
                   .sort_values("targetid").reset_index(drop=True))
    split_frame.to_csv(split_path, index=False)

    sample = manifest[manifest["in_sample"]]
    census = {name: sample.loc[sample["split"] == name, "spectype"].value_counts().to_dict()
              for name in SPLITS}
    coverage = {flag: int(sample[flag].sum()) for flag in
                PRESENCE_FLAGS + tuple(f"has_{b}" for b in WISE_BANDS)}
    comp_sizes = sample.groupby("component").size()
    log(f"[out] sample {n:,} rows; census "
        f"{sample['spectype'].value_counts().to_dict()}")
    write_ledger(STEP, cfg,
                 inputs={"crossmatch": xm_path, "spectra_source": source_path},
                 counts={"crossmatch_rows": int(len(manifest)), "sample_rows": n,
                         "split_source_rows_excluded": int(manifest["split_source"].sum()),
                         "components": int(comp_sizes.size),
                         "largest_component_rows": int(comp_sizes.max()),
                         **{f"rows_{k}": v for k, v in row_counts.items()},
                         **{f"components_{k}": v for k, v in comp_counts.items()}},
                 filters=led.rows,
                 extra={"hash": dict(s["hash"]), "fractions": list(s["fractions"]),
                        "tolerance": s["tolerance"], "drift": drift,
                        "presence_in_sample": coverage,
                        "presence_fraction_in_sample": {k: v / n for k, v in coverage.items()},
                        "presence": presence_stats, "census_by_split": census,
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
