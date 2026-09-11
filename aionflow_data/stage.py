"""Step 6: stage the sample into the per-split HDF5 files the trainer reads.

    python -m aionflow_data.stage [--config CONFIG]

Inputs only. No label is written here: labels join at load time from
work/labels.csv by targetid, so a staged copy can never go stale against the
catalogue. Per split, rows in the order of spectra/source.h5:

    targetid                           int64
    spectra, spectra_ivar              float32 (n, nbin), from spectra/source.h5
    spectra_lambda                     float32 (nbin)
    redshift, flux_w1, flux_w2, flux_w3   float32
    image_flux                         float32 (n, 4, size, size), from the cutouts
    has_z, has_wise                    bool

Every dataset is chunked along rows only (one row is read at a time in
training), gzip on the spectra, none on the images. Attributes `image_bands`
and `image_size` describe the image tensor. The ledger data/provenance/stage.json
records the counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .fetch_cutouts import CUTOUT_DIR, cutout_path, read_cutout
from .fetch_spectra import SOURCE as SPECTRA_SOURCE
from .manifest_split import MANIFEST, SPLITS

STEP = "stage"
SPLIT_FILE = "{split}.h5"
IMAGE_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")
CHUNK_TARGET_BYTES = 256 * 1024
SPECTRA_BLOCK = 2048
SCALARS = {"redshift": "z", "flux_w1": "ls10_flux_w1", "flux_w2": "ls10_flux_w2",
           "flux_w3": "ls10_flux_w3"}
FLAGS = ("has_z", "has_wise")
GZIP = {"compression": "gzip", "compression_opts": 4}


class StageError(RuntimeError):
    pass


def row_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...] | None:
    """Chunk along rows only, every other axis full, about CHUNK_TARGET_BYTES per chunk."""
    if not shape or shape[0] == 0:
        return None
    row_bytes = itemsize
    for dim in shape[1:]:
        row_bytes *= int(dim)
    rows = max(1, CHUNK_TARGET_BYTES // max(row_bytes, 1))
    return (min(int(shape[0]), rows), *(int(d) for d in shape[1:]))


def stage_split(name: str, rows: pd.DataFrame, source: h5py.File, cutout_dir: Path,
                dest: Path, size: int, bands: str, log=print) -> dict:
    """Write one split. `rows` must be ordered by their row in `source`."""
    n = len(rows)
    src_rows = rows["source_row"].to_numpy(np.int64)
    if n and not (np.diff(src_rows) > 0).all():
        raise StageError("rows must be ordered by unique source row")
    nbin = int(source["spectra"].shape[1])
    tmp = dest.with_name(dest.name + ".part")

    def create(key, data, **kw):
        arr = np.asarray(data)
        return h.create_dataset(key, data=arr, chunks=row_chunks(arr.shape, arr.dtype.itemsize),
                                track_times=False, **kw)

    with h5py.File(tmp, "w") as h:
        create("targetid", rows["targetid"].to_numpy(np.int64), **GZIP)
        for key, col in SCALARS.items():
            create(key, rows[col].to_numpy(np.float64).astype(np.float32), **GZIP)
        for flag in FLAGS:
            create(flag, rows[flag].to_numpy(bool), **GZIP)
        h.create_dataset("spectra_lambda", data=source["spectra_lambda"][:].astype(np.float32),
                         track_times=False)
        for key in ("spectra", "spectra_ivar"):
            ds = h.create_dataset(key, shape=(n, nbin), dtype=np.float32,
                                  chunks=row_chunks((n, nbin), 4), track_times=False, **GZIP)
            for lo in range(0, n, SPECTRA_BLOCK):
                hi = min(lo + SPECTRA_BLOCK, n)
                ds[lo:hi] = source[key][src_rows[lo:hi]]
        images = h.create_dataset("image_flux", shape=(n, len(IMAGE_BANDS), size, size),
                                  dtype=np.float32, track_times=False,
                                  chunks=row_chunks((n, len(IMAGE_BANDS), size, size), 4))
        for i, tid in enumerate(rows["targetid"]):
            images[i] = read_cutout(cutout_path(cutout_dir, tid), size, bands)
            if (i + 1) % 5000 == 0:
                log(f"[stage] {name}: {i + 1:,}/{n:,} images", flush=True)
        h.attrs["image_bands"] = np.asarray(IMAGE_BANDS, dtype="S")
        h.attrs["image_size"] = size
        h.attrs["split"] = name
    tmp.replace(dest)
    stats = {"rows": n, "bytes": dest.stat().st_size}
    log(f"[stage] {name}: {n:,} rows -> {dest.name} ({stats['bytes'] / 2 ** 20:.1f} MiB)")
    return stats


def run(cfg: dict, log=print) -> dict:
    ensure_dirs(cfg)
    work, staged = Path(cfg["paths"]["work"]), Path(cfg["paths"]["staged"])
    manifest_path, source_path = work / MANIFEST, work / SPECTRA_SOURCE
    cutout_dir = work / CUTOUT_DIR
    for p in (manifest_path, source_path):
        if not p.is_file():
            raise StageError(f"missing input {p}")
    manifest = pd.read_csv(manifest_path)
    sample = manifest[manifest["in_sample"].astype(bool)].copy()
    if sample["split"].isna().any():
        raise StageError("a sample row has no split; rebuild the manifest")
    size, bands = int(cfg["cutouts"]["size"]), str(cfg["cutouts"]["bands"])
    splits: dict[str, dict] = {}
    with h5py.File(source_path, "r") as source:
        if int(source["spectra"].shape[1]) != int(cfg["spectra"]["nbin"]):
            raise StageError(f"{source_path.name} has {source['spectra'].shape[1]} bins, the "
                             f"config says {cfg['spectra']['nbin']}")
        row_of = pd.Series(np.arange(source["targetid"].shape[0]),
                           index=source["targetid"][:].astype(np.int64))
        sample["source_row"] = row_of.reindex(sample["targetid"]).to_numpy()
        if sample["source_row"].isna().any():
            raise StageError("a sample target has no spectrum in spectra/source.h5")
        for name in SPLITS:
            rows = sample[sample["split"] == name].sort_values("source_row")
            splits[name] = stage_split(name, rows, source, cutout_dir,
                                       staged / SPLIT_FILE.format(split=name), size, bands,
                                       log=log)
    counts = {"sample_rows": int(len(sample)),
              **{f"rows_{k}": v["rows"] for k, v in splits.items()}}
    write_ledger(STEP, cfg,
                 inputs={"manifest": manifest_path, "spectra_source": source_path},
                 counts=counts,
                 extra={"image_bands": list(IMAGE_BANDS), "image_size": size,
                        "outputs": {name: describe_file(staged / SPLIT_FILE.format(split=name))
                                    for name in SPLITS}})
    return {"splits": splits}


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg)
    except StageError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
