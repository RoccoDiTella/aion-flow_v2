"""Step 6: stage the sample into the per-split HDF5 files the trainer reads.

    python -m aionflow_data.stage [--config CONFIG]

Inputs only. No label is written here: labels join at load time from
work/labels.csv by targetid, so a staged copy can never go stale against the
catalogue. Per split, rows sorted by `source_row`:

    source_row, desi_targetid          int64
    spectra, spectra_ivar              float32 (n, nbin), from spectra/source.h5
    spectra_lambda                     float32 (nbin)
    redshift, flux_w1, flux_w2, flux_w3, target_ra, target_dec   float32
    image_flux                         float32 (n, 4, size, size), from the cutouts
    has_spectrum, has_z, has_wise, has_image                     bool

A row whose manifest says `has_image` is false gets an all-zero frame; the
flags, not the pixels, are the modality's verdict. Every dataset is chunked
along rows only (one row is read at a time in training, and a chunk that
splits a row across columns multiplies every read), gzip on the spectra, none
on the images. Attributes `image_bands` and `image_size` describe the image
tensor. summary.json and the ledger data/provenance/stage.json record counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .fetch_cutouts import CUTOUT_DIR, cutout_path, read_cutout
from .fetch_spectra import SOURCE as SPECTRA_SOURCE
from .manifest_split import MANIFEST, PRESENCE_FLAGS, SPLITS

STEP = "stage"
SPLIT_FILE = "desi_{split}.hdf5"
SUMMARY = "summary.json"
IMAGE_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")
SPECTRA_BLOCK = 2048
SCALARS = {"redshift": "z", "flux_w1": "flux_w1", "flux_w2": "flux_w2", "flux_w3": "flux_w3",
           "target_ra": "target_ra", "target_dec": "target_dec"}


class StageError(RuntimeError):
    pass


def row_chunks(shape: tuple[int, ...], itemsize: int, target_bytes: int) -> tuple[int, ...] | None:
    """Chunk along rows only, every other axis full, about `target_bytes` per chunk."""
    if not shape or shape[0] == 0:
        return None
    row_bytes = itemsize
    for dim in shape[1:]:
        row_bytes *= int(dim)
    rows = max(1, int(target_bytes) // max(row_bytes, 1))
    return (min(int(shape[0]), rows), *(int(d) for d in shape[1:]))


def _compression(cfg_stage: dict) -> dict:
    kind = str(cfg_stage.get("spectra_compression", "gzip"))
    if kind == "none":
        return {}
    if kind == "gzip":
        return {"compression": "gzip",
                "compression_opts": int(cfg_stage.get("spectra_compression_level", 4))}
    if kind == "lzf":
        return {"compression": "lzf"}
    raise StageError(f"unsupported spectra_compression {kind!r}")


def stage_split(name: str, rows: pd.DataFrame, source: h5py.File, cutout_dir: Path,
                dest: Path, cfg: dict, log=print) -> dict:
    n = len(rows)
    src_rows = rows["source_row"].to_numpy(np.int64)
    if n and not (np.diff(src_rows) > 0).all():
        raise StageError("rows must be sorted by unique source_row")
    nbin = int(source["spectra"].shape[1])
    size = int(cfg["cutouts"]["size"])
    bands = str(cfg["cutouts"]["bands"])
    target = int(cfg["stage"]["chunk_target_bytes"])
    gz = _compression(cfg["stage"])
    tmp = dest.with_name(dest.name + ".part")

    def create(key, data, **kw):
        arr = np.asarray(data)
        return h.create_dataset(key, data=arr, chunks=row_chunks(arr.shape, arr.dtype.itemsize,
                                                                 target), track_times=False,
                                **kw)

    with h5py.File(tmp, "w") as h:
        create("source_row", src_rows, **gz)
        create("desi_targetid", rows["targetid"].to_numpy(np.int64), **gz)
        for key, col in SCALARS.items():
            create(key, rows[col].to_numpy(np.float64).astype(np.float32), **gz)
        for flag in PRESENCE_FLAGS:
            create(flag, rows[flag].to_numpy(bool), **gz)
        h.create_dataset("spectra_lambda", data=source["spectra_lambda"][:].astype(np.float32),
                         track_times=False)
        for key in ("spectra", "spectra_ivar"):
            ds = h.create_dataset(key, shape=(n, nbin), dtype=np.float32,
                                  chunks=row_chunks((n, nbin), 4, target), track_times=False,
                                  **gz)
            for lo in range(0, n, SPECTRA_BLOCK):
                hi = min(lo + SPECTRA_BLOCK, n)
                ds[lo:hi] = source[key][src_rows[lo:hi]]
        images = h.create_dataset("image_flux", shape=(n, len(IMAGE_BANDS), size, size),
                                  dtype=np.float32, track_times=False,
                                  chunks=row_chunks((n, len(IMAGE_BANDS), size, size), 4, target))
        has_image = rows["has_image"].to_numpy(bool)
        zero = np.zeros((len(IMAGE_BANDS), size, size), np.float32)
        for i, (tid, present) in enumerate(zip(rows["targetid"], has_image)):
            images[i] = read_cutout(cutout_path(cutout_dir, tid), size, bands) if present else zero
            if (i + 1) % 5000 == 0:
                log(f"[stage] {name}: {i + 1:,}/{n:,} images", flush=True)
        h.attrs["image_bands"] = np.asarray(IMAGE_BANDS, dtype="S")
        h.attrs["image_size"] = size
        h.attrs["split"] = name
    tmp.replace(dest)
    stats = {"rows": n, "has_image": int(has_image.sum()),
             "zero_images": int((~has_image).sum()), "bytes": dest.stat().st_size}
    log(f"[stage] {name}: {n:,} rows, {stats['has_image']:,} with an image -> {dest.name} "
        f"({stats['bytes'] / 2 ** 20:.1f} MiB)")
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
    sample = manifest[manifest["in_sample"].astype(bool)]
    if sample["split"].isna().any():
        raise StageError("a sample row has no split; rebuild the manifest")
    summary: dict = {"manifest": str(manifest_path), "spectra_source": str(source_path),
                     "cutout_dir": str(cutout_dir),
                     "chunk_target_bytes": int(cfg["stage"]["chunk_target_bytes"]),
                     "splits": {}}
    with h5py.File(source_path, "r") as source:
        if int(source["spectra"].shape[1]) != int(cfg["spectra"]["nbin"]):
            raise StageError(f"{source_path.name} has {source['spectra'].shape[1]} bins, the "
                             f"config says {cfg['spectra']['nbin']}")
        for name in SPLITS:
            rows = sample[sample["split"] == name].sort_values("source_row")
            summary["splits"][name] = stage_split(
                name, rows, source, cutout_dir, staged / SPLIT_FILE.format(split=name), cfg,
                log=log)
    (staged / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    counts = {"sample_rows": int(len(sample))}
    for name, st in summary["splits"].items():
        counts[f"rows_{name}"] = st["rows"]
        counts[f"has_image_{name}"] = st["has_image"]
    write_ledger(STEP, cfg,
                 inputs={"manifest": manifest_path, "spectra_source": source_path},
                 counts=counts,
                 extra={"image_bands": list(IMAGE_BANDS), "image_size": int(cfg["cutouts"]["size"]),
                        "chunk_target_bytes": summary["chunk_target_bytes"],
                        "spectra_compression": _compression(cfg["stage"]),
                        "outputs": {name: describe_file(staged / SPLIT_FILE.format(split=name))
                                    for name in SPLITS}})
    return summary


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
