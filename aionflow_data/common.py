"""Shared helpers: configuration, hashing, FITS column reads, provenance ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

REQUIRED_SECTIONS = (
    "inputs", "archives", "crossmatch", "labels", "spectra", "cutouts", "split", "stage", "paths",
)
INPUT_NAMES = ("nway", "main", "desi_zcat", "cigale")
PATH_KEYS = ("raw", "work", "staged", "provenance")


# ----------------------------------------------------------------------------- config

def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read the YAML config. Relative entries under `paths` resolve against the file.

    Resolution order: explicit argument, `AIONFLOW_CONFIG`, then `config.yaml` at the
    repository root. The resolved config path is stored under `_config_path`.
    """
    path = Path(path or os.environ.get("AIONFLOW_CONFIG") or DEFAULT_CONFIG).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"{path}: missing config sections {missing}")
    missing_inputs = [n for n in INPUT_NAMES if n not in cfg["inputs"]]
    if missing_inputs:
        raise KeyError(f"{path}: missing inputs {missing_inputs}")
    missing_paths = [k for k in PATH_KEYS if k not in cfg["paths"]]
    if missing_paths:
        raise KeyError(f"{path}: missing paths {missing_paths}")
    root = path.parent
    for key in PATH_KEYS:
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else (root / p).resolve())
    cfg["_config_path"] = str(path)
    return cfg


def step_parser(description: str) -> argparse.ArgumentParser:
    """An argument parser with the `--config` option every step takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None,
                        help="pipeline config (default: $AIONFLOW_CONFIG or config.yaml)")
    return parser


def ensure_dirs(cfg: dict) -> None:
    for key in PATH_KEYS:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- hashing

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_digest(path: str | os.PathLike, algorithm: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_digests(path: str | os.PathLike, algorithms: tuple[str, ...] = ("md5", "sha256"),
                 chunk: int = 1 << 20) -> dict[str, str]:
    """Several digests of one file in a single pass."""
    hashers = {a: hashlib.new(a) for a in algorithms}
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            for h in hashers.values():
                h.update(block)
    return {a: h.hexdigest() for a, h in hashers.items()}


def sha256(path: str | os.PathLike) -> str:
    return file_digest(path, "sha256")


def md5(path: str | os.PathLike) -> str:
    return file_digest(path, "md5")


# ----------------------------------------------------------------------------- FITS

def native(a: np.ndarray) -> np.ndarray:
    """FITS is big-endian; pandas refuses to hash or sort that on x86."""
    a = np.asarray(a)
    if a.dtype.byteorder == ">":
        return a.astype(a.dtype.newbyteorder("="))
    return a


def fits_nrows(path: str | os.PathLike, hdu: int = 1) -> int:
    from astropy.io import fits

    with fits.open(path, memmap=True) as handle:
        return int(handle[hdu].header["NAXIS2"])


def fits_column_names(path: str | os.PathLike, hdu: int = 1) -> list[str]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as handle:
        return list(handle[hdu].columns.names)


def read_fits_columns(path: str | os.PathLike, columns: list[str] | tuple[str, ...],
                      hdu: int = 1, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Read named columns one at a time off a memmap, subset on read, native byte order.

    `rows` is a boolean mask or an index array over the table; None reads every row.
    Peak memory is one column, not the table.
    """
    from astropy.io import fits

    out: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=True) as handle:
        data = handle[hdu].data
        names = set(data.columns.names)
        absent = [c for c in columns if c not in names]
        if absent:
            raise KeyError(f"{path}: columns {absent} not in HDU {hdu}")
        for name in columns:
            col = np.asarray(data[name])
            if rows is not None:
                col = col[rows]
            out[name] = native(np.ascontiguousarray(col))
            del col
    return out


# ----------------------------------------------------------------------------- ledgers

class FilterLedger:
    """Record every row cut a step makes, in order, with what it kept and dropped."""

    def __init__(self, n_in: int) -> None:
        self.n_in = int(n_in)
        self.n = int(n_in)
        self.rows: list[dict] = []

    def apply(self, name: str, keep, note: str | None = None) -> np.ndarray:
        keep = np.asarray(keep, bool)
        if keep.shape != (self.n,):
            raise ValueError(f"filter {name!r}: mask has shape {keep.shape}, expected ({self.n},)")
        kept = int(keep.sum())
        row = {"filter": name, "kept": kept, "dropped": self.n - kept}
        if note:
            row["note"] = note
        self.rows.append(row)
        print(f"[filter] {name:30s} kept {kept:>10,}  dropped {self.n - kept:>9,}", flush=True)
        self.n = kept
        return keep


def describe_file(path: str | os.PathLike, digest: str | None = None) -> dict:
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest or sha256(p)}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def ledger_path(step: str, cfg: dict) -> Path:
    return Path(cfg["paths"]["provenance"]) / f"{step}.json"


def write_ledger(step: str, cfg: dict, *, inputs: dict, counts: dict,
                 filters: list[dict] | tuple = (), extra: dict | None = None) -> Path:
    """Write `data/provenance/<step>.json`: hashed inputs, counts, filters, extras.

    `inputs` maps a name to a path (hashed here) or to a dict already holding
    `path`, `bytes` and `sha256` (used as given, for files hashed upstream).
    """
    described = {}
    for name, value in inputs.items():
        described[name] = dict(value) if isinstance(value, dict) else describe_file(value)
    record = {
        "step": step,
        "written_utc": utc_now(),
        "config": {"path": cfg["_config_path"], "sha256": sha256(cfg["_config_path"])},
        "inputs": described,
        "counts": dict(counts),
        "filters": [dict(f) for f in filters],
        "extra": dict(extra or {}),
    }
    path = ledger_path(step, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    print(f"[ledger] wrote {path}", flush=True)
    return path


def read_ledger(step: str, cfg: dict) -> dict | None:
    path = ledger_path(step, cfg)
    if not path.is_file():
        return None
    return json.loads(path.read_text())
