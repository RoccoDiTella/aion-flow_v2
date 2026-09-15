"""The staged split, its labels, and the batches the trainer reads.

Phase 1 writes the inputs to `<staged>/{split}.h5` and the labels to
`<work>/labels.csv`, so no label is ever staged and a staged copy cannot go
stale against the catalogue. `labels.csv` holds one row per cross-match row and
`manifest.csv` names the row that entered the sample, so the join runs staged
targetid -> manifest -> DETUID -> labels. This module adds that join, the
training-split standardization, and the per-source modality presence the
dropout sampler clamps to. The sampler itself is in `objective.py`: which modalities a step
hides is part of the objective, not of the data.

A target is usable for a source when its label is finite; a rate target when
its aperture triple is complete and the exposure is positive. A zero-count band
is a measurement, not a missing value, which is the point of the Poisson
objective, so it is kept.

Every batch carries the whole target registry regardless of which heads a run
trains, so the batch contract does not depend on the recipe:

    targetid       int64   (B,)
    tok_z          int32   (B, 1)         tok_spectrum_desi int32 (B, 273)
    tok_image      int32   (B, 576)       tok_flux_w1..w3   int32 (B, 1)
    present        bool    (B, 4)         over MODALITIES
    y              float32 (B, 4)         standardized scalars, 0 where not usable
    y_ok           bool    (B, 4)
    counts, bkg, expo  float64 (B, 2)     the aperture triples
    rate_ok        bool    (B, 2)

The spectra and images themselves never reach a batch: `tokenize.py` has already
run AION's codecs over them. They stay reachable on the split, which is what
that step reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aionflow_data.labels import OUTPUT as LABELS_FILE
from aionflow_data.manifest_split import MANIFEST, SPLITS
from aionflow_data.stage import SPLIT_FILE

TOKENS_FILE = "tokens_{split}.h5"

MODALITIES = ("Z", "S", "I", "W")
NET_COUNT_FLOOR = 0.5      # N - B floored at half a photon before a log or a sigma

# What each modality contributes to AION's encoder, and how many tokens of it.
TOKEN_KEYS: dict[str, tuple[tuple[str, int], ...]] = {
    "Z": (("tok_z", 1),),
    "S": (("tok_spectrum_desi", 273),),
    "I": (("tok_image", 576),),
    "W": (("tok_flux_w1", 1), ("tok_flux_w2", 1), ("tok_flux_w3", 1)),
}
ALL_TOKEN_KEYS = tuple(key for m in MODALITIES for key, _ in TOKEN_KEYS[m])
TOKEN_SIZES = {key: n for m in MODALITIES for key, n in TOKEN_KEYS[m]}
TOKENS_PER_SOURCE = sum(TOKEN_SIZES.values())


class DataError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- targets

@dataclass(frozen=True)
class Target:
    """One predicted quantity. A scalar target has a label column; a rate target
    has the aperture triple its Poisson likelihood is written on."""

    name: str
    kind: str                      # "scalar" or "rate"
    columns: tuple[str, ...]
    label: str                     # for tables and figures


TARGETS: dict[str, Target] = {
    "flux": Target("flux", "scalar", ("log_flux_1",), "X-ray flux"),
    "lx": Target("lx", "scalar", ("log_lx",), "log LX"),
    "sfr": Target("sfr", "scalar", ("log_sfr",), "log SFR"),
    "mstar": Target("mstar", "scalar", ("logmstar_cigale",), "log M*"),
    "rate_p2": Target("rate_p2", "rate", ("ape_cts_p2", "ape_bkg_p2", "ape_exp_p2"), "lambda_P2"),
    "rate_p3": Target("rate_p3", "rate", ("ape_cts_p3", "ape_bkg_p3", "ape_exp_p3"), "lambda_P3"),
}
SCALAR_TARGETS = tuple(t.name for t in TARGETS.values() if t.kind == "scalar")
RATE_TARGETS = tuple(t.name for t in TARGETS.values() if t.kind == "rate")


def log_plug_in_rate(counts, bkg, expo):
    """log10 of the plug-in rate (N - B) / t, with N - B floored at half a photon.

    This is the point the Laplace proposal recentres on and the quantity the rate
    standardizer is fitted to.
    """
    net = np.maximum(np.asarray(counts, float) - np.asarray(bkg, float), NET_COUNT_FLOOR)
    return np.log10(net / np.asarray(expo, float))


# ----------------------------------------------------------------------------- scaling

class Standardizer:
    """Zero mean and unit variance per target, from the training split."""

    def __init__(self, mean: dict[str, float], scale: dict[str, float]):
        missing = set(TARGETS) - set(mean) | set(TARGETS) - set(scale)
        if missing:
            raise DataError(f"standardizer is missing targets {sorted(missing)}")
        self.mean = {k: float(mean[k]) for k in TARGETS}
        self.scale = {k: float(scale[k]) for k in TARGETS}
        bad = [k for k, v in self.scale.items() if not np.isfinite(v) or v <= 0]
        if bad:
            raise DataError(f"non-positive standardizer scale for {bad}")

    @classmethod
    def fit(cls, split: Split) -> Standardizer:
        if split.name != "train":
            raise DataError(f"standardizers are fitted on the training split, not {split.name!r}")
        mean, scale = {}, {}
        for i, name in enumerate(SCALAR_TARGETS):
            mean[name], scale[name] = _moments(name, split.y_raw[:, i][split.y_ok[:, i]])
        for i, name in enumerate(RATE_TARGETS):
            ok = split.rate_ok[:, i]
            values = log_plug_in_rate(split.counts[ok, i], split.bkg[ok, i], split.expo[ok, i])
            mean[name], scale[name] = _moments(name, values)
        return cls(mean, scale)

    def encode(self, name: str, x):
        return (np.asarray(x, float) - self.mean[name]) / self.scale[name]

    def decode(self, name: str, u):
        return np.asarray(u, float) * self.scale[name] + self.mean[name]

    def as_dict(self) -> dict:
        """Copies, so a caller cannot reach in and change the fitted scaling."""
        return {"mean": dict(self.mean), "scale": dict(self.scale)}

    @classmethod
    def from_dict(cls, d: dict) -> Standardizer:
        return cls(d["mean"], d["scale"])

    def write(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | os.PathLike) -> Standardizer:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _moments(name: str, values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    if values.size < 2:
        raise DataError(f"target {name!r} has {values.size} usable training rows")
    return float(values.mean()), float(values.std())


# ----------------------------------------------------------------------------- a split

class Split:
    """One staged split joined to `labels.csv` and `manifest.csv` by targetid.

    The scalars, flags and labels are held in memory, in staged row order. The
    spectra and images stay on disk and are read a row at a time.
    """

    def __init__(self, staged: str | os.PathLike, work: str | os.PathLike, name: str):
        if name not in SPLITS:
            raise DataError(f"unknown split {name!r}; expected one of {SPLITS}")
        self.name = name
        self.path = Path(staged) / SPLIT_FILE.format(split=name)
        if not self.path.is_file():
            raise DataError(f"missing staged split {self.path}")
        self._files: dict[tuple[str, int], h5py.File] = {}
        self.tokens_path = Path(staged) / TOKENS_FILE.format(split=name)
        with h5py.File(self.path, "r") as h:
            self.targetid = h["targetid"][:].astype(np.int64)
            self.redshift = h["redshift"][:].astype(np.float64)
            self.wise = np.stack([h[f"flux_w{b}"][:] for b in (1, 2, 3)], 1).astype(np.float64)
            has_z, has_wise = h["has_z"][:].astype(bool), h["has_wise"][:].astype(bool)
            self.wavelength = h["spectra_lambda"][:].astype(np.float32)
            self.nbin = int(h["spectra"].shape[1])
            self.image_size = int(h.attrs["image_size"])
            self.image_bands = tuple(b.decode() for b in h.attrs["image_bands"])
        self.n = self.targetid.size
        # Every sample row has a spectrum and an image by construction of the sample.
        self.present = np.stack([has_z, np.ones(self.n, bool), np.ones(self.n, bool), has_wise], 1)
        self._join(Path(work))

    def _join(self, work: Path) -> None:
        """labels.csv is one row per cross-match row, so a targetid can appear twice.
        The manifest names the row that entered the sample; the join goes through it."""
        manifest = _read_csv(work / MANIFEST, "manifest")
        manifest = manifest[manifest["in_sample"].astype(bool)]
        rows = _index_unique(manifest, "targetid", MANIFEST).reindex(self.targetid)
        missing = self.targetid[rows["ero_detuid"].isna().to_numpy()]
        if missing.size:
            raise DataError(f"{missing.size} staged targets are not in the manifest sample, "
                            f"first {missing[0]}")
        if not (rows["split"].to_numpy() == self.name).all():
            raise DataError(f"{self.path.name} holds rows the manifest does not call {self.name}")
        self.detuid = rows["ero_detuid"].to_numpy(str)
        self.spectype = rows["spectype"].to_numpy(str)
        labels = _index_unique(_read_csv(work / LABELS_FILE, "labels"), "ero_detuid", LABELS_FILE)
        # get_indexer is a hash lookup. np.setdiff1d here was a sort, and between our
        # <U32 array and pandas' object index numpy falls back to comparing Python
        # strings one at a time: six minutes on the real sample, against a tenth of a
        # second, and invisible on fixtures of thirty rows.
        where = labels.index.get_indexer(self.detuid)
        if (where < 0).any():
            absent = self.detuid[where < 0]
            raise DataError(f"{absent.size} staged DETUIDs have no labels row, "
                            f"first {absent[0]}")
        labels = labels.take(where)
        self.y_raw = np.stack([labels[TARGETS[t].columns[0]].to_numpy(float)
                               for t in SCALAR_TARGETS], 1)
        self.y_ok = np.isfinite(self.y_raw)
        stack = np.stack([np.stack([labels[c].to_numpy(float) for c in TARGETS[t].columns], 1)
                          for t in RATE_TARGETS], 1)                 # (n, n_rates, 3)
        self.counts, self.bkg, self.expo = stack[:, :, 0], stack[:, :, 1], stack[:, :, 2]
        self.rate_ok = np.isfinite(stack).all(2) & (self.expo > 0)

    def standardized(self, standardizer: Standardizer) -> np.ndarray:
        """The scalar targets in standardized units, zero where not usable."""
        out = np.zeros_like(self.y_raw)
        for i, name in enumerate(SCALAR_TARGETS):
            ok = self.y_ok[:, i]
            out[ok, i] = standardizer.encode(name, self.y_raw[ok, i])
        return out

    def _handle(self, which: str = "inputs") -> h5py.File:
        """One open handle per process, so DataLoader workers do not share one."""
        path = self.path if which == "inputs" else self.tokens_path
        key = (which, os.getpid())
        if key not in self._files:
            if not path.is_file():
                raise DataError(f"missing {path}; run aionflow_model.tokenize"
                                if which == "tokens" else f"missing {path}")
            self._files[key] = h5py.File(path, "r")
        return self._files[key]

    def spectra(self, lo: int, hi: int) -> tuple[Tensor, Tensor]:
        """Flux and inverse variance for rows [lo, hi), for the codecs."""
        h = self._handle()
        return (torch.from_numpy(h["spectra"][lo:hi]), torch.from_numpy(h["spectra_ivar"][lo:hi]))

    def images(self, lo: int, hi: int) -> Tensor:
        return torch.from_numpy(self._handle()["image_flux"][lo:hi])

    def tokens(self, row: int) -> dict[str, np.ndarray]:
        h = self._handle("tokens")
        return {key: h[key][row] for key in ALL_TOKEN_KEYS}

    def close(self) -> None:
        for h in self._files.values():
            h.close()
        self._files.clear()

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"Split({self.name!r}, n={self.n}, nbin={self.nbin}, image={self.image_size})"


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.is_file():
        raise DataError(f"missing {what} at {path}")
    return pd.read_csv(path)


def _index_unique(frame: pd.DataFrame, key: str, what: str) -> pd.DataFrame:
    out = frame.set_index(key)
    if out.index.duplicated().any():
        raise DataError(f"{what} has duplicate {key}s among the rows that entered the sample")
    return out


# ----------------------------------------------------------------------------- batches

class TokenDataset(Dataset):
    """Rows of one split as AION token ids and standardized targets.

    Spectra and images do not appear: `aionflow_model.tokenize` has already run
    the codecs over them, and training never reads them again.
    """

    def __init__(self, split: Split, standardizer: Standardizer):
        self.split = split
        self.y = split.standardized(standardizer).astype(np.float32)

    def __len__(self) -> int:
        return self.split.n

    def __getitem__(self, row: int) -> dict:
        s = self.split
        item = {key: torch.from_numpy(np.asarray(value, np.int32))
                for key, value in s.tokens(row).items()}
        item.update({
            "targetid": torch.tensor(s.targetid[row], dtype=torch.int64),
            "present": torch.from_numpy(s.present[row].copy()),
            "y": torch.from_numpy(self.y[row].copy()),
            "y_ok": torch.from_numpy(s.y_ok[row].copy()),
            "counts": torch.from_numpy(s.counts[row].copy()),
            "bkg": torch.from_numpy(s.bkg[row].copy()),
            "expo": torch.from_numpy(s.expo[row].copy()),
            "rate_ok": torch.from_numpy(s.rate_ok[row].copy()),
        })
        return item


def loader(dataset: TokenDataset, batch_size: int, *, shuffle: bool, workers: int = 0,
           seed: int | None = None) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed if seed is not None else 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, drop_last=False,
                      persistent_workers=bool(workers))
