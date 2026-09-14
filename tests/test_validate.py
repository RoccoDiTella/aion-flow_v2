"""Cut 9: the validator passes on a good staging and names the check that a corruption breaks."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from aionflow_data import (
    common,
    crossmatch,
    fetch_spectra,
    labels,
    manifest_split,
    stage,
    validate,
)
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    """A complete good run, kept pristine; tests clone it before corrupting."""
    root = tmp_path_factory.mktemp("validate")
    cfg = make_fx_cfg(root)
    crossmatch.run(cfg, **QUIET)
    labels.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    shutil.copytree(FIXTURES / "cutouts", root / "work" / manifest_split.CUTOUT_DIR)
    manifest_split.run(cfg, **QUIET)
    stage.run(cfg, **QUIET)
    return root, cfg


@pytest.fixture
def clone(good, tmp_path):
    """A writable copy of the good run under a fresh config."""
    root, _ = good
    for sub in ("work", "staged"):
        shutil.copytree(root / sub, tmp_path / sub)
    return make_fx_cfg(tmp_path)


def _failed(cfg) -> list[str]:
    verdict = validate.run(cfg, **QUIET)
    ledger = common.read_ledger("validate", cfg)
    assert ledger["extra"]["failed"] == verdict["failed"]
    assert ledger["extra"]["passed"] == verdict["passed"]
    return verdict["failed"]


def _staged(cfg, split):
    return common.ledger_path("validate", cfg).parent.parent / "staged" / f"{split}.h5"


# ----------------------------------------------------------------------------- good run

def test_good_staging_passes_every_check(good):
    _, cfg = good
    verdict = validate.run(cfg, **QUIET)
    assert verdict["passed"] and verdict["failed"] == []
    names = [c["check"] for c in verdict["checks"]]
    assert names == ["files_present", "manifest_and_split_present", "schema", "targetids",
                     "split_sizes", "manifest_agreement", "detections_in_one_split",
                     "content", "labels"]
    ledger = common.read_ledger("validate", cfg)
    assert ledger["counts"]["staged_targets"] == ledger["counts"]["labelled_targets"]
    assert ledger["counts"]["log_flux_1_finite"] == ledger["counts"]["staged_targets"]
    assert sum(ledger["counts"][f"rows_{s}"] for s in manifest_split.SPLITS) == \
        ledger["counts"]["staged_targets"]
    assert set(ledger["extra"]["census_by_split"]) == set(manifest_split.SPLITS)
    assert validate.main(["--config", cfg["_config_path"]]) == 0


# ----------------------------------------------------------------------------- corruptions

def test_duplicate_targetid_across_splits(clone):
    with h5py.File(_staged(clone, "train")) as h:
        stolen = h["targetid"][0]
    with h5py.File(_staged(clone, "test"), "r+") as h:
        h["targetid"][0] = stolen
    failed = _failed(clone)
    # the moved target is also recorded as train in the manifest, so two checks fire
    assert failed == ["targetids", "manifest_agreement"]


def test_extra_dataset_breaks_the_schema(clone):
    with h5py.File(_staged(clone, "val"), "r+") as h:
        h.create_dataset("log_lx", data=np.zeros(h["targetid"].shape[0], np.float32))
    assert _failed(clone) == ["schema"]


def test_zero_image(clone):
    with h5py.File(_staged(clone, "train"), "r+") as h:
        h["image_flux"][0] = 0.0
    assert _failed(clone) == ["content"]


def test_nan_in_spectra_where_ivar_positive(clone):
    with h5py.File(_staged(clone, "train"), "r+") as h:
        row = h["spectra"][0]
        col = int(np.flatnonzero(h["spectra_ivar"][0] > 0)[0])
        row[col] = np.nan
        h["spectra"][0] = row
    assert _failed(clone) == ["content"]


def test_labels_missing_a_column(clone):
    work = common.ledger_path("validate", clone).parent.parent / "work"
    path = work / labels.OUTPUT
    pd.read_csv(path).drop(columns=["det_like_0"]).to_csv(path, index=False)
    assert _failed(clone) == ["labels"]
    assert validate.main(["--config", clone["_config_path"]]) == 1


def test_labels_missing_a_staged_target(clone):
    work = common.ledger_path("validate", clone).parent.parent / "work"
    path = work / labels.OUTPUT
    frame = pd.read_csv(path)
    frame.iloc[1:].to_csv(path, index=False)
    assert _failed(clone) == ["labels"]
    ledger = common.read_ledger("validate", clone)
    assert "without a label row" in [c for c in ledger["extra"]["checks"]
                                     if c["check"] == "labels"][0]["detail"]


def test_missing_staged_file_is_reported_without_crashing(clone):
    _staged(clone, "val").unlink()
    assert _failed(clone) == ["files_present"]


def test_flag_disagreeing_with_manifest(clone):
    with h5py.File(_staged(clone, "train"), "r+") as h:
        h["has_wise"][0] = not h["has_wise"][0]
    assert _failed(clone) == ["manifest_agreement"]
