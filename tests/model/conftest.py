"""A staged fixture dataset: Phase 1 run end to end on the committed synthetic inputs."""

from __future__ import annotations

import shutil

import pytest

from aionflow_data import common, crossmatch, fetch_spectra, labels, manifest_split, stage
from aionflow_model.data import Split, Standardizer
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a, **k: None)


@pytest.fixture(scope="session")
def staged(tmp_path_factory):
    """(work, staged) directories holding labels.csv, manifest.csv and the split files."""
    cfg = make_fx_cfg(tmp_path_factory.mktemp("model"))
    crossmatch.run(cfg, **QUIET)
    labels.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    work = common.ledger_path("labels", cfg).parent.parent / "work"
    shutil.copytree(FIXTURES / "cutouts", work / manifest_split.CUTOUT_DIR)
    manifest_split.run(cfg, **QUIET)
    stage.run(cfg, **QUIET)
    return work, common.ledger_path("stage", cfg).parent.parent / "staged"


@pytest.fixture(scope="session")
def splits(staged):
    work, staged_dir = staged
    out = {name: Split(staged_dir, work, name) for name in ("train", "val", "test")}
    yield out
    for split in out.values():
        split.close()


@pytest.fixture(scope="session")
def standardizer(splits) -> Standardizer:
    return Standardizer.fit(splits["train"])
