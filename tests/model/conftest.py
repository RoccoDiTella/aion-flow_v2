"""A staged fixture dataset: Phase 1 run end to end on the committed synthetic inputs."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

from aionflow_data import (
    common,
    crossmatch,
    fetch_spectra,
    labels,
    line_features,
    manifest_split,
    stage,
)
from aionflow_model.data import ALL_TOKEN_KEYS, TOKEN_SIZES, TOKENS_FILE, Split, Standardizer
from aionflow_model.tokenize import CODEC_REPO
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a, **k: None)
VOCABULARY = 1024


def write_fake_tokens(staged_dir, name: str, targetid: np.ndarray) -> None:
    """Stand-in token ids, so the suite exercises the loader without AION's weights."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    with h5py.File(staged_dir / TOKENS_FILE.format(split=name), "w") as h:
        h.create_dataset("targetid", data=targetid)
        for key in ALL_TOKEN_KEYS:
            h.create_dataset(key, data=rng.integers(0, VOCABULARY,
                                                    size=(targetid.size, TOKEN_SIZES[key]),
                                                    dtype=np.int32))
        h.attrs["codec_repo"] = CODEC_REPO
        h.attrs["split"] = name


@pytest.fixture(scope="session")
def staged(tmp_path_factory):
    """(cfg, work, staged): the pipeline config and the directories it wrote into."""
    cfg = make_fx_cfg(tmp_path_factory.mktemp("model"))
    crossmatch.run(cfg, **QUIET)
    labels.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    work = common.ledger_path("labels", cfg).parent.parent / "work"
    shutil.copytree(FIXTURES / "cutouts", work / manifest_split.CUTOUT_DIR)
    manifest_split.run(cfg, **QUIET)
    stage.run(cfg, **QUIET)
    line_features.run(cfg, nproc=1, **QUIET)
    staged_dir = common.ledger_path("stage", cfg).parent.parent / "staged"
    for name in ("train", "val", "test"):
        with h5py.File(staged_dir / f"{name}.h5", "r") as h:
            write_fake_tokens(staged_dir, name, h["targetid"][:])
    return cfg, work, staged_dir


@pytest.fixture(scope="session")
def splits(staged):
    _, work, staged_dir = staged
    out = {name: Split(staged_dir, work, name) for name in ("train", "val", "test")}
    yield out
    for split in out.values():
        split.close()


@pytest.fixture(scope="session")
def standardizer(splits) -> Standardizer:
    return Standardizer.fit(splits["train"])
