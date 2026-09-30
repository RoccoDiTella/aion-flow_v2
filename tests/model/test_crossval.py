"""Cross-validation: the fold assignment, the fold views, and one fold end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aionflow_data.manifest_split import assign
from aionflow_model import crossval
from aionflow_model.crossval import FOLDS, CrossvalError, blocks, fold_splits, roles
from aionflow_model.data import Split
from tests.model.test_train import a_backbone

QUIET = dict(log=lambda *a, **k: None)


@pytest.mark.parametrize("n", [29, 1000, 129356])
def test_the_blocks_are_the_existing_split_cut_finer(n):
    """Same sort, same seeded shuffle as manifest_split: blocks 0-7 are its training
    set, 8 its validation set and 9 its test set, whatever the sample size."""
    tids = np.random.default_rng(n).choice(10**12, size=n, replace=False)
    names = assign(tids, 42, [0.8, 0.1, 0.1])
    b = blocks(tids, 42)
    assert ((b < 8) == (names == "train")).all()
    assert ((b == 8) == (names == "val")).all()
    assert ((b == 9) == (names == "test")).all()


def test_every_source_is_tested_exactly_once_across_the_folds():
    b = blocks(np.arange(1, 1001), 42)
    tested = np.zeros(b.size, int)
    for fold in range(FOLDS):
        r = roles(b, fold)
        assert not (r["train"] & r["val"]).any() and not (r["train"] & r["test"]).any()
        assert not (r["val"] & r["test"]).any()
        assert (r["train"] | r["val"] | r["test"]).all()
        assert r["test"].mean() == pytest.approx(0.2, abs=0.01)
        assert r["val"].mean() == pytest.approx(0.1, abs=0.01)
        tested += r["test"]
    assert (tested == 1).all()
    with pytest.raises(CrossvalError):
        roles(b, FOLDS)


def test_a_fold_view_reads_the_right_rows_and_withholding_stays_inside_it(staged):
    _, work, staged_dir = staged
    parts = {name: Split(staged_dir, work, name) for name in ("train", "val", "test")}
    try:
        views, check = fold_splits(parts, 1, 42)
        assert check["blocks_match_existing_split"]
        every = np.concatenate([p.targetid for p in parts.values()])
        seen = np.concatenate([v.targetid for v in views.values()])
        assert np.sort(seen).tolist() == np.sort(every).tolist()
        lookup = {int(t): (p, i) for p in parts.values() for i, t in enumerate(p.targetid)}
        view = views["test"]
        for row in range(view.n):
            part, local = lookup[int(view.targetid[row])]
            got, want = view.tokens(row), part.tokens(local)
            assert all(np.array_equal(got[k], want[k]) for k in want)
            assert view.redshift[row] == part.redshift[local]
        before = [p.y_ok.copy() for p in parts.values()]
        mask = np.zeros(views["train"].n, bool)
        mask[:2] = True
        views["train"].withhold(mask)
        assert not views["train"].y_ok[:2].any()
        assert all(np.array_equal(b, p.y_ok) for b, p in zip(before, parts.values()))
    finally:
        for p in parts.values():
            p.close()


def test_one_fold_trains_and_writes_rho_for_its_held_out_sources(staged, tmp_path):
    cfg, work, staged_dir = staged
    crossval.run(cfg, 0, tmp_path, chunk=8, rho_chunk=8, draws=16, max_epochs=1,
                 backbone=a_backbone(), **QUIET)
    rho = pd.read_csv(tmp_path / crossval.RHO)
    parts = {name: Split(staged_dir, work, name) for name in ("train", "val", "test")}
    try:
        views, _ = fold_splits(parts, 0, 42)
        assert sorted(rho.targetid) == sorted(views["test"].targetid.tolist())
    finally:
        for p in parts.values():
            p.close()
    assert (rho.fold == 0).all() and np.isfinite(rho.rho).any()
    choices = json.loads((tmp_path / "choices.json").read_text())
    assert choices["crossval_fold"] == 0 and choices["blocks_match_existing_split"]
    assert choices["training"]["lr_flow"] == pytest.approx(5e-4)      # joint4's half rate
    assert "Mineo" in choices["sf_dominated_rule"]


def test_a_trained_fold_can_redo_only_its_rho(staged, tmp_path):
    """The draws are the memory-hungry step; a fold that trained should not retrain
    because they ran out of room."""
    cfg, _, _ = staged
    crossval.run(cfg, 3, tmp_path, chunk=8, rho_chunk=8, draws=16, max_epochs=1,
                 backbone=a_backbone(), **QUIET)
    (tmp_path / crossval.RHO).unlink()
    before = (tmp_path / "best.pt").stat().st_mtime_ns
    crossval.run(cfg, 3, tmp_path, rho_chunk=8, draws=16, rho_only=True,
                 backbone=a_backbone(), **QUIET)
    assert (tmp_path / crossval.RHO).is_file()
    assert (tmp_path / "best.pt").stat().st_mtime_ns == before, "it must not retrain"
    with pytest.raises(CrossvalError, match="needs a trained fold"):
        crossval.run(cfg, 3, tmp_path / "empty", rho_only=True, backbone=a_backbone(), **QUIET)
