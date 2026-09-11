"""M6: information gain against the KDE prior, R2, coverage, and the per-source dump."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aionflow_model.config import Head, load_run
from aionflow_model.data import RATE_TARGETS, SCALAR_TARGETS, Split, log_plug_in_rate
from aionflow_model.evaluate import (
    COVERAGE,
    PER_SOURCE,
    RESULTS,
    EvaluateError,
    PriorHead,
    common_subsample,
    coverage,
    evaluate,
    fit_prior,
    head_values,
    r_squared,
)
from aionflow_model.evaluate import run as evaluate_run
from aionflow_model.objective import SUBSET_NAMES, Model
from tests.model.fake import FakeBackbone

QUIET = dict(log=lambda *a, **k: None)


def a_backbone():
    torch.manual_seed(0)
    return FakeBackbone(width=96, heads=4, depth=2)


class PriorAsFlow(PriorHead):
    """The KDE standing in for a trained flow, to show the two terms of the gain
    meet the same quadrature. Its draws are unused by what this exercises."""

    def sample(self, context, draws):
        return torch.zeros(context.shape[0], draws, self.kde.features, dtype=context.dtype)


# ----------------------------------------------------------------------------- the prior

def test_the_prior_is_fitted_on_the_complete_training_rows_only(splits, standardizer):
    train = splits["train"]
    head = Head("sfr_mstar", ("sfr", "mstar"))
    kde = fit_prior(head, train, standardizer)
    complete = train.y_ok[:, [SCALAR_TARGETS.index("sfr"), SCALAR_TARGETS.index("mstar")]]
    assert kde.n == int(complete.all(axis=1).sum()) < train.n
    assert kde.features == 2 and kde.scott == pytest.approx(kde.n ** (-1 / 6))


def test_a_rate_prior_is_over_plug_in_rates(splits, standardizer):
    train = splits["train"]
    head = Head("rates", RATE_TARGETS)
    values = head_values(head, train, standardizer)
    for j, target in enumerate(RATE_TARGETS):
        want = standardizer.encode(target, log_plug_in_rate(
            train.counts[:, j], train.bkg[:, j], train.expo[:, j]))
        assert np.allclose(values[:, j], want)
    assert fit_prior(head, train, standardizer).features == 2


def test_a_head_with_too_little_to_fit_is_refused(splits, standardizer):
    train = splits["train"]
    empty = Split.__new__(Split)
    for attribute in ("y_raw", "y_ok", "counts", "bkg", "expo", "rate_ok"):
        setattr(empty, attribute, getattr(train, attribute)[:1])
    empty.y_ok = np.zeros_like(empty.y_ok)
    with pytest.raises(EvaluateError, match="complete training rows"):
        fit_prior(Head("sfr", ("sfr",)), empty, standardizer)


# ----------------------------------------------------------------------------- subsample

def test_the_common_subsample_is_all_four_modalities_and_every_target(splits):
    test = splits["test"]
    for head in load_run("configs/marginals.yaml").heads:
        keep = common_subsample(head, test)
        assert keep.dtype == bool and keep.shape == (test.n,)
        assert not keep[~test.present.all(axis=1)].any()
        for target in head.targets:
            assert test.y_ok[keep, SCALAR_TARGETS.index(target)].all()


# ----------------------------------------------------------------------------- metrics

def test_r_squared_is_one_for_a_perfect_prediction_and_zero_for_the_mean(splits, standardizer):
    test = splits["test"]
    head = Head("lx", ("lx",))
    keep = common_subsample(head, test)
    truth = test.y_raw[:, SCALAR_TARGETS.index("lx")]
    perfect = standardizer.encode("lx", truth)[:, None]
    assert r_squared(head, perfect, test, standardizer, keep)["lx"] == pytest.approx(1.0)
    flat = np.full_like(perfect, standardizer.encode("lx", truth[keep].mean()))
    assert r_squared(head, flat, test, standardizer, keep)["lx"] == pytest.approx(0.0, abs=1e-9)


def test_coverage_counts_the_sources_inside_the_central_interval(splits, standardizer):
    test = splits["test"]
    head = Head("lx", ("lx",))
    keep = common_subsample(head, test)
    truth = standardizer.encode("lx", test.y_raw[:, SCALAR_TARGETS.index("lx")])
    wide = truth[:, None, None] + np.linspace(-50, 50, 4001)[None, :, None]
    assert all(v == pytest.approx(1.0) for v in
               coverage(head, wide, test, standardizer, keep)["lx"].values())
    far = np.full((test.n, 101, 1), 1e3)
    assert all(v == 0.0 for v in coverage(head, far, test, standardizer, keep)["lx"].values())
    assert tuple(f"{c:.2f}" for c in COVERAGE) == ("0.68", "0.90", "0.95")


# ----------------------------------------------------------------------------- the pass

@pytest.fixture(scope="module")
def scored(splits, standardizer):
    """Evaluated on the training split, which is the only fixture split with enough
    rows for the tables to have anything in them."""
    model = Model(a_backbone(), load_run("configs/marginals.yaml"), standardizer)
    model.eval()
    return model, evaluate(model, {"train": splits["train"], "test": splits["train"]},
                           draws=16, chunk=8, **QUIET)


def test_every_head_gets_fifteen_rows_on_one_fixed_subsample(scored, splits):
    model, (results, frame) = scored
    heads = [head.name for head in model.run.heads]
    assert len(results["rows"]) == 15 * len(heads)
    for name in heads:
        rows = [r for r in results["rows"] if r["head"] == name]
        assert [r["inputs"] for r in rows] == list(SUBSET_NAMES)
        assert len({r["n"] for r in rows}) == 1                    # fixed across the 15
        assert rows[0]["n"] == results["common_subsample"][name]
        assert rows[0]["n"] == int(frame[f"common_{name}"].sum())
    assert set(results["coverage"]) == set(heads)
    assert set(results["coverage"]["lx"]["lx"]) == {"0.68", "0.90", "0.95"}


def test_the_dump_carries_every_source_under_every_combination(scored, splits):
    model, (results, frame) = scored
    train = splits["train"]
    assert len(frame) == train.n
    assert np.array_equal(frame["targetid"].to_numpy(), train.targetid)
    assert np.array_equal(frame["redshift"].to_numpy(), train.redshift)
    for head in model.run.heads:
        assert f"prior_{head.name}" in frame
        for name in SUBSET_NAMES:
            assert f"ll_{head.name}_{name}" in frame
        keep = frame[f"common_{head.name}"].to_numpy()
        row = next(r for r in results["rows"]
                   if r["head"] == head.name and r["inputs"] == "ZSIW")
        gain = (frame.loc[keep, f"ll_{head.name}_ZSIW"]
                - frame.loc[keep, f"prior_{head.name}"]).mean()
        assert row["information_gain"] == pytest.approx(gain)


def test_a_model_equal_to_the_prior_gains_nothing(splits, standardizer):
    """Both terms of the gain are the same likelihood, so a model that is the prior
    scores exactly zero. That is what makes "for rates, both terms are the counts
    likelihood of Eq. 1" a property of the code rather than a claim about it."""
    model = Model(a_backbone(), load_run("configs/rates.yaml"), standardizer)
    model.eval()
    head = model.run.heads[0]
    kde = fit_prior(head, splits["train"], standardizer)
    del model.flows                       # out of _modules, so a plain mapping can stand in
    model.flows = {head.name: PriorAsFlow(kde)}
    results, frame = evaluate(model, {"train": splits["train"], "test": splits["train"]},
                              draws=4, chunk=8, **QUIET)
    for row in results["rows"]:
        assert row["information_gain"] == pytest.approx(0.0, abs=1e-9), row["inputs"]
    assert np.allclose(frame[f"ll_{head.name}_ZSIW"], frame[f"prior_{head.name}"])


# ----------------------------------------------------------------------------- cli

def test_a_missing_checkpoint_is_reported(staged, tmp_path):
    cfg, _, _ = staged
    with pytest.raises(EvaluateError, match="no checkpoint"):
        evaluate_run(cfg, tmp_path, **QUIET)


def test_results_and_dump_are_written_next_to_the_checkpoint(staged, tmp_path):
    from aionflow_model.train import run as train_run
    cfg, _, _ = staged
    out = tmp_path / "run"
    train_run(cfg, "configs/rates.yaml", out, chunk=8, max_epochs=1,
              backbone=a_backbone(), **QUIET)
    results = evaluate_run(cfg, out, chunk=8, draws=8, backbone=a_backbone(), **QUIET)
    assert json.loads((out / RESULTS).read_text())["rows"] == results["rows"]
    assert (out / PER_SOURCE).is_file() and len(results["rows"]) == 15
