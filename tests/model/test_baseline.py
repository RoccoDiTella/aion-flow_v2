"""M7: the emission-line baseline, scored the same way as the probe."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from aionflow_model.baseline import (
    LINES,
    BaselineError,
    BaselineModel,
    LineDataset,
    line_context,
    line_scaling,
    read_lines,
)
from aionflow_model.baseline import run as baseline_run
from aionflow_model.config import load_run
from aionflow_model.encoder import readout
from aionflow_model.evaluate import PER_SOURCE, RESULTS
from aionflow_model.evaluate import run as evaluate_run
from aionflow_model.flows import CONTEXT, FlowHead
from aionflow_model.objective import SUBSET_NAMES
from aionflow_model.train import run as train_run
from tests.model.fake import FakeBackbone

QUIET = dict(log=lambda *a, **k: None)


def a_backbone():
    torch.manual_seed(0)
    return FakeBackbone(width=96, heads=4, depth=2)


# ----------------------------------------------------------------------------- the lines

def test_the_four_lines_are_the_papers_and_come_out_aligned(staged, splits):
    _, work, _ = staged
    assert LINES == ("oiii_5007", "nev_3426", "halpha", "hbeta")
    split = splits["train"]
    fluxes = read_lines(work, split.targetid)
    assert fluxes.shape == (split.n, 4) and np.isfinite(fluxes).all()
    frame = pd.read_csv(work / "line_features.csv").set_index("targetid")
    for j, line in enumerate(LINES):
        assert np.allclose(fluxes[:, j], frame.loc[split.targetid, f"{line}_flux"])
    # a line outside coverage or a failed fit is a zero, not a dropped source
    assert (fluxes == 0).any()


def test_missing_line_features_are_reported(tmp_path, splits):
    with pytest.raises(BaselineError, match="run aionflow_data.line_features"):
        read_lines(tmp_path, splits["train"].targetid)


def test_absent_targets_are_reported(staged, splits, tmp_path):
    _, work, _ = staged
    frame = pd.read_csv(work / "line_features.csv")
    frame.iloc[1:].to_csv(tmp_path / "line_features.csv", index=False)
    with pytest.raises(BaselineError, match="no line features"):
        read_lines(tmp_path, splits["train"].targetid)


def test_the_scaling_is_the_training_splits(staged, splits):
    _, work, _ = staged
    fluxes = read_lines(work, splits["train"].targetid)
    mean, scale = line_scaling(fluxes)
    assert np.allclose(mean, fluxes.mean(0)) and np.allclose(scale, fluxes.std(0))
    with pytest.raises(BaselineError, match="no spread"):
        line_scaling(np.ones((5, 4)))


# ----------------------------------------------------------------------------- the model

def test_only_the_context_encoder_differs():
    encoder, probe_readout = line_context(), readout()
    linears = [m for m in encoder if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(4, 512), (512, CONTEXT)]
    # the readout's shape, minus its leading LayerNorm: the fluxes arrive standardized
    assert [type(m).__name__ for m in encoder] == [
        type(m).__name__ for m in list(probe_readout)[1:]]
    assert isinstance(list(probe_readout)[0], nn.LayerNorm)
    assert encoder(torch.randn(3, 4)).shape == (3, CONTEXT)


def test_the_flow_is_identical_to_the_models_heads(standardizer):
    model = BaselineModel(load_run("configs/baseline.yaml"), standardizer)
    assert list(model.flows) == ["flux", "lx", "sfr", "mstar"]
    for name, flow in model.flows.items():
        assert sum(p.numel() for p in flow.parameters()) == sum(
            p.numel() for p in FlowHead(1).parameters()) == 1_099_960, name


def test_the_baseline_reads_no_modality(standardizer):
    model = BaselineModel(load_run("configs/baseline.yaml"), standardizer).eval()
    batch = {"lines": torch.randn(5, 4)}
    full = model.contexts(batch, torch.ones(5, 4, dtype=torch.bool))
    none = model.contexts(batch, torch.zeros(5, 4, dtype=torch.bool))
    assert all(torch.equal(full[k], none[k]) for k in full)


def test_the_optimizer_has_two_groups(standardizer):
    from aionflow_model.config import TRAINING
    model = BaselineModel(load_run("configs/baseline.yaml"), standardizer)
    groups = model.parameter_groups(TRAINING)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [(3e-4, 1e-4), (1e-3, 1e-4)]
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_the_dataset_carries_the_targets_and_the_scaled_fluxes(staged, splits, standardizer):
    _, work, _ = staged
    split = splits["train"]
    mean, scale = line_scaling(read_lines(work, split.targetid))
    dataset = LineDataset(split, standardizer, work, mean, scale)
    item = dataset[0]
    assert set(item) == {"targetid", "lines", "present", "y", "y_ok", "counts", "bkg",
                         "expo", "rate_ok"}
    assert item["lines"].shape == (4,)
    assert np.allclose(dataset.lines.mean(0), 0.0, atol=1e-5)
    assert np.allclose(dataset.lines.std(0), 1.0, atol=1e-4)


# ----------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def both(staged, tmp_path_factory):
    """A baseline and a probe run, trained and scored on the same fixture split."""
    cfg, _, _ = staged
    root = tmp_path_factory.mktemp("compare")
    baseline_run(cfg, root / "baseline", chunk=8, max_epochs=1, **QUIET)
    evaluate_run(cfg, root / "baseline", chunk=8, draws=8, baseline=True, **QUIET)
    train_run(cfg, "configs/marginals.yaml", root / "probe", chunk=8, max_epochs=1,
              backbone=a_backbone(), **QUIET)
    evaluate_run(cfg, root / "probe", chunk=8, draws=8, backbone=a_backbone(), **QUIET)
    return root


def test_the_baseline_trains_and_records_its_context_encoder(both):
    choices = json.loads((both / "baseline" / "choices.json").read_text())
    assert choices["run"] == "baseline" and choices["lines"] == list(LINES)
    assert "no leading LayerNorm" in choices["context_encoder"]
    assert choices["training"]["seed"] == 42
    scaling = json.loads((both / "baseline" / "lines.json").read_text())
    assert scaling["lines"] == list(LINES) and len(scaling["mean"]) == 4


def test_every_combination_scores_the_same_because_it_reads_none(both):
    results = json.loads((both / "baseline" / RESULTS).read_text())
    assert len(results["rows"]) == 15 * 4
    for head in ("flux", "lx", "sfr", "mstar"):
        gains = {round(r["information_gain"], 12) for r in results["rows"]
                 if r["head"] == head}
        assert len(gains) == 1, head
        assert [r["inputs"] for r in results["rows"] if r["head"] == head] == list(SUBSET_NAMES)


def test_the_baseline_meets_the_same_prior_on_the_same_subsample(both):
    """Figure 1 puts the two side by side, so they must be scored identically."""
    baseline = pd.read_csv(both / "baseline" / PER_SOURCE)
    probe = pd.read_csv(both / "probe" / PER_SOURCE)
    assert np.array_equal(baseline["targetid"], probe["targetid"])
    for head in ("flux", "lx", "sfr", "mstar"):
        assert np.allclose(baseline[f"prior_{head}"], probe[f"prior_{head}"])
        assert np.array_equal(baseline[f"common_{head}"], probe[f"common_{head}"])
    sizes = json.loads((both / "baseline" / RESULTS).read_text())["common_subsample"]
    assert sizes == {k: v for k, v in
                     json.loads((both / "probe" / RESULTS).read_text())
                     ["common_subsample"].items() if k in sizes}
