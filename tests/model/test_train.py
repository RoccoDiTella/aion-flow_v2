"""M5: the training loop, its selection rule, and what a checkpoint restores."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from aionflow_model import train as train_module
from aionflow_model.config import TRAINING, load_run
from aionflow_model.data import Split, Standardizer, TokenDataset, loader
from aionflow_model.objective import Model
from aionflow_model.train import CHECKPOINT, CHOICES, HISTORY, run, validate, validation_masks
from tests.model.fake import FakeBackbone

QUIET = dict(log=lambda *a, **k: None)


def a_backbone():
    torch.manual_seed(0)
    return FakeBackbone(width=96, heads=4, depth=2)


@pytest.fixture(scope="module")
def trained(staged, tmp_path_factory):
    cfg, _, _ = staged
    out = tmp_path_factory.mktemp("run")
    result = run(cfg, "configs/marginals.yaml", out, chunk=8, max_epochs=2,
                 backbone=a_backbone(), **QUIET)
    return cfg, out, result


# ----------------------------------------------------------------------------- the loop

def test_two_epochs_on_the_fixtures_leave_a_run_directory(trained):
    _, out, result = trained
    history = json.loads((out / HISTORY).read_text())
    assert len(history) == 2 == len(result["history"])
    assert [h["epoch"] for h in history] == [0, 1]
    for entry in history:
        assert set(entry["val"]) == {"flux", "lx", "sfr", "mstar", "sfr_mstar"}
        assert entry["metric"] == pytest.approx(
            sum(entry["val"].values()) / len(entry["val"]))
        assert entry["train"] and entry["seconds"] > 0
    assert (out / "standardizer.json").is_file() and (out / CHECKPOINT).is_file()
    assert result["best"]["metric"] == min(h["metric"] for h in history)


def test_the_run_directory_records_what_the_paper_leaves_open(trained):
    _, out, _ = trained
    choices = json.loads((out / CHOICES).read_text())
    assert choices["run"] == "marginals"
    assert choices["heads"]["sfr_mstar"] == ["sfr", "mstar"]
    assert "mean over heads" in choices["validation_metric"]
    assert "drawn once" in choices["validation_masks"]
    assert choices["training"]["batch_size"] == 896 and choices["training"]["seed"] == 42
    assert choices["training"]["patience"] == 5 and choices["training"]["max_epochs"] == 40


def test_the_optimizer_is_the_papers_four_groups(standardizer):
    model = Model(a_backbone(), load_run("configs/marginals.yaml"), standardizer)
    groups = model.parameter_groups(TRAINING)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [
        (3e-4, 1e-4),     # readout MLPs
        (3e-4, 0.0),      # the CLS token, no decay
        (1e-3, 1e-4),     # the flows
        (3e-5, 0.1),      # the read adapters
    ]
    assert sum(len(g["params"]) for g in groups) == len(
        [p for p in model.parameters() if p.requires_grad])
    assert groups[1]["params"] == [model.probe.cls]


# ----------------------------------------------------------------------------- selection

def test_a_reloaded_checkpoint_reproduces_the_selection_value(trained, staged):
    cfg, out, result = trained
    _, work, staged_dir = staged
    checkpoint = torch.load(out / CHECKPOINT, weights_only=False)
    assert checkpoint["run"] == "marginals"
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    assert standardizer.as_dict() == Standardizer.read(out / "standardizer.json").as_dict()

    model = Model(a_backbone(), load_run("configs/marginals.yaml"), standardizer)
    model.load_state_dict(checkpoint["model"])
    split = Split(staged_dir, work, "val")
    batches = loader(TokenDataset(split, standardizer), 896, shuffle=False)
    metric, per_head = validate(model, batches, validation_masks(split, TRAINING.seed),
                                "cpu", 8)
    split.close()
    assert metric == pytest.approx(checkpoint["metric"], abs=1e-9)
    assert per_head == pytest.approx(checkpoint["per_head"], abs=1e-9)
    assert metric == pytest.approx(result["best"]["metric"], abs=1e-9)


def test_validation_masks_are_drawn_once_and_stay_put(staged):
    _, work, staged_dir = staged
    split = Split(staged_dir, work, "val")
    first = validation_masks(split, TRAINING.seed)
    assert set(first) == {int(t) for t in split.targetid}
    assert all(m.any() for m in first.values())
    again = validation_masks(split, TRAINING.seed)
    assert all(torch.equal(first[k], again[k]) for k in first)
    other = validation_masks(split, TRAINING.seed + 1)
    assert any(not torch.equal(first[k], other[k]) for k in first)
    split.close()


def test_patience_stops_a_run_that_stops_improving(staged, tmp_path, monkeypatch):
    cfg, _, _ = staged
    frozen = dataclasses.replace(TRAINING, max_epochs=8, patience=2,
                                 lr_readout=0.0, lr_flow=0.0, lr_adapter=0.0)
    monkeypatch.setattr(train_module, "TRAINING", frozen)
    result = run(cfg, "configs/rates.yaml", tmp_path / "stop", chunk=8,
                 backbone=a_backbone(), **QUIET)
    # nothing can improve at zero learning rate, so epoch 0 wins and patience ends it
    assert result["best"]["epoch"] == 0
    assert len(result["history"]) == 1 + frozen.patience < frozen.max_epochs
    assert len({round(h["metric"], 10) for h in result["history"]}) == 1


# ----------------------------------------------------------------------------- joint-only

def test_a_joint_only_run_carries_no_scalar_head(standardizer):
    model = Model(a_backbone(), load_run("configs/rates.yaml"), standardizer)
    assert list(model.flows) == ["rates"] == list(model.probe.readouts)
    assert model.flows["rates"].features == 2
    four = Model(a_backbone(), load_run("configs/joint4.yaml"), standardizer)
    assert list(four.flows) == ["joint4"] and four.flows["joint4"].features == 4
