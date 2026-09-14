"""Appendix B: the pooling arms and the random-encoder control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from aionflow_model.ablations import (
    ARMS,
    AblationError,
    AttentivePool,
    MeanPool,
    build,
    frozen_tokens,
    modality_index,
    randomize_encoder,
)
from aionflow_model.ablations import run as ablation_run
from aionflow_model.config import TRAINING, load_run
from aionflow_model.data import MODALITIES, TOKEN_KEYS, Standardizer
from aionflow_model.objective import Model
from tests.model.fake import FakeBackbone, ShapedBackbone
from tests.model.test_encoder import a_batch

QUIET = dict(log=lambda *a, **k: None)
SMALL = 96


def a_backbone(width=SMALL):
    torch.manual_seed(0)
    return FakeBackbone(width=width, heads=4, depth=2)


@pytest.fixture
def pooling():
    return load_run("configs/pooling.yaml")


# ----------------------------------------------------------------------------- counts

def test_the_three_arms_have_the_papers_parameter_counts(standardizer, pooling):
    """6.8M for the CLS read, 5.6M for the probe and 3.3M for the mean pool."""
    backbone = ShapedBackbone(depth=12)
    counts = {arm: sum(p.numel() for p in build(arm, backbone, pooling, standardizer)
                       .parameters() if p.requires_grad) for arm in ARMS}
    assert counts["cls"] == 6_795_888          # 6.8M
    assert counts["attentive"] == 5_625_456    # 5.6M
    assert counts["mean"] == 3_256_176         # 3.3M
    assert [round(c / 1e6, 1) for c in (counts["cls"], counts["attentive"],
                                        counts["mean"])] == [6.8, 5.6, 3.3]


def test_the_attentive_probe_is_bare(standardizer, pooling):
    """"A bare attentive probe, one learned query with single-head cross-attention":
    no self-attention among queries, no feed-forward, no second layer. The count is
    what pins it - an extra block or a dropped output projection both miss 5.6M."""
    probe = AttentivePool(ShapedBackbone(depth=12), pooling, standardizer)
    assert probe.query.shape == (768,)                       # one query
    assert probe.presence.shape == (len(MODALITIES), 768)
    assert probe.scale.shape == probe.shift.shape == (len(MODALITIES), 768)
    projections = [m for name, m in probe.named_modules()
                   if isinstance(m, nn.Linear) and not name.startswith(("readouts", "flows"))]
    assert len(projections) == 4                             # Q, K, V and the output
    assert all(m.bias is None and m.in_features == m.out_features == 768
               for m in projections)
    pooler = sum(p.numel() for name, p in probe.named_parameters()
                 if not name.startswith(("readouts", "flows", "backbone")))
    assert pooler == 4 * 768 * 768 + 768 + 3 * len(MODALITIES) * 768 == 2_369_280


def test_build_rejects_an_arm_the_paper_does_not_have(standardizer, pooling):
    assert isinstance(build("cls", a_backbone(), pooling, standardizer), Model)
    with pytest.raises(AblationError, match="unknown arm 'cosine'"):
        build("cosine", a_backbone(), pooling, standardizer)
    assert set(ARMS) == {"cls", "attentive", "mean"}       # no four-token, no cosine


def test_the_main_path_never_imports_the_ablations():
    """The package trains one architecture; these arms are reachable only on purpose."""
    offenders = [p.name for p in Path("aionflow_model").glob("*.py")
                 if p.name != "ablations.py" and "ablations" in p.read_text()]
    assert offenders == []


# ----------------------------------------------------------------------------- pooling

def test_the_tokens_carry_their_modality_and_a_pad_marker():
    """Modality dropout is per source, so rows differ in length and the short ones are
    padded; the pad slots must be marked, not silently attributed to a modality."""
    backbone = a_backbone()
    batch = a_batch(rows=3)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[0, MODALITIES.index("I")] = False        # one short row, two full ones
    tokens, valid, modality = frozen_tokens(backbone, batch, mask)
    assert tokens.shape[0] == 3 and tokens.shape[-1] == SMALL
    assert valid.shape == modality.shape == tokens.shape[:2]

    image = sum(n for _, n in TOKEN_KEYS["I"])
    assert int(valid[0].sum()) == int(valid[1].sum()) - image
    assert (~valid[0]).any() and valid[1].all()   # the short row is the padded one
    assert modality[~valid].eq(-1).all()          # and its pad slots claim no modality

    for row in (0, 1):
        for m, name in enumerate(MODALITIES):
            want = 0 if (row == 0 and name == "I") else sum(n for _, n in TOKEN_KEYS[name])
            assert int((modality[row] == m).sum()) == want, (row, name)
    raw = torch.full((2, 5), 1000, dtype=torch.long)
    assert modality_index(backbone, raw).unique().tolist() == [0]    # tok_z -> Z


def test_a_masked_token_cannot_change_either_pool(standardizer, pooling):
    torch.manual_seed(0)
    for cls in (MeanPool, AttentivePool):
        pool = cls(a_backbone(), pooling, standardizer).eval()
        rows, n = 3, 12
        tokens = torch.randn(rows, n, SMALL)
        valid = torch.ones(rows, n, dtype=torch.bool)
        valid[:, -4:] = False
        modality = torch.zeros(rows, n, dtype=torch.long)
        modality[:, -4:] = -1
        present = torch.ones(rows, 4, dtype=torch.bool)
        before = pool.pool(tokens, valid, modality, present)
        tokens[:, -4:] = 1e3                                # garbage in the dead slots
        after = pool.pool(tokens, valid, modality, present)
        assert torch.allclose(before, after, atol=1e-5), cls.__name__


def test_the_affine_is_per_modality_and_the_query_knows_what_is_present(standardizer,
                                                                       pooling):
    torch.manual_seed(0)
    pool = AttentivePool(a_backbone(), pooling, standardizer).eval()
    rows, n = 2, 8
    tokens = torch.randn(rows, n, SMALL)
    valid = torch.ones(rows, n, dtype=torch.bool)
    modality = torch.zeros(rows, n, dtype=torch.long)       # every token is modality 0
    present = torch.ones(rows, 4, dtype=torch.bool)
    base = pool.pool(tokens, valid, modality, present)
    with torch.no_grad():
        pool.scale[1] += 5.0                                # a modality that is absent
    assert torch.allclose(base, pool.pool(tokens, valid, modality, present), atol=1e-6)
    with torch.no_grad():
        pool.scale[0] += 5.0                                # the one that is there
    assert not torch.allclose(base, pool.pool(tokens, valid, modality, present))
    # the presence embedding moves the query when the conditioning set changes
    with torch.no_grad():
        pool.presence.normal_()
    fewer = present.clone()
    fewer[:, 3] = False
    assert not torch.allclose(pool.pool(tokens, valid, modality, present),
                              pool.pool(tokens, valid, modality, fewer))


# ----------------------------------------------------------------------------- control

def test_the_random_encoder_keeps_the_architecture_and_loses_the_pretraining():
    trained, control = a_backbone(), a_backbone()
    batch, mask = a_batch(rows=3), torch.ones(3, 4, dtype=torch.bool)
    before, _, _ = frozen_tokens(control, batch, mask)
    reset = randomize_encoder(control)
    assert reset > 0
    after, _, _ = frozen_tokens(control, batch, mask)
    assert after.shape == before.shape                      # identical architecture
    assert not torch.allclose(after, before)                # and nothing it had learned
    assert not control.training
    assert not any(p.requires_grad for p in control.parameters())
    assert sum(p.numel() for p in control.parameters()) == sum(p.numel()
                                                               for p in trained.parameters())
    again = a_backbone()
    randomize_encoder(again, seed=TRAINING.seed)
    once = a_backbone()
    randomize_encoder(once, seed=TRAINING.seed)
    a, _, _ = frozen_tokens(again, batch, mask)
    b, _, _ = frozen_tokens(once, batch, mask)
    assert torch.allclose(a, b)                             # seeded, so reproducible


# ----------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_trains_on_the_fixtures(staged, tmp_path, arm):
    cfg, _, _ = staged
    result = ablation_run(cfg, tmp_path / arm, arm=arm, chunk=8, max_epochs=1,
                          backbone=a_backbone(), **QUIET)
    assert result["best"]["epoch"] == 0
    choices = json.loads((tmp_path / arm / "choices.json").read_text())
    assert choices["ablation_arm"] == arm and choices["run"] == "pooling"
    assert choices["random_encoder"] is False
    assert set(choices["heads"]) == {"flux", "lx"}          # "the two heads, flux and log LX"
    assert "Appendix B only" in choices["note"]


def test_the_control_sees_every_modality(staged, tmp_path):
    cfg, _, _ = staged
    ablation_run(cfg, tmp_path / "control", arm="mean", random_encoder=True, chunk=8,
                 max_epochs=1, backbone=a_backbone(), **QUIET)
    choices = json.loads((tmp_path / "control" / "choices.json").read_text())
    assert choices["random_encoder"] is True and choices["modules_reinitialized"] > 0
    # the control is the mean pool, so it trains the mean pool's parameters and no more
    assert choices["ablation_arm"] == "mean"
    assert choices["trained_parameters"] == sum(
        p.numel() for p in MeanPool(a_backbone(), load_run("configs/pooling.yaml"),
                                    Standardizer.read(tmp_path / "control"
                                                      / "standardizer.json")).parameters()
        if p.requires_grad)
