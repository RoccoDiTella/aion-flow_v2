"""Appendix B: the pooling arms and the random-encoder control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from aionflow_model.ablations import (
    ARMS,
    LR_BACKBONE,
    POOLING_ARMS,
    WD_BACKBONE,
    AblationError,
    AttentivePool,
    Finetuned,
    MeanPool,
    build,
    default_recipe,
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
    assert set(ARMS) == {"cls", "attentive", "mean", "finetune"}   # no four-token, no cosine


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

@pytest.mark.parametrize("arm", POOLING_ARMS)
def test_each_pooling_arm_trains_on_the_fixtures(staged, tmp_path, arm):
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


# ---------------------------------------------------------------------------- finetune

def a_pair(standardizer, recipe):
    """The reported model and the finetuning arm at identical weights.

    Separate backbones on purpose: `Probe.__init__` freezes the module it is given
    and `FinetunedProbe.__init__` unfreezes it, so two models sharing one backbone
    would have the later one decide for both.
    """
    frozen = Model(a_backbone(), recipe, standardizer).eval()
    tuned = Finetuned(a_backbone(), recipe, standardizer).eval()
    tuned.load_state_dict(frozen.state_dict())
    return frozen, tuned


def a_mask(rows=3):
    return torch.ones(rows, len(MODALITIES), dtype=torch.bool)


def test_the_finetune_reads_exactly_what_the_frozen_probe_reads(standardizer, pooling):
    """`FinetunedProbe.context` is `Probe.context` with the three no_grad blocks gone,
    so at identical weights the two must agree to the bit. This is what stops the copy
    drifting if the reported model ever changes, and it says the recomputation and the
    removed no_grad are not quietly arithmetic."""
    torch.manual_seed(0)
    frozen, tuned = a_pair(standardizer, pooling)
    batch = a_batch(rows=3)
    with torch.no_grad():
        want = frozen.probe.context(batch, a_mask())
        got = tuned.probe.context(batch, a_mask())
    assert torch.equal(want, got)


def test_the_encoder_gets_a_gradient_here_and_nowhere_else(standardizer, pooling):
    torch.manual_seed(0)
    frozen, tuned = a_pair(standardizer, pooling)
    batch, mask = a_batch(rows=2), a_mask(2)

    tuned.probe.context(batch, mask).square().sum().backward()
    grads = [p.grad for p in tuned.probe.backbone.parameters()]
    assert grads and all(g is not None for g in grads), "every encoder weight moved"
    assert sum(float(g.abs().sum()) for g in grads) > 0

    frozen.probe.context(batch, mask).square().sum().backward()
    assert all(p.grad is None for p in frozen.probe.backbone.parameters())
    assert frozen.probe.cls.grad is not None, "the read still trains"


def test_recomputing_the_blocks_does_not_change_the_gradient(standardizer, pooling):
    """Gradient checkpointing is what makes the arm affordable, and it is only ever
    worth having if it is exactly the same gradient."""
    torch.manual_seed(0)
    _, stored = a_pair(standardizer, pooling)
    _, recomputed = a_pair(standardizer, pooling)
    recomputed.load_state_dict(stored.state_dict())
    stored.probe.checkpointing = False
    assert recomputed.probe.checkpointing is True

    batch, mask = a_batch(rows=2), a_mask(2)
    for model in (stored, recomputed):
        model.probe.context(batch, mask).square().sum().backward()
    other = dict(recomputed.named_parameters())
    compared = 0
    for name, a in stored.named_parameters():
        b = other[name]
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), name
            compared += 1
    # context() does not reach the heads, so the flows and readouts have no gradient
    assert compared == len(list(stored.probe.backbone.parameters())) + 1 + len(
        list(stored.probe.reads.parameters()))


def test_the_encoder_gets_its_own_rate_and_its_norms_are_not_decayed(standardizer,
                                                                     pooling):
    """3e-4 on 314M pretrained weights is a way to destroy them, and decaying a
    pretrained LayerNorm gain pulls the encoder off its own scale."""
    torch.manual_seed(0)
    tuned = Finetuned(a_backbone(), pooling, standardizer)
    groups = tuned.parameter_groups(TRAINING)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [
        (3e-4, 1e-4),                 # the readouts
        (3e-4, 0.0),                  # the CLS token
        (1e-3, 1e-4),                 # the flows
        (3e-5, 0.1),                  # the read adapters
        (LR_BACKBONE, WD_BACKBONE),   # the encoder's weights
        (LR_BACKBONE, 0.0),           # and its norms
    ]
    assert sum(len(g["params"]) for g in groups) == len(
        [p for p in tuned.parameters() if p.requires_grad])
    assert all(p.ndim == 1 for p in groups[-1]["params"])
    assert all(p.ndim >= 2 for p in groups[-2]["params"])


def test_the_backbone_stays_in_eval_so_only_requires_grad_changes(standardizer, pooling):
    """Train mode would turn on dropout and drop-path as well, and then the arm would
    differ from the reported run in more than the one thing it is meant to test."""
    torch.manual_seed(0)
    tuned = Finetuned(a_backbone(), pooling, standardizer)
    tuned.train()
    assert tuned.training and not tuned.probe.backbone.training


def test_the_finetune_is_measured_against_a_reported_run(standardizer):
    """It needs no frozen reference of its own: marginals is already being trained."""
    assert default_recipe("finetune").stem == "marginals"
    assert default_recipe("mean").stem == "pooling"
    assert load_run(default_recipe("finetune")).name == "marginals"


def test_the_finetune_trains_on_the_fixtures(staged, tmp_path):
    """End to end through `fit`: the unfrozen encoder, the recomputed blocks and the
    checkpoint that now has to carry the backbone."""
    cfg, _, _ = staged
    result = ablation_run(cfg, tmp_path / "ft", arm="finetune", chunk=8, max_epochs=1,
                          recipe="configs/pooling.yaml", lr_backbone=1e-4,
                          backbone=a_backbone(), **QUIET)
    assert result["best"]["epoch"] == 0
    choices = json.loads((tmp_path / "ft" / "choices.json").read_text())
    assert choices["ablation_arm"] == "finetune"
    assert choices["lr_backbone"] == 1e-4 and choices["wd_backbone"] == WD_BACKBONE
    assert "codecs are frozen" in choices["finetunes"]
    assert "finetuning ceiling" in choices["note"]

    saved = torch.load(tmp_path / "ft" / "best.pt", weights_only=False)["model"]
    pretrained = a_backbone().state_dict()
    encoder = {k[len("probe.backbone."):]: v for k, v in saved.items()
               if k.startswith("probe.backbone.")}
    assert encoder, "the checkpoint carries the encoder"
    moved = [k for k, v in encoder.items()
             if k in pretrained and not torch.equal(v, pretrained[k])]
    assert moved, "and the encoder it carries is not the one we started from"
