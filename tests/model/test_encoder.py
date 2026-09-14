"""M4: the read path is the paper's equations, and the data stream is untouched."""

from __future__ import annotations

import os

import pytest
import torch
from aion.fourm.fm_utils import Block

from aionflow_model.config import Head, Run, load_run
from aionflow_model.data import ALL_TOKEN_KEYS, MODALITIES, TOKEN_KEYS, TOKEN_SIZES
from aionflow_model.encoder import (
    CLS_INIT_STD,
    RANK,
    WIDTH,
    BlockRead,
    Delta,
    EncoderError,
    Probe,
    readout,
)
from aionflow_model.flows import CONTEXT
from tests.model.fake import VOCABULARY, FakeBackbone, ShapedBackbone

NEEDS_AION = pytest.mark.skipif(not os.environ.get("AIONFLOW_TEST_AION"),
                                reason="set AIONFLOW_TEST_AION=1 to run the real backbone")
DEPTH, HEADS, SMALL = 3, 4, 96


def a_block(width=SMALL, heads=HEADS, seed=0) -> Block:
    torch.manual_seed(seed)
    block = Block(width, num_heads=heads, qkv_bias=False, proj_bias=False, mlp_bias=False,
                  gated_mlp=True, qk_norm=True).eval()
    for p in block.parameters():                      # a block that is not near-identity
        p.data.add_(0.05 * torch.randn_like(p))
    return block.requires_grad_(False)


def a_batch(rows=3, seed=1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {key: torch.randint(0, VOCABULARY, (rows, TOKEN_SIZES[key]), generator=generator)
            for key in ALL_TOKEN_KEYS}


# ----------------------------------------------------------------------------- the deltas

def test_a_delta_is_b_a_and_starts_at_zero():
    delta = Delta(WIDTH, RANK)
    assert delta.a.shape == (RANK, WIDTH) and delta.b.shape == (WIDTH, RANK)
    assert (delta.b == 0).all()
    assert delta(torch.randn(2, 5, WIDTH)).abs().max() == 0.0
    # A ~ N(0, 1/d)
    assert float(Delta(WIDTH, RANK).a.detach().std()) == pytest.approx(WIDTH**-0.5, rel=0.05)
    delta.b.data.normal_()
    x = torch.randn(2, 5, WIDTH)
    assert torch.allclose(delta(x), x @ delta.a.T @ delta.b.T, atol=1e-5)


def test_the_paper_s_parameter_counts():
    assert sum(p.numel() for p in readout().parameters()) == 528_128
    per_block = sum(p.numel() for p in BlockRead().parameters())
    assert per_block == 3 * 2 * WIDTH * RANK == 294_912
    assert per_block * 12 == 3_538_944


# ----------------------------------------------------------------------------- the read

def test_the_read_is_the_block_run_on_the_sequence_with_the_cls_hidden_as_a_key():
    """The CLS reads the data tokens through the block's own attention and writes
    nothing back, so appending it with its own column masked out must agree."""
    block = a_block()
    rows, tokens = 3, 7
    x = torch.randn(rows, tokens, SMALL)
    cls = torch.randn(rows, 1, SMALL)
    absent = torch.zeros(rows, tokens, dtype=torch.bool)
    absent[0, -2:] = True                                  # a source with fewer tokens

    # the reference: one block over [X; cls], with the CLS blocked as a key for every query
    joint = torch.cat([x, cls], dim=1)
    joint_mask = torch.cat([absent, torch.ones(rows, 1, dtype=torch.bool)], dim=1)
    reference = block(joint, mask=joint_mask.unsqueeze(1))

    # ours: zero deltas, so the read is exactly the frozen attention
    read = BlockRead(SMALL, RANK)
    x_hat = block.norm1(x)
    out = cls + block.drop_path(read(block, block.norm1(cls), x_hat, absent.unsqueeze(1)))
    out = out + block.drop_path(block.mlp(block.norm2(out)))
    assert torch.allclose(out, reference[:, tokens:], atol=1e-6)
    # and the data tokens are untouched by the CLS being there at all
    assert torch.allclose(reference[:, :tokens], block(x, mask=absent.unsqueeze(1)), atol=1e-6)


def test_trained_deltas_enter_before_the_qk_norm_and_after_the_pre_norm():
    """"The block's QK norm acts on q and K after the deltas", and the deltas act on the
    token the block has already pre-normed.

    The zero-delta test above cannot see either: with B = 0 the delta contributes
    nothing, so the order it is applied in is unobservable. Here the reference is a
    copy of the block with the deltas baked into its own qkv weight, so AION's own
    forward puts the norm after them structurally rather than by our arrangement. If
    we normed before adding the deltas, or fed them the raw token, this disagrees.
    """
    import copy

    block = a_block()
    rows, tokens = 3, 7
    x, cls = torch.randn(rows, tokens, SMALL), torch.randn(rows, 1, SMALL)
    absent = torch.zeros(rows, tokens, dtype=torch.bool)

    read = BlockRead(SMALL, RANK)
    torch.manual_seed(3)
    for delta in (read.query, read.key, read.value):
        delta.b.data.normal_(std=0.2)                      # deltas that actually do something

    baked = copy.deepcopy(block)
    with torch.no_grad():
        deltas = torch.cat([(d.b @ d.a) for d in (read.query, read.key, read.value)], dim=0)
        baked.attn.qkv.weight.add_(deltas)                 # W_Q + dQ, W_K + dK, W_V + dV
    joint = torch.cat([x, cls], dim=1)
    joint_mask = torch.cat([absent, torch.ones(rows, 1, dtype=torch.bool)], dim=1)
    reference = baked(joint, mask=joint_mask.unsqueeze(1))[:, tokens:]

    x_hat = block.norm1(x)
    out = cls + block.drop_path(read(block, block.norm1(cls), x_hat, absent.unsqueeze(1)))
    out = out + block.drop_path(block.mlp(block.norm2(out)))
    # 1e-4 sits three orders above this pair's float32 agreement and three below the
    # nearest way of getting it wrong: norming before the deltas misses by 1.9, feeding
    # them the raw token by 0.5.
    assert torch.allclose(out, reference, atol=1e-4)

    # and the deltas are doing enough work that the agreement means something
    zeroed = BlockRead(SMALL, RANK)
    flat = cls + block.drop_path(zeroed(block, block.norm1(cls), x_hat, absent.unsqueeze(1)))
    flat = flat + block.drop_path(block.mlp(block.norm2(flat)))
    assert (out - flat).abs().max() > 1e-3


def test_a_trained_delta_changes_the_read_and_only_the_read():
    block = a_block()
    x, cls = torch.randn(3, 7, SMALL), torch.randn(3, 1, SMALL)
    read = BlockRead(SMALL, RANK)
    before = read(block, block.norm1(cls), block.norm1(x), None)
    for delta in (read.query, read.key, read.value):
        delta.b.data.normal_(std=0.1)
    after = read(block, block.norm1(cls), block.norm1(x), None)
    assert not torch.allclose(before, after)
    assert torch.allclose(block(x, mask=None), block(x, mask=None))


def test_the_read_refuses_a_backbone_the_paper_does_not_describe():
    x, cls = torch.randn(2, 5, SMALL), torch.randn(2, 1, SMALL)
    read = BlockRead(SMALL, RANK)
    biased = Block(SMALL, num_heads=HEADS, qkv_bias=True, qk_norm=True, gated_mlp=True).eval()
    with pytest.raises(EncoderError, match="projection biases"):
        read(biased, biased.norm1(cls), biased.norm1(x), None)
    block = a_block()
    block.attn.allow_zero_attn = True
    with pytest.raises(EncoderError, match="zero token"):
        read(block, block.norm1(cls), block.norm1(x), None)


# ----------------------------------------------------------------------------- the probe

@pytest.fixture(scope="module")
def probe():
    torch.manual_seed(0)
    backbone = FakeBackbone(width=SMALL, heads=HEADS, depth=DEPTH)
    return Probe(backbone, load_run("configs/marginals.yaml")), backbone


def test_the_probe_leaves_the_backbone_stream_bit_identical(probe):
    model, backbone = probe
    batch, mask = a_batch(), torch.ones(3, 4, dtype=torch.bool)
    tokens, hidden, needed = model.inputs(batch, mask)
    x, emb, token_mask, _ = backbone.embed_inputs(tokens, mask=hidden,
                                                  num_encoder_tokens=needed)
    reference = backbone.forward_encoder(x + emb, token_mask)
    model.context(batch, mask)          # runs the same blocks with the CLS attached
    assert torch.equal(backbone.forward_encoder(x + emb, token_mask), reference)
    assert needed == sum(TOKEN_SIZES.values()) == 853


def test_hiding_a_modality_removes_its_keys(probe):
    model, _ = probe
    batch = a_batch()
    full = torch.ones(3, 4, dtype=torch.bool)
    for m, modality in enumerate(MODALITIES):
        mask = full.clone()
        mask[:, m] = False
        _, hidden, needed = model.inputs(batch, mask)
        gone = {key for key, _ in TOKEN_KEYS[modality]}
        assert needed == 853 - sum(TOKEN_SIZES[key] for key in gone)
        assert {key for key in ALL_TOKEN_KEYS if hidden[key].any()} == gone
        assert all(hidden[key].all() for key in gone)
    # a different conditioning set gives a different context
    spectrum_only = torch.zeros(3, 4, dtype=torch.bool)
    spectrum_only[:, MODALITIES.index("S")] = True
    assert not torch.allclose(model.context(batch, full), model.context(batch, spectrum_only))
    with pytest.raises(EncoderError, match="no modality"):
        model.context(batch, torch.zeros(3, 4, dtype=torch.bool))


def test_the_probe_trains_only_its_own_parameters(probe):
    model, backbone = probe
    assert not any(p.requires_grad for p in backbone.parameters())
    model.train()
    assert not backbone.training and model.training
    contexts = model(a_batch(), torch.ones(3, 4, dtype=torch.bool))
    assert set(contexts) == {"flux", "lx", "sfr", "mstar", "sfr_mstar"}
    assert all(c.shape == (3, CONTEXT) for c in contexts.values())
    sum(c.square().sum() for c in contexts.values()).backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert model.cls.grad is not None and model.cls.shape == (SMALL,)
    assert all(read.key.b.grad is not None for read in model.reads)


def test_the_trained_parameter_count_is_the_papers():
    """768 in the CLS, 3,538,944 in the deltas, 528,128 per readout, plus the flows."""
    from aionflow_model.flows import FlowHead

    backbone = ShapedBackbone(depth=12)
    totals = {}
    for name in ("marginals", "rates", "joint4"):
        run = load_run(f"configs/{name}.yaml")
        model = Probe(backbone, run)
        assert model.cls.numel() == WIDTH == 768
        assert sum(p.numel() for p in model.reads.parameters()) == 3_538_944
        assert sum(p.numel() for p in model.readouts.parameters()) == 528_128 * len(run.heads)
        flows = sum(sum(p.numel() for p in FlowHead(len(head.targets)).parameters())
                    for head in run.heads)
        totals[name] = model.trained_parameters() + flows
    assert totals["marginals"] == 11_731_536                # the paper's 11.7M
    assert totals["rates"] == 5_219_184                     # 5.2M
    assert totals["joint4"] == 5_317_856                    # 5.3M
    # Appendix B's pooling comparison trains two scalar heads: 6.8M
    pooling = Probe(backbone, Run("pooling", (Head("flux", ("flux",)), Head("lx", ("lx",)))))
    assert (pooling.trained_parameters()
            + 2 * sum(p.numel() for p in FlowHead(1).parameters())) == 6_795_888


@NEEDS_AION
def test_the_real_backbone_matches_the_paper_and_keeps_its_stream(splits, standardizer):
    from aionflow_model.encoder import load_backbone

    backbone = load_backbone()
    assert len(backbone.encoder) == 12
    attention = backbone.encoder[0].attn
    assert attention.num_heads == 12 and attention.qkv.weight.shape == (3 * WIDTH, WIDTH)
    assert attention.scale == (WIDTH // attention.num_heads) ** -0.5
    assert attention.qkv.bias is None and attention.proj.bias is None
    model = Probe(backbone, load_run("configs/rates.yaml"))
    assert model.trained_parameters() == WIDTH + 3_538_944 + 528_128
    batch, mask = a_batch(rows=2), torch.ones(2, 4, dtype=torch.bool)
    tokens, hidden, needed = model.inputs(batch, mask)
    x, emb, token_mask, _ = backbone.embed_inputs(tokens, mask=hidden,
                                                  num_encoder_tokens=needed)
    reference = backbone.forward_encoder(x + emb, token_mask)
    model.context(batch, mask)
    assert torch.equal(backbone.forward_encoder(x + emb, token_mask), reference)
    assert abs(model.cls.detach().std().item() - CLS_INIT_STD) < 0.01
