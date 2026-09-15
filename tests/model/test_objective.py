"""M5: the modality-dropout sampler and the per-head objective."""

from __future__ import annotations

import copy
import math
from collections import Counter

import pytest
import torch

from aionflow_model.config import Head, load_run
from aionflow_model.data import MODALITIES, RATE_TARGETS, SCALAR_TARGETS
from aionflow_model.objective import (
    SUBSET_NAMES,
    SUBSETS,
    Model,
    batch_loss,
    head_log_likelihood,
    observed,
    sample_subsets,
    scorable,
)
from aionflow_model.poisson import K, log_marginal, pinned, prior_axis, rate_axis
from tests.model.fake import FakeBackbone

SMALL = 96


# ----------------------------------------------------------------------------- subsets

def test_there_are_fifteen_subsets_and_they_are_every_non_empty_one():
    assert SUBSETS.shape == (15, 4) and SUBSETS.any(dim=1).all()
    assert len({tuple(row.tolist()) for row in SUBSETS}) == 15
    assert SUBSET_NAMES[0] == "Z" and SUBSET_NAMES[-1] == "ZSIW"
    assert SUBSET_NAMES == tuple(sorted(SUBSET_NAMES, key=len))       # smallest first


def test_the_sampler_is_uniform_over_sizes_then_over_subsets():
    generator = torch.Generator().manual_seed(0)
    drawn = sample_subsets(torch.ones(120_000, 4, dtype=torch.bool), generator)
    sizes = Counter(drawn.sum(1).tolist())
    assert set(sizes) == {1, 2, 3, 4}
    for size in sizes:
        assert sizes[size] == pytest.approx(30_000, rel=0.03)
    counts = Counter(tuple(row.tolist()) for row in drawn)
    assert len(counts) == 15
    for combination, count in counts.items():
        size = sum(combination)
        expected = 30_000 / math.comb(4, size)
        assert count == pytest.approx(expected, rel=0.06), combination


def test_the_sampler_clamps_to_what_a_source_has_and_never_empties():
    present = torch.tensor([[False, True, True, False],      # spectrum and image only
                            [True, True, True, True],
                            [False, True, False, False]])    # spectrum only
    generator = torch.Generator().manual_seed(1)
    drawn = sample_subsets(present.repeat(4000, 1), generator)
    assert (drawn <= present.repeat(4000, 1)).all()
    assert drawn.any(dim=1).all()
    two = drawn[0::3]
    assert set(two.sum(1).tolist()) == {1, 2}
    assert Counter(two.sum(1).tolist())[1] == pytest.approx(2000, rel=0.06)
    assert (drawn[2::3][:, MODALITIES.index("S")]).all()     # the only one it has
    with pytest.raises(ValueError, match="no modality at all"):
        sample_subsets(torch.zeros(2, 4, dtype=torch.bool), generator)


# ----------------------------------------------------------------------------- the loss

def a_batch(rows=6, seed=0) -> dict:
    torch.manual_seed(seed)
    y_ok = torch.ones(rows, len(SCALAR_TARGETS), dtype=torch.bool)
    y_ok[0, SCALAR_TARGETS.index("sfr")] = False
    y_ok[1, :] = False
    rate_ok = torch.ones(rows, len(RATE_TARGETS), dtype=torch.bool)
    rate_ok[2, 0] = False
    return {
        "y": torch.randn(rows, len(SCALAR_TARGETS), dtype=torch.float64),
        "y_ok": y_ok,
        "counts": torch.tensor([[3.0, 7.0]] * rows, dtype=torch.float64),
        "bkg": torch.tensor([[0.5, 1.0]] * rows, dtype=torch.float64),
        "expo": torch.full((rows, len(RATE_TARGETS)), 900.0, dtype=torch.float64),
        "rate_ok": rate_ok,
    }


def unit_normal(u):
    return (-0.5 * u.square() - 0.5 * math.log(2 * math.pi)).sum(-1)


class UnitNormal:
    """A stand-in flow: a standard normal whatever the context, so every quadrature
    below has a value the test can write down."""

    def log_prob(self, u, context):
        return unit_normal(u)


def test_a_scalar_head_scores_only_the_rows_that_have_the_label(standardizer):
    head = Head("sfr", ("sfr",))
    batch = a_batch()
    context = torch.zeros(6, 1, dtype=torch.float64)
    values, keep = head_log_likelihood(head, UnitNormal(), context, batch, standardizer)
    assert keep.tolist() == [False, False, True, True, True, True]
    want = -0.5 * batch["y"][:, SCALAR_TARGETS.index("sfr")] ** 2 - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(values[keep], want[keep])
    assert (values[~keep] == 0).all()


def test_a_joint_pins_what_is_observed_and_integrates_what_is_not(standardizer):
    head = Head("sfr_mstar", ("sfr", "mstar"))
    batch = a_batch()
    sfr, mstar = SCALAR_TARGETS.index("sfr"), SCALAR_TARGETS.index("mstar")
    context = torch.zeros(6, 1, dtype=torch.float64)
    values, keep = head_log_likelihood(head, UnitNormal(), context, batch, standardizer)
    assert keep.tolist() == [True, False, True, True, True, True]

    # row 0 has M* but not SFR: SFR integrates on the prior grid, M* stays pinned
    want = log_marginal([prior_axis(1, dtype=torch.float64), pinned(batch["y"][0:1, mstar])],
                        unit_normal)
    assert values[0].item() == pytest.approx(float(want), abs=1e-9)

    # row 2 has both: two pinned dimensions and no integration at all
    both = log_marginal([pinned(batch["y"][2:3, sfr]), pinned(batch["y"][2:3, mstar])],
                        unit_normal)
    assert values[2].item() == pytest.approx(float(both), abs=1e-9)
    assert values[2].item() == pytest.approx(float(unit_normal(batch["y"][2, [sfr, mstar]])),
                                             abs=1e-9)


def test_a_rate_head_integrates_the_counts_and_falls_back_when_a_band_is_missing(standardizer):
    head = Head("rates", RATE_TARGETS)
    batch = a_batch()
    context = torch.zeros(6, 1, dtype=torch.float64)
    values, keep = head_log_likelihood(head, UnitNormal(), context, batch, standardizer)
    assert keep.all()          # a zero-count band is a measurement, so every row scores

    def band(row, j):
        target = RATE_TARGETS[j]
        return rate_axis(batch["counts"][row:row + 1, j], batch["bkg"][row:row + 1, j],
                         batch["expo"][row:row + 1, j],
                         standardizer.mean[target], standardizer.scale[target])

    # row 2 has no P2 measurement, so that axis falls back to the fixed prior grid
    fallback = [prior_axis(1, dtype=torch.float64), band(2, 1)]
    assert values[2].item() == pytest.approx(float(log_marginal(fallback, unit_normal)),
                                            abs=1e-9)
    # every other row has both bands, so the grid is K by K
    full = [band(0, 0), band(0, 1)]
    assert values[0].item() == pytest.approx(float(log_marginal(full, unit_normal)), abs=1e-9)
    assert full[0].size * full[1].size == K * K == 144


def test_the_flow_runs_in_its_own_precision_and_the_sum_in_double(standardizer):
    """The nodes are built in float64 and cast to the flow's dtype, which costs about
    a part in ten million; the Poisson factor and the sum stay in float64."""
    head = Head("rates", RATE_TARGETS)
    batch = a_batch()
    double, _ = head_log_likelihood(head, UnitNormal(), torch.zeros(6, 1, dtype=torch.float64),
                                    batch, standardizer)
    single, _ = head_log_likelihood(head, UnitNormal(), torch.zeros(6, 1), batch, standardizer)
    assert double.dtype == single.dtype == torch.float64
    assert torch.allclose(double, single, atol=1e-5)
    assert not torch.equal(double, single)


def test_a_mixed_joint_skips_sources_with_no_observed_scalar(standardizer):
    """Integrating both scalars out of the four-dimensional joint tells it only what the
    dedicated rate head already carries, and costs K^2 extra nodes to say it.

    On the real sample that is 4.6% of sources and two thirds of the joint's entire
    quadrature budget, so those rows are not trained on. A pure rate head is unaffected:
    there is no scalar for it to be missing.
    """
    batch = a_batch()
    mixed = Head("joint4", ("rate_p2", "rate_p3", "sfr", "mstar"))
    rates = Head("rates", RATE_TARGETS)
    scalars = Head("sfr_mstar", ("sfr", "mstar"))

    # row 1 has both rates and neither scalar; row 0 has one scalar, row 2 one rate
    assert not batch["y_ok"][1].any() and batch["rate_ok"][1].all()
    assert scorable(mixed, batch).tolist() == [True, False, True, True, True, True]
    assert observed(mixed, batch).any(dim=1).tolist() == [True] * 6   # the old rule kept it
    assert scorable(rates, batch).all()                               # no scalar to miss
    assert scorable(scalars, batch).tolist() == [True, False, True, True, True, True]

    # and the row really is left out of the likelihood, not merely flagged
    context = torch.zeros(6, 1, dtype=torch.float64)
    values, keep = head_log_likelihood(mixed, UnitNormal(), context, batch, standardizer)
    assert not keep[1] and values[1] == 0.0
    assert keep.sum() == 5


def test_a_row_with_nothing_observed_is_not_scored(standardizer):
    batch = a_batch()
    batch["y_ok"][:] = False
    head = Head("sfr_mstar", ("sfr", "mstar"))
    context = torch.zeros(6, 1, dtype=torch.float64)
    _, keep = head_log_likelihood(head, UnitNormal(), context, batch, standardizer)
    assert not keep.any()
    assert observed(head, batch).shape == (6, 2)


# ----------------------------------------------------------------------------- weighting

def gradients_of(model):
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def accumulate(model, batch, mask, weights, rows, step=3):
    """One backward per chunk, weighted by the whole batch."""
    model.zero_grad(set_to_none=True)
    for lo in range(0, rows, step):
        part = {k: v[lo:lo + step] for k, v in batch.items()}
        batch_loss(model.log_likelihood(part, mask[lo:lo + step]), weights)[0].backward()
    return gradients_of(model)


def test_chunking_a_batch_leaves_the_loss_and_its_gradient_unchanged(standardizer):
    """What is under test is the row weighting, not arithmetic precision, so it is
    checked in double: each head's mean is taken over its scorable rows in the whole
    batch, which makes the chunks' accumulated gradient the whole batch's exactly.

    In float32 the two differ in the last bits, because the summation order differs and
    by how much depends on the BLAS. The second half holds that drift to a relative
    size over the whole gradient rather than element by element.
    """
    torch.manual_seed(0)
    backbone = FakeBackbone(width=SMALL, heads=4, depth=2)
    model = Model(backbone, load_run("configs/marginals.yaml"), standardizer).double().eval()
    rows = 8
    batch = a_batch(rows=rows)
    from tests.model.test_encoder import a_batch as token_batch
    batch.update(token_batch(rows=rows))
    batch["present"] = torch.ones(rows, 4, dtype=torch.bool)
    mask = torch.ones(rows, 4, dtype=torch.bool)
    weights = {head.name: int(observed(head, batch).any(dim=1).sum())
               for head in model.run.heads}

    model.zero_grad(set_to_none=True)
    whole, _ = batch_loss(model.log_likelihood(batch, mask), weights)
    whole.backward()
    exact = gradients_of(model)
    assert exact
    chunked = accumulate(model, batch, mask, weights, rows)
    assert len(chunked) == len(exact)
    for a, b in zip(exact, chunked):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-8)

    # the same model in the precision training runs at
    single = copy.deepcopy(model).float().eval()
    single.zero_grad(set_to_none=True)
    batch32 = dict(batch, y=batch["y"].float())
    batch_loss(single.log_likelihood(batch32, mask), weights)[0].backward()
    exact32 = gradients_of(single)
    chunked32 = accumulate(single, batch32, mask, weights, rows)
    size = torch.cat([g.reshape(-1) for g in exact32]).norm()
    drift = torch.cat([(a - b).reshape(-1) for a, b in zip(exact32, chunked32)]).norm()
    assert drift / size < 1e-4


def test_the_loss_is_the_mean_over_heads(standardizer):
    values = {"a": (torch.tensor([-1.0, -3.0], dtype=torch.float64),
                    torch.tensor([True, True])),
              "b": (torch.tensor([-5.0, 0.0], dtype=torch.float64),
                    torch.tensor([True, False]))}
    total, parts = batch_loss(values)
    assert parts == {"a": 2.0, "b": 5.0}
    assert float(total) == pytest.approx(3.5)
    # with batch-wide weights a head's mean is taken over the whole batch, not the chunk
    weighted, _ = batch_loss(values, {"a": 4, "b": 4})
    assert float(weighted) == pytest.approx((4 / 4 + 5 / 4) / 2)
