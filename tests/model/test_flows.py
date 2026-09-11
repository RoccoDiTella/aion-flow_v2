"""M1: the flow heads are the paper's, and the KDE prior is Scott's rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde

from aionflow_model.flows import BINS, CONTEXT, HIDDEN, TRANSFORMS, FlowHead, GaussianKDE

# "1,099,960 parameters for one feature, 1,151,344 for two" and the joint4 run's 5.3M
# total fixes the four-feature head at 1,250,016.
PARAMETERS = {1: 1_099_960, 2: 1_151_344, 4: 1_250_016}


# ----------------------------------------------------------------------------- the head

@pytest.mark.parametrize("features, want", PARAMETERS.items())
def test_a_head_has_the_papers_parameter_count(features, want):
    assert sum(p.numel() for p in FlowHead(features).parameters()) == want


def test_the_head_is_eight_masked_autoregressive_spline_transforms():
    head = FlowHead(2)
    transforms = list(head.flow.transform.transforms)
    assert len(transforms) == TRANSFORMS == 8
    widths = [m.out_features for m in transforms[0].hyper if hasattr(m, "out_features")]
    assert tuple(widths[:-1]) == HIDDEN == (256, 256)
    assert transforms[0].hyper[0].in_features == head.features + CONTEXT
    # 8 widths, 8 heights and 7 slopes per feature
    assert widths[-1] == head.features * (3 * BINS - 1)


def test_the_spline_domain_is_minus_five_to_five():
    torch.manual_seed(0)
    head = FlowHead(1)
    for p in head.parameters():
        p.data.add_(0.3 * torch.randn_like(p))      # a spline that is not the identity
    context = torch.randn(4, CONTEXT)
    with torch.no_grad():
        base = head.flow(context).base
        inside = torch.full((4, 1), 0.5)
        assert (head.log_prob(inside, context) - base.log_prob(inside)).abs().max() > 1.0
        for outside in (7.0, -7.0):
            u = torch.full((4, 1), outside)
            assert torch.equal(head.log_prob(u, context), base.log_prob(u))


def test_log_prob_broadcasts_a_grid_over_one_context_row():
    torch.manual_seed(0)
    head = FlowHead(2)
    context = torch.randn(5, CONTEXT)
    assert head.log_prob(torch.randn(5, 2), context).shape == (5,)
    grid = torch.randn(5, 12, 2)
    got = head.log_prob(grid, context)
    assert got.shape == (5, 12)
    # every grid node is scored against its own source's context
    one = head.log_prob(grid[:, 3, :], context)
    assert torch.allclose(got[:, 3], one, atol=1e-6)


def test_sampling_is_per_context_and_the_density_normalises():
    torch.manual_seed(0)
    head = FlowHead(1)
    for p in head.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    context = torch.randn(3, CONTEXT)
    draws = head.sample(context, 64)
    assert draws.shape == (3, 64, 1)
    assert not torch.allclose(draws[0], draws[1])
    with torch.no_grad():
        # the head is conditional: the same point scores differently per context
        assert head.log_prob(torch.full((3, 1), 0.5), context).std() > 1.0
        grid = torch.linspace(-8, 8, 16001)
        density = head.log_prob(grid.reshape(1, -1, 1).expand(3, -1, -1), context).exp()
        assert torch.trapezoid(density, grid, dim=-1).allclose(torch.ones(3), atol=1e-3)


def test_a_head_needs_a_feature():
    with pytest.raises(ValueError, match="at least one feature"):
        FlowHead(0)


# ----------------------------------------------------------------------------- the prior

@pytest.mark.parametrize("features", [1, 2, 4])
def test_the_kde_is_scipys_scott_rule_kde(features):
    rng = np.random.default_rng(0)
    mix = np.triu(rng.normal(size=(features, features))) + 2 * np.eye(features)
    points = rng.normal(size=(500, features)) @ mix
    ours, reference = GaussianKDE(torch.from_numpy(points)), gaussian_kde(points.T)
    assert ours.scott == pytest.approx(reference.scotts_factor())
    assert ours.scott == pytest.approx(500 ** (-1.0 / (features + 4)))
    u = rng.normal(size=(37, features))
    got = ours.log_prob(torch.from_numpy(u)).numpy()
    assert np.allclose(got, reference.logpdf(u.T), atol=1e-10)


def test_the_kde_normalises_and_widens_with_the_bandwidth():
    rng = np.random.default_rng(1)
    points = torch.from_numpy(rng.normal(size=(400, 1)))
    grid = torch.linspace(-10, 10, 4001).reshape(-1, 1)
    for bandwidth in (0.3, 1.0, 3.0):
        kde = GaussianKDE(points, bandwidth=bandwidth)
        assert kde.scott == pytest.approx(400 ** (-0.2))
        density = kde.log_prob(grid).exp()
        assert torch.trapezoid(density, grid[:, 0]).item() == pytest.approx(1.0, abs=1e-6)
    peak = [GaussianKDE(points, bandwidth=b).log_prob(torch.zeros(1, 1)).item()
            for b in (0.3, 1.0, 3.0)]
    assert peak[0] > peak[1] > peak[2]      # smoothing flattens the mode


def test_the_kde_chunks_without_changing_its_answer(monkeypatch):
    rng = np.random.default_rng(2)
    kde = GaussianKDE(torch.from_numpy(rng.normal(size=(300, 2))))
    u = torch.from_numpy(rng.normal(size=(9, 11, 2)))
    whole = kde.log_prob(u)
    monkeypatch.setattr("aionflow_model.flows.KDE_CHUNK", 7)
    assert whole.shape == (9, 11)
    assert torch.allclose(whole, kde.log_prob(u), atol=1e-12)


def test_the_kde_refuses_what_it_cannot_fit():
    points = torch.randn(20, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\(n, d\) points"):
        GaussianKDE(torch.randn(5, dtype=torch.float64))
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        GaussianKDE(points, bandwidth=0.0)
    with pytest.raises(ValueError, match="over 2 features got 3"):
        GaussianKDE(points).log_prob(torch.zeros(4, 3, dtype=torch.float64))


def test_the_kde_scores_its_own_support_above_a_shifted_one():
    rng = np.random.default_rng(3)
    points = torch.from_numpy(rng.normal(size=(600, 2)))
    kde = GaussianKDE(points)
    here = kde.log_prob(torch.from_numpy(rng.normal(size=(500, 2)))).mean()
    away = kde.log_prob(torch.from_numpy(rng.normal(size=(500, 2)) + 6.0)).mean()
    assert here > away
    # a standard normal's differential entropy is the scale to beat
    assert here.item() == pytest.approx(-math.log(2 * math.pi) - 1.0, abs=0.15)
