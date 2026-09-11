"""M2: the count likelihood, the node placement, and the quadrature."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import poisson as scipy_poisson

from aionflow_model.data import NET_COUNT_FLOOR
from aionflow_model.poisson import (
    COUNT_FLOOR,
    LN10,
    SPAN,
    Axis,
    K,
    laplace_proposal,
    log_marginal,
    log_pmf,
    pinned,
    prior_axis,
    product_grid,
    rate_axis,
)

MEAN, SCALE = -2.5, 0.35        # a standardization close to the paper's rate scale
COUNTS = torch.tensor([0.0, 1.0, 3.0, 12.0, 400.0, 5000.0], dtype=torch.float64)
BKG = torch.tensor([2.0, 0.0, 0.5, 1.0, 5.0, 40.0], dtype=torch.float64)
EXPO = torch.full((6,), 1000.0, dtype=torch.float64)


def standard_normal(u: torch.Tensor) -> torch.Tensor:
    """A normalized q that factorizes across dimensions, so the sum has a closed form."""
    return (-0.5 * u.square() - 0.5 * math.log(2 * math.pi)).sum(-1)


# ----------------------------------------------------------------------------- likelihood

def test_the_count_likelihood_is_poisson_in_the_total_rate():
    axis = rate_axis(COUNTS, BKG, EXPO, MEAN, SCALE)
    mu = EXPO[:, None] * 10.0 ** (SCALE * axis.nodes + MEAN) + BKG[:, None]
    want = scipy_poisson.logpmf(COUNTS[:, None].numpy(), mu.numpy())
    assert np.abs(axis.log_like.numpy() - want).max() < 1e-11
    # exposure and background enter the likelihood only, and a zero-count band scores
    assert torch.isfinite(axis.log_like).all()
    direct = log_pmf(COUNTS, BKG, EXPO, axis.nodes, MEAN, SCALE)
    assert torch.equal(direct, axis.log_like)


# ----------------------------------------------------------------------------- placement

def test_the_proposal_is_the_gaussian_posterior_under_the_unit_prior():
    centre, width = laplace_proposal(COUNTS, BKG, EXPO, MEAN, SCALE)
    net = torch.clamp(COUNTS - BKG, min=NET_COUNT_FLOOR)
    sigma = torch.sqrt(torch.clamp(COUNTS, min=COUNT_FLOOR)) / (LN10 * SCALE * net)
    u_hat = (torch.log10(net / EXPO) - MEAN) / SCALE
    assert torch.allclose(centre, u_hat / (1 + sigma**2))
    assert torch.allclose(width, sigma / torch.sqrt(1 + sigma**2))
    # a brighter band is placed more tightly and closer to its plug-in value
    assert (width[1:] < width[:-1]).all()
    assert abs(centre[-1] - u_hat[-1]) < abs(centre[0] - u_hat[0])


def test_a_band_with_counts_below_background_gets_a_scale_the_paper_quotes():
    """"a source with N <= B gives a scale of 0.92 to 0.93 rather than a sign flip"."""
    counts = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    bkg = torch.tensor([2.0, 9.0, 3.0], dtype=torch.float64)
    expo = torch.full((3,), 800.0, dtype=torch.float64)
    _, width = laplace_proposal(counts, bkg, expo, MEAN, SCALE)
    assert torch.isfinite(width).all() and (width > 0).all()
    # Both floors bite, so the scale depends on the band's standardization alone.
    assert torch.allclose(width, torch.full_like(width, float(width[0])))
    assert 0.92 < float(width[0]) < 0.93
    # The paper's range is the two bands' two scales; it pins s to a third of a dex.
    scales = [float(s) for s in np.arange(0.20, 0.60, 0.001)
              if 0.92 < float(laplace_proposal(counts[:1], bkg[:1], expo[:1],
                                               MEAN, float(s))[1]) < 0.93]
    assert min(scales) == pytest.approx(0.343, abs=5e-3)
    assert max(scales) == pytest.approx(0.370, abs=5e-3)


def test_the_fixed_prior_grid_is_closed_nodes_over_plus_minus_five():
    axis = prior_axis(4)
    assert axis.size == K == 12 and SPAN == 5.0
    assert axis.nodes[0, 0] == -5.0 and axis.nodes[0, -1] == 5.0
    spacing = torch.diff(axis.nodes[0])
    assert torch.allclose(spacing, spacing[0]) and spacing[0] == pytest.approx(10 / 11)
    assert torch.allclose(axis.log_spacing, torch.log(spacing[0]))
    assert axis.log_like is None and not pinned(torch.zeros(4)).integrated


# ----------------------------------------------------------------------------- the sum

def test_twelve_nodes_reproduce_a_dense_integral():
    coarse = log_marginal([rate_axis(COUNTS, BKG, EXPO, MEAN, SCALE)], standard_normal)
    dense = log_marginal([rate_axis(COUNTS, BKG, EXPO, MEAN, SCALE, k=20001, span=60.0)],
                         standard_normal)
    assert (coarse - dense).abs().max() < 1e-3


def test_the_jacobian_is_carried_so_a_density_integrates_to_one():
    # No likelihood factor: the sum is the mass a standard normal puts inside +-5.
    mass = log_marginal([prior_axis(3)], standard_normal).exp()
    assert torch.allclose(mass, torch.ones(3, dtype=torch.float64), atol=1e-4)
    fine = log_marginal([prior_axis(3, k=4001)], standard_normal).exp()
    assert torch.allclose(fine, torch.ones(3, dtype=torch.float64), atol=1e-6)
    # and it is the node spacing that carries it
    wide = Axis(prior_axis(3, span=10.0).nodes, prior_axis(3).log_spacing, None)
    assert not torch.allclose(log_marginal([wide], standard_normal).exp(),
                              torch.ones(3, dtype=torch.float64), atol=1e-2)


def test_a_joint_is_the_outer_product_of_its_axes():
    axes = [rate_axis(COUNTS, BKG, EXPO, MEAN, SCALE),
            rate_axis(COUNTS.flip(0), BKG, EXPO, MEAN, SCALE)]
    grid, like = product_grid(axes)
    assert grid.shape == (6, K * K, 2) and like.shape == (6, K * K)
    # the first axis varies slowest, and each node keeps its own likelihood
    assert torch.equal(grid[:, :, 0].reshape(6, K, K)[:, :, 0], axes[0].nodes)
    assert torch.equal(grid[:, :, 1].reshape(6, K, K)[:, 0, :], axes[1].nodes)
    assert torch.allclose(like.reshape(6, K, K),
                          axes[0].log_like[:, :, None] + axes[1].log_like[:, None, :])
    # a q that factorizes makes the joint sum the sum of the two single-axis sums
    joint = log_marginal(axes, standard_normal)
    apart = sum(log_marginal([axis], standard_normal) for axis in axes)
    assert torch.allclose(joint, apart, atol=1e-10)


def test_a_pinned_dimension_is_a_density_not_an_integral():
    value = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    got = log_marginal([pinned(value)], standard_normal)
    assert torch.allclose(got, standard_normal(value[:, None]))
    # mixed: two pinned dimensions and one integrated give K nodes, not K^3
    mixed = [pinned(value), prior_axis(3), pinned(-value)]
    grid, _ = product_grid(mixed)
    assert grid.shape == (3, K, 3)
    assert torch.equal(grid[:, :, 0], value[:, None].expand(3, K))
    both = log_marginal(mixed, standard_normal)
    alone = log_marginal([prior_axis(3)], standard_normal)
    assert torch.allclose(both, alone + standard_normal(value[:, None])
                          + standard_normal(-value[:, None]), atol=1e-10)


def test_a_malformed_head_is_refused():
    with pytest.raises(ValueError, match="at least one dimension"):
        product_grid([])
    with pytest.raises(ValueError, match="same rows"):
        product_grid([prior_axis(3), prior_axis(4)])
