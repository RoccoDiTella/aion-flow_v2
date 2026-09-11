"""The Poisson marginal likelihood of the observed counts, by deterministic quadrature.

"Rate heads are trained on the marginal likelihood of the observed counts,
-log integral p(N | lambda, t, B) q(lambda | c) dlambda, with N ~ Poisson(lambda t
+ B). Exposure and background enter only the likelihood, never the model input.
The integral is a uniform-weight Riemann sum on equally spaced closed nodes in
standardized latent space, K = 12 nodes per axis spanning +-5 standardized units
(a K^2 grid over the two latent axes of a joint). Nodes are placed per source and
per band by a Laplace proposal under the unit-scale standardized prior, at centre
u_hat / (1 + sigma^2) and scale sigma / sqrt(1 + sigma^2), where u_hat is the
standardized log10 plug-in rate, s the standardization scale, and sigma =
sqrt(N) / (ln 10 . s . (N - B)). Before sigma is formed, N - B is floored at half
a photon and N at one, so a source with N <= B gives a scale of 0.92 to 0.93
rather than a sign flip. A band with no usable measurement and any missing
observed joint dimension both fall back to the fixed (non-recentred) prior grid,
with the Jacobian carried exactly."

The proposal is the posterior of a Gaussian likelihood N(u_hat, sigma^2) under
the standard normal prior the targets are standardized to, and sigma is the
delta-method error of log10 of the plug-in rate (N - B) / t.

A head's dimension therefore arrives at quadrature in one of three states, and
`Axis` below is each of them: pinned at an observed standardized value, integrated
on a recentred grid against a Poisson factor, or integrated on the fixed prior
grid. Every row in one call shares the pattern; grouping rows by pattern is the
caller's business. Counts arithmetic runs in float64 throughout, because a bright
source's log-pmf is a large number built from larger ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import NET_COUNT_FLOOR

K = 12             # nodes per integrated axis
SPAN = 5.0         # standardized units either side of the centre
COUNT_FLOOR = 1.0  # N floored at one photon before sigma is formed
LN10 = math.log(10.0)


def log_pmf(counts: Tensor, bkg: Tensor, expo: Tensor, u: Tensor,
            mean: float, scale: float) -> Tensor:
    """log Poisson(N | lambda t + B) at standardized log10 rates `u`.

    `counts`, `bkg` and `expo` are (B,); `u` is (B, K); the result is (B, K).
    """
    counts, bkg, expo = counts[..., None], bkg[..., None], expo[..., None]
    mu = expo * torch.pow(10.0, scale * u + mean) + bkg
    return counts * torch.log(mu) - mu - torch.lgamma(counts + 1.0)


def laplace_proposal(counts: Tensor, bkg: Tensor, expo: Tensor,
                     mean: float, scale: float) -> tuple[Tensor, Tensor]:
    """Centre and scale of the proposal, per source, in standardized units."""
    net = torch.clamp(counts - bkg, min=NET_COUNT_FLOOR)
    u_hat = (torch.log10(net / expo) - mean) / scale
    sigma = torch.sqrt(torch.clamp(counts, min=COUNT_FLOOR)) / (LN10 * scale * net)
    variance = sigma * sigma
    return u_hat / (1.0 + variance), sigma / torch.sqrt(1.0 + variance)


# ----------------------------------------------------------------------------- axes

@dataclass(frozen=True)
class Axis:
    """One dimension of a head at quadrature time.

    `nodes` is (B, 1) for a pinned dimension and (B, K) for an integrated one;
    `log_spacing` is the log Jacobian of the node spacing, zero when pinned;
    `log_like` is the Poisson factor at the nodes, or None when the dimension
    carries no measurement.
    """

    nodes: Tensor
    log_spacing: Tensor
    log_like: Tensor | None

    @property
    def size(self) -> int:
        return int(self.nodes.shape[-1])

    @property
    def integrated(self) -> bool:
        return self.size > 1


def pinned(value: Tensor) -> Axis:
    """A dimension held at its observed standardized value: a density, not an integral."""
    value = value.to(torch.float64)
    return Axis(value[..., None], torch.zeros_like(value), None)


def prior_axis(rows: int, *, k: int = K, span: float = SPAN,
               device=None, dtype=torch.float64) -> Axis:
    """The fixed, non-recentred grid: `k` closed nodes spanning +-`span`."""
    offsets = torch.linspace(-span, span, k, device=device, dtype=dtype)
    spacing = math.log(2 * span / (k - 1))
    return Axis(offsets.expand(rows, k),
                torch.full((rows,), spacing, device=device, dtype=dtype), None)


def rate_axis(counts: Tensor, bkg: Tensor, expo: Tensor, mean: float, scale: float,
              *, k: int = K, span: float = SPAN) -> Axis:
    """A measured band: the recentred grid and the Poisson factor on it."""
    counts, bkg, expo = (t.to(torch.float64) for t in (counts, bkg, expo))
    centre, width = laplace_proposal(counts, bkg, expo, mean, scale)
    offsets = torch.linspace(-span, span, k, device=counts.device, dtype=torch.float64)
    nodes = centre[..., None] + width[..., None] * offsets
    log_spacing = torch.log(width) + math.log(2 * span / (k - 1))
    return Axis(nodes, log_spacing, log_pmf(counts, bkg, expo, nodes, mean, scale))


# ----------------------------------------------------------------------------- the sum

def product_grid(axes: list[Axis]) -> tuple[Tensor, Tensor]:
    """The outer product of the axes' nodes, (B, M, D), and the Poisson factor, (B, M)."""
    if not axes:
        raise ValueError("a head has at least one dimension")
    rows = axes[0].nodes.shape[0]
    sizes = [axis.size for axis in axes]
    if any(axis.nodes.shape[0] != rows for axis in axes):
        raise ValueError("every axis must cover the same rows")
    nodes, like = [], torch.zeros((rows, *sizes), dtype=torch.float64,
                                  device=axes[0].nodes.device)
    for d, axis in enumerate(axes):
        shape = [rows] + [1] * len(sizes)
        shape[d + 1] = sizes[d]
        nodes.append(axis.nodes.reshape(shape).expand(rows, *sizes).reshape(rows, -1))
        if axis.log_like is not None:
            like = like + axis.log_like.reshape(shape)
    return torch.stack(nodes, dim=-1), like.reshape(rows, -1)


def log_marginal(axes: list[Axis], log_q) -> Tensor:
    """-log of Eq. 1, without the sign: the log marginal likelihood per source.

    `log_q` takes the grid (B, M, D) and returns (B, M): the head's density at the
    nodes, or any unconditional density such as the KDE prior.
    """
    grid, like = product_grid(axes)
    terms = log_q(grid).to(torch.float64) + like
    jacobian = sum(axis.log_spacing for axis in axes)
    return torch.logsumexp(terms, dim=-1) + jacobian
