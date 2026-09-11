"""The flow heads, and the KDE prior the information gain is measured against.

Every head is "a conditional neural spline flow: 8 transforms, 8 monotonic
rational-quadratic bins on [-5, 5], a masked autoregressive conditioner with two
256-unit hidden layers, context dimension 256. A scalar head has one feature and
a joint head one feature per joint dimension." Targets reach the flow already
standardized, which is also what puts them inside the spline's domain: a feature
outside [-5, 5] is passed through untransformed.

The prior in the information-gain metric is "an unconditional Gaussian KDE with
Scott's bandwidth rule, fit on the standardized training-split targets", and for
a joint a KDE over training tuples. It is the same object in both cases, so it
lives here beside the heads it is compared against.
"""

from __future__ import annotations

import math

import torch
import zuko
from torch import Tensor, nn

TRANSFORMS = 8
BINS = 8
BOUND = 5.0        # zuko's MonotonicRQSTransform default; asserted in the tests
HIDDEN = (256, 256)
CONTEXT = 256
KDE_CHUNK = 4096   # evaluation points per block, to bound the pairwise matrix


class FlowHead(nn.Module):
    """One conditional flow over `features` standardized targets."""

    def __init__(self, features: int, context: int = CONTEXT):
        super().__init__()
        if features < 1:
            raise ValueError(f"a head needs at least one feature, got {features}")
        self.features = int(features)
        self.context = int(context)
        self.flow = zuko.flows.NSF(features=self.features, context=self.context,
                                   transforms=TRANSFORMS, bins=BINS, hidden_features=HIDDEN)

    def log_prob(self, u: Tensor, context: Tensor) -> Tensor:
        """log q(u | context).

        `u` may carry extra axes over a single context row, which is how the
        quadrature reaches it: u of (B, K, D) against a context of (B, C) gives
        (B, K).
        """
        for _ in range(u.dim() - context.dim()):
            context = context.unsqueeze(-2)
        return self.flow(context).log_prob(u)

    def sample(self, context: Tensor, draws: int) -> Tensor:
        """`draws` posterior draws per context row, shaped (B, draws, D)."""
        return self.flow(context).sample((draws,)).movedim(0, -2)

    def extra_repr(self) -> str:
        return f"features={self.features}, context={self.context}"


class GaussianKDE:
    """An unconditional Gaussian KDE at Scott's bandwidth over standardized points.

    `bandwidth` multiplies Scott's rule; the paper sweeps it over 0.3 to 3 to show
    the information gains are not an artifact of an oversmoothed prior.
    """

    def __init__(self, points: Tensor, bandwidth: float = 1.0):
        points = torch.as_tensor(points, dtype=torch.float64)
        if points.dim() != 2 or points.shape[0] < 2:
            raise ValueError(f"a KDE needs (n, d) points with n > 1, got {tuple(points.shape)}")
        if not (bandwidth > 0):
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        n, d = points.shape
        self.n, self.features, self.bandwidth = int(n), int(d), float(bandwidth)
        self.scott = float(n ** (-1.0 / (d + 4)))
        cov = torch.cov(points.T).reshape(d, d) * (self.scott * bandwidth) ** 2
        chol = torch.linalg.cholesky(cov)
        self.whitener = torch.linalg.inv(chol).T                 # right-multiplied
        self.points = points @ self.whitener
        self.log_norm = (-0.5 * d * math.log(2 * math.pi)
                         - float(torch.log(torch.diagonal(chol)).sum()) - math.log(n))

    def to(self, device) -> GaussianKDE:
        self.points = self.points.to(device)
        self.whitener = self.whitener.to(device)
        return self

    def log_prob(self, u: Tensor) -> Tensor:
        """log p(u) for points shaped (..., d), returning (...)."""
        u = torch.as_tensor(u, dtype=torch.float64, device=self.points.device)
        if u.shape[-1] != self.features:
            raise ValueError(f"KDE over {self.features} features got {u.shape[-1]}")
        shape = u.shape[:-1]
        flat = (u.reshape(-1, self.features) @ self.whitener).contiguous()
        out = torch.empty(flat.shape[0], dtype=torch.float64, device=flat.device)
        for lo in range(0, flat.shape[0], KDE_CHUNK):
            block = flat[lo:lo + KDE_CHUNK]
            d2 = torch.cdist(block, self.points).square()
            out[lo:lo + KDE_CHUNK] = torch.logsumexp(-0.5 * d2, dim=-1)
        return (out + self.log_norm).reshape(shape)

    def __repr__(self) -> str:
        return (f"GaussianKDE(n={self.n}, features={self.features}, "
                f"scott={self.scott:.4g}, bandwidth={self.bandwidth:g})")
