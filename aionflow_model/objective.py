"""The conditioning subsets, the per-source sampler, and the per-head objective.

"For each source and at each training step we mask out a random subset of the
modalities (modality dropout), so one model covers all combinations", drawn "by
sampling its size uniformly on {1, ..., 4} and then uniformly among subsets of
that size, clamped to the modalities the source has". Our sample always has a
spectrum and an image and over 99% of it has redshift and WISE, so for almost
every source the clamp does nothing; where it bites, the size is drawn uniformly
on {1, ..., k} over the k modalities the source does have, which is the same rule
whenever k = 4 and never returns an empty conditioning set.

A head scores a source through `poisson.log_marginal`, with each of its
dimensions in one of the three states that module defines: a scalar target with
a label is pinned at it, a rate target with a measurement is integrated against
its Poisson factor, and anything missing is integrated on the fixed prior grid.
A source with nothing observed in a head is not scored by it at all: integrating
out every dimension gives log 1 whatever the model says.

Rows are grouped by which of a head's dimensions are observed, because the grid
is a product and its size is K to the power of the integrated dimensions. Within
a group every row shares one grid shape, which is what lets a batch be scored in
a handful of calls rather than one per source.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor, nn

from .config import Head, Run
from .data import MODALITIES, RATE_TARGETS, SCALAR_TARGETS, TARGETS, Standardizer
from .encoder import Probe
from .flows import FlowHead
from .poisson import log_marginal, pinned, prior_axis, rate_axis


def subsets() -> Tensor:
    """The 15 non-empty modality combinations, smallest first, (15, 4) bool."""
    order = [c for size in range(1, len(MODALITIES) + 1)
             for c in combinations(range(len(MODALITIES)), size)]
    out = torch.zeros(len(order), len(MODALITIES), dtype=torch.bool)
    for row, combination in enumerate(order):
        out[row, list(combination)] = True
    return out


SUBSETS = subsets()
SUBSET_NAMES = tuple("".join(m for m, on in zip(MODALITIES, row) if on) for row in SUBSETS)


def sample_subsets(present: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """One conditioning subset per source: size uniform, then subset uniform, clamped."""
    rows = present.shape[0]
    available = present.sum(1)
    if not bool((available > 0).all()):
        raise ValueError("a source has no modality at all")
    # size uniform on {1, ..., k}, then a uniform subset of that size among the k
    size = (torch.rand(rows, generator=generator) * available).floor().long() + 1
    noise = torch.rand(present.shape, generator=generator).masked_fill(~present, -1.0)
    rank = noise.argsort(dim=1, descending=True).argsort(dim=1)
    return rank < size[:, None]


# ----------------------------------------------------------------------------- the model

class Heads(nn.Module):
    """Whatever turns a batch into one context per head, plus the flows on them.

    The probe is one such thing and the emission-line baseline is another; they
    share this so that both are scored by exactly the same likelihood.
    """

    run: Run
    standardizer: Standardizer
    flows: nn.ModuleDict

    def contexts(self, batch: dict, mask: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def log_likelihood(self, batch: dict, mask: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
        """Per head, the per-row log likelihood and which rows it can score."""
        contexts = self.contexts(batch, mask)
        return {head.name: head_log_likelihood(head, self.flows[head.name],
                                               contexts[head.name], batch, self.standardizer)
                for head in self.run.heads}


class Model(Heads):
    """The probe and one flow per head."""

    def __init__(self, backbone, run: Run, standardizer: Standardizer):
        super().__init__()
        self.run = run
        self.standardizer = standardizer
        self.probe = Probe(backbone, run)
        self.flows = nn.ModuleDict({head.name: FlowHead(len(head.targets))
                                    for head in run.heads})

    def contexts(self, batch: dict, mask: Tensor) -> dict[str, Tensor]:
        return self.probe(batch, mask)

    def parameter_groups(self, training) -> list[dict]:
        """The paper's four groups: readouts and CLS, flows, adapters, and no decay on CLS."""
        return [
            {"params": list(self.probe.readouts.parameters()),
             "lr": training.lr_readout, "weight_decay": training.wd_readout},
            {"params": [self.probe.cls],
             "lr": training.lr_readout, "weight_decay": training.wd_cls},
            {"params": list(self.flows.parameters()),
             "lr": training.lr_flow, "weight_decay": training.wd_readout},
            {"params": list(self.probe.reads.parameters()),
             "lr": training.lr_adapter, "weight_decay": training.wd_adapter},
        ]


# ----------------------------------------------------------------------------- the loss

def observed(head: Head, batch: dict) -> Tensor:
    """(B, D): whether each of the head's dimensions carries a measurement."""
    columns = []
    for target in head.targets:
        if TARGETS[target].kind == "scalar":
            columns.append(batch["y_ok"][:, SCALAR_TARGETS.index(target)])
        else:
            columns.append(batch["rate_ok"][:, RATE_TARGETS.index(target)])
    return torch.stack(columns, dim=1)


def scorable_rows(run: Run, split) -> dict[str, int]:
    """How many rows of a split each head can score at all.

    A head whose target is entirely missing is built, is optimised over, contributes
    nothing, and in a loss curve is indistinguishable from instant convergence. The
    standardizer refuses fewer than two usable rows, which covers the common case, but
    the objective would skip such a head without a word.
    """
    batch = {"y_ok": torch.from_numpy(split.y_ok),
             "rate_ok": torch.from_numpy(split.rate_ok)}
    return {head.name: int(observed(head, batch).any(dim=1).sum()) for head in run.heads}


def axes_for(head: Head, batch: dict, rows: Tensor, seen: Tensor,
             standardizer: Standardizer) -> list:
    """The quadrature axes of one group of rows, which share an observation pattern."""
    out = []
    for d, target in enumerate(head.targets):
        kind = TARGETS[target].kind
        if not bool(seen[d]):
            out.append(prior_axis(int(rows.numel()), device=batch["y"].device))
        elif kind == "scalar":
            out.append(pinned(batch["y"][rows, SCALAR_TARGETS.index(target)]))
        else:
            j = RATE_TARGETS.index(target)
            out.append(rate_axis(batch["counts"][rows, j], batch["bkg"][rows, j],
                                 batch["expo"][rows, j],
                                 standardizer.mean[target], standardizer.scale[target]))
    return out


def head_log_likelihood(head: Head, flow: FlowHead, context: Tensor, batch: dict,
                        standardizer: Standardizer) -> tuple[Tensor, Tensor]:
    """log p of a head's targets per row, and the rows it can score."""
    seen = observed(head, batch)
    scorable = seen.any(dim=1)
    out = torch.zeros(seen.shape[0], dtype=torch.float64, device=context.device)
    patterns = torch.unique(seen[scorable], dim=0) if bool(scorable.any()) else seen[:0]
    for pattern in patterns:
        rows = torch.nonzero((seen == pattern).all(dim=1) & scorable, as_tuple=True)[0]
        axes = axes_for(head, batch, rows, pattern, standardizer)
        here = context[rows]
        out = out.index_put((rows,), log_marginal(
            axes, lambda u, c=here: flow.log_prob(u.to(c.dtype), c).to(torch.float64)))
    return out, scorable


def batch_loss(log_likelihoods: dict[str, tuple[Tensor, Tensor]],
               weights: dict[str, int] | None = None) -> tuple[Tensor, dict[str, float]]:
    """Mean over heads of their mean per-row negative log likelihood.

    `weights` gives each head the number of scorable rows in the whole batch, so a
    batch scored in chunks accumulates exactly the same gradient as one scored
    whole. Without it the mean is taken over the rows present here. The mean over
    heads rather than the sum is what lets one set of learning rates serve runs
    with five heads and runs with one.
    """
    total, parts = 0.0, {}
    for name, (values, scorable) in log_likelihoods.items():
        count = float(weights[name] if weights else int(scorable.sum()))
        if count <= 0:
            continue
        nll = -values[scorable].sum() / count
        total = total + nll
        parts[name] = float(nll.detach())
    return total / max(len(log_likelihoods), 1), parts
