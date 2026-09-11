"""Cut M6: score a trained run on the test split.

    python -m aionflow_model.evaluate --run-dir runs/joint4 [--config CONFIG]
        [--device cuda] [--chunk N] [--draws N]

"Information gain (IG), our main metric, is the mean over test objects of
log p(y_i | c_i,g) - log p_KDE(y_i) in nats, for modality combination g. For
rates, both terms are the counts likelihood of Eq. 1. We also report R2, using
the posterior sample mean in natural units as the prediction." The prior is "an
unconditional Gaussian KDE with Scott's bandwidth rule, fit on the standardized
training-split targets", over training plug-in rates for a rate head and over
training tuples for a joint.

Both terms go through `objective.head_log_likelihood`, the prior reaching it as a
KDE dressed as a head. That is not a convenience: it is what makes "both terms
are the counts likelihood of Eq. 1" true rather than asserted, since the prior
then meets the same Laplace-placed nodes and the same Jacobian as the model.

Every combination is scored on one common subsample per head, fixed across the 15
rows so they are comparable: the test sources that hold all four modalities and
every one of the head's targets. Its size is an output, recorded, not a target.

The emission-line baseline is scored through this same function, with `--baseline`,
so Figure 1's comparison is against the same prior on the same subsample. Its 15
rows are identical to each other, since it reads no modality.

Outputs in the run directory: `results.json` with a row per head and combination,
and `per_source.csv` with each test source's log likelihood under every
combination, which is what the bootstrap, Figure 2's redshift trend and the
analyses read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aionflow_data.common import load_config

from .config import Head, load_run, recipe_path
from .data import (
    RATE_TARGETS,
    SCALAR_TARGETS,
    TARGETS,
    Split,
    Standardizer,
    TokenDataset,
    loader,
    log_plug_in_rate,
)
from .flows import GaussianKDE
from .objective import SUBSET_NAMES, SUBSETS, Heads, Model, head_log_likelihood, observed
from .train import CHECKPOINT, chunks, to_device

RESULTS = "results.json"
PER_SOURCE = "per_source.csv"
DRAWS = 1024
COVERAGE = (0.68, 0.90, 0.95)


class EvaluateError(RuntimeError):
    pass


class PriorHead:
    """A KDE dressed as a head, so the prior meets the same quadrature as the model."""

    def __init__(self, kde: GaussianKDE):
        self.kde = kde

    def log_prob(self, u: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.kde.log_prob(u)


def head_values(head: Head, split: Split, standardizer: Standardizer) -> np.ndarray:
    """The head's targets in standardized units, one column per dimension."""
    columns = []
    for target in head.targets:
        if TARGETS[target].kind == "scalar":
            i = SCALAR_TARGETS.index(target)
            columns.append(standardizer.encode(target, split.y_raw[:, i]))
        else:
            j = RATE_TARGETS.index(target)
            columns.append(standardizer.encode(
                target, log_plug_in_rate(split.counts[:, j], split.bkg[:, j], split.expo[:, j])))
    return np.stack(columns, axis=1)


def fit_prior(head: Head, train: Split, standardizer: Standardizer) -> GaussianKDE:
    """The unconditional KDE over the training rows that carry every one of the head's
    targets: standardized labels for scalars, plug-in rates for bands, tuples for a joint."""
    complete = observed(head, {"y_ok": torch.from_numpy(train.y_ok),
                               "rate_ok": torch.from_numpy(train.rate_ok)}).all(dim=1).numpy()
    values = head_values(head, train, standardizer)[complete]
    if values.shape[0] < 2:
        raise EvaluateError(f"head {head.name!r} has {values.shape[0]} complete training rows")
    return GaussianKDE(torch.from_numpy(values))


def common_subsample(head: Head, split: Split) -> np.ndarray:
    """Test sources with all four modalities and every one of the head's targets."""
    complete = observed(head, {"y_ok": torch.from_numpy(split.y_ok),
                               "rate_ok": torch.from_numpy(split.rate_ok)}).all(dim=1).numpy()
    return complete & split.present.all(axis=1)


# ----------------------------------------------------------------------------- one pass

@torch.no_grad()
def score(model: Heads, batches, mask_row: torch.Tensor, device, chunk: int, draws: int,
          keep_draws: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """Per head, the log likelihood and the posterior mean of every row, under one
    combination held fixed across sources. The draws themselves are kept only for the
    combination coverage is reported on, since they are the largest thing here."""
    out = {head.name: {"ll": [], "mean": [], "draws": []} for head in model.run.heads}
    for batch in batches:
        batch = to_device(batch, device)
        for part in chunks(batch, chunk):
            rows = part["y"].shape[0]
            mask = mask_row.to(device).expand(rows, -1)
            contexts = model.contexts(part, mask)
            for head in model.run.heads:
                flow = model.flows[head.name]
                values, _ = head_log_likelihood(head, flow, contexts[head.name], part,
                                                model.standardizer)
                sample = flow.sample(contexts[head.name], draws)
                out[head.name]["ll"].append(values.cpu().numpy())
                out[head.name]["mean"].append(sample.mean(dim=1).cpu().numpy())
                if keep_draws:
                    out[head.name]["draws"].append(sample.cpu().numpy())
    return {name: {key: np.concatenate(value) for key, value in parts.items() if value}
            for name, parts in out.items()}


def r_squared(head: Head, means: np.ndarray, split: Split, standardizer: Standardizer,
              keep: np.ndarray) -> dict[str, float]:
    """From the posterior sample mean in natural units, per scalar dimension."""
    out = {}
    for d, target in enumerate(head.targets):
        if TARGETS[target].kind != "scalar":
            continue
        truth = split.y_raw[keep, SCALAR_TARGETS.index(target)]
        predicted = standardizer.decode(target, means[keep, d])
        residual = float(((truth - predicted) ** 2).sum())
        total = float(((truth - truth.mean()) ** 2).sum())
        out[target] = 1.0 - residual / total if total > 0 else float("nan")
    return out


def coverage(head: Head, sample: np.ndarray, split: Split, standardizer: Standardizer,
             keep: np.ndarray) -> dict[str, dict[str, float]]:
    """Central interval coverage per scalar dimension, on the common subsample."""
    out = {}
    for d, target in enumerate(head.targets):
        if TARGETS[target].kind != "scalar":
            continue
        truth = standardizer.encode(target, split.y_raw[keep, SCALAR_TARGETS.index(target)])
        drawn = sample[keep, :, d]
        per_level = {}
        for level in COVERAGE:
            lo, hi = np.quantile(drawn, [(1 - level) / 2, (1 + level) / 2], axis=1)
            per_level[f"{level:.2f}"] = float(((truth >= lo) & (truth <= hi)).mean())
        out[target] = per_level
    return out


# ----------------------------------------------------------------------------- the run

def evaluate(model: Heads, splits: dict[str, Split], *, device: str = "cpu",
             chunk: int = 448, draws: int = DRAWS, workers: int = 0, dataset=None,
             log=print) -> tuple[dict, pd.DataFrame]:
    """`dataset` defaults to the staged tokens; the baseline passes its own, so both
    are scored against the same prior on the same common subsample."""
    test, train = splits["test"], splits["train"]
    standardizer = model.standardizer
    dataset = TokenDataset(test, standardizer) if dataset is None else dataset
    priors = {head.name: fit_prior(head, train, standardizer).to(device)
              for head in model.run.heads}
    keep = {head.name: common_subsample(head, test) for head in model.run.heads}

    frame = pd.DataFrame({"targetid": test.targetid, "spectype": test.spectype,
                          "redshift": test.redshift})
    prior_ll = {}
    for head in model.run.heads:
        batches = loader(dataset, 512, shuffle=False, workers=workers)
        values = []
        for batch in batches:
            batch = to_device(batch, device)
            for part in chunks(batch, chunk):
                got, _ = head_log_likelihood(head, PriorHead(priors[head.name]),
                                             torch.zeros(part["y"].shape[0], 1, device=device),
                                             part, standardizer)
                values.append(got.cpu().numpy())
        prior_ll[head.name] = np.concatenate(values)
        frame[f"prior_{head.name}"] = prior_ll[head.name]
        frame[f"common_{head.name}"] = keep[head.name]

    rows, cover = [], {}
    for g, name in enumerate(SUBSET_NAMES):
        last = name == SUBSET_NAMES[-1]         # coverage is reported on all four modalities
        scored = score(model, loader(dataset, 512, shuffle=False, workers=workers),
                       SUBSETS[g], device, chunk, draws, keep_draws=last)
        for head in model.run.heads:
            here, mask = scored[head.name], keep[head.name]
            frame[f"ll_{head.name}_{name}"] = here["ll"]
            gain = float((here["ll"][mask] - prior_ll[head.name][mask]).mean())
            rows.append({"head": head.name, "inputs": name, "n": int(mask.sum()),
                         "information_gain": gain,
                         **{f"r2_{k}": v for k, v in
                            r_squared(head, here["mean"], test, standardizer, mask).items()}})
            if last:
                cover[head.name] = coverage(head, here["draws"], test, standardizer, mask)
        log(f"[evaluate] {name:5s} " + "  ".join(
            f"{r['head']} {r['information_gain']:+.3f}" for r in rows[-len(model.run.heads):]))
    results = {"run": model.run.name, "draws": draws,
               "common_subsample": {k: int(v.sum()) for k, v in keep.items()},
               "prior": "Gaussian KDE at Scott's bandwidth on the training split, "
                        "scored through the same quadrature as the model",
               "coverage": cover, "rows": rows}
    return results, frame


def run(cfg: dict, run_dir: str | Path, *, device: str = "cpu", chunk: int = 448,
        draws: int = DRAWS, workers: int = 0, baseline: bool = False, backbone=None,
        log=print) -> dict:
    run_dir = Path(run_dir)
    if not (run_dir / CHECKPOINT).is_file():
        raise EvaluateError(f"no checkpoint at {run_dir / CHECKPOINT}")
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    splits = {name: Split(staged, work, name) for name in ("train", "test")}
    try:
        if baseline:
            from .baseline import LineDataset, load
            model, mean, scale = load(run_dir, work, device)
            dataset = LineDataset(splits["test"], model.standardizer, work, mean, scale)
        else:
            checkpoint = torch.load(run_dir / CHECKPOINT, map_location=device,
                                    weights_only=False)
            standardizer = Standardizer.from_dict(checkpoint["standardizer"])
            if backbone is None:
                from .encoder import load_backbone
                backbone = load_backbone()
            recipe = load_run(recipe_path(checkpoint["run"]))
            model = Model(backbone, recipe, standardizer).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            dataset = None
        results, frame = evaluate(model, splits, device=device, chunk=chunk, draws=draws,
                                  workers=workers, dataset=dataset, log=log)
    finally:
        for split in splits.values():
            split.close()
    (run_dir / RESULTS).write_text(json.dumps(results, indent=1) + "\n")
    frame.to_csv(run_dir / PER_SOURCE, index=False)
    log(f"[evaluate] {len(results['rows'])} rows -> {run_dir / RESULTS}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--run-dir", required=True, help="a directory train.py wrote")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=448, help="rows per forward")
    parser.add_argument("--draws", type=int, default=DRAWS, help="posterior draws per source")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline", action="store_true",
                        help="the run directory holds an emission-line baseline")
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.run_dir, device=args.device, chunk=args.chunk,
            draws=args.draws, workers=args.workers, baseline=args.baseline)
    except (EvaluateError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
