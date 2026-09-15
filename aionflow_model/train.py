"""Cut M5: train one run.

    python -m aionflow_model.train --run configs/joint4.yaml --out runs/joint4
        [--config CONFIG] [--device cuda] [--chunk N] [--workers N] [--max-epochs N]

"AdamW, (b1, b2) = (0.95, 0.999), constant learning rates without warmup: 3e-4
for the readout MLPs and the CLS token, 1e-3 for the flows and 3e-5 for the read
adapters. Weight decay is 1e-4 on readout and flows, 0.1 on the adapters and 0 on
the CLS token. Batches hold 896 sources, gradients are clipped at global norm 5,
and training runs at most 40 epochs with early stopping at patience 5 on the
validation NLL of the trained heads. The seed is 42 throughout, including the
split."

A batch of 896 sources is 896 sequences of up to 853 tokens, which no single
forward fits, so a batch is scored in chunks and the chunks' gradients are
accumulated. Each head's mean is taken over its scorable rows in the whole batch
rather than in the chunk, so the accumulated gradient is exactly the gradient of
the whole batch; a test asserts it against an unchunked pass.

Two things the paper leaves open, both written into `choices.json` in the run
directory. The validation metric is the unweighted mean over the trained heads of
their per-row NLL, on the rows each head can score. Validation draws one
conditioning subset per source from the same sampler under its own seed and keeps
it for every epoch, so the metric is comparable across epochs and runs rather
than being re-randomised each time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from aionflow_data.common import load_config

from .config import TRAINING, load_run
from .data import Split, Standardizer, TokenDataset, loader
from .objective import Model, batch_loss, sample_subsets, scorable, scorable_rows

CHUNK = 448
VALIDATION_SEED_OFFSET = 1_000
CHECKPOINT = "best.pt"
HISTORY = "history.json"
CHOICES = "choices.json"


class TrainError(RuntimeError):
    pass


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def scorable_counts(model: Model, batch: dict) -> dict[str, int]:
    """How many rows of the whole batch each head can score, before it is chunked."""
    return {head.name: int(scorable(head, batch).sum()) for head in model.run.heads}


def chunks(batch: dict, size: int):
    rows = batch["y"].shape[0]
    for lo in range(0, rows, size):
        yield {k: v[lo:lo + size] for k, v in batch.items()}


def train_epoch(model: Model, batches, optimizer, generator, device, chunk: int,
                clip: float) -> dict[str, float]:
    model.train()
    totals, seen = {}, 0
    for batch in batches:
        batch = to_device(batch, device)
        weights = scorable_counts(model, batch)
        optimizer.zero_grad(set_to_none=True)
        for part in chunks(batch, chunk):
            mask = sample_subsets(part["present"].cpu(), generator).to(device)
            loss, parts = batch_loss(model.log_likelihood(part, mask), weights)
            loss.backward()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        optimizer.step()
        seen += 1
    return {k: v / max(seen, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model: Model, batches, masks: dict[int, torch.Tensor], device,
             chunk: int) -> tuple[float, dict[str, float]]:
    """The mean over heads of each head's mean per-row NLL on the split."""
    model.eval()
    sums = {head.name: 0.0 for head in model.run.heads}
    counts = dict.fromkeys(sums, 0)
    for batch in batches:
        batch = to_device(batch, device)
        for part in chunks(batch, chunk):
            mask = torch.stack([masks[int(t)] for t in part["targetid"]]).to(device)
            for name, (values, ok) in model.log_likelihood(part, mask).items():
                sums[name] -= float(values[ok].sum())
                counts[name] += int(ok.sum())
    per_head = {name: sums[name] / counts[name] for name in sums if counts[name]}
    if not per_head:
        raise TrainError("no validation row is scorable by any head")
    return sum(per_head.values()) / len(per_head), per_head


def validation_masks(split: Split, seed: int) -> dict[int, torch.Tensor]:
    """One conditioning subset per source, drawn once and kept for every epoch."""
    generator = torch.Generator().manual_seed(seed + VALIDATION_SEED_OFFSET)
    drawn = sample_subsets(torch.from_numpy(split.present), generator)
    return {int(t): drawn[i] for i, t in enumerate(split.targetid)}


def fit(model, datasets: dict, masks: dict, out: str | Path, *, device: str = "cpu",
        chunk: int = CHUNK, workers: int = 0, max_epochs: int | None = None,
        choices: dict | None = None, log=print) -> dict:
    """Train `model` on `datasets["train"]`, select on `datasets["val"]`, write the run
    directory. The probe and the emission-line baseline both come through here, so the
    schedule, the selection rule and the checkpoint are the same for both."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trainable = scorable_rows(model.run, datasets["train"].split)
    dead = sorted(name for name, rows in trainable.items() if rows == 0)
    if dead:
        raise TrainError(f"heads {dead} can score no training row, so they would be "
                         f"optimised over in silence and look converged; check the "
                         f"label columns their targets name")
    optimizer = torch.optim.AdamW(model.parameter_groups(TRAINING), betas=TRAINING.betas)
    generator = torch.Generator().manual_seed(TRAINING.seed)
    epochs = TRAINING.max_epochs if max_epochs is None else int(max_epochs)
    (out / CHOICES).write_text(json.dumps({
        "run": model.run.name,
        "heads": {head.name: list(head.targets) for head in model.run.heads},
        "validation_metric": "unweighted mean over heads of the per-row NLL on scorable rows",
        "validation_masks": "one subset per source, drawn once at seed "
                            f"{TRAINING.seed + VALIDATION_SEED_OFFSET}",
        "objective": "mean over heads of the per-row NLL, rows weighted by the whole batch",
        "mixed_joint_rows": "a head mixing rates with scalars is trained only on sources "
                            "with at least one observed scalar; integrating both out says "
                            "only what the rate head already carries and costs K^2 nodes",
        "rows_trained_per_head": trainable,
        "batch_chunk_rows": chunk,
        "training": vars(TRAINING),
        **(choices or {}),
    }, indent=1) + "\n")

    history, best, since = [], None, 0
    for epoch in range(epochs):
        started = time.time()
        train_batches = loader(datasets["train"], TRAINING.batch_size, shuffle=True,
                               workers=workers, seed=TRAINING.seed + epoch)
        losses = train_epoch(model, train_batches, optimizer, generator, device, chunk,
                             TRAINING.grad_clip)
        val_batches = loader(datasets["val"], TRAINING.batch_size, shuffle=False,
                             workers=workers)
        metric, per_head = validate(model, val_batches, masks, device, chunk)
        history.append({"epoch": epoch, "train": losses, "val": per_head,
                        "metric": metric, "seconds": time.time() - started})
        (out / HISTORY).write_text(json.dumps(history, indent=1) + "\n")
        improved = best is None or metric < best["metric"]
        log(f"[train] epoch {epoch:3d}  val {metric:.4f}"
            f"{'  *' if improved else ''}  ({history[-1]['seconds']:.1f}s)")
        if improved:
            best, since = {"epoch": epoch, "metric": metric, "per_head": per_head}, 0
            torch.save({"model": model.state_dict(), "run": model.run.name,
                        "standardizer": model.standardizer.as_dict(), **best},
                       out / CHECKPOINT)
        else:
            since += 1
            if since >= TRAINING.patience:
                log(f"[train] no improvement for {since} epochs; stopping")
                break
    log(f"[train] best epoch {best['epoch']} at {best['metric']:.4f} -> {out / CHECKPOINT}")
    return {"best": best, "history": history}


def run(cfg: dict, recipe: str | Path, out: str | Path, *, device: str = "cpu",
        chunk: int = CHUNK, workers: int = 0, max_epochs: int | None = None,
        backbone=None, log=print) -> dict:
    recipe = load_run(recipe)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TRAINING.seed)
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    splits = {name: Split(staged, work, name) for name in ("train", "val")}
    standardizer = Standardizer.fit(splits["train"])
    standardizer.write(out / "standardizer.json")
    if backbone is None:
        from .encoder import load_backbone
        backbone = load_backbone()
    model = Model(backbone, recipe, standardizer).to(device)
    datasets = {k: TokenDataset(v, standardizer) for k, v in splits.items()}
    masks = validation_masks(splits["val"], TRAINING.seed)
    try:
        return fit(model, datasets, masks, out, device=device, chunk=chunk, workers=workers,
                   max_epochs=max_epochs, log=log)
    finally:
        for split in splits.values():
            split.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--run", required=True, help="run recipe, e.g. configs/joint4.yaml")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="rows per forward")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help=f"default {TRAINING.max_epochs}, the paper's cap")
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.run, args.out, device=args.device,
            chunk=args.chunk, workers=args.workers, max_epochs=args.max_epochs)
    except (TrainError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
