"""Cross-validated joint4, so every source gets a rho from a model that never saw it.

    python -m aionflow_model.crossval --fold 0 --out runs/cv/fold0 [--device cuda]

The sample is sorted by targetid, shuffled with the split's own seed and cut into
ten blocks, exactly as `manifest_split` does it, so the blocks are unions of the
existing split: blocks 0-7 are its training set, 8 its validation set and 9 its
test set. Fold f tests on blocks 2f and 2f+1, stops early on block 2f+2 and trains
on the other seven, a 70/10/20 split. Across the five folds every source is
tested exactly once.

Only joint4 is cross-validated and only rho is written. The other runs were
trained on the fixed split, so scoring them on a fold's test rows would score
sources they were trained on.
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

from .analysis import DRAWS, load_model, within_object
from .config import TRAINING, load_run, recipe_path
from .data import Split, Standardizer, TokenDataset
from .objective import SUBSETS, Model
from .train import CHECKPOINT, fit, validation_masks

FOLDS = 5
BLOCKS = 10
RHO = "rho.csv"
SUMMARY = "rho_summary.json"


class CrossvalError(RuntimeError):
    pass


def blocks(targetids: np.ndarray, seed: int) -> np.ndarray:
    """Block 0-9 per source, from the same shuffle `manifest_split.assign` draws."""
    tids = np.asarray(targetids, np.int64)
    if np.unique(tids).size != tids.size:
        raise CrossvalError("targets must be unique")
    n = tids.size
    order = np.argsort(tids)
    perm = np.random.RandomState(int(seed)).permutation(n)
    rank = np.empty(n, dtype=np.int64)
    rank[perm] = np.arange(n)
    edges = np.round(np.arange(1, BLOCKS) / BLOCKS * n).astype(int)
    out = np.empty(n, dtype=np.int64)
    out[order] = np.searchsorted(edges, rank, side="right")
    return out


def roles(block: np.ndarray, fold: int) -> dict[str, np.ndarray]:
    """Which sources fold `fold` trains, validates and tests on."""
    if not 0 <= fold < FOLDS:
        raise CrossvalError(f"fold must be 0-{FOLDS - 1}, got {fold}")
    test = block // (BLOCKS // FOLDS) == fold
    val = block == (2 * fold + 2) % BLOCKS
    return {"train": ~test & ~val, "val": val, "test": test}


class FoldSplit:
    """Rows of one fold role, drawn from all three staged splits.

    It carries the same per-row arrays as a `Split` and reads each row's tokens from
    the staged file that holds it. The arrays are copies, so withholding rows from
    one role cannot reach another.
    """

    ROW_ARRAYS = ("targetid", "redshift", "wise", "present", "detuid", "spectype", "y_raw",
                  "y_ok", "counts", "bkg", "expo", "rate_ok", "sf_dominated")
    standardized = Split.standardized
    withhold = Split.withhold

    def __init__(self, name: str, parts: list[Split], picks: list[np.ndarray]):
        self.name = name
        self._parts = parts
        for attr in self.ROW_ARRAYS:
            setattr(self, attr, np.concatenate([getattr(p, attr)[k] for p, k in zip(parts, picks)]))
        self._part = np.concatenate([np.full(k.size, i) for i, k in enumerate(picks)])
        self._local = np.concatenate(picks)
        self.n = int(self.targetid.size)

    def tokens(self, row: int) -> dict[str, np.ndarray]:
        return self._parts[self._part[row]].tokens(int(self._local[row]))

    def close(self) -> None:
        """The parts belong to whoever opened them."""

    def __len__(self) -> int:
        return self.n


def fold_splits(parts: dict[str, Split], fold: int, seed: int) -> tuple[dict, dict]:
    """The train, val and test views of one fold, and a check that the blocks line up
    with the existing split."""
    order = list(parts.values())
    block = blocks(np.concatenate([p.targetid for p in order]), seed)
    role = roles(block, fold)
    offsets = np.cumsum([0] + [p.n for p in order])
    views = {}
    for name, mask in role.items():
        picks = [np.flatnonzero(mask[lo:hi]) for lo, hi in zip(offsets[:-1], offsets[1:])]
        views[name] = FoldSplit(name, order, picks)
    by_split = {name: block[lo:hi] for name, lo, hi in zip(parts, offsets[:-1], offsets[1:])}
    aligned = (bool((by_split["train"] < 8).all()) and bool((by_split["val"] == 8).all())
               and bool((by_split["test"] == 9).all()))
    return views, {"blocks_match_existing_split": aligned}


RHO_CHUNK = 32


def run(cfg: dict, fold: int, out: str | Path, *, recipe: str | Path | None = None,
        device: str = "cpu", chunk: int = 448, rho_chunk: int = RHO_CHUNK, draws: int = DRAWS,
        workers: int = 0, max_epochs: int | None = None, rho_only: bool = False,
        backbone=None, log=print) -> dict:
    """Train one fold and write rho for its held-out sources.

    With `rho_only`, the fold's saved model is reused and only rho is computed: the
    draws are the memory-hungry step, and a fold whose training finished should not
    have to train again because they ran out of room.
    """
    recipe = load_run(recipe_path("joint4") if recipe is None else recipe)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TRAINING.seed)
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    parts = {name: Split(staged, work, name) for name in ("train", "val", "test")}
    try:
        views, check = fold_splits(parts, fold, int(cfg["split"]["seed"]))
        if backbone is None:
            from .encoder import load_backbone
            backbone = load_backbone()
        if rho_only:
            if not (out / CHECKPOINT).is_file():
                raise CrossvalError(f"--rho-only needs a trained fold at {out / CHECKPOINT}")
            return {"rho": write_rho(load_model(out, backbone, device), views["test"], fold,
                                     out, device, rho_chunk, draws, log)}
        choices = {"crossval_fold": fold, "crossval_folds": FOLDS,
                   "rows": {k: v.n for k, v in views.items()}, **check}
        log(f"[crossval] fold {fold}: {choices['rows']}; blocks line up with the "
            f"existing split: {check['blocks_match_existing_split']}")
        if recipe.exclude_sf_dominated:
            from .cleaning import DESCRIPTION
            choices["sf_dominated_withheld"] = {k: views[k].withhold(views[k].sf_dominated)
                                                for k in ("train", "val")}
            choices["sf_dominated_rule"] = DESCRIPTION
        standardizer = Standardizer.fit(views["train"])
        standardizer.write(out / "standardizer.json")
        model = Model(backbone, recipe, standardizer).to(device)
        datasets = {k: TokenDataset(views[k], standardizer) for k in ("train", "val")}
        masks = validation_masks(views["val"], TRAINING.seed)
        result = fit(model, datasets, masks, out, device=device, chunk=chunk, workers=workers,
                     max_epochs=max_epochs, choices=choices, log=log)
        del model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary = write_rho(load_model(out, backbone, device), views["test"], fold, out,
                            device, rho_chunk, draws, log)
        return {"best": result["best"], "rho": summary}
    finally:
        for split in parts.values():
            split.close()


def write_rho(model: Model, test: FoldSplit, fold: int, out: Path, device: str, chunk: int,
              draws: int, log=print) -> dict:
    rho, summary = within_object(model, test, device, chunk, draws, SUBSETS[-1], log=log)
    pd.DataFrame({"targetid": test.targetid, "spectype": test.spectype,
                  "redshift": test.redshift, "rho": rho,
                  "fold": fold}).to_csv(out / RHO, index=False)
    (out / SUMMARY).write_text(json.dumps({"fold": fold, **summary}, indent=1,
                                          default=float) + "\n")
    log(f"[crossval] fold {fold}: rho for {test.n} held-out sources -> {out / RHO}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--fold", type=int, required=True, help=f"0 to {FOLDS - 1}")
    parser.add_argument("--out", required=True, help="run directory for this fold")
    parser.add_argument("--run", default=None, help="run recipe (default configs/joint4.yaml)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=448, help="rows per training forward")
    parser.add_argument("--rho-chunk", type=int, default=RHO_CHUNK,
                        help="rows per forward for rho; the draws, not training, set the memory")
    parser.add_argument("--rho-only", action="store_true",
                        help="reuse this fold's trained model and only compute rho")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.fold, args.out, recipe=args.run, device=args.device,
            chunk=args.chunk, rho_chunk=args.rho_chunk, draws=args.draws, workers=args.workers,
            max_epochs=args.max_epochs, rho_only=args.rho_only)
    except (CrossvalError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
