"""Cut M7: the emission-line baseline.

"As a baseline for comparison, we train a separate normalizing-flow head on four
standard AGN emission-line fluxes ([O III] 5007, [Ne V] 3426, H-alpha, H-beta)
known to correlate with X-ray emission." "Its flow is identical to the model's
heads, 8 transforms and two 256-unit hidden layers. Only the context encoder
differs, a small MLP over the four standardized line fluxes."

    python -m aionflow_model.baseline --out runs/baseline [--config CONFIG] [--device D]
    python -m aionflow_model.evaluate --run-dir runs/baseline --baseline

"Only the context encoder differs" is meant literally, so the baseline goes
through the same `fit` and the same `evaluate` as the probe: the same schedule,
the same selection rule, the same KDE prior and the same common subsample. What
changes is one module and the four numbers it reads.

The context encoder is the readout's shape with Linear(4, 512) in place of
Linear(768, 512) and without its leading LayerNorm. The readout's leading norm is
there because the CLS state arrives unnormalized; the four fluxes arrive
standardized, and a LayerNorm across four features would undo exactly that.

A line outside the spectrograph's coverage at the source's redshift, or one whose
fit failed, is a zero in `line_features.csv` rather than a missing value, because
a line baseline has nothing to say about such a source and dropping it would let
the baseline choose its own sample. The zeros are standardized with the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from aionflow_data.common import load_config
from aionflow_data.line_features import FEATURES as LINE_FEATURES

from .config import TRAINING, load_run
from .data import Split, Standardizer
from .encoder import DROPOUT, HIDDEN
from .flows import CONTEXT, FlowHead
from .objective import Heads
from .train import CHECKPOINT, fit, validation_masks

LINES = ("oiii_5007", "nev_3426", "halpha", "hbeta")
RECIPE = "configs/baseline.yaml"


class BaselineError(RuntimeError):
    pass


def line_context(lines: int = len(LINES), hidden: int = HIDDEN, context: int = CONTEXT,
                 dropout: float = DROPOUT) -> nn.Sequential:
    """The readout's shape over four standardized fluxes instead of the CLS state."""
    return nn.Sequential(
        nn.Linear(lines, hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, context),
        nn.LayerNorm(context),
    )


class BaselineModel(Heads):
    """One context MLP and one flow per head, on the four line fluxes."""

    def __init__(self, run, standardizer: Standardizer):
        super().__init__()
        self.run = run
        self.standardizer = standardizer
        self.encoders = nn.ModuleDict({head.name: line_context() for head in run.heads})
        self.flows = nn.ModuleDict({head.name: FlowHead(len(head.targets))
                                    for head in run.heads})

    def contexts(self, batch: dict, mask: Tensor) -> dict[str, Tensor]:
        """The conditioning subset means nothing here: the baseline reads four numbers."""
        return {name: encoder(batch["lines"]) for name, encoder in self.encoders.items()}

    def parameter_groups(self, training) -> list[dict]:
        return [
            {"params": list(self.encoders.parameters()),
             "lr": training.lr_readout, "weight_decay": training.wd_readout},
            {"params": list(self.flows.parameters()),
             "lr": training.lr_flow, "weight_decay": training.wd_readout},
        ]


# ----------------------------------------------------------------------------- the data

def read_lines(work: str | Path, targetid: np.ndarray) -> np.ndarray:
    """The four fluxes for `targetid`, in the paper's order."""
    path = Path(work) / LINE_FEATURES
    if not path.is_file():
        raise BaselineError(f"missing {path}; run aionflow_data.line_features")
    frame = pd.read_csv(path).set_index("targetid")
    missing = np.setdiff1d(targetid, frame.index.to_numpy())
    if missing.size:
        raise BaselineError(f"{missing.size} sample targets have no line features, "
                            f"first {missing[0]}")
    rows = frame.reindex(targetid)
    return np.stack([rows[f"{line}_flux"].to_numpy(float) for line in LINES], axis=1)


def line_scaling(fluxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-split mean and scale of each line, as the model's targets are scaled."""
    scale = fluxes.std(axis=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise BaselineError(f"a line flux has no spread on the training split: {scale}")
    return fluxes.mean(axis=0), scale


class LineDataset(Dataset):
    """The targets a `TokenDataset` carries, with the four fluxes in place of tokens."""

    def __init__(self, split: Split, standardizer: Standardizer, work: str | Path,
                 mean: np.ndarray, scale: np.ndarray):
        self.split = split
        self.y = split.standardized(standardizer).astype(np.float32)
        self.lines = ((read_lines(work, split.targetid) - mean) / scale).astype(np.float32)

    def __len__(self) -> int:
        return self.split.n

    def __getitem__(self, row: int) -> dict:
        s = self.split
        return {
            "targetid": torch.tensor(s.targetid[row], dtype=torch.int64),
            "lines": torch.from_numpy(self.lines[row].copy()),
            "present": torch.from_numpy(s.present[row].copy()),
            "y": torch.from_numpy(self.y[row].copy()),
            "y_ok": torch.from_numpy(s.y_ok[row].copy()),
            "counts": torch.from_numpy(s.counts[row].copy()),
            "bkg": torch.from_numpy(s.bkg[row].copy()),
            "expo": torch.from_numpy(s.expo[row].copy()),
            "rate_ok": torch.from_numpy(s.rate_ok[row].copy()),
        }


# ----------------------------------------------------------------------------- the run

def run(cfg: dict, out: str | Path, *, recipe: str | Path = RECIPE, device: str = "cpu",
        chunk: int = 4096, workers: int = 0, max_epochs: int | None = None,
        log=print) -> dict:
    recipe = load_run(recipe)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TRAINING.seed)
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    splits = {name: Split(staged, work, name) for name in ("train", "val")}
    standardizer = Standardizer.fit(splits["train"])
    standardizer.write(out / "standardizer.json")
    mean, scale = line_scaling(read_lines(work, splits["train"].targetid))
    (out / "lines.json").write_text(json.dumps(
        {"lines": list(LINES), "mean": mean.tolist(), "scale": scale.tolist()}, indent=1) + "\n")
    model = BaselineModel(recipe, standardizer).to(device)
    datasets = {k: LineDataset(v, standardizer, work, mean, scale) for k, v in splits.items()}
    masks = validation_masks(splits["val"], TRAINING.seed)
    try:
        return fit(model, datasets, masks, out, device=device, chunk=chunk, workers=workers,
                   max_epochs=max_epochs, log=log,
                   choices={"context_encoder": "the readout's shape with Linear(4, 512) and "
                                               "no leading LayerNorm; the fluxes arrive "
                                               "standardized",
                            "lines": list(LINES)})
    finally:
        for split in splits.values():
            split.close()


def load(run_dir: str | Path, work: str | Path, device: str = "cpu"):
    """A trained baseline and the scaling it was trained with, for `evaluate`."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False)
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    scaling = json.loads((run_dir / "lines.json").read_text())
    model = BaselineModel(load_run(Path("configs") / f"{checkpoint['run']}.yaml"),
                          standardizer).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, np.asarray(scaling["mean"]), np.asarray(scaling["scale"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--run", default=RECIPE, help=f"run recipe (default {RECIPE})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=4096, help="rows per forward")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.out, recipe=args.run, device=args.device,
            chunk=args.chunk, workers=args.workers, max_epochs=args.max_epochs)
    except (BaselineError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
