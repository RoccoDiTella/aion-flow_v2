"""Appendix B: the pooling comparison and the random-encoder control.

"We compare the CLS read against a bare attentive probe, one learned query with
single-head cross-attention over the final tokens as in AION-1's own probing
protocol, and a masked mean over the same tokens. All three share the readout
MLPs and flows, the two heads, flux and log LX, the split, schedule, seed, batch
layout and sampled modality dropout. The attentive probe has a per-modality
affine on the tokens and a presence embedding on the query. Trained parameters
are 6.8M for the CLS read, 5.6M for the probe and 3.3M for the mean pool."

"The control runs the same mean pool and heads on the pretrained encoder and on a
randomly initialized frozen encoder of identical architecture, with all four
modalities always present."

    python -m aionflow_model.ablations --arm mean --out runs/pool-mean
    python -m aionflow_model.ablations --arm mean --out runs/random --random-encoder
    python -m aionflow_model.ablations --arm attentive --out runs/pool-attentive
    python -m aionflow_model.ablations --arm cls --out runs/pool-cls

The main path never imports this module: `Model` is the one architecture the
package trains, and these arms exist to say what is lost by reading the encoder
differently. Only the two comparisons that test a claim are here. The four-token
and cosine-schedule grid of the same appendix is a hyperparameter search rather
than an architecture claim, and is deliberately absent.

"Bare" is the load-bearing word in "a bare attentive probe": one query, one head,
cross-attention only. No self-attention among queries, no feed-forward, no second
layer. The parameter count is the check - Q, K, V and the output projection at
768 by 768, plus the query, the presence embedding and the per-modality affine,
come to 5,625,456 with two heads, which is the paper's 5.6M. Adding a
feed-forward or dropping the output projection both miss it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from aionflow_data.common import load_config

from .config import TRAINING, load_run, recipe_path
from .data import MODALITIES, TOKEN_KEYS, Split, Standardizer, TokenDataset
from .encoder import backbone_width, readout, token_inputs
from .flows import FlowHead
from .objective import Heads, Model
from .train import fit, validation_masks

ARMS = ("cls", "attentive", "mean")
RECIPE = recipe_path("pooling")


class AblationError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- the tokens

def modality_index(backbone, mod_mask: Tensor) -> Tensor:
    """Our 0..3 modality per token, from AION's own ids; -1 where there is no token."""
    out = torch.full_like(mod_mask, -1, dtype=torch.long)
    for m, modality in enumerate(MODALITIES):
        for key, _ in TOKEN_KEYS[modality]:
            out[mod_mask == int(backbone.modality_info[key]["id"])] = m
    return out


def frozen_tokens(backbone, batch: dict, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The encoder's final tokens, which of them are real, and what modality each is.

    The same frozen forward the CLS read watches, run to completion instead: these
    arms read the output rather than every block.
    """
    tokens, hidden, needed = token_inputs(batch, mask)
    with torch.no_grad():
        x, emb, token_mask, mod_mask = backbone.embed_inputs(
            tokens, mask=hidden, num_encoder_tokens=needed)
        final = backbone.forward_encoder(x + emb, token_mask)
    return final, ~token_mask.squeeze(1), modality_index(backbone, mod_mask)


# ----------------------------------------------------------------------------- the arms

class Pooled(Heads):
    """A pooler over the frozen encoder's final tokens, then the same readouts and flows."""

    def __init__(self, backbone, run, standardizer: Standardizer):
        super().__init__()
        self.run = run
        self.standardizer = standardizer
        self.backbone = backbone.eval().requires_grad_(False)
        self.width = backbone_width(backbone)
        self.readouts = nn.ModuleDict({h.name: readout(self.width) for h in run.heads})
        self.flows = nn.ModuleDict({h.name: FlowHead(len(h.targets)) for h in run.heads})

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def pool(self, tokens: Tensor, valid: Tensor, modality: Tensor, present: Tensor) -> Tensor:
        raise NotImplementedError

    def contexts(self, batch: dict, mask: Tensor) -> dict[str, Tensor]:
        tokens, valid, modality = frozen_tokens(self.backbone, batch, mask)
        summary = self.pool(tokens, valid, modality, mask)
        return {name: head(summary) for name, head in self.readouts.items()}

    def parameter_groups(self, training) -> list[dict]:
        """No adapters here, so everything runs at the base rate: "its zero-initialized
        deltas run at a learning rate tuned for them, the other two readouts at the
        base rate"."""
        base = [p for name, p in self.named_parameters()
                if p.requires_grad and not name.startswith("flows.")]
        return [
            {"params": base, "lr": training.lr_readout, "weight_decay": training.wd_readout},
            {"params": list(self.flows.parameters()),
             "lr": training.lr_flow, "weight_decay": training.wd_readout},
        ]


class MeanPool(Pooled):
    """A masked mean over the final tokens."""

    def pool(self, tokens: Tensor, valid: Tensor, modality: Tensor, present: Tensor) -> Tensor:
        weight = valid.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weight).sum(1) / weight.sum(1).clamp(min=1.0)


class AttentivePool(Pooled):
    """One learned query, single-head cross-attention over the final tokens."""

    def __init__(self, backbone, run, standardizer: Standardizer):
        super().__init__(backbone, run, standardizer)
        width = self.width
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.presence = nn.Parameter(torch.zeros(len(MODALITIES), width))
        self.scale = nn.Parameter(torch.ones(len(MODALITIES), width))
        self.shift = nn.Parameter(torch.zeros(len(MODALITIES), width))
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def pool(self, tokens: Tensor, valid: Tensor, modality: Tensor, present: Tensor) -> Tensor:
        index = modality.clamp(min=0)                     # pad tokens are masked out below
        x = tokens * self.scale[index] + self.shift[index]
        query = self.query + (present.to(tokens.dtype) @ self.presence)
        q = self.to_q(query).unsqueeze(1)                 # (B, 1, D)
        k, v = self.to_k(x), self.to_v(x)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(tokens.shape[-1])
        logits = logits.masked_fill(~valid.unsqueeze(1), -torch.finfo(logits.dtype).max)
        return self.proj((logits.softmax(-1) @ v).squeeze(1))


# ----------------------------------------------------------------------------- the control

def randomize_encoder(backbone, seed: int = TRAINING.seed) -> int:
    """Re-initialize the frozen encoder in place: identical architecture, no pretraining.

    Everything the forward pass touches is reset - the per-modality token
    embeddings, the blocks and the output norm - so what is left is the
    architecture and nothing it learned.
    """
    torch.manual_seed(seed)
    reset = 0
    parts = [backbone.encoder, backbone.encoder_norm]
    if hasattr(backbone, "encoder_embeddings"):
        parts.append(backbone.encoder_embeddings)
    for part in parts:
        for module in part.modules():
            if callable(getattr(module, "reset_parameters", None)):
                module.reset_parameters()
                reset += 1
    backbone.eval().requires_grad_(False)
    return reset


def build(arm: str, backbone, run, standardizer: Standardizer) -> Heads:
    if arm not in ARMS:
        raise AblationError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "cls":
        return Model(backbone, run, standardizer)
    return (MeanPool if arm == "mean" else AttentivePool)(backbone, run, standardizer)


# ----------------------------------------------------------------------------- the run

def run(cfg: dict, out: str | Path, *, arm: str = "mean", random_encoder: bool = False,
        recipe: str | Path = RECIPE, device: str = "cpu", chunk: int = 448,
        workers: int = 0, max_epochs: int | None = None, backbone=None, log=print) -> dict:
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
    reset = randomize_encoder(backbone) if random_encoder else 0
    model = build(arm, backbone, recipe, standardizer).to(device)
    datasets = {k: TokenDataset(v, standardizer) for k, v in splits.items()}
    masks = validation_masks(splits["val"], TRAINING.seed)
    if random_encoder:
        # "with all four modalities always present"
        masks = {k: torch.ones_like(v) for k, v in masks.items()}
    try:
        return fit(model, datasets, masks, out, device=device, chunk=chunk, workers=workers,
                   max_epochs=max_epochs, log=log,
                   choices={"ablation_arm": arm,
                            "random_encoder": random_encoder,
                            "modules_reinitialized": reset,
                            "trained_parameters": sum(p.numel() for p in model.parameters()
                                                      if p.requires_grad),
                            "note": "Appendix B only; not one of the reported runs"})
    finally:
        for split in splits.values():
            split.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--arm", choices=ARMS, default="mean", help="how the encoder is read")
    parser.add_argument("--random-encoder", action="store_true",
                        help="the control: reinitialize the frozen encoder first")
    parser.add_argument("--run", default=str(RECIPE), help=f"run recipe (default {RECIPE})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=448, help="rows per forward")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.out, arm=args.arm,
            random_encoder=args.random_encoder, recipe=args.run, device=args.device,
            chunk=args.chunk, workers=args.workers, max_epochs=args.max_epochs)
    except (AblationError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
