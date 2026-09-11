"""The read-only CLS probe on the frozen AION-1-B encoder, and the readout MLPs.

Appendix B, equations (2) to (4). With X the unmasked data tokens of a source
and a hat the block's frozen pre-norm, the data path is unchanged,

    q = W_Q x_n,   K = W_K X,   V = W_V X,

and the extra token reads them through its own copy of the same projections,

    q = (W_Q + dQ) x_cls,   K = (W_K + dK) X,   V = (W_V + dV) X,
    Attn(q, K, V) = V softmax(K' q / sqrt(d_h)),

with the block's QK norm acting on q and K after the deltas, then the frozen
output projection, residual and MLP. "The CLS is in neither K nor V, so the data
tokens' attention is the pretrained model's exactly and the CLS does not attend
to itself." Each delta is dW = B A with A in R^{64 x d} drawn N(0, 1/d) and B
zero, so the read starts as the frozen attention. The context is
c_t = MLP_t(x_cls after the encoder's output LayerNorm).

Two consequences of writing nothing back are worth naming. The data stream is
run by AION's own `Block.forward`, so it cannot diverge from the pretrained
model. And nothing trainable feeds it, so it needs no gradient at all: the whole
data pass runs under `no_grad`, and only the CLS's own thin stream is kept for
backward. Masking a modality is the encoder's native token mask, so a hidden
modality contributes no key.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Run
from .data import MODALITIES, TOKEN_KEYS, TOKEN_SIZES
from .flows import CONTEXT

RANK = 64
WIDTH = 768
HIDDEN = 512
DROPOUT = 0.05
CLS_INIT_STD = 0.02
BACKBONE_REPO = "polymathic-ai/aion-base"


class EncoderError(RuntimeError):
    pass


def load_backbone(repo: str = BACKBONE_REPO):
    """The frozen AION-1-B encoder, in eval mode with no gradients."""
    from aion import AION

    backbone = AION.from_pretrained(repo)
    backbone.eval().requires_grad_(False)
    return backbone


# ----------------------------------------------------------------------------- the read

class Delta(nn.Module):
    """One rank-64 delta, dW = B A, applied as a correction to a frozen projection."""

    def __init__(self, width: int = WIDTH, rank: int = RANK):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank, width) / width**0.5)   # N(0, 1/d)
        self.b = nn.Parameter(torch.zeros(width, rank))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.a), self.b)


class BlockRead(nn.Module):
    """The CLS's read path through one frozen block: its three deltas."""

    def __init__(self, width: int = WIDTH, rank: int = RANK):
        super().__init__()
        self.query = Delta(width, rank)
        self.key = Delta(width, rank)
        self.value = Delta(width, rank)

    def forward(self, block, cls_hat: Tensor, x_hat: Tensor, mask: Tensor | None) -> Tensor:
        """Attention of the single CLS query over the data tokens, then W_O."""
        attn = block.attn
        if attn.qkv.bias is not None or attn.proj.bias is not None:
            raise EncoderError("this backbone has projection biases; the paper's has none")
        if getattr(attn, "allow_zero_attn", False):
            raise EncoderError("this backbone attends to a zero token; the paper's does not")
        rows, tokens, width = x_hat.shape
        heads = attn.num_heads
        head_dim = width // heads
        w_q, w_k, w_v = attn.qkv.weight.split(width, dim=0)
        q = F.linear(cls_hat, w_q) + self.query(cls_hat)
        k = F.linear(x_hat, w_k) + self.key(x_hat)
        v = F.linear(x_hat, w_v) + self.value(x_hat)
        q = attn.q_norm(q.reshape(rows, 1, heads, head_dim).transpose(1, 2))
        k = attn.k_norm(k.reshape(rows, tokens, heads, head_dim).transpose(1, 2))
        v = v.reshape(rows, tokens, heads, head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * attn.scale
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), -torch.finfo(logits.dtype).max)
        out = logits.softmax(dim=-1) @ v
        return attn.proj(out.transpose(1, 2).reshape(rows, 1, width))


def readout(width: int = WIDTH, hidden: int = HIDDEN, context: int = CONTEXT,
            dropout: float = DROPOUT) -> nn.Sequential:
    """The per-target readout: 528,128 parameters."""
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, context),
        nn.LayerNorm(context),
    )


# ----------------------------------------------------------------------------- the probe

def backbone_width(backbone) -> int:
    """The residual width the CLS has to share with the data tokens."""
    norm = getattr(backbone, "encoder_norm", None)
    shape = getattr(norm, "normalized_shape", None)
    if not shape:
        raise EncoderError("this backbone has no encoder_norm to take a width from")
    return int(shape[0])


class Probe(nn.Module):
    """The frozen backbone, the CLS token that reads it, and one readout per head."""

    def __init__(self, backbone, run: Run, rank: int = RANK):
        super().__init__()
        width = backbone_width(backbone)
        self.backbone = backbone.eval().requires_grad_(False)
        self.cls = nn.Parameter(torch.randn(width) * CLS_INIT_STD)
        self.reads = nn.ModuleList(BlockRead(width, rank) for _ in backbone.encoder)
        self.readouts = nn.ModuleDict({head.name: readout(width) for head in run.heads})
        self.width = width

    def train(self, mode: bool = True):
        """The backbone never leaves eval: it is frozen, dropout included."""
        super().train(mode)
        self.backbone.eval()
        return self

    def inputs(self, batch: dict, mask: Tensor) -> tuple[dict, dict, int]:
        """Token ids, the per-modality mask, and how many tokens the batch needs."""
        if not bool(mask.any(dim=1).all()):
            raise EncoderError("a source was given no modality to condition on")
        tokens, hidden, visible = {}, {}, torch.zeros(mask.shape[0], dtype=torch.int64,
                                                      device=mask.device)
        for m, modality in enumerate(MODALITIES):
            for key, _ in TOKEN_KEYS[modality]:
                tokens[key] = batch[key]
                hidden[key] = (~mask[:, m])[:, None].expand(-1, TOKEN_SIZES[key])
                visible = visible + mask[:, m].long() * TOKEN_SIZES[key]
        return tokens, hidden, int(visible.max())

    def context(self, batch: dict, mask: Tensor) -> Tensor:
        """The final CLS state, after the encoder's output LayerNorm."""
        tokens, hidden, needed = self.inputs(batch, mask)
        with torch.no_grad():
            x, emb, token_mask, _ = self.backbone.embed_inputs(
                tokens, mask=hidden, num_encoder_tokens=needed)
            x = x + emb
        cls = self.cls.to(x.dtype).expand(x.shape[0], 1, -1)
        for block, read in zip(self.backbone.encoder, self.reads):
            with torch.no_grad():
                x_hat = block.norm1(x)
            cls = cls + block.drop_path(read(block, block.norm1(cls), x_hat, token_mask))
            cls = cls + block.drop_path(block.mlp(block.norm2(cls)))
            with torch.no_grad():
                x = block(x, mask=token_mask)
        return self.backbone.encoder_norm(cls).squeeze(1)

    def forward(self, batch: dict, mask: Tensor) -> dict[str, Tensor]:
        """One context vector per head."""
        state = self.context(batch, mask)
        return {name: head(state) for name, head in self.readouts.items()}

    def trained_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
