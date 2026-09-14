"""A small stand-in for AION-1-B, so the probe's tests need no frozen weights.

It uses AION's own `Block`, so the read path is exercised against the real
attention, and mimics `embed_inputs`: per-modality token embeddings concatenated
in a fixed order, reordered valid-first per row and truncated, with a (B, 1, N)
mask that is True where a token is not there.
"""

from __future__ import annotations

import torch
from aion.fourm.fm_utils import Block
from torch import Tensor, nn

from aionflow_model.data import ALL_TOKEN_KEYS, TOKEN_SIZES

VOCABULARY = 1024


class FakeBackbone(nn.Module):
    def __init__(self, width: int = 96, heads: int = 4, depth: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.depth = width, depth
        self.embeddings = nn.ModuleDict(
            {key: nn.Embedding(VOCABULARY, width) for key in ALL_TOKEN_KEYS})
        self.positions = nn.ParameterDict(
            {key: nn.Parameter(torch.randn(TOKEN_SIZES[key], width) * 0.02)
             for key in ALL_TOKEN_KEYS})
        self.encoder = nn.ModuleList(
            Block(width, num_heads=heads, qkv_bias=False, proj_bias=False, mlp_bias=False,
                  gated_mlp=True, qk_norm=True) for _ in range(depth))
        self.encoder_norm = nn.LayerNorm(width)
        # AION gives every token key its own arbitrary global id; the ablation arms
        # map those back to our four modalities, so the stand-in must have them too.
        self.modality_info = {key: {"id": 1000 + 7 * i}
                              for i, key in enumerate(ALL_TOKEN_KEYS)}

    def embed_inputs(self, input_dict: dict, mask: dict | None = None,
                     num_encoder_tokens: int = 256):
        mask = mask or {}
        rows = next(iter(input_dict.values())).shape[0]
        tokens, embeddings, hidden, mods = [], [], [], []
        for key in ALL_TOKEN_KEYS:
            ids = input_dict[key].to(torch.long)
            tokens.append(self.embeddings[key](ids))
            embeddings.append(self.positions[key].expand(rows, -1, -1))
            hidden.append(mask.get(key, torch.zeros(ids.shape, dtype=torch.bool)))
            mods.append(torch.full(ids.shape, self.modality_info[key]["id"],
                                   dtype=torch.long))
        tokens = torch.cat(tokens, 1)
        embeddings = torch.cat(embeddings, 1)
        hidden = torch.cat(hidden, 1)
        mods = torch.cat(mods, 1)
        keep = torch.argsort(hidden.long(), dim=1, stable=True)[:, :num_encoder_tokens]
        tokens = torch.gather(tokens, 1, keep[..., None].expand(-1, -1, self.width))
        embeddings = torch.gather(embeddings, 1, keep[..., None].expand(-1, -1, self.width))
        mods = torch.gather(mods, 1, keep)
        hidden = torch.gather(hidden, 1, keep)
        tokens = tokens.masked_fill(hidden[..., None], 0.0)
        embeddings = embeddings.masked_fill(hidden[..., None], 0.0)
        return tokens, embeddings, hidden.unsqueeze(1), mods.masked_fill(hidden, -1)

    def forward_encoder(self, x: Tensor, encoder_mask: Tensor) -> Tensor:
        for block in self.encoder:
            x = block(x, mask=encoder_mask)
        return self.encoder_norm(x)


class ShapedBackbone(nn.Module):
    """Only the shape of AION-1-B: twelve blocks, so a probe built on it has exactly
    the trained parameters the paper counts, with no 318M of frozen weights to build."""

    def __init__(self, depth: int = 12, width: int = 768):
        super().__init__()
        self.encoder = nn.ModuleList(nn.Identity() for _ in range(depth))
        self.encoder_norm = nn.LayerNorm(width)
        self.modality_info = {key: {"id": 1000 + 7 * i}
                              for i, key in enumerate(ALL_TOKEN_KEYS)}
