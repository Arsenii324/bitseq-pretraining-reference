"""Tiny bidirectional masked-diffusion language model (LLaDA-shaped, toy-sized).

The model is a *mask predictor*: given a sequence in which some completion positions hold the
MASK id, it returns logits for every position.  Prompt positions are never masked.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MaskPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size          # includes the MASK id
        self.mask_id = vocab_size - 1
        self.max_len = max_len
        self.d_model = d_model
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size - 1)   # never predict MASK
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        p = torch.arange(T, device=ids.device)
        h = self.tok(ids) + self.pos(p)[None]
        h = self.enc(h)
        return self.norm(h)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B,T) int64 -> logits (B,T,V-1)."""
        return self.head(self.hidden(ids))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LogZHead(nn.Module):
    """TraFL's partition head: 2-layer MLP over mean-pooled *prompt* hidden states.

    The backbone features are detached, exactly as the paper specifies ("with backbone features
    detached so that optimizing the partition head does not alter the backbone's prompt
    representations", §4.1).
    """

    def __init__(self, d_model: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prompt_hidden: torch.Tensor) -> torch.Tensor:
        """prompt_hidden: (B, P, d) -> (B,) log Z."""
        return self.net(prompt_hidden.detach().mean(dim=1)).squeeze(-1)
