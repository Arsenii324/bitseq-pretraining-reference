"""Local task specialization of the existing bidirectional predictor/head.

Not a new Transformer implementation or a scaled LLaDA architecture. Input IDs
are words0..3/MASK4/BOS5; only four word classes are predicted. The factory also
fixes cloned attention input-projection initialization locally, not upstream.
"""

import torch
from torch import nn

from tdlm.model import MaskPredictor


def make_denoiser(seed: int) -> MaskPredictor:
    # Seed only CPU's generator; torch.manual_seed would also disturb MPS RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = MaskPredictor(
            6, 7, d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.0
        )
        model.mask_id = 4
        model.head = nn.Linear(128, 4)
        model._init(model.head)
        for layer in model.enc.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
    return model


def prompt_hidden(model: MaskPredictor) -> torch.Tensor:
    """Only BOS is visible; completion-dependent prompt pooling is forbidden."""
    device = next(model.parameters()).device
    return model.hidden(torch.tensor([[5]], dtype=torch.long, device=device))
