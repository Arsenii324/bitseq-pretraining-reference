"""Frozen task construction and input guard around the original torchgfn environment."""

from __future__ import annotations

import torch
from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence


MODES = (
    "000000000000",
    "101010110101",
    "011001101100",
    "110011011001",
    "000111100011",
    "101101010110",
    "011110001111",
    "110100111010",
)


def make_env(device: str = "cpu") -> NonAutoregressiveBitSequence:
    """Construct the unmodified upstream class with protocol's explicit modes/reward."""
    modes = torch.tensor([[int(bit) for bit in s] for s in MODES], dtype=torch.long)
    return NonAutoregressiveBitSequence(
        word_size=2,
        seq_size=12,
        n_modes=8,
        reward_exponent=24.0,
        H=modes,
        device_str=device,
        debug=True,
    )


def terminal_log_reward(
    env: NonAutoregressiveBitSequence, y: torch.Tensor
) -> torch.Tensor:
    """Validate our rewarded-terminal boundary; delegate the actual reward unchanged."""
    if y.ndim != 2 or y.shape[1] != env.words_per_seq:
        raise ValueError("expected a batch of full word boards")
    if y.dtype not in (torch.int32, torch.int64):
        raise ValueError("word boards must have integer dtype")
    if bool(((y < 0) | (y >= env.n_words)).any()):
        raise ValueError("partial or sink boards are not rewarded terminals")
    return env.log_reward(env.States(y))
