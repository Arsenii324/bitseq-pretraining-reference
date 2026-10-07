"""Normalized masked-word action adapter; sampling/replay remain torchgfn's.

The environment supplies legal actions; this module additionally fixes position
selection to uniform available positions or the leftmost available position.
Flattening RAW word logits would silently implement another position policy.
"""

import torch
from torch import nn
from gfn.estimators import DiscretePolicyEstimator


class MaskedWordActionModule(nn.Module):
    input_dim = 6
    output_dim = 25

    def __init__(self, model: nn.Module, order: str):
        super().__init__()
        if order not in ("random", "ar"):
            raise ValueError("order must be random or ar")
        self.model, self.order = model, order

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        if (
            boards.ndim != 2
            or boards.shape[1] != 6
            or boards.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("expected integer word boards[B,6]")
        if bool(((boards < -1) | (boards > 3)).any()):
            raise ValueError("sink/invalid board must not be a neural policy input")
        empty = boards == -1
        count = empty.sum(1)
        active = count > 0
        # Full boards have no model-dependent action; skip their neural forward.
        output = torch.full(
            (len(boards), 25),
            -torch.inf,
            device=boards.device,
            dtype=next(self.model.parameters()).dtype,
        )
        output[~active, 24] = 0.0
        if bool(active.any()):
            b = boards[active]
            ids = torch.cat(
                [
                    torch.full((len(b), 1), 5, device=b.device, dtype=torch.long),
                    torch.where(b == -1, 4, b).long(),
                ],
                1,
            )
            logits = self.model(ids)[:, 1:]
            if logits.shape != (len(b), 6, 4) or not torch.isfinite(logits).all():
                raise ValueError("masked predictor must return finite four-word logits")
            words = logits.log_softmax(-1)
            available = empty[active]
            if self.order == "random":
                action_logits = (
                    words - count[active].to(words.dtype).log()[:, None, None]
                )
            else:
                available = torch.nn.functional.one_hot(
                    available.long().argmax(1), 6
                ).bool()
                action_logits = words
            action_logits = action_logits.masked_fill(
                ~available[:, :, None], -torch.inf
            )
            output[active, :24] = action_logits.reshape(-1, 24)
        return output


def make_pf(model: nn.Module, order: str) -> DiscretePolicyEstimator:
    return DiscretePolicyEstimator(
        MaskedWordActionModule(model, order), n_actions=25, debug=True
    )
