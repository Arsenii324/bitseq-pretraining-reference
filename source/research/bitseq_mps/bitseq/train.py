"""Verified denoising loss primitives; source mask/scoring functions are reused."""

import torch

from tdlm.mdm import SurrogateCfg, draw_masks, score_from_masks


def draw_pretraining_masks(batch: int, generator: torch.Generator):
    cfg = SurrogateCfg(n_replicates=1, scheme="uniform_l", scale="extensive")
    return draw_masks(6, batch, cfg, generator, "cpu")[0][0]


def pretraining_loss(model, y, masks):
    if (
        y.ndim != 2
        or y.shape[1] != 6
        or masks.shape != y.shape
        or masks.dtype != torch.bool
    ):
        raise ValueError("word batch and Boolean masks[B,6] required")
    if y.dtype not in (torch.int32, torch.int64) or bool(((y < 0) | (y > 3)).any()):
        raise ValueError("only full four-word terminal boards are denoising labels")
    if bool((masks.sum(1) == 0).any()):
        raise ValueError("pretraining mask must be nonempty")
    prompt = torch.full((len(y), 1), 5, dtype=torch.long, device=y.device)
    weights = 6 / masks.sum(1).float()
    return -score_from_masks(model, prompt, y, [(masks, weights)]).mean()


def soft_pretraining_loss(model, boards, targets, *, reduction="mean"):
    """Conditional expectation of the same hard-label loss, not a new target.

    Contexts still follow q_pre and uniform mask levels. Four-class teacher
    marginals integrate out the masked labels, preserving the population NELBO.
    """
    if boards.ndim != 2 or boards.shape[1] != 6 or targets.shape != (len(boards), 6, 4):
        raise ValueError("partial boards[B,6] and four-word teacher marginals required")
    if boards.dtype not in (torch.int32, torch.int64) or bool(
        ((boards < -1) | (boards > 3)).any()
    ):
        raise ValueError("invalid partial board")
    masks = boards == -1
    if (
        bool((masks.sum(1) == 0).any())
        or not torch.isfinite(targets).all()
        or bool((targets < 0).any())
    ):
        raise ValueError("nonempty masks and finite teacher probabilities required")
    if bool((targets.sum(-1) - 1).abs().max() > 1e-5):
        raise ValueError("teacher marginals must normalize")
    prompt = torch.full((len(boards), 1), 5, dtype=torch.long, device=boards.device)
    from tdlm.mdm import build_z

    ids = build_z(prompt, boards.clamp(min=0), masks, model.mask_id)
    lp = model(ids)[:, 1:].float().log_softmax(-1)
    cross_entropy = -(targets.detach() * lp).sum(-1)
    losses = cross_entropy.masked_fill(~masks, 0).sum(1) * 6 / masks.sum(1)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("unknown reduction")
