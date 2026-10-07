"""Small configuration adapters around original GFlowNet objective implementations."""

# JustGRPO grouped-advantage/token-ratio port: Copyright(c)2026 LeapLab,
# Tsinghua University; MIT notice in ../THIRD_PARTY_NOTICES.md. Pinned grpo.py
# 1a2fddb5... is executed independently in neural derivative tests. No author
# tokenizer/CUDA/distributed generation pipeline is claimed to run verbatim.

import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.utils.modules import UniformModule
from tdlm.mdm import SurrogateCfg, draw_masks


def make_canonical_tb(env, forward_module):
    """Configure the original extensive squared TB loss, no reference or reward exponentiation."""
    pf = DiscretePolicyEstimator(forward_module, n_actions=env.n_actions)
    pb = DiscretePolicyEstimator(
        UniformModule(output_dim=env.n_actions - 1, input_dim=env.words_per_seq),
        n_actions=env.n_actions,
        is_backward=True,
    )
    return TBGFlowNet(
        pf=pf,
        pb=pb,
        init_logZ=0.0,
        log_reward_clip_min=-float("inf"),
        debug=True,
    )


def _author_group_advantages(rewards, group_size):
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    return (rewards - mean) / (std + 1e-4)


def jg_advantages(rewards):
    """Adapt group-major author's layout to explicit[M,G]; preserve arithmetic."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("finite grouped rewards[M,G>=2] required")
    m, g = rewards.shape
    flat = rewards.detach().T.contiguous().reshape(-1)
    return _author_group_advantages(flat, g).view(g, m).T.contiguous()


def jg_loss(token_logp, advantages):
    """Original online per-token clipped ratio and one-microbatch normalization."""
    if token_logp.ndim != 2 or advantages.shape != (token_logp.shape[0],):
        raise ValueError("token logps[B,L] and advantage[B] required")
    if not torch.isfinite(token_logp).all() or not torch.isfinite(advantages).all():
        raise ValueError("nonfinite JustGRPO input")
    advantages = advantages.detach()
    valid = float((advantages != 0).sum().cpu())
    scale = 1.0 / token_logp.shape[1] / (valid + 1e-5)
    ratio = (token_logp - token_logp.detach()).exp()
    clipped = ratio.clamp(0.8, 1.2)
    loss = -torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
    return loss.mul(scale).sum()


def draw_score_masks(batch, k, scheme, generator, device):
    if scheme not in ("iid", "comp") or k < 1 or (scheme == "comp" and k % 2):
        raise ValueError("IID or complete complementary mask pairs required")
    cfg = SurrogateCfg(
        n_replicates=k,
        scheme="uniform_l" if scheme == "iid" else "antithetic_l",
        scale="intensive",
    )
    return [
        (m.to(device), w.to(device))
        for m, w in draw_masks(6, batch, cfg, generator, "cpu")
    ]


def trafl_loss(policy_scores, reference_scores, rewards, logz, beta):
    """Practical Algorithm1 residual; reference/reward detached, head learned."""
    if (
        rewards.ndim != 2
        or policy_scores.numel() != rewards.numel()
        or reference_scores.shape != policy_scores.shape
    ):
        raise ValueError("score/group layout mismatch")
    advantage = rewards.detach() - rewards.detach().mean(1, keepdim=True)
    delta = (
        (policy_scores - reference_scores.detach()).reshape(rewards.shape)
        - beta * advantage
        + logz
    )
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite TraFL residual")
    return delta.square().mean()
