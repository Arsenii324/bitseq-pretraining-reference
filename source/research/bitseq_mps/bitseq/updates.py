"""One actual native rollout/update for the declared common-model recipe arms.

No calibration or method superiority is inferred from smoke updates. Driver must
load an accepted immutable base before primary training. Position law is native
adapter-defined; JustGRPO port and paper TraFL are distinct from native TB.
"""

import copy
import hashlib
import math
import time

import torch
from gfn.samplers import Sampler
from gfn.utils.prob_calculations import get_trajectory_pfs
from tdlm.mdm import score_from_masks
from tdlm.model import LogZHead

from bitseq.env import make_env
from bitseq.artifacts import load_checkpoint, save_checkpoint
from bitseq.model import prompt_hidden
from bitseq.objectives import (
    draw_score_masks,
    jg_advantages,
    jg_loss,
    make_canonical_tb,
    trafl_loss,
)
from bitseq.policy import make_pf, MaskedWordActionModule


class RLState:
    METHODS = ("jg_ar", "pg_random", "trafl_iid", "trafl_comp", "trafl_ar", "tb_random")

    def __init__(self, base, method, *, device, seed, lr, beta=0.3):
        if method not in self.METHODS:
            raise ValueError("unknown method")
        self.method, self.device, self.seed, self.lr, self.beta = (
            method,
            device,
            seed,
            lr,
            beta,
        )
        self.order = "ar" if method in ("jg_ar", "trafl_ar") else "random"
        self.policy = copy.deepcopy(base).to(device).requires_grad_(True)
        self.reference = copy.deepcopy(base).to(device).eval().requires_grad_(False)
        self.env = make_env(device)
        self.head = None
        self.gfn = None
        self.normalizer_optimizer = None
        if method == "tb_random":
            self.gfn = make_canonical_tb(
                self.env, MaskedWordActionModule(self.policy, "random")
            ).to(device)
            self.pf = self.gfn.pf
            self.normalizer_optimizer = torch.optim.AdamW(
                [self.gfn.logZ], lr=1e-2, weight_decay=0.0
            )
        else:
            self.pf = make_pf(self.policy, self.order)
            if method.startswith("trafl"):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed + 30_000)
                    self.head = LogZHead(128).to(device)
                self.normalizer_optimizer = torch.optim.AdamW(
                    self.head.parameters(), lr=1e-2, weight_decay=0.0
                )
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.mask_rng = torch.Generator().manual_seed(seed + 10_000)
        if device == "mps":
            torch.mps.manual_seed(seed)
        elif device == "cpu":
            torch.random.default_generator.manual_seed(seed)
        else:
            raise ValueError("only declared local CPU/MPS devices")
        self.steps = 0
        self.samples = 0
        self._in_update = False
        self.forward_counts = {
            "policy_calls": 0,
            "policy_rows": 0,
            "reference_calls": 0,
            "reference_rows": 0,
            "prompt_feature_calls": 0,
        }

        def count(kind):
            def hook(module, args, output):
                self.forward_counts[kind + "_calls"] += 1
                self.forward_counts[kind + "_rows"] += len(args[0])

            return hook

        self.policy.register_forward_hook(count("policy"))
        self.reference.register_forward_hook(count("reference"))

    def _identity(self, source_hash):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.reference.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return dict(
            method=self.method,
            device=self.device,
            seed=self.seed,
            lr=self.lr,
            beta=self.beta,
            reference_sha256=digest.hexdigest(),
            source_hash=source_hash,
        )

    def save(self, path, *, source_hash, overwrite=False):
        extra = dict(
            head=None if self.head is None else self.head.state_dict(),
            logZ=None if self.gfn is None else self.gfn.logZ.detach(),
            normalizer_optimizer=None
            if self.normalizer_optimizer is None
            else self.normalizer_optimizer.state_dict(),
            forward_counts=self.forward_counts.copy(),
        )
        save_checkpoint(
            path,
            self.policy,
            self.optimizer,
            {"mask": self.mask_rng},
            dict(
                **self._identity(source_hash),
                step=self.steps,
                samples=self.samples,
                optimizer_steps=self.steps,
                checkpoint_phase="forensic_partial" if self._in_update else "committed",
            ),
            extra=extra,
            overwrite=overwrite,
        )

    def restore(self, path, *, source_hash):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["metadata"].get("checkpoint_phase") != "committed":
            raise ValueError("Checkpoint phase is not a committed update boundary")
        expected = self._identity(source_hash)
        if any(record["metadata"].get(k) != v for k, v in expected.items()):
            raise ValueError("checkpoint identity mismatch")
        extra = record["extra"]
        if (
            (extra["head"] is None) != (self.head is None)
            or (extra["logZ"] is None) != (self.gfn is None)
            or (extra["normalizer_optimizer"] is None)
            != (self.normalizer_optimizer is None)
        ):
            raise ValueError("normalizer checkpoint identity mismatch")
        metadata = load_checkpoint(
            path, self.policy, self.optimizer, {"mask": self.mask_rng}, expected
        )
        if self.head is not None:
            self.head.load_state_dict(extra["head"])
        if self.gfn is not None:
            with torch.no_grad():
                self.gfn.logZ.copy_(extra["logZ"].to(self.device))
        if self.normalizer_optimizer is not None:
            self.normalizer_optimizer.load_state_dict(extra["normalizer_optimizer"])
        self.forward_counts = extra["forward_counts"].copy()
        self.steps, self.samples = metadata["step"], metadata["samples"]
        self._in_update = False

    def step(self):
        if self._in_update:
            raise RuntimeError(
                "Failed/partial update; restore a committed checkpoint first"
            )
        self._in_update = True
        started = time.monotonic()
        before_counts = self.forward_counts.copy()
        self.policy.eval()
        with torch.no_grad():
            paths = Sampler(self.pf).sample_trajectories(
                self.env, n=80, save_logprobs=False
            )
            y = paths.terminating_states.tensor
            rewards = self.env.reward(paths.terminating_states).reshape(16, 5)
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.normalizer_optimizer is not None:
            self.normalizer_optimizer.zero_grad(set_to_none=True)
        additional = {}
        if self.method in ("jg_ar", "pg_random"):
            lp = get_trajectory_pfs(self.pf, paths, recalculate_all_logprobs=True)[:6].T
            if self.order == "random":
                lp = lp + torch.tensor(
                    [math.log(6 - i) for i in range(6)], device=self.device
                )
            advantage = jg_advantages(rewards)
            loss = jg_loss(lp, advantage.flatten())
            additional = dict(
                valid_advantages=int((advantage != 0).sum().cpu()),
                zero_advantage_fraction=float((advantage == 0).float().mean().cpu()),
                mean_group_std=float(rewards.std(1).mean().cpu()),
                mean_std_over_epsilon=float((rewards.std(1) / 1e-4).mean().cpu()),
            )
        elif self.method.startswith("trafl"):
            prompt = torch.full((80, 1), 5, device=self.device)
            masks = draw_score_masks(
                80,
                4,
                "comp" if self.method == "trafl_comp" else "iid",
                self.mask_rng,
                self.device,
            )
            policy_scores = score_from_masks(self.policy, prompt, y, masks)
            with torch.no_grad():
                ref_scores = score_from_masks(self.reference, prompt, y, masks)
            self.forward_counts["prompt_feature_calls"] += 1
            z = self.head(prompt_hidden(self.policy))
            loss = trafl_loss(policy_scores, ref_scores, rewards, z, self.beta)
            ratio = policy_scores.detach() - ref_scores
            additional = dict(
                score_ratio_mean=float(ratio.mean().cpu()),
                score_ratio_std=float(ratio.std().cpu()),
                logZ=float(z.detach().cpu()),
            )
        else:
            loss = self.gfn.loss(self.env, paths, recalculate_all_logprobs=True)
            additional = dict(logZ=float(self.gfn.logZ.detach().cpu()))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite RL loss")
        loss.backward()
        old = [p.detach().clone() for p in self.policy.parameters()]
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 1.0, error_if_nonfinite=True
        )
        z_norm = None
        if self.normalizer_optimizer is not None:
            zparams = (
                list(self.head.parameters())
                if self.head is not None
                else [self.gfn.logZ]
            )
            z_norm = torch.nn.utils.clip_grad_norm_(
                zparams, 1.0, error_if_nonfinite=True
            )
        self.optimizer.step()
        if self.normalizer_optimizer is not None:
            self.normalizer_optimizer.step()
        delta = sum(
            (p.detach() - oldp).square().sum()
            for p, oldp in zip(self.policy.parameters(), old)
        ).sqrt()
        if not torch.isfinite(
            torch.cat([p.detach().flatten() for p in self.policy.parameters()])
        ).all():
            raise RuntimeError("nonfinite policy parameters after update")
        if self.device == "mps":
            torch.mps.synchronize()
        self.steps += 1
        self.samples += 80
        self._in_update = False
        return dict(
            step=self.steps,
            samples=self.samples,
            optimizer_steps=self.steps,
            reward=float(rewards.mean().cpu()),
            loss=float(loss.detach().cpu()),
            preclip_norm=float(norm.detach().cpu()),
            clipping_factor=min(1.0, 1.0 / (float(norm.detach().cpu()) + 1e-6)),
            update_norm=float(delta.cpu()),
            normalizer_preclip_norm=None
            if z_norm is None
            else float(z_norm.detach().cpu()),
            update_seconds=time.monotonic() - started,
            forward_counts={
                key: value - before_counts[key]
                for key, value in self.forward_counts.items()
            },
            **additional,
        )
