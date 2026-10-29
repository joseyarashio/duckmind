"""The Stage-2 algorithm (s2:build): PPO with a critic warm-up and gated dual-teacher cloning, and
the v5 import with the body slots re-conditioned.

**Why** (training plan §3.3 / §7 risk 1–2): only v5's ONNX exists — no critic, no optimiser — so a
warm start trains a fresh critic, and a fresh critic can wreck a warm-started actor; and the walk
network carries walking and self-righting, which a fine-tune must not forget. So:

* **Critic warm-up** (``critic_warmup`` iterations, 100): the actor's parameters are frozen and the
  learning-rate schedule held, PPO trains the value function only; then the actor phase starts at
  ``actor_lr`` with the recipe's adaptive-KL schedule.
* **Dual teachers** (upstream ``distill.py``'s PpoWithExpertBc, re-gated): after every PPO update a
  behaviour-cloning pass on the frames the student just collected —
  the **walk anchor** on upright walking frames (tilt < 25°, |twist| > 0.1: robotd's twist slot, which
  pose mode zeroes) toward the start network (v5 for R3; R3 for R4) on the anchored action dims (all
  14 for R3; the 10 leg dims for R4, coefficient 1.0 → 0.3, the neck free to learn the gaze), and the
  **fallen expert** (v5, which rights itself 91–98 % within 8 s) on frames tilted > 35° (all dims).
  Standing and pose frames belong to PPO alone — that is what is being learned. The teachers see the
  body slots zeroed (v5 never saw a pose command). Own Adam (lr 3e-4), 5 epochs × 4 mini-batches,
  ≥ 512 frames per mini-batch, a teacher's term skipped under 64 frames.
* **Re-conditioned body slots** (:func:`recondition`): v5's normaliser divides the body block by
  0.0129 (z) / 0.0389 (roll, pitch) — fitted to ±5 mm / ±0.05 rad — so a ±0.2 rad pose would reach
  5σ. The import rescales those three slots' statistics to the pose envelope and the first layer's
  columns by the same factor: the function is unchanged (parity-checked on observations carrying
  pose commands), the optimisation is not dominated by 5σ inputs.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from duckmind.skilltrain import rl_onnx
from duckmind.skilltrain import s2_contract as C

GRAVITY = (3, 6)
TWIST = (48, 51)
BODY = (55, 61)


# ── the import ──────────────────────────────────────────────────────────────────────────────


def recondition(
    actor: Any, new_std: Mapping[int, float], eps: float = rl_onnx.NORMALIZER_EPS
) -> dict:
    """Rescale the frozen normaliser's statistics of the given observation slots to ``new_std`` and
    the first layer's matching columns by (std' + eps) / (std + eps): x' = (o − m) / (std' + eps),
    W'x' = W x — the actor's function is unchanged for every input."""
    norm = actor.obs_normalizer
    first = actor.mlp[0]
    out = {}
    with torch.no_grad():
        for slot, std_new in new_std.items():
            old_div = (
                float(norm._std[0, slot]) + eps
                if norm._std.dim() == 2
                else float(norm._std[slot]) + eps
            )
            new_div = float(std_new) + eps
            factor = new_div / old_div
            if norm._std.dim() == 2:
                norm._std[0, slot] = float(std_new)
                norm._var[0, slot] = float(std_new) ** 2
            else:
                norm._std[slot] = float(std_new)
                norm._var[slot] = float(std_new) ** 2
            first.weight[:, slot] *= factor
            out[int(slot)] = {
                "div_old": round(old_div, 6),
                "div_new": round(new_div, 6),
                "factor": round(factor, 4),
            }
    rl_onnx.freeze_normalizer(actor)
    return out


def pose_observations(onnx_path: str | Path, n: int = 512, seed: int = 3) -> np.ndarray:
    """Observations around the ONNX's own statistics, with the body slots carrying pose commands
    inside the envelope (the parity check of the re-conditioned import)."""
    w = rl_onnx.read_onnx(onnx_path)
    rng = np.random.default_rng(seed)
    x = (w["mean"] + w["div"] * rng.standard_normal((n, w["obs_dim"]))).astype(np.float32)
    x[:, list(C.BODY_UNBOUND_SLOTS)] = 0.0
    x[:, C.BODY_Z_SLOT] = rng.uniform(C.POSE.z[0], C.POSE.z[1], n)
    x[:, C.BODY_ROLL_SLOT] = rng.uniform(-C.POSE.angle, C.POSE.angle, n)
    x[:, C.BODY_PITCH_SLOT] = rng.uniform(-C.POSE.angle, C.POSE.angle, n)
    return x


def import_v5(actor: Any, onnx_path: str | Path, *, init_std: float = C.INIT_STD,
              recondition_std: Mapping[int, float] | None = None, obs: np.ndarray | None = None) -> dict:  # fmt: skip
    """v5 → the student actor (frozen normaliser), the body slots re-conditioned, parity ≤ 1e-4 on
    random, env and pose-carrying observations (else ValueError)."""
    info = rl_onnx.load_into_actor(actor, onnx_path, init_std=init_std, frozen=True)
    info["recondition"] = recondition(actor, dict(recondition_std or C.RECONDITION_STD))
    dev = next(actor.parameters()).device
    rows = [pose_observations(onnx_path)]
    if obs is not None and len(obs):
        rows.append(np.asarray(obs, dtype=np.float32))
    par = rl_onnx.parity(onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=256,
                         obs=np.concatenate(rows))  # fmt: skip
    info["parity"] = par
    if par["max_abs"] > 1e-4:
        raise ValueError(
            f"v5 import parity {par['max_abs']:.2e} > 1e-4 (re-conditioned body slots)"
        )
    return info


def teacher_from_onnx(student: Any, onnx_path: str | Path) -> Any:
    """A frozen copy of the student's architecture holding an ONNX's network (its own normaliser)."""
    t = copy.deepcopy(student)
    rl_onnx.load_into_actor(t, onnx_path, init_std=0.1, frozen=True)
    return _freeze(t)


def teacher_from_actor(student: Any) -> Any:
    return _freeze(copy.deepcopy(student))


def _freeze(t: Any) -> Any:
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    rl_onnx.freeze_normalizer(t)
    return t


# ── the algorithm ───────────────────────────────────────────────────────────────────────────


def tilt_cos(flat: torch.Tensor) -> torch.Tensor:
    g = flat[:, GRAVITY[0] : GRAVITY[1]]
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return -g[:, 2]


def teacher_input(flat: torch.Tensor) -> torch.Tensor:
    x = flat.clone()
    x[:, BODY[0] : BODY[1]] = 0.0
    return x


class S2Ppo(PPO):
    """rsl_rl PPO + the critic warm-up + gated dual-teacher behaviour cloning (see the module doc).
    The trainer attaches the teachers with :meth:`set_teachers` after the runner is built."""

    def __init__(self, actor: Any, critic: Any, storage: Any, *args: Any, s2_cfg: dict | None = None,
                 **kwargs: Any) -> None:  # fmt: skip
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.s2 = dict(s2_cfg or {})
        self.bc = dict(self.s2.get("bc") or {})
        self.warmup_left = int(self.s2.get("critic_warmup", 0))
        self.actor_lr = float(self.s2.get("actor_lr", self.learning_rate))
        self.iteration = 0
        self.anchor: Any = None
        self.expert: Any = None
        self.bc_optimizer: torch.optim.Optimizer | None = None
        self._schedule = self.schedule
        self.last_bc: dict[str, float] = {}

    def set_teachers(self, anchor: Any, expert: Any) -> None:
        self.anchor = anchor.to(self.device) if anchor is not None else None
        self.expert = expert.to(self.device) if expert is not None else None
        if self.anchor is not None or self.expert is not None:
            self.bc_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=float(self.bc.get("lr", 3e-4))
            )

    def anchor_coef(self) -> float:
        stages = self.bc.get("anchor_coef") or ((0, 1.0),)
        value = float(stages[0][1])
        for it, c in stages:
            if self.iteration >= int(it):
                value = float(c)
        return value

    # ── one update ──
    def update(self) -> dict[str, float]:
        if self.warmup_left > 0:
            losses = self._critic_only()
            self.warmup_left -= 1
            if self.warmup_left == 0:
                self.learning_rate = self.actor_lr
                for group in self.optimizer.param_groups:
                    group["lr"] = self.actor_lr
            losses["critic_warmup"] = 1.0
        else:
            losses = super().update()
            if self.bc_optimizer is not None:
                losses.update(self._bc_update())
        self.iteration += 1
        return losses

    def _critic_only(self) -> dict[str, float]:
        params = list(self.actor.parameters())
        flags = [p.requires_grad for p in params]
        schedule, lr = self.schedule, self.learning_rate
        for p in params:
            p.requires_grad_(False)
        self.schedule = "fixed"
        try:
            losses = super().update()
        finally:
            for p, f in zip(params, flags, strict=True):
                p.requires_grad_(f)
            self.schedule = schedule
            self.learning_rate = lr
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        return losses

    def _bc_update(self) -> dict[str, float]:
        cfg = self.bc
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        cos = tilt_cos(flat)
        twist = flat[:, TWIST[0] : TWIST[1]].norm(dim=1)
        fallen = cos < math.cos(math.radians(float(cfg.get("fallen_tilt_deg", 35.0))))
        upright = cos > math.cos(math.radians(float(cfg.get("anchor_tilt_deg", 25.0))))
        anchor = upright & (twist > float(cfg.get("anchor_min_twist", C.WALKING_TWIST)))
        min_samples = int(cfg.get("min_samples", 64))
        stats = {
            "bc_fallen_frac": fallen.float().mean().item(),
            "bc_anchor_frac": anchor.float().mean().item(),
        }
        if self.expert is None or fallen.sum().item() < min_samples:
            fallen = torch.zeros_like(fallen)
        if self.anchor is None or anchor.sum().item() < min_samples:
            anchor = torch.zeros_like(anchor)
        use = fallen | anchor
        idx = use.nonzero().flatten()
        coef_a = self.anchor_coef()
        stats["bc_anchor_coef"] = coef_a
        if idx.numel() == 0:
            stats["bc_loss"] = 0.0
            self.last_bc = stats
            return stats
        sub = obs_td[idx]
        f = fallen[idx]
        dims = torch.tensor(list(cfg.get("anchor_dims") or range(C.ACTION_LEN)), device=self.device)
        dim_mask = torch.zeros(C.ACTION_LEN, device=self.device)
        dim_mask[dims] = 1.0
        with torch.no_grad():
            x = teacher_input(torch.cat([sub[g] for g in groups], dim=-1))
            td = TensorDict({groups[0]: x}, batch_size=[len(idx)])
            target = torch.zeros(len(idx), C.ACTION_LEN, device=self.device)
            if f.any():
                target[f] = self.expert(td[f])
            if (~f).any():
                target[~f] = self.anchor(td[~f])
        # per-frame dim weights: fallen → all dims × fallen_coef; anchored → anchor dims × coef
        w = torch.where(f.unsqueeze(1), torch.full((len(idx), C.ACTION_LEN), float(cfg.get("fallen_coef", 1.0)),
                        device=self.device), dim_mask.unsqueeze(0) * coef_a)  # fmt: skip
        n = len(idx)
        mb = max(int(cfg.get("min_mini_batch", 512)), n // int(cfg.get("mini_batches", 4)))
        total, steps = 0.0, 0
        for _ in range(int(cfg.get("epochs", 5))):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s : s + mb]
                pred = self.actor(sub[sel])
                denom = w[sel].sum(dim=1).clamp_min(1e-6)
                loss = (
                    (w[sel] * (pred - target[sel]) ** 2).sum(dim=1) / denom * (denom > 1e-6)
                ).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += float(loss.item())
                steps += 1
        stats["bc_loss"] = total / max(steps, 1)
        self.last_bc = stats
        return stats


def anchor_dims(stage: str) -> Sequence[int]:
    return C.BC[stage].anchor_dims


__all__ = [
    "S2Ppo",
    "import_v5",
    "pose_observations",
    "recondition",
    "teacher_from_actor",
    "teacher_from_onnx",
]
