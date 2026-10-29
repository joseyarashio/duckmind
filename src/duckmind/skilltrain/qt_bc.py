"""The queue's algorithm: the panic round's gated dual-teacher cloning (:class:`~duckmind.skilltrain.pg_bc.PgPpo`)
with the walk anchor's frames chosen per item (``bc["rule"]``):

* ``zero_pose``  (ex)    upright frames with no head / body pose command: the legs stay v5's (``anchor_dims``);
* ``not_turning`` (turn) upright frames whose yaw command is below 0.3 rad/s: walking and standing stay v5's;
* ``legs``       (calm)  every upright frame, ``anchor_dims`` = the ten leg dims (the head is free to stabilise);
* ``normal_style`` (style) upright frames whose style code is (0, 0): the normal gait stays v5's.

The fallen expert (v5, tilt > 35 degrees, all dims) is the same for every item.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from tensordict import TensorDict

from duckmind.skilltrain import qt_contract as C
from duckmind.skilltrain.s2_bc import S2Ppo, tilt_cos


def anchor_mask(rule: str, flat: torch.Tensor, upright: torch.Tensor) -> torch.Tensor:
    head = flat[:, C.HEAD_SLOTS[0] : C.HEAD_SLOTS[1]]
    if rule == "zero_pose":
        quiet = (head.abs().amax(dim=1) < 0.02) & (
            flat[:, C.SLOT_Z : C.SLOT_PITCH + 1].abs().amax(dim=1) < 0.004
        )
        return upright & quiet
    if rule == "not_turning":
        return upright & (flat[:, C.SLOT_TWIST_YAW].abs() < 0.3)
    if rule == "legs":
        return upright
    if rule == "normal_style":
        return upright & (flat[:, C.SLOT_X : C.SLOT_Y + 1].abs().amax(dim=1) < 0.2)
    raise ValueError(f"anchor rule {rule!r}")


class QtPpo(S2Ppo):
    def _bc_update(self) -> dict[str, float]:  # noqa: C901
        cfg = self.bc
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        cos = tilt_cos(flat)
        fallen = cos < math.cos(math.radians(float(cfg.get("fallen_tilt_deg", 35.0))))
        upright = cos > math.cos(math.radians(float(cfg.get("anchor_tilt_deg", 25.0))))
        anchor = anchor_mask(str(cfg.get("rule", "zero_pose")), flat, upright)
        min_samples = int(cfg.get("min_samples", 64))
        stats = {
            "bc_fallen_frac": fallen.float().mean().item(),
            "bc_anchor_frac": anchor.float().mean().item(),
        }
        if self.expert is None or fallen.sum().item() < min_samples:
            fallen = torch.zeros_like(fallen)
        if self.anchor is None or anchor.sum().item() < min_samples:
            anchor = torch.zeros_like(anchor)
        idx = (fallen | anchor).nonzero().flatten()
        coef_a = self.anchor_coef()
        stats["bc_anchor_coef"] = coef_a
        if idx.numel() == 0:
            stats["bc_loss"] = 0.0
            self.last_bc = stats
            return stats
        sub = obs_td[idx]
        f = fallen[idx]
        n_act = C.ACTION_LEN
        dims = torch.tensor(list(cfg.get("anchor_dims") or range(n_act)), device=self.device)
        dim_mask = torch.zeros(n_act, device=self.device)
        dim_mask[dims] = 1.0
        with torch.no_grad():
            target = torch.zeros(len(idx), n_act, device=self.device)
            if f.any():
                target[f] = self.expert(sub[f])
            if (~f).any():
                target[~f] = self.anchor(sub[~f])
        w = torch.where(
            f.unsqueeze(1),
            torch.full((len(idx), n_act), float(cfg.get("fallen_coef", 1.0)), device=self.device),
            dim_mask.unsqueeze(0) * coef_a,
        )
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


__all__ = ["QtPpo", "anchor_mask"]
_ = Any
