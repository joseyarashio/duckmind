"""The panic-gait algorithm: Stage 2's critic warm-up and gated dual-teacher cloning
(:class:`~duckmind.skilltrain.s2_bc.S2Ppo`), with the walk anchor re-gated for the panic fine-tune.

* **Walk anchor** (v5, all 14 dims): upright frames (tilt < 25°) commanded at most
  ``anchor_max_vx`` (0.42 m/s: v5's own range) — strolling, turning and standing stay v5; frames commanded
  faster are free for PPO to speed up.
* **Fallen expert** (v5): frames tilted > 35° (v5 rights itself 91–98 % within 8 s); no wandb stand expert is
  needed. The teachers see the raw observation (the body block here is upstream's own small ±5 mm / ±0.05 rad
  draw, which v5 saw in training; nothing is zeroed).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from tensordict import TensorDict

from duckmind.skilltrain import pg_contract as C
from duckmind.skilltrain.s2_bc import S2Ppo, tilt_cos


class PgPpo(S2Ppo):
    def _bc_update(self) -> dict[str, float]:  # noqa: C901
        cfg = self.bc
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        cos = tilt_cos(flat)
        vx = flat[:, C.VX_SLOT]
        fallen = cos < math.cos(math.radians(float(cfg.get("fallen_tilt_deg", 35.0))))
        upright = cos > math.cos(math.radians(float(cfg.get("anchor_tilt_deg", 25.0))))
        anchor = upright & (vx <= float(cfg.get("anchor_max_vx", C.ANCHOR_MAX_VX)))
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
        w = torch.where(f.unsqueeze(1), torch.full((len(idx), n_act), float(cfg.get("fallen_coef", 1.0)),
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


__all__ = ["PgPpo"]
_ = Any
