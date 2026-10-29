"""The simulator's XML servo as a per-episode alternative to the BAM XL330 in the training physics.

The panic-gait round 1 trained on the BAM model only; on the XML position servos of the owner's simulator
(kp 0.55 N m/rad, kv 0, force range ±0.96 N m, joint damping 0.053, dry friction 0.0048 N m: ``chosen_actuator``
in the official MJCF) its 40 ms falls were far worse than v5's. :class:`XmlMixBamActuator` is the BAM actuator
with a per-env flag: a flagged env gets the XML servo's torque law and friction instead (the BAM friction
budget is overwritten with the XML's constants after BAM wrote it), and a nominal 7.4 V so robotd's voltage
adaptation is the identity there, as in the live simulator. :class:`PgAction` draws the flag at every reset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator, FrictionDRBamActuatorCfg

from duckmind.skilltrain.s2_mdp import S2Action, S2ActionCfg

XML_KP = 0.55  # N m / rad (position servo, gain 200)
XML_FORCE = 0.96  # N m
XML_DAMPING = 0.053
XML_FRICTIONLOSS = 0.0048
XML_VIN = 7.4  # the live simulator's nominal battery: voltage adaptation = 1


class XmlMixBamActuator(FrictionDRBamActuator):
    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        n = self.kp_scale.shape[0]
        self.xml = torch.zeros(n, 1, dtype=torch.bool, device=device)
        self.vin_bam = self.vin_tensor.clone()

    def set_xml(self, env_ids: Any, flag: torch.Tensor) -> None:
        """``flag`` (m,) bool for the given envs."""
        f = flag.reshape(-1, 1)
        self.xml[env_ids] = f
        self.vin_tensor[env_ids] = torch.where(
            f, torch.full_like(self.vin_bam[env_ids], XML_VIN), self.vin_bam[env_ids]
        )

    def compute(self, cmd: Any) -> torch.Tensor:
        bam_tau = super().compute(cmd)
        xml = self.xml
        if not bool(xml.any()):
            return bam_tau
        # the XML servo's torque: a plain position servo, no velocity term (kv 0), clipped to its force range
        tau = (XML_KP * self.kp_scale * (cmd.position_target - cmd.pos)).clamp(
            -XML_FORCE, XML_FORCE
        )
        fl = self._mjwarp_model.dof_frictionloss
        dm = self._mjwarp_model.dof_damping
        fl[:, self._dof_ids] = torch.where(
            xml, torch.full_like(bam_tau, XML_FRICTIONLOSS), fl[:, self._dof_ids]
        )
        dm[:, self._dof_ids] = torch.where(
            xml, torch.full_like(bam_tau, XML_DAMPING), dm[:, self._dof_ids]
        )
        return torch.where(xml, tau, bam_tau)


@dataclass(kw_only=True)
class XmlMixBamActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> XmlMixBamActuator:
        return XmlMixBamActuator(self, entity, target_ids, target_names)


def mix_actuators(cfg_actuators: tuple) -> tuple:
    """The same actuator configs, as :class:`XmlMixBamActuatorCfg` (every init field kept)."""
    out = []
    for a in cfg_actuators:
        kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.init}
        out.append(XmlMixBamActuatorCfg(**kw))
    return tuple(out)


@dataclass(kw_only=True)
class PgActionCfg(S2ActionCfg):
    p_xml: float = 0.5

    def build(self, env: Any) -> PgAction:
        return PgAction(self, env)


class PgAction(S2Action):
    """Stage 2's action term + the per-episode servo model (XML with probability ``p_xml``)."""

    cfg: PgActionCfg

    def __init__(self, cfg: PgActionCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self._mix = [a for a in self._entity.actuators if hasattr(a, "set_xml")]
        self.xml = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        if not self._mix:
            return
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = env_ids
        if len(ids) == 0:
            return
        flag = torch.rand(len(ids), device=self.device) < self.cfg.p_xml
        self.xml[ids] = flag
        for a in self._mix:
            a.set_xml(ids, flag)


def _fast_xml(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cmd = env.command_manager.get_command("twist")
    xml = env.action_manager.get_term("joint_pos").xml
    return (cmd[:, 0] >= 0.8) & xml, cmd


def m_xml(env: Any) -> torch.Tensor:
    return env.action_manager.get_term("joint_pos").xml.float()


def m_fast_xml_n(env: Any) -> torch.Tensor:
    on, _ = _fast_xml(env)
    return on.float()


def m_fast_xml_cmd(env: Any) -> torch.Tensor:
    on, cmd = _fast_xml(env)
    return torch.where(on, cmd[:, 0], torch.zeros_like(cmd[:, 0]))


def m_fast_xml_vx(env: Any) -> torch.Tensor:
    on, cmd = _fast_xml(env)
    return torch.where(
        on, env.scene["robot"].data.root_link_lin_vel_b[:, 0], torch.zeros_like(cmd[:, 0])
    )
