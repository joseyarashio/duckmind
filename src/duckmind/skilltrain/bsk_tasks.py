"""The three B-skill mjlab tasks (b:feasibility, 2026-09-28), built like Stage 1's (``rl_tasks``).

From ``make_microduck_velstand_env_cfg`` (read-only upstream: ``robot_allcollisions``, the BAM XL330
actuator, the velocity recipe's DR, observation noise and delays, the NaN guard, VelStand's servo-impact
sensors). What changes, for every net:

* **Physics**: SitStand's contact hardening (nconmax 200, 30 / 50 solver iterations).
* **Servo command delay**: the BAM actuator's lag redrawn 0–8 physics steps (0–40 ms) every 0.5 s,
  staggered (Stage 1b's finding; the simulator will run a fixed 20 ms; the skills must not depend on it).
* **Action**: robotd's path (``rl_action``) at the pinned scale 0.9 (velstand's walking scale: the
  networks are warm-started from v5 velstand, so iteration 0 behaves like v5).
* **Command**: ``rl_mdp.SkillCommand`` (hold → unwind), head/body slots zero-padded; the critic adds
  ``bsk_mdp.bsk_state_obs``.
* **Events**: v5 library entries (``bsk_mdp.reset_bsk_episode``), the recipe's DR with foot μ 0.5–1.3,
  late mild pushes on the flat nets.
* **Terrain** (``bskmat`` only): the hop workflow's strip tiles (``hop_terrain.strip_geoms``) made
  0.40 m deep — a mat — 2–5 cm by curriculum row, rigid or with a compliant top (7 mm sink).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import bsk_mdp as M
from duckmind.skilltrain import hop_terrain
from duckmind.skilltrain import rl_mdp as RM
from duckmind.skilltrain.rl_action import RobotdJointPositionActionCfg, pin_gain_ratio

STEPS = 24
ITERATIONS = {"bskstep": 1200, "bskmat": 1500, "bskbow": 800}
REWARD_REV = "b3"  # b3: march heading + measured beak-drop hold; gate thresholds unchanged.
REWARD_REVS = ("b3", "b4")
# b0 = the first calibration run; b1 (bskstep): stepping-gated in-place rewards, yaw progress,
# cadence and foot-slide costs — the b0 run stopped stepping by iteration 54 (steps 10.4 → 1.0 per episode, lift
# 10.7 → 0.25 mm) and turned 12–15 °/s by sliding its soles; b1 (bskbow): a linear pitch-progress reward and
# broad composite / leg stds — the b0 bow had no gradient toward a 25–33° pitch (it only dropped 11 mm by it 74).
# b2 (b:train, after S0-B): every net — a one-shot ``fall_event`` cost (−300: −6 per fall at once); bskmat — the
# curriculum promotes only a success without a fall (``mat_levels``), ``mat_clean`` logged; bskbow — pitch
# progress paid in the hold only (b1 paid for staying pitched in the unwind) and weight 3 → 4; bskstep — the
# march's heading drift −1 → −3 (S0-B Mac check: heading p90 14.7–22° in 4 s against the 15° gate).
MAIN_TERMS = {"bskstep": ("yaw_track", "steps"), "bskmat": ("mat_progress", "mat_settle"),
              "bskbow": ("composite", "handback")}  # fmt: skip
SIGNED_TERMS = ("mat_progress",)
FOOT_MU = (0.5, 1.3)
KEEP_EVENTS = (
    "foot_friction",
    "encoder_bias",
    "base_com",
    "expand_bam_friction_fields",
    "reset_action_history",
    "randomize_com",
    "randomize_head_com",
    "randomize_mass_inertia",
    "randomize_joint_friction",
    "randomize_armature",
    "push_robot",
)  # fmt: skip  (Stage 1's list, copied: rl_tasks belongs to Stage 1b)
SOFT = {
    "solref": (0.06, 1.0),
    "solimp": (0.4, 0.95, 0.015, 0.5, 2.0),
    "friction": (0.9, 0.005, 0.0001),
}


# ── the mat terrain (the hop workflow's strip, 0.40 m deep) ─────────────────────────────────


@dataclass(kw_only=True)
class MatTerrainCfg(hop_terrain.StripTerrainCfg):
    """A floor tile with one mat across it (``hop_terrain.strip_geoms``, sharp edge, 0.40 m deep);
    height from the curriculum row (``bsk_mdp.mat_height_of``); an optional compliant top."""

    soft: bool = False
    rows: int = B.MAT_ROWS

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        sx, sy = self.size
        row = min(self.rows - 1, max(0, int(math.floor(difficulty * self.rows + 1e-9))))
        height = float(M.mat_height_of(row, self.rows))
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, 0.05), pos=(sx / 2, sy / 2, -0.05)
        )
        geoms = [TerrainGeometry(geom=floor, color=hop_terrain.FLOOR_COLOR)]
        for g in hop_terrain.strip_geoms(
            body, self.face_x, sy / 2, height, B.MAT_DEPTH_M, sy, "sharp"
        ):
            if self.soft:
                g.priority = 2
                g.solref = list(SOFT["solref"])
                g.solimp = list(SOFT["solimp"])
                g.friction = list(SOFT["friction"])
            geoms.append(
                TerrainGeometry(
                    geom=g, color=(0.35, 0.45, 0.70, 1.0) if self.soft else hop_terrain.STRIP_COLOR
                )
            )
        return TerrainOutput(origin=np.array([self.face_x, sy / 2, 0.0]), geometries=geoms)


def mat_terrain_cfg(seed: int = 7) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=(2.0, 2.0),
        border_width=1.0,
        border_height=0.1,
        num_rows=B.MAT_ROWS,
        num_cols=2,
        color_scheme="none",
        sub_terrains={
            "rigid": MatTerrainCfg(soft=False, proportion=1.0 - B.SOFT_SHARE),
            "soft": MatTerrainCfg(soft=True, proportion=B.SOFT_SHARE),
        },
        difficulty_range=(0.0, 1.0),
        add_lights=False,
    )


def _delayed_robot(robot_cfg: Any, delay: tuple[int, int]) -> Any:
    rc = copy.deepcopy(robot_cfg)
    acts = tuple(
        dataclasses.replace(a, delay_min_lag=int(delay[0]), delay_max_lag=int(delay[1]),
                            delay_update_period=B.DELAY_UPDATE_STEPS, delay_hold_prob=0.5,
                            delay_per_env_phase=True)
        if hasattr(a, "delay_max_lag") else a
        for a in rc.articulation.actuators
    )  # fmt: skip
    rc.articulation = dataclasses.replace(rc.articulation, actuators=acts)
    return rc


# ── rewards ─────────────────────────────────────────────────────────────────────────────────


def _trunk_cfg():
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    return SceneEntityCfg("robot", body_names=("trunk_base",))


def _servo_stack(scale: float = 1.0) -> dict[str, RewardTermCfg]:
    """VelStand's servo-protection costs (Stage 1's ``rl_tasks._servo_stack``, copied: rl_tasks is
    Stage 1b's file and may change under us)."""
    R = RewardTermCfg
    return {
        "servo_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.02,
                          params={"sensor_name": "servo_ground_contact", "threshold": 2.0}),
        "head_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.01,
                         params={"sensor_name": "head_ground_contact", "threshold": 15.0}),
        "trunk_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.01,
                          params={"sensor_name": "trunk_ground_contact", "threshold": 20.0}),
        "servo_acc_spike": R(func=microduck_mdp.servo_acc_spike_penalty, weight=-1e-3 * scale,
                             params={"acc_thresh": 300.0}),
        "servo_stall": R(func=microduck_mdp.servo_stall_penalty, weight=-0.05 * scale,
                         params={"torque_thresh": 0.4, "vel_thresh": 0.5}),
    }  # fmt: skip


def _regularisers() -> dict[str, RewardTermCfg]:
    R = RewardTermCfg
    return {
        "action_rate_l2": R(func=vel_mdp.action_rate_l2, weight=-0.02),
        "joint_torque_rate_l2": R(func=microduck_mdp.joint_torque_rate_l2, weight=0.0),
        "body_ang_vel": R(func=vel_mdp.body_angular_velocity_penalty, weight=-0.02,
                          params={"asset_cfg": _trunk_cfg()}),
        "angular_momentum": R(func=vel_mdp.angular_momentum_penalty, weight=-0.01,
                              params={"sensor_name": "robot/root_angmom"}),
        "dof_pos_limits": R(func=vel_mdp.joint_pos_limits, weight=-1.0),
        "self_collisions": R(func=vel_mdp.self_collision_cost, weight=-1.0,
                             params={"sensor_name": "self_collision"}),
    }  # fmt: skip


def _common_costs(scale: float = 1.0, spin: bool = False) -> dict[str, RewardTermCfg]:
    R = RewardTermCfg
    out = {
        "fallen": R(func=RM.fallen, weight=-3.0),
        "fall_event": R(func=M.fall_event, weight=-300.0),  # rev b2: −6 per fall, paid at once
        "leg_torque": R(func=M.leg_torque, weight=-5.0),
        **_servo_stack(scale),
        **_regularisers(),
    }
    if spin:
        out.pop("angular_momentum", None)  # it would tax the yaw of a turn on the spot
    return out


def step_rewards() -> dict[str, RewardTermCfg]:
    R = RewardTermCfg
    return {
        "yaw_track": R(func=M.yaw_track, weight=2.0, params={"std": 0.5}),  # first: bookkeeping
        "yaw_progress": R(func=M.yaw_progress, weight=3.0),
        "stay": R(func=M.stay_stepping, weight=2.0, params={"std": 0.03}),
        "trunk_speed": R(func=M.speed_stepping, weight=1.0, params={"std": 0.08}),
        "steps": R(func=M.steps, weight=3.0),
        "clearance": R(func=M.clearance, weight=2.0),
        "cadence": R(func=M.cadence, weight=-2.0),
        "foot_slide": R(func=M.foot_slide, weight=-2.0),
        "upright": R(func=RM.upright, weight=1.0),
        "travel": R(func=M.travel, weight=-20.0),
        "heading_drift": R(func=M.heading_drift, weight=-8.0),
        "march_heading": R(func=M.march_heading, weight=3.0),
        "double_flight": R(func=M.double_flight, weight=-2.0),
        "touchdown_impact": R(func=M.touchdown_impact, weight=-1.0),
        **_common_costs(spin=True),
    }


def bow_rewards() -> dict[str, RewardTermCfg]:
    R = RewardTermCfg
    return {
        "composite": R(
            func=M.bow_composite,
            weight=3.0,  # first: bookkeeping; rev b1 broad stds
            params={"orient_std": 0.3, "legs_std": 0.6, "head_std": 0.4},
        ),
        "pitch_progress": R(func=M.pitch_progress, weight=4.0),  # rev b1; b2: hold only, 3 → 4
        "bow_gate_pose": R(func=M.bow_gate_pose, weight=5.0),
        "pose_legs": R(func=RM.pose_legs, weight=2.0, params={"std": 0.6}),
        "pose_head": R(func=RM.pose_head, weight=1.5, params={"std": 0.15}),
        "height": R(func=RM.height_track, weight=1.5, params={"std": 0.02}),
        "height_sharp": R(func=RM.height_track, weight=1.0, params={"std": 0.006}),
        "pitch": R(func=RM.pitch_track, weight=1.5, params={"std_deg": 5.0}),
        "still": R(func=RM.still, weight=1.0, params={"vel_std": 0.05}),
        "planted": R(func=RM.planted, weight=0.5),
        "upright": R(func=RM.upright, weight=0.5),
        "handback": R(func=RM.handback_bonus, weight=250.0),
        "pose_l1": R(func=RM.pose_l1, weight=-1.0),
        "height_l1": R(func=RM.height_l1, weight=-20.0),
        "steps_still": R(func=RM.steps_when_still, weight=-1.0),
        "rise_speed": R(func=RM.rise_speed, weight=-10.0),
        "double_flight": R(func=M.double_flight, weight=-2.0),
        "vertical_accel": R(func=RM.vertical_accel_cost, weight=-0.005),
        **_common_costs(),
    }


def rewards(net: str, rev: str = REWARD_REV) -> dict[str, RewardTermCfg]:
    """Build a task reward set; b4 changes only bow leg-torque pressure."""
    if rev not in REWARD_REVS:
        raise ValueError(f"reward revision {rev!r}; use one of {REWARD_REVS}")
    out = REWARDS[net]()
    if rev == "b4" and net == "bskbow":
        out["leg_torque"].weight = -15.0
    return out


def mat_rewards() -> dict[str, RewardTermCfg]:
    R = RewardTermCfg
    return {
        "mat_progress": R(func=M.mat_progress, weight=20.0),  # first: bookkeeping; signed
        "mat_settle": R(func=M.mat_settle, weight=250.0),
        "mat_clearance": R(func=M.mat_clearance, weight=1.0),
        "heading": R(func=M.heading_keep, weight=0.5),
        "upright": R(func=RM.upright, weight=1.0),
        "after_speed": R(func=M.after_speed, weight=-2.0),
        **_common_costs(),
    }


REWARDS = {"bskstep": step_rewards, "bskbow": bow_rewards, "bskmat": mat_rewards}


def _metrics(net: str) -> dict[str, MetricsTermCfg]:
    Mt = MetricsTermCfg
    common = {"fell": Mt(func=M.m_fell, reduce="last"), "fallen45": Mt(func=RM.m_fallen)}
    if net == "bskstep":
        return {**common, "turn_rate_dps": Mt(func=M.m_turn_rate_dps), "travel_cm": Mt(func=M.m_travel_cm, reduce="last"),
                "steps": Mt(func=M.m_steps, reduce="last"), "lift_mm": Mt(func=M.m_lift_mm),
                "share_turn": Mt(func=M.m_share("turn"), reduce="last"), "share_march": Mt(func=M.m_share("march"), reduce="last")}  # fmt: skip
    if net == "bskmat":
        return {**common, "mat_success": Mt(func=M.m_mat_success, reduce="last"),
                "mat_clean": Mt(func=M.m_mat_clean, reduce="last"),
                "mat_up_success": Mt(func=M.m_mat_up_success, reduce="last"), "mat_mm": Mt(func=M.m_mat_mm, reduce="last"),
                "share_up": Mt(func=M.m_share("step_up"), reduce="last")}  # fmt: skip
    return {**common, "bow_pitch_deg": Mt(func=M.m_bow_pitch_deg), "bow_drop_mm": Mt(func=M.m_bow_drop_mm),
            "share_bow": Mt(func=M.m_share("bow"), reduce="last")}  # fmt: skip


def _curricula(net: str, *, eval_mode: bool, scale: float = 1.0) -> dict[str, CurriculumTermCfg]:
    if eval_mode:
        return {}
    Cu = CurriculumTermCfg

    def it(i: int) -> int:
        return int(round(i * STEPS * scale))

    cur = {
        "action_rate_weight": Cu(func=microduck_mdp.reward_weight, params={
            "reward_name": "action_rate_l2",
            "weight_stages": [{"step": 0, "weight": -0.02}, {"step": it(300), "weight": -0.1},
                              {"step": it(700), "weight": -0.2}]}),
        "com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(200), "range": 0.010},
                             {"step": it(400), "range": 0.015}]}),
        "head_com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_head_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(200), "range": 0.010}]}),
        "entry_mix": Cu(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "skill_episode",
            "param_stages": [{"step": 0, "params": {"mix_stage": "early"}},
                             {"step": it(200), "params": {"mix_stage": "full"}}]}),
    }  # fmt: skip
    if net in ("bskstep", "bskbow"):
        cur["push_magnitude"] = Cu(func=microduck_mdp.push_curriculum, params={
            "event_name": "push_robot",
            "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                            {"step": it(400), "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                            {"step": it(800), "velocity_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12)}}]})  # fmt: skip
    if net == "bskmat":
        cur["mat_levels"] = Cu(func=M.mat_levels, params={})
    return cur


def make_env_cfg(
    net: str,
    *,
    library_path: str | None = None,
    eval_cells: list[tuple[str, str]] | None = None,
    foot_mu: tuple[float, float] | None = None,
    pushes: bool = True,
    curriculum_scale: float = 1.0,
    delay_steps: tuple[int, int] = B.SERVO_DELAY_STEPS,
    max_init_level: int = 2,
    reward_rev: str = REWARD_REV,
) -> ManagerBasedRlEnvCfg:
    if net not in B.NETS:
        raise ValueError(f"net {net!r}; use one of {B.NETS}")
    if reward_rev not in REWARD_REVS:
        raise ValueError(f"reward revision {reward_rev!r}; use one of {REWARD_REVS}")
    eval_mode = eval_cells is not None
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.episode_length_s = B.episode_s(net) + 0.5
    if net == "bskmat":
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = mat_terrain_cfg()
        cfg.scene.terrain.max_init_terrain_level = None if eval_mode else int(max_init_level)
    cfg.scene.entities = {
        **cfg.scene.entities,
        "robot": _delayed_robot(cfg.scene.entities["robot"], delay_steps),
    }

    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = RobotdJointPositionActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names, scale=1.0,
        use_default_offset=True, skill_action_scale=B.ACTION_SCALE,
    )  # fmt: skip
    cfg.commands = {"twist": RM.SkillCommandCfg()}
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )
    cfg.observations["critic"].terms["bsk_state"] = ObservationTermCfg(func=M.bsk_state_obs)

    cfg.events = {k: v for k, v in cfg.events.items() if k in KEEP_EVENTS}
    cfg.events["foot_friction"].params["ranges"] = foot_mu or ((0.7, 1.3) if eval_mode else FOOT_MU)
    if eval_mode or not pushes or net == "bskmat":
        cfg.events.pop("push_robot", None)
    else:
        push = cfg.events["push_robot"]
        push.interval_range_s = (1.5, 3.0)
        push.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    if eval_mode:
        cfg.events["randomize_com"].params["ranges"] = (-0.015, 0.015)
        cfg.events["randomize_head_com"].params["ranges"] = (-0.010, 0.010)
    cfg.events["pin_gain_ratio"] = EventTermCfg(
        func=pin_gain_ratio, mode="startup", params={"gain_ratio": B.GAIN_RATIO}
    )
    cfg.events["skill_episode"] = EventTermCfg(
        func=M.reset_bsk_episode, mode="reset",
        params={"net": net, "mix_stage": "full" if eval_mode else "early", "library_path": library_path,
                "eval_cells": eval_cells, "nominal_windows": eval_mode,
                "entry_mix_overrides": ({"bow": {"stand": .35, "walk": .35, "coast": .20,
                                                       "head": .05, "risen": .05},
                                          "lie": {"stand": .35, "walk": .35, "coast": .20,
                                                  "head": .05, "risen": .05}}
                                         if reward_rev == "b4" and net == "bskbow" and not eval_mode
                                         else None)},
    )  # fmt: skip

    cfg.rewards = rewards(net, reward_rev)
    cfg.terminations = {
        "skill_done": TerminationTermCfg(func=M.bsk_done, time_out=False),
        "nan_state": TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False,
                                        params={"sensor_names": ("feet_ground_contact",)}),
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    }  # fmt: skip
    if not eval_mode:
        cfg.terminations["fallen"] = TerminationTermCfg(
            func=microduck_mdp.fallen_too_long, time_out=False,
            params={"gate_z_below": 0.0, "gate_tilt_above_deg": 60.0, "max_duration_s": 1.0},
        )  # fmt: skip
    cfg.curriculum = _curricula(net, eval_mode=eval_mode, scale=curriculum_scale)
    cfg.metrics = _metrics(net)
    return cfg


def runner_cfg(net: str, *, iterations: int | None = None, init_std: float = 0.3, seed: int = 1,
               learning_rate: float = 3.0e-4) -> RslRlOnPolicyRunnerCfg:  # fmt: skip
    """PPO as the Stage-1 skills (512-256-128 ELU, adaptive KL 0.01), for a v5 warm start: a low
    initial std and learning rate (the S0 warm start ran at std 0.25, lr ≈ 2–4e-4)."""
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=STEPS, max_iterations=int(iterations or ITERATIONS[net]),
        save_interval=250, experiment_name=f"bsk_{net}", run_name=f"bsk_{net}", logger="tensorboard",
        upload_model=False,
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": init_std,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                       entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                       learning_rate=learning_rate, schedule="adaptive", gamma=0.99, lam=0.95,
                                       desired_kl=0.01, max_grad_norm=1.0),
    )  # fmt: skip


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    names = {n for n, t in cfg.rewards.items() if t.weight < 0}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


__all__ = ["ITERATIONS", "MAIN_TERMS", "MatTerrainCfg", "REWARD_REV", "REWARD_REVS", "cost_terms", "make_env_cfg",
           "mat_terrain_cfg", "runner_cfg"]  # fmt: skip
