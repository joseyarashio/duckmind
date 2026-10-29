"""The forward hop-over mjlab task (hop:feasibility, 2026-09-28), on the official recipe.

``Mjlab-DuckHopOver-Threshold-MicroDuck``: one network (``hopover.onnx``), four robotd table
entries (``hop_over_1..4``: 1–4 cm classes). Built like Stage 1's tasks (``rl_tasks``) from
``make_microduck_velstand_env_cfg`` (read-only upstream: the ``robot_allcollisions`` body with its
named XL330 housings, the BAM XL330 actuator, the velocity recipe's domain randomisation,
observation noise and delays, the NaN guard, VelStand's servo-impact sensors). What changes:

* **Terrain**: ``hop_terrain`` — a curriculum grid of 2 m tiles, one strip per tile (rows: 0.5 → 4 cm,
  columns: sharp / round edge); the curriculum moves a duck up a row after a clean crossing and down
  after a fall or a failure (``hop_mdp.strip_levels``).
* **Servo command delay**: the BAM actuator's delay 0–8 physics steps (0–40 ms; Stage 1b's finding,
  the orchestrator's rule) instead of the recipe's 3–6 — hop:train draws it piecewise constant (a
  fresh lag every 0.5 s with p 0.5, as Stage 1b's retrain); the calibration re-drew it every step.
* **Physics / action / command**: as Stage 1 (nconmax 200, 30 / 50 solver iterations; robotd's
  action path at the pinned scale; ``SkillCommand`` phases; head/body slots zero-padded).
* **Entries**: the v5 state library (walk 55 %, stand 35 %, coast 10 %; early: stand-heavy), placed
  6–18 cm before the strip, heading ±12°.
* **Rewards / terminations / metrics**: ``hop_mdp`` (progress potential, crossing bonus, lift over
  the strip, settle-for-velstand bonus; toe stubs, falls, leg strain, VelStand's servo protection).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

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
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import hop_mdp as M
from duckmind.skilltrain import hop_terrain
from duckmind.skilltrain import rl_mdp as RM
from duckmind.skilltrain.rl_action import RobotdJointPositionActionCfg, pin_gain_ratio

ITERATIONS = 1500
STEPS = 24
# Reward revisions. "h1" (hop:feasibility calibration, 240 it, 2026-09-28): the task trains — the network
# leap-steps over 1–4 cm strips (Mac gate, both servo models, 20 ms delay: crossed 104–120/120 per height)
# but dashes (peak 0.7–0.8 m/s), lands at 3.6–5.7 × body weight (p50 of per-trial peaks; one foot), runs
# its legs at the 0.64 N·m ceiling (per-tick leg max p95 0.50–0.64 vs the 0.35 strain gate) and hits a
# servo housing in 13–51 % of 3–4 cm crossings. "h2" (proposed, CPU-smoked only): progress paid up to
# 0.35 m/s; foot_impact (vertical sole force above 2.5 × body weight, −1.0 per N); servo_impact −0.02 → −0.5;
# leg_torque −5 → −15 (Stage 1's a3 lesson: 0.55 → 0.35 p95).
REWARD_REV = "h2"
REWARD_REVS = ("h1", "h2", "h3", "h4")
FOOT_MU = (0.5, 1.3)
KEEP_EVENTS = (
    "foot_friction", "encoder_bias", "base_com", "expand_bam_friction_fields",
    "reset_action_history", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
    "randomize_joint_friction", "randomize_armature",
)  # fmt: skip


def _it(i: int, scale: float = 1.0) -> int:
    return int(round(i * STEPS * scale))


def _trunk_cfg():
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    return SceneEntityCfg("robot", body_names=("trunk_base",))


def rewards(rev: str = REWARD_REV) -> dict[str, RewardTermCfg]:
    """``progress`` first: it also does the per-step bookkeeping the others read."""
    if rev not in REWARD_REVS:
        raise ValueError(f"reward revision {rev!r}: one of {REWARD_REVS}")
    h2 = rev in ("h2", "h3", "h4")
    R = RewardTermCfg
    out = {
        "progress": R(func=M.progress, weight=20.0, params={"max_mps": 0.35 if h2 else 2.0}),
        "cross_bonus": R(func=M.cross_bonus, weight=250.0),
        "upright": R(func=M.upright, weight=1.0),
        "heading": R(func=M.heading, weight=0.5),
        "swing_clearance": R(func=M.swing_clearance, weight=1.0),
        "settle": R(func=M.settle_bonus, weight=250.0),
        "toe_stub": R(func=M.toe_stub, weight=-0.2),
        "fallen": R(func=M.fallen, weight=-3.0),
        "leg_torque": R(func=M.leg_torque, weight=-15.0 if h2 else -5.0),
        "after_cross_speed": R(func=M.after_cross_speed, weight=-2.0),
        "servo_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.5 if h2 else -0.02,
                          params={"sensor_name": "servo_ground_contact", "threshold": 2.0}),
        "head_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.01,
                         params={"sensor_name": "head_ground_contact", "threshold": 15.0}),
        "trunk_impact": R(func=microduck_mdp.body_impact_cost, weight=-0.01,
                          params={"sensor_name": "trunk_ground_contact", "threshold": 20.0}),
        "servo_acc_spike": R(func=microduck_mdp.servo_acc_spike_penalty, weight=-1e-3,
                             params={"acc_thresh": 300.0}),
        "servo_stall": R(func=microduck_mdp.servo_stall_penalty, weight=-0.05,
                         params={"torque_thresh": 0.4, "vel_thresh": 0.5}),
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
    if h2:
        out["foot_impact"] = R(func=M.foot_impact, weight=-1.0, params={"free_bw": 2.5})
        # hop:train: two costs present at weight 0, so a run can switch them on mid-way (OVERRIDE.json)
        out["housing_touch"] = R(func=M.housing_touch, weight=0.0, params={"threshold": 2.0})
        out["dash_speed"] = R(func=M.dash_speed, weight=0.0, params={"free_mps": 0.45})
    if rev in ("h3", "h4"):
        out["housing_touch"].weight = -4.0
        out["dash_speed"].weight = -2.0
        out["fall_event"] = R(func=M.fall_event, weight=-300.0)
    if rev == "h4":
        out["housing_touch"].weight = -12.0
        out["fall_event"].weight = -600.0
    return out


OVERRIDABLE = H.OVERRIDABLE  # the mid-run re-weightable terms (hop_contract)


def _curricula(*, eval_mode: bool, scale: float = 1.0) -> dict[str, CurriculumTermCfg]:
    if eval_mode:
        return {}
    Cu = CurriculumTermCfg

    def it(i: int) -> int:
        return _it(i, scale)

    return {
        "strip_levels": Cu(func=M.strip_levels, params={}),
        "action_rate_weight": Cu(func=microduck_mdp.reward_weight, params={
            "reward_name": "action_rate_l2",
            "weight_stages": [{"step": 0, "weight": -0.02}, {"step": it(400), "weight": -0.1},
                              {"step": it(900), "weight": -0.2}]}),
        "torque_rate_weight": Cu(func=microduck_mdp.reward_weight, params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [{"step": 0, "weight": 0.0}, {"step": it(700), "weight": -5e-4}]}),
        "com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(250), "range": 0.010},
                             {"step": it(500), "range": 0.015}]}),
        "head_com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_head_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(250), "range": 0.010}]}),
        "entry_mix": Cu(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "hop_episode",
            "param_stages": [{"step": 0, "params": {"mix_stage": "early"}},
                             {"step": it(300), "params": {"mix_stage": "full"}}]}),
    }  # fmt: skip


def _metrics() -> dict[str, MetricsTermCfg]:
    return {
        "crossed": MetricsTermCfg(func=M.m_crossed, reduce="last"),
        "fell": MetricsTermCfg(func=M.m_fell, reduce="last"),
        "strip_mm": MetricsTermCfg(func=M.m_height_mm, reduce="last"),
        "progress_mm": MetricsTermCfg(func=M.m_progress_mm, reduce="last"),
        "stubs": MetricsTermCfg(func=M.m_stubs, reduce="last"),
        "max_tilt": MetricsTermCfg(func=M.m_max_tilt, reduce="last"),
    }


def _delayed_robot(robot_cfg: Any, delay: str | int | tuple[int, int]) -> Any:
    """A copy of the robot entity whose BAM actuator carries this servo command delay:
    ``hop_contract.delay_fields`` modes (``"train"``: piecewise constant 0–40 ms; ``"jitter"``:
    re-drawn every physics step; an int: a fixed lag in physics steps), or a legacy (min, max)
    tuple (re-drawn every step, the calibration's)."""
    if isinstance(delay, tuple):
        fields: dict[str, Any] = {"delay_min_lag": int(delay[0]), "delay_max_lag": int(delay[1]),
                                  "delay_update_period": 0, "delay_hold_prob": 0.0}  # fmt: skip
    else:
        fields = dict(H.delay_fields(delay))
    rc = copy.deepcopy(robot_cfg)
    acts = tuple(
        dataclasses.replace(a, **fields) if hasattr(a, "delay_max_lag") else a
        for a in rc.articulation.actuators
    )
    rc.articulation = dataclasses.replace(rc.articulation, actuators=acts)
    return rc


def make_env_cfg(
    *,
    library_path: str | None = None,
    eval_cells: list[tuple[int, int, str]] | None = None,
    handback_s: float = 0.0,
    foot_mu: tuple[float, float] | None = None,
    curriculum_scale: float = 1.0,
    max_init_level: int = 2,
    delay_steps: str | int | tuple[int, int] = "train",
    reward_rev: str = REWARD_REV,
) -> ManagerBasedRlEnvCfg:
    """The training (or, with ``eval_cells``, the evaluation) env cfg. ``delay_steps``: the servo
    command delay mode (:func:`hop_contract.delay_fields`; hop:train's default ``"train"``)."""
    eval_mode = eval_cells is not None
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.episode_length_s = H.HOLD_S * H.HOLD_JITTER[1] + H.UNWIND_S + handback_s + 0.5

    # terrain: the strip grid (the velstand recipe's plane replaced)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = hop_terrain.terrain_cfg()
    cfg.scene.terrain.max_init_terrain_level = None if eval_mode else int(max_init_level)
    cfg.scene.entities = {
        **cfg.scene.entities,
        "robot": _delayed_robot(cfg.scene.entities["robot"], delay_steps),
    }

    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = RobotdJointPositionActionCfg(
        entity_name=old.entity_name,
        actuator_names=old.actuator_names,
        scale=1.0,
        use_default_offset=True,
        skill_action_scale=H.ACTION_SCALE,
    )
    cfg.commands = {"twist": RM.SkillCommandCfg()}
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )
    cfg.observations["critic"].terms["hop_state"] = ObservationTermCfg(func=M.hop_state_obs)

    cfg.events = {k: v for k, v in cfg.events.items() if k in KEEP_EVENTS}
    cfg.events["foot_friction"].params["ranges"] = foot_mu or ((0.7, 1.3) if eval_mode else FOOT_MU)
    if eval_mode:
        cfg.events["randomize_com"].params["ranges"] = (-0.015, 0.015)
        cfg.events["randomize_head_com"].params["ranges"] = (-0.010, 0.010)
    cfg.events["pin_gain_ratio"] = EventTermCfg(func=pin_gain_ratio, mode="startup",
                                                params={"gain_ratio": H.GAIN_RATIO})  # fmt: skip
    cfg.events["hop_episode"] = EventTermCfg(
        func=M.reset_hop_episode,
        mode="reset",
        params={
            "mix_stage": "full" if eval_mode else "early",
            "library_path": library_path,
            "eval_cells": eval_cells,
            "handback_s": handback_s,
            "nominal_windows": eval_mode,
        },
    )
    cfg.rewards = rewards(reward_rev)
    cfg.terminations = {
        "skill_done": TerminationTermCfg(func=RM.skill_done, time_out=False),
        "nan_state": TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False,
                                        params={"sensor_names": ("feet_ground_contact",)}),
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    }  # fmt: skip
    if not eval_mode:
        cfg.terminations["fallen"] = TerminationTermCfg(
            func=microduck_mdp.fallen_too_long,
            time_out=False,
            params={"gate_z_below": 0.0, "gate_tilt_above_deg": 60.0, "max_duration_s": 1.0},
        )
    cfg.curriculum = _curricula(eval_mode=eval_mode, scale=curriculum_scale)
    if reward_rev in ("h3", "h4") and not eval_mode:
        cfg.curriculum["strip_levels"].params["max_level"] = 3  # 5–20 mm, not 30–40 mm
    cfg.metrics = _metrics()
    return cfg


def runner_cfg(
    *, iterations: int | None = None, init_std: float = 1.0, seed: int = 1
) -> RslRlOnPolicyRunnerCfg:
    """PPO as the microduck recipes run it (Stage 1's settings)."""
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=STEPS,
        max_iterations=int(iterations or ITERATIONS),
        save_interval=250,
        experiment_name="skill_hopover",
        run_name="skill_hopover",
        logger="tensorboard",
        upload_model=False,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),  # fmt: skip
    )


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    """Every cost term: negative weights, the curriculum's negative stages, and our cost functions
    even at weight 0 (they may be switched on mid-run)."""
    names = {n for n, t in cfg.rewards.items() if t.weight < 0 or t.func in M.COST_FUNCS}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


def eval_grid(kinds: tuple[str, ...] = ("walk", "stand")) -> list[tuple[int, int, str]]:
    """Every (row, edge column, entry kind): rows 1 cm, 2 cm, 3 cm, 4 cm (and 0.5 cm)."""
    rows = sorted({min(H.TERRAIN_ROWS - 1, round((h - H.HEIGHT_RANGE[0]) / (H.HEIGHT_RANGE[1] - H.HEIGHT_RANGE[0])
                                                  * (H.TERRAIN_ROWS - 1)))
                   for h in (0.005, 0.01, 0.02, 0.03, 0.04)})  # fmt: skip
    return [(r, c, k) for r in rows for c in range(len(H.EDGES)) for k in kinds]


__all__ = [
    "ITERATIONS",
    "REWARD_REV",
    "cost_terms",
    "eval_grid",
    "make_env_cfg",
    "rewards",
    "runner_cfg",
]
