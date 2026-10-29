"""The two Stage-1 mjlab tasks (training plan §3.1–3.2), built on the official recipe.

``Mjlab-DuckStartle-Flat-MicroDuck`` (R1: flinch → crouch-and-freeze, hop-back as quick
back-steps) and ``Mjlab-DuckComfort-Flat-MicroDuck`` (R2: shake-off, stretch standing and
seated) — each one network, several robotd table entries.

Both start from ``make_microduck_velstand_env_cfg`` (imported read-only from
``mjlab_microduck``): the official ``robot_allcollisions`` body with its named XL330 housings, the
BAM XL330 actuator, the velocity recipe's domain randomisation / observation noise / delays /
NaN guard, and VelStand's servo-impact sensors. What changes:

* **Physics**: SitStand's contact-solver hardening (nconmax 200, 30 / 50 solver iterations —
  crouches and seats put many parts near the floor; the gate harness assumes the same).
* **Action**: robotd's path (``rl_action``): the table's pinned scale, voltage adaptation,
  low-pass head 0.5 / legs 0.7, entry filter state. Gain ratio pinned via ``kp_scale``.
* **Command**: the skill window (``rl_mdp.SkillCommand`` as ``twist``); head and body slots
  zero-padded (robotd zeroes them during a skill) — the 61-D actor layout is unchanged.
  The critic adds privileged skill state (phase, time left, targets).
* **Events**: entries from the v5 state library (``rl_library``), the recipe's DR with the floor
  friction widened to 0.5–1.3 (plan: "floor μ down to 0.5"), pushes ±0.15 m/s (ground pick's
  value) introduced late; VelStand's topple pushes and prone spawns removed.
* **Rewards / curricula / terminations**: replaced (below); the episode ends with the window.

``make_env_cfg(net, eval_cells=…, handback_s=…)`` builds the evaluator's variant (final DR, no
pushes, nominal windows, the hand-back phase).
"""

from __future__ import annotations

import math
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

from duckmind.skilltrain import rl_contract as C
from duckmind.skilltrain import rl_mdp as M
from duckmind.skilltrain.rl_action import RobotdJointPositionActionCfg, pin_gain_ratio

TASK_IDS = {
    "startle": "Mjlab-DuckStartle-Flat-MicroDuck",
    "comfort": "Mjlab-DuckComfort-Flat-MicroDuck",
}
ITERATIONS = {"startle": 2500, "comfort": 2000}  # plan §3.1 / §3.2 full-run budgets
STEPS = 24  # num_steps_per_env (curriculum steps are iteration × 24)
# Reward revision (recorded in every run's metrics.json). "a1" (tr:train, 2026-09-28, from the S0
# calibration's 100-iteration evidence: hop_back 0/32 performed in every posture, hop_progress ≈ 0,
# while holding the crouch paid ≈ 13/s and stepping back at 0.125 m/s cost ≈ 2.5–3/s of posture
# for 1.25/s of progress — stepping never paid; the comfort run's steps_still cost grew −0.10 → −0.26):
# the hop window frees the legs (no leg-pose tracking: they must step; trunk height, pitch and the
# head stay tracked), hop_progress 10 → 30, hop_steps 1 → 5; the shake's band rewards are scaled by
# exp(−0.7 · touchdowns in the statistics window), so a shake by stepping does not pay.
# "a2" (tr:train, 2026-09-28, from the attempt-1 250-iteration evaluation, 32 ducks per cell: the hop
# moved back 7–12 cm p50 by 0.7 s — the gate's 6 cm — but turned 39–55° p50, gate ≤ 15°; yaw_hold's
# σ 0.15 rad Gaussian is flat there): hop_yaw, a linear |Δheading| cost (weight −3) through the hop's
# hold and unwind; backing 10 cm straight still nets ≈ +2 per episode, the 55° turn ≈ break-even.
# "c2" (tr:train, 2026-09-28, comfort attempt 1 at iteration 124–148: shake 0/32 — roll 0.2–0.3°, head yaw
# 0.01 rad — with ≈ 9 contact-chatter "touchdowns" per 0.5 s window under exploration noise zeroing the
# planted factor): the planted factor counts only real steps (a foot airborne ≥ 0.06 s).
# "a3" (tr:train, 2026-09-28, the a2 startle network's final evaluation, 128 per cell: hop leg torque p95
# 0.49–0.62 N·m against the ≤ 0.35 strain gate): the hop's trunk dip at k = 0.5 (≈ 14 mm; the neck pull-back
# stays k = 1.0) and hop_torque, Σ leg |τ| above 0.30 N·m (weight −15) through the hop's hold and unwind.
# "c3" (tr:train, 2026-09-28, the c2 comfort network's final evaluation, 128 per cell: every cell performed ≥ 116/128
# on the Mac gate's footfall definition, but shake_2 head |τ| p95 0.43–0.44 and legs 0.36–0.37 N·m against the 0.35
# strain gate, and the peak-amplitude window at 2.35–2.49 Hz next to the 2.5 Hz edge): the shake's training bands move
# to margins over the gate with smaller amplitudes (SHAKE_BANDS) and shake_torque, Σ |τ| above 0.30 N·m over all servos
# (weight −15), through the shake's hold and unwind.
# "a4" / "c4" (s1b:retrain, 2026-09-28, the review's majors (4) and minors; resumed from a3 it 1549 / c3 it 1249):
# * both: the servo command delay randomised 0–40 ms, piecewise constant (rl_contract.SERVO_LAG_TRAIN; the s1b
#   latency map: rev a3/c3 were fall-free on the XML servos only at 15–30 ms, the hop performed only at 20–25 ms);
# * startle: a real 2–3-step hop-back (hop_long_steps pays real steps by length, hop_extra_steps costs every step
#   beyond the 4th, hop_flight costs both feet in the air; rev a1's hop_steps paid every touchdown and is gone:
#   rev a3 took 7–8 steps of ≈ 1.2 cm); walk_stop_torque (leg |τ| above 0.35 N·m in the first 0.7 s of a walking
#   entry) and walk_yaw (the heading while stopping); 5 % of the crouch episodes held 2–6 s;
# * comfort: a 1.5-s shake window with 1° more trunk roll (rl_mdp.SHAKE_BANDS), the seated stretch lifting the
#   head (camera 30°, rl_contract.STRETCH_SIT), stretch_acc (joint acceleration above 80 rad/s² in a stretch).
REWARD_REV = "a4c4"
FOOT_MU = (0.5, 1.3)  # training (plan R1 extra DR); evaluation also runs the recipe's 0.7–1.3
PUSH_FINAL = 0.15  # m/s, ground pick's quasi-static value (plan §3.1)
PUSH_INTERVAL_S = (1.5, 3.0)
KEEP_EVENTS = (
    "foot_friction", "encoder_bias", "base_com", "expand_bam_friction_fields",
    "reset_action_history", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
    "randomize_joint_friction", "randomize_armature", "push_robot",
)  # fmt: skip


def _it(i: int, scale: float = 1.0) -> int:
    """Curriculum step (env steps) of iteration ``i``; ``scale`` < 1 compresses the schedule for
    runs shorter than the plan's budgets (S0: 1,200 / 1,000 iterations → × 0.5, final stages by
    iteration 600)."""
    return int(round(i * STEPS * scale))


def _servo_stack(scale: float = 1.0) -> dict[str, RewardTermCfg]:
    """VelStand's servo-protection costs at their full weights (the skills need no fall-recovery
    discovery window); ``scale`` multiplies the wear terms (stall, acceleration spikes)."""
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
        "action_rate_l2": R(func=vel_mdp.action_rate_l2, weight=-0.02),  # curriculum → −0.3
        "joint_torque_rate_l2": R(func=microduck_mdp.joint_torque_rate_l2, weight=0.0),
        "body_ang_vel": R(func=vel_mdp.body_angular_velocity_penalty, weight=-0.02,
                          params={"asset_cfg": _trunk_cfg()}),
        "angular_momentum": R(func=vel_mdp.angular_momentum_penalty, weight=-0.01,
                              params={"sensor_name": "robot/root_angmom"}),
        "dof_pos_limits": R(func=vel_mdp.joint_pos_limits, weight=-1.0),
        "self_collisions": R(func=vel_mdp.self_collision_cost, weight=-1.0,
                             params={"sensor_name": "self_collision"}),
    }  # fmt: skip


def _trunk_cfg():
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    return SceneEntityCfg("robot", body_names=("trunk_base",))


def startle_rewards() -> dict[str, RewardTermCfg]:
    """R1: posture-conditioned single targets with a slewed blend (SitStand), the freeze (still,
    planted), the back-steps (potential-based, capped at 10 cm, inside 0.8 s, real steps), the
    hand-back test, VelStand's servo protection."""
    R = RewardTermCfg
    return {
        "pose_legs": R(func=M.pose_legs, weight=3.0, params={"std": 0.3}),
        "pose_head": R(func=M.pose_head, weight=2.0, params={"std": 0.12}),
        "height": R(func=M.height_track, weight=1.5, params={"std": 0.02}),
        "height_sharp": R(func=M.height_track, weight=1.5, params={"std": 0.006}),
        "pitch": R(func=M.pitch_track, weight=0.5, params={"std_deg": 4.0}),
        "composite": R(func=M.composite, weight=3.0),
        "still": R(func=M.still, weight=1.5, params={"vel_std": 0.05}),
        "planted": R(func=M.planted, weight=0.5),
        "yaw_hold": R(func=M.yaw_hold, weight=0.5, params={"std": 0.15}),
        "hop_progress": R(func=M.hop_progress, weight=30.0),  # rev a1 (was 10)
        # rev a4: hop_steps (every touchdown, weight 5 since rev a1) replaced by real steps paid by length
        "hop_long_steps": R(func=M.hop_long_steps, weight=15.0),  # rev a4
        "hop_extra_steps": R(func=M.hop_extra_steps, weight=-10.0),  # rev a4
        "hop_flight": R(func=M.hop_flight, weight=-10.0),  # rev a4
        "hop_yaw": R(func=M.hop_yaw, weight=-3.0),  # rev a2
        "hop_torque": R(func=M.hop_leg_torque, weight=-15.0),  # rev a3
        "walk_stop_torque": R(func=M.walk_stop_torque, weight=-15.0),  # rev a4
        "walk_yaw": R(func=M.walk_yaw, weight=-3.0),  # rev a4
        "upright": R(func=M.upright, weight=1.0),
        "handback": R(func=M.handback_bonus, weight=250.0),
        "pose_l1": R(func=M.pose_l1, weight=-1.0),
        "height_l1": R(func=M.height_l1, weight=-20.0),
        "steps_still": R(func=M.steps_when_still, weight=-1.0),
        "foot_slip": R(func=M.foot_slip, weight=-0.5),
        "rise_speed": R(func=M.rise_speed, weight=-10.0),
        "fallen": R(func=M.fallen, weight=-3.0),
        "vertical_accel": R(func=M.vertical_accel_cost, weight=-0.005),
        **_servo_stack(),
        **_regularisers(),
    }


def comfort_rewards() -> dict[str, RewardTermCfg]:
    """R2: phase-invariant oscillation bands for the shake (trunk roll 2.5–4 Hz, head yaw
    3–5 Hz, amplitudes by the command), feet planted, in place, heavier wear costs; the
    stretches' slewed height / camera-pitch targets held still; the hand-back test."""
    R = RewardTermCfg
    return {
        "pose_legs": R(func=M.pose_legs, weight=2.0, params={"std": 0.3}),
        "pose_head": R(func=M.pose_head, weight=1.5, params={"std": 0.15}),
        "height": R(func=M.height_track, weight=1.0, params={"std": 0.015}),
        "height_sharp": R(func=M.height_track, weight=1.5, params={"std": 0.004}),
        "composite": R(func=M.composite, weight=3.0),
        "camera": R(func=M.camera_pitch, weight=2.0, params={"std_deg": 6.0}),
        "shake_roll": R(func=M.shake_roll, weight=5.0, params={"sigma_hz": 1.5}),
        "shake_head": R(func=M.shake_head, weight=3.0, params={"sigma_hz": 1.5}),
        "still": R(func=M.still, weight=1.0, params={"vel_std": 0.05}),
        "planted": R(func=M.planted, weight=1.0),
        "yaw_hold": R(func=M.yaw_hold, weight=0.5, params={"std": 0.15}),
        "upright": R(func=M.upright, weight=1.0),
        "handback": R(func=M.handback_bonus, weight=250.0),
        "pose_l1": R(func=M.pose_l1, weight=-1.0),
        "height_l1": R(func=M.height_l1, weight=-20.0),
        "steps_still": R(func=M.steps_when_still, weight=-1.5),
        "drift": R(func=M.drift, weight=-20.0, params={"free_m": 0.01}),
        "rise_speed": R(func=M.rise_speed, weight=-10.0),
        "fallen": R(func=M.fallen, weight=-3.0),
        "vertical_accel": R(func=M.vertical_accel_cost, weight=-0.005),
        "torque_sq": R(func=M.torque_sq, weight=-0.02),
        "shake_torque": R(func=M.shake_servo_torque, weight=-15.0),  # rev c3
        "stretch_acc": R(func=M.stretch_acc, weight=-0.03),  # rev c4
        **_servo_stack(scale=2.0),
        **_regularisers(),
    }


def _curricula(net: str, *, eval_mode: bool, scale: float = 1.0) -> dict[str, CurriculumTermCfg]:
    if eval_mode:
        return {}
    if not 0.05 <= scale <= 2.0:
        raise ValueError(f"curriculum scale {scale} outside 0.05–2")
    Cu = CurriculumTermCfg

    def it(i: int) -> int:
        return _it(i, scale)

    cur = {
        "action_rate_weight": Cu(func=microduck_mdp.reward_weight, params={
            "reward_name": "action_rate_l2",
            "weight_stages": [{"step": 0, "weight": -0.02}, {"step": it(500), "weight": -0.1},
                              {"step": it(1000), "weight": -0.3}]}),
        "torque_rate_weight": Cu(func=microduck_mdp.reward_weight, params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [{"step": 0, "weight": 0.0}, {"step": it(800), "weight": -5e-4},
                              {"step": it(1200), "weight": -1e-3}]}),
        "com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(300), "range": 0.010},
                             {"step": it(600), "range": 0.015}]}),
        "head_com_range": Cu(func=microduck_mdp.com_range_curriculum, params={
            "event_name": "randomize_head_com",
            "range_stages": [{"step": 0, "range": 0.005}, {"step": it(300), "range": 0.010}]}),
        "push_magnitude": Cu(func=microduck_mdp.push_curriculum, params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": it(400), "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": it(800), "velocity_range": {"x": (-PUSH_FINAL, PUSH_FINAL),
                                                      "y": (-PUSH_FINAL, PUSH_FINAL)}}]}),
        "entry_mix": Cu(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "skill_episode",
            "param_stages": [{"step": 0, "params": {"mix_stage": "early"}},
                             {"step": it(600), "params": {"mix_stage": "full"}}]}),
    }  # fmt: skip
    if net == "comfort":
        for name in ("shake_roll", "shake_head"):
            cur[f"{name}_band"] = Cu(func=reward_param_curriculum, params={
                "reward_name": name,
                "param_stages": [{"step": 0, "params": {"sigma_hz": 1.5}},
                                 {"step": it(600), "params": {"sigma_hz": 1.0}},
                                 {"step": it(1200), "params": {"sigma_hz": 0.6}}]})  # fmt: skip
    return cur


def reward_param_curriculum(
    env: Any, env_ids: Any, reward_name: str, param_stages: list[dict]
) -> Any:
    """Mutate a live reward term's params at scheduled env steps (the reward manager deep-copies
    its cfg, so ``env.cfg`` writes would be silent no-ops — upstream's lesson)."""
    import torch

    del env_ids
    term = env.reward_manager.get_term_cfg(reward_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term.params.update(current)
    first = next(iter(current.values()))
    return torch.tensor(float(first) if isinstance(first, (int, float)) else 0.0)


def _metrics() -> dict[str, MetricsTermCfg]:
    return {
        "fallen45": MetricsTermCfg(func=M.m_fallen),
        "height_err_mm": MetricsTermCfg(func=M.m_height_err_mm),
        "hold_speed": MetricsTermCfg(func=M.m_speed),
        "hop_back_mm": MetricsTermCfg(func=M.m_back_mm, reduce="last"),
        "shake_roll_deg": MetricsTermCfg(func=M.m_roll_amp_deg),
        "shake_touchdowns": MetricsTermCfg(func=M.m_shake_touchdowns),
        "stretch_cam_deg": MetricsTermCfg(func=M.m_camera_deg),
        "neck_delta": MetricsTermCfg(func=M.m_neck_delta),
    }


def make_env_cfg(
    net: str,
    *,
    library_path: str | None = None,
    eval_cells: list[tuple[str, str]] | None = None,
    handback_s: float = 0.0,
    foot_mu: tuple[float, float] | None = None,
    pushes: bool = True,
    curriculum_scale: float = 1.0,
    servo_lag: str | int | None = None,
) -> ManagerBasedRlEnvCfg:
    """The training (or, with ``eval_cells``, the evaluation) env cfg of ``net``;
    ``curriculum_scale`` compresses (< 1) the curricula's iteration schedule. ``servo_lag``: the servo command
    delay — ``"train"`` (rev a4/c4: 0–40 ms, piecewise constant; the training default), ``"recipe"`` (the
    velocity recipe's 15–30 ms redrawn every physics step; the evaluation default), or an int: a fixed lag in
    physics steps of 5 ms (0 = none)."""
    if net not in C.NETS:
        raise ValueError(f"net {net!r}; use one of {C.NETS}")
    eval_mode = eval_cells is not None
    scale, gain = C.pinned(net)
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.episode_length_s = C.max_episode_s(net) + handback_s + 0.5
    lag_mode = servo_lag if servo_lag is not None else ("recipe" if eval_mode else "train")
    _set_servo_lag(cfg, lag_mode)

    # action: robotd's path at the table's pinned scale
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = RobotdJointPositionActionCfg(
        entity_name=old.entity_name,
        actuator_names=old.actuator_names,
        scale=1.0,
        use_default_offset=True,
        skill_action_scale=scale,
    )

    # command: the skill window; head/body slots zero (robotd zeroes them in a skill)
    cfg.commands = {"twist": M.SkillCommandCfg()}
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )
    cfg.observations["critic"].terms["skill_state"] = ObservationTermCfg(func=M.skill_state_obs)

    # events: the recipe's DR + our entries; no topple pushes / prone spawns / velocity resets
    cfg.events = {k: v for k, v in cfg.events.items() if k in KEEP_EVENTS}
    cfg.events["foot_friction"].params["ranges"] = foot_mu or ((0.7, 1.3) if eval_mode else FOOT_MU)
    if eval_mode or not pushes:
        cfg.events.pop("push_robot", None)
        cfg.events["randomize_com"].params["ranges"] = (-0.015, 0.015)
        cfg.events["randomize_head_com"].params["ranges"] = (-0.010, 0.010)
    else:
        push = cfg.events["push_robot"]
        push.interval_range_s = PUSH_INTERVAL_S
        push.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    cfg.events["pin_gain_ratio"] = EventTermCfg(
        func=pin_gain_ratio, mode="startup", params={"gain_ratio": gain}
    )
    if (
        lag_mode == "train"
    ):  # rev a4/c4: every episode starts with a fresh lag (the buffer zeroes it at reset)
        lag = C.SERVO_LAG_TRAIN
        cfg.events["servo_lag"] = EventTermCfg(
            func=M.randomize_servo_lag,
            mode="reset",
            params={"min_lag": lag["min_lag"], "max_lag": lag["max_lag"]},
        )
    cfg.events["skill_episode"] = EventTermCfg(
        func=M.reset_skill_episode,
        mode="reset",
        params={
            "net": net,
            "mix_stage": "full" if eval_mode else "early",
            "library_path": library_path,
            "eval_cells": eval_cells,
            "handback_s": handback_s,
            "nominal_windows": eval_mode,
        },
    )

    cfg.rewards = startle_rewards() if net == "startle" else comfort_rewards()
    cfg.terminations = {
        "skill_done": TerminationTermCfg(func=M.skill_done, time_out=False),
        "nan_state": TerminationTermCfg(
            func=microduck_mdp.robot_state_is_nan, time_out=False,
            params={"sensor_names": ("feet_ground_contact",)},
        ),
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    }  # fmt: skip
    if not eval_mode:
        # A duck down (> 60°) for 1 s ends the episode: the skill's MDP is over (velstand would
        # take over and right it), the landing's impact costs were already paid, and the rest of
        # the window would be spent lying on the floor. The evaluator keeps every window.
        cfg.terminations["fallen"] = TerminationTermCfg(
            func=microduck_mdp.fallen_too_long,
            time_out=False,
            params={"gate_z_below": 0.0, "gate_tilt_above_deg": 60.0, "max_duration_s": 1.0},
        )
    cfg.curriculum = _curricula(net, eval_mode=eval_mode, scale=curriculum_scale)
    cfg.metrics = _metrics()
    return cfg


def _set_servo_lag(cfg: ManagerBasedRlEnvCfg, mode: str | int) -> None:
    """Replace the robot's actuator cfgs with copies carrying this servo-delay mode (never mutates the
    upstream module-level cfg other tasks share)."""
    import dataclasses

    if mode == "recipe":
        return
    if mode == "train":
        lag = C.SERVO_LAG_TRAIN
        fields = {
            "delay_min_lag": int(lag["min_lag"]),
            "delay_max_lag": int(lag["max_lag"]),
            "delay_update_period": int(lag["update_period"]),
            "delay_hold_prob": float(lag["hold_prob"]),
            "delay_per_env_phase": True,
        }
    else:
        steps = int(mode)
        if not 0 <= steps <= 20:
            raise ValueError(f"servo lag {steps} physics steps outside 0–20 (0–100 ms)")
        fields = {
            "delay_min_lag": steps,
            "delay_max_lag": steps,
            "delay_update_period": 0,
            "delay_hold_prob": 0.0,
        }
    # upstream's make_* functions hand every env cfg the SAME module-level robot EntityCfg: copy the chain
    # (entity → articulation → actuators) instead of mutating it, or the lag would leak into every later cfg
    ent = cfg.scene.entities["robot"]
    art = dataclasses.replace(
        ent.articulation,
        actuators=tuple(dataclasses.replace(a, **fields) for a in ent.articulation.actuators),
    )
    entities = dict(cfg.scene.entities)
    entities["robot"] = dataclasses.replace(ent, articulation=art)
    cfg.scene.entities = entities


def runner_cfg(
    net: str, *, iterations: int | None = None, init_std: float = 1.0, seed: int = 1
) -> RslRlOnPolicyRunnerCfg:
    """PPO as the microduck recipes run it (512-256-128 ELU, obs normalisation baked at export,
    adaptive KL 0.01, 5 epochs × 4 mini-batches, 24 steps per env)."""
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=STEPS,
        max_iterations=int(iterations or ITERATIONS[net]),
        save_interval=250,
        experiment_name=f"skill_{net}",
        run_name=f"skill_{net}",
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
        ),
    )


# Positive-weight terms that are signed by design (potential-based progress: moving the wrong
# way costs), exempt from the "rewards log ≥ 0" check (costs are always checked).
SIGNED_TERMS = ("hop_progress",)


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    """Reward terms that are costs (negative weight at any curriculum stage)."""
    names = {n for n, t in cfg.rewards.items() if t.weight < 0}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


def register() -> None:
    """Register both tasks with mjlab's registry (``play`` = the evaluation variant)."""
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import register_mjlab_task

    for net, task_id in TASK_IDS.items():
        cells = [(v.name, _default_kind(v)) for v in C.variants(net)]
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_env_cfg(net),
            play_env_cfg=make_env_cfg(net, eval_cells=cells),
            rl_cfg=runner_cfg(net),
            runner_cls=MjlabOnPolicyRunner,
        )


def _default_kind(v: C.Variant) -> str:
    return "seated" if v.postures == ("sit",) else "stand"


def radians(deg: float) -> float:
    return math.radians(deg)


__all__ = [
    "ITERATIONS",
    "REWARD_REV",
    "TASK_IDS",
    "comfort_rewards",
    "cost_terms",
    "make_env_cfg",
    "register",
    "runner_cfg",
    "startle_rewards",
]
