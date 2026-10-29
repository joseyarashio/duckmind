"""The two Stage-2 mjlab tasks (training plan §3.3 R3, §3.4 R4): fine-tunes of VelStand.

``Mjlab-DuckWalkPose-Flat-MicroDuck`` (R3: sway, weight shift, lean, chest-out, crouch via robot.pose
and the fine standing head, at zero twist) and ``Mjlab-DuckWalkGaze-Flat-MicroDuck`` (R4: a steady gaze
while walking) — one walk network each, **the same task and observation as v5** (61-D actor, 14
actions), warm-started from v5 (R3) or from R3 (R4).

Both start from ``make_microduck_velstand_env_cfg`` (imported read-only from ``mjlab_microduck``) and
**collapse every curriculum to its final stage** — the network already walks, falls protectively and
rights itself (upstream's own warm-start rule: re-running the curricula re-teaches the walk under
easier conditions). What changes:

* **Robot**: the official ``robot_allcollisions`` + BAM XL330 (or the backlash twin for a final run),
  with the actuator's delay buffer widened to 0–40 ms and held (``hold_prob`` 1) so the action term
  draws the per-episode delay (``s2_mdp.S2Action``).
* **Action**: robotd's walk-slot path (scale 0.9 × voltage adaptation, low-pass 0.5 / 0.7, the raw
  previous output in the observation) — training value = deployment value.
* **Commands**: the Stage-2 program with robotd's smoothing (``s2_mdp``); the actor's observation
  terms read ``twist`` / ``head_pose`` / ``body_pose`` exactly as v5's did.
* **Rewards**: the velstand set at its final weights, with ``pose`` → the pose-aware ``posture`` and
  ``upright`` → ``upright_pose``, ``body_pose_tracking`` (weight 0) removed, and the Stage-2 terms
  ramped in after the critic warm-up (plan §3.3 / §3.4).
* **DR**: foot friction 0.5–1.3 (plan); R4 adds head mass ±5 %, gyro noise × 1.5 and 0–20 ms more
  IMU latency. Pushes (±0.3 m/s every 3–6 s, also during pose holds) and topple pushes stay.
* **Critic**: + the privileged ``s2_state`` (25 dims).

``make_env_cfg(stage, program=…, fixed_lag=…)`` builds evaluation variants (one program for every
env, one delay, no topple pushes).
"""

from __future__ import annotations

import dataclasses
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

from duckmind.skilltrain import s2_contract as C
from duckmind.skilltrain import s2_mdp as M

TASK_IDS = C.TASK_IDS
STEPS = C.STEPS_PER_ENV
# Reward revision (recorded in every run's metrics.json; later revisions say what changed and why).
# "p1" (s2:build, 2026-09-28): the first Stage-2 set — pose_track 3.0 (upstream StandUp's z/roll/pitch
# stds 10 mm / 5°), pose_composite 1.0, pose-aware posture and upright, still_stand 0.5,
# steps_standing −1, stand_leg_torque −20 above v5 + 10 %, release_rate −2 above 40 °/s, head_fine 1.0;
# R4 adds cam_smear −0.3 (⊥ rate above 10 °/s, squared, rad/s), gaze_yaw 1.0, head_torque −2, head_vel −0.05
# (Σ head qd², walking), neck_accel −1.
# P0 (Mac, n = 30, 2026-09-28): half of v5's walking camera shake on BAM is its own head wobble
# (head targets held at HOME: 61.8 → 30.2 °/s at 0.40), so cam_smear first pays for a still head.
REWARD_REV = "p1"
# "p2" (s2:train, 2026-09-29; opt-in with ``--reward-rev p2``, p1 stays the default): p1's battery at
# it 499 learned lean / sway / chest-out but crouch stayed 1–2 mm and the standing neck pull-back 0 (both
# Gaussians flat where the policy stands), and stepping while posed rose to 34/336 → pose_track's z also
# at σ 25 mm (averaged with 10 mm), head_fine also at σ 0.35 rad (averaged with 0.15), steps_standing −2.
REWARD_REVS: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {},
    "p2": {
        "pose_track": {"params": {"z_std_coarse": 0.025}},
        "head_fine": {"params": {"std_coarse": 0.35}},
        "steps_standing": {"final": -2.0, "stages": [(0, -2.0)]},
    },
}


def terms_for(
    stage: str, reward_rev: str = REWARD_REV
) -> dict[str, tuple[Any, float, list[tuple[int, float]], dict]]:
    """The Stage-2 reward terms of ``stage`` under ``reward_rev`` (R3's terms, + R4's for gaze)."""
    if reward_rev not in REWARD_REVS:
        raise ValueError(f"reward rev {reward_rev!r}; use one of {sorted(REWARD_REVS)}")
    terms = dict(R3_TERMS)
    if stage == "gaze":
        terms.update(R4_TERMS)
    for name, ch in REWARD_REVS[reward_rev].items():
        func, final, stages, params = terms[name]
        terms[name] = (
            func,
            ch.get("final", final),
            list(ch.get("stages", stages)),
            {**params, **ch.get("params", {})},
        )
    return terms


KEEP_CURRICULA_OFF = ("standing_envs", "head_pose_range", "body_pose_range")


def _it(i: int, scale: float = 1.0) -> int:
    return int(round(i * STEPS * scale))


def collapse_curricula(cfg: ManagerBasedRlEnvCfg) -> None:
    """Every curriculum at its final stage from step 0 (upstream's warm-start rule, applied to
    velstand's own phases too): reward-weight stages also write the final weight into the reward."""
    for _name, term in cfg.curriculum.items():
        for key, val in list(term.params.items()):
            if (
                isinstance(val, list)
                and val
                and all(isinstance(v, dict) and "step" in v for v in val)
            ):
                final = {**val[-1], "step": 0}
                term.params[key] = [final]
                if key == "weight_stages" and term.params.get("reward_name") in cfg.rewards:
                    cfg.rewards[term.params["reward_name"]].weight = final["weight"]
    for name in KEEP_CURRICULA_OFF:
        cfg.curriculum.pop(name, None)


def robot_cfg(backlash: bool = False, delay_steps: tuple[int, int] = C.DELAY_STEPS) -> Any:
    """The official all-collisions robot (or its backlash twin) with the delay buffer widened and
    held: lags 0..8 physics steps, set only by the action term."""
    from mjlab_microduck.robot import microduck_constants as K

    base = (
        K.MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
        if backlash
        else K.MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    )
    cfg = deepcopy(base)
    acts = tuple(
        dataclasses.replace(
            a,
            delay_min_lag=int(delay_steps[0]),
            delay_max_lag=int(delay_steps[1]),
            delay_hold_prob=1.0,
            delay_update_period=0,
        )  # fmt: skip
        for a in cfg.articulation.actuators
    )
    cfg.articulation = dataclasses.replace(cfg.articulation, actuators=acts)
    return cfg


def _ramp(name: str, stages: list[tuple[int, float]], scale: float) -> CurriculumTermCfg:
    return CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": name,
            "weight_stages": [{"step": _it(i, scale), "weight": w} for i, w in stages],
        },
    )


# (reward, final weight, ramp stages in iterations) — ramps start after the critic warm-up (100 it)
R3_TERMS: dict[str, tuple[Any, float, list[tuple[int, float]], dict]] = {
    "pose_track": (M.pose_track, 3.0, [(0, 0.0), (100, 1.5), (400, 3.0)], {}),
    "pose_composite": (M.pose_composite, 1.0, [(0, 0.0), (100, 0.5), (400, 1.0)], {}),
    "still_stand": (M.still_stand, 0.5, [(0, 0.5)], {}),
    "steps_standing": (M.steps_standing, -1.0, [(0, -1.0)], {}),
    "stand_leg_torque": (M.stand_leg_torque, -20.0, [(0, 0.0), (200, -10.0), (500, -20.0)], {}),
    "release_rate": (M.release_rate, -2.0, [(0, 0.0), (200, -1.0), (500, -2.0)], {}),
    "head_fine": (M.head_fine, 1.0, [(0, 0.0), (100, 0.5), (400, 1.0)], {}),
}
R4_TERMS: dict[str, tuple[Any, float, list[tuple[int, float]], dict]] = {
    # CPU smoke with v5 walking: cam_smear ≈ 1.9 (rad/s)² per walking tick (spikes to 7) — at −1.0 it would
    # rival the tracking terms (a "stop walking" hack); −0.3 prices v5's smear at ≈ −0.56 per tick
    "cam_smear": (M.cam_smear, -0.3, [(0, 0.0), (100, -0.1), (400, -0.2), (1000, -0.3)], {}),
    "gaze_yaw": (M.gaze_yaw, 1.0, [(0, 0.0), (100, 0.5), (400, 1.0)], {}),
    "head_torque": (M.head_torque, -2.0, [(0, 0.0), (200, -1.0), (600, -2.0)], {}),
    "head_vel": (M.head_vel, -0.05, [(0, 0.0), (100, -0.02), (400, -0.05)], {}),
    "neck_accel": (M.neck_accel, -1.0, [(0, 0.0), (200, -1.0)], {}),
}
POSE_RANGE_STAGES = [(0, 0.5), (250, 0.75), (500, 1.0)]


def _metrics(stage: str) -> dict[str, MetricsTermCfg]:
    out = {
        "fallen45": MetricsTermCfg(func=M.m_fallen),
        "pose_on": MetricsTermCfg(func=M.m_pose_on),
        "pose_err_deg": MetricsTermCfg(func=M.m_pose_err_deg),
        "lean_deg": MetricsTermCfg(func=M.m_lean_deg),
        "z_err_mm": MetricsTermCfg(func=M.m_z_err_mm),
        "stand_leg_rms": MetricsTermCfg(func=M.m_stand_leg_rms),
        "steps_standing": MetricsTermCfg(func=M.m_steps_standing),
        "neck_dev": MetricsTermCfg(func=M.m_neck_dev),
        "walking": MetricsTermCfg(func=M.m_walking),
        "lag_ms": MetricsTermCfg(func=M.m_lag_ms),
        "cam_perp_dps": MetricsTermCfg(func=M.m_cam_perp_dps),
        "gaze_err_deg": MetricsTermCfg(func=M.m_gaze_err_deg),
    }
    return out


def make_env_cfg(
    stage: str,
    *,
    program: str | None = None,
    fixed_lag: int | None = None,
    foot_mu: tuple[float, float] | None = None,
    pushes: bool = True,
    topple: bool = True,
    backlash: bool = False,
    curriculum_scale: float = 1.0,
    warm_weights: bool = False,
    topple_scale: float = 1.0,
    prone_scale: float = 1.0,
    reward_rev: str = REWARD_REV,
) -> ManagerBasedRlEnvCfg:
    """The training env cfg of ``stage`` (or, with ``program``, an evaluation variant: one program
    for every env, no curricula ramps — the final weights — and, unless asked, no topple pushes).
    ``warm_weights``: every Stage-2 term at its final weight from step 0 (R4 keeps R3's terms so)."""
    if stage not in C.STAGES:
        raise ValueError(f"stage {stage!r}; use one of {C.STAGES}")
    if not 0.05 <= curriculum_scale <= 2.0:
        raise ValueError(f"curriculum scale {curriculum_scale} outside 0.05–2")
    eval_mode = program is not None
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    collapse_curricula(cfg)
    if backlash:
        from mjlab_microduck.robot.microduck_constants import (
            MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
        )
        from mjlab_microduck.tasks.backlash import make_backlash_variant

        make_backlash_variant(cfg, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg(backlash)}

    # action: robotd's walk slot + the per-episode delay
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = M.S2ActionCfg(
        entity_name=old.entity_name,
        actuator_names=old.actuator_names,
        scale=1.0,
        use_default_offset=True,
        skill_action_scale=C.WALK_ACTION_SCALE,
        fixed_lag=fixed_lag,
    )
    # commands: the program + robotd's smoothing (the obs terms read these three by name)
    cfg.commands = {
        "twist": M.S2TwistCommandCfg(stage=stage, fixed_program=program),
        "head_pose": M.S2HeadCommandCfg(),
        "body_pose": M.S2BodyCommandCfg(),
    }
    cfg.observations["critic"].terms["s2_state"] = ObservationTermCfg(func=M.s2_state_obs)
    if stage == "gaze":
        actor = cfg.observations["actor"].terms
        from mjlab.utils.noise import UniformNoiseCfg as Unoise

        n = actor["base_ang_vel"].noise
        lim = C.GYRO_NOISE_SCALE * (getattr(n, "n_max", 0.03) or 0.03)
        actor["base_ang_vel"].noise = Unoise(n_min=-lim, n_max=lim)
        for name in ("base_ang_vel", "projected_gravity"):
            if name in actor:
                actor[name].delay_max_lag = 1 + C.IMU_EXTRA_LAG_TICKS[1]

    # events: the recipe's final DR + the floor down to μ 0.5; R4 head mass ±5 %
    cfg.events["foot_friction"].params["ranges"] = foot_mu or C.FOOT_MU
    if stage == "gaze":
        from mjlab.envs.mdp import dr
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES

        lo, hi = C.HEAD_MASS_SCALE
        cfg.events["randomize_head_mass"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0),
            },  # fmt: skip
        )
    if eval_mode and not topple:
        cfg.events.pop("topple_push", None)
    # the recipe's final topple pushes (±1.2 m/s every 4–8 s) and prone / crouch spawns (45 % / 20 %) keep
    # the righting alive; S0 measures the share of fallen time and may thin them (the BC fallen expert
    # keeps v5's righting on the fallen frames either way)
    if not eval_mode and "topple_push" in cfg.events and topple_scale != 1.0:
        if topple_scale <= 0:
            cfg.events.pop("topple_push", None)
            cfg.curriculum.pop("topple_push_range", None)
        else:
            lo, hi = cfg.events["topple_push"].interval_range_s
            cfg.events["topple_push"].interval_range_s = (lo / topple_scale, hi / topple_scale)
    if not eval_mode and prone_scale != 1.0 and "prone_init_prob" in cfg.curriculum:
        stages = cfg.curriculum["prone_init_prob"].params["param_stages"]
        for st in stages:
            for key in ("prone_prob", "crouch_prob"):
                if key in st["params"]:
                    st["params"][key] = float(st["params"][key]) * max(0.0, prone_scale)
    if not pushes:
        cfg.events.pop("push_robot", None)
        cfg.events.pop("topple_push", None)
    if eval_mode:
        # evaluation spawns standing (no prone / crouch slices)
        if "random_prone_init" in cfg.events:
            cfg.events["random_prone_init"].params.update(prone_prob=0.0, crouch_prob=0.0)
        cfg.curriculum.pop("prone_init_prob", None)
        cfg.curriculum.pop("topple_push_range", None)

    # rewards
    rw = cfg.rewards
    rw.pop("pose", None)
    rw.pop("body_pose_tracking", None)
    rw["posture"] = RewardTermCfg(func=M.posture, weight=1.0)
    rw["upright"] = RewardTermCfg(func=M.upright_pose, weight=rw["upright"].weight)
    terms = terms_for(stage, reward_rev)
    ramp_all = not (eval_mode or warm_weights)
    for name, (func, final, stages, params) in terms.items():
        is_r3_in_gaze = stage == "gaze" and name in R3_TERMS
        start = final if (not ramp_all or is_r3_in_gaze) else stages[0][1]
        rw[name] = RewardTermCfg(func=func, weight=start, params=dict(params))
        if ramp_all and not is_r3_in_gaze and len(stages) > 1:
            cfg.curriculum[f"{name}_weight"] = _ramp(name, stages, curriculum_scale)
    if ramp_all and stage == "pose":
        cfg.curriculum["pose_range"] = CurriculumTermCfg(
            func=M.set_pose_range,
            params={
                "scale_stages": [
                    {"step": _it(i, curriculum_scale), "scale": s} for i, s in POSE_RANGE_STAGES
                ]
            },
        )
    else:
        cfg.curriculum["pose_range"] = CurriculumTermCfg(
            func=M.set_pose_range, params={"scale_stages": [{"step": 0, "scale": 1.0}]}
        )
    cfg.metrics = _metrics(stage)
    return cfg


@dataclass
class S2PpoCfg(RslRlPpoAlgorithmCfg):
    """PPO + the Stage-2 extras (critic warm-up, gated dual-teacher BC): ``s2_bc.S2Ppo``."""

    s2_cfg: dict | None = None
    class_name: str = "duckmind.skilltrain.s2_bc.S2Ppo"


def runner_cfg(
    stage: str,
    *,
    iterations: int | None = None,
    init_std: float = C.INIT_STD,
    seed: int = 1,
    critic_warmup: int = C.CRITIC_WARMUP_ITERS,
    learning_rate: float = C.LEARNING_RATE,
) -> RslRlOnPolicyRunnerCfg:
    """PPO as the microduck recipes run it (512-256-128 ELU, adaptive KL 0.01, 5 × 4, 24 steps),
    with the Stage-2 algorithm (``s2_cfg`` carries the stage, the BC spec and the warm-up)."""
    bc = C.BC[stage]
    s2 = {
        "stage": stage,
        "critic_warmup": int(critic_warmup),
        "actor_lr": float(learning_rate),
        "bc": dataclasses.asdict(bc),
    }
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=STEPS,
        max_iterations=int(iterations or C.DEFAULT_ITERATIONS[stage]),
        save_interval=250,
        experiment_name=f"s2_{stage}",
        run_name=f"s2_{stage}",
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
        algorithm=S2PpoCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,  # the critic's warm-up; the actor phase starts at ``actor_lr``
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            s2_cfg=s2,
        ),
    )


# Positive-weight terms that are signed by design (upstream's self-negating or potential-based
# terms), exempt from the "rewards log ≥ 0" check; costs are always checked.
SIGNED_TERMS = ("head_pose_bias", "gentle_rise", "upright_progress", "height_progress")
MAIN_TERMS = {"pose": ("pose_track",), "gaze": ("gaze_yaw", "pose_track")}


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    names = {n for n, t in cfg.rewards.items() if t.weight < 0}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


def register() -> None:
    """Register both tasks with mjlab's registry (``play`` = a pose / walking evaluation)."""
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import register_mjlab_task

    for stage, task_id in TASK_IDS.items():
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_env_cfg(stage),
            play_env_cfg=make_env_cfg(stage, program="pose" if stage == "pose" else "walk_neuro"),
            rl_cfg=runner_cfg(stage),
            runner_cls=MjlabOnPolicyRunner,
        )


__all__ = [
    "MAIN_TERMS",
    "R3_TERMS",
    "R4_TERMS",
    "REWARD_REV",
    "REWARD_REVS",
    "SIGNED_TERMS",
    "terms_for",
    "TASK_IDS",
    "collapse_curricula",
    "cost_terms",
    "make_env_cfg",
    "register",
    "robot_cfg",
    "runner_cfg",
]
