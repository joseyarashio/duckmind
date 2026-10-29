"""The panic-gait mjlab task (``Mjlab-DuckPanicWalk-Flat-MicroDuck``): VelStand, fine-tuned from v5.

Built on ``make_microduck_velstand_env_cfg`` like Stage 2 (``s2_tasks``): every inherited curriculum is
collapsed to its final stage (the network already walks), the robot is the official all-collisions + BAM
XL330 with the delay buffer widened to 0–40 ms (a per-episode delay, ``s2_mdp.S2Action``), and the action
is robotd's walk-slot path (scale 0.9 × voltage adaptation, low-pass 0.5 / 0.7). What changes:

* **Command**: ``pg_mdp.PgTwistCommand`` — upstream's draw (50 % of the non-standing walks) plus a
  strolling band (vx 0.35–0.70) and a panic band (vx 0.6 → 1.0, straight or arcs), through robotd's twist
  EMA; the standing, forward-only and turn-in-place buckets are kept. The panic ceiling widens by curriculum.
* **Rewards**: upstream's set at its final weights, plus ``track_vel_sharp``, ``track_vel_forward`` (progress
  capped at the command), ``track_yaw_sharp`` and ``fall_event`` (a one-shot cost on tipping past 60°);
  ``pose`` gets a wider ``std_running`` from |twist| 0.65, ``air_time`` accepts a 0.09 s swing.
* **DR**: the recipe's, foot friction 0.5–1.3, topple pushes / prone spawns thinned (v5 is the fallen expert).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

from duckmind.skilltrain import pg_contract as C
from duckmind.skilltrain import pg_mdp as M
from duckmind.skilltrain import pg_xml
from duckmind.skilltrain import s2_mdp as S2M
from duckmind.skilltrain import s2_tasks as S2T

STEPS = C.STEPS_PER_ENV
BASE_TERMS: dict[str, tuple[Any, float, dict]] = {
    "track_vel_sharp": (M.track_vel_sharp, 2.0, {"std": 0.18}),
    "track_vel_forward": (M.track_vel_forward, 3.0, {"gate": 0.3}),
    "track_yaw_sharp": (M.track_yaw_sharp, 1.5, {"std": 0.25}),
    "fall_event": (M.fall_event, -10.0, {"tilt_deg": 60.0}),
}
YAWMEAN = (M.yaw_mean, 4.0, {"sigma": 0.07})  # g10
HEADING = (M.heading_hold, 2.0, {"sigma": 0.25})  # g8 only
COARSE = (
    M.track_vel_coarse,
    2.0,
    {"std": 0.31622776601683794, "mapped": True},
)  # replaces upstream's, g5 only


def terms_for(reward_rev: str) -> dict[str, tuple[Any, float, dict]]:
    if reward_rev not in C.REWARD_REVS:
        raise ValueError(f"reward rev {reward_rev!r}; use one of {sorted(C.REWARD_REVS)}")
    out = {k: (f, w, dict(p)) for k, (f, w, p) in BASE_TERMS.items()}
    if "track_vel_coarse" in C.REWARD_REVS[reward_rev]:
        out["track_vel_coarse"] = (COARSE[0], COARSE[1], dict(COARSE[2]))
    if "heading_hold" in C.REWARD_REVS[reward_rev]:
        out["heading_hold"] = (HEADING[0], HEADING[1], dict(HEADING[2]))
    if "yaw_mean" in C.REWARD_REVS[reward_rev]:
        out["yaw_mean"] = (YAWMEAN[0], YAWMEAN[1], dict(YAWMEAN[2]))
    for name, ch in C.REWARD_REVS[reward_rev].items():
        f, w, p = out[name]
        out[name] = (f, ch.get("weight", w), {**p, **ch.get("params", {})})
    return out


def collapse_curricula(cfg: ManagerBasedRlEnvCfg) -> None:
    """Every curriculum at its final stage from step 0 (the standing / head / body-pose draws included:
    their curricula would otherwise start from the velocity recipe's tiny first stage)."""
    for term in cfg.curriculum.values():
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


def _metrics() -> dict[str, MetricsTermCfg]:
    names = (
        "fast_n",
        "fast_cmd",
        "fast_vx",
        "ord_n",
        "ord_cmd",
        "ord_vx",
        "arc_n",
        "arc_cmd",
        "arc_yaw",
        "fell60",
        "str_n",
        "str_yawerr",
        "yawlp_abs",
    )
    out = {n: MetricsTermCfg(func=getattr(M, f"m_{n}")) for n in names}
    out["lag_ms"] = MetricsTermCfg(func=S2M.m_lag_ms)
    return out


def make_env_cfg(
    *,
    reward_rev: str = C.DEFAULT_REWARD_REV,
    fixed_lag: int | None = None,
    topple_scale: float = 0.5,
    prone_scale: float = 0.4,
    pushes: bool = True,
    vmax_scale: float = 1.0,
    p_fast: float | None = None,
    p_xml: float = 0.0,
    action_rate: float | None = None,
    vmax_stages: tuple[tuple[int, float], ...] | None = None,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    collapse_curricula(cfg)
    cfg.scene.entities = {**cfg.scene.entities, "robot": S2T.robot_cfg(False)}
    old = cfg.actions["joint_pos"]
    if p_xml > 0:
        robot = cfg.scene.entities["robot"]
        robot.articulation = dataclasses.replace(
            robot.articulation, actuators=pg_xml.mix_actuators(robot.articulation.actuators)
        )
        cfg.actions["joint_pos"] = pg_xml.PgActionCfg(
            entity_name=old.entity_name, actuator_names=old.actuator_names, scale=1.0, use_default_offset=True,
            skill_action_scale=C.WALK_ACTION_SCALE, fixed_lag=fixed_lag, p_xml=float(p_xml),
        )  # fmt: skip
    else:
        cfg.actions["joint_pos"] = S2M.S2ActionCfg(
            entity_name=old.entity_name,
            actuator_names=old.actuator_names,
            scale=1.0,
            use_default_offset=True,
            skill_action_scale=C.WALK_ACTION_SCALE,
            fixed_lag=fixed_lag,
        )
    # command: upstream's term → the panic mixture (same fields)
    base = cfg.commands["twist"]
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    if p_fast is not None:
        fields["p_fast"] = float(p_fast)
    cfg.commands["twist"] = M.PgTwistCommandCfg(**fields)
    cfg.curriculum["panic_vmax"] = CurriculumTermCfg(
        func=M.set_vmax,
        params={
            "vmax_stages": [
                {"step": it * STEPS, "vmax": v * vmax_scale}
                for it, v in (vmax_stages or C.VMAX_STAGES)
            ]
        },
    )
    cfg.events["foot_friction"].params["ranges"] = C.FOOT_MU
    if "topple_push" in cfg.events and topple_scale != 1.0:
        if topple_scale <= 0:
            cfg.events.pop("topple_push", None)
            cfg.curriculum.pop("topple_push_range", None)
        else:
            lo, hi = cfg.events["topple_push"].interval_range_s
            cfg.events["topple_push"].interval_range_s = (lo / topple_scale, hi / topple_scale)
    if prone_scale != 1.0 and "prone_init_prob" in cfg.curriculum:
        for st in cfg.curriculum["prone_init_prob"].params["param_stages"]:
            for key in ("prone_prob", "crouch_prob"):
                if key in st["params"]:
                    st["params"][key] = float(st["params"][key]) * max(0.0, prone_scale)
    if not pushes:
        cfg.events.pop("push_robot", None)
        cfg.events.pop("topple_push", None)
    rw = cfg.rewards
    if (
        action_rate is not None
    ):  # the collapsed curriculum rewrites the weight at every reset: change its stage too
        rw["action_rate_l2"].weight = float(action_rate)
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": float(action_rate)}
        ]
    rw["pose"].params["std_running"] = dict(C.STD_RUNNING)
    rw["pose"].params["running_threshold"] = C.RUNNING_THRESHOLD
    rw["air_time"].params["threshold_min"] = C.AIR_TIME_MIN
    if "track_vel_coarse" in C.REWARD_REVS[reward_rev]:
        rw.pop("track_linear_velocity", None)  # the same term against the mapped goal (below)
    for name, (func, weight, params) in terms_for(reward_rev).items():
        rw[name] = RewardTermCfg(func=func, weight=weight, params=dict(params))
    if "heading_hold" in C.REWARD_REVS[reward_rev]:
        from mjlab.managers import ObservationTermCfg

        cfg.observations["critic"].terms["pg_heading_err"] = ObservationTermCfg(
            func=M.heading_err_obs
        )
    cfg.metrics = _metrics()
    if p_xml > 0:
        for n in ("xml", "fast_xml_n", "fast_xml_cmd", "fast_xml_vx"):
            cfg.metrics[n] = MetricsTermCfg(func=getattr(pg_xml, f"m_{n}"))
    return cfg


@dataclass
class PgPpoCfg(RslRlPpoAlgorithmCfg):
    """PPO + the critic warm-up and gated dual-teacher cloning (``pg_bc.PgPpo``)."""

    s2_cfg: dict | None = None
    class_name: str = "duckmind.skilltrain.pg_bc.PgPpo"


def runner_cfg(
    *,
    iterations: int = 2000,
    init_std: float = C.INIT_STD,
    seed: int = 1,
    critic_warmup: int = C.CRITIC_WARMUP_ITERS,
    learning_rate: float = C.LEARNING_RATE,
    anchor_coef: float | None = None,
    entropy_coef: float = 0.01,
) -> RslRlOnPolicyRunnerCfg:
    s2 = {
        "stage": "panic",
        "critic_warmup": int(critic_warmup),
        "actor_lr": float(learning_rate),
        "bc": {k: (list(v) if isinstance(v, tuple) else v) for k, v in C.BC.items()},
    }
    s2["bc"]["anchor_coef"] = [list(x) for x in C.BC["anchor_coef"]]
    if anchor_coef is not None:
        s2["bc"]["anchor_coef"] = [[0, float(anchor_coef)]]
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=STEPS,
        max_iterations=int(iterations),
        save_interval=250,
        experiment_name="pg_panic",
        run_name="pg_panic",
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
        algorithm=PgPpoCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=entropy_coef,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            s2_cfg=s2,
        ),
    )


SIGNED_TERMS = ("head_pose_bias", "gentle_rise", "upright_progress", "height_progress")
MAIN_TERMS = ("track_vel_sharp", "track_vel_forward")


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    names = {n for n, t in cfg.rewards.items() if t.weight < 0}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


__all__ = ["MAIN_TERMS", "SIGNED_TERMS", "cost_terms", "make_env_cfg", "runner_cfg", "terms_for"]
