"""The queue's mjlab tasks (``Mjlab-DuckExpressWalk / QuickTurn / CalmWalk / StyleWalk-Flat-MicroDuck``): VelStand fine-tunes.

Built like the panic round's task (``pg_tasks``): ``make_microduck_velstand_env_cfg`` with every curriculum collapsed to its
final stage (the network already walks), the official all-collisions robot with the BAM XL330 mixed with the simulator's XML
position servo per episode (``pg_xml``), a 0-40 ms servo delay, robotd's action path (scale 0.9 x voltage adaptation,
low-pass 0.5 / 0.7), the recipe's DR with foot friction 0.5-1.3 and thinned topple pushes / prone spawns. What changes per item:

* ``ex``    the head / body pose commands are drawn over the ranges the 14 emotions use and tracked (``qt_mdp.ex_head``,
            ``ex_trunk``); upstream's ``pose`` / ``upright`` (which pull toward a level, default stance) are made pose-aware.
* ``turn``  the twist mixture is mostly turning in place at 0.5-3.0 rad/s (a share unsmoothed: a skill's constant command),
            paid by ``turn_rate`` / ``turn_track`` / ``turn_still``.
* ``calm``  the camera's world angular speed is charged while walking with no head command; the legs stay v5's (anchor).
* ``style`` the style code rides in obs[55:57]; each style has its own height / lean / clearance / cadence / speed targets.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict  # noqa: F401
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

from duckmind.skilltrain import pg_contract as P
from duckmind.skilltrain import pg_xml
from duckmind.skilltrain import qt_contract as C
from duckmind.skilltrain import qt_mdp as M
from duckmind.skilltrain import s2_mdp as S2M
from duckmind.skilltrain import s2_tasks as S2T
from duckmind.skilltrain.pg_tasks import collapse_curricula

STEPS = C.STEPS_PER_ENV


def item_terms(item: str, rev: str) -> tuple[dict[str, tuple[Any, float, dict]], list[str]]:
    """(new reward terms {name: (func, weight, params)}, upstream terms to drop) for an item and reward revision."""
    fall = ("fall_event", (M.fall_event, -10.0, {"tilt_deg": 60.0}))
    if item == "ex":
        new = dict(
            [
                ("ex_head", (M.ex_head, 6.0, {"std": 0.5, "mid_std": 0.2, "fine_std": 0.08})),
                ("ex_neck", (M.ex_neck, 4.0, {"std": 0.15})),
                (
                    "ex_trunk",
                    (M.ex_trunk, 5.0, {"pitch_std": 0.045, "z_std": 0.007, "roll_std": 0.06}),
                ),
                ("ex_upright", (M.ex_upright, 2.0, {})),
                ("ex_posture", (M.posture, 1.0, {})),
                fall,
            ]
        )
        return new, ["head_pose_tracking", "upright", "pose"]
    if item == "turn":
        new = dict(
            [
                ("turn_rate", (M.turn_rate, 5.0, {"goal_k": C.TURN_GOAL_K, "gate": 0.4})),
                ("turn_low", (M.turn_low, 4.0, {})),
                ("turn_track", (M.turn_track, 4.0, {"std": 0.3})),
                ("turn_still", (M.turn_still, 3.0, {"std": 0.06})),
                fall,
            ]
        )
        return new, []
    if item == "calm":
        new = dict(
            [
                ("calm_cam", (M.calm_cam, -0.6, {"floor_dps": 8.0})),
                ("calm_head_vel", (M.calm_head_vel, -0.01, {})),
                fall,
            ]
        )
        return new, []
    if item == "style":
        from duckmind.skilltrain import qt_style as QS

        return QS.style_terms(rev), ["pose", "upright", "foot_clearance", "foot_swing_height"]
    raise ValueError(f"item {item!r}; use one of {C.ITEMS}")


def _metrics(item: str) -> dict[str, MetricsTermCfg]:
    names: dict[str, tuple[str, ...]] = {
        "ex": (
            "head_err", "pitch_err_deg", "z_err_mm", "pitch_cmd_on", "pitch_got", "pitch_cmd", "head_cmd_on",
            "walk_n", "walk_cmd", "walk_vx", "walk_zero_pose_n", "walk_zero_pose_cmd", "walk_zero_pose_vx", "fell60",
        ),
        "turn": ("turn_n", "turn_cmd", "turn_rate", "turn_drift", "walk_n", "walk_cmd", "walk_vx", "fell60"),
        "calm": ("cam_dps", "cam_n", "walk_n", "walk_cmd", "walk_vx", "fell60"),
        "style": ("walk_n", "walk_cmd", "walk_vx", "fell60"),
    }  # fmt: skip
    out = {n: MetricsTermCfg(func=getattr(M, f"m_{n}")) for n in names[item]}
    out["lag_ms"] = MetricsTermCfg(func=S2M.m_lag_ms)
    if item == "style":
        from duckmind.skilltrain import qt_style as QS

        out.update(QS.style_metrics())
    return out


def make_env_cfg(
    item: str,
    *,
    rev: str | None = None,
    fixed_lag: int | None = None,
    topple_scale: float = 0.5,
    prone_scale: float = 0.4,
    pushes: bool = True,
    p_xml: float = 0.5,
    action_rate: float | None = None,
    panic: bool = False,
    style_probs: tuple[float, ...] | None = None,
    stand_tune: float | None = None,
) -> ManagerBasedRlEnvCfg:
    if item not in C.ITEMS:
        raise ValueError(f"item {item!r}; use one of {C.ITEMS}")
    rev = rev or C.DEFAULT_REV[item]
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    collapse_curricula(cfg)
    cfg.scene.entities = {**cfg.scene.entities, "robot": S2T.robot_cfg(False)}
    old = cfg.actions["joint_pos"]
    robot = cfg.scene.entities["robot"]
    robot.articulation = dataclasses.replace(
        robot.articulation, actuators=pg_xml.mix_actuators(robot.articulation.actuators)
    )
    cfg.actions["joint_pos"] = M.QtActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names, scale=1.0, use_default_offset=True,
        skill_action_scale=C.WALK_ACTION_SCALE, fixed_lag=fixed_lag, p_xml=float(p_xml),
        p_stand_tune=float(stand_tune if stand_tune is not None else (C.EX_P_STAND_TUNE if item == "ex" else 0.0)),
    )  # fmt: skip
    # ── commands ──
    base = cfg.commands["twist"]
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    mix = {
        "ex": C.EX_TWIST_MIX,
        "turn": C.TURN_TWIST_MIX,
        "calm": C.CALM_TWIST_MIX,
        "style": C.CALM_TWIST_MIX,
    }[item]
    if item == "style":
        from duckmind.skilltrain import qt_style as QS

        mix = QS.TWIST_MIX
    if item == "calm" and not panic:  # the panic band only for the panic variant
        mix = tuple(m for m in mix if m[2] != 1.0)
    fields["resampling_time_range"] = C.TURN_RESAMPLE_S if item == "turn" else (2.0, 6.0)
    cfg.commands["twist"] = M.QtTwistCommandCfg(
        **fields, mix=mix, p_instant=C.TURN_P_INSTANT if item == "turn" else 0.0
    )
    # the head: ex draws the emotions' ranges; the others a mostly-quiet head inside upstream's final ranges
    if item == "ex":
        cfg.commands["head_pose"] = M.QtPoseCommandCfg(
            resampling_time_range=C.EX_HEAD_RESAMPLE_S, ranges=C.EX_HEAD_RANGES, alphas=C.EX_HEAD_ALPHA,
            rate=C.EX_HEAD_RATE, p_zero=C.EX_P_HEAD_ZERO, p_axis_off=C.EX_P_AXIS_OFF, start_scale=C.EX_RANGE_STAGES[0][1],
        )  # fmt: skip
        cfg.commands["body_pose"] = M.QtPoseCommandCfg(
            resampling_time_range=C.EX_BODY_RESAMPLE_S, ranges=C.EX_BODY_RANGES, alphas=((P.CMD_ALPHA, 1.0),),
            rate=C.EX_BODY_RATE, axis_rate=(1.0, 1.0, 0.1, 1.0, 1.0, 1.0), p_zero=C.EX_P_BODY_ZERO,
            p_axis_off=C.EX_P_AXIS_OFF, p_snap=0.2, start_scale=C.EX_RANGE_STAGES[0][1],
        )  # fmt: skip
        cfg.curriculum["qt_pose_scale"] = CurriculumTermCfg(
            func=M.set_pose_scale,
            params={"stages": [{"step": it * STEPS, "scale": s} for it, s in C.EX_RANGE_STAGES]},
        )
    else:
        quiet = 0.85 if item == "calm" else 0.6
        cfg.commands["head_pose"] = M.QtPoseCommandCfg(
            resampling_time_range=(2.0, 5.0), ranges=tuple((-r, r) for r in (0.55, 0.55, 0.7, 0.15)),
            alphas=C.EX_HEAD_ALPHA, rate=(0.5, 3.0), p_zero=quiet, p_axis_off=0.3,
        )  # fmt: skip
        if item == "style":
            from duckmind.skilltrain import qt_style as QS

            cfg.commands["body_pose"] = M.StyleCommandCfg(
                resampling_time_range=(3.0, 8.0), ranges=((0.0, 0.0),) * 6, style_probs=style_probs or QS.STYLE_PROBS,
            )  # fmt: skip
        else:
            cfg.commands["body_pose"] = M.QtPoseCommandCfg(
                resampling_time_range=(3.0, 8.0), ranges=((0.0, 0.0),) * 6, p_zero=1.0
            )
    # ── DR as the panic round ──
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
    # ── rewards ──
    rw = cfg.rewards
    if action_rate is not None:
        rw["action_rate_l2"].weight = float(action_rate)
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": float(action_rate)}
        ]
    if (
        item == "turn"
    ):  # a faster swing is allowed, the pose tolerance widens at speed (the panic round's numbers)
        rw["pose"].params["std_running"] = dict(P.STD_RUNNING)
        rw["pose"].params["running_threshold"] = P.RUNNING_THRESHOLD
        rw["air_time"].params["threshold_min"] = P.AIR_TIME_MIN
    new, drop = item_terms(item, rev)
    for name in drop:
        rw.pop(name, None)
    for name, (func, weight, params) in new.items():
        rw[name] = RewardTermCfg(func=func, weight=weight, params=dict(params))
    cfg.metrics = _metrics(item)
    for n in ("xml",):
        cfg.metrics[n] = MetricsTermCfg(func=getattr(pg_xml, f"m_{n}"))
    cfg.metrics["stand_tuned"] = MetricsTermCfg(func=M.m_stand_tuned)
    return cfg


@dataclasses.dataclass
class QtPpoCfg(RslRlPpoAlgorithmCfg):
    """PPO + the critic warm-up and gated dual-teacher cloning (``qt_bc.QtPpo``)."""

    s2_cfg: dict | None = None
    class_name: str = "duckmind.skilltrain.qt_bc.QtPpo"


def runner_cfg(
    item: str,
    *,
    iterations: int = 1000,
    init_std: float = C.INIT_STD,
    seed: int = 1,
    critic_warmup: int = C.CRITIC_WARMUP_ITERS,
    learning_rate: float = C.LEARNING_RATE,
    anchor_coef: float | None = None,
    entropy_coef: float = 0.002,
    anchor_dims: tuple[int, ...] | None = None,
) -> RslRlOnPolicyRunnerCfg:
    bc = {k: (list(v) if isinstance(v, tuple) else v) for k, v in C.BC.items()}
    bc["rule"] = C.ANCHOR_RULE[item]
    bc["anchor_coef"] = [list(x) for x in C.BC["anchor_coef"]]
    if anchor_coef is not None:
        bc["anchor_coef"] = [[0, float(anchor_coef)]]
    if anchor_dims is not None:
        bc["anchor_dims"] = list(anchor_dims)
    elif item in ("ex", "calm"):
        bc["anchor_dims"] = list(
            C.LEG_IDX
        )  # the head is free (ex: head tracking; calm: stabilising)
    s2 = {
        "stage": "queue", "critic_warmup": int(critic_warmup), "actor_lr": float(learning_rate), "bc": bc,
    }  # fmt: skip
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=STEPS,
        max_iterations=int(iterations),
        save_interval=250,
        experiment_name=f"qt_{item}",
        run_name=f"qt_{item}",
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
        algorithm=QtPpoCfg(
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


def cost_terms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
    names = {n for n, t in cfg.rewards.items() if t.weight < 0}
    for term in cfg.curriculum.values():
        stages = term.params.get("weight_stages") if isinstance(term.params, dict) else None
        if stages and any(s["weight"] < 0 for s in stages):
            names.add(term.params["reward_name"])
    return sorted(names)


__all__ = ["SIGNED_TERMS", "cost_terms", "item_terms", "make_env_cfg", "runner_cfg"]
