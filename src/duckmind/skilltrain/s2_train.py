"""Fine-tune the walk network for Stage 2 (s2:build): PPO on the official physics → robotd's ONNX.

    python -m duckmind.skilltrain.s2_train --stage pose --envs 4096 --iterations 1500 \
        --device cuda:0 --v5 /content/stage2/policies --init v5 --out /content/out/pose-p1 \
        --time-budget-min 110
    python -m duckmind.skilltrain.s2_train --stage gaze ... --init /content/out/pose-p1/checkpoints/model_1499.pt \
        --anchor-onnx /content/out/pose-p1/final/velstand_duck.onnx --out /content/out/gaze-g1

``--init``: ``v5`` (R3; the v5 ONNX imported with the body slots re-conditioned, a fresh critic, a
100-iteration critic warm-up) or a Stage-2 checkpoint (R4 from R3: actor and critic loaded, the
curricula start at 0, a 30-iteration warm-up); ``--resume`` continues a run of the same stage
(iteration, optimiser and curricula restored). The teachers (``s2_bc``): the fallen expert is always
v5; the walk anchor is v5 for R3 and the start network (``--anchor-onnx``, else the init actor) for R4.

Writes into ``--out`` (all small enough to download): ``progress.jsonl`` (one line per iteration:
s/it, every ``Episode_Reward/…`` and ``Episode_Metrics/…``, the PPO and BC losses, lr, std, the
sign-check verdicts), ``checkpoints/model_<it>.pt`` every ``--save-every`` (and ``latest.pt``),
``final/velstand_duck.onnx`` (v5's graph shape, normaliser baked) + ``parity.json`` +
``manifest.json`` (schema 2, the walk slot) and ``metrics.json`` (timings, units, stop reason).

Stop rules (training plan §5): an ``Episode_Reward`` of a cost above 0 or of a reward below 0 (a sign
bug; upstream's self-negating terms exempt) stops the run at once; the main term flat over 500
iterations stops it at the 500 / 1,000 checks; the time budget saves and exports before the cap.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from duckmind.skilltrain import s2_contract as C
from duckmind.skilltrain.rl_train import (
    Recorder,
    _jsonable,
    _now,
    _s_per_it,
    _tail_means,
    flat_terms,
    save_checkpoint,
    sign_problems,
)

FLAT_CHECKS = (500, 1000)


def build(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401  (upstream's NaN-safe patches ride on this import)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from duckmind.skilltrain import s2_tasks as T

    torch.manual_seed(args.seed)
    env_cfg = T.make_env_cfg(
        args.stage,
        backlash=args.backlash,
        curriculum_scale=args.curriculum_scale,
        pushes=not args.no_pushes,
        topple_scale=args.topple_scale,
        prone_scale=args.prone_scale,
        reward_rev=args.reward_rev,
    )
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    warmup = args.critic_warmup
    if warmup < 0:
        warmup = C.CRITIC_WARMUP_ITERS if args.init == "v5" else (0 if args.resume else 30)
    agent = T.runner_cfg(args.stage, iterations=args.iterations, init_std=args.init_std, seed=args.seed,
                         critic_warmup=warmup, learning_rate=args.lr)  # fmt: skip
    if args.desired_kl:
        agent.algorithm.desired_kl = float(args.desired_kl)
    agent_dict = asdict(agent)
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, log_dir=None, device=args.device)
    return env, wrapped, runner, {"env_cfg": env_cfg, "agent": agent_dict, "critic_warmup": warmup}


def warm(runner: Any, wrapped: Any, args: argparse.Namespace, log: Any) -> dict:
    """The actor's start (v5 import or a Stage-2 checkpoint) and the teachers."""
    import torch

    from duckmind.skilltrain import rl_onnx, s2_bc

    actor = runner.alg.actor
    v5 = Path(args.v5) / "velstand.onnx"
    got = rl_onnx.sha256(v5)
    if got != C.V5_SHA256["velstand.onnx"]:
        raise RuntimeError(f"{v5}: sha256 {got[:16]} is not the validated v5")
    info: dict[str, Any] = {}
    obs = wrapped.get_observations()["actor"].detach().cpu().numpy()
    if args.resume:
        runner.load(args.resume, map_location=args.device)
        rl_onnx.freeze_normalizer(actor)
        runner.alg.iteration = int(runner.current_learning_iteration) + 1
        runner.alg.warmup_left = 0
        info["resumed_from"] = {"path": args.resume, "iteration": runner.current_learning_iteration}
        log(f"resumed {args.resume} at iteration {runner.current_learning_iteration}")
    elif args.init == "v5":
        info["import"] = s2_bc.import_v5(actor, v5, init_std=args.init_std, obs=obs)
        par = info["import"]["parity"]["max_abs"]
        log(f"actor ← v5 velstand ({got[:12]}), body slots re-conditioned {info['import']['recondition']}, "
            f"parity {par:.2e}, init std {args.init_std}")  # fmt: skip
    else:
        ck = torch.load(args.init, map_location=args.device, weights_only=False)
        actor.load_state_dict(ck["actor_state_dict"], strict=True)
        runner.alg.critic.load_state_dict(ck["critic_state_dict"], strict=True)
        rl_onnx.freeze_normalizer(actor)
        with torch.no_grad():
            actor.distribution.std_param.fill_(float(args.init_std))
        info["init"] = {
            "path": args.init,
            "sha256": rl_onnx.sha256(args.init),
            "iteration": int(ck.get("iter", -1)),
        }
        log(
            f"actor + critic ← {args.init} (iteration {ck.get('iter')}), std reset to {args.init_std}"
        )
    expert = s2_bc.teacher_from_onnx(actor, v5)
    spec = C.BC[args.stage]
    if spec.anchor == "v5":
        anchor = expert
        info["anchor"] = "v5"
    elif args.anchor_onnx:
        anchor = s2_bc.teacher_from_onnx(actor, args.anchor_onnx)
        info["anchor"] = {"onnx": args.anchor_onnx, "sha256": rl_onnx.sha256(args.anchor_onnx)}
    else:
        anchor = s2_bc.teacher_from_actor(actor)
        info["anchor"] = "the start actor"
    runner.alg.set_teachers(anchor, expert)
    info["bc"] = asdict(spec)
    log(f"teachers: fallen expert v5, walk anchor {info['anchor'] if isinstance(info['anchor'], str) else 'R3 ONNX'} "
        f"(dims {len(spec.anchor_dims)}, coef {spec.anchor_coef})")  # fmt: skip
    return info


def train(args: argparse.Namespace) -> dict:
    import torch

    from duckmind.skilltrain import s2_tasks as T

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    progress = (out / "progress.jsonl").open("a", encoding="utf-8")
    log_file = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{args.stage} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t_start = time.time()
    env, wrapped, runner, built = build(args)
    t_build = time.time() - t_start
    env_cfg = built["env_cfg"]
    costs = set(T.cost_terms(env_cfg))
    rewards = {
        n
        for n, t in env_cfg.rewards.items()
        if n not in costs and t.weight > 0 and n not in T.SIGNED_TERMS
    }
    meta: dict[str, Any] = {
        "format": "duckmind-s2-run-1",
        "stage": args.stage,
        "task_id": T.TASK_IDS[args.stage],
        "started": _now(),
        "args": vars(args),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and "cuda" in args.device
        else None,
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
        "reward_rev": args.reward_rev,
        "critic_warmup": built["critic_warmup"],
        "build_s": round(t_build, 1),
        "obs_dims": {k: list(v.shape) for k, v in wrapped.get_observations().items()},
        "rewards": {n: t.weight for n, t in env_cfg.rewards.items()},
    }
    log(f"built {args.envs} envs on {args.device} in {t_build:.1f} s; obs {meta['obs_dims']}")
    meta["warm"] = warm(runner, wrapped, args, log)

    if args.random_ep_len and not args.resume:
        # rsl_rl's init_at_random_ep_len: the 20-s episodes end at spread times (the first episode
        # logs arrive within a few iterations instead of ~42, and the resets decorrelate)
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    stop_reason = "iterations"
    history: list[dict] = []
    main_hist: dict[str, list[float]] = {k: [] for k in T.MAIN_TERMS[args.stage]}
    obs = wrapped.get_observations().to(runner.device)
    runner.alg.train_mode()
    start_it = runner.current_learning_iteration + (1 if args.resume else 0)
    total_it = start_it + args.iterations
    it = start_it
    try:
        for it in range(start_it, total_it):
            t0 = time.time()
            rec = Recorder()
            with torch.inference_mode():
                for _ in range(runner.cfg["num_steps_per_env"]):
                    actions = runner.alg.act(obs)
                    obs, rew, dones, extras = wrapped.step(actions.to(wrapped.device))
                    if not (torch.isfinite(rew).all() and torch.isfinite(obs["actor"]).all()):
                        raise FloatingPointError("non-finite reward or observation from the env")
                    obs, rew, dones = (
                        obs.to(runner.device),
                        rew.to(runner.device),
                        dones.to(runner.device),
                    )
                    runner.alg.process_env_step(obs, rew, dones, extras)
                    resets = int(dones.sum().item())
                    if "log" in extras:
                        rec.add_log(extras["log"], resets)
                t_collect = time.time() - t0
                runner.alg.compute_returns(obs)
            losses = runner.alg.update()
            runner.current_learning_iteration = it
            dt_it = time.time() - t0
            means = rec.means()
            std = float(runner.alg.get_policy().output_std.mean().item())
            row = {
                "it": it, "s": round(dt_it, 3), "collect_s": round(t_collect, 3), "lr": runner.alg.learning_rate,
                "std": round(std, 4), "warmup": int(runner.alg.warmup_left > 0 or "critic_warmup" in losses),
                "losses": {k: round(float(v), 5) for k, v in losses.items()},
                **{k: round(v, 5) for k, v in means.items()},
            }  # fmt: skip
            for term in main_hist:
                key = f"Episode_Reward/{term}"
                if key in means:
                    main_hist[term].append(means[key])
            problems = sign_problems(means, costs, rewards)
            if problems:
                row["sign_problems"] = problems
            progress.write(json.dumps(_jsonable(row)) + "\n")
            progress.flush()
            history.append(row)
            done_its = it - start_it + 1
            if it % args.print_every == 0 or done_its == args.iterations:
                eta = (total_it - it - 1) * dt_it / 60
                mains = " ".join(
                    f"{k} {row.get('Episode_Reward/' + k, float('nan')):.3f}" for k in main_hist
                )
                fall = row.get("Episode_Metrics/fallen45", float("nan"))
                lean = row.get("Episode_Metrics/lean_deg", float("nan"))
                cam = row.get("Episode_Metrics/cam_perp_dps", float("nan"))
                log(f"it {it}/{total_it - 1} {dt_it:.2f} s/it | {mains} | lean {lean:.2f}° cam {cam:.1f}°/s | fallen "
                    f"{fall:.3f} | std {std:.3f} lr {runner.alg.learning_rate:.1e} bc {losses.get('bc_loss', 0):.4f} | "
                    f"eta {eta:.0f} min")  # fmt: skip
            if problems:
                stop_reason = "sign: " + "; ".join(problems)
                log("STOP " + stop_reason)
                break
            if (it + 1) % args.save_every == 0:
                save_checkpoint(runner, env, out, it)
            if args.kill_flat and (it - start_it + 1) in FLAT_CHECKS:
                flat = flat_terms(main_hist)
                if flat:
                    stop_reason = f"flat: {flat} over the last 500 iterations (plan §5 kill rule)"
                    log("STOP " + stop_reason)
                    break
            elapsed = (time.time() - t_start) / 60
            if (
                args.time_budget_min
                and elapsed + dt_it / 60 + args.reserve_min > args.time_budget_min
            ):
                stop_reason = f"time budget {args.time_budget_min} min"
                log("STOP " + stop_reason)
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as exc:  # noqa: BLE001
        stop_reason = f"error: {type(exc).__name__}: {exc}"
        log("STOP " + stop_reason)
        meta["error"] = stop_reason
    final_ckpt = save_checkpoint(runner, env, out, it, final=True)
    summary: dict[str, Any] = {
        **meta,
        "stopped": _now(),
        "stop_reason": stop_reason,
        "iterations_done": it - start_it + 1,
        "last_iteration": it,
        "env_steps": int(env.common_step_counter),
        "seconds_per_iteration": _s_per_it(history),
        "wall_min": round((time.time() - t_start) / 60, 2),
        "a100_units_estimate": round((time.time() - t_start) / 3600 * C.UNITS_PER_HOUR, 2)
        if "cuda" in args.device
        else None,
        "final_values": _tail_means(history),
        "checkpoint": str(final_ckpt),
    }
    if args.export and not stop_reason.startswith("error"):
        summary["export"] = export(runner, env, wrapped, out, args, log, summary)
    (out / "metrics.json").write_text(
        json.dumps(_jsonable(summary), indent=1, ensure_ascii=False) + "\n"
    )
    log(f"done: {stop_reason}; metrics → {out / 'metrics.json'}")
    progress.close()
    log_file.close()
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return summary


def walk_manifest(stage: str, onnx_path: Path, iteration: int, run: dict[str, Any]) -> dict:
    """A schema-2 single-policy manifest for the walk slot (upstream docs/policy-manifest.md), with
    the Stage-2 semantics in its ``duckmind`` block."""
    from duckmind.skilltrain import rl_onnx

    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": C.OBS_LEN,
        "action_len": C.ACTION_LEN,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": C.CONTROL_HZ},
        "name": onnx_path.stem,
        "kind": "perpetual",
        "slot": "walk",
        "entry_pose": "standing",
        "description": "duckmind Stage-2 walk network: v5 velstand fine-tuned to lean / sway / crouch on robot.pose"
        + (" and to keep a steady gaze while walking" if stage == "gaze" else "")
        + " (simulator only until phase 8)",
        "command": {
            "twist": "vx, vy, vyaw as v5 (robotd smooths at cmd_alpha 0.2; zero while pose mode is held)",
            "head": "neck_pitch, head_pitch, head_yaw, head_roll offsets from HOME"
            + ("; walking: a gaze offset in the slow heading frame" if stage == "gaze" else ""),
            "body": "robot.pose z (m), roll, pitch (rad) at zero twist, envelope z -0.025..+0.008, |roll|,|pitch| <= 0.2",
        },
        "training": {
            "repo": "duckmind (skilltrain s2)",
            "task_id": C.TASK_IDS[stage],
            "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
            "checkpoint": iteration,
            "sha256": rl_onnx.sha256(onnx_path),
            "exported": _now(),
            "warm_start": "v5 velstand 1c659be5" if stage == "pose" else "Stage-2 pose network",
        },
        "eval": {},
        "duckmind": {
            "stage": C.STAGE,
            "net": stage,
            "pose_capable": True,
            "gaze_stable": stage == "gaze",
            "pose_envelope": {
                "z": list(C.POSE.z),
                "angle": C.POSE.angle,
                "rate_rad_s": C.POSE.rate_rad_s,
                "rate_m_s": C.POSE.rate_m_s,
                "ramp_to_zero_before_release": True,
            },  # fmt: skip
            "robotd": {
                "walk_action_scale": C.WALK_ACTION_SCALE,
                "gain_ratio": C.GAIN_RATIO,
                "stand": "none",
                "head_alpha": C.HEAD_ALPHA_DEMO,
                "lowpass": [0.5, 0.7],
            },  # fmt: skip
            "servo_delay_trained_ms": [C.DELAY_STEPS[0] * 5, C.DELAY_STEPS[1] * 5],
            "reward_rev": run.get("reward_rev"),
            "iterations": run.get("iterations_done"),
        },
    }


def export(
    runner: Any, env: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any, run: dict
) -> dict:
    from duckmind.skilltrain import rl_action, rl_onnx, s2_bc

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy()
    actor.eval()
    action_cfg = env.cfg.actions["joint_pos"]
    metadata = {
        "duckmind_walk_net": args.stage,
        "duckmind_stage": C.STAGE,
        "robotd": rl_action.robotd_numbers(action_cfg, C.GAIN_RATIO),
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
        "iteration": runner.current_learning_iteration,
        "joint_names": list(C.JOINTS),
        "default_joint_pos": list(C.HOME),
        "observation_names": [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "head_command",
            "body_command",
        ],  # fmt: skip
        "slot": "walk",
    }
    onnx_path = rl_onnx.export_actor(actor, final / C.WALK_FILE, metadata)
    dev = next(actor.parameters()).device
    import numpy as np

    obs = wrapped.get_observations()["actor"][:256].detach().cpu().numpy()
    rows = np.concatenate([obs, s2_bc.pose_observations(onnx_path, n=256)])
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=512, obs=rows
    )
    problems = rl_onnx.robotd_problems(onnx_path)
    (final / "parity.json").write_text(
        json.dumps({"parity": par, "robotd_problems": problems}, indent=1) + "\n"
    )
    manifest = walk_manifest(args.stage, onnx_path, runner.current_learning_iteration, run)
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    log(f"exported {onnx_path.name} ({onnx_path.stat().st_size / 1e3:.0f} kB): parity max {par['max_abs']:.2e}, "
        f"robotd problems {problems or 'none'}")  # fmt: skip
    return {"onnx": str(onnx_path), "sha256": rl_onnx.sha256(onnx_path), "parity": par, "robotd_problems": problems,
            "manifest": str(final / "manifest.json")}  # fmt: skip


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fine-tune the walk network for Stage 2 (pose or gaze)"
    )
    ap.add_argument("--stage", choices=C.STAGES, required=True)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=0, help="0 = the stage's default")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--v5", required=True, help="folder with the v5 velstand.onnx (and alpha_sitstand.onnx)"
    )
    ap.add_argument("--init", default="v5", help="v5, or a Stage-2 checkpoint (R4 from R3)")
    ap.add_argument(
        "--anchor-onnx", default=None, help="R4: the start network's ONNX for the leg anchor"
    )
    ap.add_argument("--resume", default=None, help="a checkpoint of this stage's run (continue it)")
    ap.add_argument("--init-std", type=float, default=C.INIT_STD)
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument(
        "--critic-warmup", type=int, default=-1, help="-1 = 100 from v5, 30 from a checkpoint"
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--reserve-min", type=float, default=3.0)
    ap.add_argument("--kill-flat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--backlash", action="store_true", help="the backlash twin (a final run)")
    ap.add_argument("--no-pushes", action="store_true")
    ap.add_argument("--curriculum-scale", type=float, default=1.0)
    ap.add_argument("--desired-kl", type=float, default=0.0, help="0 = the recipe's 0.01")
    ap.add_argument(
        "--topple-scale", type=float, default=1.0, help="topple-push rate × this (0 = none)"
    )
    ap.add_argument(
        "--prone-scale", type=float, default=1.0, help="prone / crouch spawn shares × this"
    )
    ap.add_argument("--random-ep-len", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--reward-rev", default="p1", help="the Stage-2 reward revision (s2_tasks.REWARD_REVS)"
    )
    args = ap.parse_args(argv)
    if not args.iterations:
        args.iterations = C.DEFAULT_ITERATIONS[args.stage]
    if args.device.startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    summary = train(args)
    ok = not str(summary.get("stop_reason", "")).startswith(("error", "sign"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
