"""Train one B-skill network (b:feasibility, 2026-09-28): PPO on the official physics → robotd's ONNX.

    python -m duckmind.skilltrain.bsk_train --net bskstep --envs 4096 --iterations 1200 \
        --device cuda:0 --library /content/out/library/v5_entries.npz \
        --policies /content/bsk/policies --out /content/out/bskstep-s1 --time-budget-min 70

Stage 1's trainer (``rl_train``) re-used piece by piece: the v5 warm start (``rl_train.warm_start``:
the velstand ONNX into the actor, frozen normaliser, parity ≤ 1e-4 on this env's observations, a fresh
critic), the per-iteration recorder and the sign check (every cost logs ≤ 0, every reward ≥ 0; the mat
progress potential is signed by design and exempt), checkpoints, the time budget. Writes into ``--out``:
``progress.jsonl``, ``train.log``, ``checkpoints/``, ``final/<net>.onnx`` + ``parity.json`` +
``robotd_skills.toml``, ``eval.json`` (a deterministic rollout of the final actor: the task metrics)
and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain.rl_train import (
    Recorder,
    _jsonable,
    _now,
    _s_per_it,
    _tail_means,
    flat_terms,
    save_checkpoint,
    sign_problems,
    warm_start,
)

UNITS_PER_HOUR = 6.77


def build(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401  (upstream's NaN-safe patches ride on this import)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from duckmind.skilltrain import bsk_tasks as T

    torch.manual_seed(args.seed)
    delay = parse_delay(args.delay)
    delay_steps = B.SERVO_DELAY_STEPS if delay == "train" else (delay, delay)
    env_cfg = T.make_env_cfg(args.net, library_path=args.library, pushes=not args.no_pushes,
                             curriculum_scale=args.curriculum_scale, max_init_level=args.max_init_level,
                             delay_steps=delay_steps, reward_rev=args.reward_rev)  # fmt: skip
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent = T.runner_cfg(args.net, iterations=args.iterations, init_std=args.init_std, seed=args.seed,
                         learning_rate=args.lr)  # fmt: skip
    agent_dict = asdict(agent)
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, log_dir=None, device=args.device)
    return (
        env,
        wrapped,
        runner,
        {
            "env_cfg": env_cfg,
            "agent": agent_dict,
            "servo_delay_steps": list(delay_steps),
            "delay_mode": args.delay,
        },
    )


def parse_delay(value: str) -> str | int:
    """Parse the B-skill delay CLI: randomized training or a fixed 0–8 physics-step lag."""
    text = str(value).strip().lower()
    if text == "train":
        return text
    if text.endswith("steps"):
        text = text[:-5].strip()
    parsed = H.parse_delay(text)
    if not isinstance(parsed, int) or not 0 <= parsed <= 8:
        raise ValueError("B-skill delay must be 'train' or a fixed 0–8 steps (0–40 ms)")
    return parsed


def train(args: argparse.Namespace) -> dict:
    import torch

    from duckmind.skilltrain import bsk_tasks as T

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    progress = (out / "progress.jsonl").open("a", encoding="utf-8")
    log_file = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{args.net} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t_start = time.time()
    env, wrapped, runner, built = build(args)
    env_cfg = built["env_cfg"]
    costs = set(T.cost_terms(env_cfg))
    rewards = {
        n
        for n, t in env_cfg.rewards.items()
        if n not in costs and t.weight > 0 and n not in T.SIGNED_TERMS
    }
    meta: dict[str, Any] = {
        "format": "duckmind-bsk-run-1", "net": args.net, "task_id": B.TASK_IDS[args.net], "started": _now(),
        "args": vars(args), "host": platform.node(), "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in args.device else None,
        "entries": [e.name for e in B.entries(args.net)], "pinned": {"action_scale": B.ACTION_SCALE, "gain_ratio": B.GAIN_RATIO},
        "reward_rev": args.reward_rev, "delay_mode": built["delay_mode"],
        "servo_delay_steps": built["servo_delay_steps"],
        "build_s": round(time.time() - t_start, 1),
        "obs_dims": {k: list(v.shape) for k, v in wrapped.get_observations().items()},
    }  # fmt: skip
    log(f"built {args.envs} envs on {args.device} in {meta['build_s']} s; obs {meta['obs_dims']}")
    if args.warm_start and args.warm_start != "none":
        meta["warm_start"] = warm_start(runner, wrapped, args.warm_start, args.init_std, log)
    if args.resume:
        runner.load(args.resume, map_location=args.device)
        if args.warm_start and args.warm_start != "none":
            from duckmind.skilltrain.rl_onnx import freeze_normalizer

            freeze_normalizer(runner.alg.actor)
        meta["resumed_from"] = {"path": args.resume, "iteration": runner.current_learning_iteration}
        log(f"resumed {args.resume} at iteration {runner.current_learning_iteration}")

    stop_reason = "iterations"
    history: list[dict] = []
    main_hist: dict[str, list[float]] = {k: [] for k in T.MAIN_TERMS[args.net]}
    obs = wrapped.get_observations().to(runner.device)
    runner.alg.train_mode()
    start_it = runner.current_learning_iteration + (1 if args.resume else 0)
    total_it = start_it + args.iterations
    it = start_it
    last_s = 0.0
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
                    if "log" in extras:
                        rec.add_log(extras["log"], int(dones.sum().item()))
                t_collect = time.time() - t0
                runner.alg.compute_returns(obs)
            losses = runner.alg.update()
            runner.current_learning_iteration = it
            last_s = time.time() - t0
            means = rec.means()
            std = float(runner.alg.get_policy().output_std.mean().item())
            row = {"it": it, "s": round(last_s, 3), "collect_s": round(t_collect, 3), "lr": runner.alg.learning_rate,
                   "std": round(std, 4), "losses": {k: round(float(v), 5) for k, v in losses.items()},
                   **{k: round(v, 5) for k, v in means.items()}}  # fmt: skip
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
            done = it - start_it + 1
            if it % args.print_every == 0 or done == args.iterations:
                mains = " ".join(
                    f"{k} {row.get('Episode_Reward/' + k, float('nan')):.3f}" for k in main_hist
                )
                extra = " ".join(f"{k.split('/')[-1]} {v:.3g}" for k, v in row.items()
                                 if k.startswith("Episode_Metrics/") and not k.endswith("fallen45"))  # fmt: skip
                log(
                    f"it {it}/{total_it - 1} {last_s:.2f} s/it | {mains} | {extra} | std {std:.3f} lr {runner.alg.learning_rate:.1e}"
                )
            if problems:
                stop_reason = "sign: " + "; ".join(problems)
                log("STOP " + stop_reason)
                break
            if (it + 1) % args.save_every == 0:
                save_checkpoint(runner, env, out, it)
            if args.kill_flat and done in (500, 1000):
                flat = flat_terms(main_hist)
                if flat:
                    stop_reason = f"flat: {flat}"
                    log("STOP " + stop_reason)
                    break
            if (
                args.time_budget_min
                and (time.time() - t_start) / 60 + last_s / 60 + args.reserve_min
                > args.time_budget_min
            ):
                stop_reason = f"time budget {args.time_budget_min} min"
                log("STOP " + stop_reason)
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as exc:  # noqa: BLE001
        stop_reason = f"error: {type(exc).__name__}: {exc}"
        log("STOP " + stop_reason)
    final_ckpt = save_checkpoint(runner, env, out, it, final=True)
    summary: dict[str, Any] = {
        **meta, "stopped": _now(), "stop_reason": stop_reason, "iterations_done": it - start_it + 1,
        "last_iteration": it, "env_steps": int(env.common_step_counter), "seconds_per_iteration": _s_per_it(history),
        "wall_min": round((time.time() - t_start) / 60, 2),
        "a100_units_estimate": round((time.time() - t_start) / 3600 * UNITS_PER_HOUR, 2) if "cuda" in args.device else None,
        "final_values": _tail_means(history), "checkpoint": str(final_ckpt),
        "learning": learning_trend(history),
    }  # fmt: skip
    if not stop_reason.startswith("error"):
        if args.export:
            summary["export"] = export(runner, wrapped, out, args, log)
        if args.eval_iters > 0:
            summary["eval"] = evaluate(runner, wrapped, out, args, log)
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


def learning_trend(history: list[dict], n: int = 10) -> dict[str, Any]:
    """First vs last ``n`` iterations of every Episode_Reward / Episode_Metrics value."""
    if not history:
        return {}
    keys = sorted(
        {k for r in history for k in r if k.startswith(("Episode_Reward/", "Episode_Metrics/"))}
    )

    def mean(part: list[dict], k: str) -> float | None:
        v = [r[k] for r in part if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
        return round(sum(v) / len(v), 5) if v else None

    return {k: {"first": mean(history[1 : n + 1], k), "last": mean(history[-n:], k)} for k in keys}


def export(runner: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any) -> dict:
    from duckmind.skilltrain import rl_onnx
    from duckmind.skilltrain import table as TB

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy()
    actor.eval()
    delay_meta = _delay_meta(getattr(wrapped, "unwrapped", wrapped))
    meta = {"duckmind_skill_net": args.net, "duckmind_stage": B.STAGE, "skills": [e.name for e in B.entries(args.net)],
            "reward_rev": args.reward_rev,
            "iteration": runner.current_learning_iteration,
            "command_encoding": {e.name: list(e.command) for e in B.entries(args.net)},
            "action_scale": B.ACTION_SCALE, "gain_ratio": B.GAIN_RATIO, "trained_with_lowpass": [0.5, 0.7],
            "servo_delay_ms": [5 * delay_meta["delay_min_lag"], 5 * delay_meta["delay_max_lag"]],
            "servo_delay_mode_trained": delay_meta}  # fmt: skip
    onnx_path = rl_onnx.export_actor(actor, final / f"{args.net}.onnx", meta)
    dev = next(actor.parameters()).device
    obs = wrapped.get_observations()["actor"][:256].detach().cpu().numpy()
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=512, obs=obs
    )
    problems = rl_onnx.robotd_problems(onnx_path)
    (final / "parity.json").write_text(
        json.dumps({"parity": par, "robotd_problems": problems}, indent=1) + "\n"
    )
    defs = [
        TB.SkillDef(**d)
        for d in B.skill_defs({args.net: str(onnx_path.resolve())}, nets=(args.net,))
    ]
    (final / "robotd_skills.toml").write_text(TB.toml_section(defs), encoding="utf-8")
    log(
        f"exported {onnx_path.name}: parity {par['max_abs']:.2e}, robotd problems {problems or 'none'}"
    )
    return {
        "onnx": str(onnx_path),
        "sha256": rl_onnx.sha256(onnx_path),
        "parity": par,
        "robotd_problems": problems,
    }


def _delay_meta(env: Any) -> dict[str, Any]:
    """Return the actual configured actuator lag for run and ONNX metadata."""
    for actuator in env.cfg.scene.entities["robot"].articulation.actuators:
        if hasattr(actuator, "delay_max_lag"):
            return {
                k: getattr(actuator, k)
                for k in (
                    "delay_min_lag",
                    "delay_max_lag",
                    "delay_update_period",
                    "delay_hold_prob",
                )
            }
    raise RuntimeError("configured robot actuator has no delay metadata")


def evaluate(runner: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any) -> dict:
    """The final actor's mean action (no exploration noise) for ``--eval-iters`` × 24 steps on the
    training env: the task metrics as the episodes end (the env's own DR and curriculum state)."""
    import torch

    policy = runner.alg.get_policy()
    policy.eval()
    obs = wrapped.get_observations().to(runner.device)
    rec = Recorder()
    with torch.inference_mode():
        for _ in range(args.eval_iters * 24):
            act = (
                policy.act_inference(obs)
                if hasattr(policy, "act_inference")
                else runner.alg.act(obs)
            )
            obs, _, dones, extras = wrapped.step(act.to(wrapped.device))
            obs = obs.to(runner.device)
            if "log" in extras:
                rec.add_log(extras["log"], int(dones.sum().item()))
    res = {
        k: round(v, 5)
        for k, v in rec.means().items()
        if k.startswith(("Episode_Metrics/", "Curriculum/"))
    }
    (out / "eval.json").write_text(json.dumps(res, indent=1) + "\n")
    log("eval (mean action): " + ", ".join(f"{k.split('/')[-1]} {v:.3g}" for k, v in res.items()))
    return res


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bsk_train")
    ap.add_argument("--net", choices=B.NETS, required=True)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--library", default=None)
    ap.add_argument(
        "--policies", required=True, help="folder with the v5 velstand.onnx (the warm start)"
    )
    ap.add_argument(
        "--warm-start", default="auto", help="auto = <policies>/velstand.onnx; none; or a path"
    )
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--reserve-min", type=float, default=2.0)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--curriculum-scale", type=float, default=1.0)
    ap.add_argument("--max-init-level", type=int, default=2)
    ap.add_argument("--reward-rev", choices=("b3", "b4"), default="b3")
    ap.add_argument("--delay", default="train", help="train or fixed 0–8 steps / 0–40 ms")
    ap.add_argument("--no-pushes", action="store_true")
    ap.add_argument("--kill-flat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-iters", type=int, default=8)
    args = ap.parse_args(argv)
    args.delay = parse_delay(args.delay)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.warm_start == "auto":
        args.warm_start = str(Path(args.policies) / "velstand.onnx")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
