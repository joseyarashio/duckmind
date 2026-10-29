"""Fine-tune the v5 walk network for the panic gait (2026-10-08): PPO on the official physics → robotd's ONNX.

    python -m duckmind.skilltrain.pg_train --envs 4096 --iterations 1500 --device cuda:0 \
        --v5 POLICIES/releases/seed-v5 --out runs/pg-a --init v5 --reward-rev g1

``--init``: ``v5`` (the v5 ONNX imported, vx slot re-conditioned, a fresh critic, a 100-iteration critic
warm-up, v5 as walk anchor and fallen expert) or a checkpoint of an earlier panic run (actor and critic
loaded, warm-up 30, ``--curriculum-offset`` iterations of the command curriculum already done).

Writes into ``--out``: ``progress.jsonl`` (one line per iteration), ``checkpoints/model_<it>.pt`` and — every
``--save-every`` — ``checkpoints/panic_<it>.onnx`` (robotd's format, normaliser baked; the quick gate reads these
while training continues), ``final/velstand_panic.onnx`` + parity + manifest, ``metrics.json``. Stop rules: a
reward-sign problem or non-finite state stops at once; the time budget saves, exports and exits; a STOP file
stops at the next iteration; ``OVERRIDE.json`` re-weights overridable terms mid-run.
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

from duckmind.skilltrain import pg_contract as C
from duckmind.skilltrain.rl_train import (
    Recorder,
    _jsonable,
    _now,
    _s_per_it,
    _tail_means,
    save_checkpoint,
    sign_problems,
)


def _parse_stages(text: str | None) -> tuple[tuple[int, float], ...] | None:
    """'0:0.7,150:0.85,350:1.0' → ((0, 0.7), (150, 0.85), (350, 1.0))."""
    if not text:
        return None
    return tuple((int(a), float(b)) for a, b in (x.split(":") for x in text.split(",")))


def build(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from duckmind.skilltrain import pg_tasks as T

    torch.manual_seed(args.seed)
    env_cfg = T.make_env_cfg(reward_rev=args.reward_rev, topple_scale=args.topple_scale,
                             prone_scale=args.prone_scale, vmax_scale=args.vmax_scale, p_fast=args.p_fast, p_xml=args.p_xml,
                             action_rate=args.action_rate, vmax_stages=_parse_stages(args.vmax_stages))  # fmt: skip
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    warmup = args.critic_warmup
    if warmup < 0:
        warmup = C.CRITIC_WARMUP_ITERS if args.init == "v5" else (0 if args.resume else 30)
    agent = T.runner_cfg(iterations=args.iterations, init_std=args.init_std, seed=args.seed,
                         critic_warmup=warmup, learning_rate=args.lr, anchor_coef=args.anchor_coef, entropy_coef=args.entropy)  # fmt: skip
    if args.desired_kl:
        agent.algorithm.desired_kl = float(args.desired_kl)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), log_dir=None, device=args.device)
    return env, wrapped, runner, {"env_cfg": env_cfg, "critic_warmup": warmup}


def warm(runner: Any, wrapped: Any, args: argparse.Namespace, log: Any) -> dict:
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
        info["import"] = s2_bc.import_v5(
            actor, v5, init_std=args.init_std, recondition_std=C.RECONDITION_STD, obs=obs
        )
        log(f"actor ← v5 velstand ({got[:12]}), vx slot re-conditioned {info['import']['recondition']}, "
            f"parity {info['import']['parity']['max_abs']:.2e}, init std {args.init_std}")  # fmt: skip
    else:
        ck = torch.load(args.init, map_location=args.device, weights_only=False)
        actor.load_state_dict(ck["actor_state_dict"], strict=True)
        if not args.actor_only:
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
    teacher = s2_bc.teacher_from_onnx(actor, v5)
    runner.alg.set_teachers(teacher, teacher)
    runner.alg.bc = {
        **runner.alg.bc,
        "anchor_coef": [list(x) for x in runner.alg.bc.get("anchor_coef", [[0, 1.0]])],
    }
    info["bc"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in C.BC.items()}
    log(
        f"teachers: v5 as walk anchor (upright, vx <= {C.ANCHOR_MAX_VX}) and fallen expert (tilt > 35°)"
    )
    return info


def checkpoint_onnx(runner: Any, env: Any, out: Path, it: int, log: Any) -> Path | None:
    from duckmind.skilltrain import rl_action, rl_onnx

    try:
        metadata = {
            "duckmind_walk_net": C.NET,
            "duckmind_stage": C.STAGE,
            "robotd": rl_action.robotd_numbers(env.cfg.actions["joint_pos"], 1.0),
            "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
            "iteration": int(it),
            "checkpoint": True,
            "slot": "walk",
        }
        path = rl_onnx.export_actor(
            runner.alg.get_policy(), out / "checkpoints" / f"panic_{it}.onnx", metadata
        )
        problems = rl_onnx.robotd_problems(path)
        log(f"checkpoint {it} → {path.name}; robotd problems {problems or 'none'}")
        return path
    except Exception as exc:  # noqa: BLE001
        log(f"checkpoint {it} export failed: {type(exc).__name__}: {exc}")
        return None


def poll_override(path: Path, state: dict) -> dict | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if state.get("mtime") == mtime:
        return None
    state["mtime"] = mtime
    try:
        spec = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"_error": f"unreadable override: {exc}"}
    return spec if isinstance(spec, dict) else {"_error": "the override is not a JSON object"}


def apply_override(env: Any, spec: dict, costs: set[str], log: Any) -> dict:
    """Re-weight overridable reward terms / params mid-run (signs kept); rejects, never raises."""
    out: dict[str, Any] = {"id": spec.get("id"), "applied": {}, "rejected": {}}
    if "_error" in spec:
        out["rejected"]["file"] = spec["_error"]
        log(f"override rejected: {spec['_error']}")
        return out
    rm = getattr(env, "unwrapped", env).reward_manager
    names = set(rm.active_terms)
    for name, w in (spec.get("weights") or {}).items():
        if name not in C.OVERRIDABLE or name not in names:
            out["rejected"][name] = "not an overridable term"
            continue
        try:
            w = float(w)
        except (TypeError, ValueError):
            out["rejected"][name] = f"weight {w!r} is not a number"
            continue
        if (name in costs and w > 0) or (name not in costs and w < 0):
            out["rejected"][name] = f"sign: weight {w}"
            continue
        cfg = rm.get_term_cfg(name)
        out["applied"][name] = {"weight": [cfg.weight, w]}
        cfg.weight = w
    for name, params in (spec.get("params") or {}).items():
        allowed = C.OVERRIDABLE.get(name, ())
        if name not in names or not isinstance(params, dict):
            out["rejected"][f"{name}.params"] = "not an overridable term"
            continue
        cfg = rm.get_term_cfg(name)
        for key, value in params.items():
            if key not in allowed:
                out["rejected"][f"{name}.{key}"] = "not an overridable parameter"
                continue
            out["applied"].setdefault(name, {})[key] = [cfg.params.get(key), float(value)]
            cfg.params[key] = float(value)
    log(f"override {out['id']}: applied {out['applied']}; rejected {out['rejected'] or 'none'}")
    return out


def train(args: argparse.Namespace) -> dict:
    import torch

    from duckmind.skilltrain import pg_tasks as T

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    progress = (out / "progress.jsonl").open("a", encoding="utf-8")
    log_file = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[panic {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    if args.target_k0 is not None:
        C.TARGET_OVERRIDE["k0"] = float(args.target_k0)
    if args.target_top is not None:
        C.TARGET_OVERRIDE["top"] = float(args.target_top)
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
        "format": "duckmind-panic-run-1", "stage": C.STAGE, "task_id": C.TASK_ID, "started": _now(),
        "args": vars(args), "host": platform.node(), "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in args.device else None,
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT, "reward_rev": args.reward_rev,
        "critic_warmup": built["critic_warmup"], "build_s": round(t_build, 1),
        "obs_dims": {k: list(v.shape) for k, v in wrapped.get_observations().items()},
        "rewards": {n: t.weight for n, t in env_cfg.rewards.items()},
    }  # fmt: skip
    log(f"built {args.envs} envs on {args.device} in {t_build:.1f} s; obs {meta['obs_dims']}")
    meta["warm"] = warm(runner, wrapped, args, log)
    if args.curriculum_offset:
        env.common_step_counter = int(args.curriculum_offset) * C.STEPS_PER_ENV
        log(f"command/reward curricula start at iteration {args.curriculum_offset}")
    if args.random_ep_len and not args.resume:
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    stop_reason = "iterations"
    history: list[dict] = []
    obs = wrapped.get_observations().to(runner.device)
    runner.alg.train_mode()
    override_path = Path(args.override_file) if args.override_file else out / "OVERRIDE.json"
    override_state: dict[str, Any] = {"mtime": None}
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
            row = {"it": it, "s": round(dt_it, 3), "collect_s": round(t_collect, 3), "lr": runner.alg.learning_rate,
                   "std": round(std, 4), "warmup": int(runner.alg.warmup_left > 0 or "critic_warmup" in losses),
                   "losses": {k: round(float(v), 5) for k, v in losses.items()},
                   **{k: round(v, 5) for k, v in means.items()}}  # fmt: skip
            problems = sign_problems(means, costs, rewards)
            if problems:
                row["sign_problems"] = problems
            spec = poll_override(override_path, override_state)
            if spec is not None:
                applied = apply_override(env, spec, costs, log)
                row["override"] = applied
                with (out / "overrides.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_jsonable({"it": it, "at": _now(), **applied})) + "\n")
            progress.write(json.dumps(_jsonable(row)) + "\n")
            progress.flush()
            history.append(row)
            if it % args.print_every == 0 or it == total_it - 1:
                m = lambda k: row.get("Episode_Metrics/" + k, 0.0)  # noqa: E731
                fv = m("fast_vx") / m("fast_cmd") if m("fast_cmd") > 1e-6 else float("nan")
                ov = m("ord_vx") / m("ord_cmd") if m("ord_cmd") > 1e-6 else float("nan")
                ay = m("arc_yaw") / m("arc_cmd") if m("arc_cmd") > 1e-6 else float("nan")
                log(f"it {it}/{total_it - 1} {dt_it:.2f} s/it | fast v/cmd {fv:.2f} ord {ov:.2f} arc-yaw {ay:.2f} | "
                    f"fell60 {m('fell60'):.4f} | tr_sharp {row.get('Episode_Reward/track_vel_sharp', 0):.2f} "
                    f"std {std:.3f} lr {runner.alg.learning_rate:.1e} bc {losses.get('bc_loss', 0):.4f} | "
                    f"eta {(total_it - it - 1) * dt_it / 60:.0f} min")  # fmt: skip
            if problems:
                stop_reason = "sign: " + "; ".join(problems)
                log("STOP " + stop_reason)
                break
            if (it + 1) % args.save_every == 0:
                save_checkpoint(runner, env, out, it)
                if args.ckpt_onnx:
                    checkpoint_onnx(runner, env, out, it, log)
            elapsed = (time.time() - t_start) / 60
            if (
                args.time_budget_min
                and elapsed + dt_it / 60 + args.reserve_min > args.time_budget_min
            ):
                stop_reason = f"time budget {args.time_budget_min} min"
                log("STOP " + stop_reason)
                break
            if (out / "STOP").exists():
                stop_reason = "STOP file"
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
        **meta, "stopped": _now(), "stop_reason": stop_reason, "iterations_done": it - start_it + 1,
        "last_iteration": it, "env_steps": int(env.common_step_counter),
        "seconds_per_iteration": _s_per_it(history), "wall_min": round((time.time() - t_start) / 60, 2),
        "final_values": _tail_means(history), "checkpoint": str(final_ckpt),
    }  # fmt: skip
    if args.export and not stop_reason.startswith("error"):
        t_exp = time.time()
        summary["export"] = export(runner, env, wrapped, out, args, log)
        summary["export_s"] = round(time.time() - t_exp, 2)
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


def export(
    runner: Any, env: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any
) -> dict:
    from duckmind.skilltrain import rl_action, rl_onnx
    from duckmind.skilltrain.s2_train import walk_manifest

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy()
    actor.eval()
    metadata = {
        "duckmind_walk_net": C.NET, "duckmind_stage": C.STAGE,
        "robotd": rl_action.robotd_numbers(env.cfg.actions["joint_pos"], 1.0),
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT, "iteration": runner.current_learning_iteration, "slot": "walk",
        "reward_rev": args.reward_rev,
    }  # fmt: skip
    onnx_path = rl_onnx.export_actor(actor, final / C.WALK_FILE, metadata)
    dev = next(actor.parameters()).device
    obs = wrapped.get_observations()["actor"][:256].detach().cpu().numpy()
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=512, obs=obs
    )
    problems = rl_onnx.robotd_problems(onnx_path)
    (final / "parity.json").write_text(
        json.dumps({"parity": par, "robotd_problems": problems}, indent=1) + "\n"
    )
    manifest = walk_manifest(
        "pose", onnx_path, runner.current_learning_iteration, {"reward_rev": args.reward_rev}
    )
    manifest["description"] = (
        "duckmind panic-gait walk network: v5 velstand fine-tuned to run to vx 1.0 (simulator only)"
    )
    manifest["training"]["task_id"] = C.TASK_ID
    manifest["training"]["warm_start"] = "v5 velstand 1c659be5"
    manifest["duckmind"] = {"stage": C.STAGE, "net": C.NET, "reward_rev": args.reward_rev}
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    log(
        f"exported {onnx_path.name}: parity max {par['max_abs']:.2e}, robotd problems {problems or 'none'}"
    )
    return {
        "onnx": str(onnx_path),
        "sha256": rl_onnx.sha256(onnx_path),
        "parity": par,
        "robotd_problems": problems,
    }


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune the v5 walk network for the panic gait")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v5", required=True, help="folder with the v5 velstand.onnx")
    ap.add_argument("--init", default="v5", help="v5, or a panic checkpoint")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-std", type=float, default=C.INIT_STD)
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--critic-warmup", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--reserve-min", type=float, default=3.0)
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ckpt-onnx", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--desired-kl", type=float, default=0.0)
    ap.add_argument("--topple-scale", type=float, default=0.5)
    ap.add_argument("--prone-scale", type=float, default=0.4)
    ap.add_argument("--vmax-scale", type=float, default=1.0)
    ap.add_argument(
        "--action-rate", type=float, default=None, help="action_rate_l2 weight (default -1.0)"
    )
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument(
        "--target-k0",
        type=float,
        default=None,
        help="target map slope up to the knee (default 0.50)",
    )
    ap.add_argument(
        "--target-top", type=float, default=None, help="target map vx at command 1.0 (default 0.70)"
    )
    ap.add_argument(
        "--actor-only",
        action="store_true",
        help="--init loads the actor only (a fresh critic: g8 adds a critic input)",
    )
    ap.add_argument(
        "--vmax-stages", default=None, help="'it:vmax,...' (default 0:0.7,150:0.85,350:1.0)"
    )
    ap.add_argument(
        "--p-xml",
        type=float,
        default=0.0,
        help="share of episodes on the XML servo model (default 0)",
    )
    ap.add_argument(
        "--p-fast", type=float, default=None, help="share of the panic band (default 0.30)"
    )
    ap.add_argument(
        "--anchor-coef", type=float, default=None, help="v5 walk-anchor weight (default 1.0)"
    )
    ap.add_argument("--curriculum-offset", type=int, default=0)
    ap.add_argument("--random-ep-len", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reward-rev", default=C.DEFAULT_REWARD_REV)
    ap.add_argument("--override-file", default=None)
    args = ap.parse_args(argv)
    if args.device.startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return args


def main(argv: list[str] | None = None) -> int:
    summary = train(parse(argv))
    return 0 if not str(summary.get("stop_reason", "")).startswith(("error", "sign")) else 1


if __name__ == "__main__":
    sys.exit(main())
