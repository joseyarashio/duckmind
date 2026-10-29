"""Train one Stage-1 skill network (tr:tasks): PPO on the official physics → robotd's ONNX.

    python -m duckmind.skilltrain.rl_train --net startle --envs 4096 --iterations 2500 \
        --device cuda:0 --library /content/out/library/v5_entries.npz \
        --policies /content/stage1/policies --out /content/out/startle-a1 --time-budget-min 80

Writes into ``--out`` (all small enough to download):

* ``progress.jsonl`` — one line per iteration: seconds per iteration, mean reward and episode
  length, every ``Episode_Reward/…`` / ``Episode_Metrics/…`` / curriculum value, the PPO
  losses, learning rate, action std, and the stop-rule verdicts;
* ``checkpoints/model_<it>.pt`` every ``--save-every`` iterations (and ``latest.pt``);
* ``final/<net>.onnx`` (robotd's format, normaliser baked), ``final/parity.json`` (torch vs
  onnxruntime ≤ 1e-5), ``final/manifest.json`` (schema-2 set entries with the pinned scale /
  gain), ``final/robotd_skills.toml`` (the table section for these entries),
  ``final/eval.json`` (``--eval-n`` ducks per cell, the plan's gates, when asked);
* ``metrics.json`` — the run's summary (timings, units estimate, stop reason, final values).

Stop rules (training plan §5): an ``Episode_Reward`` of a cost term above 0 or of a reward term
below 0 (a sign bug) stops the run at once; the main task term flat for 500 iterations stops it
at the 500 / 1,000 checks (``--kill-flat``); the time budget saves, exports and exits cleanly.
Warm start (``--warm-start v5.onnx``): the actor from the ONNX (frozen normaliser, ``--init-std``),
a fresh critic; the imported actor must reproduce the ONNX on this env's observations to ≤ 1e-4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from duckmind.skilltrain import rl_contract as C

UNITS_PER_HOUR = 6.77  # Colab A100 (measured 2026-09-25 / 28)
MAIN_TERMS = {"startle": ("composite", "hop_progress"), "comfort": ("composite", "shake_roll")}
FLAT_CHECKS = (500, 1000)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class Recorder:
    """Per-iteration aggregation of the env's ``extras['log']`` (weighted by resets)."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def add_log(self, log: dict, resets: int) -> None:
        if resets <= 0:
            return
        for key, value in log.items():
            try:
                v = float(value.item() if hasattr(value, "item") else value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + v * resets
            self.weights[key] = self.weights.get(key, 0.0) + resets

    def means(self) -> dict[str, float]:
        return {k: self.sums[k] / self.weights[k] for k in self.sums if self.weights.get(k)}


def sign_problems(
    means: dict[str, float], costs: set[str], rewards: set[str], tol: float = 1e-7
) -> list[str]:
    """AGENTS.md's infallible check: every cost logs ≤ 0 and every reward ≥ 0."""
    found = []
    for key, value in means.items():
        if not key.startswith("Episode_Reward/"):
            continue
        name = key.split("/", 1)[1]
        if name in costs and value > tol:
            found.append(
                f"cost {name} logged {value:+.3g} > 0 (a sign bug: the policy is paid for it)"
            )
        if name in rewards and value < -tol:
            found.append(f"reward {name} logged {value:+.3g} < 0")
    return found


def build(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401  (upstream's NaN-safe patches ride on this import)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from duckmind.skilltrain import rl_tasks as T

    torch.manual_seed(args.seed)
    env_cfg = T.make_env_cfg(
        args.net,
        library_path=args.library,
        pushes=not args.no_pushes,
        curriculum_scale=args.curriculum_scale,
    )
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent = T.runner_cfg(
        args.net, iterations=args.iterations, init_std=args.init_std, seed=args.seed
    )
    if args.lr:
        agent.algorithm.learning_rate = args.lr
    agent.algorithm.schedule = args.schedule
    if args.entropy is not None:
        agent.algorithm.entropy_coef = args.entropy
    agent_dict = asdict(agent)
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, log_dir=None, device=args.device)
    return env, wrapped, runner, {"env_cfg": env_cfg, "agent": agent_dict}


def warm_start(runner: Any, wrapped: Any, onnx_path: str, init_std: float, log: Any) -> dict:
    """The actor from a microduck ONNX (frozen normaliser), a fresh critic; parity on this
    env's own observations must be ≤ 1e-4 or the run stops."""
    from duckmind.skilltrain import rl_onnx

    actor = runner.alg.actor
    info = rl_onnx.load_into_actor(actor, onnx_path, init_std=init_std, frozen=True)
    obs = wrapped.get_observations()["actor"].detach().cpu().numpy()
    par = rl_onnx.parity(onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(next(actor.parameters()).device)).cpu(),
                         samples=256, obs=obs)  # fmt: skip
    info["parity"] = par
    log(f"[warm] actor ← {Path(onnx_path).name} ({info['sha256'][:12]}), parity max {par['max_abs']:.2e} "
        f"over {par['rows']} obs (env rows {len(obs)}), init std {init_std}")  # fmt: skip
    if par["max_abs"] > 1e-4:
        raise RuntimeError(
            f"warm-start parity {par['max_abs']:.2e} > 1e-4: refusing to train on a wrong actor"
        )
    # the runner's optimizer holds the actor params by reference: its moments are fresh anyway
    return info


def train(args: argparse.Namespace) -> dict:
    import torch

    from duckmind.skilltrain import rl_tasks as T

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
    t_build = time.time() - t_start
    env_cfg = built["env_cfg"]
    costs = set(T.cost_terms(env_cfg))
    rewards = {
        n
        for n, t in env_cfg.rewards.items()
        if n not in costs and t.weight > 0 and n not in T.SIGNED_TERMS
    }
    meta: dict[str, Any] = {
        "format": "duckmind-skill-run-1",
        "net": args.net,
        "task_id": T.TASK_IDS[args.net],
        "started": _now(),
        "args": vars(args),
        "host": platform.node(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and "cuda" in args.device
        else None,
        "variants": [v.name for v in C.variants(args.net)],
        "pinned": dict(zip(("action_scale", "gain_ratio"), C.pinned(args.net), strict=True)),
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
        "reward_rev": T.REWARD_REV,
        "curriculum_scale": args.curriculum_scale,
        "build_s": round(t_build, 1),
        "obs_dims": {k: list(v.shape) for k, v in wrapped.get_observations().items()},
    }
    log(f"built {args.envs} envs on {args.device} in {t_build:.1f} s; obs {meta['obs_dims']}")
    if args.warm_start and args.warm_start != "none":
        meta["warm_start"] = warm_start(runner, wrapped, args.warm_start, args.init_std, log)
    if args.resume:
        infos = runner.load(args.resume, map_location=args.device)
        if args.warm_start and args.warm_start != "none":
            from duckmind.skilltrain.rl_onnx import freeze_normalizer

            freeze_normalizer(runner.alg.actor)  # ``until`` is not in the checkpoint
        meta["resumed_from"] = {"path": args.resume, "iteration": runner.current_learning_iteration,
                                "env_steps": env.common_step_counter, "infos": str(infos)[:200]}  # fmt: skip
        log(f"resumed {args.resume} at iteration {runner.current_learning_iteration}")

    stop_reason = "iterations"
    history: list[dict] = []
    main_hist: dict[str, list[float]] = {k: [] for k in MAIN_TERMS[args.net]}
    obs = wrapped.get_observations().to(runner.device)
    runner.alg.train_mode()
    start_it = runner.current_learning_iteration + (1 if args.resume else 0)
    total_it = start_it + args.iterations
    it = start_it
    last_iter_s = 0.0
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
            last_iter_s = time.time() - t0
            means = rec.means()
            std = float(runner.alg.get_policy().output_std.mean().item())
            row = {
                "it": it, "s": round(last_iter_s, 3), "collect_s": round(t_collect, 3),
                "lr": runner.alg.learning_rate, "std": round(std, 4),
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
                eta = (total_it - it - 1) * last_iter_s / 60
                mains = " ".join(
                    f"{k} {row.get('Episode_Reward/' + k, float('nan')):.3f}"
                    for k in MAIN_TERMS[args.net]
                )
                fall = row.get("Episode_Metrics/fallen45", float("nan"))
                log(f"it {it}/{total_it - 1} ({100 * done_its / args.iterations:.1f}%) {last_iter_s:.2f} s/it | "
                    f"{mains} | fallen {fall:.3f} | std {std:.3f} lr {runner.alg.learning_rate:.1e} | eta {eta:.0f} min")  # fmt: skip
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
            elapsed_min = (time.time() - t_start) / 60
            if (
                args.time_budget_min
                and elapsed_min + last_iter_s / 60 + args.reserve_min > args.time_budget_min
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
        "a100_units_estimate": round((time.time() - t_start) / 3600 * UNITS_PER_HOUR, 2)
        if "cuda" in args.device
        else None,
        "final_values": _tail_means(history),
        "checkpoint": str(final_ckpt),
    }
    if args.export and not stop_reason.startswith("error"):
        summary["export"] = export(runner, env, wrapped, out, args, log)
        if args.eval_n > 0:
            summary["eval"] = run_eval(runner, out, args, log)
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


def _tail_means(history: list[dict], n: int = 20) -> dict[str, float]:
    """Mean of every numeric field over the last ``n`` iterations (one iteration is noisy)."""
    tail = history[-n:]
    keys = {k for row in tail for k, v in row.items() if isinstance(v, (int, float)) and k != "it"}
    out = {}
    for k in sorted(keys):
        vals = [row[k] for row in tail if isinstance(row.get(k), (int, float))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 5)
    out["iterations_averaged"] = len(tail)
    return out


def _s_per_it(history: list[dict]) -> dict[str, float | None]:
    s = [h["s"] for h in history[1:]] or [h["s"] for h in history]
    if not s:
        return {"median": None, "mean": None, "n": 0}
    srt = sorted(s)
    return {"median": srt[len(srt) // 2], "mean": round(sum(s) / len(s), 3), "n": len(s)}


def flat_terms(
    main_hist: dict[str, list[float]], window: int = 100, span: int = 500, gain: float = 0.02
) -> list[str]:
    """Main terms whose last-window mean is not ``gain`` above the mean ``span`` iterations
    earlier (the plan's "main task term flat for 500 iterations")."""
    flat = []
    for name, values in main_hist.items():
        if len(values) < span:
            continue
        recent = sum(values[-window:]) / window
        before = sum(values[-span : -span + window]) / window
        if recent <= before + gain * max(abs(before), 1e-3):
            flat.append(name)
    # both main terms flat (hop/shake may legitimately lag the posture early)
    return flat if len(flat) == len(main_hist) else []


def save_checkpoint(runner: Any, env: Any, out: Path, it: int, *, final: bool = False) -> Path:
    path = out / "checkpoints" / (f"model_{it}.pt" if not final else f"model_{it}_final.pt")
    runner.save(str(path))
    shutil.copyfile(path, out / "checkpoints" / "latest.pt")
    return path


def export(
    runner: Any, env: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any
) -> dict:
    """The trained actor → ``final/<net>.onnx`` + parity + manifest + the table section."""
    from duckmind.skilltrain import rl_action, rl_onnx
    from duckmind.skilltrain import table as TB

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy()
    actor.eval()
    scale, gain = C.pinned(args.net)
    action_cfg = env.cfg.actions["joint_pos"]
    metadata = {
        "duckmind_skill_net": args.net,
        "duckmind_stage": C.STAGE,
        "skills": [v.name for v in C.variants(args.net)],
        "robotd": rl_action.robotd_numbers(action_cfg, gain),
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
        "command_encoding": {v.name: list(v.command) for v in C.variants(args.net)},
    }
    onnx_path = rl_onnx.export_actor(actor, final / f"{args.net}.onnx", metadata)
    dev = next(actor.parameters()).device
    obs = wrapped.get_observations()["actor"][:256].detach().cpu().numpy()
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=512, obs=obs
    )
    problems = rl_onnx.robotd_problems(onnx_path)
    (final / "parity.json").write_text(
        json.dumps({"parity": par, "robotd_problems": problems}, indent=1) + "\n"
    )
    manifest = skill_manifest(args.net, onnx_path, runner.current_learning_iteration)
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    entries = [e for e in C.stage1_entries() if e.skill.path == C.PLACEHOLDER[args.net]]
    toml = TB.toml_section([
        TB.SkillDef(**{**e.skill.__dict__, "path": str(onnx_path.resolve())}) for e in entries
    ])  # fmt: skip
    (final / "robotd_skills.toml").write_text(toml, encoding="utf-8")
    log(f"exported {onnx_path.name} ({onnx_path.stat().st_size / 1e3:.0f} kB): parity max {par['max_abs']:.2e}, "
        f"robotd problems {problems or 'none'}")  # fmt: skip
    return {"onnx": str(onnx_path), "sha256": rl_onnx.sha256(onnx_path), "parity": par, "robotd_problems": problems,
            "manifest": str(final / "manifest.json"), "toml": str(final / "robotd_skills.toml"),
            "action_scale": scale, "gain_ratio": gain}  # fmt: skip


def skill_manifest(net: str, onnx_path: Path, iteration: int) -> dict:
    """A schema-2 *set* manifest (``policies: [...]``) for the file's table entries — the
    vocabulary robotd and upstream's publish tool share. Entries with an unwind are held poses in
    upstream's words (``perpetual`` + ``unwind_s``); ``hold_s`` is our table's window."""
    from duckmind.skilltrain import rl_onnx

    scale, gain = C.pinned(net)
    policies = []
    for v in C.variants(net):
        entry: dict[str, Any] = {
            "file": onnx_path.name,
            "name": v.name,
            "entry_pose": "sitting" if v.postures == ("sit",) else "standing",
            "action_scale": scale,
            "command": {
                "encoding": "constant",
                "twist": list(v.command),
                "idle": list(v.unwind),
                "head": "unused (zeros)",
                "body": "unused (zeros)",
            },  # fmt: skip
            "duckmind": {
                "family": v.family,
                "level": v.level,
                "postures": list(v.postures),
                "hold_s": v.hold_s,
                "gain_ratio": gain,
                "trained_with_lowpass": [C.LOWPASS_HEAD, C.LOWPASS_LEGS],
            },  # fmt: skip
        }
        if v.unwind_s > 0:
            entry.update(kind="perpetual", duration_s=None, unwind_s=v.unwind_s)
        else:
            entry.update(kind="episodic", duration_s=v.hold_s, chain=v.chain)
        policies.append(entry)
    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": C.OBS_LEN,
        "action_len": C.ACTION_LEN,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": C.CONTROL_HZ},
        "description": f"duckmind Stage-1 {net} skills (simulator only until phase 8)",
        "training": {
            "repo": "duckmind (skilltrain)",
            "task": f"Mjlab-Duck{net.capitalize()}-Flat-MicroDuck",
            "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
            "iteration": iteration,
            "sha256": rl_onnx.sha256(onnx_path),
            "exported": _now(),
        },  # fmt: skip
        "policies": policies,
    }


def run_eval(runner: Any, out: Path, args: argparse.Namespace, log: Any) -> dict:
    from duckmind.skilltrain import rl_eval, rl_onnx

    onnx = out / "final" / f"{args.net}.onnx"
    fn = rl_onnx.torch_policy(onnx, args.device)
    try:
        # the standard grid on the main library only (a joined "main,walks" training library would mix the Mac-like
        # walks into every walking cell and break the comparison with earlier runs; those get their own evals)
        res = rl_eval.evaluate(args.net, fn, policies_dir=args.policies, n_per_cell=args.eval_n, device=args.device,
                               library_path=_main_library(args.library), log=log)  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        log(f"eval failed: {type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}
    (out / "final" / "eval.json").write_text(
        json.dumps(_jsonable(res), indent=1, ensure_ascii=False) + "\n"
    )
    brief = {k: {kk: vv for kk, vv in c.items() if kk in ("n", "fall_skill", "fall_handback", "performed")}
             for k, c in res["cells"].items()}  # fmt: skip
    for key, c in brief.items():
        log(f"eval {key}: {c}")
    extra = _extra_evals(args, fn, out, log)
    return {
        "file": str(out / "final" / "eval.json"),
        "worst_fall_rate": res.get("worst_fall_rate"),
        "cells": brief,
        "extra": extra,
    }


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _main_library(library: str | None) -> str | None:
    return library.split(",")[0] if library else library


def _pooled(res: dict) -> dict:
    tot: dict[str, list[int]] = {}
    for c in res["cells"].values():
        for flag in ("fall_skill", "fall_handback", "performed"):
            if flag in c:
                k, n = tot.setdefault(flag, [0, 0])
                tot[flag] = [k + int(c[flag][0]), n + int(c[flag][1])]
    return {flag: C.rate(k, n) for flag, (k, n) in tot.items()}


def _extra_evals(args: argparse.Namespace, fn: Any, out: Path, log: Any) -> dict:
    """s1b:retrain (the V1 battery's delay cells): the same grid with a fixed servo command lag per run
    (``--eval-lags``, physics steps of 5 ms), and the walking cells on the Mac-like walk library
    (``--eval-walk-library``, at the recipe's delay and at every extra lag)."""
    from duckmind.skilltrain import rl_eval

    done: dict[str, Any] = {}
    lags = [int(v) for v in str(args.eval_lags or "").split(",") if v.strip() != ""]
    main = _main_library(args.library)
    jobs: list[tuple[str, Any, str | None, list | None]] = [
        (f"lag{lag}", lag, main, None) for lag in lags
    ]
    if args.eval_walk_library and args.net == "startle":
        # walking cells only: the Mac-like library holds walks (a coast cell would fall back to synthetic states)
        walk_cells = [c for c in rl_eval.default_cells(args.net) if c[1] == "walk"]
        for lag in [None, *lags]:
            tag = "macwalk" + ("" if lag is None else f"_lag{lag}")
            jobs.append((tag, lag, args.eval_walk_library, walk_cells))
    for tag, lag, library, cells in jobs:
        _free_gpu()  # each evaluation builds its own env next to the training env: free the last one first
        t0 = time.time()
        try:
            res = rl_eval.evaluate(args.net, fn, policies_dir=args.policies, n_per_cell=args.eval_n, device=args.device,
                                   library_path=library, log=log, servo_lag=lag, cells=cells)  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            log(f"extra eval {tag} failed: {type(exc).__name__}: {exc}")
            done[tag] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        path = out / "final" / f"eval_{tag}.json"
        path.write_text(json.dumps(_jsonable(res), indent=1, ensure_ascii=False) + "\n")
        pooled = _pooled(res)
        done[tag] = {"file": str(path), "worst_fall_rate": res.get("worst_fall_rate"), **pooled,
                     "wall_s": round(time.time() - t0, 1)}  # fmt: skip
        log(f"extra eval {tag}: {pooled} in {time.time() - t0:.0f} s")
    return done


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a Stage-1 skill network (startle or comfort)")
    ap.add_argument("--net", choices=C.NETS, required=True)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=0, help="0 = the plan's budget (2500 / 2000)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--library", default=None, help="entry-state library (npz); synthetic entries without"
    )
    ap.add_argument(
        "--policies", default=None, help="v5 folder (velstand, alpha_sitstand) for the final eval"
    )
    ap.add_argument(
        "--warm-start", default="none", help="an ONNX to import into the actor (plan S0 test)"
    )
    ap.add_argument("--init-std", type=float, default=1.0)
    ap.add_argument("--resume", default=None, help="a checkpoint of this net's run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument(
        "--schedule",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="PPO learning-rate schedule (adaptive KL 0.01 = the microduck recipes; fixed for diagnostics)",
    )
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument(
        "--reserve-min", type=float, default=4.0, help="kept for export + eval at the budget's end"
    )
    ap.add_argument("--kill-flat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--eval-n", type=int, default=0, help="ducks per eval cell after export (0 = no eval)"
    )
    ap.add_argument(
        "--eval-lags",
        default="",
        help="s1b: extra final evals with a fixed servo command lag each (physics steps of 5 ms), e.g. 0,4,8",
    )
    ap.add_argument(
        "--eval-walk-library",
        default="",
        help="s1b: startle's walking cells again on this library (the Mac-like walks)",
    )
    ap.add_argument("--no-pushes", action="store_true")
    ap.add_argument(
        "--curriculum-scale",
        type=float,
        default=1.0,
        help="compress (< 1) the curricula's iteration schedule for runs shorter than the plan's",
    )
    args = ap.parse_args(argv)
    if not args.iterations:
        from duckmind.skilltrain.rl_tasks import ITERATIONS

        args.iterations = ITERATIONS[args.net]
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
