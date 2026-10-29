"""Train the forward hop-over network (hop:feasibility, 2026-09-28): PPO on the official physics
over a strip terrain → robotd's ONNX. Stage 1's trainer loop (``rl_train``'s helpers, unchanged)
around the hop task (``hop_tasks``).

    python -m duckmind.skilltrain.hop_train --envs 4096 --iterations 1500 --device cuda:0 \
        --library /content/out/library/v5_entries.npz --policies /content/stage1/policies \
        --out /content/out/hopover-h1 --time-budget-min 120 --eval-n 32

Writes into ``--out``: ``progress.jsonl`` (one line per iteration: s/it, every Episode_Reward /
Episode_Metrics — crossed, fell, strip_mm — and Curriculum value, PPO losses, std), checkpoints
every ``--save-every`` (+ ``latest.pt``; hop:train: + ``hopover_<it>.onnx``), ``final/hopover.onnx``
(robotd's format, normaliser baked) + parity + manifest + the table section, ``final/eval.json``
(``--eval-n`` ducks per cell, hand-back to v5; hop:train: ``final/eval-<delay>.json`` for every
``--eval-delays`` mode, and ``checkpoints/eval_<it>.json`` with ``--ckpt-eval-n``), ``baseline.json``
(``--baseline-n``: the v5 walk through the same grid), ``metrics.json``. The servo command delay while
training is ``--delay`` (hop:train default ``train``: piecewise constant 0–40 ms).

Stop rules (Stage 1's): a cost logged above 0 or a reward below 0 stops at once (a sign bug; the
progress potential is signed by design and exempt); ``progress`` and ``cross_bonus`` both flat over
500 iterations stops at the 500 / 1,000 checks; the time budget saves, exports, evaluates, exits.
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

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import rl_contract as C
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

UNITS_PER_HOUR = 6.77
MAIN_TERMS = ("progress", "cross_bonus")
FLAT_CHECKS = (500, 1000)


def build(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401  (upstream's NaN-safe patches ride on this import)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from duckmind.skilltrain import hop_tasks as T

    torch.manual_seed(args.seed)
    env_cfg = T.make_env_cfg(library_path=args.library, curriculum_scale=args.curriculum_scale,
                             max_init_level=args.max_init_level, reward_rev=args.reward_rev,
                             delay_steps=H.parse_delay(args.delay),
                             handback_s=2.0 if args.reward_rev in ("h3", "h4") else 0.0)  # fmt: skip
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent = T.runner_cfg(iterations=args.iterations, init_std=args.init_std, seed=args.seed)
    if args.lr:
        agent.algorithm.learning_rate = args.lr
    if args.entropy is not None:
        agent.algorithm.entropy_coef = args.entropy
    agent_dict = asdict(agent)
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, log_dir=None, device=args.device)
    return env, wrapped, runner, {"env_cfg": env_cfg, "agent": agent_dict}


def poll_override(path: Path, state: dict) -> dict | None:
    """A new (or rewritten) ``OVERRIDE.json`` → its contents once; else None. Never raises."""
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
        return {"_error": f"unreadable override: {type(exc).__name__}: {exc}"}
    return spec if isinstance(spec, dict) else {"_error": "the override is not a JSON object"}


def apply_override(env: Any, spec: dict, costs: set[str], log: Any) -> dict:
    """Re-weight reward terms of the live reward manager mid-run (hop:train): ``{"id": "...", "weights":
    {term: w}, "params": {term: {key: value}}}``. Only :data:`hop_contract.OVERRIDABLE` terms/keys; a cost
    keeps a weight ≤ 0 and a reward ≥ 0 (the sign check stays valid); anything else is rejected and
    logged, never raised."""
    out: dict[str, Any] = {"id": spec.get("id"), "applied": {}, "rejected": {}}
    if "_error" in spec:
        out["rejected"]["file"] = spec["_error"]
        log(f"override rejected: {spec['_error']}")
        return out
    base = getattr(env, "unwrapped", env)
    rm = base.reward_manager
    names = set(rm.active_terms)
    for name, w in (spec.get("weights") or {}).items():
        if name not in H.OVERRIDABLE or name not in names:
            out["rejected"][name] = "not an overridable term"
            continue
        try:
            w = float(w)
        except (TypeError, ValueError):
            out["rejected"][name] = f"weight {w!r} is not a number"
            continue
        is_cost = name in costs
        if (is_cost and w > 0) or (not is_cost and w < 0):
            out["rejected"][name] = f"sign: {'cost' if is_cost else 'reward'} weight {w}"
            continue
        cfg = rm.get_term_cfg(name)
        out["applied"][name] = {"weight": [cfg.weight, w]}
        cfg.weight = w
    for name, params in (spec.get("params") or {}).items():
        allowed = H.OVERRIDABLE.get(name, ())
        if name not in names or not isinstance(params, dict):
            out["rejected"][f"{name}.params"] = "not an overridable term"
            continue
        cfg = rm.get_term_cfg(name)
        for key, value in params.items():
            if key not in allowed:
                out["rejected"][f"{name}.{key}"] = "not an overridable parameter"
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                out["rejected"][f"{name}.{key}"] = f"{value!r} is not a number"
                continue
            out["applied"].setdefault(name, {})[key] = [cfg.params.get(key), value]
            cfg.params[key] = value
    log(f"override {out['id']}: applied {out['applied']}; rejected {out['rejected'] or 'none'}")
    return out


def load_handback_policy(reward_rev: str, policies: str | None, device: str) -> Any | None:
    """Load the actual shipped v5 walk actor for the h3/h4 post-skill hand-back."""
    if reward_rev not in ("h3", "h4"):
        return None
    if not policies:
        raise ValueError(f"{reward_rev} requires --policies for the real v5 hand-back")
    from duckmind.skilltrain.rl_onnx import torch_policy

    return torch_policy(Path(policies) / "velstand.onnx", device)


def warm_start(runner: Any, wrapped: Any, onnx_path: str, init_std: float, log: Any) -> dict:
    """The actor from a microduck ONNX (velstand: it already walks), a fresh critic."""
    from duckmind.skilltrain import rl_onnx

    actor = runner.alg.actor
    info = rl_onnx.load_into_actor(actor, onnx_path, init_std=init_std, frozen=True)
    obs = wrapped.get_observations()["actor"].detach().cpu().numpy()
    dev = next(actor.parameters()).device
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=256, obs=obs
    )
    info["parity"] = par
    log(
        f"[warm] actor ← {Path(onnx_path).name}, parity max {par['max_abs']:.2e}, init std {init_std}"
    )
    if par["max_abs"] > 1e-4:
        raise RuntimeError(f"warm-start parity {par['max_abs']:.2e} > 1e-4")
    return info


def train(args: argparse.Namespace) -> dict:
    import torch

    from duckmind.skilltrain import hop_mdp as M
    from duckmind.skilltrain import hop_tasks as T

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    progress = (out / "progress.jsonl").open("a", encoding="utf-8")
    log_file = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[hopover {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t_start = time.time()
    baseline = None
    if args.baseline_n > 0 and args.policies:
        from duckmind.skilltrain import hop_eval

        try:
            baseline = hop_eval.evaluate("v5walk", policies_dir=args.policies, n_per_cell=args.baseline_n,
                                         device=args.device, library_path=args.library, log=log,
                                         delay=H.parse_delay(args.delay))  # fmt: skip
            (out / "baseline.json").write_text(json.dumps(_jsonable(baseline), indent=1) + "\n")
            for key, v in baseline["by_height"].items():
                log(f"baseline v5walk {key}: {v}")
        except Exception as exc:  # noqa: BLE001
            log(f"baseline failed: {type(exc).__name__}: {exc}")
    env, wrapped, runner, built = build(args)
    t_build = time.time() - t_start
    env_cfg = built["env_cfg"]
    costs = set(T.cost_terms(env_cfg))
    rewards = {
        n
        for n, t in env_cfg.rewards.items()
        if n not in costs and t.weight > 0 and n not in M.SIGNED_TERMS
    }
    meta: dict[str, Any] = {
        "format": "duckmind-skill-run-1",
        "net": H.NET,
        "task_id": H.TASK_ID,
        "started": _now(),
        "args": vars(args),
        "host": platform.node(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and "cuda" in args.device
        else None,
        "variants": [v.name for v in H.variants()],
        "pinned": {"action_scale": H.ACTION_SCALE, "gain_ratio": H.GAIN_RATIO},
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
        "reward_rev": args.reward_rev,
        "servo_delay_steps": [_delay_meta(env)["delay_min_lag"], _delay_meta(env)["delay_max_lag"]],
        "servo_delay_ms_trained": [
            5 * _delay_meta(env)["delay_min_lag"],
            5 * _delay_meta(env)["delay_max_lag"],
        ],
        "servo_delay": {
            "mode": H.delay_label(H.parse_delay(args.delay)),
            "fields": H.delay_fields(H.parse_delay(args.delay)),
        },  # fmt: skip
        "build_s": round(t_build, 1),
        "obs_dims": {k: list(v.shape) for k, v in wrapped.get_observations().items()},
    }
    log(f"built {args.envs} envs on {args.device} in {t_build:.1f} s; obs {meta['obs_dims']}")
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
    main_hist: dict[str, list[float]] = {k: [] for k in MAIN_TERMS}
    override_path = Path(args.override_file) if args.override_file else out / "OVERRIDE.json"
    override_state: dict[str, Any] = {"mtime": None}
    meta["override_file"] = str(override_path)
    obs = wrapped.get_observations().to(runner.device)
    from duckmind.skilltrain import rl_mdp as RM

    handback_policy = load_handback_policy(args.reward_rev, args.policies, args.device)
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
                    executed = actions.to(wrapped.device)
                    if handback_policy is not None:
                        # The environment ignores the candidate's action after ownership returns
                        # to v5. PPO retains the sampled action, as for any action-independent phase.
                        hb = RM.skill_term(env).phase >= RM.HANDBACK
                        executed = torch.where(
                            hb.unsqueeze(1), handback_policy(obs["actor"]), executed
                        )
                    obs, rew, dones, extras = wrapped.step(executed)
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
            row = {"it": it, "s": round(last_iter_s, 3), "collect_s": round(t_collect, 3),
                   "lr": runner.alg.learning_rate, "std": round(std, 4),
                   "losses": {k: round(float(v), 5) for k, v in losses.items()},
                   **{k: round(v, 5) for k, v in means.items()}}  # fmt: skip
            for term in main_hist:
                key = f"Episode_Reward/{term}"
                if key in means:
                    main_hist[term].append(means[key])
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
            done_its = it - start_it + 1
            if it % args.print_every == 0 or done_its == args.iterations:
                eta = (total_it - it - 1) * last_iter_s / 60
                log(f"it {it}/{total_it - 1} {last_iter_s:.2f} s/it | crossed {row.get('Episode_Metrics/crossed', float('nan')):.3f} "
                    f"fell {row.get('Episode_Metrics/fell', float('nan')):.3f} strip {row.get('Episode_Metrics/strip_mm', float('nan')):.1f} mm "
                    f"level {row.get('Curriculum/strip_levels/mean', float('nan')):.2f} | progress {row.get('Episode_Reward/progress', float('nan')):.3f} "
                    f"| std {std:.3f} lr {runner.alg.learning_rate:.1e} | eta {eta:.0f} min")  # fmt: skip
            if problems:
                stop_reason = "sign: " + "; ".join(problems)
                log("STOP " + stop_reason)
                break
            if (it + 1) % args.save_every == 0:
                save_checkpoint(runner, env, out, it)
                if args.ckpt_onnx:
                    checkpoint_onnx(runner, env, out, it, log)
            if args.kill_flat and done_its in FLAT_CHECKS:
                flat = flat_terms(main_hist)
                if flat:
                    stop_reason = f"flat: {flat} over the last 500 iterations"
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
        "baseline_by_height": baseline["by_height"] if baseline else None,
    }
    if args.export and not stop_reason.startswith("error"):
        summary["export"] = export(runner, env, wrapped, out, args, log)
        (out / "metrics.json").write_text(
            json.dumps(_jsonable(summary), indent=1, ensure_ascii=False) + "\n"
        )
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        if args.eval_n > 0 and args.policies:
            modes = [H.parse_delay(m) for m in args.eval_delays.split(",") if m.strip()]
            summary["eval"] = run_eval(out, args, log, delay=modes[0])
            summary["eval_by_delay"] = {H.delay_label(modes[0]): summary["eval"]}
            for mode in modes[1:]:
                summary["eval_by_delay"][H.delay_label(mode)] = run_eval(out, args, log, delay=mode)
            (out / "metrics.json").write_text(
                json.dumps(_jsonable(summary), indent=1, ensure_ascii=False) + "\n"
            )
        if args.ckpt_eval_n > 0 and args.policies:
            summary["checkpoint_evals"] = eval_checkpoints(out, args, log)
    (out / "metrics.json").write_text(
        json.dumps(_jsonable(summary), indent=1, ensure_ascii=False) + "\n"
    )
    log(f"done: {stop_reason}; metrics → {out / 'metrics.json'}")
    progress.close()
    log_file.close()
    return summary


def export(
    runner: Any, env: Any, wrapped: Any, out: Path, args: argparse.Namespace, log: Any
) -> dict:
    from duckmind.skilltrain import rl_action, rl_onnx
    from duckmind.skilltrain import table as TB

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy()
    actor.eval()
    action_cfg = env.cfg.actions["joint_pos"]
    metadata = {
        "duckmind_skill_net": H.NET,
        "duckmind_stage": H.STAGE,
        "skills": [v.name for v in H.variants()],
        "robotd": rl_action.robotd_numbers(action_cfg, H.GAIN_RATIO),
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
        "command_encoding": {v.name: list(v.command) for v in H.variants()},
        "reward_rev": args.reward_rev,
        "curriculum_scale": args.curriculum_scale,
        "handback_s": 2.0 if args.reward_rev in ("h3", "h4") else 0.0,
        "servo_delay": {
            "mode": H.delay_label(H.parse_delay(args.delay)),
            "fields": _delay_meta(env),
        },
        "servo_delay_ms_trained": [
            5 * _delay_meta(env)["delay_min_lag"],
            5 * _delay_meta(env)["delay_max_lag"],
        ],
        "servo_delay_mode_trained": _delay_meta(env),
    }
    onnx_path = rl_onnx.export_actor(actor, final / f"{H.NET}.onnx", metadata)
    dev = next(actor.parameters()).device
    obs = wrapped.get_observations()["actor"][:256].detach().cpu().numpy()
    par = rl_onnx.parity(
        onnx_path, lambda x: rl_onnx.actor_forward(actor, x.to(dev)).cpu(), samples=512, obs=obs
    )
    problems = rl_onnx.robotd_problems(onnx_path)
    (final / "parity.json").write_text(
        json.dumps({"parity": par, "robotd_problems": problems}, indent=1) + "\n"
    )
    defs = [TB.SkillDef(**{**d, "path": str(onnx_path.resolve())}) for d in H.skill_defs()]
    (final / "robotd_skills.toml").write_text(TB.toml_section(defs), encoding="utf-8")
    manifest = {
        "schema_version": 2, "model_api": 1, "obs_len": C.OBS_LEN, "action_len": C.ACTION_LEN,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": C.CONTROL_HZ},
        "description": "duckmind forward hop-over skill (simulator only until phase 8)",
        "training": {"repo": "duckmind (skilltrain hop_*)", "task": H.TASK_ID, "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
                     "iteration": runner.current_learning_iteration, "sha256": rl_onnx.sha256(onnx_path), "exported": _now()},
        "policies": [{"file": onnx_path.name, "name": v.name, "entry_pose": "standing", "action_scale": v.action_scale,
                      "kind": "perpetual", "duration_s": None, "unwind_s": v.unwind_s,
                      "command": {"encoding": "constant", "twist": list(v.command), "idle": list(v.unwind),
                                  "head": "unused (zeros)", "body": "unused (zeros)"},
                      "duckmind": {"family": "hop_over", "height_m": v.height_m, "postures": list(v.postures),
                                   "hold_s": v.hold_s, "gain_ratio": v.gain_ratio,
                                   "trained_with_lowpass": [C.LOWPASS_HEAD, C.LOWPASS_LEGS]}}
                     for v in H.variants()],
    }  # fmt: skip
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


def _delay_meta(env: Any) -> dict:
    """The robot's BAM actuator delay fields as trained (for the export metadata)."""
    try:
        for a in env.cfg.scene.entities["robot"].articulation.actuators:
            if hasattr(a, "delay_max_lag"):
                return {k: getattr(a, k) for k in ("delay_min_lag", "delay_max_lag",
                                                   "delay_update_period", "delay_hold_prob")}  # fmt: skip
    except Exception:  # noqa: BLE001
        pass
    return {}


def checkpoint_onnx(runner: Any, env: Any, out: Path, it: int, log: Any) -> Path | None:
    """The actor at a checkpoint → ``checkpoints/hopover_<it>.onnx`` (robotd's format; the live actor
    is untouched: ``export_actor`` exports a CPU copy), so the Mac gates can run any checkpoint."""
    from duckmind.skilltrain import rl_action, rl_onnx

    try:
        metadata = {
            "duckmind_skill_net": H.NET,
            "duckmind_stage": H.STAGE,
            "skills": [v.name for v in H.variants()],
            "robotd": rl_action.robotd_numbers(env.cfg.actions["joint_pos"], H.GAIN_RATIO),
            "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
            "iteration": int(it),
            "checkpoint": True,
            "servo_delay_mode_trained": _delay_meta(env),
        }
        path = rl_onnx.export_actor(
            runner.alg.get_policy(), out / "checkpoints" / f"{H.NET}_{it}.onnx", metadata
        )
        problems = rl_onnx.robotd_problems(path)
        log(f"checkpoint {it} → {path.name}; robotd problems {problems or 'none'}")
        return path
    except Exception as exc:  # noqa: BLE001
        log(f"checkpoint {it} export failed: {type(exc).__name__}: {exc}")
        return None


def run_eval(
    out: Path, args: argparse.Namespace, log: Any, *, delay: str | int = "train",
    onnx: Path | None = None, n: int | None = None, dest: Path | None = None,
) -> dict:  # fmt: skip
    """``final/eval.json`` (training's delay mode) and ``final/eval-<delay>.json`` for each mode."""
    from duckmind.skilltrain import hop_eval, rl_onnx

    onnx = onnx or out / "final" / f"{H.NET}.onnx"
    fn = rl_onnx.torch_policy(onnx, args.device)
    label = H.delay_label(delay)
    try:
        res = hop_eval.evaluate(fn, policies_dir=args.policies, n_per_cell=n or args.eval_n, device=args.device,
                                library_path=args.library, log=log, delay=delay)  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        log(f"eval ({label}) failed: {type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}", "delay": label}
    text = json.dumps(_jsonable(res), indent=1, ensure_ascii=False) + "\n"
    files = [dest] if dest else [out / "final" / f"eval-{label}.json"]
    if dest is None and label == H.delay_label(H.parse_delay(args.delay)):
        files.append(out / "final" / "eval.json")
    for f in files:
        f.write_text(text)
    for key, v in res["by_height"].items():
        log(f"eval {onnx.name} ({label}) {key}: {v}")
    return {"file": str(files[0]), "delay": label, "by_height": res["by_height"]}


def eval_checkpoints(out: Path, args: argparse.Namespace, log: Any) -> dict:
    """Every ``checkpoints/hopover_<it>.onnx`` through the evaluator (training's delay mode, ``--ckpt-eval-n``
    per cell) → ``checkpoints/eval_<it>.json``: picks the checkpoint if a later one is worse."""
    found = sorted(
        (int(p.stem.rsplit("_", 1)[1]), p) for p in (out / "checkpoints").glob(f"{H.NET}_*.onnx")
    )
    res = {}
    for it, p in found:
        if (out / "STOP_EVALS").exists():
            log("checkpoint evaluations stopped (STOP_EVALS)")
            break
        res[str(it)] = run_eval(out, args, log, delay=H.parse_delay(args.delay), onnx=p,
                                n=args.ckpt_eval_n, dest=out / "checkpoints" / f"eval_{it}.json")  # fmt: skip
    return res


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the forward hop-over network")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument(
        "--policies", default=None, help="v5 folder (velstand.onnx) for the baseline / eval"
    )
    ap.add_argument("--warm-start", default="none")
    ap.add_argument("--init-std", type=float, default=1.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--reserve-min", type=float, default=4.0)
    ap.add_argument("--kill-flat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-n", type=int, default=0)
    ap.add_argument("--baseline-n", type=int, default=0)
    ap.add_argument("--curriculum-scale", type=float, default=1.0)
    ap.add_argument("--max-init-level", type=int, default=2)
    ap.add_argument(
        "--reward-rev", default="h2", help="hop_tasks.REWARD_REVS (h1 = the calibration's)"
    )
    ap.add_argument(
        "--delay", default="train",
        help="servo command delay while training: train (piecewise constant 0–40 ms), jitter (the "
             "calibration's per-step 0–40 ms) or a fixed delay (20ms / 4 steps)",
    )  # fmt: skip
    ap.add_argument(
        "--eval-delays", default="train",
        help="comma list of delay modes for the final evaluation (the first → final/eval.json too)",
    )  # fmt: skip
    ap.add_argument("--ckpt-onnx", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--override-file", default=None,
        help="mid-run reward re-weighting (default <out>/OVERRIDE.json; see apply_override)",
    )  # fmt: skip
    ap.add_argument(
        "--ckpt-eval-n", type=int, default=0, help="evaluate every checkpoint ONNX (n per cell)"
    )
    args = ap.parse_args(argv)
    H.parse_delay(args.delay)  # validate early
    for m in args.eval_delays.split(","):
        if m.strip():
            H.parse_delay(m)
    if args.device.startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return args


def main(argv: list[str] | None = None) -> int:
    summary = train(parse(argv))
    return 0 if not str(summary.get("stop_reason", "")).startswith(("error", "sign")) else 1


if __name__ == "__main__":
    sys.exit(main())
