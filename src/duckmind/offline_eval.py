"""Manual, CPU-only official physics/parity probe; not a pytest test.

Run with the prepared official trainer interpreter and a TRUSTED local checkpoint
(official runner.load uses torch.load(weights_only=False)). No training/export,
robotd, servo, rendering, network, or policy publication is performed. The caller
must provide a wall timeout/process-group guard, e.g. training.run_phase.

Source contracts: microduck_rl cb70b792312d559a4da09064d92009079671815f
src/mjlab_microduck/export.py:214-261; mjlab 1.3.0 rl/runner.py and
envs/manager_based_rl_env.py:351-473; rsl-rl-lib 5.0.1 models/mlp_model.py.
The official inference actor already includes its fitted observation normalizer.
auto_reset=False preserves terminal physics instead of measuring a reset pose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

TASK = "Mjlab-Velocity-Flat-MicroDuck"
UPSTREAM_COMMIT = "cb70b792312d559a4da09064d92009079671815f"
SEEDS = (10001, 10002, 10003, 10004)
HORIZON = 250
RTOL, ATOL = 1e-4, 1e-5


@dataclass(frozen=True)
class EvaluationProfile:
    play: bool
    diagnostic_only: bool
    push_enabled: bool
    push_velocity_zeroed: bool = False


EVALUATION_PROFILES = {
    "play": EvaluationProfile(play=True, diagnostic_only=False, push_enabled=True),
    "nonplay": EvaluationProfile(play=False, diagnostic_only=True, push_enabled=True),
    "play-no-push": EvaluationProfile(play=True, diagnostic_only=True, push_enabled=False),
    "play-zero-push": EvaluationProfile(
        play=True, diagnostic_only=True, push_enabled=True, push_velocity_zeroed=True
    ),
}


def configure_probe_events(config, profile: EvaluationProfile) -> None:
    """Ablate push effect while keeping the remaining official play settings."""

    if not profile.push_enabled or profile.push_velocity_zeroed:
        push = config.events.get("push_robot")
        if push is None:
            raise ValueError("official play config is missing push_robot")
        if profile.push_velocity_zeroed:
            velocity_range = push.params.get("velocity_range")
            if not isinstance(velocity_range, dict) or set(velocity_range) != {"x", "y"}:
                raise ValueError("official push event has an unexpected velocity_range")
            push.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
            return
        del config.events["push_robot"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_identity(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT or git("status", "--porcelain"):
        raise ValueError("upstream must be the clean reviewed revision")
    hashes = {name: digest(root / name) for name in git("ls-files", "-z").split("\0") if name}
    return {
        "commit": UPSTREAM_COMMIT,
        "tracked_tree_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "uv_lock_sha256": hashes["uv.lock"],
    }


def write_report(path, report):
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--reference-onnx", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--evaluation-profile",
        choices=tuple(EVALUATION_PROFILES),
        default="play",
        help="nonplay and play-*-push are diagnostics and cannot qualify a policy",
    )
    args = parser.parse_args()
    profile = EVALUATION_PROFILES[args.evaluation_profile]
    play = profile.play
    for name in ("upstream", "checkpoint", "onnx", "report"):
        setattr(args, name, getattr(args, name).resolve())
    if args.reference_onnx:
        args.reference_onnx = args.reference_onnx.resolve()
    if args.report.exists():
        parser.error("report already exists; choose a new path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream / "src"))
    report = {
        "schema_version": 1,
        "status": "running",
        "experimental": True,
        "task_id": TASK,
        "device": "cpu",
        "cpu_threads": 2,
        "seeds": list(SEEDS),
        "horizon": HORIZON,
        "num_envs": 1,
        "play": play,
        "evaluation_profile": args.evaluation_profile,
        "diagnostic_only": profile.diagnostic_only,
        "push_event_enabled": profile.push_enabled,
        "push_velocity_zeroed": profile.push_velocity_zeroed,
        "auto_reset": False,
        "physical_hardware_gate": "not_run",
        "robotd_gate": "not_run",
        "walking_success": "not_claimed",
        "policy_quality_gate": "not_assessed",
        "scope": "Checkpoint/ONNX parity and finite-horizon official BAM simulation observations only.",
        "started_at": datetime.now(UTC).isoformat(),
        "probe_script_sha256": digest(__file__),
        "candidate_episodes": [],
        "reference_episodes": [],
        "parity": {
            "rtol": RTOL,
            "atol": ATOL,
            "samples": 0,
            "failed_samples": 0,
            "max_abs_error": 0.0,
            "all_finite": True,
            "passed": False,
        },
        "limitations": [
            "Four paired seeds and a short horizon are not a locomotion or safety qualification.",
            "Official play configuration retains its configured noise, events and curricula.",
            "Termination/reward derived state can lag one physics substep per mjlab's step contract.",
            "Reference velstand, if supplied, is a different-task baseline, not a training-quality oracle.",
        ],
    }
    if not profile.push_enabled:
        report["limitations"].append(
            "Pushes were removed from the official play configuration for causal diagnosis only."
        )
    if profile.push_velocity_zeroed:
        report["limitations"].append(
            "Push scheduling and sampling remain active, but x/y velocity ranges are zeroed for diagnosis only."
        )
    write_report(args.report, report)
    started = time.monotonic()
    try:
        identity = upstream_identity(args.upstream)
        report["upstream"] = identity
        model_paths = {"checkpoint": args.checkpoint, "onnx": args.onnx}
        if args.reference_onnx:
            model_paths["reference_onnx"] = args.reference_onnx
        report["artifacts"] = {
            name: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
            for name, path in model_paths.items()
        }
        versions = {
            name: metadata.version(name)
            for name in (
                "mjlab",
                "rsl-rl-lib",
                "torch",
                "warp-lang",
                "mujoco",
                "mujoco-warp",
                "better-actuator-models",
                "onnxruntime",
                "numpy",
            )
        }
        for name, expected in {
            "mjlab": "1.3.0",
            "rsl-rl-lib": "5.0.1",
            "torch": "2.9.1",
            "warp-lang": "1.12.0",
            "mujoco": "3.10.0",
            "mujoco-warp": "3.8.1",
        }.items():
            if versions[name].split("+")[0] != expected:
                raise ValueError(f"unexpected dependency: {name}={versions[name]}")
        report["versions"] = versions
        import bam
        import mjlab
        import mjlab.tasks  # noqa: F401 -- official task-plugin registration
        import mjlab_microduck
        import mujoco
        import numpy as np
        import onnxruntime as ort
        import rsl_rl
        import torch
        import yaml
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from tensordict import TensorDict

        if (
            Path(mjlab_microduck.__file__).resolve()
            != args.upstream / "src/mjlab_microduck/__init__.py"
        ):
            raise ValueError("wrong microduck import")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        bam_root = Path(bam.__file__).parent
        physics_files = {
            "bam_mjlab": bam_root / "mjlab.py",
            "bam_model": bam_root / "model.py",
            "bam_xl330_m6": bam_root / "params/xl330/m6.json",
            "microduck_robot": args.upstream / "src/mjlab_microduck/robot/microduck_constants.py",
            "microduck_friction_dr": args.upstream
            / "src/mjlab_microduck/actuator/friction_dr_bam.py",
            "microduck_xml": args.upstream / "src/mjlab_microduck/robot/microduck/robot_walk.xml",
            "mjlab_environment": Path(mjlab.__file__).parent / "envs/manager_based_rl_env.py",
            "mjlab_runner": Path(mjlab.__file__).parent / "rl/runner.py",
            "rsl_actor": Path(rsl_rl.__file__).parent / "models/mlp_model.py",
        }
        report["physics_source_sha256"] = {
            name: digest(path) for name, path in physics_files.items()
        }
        bam_origin = metadata.distribution("better-actuator-models").read_text("direct_url.json")
        report["bam_origin"] = json.loads(bam_origin) if bam_origin else None

        def session(path):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            model = ort.InferenceSession(
                path.read_bytes(), options, providers=["CPUExecutionProvider"]
            )
            inputs, outputs = model.get_inputs(), model.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("only feed-forward one-input/one-output policies supported")
            for outlet, width in ((inputs[0], 61), (outputs[0], 14)):
                if (
                    outlet.type != "tensor(float)"
                    or len(outlet.shape) != 2
                    or outlet.shape[1] != width
                ):
                    raise ValueError("policy must have float32 [batch,61] -> [batch,14]")
                if isinstance(outlet.shape[0], int) and outlet.shape[0] not in (1, -1):
                    raise ValueError("policy batch must be one or dynamic")
            if inputs[0].name != "obs":
                raise ValueError("policy input must be obs")
            return model

        candidate = session(args.onnx)
        reference = session(args.reference_onnx) if args.reference_onnx else None

        def config(seed):
            cfg = load_env_cfg(TASK, play=play)
            cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
            configure_probe_events(cfg, profile)
            return cfg

        agent_cfg = load_rl_cfg(TASK)
        agent_cfg.logger, agent_cfg.upload_model = "tensorboard", False
        env = ManagerBasedRlEnv(config(SEEDS[0]), device="cpu", render_mode=None)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_cls = load_runner_cls(TASK)
            if runner_cls is None:
                raise ValueError("expected official Microduck runner")
            runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
            runner.load(str(args.checkpoint), map_location="cpu")
            actor = runner.get_inference_policy(device="cpu")
            if actor.is_recurrent or tuple(actor.obs_groups) != ("actor",) or actor.obs_dim != 61:
                raise ValueError("unsupported checkpoint actor contract")
            counter = int(env.common_step_counter)
            normalizer_hash = hashlib.sha256()
            for name, tensor in sorted(actor.obs_normalizer.state_dict().items()):
                normalizer_hash.update(name.encode())
                normalizer_hash.update(tensor.detach().cpu().numpy().tobytes())
            report["normalizer"] = {
                "source": "official runner.load checkpoint actor; no manual preprocessing",
                "class": type(actor.obs_normalizer).__qualname__,
                "state_sha256": normalizer_hash.hexdigest(),
            }
            report["checkpoint_common_step_counter"] = counter
        finally:
            env.close()

        def episode(model, seed, compare):
            cfg = config(seed)
            config_yaml = yaml.dump(asdict(cfg), sort_keys=True)
            result = {
                "seed": seed,
                "steps": 0,
                "fell_over": False,
                "terminated": False,
                "truncated": False,
                "all_finite": True,
                "reward_sum": 0.0,
                "config_yaml_sha256": hashlib.sha256(config_yaml.encode()).hexdigest(),
                "trace": [],
            }
            environment = ManagerBasedRlEnv(cfg, device="cpu", render_mode=None)
            try:
                environment.common_step_counter = counter
                obs, _ = environment.reset(seed=seed)
                robot = environment.scene["robot"]
                indices, names = robot.find_bodies("trunk_base")
                if names != ["trunk_base"]:
                    raise ValueError("cannot identify trunk body")
                trunk = indices[0]
                initial = obs["actor"].detach().cpu().numpy()
                result["initial_actor_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
                result["initial_trunk_height_m"] = float(robot.data.body_link_pos_w[0, trunk, 2])
                result["min_trunk_height_m"] = result["initial_trunk_height_m"]
                result["step_dt_s"] = environment.step_dt
                result["physics_dt_s"] = environment.physics_dt
                result["clip_actions"] = agent_cfg.clip_actions
                result["termination_terms"] = list(environment.termination_manager.active_terms)
                with tempfile.NamedTemporaryFile(suffix=".mjb") as binary:
                    mujoco.mj_saveModel(environment.sim.mj_model, binary.name)
                    result["compiled_model_sha256"] = digest(binary.name)
                observation_digest = hashlib.sha256()
                for step in range(HORIZON):
                    raw = obs["actor"].detach().cpu().numpy().astype(np.float32, copy=False)
                    if raw.shape != (1, 61) or not all(
                        torch.isfinite(value).all() for value in obs.values()
                    ):
                        raise ValueError("nonfinite or wrong-shaped observations")
                    observation_digest.update(raw.tobytes())
                    actions = model.run([model.get_outputs()[0].name], {"obs": raw})[0]
                    if actions.shape != (1, 14) or not np.isfinite(actions).all():
                        report["parity"]["all_finite"] = False
                        raise ValueError("nonfinite or wrong-shaped actions")
                    if compare:
                        with torch.inference_mode():
                            expected = actor(TensorDict(obs, batch_size=[1])).detach().cpu().numpy()
                        parity = report["parity"]
                        parity["samples"] += 1
                        if expected.shape != (1, 14) or not np.isfinite(expected).all():
                            parity["all_finite"] = False
                            raise ValueError("nonfinite checkpoint actions")
                        parity["max_abs_error"] = max(
                            parity["max_abs_error"], float(np.max(np.abs(expected - actions)))
                        )
                        parity["failed_samples"] += int(
                            not np.allclose(actions, expected, rtol=RTOL, atol=ATOL)
                        )
                    commands = {
                        name: environment.command_manager.get_command(name)[0].tolist()
                        for name in environment.command_manager.active_terms
                    }
                    action_tensor = torch.from_numpy(actions.copy())
                    if agent_cfg.clip_actions is not None:
                        action_tensor = action_tensor.clamp(
                            -agent_cfg.clip_actions, agent_cfg.clip_actions
                        )
                    obs, reward, terminated, truncated, _ = environment.step(action_tensor)
                    height = float(robot.data.body_link_pos_w[0, trunk, 2])
                    finite = bool(
                        torch.isfinite(robot.data.joint_pos).all()
                        and torch.isfinite(robot.data.joint_vel).all()
                        and torch.isfinite(robot.data.body_link_pos_w).all()
                        and torch.isfinite(reward).all()
                    )
                    if not finite:
                        raise ValueError("nonfinite physics state/reward")
                    terms = {
                        name: bool(environment.termination_manager.get_term(name)[0])
                        for name in environment.termination_manager.active_terms
                    }
                    result["min_trunk_height_m"] = min(result["min_trunk_height_m"], height)
                    result["reward_sum"] += float(reward[0])
                    result["steps"] = step + 1
                    result["fell_over"] = result["fell_over"] or terms.get("fell_over", False)
                    result["terminated"], result["truncated"] = (
                        bool(terminated[0]),
                        bool(truncated[0]),
                    )
                    result["trace"].append(
                        {
                            "step": step + 1,
                            "trunk_height_m": height,
                            "reward": float(reward[0]),
                            "commands_before_step": commands,
                            "raw_actions": actions[0].tolist(),
                            "termination_terms": terms,
                        }
                    )
                    if result["terminated"] or result["truncated"]:
                        break
                result["episode_length_steps"] = int(environment.episode_length_buf[0])
                result["observation_stream_sha256"] = observation_digest.hexdigest()
                result["stopped_at"] = (
                    "environment_done"
                    if result["terminated"] or result["truncated"]
                    else "evaluation_horizon"
                )
                return result
            finally:
                environment.close()

        for seed in SEEDS:
            report["candidate_episodes"].append(episode(candidate, seed, True))
            write_report(args.report, report)
            if reference:
                report["reference_episodes"].append(episode(reference, seed, False))
                write_report(args.report, report)
        report["parity"]["passed"] = bool(
            report["parity"]["samples"]
            and report["parity"]["failed_samples"] == 0
            and report["parity"]["all_finite"]
        )
        if reference:
            report["paired_initial_observations_match"] = all(
                left["initial_actor_sha256"] == right["initial_actor_sha256"]
                and left["config_yaml_sha256"] == right["config_yaml_sha256"]
                for left, right in zip(
                    report["candidate_episodes"], report["reference_episodes"], strict=True
                )
            )
        for name, episodes in (
            ("candidate", report["candidate_episodes"]),
            ("reference", report["reference_episodes"]),
        ):
            if episodes:
                report[f"{name}_summary"] = {
                    "episodes": len(episodes),
                    "falls": sum(item["fell_over"] for item in episodes),
                    "total_steps": sum(item["steps"] for item in episodes),
                    "minimum_trunk_height_m": min(item["min_trunk_height_m"] for item in episodes),
                    "reward_sum": sum(item["reward_sum"] for item in episodes),
                }
        if upstream_identity(args.upstream) != identity or any(
            digest(path) != report["artifacts"][name]["sha256"]
            for name, path in model_paths.items()
        ):
            raise ValueError("source/artifact changed during probe")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report["finished_at"] = datetime.now(UTC).isoformat()
        write_report(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "parity": report["parity"],
                "candidate": report.get("candidate_summary"),
                "reference": report.get("reference_summary"),
            },
            indent=2,
        )
    )
    return 0 if report["parity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
