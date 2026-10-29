"""Experimental Mac/CPU PPO balance training, never a robot deployment command.

Uses official native MuJoCo body geometry/PD, not mjlab's Warp/BAM training task.
SB3 2.7.1 PPO/export contracts:
https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html
https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/export.html
All artifacts remain quarantined in a fresh local job directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duckmind.training import child_environment, run_phase, verify_upstream
from duckmind.versions import package_fingerprint

__all__ = ["LocalTrainingSpec", "compare_evaluations", "run_local_job"]

EVAL_SEEDS = tuple(range(10001, 10009))
REQUIRED_VERSIONS = {
    "stable-baselines3": "2.7.1",
    "torch": "2.9.1",
    "gymnasium": "1.2.3",
    "mujoco": "3.10.0",
    "onnx": "1.20.1",
    "onnxruntime": "1.27.0",
}


@dataclass(frozen=True)
class LocalTrainingSpec:
    timesteps: int = 32768
    seed: int = 7
    threads: int = 2
    timeout_s: int = 900
    horizon_steps: int = 250

    def __post_init__(self) -> None:
        for name, low, high in (
            ("timesteps", 256, 1_048_576),
            ("seed", 0, 9999),
            ("threads", 1, 8),
            ("timeout_s", 1, 7200),
            ("horizon_steps", 10, 1000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.timesteps % 256:
            raise ValueError("timesteps must be a multiple of the 256-step PPO rollout")


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_evaluations(baseline: list[dict], candidate: list[dict]) -> dict[str, Any]:
    """A paired diagnostic, deliberately not a hardware promotion criterion."""
    seeds = [row["seed"] for row in baseline]
    if not seeds or len(set(seeds)) != len(seeds) or seeds != [r["seed"] for r in candidate]:
        raise ValueError("evaluations must contain paired unique seeds in the same order")
    for row in (*baseline, *candidate):
        if not all(math.isfinite(row[key]) for key in ("steps", "return")):
            raise ValueError("non-finite evaluation")
    reward_delta = sum(c["return"] - b["return"] for b, c in zip(baseline, candidate)) / len(seeds)
    steps_delta = sum(c["steps"] - b["steps"] for b, c in zip(baseline, candidate)) / len(seeds)
    baseline_falls = sum(bool(r["fell"]) for r in baseline)
    candidate_falls = sum(bool(r["fell"]) for r in candidate)
    return {
        "paired_seeds": seeds,
        "reward_delta": reward_delta,
        "survival_steps_delta": steps_delta,
        "baseline_falls": baseline_falls,
        "candidate_falls": candidate_falls,
        "non_regression": reward_delta >= 0
        and steps_delta >= 0
        and candidate_falls <= baseline_falls,
        "hardware_approved": False,
    }


def parity_error(expected: Any, actual: Any) -> float:
    pairs = list(zip(expected, actual, strict=True))
    if not pairs or any(not math.isfinite(float(x)) for pair in pairs for x in pair):
        raise ValueError("export parity needs nonempty finite values on both sides")
    difference = max(abs(float(left) - float(right)) for left, right in pairs)
    if not math.isfinite(difference):
        raise ValueError("non-finite parity difference")
    return difference


def close_environments(environments: Any) -> list[str]:
    errors = []
    for instance in environments:
        if instance is not None:
            try:
                instance.close()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}"[:200])
    return errors


def export_actor(policy: Any, path: Path, *, action_limit: float) -> None:
    import torch

    # SB3's documented example omits action postprocessing. Bake the same clip
    # used by model.predict()/training into the ONE action output robotd loads.
    class ClippedActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            action, _, _ = self.policy(obs, deterministic=True)
            return torch.clamp(action, -action_limit, action_limit)

    actor = ClippedActor().eval()
    torch.onnx.export(
        actor,
        torch.zeros((1, 61), dtype=torch.float32),
        str(path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )


def _parameter_hash(policy: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(env: Any, predict: Any) -> tuple[list[dict], list[Any]]:
    import numpy as np

    episodes, observations = [], []
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=seed)
        total, steps, heights, tilts = 0.0, 0, [], []
        terminated = truncated = False
        while not (terminated or truncated):
            if steps % 10 == 0:
                observations.append(obs.copy())
            action = np.asarray(predict(obs), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            heights.append(float(info["trunk_z"]))
            tilts.append(float(info["tilt_rad"]))
        episodes.append(
            {
                "seed": seed,
                "steps": steps,
                "survival_s": steps * 0.02,
                "return": total,
                "fell": bool(terminated),
                "horizon_reached": bool(truncated),
                "min_trunk_z": min(heights),
                "max_tilt_rad": max(tilts),
            }
        )
    return episodes, observations


def _worker(upstream: Path, job: Path, spec: LocalTrainingSpec) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure

    from duckmind.local_env import ACTION_LIMIT, LocalBalanceEnv
    from duckmind.policy_check import inspect_policy

    report_path = job / "report.json"
    report = json.loads(report_path.read_text())
    env = evaluation = None
    try:
        packages = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
        if any(
            packages[name].split("+")[0] != version for name, version in REQUIRED_VERSIONS.items()
        ):
            raise ValueError(f"unreviewed local training dependency versions: {packages}")
        report["environment"] = {
            "packages": packages,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "torch_threads": spec.threads,
        }
        torch.set_num_threads(spec.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        env = LocalBalanceEnv(upstream, horizon_steps=spec.horizon_steps)
        evaluation = LocalBalanceEnv(upstream, horizon_steps=spec.horizon_steps)
        check_env(env, warn=True)
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            seed=spec.seed,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [64, 64], "log_std_init": -2.0},
            verbose=1,
        )
        model.set_logger(configure(str(job / "logs"), ["stdout", "csv"]))
        report["ppo"] = {
            "algorithm": "SB3 PPO",
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "net_arch": [64, 64],
            "log_std_init": -2.0,
        }
        report["initial_parameter_sha256"] = _parameter_hash(model.policy)
        model.save(job / "initial.zip")
        zero, _ = _evaluate(evaluation, lambda _: np.zeros(14, dtype=np.float32))
        initial, initial_obs = _evaluate(
            evaluation, lambda obs: model.predict(obs, deterministic=True)[0]
        )
        report.update(status="training", zero_action_baseline=zero, initial_policy=initial)
        _write(report_path, report)
        # Explicit reset removes check_env's random probe state from the training stream.
        env.reset(seed=spec.seed)
        model.learn(total_timesteps=spec.timesteps)
        model.save(job / "trained.zip")
        report["actual_timesteps"] = int(model.num_timesteps)
        report["trained_parameter_sha256"] = _parameter_hash(model.policy)
        report["weights_changed"] = (
            report["initial_parameter_sha256"] != report["trained_parameter_sha256"]
        )
        if not report["weights_changed"]:
            raise RuntimeError("training produced no parameter changes")
        # Validate the saved artifact, not only the still-live in-memory model.
        loaded = PPO.load(job / "trained.zip", device="cpu")
        if _parameter_hash(loaded.policy) != report["trained_parameter_sha256"]:
            raise RuntimeError("checkpoint reload changed model parameters")
        candidate, candidate_obs = _evaluate(
            evaluation, lambda obs: loaded.predict(obs, deterministic=True)[0]
        )
        report["trained_policy"] = candidate
        report["paired_comparison"] = compare_evaluations(initial, candidate)
        report["zero_action_comparison"] = compare_evaluations(zero, candidate)
        onnx = job / "policy.onnx"
        export_actor(loaded.policy, onnx, action_limit=ACTION_LIMIT)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx), sess_options=options, providers=["CPUExecutionProvider"]
        )
        max_error = 0.0
        for obs in initial_obs + candidate_obs:
            expected = loaded.predict(obs, deterministic=True)[0]
            actual = session.run(None, {"obs": obs.reshape(1, 61)})[0][0]
            max_error = max(max_error, parity_error(expected, actual))
        if max_error > 1e-5:
            raise RuntimeError(f"ONNX/PyTorch parity failed: {max_error}")
        report["export_parity"] = {
            "passed": True,
            "samples": len(initial_obs + candidate_obs),
            "max_abs_error": max_error,
            "tolerance": 1e-5,
            "normalization": "identity_no_running_normalizer",
            "baked_action_clip": ACTION_LIMIT,
        }
        # Evaluate the actual ONNX in physics as well as numerical samples.
        exported, _ = _evaluate(
            evaluation, lambda obs: session.run(None, {"obs": obs.reshape(1, 61)})[0][0]
        )
        report["onnx_policy"] = exported
        inspection = inspect_policy(onnx, seed=spec.seed, command_profile="zero").as_dict()
        _write(job / "compatibility.json", inspection)
        report["compatibility_gate"] = "pass" if inspection["passed"] else "failed"
        report["simulator_gate"] = "evaluated_native_pd_only"
        report["robotd_candidate_gate"] = "not_run"
        report["artifacts"] = {
            name: {"sha256": _sha(job / name), "bytes": (job / name).stat().st_size}
            for name in ("initial.zip", "trained.zip", "policy.onnx", "compatibility.json")
        }
        if (
            verify_upstream(upstream) != report["upstream"]
            or package_fingerprint() != report["duckmind_source_sha256"]
        ):
            raise RuntimeError("source changed during experiment")
        report["status"] = (
            "completed_experiment" if inspection["passed"] else "compatibility_failed"
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
        raise
    finally:
        cleanup_errors = close_environments((env, evaluation))
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)


def run_local_job(upstream: Path, job: Path, spec: LocalTrainingSpec) -> dict[str, Any]:
    """Own one bounded worker; never install/deploy/swap a robot policy."""
    upstream, job = upstream.resolve(), job.absolute()
    if job.exists() or job.is_symlink():
        raise ValueError("job directory exists; choose a new run, no implicit resume")
    if job.resolve().is_relative_to(upstream):
        raise ValueError("job must be outside the upstream checkout")
    source = verify_upstream(upstream)
    job.mkdir(parents=True, exist_ok=False)
    (job / "logs").mkdir()
    report = {
        "schema_version": 1,
        "profile": "experimental-native-pd-balance-ppo-v1",
        "status": "starting",
        "spec": asdict(spec),
        "upstream": source,
        "duckmind_source_sha256": package_fingerprint(),
        "started_at": datetime.now(UTC).isoformat(),
        "physical_hardware_gate": "not_run",
        "robotd_candidate_gate": "not_run",
        "promotion": "quarantined_no_deployment",
        "official_mjlab_training": False,
        "bam_actuator_training": False,
        "commands_trained": "zero_only_no_walking_or_head_tracking",
        "evaluation_seeds": list(EVAL_SEEDS),
    }
    _write(job / "report.json", report)
    command = [
        sys.executable,
        "-m",
        "duckmind.local_train",
        "--worker",
        "--upstream",
        str(upstream),
        "--job",
        str(job),
    ]
    for key, value in asdict(spec).items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    environment = child_environment(upstream)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).resolve().parents[1]), str(upstream / "src"))
    )
    environment["MPLCONFIGDIR"] = str(job / "plot-cache")
    started = time.monotonic()
    try:
        phase = run_phase(
            command, cwd=job, env=environment, log=job / "worker.log", timeout_s=spec.timeout_s
        )
        report = json.loads((job / "report.json").read_text())
        if phase.timed_out:
            report["status"] = "timed_out"
        elif phase.returncode or report.get("status") != "completed_experiment":
            report["status"] = "failed"
        report["worker"] = asdict(phase)
    except KeyboardInterrupt:
        report = json.loads((job / "report.json").read_text())
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
    finally:
        report["wall_elapsed_s"] = time.monotonic() - started
        _write(job / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--horizon-steps", type=int, default=250)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = LocalTrainingSpec(
        args.timesteps, args.seed, args.threads, args.timeout_s, args.horizon_steps
    )
    if args.worker:
        _worker(args.upstream, args.job, spec)
    else:
        result = run_local_job(args.upstream, args.job, spec)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if result["status"] != "completed_experiment":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
