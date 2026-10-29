"""Isolated candidate-vs-v5 robotd/MuJoCo diagnostic; never deploys a policy.

The official `robotd --sim` path changes RobotIo below the real control loop,
while the body server preserves the BAM model and exposes validation-only
`trunk_z` and `sim_time`. A report of four short runs can reject an observed
failure, never qualify a gait or physical hardware.

Sources (reviewed revisions):
https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/robot/simulation.md
https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/policy-manifest.md
https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from duckmind.body import BodyAck, BodyIntent, BodyPort
from duckmind.robotd import RobotdBodyPort, RobotdOperatorClient
from duckmind.sim_live import (
    POLICY_FILES,
    SimOracle,
    _free_local_port,
    _git_commit,
    _installed_ort_library,
    _policy_config,
    _port_open,
    _socket_open,
    _stop_child,
    _wait_ready,
)
from duckmind.tmpdir import short_tmp
from duckmind.training import UPSTREAM_COMMIT, verify_upstream
from duckmind.versions import package_fingerprint

MICRODUCK_COMMIT = "a9ec4b2079ef8ee7904014089c885bb07d57d63c"
SCENARIOS = ("idle", "forward_stop")
MIN_REALTIME_FACTOR = 0.95


class FixedForwardProbeSkill:
    """Validation-only fixed semantic walk/stop skill; cognition never gets vx."""

    def __init__(self, body: BodyPort) -> None:
        self._body = body

    def forward_tick(self) -> BodyAck:
        ack = self._body.dispatch_intent(
            BodyIntent("robot.move", {"vx": 0.15, "vy": 0.0, "vyaw": 0.0})
        )
        if not ack.accepted:
            raise RuntimeError("robotd refused fixed forward probe")
        return ack

    def stop(self) -> BodyAck:
        ack = self._body.stop()
        if not ack.accepted:
            raise RuntimeError("robotd refused probe stop")
        return ack


def _child_env(**overrides: str) -> dict[str, str]:
    environment = dict(os.environ)
    for secret in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(secret, None)
    environment.update(overrides)
    return environment


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_candidate_job(job: Path) -> dict[str, Any]:
    """Accept only a source-bound local candidate for *diagnostic* simulator use."""
    if job.is_symlink() or not job.is_dir():
        raise ValueError("training job must be a real local directory")
    report_path = job / "job.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("training job report missing or linked")
    report = json.loads(report_path.read_text())
    try:
        status = report["status"]
        expected_verdict = {
            "candidate_rejected": "rejected",
            "candidate_unqualified": "inconclusive",
        }[status]
        if (
            report["compatibility_gate"] != "pass"
            or report["qualification"]["verdict"] != expected_verdict
            or report["simulator_gate"]
            != ("failed" if status == "candidate_rejected" else "inconclusive")
            or report["physical_hardware_gate"] != "not_run"
            or report["upstream_identity"]["commit"] != UPSTREAM_COMMIT
        ):
            raise ValueError("training job gates are inconsistent")
        record = report["artifacts"]["onnx"]
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("candidate model path escapes training job")
        model = job / relative
        if (
            model.is_symlink()
            or not model.is_file()
            or not model.resolve().is_relative_to(job.resolve())
        ):
            raise ValueError("candidate ONNX missing or linked")
        model_sha256 = _sha(model)
        if model_sha256 != record["sha256"]:
            raise ValueError("candidate ONNX bytes differ from training report")
        return {
            "job": job.resolve(),
            "model": model.resolve(),
            "model_sha256": model_sha256,
            "job_report_sha256": _sha(report_path),
            "quality_status": status,
            "training_source_sha256": report["duckmind_source_sha256"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete training job evidence") from error


def make_walk_config(policies: Path, walk_model: Path) -> str:
    """Override only the walking slot in a fresh simulator params file.

    The installed set is read-only. `robot.subscribe` publishes a basename,
    not its resolved path, so the candidate copy also gets a unique hash name.
    """
    return _policy_config(policies, walk_override=walk_model)


def assess_robotd_pair(runs: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on invalid timing/pairing and never return an approval."""
    reasons: list[str] = []
    try:
        for scenario in SCENARIOS:
            baseline = runs["baseline"][scenario]
            candidate = runs["candidate"][scenario]
            left, right = baseline["initial_pose_sha256"], candidate["initial_pose_sha256"]
            if (
                not isinstance(left, str)
                or re.fullmatch(r"[0-9a-f]{64}", left) is None
                or left != right
            ):
                reasons.append(f"unpaired_initial_pose:{scenario}")
            for label, item in (("baseline", baseline), ("candidate", candidate)):
                rate = item["sim_realtime_factor"]
                if (
                    item["status"] != "completed"
                    or type(item["samples"]) is not int
                    or item["samples"] < 1
                    or type(rate) not in (int, float)
                    or not math.isfinite(rate)
                    or rate < MIN_REALTIME_FACTOR
                    or type(item["fallen"]) is not bool
                    or type(item["limp"]) is not bool
                    or type(item["walk_reached"]) is not bool
                    or type(item["min_trunk_z_m"]) not in (int, float)
                    or not math.isfinite(item["min_trunk_z_m"])
                    or (
                        scenario == "forward_stop"
                        and (
                            type(item["command_applied"]) is not bool
                            or type(item["stop_confirmed"]) is not bool
                        )
                    )
                ):
                    reasons.append(f"invalid_run:{label}:{scenario}")
    except (KeyError, TypeError, AttributeError):
        reasons.append("incomplete_robotd_pair")
    if reasons:
        return {
            "verdict": "invalid",
            "reasons": reasons,
            "physical_hardware_gate": "not_run",
        }

    baseline_runs = runs["baseline"]
    for scenario in SCENARIOS:
        item = baseline_runs[scenario]
        if (
            not item["walk_reached"]
            or item["fallen"]
            or item["limp"]
            or item["min_trunk_z_m"] < 0.05
        ):
            reasons.append(f"baseline_unstable:{scenario}")
    baseline_forward = baseline_runs["forward_stop"]
    if not baseline_forward["command_applied"] or not baseline_forward["stop_confirmed"]:
        reasons.append("baseline_command_path_unverified")
    if reasons:
        return {
            "verdict": "invalid",
            "reasons": reasons,
            "physical_hardware_gate": "not_run",
        }

    candidate_runs = runs["candidate"]
    for scenario in SCENARIOS:
        item = candidate_runs[scenario]
        if not item["walk_reached"]:
            reasons.append(f"candidate_walk_not_reached:{scenario}")
        if item["fallen"] or item["limp"]:
            reasons.append(f"candidate_fell:{scenario}")
        if item["min_trunk_z_m"] < 0.05:
            reasons.append(f"candidate_low_trunk:{scenario}")
    candidate_forward = candidate_runs["forward_stop"]
    if candidate_forward["walk_reached"]:
        for name in ("command_applied", "stop_confirmed"):
            if not candidate_forward[name]:
                reasons.append(f"candidate_{name}_failed")
    if reasons:
        verdict = "rejected"
    else:
        verdict = "inconclusive"
        reasons.append("coverage_too_small")
    return {"verdict": verdict, "reasons": reasons, "physical_hardware_gate": "not_run"}


def _pose_digest(snapshot: dict[str, Any]) -> str:
    pose = {"positions": snapshot["positions"], "trunk": snapshot["trunk"]}
    return hashlib.sha256(json.dumps(pose, sort_keys=True).encode()).hexdigest()


def _capture_first_walk_truth(
    result: dict[str, Any], observation: Any, read_truth: Callable[[], dict[str, Any]]
) -> None:
    """One validation-only MuJoCo read at first observed walk, never sent to cognition."""
    if observation.policy != "walk" or result.get("first_walk_truth") is not None:
        return
    truth = read_truth()
    imu = truth.get("imu")
    gravity = imu.get("gravity") if isinstance(imu, dict) else None
    numbers = (truth.get("sim_time"), truth.get("trunk_z"), observation.robot_time_s)

    def finite_scalar(value: object) -> bool:
        return (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )

    if (
        any(not finite_scalar(value) for value in numbers)
        or not isinstance(gravity, list)
        or len(gravity) != 3
        or any(not finite_scalar(value) for value in gravity)
    ):
        raise ValueError("first walk oracle has invalid ground truth")
    result["first_walk_truth"] = {
        "robot_time_s": float(observation.robot_time_s),
        "sim_time_s": float(truth["sim_time"]),
        "trunk_z_m": float(truth["trunk_z"]),
        "projected_gravity": [float(value) for value in gravity],
        "safety_fallen": observation.safety.fallen,
    }


def _run_one(
    *,
    label: str,
    scenario: str,
    state: Path,
    walk_model: Path,
    expected_name: str,
    microduck: Path,
    microduck_rl: Path,
    policies: Path,
    robotd_binary: Path,
    ort_lib: Path,
) -> dict[str, Any]:
    state.mkdir(parents=True, exist_ok=False)
    config = state / "robotd.toml"
    config.write_text(make_walk_config(policies, walk_model))
    socket_path = state / "robotd.sock"
    port = _free_local_port()
    result: dict[str, Any] = {
        "status": "failed",
        "label": label,
        "scenario": scenario,
        "config_sha256": _sha(config),
        "walk_model_sha256": _sha(walk_model),
        "expected_walk_name": expected_name,
        "physical_hardware_gate": "not_run",
        "samples": 0,
        "command_applied": False,
        "stop_confirmed": False,
        "first_walk_truth": None,
        "ready_states": [],
        "trace": [],
    }
    body_process = robot_process = None
    with (state / "body.log").open("w") as body_log, (state / "robotd.log").open("w") as robot_log:
        try:
            body_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mjlab_microduck.sim.body_server",
                    "--headless",
                    "--keyframe",
                    "HOME",
                    "--port",
                    str(port),
                ],
                cwd=microduck_rl,
                env=_child_env(PYTHONPATH=str(microduck_rl / "src")),
                stdin=subprocess.DEVNULL,
                stdout=body_log,
                stderr=subprocess.STDOUT,
            )
            _wait_ready(body_process, lambda: _port_open(port), deadline_s=30, label="duck-body")
            robot_process = subprocess.Popen(
                [
                    str(robotd_binary),
                    "--sim",
                    f"127.0.0.1:{port}",
                    "--params",
                    str(config),
                    "--socket",
                    str(socket_path),
                ],
                cwd=microduck,
                env=_child_env(
                    ORT_DYLIB_PATH=str(ort_lib),
                    DUCK_RUNTIME_DIR=str(state),
                    RUST_LOG="info",
                ),
                stdin=subprocess.DEVNULL,
                stdout=robot_log,
                stderr=subprocess.STDOUT,
            )
            _wait_ready(
                robot_process, lambda: _socket_open(socket_path), deadline_s=30, label="robotd"
            )
            with (
                RobotdBodyPort(socket_path, backend="simulator", hz=25) as body,
                RobotdOperatorClient(socket_path) as operator,
                SimOracle(port) as oracle,
            ):
                capabilities = body.capabilities
                result["capabilities"] = capabilities
                if capabilities.get("walk") != expected_name or capabilities.get("unavailable"):
                    raise RuntimeError("robotd did not resolve the isolated walking policy")
                initial = oracle.snapshot()
                result["initial_pose_sha256"] = _pose_digest(initial)
                result["initial_trunk_z_m"] = float(initial["trunk_z"])
                sim_started = float(initial["sim_time"])
                sim_wall_started = time.monotonic()
                result["min_trunk_z_m"] = float(initial["trunk_z"])
                result["fallen"] = False
                result["limp"] = False
                result["ready_fallen"] = False
                result["ready_limp"] = False
                result["limited_by"] = []
                motion = FixedForwardProbeSkill(body)
                ack = operator.enable(True)
                result["enable_ack"] = {"accepted": ack.accepted, "reason": ack.reason}
                if not ack.accepted:
                    raise RuntimeError("simulator operator enable refused")
                try:
                    # The daemon is the command owner; the oracle stays solely
                    # in this validation process, never in cognition.
                    deadline = time.monotonic() + 12.0
                    walking = False
                    while time.monotonic() < deadline:
                        observation = body.observe()
                        # An extra MuJoCo TCP read on every homing tick
                        # perturbed the baseline enough to make it fall.
                        # Keep the readiness lane observationally light, as
                        # the established sim_live gate does.
                        result["samples"] += 1
                        result["ready_fallen"] = result["ready_fallen"] or observation.safety.fallen
                        result["ready_limp"] = result["ready_limp"] or observation.safety.limp
                        result["ready_states"].append(
                            {
                                "policy": observation.policy,
                                "fallen": observation.safety.fallen,
                                "limp": observation.safety.limp,
                                "robot_time_s": observation.robot_time_s,
                            }
                        )
                        _capture_first_walk_truth(result, observation, oracle.snapshot)
                        if observation.policy == "walk" and not observation.safety.fallen:
                            walking = True
                            break
                    result["walk_reached"] = walking
                    if walking:
                        phases = (
                            (("idle", 3.0),)
                            if scenario == "idle"
                            else (
                                ("forward", 2.0),
                                ("settle", 1.5),
                            )
                        )
                        for phase, duration in phases:
                            phase_deadline = time.monotonic() + duration
                            next_move = 0.0
                            if phase == "settle":
                                motion.stop()
                            while time.monotonic() < phase_deadline:
                                if phase == "forward" and time.monotonic() >= next_move:
                                    motion.forward_tick()
                                    next_move = time.monotonic() + 0.10
                                observation = body.observe()
                                truth = oracle.snapshot()
                                _record(result, observation, truth, phase=phase)
                                if observation.safety.fallen or observation.safety.limp:
                                    break
                            if result["fallen"] or result["limp"]:
                                break
                    forward = [item for item in result["trace"] if item["phase"] == "forward"]
                    settled = [item for item in result["trace"] if item["phase"] == "settle"]
                    result["command_applied"] = any(
                        abs(item["move_applied"][0]) >= 0.05 for item in forward
                    )
                    result["stop_confirmed"] = len(settled) >= 2 and all(
                        max(abs(value) for value in item["move_applied"]) <= 0.02
                        for item in settled[-2:]
                    )
                    motion.stop()
                    final = oracle.snapshot()
                    result["min_trunk_z_m"] = min(result["min_trunk_z_m"], float(final["trunk_z"]))
                    result["final_trunk_z_m"] = float(final["trunk_z"])
                    result["forward_displacement_m"] = float(final["trunk"][0]) - float(
                        initial["trunk"][0]
                    )
                    result["sim_realtime_factor"] = (float(final["sim_time"]) - sim_started) / max(
                        time.monotonic() - sim_wall_started, 1e-6
                    )
                    result["status"] = "completed"
                finally:
                    if not operator.enable(False).accepted:
                        raise RuntimeError("simulator operator disable refused")
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"[:500]
        finally:
            _stop_child(robot_process)
            _stop_child(body_process)
    if result["status"] != "completed":
        for name in ("robotd", "body"):
            log = state / f"{name}.log"
            result[f"{name}_log_tail"] = log.read_text(errors="replace")[-1500:]
    return result


def _record(
    result: dict[str, Any], observation: Any, truth: dict[str, Any], *, phase: str = "ready"
) -> None:
    result["samples"] += 1
    result["fallen"] = result["fallen"] or observation.safety.fallen
    result["limp"] = result["limp"] or observation.safety.limp
    result["min_trunk_z_m"] = min(result["min_trunk_z_m"], float(truth["trunk_z"]))
    result["limited_by"] = sorted(set(result["limited_by"]) | set(observation.limited_by))
    result["trace"].append(
        {
            "phase": phase,
            "wall_monotonic_s": observation.monotonic_s,
            "robot_time_s": observation.robot_time_s,
            "sim_time_s": truth["sim_time"],
            "policy": observation.policy,
            "fallen": observation.safety.fallen,
            "limp": observation.safety.limp,
            "trunk_z_m": truth["trunk_z"],
            "move_requested": observation.movement_requested,
            "move_applied": observation.movement_applied,
            "limited_by": observation.limited_by,
        }
    )


def run_robotd_comparison(
    *,
    candidate_job: Path,
    microduck: Path,
    microduck_rl: Path,
    policies: Path,
    robotd_binary: Path,
    ort_lib: Path,
    report_path: Path,
) -> dict[str, Any]:
    if report_path.exists() or report_path.is_symlink():
        raise ValueError("comparison report already exists; never overwrite")
    identity = verify_candidate_job(candidate_job)
    if _git_commit(microduck) != MICRODUCK_COMMIT or _git_commit(microduck_rl) != UPSTREAM_COMMIT:
        raise ValueError("official source revision differs from reviewed robotd comparison")
    rl_source = verify_upstream(microduck_rl)
    for name, path in (
        ("robotd", robotd_binary),
        ("ORT", ort_lib),
        ("manifest", policies / "manifest.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} missing: {path}")
    manifest = json.loads((policies / "manifest.json").read_text())
    if manifest.get("obs_len") != 61 or manifest.get("action_len") != 14:
        raise ValueError("official policy set observation/action contract differs")
    baseline = policies / "velstand.onnx"
    _policy_config(policies)  # verify all configured v5 policy files before launch
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "duckmind_source_sha256": package_fingerprint(),
        "training_job": str(identity["job"]),
        "training_report_sha256": identity["job_report_sha256"],
        "training_quality_status": identity["quality_status"],
        "microduck_commit": MICRODUCK_COMMIT,
        "microduck_rl_commit": UPSTREAM_COMMIT,
        "microduck_rl_source_sha256": rl_source["source_sha256"],
        "robotd_binary_sha256": _sha(robotd_binary),
        "ort_lib_sha256": _sha(ort_lib),
        "policy_manifest_sha256": _sha(policies / "manifest.json"),
        "baseline_onnx_sha256": _sha(baseline),
        "official_policy_sha256": {name: _sha(policies / name) for name in POLICY_FILES},
        "candidate_onnx_sha256": identity["model_sha256"],
        "scene_sha256": _sha(microduck_rl / "src/mjlab_microduck/robot/microduck/scene.xml"),
        "body_server_sha256": _sha(microduck_rl / "src/mjlab_microduck/sim/body_server.py"),
        "scenarios": list(SCENARIOS),
        "runs": {"baseline": {}, "candidate": {}},
        "physical_hardware_gate": "not_run",
        "policy_deployment": "not_run",
        "started_at": datetime.now(UTC).isoformat(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    try:
        with tempfile.TemporaryDirectory(prefix="dm-pair-", dir=short_tmp()) as name:
            root = Path(name)
            copied_candidate = root / f"dm-candidate-{identity['model_sha256'][:12]}.onnx"
            shutil.copy2(identity["model"], copied_candidate)
            if _sha(copied_candidate) != identity["model_sha256"]:
                raise ValueError("candidate copy differs before robotd launch")
            for scenario in SCENARIOS:
                for label, model in (("baseline", baseline), ("candidate", copied_candidate)):
                    report["runs"][label][scenario] = _run_one(
                        label=label,
                        scenario=scenario,
                        state=root / f"{label}-{scenario}",
                        walk_model=model,
                        expected_name=model.name,
                        microduck=microduck,
                        microduck_rl=microduck_rl,
                        policies=policies,
                        robotd_binary=robotd_binary,
                        ort_lib=ort_lib,
                    )
                    _write_json(report_path, report)
            if _sha(copied_candidate) != identity["model_sha256"]:
                raise ValueError("candidate copy changed during robotd comparison")
        if (
            _sha(identity["model"]) != identity["model_sha256"]
            or _sha(candidate_job / "job.json") != identity["job_report_sha256"]
            or _sha(policies / "manifest.json") != report["policy_manifest_sha256"]
            or _sha(baseline) != report["baseline_onnx_sha256"]
            or any(
                _sha(policies / name) != digest
                for name, digest in report["official_policy_sha256"].items()
            )
            or _sha(robotd_binary) != report["robotd_binary_sha256"]
            or _sha(ort_lib) != report["ort_lib_sha256"]
            or verify_upstream(microduck_rl) != rl_source
            or package_fingerprint() != report["duckmind_source_sha256"]
        ):
            raise ValueError("comparison source, binary or model changed during run")
        report["assessment"] = assess_robotd_pair(report["runs"])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-job", type=Path, required=True)
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--microduck-rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--robotd-bin", type=Path, required=True)
    parser.add_argument("--ort-lib", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_robotd_comparison(
        candidate_job=args.candidate_job.absolute(),
        microduck=args.microduck.resolve(),
        microduck_rl=args.microduck_rl.resolve(),
        policies=args.policies.resolve(),
        robotd_binary=args.robotd_bin.resolve(),
        ort_lib=args.ort_lib.resolve() if args.ort_lib else _installed_ort_library(),
        report_path=args.report.absolute(),
    )
    print(json.dumps({"status": report["status"], "assessment": report["assessment"]}, indent=2))
    # Completion of four short scenarios is not promotion. A clean pair is
    # inconclusive and still exits nonzero for automation that only sees $?.
    raise SystemExit(
        {"rejected": 1, "invalid": 2, "inconclusive": 3}[report["assessment"]["verdict"]]
    )


if __name__ == "__main__":
    main()
