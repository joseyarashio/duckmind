"""Bounded, off-board PPO -> ONNX -> compatibility -> physics probe jobs.

No cloud submission, package installation, policy publication or robot controls.
Recipe: microduck_rl cb70b792312d559a4da09064d92009079671815f and mjlab 1.3.0.
Explicit CPU execution is experimental, not a CUDA-equivalent quality claim.
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/train.py
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/utils/gpu.py
https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/export.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from duckmind.policy_check import CONTRACT_COMMIT
from duckmind.qualification import assess_official_probe
from duckmind.versions import package_fingerprint

__all__ = ["TrainingSpec", "make_plan", "probe_environment", "run_job"]

UPSTREAM_COMMIT = "cb70b792312d559a4da09064d92009079671815f"
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
VERSIONS = {
    "mjlab": "1.3.0",
    "warp-lang": "1.12.0",
    "torch": "2.9.1",
    "rsl-rl-lib": "5.0.1",
    "mjlab-microduck": "0.1.0",
}

ENVIRONMENT_PROBE = """
import importlib.metadata as metadata
import importlib.util
import json
import sys
result = {"python": list(sys.version_info[:3]), "platform": sys.platform,
          "packages": {}, "cuda_available": False, "cuda_devices": 0,
          "cpu_available": False, "requested_device": sys.argv[1], "module_path": None}
for name in ("mjlab", "warp-lang", "torch", "rsl-rl-lib", "mjlab-microduck"):
    try:
        result["packages"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result["packages"][name] = None
spec = importlib.util.find_spec("mjlab_microduck")
result["module_path"] = spec.origin if spec else None
result["task_entrypoint"] = any(
    entry.name == "mjlab_microduck" and entry.value == "mjlab_microduck.tasks"
    for entry in metadata.entry_points(group="mjlab.tasks")
)
if result["packages"]["torch"]:
    try:
        import torch
        result["cpu_available"] = bool(
            torch.isfinite(torch.tensor([1.0], device="cpu") + 1.0).all().item()
        )
        if sys.argv[1] == "cuda":
            result["cuda_available"] = torch.cuda.is_available()
            result["cuda_devices"] = torch.cuda.device_count()
        result["torch_cuda"] = torch.version.cuda
    except Exception as error:
        result["torch_error"] = type(error).__name__
print(json.dumps(result))
"""


@dataclass(frozen=True)
class TrainingSpec:
    num_envs: int = 64
    max_iterations: int = 5
    seed: int = 0
    timeout_s: int = 600
    device: str = "cuda"
    push_interval_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("num_envs", 1, 16384),
            ("max_iterations", 1, 100000),
            ("seed", 0, 2**31 - 1),
            ("timeout_s", 1, 86400),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        _check_device(self.device)
        interval = self.push_interval_s
        if interval is not None and (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or any(
                type(value) not in (int, float) or not 0.1 <= value <= 60.0 for value in interval
            )
            or interval[0] > interval[1]
        ):
            raise ValueError("push_interval_s must be a finite ascending pair in [0.1, 60.0]")


def _check_device(device: str) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")


@dataclass(frozen=True)
class PhaseResult:
    returncode: int
    timed_out: bool
    elapsed_s: float


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_upstream(upstream: Path) -> dict[str, str]:
    """Require the reviewed revision and record its actual checked-out bytes."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=upstream, text=True).strip()

    commit = git("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError("microduck_rl revision differs from reviewed training profile")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("microduck_rl worktree is dirty; use a clean reviewed checkout")
    files = git("ls-files", "-z").split("\0")
    hashes = {}
    for name in files:
        if not name:
            continue
        path = upstream / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsupported upstream source entry: {name}")
        hashes[name] = _sha(path)
    source_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "commit": commit,
        "source_sha256": source_hash,
        "uv_lock_sha256": _sha(upstream / "uv.lock"),
    }


def child_environment(upstream: Path, *, device: str = "cuda") -> dict[str, str]:
    _check_device(device)
    environment = dict(os.environ)
    # One GPU or explicit CPU, never join a parent torchrun rendezvous.
    # Preserve the operator's GPU boundary only for the CUDA profile.
    removed = {
        "WANDB_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PYTHONHOME",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    }
    for key in tuple(environment):
        if key in removed or key.startswith("TORCHELASTIC_"):
            environment.pop(key)
    environment.update(
        PYTHONPATH=str(upstream / "src"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    if device == "cpu":
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
    return environment


def environment_reasons(
    report: dict[str, Any], upstream: Path, *, device: str = "cuda"
) -> list[str]:
    _check_device(device)
    reasons = []
    if report.get("python", [])[:2] != [3, 12]:
        reasons.append("python_3_12_required")
    if device == "cpu":
        if report.get("platform") not in ("darwin", "linux"):
            reasons.append("macos_linux_cpu_profile_required")
        if report.get("cpu_available") is not True:
            reasons.append("cpu_unavailable")
    else:
        if report.get("platform") != "linux":
            reasons.append("linux_cuda_profile_required")
        if report.get("cuda_available") is not True or report.get("cuda_devices", 0) < 1:
            reasons.append("cuda_unavailable")
    for name, version in VERSIONS.items():
        installed = report.get("packages", {}).get(name)
        if not isinstance(installed, str) or installed.split("+")[0] != version:
            reasons.append(f"version_mismatch:{name}")
    if report.get("task_entrypoint") is not True:
        reasons.append("missing_task_entrypoint")
    expected = (upstream / "src/mjlab_microduck/__init__.py").resolve()
    if (
        not isinstance(report.get("module_path"), str)
        or Path(report["module_path"]).resolve() != expected
    ):
        reasons.append("wrong_microduck_import")
    return reasons


def probe_environment(
    python: Path, upstream: Path, *, timeout_s: float = 60, device: str = "cuda"
) -> dict[str, Any]:
    _check_device(device)
    command = [str(python), "-c", ENVIRONMENT_PROBE, device]
    # File-backed channels avoid pipe deadlocks and keep diagnostics out of JSON.
    # The probe shares the phase cancellation guard.
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        result = _run_process(
            command,
            cwd=upstream,
            env=child_environment(upstream, device=device),
            stdout=stdout,
            stderr=stderr,
            timeout_s=timeout_s,
        )
        stdout.seek(0)
        stderr.seek(0)
        output, errors = stdout.read(), stderr.read()
    if result.timed_out:
        raise subprocess.TimeoutExpired(command, timeout_s, output=output, stderr=errors)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=output, stderr=errors
        )
    report = json.loads(output)
    if not isinstance(report, dict):
        raise ValueError("invalid environment probe response")
    return report


def _resume_source(
    parent_job: Path, upstream: Path, upstream_identity: dict[str, str], spec: TrainingSpec
) -> dict[str, Any]:
    """Inspect a locally produced job before its pickle-backed checkpoint is loaded."""

    parent_job = parent_job.absolute()
    if (
        parent_job.is_symlink()
        or not parent_job.is_dir()
        or parent_job.resolve().is_relative_to(upstream)
    ):
        raise ValueError("resume parent must be a real local job outside upstream")
    report_path = parent_job / "job.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("resume parent job report is missing or symlinked")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("resume parent job report is invalid") from error
    if not isinstance(report, dict):
        raise ValueError("resume parent job report must be an object")
    status = report.get("status")
    expected_gate = (
        {
            "candidate_rejected": ("failed", "rejected"),
            "candidate_unqualified": ("inconclusive", "inconclusive"),
        }.get(status)
        if isinstance(status, str)
        else None
    )
    qualification = report.get("qualification")
    if (
        expected_gate is None
        or report.get("simulator_gate") != expected_gate[0]
        or not isinstance(qualification, dict)
        or qualification.get("verdict") != expected_gate[1]
        or qualification.get("physical_hardware_gate") != "not_run"
        or report.get("task_id") != TASK_ID
        or report.get("training_gate") != "checkpoint_produced"
        or report.get("export_gate") != "exported"
        or report.get("compatibility_gate") != "pass"
        or report.get("upstream_identity") != upstream_identity
        or not isinstance(report.get("spec"), dict)
        or any(
            report["spec"].get(key) != getattr(spec, key) for key in ("device", "num_envs", "seed")
        )
        or not isinstance(report.get("duckmind_source_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", report["duckmind_source_sha256"])
    ):
        raise ValueError("resume parent is not a compatible completed official job")
    artifacts = report.get("artifacts")
    record = artifacts.get("checkpoint") if isinstance(artifacts, dict) else None
    if not isinstance(record, dict):
        raise ValueError("resume parent checkpoint record is missing")
    name = record.get("path")
    if not isinstance(name, str):
        raise ValueError("resume checkpoint path is invalid")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or relative.as_posix() != name
        or len(relative.parts) != 3
        or relative.parts[0] != "checkpoints"
        or ".." in relative.parts
        or not re.fullmatch(r"model_\d+\.pt", relative.name)
    ):
        raise ValueError("resume checkpoint must be inside a local official run")
    checkpoint = parent_job / name
    if checkpoint.parent.is_symlink() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("resume checkpoint is missing or symlinked")
    if (
        not checkpoint.resolve().is_relative_to(parent_job.resolve())
        or record.get("sha256") != _sha(checkpoint)
        or type(record.get("bytes")) is not int
        or record["bytes"] != checkpoint.stat().st_size
        or record["bytes"] <= 0
    ):
        raise ValueError("resume checkpoint differs from its parent job record")
    return {
        "parent_job": str(parent_job),
        "parent_report_sha256": _sha(report_path),
        "parent_source_sha256": report["duckmind_source_sha256"],
        "parent_status": report["status"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": record["sha256"],
        "checkpoint_bytes": record["bytes"],
        "checkpoint_index": int(relative.stem.split("_")[-1]),
    }


def make_plan(
    upstream: Path,
    python: Path,
    job: Path,
    spec: TrainingSpec,
    *,
    resume_job: Path | None = None,
) -> dict[str, Any]:
    upstream, job = upstream.resolve(), job.resolve()
    # Preserve a venv interpreter's symlink path: resolving it loses its venv.
    python = python.absolute()
    if job.is_relative_to(upstream):
        raise ValueError("job outputs must be outside the upstream checkout")
    if resume_job is not None and job.is_relative_to(resume_job.resolve()):
        raise ValueError("resume output must be outside its parent job")
    resume = (
        _resume_source(resume_job, upstream, verify_upstream(upstream), spec)
        if resume_job is not None
        else None
    )
    command = [
        str(python),
        "-m",
        "mjlab.scripts.train",
        TASK_ID,
        "--gpu-ids",
        # Official mjlab 1.3.0 TrainConfig.gpu_ids=None selects CPU directly.
        "None" if spec.device == "cpu" else "[0]",
        "--env.scene.num-envs",
        str(spec.num_envs),
        "--agent.max-iterations",
        str(spec.max_iterations),
        "--agent.seed",
        str(spec.seed),
        "--agent.logger",
        "tensorboard",
        "--agent.upload-model",
        "False",
        "--agent.resume",
        "True" if resume is not None else "False",
        "--agent.experiment-name",
        str(job / "checkpoints"),
        "--agent.run-name",
        "duckmind",
    ]
    if resume is not None:
        filename = Path(resume["checkpoint_path"]).name
        command.extend(
            [
                "--agent.load-run",
                "^resume-source$",
                "--agent.load-checkpoint",
                f"^{re.escape(filename)}$",
            ]
        )
    if spec.push_interval_s is not None:
        command.extend(
            [
                "--env.events.push-robot.interval-range-s",
                f"({spec.push_interval_s[0]}, {spec.push_interval_s[1]})",
            ]
        )
    plan = {
        "schema_version": 1,
        "profile": (
            f"official-offline-velocity-{spec.device}-push-interval-experimental-v1"
            if spec.push_interval_s is not None
            else (
                "official-offline-velocity-cpu-experimental-v1"
                if spec.device == "cpu"
                else "official-offline-velocity-v1"
            )
        ),
        "experimental": spec.device == "cpu" or spec.push_interval_s is not None,
        "cpu_thread_limit": 2 if spec.device == "cpu" else None,
        "cuda_quality_equivalence": "not_claimed",
        "task_id": TASK_ID,
        "upstream": str(upstream),
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "python": str(python),
        "job_directory": str(job),
        "spec": asdict(spec),
        "train_command": command,
        "logger": "tensorboard",
        "cloud_submission": False,
        "physical_hardware_gate": "not_run",
    }
    if resume is not None:
        plan["resume_source"] = resume
    return plan


def _stop_process(process: subprocess.Popen) -> None:
    # Own group created by start_new_session; a dead/reaped leader does not
    # imply that its workers exited. Never enumerate or target unrelated PIDs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=3)
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader even while descendants are exiting.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait(timeout=3)
            return
        except PermissionError:
            # A restricted host can deny a zero-signal probe while the group
            # is exiting. This is not proof of absence: retain the deadline
            # and the actual SIGKILL fallback (whose errors are not hidden).
            pass
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    # Escalate for the group, not only when waiting on its leader times out.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: TextIO,
    stderr: TextIO | int,
    timeout_s: float,
) -> PhaseResult:
    started = time.monotonic()
    if timeout_s <= 0:
        return PhaseResult(-1, True, 0.0)
    manages_signals = threading.current_thread() is threading.main_thread()
    previous_term = signal.getsignal(signal.SIGTERM) if manages_signals else None
    previous_int = signal.getsignal(signal.SIGINT) if manages_signals else None
    process = None
    timed_out = False
    cleanup_signal = None

    def terminate_job(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"training interrupted by signal {signum}")

    def defer_interrupt(signum: int, _frame: object) -> None:
        nonlocal cleanup_signal
        cleanup_signal = signum

    try:
        if manages_signals:
            signal.signal(signal.SIGTERM, terminate_job)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        # TERM/INT must not interrupt cleanup or silently turn cancellation
        # into success if the signal arrives after the leader already exited.
        if manages_signals:
            signal.signal(signal.SIGTERM, defer_interrupt)
            signal.signal(signal.SIGINT, defer_interrupt)
        try:
            if process is not None:
                _stop_process(process)
        finally:
            if manages_signals:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
        if cleanup_signal is not None:
            raise KeyboardInterrupt(f"training interrupted by signal {cleanup_signal}")
    return PhaseResult(process.returncode, timed_out, time.monotonic() - started)


def run_phase(
    command: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout_s: float
) -> PhaseResult:
    with log.open("x") as stream:
        return _run_process(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout_s=timeout_s,
        )


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes job directory")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def run_job(
    upstream: Path,
    python: Path,
    job: Path,
    spec: TrainingSpec,
    *,
    execute: bool = False,
    resume_job: Path | None = None,
) -> dict[str, Any]:
    if not execute:
        raise ValueError("training requires explicit --execute consent")
    if job.exists() or job.is_symlink():
        raise ValueError("job directory already exists; never overwrite or silently resume")
    plan = make_plan(upstream, python, job, spec, resume_job=resume_job)
    upstream, job, python = (
        Path(plan["upstream"]),
        Path(plan["job_directory"]),
        Path(plan["python"]),
    )
    source = verify_upstream(upstream)
    job.mkdir(parents=True, exist_ok=False)
    (job / "artifacts").mkdir()
    (job / "logs").mkdir()
    report: dict[str, Any] = {
        **plan,
        "started_at": datetime.now(UTC).isoformat(),
        "duckmind_source_sha256": package_fingerprint(),
        "upstream_identity": source,
        "status": "preflight",
        "training_gate": "not_run",
        "export_gate": "not_run",
        "compatibility_gate": "not_run",
        "simulator_gate": "not_run",
        "normalizer_provenance": "unverified",
        "phases": [],
        "artifacts": {},
    }
    report_path = job / "job.json"
    _write_report(report_path, report)
    deadline = time.monotonic() + spec.timeout_s
    resume = report.get("resume_source")

    def verify_resume_input() -> None:
        if resume is None:
            return
        staged = job / resume["staged_checkpoint"]["path"]
        if (
            _artifact(staged, job) != resume["staged_checkpoint"]
            or _sha(Path(resume["checkpoint_path"])) != resume["checkpoint_sha256"]
            or _sha(Path(resume["parent_job"]) / "job.json") != resume["parent_report_sha256"]
        ):
            raise ValueError("resume source changed during training job")

    def phase(name: str, command: list[str]) -> bool:
        verify_resume_input()
        report["status"] = f"{name}_running"
        _write_report(report_path, report)
        result = run_phase(
            command,
            cwd=upstream,
            env=child_environment(upstream, device=spec.device),
            log=job / "logs" / f"{name}.log",
            timeout_s=deadline - time.monotonic(),
        )
        report["phases"].append({"name": name, "command": command, **asdict(result)})
        if (
            verify_upstream(upstream) != source
            or package_fingerprint() != report["duckmind_source_sha256"]
        ):
            raise ValueError("source changed during training job")
        verify_resume_input()
        return result.returncode == 0 and not result.timed_out

    try:
        if resume is not None:
            staged_dir = job / "checkpoints/resume-source"
            staged_dir.mkdir(parents=True, exist_ok=False)
            staged = staged_dir / Path(resume["checkpoint_path"]).name
            shutil.copyfile(resume["checkpoint_path"], staged)
            resume["staged_checkpoint"] = _artifact(staged, job)
            verify_resume_input()
            _write_report(report_path, report)
        environment = probe_environment(
            python, upstream, timeout_s=min(60, deadline - time.monotonic()), device=spec.device
        )
        report["environment"] = environment
        reasons = environment_reasons(environment, upstream, device=spec.device)
        if reasons:
            report.update(status="blocked_preflight", reasons=reasons)
            return report
        report["training_gate"] = "failed"
        if not phase("train", plan["train_command"]):
            report["status"] = "training_failed"
            return report
        checkpoints = [
            path
            for path in (job / "checkpoints").rglob("model_*.pt")
            if re.fullmatch(r"model_\d+\.pt", path.name)
            and not path.is_relative_to(job / "checkpoints/resume-source")
        ]
        if not checkpoints:
            report.update(status="training_failed", reasons=["no_checkpoint_produced"])
            return report
        checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        report["artifacts"]["checkpoint"] = _artifact(checkpoint, job)
        if (
            resume is not None
            and report["artifacts"]["checkpoint"]["sha256"] == resume["checkpoint_sha256"]
        ):
            report.update(status="training_failed", reasons=["resume_checkpoint_unchanged"])
            return report
        report["training_gate"] = "checkpoint_produced"
        onnx = job / "artifacts/policy.onnx"
        export = [
            str(python),
            str(upstream / "scripts/export.py"),
            TASK_ID,
            "--checkpoint-file",
            str(checkpoint),
            "--onnx-file",
            str(onnx),
            "--num-envs",
            "1",
            "--device",
            "cpu" if spec.device == "cpu" else "cuda:0",
        ]
        report["export_gate"] = "failed"
        if not phase("export", export):
            report["status"] = "export_failed"
            return report
        report["artifacts"]["onnx"] = _artifact(onnx, job)
        if _artifact(checkpoint, job) != report["artifacts"]["checkpoint"]:
            raise ValueError("checkpoint changed during export")
        report["export_gate"] = "exported"
        report["normalizer_provenance"] = "official_checkpoint_export"
        compatibility = job / "artifacts/compatibility.json"
        inspect = [
            str(python),
            str(Path(__file__).with_name("policy_check.py")),
            "--onnx",
            str(onnx),
            "--report",
            str(compatibility),
            "--command-profile",
            "locomotion",
            "--steps",
            "50",
            "--seed",
            str(spec.seed),
        ]
        report["compatibility_gate"] = "failed"
        if not phase("inspect", inspect):
            report["status"] = "compatibility_failed"
            return report
        inspected = json.loads(compatibility.read_text())
        checks = {
            "passed": inspected.get("passed") is True,
            "compatibility_only": inspected.get("compatibility_only") is True,
            "physical_hardware_gate": inspected.get("physical_hardware_gate") == "not_run",
            "model_sha256": inspected.get("model_sha256") == report["artifacts"]["onnx"]["sha256"],
            "contract_commit": inspected.get("contract_commit") == CONTRACT_COMMIT,
            "profile": inspected.get("profile") == "microduck-feedforward-api1",
        }
        mismatches = [name for name, valid in checks.items() if not valid]
        if mismatches:
            report.update(
                status="compatibility_failed",
                reasons=[f"inspection_mismatch:{name}" for name in mismatches],
            )
            return report
        if _artifact(onnx, job) != report["artifacts"]["onnx"]:
            raise ValueError("ONNX changed during inspection")
        report["artifacts"]["compatibility"] = _artifact(compatibility, job)
        report["compatibility_gate"] = "pass"
        simulation = job / "artifacts/simulation.json"
        probe_script = Path(__file__).with_name("offline_eval.py")
        simulate = [
            str(python),
            str(probe_script),
            "--upstream",
            str(upstream),
            "--checkpoint",
            str(checkpoint),
            "--onnx",
            str(onnx),
            "--report",
            str(simulation),
        ]
        report["simulator_gate"] = "failed"
        if not phase("simulation", simulate):
            report.update(status="simulation_failed", reasons=["probe_process_failed"])
            return report
        try:
            simulation_artifact = _artifact(simulation, job)
            evidence = json.loads(simulation.read_text())
            if not isinstance(evidence, dict):
                raise ValueError("official physics report must be an object")
            qualification = assess_official_probe(
                evidence,
                checkpoint_sha256=report["artifacts"]["checkpoint"]["sha256"],
                onnx_sha256=report["artifacts"]["onnx"]["sha256"],
                script_sha256=_sha(probe_script),
                upstream_commit=source["commit"],
                uv_lock_sha256=source["uv_lock_sha256"],
            )
            if (
                _artifact(simulation, job) != simulation_artifact
                or _artifact(checkpoint, job) != report["artifacts"]["checkpoint"]
                or _artifact(onnx, job) != report["artifacts"]["onnx"]
            ):
                raise ValueError("simulation evidence or model changed during assessment")
        except (OSError, ValueError, TypeError) as error:
            report.update(
                status="simulation_failed",
                reasons=["invalid_or_missing_probe_evidence"],
                error=f"{type(error).__name__}: {error}"[:500],
            )
            return report
        report["artifacts"]["simulation"] = simulation_artifact
        report["qualification"] = qualification
        if qualification["verdict"] == "rejected":
            report.update(status="candidate_rejected", simulator_gate="failed")
        else:
            report.update(status="candidate_unqualified", simulator_gate="inconclusive")
        return report
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
        return report
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_report(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, required=True, help="prepared official trainer Python 3.12"
    )
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--push-interval-s",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="experimental official push interval override in seconds; formal play gate unchanged",
    )
    parser.add_argument("--resume-job", type=Path, help="trusted completed local Duck Mind job")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = TrainingSpec(
        args.num_envs,
        args.max_iterations,
        args.seed,
        args.timeout_s,
        args.device,
        tuple(args.push_interval_s) if args.push_interval_s is not None else None,
    )
    if args.action == "plan":
        result = make_plan(args.upstream, args.python, args.job, spec, resume_job=args.resume_job)
    elif args.action == "preflight":
        source = verify_upstream(args.upstream)
        environment = probe_environment(args.python, args.upstream, device=spec.device)
        reasons = environment_reasons(environment, args.upstream, device=spec.device)
        result = {
            "upstream": source,
            "environment": environment,
            "passed": not reasons,
            "reasons": reasons,
            "device": spec.device,
            "experimental": spec.device == "cpu",
        }
        if args.resume_job is not None:
            result["resume_source"] = make_plan(
                args.upstream, args.python, args.job, spec, resume_job=args.resume_job
            )["resume_source"]
    else:
        result = run_job(
            args.upstream,
            args.python,
            args.job,
            spec,
            execute=args.execute,
            resume_job=args.resume_job,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result.get("passed") is False or result.get("status") is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
