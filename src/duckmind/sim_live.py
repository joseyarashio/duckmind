"""One-shot integration gate: Duck Mind -> robotd -> ONNX -> MuJoCo -> outcome.

Run from an environment with Python 3.12, mujoco and numpy. This harness owns
only its own child processes and a fresh short temporary state directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from duckmind.robotd import RobotdBodyPort, RobotdOperatorClient
from duckmind.skills import EmbodiedSkillLayer, OneShotSkillIntent, SkillPhase
from duckmind.tmpdir import short_tmp
from duckmind.versions import package_fingerprint

POLICY_FILES = (
    "velstand.onnx",
    "alpha_sitstand.onnx",
    "alpha_ground_pick.onnx",
    "ball_kick_left.onnx",
    "ball_kick_right.onnx",
    "roulade.onnx",
)


def policy_file_hashes(directory: Path) -> dict[str, str]:
    """Bind the policy bytes the isolated simulator config selects."""

    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in POLICY_FILES
    }


class SimOracle:
    """Validation-only view of MuJoCo. Never passed to cognition/BodyPort.

    One request, one answer, on one connection. A call that fails halfway (the body server was
    slow to answer, the link broke) leaves the stream in an unknown place: a late answer could
    still be on its way and would be read as the answer to the *next* request. So a failed call
    drops the connection (``lost``) and the oracle answers nothing until :meth:`reconnect` opens a
    new one; whatever late answer was still on the old connection goes with it. :meth:`close` is
    final: a closed oracle never reconnects."""

    def __init__(self, port: int, *, timeout_s: float = 2.0) -> None:
        self._port = port
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._stream: Any = None
        self._final = False
        self._open()

    @property
    def lost(self) -> bool:
        """The connection was dropped after a failed call: :meth:`reconnect` before the next one."""
        return self._stream is None

    def _open(self) -> None:
        sock = socket.create_connection(("127.0.0.1", self._port), timeout=self._timeout_s)
        sock.settimeout(self._timeout_s)
        self._socket, self._stream = sock, sock.makefile("rwb")
        try:
            hello = self._call({"op": "hello", "protocol": 1, "joints": 15})
            if hello.get("protocol") != 1:
                raise RuntimeError("unexpected MuJoCo body protocol")
        except BaseException:
            self._drop()
            raise

    def reconnect(self) -> None:
        """Drop the connection (and any late answer still on it) and open a new one."""
        if self._final:
            raise ConnectionError("MuJoCo body oracle is closed")
        self._drop()
        self._open()

    def _call(self, payload: dict[str, object]) -> dict[str, Any]:
        stream = self._stream
        if stream is None:
            raise ConnectionError("MuJoCo body oracle has no connection")
        try:
            stream.write((json.dumps(payload) + "\n").encode())
            stream.flush()
            answer = stream.readline()
            if not answer:
                raise ConnectionError("MuJoCo body oracle disconnected")
            result = json.loads(answer)
        except Exception:
            self._drop()  # (timed out, broken or garbled: the stream cannot be trusted any more)
            raise
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError(f"MuJoCo body oracle error: {result}")
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self._call({"op": "read"})
        if not isinstance(result.get("trunk_z"), (int, float)) or not isinstance(
            result.get("positions"), list
        ):
            raise RuntimeError("MuJoCo body read lacks ground truth")
        return result

    def _drop(self) -> None:
        stream, sock = self._stream, self._socket
        self._stream = self._socket = None
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
        finally:
            if sock is not None:
                sock.close()

    def close(self) -> None:
        self._final = True
        self._drop()

    def __enter__(self) -> SimOracle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_ready(process: subprocess.Popen, check, *, deadline_s: float, label: str) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited early with code {process.returncode}")
        if check():
            return
        time.sleep(0.1)
    raise TimeoutError(f"{label} did not become ready in {deadline_s}s")


def _stop_child(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _policy_config(
    directory: Path, *, walk_override: Path | None = None, none_slots: tuple[str, ...] = ()
) -> str:
    """The ``[policy]`` table of this simulator's robotd.toml. ``none_slots``: skill slots switched off
    with the ``"none"`` sentinel (only ``roulade`` here: a slot's path in this table outranks a
    ``[[policy.skill]]`` entry of the same name, so a table that needs the room switches the slot)."""
    values = {name: directory / name for name in POLICY_FILES}
    missing = [name for name, path in values.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official policy files missing: {', '.join(missing)}")
    if walk_override is not None:
        if (
            not walk_override.is_absolute()
            or walk_override.is_symlink()
            or not walk_override.is_file()
        ):
            raise ValueError("walk override must be an absolute, regular local model file")
        values["velstand.onnx"] = walk_override
    return "\n".join(
        (
            "[policy]",
            "enabled = true",
            f"walk = {json.dumps(str(values['velstand.onnx']))}",
            'stand = "none"',
            f"sitstand = {json.dumps(str(values['alpha_sitstand.onnx']))}",
            f"ground_pick = {json.dumps(str(values['alpha_ground_pick.onnx']))}",
            f"kick_left = {json.dumps(str(values['ball_kick_left.onnx']))}",
            f"kick_right = {json.dumps(str(values['ball_kick_right.onnx']))}",
            'roulade = "none"'
            if "roulade" in none_slots
            else f"roulade = {json.dumps(str(values['roulade.onnx']))}",
            "",
        )
    )


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _socket_open(path: Path) -> bool:
    if not path.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(str(path)) == 0


def _max_joint_delta(before: list[float], after: list[float]) -> float:
    if len(before) != 15 or len(after) != 15:
        raise RuntimeError("unexpected simulator joint count")
    # Exclude the mouth, which no locomotion policy controls.
    return max(abs(a - b) for index, (a, b) in enumerate(zip(before, after)) if index != 9)


def _installed_ort_library() -> Path:
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "onnxruntime is not installed; use --ort-lib or uv --with onnxruntime"
        )
    capi = Path(spec.origin).parent / "capi"
    candidates = sorted(
        path
        for path in capi.iterdir()
        if path.name.startswith("libonnxruntime.") and (".dylib" in path.name or ".so" in path.name)
    )
    if not candidates:
        raise FileNotFoundError(f"onnxruntime shared library missing under {capi}")
    return candidates[-1]


def run_sim_skill(
    *,
    microduck: Path,
    microduck_rl: Path,
    policies: Path,
    ort_lib: Path,
    robotd_binary: Path | None = None,
    trace_path: Path,
    skill: str = "kick_left",
) -> dict[str, Any]:
    """Run one episodic skill, recording ACK, state and simulator-only evidence."""

    robotd = robotd_binary or microduck / "target" / "debug" / "robotd"
    scene = microduck_rl / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"
    body_server = microduck_rl / "src/mjlab_microduck/sim/body_server.py"
    for name, path in (("robotd", robotd), ("scene", scene), ("ONNX Runtime", ort_lib)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    config = _policy_config(policies)
    manifest = json.loads((policies / "manifest.json").read_text())
    if manifest.get("obs_len") != 61 or manifest.get("action_len") != 14:
        raise RuntimeError("policy manifest does not match robotd's 61 -> 14 contract")
    trace_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    started_utc = datetime.now(timezone.utc).isoformat()
    port = _free_local_port()
    records: list[dict[str, Any]] = [
        {
            "type": "run",
            "schema_version": 1,
            "trace_id": trace_id,
            "episode_id": episode_id,
            "started_at": started_utc,
            "duckmind_source_sha256": package_fingerprint(),
            "microduck_commit": _git_commit(microduck),
            "microduck_rl_commit": _git_commit(microduck_rl),
            "robotd_binary_sha256": hashlib.sha256(robotd.read_bytes()).hexdigest(),
            "ort_lib_sha256": hashlib.sha256(ort_lib.read_bytes()).hexdigest(),
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "body_server_sha256": hashlib.sha256(body_server.read_bytes()).hexdigest(),
            "policy_manifest_sha256": hashlib.sha256(
                (policies / "manifest.json").read_bytes()
            ).hexdigest(),
            "policy_files_sha256": policy_file_hashes(policies),
            "policy_source": (policies / ".source").read_text().strip()
            if (policies / ".source").is_file()
            else "unrecorded",
            "policy_walk": "velstand.onnx",
            "skill": skill,
            "seed": 0,
            "scene": "scene.xml",
            "start_timeout_s": 2.0,
            "completion_timeout_s": 8.0,
        }
    ]
    body_process: subprocess.Popen | None = None
    robot_process: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="dm-sim-", dir=short_tmp()) as state_name:
            state = Path(state_name)
            socket_path = state / "robotd.sock"
            params_path = state / "robotd.toml"
            params_path.write_text(config)
            with (
                (state / "body.log").open("w") as body_log,
                (state / "robotd.log").open("w") as robot_log,
            ):
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
                    env={**os.environ, "PYTHONPATH": str(microduck_rl / "src")},
                    stdout=body_log,
                    stderr=subprocess.STDOUT,
                )
                _wait_ready(
                    body_process,
                    lambda: _port_open(port),
                    deadline_s=30,
                    label="duck-body",
                )
                robot_process = subprocess.Popen(
                    [
                        str(robotd),
                        "--sim",
                        f"127.0.0.1:{port}",
                        "--params",
                        str(params_path),
                        "--socket",
                        str(socket_path),
                    ],
                    cwd=microduck,
                    env={
                        **os.environ,
                        "ORT_DYLIB_PATH": str(ort_lib),
                        "DUCK_RUNTIME_DIR": str(state),
                        "RUST_LOG": "info",
                    },
                    stdout=robot_log,
                    stderr=subprocess.STDOUT,
                )
                _wait_ready(
                    robot_process,
                    lambda: _socket_open(socket_path),
                    deadline_s=30,
                    label="robotd",
                )
                with RobotdBodyPort(socket_path, backend="simulator", hz=25) as body:
                    if body.capabilities.get("walk") != "velstand.onnx":
                        raise RuntimeError(f"wrong walking policy: {body.capabilities}")
                    if body.capabilities.get("unavailable"):
                        raise RuntimeError(
                            f"policy unavailable: {body.capabilities['unavailable']}"
                        )
                    if skill not in body.capabilities.get("skills", []):
                        raise RuntimeError(f"skill {skill} is not loaded: {body.capabilities}")
                    with (
                        RobotdOperatorClient(socket_path) as operator,
                        SimOracle(port) as oracle,
                    ):
                        enable_ack = operator.enable(True)
                        records.append({"type": "operator_enable", "ack": asdict(enable_ack)})
                        if not enable_ack.accepted:
                            raise RuntimeError(
                                f"robotd refused operator enable: {enable_ack.reason}"
                            )
                        deadline = time.monotonic() + 12
                        baseline = None
                        while time.monotonic() < deadline:
                            observed = body.observe()
                            if observed.policy == "walk" and not observed.safety.fallen:
                                baseline = observed
                                break
                        if baseline is None:
                            raise TimeoutError("robotd never reached upright walk policy")
                        initial_truth = oracle.snapshot()
                        records.append(
                            {
                                "type": "baseline",
                                "policy": baseline.policy,
                                "robot_time_s": baseline.robot_time_s,
                                "trunk_z": initial_truth["trunk_z"],
                            }
                        )
                        layer = EmbodiedSkillLayer(body)
                        execution = layer.start_one_shot(
                            OneShotSkillIntent(intent_id=intent_id, skill=skill),
                            start_timeout_s=2,
                            completion_timeout_s=8,
                        )
                        dispatch_at = execution.dispatched_at
                        records.append(
                            {
                                "type": "dispatch",
                                "intent_id": intent_id,
                                "local_monotonic_s": dispatch_at,
                                "ack": asdict(execution.ack),
                                "skill": skill,
                            }
                        )
                        min_trunk_z = float(initial_truth["trunk_z"])
                        max_joint_delta = 0.0
                        seen_phases: set[str] = {execution.status.phase.value}
                        limit_reasons: set[str] = set()
                        sample_count = 0
                        while not execution.status.terminal:
                            try:
                                observed, status = execution.poll()
                            except (ConnectionError, TimeoutError):
                                break
                            truth = oracle.snapshot()
                            seen_phases.add(status.phase.value)
                            sample_count += 1
                            limit_reasons.update(observed.limited_by)
                            min_trunk_z = min(min_trunk_z, float(truth["trunk_z"]))
                            max_joint_delta = max(
                                max_joint_delta,
                                _max_joint_delta(initial_truth["positions"], truth["positions"]),
                            )
                            records.append(
                                {
                                    "type": "sample",
                                    "event_id": f"{trace_id}:{sample_count}",
                                    "episode_id": episode_id,
                                    "source": {
                                        "kind": "robotd_state",
                                        "source_id": "robotd",
                                        "reliability": 1.0,
                                        "version": records[0]["microduck_commit"],
                                    },
                                    "local_monotonic_s": observed.monotonic_s,
                                    "robot_time_s": observed.robot_time_s,
                                    "policy": observed.policy,
                                    "safety": asdict(observed.safety),
                                    "move": {
                                        "requested": observed.movement_requested,
                                        "applied": observed.movement_applied,
                                        "limited_by": observed.limited_by,
                                    },
                                    "oracle": {
                                        "trunk_z": truth["trunk_z"],
                                        "max_joint_delta_rad": max_joint_delta,
                                    },
                                    "phase": status.phase.value,
                                }
                            )
                        body.stop()
                        operator.enable(False)
                        status = execution.status
                        # This proves the execution path, not that a kick hit
                        # a ball: the scene has no goal/contact oracle for that.
                        gate_passed = (
                            status.phase is SkillPhase.COMPLETED
                            and SkillPhase.STARTED.value in seen_phases
                            and max_joint_delta >= 0.02
                        )
                        warnings = []
                        if min_trunk_z < 0.06:
                            warnings.append("low_trunk_height_observed")
                        if "joint_range" in limit_reasons:
                            warnings.append("joint_range_limit_observed")
                        summary = {
                            "type": "outcome",
                            "outcome_id": str(uuid.uuid4()),
                            "trace_id": trace_id,
                            "episode_id": episode_id,
                            "execution_id": execution.tracker.execution_id,
                            "evaluator_version": "sim-one-shot-execution-v2",
                            "local_monotonic_s": time.monotonic(),
                            "execution_gate": "pass" if gate_passed else "fail",
                            "verdict": "unknown",
                            "phase": status.phase.value,
                            "reason": "no_task_goal_evidence"
                            if gate_passed
                            else status.reason or "execution_gate_failed",
                            "warnings": warnings,
                            "evidence_event_ids": [
                                item["event_id"] for item in records if item["type"] == "sample"
                            ],
                            "sample_count": sample_count,
                            "min_trunk_z_m": min_trunk_z,
                            "max_joint_delta_rad": max_joint_delta,
                            "phase_evidence": sorted(seen_phases),
                        }
                        records.append(summary)
                        records.append(
                            {
                                "type": "episode",
                                "episode_id": episode_id,
                                "trace_id": trace_id,
                                "started_at": started_utc,
                                "ended_at": datetime.now(timezone.utc).isoformat(),
                                "event_ids": summary["evidence_event_ids"],
                                "intent_ids": [intent_id],
                                "outcome_ids": [summary["outcome_id"]],
                                "runtime_versions": {
                                    "microduck": records[0]["microduck_commit"],
                                    "microduck_rl": records[0]["microduck_rl_commit"],
                                    "policy_set": "v5",
                                    "evaluator": summary["evaluator_version"],
                                },
                            }
                        )
                        return summary
    finally:
        _stop_child(robot_process)
        _stop_child(body_process)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--microduck-rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--ort-lib", type=Path, help="defaults to the installed onnxruntime wheel")
    parser.add_argument("--robotd-bin", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--skill", default="kick_left")
    args = parser.parse_args()
    summary = run_sim_skill(
        microduck=args.microduck.resolve(),
        microduck_rl=args.microduck_rl.resolve(),
        policies=args.policies.resolve(),
        ort_lib=args.ort_lib.resolve() if args.ort_lib else _installed_ort_library(),
        robotd_binary=args.robotd_bin.resolve() if args.robotd_bin else None,
        trace_path=args.trace.resolve(),
        skill=args.skill,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["execution_gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
