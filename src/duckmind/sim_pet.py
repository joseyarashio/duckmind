"""Run the local pet service against the official MuJoCo/robotd body path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from duckmind.behavior import BehaviorChoice, choose_invitation
from duckmind.cli import request as service_request
from duckmind.core import DuckSelf, reduce_event
from duckmind.pet_body_trace import replay_front_gaze_trace
from duckmind.provenance import Event, SourceRef
from duckmind.robotd import RobotdBodyPort, RobotdOperatorClient, RobotdSafetyClient
from duckmind.sim_live import (
    _free_local_port,
    _git_commit,
    _installed_ort_library,
    _policy_config,
    _port_open,
    _socket_open,
    _stop_child,
    _wait_ready,
    policy_file_hashes,
)
from duckmind.tmpdir import short_tmp
from duckmind.versions import package_fingerprint

VALID_KEYFRAMES = ("HOME", "SIT", "STAND", "FOLD")


def _body_trace_path(report_path: Path) -> Path:
    return report_path.with_suffix(".body.json")


def _body_server_command(*, port: int, keyframe: str) -> list[str]:
    if keyframe not in VALID_KEYFRAMES:
        raise ValueError(f"unsupported simulator keyframe: {keyframe}")
    return [
        sys.executable,
        "-m",
        "mjlab_microduck.sim.body_server",
        "--headless",
        "--keyframe",
        keyframe,
        "--port",
        str(port),
    ]


def _choose_actor() -> str:
    now = datetime.now(UTC)
    enable = Event(
        event_id="sim-owner-enable",
        kind="owner_play_enabled",
        occurred_at=now,
        observed_at=now,
        source=SourceRef("owner_control", "sim-harness", 1.0),
        trace_id="sim",
        payload={},
    )
    state = reduce_event(DuckSelf.new(identity="duck", born_at=now), enable)
    for index in range(1000):
        actor = f"simowner{index}"
        invitation = Event(
            event_id=f"sim-invite-{index}",
            kind="invitation",
            occurred_at=now,
            observed_at=now,
            source=SourceRef("owner_request", "sim-harness", 1.0),
            trace_id="sim",
            payload={"actor": actor, "activity": "look_game"},
        )
        if choose_invitation(state, invitation).choice is BehaviorChoice.ENGAGE:
            return actor
    raise RuntimeError("seeded exploration never selected a first interaction")


def _ask(path: Path, action: str, **params: object) -> dict[str, object]:
    response = service_request(path, {"action": action, "request_id": str(uuid.uuid4()), **params})
    if response.get("ok") is not True:
        raise RuntimeError(f"Duck Mind {action} refused: {response.get('error')}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Duck Mind response lacked result")
    return result


def run_sim_pet(
    *,
    microduck: Path,
    microduck_rl: Path,
    policies: Path,
    robotd_binary: Path,
    ort_lib: Path,
    report_path: Path,
    keyframe: str = "HOME",
    feedback_score: int = 1,
) -> dict[str, object]:
    if keyframe not in VALID_KEYFRAMES:
        raise ValueError(f"unsupported simulator keyframe: {keyframe}")
    if type(feedback_score) is not int or feedback_score not in (0, 1):
        raise ValueError("binary feedback score must be 0 or 1")
    scene = microduck_rl / "src/mjlab_microduck/robot/microduck/scene.xml"
    body_server = microduck_rl / "src/mjlab_microduck/sim/body_server.py"
    for name, path in (("robotd", robotd_binary), ("scene", scene), ("ORT", ort_lib)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} missing: {path}")
    config = _policy_config(policies)
    manifest = json.loads((policies / "manifest.json").read_text())
    if manifest.get("obs_len") != 61 or manifest.get("action_len") != 14:
        raise RuntimeError("unexpected policy observation/action dimensions")
    port = _free_local_port()
    body_process = robot_process = pet_process = None
    report: dict[str, object] = {
        "schema_version": 1,
        "scenario": "official-v5-first-pet-slice",
        "initial_keyframe": keyframe,
        "duckmind_source_sha256": package_fingerprint(),
        "microduck_commit": _git_commit(microduck),
        "microduck_rl_commit": _git_commit(microduck_rl),
        "robotd_binary_sha256": hashlib.sha256(robotd_binary.read_bytes()).hexdigest(),
        "ort_lib_sha256": hashlib.sha256(ort_lib.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "body_server_sha256": hashlib.sha256(body_server.read_bytes()).hexdigest(),
        "policy_manifest_sha256": hashlib.sha256(
            (policies / "manifest.json").read_bytes()
        ).hexdigest(),
        "policy_files_sha256": policy_file_hashes(policies),
        "started_at": datetime.now(UTC).isoformat(),
        "execution_gate": "not_run",
        "body_replay_gate": "not_run",
        "learning_gate": "not_run",
        "physical_hardware_gate": "not_run",
        "feedback_source": "synthetic_simulation_harness",
        "feedback_score": feedback_score,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="dm-pet-sim-", dir=short_tmp()) as name:
            root = Path(name)
            params = root / "robotd.toml"
            params.write_text(config)
            robot_socket = root / "robotd.sock"
            pet_socket = root / "pet.sock"
            database = root / "pet.sqlite3"
            inner_trace = root / "look-game.body.json"
            with (
                (root / "body.log").open("w") as body_log,
                (root / "robotd.log").open("w") as robot_log,
                (root / "pet.log").open("w") as pet_log,
            ):
                try:
                    body_process = subprocess.Popen(
                        _body_server_command(port=port, keyframe=keyframe),
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
                            str(robotd_binary),
                            "--sim",
                            f"127.0.0.1:{port}",
                            "--params",
                            str(params),
                            "--socket",
                            str(robot_socket),
                        ],
                        cwd=microduck,
                        env={
                            **os.environ,
                            "ORT_DYLIB_PATH": str(ort_lib),
                            "DUCK_RUNTIME_DIR": str(root),
                            "RUST_LOG": "info",
                        },
                        stdout=robot_log,
                        stderr=subprocess.STDOUT,
                    )
                    _wait_ready(
                        robot_process,
                        lambda: _socket_open(robot_socket),
                        deadline_s=30,
                        label="robotd",
                    )
                    with (
                        RobotdOperatorClient(robot_socket) as operator,
                        RobotdBodyPort(robot_socket, backend="simulator") as body,
                    ):
                        report["robotd_hello"] = body.hello
                        report["robotd_capabilities"] = body.capabilities
                        if body.capabilities.get(
                            "walk"
                        ) != "velstand.onnx" or body.capabilities.get("unavailable"):
                            raise RuntimeError(f"unexpected robotd policies: {body.capabilities}")
                        if not operator.enable(True).accepted:
                            raise RuntimeError("sim operator enable was refused")
                        deadline = time.monotonic() + 12
                        upright_samples = 0
                        while time.monotonic() < deadline:
                            observed = body.observe()
                            upright_samples = (
                                upright_samples + 1
                                if observed.policy == "walk"
                                and not observed.safety.fallen
                                and not observed.safety.limp
                                else 0
                            )
                            if upright_samples >= 10:
                                break
                        else:
                            raise TimeoutError("sim body never reached walk policy")
                        # Record the official daemon's loaded slots, not just
                        # robot.subscribe basenames, for contract review.
                        with RobotdSafetyClient(robot_socket) as safety:
                            report["robotd_loaded_policies"] = safety.policies()
                        pet_process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "duckmind.service",
                                "--state",
                                str(database),
                                "--socket",
                                str(pet_socket),
                                "--robotd",
                                str(robot_socket),
                                "--backend",
                                "simulator",
                                "--sim-body-trace",
                                str(inner_trace),
                            ],
                            cwd=report_path.parent,
                            stdout=pet_log,
                            stderr=subprocess.STDOUT,
                        )
                        _wait_ready(
                            pet_process,
                            lambda: _socket_open(pet_socket),
                            deadline_s=10,
                            label="duckmindd",
                        )
                        enabled = _ask(pet_socket, "enable")
                        if enabled.get("play_enabled") is not True:
                            raise RuntimeError(f"Duck Mind did not enable play: {enabled}")
                        actor = _choose_actor()
                        report["actor"] = actor
                        interaction = _ask(pet_socket, "invite", actor=actor, activity="look_game")
                        report["interaction"] = interaction
                        if not inner_trace.is_file():
                            raise RuntimeError("pet service did not record the gaze body trace")
                        trace_output = _body_trace_path(report_path)
                        trace_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(inner_trace, trace_output)
                        report["body_trace_file"] = trace_output.name
                        report["body_trace_sha256"] = hashlib.sha256(
                            trace_output.read_bytes()
                        ).hexdigest()
                        if not replay_front_gaze_trace(trace_output).passed:
                            raise RuntimeError("recorded gaze body trace did not replay")
                        report["body_replay_gate"] = "pass"
                        if interaction.get("execution") != "completed":
                            raise RuntimeError(f"gaze did not complete: {interaction}")
                        report["execution_gate"] = "pass"
                        feedback = _ask(
                            pet_socket,
                            "feedback",
                            interaction_id=interaction["interaction_id"],
                            score=feedback_score,
                        )
                        report["feedback"] = feedback
                        expected_outcome = "success" if feedback_score else "failure"
                        if (
                            feedback.get("learning_update_id") is None
                            or feedback.get("outcome") != expected_outcome
                        ):
                            raise RuntimeError(
                                "explicit simulated feedback did not produce the expected update"
                            )
                        report["before_stop"] = _ask(pet_socket, "status")
                        report["stop"] = _ask(pet_socket, "stop")
                        after_stop = _ask(pet_socket, "status")
                        report["after_stop"] = after_stop
                        if after_stop.get("play_enabled") is not False:
                            raise RuntimeError("owner stop did not disable play")
                        _stop_child(pet_process)
                        pet_process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "duckmind.service",
                                "--state",
                                str(database),
                                "--socket",
                                str(pet_socket),
                                "--robotd",
                                str(robot_socket),
                                "--backend",
                                "simulator",
                            ],
                            cwd=Path.cwd(),
                            stdout=pet_log,
                            stderr=subprocess.STDOUT,
                        )
                        _wait_ready(
                            pet_process,
                            lambda: _socket_open(pet_socket),
                            deadline_s=10,
                            label="restarted duckmindd",
                        )
                        after_restart = _ask(pet_socket, "status")
                        report["after_restart"] = after_restart
                        preferences = after_restart.get("preferences")
                        preference = (
                            preferences.get(f"preferences.{actor}.look_game")
                            if isinstance(preferences, dict)
                            else None
                        )
                        bounded_direction = (
                            isinstance(preference, (int, float))
                            and not isinstance(preference, bool)
                            and (
                                0.5 < preference <= 0.505000001
                                if feedback_score
                                else 0.494999999 <= preference < 0.5
                            )
                        )
                        if (
                            after_restart.get("play_enabled") is not False
                            or after_restart.get("learning_updates") != 1
                            or not bounded_direction
                        ):
                            raise RuntimeError(
                                "restart did not preserve bounded learning while staying passive"
                            )
                        report["learning_gate"] = "pass"
                        operator.enable(False)
                finally:
                    for process in (pet_process, robot_process, body_process):
                        _stop_child(process)
                    if report["execution_gate"] != "pass" or report["learning_gate"] != "pass":
                        for label in ("pet", "robotd", "body"):
                            log = root / f"{label}.log"
                            if log.exists():
                                report[f"{label}_log_tail"] = log.read_text(errors="replace")[
                                    -2000:
                                ]
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["ended_at"] = datetime.now(UTC).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--microduck-rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--robotd-bin", type=Path, required=True)
    parser.add_argument("--ort-lib", type=Path)
    parser.add_argument("--keyframe", choices=VALID_KEYFRAMES, default="HOME")
    parser.add_argument("--feedback-score", type=int, choices=(0, 1), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_sim_pet(
        microduck=args.microduck,
        microduck_rl=args.microduck_rl,
        policies=args.policies,
        robotd_binary=args.robotd_bin,
        ort_lib=args.ort_lib or _installed_ort_library(),
        report_path=args.report,
        keyframe=args.keyframe,
        feedback_score=args.feedback_score,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("execution_gate", "learning_gate", "physical_hardware_gate")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
