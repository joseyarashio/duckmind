import math
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from duckmind.body import BodyIntent
from duckmind.robotd import (
    RobotdBodyPort,
    RobotdOperatorClient,
    RobotdProtocolError,
    RobotdSafetyClient,
)
from duckmind.tmpdir import short_tmp


@pytest.fixture
def short_socket_dir():
    # macOS AF_UNIX paths are shorter than pytest's usual nested tmp_path.
    with tempfile.TemporaryDirectory(prefix="dm-", dir=short_tmp()) as directory:
        yield Path(directory)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("robot.rebootMotors", {}),
        ("robot.enable", {"on": True}),
        ("robot.move", {"vx": math.nan}),
        ("robot.move", {"joint": 3}),
        ("robot.do", {"skill": ""}),
        ("robot.stop", {"force": True}),
    ],
)
def test_body_intent_rejects_unsafe_or_invalid_commands(method, params) -> None:
    with pytest.raises(ValueError):
        BodyIntent(method, params)


def test_body_intent_parameters_cannot_be_changed_after_validation() -> None:
    original = {"vx": 0.1}
    intent = BodyIntent("robot.move", original)
    original["vx"] = 1.0
    assert intent.params["vx"] == 0.1
    with pytest.raises(TypeError):
        intent.params["joint"] = 5


def test_robotd_body_port_uses_same_state_and_command_contract(
    short_socket_dir,
) -> None:
    path = short_socket_dir / "robot.sock"
    ready = threading.Event()
    commands = []

    def serve() -> None:
        import json

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(3)
            ready.set()
            command_sock, _ = listener.accept()
            with command_sock, command_sock.makefile("rwb") as command_stream:
                hello = json.loads(command_stream.readline())
                commands.append(hello)
                command_stream.write(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": hello["id"],
                                "result": {"api_version": 37},
                            }
                        )
                        + "\n"
                    ).encode()
                )
                command_stream.flush()
                state_sock, _ = listener.accept()
                with state_sock, state_sock.makefile("rwb") as state_stream:
                    subscribe = json.loads(state_stream.readline())
                    commands.append(subscribe)
                    state_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": subscribe["id"],
                                    "result": {
                                        "accepted": True,
                                        "walk": "velstand.onnx",
                                        "skills": ["kick_left"],
                                    },
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    state_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "robot.state",
                                    "params": {
                                        "t": 1.4,
                                        "policy": "walk",
                                        "move": {
                                            "requested": [0.1, 0, 0],
                                            "applied": [0.08, 0, 0],
                                            "limited_by": ["speed"],
                                        },
                                        "safety": {"fallen": False, "limp": False},
                                        "odom": {"position": [0.01, 0, 0], "yaw": 0.0},
                                    },
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    state_stream.flush()
                    action = json.loads(command_stream.readline())
                    commands.append(action)
                    command_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": action["id"],
                                    "result": {"accepted": True},
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    command_stream.flush()
                    stop = json.loads(command_stream.readline())
                    commands.append(stop)
                    command_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": stop["id"],
                                    "result": {"accepted": True},
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    command_stream.flush()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    with RobotdBodyPort(path, backend="simulator") as port:
        observation = port.observe()
        ack = port.dispatch_intent(BodyIntent("robot.do", {"skill": "kick_left"}))
        stopped = port.stop()

    thread.join(2)
    assert not thread.is_alive()
    assert observation.policy == "walk"
    assert observation.movement_applied == (0.08, 0.0, 0.0)
    assert observation.robot_time_s == 1.4
    assert observation.metadata["odom"]["position"] == [0.01, 0, 0]
    assert ack.accepted and stopped.accepted
    assert [item["method"] for item in commands] == [
        "hello",
        "robot.subscribe",
        "robot.do",
        "robot.stop",
    ]


def test_operator_control_is_not_part_of_body_port() -> None:
    assert "enable" not in RobotdBodyPort.__dict__
    assert hasattr(RobotdOperatorClient, "enable")


def test_safety_client_reads_loaded_policy_slots_without_writing_to_robotd(
    short_socket_dir,
) -> None:
    import json

    path = short_socket_dir / "robot.sock"
    ready = threading.Event()
    commands = []

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(1)
            ready.set()
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb") as stream:
                for result in (
                    {"api_version": 37},
                    {
                        "mode": "walk",
                        "enabled": True,
                        "homed": True,
                        "slots": [
                            {
                                "slot": "walk",
                                "path": "/opt/robot/policies/current/velstand.onnx",
                                "origin": "official",
                                "overridden": False,
                                "error": None,
                            }
                        ],
                    },
                ):
                    request = json.loads(stream.readline())
                    commands.append(request)
                    stream.write(
                        (
                            json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result})
                            + "\n"
                        ).encode()
                    )
                    stream.flush()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    with RobotdSafetyClient(path) as client:
        report = client.policies()
    thread.join(2)
    assert not thread.is_alive()
    assert report["slots"][0]["origin"] == "official"
    assert [command["method"] for command in commands] == ["hello", "robot.policies"]


def test_v5_head_delta_profile_is_checked_against_measured_joints(
    short_socket_dir,
) -> None:
    import json

    path = short_socket_dir / "robot.sock"
    ready = threading.Event()
    commands = []

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(2)
            ready.set()
            command_sock, _ = listener.accept()
            with command_sock, command_sock.makefile("rwb") as command_stream:
                hello = json.loads(command_stream.readline())
                command_stream.write(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": hello["id"],
                                "result": {"api_version": 37},
                            }
                        )
                        + "\n"
                    ).encode()
                )
                command_stream.flush()
                state_sock, _ = listener.accept()
                with state_sock, state_sock.makefile("rwb") as state_stream:
                    subscribe = json.loads(state_stream.readline())
                    state_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": subscribe["id"],
                                    "result": {
                                        "accepted": True,
                                        "walk": "velstand.onnx",
                                    },
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    state_stream.flush()
                    action = json.loads(command_stream.readline())
                    commands.append(action)
                    joints = [0.0] * 15
                    joints[5:9] = [0.3491, 0.4991, 0.0, 0.0]
                    state_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "robot.state",
                                    "params": {
                                        "t": 2.0,
                                        "policy": "walk",
                                        "move": {
                                            "requested": [0, 0, 0],
                                            "applied": [0, 0, 0],
                                            "limited_by": [],
                                        },
                                        "safety": {"fallen": False, "limp": False},
                                        "joints": joints,
                                    },
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    state_stream.flush()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    with RobotdBodyPort(path, backend="simulator") as port:
        assert port.supports_v5_head_profile
        ack = port.dispatch_intent(
            BodyIntent(
                "robot.head",
                {
                    "neck_pitch": 0.0,
                    "head_pitch": 0.15,
                    "head_yaw": 0.0,
                    "head_roll": 0.0,
                },
            )
        )
        observed = port.observe()
    thread.join(2)
    assert ack.accepted and not ack.confirmed
    assert observed.head_tracking_error_rad == pytest.approx(0.0)
    assert commands[0]["method"] == "robot.head"


def test_mismatched_head_profile_is_refused_before_notification(
    short_socket_dir,
) -> None:
    import json

    path = short_socket_dir / "robot.sock"
    ready = threading.Event()
    commands = []

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(2)
            ready.set()
            command_sock, _ = listener.accept()
            with command_sock, command_sock.makefile("rwb") as command_stream:
                hello = json.loads(command_stream.readline())
                command_stream.write(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": hello["id"],
                                "result": {"api_version": 38},
                            }
                        )
                        + "\n"
                    ).encode()
                )
                command_stream.flush()
                state_sock, _ = listener.accept()
                with state_sock, state_sock.makefile("rwb") as state_stream:
                    subscribe = json.loads(state_stream.readline())
                    state_stream.write(
                        (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": subscribe["id"],
                                    "result": {"accepted": True, "walk": "future.onnx"},
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    state_stream.flush()
                    command_sock.settimeout(0.2)
                    try:
                        data = command_sock.recv(1)
                        commands.append(data)
                    except socket.timeout:
                        pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    with RobotdBodyPort(path, backend="simulator") as port:
        assert not port.supports_v5_head_profile
        with pytest.raises(RobotdProtocolError, match="not verified"):
            port.dispatch_intent(BodyIntent("robot.head", {"head_pitch": 0.15}))
    thread.join(2)
    assert not any(commands)
