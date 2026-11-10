"""The page's real-duck mode, body side (docs/real-microduck.md): a bridge that says ``backend: real``
is driven with only the duck's own senses — robotd's state over the link — and no simulated world.

A fake robotd on a unix socket stands in for the duck; its bridge and the brain's LiveBody run in this
process over loopback TCP (no simulator, no MuJoCo).
"""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind import livebody
from duckmind.i18n import msg
from duckmind.livebody import (
    LiveBody,
    LivePaths,
    NoWorld,
    RemoteBodySpec,
    real_frame_pose,
)
from duckmind.remotebody.bridge import Bridge, Channel
from duckmind.remotebody.client import RemoteLink
from duckmind.remotebody.simbody import body_info, channels_for
from duckmind.tmpdir import short_tmp

TOKEN = b"0123456789abcdef0123456789abcdef0123456789abcdef"
JOINTS = [0.01 * (index + 1) for index in range(15)]
OPERATOR = {"robot.enable", "robot.init", "robot.poseJoints"}


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class StateRobotd:
    """hello / subscribe / anything answered; robot.state at ~50 Hz from a settable ``state``.
    Every method asked of it is kept (``methods``); ``robot.enable`` brings its policy up."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.state: dict[str, Any] = {
            "policy": "stand",
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0], "limited_by": []},
            "safety": {"fallen": False, "limp": False, "picked_up": False},
            "joints": JOINTS,
            "odom": {"position": [0.4, -0.2, 0.118], "yaw": 0.5},
            "imu": {"quat": _yaw_quat(2.0), "gyro": [0, 0, 0]},
        }
        self.skills: list[str] = []  # robot.do asked of it (sit_toggle: change posture)
        self.methods: list[str] = []  # every request and notification, in order
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen(16)
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self._server.close()

    def _accept(self) -> None:
        self._server.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buffer, subscribed = b"", False
        conn.settimeout(0.02)
        while not self._stop.is_set():
            try:
                data = conn.recv(65536)
                if not data:
                    return
                buffer += data
            except TimeoutError:
                pass
            except OSError:
                return
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                message = json.loads(line)
                self.methods.append(str(message.get("method")))
                if "id" not in message:
                    continue
                result: dict[str, Any] = {"accepted": True}
                if message["method"] == "robot.subscribe":
                    subscribed = True
                    result = {"accepted": True, "walk": "velstand.onnx", "skills": []}
                elif message["method"] == "hello":
                    result = {"api_version": 49}
                elif message["method"] == "robot.do":
                    self.skills.append(str((message.get("params") or {}).get("skill")))
                elif message["method"] == "robot.enable":
                    self.state = {**self.state, "policy": "stand"}
                reply = {"jsonrpc": "2.0", "id": message["id"], "result": result}
                try:
                    conn.sendall(json.dumps(reply).encode() + b"\n")
                except OSError:
                    return
            if subscribed:
                params = {"t": time.monotonic(), **self.state}
                frame = {"jsonrpc": "2.0", "method": "robot.state", "params": params}
                try:
                    conn.sendall(json.dumps(frame).encode() + b"\n")
                except OSError:
                    return


def _wait_for(check: Any, seconds: float = 4.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def duck(tmp_path: Path):
    """A fake real duck behind a bridge that says ``backend: real`` (robotd only: no world)."""
    with tempfile.TemporaryDirectory(prefix="dm-rd-t-", dir=short_tmp()) as directory:
        robotd = StateRobotd(os.path.join(directory, "r.sock"))
        bridge = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={"robotd": Channel("robotd", unix=robotd.path)},
            keeper=None,
            body_info={"backend": "real", "rehearsal": True},
        )
        bridge.start()
        token = tmp_path / "token"
        token.write_bytes(TOKEN)
        token.chmod(0o600)
        yield SimpleNamespace(robotd=robotd, bridge=bridge, token=token)
        bridge.close()
        robotd.close()


def _body(tmp_path: Path, duck: Any) -> LiveBody:
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    url = f"remote://127.0.0.1:{duck.bridge.port}"
    return LiveBody(paths, remote=RemoteBodySpec(url, duck.token))


def test_a_real_duck_is_driven_with_its_own_senses_and_no_world(tmp_path: Path, duck: Any) -> None:
    """Before: the page refused a bridge without a world ("drives simulated remote bodies only")."""
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert body.real and body.rehearsal
        assert body._oracle is None and body.home is None and body.body_port is None
        assert body.world() is None
        assert body.host_info == {"ops": ["x_sensors"], "real": True}
        status = body.status()
        assert status["real"] is True and status["remote"]["up"] is True
        assert status["monitor"]["trust"] == "ok" and status["realtime"] == 1.0
        # The 3D view's frames are robotd's joints and contact odometry, nothing else.
        assert _wait_for(lambda: len(body.frames_since(None)) >= 3)
        frame = body.frames_since(None)[-1]
        assert frame["j"] == [round(v, 4) for i, v in enumerate(JOINTS) if i != 9]
        assert frame["p"] == [0.4, -0.2, 0.118]
        w, _, _, z = frame["q"]
        assert math.isclose(2 * math.atan2(z, w), 0.5, abs_tol=1e-3)  # the odometry's heading
        assert body.frame()["source"] == "odometry"
    finally:
        body.stop()


def test_a_real_duck_refuses_every_world_op_with_the_plain_reason(
    tmp_path: Path, duck: Any
) -> None:
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        for op in ("x_world", "x_hand", "x_place", "x_people", "x_battery", "x_place_prop", "pg"):
            with pytest.raises(NoWorld) as refused:
                body.world_call(op, do="pickup")
            assert str(refused.value) == str(msg("livebody.no_world"))
            assert refused.value.op == op
        # Its own sensors are offered as the link offers them: this duck has no camera channel.
        sensors = body.world_call("x_sensors")
        assert sensors["media"]["socket"] is None and sensors["tof"]["fitted"] is False
        assert [str(e) for e in sensors["errors"]] == [str(msg("livebody.real_no_camera"))]
    finally:
        body.stop()


def test_a_real_duck_is_held_only_when_its_own_robotd_says_it_was_picked_up(
    tmp_path: Path, duck: Any
) -> None:
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert _wait_for(lambda: body.frame() is not None)
        assert body.held is False
        duck.robotd.state = {**duck.robotd.state, "policy": "held"}
        duck.robotd.state["safety"] = {"fallen": False, "limp": False, "picked_up": True}
        assert _wait_for(lambda: body.held is True)
        duck.robotd.state["safety"] = {"fallen": False, "limp": False, "picked_up": False}
        assert _wait_for(lambda: body.held is False)
    finally:
        body.stop()


def test_a_real_duck_set_down_goes_back_to_the_posture_duck_core_asked_for(
    tmp_path: Path, duck: Any
) -> None:
    """Before: robotd's pick-up pause made the body ask to sit and forget Duck Core's posture, so
    a duck set down seated stayed seated (a simulated duck's set-down asks for it again)."""
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert _wait_for(lambda: body.frame() is not None)
        body.consume([{"sync": "stand"}])  # Duck Core wants it standing (and it stands)
        lifted = {"fallen": False, "limp": False, "picked_up": True}
        duck.robotd.state = {**duck.robotd.state, "policy": "held", "safety": lifted}
        assert _wait_for(lambda: body.held is True)
        time.sleep(0.3)
        assert duck.robotd.skills == []  # (paused by robotd: nothing asked of it while held)
        down = {"fallen": False, "limp": False, "picked_up": False}
        duck.robotd.state = {**duck.robotd.state, "policy": "sit", "safety": down}
        assert _wait_for(lambda: body.held is False)
        assert _wait_for(lambda: "sit_toggle" in duck.robotd.skills)  # back up to standing
    finally:
        body.stop()


def test_a_simulated_body_without_a_world_is_still_refused(tmp_path: Path, duck: Any) -> None:
    duck.bridge.body_info = {"backend": "simulator"}
    body = _body(tmp_path, duck)
    with pytest.raises(RuntimeError, match="offers no simulated world"):
        body.start(timeout_s=8.0)
    body.stop()


# ── duckmind never turns a real duck on (SAFETY.md) ─────────────────────────────────────────────


def _start_behind(body: LiveBody) -> tuple[threading.Thread, list[BaseException]]:
    """``body.start()`` on a thread of its own (it waits while the duck is not on)."""
    errors: list[BaseException] = []

    def start() -> None:
        try:
            body.start(timeout_s=3.0)
        except BaseException as error:  # noqa: BLE001 - the test reads it
            errors.append(error)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    return thread, errors


def test_a_real_duck_that_is_not_on_is_waited_for_and_never_enabled(
    tmp_path: Path, duck: Any
) -> None:
    """Before: a real duck whose walking policy was not on yet got the operator's ``robot.enable``
    from the brain as it attached, against SAFETY.md ("duckmind never turns the robot on")."""
    refused: list[str] = []
    duck.bridge.log = lambda event, **fields: (
        refused.append(str(fields.get("method"))) if event == "refused_operator" else None
    )
    duck.robotd.state = {**duck.robotd.state, "policy": "off"}
    body = _body(tmp_path, duck)
    starting, errors = _start_behind(body)
    try:
        assert _wait_for(lambda: body.policy != "starting")  # (robotd's first state frame)
        time.sleep(0.6)
        # Nothing but the handshake and the state subscription reached the duck, and the brain
        # did not even ask its bridge for more.
        assert set(duck.robotd.methods) <= {"hello", "robot.subscribe"}, duck.robotd.methods
        assert refused == []
        assert starting.is_alive() and errors == []  # (still waiting for a person)
        assert body.waits_turn_on is True and body.status()["waits_turn_on"] is True
        assert body._operator is None
        # A person turns the duck on with its own control: the brain takes it from there.
        duck.robotd.state = {**duck.robotd.state, "policy": "sit"}
        starting.join(timeout=4.0)
        assert not starting.is_alive() and errors == []
        assert body.waits_turn_on is False and body.status()["waits_turn_on"] is False
        body.consume([{"sync": "stand"}])
        assert _wait_for(lambda: "sit_toggle" in duck.robotd.skills)
        assert not OPERATOR & set(duck.robotd.methods) and refused == []
    finally:
        body.stop()
        starting.join(timeout=4.0)


def test_a_real_duck_gets_no_operator_request_from_any_helper(tmp_path: Path, duck: Any) -> None:
    """Play dead's scripted pose and its wake (``robot.poseJoints``, ``robot.init``,
    ``robot.enable``) go through these helpers: on a real duck each one answers False and sends
    nothing, whatever the caller believes the robotd can do."""
    refused: list[str] = []
    duck.bridge.log = lambda event, **fields: (
        refused.append(str(fields.get("method"))) if event == "refused_operator" else None
    )
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert body.real and body._operator is None and body.pose_joints_ok is False
        body.pose_joints_ok = True  # (a caller that thinks this robotd can lie the duck down)
        assert body.pose_joints([None] * 15, [], gain=160, ramp_s=1.0) is False
        assert body.operator_init() is False
        assert body.operator_enable(True) is False
        assert body.error is None
        time.sleep(0.3)
        assert not OPERATOR & set(duck.robotd.methods) and refused == []
    finally:
        body.stop()


def test_stopping_ends_the_wait_for_a_real_duck_that_is_never_turned_on(
    tmp_path: Path, duck: Any
) -> None:
    duck.robotd.state = {**duck.robotd.state, "policy": "off"}
    body = _body(tmp_path, duck)
    starting, errors = _start_behind(body)
    assert _wait_for(lambda: body.waits_turn_on)
    body.stop()
    starting.join(timeout=4.0)
    assert not starting.is_alive() and body.waits_turn_on is False
    assert len(errors) == 1 and "not turned on" in str(errors[0])
    assert not OPERATOR & set(duck.robotd.methods)


def test_a_simulated_remote_body_that_is_not_up_is_still_enabled(
    tmp_path: Path, duck: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated remote body (``sim-body``) keeps what it had: the brain's operator enables a
    body that is not up yet, through a bridge that says ``backend: simulator``."""
    duck.bridge.body_info = {"backend": "simulator"}
    duck.robotd.state = {**duck.robotd.state, "policy": "off"}
    monkeypatch.setattr(livebody, "SimOracle", lambda port: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(LiveBody, "_sample", lambda self: None)
    monkeypatch.setattr(LiveBody, "_watch", lambda self: None)
    url = f"remote://127.0.0.1:{duck.bridge.port}"
    link = RemoteLink(url, TOKEN)
    link.start(timeout_s=3)
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    body = LiveBody(paths, seated=False, remote=RemoteBodySpec(url, duck.token))
    # (what _start_remote sets from the bridge's words; the world's oracle is a stand-in here)
    body._link, body.body_port, body.home = link, 1, None
    try:
        body._attach(link.robotd_socket, 4.0)
        assert body.real is False
        assert "robot.enable" in duck.robotd.methods and body.policy == "stand"
    finally:
        body.stop()


def _ask(robotd_socket: Path, method: str) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(robotd_socket))
        sock.settimeout(2)
        request = {"jsonrpc": "2.0", "id": 7, "method": method, "params": {}}
        sock.sendall(json.dumps(request).encode() + b"\n")
        return dict(json.loads(sock.makefile("rb").readline()))


def test_sim_body_as_real_is_never_enabled_through_its_bridge(duck: Any) -> None:
    """``sim-body --as-real`` behaves like a real duck's bridge here too: the operator's three
    methods are refused and never reach its robotd."""
    hashes = {"walk_sha256": "w", "sitstand_sha256": "s"}
    duck.bridge.body_info = body_info(SimpleNamespace(policy_hashes=lambda: hashes), as_real=True)
    link = RemoteLink(f"remote://127.0.0.1:{duck.bridge.port}", TOKEN)
    link.start(timeout_s=3)
    try:
        for method in sorted(OPERATOR):
            reply = _ask(link.robotd_socket, method)
            assert "result" not in reply and "real duck" in reply["error"]["message"], reply
        assert _ask(link.robotd_socket, "robot.health")["result"] == {"accepted": True}
        assert not OPERATOR & set(duck.robotd.methods)
        assert "robot.health" in duck.robotd.methods
    finally:
        link.close()


def test_the_real_pose_keeps_the_imu_tilt_and_takes_the_odometry_heading() -> None:
    # 0.2 rad pitch about y, with an IMU heading of 1.0 rad that the odometry says is 0.3
    pitch = 0.2
    tilt = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    yaw = _yaw_quat(1.0)
    imu = [  # yaw ⊗ tilt
        yaw[0] * tilt[0] - yaw[3] * tilt[3],
        yaw[0] * tilt[1] - yaw[3] * tilt[2],
        yaw[0] * tilt[2] + yaw[3] * tilt[1],
        yaw[0] * tilt[3] + yaw[3] * tilt[0],
    ]
    trunk, quat = real_frame_pose((1.0, 2.0, 0.0), 0.3, imu, "sit")
    assert trunk == [1.0, 2.0, 0.06]  # (robotd gave no height: the seated one)
    expect = _yaw_quat(0.3)
    want = [
        expect[0] * tilt[0] - expect[3] * tilt[3],
        expect[0] * tilt[1] - expect[3] * tilt[2],
        expect[0] * tilt[2] + expect[3] * tilt[1],
        expect[0] * tilt[3] + expect[3] * tilt[0],
    ]
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(quat, want, strict=True))
    assert real_frame_pose((0.0, 0.0, 0.0), 0.0, None, "stand")[0][2] == 0.12


def test_sim_body_as_real_offers_what_a_real_bridge_offers() -> None:
    body = SimpleNamespace(
        robotd_socket=Path("/tmp/r.sock"),
        body_port=1234,
        home=None,
        policy_hashes=lambda: {"walk_sha256": "w", "sitstand_sha256": "s"},
        world=lambda: None,
    )
    assert sorted(channels_for(body, as_real=True)) == ["robotd"]
    assert "world" in channels_for(body)
    info = body_info(body, as_real=True)
    assert info["backend"] == "real" and info["rehearsal"] is True
    assert "home" not in info and info["policies"]["walk_sha256"] == "w"


def test_a_remote_bodys_robotd_battery_is_a_real_sense_taken_as_it_is() -> None:
    """Before: the brain's power sense said "model" (simulated stand-in) and added the world
    meter's noise even when the pack was robotd's own reading over the link."""
    from duckmind.world.neuro_senses import SenseHub

    hub = SenseHub()
    world = {"battery": {"percent": 63.0, "volts": 7.6, "charging": False, "source": "robotd"}}
    power = hub._power(5_000, world, None)
    assert power.source == "real" and power.battery == 0.63
    modelled = SenseHub()._power(5_000, {"battery": {"percent": 63.0}}, None)
    assert modelled.source == "model"
    assert SenseHub()._power(5_000, None, None).source == "none"
