"""eye.server: the duck-eye process — the pure core (one look from bytes to integers, P2) and the
socket server duck-body talks to (spec §4.3, §6)."""

from __future__ import annotations

import json
import math
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from duckmind.eye import frames as f
from duckmind.eye import light as eye_light
from duckmind.eye import protocol as p
from duckmind.eye import server as es
from duckmind.eye.camera import q_about_z, q_mul

LEVEL_CAM = (0.5, -0.5, 0.5, -0.5)
W, H = 64, 36  # a small frame keeps the tests quick; K scales to it
EPOCH_OFFSET_NS = 1_790_000_000_000_000_000


def _frame(mean: float = 90.0, sigma: float = 2.0, seed: int = 1) -> bytes:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    y = np.clip(np.rint(np.full((H, W), mean) + rng.normal(0.0, sigma, (H, W))), 0, 255)
    packed = np.full((H, W // 2, 4), 128, dtype=np.uint8)
    packed[:, :, 1] = y[:, 0::2]
    packed[:, :, 3] = y[:, 1::2]
    return packed.tobytes()


def _header(capture_ns: int) -> f.MediaFrameHeader:
    return f.MediaFrameHeader(W, H, "UYVY", W * H * 2, (EPOCH_OFFSET_NS + capture_ns) // 1000, 90)


def _pose(seq: int, t_ns: int, yaw_deg: float = 0.0) -> p.PoseSample:
    return p.PoseSample(
        seq=seq,
        t_ns=t_ns,
        trunk_height_mm=210,
        cam_pos_m=(0.05, 0.0, 0.10),
        cam_quat=q_mul(q_about_z(math.radians(yaw_deg)), LEVEL_CAM),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        head=(0.0, 0.0, math.radians(yaw_deg), 0.0),
    )


class FloorStub:
    name = "floor"

    def __init__(self) -> None:
        self.learned: list[Any] = []

    def run(self, view: f.LookView) -> dict[str, Any]:
        assert view.pose is not None and view.camera_pose is not None
        assert view.planes.y.shape == (W // 4, H // 4)  # upright 36×64 every 4th pixel
        assert view.light["level"] in eye_light.LEVELS
        return {"free_mm": [300, 400, 500, 400, 300], "look": view.look.look_id}

    def learn(self, event: p.EyeLearn) -> int:
        self.learned.append(event)
        return len(event.floor)


class Broken:
    name = "nest"

    def run(self, view: f.LookView) -> dict[str, Any]:
        raise RuntimeError("tag library missing")


class Floaty:
    name = "things"

    def run(self, view: f.LookView) -> dict[str, Any]:
        return {"b_rad": 0.25}


def _core(**tasks: Any) -> es.EyeCore:
    return es.EyeCore(es.EyeConfig(), tasks=tasks)


def _look(**kw: Any) -> p.EyeLook:
    base: dict[str, Any] = dict(
        look_id="L1", tasks=("floor", "nest", "things"), pose_tap_ref=1, not_before_ns=0
    )
    base.update(kw)
    return p.EyeLook(**base)


def _run(core: es.EyeCore, look: p.EyeLook, capture_ns: int, poses, data=None) -> p.EyeResult:
    return core.run_frame(
        look,
        _header(capture_ns),
        data if data is not None else _frame(),
        captured_at_ns=capture_ns,
        poses=poses,
    )


# ── the core ────────────────────────────────────────────────────────────────


def test_one_look_from_bytes_to_integers():
    core = _core(floor=FloorStub(), nest=Broken(), things=Floaty())
    capture = 5_000_000_000
    result = _run(
        core, _look(), capture, [_pose(1, capture - 20_000_000), _pose(2, capture + 20_000_000)]
    )
    assert result.status == "ok" and result.why is None and result.problems() == []
    assert result.captured_at_ns == capture
    assert (
        result.pose is not None and result.pose["interp"] == "lerp" and result.pose["seq"] == [1, 2]
    )
    light = result.results["light"]
    assert light["level"] == "bright" and light["luma"] == 90 and light["settle_ms"] == 0
    assert result.results["floor"] == {"free_mm": [300, 400, 500, 400, 300], "look": "L1"}
    assert result.results["nest"] == {"error": "RuntimeError"}  # the look goes on
    assert result.results["things"] == {"error": "not_integers"}  # no float leaves the eye
    assert result.frame is not None and len(result.frame["sha16"]) == 16
    assert result.frame["w"] == W and result.pv == core.pv and p.PV_PATTERN.match(core.pv)
    line = p.encode(result.to_json())
    assert p.EyeResult.from_json(p.decode(line)).results == result.results


def test_a_head_that_moved_means_retake_then_the_light_alone():
    core = _core(floor=FloorStub())
    capture = 5_000_000_000
    turning = [_pose(1, capture - 20_000_000, 0.0), _pose(2, capture + 20_000_000, 2.0)]
    first = _run(core, _look(), capture, turning)
    assert (first.status, first.why, first.retake_in_ms) == ("retake", "head_moved", 150)
    assert first.results == {} and first.pose is not None and first.pose["moved_mdeg"] == 2000
    assert core.light.level is None  # a retaken frame counts nowhere
    last = _run(
        core,
        _look(attempt=2),
        capture + 400_000_000,
        [_pose(3, capture + 380_000_000, 2.0), _pose(4, capture + 420_000_000, 4.0)],
    )
    assert last.status == "moved" and set(last.results) == {"light"}
    assert core.light.level == "bright"
    none = _run(core, _look(attempt=2, look_id="L2"), capture + 2_000_000_000, [])
    assert (none.status, none.why) == ("moved", "no_pose") and none.pose is None


def test_the_governor_refuses_what_the_heat_does_not_allow():
    core = _core()
    core.on_health(p.EyeHealth(t_ns=0, cpu_temp_c=90.0))
    capture = 5_000_000_000
    poses = [_pose(1, capture - 1), _pose(2, capture + 1)]
    refused = _run(core, _look(kind="scan"), capture, poses)
    assert (refused.status, refused.why) == ("none", "governor")
    assert refused.gov["tier"] == 3 and refused.retake_in_ms == 1000
    safe = _run(
        core,
        _look(kind="glance_down", look_id="L2"),
        capture + 300_000_000,
        [_pose(3, capture + 299_000_000), _pose(4, capture + 301_000_000)],
    )
    assert safe.status == "ok" and safe.gov["safe_only"] is True


def test_the_same_bytes_and_poses_give_the_same_integers_p2():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(9)
    frames = [
        _frame(mean=float(m), sigma=float(s), seed=i)
        for i, (m, s) in enumerate(
            zip(rng.uniform(30, 110, 12), rng.uniform(1, 22, 12), strict=True)
        )
    ]

    def replay() -> list[str]:
        core = _core(floor=FloorStub())
        out = []
        for i, data in enumerate(frames):
            capture = (i + 1) * 1_000_000_000
            poses = [
                _pose(2 * i, capture - 10_000_000, 0.1 * i),
                _pose(2 * i + 1, capture + 30_000_000, 0.1 * i),
            ]
            result = _run(core, _look(look_id=f"L{i}", tasks=("floor",)), capture, poses, data)
            record = result.to_json()
            record.pop("cost_us")  # timings are the only thing allowed to differ
            out.append(json.dumps(record, sort_keys=True))
        return out

    assert replay() == replay()


def test_learning_reaches_the_tasks_that_learn():
    stub = FloorStub()
    core = _core(floor=stub, nest=Broken())
    event = p.EyeLearn("W1", "L1", "clear", ((300, 0), (600, 0)))
    assert core.on_learn(event) == {"floor": 2}
    assert stub.learned == [event]


def test_the_perception_hash_changes_with_the_thresholds():
    a = es.perception_hash({}, eye_light.THRESHOLDS)
    b = es.perception_hash({}, eye_light.Thresholds(60, 3100, 5500))
    assert a != b and p.PV_PATTERN.match(a) and len(a) == 18
    assert es.perception_hash({"floor": FloorStub()}, eye_light.THRESHOLDS) != a


def test_the_eye_vision_tasks_are_found_by_their_shape():
    tasks = es.load_tasks(["floor", "nest", "things", "duck", "nonsense"])
    for name, task in tasks.items():
        assert task.name == name and callable(task.run)
    assert "nonsense" not in tasks


# ── the process over real sockets ─────────────────────────────────────────────────


@pytest.fixture
def folder() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dmeye-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class Body:
    """A stand-in duck-body: streams the pose tap and hands media fds over with each look."""

    def __init__(self, eye_path: Path, media_path: Path, yaw_deg: float = 0.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect(str(eye_path))
        self.media_path = media_path
        self.seq = 0
        self.yaw_deg = yaw_deg
        self.turn_deg_per_sample = 0.0  # the head turns this much between two pose samples
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.buffer = b""
        self.thread = threading.Thread(target=self._tap, daemon=True)

    def send(self, message: dict[str, Any]) -> None:
        with self.lock:
            self.sock.sendall(p.encode(message))

    def read(self) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            assert chunk, "the eye hung up"
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def _tap(self) -> None:
        while not self.stop.is_set():
            self.seq += 1
            self.yaw_deg += self.turn_deg_per_sample
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            self.send(_pose(self.seq, now, self.yaw_deg).to_json())
            time.sleep(0.04)

    def look(self, look: p.EyeLook, *, fd: bool = True) -> dict[str, Any]:
        if fd:
            media = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            media.settimeout(3.0)
            media.connect(str(self.media_path))
            f.say_hello(media)
            with self.lock:
                f.send_with_fd(self.sock, p.encode(look.to_json()), media.fileno())
            media.close()
        else:
            self.send(look.to_json())
        while True:
            message = self.read()
            if message.get("type") in (p.RESULT, p.ERROR):
                return message

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.sock.close()


@pytest.fixture
def rig(folder: Path) -> Iterator[tuple[es.EyeServer, Any, Path]]:
    pytest.importorskip("numpy")
    from duckmind.world import media_shim as ms

    shim = ms.MediaShim(folder / "media.sock", ms.Frames(), log=lambda line: None)
    shim.start()
    stop = threading.Event()
    data = _frame()

    def camera() -> None:
        while not stop.is_set():
            if shim.frames.demand.wait(0.05) and shim.frames.take_request():
                time.sleep(0.03)  # a render's worth
                shim.frames.deliver(ms.Frame(W, H, data, time.time_ns() // 1000))

    thread = threading.Thread(target=camera, daemon=True)
    thread.start()
    server = es.EyeServer(folder / "eye.sock", _core(floor=FloorStub()), log=lambda line: None)
    server.start()
    try:
        yield server, shim, folder
    finally:
        server.close()
        stop.set()
        thread.join(timeout=2)
        shim.close()


def test_duck_body_hands_the_eye_a_frame_and_gets_integers_back(rig):
    server, shim, folder = rig
    body = Body(folder / "eye.sock", shim.path)
    try:
        body.send(p.Hello().to_json())
        hello = p.EyeHello.from_json(body.read())
        assert hello.perception_hash == server.core.pv and "floor" in hello.tasks
        body.thread.start()
        time.sleep(0.2)
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        look = _look(tasks=("floor",), pose_tap_ref=body.seq, not_before_ns=now + 50_000_000)
        answer = p.EyeResult.from_json(body.look(look))
        assert answer.status == "ok", answer
        assert (
            answer.captured_at_ns is not None
            and answer.captured_at_ns >= look.not_before_ns - 50_000_000
        )
        assert answer.pose is not None and answer.pose["interp"] == "lerp"
        assert (
            answer.results["floor"]["look"] == "L1" and answer.results["light"]["level"] == "bright"
        )
        assert answer.cost_us["wait"] >= 30_000 and answer.cost_us["fetch"] > 0
        deadline = time.monotonic() + 2.0  # the shim counts after the bytes went out
        while shim.stats()["frames"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert shim.stats()["hello"] >= 1 and shim.stats()["frames"] == 1
        # a pose sample duck-body could not complete: skipped silently (25 Hz, no error lines)
        body.send({"type": "eye.pose", "seq": 1, "t_ns": 1, "trunk_height_mm": None})
        # no fd: answered, not hung
        missing = body.look(_look(look_id="L2", kind="glance_down"), fd=False)
        assert (missing["status"], missing["why"]) == ("none", "no_fd")
        # a line that is not eye-v1: an error, and the eye stays up
        body.send({"type": "eye.look", "look_id": "bad id"})
        err = body.read()
        while err.get("type") != p.ERROR:
            err = body.read()
        assert err["code"] == "bad_request"
        body.send(p.EyeLearn("W1", "L1", "clear", ((300, 0),)).to_json())
        time.sleep(0.3)  # the governor spaces looks 250 ms apart
        again = body.look(_look(look_id="L3", kind="glance_down", pose_tap_ref=body.seq))
        assert again["status"] == "ok", again
        assert server.core.tasks["floor"].learned[0].walk_id == "W1"
        assert server.stats["ok"] == 2 and server.stats["none"] == 1
        assert server.stats["incomplete_poses"] == 1
    finally:
        body.close()


def test_a_second_client_is_turned_away_and_hello_comes_first(rig):
    server, shim, folder = rig
    body = Body(folder / "eye.sock", shim.path)
    try:
        body.send(p.EyeHealth(t_ns=0).to_json())
        assert body.read()["code"] == "hello"
        other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        other.settimeout(5.0)
        other.connect(str(folder / "eye.sock"))
        reply = json.loads(other.recv(4096).split(b"\n")[0])
        assert reply["type"] == p.ERROR and reply["code"] == "busy"
        other.close()
        body.send({"type": "eye.hello", "api": "eye-v9"})
        assert body.read()["code"] == "api"
    finally:
        body.close()


def test_the_eye_answers_a_look_whose_head_keeps_turning_with_retake(rig):
    server, shim, folder = rig
    body = Body(folder / "eye.sock", shim.path)

    # The head turns 30°/s: 1.2° between samples. The turn is made by the tap itself, sample by sample
    # (a free-running spinner thread beside the tap starved under load, and two samples with the same
    # yaw made the look "ok": seen once in a full run beside another full run, 2026-10-04).
    body.turn_deg_per_sample = 1.2
    try:
        body.send(p.Hello().to_json())
        body.read()
        body.thread.start()
        time.sleep(0.2)
        answer = body.look(_look(tasks=("floor",), pose_tap_ref=body.seq))
        assert (answer["status"], answer["why"]) == ("retake", "head_moved")
        assert answer["retake_in_ms"] == p.RETAKE_IN_MS and answer["results"] == {}
    finally:
        body.close()


def test_tasks_that_share_work_are_bound_and_the_floor_runs_first():
    order: list[str] = []

    class Floor:
        name = "floor"

        def run(self, view: f.LookView) -> dict[str, Any]:
            order.append("floor")
            return {"n": 1}

    class Things:
        name = "things"

        def __init__(self) -> None:
            self.bound: dict[str, Any] | None = None

        def bind(self, tasks: dict[str, Any]) -> None:
            self.bound = tasks

        def run(self, view: f.LookView) -> dict[str, Any]:
            order.append("things")
            return {"n": 2}

    floor, things = Floor(), Things()
    core = _core(floor=floor, things=things)
    assert things.bound is not None and things.bound["floor"] is floor
    capture = 5_000_000_000
    poses = [_pose(1, capture - 10_000_000), _pose(2, capture + 30_000_000)]
    result = _run(core, _look(tasks=("things", "floor")), capture, poses)
    assert result.status == "ok" and order == ["floor", "things"]
    assert list(result.results) == ["light", "floor", "things"]


def test_no_frame_from_the_camera_is_a_timeout(folder):
    from duckmind.world import media_shim as ms

    shim = ms.MediaShim(folder / "media.sock", ms.Frames(), log=lambda line: None)
    shim.start()  # no camera behind it: mediad answers -32603 after 500 ms
    server = es.EyeServer(folder / "eye.sock", _core(), log=lambda line: None).start()
    body = Body(folder / "eye.sock", shim.path)
    try:
        body.send(p.Hello().to_json())
        body.read()
        answer = body.look(_look(kind="glance_down", tasks=("light",)))
        assert (answer["status"], answer["why"]) == ("none", "timeout")
    finally:
        body.close()
        server.close()
        shim.close()
