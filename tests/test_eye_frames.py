"""eye.frames v1: the fd hand-over, the clock bridge, pose pairing and the one-pass sampler
(spec §4.3 steps 3–6)."""

from __future__ import annotations

import math
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from duckmind.eye import frames as f
from duckmind.eye.camera import q_about_z, q_mul
from duckmind.eye.protocol import PoseSample

LEVEL_CAM = (0.5, -0.5, 0.5, -0.5)  # sensor → trunk, an unrotated level mount (x right, y down)


def _uyvy(width: int, height: int, seed: int = 0) -> bytes:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, width * height * 2, dtype=np.uint8).tobytes()


# ── the sampler ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_the_one_pass_sampler_is_the_upright_frame_every_fourth_pixel(rotate):
    np = pytest.importorskip("numpy")
    w, h = 64, 36
    data = _uyvy(w, h, rotate)
    q = f.sample_quarter(data, w, h, rotate)
    assert (q.y == f.upright_luma(data, w, h, rotate, step=4)).all()
    y, u, v = f._planes(data, w, h)
    u_full, v_full = np.repeat(u, 2, axis=1), np.repeat(v, 2, axis=1)
    assert (q.u == f._turn(u_full, rotate)[::8, ::8]).all()
    assert (q.v == f._turn(v_full, rotate)[::8, ::8]).all()
    assert q.u.shape == q.y[::2, ::2].shape
    rgb = q.rgb_at_chroma()
    assert rgb.shape == (*q.u.shape, 3)


def test_the_sampler_at_720p_is_180_by_320_and_90_by_160():
    data = _uyvy(1280, 720)
    q = f.sample_quarter(data, 1280, 720, 90)
    assert q.y.shape == (320, 180) and q.u.shape == (160, 90) and q.v.shape == (160, 90)
    with pytest.raises(f.FrameError):
        f.sample_quarter(data[:-2], 1280, 720, 90)


def test_the_roi_cut_is_full_resolution_upright():
    w, h = 64, 36
    data = _uyvy(w, h, 5)
    full = f.upright_luma(data, w, h, 90)
    assert (f.roi_luma(data, w, h, 90, (3, 10, 20, 31)) == full[10:31, 3:20]).all()
    assert (f.roi_luma(data, w, h, 90, (-5, -5, 4, 4)) == full[0:4, 0:4]).all()  # clipped
    assert f.roi_luma(data, w, h, 90, (30, 30, 30, 40)).size == 0
    assert (f.roi_luma(data, w, h, 90, (0, 0, 36, 64), step=2) == full[::2, ::2]).all()
    assert len(f.frame_sha16(data)) == 16


# ── the clock bridge ───────────────────────────────────────────────────────────


def _head(unix_us: int) -> f.MediaFrameHeader:
    return f.MediaFrameHeader(1280, 720, "UYVY", 1280 * 720 * 2, unix_us, 90)


def test_the_wall_clock_stamp_lands_on_monotonic():
    offset = 1_790_000_000_000_000_000  # REALTIME − MONOTONIC
    before = f.ClockPair(offset + 1_000_000_000, 1_000_000_000)
    after = f.ClockPair(offset + 1_080_000_000, 1_080_000_000)
    bridge = f.ClockBridge(before, after)
    assert bridge.steady and bridge.offset_ns == offset
    captured = 1_030_000_000  # 30 ms after the request
    assert bridge.capture_ns(_head((offset + captured) // 1000)) == captured
    # a stamp after the bytes arrived, or long before the request, is not this frame's
    assert bridge.capture_ns(_head((offset + 1_090_000_000) // 1000)) is None
    assert bridge.capture_ns(_head((offset + 900_000_000) // 1000)) is None
    # the wall clock stepped during the look: nothing is trusted
    stepped = f.ClockBridge(before, f.ClockPair(offset + 1_080_000_000 + 5_000_000, 1_080_000_000))
    assert not stepped.steady and stepped.capture_ns(_head((offset + captured) // 1000)) is None
    assert f.wait_ns(100, 40) == 60 and f.wait_ns(100, 140) == 0


# ── pose pairing ───────────────────────────────────────────────────────────────


def _sample(seq: int, t_ms: int, yaw_deg: float = 0.0, height: int = 210) -> PoseSample:
    turn = q_about_z(math.radians(yaw_deg))  # the head turned about the trunk's up axis
    return PoseSample(
        seq=seq,
        t_ns=t_ms * 1_000_000,
        trunk_height_mm=height,
        cam_pos_m=(0.05, 0.0, 0.10 + seq * 0.001),
        cam_quat=q_mul(turn, LEVEL_CAM),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        head=(0.0, 0.0, math.radians(yaw_deg), 0.0),
    )


def test_the_pose_is_interpolated_to_the_capture():
    samples = [_sample(1, 1000, 0.0, 200), _sample(2, 1040, 0.8, 220)]
    got = f.pair_pose(samples, 1_010_000_000, now_ns=1_050_000_000)
    assert got.status == "ok" and got.pose is not None
    pose = got.pose
    assert pose.interp == "lerp" and pose.seqs == (1, 2)
    assert pose.trunk_height_mm == 205 and pose.cam_pos_m[2] == pytest.approx(0.10125)
    assert pose.head[2] == pytest.approx(math.radians(0.2))
    assert pose.moved_mdeg == 800
    out = pose.to_json()
    assert out["head_mrad"] == [0, 0, 3, 0] and out["moved_mdeg"] == 800
    assert all(isinstance(v, int) for v in out["cam_quat_e6"])


def test_more_than_a_degree_between_the_samples_means_retake():
    samples = [_sample(1, 1000, 0.0), _sample(2, 1040, 1.2)]
    got = f.pair_pose(samples, 1_020_000_000, now_ns=1_050_000_000)
    assert got.status == "moved" and got.pose is not None and got.pose.moved_mdeg == 1200


def test_waiting_for_the_sample_after_the_capture_then_holding_the_last_one():
    samples = [_sample(1, 960), _sample(2, 1000)]
    capture = 1_010_000_000
    assert f.pair_pose(samples, capture, now_ns=1_020_000_000).status == "pending"
    late = f.pair_pose(samples, capture, now_ns=capture + f.POSE_WAIT_NS)
    assert late.status == "ok" and late.pose is not None and late.pose.interp == "hold"
    assert late.pose.seqs == (2, 2)
    # the head was turning before the capture: held, but not trusted
    turning = [_sample(1, 960, 0.0), _sample(2, 1000, 1.5)]
    assert f.pair_pose(turning, capture, now_ns=capture + f.POSE_WAIT_NS).status == "moved"
    # nothing near the capture at all
    far = [_sample(1, 700), _sample(2, 740)]
    assert f.pair_pose(far, capture, now_ns=capture + f.POSE_WAIT_NS).status == "no_pose"
    assert f.pair_pose([], capture, now_ns=capture + f.POSE_WAIT_NS).status == "no_pose"
    # a gap in the tap: the samples do not bracket it, the nearer one holds it
    gap = [_sample(1, 990), _sample(2, 1200)]
    assert f.pair_pose(gap, capture, now_ns=capture).pose.interp == "hold"


def test_the_tap_keeps_time_order_and_its_length():
    tap = f.PoseTap(keep=5)
    for i in range(10):
        tap.add(_sample(i, 1000 + 40 * i))
    assert [s.seq for s in tap.samples] == [5, 6, 7, 8, 9] and tap.latest_seq() == 9
    tap.add(_sample(99, 1000 + 40 * 9))  # same instant: ignored
    tap.add(_sample(3, 1000 + 40 * 6 + 10))  # a restarted tap: time order kept
    assert [s.t_ns // 1_000_000 for s in tap.samples] == [1200, 1240, 1250]


# ── the hand-over (SCM_RIGHTS) ─────────────────────────────────────────────────


def test_an_fd_rides_on_its_look_line_and_nowhere_else():
    body, eye = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()
    try:
        body.sendall(b'{"type":"eye.pose","n":1}\n')
        f.send_with_fd(body, b'{"type":"eye.look","n":1}\n', w1)
        body.sendall(b'{"type":"eye.pose","n":2}\n{"type":"eye.pose","n":3}\n')
        f.send_with_fd(body, b'{"type":"eye.look","n":2}\n', w2)
        body.close()
        reader = f.FdLineReader(eye)
        got = []
        while (item := reader.read_line()) is not None:
            line, offset = item
            fd = reader.take_fd(offset) if b"eye.look" in line else None
            got.append((line, fd))
        looks = [(line, fd) for line, fd in got if b"eye.look" in line]
        assert [fd is not None for _, fd in got] == [False, True, False, False, True]
        for (_, fd), (read_end, n) in zip(looks, ((r1, b"1"), (r2, b"2")), strict=True):
            os.write(fd, n)  # the received fd is the pipe's write end
            os.close(fd)
            assert os.read(read_end, 1) == n
        assert reader.fds == []
    finally:
        eye.close()
        for fd in (r1, w1, r2, w2):
            try:
                os.close(fd)
            except OSError:
                pass


def test_fds_nobody_took_are_closed():
    body, eye = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    r, w = os.pipe()
    try:
        f.send_with_fd(body, b'{"type":"eye.pose"}\n', w)  # an fd on the wrong line
        body.sendall(b'{"type":"eye.pose"}\n')
        body.close()
        reader = f.FdLineReader(eye)
        first = reader.read_line()
        assert first is not None
        reader.drop_stale(first[1] + len(first[0]))
        second = reader.read_line()
        assert second is not None
        reader.drop_stale(second[1] + len(second[0]))
        assert reader.fds == []
        reader.close()
    finally:
        eye.close()
        os.close(r)
        try:
            os.close(w)
        except OSError:
            pass


class _OneRead(f.FdLineReader):
    """Linux may hand over earlier lines and the fd-carrying line in one read."""

    def __init__(self, reads):
        super().__init__(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        self.reads = list(reads)

    def _recv(self):
        return self.reads.pop(0) if self.reads else (b"", [])


def test_an_fd_behind_earlier_lines_in_the_same_read_reaches_its_look():
    r, w = os.pipe()
    lines = [
        b'{"type":"eye.pose","n":1}\n',
        b'{"type":"eye.pose","n":2}\n',
        b'{"type":"eye.look","n":1}\n',
    ]
    reader = _OneRead([(b"".join(lines), [w]), (b'{"type":"eye.pose","n":3}\n', [])])
    try:
        taken = []
        while (item := reader.read_line()) is not None:
            line, offset = item
            if b"eye.look" in line:
                taken.append(reader.take_fd(offset))
            reader.drop_stale(offset + len(line))  # the server's order
        assert taken == [w] and reader.fds == []
    finally:
        reader.sock.close()
        os.close(r)
        os.close(w)


def test_a_line_too_long_is_refused():
    body, eye = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        body.sendall(b"x" * 200 + b"\n")
        body.close()
        with pytest.raises(ValueError):
            f.FdLineReader(eye, max_line=100).read_line()
    finally:
        eye.close()


# ── the frame on a handed-over media connection ───────────────────────────────


@pytest.fixture
def folder() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dmeye-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_duck_body_says_hello_and_the_eye_asks_for_the_frame_with_id_2(folder):
    from duckmind.world import media_shim as ms

    shim = ms.MediaShim(folder / "media.sock", ms.Frames(), log=lambda line: None)
    shim.start()
    stop = threading.Event()

    def camera() -> None:
        while not stop.is_set():
            if shim.frames.demand.wait(0.05) and shim.frames.take_request():
                shim.frames.deliver(ms.Frame(4, 2, bytes(range(16)), 1_234_567))

    thread = threading.Thread(target=camera, daemon=True)
    thread.start()
    try:
        # duck-body's side: connect, hello, hand the fd over
        media = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        media.settimeout(3.0)
        media.connect(str(shim.path))
        f.say_hello(media)
        body, eye = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        f.send_with_fd(body, b'{"type":"eye.look"}\n', media.fileno())
        media.close()  # duck-body keeps nothing
        reader = f.FdLineReader(eye)
        line, offset = reader.read_line()
        fd = reader.take_fd(offset)
        assert fd is not None
        with socket.socket(fileno=fd) as handed:
            head, data = f.fetch_on(handed, media_id=2, timeout_s=3.0)
        assert (head.width, head.height, head.captured_at_unix_us) == (4, 2, 1_234_567)
        assert data == bytes(range(16))
        deadline = time.monotonic() + 2.0  # the shim counts after the bytes went out
        while shim.stats()["frames"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert shim.stats()["hello"] == 1 and shim.stats()["frames"] == 1
        body.close()
        eye.close()
    finally:
        stop.set()
        thread.join(timeout=2)
        shim.close()


def test_the_eye_s_modules_know_nothing_of_the_simulator():
    """duck-eye runs the same on the board: no world, no page, no physics, no Duck Core
    internals, in any of its modules."""
    import ast

    here = Path(f.__file__).parent
    forbidden = ("duckmind.world", "duckmind.simweb", "mujoco", "mjlab", "duckmind.livebody")
    for path in sorted(here.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith(forbidden), f"eye/{path.name} imports {name}"
                if name.startswith("duckmind.duckcore"):
                    assert name in ("duckmind.duckcore.fixed", "duckmind.duckcore.rng"), name


def test_time_is_never_read_in_the_pure_modules():
    """Clocks live in server.py (and the page's eye window): the step functions take time."""
    import ast

    here = Path(f.__file__).parent
    for name in ("protocol.py", "camera.py", "light.py", "governor.py"):
        tree = ast.parse((here / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "time" and node.attr not in ("perf_counter_ns",):
                    pytest.fail(f"eye/{name} reads time.{node.attr}")
