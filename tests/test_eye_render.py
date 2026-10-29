"""The world's camera branch (:mod:`duckmind.world.eye_render`): it meters twice a second
whatever happens, renders only when a reader waits, on the camera's own clock, and logs the
truth of every frame for evaluators only.

The thread's behaviour is tested with a stand-in eye (no GL, runs anywhere); one test renders
through the real camera model on the official scene (MuJoCo, the render check).
"""

from __future__ import annotations

import os
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import eye_render as er
from duckmind.world import media_shim as ms


class FakeCapture:
    def __init__(self, index: int) -> None:
        self.index = index
        self.width, self.height, self.rotate = 4, 2, 90
        self.glow = False
        self.timings = {"render_ms": 1.0, "frame_ms": 2.0}
        self.extras = {"ae": {"exposure_lines": 600}, "wb": [1.0, 1.0, 1.0], "room": "living"}

    def uyvy(self) -> bytes:
        return bytes([128, self.index % 256]) * 8


class FakeEye:
    """What EyeRender needs of a HomeEye, counting what it is asked."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.frame_size = (4, 2)
        self.meters: list[float] = []
        self.captures: list[float] = []
        self.states: list[float] = []
        self.fail_at = fail_at
        self.closed = False
        self.last_light = None
        self.ae = types.SimpleNamespace(
            sensor=None, settled=lambda: True
        )  # patched out by _ae below
        self.awb = types.SimpleNamespace(gains=(1.0, 1.0, 1.0))
        data = types.SimpleNamespace(
            cam_xpos=[[0.1, 0.0, 0.2]], cam_xmat=[[1.0, 0, 0, 0, 1.0, 0, 0, 0, -1.0]]
        )
        self.eye = types.SimpleNamespace(data=data, camera=0)

    def set_state(self, qpos: Any, *, mocap_pos: Any = None, mocap_quat: Any = None) -> None:
        self.states.append(time.monotonic())

    def converge(self, state: Any) -> int:
        return 1

    def meter(self, state: Any) -> float:
        self.meters.append(time.monotonic())
        return 90.0

    def capture(self, state: Any, *, index: int | None = None) -> FakeCapture:
        self.captures.append(time.monotonic())
        if self.fail_at is not None and len(self.captures) == self.fail_at:
            raise RuntimeError("a bad frame")
        return FakeCapture(int(index or 0))

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_camera_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(er, "_ae", lambda ae: {"exposure_lines": 600, "settled": True})


class World:
    """A world whose physics time the test moves."""

    def __init__(self) -> None:
        self.sim_time = 0.0

    def snapshot(self) -> er.Snapshot:
        return er.Snapshot([0.0], [], [], self.sim_time, time.time_ns() // 1000, False)


def _render(eye: FakeEye, world: World, **more: Any) -> tuple[er.EyeRender, ms.Frames]:
    frames = ms.Frames()
    render = er.EyeRender(
        open_eye=lambda: eye,
        snapshot=world.snapshot,
        light=lambda snap: "day",
        frames=frames,
        **more,
    )
    render.start()
    assert render.wait_ready(5.0)
    return render, frames


def test_it_meters_twice_a_second_and_renders_nothing_nobody_asked_for():
    eye, world = FakeEye(), World()
    render, frames = _render(eye, world, meter_s=0.1)
    try:
        time.sleep(0.55)
        assert eye.captures == []
        assert 4 <= len(eye.meters) <= 7
    finally:
        render.stop()
    assert eye.closed and not render.running


def test_a_reader_gets_the_next_frame_on_the_camera_s_clock():
    eye, world = FakeEye(), World()
    period = 0.05
    render, frames = _render(eye, world, frame_period_s=period, meter_s=10.0)
    try:
        for _ in range(5):
            asked = time.monotonic()
            frame = frames.next_frame(0.5)
            assert frame is not None and frame.width == 4 and len(frame.data) == 16
            taken = eye.captures[-1]
            assert taken >= asked - 1e-3
            assert taken - asked <= period + 0.03  # at most one period's wait (plus slack)
            phase = (taken - render._phase) / period
            assert abs(phase - round(phase)) * period < 0.012  # on a tick of the clock
            time.sleep(0.013)
        entries = render.entries(limit=10)
        assert [e["seq"] for e in entries] == [0, 1, 2, 3, 4]
        first = entries[0]
        for key in ("captured_at_unix_us", "sim_time", "tick_late_ms", "deliver_ms", "light"):
            assert key in first
        assert first["camera"]["pos_mm"] == [100, 0, 200]
        assert render.entries(since=3) == entries[-1:]
    finally:
        render.stop()


def test_without_the_clock_a_frame_is_rendered_at_once():
    eye, world = FakeEye(), World()
    render, frames = _render(eye, world, clock=False, meter_s=10.0)
    try:
        asked = time.monotonic()
        assert frames.next_frame(0.5) is not None
        assert eye.captures[-1] - asked < 0.02
    finally:
        render.stop()


def test_readers_all_along_still_let_the_camera_meter():
    eye, world = FakeEye(), World()
    render, frames = _render(eye, world, frame_period_s=0.01, meter_s=0.1)
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            frames.next_frame(0.5)

    reader = threading.Thread(target=hammer)
    reader.start()
    try:
        time.sleep(0.6)
    finally:
        stop.set()
        reader.join()
        render.stop()
    assert len(eye.captures) > 20 and len(eye.meters) >= 3


def test_a_stopped_world_is_captured_and_said_to_be_frozen():
    eye, world = FakeEye(), World()
    render, frames = _render(eye, world, meter_s=0.05, clock=False)
    try:
        time.sleep(1.2)  # the physics time has not moved for over a second
        assert frames.next_frame(0.5) is not None
        assert render.entries(limit=1)[0]["frozen"] is True and render.status()["frozen"]
        world.sim_time = 1.0
        time.sleep(0.12)
        assert frames.next_frame(0.5) is not None
        assert render.entries(limit=1)[0]["frozen"] is False
    finally:
        render.stop()


def test_one_bad_frame_does_not_stop_the_camera():
    eye, world = FakeEye(fail_at=1), World()
    render, frames = _render(eye, world, clock=False, meter_s=10.0)
    try:
        assert frames.next_frame(0.3) is None  # that capture failed: the reader times out
        assert render.status()["error"]
        assert frames.next_frame(0.5) is not None
        assert render.status()["error"] is None and render.running
    finally:
        render.stop()


def test_an_eye_that_cannot_be_built_says_so_in_plain_words():
    def broken() -> Any:
        raise RuntimeError("no GL here")

    render = er.EyeRender(
        open_eye=broken,
        snapshot=World().snapshot,
        light=lambda snap: None,
        frames=ms.Frames(),
    )
    render.start()
    assert render.wait_ready(5.0) is False
    status = render.status()
    assert not status["running"] and "鴨眼" in status["error"]
    assert "no GL here" in status["error_detail"]
    render.stop()


# ── the real camera (MuJoCo, the render check) ─────────────────────────────────


def test_a_real_frame_through_the_camera_model_on_the_official_scene():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not mjcf or not Path(mjcf, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    from duckmind.world import camera_model as cm
    from duckmind.world import render_scene

    spec = mujoco.MjSpec.from_file(str(Path(mjcf, "scene_allcollisions.xml")))
    model = mujoco.MjSpec.from_file(str(Path(mjcf, "scene_allcollisions.xml"))).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "HOME")
    )
    mujoco.mj_forward(model, data)

    def snapshot() -> er.Snapshot:
        return er.Snapshot(
            data.qpos.copy(),
            data.mocap_pos.copy(),
            data.mocap_quat.copy(),
            1.0,
            time.time_ns() // 1000,
        )

    from datetime import datetime

    frames = ms.Frames()
    render = er.EyeRender(
        open_eye=lambda: render_scene.HomeEye.open(spec, config=cm.CameraConfig(quality="360p30")),
        snapshot=snapshot,
        light=lambda snap: render_scene.default_home_state(datetime(2026, 9, 27, 12, 0)),
        frames=frames,
        frame_period_s=cm.quality_rung("360p30").frame_period_s,
    )
    render.start()
    try:
        assert render.wait_ready(60.0), render.status()
        before = time.time_ns() // 1000
        frame = frames.next_frame(0.5) or frames.next_frame(0.5)
        assert frame is not None
        assert (frame.width, frame.height) == (640, 360) and len(frame.data) == 640 * 360 * 2
        assert frame.captured_at_unix_us >= before - 50_000
        rgb = cm.uyvy_to_rgb(frame.data, 640, 360)
        assert 40 < float(rgb.mean()) < 200  # exposed, not black or white
        entry = render.entries(limit=1)[0]
        assert entry["light"]["key"] == "day" and entry["ae"]["exposure_lines"] > 0
        head = entry["camera"]["pos_mm"]
        assert 150 <= head[2] <= 300  # the head camera's height at HOME
        assert render.status()["meter_ticks"] >= 1
    finally:
        render.stop()
