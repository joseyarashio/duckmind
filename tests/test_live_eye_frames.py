"""Live voice, LV0: the duck-eye frame source for the live connection (``duckmind.simweb.eyeframes``):
the simulator's own eye pictures, at most one a second, each picture once, no network. One test
renders a real frame through the existing eye helpers of ``test_eyecam`` (it needs MuJoCo, GL and
the official checkout, like the other render tests, and skips without them)."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.simweb import eyeframes as EF
from duckmind.simweb import jpeg
from duckmind.simweb.eyecam import EyeCam


def _picture(seed: int, size: tuple[int, int] = (24, 16)) -> bytes:
    """A real (tiny) baseline JPEG from the page's own encoder; ``seed`` makes it a different one."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return jpeg.encode_numpy(rgb, 80)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Eye:
    """What ``EyeCam`` and ``WindowEye`` answer: ``latest()`` with the newest picture."""

    def __init__(self) -> None:
        self.frame: dict[str, Any] | None = None
        self.touched = 0
        self.note = ""

    def publish(self, count: int, wall: float) -> None:
        self.frame = {
            "jpeg": _picture(count),
            "wall": wall,
            "frames": count,
            "width": 24,
            "height": 16,
        }

    def latest(self) -> dict[str, Any] | None:
        return dict(self.frame) if self.frame else None

    def status(self) -> dict[str, Any]:
        return {"note": self.note, "error": None}


def test_the_pictures_the_simulator_makes_come_out_once_each_at_most_a_second() -> None:
    clock, eye = _Clock(), _Eye()
    frames = EF.EyeFrames.from_eye(eye, clock=clock, wall=lambda: 1000.0 + clock.t)
    out: list[EF.EyeFrame] = []
    count = 0
    for step in range(200):  # twenty seconds of the host asking every 0.1 s
        clock.t = step * 0.1
        count += 1  # (the eye renders a new picture every 0.1 s: ten times too many)
        eye.publish(count, clock.t)
        got = frames.poll()
        if got is not None:
            out.append(got)
    assert 18 <= len(out) <= 20, len(out)
    assert [f.seq for f in out] == list(range(1, len(out) + 1))
    assert all(b.wall - a.wall >= 1.0 - 1e-9 for a, b in zip(out, out[1:], strict=False))
    assert len({f.tag for f in out}) == len(out)
    assert frames.sent == len(out) and frames.status()["max_hz"] == 1.0


def test_a_picture_that_has_not_changed_is_not_sent_again() -> None:
    clock, eye = _Clock(), _Eye()
    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    eye.publish(1, 0.0)
    assert frames.poll() is not None
    for step in range(1, 40):
        clock.t = step * 0.5
        assert frames.poll() is None
    assert frames.same >= 30 and frames.sent == 1
    eye.publish(2, 20.0)
    clock.t = 20.0
    assert frames.poll() is not None


def test_a_slow_eye_gives_frames_at_its_own_pace() -> None:
    clock, eye = _Clock(), _Eye()
    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    got = 0
    for step in range(400):  # EyeCam renders every 2 s
        clock.t = step * 0.05
        if step % 40 == 0:
            eye.publish(step // 40 + 1, clock.t)
        got += frames.poll() is not None
    assert got == 10


def test_a_frame_is_a_jpeg_with_its_size_and_its_tag() -> None:
    clock, eye = _Clock(), _Eye()
    frames = EF.EyeFrames.from_eye(eye, clock=clock, wall=lambda: 5.0)
    eye.publish(7, 123.456)
    frame = frames.poll()
    assert frame is not None
    assert frame.jpeg[:2] == b"\xff\xd8" and frame.jpeg[-2:] == b"\xff\xd9"
    assert frame.mime == "image/jpeg" and (frame.width, frame.height) == (24, 16)
    assert frame.tag == "123456-7" and frame.wall == 5.0 and frame.seq == 1


def test_what_is_not_a_picture_is_dropped_and_counted() -> None:
    clock, eye = _Clock(), _Eye()
    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    eye.frame = {"jpeg": b"not a jpeg at all", "wall": 1.0, "frames": 1}
    assert frames.poll() is None and frames.bad == 1
    eye.frame = {"jpeg": b"\xff\xd8" + bytes(EF.MAX_JPEG_BYTES) + b"\xff\xd9", "frames": 2}
    assert frames.poll() is None and frames.bad == 2
    eye.frame = {"jpeg": b"", "frames": 3}
    assert frames.poll() is None and frames.bad == 2


def test_no_picture_yet_says_why_in_the_eyes_own_words() -> None:
    clock, eye = _Clock(), _Eye()
    eye.note = "等物理身體的姿勢"
    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    assert frames.poll() is None and frames.last_why == "等物理身體的姿勢"
    assert frames.status()["why"] == "等物理身體的姿勢" and frames.sent == 0


def test_a_window_eye_is_touched_so_the_worlds_camera_renders() -> None:
    clock, eye = _Clock(), _Eye()
    eye.touch = lambda: setattr(eye, "touched", eye.touched + 1)  # type: ignore[attr-defined]
    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    frames.poll()
    frames.poll()
    assert eye.touched == 2  # (it only renders while somebody keeps asking: owner decision F4)


def test_the_real_eye_camera_without_a_scene_gives_no_frames_and_its_reason(tmp_path) -> None:
    cam = EyeCam(lambda: {}, scene=tmp_path / "nope.xml")  # the render thread cannot start
    cam.start()
    cam.stop()
    frames = EF.EyeFrames.from_eye(cam, clock=_Clock())
    assert frames.poll() is None
    assert "找不到場景檔" in frames.last_why


def test_the_real_eye_camera_hands_its_latest_picture_over() -> None:
    cam = EyeCam(lambda: None, scene=None)
    data = _picture(3)
    cam._latest = {"jpeg": data, "wall": 50.0, "frames": 4, "width": 24, "height": 16}
    frame = EF.EyeFrames.from_eye(cam, clock=_Clock()).poll()
    assert frame is not None and frame.jpeg == data and frame.tag == "50000-4"


def test_the_live_bridges_eye_image_is_a_source_too() -> None:
    clock = _Clock()
    answers: list[tuple[bytes | None, str, str]] = [
        (None, "", "等第一張"),  # (the window just opened)
        (_picture(1), '"1-1"', ""),
        (_picture(1), '"1-1"', ""),  # (the same picture again)
        (_picture(2), '"2-2"', ""),
    ]
    asked: list[int] = []

    def eye_image() -> tuple[bytes | None, str, str]:
        out = answers[len(asked)]
        asked.append(1)
        return out

    frames = EF.EyeFrames.from_bridge(SimpleNamespace(eye_image=eye_image), clock=clock)
    assert frames.poll() is None and frames.last_why == "等第一張"
    first = frames.poll()
    assert first is not None and first.tag == '"1-1"'
    clock.t = 1.5
    assert frames.poll() is None and frames.same == 1
    second = frames.poll()
    assert second is not None and second.tag == '"2-2"' and second.seq == 2


def test_a_source_without_tags_is_told_apart_by_its_bytes() -> None:
    clock = _Clock()
    pictures = [_picture(1), _picture(1), _picture(2)]
    frames = EF.EyeFrames(lambda: (pictures.pop(0), "", ""), clock=clock)
    assert frames.poll() is not None
    clock.t = 2.0
    assert frames.poll() is None and frames.same == 1
    assert frames.poll() is not None


def test_more_than_one_frame_a_second_is_not_allowed_and_less_is_kept() -> None:
    assert EF.EyeFrames(lambda: (None, "", ""), max_hz=10.0).period_s == 1.0
    assert EF.EyeFrames(lambda: (None, "", ""), max_hz=0.5).period_s == 2.0


def test_the_generator_sleeps_between_frames_and_stops_on_request() -> None:
    clock, eye = _Clock(), _Eye()
    stop = threading.Event()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.t += seconds
        eye.publish(int(clock.t * 10) + 1, clock.t)

    frames = EF.EyeFrames.from_eye(eye, clock=clock)
    got = []
    for frame in frames.frames(stop, sleep=sleep, poll_s=0.1):
        got.append(frame)
        if len(got) == 4:
            stop.set()
    assert len(got) == 4 and slept and max(slept) == 0.1
    assert clock.t == pytest.approx(3.0, abs=0.35)  # four frames: about three seconds of sim time


def test_a_real_rendered_frame_of_the_eye_goes_through_unchanged() -> None:
    """The existing eye helpers (tests/test_eyecam.py): render one frame of the head camera and
    hand it over like the live connection would take it. Skips without MuJoCo and the checkout."""
    from datetime import datetime

    from test_eyecam import _needs_render, _renderer

    from duckmind.world import camera_model as cm
    from duckmind.world import lights

    scene = _needs_render()
    renderer = _renderer(scene)
    try:
        pose = cm.keyframe_pose(renderer.eye, "STAND")
        state = lights.light_state(datetime(2026, 9, 27, 12, 0))
        renderer.converge(pose, state)
        out = renderer.frame(pose, state)
    finally:
        renderer.close()
    clock = _Clock()
    frames = EF.EyeFrames.from_eye(SimpleNamespace(latest=lambda: out), clock=clock)
    frame = frames.poll()
    assert frame is not None and frame.jpeg == out["jpeg"]
    assert (frame.width, frame.height) == (360, 640)
    assert frame.jpeg[:2] == b"\xff\xd8" and len(frame.jpeg) < EF.MAX_JPEG_BYTES
    clock.t = 0.5
    assert frames.poll() is None  # (the same picture is not sent twice, nor within a second)
