"""The world's sensors as a simhome extension (:mod:`duckmind.world.sensors`).

In process: the official body server (by path, never edited) hosted by
:class:`duckmind.world.simhome.SimHome` in ``default_home``, with the suite registered as an
extension — the camera answers ``media.frame`` on its socket, the ``tof`` op is the multi-ray
sensor, the ``x_`` ops carry the truth, and (when the official ``tofd`` is built) the daemon
reads the op through the real body port. MuJoCo tests (the render check); the settings and the
light clock run anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import sensors, simhome, tofd_launch

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


# ── settings and the light clock (anywhere) ────────────────────────────────────


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DUCKMIND_SENSORS",
        "DUCKMIND_SIM_TOF",
        "DUCKMIND_SENSORS_DIR",
        "DUCKMIND_MEDIA_QUALITY",
        "DUCKMIND_EYE_CLOCK",
        "DUCKMIND_TOFD",
    ):
        monkeypatch.delenv(name, raising=False)
    default = sensors.Settings.from_env()
    assert (default.eye, default.tof, default.fitted, default.clock) == (True, True, True, True)
    assert default.quality == "720p30" and default.tofd == "" and default.folder is None
    monkeypatch.setenv("DUCKMIND_SENSORS", "eye")
    monkeypatch.setenv("DUCKMIND_SIM_TOF", "0")
    monkeypatch.setenv("DUCKMIND_MEDIA_QUALITY", "360p30")
    monkeypatch.setenv("DUCKMIND_EYE_CLOCK", "off")
    monkeypatch.setenv("DUCKMIND_SENSORS_DIR", "/tmp/x")
    monkeypatch.setenv("DUCKMIND_TOFD", "auto")
    s = sensors.Settings.from_env()
    assert (s.eye, s.tof, s.fitted, s.clock, s.quality, s.tofd) == (
        True,
        False,
        False,
        False,
        "360p30",
        "auto",
    )
    assert s.folder == Path("/tmp/x")
    monkeypatch.setenv("DUCKMIND_SENSORS", "none")
    assert not sensors.Settings.from_env().eye


def test_a_sensors_folder_too_deep_for_a_socket_falls_back_to_a_short_one() -> None:
    """Phase-2 review: a sensors folder whose socket path would not fit sun_path (104 bytes on
    macOS) made the shim raise inside on_ready; the suite now uses its own short folder."""
    assert not sensors._socket_too_long(Path("/private/tmp/dm-live-abc123/sensors"))
    assert sensors._socket_too_long(Path("/private/tmp/" + "d" * 90))


def test_the_light_clock_follows_the_page_and_keeps_now_continuous() -> None:
    clock = sensors.LightClock()
    assert clock.view()["source"] == "wall"
    clock.set({"local": "2026-09-27T19:30:00", "rate": 60.0})
    first = clock.now()
    assert datetime(2026, 9, 27, 19, 30) <= first < datetime(2026, 9, 27, 19, 31)
    time.sleep(0.05)
    assert (clock.now() - first).total_seconds() == pytest.approx(3.0, abs=1.5)
    before = clock.now()
    clock.set({"rate": 1.0})  # only the rate: the time goes on from where it was
    assert abs((clock.now() - before).total_seconds()) < 1.0
    clock.set({"lamps_on": False, "moon": 0.0, "charging": True})
    view = clock.view()
    assert (view["lamps_on"], view["moon"], view["charging"], view["source"]) == (
        False,
        0.0,
        True,
        "set",
    )
    for bad in ({"local": "tonight"}, {"rate": -1}, {"lamps_on": 1}, {"charging": "yes"}):
        with pytest.raises(ValueError):
            clock.set(bad)
    clock.set({"local": "wall"})
    assert clock.view()["source"] == "wall"


def test_the_body_address_is_read_from_the_server_s_arguments() -> None:
    assert sensors.body_address(["--headless", "--port", "8123"]) == ("127.0.0.1", 8123)
    assert sensors.body_address(["--host", "0.0.0.0", "--port=9"]) == ("0.0.0.0", 9)
    assert sensors.body_address([]) == ("127.0.0.1", 7801)


def test_log_requests_are_checked() -> None:
    assert sensors._since_limit({}) == (None, 20)
    assert sensors._since_limit({"since": 4, "limit": 3}) == (4, 3)
    for bad in ({"since": "x"}, {"limit": 0}, {"limit": 501}, {"limit": True}):
        with pytest.raises(ValueError):
            sensors._since_limit(bad)


def test_register_hooks_the_suite_into_the_world_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUCKMIND_SENSORS", raising=False)
    monkeypatch.delenv("DUCKMIND_SIM_TOF", raising=False)
    registry = simhome.Registry()
    suite = sensors.register(registry)
    view = registry.view()
    assert set(view["registered_ops"]) == {"tof", "x_sensors", "x_eye_log", "x_tof_log", "x_lights"}
    assert view["sections"] == ["lights"]
    assert view["hooks"]["edit_spec"] == ["sensors.keep_spec"]
    assert view["hooks"]["on_ready"] == ["sensors.start"]
    assert view["hooks"]["on_stop"] == ["sensors.stop"]
    assert isinstance(suite, sensors.SensorSuite)


# ── in the world server (MuJoCo) ───────────────────────────────────────────────


def _mjcf() -> Path:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(MJCF)


def _official(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    server = _mjcf().parents[1] / "sim" / "body_server.py"
    monkeypatch.syspath_prepend(str(server.parents[2]))
    spec = importlib.util.spec_from_file_location(f"official_bs_{time.monotonic_ns()}", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Running:
    """A world server in this process: the official module, simhome, the sensors."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, folder: Path, **env: str) -> None:
        for name in ("DUCKMIND_SENSORS", "DUCKMIND_SIM_TOF", "DUCKMIND_TOFD", "DUCKMIND_EYE_CLOCK"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("DUCKMIND_SENSORS_DIR", str(folder))
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        self.bs = _official(monkeypatch)
        self.port = _free_port()
        registry = simhome.Registry()
        self.charging: list[bool] = [False]
        registry.snapshot("nest", lambda world: {"charging": self.charging[0]})
        self.suite = sensors.register(registry)
        self.suite.address = ("127.0.0.1", self.port)
        self.host = simhome.SimHome(
            collision="allcollisions",
            lockstep=True,
            mjcf_dir=_mjcf(),
            home="default_home,no_nest",
            registry=registry,
        )
        self.host.install(self.bs)
        world = self.bs.World(self.host.keyframe_scene, 1)
        pose, trunk_z = self.bs.pose_table(self.host.keyframe_scene, "SIT")
        body = self.bs.Body(world, 0)
        body.place(pose, trunk_z, offset_y=0.0)
        world.bodies.append(body)
        self.bs.mujoco.mj_forward(world.model, world.data)
        self.world, self.body = world, body
        self.handler = self.bs.Handler.__new__(self.bs.Handler)
        self.server: Any = None

    def serve(self) -> None:
        """The body port, as body_server.main opens it (robotd, tofd and evaluators share it)."""
        self.server = self.bs.Server(("127.0.0.1", self.port), self.bs.Handler)
        self.server.body = self.body
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def op(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self.bs.Handler.dispatch(self.handler, self.body, request))

    def step(self, writes: int = 5) -> None:
        for _ in range(writes):
            self.op({"op": "write", "targets": [0.0] * 15})

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.host.stop()


@pytest.fixture
def folder() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dmsens-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _fetch(eye_frames, sock):  # type: ignore[no-untyped-def]
    """``eye_frames.fetch`` with a little patience: mediad's own capture timeout is 0.5 s, and the first render on a shared
    GPU (the DGX Spark with other jobs on it, EGL) now and then needs longer (2026-10-09: 2 of 6 file runs, 0 of 3 single runs).
    Only that refusal (-32603, no frame in time) is tried again, three times; every assertion on the frame stays."""
    last = None
    for _ in range(4):
        try:
            return eye_frames.fetch(sock)
        except eye_frames.FrameError as error:
            if error.code != -32603:
                raise
            last = error
    raise last


def test_the_camera_answers_media_frame_and_logs_the_truth(monkeypatch, folder):
    from duckmind.eye import frames as eye_frames

    running = Running(monkeypatch, folder)
    try:
        suite = running.suite
        assert suite.eye.wait_ready(60.0), suite.eye.status()
        running.step(5)
        sim_time = float(running.world.data.time)
        head, data = _fetch(eye_frames, folder / "mediad" / "media.sock")
        assert (head.width, head.height, head.rotate, head.format) == (1280, 720, 90, "UYVY")
        assert len(data) == head.bytes and head.valid_uyvy()
        log = running.op({"op": "x_eye_log", "limit": 5})
        entry = log["frames"][-1]
        assert entry["captured_at_unix_us"] == head.captured_at_unix_us
        assert entry["sim_time"] == pytest.approx(sim_time)
        assert entry["light"]["room"] == "living" and entry["ae"]["exposure_lines"] > 0
        assert entry["camera"]["pos_mm"][2] > 60 and log["status"]["running"]
        assert log["meter"], "the camera meters whether or not anyone looks"
        info = running.op({"op": "x_sensors"})
        assert info["media"]["socket"] == str(folder / "mediad" / "media.sock")
        assert info["media"]["quality"] == "720p30" and info["media"]["stats"]["frames"] == 1
        assert info["eye"]["frames"] == 1 and info["errors"] == []
        assert oct((folder / "mediad" / "media.sock").stat().st_mode & 0o777) == "0o660"
    finally:
        running.stop()
    assert not running.suite.eye.running


def test_the_tof_op_is_the_multi_ray_sensor_and_its_truth_goes_to_x_tof_log(monkeypatch, folder):
    running = Running(monkeypatch, folder, DUCKMIND_SENSORS="tof")
    try:
        frame = running.op({"op": "tof"})
        assert list(frame) == ["rows", "cols", "distance_mm", "status"]
        assert len(frame["distance_mm"]) == 64
        seated = [d for d, s in zip(frame["distance_mm"], frame["status"], strict=True) if s == 5]
        assert seated and max(seated) < 400  # seated, it looks at the floor in front of it
        log = running.op({"op": "x_tof_log", "limit": 1})
        truth = log["frames"][-1]
        assert truth["distance_mm"] == frame["distance_mm"] and len(truth["why"]) == 64
        assert truth["sim_time"] == pytest.approx(float(running.world.data.time))
        assert log["fitted"] is True and log["sensor"]["rays"] == 576
        assert running.suite.eye is None and running.op({"op": "x_eye_log"})["frames"] == []
    finally:
        running.stop()


def test_a_duck_without_the_module_has_no_tof_op(monkeypatch, folder):
    running = Running(monkeypatch, folder, DUCKMIND_SENSORS="eye", DUCKMIND_SIM_TOF="0")
    try:
        with pytest.raises(ValueError, match="ToF"):
            running.op({"op": "tof"})
        assert running.op({"op": "x_tof_log"}) == {"frames": [], "fitted": False}
    finally:
        running.stop()


def test_the_lights_follow_the_clock_the_page_sets_and_fill_x_world(monkeypatch, folder):
    running = Running(monkeypatch, folder, DUCKMIND_SENSORS="eye")
    try:
        assert running.suite.eye.wait_ready(60.0)
        night = running.op({"op": "x_lights", "local": "2026-09-27T21:00:00", "rate": 1.0})
        assert night["clock"]["source"] == "set" and night["clock"]["local"].startswith(
            "2026-09-27T21"
        )
        assert night["state"]["rooms"]["living"]["key"] == "lamps"
        truth = running.op({"op": "x_world"})
        assert truth["lights"]["rooms"]["living"]["key"] == "lamps"
        running.op({"op": "x_lights", "lamps_on": False})
        from duckmind.eye import frames as eye_frames

        _fetch(eye_frames, folder / "mediad" / "media.sock")
        entry = running.op({"op": "x_eye_log", "limit": 1})["frames"][-1]
        assert entry["light"]["key"] in ("moon", "black") and entry["charging"] is False
        running.charging[0] = True  # the nest module says the duck is charging
        _fetch(eye_frames, folder / "mediad" / "media.sock")
        assert running.op({"op": "x_eye_log", "limit": 1})["frames"][-1]["charging"] is True
    finally:
        running.stop()


def _tofd() -> Path | None:
    found = tofd_launch.find_tofd(os.environ.get("DUCKMIND_TOFD") or None)
    parents = Path(MJCF).resolve().parents if MJCF else []
    if found is None and len(parents) > 4:
        built = parents[4] / "microduck" / "target" / tofd_launch.TARGET / "debug" / "tofd"
        found = built if built.is_file() else None
    return found


def test_the_official_tofd_reads_the_world_s_tof_op_through_the_body_port(monkeypatch, folder):
    binary = _tofd()
    if binary is None:
        pytest.skip("the official tofd is not built here (set DUCKMIND_TOFD)")
    running = Running(monkeypatch, folder, DUCKMIND_SENSORS="tof", DUCKMIND_TOFD=str(binary))
    try:
        running.serve()
        suite = running.suite
        assert suite.tofd is not None and suite.tofd.wait_ready(10.0)
        result, frames = tofd_launch.subscribe(suite.tofd.socket_path, timeout_s=3.0)
        assert result["sensor"] == "sim" and result["hz"] == 15
        first = next(frames)
        truth = {
            json.dumps(e["distance_mm"]): e
            for e in running.op({"op": "x_tof_log", "limit": 50})["frames"]
        }
        assert json.dumps(first["distance_mm"]) in truth  # published verbatim
        view = running.op({"op": "x_sensors"})["tofd"]
        assert view["running"] and view["fitted"] and view["stream"]["sensor"] == "sim"
    finally:
        running.stop()
    assert not running.suite.tofd.running
