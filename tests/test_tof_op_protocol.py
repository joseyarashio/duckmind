"""P1 for the ToF: the world's ``tof`` op is what the official ``tofd --sim`` reads, and the
official ``tofd`` (built from the checkout, never edited) publishes it unchanged.

* **The contract, from upstream by path**: our op's answer fits ``tofd``'s ``SimDepth`` (its
  fields and their Rust types: ``rows: u8``, ``cols: u8``, ``distance_mm: Vec<i16>``, ``status:
  Vec<u8>``); the grid and status constants are ``tof/src/lib.rs``'s; the reach, the field of
  view and the range noise are the official ``sim/tof.py``'s (run here; MuJoCo); the launcher's
  socket is ``duck-ipc-proto``'s — with digests of the code this depends on (``tofd``'s
  simulated-body reader: its handshake line, its 500 ms read timeout, its ``{"op":"tof"}``
  request; and ``sim/tof.py``), so an upstream change fails here with the file named.
* **Our op** (:mod:`duckmind.world.tof_model`; MuJoCo): exactly those four fields, their
  types and ranges, on the official scene.
* **The real daemon** (``DUCKMIND_TOFD`` or the checkout's ``target/…/tofd``, skipped when not
  built): ``tofd --sim --hz 15`` against a stand-in world answers ``tof.stream`` with ``accepted,
  sensor "sim", rows 8, cols 8, hz 15`` and publishes the world's frames verbatim, in order, at
  about 14 frames a second (it sleeps a whole period after each answer); without the module
  (owner decision F2) the same daemon reports the sensor unavailable and sends nothing; and its
  guardian stops it when the process that started it is killed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import tof_model, tofd_launch


def _upstream() -> Path | None:
    raw = os.environ.get("DUCKMIND_UPSTREAM")
    if raw:
        return Path(raw)
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if mjcf:
        parents = Path(mjcf).expanduser().resolve().parents
        if len(parents) > 4 and (parents[4] / "microduck").is_dir():
            return parents[4]
    return None


UP = _upstream()
needs_upstream = pytest.mark.skipif(
    UP is None or not (UP / "microduck" / "tof" / "src" / "main.rs").is_file(),
    reason="needs the official checkout (DUCKMIND_MICRODUCK_MJCF)",
)
# The upstream code the op depends on, as reviewed on 2026-09-27 (microduck 0.15.0, a9ec4b2;
# microduck_rl cb70b792). A change fails with the file named: read it, then re-pin.
PINNED = {
    "microduck/tof/src/main.rs#sim": (
        "50fcc9d1cb880bffcbdf9309e4350adb8fd97e5d9c6ededf6b23706a66ef7ec7"
    ),
    "microduck_rl/src/mjlab_microduck/sim/tof.py": (
        "256c06bac5ac64a02f3e065785bd70314b616c58da3cb94f93484c9e00d42663"
    ),
}


def _text(key: str) -> str:
    assert UP is not None
    path, _, part = key.partition("#")
    text = (UP / path).read_text()
    if part == "sim":
        start = text.index("/// Frames from a simulated body")
        end = text.index("/// Try the named bus and address")
        return text[start:end]
    return text


def _tofd_binary() -> Path | None:
    found = tofd_launch.find_tofd(os.environ.get("DUCKMIND_TOFD") or None)
    if found is None and UP is not None:
        built = UP / "microduck" / "target" / tofd_launch.TARGET / "debug" / "tofd"
        found = built if built.is_file() else None
    return found


TOFD = _tofd_binary()
needs_tofd = pytest.mark.skipif(
    TOFD is None,
    reason="the official tofd is not built here (set DUCKMIND_TOFD, or build it: "
    "cargo build --offline -p tof --bin tofd --target aarch64-apple-darwin)",
)


# ── the contract, from upstream ─────────────────────────────────────────────────


def _const(text: str, name: str) -> int:
    found = re.search(rf"pub const {name}: \w+ = (\d+);", text)
    assert found, f"{name} not found upstream"
    return int(found.group(1))


@needs_upstream
def test_the_grid_and_statuses_are_the_tof_crate_s() -> None:
    assert UP is not None
    lib = (UP / "microduck" / "tof" / "src" / "lib.rs").read_text()
    assert (_const(lib, "ROWS"), _const(lib, "COLS")) == (tof_model.ROWS, tof_model.COLS)
    valid = re.search(r"pub const STATUS_VALID: \[u8; 2\] = \[(\d+), (\d+)\];", lib)
    assert valid and {int(v) for v in valid.groups()} == {
        tof_model.STATUS_VALID,
        tof_model.STATUS_LARGE_PULSE,
    }
    assert _const(lib, "STATUS_NO_TARGET") == tof_model.STATUS_NO_TARGET


def _official_sim_tof() -> Any:
    """The official ``sim/tof.py``, loaded by path (it imports MuJoCo and numpy)."""
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    assert UP is not None
    path = UP / "microduck_rl" / "src" / "mjlab_microduck" / "sim" / "tof.py"
    spec = importlib.util.spec_from_file_location("official_sim_tof", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_upstream
def test_the_reach_and_the_range_noise_are_the_official_simulator_s() -> None:
    """The official ``sim/tof.py`` and ours look at the same wall, near and far: the same grid,
    reach and statuses, the same whole millimetres, and the same spread of the readings (its
    sigma grows with the range: 3 mm plus 0.5 % of it; ours must grow the same)."""
    official = _official_sim_tof()
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    assert (official.ROWS, official.COLS) == (tof_model.ROWS, tof_model.COLS)
    assert (official.FOV_DEG, official.MAX_RANGE) == (tof_model.FOV_DEG, tof_model.MAX_RANGE_M)
    assert official.STATUS_VALID == tof_model.STATUS_VALID
    assert official.STATUS_NO_TARGET == tof_model.STATUS_NO_TARGET
    # one ray a zone and no status randomness: what is left is the range noise
    plain = tof_model.TofParams(sub=1, weak_per_m=0.0, weak_incidence=0.0, phantom_p=0.0)
    for wall_m in (0.5, 3.5):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><site name="tof"/>'
            f'<geom type="box" size="0.01 9 9" pos="{wall_m + 0.01} 0 0"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
        theirs = official.Tof(model, site, seed=1)
        ours = tof_model.TofSensor(model, seed=1, params=plain)
        a = np.array([theirs.frame(data)[0] for _ in range(200)], dtype=float)
        b = np.array([ours.frame_of(data)["distance_mm"] for _ in range(200)], dtype=float)
        assert a.min() > 0 and b.min() > 0  # every zone sees the wall in both
        assert b.mean() == pytest.approx(a.mean(), abs=1.0), wall_m
        spread_theirs = float((a - a.mean(axis=0)).std())
        spread_ours = float((b - b.mean(axis=0)).std())
        assert spread_ours == pytest.approx(spread_theirs, rel=0.05), wall_m


@needs_upstream
def test_the_launcher_looks_for_the_socket_where_the_proto_puts_it(tmp_path: Path) -> None:
    """``duck-ipc-proto`` names tofd's socket under ``/run``; the launcher's is the same path
    under its runtime directory (``DUCK_RUNTIME_DIR``)."""
    assert UP is not None
    proto = (UP / "microduck/duck-ipc-proto/src/lib.rs").read_text()
    found = re.search(r'pub const TOF: &str = "/run/([^"]+)";', proto)
    assert found, "the TOF socket path is not in duck-ipc-proto any more"
    launch = tofd_launch.TofdLaunch("tofd", runtime_dir=tmp_path, body=None)
    assert launch.socket_path == tmp_path / found.group(1)


@needs_upstream
def test_the_upstream_code_the_op_depends_on_is_as_reviewed() -> None:
    changed = {
        key: hashlib.sha256(_text(key).encode()).hexdigest()
        for key, want in PINNED.items()
        if hashlib.sha256(_text(key).encode()).hexdigest() != want
    }
    assert not changed, (
        f"upstream changed since the tof op was checked against it: {sorted(changed)}; read "
        "the change, adjust src/duckmind/world/tof_model.py, then update PINNED here"
    )


# ── our op on the official scene (MuJoCo) ──────────────────────────────────────


def _scene() -> Path:
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not mjcf or not Path(mjcf, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(mjcf, "scene_allcollisions.xml")


def _sensor_on_the_official_scene() -> tuple[Any, Any]:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    model = mujoco.MjModel.from_xml_path(str(_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "HOME")
    )
    mujoco.mj_forward(model, data)
    return tof_model.TofSensor(model, seed=3), data


def test_the_op_answers_exactly_simdepth() -> None:
    sensor, data = _sensor_on_the_official_scene()
    for _ in range(5):
        frame = sensor.frame_of(data)
        assert list(frame) == ["rows", "cols", "distance_mm", "status"]
        assert frame["rows"] == 8 and frame["cols"] == 8
        assert len(frame["distance_mm"]) == len(frame["status"]) == 64
        assert all(type(v) is int and 0 <= v <= 32767 for v in frame["distance_mm"])
        assert all(type(v) is int and 0 <= v <= 255 for v in frame["status"])
        assert set(frame["status"]) <= {4, 5, 9, 13, 255}
        assert json.loads(json.dumps(frame)) == frame
        for d, s in zip(frame["distance_mm"], frame["status"], strict=True):
            assert (d == 0) == (s == 255)


RUST_INTS = {"u8": (0, 255), "i16": (-32768, 32767)}


@needs_upstream
def test_the_op_s_answer_fits_the_simdepth_tofd_reads_it_into() -> None:
    """``tofd --sim`` deserializes the op's answer into its ``SimDepth``: the fields and their
    Rust types are read from its source, and a frame of ours must fit every one of them."""
    main = _text("microduck/tof/src/main.rs#sim")
    body = main[main.index("struct SimDepth {") :]
    fields = re.findall(r"(\w+): ([\w<>]+),", body[: body.index("}")])
    assert fields, "SimDepth has no fields upstream any more"
    sensor, data = _sensor_on_the_official_scene()
    frame = sensor.frame_of(data)
    assert list(frame) == [name for name, _ in fields]
    for name, kind in fields:
        listed = kind.startswith("Vec<")
        low, high = RUST_INTS[kind[4:-1] if listed else kind]
        values = frame[name] if listed else [frame[name]]
        assert isinstance(frame[name], list) == listed, name
        assert all(type(v) is int and low <= v <= high for v in values), name


# ── the real daemon ─────────────────────────────────────────────────────────────


class _World(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.served: list[dict[str, Any]] = []
        self.hellos: list[dict[str, Any]] = []
        self.lock = threading.Lock()


class _Handler(socketserver.StreamRequestHandler):
    server: _World

    def handle(self) -> None:
        for line in self.rfile:
            request = json.loads(line)
            if request.get("op") == "hello":
                self.server.hellos.append(request)
                answer: dict[str, Any] = {}
            elif request.get("op") == "tof":
                with self.server.lock:
                    n = len(self.server.served)
                    answer = {
                        "rows": 8,
                        "cols": 8,
                        "distance_mm": [(n * 64 + i) % 4000 for i in range(64)],
                        "status": [(5, 9, 4, 13, 255)[(n + i) % 5] for i in range(64)],
                    }
                    self.server.served.append(answer)
            else:
                answer = {"error": "unknown op"}
            self.wfile.write(json.dumps(answer).encode() + b"\n")


@pytest.fixture
def folder() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dmtof-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def world() -> Iterator[_World]:
    server = _World()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@needs_tofd
def test_tofd_sim_publishes_the_world_s_frames_verbatim(folder: Path, world: _World) -> None:
    assert TOFD is not None
    port = world.server_address[1]
    launch = tofd_launch.TofdLaunch(TOFD, runtime_dir=folder, body=("127.0.0.1", port))
    launch.start()
    try:
        assert launch.wait_ready(10.0), (folder / "tofd.log").read_text()
        result, frames = tofd_launch.subscribe(launch.socket_path, timeout_s=3.0)
        assert result == {"accepted": True, "sensor": "sim", "rows": 8, "cols": 8, "hz": 15}
        got: list[dict[str, Any]] = []
        started = time.monotonic()
        for frame in frames:
            got.append(frame)
            if len(got) == 30:
                break
        elapsed = time.monotonic() - started
    finally:
        launch.stop()
    assert world.hellos and world.hellos[0] == {"op": "hello", "protocol": 1, "joints": 15}
    served = {json.dumps(f["distance_mm"]): f for f in world.served}
    seqs = [f["seq"] for f in got]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    for frame in got:
        assert set(frame) == {"seq", "at_us", "t_ns", "rows", "cols", "distance_mm", "status"}
        origin = served[json.dumps(frame["distance_mm"])]
        assert frame["status"] == origin["status"] and (frame["rows"], frame["cols"]) == (8, 8)
        assert frame["t_ns"] > 0
    rate = (len(got) - 1) / elapsed
    assert 11.0 <= rate <= 15.5, rate  # "--hz 15" sleeps a whole period after each answer


@needs_tofd
def test_a_duck_without_the_module_runs_tofd_and_it_says_unavailable(folder: Path) -> None:
    assert TOFD is not None
    launch = tofd_launch.TofdLaunch(TOFD, runtime_dir=folder, body=None)
    launch.start()
    try:
        assert launch.wait_ready(10.0), (folder / "tofd.log").read_text()
        result, frames = tofd_launch.subscribe(launch.socket_path, timeout_s=1.0)
        assert result["accepted"] is True and "sensor" not in result
        assert result["unavailable"] and (result["rows"], result["cols"], result["hz"]) == (
            8,
            8,
            15,
        )
        with pytest.raises(TimeoutError):
            next(frames)
        status = launch.status()
        assert status["fitted"] is False and status["stream"]["unavailable"]
    finally:
        launch.stop()
    assert not launch.running


@needs_tofd
def test_the_guardian_stops_tofd_when_its_starter_is_killed(folder: Path) -> None:
    assert TOFD is not None
    starter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from duckmind.world import tofd_launch as tl\n"
            f"launch = tl.TofdLaunch({str(TOFD)!r}, runtime_dir={str(folder)!r}, body=None)\n"
            "launch.start()\n"
            "assert launch.wait_ready(10.0)\n"
            "print(launch.process.pid, flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(Path(tofd_launch.__file__).resolve().parents[2])},
    )
    try:
        assert starter.stdout is not None
        guardian = int(starter.stdout.readline())
        children = subprocess.run(
            ["pgrep", "-P", str(guardian)], capture_output=True, text=True, check=False
        ).stdout.split()
        assert len(children) == 1
        tofd = int(children[0])
        os.kill(starter.pid, signal.SIGKILL)
        starter.wait(timeout=5)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                os.kill(tofd, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(tofd, signal.SIGKILL)
            pytest.fail("tofd outlived the process that started it")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.kill(guardian, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    finally:
        if starter.poll() is None:
            starter.kill()
            starter.wait()
