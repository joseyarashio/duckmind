"""duckmind.bodyhost: the upstream guards, the collision model choice, the stall monitor's
maths and lockstep accounting — fast, no live simulation (one opt-in live smoke test)."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import socketserver
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind import bodyhost
from duckmind.bodyhost import (
    COLLISIONS,
    REVIEWED_SHA256,
    UPSTREAM_SHA256,
    Host,
    HostClient,
    StallMonitor,
    assess,
    check_reviewed,
    check_scene_pair,
    check_upstream,
    mesh_digest,
    scene_for,
    source_digests,
    steps_per_write,
)
from duckmind.livebody import MONITOR_FIELDS, LiveBody, LivePaths, live_choice

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> Path:
    if not MJCF or not Path(MJCF, "scene.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(MJCF)


def _official_body_server() -> Path:
    path = _mjcf().parents[1] / "sim" / "body_server.py"
    if not path.is_file():
        pytest.skip("no official body_server.py next to the MJCF folder")
    return path


# ── the upstream guard ──────────────────────────────────────────────────────────

FAKE_UPSTREAM = '''\
"""A stand-in for the official body server, shaped like it."""
import functools


class World:
    """The physics."""

    def __init__(self):
        self.n = 0

    def step(self, times: int = 1) -> None:
        # batched, like upstream
        for _ in range(times):
            self.n += 1


class Body:
    def _apply_torque(self) -> None:
        pass


class Handler:
    def handle(self) -> None:
        pass

    @staticmethod
    def dispatch(body, request):
        return {}


def run(world, headless):
    """Step in real time."""
    while False:
        pass


@functools.lru_cache(maxsize=None)
def main():
    return 1
'''


def _load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"fake_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_digest_is_the_one_inspect_getsource_gives(tmp_path):
    path = tmp_path / "fake_upstream.py"
    path.write_text(FAKE_UPSTREAM)
    module = _load(path)
    digests = source_digests(FAKE_UPSTREAM, UPSTREAM_SHA256)
    for name, obj in {
        "World.step": module.World.step,
        "Handler.dispatch": module.Handler.dispatch,
        "Handler.handle": module.Handler.handle,
        "Body._apply_torque": module.Body._apply_torque,
        "run": module.run,
        "main": module.main,
    }.items():
        assert digests[name] == hashlib.sha256(inspect.getsource(obj).encode()).hexdigest(), name


def test_a_changed_or_missing_upstream_function_is_named(tmp_path):
    path = tmp_path / "body_server.py"
    path.write_text(FAKE_UPSTREAM)
    pinned = {
        name: digest or ""
        for name, digest in source_digests(FAKE_UPSTREAM, UPSTREAM_SHA256).items()
    }
    assert check_upstream(path, pinned) == []
    changed = FAKE_UPSTREAM.replace("self.n += 1", "self.n += 2").replace("def run(", "def walk(")
    path.write_text(changed)
    problems = check_upstream(path, pinned)
    assert len(problems) == 2
    assert problems[0].startswith("World.step：sha256 ")
    assert problems[1] == "run：找不到這個函式"


def test_the_pinned_upstream_functions_are_the_ones_reviewed():
    assert check_upstream(_official_body_server()) == []


@pytest.mark.parametrize("collision", sorted(COLLISIONS))
def test_every_physics_file_is_the_reviewed_one(collision):
    assert check_reviewed(_official_body_server(), _mjcf(), collision) == []


def test_an_unreviewed_physics_file_is_named_and_the_physics_not_trusted(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "leg.stl").write_bytes(b"solid leg")
    robot = '<mujoco><compiler meshdir="assets"/><asset><mesh file="leg.stl"/></asset></mujoco>'
    (tmp_path / "robot_allcollisions.xml").write_text(robot)
    (tmp_path / "scene_allcollisions.xml").write_text("<mujoco/>")
    server = tmp_path / "body_server.py"
    server.write_text("TIMESTEP = 0.005\n")
    digest = hashlib.sha256
    pinned = {
        "body_server.py": digest(server.read_bytes()).hexdigest(),
        "scene_allcollisions.xml": digest(b"<mujoco/>").hexdigest(),
        "robot_allcollisions.xml": digest(robot.encode()).hexdigest(),
        "meshes": str(mesh_digest(tmp_path / "robot_allcollisions.xml")),
    }
    assert check_reviewed(server, tmp_path, "allcollisions", pinned) == []
    server.write_text("TIMESTEP = 0.002\n")  # a physics constant changed upstream
    (tmp_path / "assets" / "leg.stl").write_bytes(b"solid longer leg")
    problems = check_reviewed(server, tmp_path, "allcollisions", pinned)
    assert len(problems) == 2
    assert "body_server.py" in problems[0] and "還沒有人核對" in problems[0]
    assert "網格" in problems[1]
    (tmp_path / "assets" / "leg.stl").unlink()
    assert mesh_digest(tmp_path / "robot_allcollisions.xml") is None
    host = Host(
        collision="allcollisions", lockstep=True, scene=tmp_path / "scene_allcollisions.xml"
    )
    host.unreviewed = problems
    _lockstep(host.monitor, time.monotonic() - 12.0, time.monotonic())
    report = host.report()
    assert report["trust"] == "suspect" and report["why_zh"][-2:] == problems
    assert set(REVIEWED_SHA256) == {
        "body_server.py",
        "meshes",
        *(name for pair in COLLISIONS.values() for name in pair),
    }


def _fake_official(monkeypatch, file: Path, scenes: Path | None = None) -> None:
    fake = types.ModuleType("mjlab_microduck.sim.body_server")
    fake.__file__ = str(file)
    if scenes is not None:
        fake.SCENES = scenes  # type: ignore[attr-defined]
    package = types.ModuleType("mjlab_microduck")
    sim = types.ModuleType("mjlab_microduck.sim")
    sim.body_server = fake  # type: ignore[attr-defined]
    package.sim = sim  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mjlab_microduck", package)
    monkeypatch.setitem(sys.modules, "mjlab_microduck.sim", sim)
    monkeypatch.setitem(sys.modules, "mjlab_microduck.sim.body_server", fake)


def test_the_host_refuses_to_start_on_a_changed_upstream_and_says_what(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "body_server.py"
    path.write_text(FAKE_UPSTREAM)  # none of the pinned sources
    _fake_official(monkeypatch, path, scenes=tmp_path)
    with pytest.raises(SystemExit) as stopped:
        bodyhost.main(["--headless", "--port", "1"])
    assert stopped.value.code == 3
    said = capsys.readouterr().err
    assert "拒絕啟動" in said
    for name in UPSTREAM_SHA256:
        assert name in said


def test_the_host_refuses_scenes_that_differ_beyond_the_robot(tmp_path, monkeypatch, capsys):
    _write_scenes(tmp_path, friction_allcollisions="0.5 0.005 0.0001")
    _fake_official(monkeypatch, _official_body_server(), scenes=tmp_path)
    with pytest.raises(SystemExit) as stopped:
        bodyhost.main(["--headless", "--port", "1"])
    assert stopped.value.code == 3
    assert "只差碰撞模型" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--headless", "--scene", "other.xml"],
        ["--headless", "--ducks", "2"],
        ["--headless", "--cameras", "a"],
        ["--keyframe", "SIT"],  # a viewer
    ],
)
def test_what_lockstep_cannot_do_is_refused_before_anything_starts(argv, capsys):
    with pytest.raises(SystemExit) as stopped:
        bodyhost.main(argv)
    assert stopped.value.code == 3
    assert "拒絕啟動" in capsys.readouterr().err


# ── the collision model ─────────────────────────────────────────────────────────


def _write_scenes(folder: Path, friction_allcollisions: str = "1 0.005 0.0001") -> None:
    template = """<mujoco model="scene">
    <include file="{robot}" />
    <!-- {comment} -->
    <worldbody>
        <geom name="floor" size="0 0 0.05" type="plane" friction="{friction}" />
    </worldbody>
    <keyframe><key name="SIT" qpos="0 0 0.07 1 0 0 0" /></keyframe>
</mujoco>"""
    for collision, (scene, robot) in COLLISIONS.items():
        friction = friction_allcollisions if collision == "allcollisions" else "1 0.005 0.0001"
        (folder / scene).write_text(
            template.format(robot=robot, comment=collision, friction=friction)
        )
        (folder / robot).write_text("<mujoco/>")


def test_each_collision_model_is_an_official_scene():
    folder = Path("/nowhere")
    assert bodyhost.DEFAULT_COLLISION == "allcollisions"
    assert bodyhost.DEFAULT_LOCKSTEP is True
    assert scene_for(folder, "allcollisions") == folder / "scene_allcollisions.xml"
    assert scene_for(folder, "groundcontact") == folder / "scene.xml"
    with pytest.raises(ValueError):
        scene_for(folder, "walk")


def test_the_scene_pair_check_sees_only_real_differences(tmp_path):
    _write_scenes(tmp_path)
    assert check_scene_pair(tmp_path) == []  # comments differ, nothing else does
    _write_scenes(tmp_path, friction_allcollisions="0.5 0.005 0.0001")
    problems = check_scene_pair(tmp_path)
    assert len(problems) == 1 and "friction" in problems[0]
    (tmp_path / "robot_allcollisions.xml").unlink()
    assert check_scene_pair(tmp_path) == ["robot_allcollisions.xml：找不到"]


def test_the_official_scenes_differ_only_in_the_robot_they_include():
    assert check_scene_pair(_mjcf()) == []


def test_both_collision_models_are_the_same_duck_except_what_collides():
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    folder = _mjcf()
    ground = mujoco.MjModel.from_xml_path(str(scene_for(folder, "groundcontact")))
    whole = mujoco.MjModel.from_xml_path(str(scene_for(folder, "allcollisions")))

    def named(model: Any, kind: Any, values: Any) -> dict[str, Any]:
        return {mujoco.mj_id2name(model, kind, i): values[i] for i in range(len(values))}

    body, joint, actuator = (
        mujoco.mjtObj.mjOBJ_BODY,
        mujoco.mjtObj.mjOBJ_JOINT,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
    )
    for kind, field, rtol in (
        (body, "body_mass", 0.0),
        (body, "body_pos", 0.0),
        (body, "body_ipos", 0.0),
        (body, "body_inertia", 1e-5),  # the jaw's export differs in the 7th digit
        (joint, "jnt_range", 0.0),
        (joint, "jnt_axis", 0.0),
        (actuator, "actuator_gainprm", 0.0),
        (actuator, "actuator_biasprm", 0.0),
        (actuator, "actuator_ctrlrange", 0.0),
        (actuator, "actuator_forcerange", 0.0),
    ):
        a = named(ground, kind, getattr(ground, field))
        b = named(whole, kind, getattr(whole, field))
        assert a.keys() == b.keys(), field
        for name in a:
            assert np.allclose(a[name], b[name], rtol=rtol, atol=0.0), (field, name)
    for field in ("dof_damping", "dof_armature", "dof_frictionloss", "key_qpos", "key_ctrl"):
        assert np.array_equal(getattr(ground, field), getattr(whole, field)), field
    for field in ("timestep", "gravity", "integrator", "cone", "iterations", "impratio"):
        assert np.array_equal(getattr(ground.opt, field), getattr(whole.opt, field)), field
    floor = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") for m in (ground, whole)]
    assert np.array_equal(ground.geom_friction[floor[0]], whole.geom_friction[floor[1]])

    def colliding(model: Any) -> int:
        return sum(
            1
            for g in range(model.ngeom)
            if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
        )

    assert (colliding(ground), colliding(whole)) == (11, 70)


def test_the_live_body_runs_allcollisions_in_lockstep_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DUCKMIND_LIVE_COLLISION", raising=False)
    monkeypatch.delenv("DUCKMIND_LIVE_LOCKSTEP", raising=False)
    monkeypatch.delenv("DUCKMIND_LIVE_HOME", raising=False)
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    # Phase 2: by default the body runs in the world server (default_home), with the same
    # collision model and lockstep; phase 0's body host runs on request.
    world = LiveBody(paths)
    assert (world.collision, world.lockstep) == ("allcollisions", True)
    home = world._body_command(7801)
    assert home[1:3] == ["-m", "duckmind.world.simhome"] and home[-2:] == ["--home", "default_home"]
    assert home[-5:-2] == ["--collision", "allcollisions", "--lockstep"]
    body = LiveBody(paths, home="bodyhost")
    assert (body.collision, body.lockstep) == ("allcollisions", True)
    command = body._body_command(7801)
    assert command[1:3] == ["-m", "duckmind.bodyhost"]
    assert command[-3:] == ["--collision", "allcollisions", "--lockstep"]
    assert command[command.index("--port") + 1] == "7801" and "--headless" in command
    monkeypatch.setenv("DUCKMIND_LIVE_COLLISION", "groundcontact")
    monkeypatch.setenv("DUCKMIND_LIVE_LOCKSTEP", "0")
    other = LiveBody(paths, seated=False, home="bodyhost")
    assert other._body_command(1)[-3:] == ["--collision", "groundcontact", "--no-lockstep"]
    assert other._body_command(1)[other._body_command(1).index("--keyframe") + 1] == "HOME"
    assert live_choice("allcollisions", True) == ("allcollisions", True)  # given beats env
    monkeypatch.setenv("DUCKMIND_LIVE_LOCKSTEP", "maybe")
    with pytest.raises(ValueError):
        live_choice()
    with pytest.raises(ValueError):
        live_choice("walk", True)


def test_status_and_frames_carry_a_monitor_that_is_suspect_until_read(tmp_path, monkeypatch):
    monkeypatch.delenv("DUCKMIND_LIVE_LOCKSTEP", raising=False)
    monkeypatch.delenv("DUCKMIND_LIVE_COLLISION", raising=False)
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    body = LiveBody(paths)
    status = body.status()
    assert {"running", "policy", "fallen", "realtime", "error"} <= set(status)
    monitor = status["monitor"]
    assert set(MONITOR_FIELDS) <= set(monitor)
    assert monitor["trust"] == "suspect" and monitor["why_zh"] == ["物理監看還沒開始"]
    assert (monitor["collision"], monitor["lockstep"]) == ("allcollisions", True)
    body.monitor = {**monitor, "trust": "ok", "why_zh": []}
    body._monitor_at = time.monotonic()
    assert body.monitor_view()["trust"] == "ok"
    body._monitor_at = time.monotonic() - 5.0  # the reader stopped getting answers
    stale = body.monitor_view()
    assert stale["trust"] == "suspect" and "沒有更新" in stale["why_zh"][-1]


# ── the monitor's maths ─────────────────────────────────────────────────────────


def _physics(
    monitor: StallMonitor, start: float, end: float, *, period: float = 0.020, sim: float = 0.0
) -> float:
    """The official loop: 20 ms of world every ``period`` wall seconds; returns the sim time."""
    t = start
    while t < end - 1e-9:
        sim += 0.020
        monitor.advanced(sim, wall=t)
        t += period
    return sim


def _robotd(monitor: StallMonitor, start: float, end: float, *, period: float = 0.020) -> None:
    w = start
    while w < end - 1e-9:
        monitor.wrote(wall=w)
        w += period


def _lockstep(
    monitor: StallMonitor, start: float, end: float, *, period: float = 0.020, sim: float = 0.0
) -> float:
    """robotd writes every ``period``; each write moves the world 20 ms just after."""
    w = start
    while w < end - 1e-9:
        monitor.wrote(wall=w)
        sim += 0.020
        monitor.advanced(sim, wall=w + 0.0003)
        w += period
    return sim


def _judge(
    monitor: StallMonitor, now: float, lockstep: bool
) -> tuple[dict[str, Any], str, list[str]]:
    stats = {"lockstep": lockstep, **monitor.stats(now=now)}
    trust, why = assess(stats)
    return stats, trust, why


def test_a_body_that_keeps_up_is_trusted():
    monitor = StallMonitor()
    _lockstep(monitor, 0.0, 12.0)
    stats, trust, why = _judge(monitor, 12.0, lockstep=True)
    assert (trust, why) == ("ok", [])
    assert stats["max_stall_ms_10s"] == pytest.approx(20.0, abs=0.5)
    assert 49 <= stats["robotd_hz"] <= 51 and stats["robotd_hz_min_10s"] >= 49
    assert stats["realtime"] == pytest.approx(1.0, abs=0.01)
    assert stats["writes"] == stats["advances"] == 600 and stats["stalls_10s"] == 0


def test_a_stall_is_seen_for_ten_seconds_and_remembered_for_sixty():
    monitor = StallMonitor()
    sim = _physics(monitor, 0.0, 5.0)  # the last pass on time: 4.98 s
    for k in range(5):  # the next one 120 ms later, then catching up back to back
        sim += 0.020
        monitor.advanced(sim, wall=5.10 + 0.0002 * k)
    sim = _physics(monitor, 5.12, 7.0, sim=sim)
    _robotd(monitor, 0.0, 7.0)  # robotd wrote on all along
    stats, trust, why = _judge(monitor, 7.0, lockstep=False)
    assert stats["max_stall_ms_10s"] == pytest.approx(120.0, abs=1.0)
    assert stats["stalls_10s"] == 1 and trust == "suspect"
    assert len(why) == 1 and "120 毫秒" in why[0] and "補跑" in why[0]
    _physics(monitor, 7.0, 15.95, sim=sim)
    _robotd(monitor, 7.0, 15.95)
    stats, trust, why = _judge(monitor, 15.95, lockstep=False)
    assert (trust, why) == ("ok", [])
    assert stats["max_stall_ms_60s"] == pytest.approx(120.0, abs=1.0)
    assert stats["stalls_60s"] == 1 and stats["max_stall_ms_10s"] < 25


def test_a_slow_robotd_is_suspect():
    monitor = StallMonitor()
    _physics(monitor, 0.0, 12.0)
    _robotd(monitor, 0.0, 12.0, period=0.025)  # robotd at 40 Hz
    stats, trust, why = _judge(monitor, 12.0, lockstep=False)
    assert 39 <= stats["robotd_hz"] <= 41 and 39 <= stats["robotd_hz_min_10s"] <= 41
    assert trust == "suspect" and len(why) == 1
    assert f"只送了 {stats['robotd_hz_min_10s']} 次指令" in why[0]


def test_a_simulation_behind_the_wall_clock_is_suspect():
    monitor = StallMonitor()
    _lockstep(monitor, 0.0, 12.0, period=0.021)  # robotd a millisecond late every tick
    stats, trust, why = _judge(monitor, 12.0, lockstep=True)
    assert stats["realtime"] == pytest.approx(0.02 / 0.021, abs=0.005)
    assert stats["robotd_hz_min_10s"] >= 46 and stats["max_stall_ms_10s"] < 60
    assert trust == "suspect" and len(why) == 1
    assert "0.95 倍" in why[0] and "電腦太忙" in why[0]


def test_when_robotd_stops_lockstep_physics_stops_and_the_monitor_says_so():
    monitor = StallMonitor()
    sim = _lockstep(monitor, 0.0, 10.0)
    stats, trust, why = _judge(monitor, 13.0, lockstep=True)
    assert stats["since_write_ms"] == pytest.approx(3020.0, abs=1.0)
    assert stats["max_stall_ms_10s"] == pytest.approx(3020.0, abs=1.0)  # the gap still open
    assert stats["sim_time"] == pytest.approx(sim) and stats["realtime"] < 0.75
    assert trust == "suspect" and why == ["控制程式（robotd）已經 3.0 秒沒有送指令，物理停住了"]
    # Without lockstep the world runs on, the servos holding the last targets.
    monitor = StallMonitor()
    _physics(monitor, 0.0, 13.0)
    _robotd(monitor, 0.0, 0.01)
    _, trust, why = _judge(monitor, 13.0, lockstep=False)
    assert trust == "suspect"
    assert why == ["控制程式（robotd）已經 13.0 秒沒有送指令，身體沒人控制（物理照跑）"]


def test_robotd_coming_back_resumes_the_world_and_the_gap_is_reported():
    monitor = StallMonitor()
    sim = _lockstep(monitor, 0.0, 10.0)
    sim = _lockstep(monitor, 12.0, 15.0, sim=sim)  # a new robotd, 2 s later
    stats, trust, why = _judge(monitor, 15.0, lockstep=True)
    assert stats["max_stall_ms_10s"] == pytest.approx(2020.0, abs=1.0)
    assert stats["sim_time"] == pytest.approx(sim) == pytest.approx(13.0)
    assert trust == "suspect" and any("停下來等" in reason for reason in why)
    _lockstep(monitor, 15.0, 22.5, sim=sim)
    stats, trust, why = _judge(monitor, 22.5, lockstep=True)  # the gap left the window
    assert (trust, why) == ("ok", [])
    assert stats["max_stall_ms_60s"] == pytest.approx(2020.0, abs=1.0)


def _stall(monitor: StallMonitor, gap_s: float) -> None:
    """Lockstep for 5 s, one late robotd tick ``gap_s`` after the last, then on time to 7 s."""
    sim = _lockstep(monitor, 0.0, 5.0)
    _lockstep(monitor, 4.98 + gap_s, 7.0, sim=sim)


@pytest.mark.parametrize("gap_ms, flagged", [(120, False), (180, False), (260, True)])
def test_in_lockstep_a_late_tick_is_a_reason_only_past_robotds_own_timeout(gap_ms, flagged):
    """In lockstep a stall is a late robotd tick: the world waits for it and robotd's next
    read is still 20 ms of physics after its write (phase 1: 30/30 rises under 10 % × 100 ms
    stalls). Past robotd's simulator timeout (200 ms) robotd drops the link: not trusted."""
    monitor = StallMonitor()
    _stall(monitor, gap_ms / 1000)
    stats, trust, why = _judge(monitor, 7.0, lockstep=True)
    assert stats["max_stall_ms_10s"] == pytest.approx(gap_ms, abs=1.0)  # always in the numbers
    stalled = [reason for reason in why if "物理停過" in reason]
    assert bool(stalled) is flagged, why
    if flagged:
        assert trust == "suspect" and "沒有偷跑" in stalled[0]
    # The same gap without lockstep (the world caught up, back to back) is always a reason.
    catching_up = StallMonitor()
    sim = _physics(catching_up, 0.0, 5.0)
    for k in range(int(gap_ms // 20)):
        sim += 0.020
        catching_up.advanced(sim, wall=4.98 + gap_ms / 1000 + 0.0002 * k)
    _physics(catching_up, 5.0 + gap_ms / 1000, 7.0, sim=sim)
    _robotd(catching_up, 0.0, 7.0)
    _, trust, why = _judge(catching_up, 7.0, lockstep=False)
    assert trust == "suspect" and any("補跑" in reason for reason in why)


def test_before_robotd_writes_the_body_is_not_trusted_yet():
    _, trust, why = _judge(StallMonitor(), 3.0, lockstep=True)
    assert trust == "suspect" and why == ["控制程式（robotd）還沒開始送指令，物理還沒開始跑"]


def test_the_history_stays_bounded():
    monitor = StallMonitor()
    _lockstep(monitor, 0.0, 400.0)
    assert len(monitor._advances) <= 70 * 50 + 2 and len(monitor._writes) <= 70 * 50 + 2
    stats = monitor.stats(now=400.0)
    assert stats["writes"] == stats["advances"] == 20000
    assert stats["realtime"] == pytest.approx(1.0, abs=0.01)


# ── lockstep accounting (a stand-in for the official module) ────────────────────


def _fake_module(tmp_path: Path) -> types.ModuleType:
    module = types.ModuleType("fake_body_server")
    module.__file__ = str(tmp_path / "body_server.py")

    class Model:
        opt = types.SimpleNamespace(timestep=0.005)
        ngeom = 0
        nbody = 1

    class World:
        def __init__(self) -> None:
            self.model = Model()
            self.data = types.SimpleNamespace(time=0.0)
            self.lock = threading.Lock()
            self.batches: list[int] = []

        def step(self, times: int = 1) -> None:
            with self.lock:
                for _ in range(times):
                    self.data.time += self.model.opt.timestep
            self.batches.append(times)

    class Handler:
        def dispatch(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
            op = request.get("op")
            if op == "read":
                return {"sim_time": body.world.data.time}
            if op == "write":
                if len(request["targets"]) != 15:
                    raise ValueError("expected 15 targets")
                body.targets = request["targets"]
                return {}
            raise ValueError(f"unknown op {op!r}")

    class Body:
        """Just the gain scaling of the official Body (15 actuators, wire order = slot order)."""

        def __init__(self, world: Any) -> None:
            import numpy as np

            self.world = world
            self.to_wire = list(range(15))
            self.actuator_slice = np.arange(15)
            self.torque_on = True
            self.kp = 200.0
            self._gain = np.full(15, 10.0)
            world.model.actuator_gainprm = np.zeros((15, 3))
            world.model.actuator_biasprm = np.zeros((15, 3))
            self._apply_torque()

        def _apply_torque(self) -> None:
            scale = (self.kp / 200.0) if self.torque_on else 0.0
            model = self.world.model
            model.actuator_gainprm[self.actuator_slice, 0] = self._gain * scale
            model.actuator_biasprm[self.actuator_slice, 1] = -self._gain * scale

    def run(world: Any, headless: bool) -> None:
        raise AssertionError("the official real-time loop")

    module.Body = Body  # type: ignore[attr-defined]
    module.World = World  # type: ignore[attr-defined]
    module.Handler = Handler  # type: ignore[attr-defined]
    module.run = run  # type: ignore[attr-defined]
    return module


def test_steps_per_write_is_one_control_period():
    assert steps_per_write(0.005) == 4
    assert steps_per_write(0.002) == 10
    assert steps_per_write(0.05) == 1


def test_in_lockstep_each_write_advances_one_control_period_and_nothing_else_does(tmp_path):
    fake = _fake_module(tmp_path)
    host = Host(
        collision="allcollisions", lockstep=True, scene=tmp_path / "scene_allcollisions.xml"
    )
    host.install(fake)
    assert fake.run is bodyhost.idle_run  # the real-time loop no longer steps
    world = fake.World()
    body = types.SimpleNamespace(world=world, targets=None)
    handler = fake.Handler()
    for _ in range(25):
        assert handler.dispatch(body, {"op": "write", "targets": [0.0] * 15}) == {}
    assert world.batches == [4] * 25 and world.data.time == pytest.approx(0.5)
    for _ in range(10):
        handler.dispatch(body, {"op": "read"})
    report = handler.dispatch(body, {"op": "x_monitor"})
    assert world.data.time == pytest.approx(0.5)  # reads and the monitor move nothing
    assert (report["writes"], report["advances"]) == (25, 25)
    assert report["sim_time"] == pytest.approx(0.5)
    assert (report["collision"], report["lockstep"]) == ("allcollisions", True)
    with pytest.raises(ValueError):  # a refused write is still a tick of the world
        handler.dispatch(body, {"op": "write", "targets": [0.0] * 3})
    assert world.batches == [4] * 26
    report = handler.dispatch(body, {"op": "x_monitor"})
    assert (report["writes"], report["write_errors"], report["advances"]) == (26, 1, 26)


def test_torque_joints_hangs_only_the_named_servos_until_a_full_torque_on(tmp_path):
    fake = _fake_module(tmp_path)
    Host(
        collision="allcollisions", lockstep=True, scene=tmp_path / "scene_allcollisions.xml"
    ).install(fake)
    world = fake.World()
    body = fake.Body(world)
    handler = fake.Handler()
    assert bodyhost.SERVO_IDS[5:8] == (30, 31, 32)  # neck_pitch, head_pitch, head_yaw
    assert handler.dispatch(body, {"op": "torque_joints", "ids": [30, 31, 32], "on": False}) == {}
    gain = world.model.actuator_gainprm[:, 0]
    assert list(gain[5:8]) == [0.0, 0.0, 0.0]
    assert all(gain[i] == 10.0 for i in (0, 4, 8, 14))  # the legs and the rest still hold
    assert all(world.model.actuator_biasprm[i, 1] == -10.0 for i in (0, 14))
    # a gain change (robotd re-writes gains) keeps the hung ones at zero
    body.kp = 160.0
    body._apply_torque()
    assert list(world.model.actuator_gainprm[5:8, 0]) == [0.0, 0.0, 0.0]
    assert world.model.actuator_gainprm[0, 0] == pytest.approx(8.0)
    # the hung servos are listed by id; an unknown id changes nothing
    handler.dispatch(body, {"op": "torque_joints", "ids": [99], "on": False})
    assert body._x_torque_off == {5, 6, 7}
    # a single servo back on, then a whole-duck torque on clears the rest
    handler.dispatch(body, {"op": "torque_joints", "ids": [30], "on": True})
    assert body._x_torque_off == {6, 7}
    assert world.model.actuator_gainprm[5, 0] == pytest.approx(8.0)
    assert world.batches == []  # no physics step: it is not a write
    # (the fake official dispatch knows no 'torque' op; the host clears before delegating)
    with pytest.raises(ValueError):
        handler.dispatch(body, {"op": "torque", "on": True})
    assert body._x_torque_off == set()


def test_without_lockstep_writes_only_set_targets_and_the_official_loop_steps(tmp_path):
    fake = _fake_module(tmp_path)
    official_run = fake.run
    host = Host(collision="groundcontact", lockstep=False, scene=tmp_path / "scene.xml")
    host.install(fake)
    assert fake.run is official_run
    world = fake.World()
    body = types.SimpleNamespace(world=world, targets=None)
    handler = fake.Handler()
    handler.dispatch(body, {"op": "write", "targets": [0.1] * 15})
    assert world.batches == [] and body.targets == [0.1] * 15
    world.step(4)  # what the official loop does every 20 ms: measured on the way
    report = handler.dispatch(body, {"op": "x_monitor"})
    assert (report["writes"], report["advances"], report["lockstep"]) == (1, 1, False)


def test_extra_ops_never_reach_the_official_server(tmp_path):
    fake = _fake_module(tmp_path)
    Host(
        collision="allcollisions", lockstep=True, scene=tmp_path / "scene_allcollisions.xml"
    ).install(fake)
    world = fake.World()
    body = types.SimpleNamespace(world=world, targets=None)
    info = fake.Handler().dispatch(body, {"op": "x_info"})
    assert info["collision"] == "allcollisions" and info["robot_file"] == "robot_allcollisions.xml"
    assert info["lockstep"] is True and info["steps_per_write"] == 4
    assert info["upstream_sha256"] == UPSTREAM_SHA256
    with pytest.raises(ValueError):
        fake.Handler().dispatch(body, {"op": "x_whatever"})
    assert world.batches == []


def test_the_client_reads_the_monitor_on_its_own_connection():
    class Answer(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:
                op = json.loads(raw)["op"]
                answer = {"trust": "ok"} if op == "x_monitor" else {"error": "unknown op"}
                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Answer)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True).start()
    try:
        client = HostClient(server.server_address[1])
        assert client.monitor() == {"trust": "ok"}
        with pytest.raises(RuntimeError):
            client.info()
        client.close()
    finally:
        server.shutdown()
        server.server_close()


# ── the real thing, on request ──────────────────────────────────────────────────


def _live_paths() -> LivePaths:
    paths = LivePaths.from_env()
    if paths is None:
        names = (
            "DUCKMIND_MICRODUCK",
            "DUCKMIND_MICRODUCK_RL",
            "DUCKMIND_POLICIES",
            "DUCKMIND_ROBOTD",
        )
        values = [os.environ.get(name) for name in names]
        if not all(values):
            pytest.skip("set the DUCKMIND_MICRODUCK* / POLICIES / ROBOTD paths")
        from duckmind.sim_live import _installed_ort_library

        paths = LivePaths(*(Path(str(v)).expanduser() for v in values), _installed_ort_library())
    if paths.missing():
        pytest.skip(f"live body needs: {paths.missing()}")
    return paths


@pytest.mark.skipif(
    os.environ.get("DUCKMIND_LIVE_SMOKE") != "1",
    reason="live smoke run (robotd + MuJoCo): set DUCKMIND_LIVE_SMOKE=1",
)
def test_live_smoke_allcollisions_in_lockstep_with_a_monitor():
    pytest.importorskip("mujoco")
    body = LiveBody(_live_paths(), collision="allcollisions", lockstep=True)
    try:
        body.start()  # rises from the seated keyframe, then sits back down
        info = body.host_info or {}
        assert info["robot_file"] == "robot_allcollisions.xml"
        assert info["robot_colliding_geoms"] == 70 and info["lockstep"] is True
        assert info["unreviewed"] == []  # every physics file is the reviewed one
        deadline = time.monotonic() + 20.0
        monitor: dict[str, Any] = {}
        while time.monotonic() < deadline:
            monitor = (body.frame() or {}).get("monitor") or {}
            if monitor.get("realtime") is not None and monitor.get("robotd_hz_min_10s"):
                break
            time.sleep(0.25)
        assert set(MONITOR_FIELDS) <= set(monitor)
        assert (monitor["collision"], monitor["lockstep"]) == ("allcollisions", True)
        # Every physics advance came from one robotd write, 20 ms of world each (a write
        # is counted just before its advance, so a reading can fall between the two).
        assert monitor["advances"] > 0 and 0 <= monitor["writes"] - monitor["advances"] <= 1
        assert monitor["sim_time"] == pytest.approx(0.02 * monitor["advances"], abs=1e-6)
        assert body.policy == "sit" and not body.fallen
    finally:
        body.stop()
