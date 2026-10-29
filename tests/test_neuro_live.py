"""neuro:live — the neural duck drives the live body in prototype mode (``--brain neuro``):
the driver's tick (senses → brain → motor → body link), its telemetry, and the seam in the
simulator (LiveBridge, the server's flags and ``/api/neuro``) with the normal mode unchanged."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.neuro import contract as C
from duckmind.neuro.livestub import StubBrain, StubMotor, body_frame
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimServer, SimState, make_handler

HOME_JOINTS = (0.3491, 0.3491, 0.0, 0.0)


def robot_state(
    t: float,
    *,
    policy: str = "stand",
    load: float = 45.0,
    fallen: bool = False,
    gravity: tuple[float, float, float] = (0.02, 0.0, -1.0),
) -> RobotState:
    joints = [0.0] * 15
    joints[5:9] = HOME_JOINTS
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = load / 10.0
    return RobotState.from_wire(
        {
            "t_ns": 5 * 10**12 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": fallen, "limp": False, "gravity": list(gravity)},
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.12], "yaw": 0.0},
        }
    )


def stub_driver(seed: int = 7) -> NeuroDriver:
    lanes = Lanes(
        brain=StubBrain(seed=seed), names={"brain": "stub", "senses": "stub", "motor": "stub"}
    )
    return NeuroDriver(NeuroSpec(seed=seed), lanes=lanes)


def drive(driver: NeuroDriver, seconds: float, **kw) -> list[list]:
    return [driver.tick(robot_state(k * 0.02, **kw)) for k in range(int(seconds / 0.02))]


# ── the driver's tick ──


def test_the_brain_clock_starts_at_zero_and_never_goes_back() -> None:
    driver = stub_driver()
    driver.tick(robot_state(0.0))
    driver.tick(robot_state(0.02))
    assert driver._last_t_ms == 20
    driver._t0_ns, driver._t_offset_ms = None, driver._last_t_ms  # a re-attach (new body)
    driver.tick(robot_state(-100.0))
    assert driver._last_t_ms == 20  # the new body's clock restarts; the brain's does not


def test_the_stub_wanders_through_the_body_link_and_the_page_gets_its_telemetry() -> None:
    driver = stub_driver()
    ticks = drive(driver, 40.0)
    sent = [i for tick in ticks for i in tick]
    methods = {i.method for i in sent}
    assert {"robot.head", "robot.mouth"} <= methods
    walked = [i.twist for i in sent if i.method == "robot.move" and i.twist != (0.0, 0.0)]
    assert walked, "the stand-in wanders"
    assert all(0.35 <= vx <= 0.55 and abs(vyaw) <= 1.0 for vx, vyaw in walked)
    assert len({round(vx, 2) for vx, _ in walked}) > 3  # never one constant speed
    telemetry = driver.telemetry(0)
    assert telemetry["lanes"] == {"brain": "stub", "senses": "stub", "motor": "stub"}
    samples = telemetry["samples"]
    assert 390 <= len(samples) <= 410  # 10 Hz
    last = samples[-1]
    assert {"behaviour", "behaviour_zh", "twist", "wanted", "head", "gate", "gate_zh"} <= set(last)
    assert telemetry["vocals"] and telemetry["vocals"][0]["zh"] in ("啾", "嗯？")
    assert telemetry["frame"]["sources"]["camera"] == "none"  # honest: no camera in the stub
    later = driver.telemetry(samples[-10]["seq"])
    assert len(later["samples"]) == 9
    json.dumps(telemetry, ensure_ascii=False)  # the page gets plain JSON


def test_the_same_seed_and_state_stream_give_the_same_intents() -> None:
    a, b = stub_driver(11), stub_driver(11)
    run_a = [[(i.method, dict(i.params)) for i in tick] for tick in drive(a, 20.0)]
    run_b = [[(i.method, dict(i.params)) for i in tick] for tick in drive(b, 20.0)]
    assert run_a == run_b
    other = [[(i.method, dict(i.params)) for i in tick] for tick in drive(stub_driver(12), 20.0)]
    assert other != run_a


def test_a_seated_body_is_stood_up_by_the_link_when_the_stub_wants_to_walk() -> None:
    driver = stub_driver()
    ticks = drive(driver, 30.0, policy="sit", load=25.0)
    dos = [i.params for tick in ticks for i in tick if i.method == "robot.do"]
    assert dos and all(d == {"skill": "sit_toggle"} for d in dos)
    assert not [i for tick in ticks for i in tick if i.method == "robot.move" and i.twist[0] > 0]


def test_the_neuro_duck_spec_reads_a_seed_a_growth_day_or_a_preset() -> None:
    assert NeuroSpec(seed=7).parsed() == (7, None, None)
    assert NeuroSpec(duck="12", seed=7).parsed() == (12, None, None)
    assert NeuroSpec(duck="12:62", seed=7).parsed() == (12, 62, None)
    assert NeuroSpec(duck="calm_snuggler", seed=7).parsed() == (None, None, "calm_snuggler")


def test_the_stub_frame_is_only_what_the_real_doors_give() -> None:
    frame = body_frame(1000, robot_state(1.0), battery=0.62, on_dock=True, charging=True)
    assert isinstance(frame, C.SensorFrame)
    assert frame.sources["tof"] == frame.sources["sound"] == frame.sources["touch"] == "none"
    assert frame.sources["body"] == "real" and frame.power.source == "model"
    assert frame.body.posture == "stand" and abs(frame.body.head[0] - 0.3491) < 1e-6
    assert frame.power.on_dock and frame.power.battery == pytest.approx(0.62)


def test_the_stub_motor_opens_the_beak_once_per_call() -> None:
    motor = StubMotor()
    cmd = C.MotorCommand(
        t_ms=0, behaviour="look_around", bout=1, variant=1, vocal="chirp", vocal_seq=1
    )
    request, vocals = motor.step(0, cmd)
    assert vocals == [{"tag": "chirp", "level": 0.5, "behaviour": "look_around"}]
    assert request.mouth > 0.3
    again, more = motor.step(20, cmd)
    assert more == []  # the same call is voiced once
    later, _ = motor.step(400, cmd)
    assert later.mouth == 0.0


# ── the seam in the simulator ──


class _Body:
    error = None
    host_info = {"ops": ["x_place", "x_world", "x_hand", "x_people"]}
    home = "default_home"
    policy = "sit"
    passive = False

    def __init__(self) -> None:
        self.calls: list = []
        self.consumed: list = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def frame(self):
        return {"t": 1.0, "joints": {}, "trunk": [0.0, 0.0, 0.06], "quat": [1, 0, 0, 0]}

    def world(self):
        return {"hand": {"held": False}, "people": [], "mocap": []}

    def world_call(self, op, **params):
        self.calls.append((op, params))
        return {"ok": True, "eta_s": 1.0}

    def status(self):
        return {"realtime": 1.0, "policy": "sit", "fallen": False, "monitor": {}, "held": False}

    def consume(self, motions):
        self.consumed.extend(motions)


class _Driver:
    def __init__(self) -> None:
        self.attached: list = []
        self.detached = 0
        self.stopped = 0
        self.lanes = type("L", (), {"names": {"brain": "stub"}})()

    def attach(self, body) -> None:
        self.attached.append(body)

    def detach(self) -> None:
        self.detached += 1

    def stop(self) -> None:
        self.stopped += 1

    def view(self):
        return {"enabled": True, "state": "running"}

    def telemetry(self, since: int = 0):
        return {"state": "running", "since": since, "samples": []}


def _bridge(tmp_path):
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes
    bridge.body = _Body()  # type: ignore[assignment]
    return bridge


def test_the_neural_duck_takes_the_body_when_it_is_ready_and_lets_go_on_a_restart(tmp_path):
    bridge = _bridge(tmp_path)
    driver = _Driver()
    bridge.neuro = driver
    bridge._start(bridge.body)  # type: ignore[arg-type]
    assert bridge.ready and driver.attached == [bridge.body]
    assert bridge.snapshot()["neuro"] == {"enabled": True, "state": "running"}
    bridge.restart()
    assert driver.detached == 1
    bridge.stop()
    assert driver.stopped == 1


def test_in_prototype_mode_duck_core_moves_only_the_worlds_hand(tmp_path):
    from duckmind.simweb.live import NEURO_BUSY_ZH

    bridge = _bridge(tmp_path)
    bridge.neuro = _Driver()
    bridge.ready = True
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "act", {"action": "power_on"})
    before = len(session.core.log.events)
    dispatch(session, "act", {"action": "pickup", "params": {}})
    dispatch(session, "advance", {"minutes": 3})
    bridge.after_call(session, before)
    items = [item[2] for item in bridge._due]
    assert items and all(set(item) == {"hand"} for item in items)  # no clip, walk, sync, power
    bridge.resync(session, "start")
    assert not [c for c in bridge.body.calls if c[0] == "x_place"]  # Duck Core's plan: no move
    bridge.skipped(session)
    assert bridge._sync_pending is None
    with pytest.raises(ValueError, match="神經鴨"):
        bridge.demo_walk()
    assert NEURO_BUSY_ZH.startswith("神經鴨模式")


def test_the_normal_mode_still_plays_duck_cores_motions(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.ready = True
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    before = len(session.core.log.events)
    dispatch(session, "act", {"action": "power_on"})
    dispatch(session, "advance", {"minutes": 3})
    bridge.after_call(session, before)
    items = [item[2] for item in bridge._due]
    assert any("sync" in item or "clip" in item or "power" in item for item in items)
    assert "neuro" not in bridge.snapshot()


@pytest.fixture()
def neuro_server(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.ready = True
    bridge.neuro = _Driver()
    state = SimState(Config(seed=7, name="測試"), live=bridge)
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_api_neuro_serves_the_telemetry_only_in_prototype_mode(neuro_server):
    status, body = _get(neuro_server + "/api/neuro?since=12")
    data = json.loads(body)
    assert status == 200 and data["ok"] and data["since"] == 12
    plain = SimServer(("127.0.0.1", 0), make_handler(SimState(Config(seed=7))))
    thread = threading.Thread(target=plain.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(f"http://127.0.0.1:{plain.server_address[1]}/api/neuro")
        assert status == 404 and "--brain neuro" in json.loads(body)["error"]
    finally:
        plain.shutdown()
        plain.server_close()
    body_view = json.loads(_get(neuro_server + "/api/body")[1])
    assert body_view["neuro"]["state"] == "running"


def test_brain_neuro_needs_live_physics() -> None:
    from duckmind.simweb.server import main

    with pytest.raises(SystemExit):
        main(["--brain", "neuro", "--port", "0"])


def test_math_sanity_of_the_stub_brain_rhythms() -> None:
    brain = StubBrain(seed=3)
    frame = body_frame(0, robot_state(0.0))
    speeds = []
    for k in range(3000):
        cmd, tel = brain.step(k * 20, frame)
        if cmd.vx:
            speeds.append(cmd.vx)
        assert tel.reason_zh and math.isfinite(cmd.vyaw)
    assert speeds and min(speeds) >= 0.30


# ── the page's buttons: stimuli into the world, another duck ──


class _WorldBody(_Body):
    def frame(self):
        return {"t": 1.0, "joints": {}, "trunk": [1.0, 0.5, 0.12], "quat": [1, 0, 0, 0]}

    def world(self):
        return {"people": [{"who": "child", "pos_mm": [2000, 500, 0]}], "hand": {"held": False}}


class _StimBody(_WorldBody):
    host_info = {"ops": ["x_place", "x_world", "x_hand", "x_people", "x_neuro_stim"]}


def test_the_pages_bang_and_push_are_as_strong_as_their_words() -> None:
    """The bang is a very loud one (the verification's 100 dB at 1 m, not the world's everyday 90)
    and the push is the world's top force (7 N: 「真的力氣，可能會跌倒」), unless the press says otherwise
    (live check 2026-10-05: at 90 dB / 4 N the bold explorer neither startled nor fled)."""
    driver = stub_driver()
    body = _StimBody()
    driver._body = body
    driver._world, driver._truth = body.world(), body.frame()
    driver.inject("bang", {})
    assert body.calls[-1] == ("x_neuro_stim", {"stim": "bang", "db1m": 100.0})
    driver.inject("push", {})
    assert body.calls[-1] == ("x_neuro_stim", {"stim": "push", "strength": 1.0})
    driver.inject("bang", {"db1m": 70.0})
    driver.inject("push", {"strength": 0.2})
    assert body.calls[-2][1]["db1m"] == 70.0 and body.calls[-1][1]["strength"] == 0.2


def test_a_stimulus_goes_into_the_world_never_into_the_brain() -> None:
    driver = stub_driver()
    body = _WorldBody()
    driver._body = body
    driver._world, driver._truth = body.world(), body.frame()
    patted = driver.inject("pet_head", {})
    assert body.calls[-1] == (
        "x_hand",
        {"do": "touch", "where": "head", "seconds": 3.0, "speed": "slow"},
    )
    assert patted["zh"] == "一隻手摸牠的頭"
    clap = driver.inject("clap", {})["event"]
    assert clap["sound"] == "clap" and clap["pos_mm"][:2] == [2000, 500]  # the person clapped
    assert clap["who"] == "child" and clap["source"] == "model"
    wave = driver.inject("wave", {})["event"]
    assert wave["visual"] == "motion" and wave["pos_mm"][:2] == [1700, 500]  # in front of it
    driver.inject("lights_off", {})
    assert body.calls[-1] == ("x_lights", {"lamps_on": False})
    driver.inject("walk_by", {"who": "child"})
    assert body.calls[-1] == ("x_people", {"do": "come", "who": "child", "to": "duck"})
    assert driver._leaves and driver._leaves[0][1] == "child"
    with pytest.raises(ValueError):
        driver.inject("teleport", {})
    # the senses get the sounds and the wave as world events; the brain never sees them
    driver.tick(robot_state(0.0))
    kinds = [e["kind"] for e in driver.recent_stimuli(driver._last_t_ms)]
    assert kinds == ["clap", "wave", "footsteps"]
    events = driver.telemetry(0)["events"]
    assert [e["stimulus"] for e in events if e["kind"] == "world.stimulus"][:2] == [
        "pet_head",
        "clap",
    ]


def test_switching_ducks_starts_a_new_brain_and_fresh_telemetry(monkeypatch) -> None:
    import duckmind.simweb.neuro_live as NL

    def fake_lanes(spec):
        seed, gd, _ = spec.parsed()
        return Lanes(
            brain=StubBrain(seed=seed or 0),
            names={"brain": "stub"},
            genome={"seed": seed, "gd": gd},
        )

    monkeypatch.setattr(NL, "load_lanes", fake_lanes)
    driver = stub_driver()
    drive(driver, 2.0)
    assert driver.telemetry(0)["seq"] > 0
    answer = driver.switch("21:40")
    assert answer["genome"] == {"seed": 21, "gd": 40}
    assert driver.spec.parsed() == (21, 40, None) and driver.telemetry(0)["seq"] == 0
    drive(driver, 1.0)
    assert driver.telemetry(0)["samples"]


def test_the_neural_duck_starts_on_the_mat_unless_asked_for_its_dock(tmp_path) -> None:
    class Mat:
        x_mm, y_mm = 0, 0

    class Layout:
        mat = Mat()

    bridge = _bridge(tmp_path)
    bridge._layout, bridge._layout_home = Layout(), bridge.body.home
    bridge.neuro = _Driver()
    bridge._start(bridge.body)  # type: ignore[arg-type]
    placed = [c for c in bridge.body.calls if c[0] == "x_place"]
    assert placed and placed[0][1]["x_mm"] == 0 and placed[0][1]["y_mm"] == 0
    assert bridge.neuro.attached == [bridge.body]
    nest = _bridge(tmp_path)
    nest._layout, nest._layout_home = Layout(), nest.body.home
    nest.neuro = _Driver()
    nest.neuro.start_at = "nest"
    nest._start(nest.body)  # type: ignore[arg-type]
    assert not [c for c in nest.body.calls if c[0] == "x_place"]


def test_a_stimulus_goes_to_the_worlds_own_op_when_the_world_has_it() -> None:
    driver = stub_driver()

    class Body(_WorldBody):
        host_info = {"ops": ["x_hand", "x_neuro_stim"]}

        def world_call(self, op, **params):
            self.calls.append((op, params))
            return {"ok": True, "note_zh": "拍手聲從那裡傳過來"}

    body = Body()
    driver._body = body
    answer = driver.inject("clap", {"kind": "clap"})
    assert body.calls == [("x_neuro_stim", {"stim": "clap"})]
    assert answer["note_zh"].startswith("拍手")
    assert driver.telemetry(0)["events"][-1]["stimulus"] == "clap"


def test_the_tof_door_gives_the_senses_and_the_link_duck_bodys_depth_summary() -> None:
    """A duck with the ToF module: tofd's newest frame → duck-body's DS-v1 summary (the sensor's
    pose from robot.state's frames.tof) → Doors.depth and the body link's reflex stops."""
    import math as _m

    driver = stub_driver()

    def with_frames(t: float) -> RobotState:
        base = robot_state(t)
        wire = {
            "t_ns": base.t_ns,
            "policy": "stand",
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": list(base.joints),
            "currents_ma": [4.5] * 15,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            # the ToF on the head, 12 cm up and 5 cm ahead of the trunk origin, looking ahead
            "frames": {
                "tof": {
                    "pos": [0.05, 0.0, 0.12],
                    "quat": [_m.cos(_m.pi / 4), 0.0, _m.sin(_m.pi / 4), 0.0],
                },
                "camera": {"pos": [0.05, 0.0, 0.13], "quat": [1.0, 0.0, 0.0, 0.0]},
            },
        }
        return RobotState.from_wire(wire)

    state = with_frames(0.0)
    assert state.tof is not None
    assert driver._depth_step(state, False) is None  # no tofd frame yet: no depth
    driver._tof_frame = {"t_ns": state.t_ns, "distance_mm": [200] * 64, "status": [5] * 64}
    depth = driver._depth_step(state, False)
    assert depth is not None and depth.t_ns == state.t_ns
    later = with_frames(1.0)
    assert driver._depth_step(later, False) is None  # the same old frame 1 s later: stale


def test_a_recorded_run_replays_to_the_same_commands(tmp_path) -> None:
    """--neuro-record: every tick's frame and command; the same brain on the recorded frames
    gives the same commands (the brain is deterministic for a seed and an input stream)."""
    driver = stub_driver(5)
    path = tmp_path / "run.jsonl"
    driver._record = path.open("w", encoding="utf-8")
    drive(driver, 10.0)
    driver._record.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 500
    fresh = StubBrain(seed=5)
    for row in rows:
        cmd, _ = fresh.step(int(row["t_ms"]), C.SensorFrame.from_json(row["frame"]))
        assert json.loads(json.dumps(cmd.to_json())) == row["cmd"]


def test_a_lane_that_fails_a_tick_holds_the_body_still_and_the_driver_carries_on() -> None:
    class Flaky(StubBrain):
        def step(self, t_ms, frame):
            if 200 <= t_ms < 260:
                raise RuntimeError("a rare bug")
            return super().step(t_ms, frame)

    class Client:
        def __init__(self) -> None:
            self.sent: list = []
            self.states = [robot_state(k * 0.02) for k in range(40)]

        def read_state(self, timeout):
            if timeout == 0.0:
                return None  # (no backlog: one tick at a time, as robotd sends them)
            if not self.states:
                stop.set()
                return None
            return self.states.pop(0)

        def send(self, intent) -> None:
            self.sent.append(intent)

    lanes = Lanes(brain=Flaky(seed=3), names={"brain": "stub"})
    driver = NeuroDriver(NeuroSpec(seed=3), lanes=lanes)
    stop = threading.Event()
    client = Client()
    driver._loop(client, stop)  # type: ignore[arg-type]
    assert driver.stats.errors == 3 and "a rare bug" in (driver.stats.last_error or "")
    assert driver.stats.ticks == 37
    zeros = [i for i in client.sent if i.method == "robot.move" and i.twist == (0.0, 0.0)]
    assert zeros  # the failing ticks held the body still


# ── the flip (real-simulator measurement 2026-10-26): robotd's fall flag rises in every roll ──


class _Seeing(StubBrain):
    """A stand-in brain that keeps what each frame it got said: a fall, a touch, a looming thing, a bump, a thump, a word."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self.fallen: list[bool] = []
        self.fright: list[bool] = []
        self.words: list[bool] = []

    def step(self, t_ms, frame):
        self.fallen.append(bool(frame.body.fallen))
        self.fright.append(
            frame.touch.site is not None
            or frame.camera.motion > 0.0
            or any(s.looming > 0.0 or s.kind in ("motion", "looming") for s in frame.camera.sightings)
            or any(e in frame.events for e in ("blocked", "brushed", "stumbled"))
            or (frame.sound.kind == "bang" and frame.sound.onset > 0.0)
        )
        self.words.append(frame.sound.kind == "voice")
        return super().step(t_ms, frame)


def _rough_senses(doors):
    """The senses say: a sharp touch on the back, the picture streaming, a thing looming, a bump, and every other tick a
    thump (else a word) — what its own roll does to them, and somebody talking."""
    f = body_frame(doors.t_ms, doors.state, None if doors.link is None else doors.link.prox.view, events=("brushed",))
    looming = C.Sighting(kind="looming", bearing=0.0, size=0.3, looming=2.0)
    thump = (doors.t_ms // 20) % 2 == 0
    return replace(
        f,
        touch=C.TouchSense(site="back", pressure=0.9),
        sound=C.SoundSense(level_db=70.0, onset=0.9, kind="bang" if thump else "voice"),
        camera=replace(f.camera, usable=True, motion=0.9, sightings=(looming,)),
    )


def _seeing_driver() -> tuple[NeuroDriver, _Seeing]:
    brain = _Seeing(seed=5)
    lanes = Lanes(brain=brain, names={"brain": "stub", "senses": "stub", "motor": "stub"})
    return NeuroDriver(NeuroSpec(seed=5), lanes=lanes), brain


def test_what_its_own_roll_does_to_its_senses_is_no_fright_until_two_seconds_after() -> None:
    """(real simulator 2026-10-26) right after a told roll the duck retreated into a panic run and fell, and mid-roll a
    startle at its own thump or a withdraw at the floor's touch cut rolls short: the bumps and thumps, the floor on its back
    and the streaming picture of its own roll read as a fright. It asked for the roll (the body link holds it as a lie-down
    on purpose): those are its own doing until 2 s after it is up again, then the senses count again; words reach it all along."""
    lanes = Lanes(brain=_Seeing(seed=5), senses=_rough_senses, names={"brain": "stub", "senses": "stub", "motor": "stub"})
    driver, brain = NeuroDriver(NeuroSpec(seed=5), lanes=lanes), lanes.brain
    t = 0.0
    for _ in range(50):
        driver.tick(robot_state(t))
        t += 0.02
    assert all(brain.fright[-10:])  # (standing, not rolling: they count)
    n0 = len(brain.fright)
    for _ in range(50):  # the roll
        driver.tick(robot_state(t, policy="roulade", fallen=True, gravity=(0.0, 0.0, 1.0)))
        t += 0.02
    for _ in range(150):  # up again: the link's hold for 1 s, then 2 s more
        driver.tick(robot_state(t))
        t += 0.02
    n1 = len(brain.fright)
    for _ in range(50):
        driver.tick(robot_state(t))
        t += 0.02
    assert not any(brain.fright[n0:n1])
    assert sum(brain.words[n0:n1]) >= (n1 - n0) // 2 - 1  # (every other tick somebody spoke: it heard them)
    assert all(brain.fright[-10:])


def test_a_roll_it_asked_for_is_no_fall_to_the_brain_while_the_link_holds_it() -> None:
    driver, brain = _seeing_driver()
    t = 0.0
    for _ in range(50):
        driver.tick(robot_state(t))
        t += 0.02
    n0 = len(brain.fallen)
    for _ in range(40):  # robotd runs the roulade and its gravity filter says "fallen" (it is upside down)
        driver.tick(robot_state(t, policy="roulade", fallen=True, gravity=(0.0, 0.0, 1.0)))
        t += 0.02
    for _ in range(10):  # the roll window is over, robotd's flag still up for a moment, the trunk coming back
        driver.tick(robot_state(t, policy="stand", fallen=True, gravity=(0.5, 0.0, -0.8)))
        t += 0.02
    assert len(brain.fallen) - n0 == 50
    assert not any(brain.fallen[n0:])  # (else the flip command ends "interrupted: fallen" mid-roll)


def test_a_duck_still_down_long_after_its_roll_has_fallen_for_the_brain_too() -> None:
    driver, brain = _seeing_driver()
    t = 0.0
    for _ in range(50):
        driver.tick(robot_state(t))
        t += 0.02
    for _ in range(40):
        driver.tick(robot_state(t, policy="roulade", fallen=True, gravity=(0.0, 0.0, 1.0)))
        t += 0.02
    n0 = len(brain.fallen)
    for _ in range(int(32.0 / 0.02)):  # never up again: past the link's hold (30 s) it is an ordinary fall
        driver.tick(robot_state(t, policy="stand", fallen=True, gravity=(0.0, 0.0, 1.0)))
        t += 0.02
    assert not brain.fallen[n0]
    assert brain.fallen[-1]


# ── integration 2026-09-28: the day clock, the helping hand, rhythms, the nest ──


def test_the_duck_clock_starts_in_the_daytime_and_runs_at_the_real_pace() -> None:
    from duckmind.simweb.neuro_live import DAY_START_MINUTE, DayClock, minute_zh, parse_day_time

    wall = [1000.0]
    clock = DayClock(wall=lambda: wall[0])
    assert clock.minute() == DAY_START_MINUTE == 600.0  # 10:00: its active day, not 19:00
    wall[0] += 90.0
    assert abs(clock.minute() - 601.5) < 1e-9  # the real pace
    clock.set(21 * 60 + 30)
    assert clock.view()["zh"] == "晚上 9:30" and clock.local().hour == 21
    assert parse_day_time("07:30") == 450 and parse_day_time("") == 600.0
    assert minute_zh(15 * 60) == "下午 3:00" and minute_zh(60) == "半夜 1:00"
    # the driver's frames carry it (the circadian drive reads frame.day_minute)
    driver = stub_driver()
    driver.set_day(15 * 60)
    driver._day_minute = driver.day_clock.minute()
    frame_minutes = []
    real = driver.lanes.brain.step

    def spy(t_ms, frame):
        frame_minutes.append(frame.day_minute)
        return real(t_ms, frame)

    driver.lanes.brain.step = spy  # type: ignore[method-assign]
    driver.tick(robot_state(0.0))
    assert 899.0 < frame_minutes[-1] < 901.0
    tel = driver.telemetry(0)
    assert tel["day"]["zh"].startswith("下午")
    assert any(e["kind"] == "world.clock" for e in tel["events"])


def test_help_up_asks_the_worlds_hand_and_says_so() -> None:
    driver = stub_driver()
    body = _WorldBody()
    driver._body = body
    answer = driver.help_up()
    assert body.calls[-1] == ("x_hand", {"do": "upright"})
    assert answer["ok"] and driver.helps == 1
    assert driver.telemetry(0)["events"][-1]["stimulus"] == "help_up"


def test_a_rhythm_is_a_train_of_world_sounds() -> None:
    driver = stub_driver()

    class Body(_WorldBody):
        host_info = {"ops": ["x_hand", "x_neuro_stim"]}

        def world_call(self, op, **params):
            self.calls.append((op, params))
            return {"ok": True, "note_zh": "咚"}

    body = Body()
    driver._body = body
    answer = driver.inject("clap_train", {})
    assert answer["kind"] == "clap_train" and "8" in answer["note_zh"]
    deadline = __import__("time").monotonic() + 5.0
    while len(body.calls) < 8 and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.05)
    assert [c[1]["stim"] for c in body.calls] == ["clap"] * 8
    assert {"knock_train", "clap_train", "push", "tv"} <= driver.stimuli_supported()


def test_the_nest_reverse_reaches_the_body_link() -> None:
    from duckmind.neuro import motor as M
    from duckmind.neuro.genome import Genome
    from duckmind.simweb.neuro_live import LiveMotor

    motor = LiveMotor(M, Genome.from_seed(7), 7)
    cmd = C.MotorCommand(
        t_ms=0, behaviour="wander", bout=1, variant=1, vx=-0.34, docking=True, posture="stand"
    )
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    request, _ = motor(0, cmd, frame, robot_state(0.0))
    assert request.docking and request.vx < 0.0  # backing off the nest (owner: only there)


def test_a_person_can_swap_in_a_charged_battery() -> None:
    """The world's pack lasts ~30–60 min (battery.py); for a long watch a person swaps in a
    charged one — a world act (x_battery swap) the duck feels only through its power sense."""
    driver = stub_driver()

    class Body(_WorldBody):
        host_info = {"ops": ["x_hand", "x_neuro_stim", "x_battery"]}

        def world_call(self, op, **params):
            self.calls.append((op, params))
            return {"percent": 100}

    body = Body()
    driver._body = body
    assert "swap_battery" in driver.stimuli_supported()
    answer = driver.inject("swap_battery", {})
    assert body.calls == [("x_battery", {"do": "swap"})]
    assert answer["ok"] and "電池" in answer["note_zh"]
    assert driver.telemetry(0)["events"][-1]["stimulus"] == "swap_battery"
    assert "swap_battery" in C.STIMULI


def test_a_world_refusal_comes_back_in_the_worlds_own_words() -> None:
    """The world's hand is busy (it just set the duck down): the page's button gets the
    world's plain words, never a dropped connection (live page check 2026-09-28)."""
    driver = stub_driver()

    class Body(_WorldBody):
        host_info = {"ops": ["x_hand", "x_neuro_stim"]}

        def world_call(self, op, **params):
            raise RuntimeError(
                "body host x_neuro_stim: {'error': '家裡的手正在做別的事，等一下再試'}"
            )

    driver._body = Body()
    with pytest.raises(ValueError, match="家裡的手正在做別的事"):
        driver.inject("push", {})
