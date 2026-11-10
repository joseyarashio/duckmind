"""The page's real-duck mode, server side (docs/real-microduck.md): with a body whose bridge said
``backend: real`` the simulator page's server asks no world for anything. Every world action is
refused with the one plain reason (``livebody.no_world``), the page is told what to disable
(``real`` in ``/api/body`` and ``/api/neuro``), no world truth reaches the brain, and the eye is the
duck's own camera over the link or no picture with the reason — never a render of a scene.

A fake real body (the contract's LiveBody answers, no processes) stands in for the duck; one test
runs the real LiveBody against a fake real bridge (tests/test_real_duck_body.py). No simulator.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from test_real_duck_body import _wait_for, duck  # noqa: F401 - a fake real duck behind its bridge

from duckmind.duckcore.api import Config, Session
from duckmind.i18n import msg
from duckmind.livebody import LivePaths, NoWorld, RemoteBodySpec
from duckmind.neuro.livestub import StubBrain
from duckmind.simweb import battery_help, neuro_api
from duckmind.simweb.eyecam import EyeCam
from duckmind.simweb.live import REAL_REFUSES, LiveBridge, NoEye
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimServer, SimState, make_handler

NO_WORLD = str(msg("livebody.no_world"))
NO_CAMERA = str(msg("livebody.real_no_camera"))


class FakeRealBody:
    """What LiveBody answers for a real duck (local://rdpage-contract.md): no home, no world,
    ``x_sensors`` the link's own (here: no camera), every other world op NoWorld, frames from
    robotd's contact odometry. ``ops`` may list more (a defence test: still no world)."""

    real = True
    rehearsal = True
    home = None
    body_port = None
    error: str | None = None
    policy = "stand"
    passive = False

    def __init__(self, ops: tuple[str, ...] = ("x_sensors",)) -> None:
        self.host_info = {"ops": list(ops), "real": True}
        self.calls: list[str] = []
        self.consumed: list[dict[str, Any]] = []
        self.pack: dict[str, Any] | None = {
            "percent": 64,
            "volts": 7.6,
            "charging": None,
            "source": "robotd",
            "age_s": 0.4,
        }

    def frame(self) -> dict[str, Any]:
        yaw = 0.5
        return {
            "t": time.monotonic(),
            "joints": {"left_knee": 0.1},
            "trunk": [0.4, -0.2, 0.118],
            "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
            "policy": self.policy,
            "fallen": False,
            "realtime": 1.0,
            "source": "odometry",
        }

    def status(self) -> dict[str, Any]:
        return {
            "running": True,
            "policy": self.policy,
            "fallen": False,
            "realtime": 1.0,
            "error": None,
            "monitor": {"trust": "ok", "collision": "real", "why_zh": []},
            "real": True,
            "rehearsal": True,
            "remote": {"url": "remote://127.0.0.1:1", "up": True, "rtt_ms_p50": 3.0},
        }

    def battery(self) -> dict[str, Any] | None:
        return self.pack

    def world(self) -> None:
        self.calls.append("world()")
        return None

    def world_call(self, op: str, **params: Any) -> dict[str, Any]:
        self.calls.append(op)
        if op == "x_sensors":
            return {
                "media": {"socket": None},
                "tofd": {"socket": None},
                "tof": {"fitted": False},
                "eye": {},
                "errors": [msg("livebody.real_no_camera")],
            }
        raise NoWorld(op)

    def consume(self, motions: list[dict[str, Any]]) -> None:
        self.consumed.extend(motions)

    def frames_since(self, since_t: float | None = None) -> list[dict[str, Any]]:
        return []

    def stop(self) -> None:
        return None


def _bridge(tmp_path: Path, body: Any = None, *, eye: bool = False) -> tuple[LiveBridge, Any]:
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=eye)  # never started: no processes
    bridge.sync_quiet_s = 0
    body = body or FakeRealBody()
    bridge.body = body
    bridge.ready = True
    return bridge, body


def _world_ops(body: FakeRealBody) -> list[str]:
    return [op for op in body.calls if op != "x_sensors"]


# ── LiveBridge: the snapshot, the truth, every world action ─────────────────────────────


def test_the_snapshot_tells_the_page_it_is_a_real_duck_and_draws_no_world(tmp_path):
    bridge, body = _bridge(tmp_path)
    snap = bridge.snapshot()
    assert snap["real"] == {
        "rehearsal": True,
        "reason": NO_WORLD,
        "refuses": sorted(REAL_REFUSES),
        "turn_on": None,
        "walk_lock": None,
        "battery": body.pack,
        "has_dock": False,  # (no dock hardware exists: the page says so beside the battery)
    }
    assert snap["remote"]["up"] is True  # (the link's own numbers stay)
    truth = snap["frame"]["truth"]
    assert truth == {"x_mm": 400, "y_mm": -200, "yaw_mrad": 500, "frame": "odometry"}
    world = snap["world"]
    assert world["home"] is None and world["physical"] is False and world["real"] is True
    for key in ("hand", "people", "things", "battery", "nest", "backlight"):
        assert key not in world
    assert world["bodies"] == {}
    body.pack = None  # (robotd has not said yet: the page shows its placeholder)
    assert bridge.snapshot()["real"]["battery"] is None
    body.has_dock = True  # (its bridge was started with --has-dock)
    assert bridge.snapshot()["real"]["has_dock"] is True


def test_a_simulated_body_has_no_real_block(tmp_path):
    class Simulated(FakeRealBody):
        real = False

    bridge, _ = _bridge(tmp_path, Simulated())
    snap = bridge.snapshot()
    assert snap["real"] is None and snap["frame"]["truth"]["frame"] == "world"


def test_the_body_snapshot_keeps_the_real_mode_while_the_link_dials_again(tmp_path):
    """Before: a body that had not said what it is yet (the link dialling again after a drop, a
    restart) made ``/api/body`` say ``real: null`` and the page flipped to the simulated mode."""
    bridge, body = _bridge(tmp_path)
    body.has_dock = True
    assert bridge.snapshot()["real"]["rehearsal"] is True

    class Dialling(FakeRealBody):
        real = False  # (its bridge has not answered yet)
        rehearsal = False

        def battery(self) -> None:
            return None

    bridge.body = Dialling()
    assert bridge.snapshot()["real"] == {
        "rehearsal": True,
        "reason": NO_WORLD,
        "refuses": sorted(REAL_REFUSES),
        "turn_on": None,
        "walk_lock": None,
        "battery": None,
        "has_dock": True,  # (what its bridge said is kept while the link dials again)
    }


def test_the_snapshot_asks_a_person_to_turn_a_real_duck_on_while_the_body_waits(tmp_path):
    """A real duck that is not on is never enabled by the brain: the body waits and the page is
    told, in plain words, that a person turns it on with the duck's own control."""
    bridge, body = _bridge(tmp_path)
    bridge.ready = False  # (the body's start has not returned: it waits)
    body.waits_turn_on = True
    snap = bridge.snapshot()
    assert snap["starting"] is True and snap["available"] is False
    assert snap["real"]["turn_on"] == str(msg("livebody.real_turn_on"))
    assert snap["real"]["turn_on"].key == "livebody.real_turn_on"
    body.waits_turn_on = False  # (a person turned it on)
    assert bridge.snapshot()["real"]["turn_on"] is None


def test_the_live_bridge_waits_for_a_real_duck_to_be_turned_on(tmp_path, duck):  # noqa: F811
    """End to end over a fake real bridge: the duck is not on, the page's body says so and is not
    ready; nothing enables it; once a person has, the bridge is ready and drives it."""
    duck.robotd.state = {**duck.robotd.state, "policy": "off"}
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    url = f"remote://127.0.0.1:{duck.bridge.port}"
    bridge = LiveBridge(paths, remote=RemoteBodySpec(url, duck.token))
    bridge.start()
    try:
        assert _wait_for(lambda: (bridge.snapshot()["real"] or {}).get("turn_on") is not None, 10.0)
        snap = bridge.snapshot()
        assert snap["starting"] is True and snap["error"] is None and bridge.ready is False
        assert snap["real"]["turn_on"] == str(msg("livebody.real_turn_on"))
        time.sleep(0.4)
        assert "robot.enable" not in duck.robotd.methods
        duck.robotd.state = {**duck.robotd.state, "policy": "sit"}  # a person turned it on
        assert _wait_for(lambda: bridge.ready or bridge.error is not None, 10.0), "never ready"
        assert bridge.error is None and bridge.snapshot()["real"]["turn_on"] is None
        assert not {"robot.enable", "robot.init", "robot.poseJoints"} & set(duck.robotd.methods)
    finally:
        bridge.stop()


def test_every_world_action_of_the_bridge_is_refused_with_the_plain_reason(tmp_path):
    bridge, body = _bridge(tmp_path, FakeRealBody(ops=("x_sensors", "x_hand", "x_people")))
    asks = [
        bridge.upright,
        lambda: bridge.place_prop("ball", 100, 100),
        lambda: bridge.world_action("x_roll_ball", toward="duck"),
        lambda: bridge.world_action("x_tidy", to="toy_corner"),
        lambda: bridge.demo("kick", "ball"),
        lambda: bridge.eye_demo("kick", "ball"),
        bridge.demo_walk,
    ]
    for ask in asks:
        with pytest.raises(ValueError) as refused:
            ask()
        assert str(refused.value) == NO_WORLD
    assert _world_ops(body) == []
    with pytest.raises(ConnectionError) as refused:
        bridge.world_port().call("x_people", do="come", who="owner", to="duck")
    assert str(refused.value) == NO_WORLD
    assert _world_ops(body) == []


def test_resync_and_skipped_never_place_a_real_duck_but_still_match_its_posture(tmp_path):
    bridge, body = _bridge(tmp_path)
    session = Session(Config(seed=7))
    bridge.resync(session, "start")
    bridge.skipped(session)
    note = bridge.place((1.0, 1.0, 0.0), "sync")
    assert note["ok"] is False and [str(w) for w in note["why_zh"]] == [NO_WORLD]
    assert _world_ops(body) == []
    queued = [item[2] for item in bridge._due]
    assert any("power" in item for item in queued)  # (the duck's own catch-up goes on)
    assert bridge.snapshot()["world"]["placed"] is None


def test_duck_cores_hand_events_and_the_light_clock_never_reach_a_real_duck(tmp_path):
    from datetime import datetime

    from duckmind.duckcore.api import dispatch

    bridge, body = _bridge(tmp_path)
    bridge.neuro = object()  # (prototype mode: only the world's hand would go out)
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    before = len(session.core.log.events)
    dispatch(session, "act", {"action": "pickup", "params": {}})
    bridge.after_call(session, before)
    assert bridge._due == []
    bridge._hand({"do": "pickup"})
    bridge.set_clock(datetime(2026, 11, 9, 10, 0), 1.0)
    assert bridge._due == [] and _world_ops(body) == []


def test_the_household_never_visits_or_swaps_a_real_ducks_pack(tmp_path):
    # (a defence: even a body that listed the hand and people ops is no world)
    bridge, _ = _bridge(tmp_path, FakeRealBody(ops=("x_sensors", "x_hand", "x_people")))
    assert bridge.world_ready() is False
    session = Session(Config(seed=7, battery_help=True, nest=False))
    core = session.core
    core.s.bridged = True
    core.s.awake = True
    assert not battery_help.helping(core, on_screen=True, world_ready=bridge.world_ready())
    assert battery_help.pack_of(bridge.world_truth()) is None


# ── the eye: the link's camera or no picture with the reason ───────────────────────────


def test_the_eye_of_a_real_duck_is_never_a_scene_render(tmp_path):
    bridge, body = _bridge(tmp_path, eye=True)
    bridge._start_eye(body)
    assert isinstance(bridge.eye, NoEye)
    assert bridge.eye_status()["error"] == NO_CAMERA
    image, _, why = bridge.eye_image()
    assert image is None and why == NO_CAMERA
    # A real body that offers no x_sensors at all still gets no render of a scene.
    bare = FakeRealBody(ops=())
    bridge.body = bare
    bridge._start_eye(bare)
    assert isinstance(bridge.eye, NoEye) and not isinstance(bridge.eye, EyeCam)
    assert bridge.eye_status()["error"] == NO_CAMERA
    bridge._stop_eye(wait=True)


# ── the brain's driver: no world truth, no stimuli, no help, no go-there ────────────────


def _driver() -> NeuroDriver:
    lanes = Lanes(
        brain=StubBrain(seed=7), names={"brain": "stub", "senses": "stub", "motor": "stub"}
    )
    return NeuroDriver(NeuroSpec(seed=7), lanes=lanes)


def test_no_world_truth_reaches_the_brain_of_a_real_duck():
    driver = _driver()
    body = FakeRealBody()
    driver._leaves = [(0.0, "owner")]  # (a walk-by that would end now: no person to send off)
    stop = threading.Event()
    loop = threading.Thread(target=driver._aux, args=(body, stop), daemon=True)
    loop.start()
    time.sleep(0.4)
    stop.set()
    driver._aux_wake.set()
    loop.join(timeout=2)
    assert body.frame()["source"] == "odometry"  # (it has a frame: it is the duck's own, not truth)
    assert driver._truth is None and driver._world is None and driver._pg is None
    assert body.calls == []  # no pg, no x_world, no x_people leave


def test_the_driver_refuses_every_stimulus_help_and_go_there_on_a_real_duck():
    driver = _driver()
    body = FakeRealBody(ops=("x_sensors", "x_hand", "x_people", "x_battery", "x_neuro_stim"))
    driver._body = body
    assert driver.stimuli_supported() == set()
    for name in ("pet_head", "bang", "roll_ball", "walk_by", "lights_off", "swap_battery"):
        with pytest.raises(ValueError) as refused:
            driver.inject(name, {})
        assert str(refused.value) == NO_WORLD, name
    with pytest.raises(ValueError) as refused:
        driver.help_up()
    assert str(refused.value) == NO_WORLD
    answer = driver.go_there(0.5, 0.5)
    assert answer["ok"] is False and answer["reason"] == "real"
    assert str(answer["reason_zh"]) == NO_WORLD
    assert _world_ops(body) == []


def test_api_neuro_advertises_the_real_mode_and_refuses_with_its_reason():
    driver = _driver()
    driver._body = FakeRealBody()
    hub = neuro_api.NeuroHub(enabled=True)
    hub.attach(neuro_api.DriverSource(lambda: driver))
    expected = {
        "rehearsal": True,
        "reason": NO_WORLD,
        "walk_lock": None,  # (no body link yet: nothing is held)
        "refuses": sorted(REAL_REFUSES),
    }
    assert hub.status()["real"] == expected
    telemetry = hub.telemetry()
    assert telemetry["real"] == expected
    assert not any(telemetry["stimuli"].values())  # (every stimulus button off)
    refused = hub.stimulus({"kind": "pet_head"})
    assert refused["ok"] is False and str(refused["error"]) == NO_WORLD
    with pytest.raises(ValueError) as error:
        hub.help_up()
    assert str(error.value) == NO_WORLD
    gone = hub.go_there({"point": {"x_m": 0.5, "y_m": 0.5}})
    assert gone["ok"] is False and str(gone["reason_zh"]) == NO_WORLD
    # (the words carry their message for the page's other languages)
    wired = neuro_api.boundary.attach({"real": hub.status()["real"]})
    assert wired["real"]["_i18n"]["reason"]["k"] == "livebody.no_world"


def test_api_neuro_leaves_the_mode_out_while_the_brain_has_no_body():
    """Before: with no body attached (starting, a restart, a link that does not answer)
    ``/api/neuro`` said ``real: null`` and the page flipped between the real and the simulated mode."""

    class Simulated(FakeRealBody):
        real = False

    for driver in (None, _driver()):
        hub = neuro_api.NeuroHub(enabled=True)
        hub.attach(neuro_api.DriverSource(lambda driver=driver: driver))
        assert "real" not in hub.status() and "real" not in hub.telemetry()
    driver = _driver()
    hub = neuro_api.NeuroHub(enabled=True)
    hub.attach(neuro_api.DriverSource(lambda: driver))
    driver._body = FakeRealBody()
    assert hub.status()["real"]["rehearsal"] is True and hub.telemetry()["real"]["rehearsal"]
    driver._body = None  # (let go for a restart: the mode stays what the page knows)
    assert "real" not in hub.status() and "real" not in hub.telemetry()
    driver._body = Simulated()  # (a simulated body says so)
    assert hub.status()["real"] is None and hub.telemetry()["real"] is None


# ── the server's endpoints ──────────────────────────────────────────────────────────


@pytest.fixture()
def real_server(tmp_path):
    bridge, body = _bridge(tmp_path, FakeRealBody(ops=("x_sensors", "x_hand", "x_people")))
    state = SimState(Config(seed=7, name="測試"), live=bridge)
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state, body
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_server_refuses_every_world_button_of_a_real_duck(real_server):
    base, state, body = real_server
    calls = [
        ("body_upright", {}),
        ("body_show", {"kind": "kick", "thing": "ball"}),
        ("body_eye", {"kind": "kick", "thing": "ball"}),
        ("body_roll", {}),
        ("body_tidy", {}),
        ("body_place", {"thing": "ball", "x_mm": 100, "y_mm": 100}),
        ("body_demo", {}),
        ("body_reset", {}),
    ]
    old_body = state.live.body
    for method, params in calls:
        status, answer = _post(base + "/api/call", {"method": method, "params": params})
        assert status == 400 and answer["ok"] is False, method
        assert answer["error"] == NO_WORLD, method
        assert answer["_i18n"]["error"]["k"] == "livebody.no_world", method
    assert state.live.body is old_body  # (「重設」 did not rebuild anything)
    assert _world_ops(body) == []


def test_the_layout_editor_reads_but_never_applies_on_a_real_duck(real_server):
    base, state, _ = real_server
    status, layout = _get(base + "/api/layout")
    assert status == 200 and layout["real"]["reason"] == NO_WORLD
    for verb, payload in (("reset", {}), ("load", {"name": "x"})):
        status, answer = _post(base + f"/api/layout/{verb}", payload)
        assert status == 200 and answer["ok"] is False and answer["code"] == "real", verb
        assert answer["error"] == NO_WORLD and answer["_i18n"]["error"]["k"] == "livebody.no_world"
    with pytest.raises(ValueError):
        state.layout_apply({})


def test_api_body_and_api_home_carry_no_home_for_a_real_duck(real_server):
    base, state, _ = real_server
    status, view = _get(base + "/api/body?home=someone-elses")
    assert status == 200 and view["enabled"] is True
    assert view["home"] is None and "home_json" not in view
    assert view["real"]["reason"] == NO_WORLD
    assert view["real"]["_i18n"]["reason"]["k"] == "livebody.no_world"
    assert view["frame"]["truth"]["frame"] == "odometry"
    assert "family" not in view["world"]
    status, home = _get(base + "/api/home")
    assert status == 200 and home == {
        "name": None,
        "layout": None,
        "fingerprint": None,
        "real": True,
    }
    assert state.visit("pet_head", "owner", {}) is None  # (no one walks over: Duck Core hears it)


# ── the real LiveBody over a fake real bridge (no simulator) ────────────────────────────


def test_the_live_bridge_on_a_real_bridge_drives_with_no_world(tmp_path, duck):  # noqa: F811
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    url = f"remote://127.0.0.1:{duck.bridge.port}"
    bridge = LiveBridge(paths, remote=RemoteBodySpec(url, duck.token))
    bridge.start()
    try:
        assert _wait_for(lambda: bridge.ready or bridge.error is not None, 10.0), "never ready"
        assert bridge.error is None and bridge.real is True
        assert _wait_for(lambda: bridge.snapshot()["frame"] is not None)
        snap = bridge.snapshot()
        assert snap["real"]["rehearsal"] is True and snap["real"]["reason"] == NO_WORLD
        assert snap["frame"]["truth"]["frame"] == "odometry"
        assert "placed_at" not in snap["frame"]["truth"]
        assert snap["world"]["home"] is None and "hand" not in snap["world"]
        assert isinstance(bridge.eye, NoEye) and snap["eye"]["error"] == NO_CAMERA
        with pytest.raises(ValueError) as refused:
            bridge.upright()
        assert str(refused.value) == NO_WORLD
    finally:
        bridge.stop()


def test_the_sound_scenarios_refuse_a_real_duck_in_the_viewers_language() -> None:
    """Before: the refusal came back as bare 繁中 words (no ``_i18n``) for every viewer."""
    import io
    from types import SimpleNamespace

    from duckmind.livebody import NoWorld
    from duckmind.simweb.neuro_sound_live import SoundHub

    def world_call(op: str, **params: Any) -> dict[str, Any]:
        raise NoWorld(op)

    driver = SimpleNamespace(_body=SimpleNamespace(world_call=world_call), _aux_wake=None)
    raw = json.dumps({"do": "music", "tempo": "slow"}).encode()
    sent: dict[str, Any] = {}
    handler = SimpleNamespace(
        headers={"Content-Length": str(len(raw))},
        rfile=io.BytesIO(raw),
        wfile=io.BytesIO(),
        send_response=lambda status: sent.setdefault("status", status),
        send_header=lambda *a: None,
        end_headers=lambda: None,
    )
    SoundHub(lambda: driver).post(handler, "/api/neuro/sound/world")
    answer = json.loads(handler.wfile.getvalue())
    assert sent["status"] == 200 and answer["ok"] is False
    assert answer["error"] == str(msg("livebody.no_world"))
    assert answer["_i18n"]["error"]["k"] == "livebody.no_world"
