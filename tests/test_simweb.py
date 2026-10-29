import gzip
import json
import math
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from duckmind.duckcore import content
from duckmind.duckcore.api import Config
from duckmind.simweb import robot_model
from duckmind.simweb.server import MAX_BODY, SimServer, SimState, make_handler

STATIC = Path(robot_model.__file__).parent / "static"


def _worded(text: str) -> str:
    """A page source as its words were written before the page spoke six languages: every
    quoted catalog key replaced by its 繁體中文 words (static/i18n.zh-Hant.*.js; docs/i18n.md)."""
    words: dict[str, str] = {}
    for path in sorted(STATIC.glob("i18n.zh-Hant.*.js")):
        body = path.read_text(encoding="utf-8")
        words.update(
            json.loads(body[body.index("{", body.index("DuckI18n.load(")) : body.rindex("}") + 1])
        )
    return re.sub(
        r"([\"'])((?:page|fmt|web)\.[A-Za-z0-9_.|<>-]+)\1",
        lambda m: m.group(1) + words.get(m.group(2), m.group(2)) + m.group(1),
        text,
    )


# The official checkout is optional: point DUCKMIND_MICRODUCK_MJCF at
# microduck_rl/src/mjlab_microduck/robot/microduck to run the model tests.
_CONFIGURED = robot_model.robot_root()
OFFICIAL = _CONFIGURED / "robot_walk.xml" if _CONFIGURED else None
needs_official = pytest.mark.skipif(
    OFFICIAL is None or not OFFICIAL.is_file(),
    reason="official microduck_rl checkout not configured",
)


@pytest.fixture()
def server():
    httpd = SimServer(("127.0.0.1", 0), make_handler(SimState(Config(seed=7, name="測試"))))
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


def _call(base: str, method: str, params: dict | None = None, cursor: dict | None = None) -> dict:
    body = json.dumps({"method": method, "params": params or {}, "cursor": cursor or {}}).encode()
    request = urllib.request.Request(
        f"{base}/api/call", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def test_the_page_and_its_scripts_are_served(server):
    for path in (
        "/",
        "/app.js",
        "/duck3d.js",
        "/home3d.js",
        "/truth.js",
        "/style.css",
        "/shell.js",
    ):
        status, body = _get(server + path)
        assert status == 200 and body, path
    # 「資料」 is a view of the same page (shell.js shows it): /data serves the page itself
    assert _get(server + "/data") == _get(server + "/")


def test_the_api_runs_the_real_core_and_streams_a_feed(server):
    reply = _call(server, "act", {"action": "power_on", "persona": "owner"})
    assert reply["ok"] and reply["state"]["body"]["awake"]
    assert reply["feed"]["events"], "powering on should produce events for the page"
    later = _call(server, "advance", {"minutes": 10}, reply["feed"]["cursor"])
    assert later["feed"]["cursor"]["events"] > reply["feed"]["cursor"]["events"]
    assert _call(server, "new", {"seed": 3, "name": "Pip"})["state"]["identity"]["name"] == "Pip"


def test_bad_paths_and_oversized_bodies_are_refused(server):
    for path in (
        "/../pyproject.toml",
        "/%2e%2e/server.py",
        "/mesh/..%2Frobot_walk.xml",
        "/mesh/nope",
        "/.hidden.js",
    ):
        assert _get(server + path)[0] == 404, path
    request = urllib.request.Request(
        f"{server}/api/call",
        data=b"{}",
        headers={"Content-Length": str(MAX_BODY + 1)},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 413


def test_a_burst_of_parallel_requests_is_served(server):
    # `tailscale serve` forwards all ~30 mesh requests at once; none may be refused.
    assert SimServer.request_queue_size >= 64
    with ThreadPoolExecutor(max_workers=40) as pool:
        statuses = list(pool.map(lambda _: _get(server + "/api/clips")[0], range(80)))
    assert statuses == [200] * 80


@needs_official
def test_meshes_are_gzipped_and_hidden_parts_are_not_served(server):
    robot_model.load_model.cache_clear()
    request = urllib.request.Request(
        f"{server}/mesh/top_head_shell", headers={"Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["Content-Encoding"] == "gzip"
        body = gzip.decompress(response.read())
    assert body == robot_model.mesh_path("top_head_shell").read_bytes()
    assert _get(f"{server}/mesh/np_f970")[0] == 404
    drawn = set()

    def collect(part):
        drawn.update(geom["mesh"] for geom in part["geoms"])
        for child in part["children"]:
            collect(child)

    collect(robot_model.load_model()["trunk"])
    assert drawn and not drawn & robot_model.HIDDEN_MESHES


def test_without_the_official_checkout_the_page_falls_back_to_2d(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCKMIND_MICRODUCK_MJCF", str(tmp_path))
    robot_model.load_model.cache_clear()
    try:
        model = robot_model.load_model()
        assert model["available"] is False and model["attribution"]
        assert robot_model.mesh_path("jaw") is None
    finally:
        robot_model.load_model.cache_clear()


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate(q, v):
    return _quat_mul(_quat_mul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))[1:]


def _head_pose(model, angles):
    """World position and facing of the head body (its face points along local −z)."""
    found = {}

    def walk(body, position, orientation):
        position = tuple(
            a + b for a, b in zip(position, _rotate(orientation, body["pos"]), strict=True)
        )
        orientation = _quat_mul(orientation, tuple(body["quat"]))
        if body["joint"]:
            axis = body["joint"]["axis"]
            norm = math.sqrt(sum(value * value for value in axis))
            half = angles.get(body["joint"]["name"], 0.0) / 2
            turn = (math.cos(half), *(value / norm * math.sin(half) for value in axis))
            orientation = _quat_mul(orientation, turn)
        found[body["name"]] = (position, orientation)
        for child in body["children"]:
            walk(child, position, orientation)

    walk(model["trunk"], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    position, orientation = found["jaw_soft"]
    return position, _rotate(orientation, (0.0, 0.0, -1.0))


@needs_official
def test_the_sign_table_matches_the_official_model():
    robot_model.load_model.cache_clear()
    model = robot_model.load_model()
    assert model["available"] and len(model["meshes"]) > 20
    assert {"SIT", "STAND"} <= set(model["keyframes"])
    home = {**model["keyframes"]["SIT"], **content.HEAD_HOME}
    _, facing = _head_pose(model, home)
    assert facing[0] > 0.99  # the face points forward (+x) at HOME

    def moved(frame):
        targets = content.head_targets(frame, side=1)
        return _head_pose(model, {**home, **{k: v for k, v in targets.items() if k != "mouth"}})

    _, up = moved({"hp": 1.0})
    assert up[2] > 0.2  # "look up" looks up
    back_position, back_facing = moved({"np": -1.0})
    assert back_facing[2] > 0.05  # leaning back tips the face up
    _, left = moved({"hy": 1.0})
    assert left[1] > 0.3  # side +1 is the duck's left
    hinge = model["jaw_hinge"]
    assert hinge and hinge["pos"][2] < 0 and hinge["axis"] == [0.0, 1.0, 0.0]


def test_the_duck_survives_a_server_restart(tmp_path):
    from duckmind.duckcore.api import Config, dispatch
    from duckmind.simweb.server import SimState

    path = tmp_path / "duck.json"
    first = SimState(Config(seed=11, name="蛋黃"), state_file=path)
    dispatch(first.session, "act", {"action": "power_on", "params": {}})
    dispatch(first.session, "advance", {"minutes": 90})
    first.save()
    again = SimState(Config(seed=1, name="別的"), state_file=path)
    assert again.session.config.name == "蛋黃"
    assert again.session.core.log.head == first.session.core.log.head


def test_real_time_resumes_after_a_restart(tmp_path):
    from duckmind.duckcore.api import Config
    from duckmind.simweb.server import SimState

    path = tmp_path / "duck.json"
    first = SimState(Config(seed=3), state_file=path)
    first.realtime = True
    first._saved_inputs = -1
    first.save()
    again = SimState(Config(seed=3), state_file=path)
    assert again.realtime


def test_the_page_is_told_how_to_show_each_event_and_why_the_duck_is_still(server):
    first = _call(server, "state")
    assert first["state"]["body"]["banner"]["title"]
    assert {"realtime", "offset_min", "now_ms"} <= set(first["server"])
    cursor = first["feed"]["cursor"]
    played = _call(server, "advance", {"minutes": 1}, cursor)
    assert not any(item.get("settle") for item in played["feed"]["events"])
    summary = _call(server, "advance", {"minutes": 180}, played["feed"]["cursor"])
    assert summary["feed"]["events"] and all(
        item.get("settle") for item in summary["feed"]["events"]
    )
    fast = _call(server, "advance", {"minutes": 15}, summary["feed"]["cursor"])
    kept = [item for item in fast["feed"]["events"] if item["motion"] and not item.get("settle")]
    assert len(kept) <= 2  # a fast-forward step shows only its newest walk and move
    morning = _call(server, "advance", {"until": "07:00"}, fast["feed"]["cursor"])
    assert morning["state"]["clock"]["now"].endswith("07:00")
    page = _get(server + "/")[1].decode()
    assert 'id="duckBanner"' in page and 'id="bannerPlay"' in page


def test_real_time_lives_each_second_and_shows_it_a_moment_later():
    from duckmind.duckcore.api import dispatch
    from duckmind.simweb.server import SHOW_DELAY_S

    state = SimState(Config(seed=7, name="測試"))
    dispatch(state.session, "advance", {"until": "10:00"})
    dispatch(state.session, "set_setting", {"key": "floor", "value": True})
    core = state.session.core
    while core.s.napping_until is not None and core.s.napping_until > core.s.now:
        dispatch(state.session, "advance", {"minutes": 1})  # measure an awake duck
    wall = 1_900_000_000.0
    state.set_realtime(True)
    state.anchor = (wall, core.s.now)
    begin = core.s.now
    inputs = len(state.session.inputs)
    for second in range(1, 8 * 60 + 1):
        state.step_realtime(wall=wall + second)
        # The duck's clock is the wall clock: nothing is decided ahead of time.
        assert (core.s.now - begin).total_seconds() == second
    assert len(state.session.inputs) == inputs + 1  # one input for the whole stretch
    moved = [
        index for index in range(len(core.log.events)) if core.log.events[index].occurred_at > begin
    ]
    assert moved
    for index in moved:
        event = core.log.events[index]
        due = state.marks[index]["due_ms"]
        offset = (event.occurred_at - begin).total_seconds()
        assert due == round((wall + offset + SHOW_DELAY_S) * 1000)
    # Decisions fall anywhere in the minute, not all at its end.
    seconds = {
        int((core.log.events[index].occurred_at - begin).total_seconds()) % 60
        for index in moved
        if core.log.events[index].kind == "duck.decision"
    }
    assert len(seconds) >= 4
    glances = state.fidgets
    assert glances and len({item["id"] for item in glances}) == len(glances)
    assert all(wall * 1000 < item["due_ms"] <= (wall + 8 * 60 + 3) * 1000 for item in glances)
    assert all(item["weak"] and item["clip"] in content.CLIPS for item in glances)
    # A hand now comes after everything the duck already did, never before it.
    dispatch(state.session, "act", {"action": "pickup", "params": {}})
    assert core.s.posture == "sit" and not core.s.floor
    after = len(core.log.events)
    for second in range(8 * 60 + 6, 8 * 60 + 60):
        state.step_realtime(wall=wall + second)
    assert not [e for e in core.log.events[after:] if e.kind == "duck.walk"]
    assert dispatch(state.session, "replay_check", {})["identical"]


def test_real_time_never_skips_a_day_when_the_duck_is_a_little_ahead():
    from datetime import datetime, timedelta

    from duckmind.duckcore.api import dispatch

    ahead = SimState(Config(seed=3))
    core = ahead.session.core
    wall = datetime.now(core.s.now.tzinfo)
    dispatch(ahead.session, "advance", {"until": (wall + timedelta(minutes=10)).strftime("%H:%M")})
    was = core.s.now
    assert ahead.catch_up_to_wall_clock() == 0 and core.s.now == was  # it lives on from there
    assert 8 <= ahead.offset_min() <= 11
    behind = SimState(Config(seed=3))
    dispatch(behind.session, "advance", {"until": (wall - timedelta(minutes=30)).strftime("%H:%M")})
    assert 28 <= behind.catch_up_to_wall_clock() <= 32  # a little behind: it catches up
    assert abs(behind.offset_min()) <= 1


def test_the_page_can_tell_an_old_server_from_a_current_one(server):
    from duckmind.simweb.server import API_VERSION

    reply = _call(server, "state")
    assert reply["server"]["api"] == API_VERSION
    assert reply["server"]["code_changed"] is False
    assert reply["server"]["show_delay_ms"] > 0
    script = _get(server + "/app.js")[1].decode()
    assert f"const API_VERSION = {API_VERSION};" in script  # the page and server agree
    page = _get(server + "/")[1].decode()
    assert 'id="serverWarn"' in page


def test_something_done_in_real_time_comes_after_what_the_duck_already_did():
    from duckmind.duckcore.api import dispatch

    state = SimState(Config(seed=5))
    dispatch(state.session, "advance", {"until": "10:00"})
    dispatch(state.session, "set_setting", {"key": "floor", "value": True})
    state.set_realtime(True)
    core = state.session.core
    wall, begin = state.anchor or (0.0, core.s.now)
    state.step_realtime(wall=wall + 95)
    assert (core.s.now - begin).total_seconds() == 95
    # Clicking later on the wall clock: the server first lives up to that second.
    state.step_realtime(wall=wall + 130)
    assert (core.s.now - begin).total_seconds() == 130
    # Skipping ahead (a banner button) re-anchors instead of freezing the duck for hours.
    dispatch(state.session, "advance", {"minutes": 90})
    state.step_realtime(wall=wall + 131)
    assert state.anchor is not None and state.anchor[1] == core.s.now
    moved = core.s.now
    state.step_realtime(wall=wall + 141)
    assert (core.s.now - moved).total_seconds() == 10


def test_a_real_time_stretch_is_saved_as_one_input_and_comes_back_the_same(tmp_path):
    path = tmp_path / "duck.json"
    first = SimState(Config(seed=3), state_file=path)
    first.set_realtime(True)
    assert first.anchor is not None
    wall = first.anchor[0]
    for second in range(1, 200):
        first.step_realtime(wall=wall + second)
        first.save(soon=True)
    first.set_realtime(False)
    first.save()
    saved = json.loads(path.read_text())
    assert saved["inputs"][-1] == {"method": "advance", "params": {"seconds": 199}}
    again = SimState(Config(seed=1), state_file=path)
    assert again.session.core.log.head == first.session.core.log.head


def test_the_page_keeps_its_banner_buttons_and_shows_words_with_the_moves(server):
    """Round-3 verifier: rebuilding the banner's buttons on every real-time poll swallowed
    clicks; the words said "fell" 2.5 s before the 3D duck fell; at night ▶ gave no hint."""
    script = _worded(_get(server + "/app.js")[1].decode())
    # The buttons are rebuilt only when the list of actions changes.
    assert "function renderBannerActions" in script
    assert script.count("box.innerHTML = actions") == 1
    assert '$("#bannerActions").innerHTML' not in script
    # Real-time polls show their words when the 3D view shows the moves (show_delay_ms).
    assert "queueState(data.state" in script and "show_delay_ms" in script
    page = _get(server + "/")[1].decode()
    assert 'id="bannerMorning"' in page and 'data-act="owner_walk"' in page
    assert "function alignLabel" in script and "先快轉" in script


def test_the_owners_walk_button_reaches_the_duck_through_the_server(server):
    _call(server, "advance", {"until": "09:00"})
    _call(server, "set_setting", {"key": "floor", "value": True})
    reply = _call(server, "act", {"action": "owner_walk", "params": {}})
    assert reply["ok"] and reply["result"]["note"]
    kinds = {item["kind"] for item in reply["feed"]["events"]}
    assert "owner.walk" in kinds


# ---------------------------------------------------------------------------- polish: the leftover page issues


def test_the_page_can_tell_the_server_stopped_or_restarted(server):
    """Verifier: with the server stopped the page kept showing '● 真實速度…' and a stale
    banner for 40 s. The server now names its process (boot) and answers a cheap ping; the
    page says within a few seconds that it cannot reach it, and follows a restarted one
    afresh."""
    status, body = _get(server + "/api/ping")
    ping = json.loads(body)
    reply = _call(server, "state")
    assert status == 200 and ping["ok"] and ping["api"] == reply["server"]["api"]
    assert ping["boot"] and ping["boot"] == reply["server"]["boot"]
    assert json.loads(_get(server + "/api/body")[1])["boot"] == ping["boot"]
    script = _worded(_get(server + "/app.js")[1].decode())
    for hook in ("/api/ping", "function netFail", "function serverRestarted", "server-down"):
        assert hook in script, hook
    assert "連不上模擬器伺服器" in script and "Failed to fetch" not in script
    three = _worded(_get(server + "/duck3d.js")[1].decode())
    assert 'addEventListener("duck:net"' in three and "連不上伺服器" in three


def test_leaving_a_fast_forward_never_plays_its_backlog_as_now(server):
    """Verifier: after a switch from 1 s = 1 min to real time, six stale clips played for
    33 s; after pausing a fast-forward, its backlog went on while time stood still. The 3D
    view now applies that backlog at once, and a fast-forward step that answers after the
    switch is applied, not played; ▶ says what it did and never shows 快轉 in between."""
    three = _worded(_get(server + "/duck3d.js")[1].decode())
    assert "function flushBacklog" in three and "detail.stale" in three
    script = _worded(_get(server + "/app.js")[1].decode())
    assert "opts.ff" in script and "ui.rtPending" in script and "function announcePlay" in script
    assert "● 真實時間：開始中…" in script
    # the TV button names what was pressed, read before the answer relabels it
    assert 'label = T(params.on ? "開電視" : "關電視")' in script
    # every old routine is offered again (they keep a nest duck charged now)
    assert "沒有窩的小鴨才用" not in script


def test_the_feed_says_in_words_what_the_duck_is_up_to():
    """The 3D view captions what has no move of its own: setting off, the way home, the
    nest, guesses and clever or goofy moments."""
    from duckmind.duckcore.api import NOTE_KINDS, Session, dispatch, feed

    session = Session(Config(seed=3, nest=True))
    dispatch(session, "live", {"family": "elder_home", "days": 10})
    items = feed(session, max(0, len(session.core.log.events) - 3000), 0, limit=3000)["events"]
    noted = {item["kind"] for item in items if item.get("note")}
    assert {"duck.excursion", "nav.home_start", "nest.dock_try"} <= noted, noted
    assert all(
        item["kind"] in NOTE_KINDS and item["motion"] is None for item in items if item.get("note")
    )


def test_a_restarted_physics_body_is_told_what_the_duck_is_doing(tmp_path):
    """Verifier: after a re-hatch the physics body restarted sitting while the paused duck
    stood (whatever happened while it started was dropped). A body that becomes ready is
    now resynced by the server's clock, paused or not."""
    from duckmind.duckcore.api import Session, dispatch
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    class Body:
        error = None

        def start(self) -> None:
            return None

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths)  # never started: no processes
    bridge.body = Body()  # type: ignore[assignment]
    bridge._start(bridge.body)  # type: ignore[arg-type]
    assert bridge.ready and bridge.take_fresh() and not bridge.take_fresh()
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "act", {"action": "power_on"})
    bridge.resync(session)
    assert {"power": "on"} in [item[2] for item in bridge._due]


# ---------------------------------------------------------------------------- phase 0: the truth on the page


class _Body:
    """A live body without processes (robotd, MuJoCo): a still frame, a monitor and a world
    that answers the world ops (x_place moves the frame; the others are recorded)."""

    error = None
    host_info = {"ops": ["x_place", "x_world", "x_hand", "x_people", "x_roll_ball", "x_tidy"]}
    home = "default_home"
    policy = "sit"

    def __init__(self):
        self.trunk = [0.02, 0.0, 0.06]
        self.calls = []

    def frame(self):
        return {
            "t": 4.0,
            "joints": {"left_knee": 1.0},
            "trunk": list(self.trunk),
            "quat": [1.0, 0.0, 0.0, 0.0],
            "policy": "sit",
            "fallen": False,
            "realtime": 1.0,
        }

    def world(self):
        return {
            "sim_time": 4.0,
            "home": {"name": "default_home", "source": "layout", "fingerprint": "f"},
            "hand": {"state": "idle", "holding": False, "shown": False, "held": False},
            "people": [
                {
                    "who": "child",
                    "zh": "小孩",
                    "state": "standing",
                    "pos_mm": [800, 0, 0],
                    "clearance_mm": 760,
                }
            ],
            "mocap": [
                {"name": "person_child", "pos_mm": [800, 0, 900], "quat": [1, 0, 0, 0]},
                {"name": "person_owner", "pos_mm": [0, 0, -3000], "quat": [1, 0, 0, 0]},
            ],
            "things": [{"id": "ball_felt", "pos_mm": [800, -600, 0], "quat": [1, 0, 0, 0]}],
            "battery": {"percent": 83, "volts_mv": 7900, "charging": False, "fed_to_robotd": False},
            "nest": None,
            "lights": {"backlight": None, "rooms": {"living": {"label": "白天", "lux": 300.0}}},
        }

    def world_call(self, op, **params):
        self.calls.append((op, params))
        if op == "x_place":
            self.trunk = [params["x_mm"] / 1000, params["y_mm"] / 1000, 0.06]
            return {"ok": True, "x_mm": params["x_mm"], "y_mm": params["y_mm"]}
        if op == "x_hand":
            return {"ok": True, "script": 1, "eta_s": 1.8, "why_zh": []}
        return {"ok": True, "op": op}

    def status(self):
        monitor = {
            "collision": "allcollisions",
            "lockstep": True,
            "max_stall_ms_10s": 151.0,
            "max_stall_ms_60s": 151.0,
            "robotd_hz": 50,
            "realtime": 0.99,
            "trust": "suspect",
            "why_zh": ["最近 10 秒物理停過 151 毫秒"],
        }
        return {
            "running": True,
            "policy": "sit",
            "fallen": False,
            "realtime": 1.0,
            "error": None,
            "monitor": monitor,
        }

    def consume(self, motions):
        return None

    def stop(self):
        return None


class _Eye:
    """The duck's-eye camera as the server reads it (no rendering)."""

    def __init__(self, jpeg):
        self.jpeg = jpeg
        self.wall = 1_700_000_000.0

        self.touched = 0

    def touch(self):
        self.touched += 1  # (F4: asking is what keeps the world's camera window open)

    def latest(self):
        if self.jpeg is None:
            return None
        return {"jpeg": self.jpeg, "wall": self.wall, "frames": 5, "t": 12.5}

    def status(self):
        if self.jpeg is None:
            return {"running": True, "error": None, "note": "等物理身體的姿勢"}
        return {
            "running": True,
            "error": None,
            "note": "",
            "t": 12.5,
            "wall": self.wall,
            "light": {"level": "bright", "level_zh": "亮"},
            "label": "白天",
            "lux": 300.0,
            "render_ms": 7.1,
            "frames": 5,
        }

    def stop(self):
        return None


JPEG = b"\xff\xd8\xff\xe0" + b"duck" * 10 + b"\xff\xd9"


@pytest.fixture()
def live_server(tmp_path):
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes
    bridge.sync_quiet_s = 0
    bridge.body = _Body()  # type: ignore[assignment]
    bridge.ready = True
    bridge.eye = _Eye(JPEG)  # type: ignore[assignment]
    state = SimState(Config(seed=7, name="測試"), live=bridge)
    bridge.resync(state.session, "start")
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", bridge, state
    httpd.shutdown()
    httpd.server_close()


def test_the_duck_eye_picture_is_served_or_the_page_is_told_why_not(server, live_server):
    status, body = _get(server + "/api/eye.jpg")
    assert status == 404 and "--live" in body.decode("utf-8")  # no live physics, in words
    base, bridge, _ = live_server
    with urllib.request.urlopen(base + "/api/eye.jpg", timeout=10) as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.headers["Cache-Control"] == "no-store"
        tag = response.headers["ETag"]
        assert response.read() == JPEG and tag
    assert bridge.eye.touched == 1  # the page asking is what keeps the window open (F4)
    again = urllib.request.Request(base + "/api/eye.jpg", headers={"If-None-Match": tag})
    with pytest.raises(urllib.error.HTTPError) as unchanged:
        urllib.request.urlopen(again, timeout=10)
    assert unchanged.value.code == 304  # the same picture is not sent twice
    bridge.eye = _Eye(None)  # the window just opened: the first picture is on its way (F4)
    status, body = _get(base + "/api/eye.jpg")
    assert status == 202 and "還沒有第一張" in body.decode("utf-8")
    assert bridge.eye.touched == 1
    bridge.eye = None
    status, body = _get(base + "/api/eye.jpg")
    assert status == 404 and body.decode("utf-8")


def test_the_body_carries_its_truth_the_monitor_and_the_eye(live_server):
    base, _, state = live_server
    body = json.loads(_get(base + "/api/body")[1])
    assert body["enabled"] and body["available"]
    monitor = body["monitor"]  # at the top level, not only inside a frame
    assert monitor["trust"] == "suspect" and monitor["why_zh"]
    assert {"max_stall_ms_10s", "robotd_hz", "realtime", "collision", "lockstep"} <= set(monitor)
    eye = body["eye"]
    assert {"t", "age_s", "level", "label", "render_ms", "error"} <= set(eye)
    assert (eye["level"], eye["label"], eye["error"]) == ("bright", "白天", None)
    assert eye["age_s"] > 0 and eye["frames"] == 5
    truth = body["frame"]["truth"]
    plan = state.session.core.pose_now()
    # Phase 2: the body was put where the duck is (x_place, "start"); the truth is the world's.
    assert (truth["x_mm"], truth["y_mm"]) == (round(plan[0] * 1000), round(plan[1] * 1000))
    assert truth["placed_why"] == "start" and truth["placed_at"] > 0 and truth["frame"] == "world"
    assert {"joints", "trunk", "quat"} <= set(body["frame"])  # the physics frame, as before
    world = body["world"]  # the world for the page (the home, people, things, the battery)
    assert world["physical"] and world["frame"] == {"x_mm": 0, "y_mm": 0, "heading_mrad": 0}
    assert world["people"][0]["who"] == "child" and set(world["bodies"]) == {"person_child"}
    assert world["battery"]["percent"] == 83 and world["battery"]["fed_to_robotd"] is False
    assert world["things"][0]["id"] == "ball_felt" and world["placed"]["ok"]


def test_every_answer_carries_the_light_at_the_ducks_time_and_its_position_error(server):
    from datetime import datetime

    from duckmind.world import lights

    reply = _call(server, "advance", {"until": "20:00"})
    light = reply["server"]["lights"]
    when = datetime.fromisoformat(reply["state"]["clock"]["iso"])
    expected = lights.light_state(when)
    assert (light["key"], light["label"]) == (expected.key, expected.label) == ("lamps", "晚上開燈")
    assert light["lux"] == pytest.approx(expected.lux, rel=1e-3)
    assert light["three"]["lamp"]["intensity"] == pytest.approx(
        expected.three()["lamp"]["intensity"]
    )
    assert {"daylight", "glow", "lamps", "elevation", "phase"} <= set(light)
    assert _call(server, "advance", {"until": "12:00"})["server"]["lights"]["label"] == "白天"
    assert isinstance(reply["state"]["body"]["pose_err_mm"], int)


def test_the_page_contract_for_the_truth(server):
    from duckmind.simweb import server as simserver

    assert simserver.API_VERSION >= 8  # truth in the world, the world, the home, the buttons
    assert {"bodyhost.py", "world", "eye"} <= set(simserver.WATCHED)  # a change asks for a restart
    assert _get(server + "/truth.js")[0] == 200
    three = _get(server + "/duck3d.js")[1].decode()
    assert 'from "./truth.js"' in three and 'CustomEvent("duck:body"' in three


def test_a_broken_body_is_said_in_plain_words_and_its_picture_is_not_now(live_server):
    """robotd died (LiveBody keeps its error): /api/body says so plainly — the program's own
    words stay in ``error`` — and the duck's-eye status stops calling its last picture fresh."""
    base, bridge, _ = live_server
    bridge.body.error = "robotd command: [Errno 32] Broken pipe"
    body = json.loads(_get(base + "/api/body")[1])
    assert body["enabled"] and not body["available"] and not body["starting"]
    assert body["error"] == "robotd command: [Errno 32] Broken pipe"
    assert "控制程式（robotd）" in body["error_zh"] and "重設物理身體" in body["error_zh"]
    assert "出錯" in body["eye"]["error"]
    assert body["monitor"]["trust"] == "suspect"  # (the monitor keeps reporting)
    from duckmind.simweb.live import plain_error

    refused = "RuntimeError: duck-body exited early with code 3: == duckmind bodyhost 拒絕啟動：…"
    assert "核對" in (plain_error(refused) or "") and plain_error(None) is None
    for raw in ("body oracle: timed out", "TimeoutError: live body never reached stand"):
        said = plain_error(raw) or ""
        assert said and "Error" not in said and "oracle" not in said, raw


# ---------------------------------------------------------------------------- phase 2: the physical home


def test_the_page_gets_the_home_the_physics_runs(server, monkeypatch):
    """/api/home and /api/body?home=: the layout the physics runs (default_home; 蛋黃 without a
    nest gets its no_nest variant: mat and desk, A2), plus the hand's, the family's and the
    nest's shapes; /api/body carries it only when the page's copy is another one."""
    monkeypatch.delenv("DUCKMIND_LIVE_HOME", raising=False)
    plain = json.loads(_get(server + "/api/home")[1])  # (Config's default: a duck without a nest)
    assert plain["name"] == "default_home,no_nest" and plain["layout"]["nest"] is None
    assert any(el["kind"] == "mat" for el in plain["layout"]["elements"])  # its mat (A2)
    _call(server, "new", {"seed": 7, "name": "測試", "nest": True})
    home = json.loads(_get(server + "/api/home")[1])
    assert home["name"] == "default_home" and home["fingerprint"] and home["error"] is None
    layout = home["layout"]
    assert layout["schema"] == "home-page-v1" and layout["nest"] is not None  # (a nest duck)
    walls = [el for el in layout["elements"] if el["kind"] == "wall"]
    assert len(walls) == 4 and all(el["flags"]["collide"] for el in walls)  # walls are physical
    assert home["hand"]["body"] == "kit_hand" and "child" in home["people"]
    assert home["nest_parts"]["present"] and home["nest_parts"]["tags"][0]["id"] == 17
    first = json.loads(_get(server + "/api/body?home=")[1])
    assert first["home_json"]["fingerprint"] == home["fingerprint"]
    known = json.loads(_get(server + "/api/body?home=" + home["fingerprint"])[1])
    assert "home_json" not in known and known["home"]["fingerprint"] == home["fingerprint"]
    assert "home_json" not in json.loads(_get(server + "/api/body")[1])  # (not asked)
    reply = _call(server, "state")
    assert reply["server"]["home"]["fingerprint"] == home["fingerprint"]
    from duckmind.simweb import homeview

    assert homeview.page_home(nest=False) == "default_home,no_nest"
    no_nest = homeview.home_payload("default_home,no_nest")
    assert no_nest["layout"]["nest"] is None and no_nest["nest_parts"] == {"present": False}
    monkeypatch.setenv("DUCKMIND_LIVE_HOME", "bodyhost")  # phase 0's body host: drawn only
    assert homeview.page_home(nest=True) == "default_home"
    broken = homeview.home_payload("no_such_home")
    assert broken["layout"] is None and "讀不到" in broken["error"]


def test_the_world_buttons_reach_the_world_and_never_duck_core(server, live_server):
    """D2 「扶起物理身體」 (the world's hand), the curiosity demonstrations (示範), rolling a ball
    and tidying up: world ops, and Duck Core's log does not change. Without live physics they
    say so plainly."""
    reply = urllib.request.Request(
        f"{server}/api/call",
        data=json.dumps({"method": "body_upright"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as off:
        urllib.request.urlopen(reply, timeout=10)
    assert off.value.code == 400 and "--live" in json.loads(off.value.read())["error"]
    base, bridge, state = live_server
    events = len(state.session.core.log.events)
    upright = _call(base, "body_upright")
    assert upright["ok"] and "扶起來" in upright["result"]["note"]
    assert bridge.body.calls[-1] == ("x_hand", {"do": "upright"})
    roll = _call(base, "body_roll", {"thing": "ball_felt"})
    assert roll["ok"] and bridge.body.calls[-1] == (
        "x_roll_ball",
        {"toward": "duck", "id": "ball_felt"},
    )
    tidy = _call(base, "body_tidy")
    assert tidy["ok"] and bridge.body.calls[-1] == ("x_tidy", {"to": "toy_corner"})
    wrong = urllib.request.Request(
        f"{base}/api/call",
        data=json.dumps({"method": "body_show", "params": {"kind": "dance"}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(wrong, timeout=10)
    assert refused.value.code == 400 and "示範只有" in json.loads(refused.value.read())["error"]
    assert len(state.session.core.log.events) == events  # Duck Core was not told anything


def test_a_curiosity_demonstration_runs_on_the_body_and_reports_from_the_world(live_server):
    """示範 (curiosity P0): the runner steers the body (drive / posture / skill), the world puts
    the thing where the skill meets it (x_place_prop), and the result comes from the world."""
    base, bridge, _ = live_server

    class World(type(bridge.body)):  # the stand-in, with a thing and a standing body
        policy = "stand"

        def __init__(self):
            super().__init__()
            self.drives, self.skills = [], []

        def world_call(self, op, **params):
            self.calls.append((op, params))
            if op == "x_world":
                return {
                    "duck": {"pos_mm": [0, 0, 120], "yaw_mrad": 0, "tilt_mrad": 20},
                    "things": [{"id": "ball_felt", "pos_mm": [300, 0, 0]}],
                }
            if op == "x_place_prop":
                return {"ok": True, "pos_mm": [params["x_mm"], params["y_mm"], 0]}
            return super().world_call(op, **params)

        def drive(self, vx, vy, vyaw, seconds=0.6):
            self.drives.append((vx, vy, vyaw))

        def skill(self, name):
            self.skills.append(name)

        def request_posture(self, posture):
            return None

    body = World()
    bridge.body = body
    started = _call(base, "body_show", {"kind": "peck", "thing": "ball_felt"})
    assert started["ok"] and "示範" in started["result"]["note"]
    import time as _time

    deadline = _time.monotonic() + 20
    while bridge.demo_result is None and _time.monotonic() < deadline:
        _time.sleep(0.1)
    result = bridge.demo_result
    assert result is not None and result["demo"] is True and result["kind"] == "peck"
    assert body.skills == ["ground_pick"]
    placed = [params for op, params in body.calls if op == "x_place_prop"]
    assert placed and placed[0]["id"] == "ball_felt" and placed[0]["x_mm"] == 115
    view = json.loads(_get(base + "/api/body")[1])["world"]["demo"]
    assert view["running"] is False and view["kind"] == "peck"


def test_a_touch_as_a_family_member_walks_over_first_then_reaches_duck_core(live_server):
    """D3: with a world that has the household, a touch pressed as a family member brings that
    person's body over first; Duck Core hears it when the hand gets there (a Due), and the
    body is not moved again for what the hand already did."""
    base, bridge, state = live_server

    class Visits:
        def __init__(self):
            self.visits, self.asked = [], []

        def request(self, action, persona, params):
            self.asked.append((action, persona, params))
            if action not in ("pickup", "touch"):
                return None
            return {"visit": 1, "persona": persona, "state": "walking", "eta_s": 3.2, "why_zh": []}

        def people(self):
            return {"child": {"zh": "小孩"}}

        def view(self):
            return [{"visit": 1, "action": "pickup", "persona": "child", "state": "walking"}]

    state.family = Visits()
    inputs_before = len(state.session.inputs)
    # A banner's suggested steps (no "visit") act at once, in their order.
    now = _call(base, "act", {"action": "touch", "persona": "child", "params": {"seconds": 2}})
    assert now["ok"] and not now["result"].get("deferred")
    assert len(state.session.inputs) == inputs_before + 1 and state.family.asked == []
    events = len(state.session.core.log.events)
    reply = _call(
        base, "act", {"action": "pickup", "persona": "child", "params": {}, "visit": True}
    )
    assert (
        reply["ok"]
        and reply["result"]["deferred"]
        and "小孩先走到小鴨旁邊" in reply["result"]["note"]
    )
    assert len(state.session.core.log.events) == events  # not yet: the person is walking
    said = _call(
        base, "act", {"action": "call_name", "persona": "child", "params": {}, "visit": True}
    )
    assert said["ok"] and not said["result"].get("deferred")  # voice is no touch: at once
    from duckmind.simweb.family import Due

    before = len(state.session.core.log.events)
    with state.lock:
        state.act_due(Due(1, "pickup", "child", {}, {"hand": "pickup", "holding": True}, 3.4))
    kinds = [event.kind for event in state.session.core.log.events[before:]]
    assert "sense.pickup" in kinds
    # A duck with a nest is held up by the hand already: no second hand for the pickup.
    hands = [item[2]["hand"] for item in bridge._due if "hand" in item[2]]
    assert {"do": "pickup"} not in hands
    world = json.loads(_get(base + "/api/body")[1])["world"]
    assert world["family"]["visits"][0]["persona"] == "child"
    assert world["family"]["done"][-1]["action"] == "pickup"


class _CountingLive:
    """A stand-in live bridge that only records how the server asks it to follow the plan."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.clock = None

    def take_fresh(self) -> bool:
        return False

    def set_clock(self, *args: object) -> None:
        return None

    def world_port(self) -> None:
        return None

    def world_ready(self) -> bool:
        return False

    def resync(self, session: object, why: str = "sync") -> None:
        self.calls.append("resync")

    def skipped(self, session: object) -> None:
        self.calls.append("skipped")

    def after_call(self, *args: object, **kwargs: object) -> None:
        self.calls.append("after_call")

    def snapshot(self) -> dict:
        return {}


@pytest.mark.parametrize("minutes", [1, 5, 15])
def test_a_fast_forward_at_any_page_speed_places_the_body_once_it_stops(minutes):
    """The page's speeds (1, 5 and 15 min/s) all go through the debounced placement: a run of
    steps never makes a synchronous x_place on every step (which would also cancel the hand
    and reset the joints each second, and re-roll where the body lands)."""
    live = _CountingLive()
    state = SimState(Config(seed=7, name="測試"), live=live)  # type: ignore[arg-type]
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _call(base, "advance", {"until": "09:00"})
        live.calls.clear()
        for _ in range(10):
            assert _call(base, "advance", {"minutes": minutes})["ok"]
        assert "resync" not in live.calls and live.calls.count("skipped") >= 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_pages_light_is_the_light_of_the_room_the_duck_is_in():
    """Spec §11.3: the page reads the light from the same layout and model the world's camera is
    lit by (world.lights per room). In default_home that is phase 0's room, value for value; in
    a home with rooms, the room the duck is in."""
    from datetime import datetime, timedelta

    from duckmind.simweb import homeview
    from duckmind.simweb.server import lights_payload

    moment = datetime(2026, 12, 20, 0, 0)
    for _ in range(48):  # a day, every half hour
        for home in ("default_home", "default_home,no_nest"):
            assert lights_payload(moment, home, (0.3, -0.2)) == lights_payload(moment)
        moment += timedelta(minutes=30)
    assert lights_payload(moment, "no_such_home", (0.0, 0.0)) == lights_payload(moment)
    assert homeview.plan_in_world("default_home", 1.2, -0.4) == pytest.approx((1.2, -0.4))
    if _CONFIGURED is None or not (_CONFIGURED / "apartment.xml").is_file():
        return
    noon = datetime(2026, 12, 20, 12, 0)
    living = lights_payload(noon, "apartment", (-2.5, -1.5))  # windows (B3)
    bathroom = lights_payload(noon, "apartment", (2.2, -2.0))  # no window: bounced light only
    assert living["lux"] > 5 * bathroom["lux"] > 0
