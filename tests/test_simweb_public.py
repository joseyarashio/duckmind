"""Public play (``python -m duckmind.simweb --public``, duckmind.simweb.public): only the visitor's endpoints
exist (a refused one per category answers 404 with the plain reason), no answer carries a path of this machine,
the visit ends on a fake clock (the duck sits, writes answer 410, the duck file still downloads, then the server
stops), the same seed is the same duck, and the process opens no connection off the machine."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.i18n import msg
from duckmind.life import duckfile as df
from duckmind.neuro.bodylink import BodyRequest
from duckmind.neuro.drives import Drives
from duckmind.neuro.genome import Genome
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.learn import Learner
from duckmind.simweb import duckfile_api, public
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.neuro_live import NeuroDriver
from duckmind.simweb.server import (
    SimServer,
    SimState,
    _make_clock,
    _open_life,
    _public_rest,
    make_handler,
)

SEED = 1234
# (long-lived streams: read by the page for minutes at a time)
STREAMS = {"/api/neuro/stream", "/api/neuro/sound/pcm"}
SRC = Path(__file__).resolve().parents[1] / "src"


class FakeBrain:
    """The parts of a NeuroBrain the snapshot touches (as tests/test_duckfile.py)."""

    def __init__(self, seed: int) -> None:
        p = Genome.from_seed(seed, gd=0).p
        self.learn = Learner(p)
        self.hab = Habituation(p)
        self.drives = Drives()
        self._t0, self._t_prev = 0, 400_000


# What main() reads that public play picks or keeps (the parser's defaults).
DEFAULTS: dict[str, Any] = {
    "seed": 7,
    "name": None,
    "life": None,
    "life_backdate_days": 0,
    "life_birth": None,
    "life_eye_from": None,
    "live_voice": False,
    "live_voice_name": None,
    "live_voice_cap_minutes": None,
    "body": None,
    "body_token_file": None,
    "neuro_stub": False,
    "neuro_duck": "",
    "neuro_time": None,
    "neuro_record": None,
    "clock_speed": 1.0,
    "layouts_dir": None,
}


def args_for(**given: Any) -> SimpleNamespace:
    """The command line PublicImage's container runs (``--public --live … --brain neuro --home real_home``)."""
    fields = {
        **DEFAULTS,
        "public": True,
        "live": True,
        "brain": "neuro",
        "home": "real_home",
        "no_nest": True,
        "tof": True,
        "voice_commands": "auto",
        "expression_caps": "",
        **given,
    }
    return SimpleNamespace(**fields)


def hatch(env: dict[str, str]) -> tuple[public.PublicPlan, Any, SimState]:
    """A public duck made the way main() makes it: its arguments, its clock, its life, its Duck Core."""
    args = args_for()
    plan = public.apply_args(args, DEFAULTS.get, env)
    args.name_given = args.name
    clock = _make_clock(args, neuro=True)
    life, config = _open_life(args, SimpleNamespace(error=pytest.fail), birth=clock.calendar())
    assert life is not None and config is not None
    state = SimState(config, None, life.core_path, life=life, clock=clock, attach_neural=True)
    return plan, life, state


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One computer: its own DUCKMIND_HOME and temp folder (the public duck's folder goes there)."""
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    for name in ("DUCKMIND_DATA_DIR", "DUCKMIND_PATHS_FILE", "DUCKMIND_LINEAGE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    return tmp_path


@pytest.fixture()
def visit(home: Path):
    """A public server living a fresh duck (its mind on a stand-in brain), on a clock the test moves."""
    plan, life, state = hatch({public.SEED_ENV: str(SEED)})
    state.public = True
    host = LifeHost(life, state, state.clock, every_s=600.0)
    state.life_host = host
    host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True),
        lanes=SimpleNamespace(brain=FakeBrain(SEED)),
        _tick_lock=threading.Lock(),
        start_at="mat",
    )
    wall = [1_000_000.0]
    session = public.PublicSession(plan.seed, plan.name, plan.minutes, wall=lambda: wall[0])
    rested: list[str] = []
    session.on_end.append(lambda: rested.append("sat"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, public=session))
    session.on_exit.append(httpd.shutdown)
    serving = threading.Thread(target=httpd.serve_forever, daemon=True)
    serving.start()
    yield SimpleNamespace(
        url=f"http://127.0.0.1:{httpd.server_address[1]}",
        wall=wall,
        session=session,
        rested=rested,
        serving=serving,
        plan=plan,
        life=life,
    )
    httpd.shutdown()
    httpd.server_close()


def call(
    url: str, body: Any = None, *, raw: bytes | None = None
) -> tuple[int, dict[str, str], bytes]:
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data, headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def reason(body: bytes) -> str:
    return json.loads(body)["_i18n"]["error"]["k"]


# ── what exists ──────────────────────────────────────────────────────────────────────────────


REFUSED_GET = [
    "/lab",  # the lab (developer tools)
    "/data",
    "/api/layout",  # the layout editor's reads
    "/api/layout/library",
    "/api/skill-evidence",  # the skill test evidence
    "/skill-evidence/clip.mp4",
    "/api/neuro/sound/log",
    "/api/neuro/remote/ws",  # the remote control's socket
    "/api/neuro/mic",  # the browser microphone
    "/api/voice-nope",
    "/api/../server.py",
    "/floors/x.json",
]
REFUSED_POST = [
    ("/api/live/cloud", {"on": True}),  # cloud live voice
    ("/api/live/say", {"text": "hi"}),
    ("/api/neuro/voice/command", {}),  # voice commands
    ("/api/neuro/mic/chunk", {}),
    ("/api/life/duckfile/import", {}),  # duck file import
    ("/api/life/duckfile/inspect", {}),
    ("/api/layout/apply", {"layout": {}}),  # layout apply / editor saves
    ("/api/layout/save", {"name": "x", "layout": {}}),
    ("/api/layout/reset", {}),
    ("/api/life/fastforward", {"days": 3}),  # lab tools
    ("/api/neuro/clock", {"minute": 120}),  # the duck's time
    ("/api/neuro/remote", {"on": True}),  # remote control of the body
    ("/api/neuro/remote/twist", {"vx": 1}),
    ("/api/neuro/duck", {"preset": "explorer"}),  # the duck picker
    ("/api/call", {"method": "new", "params": {}}),  # Duck Core's writes and world-debug buttons
    ("/api/call", {"method": "advance", "params": {"minutes": 60}}),
    ("/api/call", {"method": "body_show", "params": {}}),
    ("/api/call", {"method": "act", "params": {"action": "pickup"}}),
    ("/api/call", {"method": "set_setting", "params": {"key": "floor", "value": True}}),
    ("/api/call", {"method": "export", "params": {}}),
]


def test_only_the_visitors_endpoints_exist(visit: Any) -> None:
    for path in REFUSED_GET:
        status, _, body = call(visit.url + path)
        assert status == 404, path
        assert reason(body) == "web.public.refused", path
    for path, body in REFUSED_POST:
        status, _, answer = call(visit.url + path, body)
        assert status == 404, (path, body)
        assert reason(answer) == "web.public.refused", (path, body)
    # what the visitor's page reads and does
    for path in ("/", "/index.html", "/public.js", "/style.css", "/api/life", "/api/life/growth",
                 "/api/neuro/status", "/api/live/status", "/api/neuro/voice/status"):  # fmt: skip
        assert call(visit.url + path)[0] == 200, path
    status, _, body = call(visit.url + "/api/ping")
    assert status == 200 and json.loads(body)["public"] is True
    status, _, body = call(visit.url + "/api/call", {"method": "state", "params": {}})
    assert status == 200 and json.loads(body)["ok"]
    # a visitor's action passes on to the duck (no brain here: its own refusal, not public mode's)
    status, _, body = call(visit.url + "/api/neuro/command", {"verb": "sit"})
    assert status not in (404, 410) and json.loads(body).get("public") is None


def test_a_normal_server_has_no_public_mode(home: Path) -> None:
    _plan, _life, state = hatch({public.SEED_ENV: str(SEED)})
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert json.loads(call(url + "/api/ping")[2])["public"] is False
        assert call(url + public.SESSION_ROUTE)[0] == 404
        assert call(url + "/api/layout/library")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── no paths or host details ─────────────────────────────────────────────────────────────────


def test_no_answer_carries_a_path_or_the_host_name(
    visit: Any, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = str(visit.life.folder)
    secret = [folder, str(home), socket.gethostname(), os.path.expanduser("~")]
    for path in sorted(public.GET_EXACT - STREAMS - {"/api/life/duckfile/export"}):
        status, headers, body = call(visit.url + path)
        if "json" not in headers.get("Content-Type", ""):
            continue
        text = body.decode()
        for word in secret:
            assert word not in text, (path, word)
        assert int(headers["Content-Length"]) == len(body)
        json.loads(body)

    # an answer that would name the duck's folder (a failed export): the path is hidden, the rest stays
    def broken(path: Path, **_: Any) -> Any:
        raise df.DuckFileError(msg("web.duckfile.no_duck", folder=str(path)))

    monkeypatch.setattr(df, "pack", broken)
    status, headers, body = call(visit.url + "/api/life/duckfile/export")
    answer = json.loads(body)
    assert status == 409 and folder not in body.decode() and public.HIDDEN in answer["error"]
    assert int(headers["Content-Length"]) == len(body)
    assert "Python" not in headers.get("Server", "")


def test_scrub_hides_machine_paths_and_keeps_page_addresses() -> None:
    text = 'robotd: [Errno 2] "/var/run/x/robotd.sock" ~/.duckmind/lives/a, /api/voice/v3/a.wav and /mesh/b.stl'
    out = public.scrub(text, ["myhost"])
    assert "/var/run" not in out and "~/.duckmind" not in out
    assert "/api/voice/v3/a.wav" in out and "/mesh/b.stl" in out
    assert public.scrub("on myhost now", ["myhost"]) == f"on {public.HIDDEN} now"
    assert public.scrub("1/2 km/h, https://example.org/a/b") == "1/2 km/h, https://example.org/a/b"


# ── the visit's time ─────────────────────────────────────────────────────────────────────────


def test_the_visit_ends_writes_410_the_duck_file_still_downloads_then_the_server_stops(
    visit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _, body = call(visit.url + public.SESSION_ROUTE)
    view = json.loads(body)
    assert status == 200 and view["ok"] and view["seed"] == SEED and view["name"] == visit.plan.name
    assert view["left_s"] == 15 * 60 and not view["ended"]
    assert set(public.REFUSES) <= set(view["refuses"])
    assert visit.session.check() == "play" and visit.rested == []

    visit.wall[0] += 15 * 60  # the time is up
    assert visit.session.check() == "ended" and visit.rested == ["sat"]
    assert visit.session.check() == "ended" and visit.rested == ["sat"]  # (once)
    view = json.loads(call(visit.url + public.SESSION_ROUTE)[2])
    assert view["ended"] and view["left_s"] == 0 and view["export_left_s"] == 180  # (3 minutes)
    for path, body in [
        ("/api/neuro/command", {"verb": "sit"}),
        ("/api/neuro/stimulus", {"name": "pat_head"}),
        ("/api/neuro/feedback", {"kind": "praise"}),
        ("/api/neuro/motion", {"id": "nod"}),
        ("/api/neuro/gothere", {"x_m": 0.1, "y_m": 0.2}),
        ("/api/call", {"method": "act", "params": {"action": "owner_hello"}}),
    ]:
        status, _, answer = call(visit.url + path, body)
        assert status == 410 and reason(answer) == "web.public.ended_grace", path
        assert json.loads(answer)["_i18n"]["error"]["p"]["minutes"] == 3
    # reading goes on, and the duck file downloads — from wherever the visitor is
    assert call(visit.url + "/api/call", {"method": "state", "params": {}})[0] == 200
    monkeypatch.setattr(duckfile_api, "_local", lambda handler: False)
    status, headers, data = call(visit.url + "/api/life/duckfile/export")
    assert status == 200 and headers["Content-Type"] == duckfile_api.MEDIA_TYPE
    assert df.read(data).manifest["duck"]["name"] == visit.plan.name

    visit.wall[0] += public.EXPORT_GRACE_S  # the download's time is over too: the server stops
    assert visit.session.check() == "over"
    visit.serving.join(10)
    assert not visit.serving.is_alive()


def test_at_the_end_the_mind_is_saved_and_the_duck_sits_and_stays_seated() -> None:
    did: list[Any] = []
    host = SimpleNamespace(snapshot_now=lambda why: did.append(("saved", why)))
    _public_rest(SimpleNamespace(neuro=SimpleNamespace(rest=lambda: did.append(("rest",)))), host)
    assert did == [("saved", "public_end"), ("rest",)]
    # what the legs get from then on, whatever the brain asks: no walk, no skill or flight, seated
    asked = BodyRequest(
        vx=0.2, vyaw=0.4, posture="stand", skill="startle", panic=True, approach=True
    )
    rested = NeuroDriver.rest_request(asked)
    assert (rested.vx, rested.vyaw, rested.posture, rested.skill) == (0.0, 0.0, "sit", None)
    assert not (rested.panic or rested.approach)
    assert rested.head == asked.head and rested.mouth == asked.mouth  # (its head and beak go on)


# ── the duck ─────────────────────────────────────────────────────────────────────────────────


def test_the_same_seed_is_the_same_duck_and_it_walks(home: Path) -> None:
    env = {public.SEED_ENV: str(SEED)}
    ducks = [hatch(env) for _ in range(2)]
    other = hatch({public.SEED_ENV: str(SEED + 5)})
    (plan_a, life_a, state_a), (plan_b, life_b, state_b) = ducks
    assert plan_a.folder != plan_b.folder and plan_a.folder.parent == Path(os.environ["TMPDIR"])
    assert life_a.name == life_b.name == public.name_for(SEED) and life_a.nature == life_b.nature
    genomes = [
        Genome.from_core_state(state.session.core.s) for state in (state_a, state_b, other[2])
    ]
    assert state_a.session.core.s.gd == state_b.session.core.s.gd == public.PUBLIC_GD
    assert genomes[0].p == genomes[1].p and genomes[0].unlocked == genomes[1].unlocked
    assert {"wander", "forage", "turn_around"} <= genomes[0].unlocked  # (it walks)
    assert genomes[2].p != genomes[0].p and other[1].name != life_a.name
    # its day starts in the morning (10:00 on its clock), whatever the hour of the visit
    assert state_a.clock.local().hour == 10
    assert args_for().home == "real_home"


def test_every_seed_walks_at_the_public_age() -> None:
    for seed in range(0, 2000, 37):
        assert "wander" in Genome.from_seed(seed, gd=public.PUBLIC_GD).unlocked, seed


def test_public_play_picks_the_duck_itself_and_needs_its_seed(home: Path) -> None:
    with pytest.raises(public.PublicConfigError, match="--seed"):
        public.apply_args(args_for(seed=99), DEFAULTS.get, {public.SEED_ENV: "1"})
    with pytest.raises(public.PublicConfigError, match="--live-voice"):
        public.apply_args(args_for(live_voice=True), DEFAULTS.get, {public.SEED_ENV: "1"})
    with pytest.raises(public.PublicConfigError, match="voice commands"):
        public.apply_args(args_for(voice_commands="mlx:x"), DEFAULTS.get, {public.SEED_ENV: "1"})
    with pytest.raises(public.PublicConfigError, match="--live"):
        public.apply_args(args_for(live=False), DEFAULTS.get, {public.SEED_ENV: "1"})
    for bad in ("", "x", "-3", "99999999999"):
        with pytest.raises(public.PublicConfigError, match=public.SEED_ENV):
            public.apply_args(args_for(), DEFAULTS.get, {public.SEED_ENV: bad})
    with pytest.raises(public.PublicConfigError, match=public.MINUTES_ENV):
        public.minutes_from_env({public.MINUTES_ENV: "0"})
    assert public.minutes_from_env({}) == 15.0
    assert list(Path(os.environ["TMPDIR"]).iterdir()) == []  # (no folder made for a refused start)
    args = args_for()
    plan = public.apply_args(args, DEFAULTS.get, {public.SEED_ENV: "7", public.MINUTES_ENV: "3"})
    assert (args.brain, args.no_nest, args.voice_commands, args.seed) == ("neuro", True, "off", 7)
    assert args.life_backdate_days == public.PUBLIC_GD and plan.minutes == 3.0


# ── no network ───────────────────────────────────────────────────────────────────────────────


def test_a_public_process_opens_no_connection_off_the_machine() -> None:
    code = """
import socket, threading
from duckmind.simweb import public
listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
public.install_network_guard()
for target in (lambda: socket.create_connection(("192.0.2.1", 80), timeout=1),
               lambda: socket.getaddrinfo("example.org", 443)):
    try:
        target()
        raise SystemExit("reached the network")
    except PermissionError:
        pass
socket.create_connection(listener.getsockname(), timeout=2).close()
print("ok")
"""
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
