"""Growth G4: the 「成長」 view and its read-only endpoint (docs/design/GROWTH-G4-2026-10-11.md).

No physics and no real brain: a Duck Core session fed with the bridge's events, a ``LifeHost`` over
it (no body on screen) and the server's handler on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import bridge
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.life import growth_view as gv
from duckmind.life.clock import DuckClock
from duckmind.life.record import LifeRecord
from duckmind.simweb import homeview, i18nwire
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.server import SimServer, SimState, make_handler

TPE = timezone(timedelta(hours=8))
BIRTH = datetime(2026, 10, 1, 8, 0, tzinfo=TPE)


def _session(days: int = 3) -> Session:
    session = Session(Config(seed=4107, name="蛋黃", tof=True, nest=False, start=BIRTH.isoformat()))
    dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    dispatch(session, "advance", {"seconds": days * 86400})
    return session


def _lived(session: Session) -> None:
    t = int(session.core.s.now.timestamp())
    events = [
        {
            "k": "touch",
            "t": t + 1,
            "where": "head",
            "ms": 4000,
            "asleep": 0,
            "settled": 0,
            "doing": "",
        },
        {
            "k": "touch",
            "t": t + 30,
            "where": "back",
            "ms": 3000,
            "asleep": 0,
            "settled": 0,
            "doing": "",
        },
        {"k": "name", "t": t + 60, "tv": 0, "asleep": 0},
        {"k": "fall", "t": t + 90, "x_mm": 100, "y_mm": 200, "outcome": "easy"},
        {"k": "sleep", "t": t + 120, "minutes": 30, "night": 0},
    ]
    dispatch(session, "experience", {"events": events})


def _view(session: Session, **kw: Any) -> dict[str, Any]:
    tally = gv.Tally()
    tally.update(session.inputs)
    defaults: dict[str, Any] = {
        "name": "蛋黃",
        "live": True,
        "branch_id": "explorer",
        "pinned": True,
        "nature_branch": "explorer",
    }
    return gv.build(session.core, tally=tally, **{**defaults, **kw})


# ── the view ────────────────────────────────────────────────────────────────────────────────


def test_the_view_says_who_it_is_and_what_it_lived() -> None:
    session = _session(3)
    _lived(session)
    view = _view(session)
    ident = view["identity"]
    assert ident["name"] == "蛋黃" and ident["seed"] == 4107
    assert ident["growth_day"] == session.core.s.gd >= 3
    assert ident["stage"]["id"] == "hatchling" and ident["next_stage"]["id"] == "duckling"
    assert ident["next_stage"]["in_days"] == ident["next_stage"]["gd"] - ident["growth_day"]
    assert [s["id"] for s in ident["stages"]] == ["hatchling", "duckling", "juvenile", "adult"]
    # a pinned branch is the family's choice, and the engine's own is kept apart
    assert ident["branch"]["id"] == "explorer" and ident["branch"]["source"] == "family"
    assert ident["branch"]["core_id"] == session.core.s.branch
    exp = view["experiences"]
    assert exp["by_kind"] == {"touch": 2, "name": 1, "fall": 1, "sleep": 1} and exp["total"] == 5
    assert [r["kind"] for r in exp["recent"]][0] == "sleep"  # newest first
    assert exp["recent"][0]["minutes"] == 30 and exp["recent"][1]["outcome"] == "easy"
    # the same events as a day's tally in the diary, whatever Duck Core wrote about the day
    today = view["diary"][0]
    assert today["lived"] == {"touch": 2, "name": 1, "fall": 1, "sleep": 1}


def test_a_duck_that_follows_its_growth_has_no_family_branch() -> None:
    session = _session(1)
    view = _view(session, branch_id=session.core.s.branch, pinned=False)
    assert view["identity"]["branch"]["source"] == ("growth" if session.core.s.branch else "none")
    view = _view(session, branch_id=None, pinned=False)
    assert view["identity"]["branch"]["source"] == "none"


def test_temperament_quirks_and_learning_are_plain_numbers_with_messages() -> None:
    session = _session(2)
    _lived(session)
    view = _view(session)
    axes = view["temperament"]["axes"]
    assert [a["axis"] for a in axes] == list(gv.AXES) and all(0 <= a["pct"] <= 100 for a in axes)
    assert len(view["temperament"]["nature"]) == 4
    assert sum(n["pct"] for n in view["temperament"]["nature"]) in range(95, 105)
    learned = view["learned"]
    assert learned["touches"] == session.core.s.touches and learned["falls"] == session.core.s.falls
    assert learned["likes"] + learned["dislikes"] <= learned["prefs_total"]
    assert view["bonds"]["family_pct"] == session.core.s.bond_owner * 100 // 10_000
    assert view["bonds"]["people"] == [] and view["bonds"]["unnamed"] == 0  # (no learner here)
    json.dumps(view)  # (plain data)


def test_a_neural_learner_gives_its_people_by_name_and_never_their_looks() -> None:
    session = _session(1)

    def person(pid: int, name: str | None, bond: float) -> SimpleNamespace:
        return SimpleNamespace(
            pid=pid, name=name, bond=bond, greeted=2, exchanges=3, sightings=5, mean=[0.1, 0.2, 0.3]
        )

    learner = SimpleNamespace(
        people={1: person(1, "小明", 0.4), 2: person(2, None, 0.1), 3: person(3, "阿嬤", 0.9)},
        places={
            (0, 0): SimpleNamespace(value=0.5, hard_falls=0),
            (1, 0): SimpleNamespace(value=-0.5, hard_falls=3),
        },
        things={1: SimpleNamespace(value=0.3)},
    )
    view = _view(session, learner=learner)
    people = view["bonds"]["people"]
    assert [p["name"] for p in people] == ["阿嬤", "小明"] and [p["pct"] for p in people] == [
        90,
        40,
    ]
    assert view["bonds"]["unnamed"] == 1
    assert "mean" not in json.dumps(view) and "0.3" not in json.dumps(people)
    mind = view["learned"]["mind"]
    assert mind == {
        "places": 2, "places_liked": 1, "places_disliked": 1, "places_avoided": 1,
        "things": 1, "things_liked": 1, "things_disliked": 0, "people": 3,
    }  # fmt: skip
    # a names is data: the boundary does not key it, even when it is a word of the engine
    answer = i18nwire.growth_answer({"ok": True, "growth": view})
    assert all("_i18n" not in p for p in answer["growth"]["bonds"]["people"])

    # a learner the brain is changing under us is read again, not an error
    class Busy:
        calls = 0

        @property
        def places(self) -> dict[Any, Any]:
            Busy.calls += 1
            if Busy.calls < 3:
                raise RuntimeError("dictionary changed size during iteration")
            return {}

        things: dict[Any, Any] = {}  # noqa: RUF012
        people: dict[Any, Any] = {}  # noqa: RUF012

    assert _view(session, learner=Busy())["learned"]["mind"]["places"] == 0


def test_words_are_catalog_messages_so_the_page_can_speak_six_languages() -> None:
    session = _session(2)
    answer = i18nwire.growth_answer({"ok": True, "growth": _view(session)})["growth"]
    stage = answer["identity"]["stage"]
    assert stage["_i18n"]["label"] == {"k": "core.content.STAGES.hatchling_zh"}
    assert (
        answer["identity"]["branch"]["_i18n"]["label"]["k"] == "core.content.BRANCHES.explorer_zh"
    )
    axis = answer["temperament"]["axes"][0]
    assert axis["_i18n"]["name"]["k"].startswith("neuro.axis.boldness")
    # the diary's lines are the engine's own sentences, matched to its templates
    line = next(
        (line for day in answer["diary"] for line in day["lines"]),
        None,
    )
    assert line is None or "_i18n" in line
    # the name is data: never keyed
    assert "name" not in answer["identity"].get("_i18n", {})


def test_the_tally_reads_the_inputs_once_and_follows_a_new_session() -> None:
    session = _session(1)
    tally = gv.Tally()
    _lived(session)
    tally.update(session.inputs)
    first = dict(tally.by_kind)
    assert tally.seen == len(session.inputs)
    tally.update(session.inputs)  # (nothing new: nothing counted twice)
    assert dict(tally.by_kind) == first
    t = int(session.core.s.now.timestamp())
    dispatch(session, "experience", {"events": [{"k": "lift", "t": t + 5}]})
    tally.update(session.inputs)
    assert tally.by_kind["lift"] == 1 and tally.by_kind["touch"] == first["touch"]
    fresh = _session(1)  # a hatched-again duck has fewer inputs: start over
    tally.update(fresh.inputs)
    assert tally.by_kind == {} or sum(tally.by_kind.values()) == 0


# ── the endpoint ────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def life_server(tmp_path: Path):
    homeview.set_home("real_home")
    record = LifeRecord.create(
        tmp_path / "life",
        name="蛋黃",
        seed=4107,
        birth=BIRTH,
        nest=False,
        tof=True,
        backdate_days=0,
    )
    wall = [BIRTH.timestamp() + 3 * 86400]
    clock = DuckClock(wall=lambda: wall[0])
    config = Config(seed=4107, name="蛋黃", tof=True, nest=False, start=record.birth.isoformat())
    state = SimState(config, None, record.core_path, life=record, clock=clock, attach_neural=True)
    host = LifeHost(record, state, clock, every_s=600.0, wall=lambda: wall[0])
    state.life_host = host
    host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True), lanes=SimpleNamespace(brain=None)
    )
    dispatch(state.session, "advance", {"seconds": 3 * 86400})
    _lived(state.session)
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state, host, wall
    httpd.shutdown()
    httpd.server_close()
    homeview.set_home(None)


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def test_the_growth_endpoint_is_read_only_and_says_the_life(life_server) -> None:
    base, state, host, wall = life_server
    before = (len(state.session.inputs), state.session.core.s.gd, state.session.core.log.head)
    answer = _get(f"{base}/api/life/growth")
    growth = answer["growth"]
    assert answer["ok"] and growth["api"] == gv.API and growth["live"] is True
    assert (
        growth["identity"]["name"] == "蛋黃" and growth["identity"]["branch"]["source"] == "family"
    )
    assert growth["identity"]["stage"]["_i18n"]["label"]["k"].startswith("core.content.STAGES.")
    assert growth["experiences"]["by_kind"]["touch"] == 2
    assert growth["regrow"]["count"] == 0 and growth["regrow"]["genome"] is None  # (no body here)
    assert growth["on_screen"] is False
    # reading changes nothing
    assert (
        len(state.session.inputs),
        state.session.core.s.gd,
        state.session.core.log.head,
    ) == before
    # the picture is kept for a few seconds, and made again after
    again = _get(f"{base}/api/life/growth")["growth"]
    assert again["calendar"] == growth["calendar"]
    wall[0] += host.GROWTH_CACHE_S + 1
    assert _get(f"{base}/api/life/growth")["growth"]["calendar"] != growth["calendar"]


def test_without_a_life_the_growth_endpoint_says_so() -> None:
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        answer = _get(f"http://127.0.0.1:{httpd.server_address[1]}/api/life/growth")
        assert answer == {"ok": True, "growth": None}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_growth_view_of_a_neural_learner_through_the_host(life_server) -> None:
    _, state, host, _ = life_server
    people = {1: SimpleNamespace(pid=1, name="小明", bond=0.5, greeted=1, exchanges=1, sightings=2)}
    learner = SimpleNamespace(people=people, places={}, things={})
    host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True),
        lanes=SimpleNamespace(brain=SimpleNamespace(learn=learner, hab=object(), genome=None)),
    )
    host._growth_cache = None
    with state.lock:
        view = host.growth_view()
    assert view["bonds"]["people"][0]["name"] == "小明" and view["on_screen"] is True
    # the server never lets the bridge's counts reset a page: the tally is from Duck Core's inputs
    assert view["experiences"]["total"] == 5
