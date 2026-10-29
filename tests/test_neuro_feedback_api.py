"""``POST /api/neuro/feedback`` (phase 2 「學會聽話」): the owner's 「乖」, a treat or 「不對」 for a trick
reaches the duck's brain (``Brain.trick_feedback``, here a stand-in with the contract's signature over a real
:class:`~duckmind.neuro.tricks.TrickBook`) through the driver, and Duck Core (the life duck on screen) hears the
phone's word and bridge-v2's praise / scold; what it cannot take is a 400 with a reason in six languages."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore.api import Config
from duckmind.i18n import msg, render
from duckmind.life.clock import DuckClock
from duckmind.life.record import LifeRecord
from duckmind.neuro.command_words import LANGS
from duckmind.neuro.tricks import TrickBook
from duckmind.simweb import homeview
from duckmind.simweb import neuro_api as N
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimServer, SimState, make_handler

TPE = timezone(timedelta(hours=8))
BIRTH = datetime(2026, 10, 1, 8, 0, tzinfo=TPE)


class FakeBrain:
    """``Brain.trick_feedback(kind, trick=None) -> dict`` as the brain slice promises it: the trick named, else
    the one it did last (``did``), stepped in a real trick book; ``no_trick`` when there is none."""

    def __init__(self) -> None:
        self.book = TrickBook()
        self.did: str | None = None
        self.calls: list[tuple[str, str | None]] = []
        self.learn = SimpleNamespace(people={}, places={}, things={}, tricks=self.book)
        self.hab = object()

    def trick_feedback(self, kind: str, trick: str | None = None) -> dict[str, Any]:
        self.calls.append((kind, trick))
        chosen = trick or self.did
        if chosen is None:
            return {"trick": None, "proficiency": None, "step": 0.0, "capped": False,
                    "shaped": False, "counted": False, "reason": "no_trick"}  # fmt: skip
        step = self.book.scold(chosen, 3) if kind == "scold" else self.book.praise(chosen, 3)
        return {"trick": chosen, "proficiency": step.proficiency, "step": step.delta,
                "capped": step.capped, "shaped": step.shaped, "counted": kind != "scold",
                "reason": None}  # fmt: skip


def _driver(brain: Any, *, running: bool = True) -> NeuroDriver:
    lanes = Lanes(brain=brain, names={"brain": "fake", "senses": "stub", "motor": "stub"})
    driver = NeuroDriver(NeuroSpec(seed=4107), lanes=lanes)
    driver.state = "running" if running else "stopped"
    return driver


def _post(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


@pytest.fixture()
def life(tmp_path: Path):
    """A server with a life whose duck is on screen: the hub's source is the driver over the fake brain,
    the life host hears the feedback (``hub.core_feedback``), as ``server.main`` wires it."""
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
    brain = FakeBrain()
    host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True), lanes=SimpleNamespace(brain=brain)
    )
    hub = N.NeuroHub(enabled=True)
    hub.attach(N.DriverSource(lambda: _driver(brain)))
    hub.core_feedback = host.feedback
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api/neuro/feedback", state, brain
    httpd.shutdown()
    httpd.server_close()
    homeview.set_home(None)


def _kinds_since(state: SimState, before: int) -> list[str]:
    return [e.kind for e in state.session.core.log.events[before:]]


# ── what it cannot take ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"kind": "pat"}, "bad_kind"),
        ({}, "bad_kind"),
        ({"kind": "praise", "trick": "fly"}, "bad_trick"),
        ({"kind": "praise"}, "no_trick"),  # (it did no trick just now, and none was named)
    ],
)
def test_what_it_cannot_take_is_a_400_with_its_reason_and_duck_core_hears_nothing(
    life: Any, body: dict[str, Any], reason: str
) -> None:
    url, state, brain = life
    before = len(state.session.core.log.events)
    status, answer = _post(url, body)
    assert status == 400
    assert answer["ok"] is False and answer["reason"] == reason
    assert answer["error"] == msg(f"neuro.feedback.error.{reason}")
    assert answer["_i18n"]["error"]["k"] == f"neuro.feedback.error.{reason}"
    assert not [k for k in _kinds_since(state, before) if k.startswith(("owner.", "pad."))]
    assert brain.book.to_state() == TrickBook().to_state()


def test_a_duck_whose_brain_is_not_running_or_the_stub_says_so() -> None:
    for source in (
        N.DriverSource(lambda: _driver(FakeBrain(), running=False)),
        N.DriverSource(lambda: None),
    ):
        hub = N.NeuroHub(enabled=True)
        hub.attach(source)
        with pytest.raises(N.FeedbackError) as caught:
            hub.feedback({"kind": "praise", "trick": "bow"})
        assert caught.value.reason == "not_running"
    hub = N.NeuroHub(enabled=True)
    hub.attach(N.StubSource(hub.card()))
    with pytest.raises(N.FeedbackError) as caught:
        hub.feedback({"kind": "scold"})
    assert caught.value.reason == "not_running"
    off = N.NeuroHub(enabled=False)
    with pytest.raises(N.FeedbackError) as caught:
        off.feedback({"kind": "praise"})
    assert caught.value.reason == "not_running"


# ── what it takes ────────────────────────────────────────────────────────────────────────────


def test_praise_steps_the_trick_in_the_brain_and_duck_core_hears_the_phone_and_the_trick(
    life: Any,
) -> None:
    url, state, brain = life
    before = len(state.session.core.log.events)
    status, answer = _post(url, {"kind": "praise", "trick": "bow"})
    assert status == 200, answer
    assert brain.calls == [("praise", "bow")]
    assert answer["ok"] is True and (answer["kind"], answer["trick"]) == ("praise", "bow")
    assert (
        answer["proficiency"] == pytest.approx(brain.book.proficiency("bow")) and answer["step"] > 0
    )
    assert (answer["capped"], answer["shaped"]) == (False, False)
    assert answer["note_key"] == "neuro.feedback.note.praised"
    assert answer["_i18n"]["note"]["k"] == "neuro.feedback.note.praised"
    assert answer["brain"] == "FakeBrain"
    assert answer["core"] == {"act": "owner_good", "perceived": True, "event": "praise", "taken": True,
                              "result": "praised"}  # fmt: skip
    kinds = _kinds_since(state, before)
    assert "owner.good" in kinds and "owner.trick_good" in kinds
    assert state.session.core.s.prefs["household.trick:bow"] > 5000


def test_a_treat_is_the_handsets_praise_and_a_scold_is_not_that(life: Any) -> None:
    url, state, brain = life
    before = len(state.session.core.log.events)
    status, treat = _post(url, {"kind": "treat", "trick": "spin"})
    assert status == 200 and treat["core"]["act"] == "pad_good" and treat["step"] > 0
    status, scold = _post(url, {"kind": "scold", "trick": "spin"})
    assert status == 200 and scold["core"]["act"] == "owner_not_that"
    # (a trick still below the floor keeps what it has: the brain's book says by how much)
    assert scold["step"] == pytest.approx(scold["proficiency"] - treat["proficiency"])
    assert scold["step"] <= 0 and scold["note_key"] == "neuro.feedback.note.scolded"
    kinds = _kinds_since(state, before)
    assert {"pad.good", "owner.trick_good", "owner.not_that", "owner.trick_not_that"} <= set(kinds)
    assert brain.calls == [("treat", "spin"), ("scold", "spin")]


def test_no_trick_named_is_the_one_it_did_last(life: Any) -> None:
    url, _state, brain = life
    brain.did = "shake"
    status, answer = _post(url, {"kind": "praise"})
    assert status == 200 and answer["trick"] == "shake"
    assert brain.calls == [("praise", None)]


def test_the_growth_picture_shows_the_new_proficiency_at_once(life: Any) -> None:
    url, state, brain = life
    host = state.life_host

    def bow_pct() -> int:
        with state.lock:
            rows = host.growth_view()["learned"]["tricks"]
        return next(row for row in rows if row["id"] == "bow")["proficiency_pct"]

    assert bow_pct() == 0  # (the picture is now kept for a few seconds)
    status, answer = _post(url, {"kind": "praise", "trick": "bow"})
    assert status == 200 and answer["step"] > 0
    assert bow_pct() == round(100 * brain.book.proficiency("bow")) > 0


def test_the_days_cap_says_practise_tomorrow(life: Any) -> None:
    url, _state, _brain = life
    notes = [_post(url, {"kind": "praise", "trick": "lie_down"})[1]["note_key"] for _ in range(6)]
    assert notes[0] == "neuro.feedback.note.praised"
    assert notes[-1] == "neuro.feedback.note.capped"


def test_a_server_without_a_life_steps_the_brain_alone() -> None:
    brain = FakeBrain()
    hub = N.NeuroHub(enabled=True)
    hub.attach(N.DriverSource(lambda: _driver(brain)))
    answer = hub.feedback({"kind": "praise", "trick": "flip"})
    assert answer["ok"] and answer["core"] is None and brain.book.proficiency("flip") > 0


# ── the words ────────────────────────────────────────────────────────────────────────────────


def test_every_feedback_word_is_in_the_six_languages() -> None:
    keys = [
        f"neuro.feedback.error.{r}" for r in ("bad_kind", "bad_trick", "no_trick", "not_running")
    ]
    keys += [
        f"neuro.feedback.note.{n}"
        for n in ("praised", "shaped", "capped", "scolded", "not_counted")
    ]
    for key in keys:
        source = msg(key, trick="bow", pct=40)
        for lang in LANGS:
            words = render(source, lang, strict=True)
            assert words.strip()
            if lang not in ("zh-Hant", "zh-Hans"):
                assert words != str(source), (key, lang)
    assert (
        render(msg("neuro.feedback.note.praised", trick="bow", pct=40), "en")
        == "“Bow” got a bit better: 40%"
    )
