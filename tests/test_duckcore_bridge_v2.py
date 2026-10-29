"""``bridge-v2`` (duckmind.duckcore.bridge_v2): bridge-v1's frozen vocabulary plus the trick lessons
(phase 2 「學會聽話」) — what Duck Core writes and learns of each, and what its save says."""

from __future__ import annotations

import json
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api, bridge, bridge_v2, narrate
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.neuro.command_words import TRICKS

START = "2026-03-02T07:00:00+08:00"


def make(seed: int = 4107, *, attach: bool = True) -> Session:
    session = Session(Config(seed=seed, name="蛋黃", tof=True, nest=False, start=START))
    if attach:
        dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    return session


def now_s(session: Session) -> int:
    return int(session.core.s.now.timestamp())


def give(session: Session, kind: str, **fields: Any) -> dict[str, Any]:
    dispatch(session, "advance", {"seconds": 5})
    answer = dispatch(
        session, "experience", {"events": [{"k": kind, "t": now_s(session), **fields}]}
    )
    return dict(answer["taken"][0])


def written(session: Session, since: int) -> list[Any]:
    return list(session.core.log.events[since:])


# ── the vocabulary ───────────────────────────────────────────────────────────────────────────


def test_v2_is_v1_unchanged_plus_three_trick_kinds_each_with_a_handler() -> None:
    assert bridge.BRIDGE_API == "bridge-v1" and bridge_v2.BRIDGE_API == "bridge-v2"
    assert set(bridge_v2.KINDS) - set(bridge.KINDS) == {"teach_trick", "praise", "scold"}
    for kind, fields in bridge.KINDS.items():
        assert bridge_v2.KINDS[kind] is fields
        assert bridge_v2.HANDLERS[kind] is bridge.HANDLERS[kind]
    assert set(bridge_v2.HANDLERS) == set(bridge_v2.KINDS)
    assert bridge_v2.TRICKS == TRICKS  # (Duck Core imports nothing of the neural duck)


def test_a_trick_event_is_a_known_trick_an_outcome_and_integers() -> None:
    ok = bridge_v2.check({"k": "teach_trick", "t": 100, "trick": "bow", "outcome": "done"})
    assert ok["ms"] == 0
    assert bridge_v2.check({"k": "praise", "t": 100})["trick"] == ""
    for bad in (
        {"k": "teach_trick", "t": 100, "trick": "bow"},
        {"k": "teach_trick", "t": 100, "trick": "fly", "outcome": "done"},
        {"k": "teach_trick", "t": 100, "trick": "bow", "outcome": "interrupted"},
        {"k": "teach_trick", "t": 100, "trick": "bow", "outcome": "done", "ms": 1.5},
        {"k": "teach_trick", "t": 100, "trick": "bow", "outcome": "done", "ms": 600_001},
        {"k": "praise", "t": 100, "trick": "fly"},
        {"k": "scold", "t": 100, "trick": "bow", "why": "x"},
    ):
        with pytest.raises(bridge.BridgeError):
            bridge_v2.check(bad)
    with pytest.raises(bridge.BridgeError, match="unknown_kind"):
        bridge.check({"k": "praise", "t": 100})  # (bridge-v1 itself still does not know them)


# ── what Duck Core makes of each ─────────────────────────────────────────────────────────────


def test_a_trick_it_did_is_its_record_and_play_care_one_it_refused_is_no_care() -> None:
    session = make()
    before = len(session.core.log.events)
    assert give(session, "teach_trick", trick="bow", ms=4200, outcome="done")["note"] == "taught"
    new = written(session, before)
    record = next(e for e in new if e.kind == "sense.teach_trick")
    assert record.payload == {"trick": "bow", "seconds": 4, "outcome": "done"}
    care = [e for e in new if e.kind == "care.episode"]
    assert [(e.payload["type"], e.payload.get("why")) for e in care] == [("play", "teach")]
    assert care[0].parent_event_ids == (record.event_id,)

    before = len(session.core.log.events)
    assert give(session, "teach_trick", trick="flip", outcome="refused")["note"] == "refused"
    new = written(session, before)
    assert [e.kind for e in new if e.kind == "sense.teach_trick"] == ["sense.teach_trick"]
    assert not [e for e in new if e.kind == "care.episode"]


def test_praise_for_a_trick_teaches_the_family_likes_it_and_a_scold_once_a_day_the_opposite() -> (
    None
):
    session = make()
    s = session.core.s
    assert give(session, "praise", trick="spin")["note"] == "praised"
    liked = s.prefs["household.trick:spin"]
    assert liked > 5000
    assert any(
        e.kind == "learning.committed" and e.payload["parameter"] == "household.trick:spin"
        for e in session.core.log.events
    )
    assert give(session, "scold", trick="spin")["note"] == "scolded"
    assert s.prefs["household.trick:spin"] < liked
    once = s.prefs["household.trick:spin"]
    assert give(session, "scold", trick="spin")["note"] == "said today"
    assert s.prefs["household.trick:spin"] == once


def test_praise_for_nothing_in_particular_writes_and_learns_nothing_here() -> None:
    session = make()
    before = len(session.core.log.events)
    prefs = dict(session.core.s.prefs)
    assert give(session, "praise")["note"] == "no trick"
    assert give(session, "scold")["note"] == "no trick"
    kinds = {e.kind for e in written(session, before)}
    assert not kinds & {"owner.trick_good", "owner.trick_not_that", "learning.committed"}
    assert session.core.s.prefs == prefs


def test_every_trick_record_has_a_sentence_of_its_own() -> None:
    session = make()
    give(session, "teach_trick", trick="lie_down", outcome="gave_up")
    give(session, "praise", trick="shake")
    give(session, "scold", trick="bow")
    lines = {
        e.kind: narrate.describe(e)
        for e in session.core.log.events
        if e.kind in {"sense.teach_trick", "owner.trick_good", "owner.trick_not_that"}
    }
    assert lines == {
        "sense.teach_trick": "主人教我「趴下」，我試了一下，沒做完",
        "owner.trick_good": "主人說我「甩一甩」做得乖",
        "owner.trick_not_that": "主人說我的「鞠躬」不對",
    }
    learnt = [
        narrate.describe(e)
        for e in session.core.log.events
        if e.kind == "learning.committed" and "trick:" in e.payload["parameter"]
    ]
    assert learnt and all("家裡喜歡我做「" in line for line in learnt)


# ── the save ─────────────────────────────────────────────────────────────────────────────────


def test_a_duck_fed_only_v1_saves_as_v1_and_one_taught_a_trick_as_v2_and_replays_the_same() -> None:
    session = make()
    give(session, "lift")
    assert api.save_json(session)["bridge"] == {"api": "bridge-v1"}
    give(session, "teach_trick", trick="bow", outcome="done")
    give(session, "praise", trick="bow")
    saved = json.loads(json.dumps(api.save_json(session)))
    assert saved["bridge"] == {"api": "bridge-v2"}
    again = api.restore(saved)
    assert again.core.log.head == session.core.log.head
    assert again.core.s.prefs == session.core.s.prefs
    saved["bridge"] = {"api": "bridge-v3"}
    with pytest.raises(api.SaveBridgeMismatch):
        api.restore(saved)


def test_a_duck_can_be_attached_under_either_name() -> None:
    session = make(attach=False)
    assert dispatch(session, "bridge", {"api": "bridge-v2"})["api"] == "bridge-v2"
    assert give(session, "teach_trick", trick="spin", outcome="done")["taken"]
