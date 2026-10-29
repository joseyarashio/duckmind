"""The 「成長」 view's trick book (phase 2 「學會聽話」): ``learned.tricks`` from the neural duck's own
learner, the trick lessons in what it lived, and a trick the family likes as a learnt preference."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import bridge
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.life import growth_view as gv
from duckmind.neuro.command_words import TRICKS
from duckmind.neuro.tricks import TrickBook

TPE = timezone(timedelta(hours=8))
BIRTH = datetime(2026, 10, 1, 8, 0, tzinfo=TPE)


def _session(days: int = 3) -> Session:
    session = Session(Config(seed=4107, name="蛋黃", tof=True, nest=False, start=BIRTH.isoformat()))
    dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    dispatch(session, "advance", {"seconds": days * 86400})
    return session


def _view(session: Session, learner: Any = None) -> dict[str, Any]:
    tally = gv.Tally()
    tally.update(session.inputs)
    return gv.build(
        session.core,
        tally=tally,
        name="蛋黃",
        live=True,
        branch_id="explorer",
        pinned=True,
        nature_branch="explorer",
        learner=learner,
    )


def _learner(book: TrickBook) -> SimpleNamespace:
    return SimpleNamespace(people={}, places={}, things={}, tricks=book)


def _give(session: Session, kind: str, **fields: Any) -> None:
    dispatch(session, "advance", {"seconds": 5})
    t = int(session.core.s.now.timestamp())
    dispatch(session, "experience", {"events": [{"k": kind, "t": t, **fields}]})


def test_the_trick_book_is_one_row_per_trick_from_the_learner() -> None:
    session = _session(3)
    gd = int(session.core.s.gd)
    book = TrickBook()
    for _ in range(3):
        book.praise("bow", gd)
    book.praise("bow", gd, shaped=True)
    tricks = _view(session, _learner(book))["learned"]["tricks"]
    assert [row["id"] for row in tricks] == list(TRICKS)
    bow = next(row for row in tricks if row["id"] == "bow")
    assert bow["proficiency_pct"] == round(100 * book.proficiency("bow"))
    assert bow["proficiency_pct"] > 0
    assert (bow["practised_today"], bow["total"], bow["shaped"]) == (4, 4, 1)
    assert bow["unlocked"] and bow["label"] == "鞠躬"
    untaught = next(row for row in tricks if row["id"] == "spin")
    assert (untaught["proficiency_pct"], untaught["total"], untaught["state"]) == (0, 0, "none")


def test_flip_opens_at_the_juvenile_stage_the_others_from_the_start() -> None:
    session = _session(3)
    young = {
        row["id"]: row["unlocked"]
        for row in _view(session, _learner(TrickBook()))["learned"]["tricks"]
    }
    assert young == {"lie_down": True, "spin": True, "bow": True, "shake": True, "flip": False}
    s = session.core.s
    s.stage_days = {**s.stage_days, "juvenile": int(s.gd)}  # (a duck whose own pace got it there)
    older = _view(session, _learner(TrickBook()))["learned"]["tricks"]
    assert all(row["unlocked"] for row in older)


def test_the_view_only_looks_at_the_book() -> None:
    session = _session(3)
    book = TrickBook()
    book.praise("shake", int(session.core.s.gd) - 2)
    state = book.to_state()
    _view(session, _learner(book))
    assert book.to_state() == state  # (no decay, no new day: the brain's thread keeps the book)


def test_no_learner_no_trick_book() -> None:
    assert _view(_session(1))["learned"]["tricks"] is None


def test_the_lessons_are_counted_with_what_it_lived_and_a_praised_trick_is_a_liking() -> None:
    session = _session(2)
    _give(session, "teach_trick", trick="bow", ms=3000, outcome="done")
    _give(session, "praise", trick="bow")
    _give(session, "scold", trick="spin")
    view = _view(session)
    lived = view["experiences"]
    assert {k: lived["by_kind"].get(k) for k in ("teach_trick", "praise", "scold")} == {
        "teach_trick": 1,
        "praise": 1,
        "scold": 1,
    }
    recent = lived["recent"]
    lesson = next(item for item in recent if item["kind"] == "teach_trick")
    assert (lesson["trick"], lesson["outcome"]) == ("bow", "done")
    prefs = view["learned"]["prefs"]
    liked = next(p for p in prefs if p["kind"] == "trick")
    assert liked["label"] == "鞠躬" and liked["liked"]
