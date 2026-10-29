import sys
from datetime import timedelta

import pytest

from duckmind.duckcore import engine
from duckmind.duckcore.api import Config, LegacyActs, Session, dispatch, legacy_world
from duckmind.duckcore.engine import PROMOTION_EVERY_GD, DuckCore
from duckmind.worldport import make_world


def _awake(seed: int = 5, persona: str = "owner") -> LegacyActs:
    """A duck built directly, living in v4's world (LegacyWorld). The family's hands reach it
    through the world, as on the page (api.LegacyActs: a touch is felt only when the world's
    detector senses it); everything else goes straight to Duck Core."""
    port = make_world(legacy_world(Config(seed=seed)))
    core = LegacyActs(DuckCore(seed=seed, port=port), port)
    core.power_on(persona)
    core.advance(5)
    return core


def _learned(core: DuckCore | LegacyActs, start: int, parameter: str) -> list[tuple[float, float]]:
    return [
        (event.payload["before"], event.payload["after"])
        for event in core.log.events[start:]
        if event.kind == "learning.committed" and event.payload["parameter"] == parameter
    ]


def _flag_verb(core: DuckCore | LegacyActs, verb: str = "look_around") -> None:
    """Pretend the duck just tried this behaviour, so praise has something to attach to."""
    core.s.last_flag = {"kind": "verb", "id": verb, "at": core.s.now}


def test_praise_teaches_with_diminishing_steps_per_growth_day():
    core = _awake()
    start = len(core.log.events)
    for _ in range(5):
        _flag_verb(core)
        core.pad_good("child")
    assert _learned(core, start, "household.verb:look_around") == [
        (0.5, 0.506),
        (0.506, 0.509),
        (0.509, 0.5105),
    ]


def test_not_that_lowers_a_preference_once_per_growth_day():
    core = _awake()
    start = len(core.log.events)
    for _ in range(3):
        _flag_verb(core)
        core.owner_not_that()
    assert _learned(core, start, "household.verb:look_around") == [(0.5, 0.494)]


def test_enough_quiets_the_duck_and_teaches_nothing():
    core = _awake()
    start = len(core.log.events)
    _flag_verb(core)
    core.owner_enough()
    kinds = [event.kind for event in core.log.events[start:]]
    assert "duck.quiet" in kinds
    assert "learning.committed" not in kinds
    assert timedelta(minutes=14) < core.s.quiet_until - core.s.now <= timedelta(minutes=15)


def test_owner_bond_moves_at_most_three_times_per_growth_day():
    core = _awake()
    start = len(core.log.events)
    for _ in range(4):
        core.owner_hello()
        core.owner_good()
    assert len(_learned(core, start, "owner.bond")) == 3


def test_only_the_owner_phone_can_change_the_owner_bond():
    core = _awake(persona="child")
    start = len(core.log.events)
    for persona in ("owner", "partner", "child", "guest"):
        _flag_verb(core)
        core.touch(persona, seconds=10)
        core.pad_good(persona)
        core.say(persona)
        core.pickup(persona)
        core.hand_near(persona)
    assert _learned(core, start, "owner.bond") == []
    assert core.s.bond_owner == 2000


def test_guest_mode_pauses_learning_and_care_counting():
    core = _awake()
    core.pad_guest("owner")
    start = len(core.log.events)
    _flag_verb(core)
    core.pad_good("guest")
    core.touch("guest", seconds=10)
    new = core.log.events[start:]
    assert not [event for event in new if event.kind == "learning.committed"]
    assert all(not event.payload["counted"] for event in new if event.kind == "care.episode")


def test_attention_bids_do_not_grow_with_absence(monkeypatch):
    """The chance to ask for attention is constant; being away never raises it."""
    chances: dict[str, list[int]] = {"before": [], "after": []}
    phase = "before"
    below = engine.rng.below

    def recording(raw: int, probability: int) -> bool:
        if sys._getframe(1).f_code.co_name == "_maybe_bid":
            chances[phase].append(probability)
        return below(raw, probability)

    monkeypatch.setattr(engine.rng, "below", recording)
    session = Session(Config(seed=9))
    dispatch(session, "live", {"days": 14, "routine": "daily"})
    core = session.core
    core.power_off("owner")
    core.advance(30 * 24 * 60)
    phase = "after"
    core.power_on("owner")
    core.advance(120)
    assert chances["before"] and chances["after"], "the duck never considered asking for attention"
    assert max(chances["after"]) <= max(chances["before"]) <= 800


@pytest.fixture(scope="module")
def long_life() -> Session:
    session = Session(Config(seed=7))
    dispatch(session, "live", {"days": 120, "routine": "daily"})
    return session


def test_family_phrases_are_promoted_at_most_once_a_week(long_life):
    promoted = [
        event.payload["gd"] for event in long_life.core.log.events if event.kind == "voice.promoted"
    ]
    assert promoted, "four months of daily care should keep at least one phrase"
    assert all(
        later - earlier >= PROMOTION_EVERY_GD
        for earlier, later in zip(promoted, promoted[1:], strict=False)
    )


def test_household_preferences_stay_away_from_the_rails(long_life):
    assert long_life.core.s.prefs
    assert max(long_life.core.s.prefs.values()) < 9000
    assert 2000 < long_life.core.s.bond_owner < 9000
