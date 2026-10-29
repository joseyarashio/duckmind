"""Company (Duck Core v4, part D): never lonely for nothing, and its calls get answered.

All contact lowers the wish for company (never to "does not need you"); a voice to it is
someone being here; asleep it settles; it asks for attention only when someone was just
around, and any attention restarts its asking; stuck, it calls for help.
"""

from __future__ import annotations

import dataclasses

from duckmind.duckcore import content
from duckmind.duckcore.api import Config, LegacyActs, Session, dispatch, open_session
from duckmind.duckcore.engine import DuckCore, HardwareConfig
from duckmind.provenance import Event

SOCIAL = content.SOCIAL


def _kinds(core: DuckCore | LegacyActs, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


def _awake_at(seed: int, hour: int = 10, **config: object) -> LegacyActs:
    """The duck, awake at ``hour``, as the family's hands reach it (api.LegacyActs: a touch
    goes through the world's detector first; everything else straight to Duck Core)."""
    core = Session(Config(seed=seed, **config)).hands  # type: ignore[arg-type]
    core.advance(core.minutes_until(f"{hour:02d}:00"))
    if core.s.napping_until is not None:
        core.wake("owner")
    return core


def test_every_kind_of_contact_lowers_the_wish_for_company_but_never_below_the_floor():
    for action in ("touch", "voice", "name", "hello", "praise", "pickup"):
        core = _awake_at(3)
        core.s.social = 7000
        before = core.s.social
        if action == "touch":
            for _ in range(5):
                core.touch("owner", seconds=3)
        elif action == "voice":
            core.say("child", kind="voice")
        elif action == "name":
            core.call_name("child")
        elif action == "hello":
            core.owner_hello()
        elif action == "praise":
            core.pad_good("child")
        else:
            core.pickup("partner")
        assert core.s.social < before, action
    core = _awake_at(4)
    for _ in range(30):
        core.say("child", kind="voice")
        core.touch("child", seconds=3)
    assert core.s.social >= SOCIAL["floor"]


def test_asleep_it_does_not_grow_lonelier():
    core = _awake_at(5)
    core.s.social = 7000
    core.nap_now("owner")
    minutes = core.nap_left_min()
    core.advance(max(1, minutes - 1))
    assert core.s.napping_until is not None
    assert core.s.social < 7000  # settling toward calm while it sleeps


def test_a_voice_to_it_is_someone_here_and_answers_its_call():
    core = _awake_at(6)
    core.s.bid_pending_since = core.s.now
    core.s.bids_stopped = True
    core.say("owner", kind="voice")
    assert core.s.bid_pending_since is None and not core.s.bids_stopped
    assert _kinds(core, "duck.bid_answered")
    assert core.s.presence_at == core.s.now - __import__("datetime").timedelta(seconds=3)


def test_without_ears_a_voice_is_not_heard():
    session = Session(Config(seed=7, ear=False))
    core = session.core
    before = core.s.day_voice
    result = core.say("owner", kind="voice")
    assert result["perceived"] is False and core.s.day_voice == before


def test_a_call_waits_ninety_seconds_for_an_answer():
    core = _awake_at(8)
    core.s.bid_pending_since = core.s.now
    before = core.s.unanswered
    core.advance_seconds(SOCIAL["bid_answer_s"] - 5)
    core._check_bid_answer()
    assert core.s.bid_pending_since is not None
    core.advance_seconds(10)
    core._check_bid_answer()
    assert core.s.bid_pending_since is None and core.s.unanswered == before + 1


def test_any_attention_restarts_its_asking():
    core = _awake_at(9)
    core.s.bids_stopped, core.s.unanswered, core.s.bid_backoff = True, 2, 2500
    if not core.touch("child", seconds=3)["perceived"]:
        core.say("child", kind="voice")  # the detector may miss a touch; a voice never is
    assert not core.s.bids_stopped and core.s.unanswered == 0 and core.s.bid_backoff == 10000


def test_it_asks_for_attention_only_when_someone_was_just_around():
    session = Session(Config(seed=10, nest=True, conservative=False, household="school_family"))
    dispatch(session, "live", {"family": "school_family", "days": 21})
    core = session.core
    bids = [e for e in _kinds(core, "duck.bid") if not e.payload.get("need")]
    assert bids
    phases = {}
    for event in core.log.events:
        if event.kind == "duck.phase":
            phases["now"] = event.payload["to"]
        if event.kind == "duck.bid" and not event.payload.get("need"):
            assert phases.get("now", "family") == "family", event.payload
            assert event.payload["room_min"] is not None
            assert event.payload["room_min"] <= SOCIAL["bid_room_min"]
            assert 7 <= event.occurred_at.hour < 22
    answered = len(_kinds(core, "duck.bid_answered"))
    unanswered = len(_kinds(core, "duck.bid_unanswered"))
    assert answered >= unanswered  # most calls are heard, because someone is there


def test_stuck_after_a_fall_it_calls_for_help():
    # (the outcome rates are the world's: a v4 config's rate fields become its LegacyWorld
    # header when the duck is opened)
    session = open_session(
        {"engine_rev": "v4_1", "seed": 11, "fall_ppm_per_s": 1_000_000, "righting_ppm": 0}
    )
    assert isinstance(session, Session)
    dispatch(session, "live", {"days": 8, "routine": "floor"})
    core = session.core
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "act", {"action": "power_on"})
    dispatch(session, "set_setting", {"key": "floor", "value": True})
    for _ in range(240):
        if core.s.posture == "fallen" and core.s.righting is None:
            break
        core.advance(1)
    help_calls = [e for e in _kinds(core, "duck.bid") if e.payload["kind"] == "help"]
    assert help_calls and help_calls[-1].payload["need"]
    session.hands.set_upright("owner")
    assert core.s.need is None
    nest_duck = open_session({"engine_rev": "v4_1", "seed": 11, "nest": True, "righting_ppm": 0})
    assert isinstance(nest_duck, Session) and nest_duck.world is not None
    assert nest_duck.world["rates"]["righting_ppm"] == 0  # the world's, not Duck Core's
    assert "righting_ppm" not in {field.name for field in dataclasses.fields(HardwareConfig)}


def test_the_lonely_share_stays_low_when_the_family_is_out_all_day():
    """Nobody home is not lonely: it rests in its nest (or on its cable) instead."""
    for kind, nest in (("single_long", True), ("cable_only", False)):
        session = Session(Config(seed=12, nest=nest, household=kind))
        dispatch(session, "live", {"family": kind, "days": 14})
        core = session.core
        decisions = [e for e in _kinds(core, "duck.decision") if "inputs" in e.payload]
        lonely = sum(1 for e in decisions if e.payload["inputs"].get("social", 0) >= 6000)
        assert lonely <= len(decisions) * 15 // 100, (kind, lonely, len(decisions))
