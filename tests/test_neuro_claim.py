"""A person's answer claims the body (live voice, interplay fix 2026-10-04): the cloud's reply to a
person the duck sees but has not been told the name of stands at 0.2–1.0 against a walk that holds
the body at 2.0–2.6 (the bridge's urge to answer is kept from such a person: LV4's reserve), so the
plain competition never gave it the body — 2 of 11 voiced for the explorer, 1 of 12 for the snuggler,
0 of 12 for the dreamer, live. What is said to the duck is an event, not a drift: a live reply whose
input is at least CLAIM_MIN takes the body from a bout in hand, and keeps it until the answer is given;
it never takes it from urgent business, sleep, a duck that is afraid, going home, a skill in the middle
of something, and a reflex never waits for it."""

from __future__ import annotations

import pytest
from test_neuro_brain import _genome

from duckmind.neuro import select as S

STANDING = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)
DT = 0.05


def _explorer(restless: float = 0.7) -> tuple[S.Selector, list[float]]:
    sel = S.Selector(_genome("bold_explorer"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = restless
    feat[S.F["curiosity"]] = 0.7
    feat[S.F["energy"]] = 0.9
    return sel, feat


def _settle(sel: S.Selector, feat: list[float], seconds: float = 8.0) -> float:
    t = 0.0
    while t < seconds:
        sel.step(DT, feat, STANDING, False, 0.0)
        t += DT
    return t


def _answer_after(
    sel: S.Selector,
    feat: list[float],
    t: float,
    value: float,
    *,
    claim: bool = True,
    seconds: float = 4.0,
    ctx: S.Context = STANDING,
) -> tuple[float | None, str]:
    """The second (after ``t``) the greeting took the body and the act at the end."""
    sel.set_extra({"greet": value}, claims={"greet": value} if claim else None)
    start = t
    first: float | None = None
    name = sel.names[sel.current]
    while t < start + seconds:
        name, new = sel.step(DT, feat, ctx, False, 0.0)
        if new and name == "greet" and first is None:
            first = round(t - start, 2)
        t += DT
    return first, name


def test_the_explorer_is_busy_walking_before_anyone_speaks() -> None:
    sel, feat = _explorer()
    _settle(sel, feat)
    assert sel.names[sel.current] in S.LOCOMOTES, sel.names[sel.current]
    assert sel.sat[sel.current] < S.FINISH_SAT  # …and the bout is not done


@pytest.mark.parametrize("value", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_a_live_answer_takes_the_body_from_a_bout_that_is_not_done(value: float) -> None:
    sel, feat = _explorer()
    t = _settle(sel, feat)
    first, name = _answer_after(sel, feat, t, value)
    assert first is not None and first <= 0.5, (first, name)
    assert name == "greet"  # …and keeps it while the answer is waited for


def test_the_same_input_that_is_not_a_claim_loses_to_the_walk() -> None:
    """The plain competition (what the bridge did before): 0.5 against a walk at ~2 never wins."""
    sel, feat = _explorer()
    t = _settle(sel, feat)
    first, name = _answer_after(sel, feat, t, 0.5, claim=False, seconds=6.0)
    assert first is None and name != "greet"


def test_an_input_under_the_claim_minimum_competes_like_any_other_reason() -> None:
    sel, feat = _explorer()
    t = _settle(sel, feat)
    first, _name = _answer_after(sel, feat, t, S.CLAIM_MIN * 0.9, seconds=6.0)
    assert first is None


def test_a_rival_cannot_take_the_answer_back_while_it_is_waited_for() -> None:
    sel, feat = _explorer()
    t = _settle(sel, feat)
    sel.set_extra({"greet": 0.4}, claims={"greet": 0.4})
    taken = []
    for _ in range(int(6.0 / DT)):
        feat[S.F["restless"]] = 0.95  # a strong rival want all the time
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        if new:
            taken.append(name)
        t += DT
    assert taken == ["greet"], taken
    # the answer is given (the bridge ends it): the duck goes back to what it wants
    sel.set_extra({}, claims={})
    later = []
    for _ in range(int(20.0 / DT)):
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        if new:
            later.append(name)
    assert later and later[0] != "greet"


def test_a_frightened_duck_does_not_give_up_the_body_to_an_answer() -> None:
    sel, feat = _explorer()
    t = _settle(sel, feat)
    feat[S.F["fear"]] = S.CLAIM_FEAR_MAX + 0.05
    first, _name = _answer_after(sel, feat, t, 1.0, seconds=6.0)
    assert first is None


def test_a_reflex_tick_is_not_given_to_an_answer() -> None:
    sel, feat = _explorer()
    _settle(sel, feat)
    sel.set_extra({"greet": 1.0}, claims={"greet": 1.0})
    name, new = sel.step(DT, feat, STANDING, True, 0.0)  # a reflex halts the body this tick
    assert not new and name != "greet"


def test_sleep_is_not_taken_by_an_answer() -> None:
    ctx = S.Context(sitting=True, standing=False, asleep=True, can_walk=True, wakefulness=0.2)
    sel = S.Selector(_genome("lazy_dreamer"), seed=1)
    feat = S.empty_features()
    feat[S.F["sleepy"]] = 0.9
    t = 0.0
    while t < 30.0:
        sel.step(DT, feat, ctx, False, 0.0)
        t += DT
    assert S.BEHAVIOURS[sel.names[sel.current]].sleep, sel.names[sel.current]
    sel.set_extra({"greet": 1.0}, claims={"greet": 1.0})
    assert sel._claimant(sel.current, feat, ctx, False) is None
    first, _name = _answer_after(sel, feat, t, 1.0, seconds=5.0, ctx=ctx)
    assert first is None  # (the plain competition decides: a sleeper's wakefulness)


def test_urgent_business_is_not_taken_by_an_answer() -> None:
    sel, feat = _explorer()
    _settle(sel, feat)
    feat[S.F["startled"]] = 0.9
    feat[S.F["fear"]] = 0.3  # (under the claim's own fear limit: it is the act that is urgent)
    sel.current = sel.index("flee")
    assert sel.urgent_now(feat)
    sel.set_extra({"greet": 1.0}, claims={"greet": 1.0})
    assert sel._claimant(sel.current, feat, STANDING, False) is None
    # …and the same duck in an ordinary bout gives the body up to it
    sel.current = sel.index("wander")
    assert (
        not sel.urgent_now(feat) and sel._claimant(sel.current, feat, STANDING, False) is not None
    )


def test_going_home_and_a_docking_are_not_taken_by_an_answer() -> None:
    sel, feat = _explorer()
    _settle(sel, feat)
    sel.set_extra({"greet": 1.0}, claims={"greet": 1.0})
    assert sel._claimant(sel.current, feat, STANDING, False) is not None
    feat[S.F["docking"]] = 1.0
    assert sel._claimant(sel.current, feat, STANDING, False) is None
    feat[S.F["docking"]] = 0.0
    feat[S.F["homing"]] = 1.0
    assert sel._claimant(sel.current, feat, STANDING, False) is None


def test_a_told_command_takes_the_body_from_an_unhurried_walk_home() -> None:
    """A person's command ("sit", "stop") is heard while the duck walks home with a full pack (live
    :8796, 2026-10-23: 'gave_up busy' after 5 s of ``return_dock``): it takes the body; a live answer
    still does not, and a pack running low (``homing``) or a docking still keeps it."""
    sel, feat = _explorer()
    _settle(sel, feat)
    sel.current = sel.index("return_dock")
    sel.set_command({"sit_down": 1.2}, claims={"sit_down": 1.2})
    assert sel.names[sel._claimant(sel.current, feat, STANDING, False)] == "sit_down"
    feat[S.F["homing"]] = S.HOMING_RELEASE
    assert sel._claimant(sel.current, feat, STANDING, False) is None
    feat[S.F["homing"]] = 0.0
    feat[S.F["docking"]] = 1.0
    assert sel._claimant(sel.current, feat, STANDING, False) is None
    feat[S.F["docking"]] = 0.0
    sel.set_command({}, claims={})
    sel.set_extra({"greet": 1.0}, claims={"greet": 1.0})
    assert sel._claimant(sel.current, feat, STANDING, False) is None


def test_the_duck_s_own_wish_is_not_a_claim() -> None:
    """``set_extra`` without ``claims`` (the bridge's own wishes: call_out, follow) never claims."""
    sel, feat = _explorer()
    _settle(sel, feat)
    sel.set_extra({"call_out": 1.0})
    assert sel._claimant(sel.current, feat, STANDING, False) is None


def test_the_answering_acts_do_not_trade_the_body_among_themselves() -> None:
    """The bridge puts one answer on up to three channels (greet, call_response, call_out): once one of
    them has the body the others do not take it, however their input moves."""
    sel, feat = _explorer()
    t = _settle(sel, feat)
    claims = {"greet": 0.4, "call_response": 0.4, "call_out": 0.4}
    sel.set_extra(claims, claims=claims)
    started = []
    for k in range(int(5.0 / DT)):
        # (the inputs of the channels that are not the act's swing up and down)
        swing = 0.4 + 0.3 * (1.0 if (k // 10) % 2 else -1.0)
        c = {"greet": 0.4, "call_response": swing, "call_out": 0.8 - swing}
        sel.set_extra(c, claims=c)
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        if new:
            started.append(name)
        t += DT
    assert len(started) == 1 and started[0] in claims, started


def test_a_one_shot_claim_takes_the_body_for_one_tick_and_the_bout_it_starts_is_held() -> None:
    """``claim_next``: what a thing that has just come into view gets for the curiosity act. It is the
    next tick's only; the bout it starts is then a bout like any other (the finishing rule holds it
    against a rival that only crept up)."""
    sel, feat = _explorer(restless=0.9)
    feat[S.F["thing"]] = 1.0  # (a new thing in view: what makes the curiosity act want the body)
    feat[S.F["novel"]] = 0.8
    feat[S.F["thing_near"]] = 0.3
    _settle(sel, feat)
    assert sel.names[sel.current] != "investigate"
    sel.claim_next("investigate")
    name, new = sel.step(DT, feat, STANDING, False, 0.0)
    assert new and name == "investigate"
    # held: with the wander want as strong as it was, the next seconds are investigate's own
    later = []
    for _ in range(int(2.0 / DT)):
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        later.append((name, new))
    assert [n for n, _ in later] == ["investigate"] * len(later)
    # and it was one tick's: the claim is not still on
    sel2, feat2 = _explorer(restless=0.9)
    _settle(sel2, feat2)
    sel2.claim_next("investigate")
    sel2.step(DT, feat2, STANDING, False, 0.0)
    assert sel2._once is None


def test_a_one_shot_claim_is_refused_where_a_persons_answer_is() -> None:
    sel, feat = _explorer()
    _settle(sel, feat)
    feat[S.F["fear"]] = S.CLAIM_FEAR_MAX + 0.05
    sel.claim_next("investigate")
    name, new = sel.step(DT, feat, STANDING, False, 0.0)
    assert not (new and name == "investigate")
    sel, feat = _explorer()
    _settle(sel, feat)
    sel.claim_next("investigate")
    name, new = sel.step(DT, feat, STANDING, True, 0.0)  # a reflex tick
    assert not (new and name == "investigate")


def test_a_skills_busy_phase_is_waited_for_a_moment_and_then_the_answer_takes_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kick or a grab (``ball_busy``, ``toy_busy``) is waited for as long as such a phase lasts
    (``BUSY_WAIT_S``): live, a line-up on the ball held ``ball_busy`` for some 15 s, longer than a reply
    lives, and two of 49 replies were lost to it. (The feature is stood in for by ``bias``, which is 1.)"""
    monkeypatch.setattr(S, "BUSY_FEATURES", ("docking", "bias"))
    sel, feat = _explorer()
    _settle(sel, feat)
    claims = {"greet": 0.4}
    sel.set_extra(claims, claims=claims)
    taken = None
    t = 0.0
    while t < 4.0:
        sel.set_extra(claims, claims=claims)
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        if new and name == "greet" and taken is None:
            taken = t
        t += DT
    assert taken is not None and S.BUSY_WAIT_S - 0.1 <= taken <= S.BUSY_WAIT_S + 0.5, taken
    # a docking is never broken off, however long the answer waits
    sel, feat = _explorer()
    _settle(sel, feat)
    feat[S.F["docking"]] = 1.0
    t = 0.0
    while t < 6.0:
        sel.set_extra(claims, claims=claims)
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        assert not (new and name == "greet")
        t += DT


def test_a_one_shot_claim_never_breaks_a_busy_phase_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "BUSY_FEATURES", ("docking", "bias"))
    sel, feat = _explorer(restless=0.9)
    feat[S.F["thing"]] = 1.0
    feat[S.F["novel"]] = 0.8
    _settle(sel, feat)
    for _ in range(int(5.0 / DT)):
        sel.claim_next("investigate")
        name, new = sel.step(DT, feat, STANDING, False, 0.0)
        assert not (new and name == "investigate")
