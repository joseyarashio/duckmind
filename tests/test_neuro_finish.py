"""Finishing what it started (M5, owner 2026-10-04: 「行為常常還沒做完就換成別的」): a bout that is not done
is not taken over by a rival that only crept up; a reflex, a released commitment, the pack running low and
a rival that *surged* (something happened) still take it at once; the bouts of a whole life end satisfied."""

from __future__ import annotations

import copy
from collections import defaultdict

import pytest
from test_neuro_brain import TEMPERAMENTS, _genome

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.synth import SynthWorld, iterate

CTX = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)
DT = 0.05


def _walk(sel: S.Selector, feat: list[float], seconds: float, start: float = 0.0) -> float:
    t = start
    while t < start + seconds:
        sel.step(DT, feat, CTX, False, 0.0)
        t += DT
    return t


def _rival(x: float, feat: list[float]) -> None:
    """A thing coming near and new: how much of it is there (0..1) — ``thing_near`` and ``novel``."""
    feat[S.F["thing_near"]] = x
    feat[S.F["novel"]] = 0.8 * x


def _rival_after(
    old_logic: bool, monkeypatch: pytest.MonkeyPatch, ramp_s: float
) -> list[tuple[float, str, float]]:
    """A wandering duck (restless 0.55) in an empty room; at 5 s a thing comes near — all at once
    (``ramp_s`` 0) or over ``ramp_s`` seconds. Returns the bout changes as (second, behaviour taken,
    satiation of the bout that was left). ``old_logic``: the selector as it was before the finishing
    rule (every behaviour exempt from it)."""
    if old_logic:
        monkeypatch.setattr(S, "UNFINISHED_OK", frozenset(C.BEHAVIOURS))
    sel = S.Selector(_genome("social_songbird"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.55
    out: list[tuple[float, str, float]] = []
    t = 0.0
    while t < 100.0:
        _rival(0.0 if t < 5.0 else (1.0 if ramp_s == 0 else min(1.0, (t - 5.0) / ramp_s)), feat)
        left = sel.sat[sel.current]
        name, new = sel.step(DT, feat, CTX, False, 0.0)
        if new:
            out.append((round(t, 2), name, left))
        t += DT
    return out


def test_a_rival_that_only_crept_up_does_not_take_a_bout_that_is_not_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _rival_after(True, monkeypatch, 20.0)
    # (the first change is the start: rest_sit → wander) the old logic hands the walk to the creeping
    # rival while the walk has used up well under half of itself …
    assert old[1][1] == "nibble" and old[1][2] < S.FINISH_SAT - 0.1, old
    monkeypatch.undo()
    new = _rival_after(False, monkeypatch, 20.0)
    # … the new one lets the walk go on until its own satiation has used up half of it
    assert new[1][2] >= S.FINISH_SAT, new
    assert new[1][0] > old[1][0] + 20.0, (old, new)


def test_a_rival_that_surged_takes_it_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    new = _rival_after(False, monkeypatch, 0.0)  # the same thing, all at once
    assert new[1][1] == "nibble" and new[1][0] < 5.0 + 3.0, new
    assert new[1][2] < 0.2  # … the walk had hardly begun


def test_a_reflex_a_released_commitment_and_a_low_pack_take_a_bout_that_is_not_done() -> None:
    sel = S.Selector(_genome("social_songbird"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.55
    t = _walk(sel, feat, 5.0)
    for k in range(int(35.0 / DT)):  # a thing creeping near and new, 20 s, then there
        _rival(min(1.0, k * DT / 20.0), feat)
        sel.step(DT, feat, CTX, False, 0.0)
    assert t > 0 and sel.names[sel.current] == "wander" and sel.sat[sel.current] < S.FINISH_SAT
    rival_i = sel.index("nibble")
    assert sel.act[rival_i] > sel.act[sel.current] * 1.2, "the rival must beat the plain hold"
    # held: the rival crept up and the walk is not done
    assert copy.deepcopy(sel).step(DT, feat, CTX, False, 0.0) == ("wander", False)
    # a reflex that halts the body: commitment is not asked for (the selector's own interruption)
    reflexed = copy.deepcopy(sel)
    assert reflexed.step(DT, feat, CTX, True, 0.0)[0] == "nibble"
    # a new thing to look at releases it (brain.attention_interrupt)
    released = copy.deepcopy(sel)
    assert released.step(DT, feat, CTX, False, 0.0, release_commitment=True)[0] == "nibble"
    # the pack running low: going home is not made to wait for the walk to finish
    homing = copy.deepcopy(sel)
    feat_low = list(feat)
    feat_low[S.F["homing"]] = 0.3
    assert homing.step(DT, feat_low, CTX, False, 0.0)[0] != "wander"


def _threat_after(
    old_logic: bool, monkeypatch: pytest.MonkeyPatch, feature: str = "fear", ramp_s: float = 20.0
) -> list[tuple[float, str, float]]:
    """The wandering duck of :func:`_rival_after`, and at 5 s a threat that builds up slowly over
    ``ramp_s`` (``fear`` rising): the bout changes as (second, behaviour taken, satiation of the bout
    that was left)."""
    if old_logic:
        monkeypatch.setattr(S, "UNFINISHED_OK", frozenset(C.BEHAVIOURS))
    sel = S.Selector(_genome("social_songbird"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.55
    out: list[tuple[float, str, float]] = []
    t = 0.0
    while t < 60.0:
        feat[S.F[feature]] = 0.0 if t < 5.0 else min(1.0, (t - 5.0) / ramp_s)
        left = sel.sat[sel.current]
        name, new = sel.step(DT, feat, CTX, False, 0.0)
        if new:
            out.append((round(t, 2), name, left))
        t += DT
    return out


def test_a_threat_that_crept_up_still_takes_a_bout_that_is_not_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _threat_after(True, monkeypatch)
    monkeypatch.undo()
    new = _threat_after(False, monkeypatch)
    # fear is urgent business: the walk is given up for fleeing when it was, creeping or not …
    assert new[1][1] == old[1][1] == "flee", (old, new)
    assert new[1][0] == pytest.approx(old[1][0], abs=0.5), (old, new)
    assert new[1][2] < S.FINISH_SAT, new  # … long before it was done


def test_the_fear_acts_do_not_take_each_other_over_half_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flee, retreat, freeze, tuck away in turn — the shy duck's reopening: once afraid, each act is
    carried to where its own satiation ends it (the old logic left one of them at 0.46)."""
    old = _threat_after(True, monkeypatch)
    monkeypatch.undo()
    new = _threat_after(False, monkeypatch)
    assert any(left < S.FINISH_SAT for _, _, left in old[2:5]), old
    assert all(left >= S.FINISH_SAT for _, _, left in new[2:5]), new
    assert len(new) >= 4, new


def _proposal_after(
    old_logic: bool, monkeypatch: pytest.MonkeyPatch, ramp_s: float
) -> list[tuple[float, str, float]]:
    """The wandering duck again; at 5 s the live voice's proposal for a greeting (``set_extra``) builds
    up over ``ramp_s`` seconds."""
    if old_logic:
        monkeypatch.setattr(S, "UNFINISHED_OK", frozenset(C.BEHAVIOURS))
    sel = S.Selector(_genome("social_songbird"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.55
    out: list[tuple[float, str, float]] = []
    t = 0.0
    while t < 40.0:
        sel.set_extra({"greet": 0.0 if t < 5.0 else 2.0 * min(1.0, (t - 5.0) / ramp_s)})
        left = sel.sat[sel.current]
        name, new = sel.step(DT, feat, CTX, False, 0.0)
        if new:
            out.append((round(t, 2), name, left))
        t += DT
    return out


def test_what_a_person_proposes_takes_a_bout_that_is_not_done_however_slowly_it_builds_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What is said to the duck is an event, not a drift: the proposal wins when it beats the plain hold,
    as it did before the finishing rule (live voice LV2: the social ducks answer)."""
    old = _proposal_after(True, monkeypatch, 20.0)
    monkeypatch.undo()
    new = _proposal_after(False, monkeypatch, 20.0)
    assert old[1][1] == new[1][1] == "greet" and new[1][2] < S.FINISH_SAT, (old, new)
    assert new[1][0] == pytest.approx(old[1][0], abs=0.2), (old, new)


def test_going_home_and_answering_a_call_are_not_held_to_the_half_satiation_bar() -> None:
    assert {"return_dock", "call_response"} <= S.UNFINISHED_OK


def _sleep_after(
    old_logic: bool, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[float, str, float]]:
    """A sleepy duck (``sleepy`` 0.4) napping on the floor; at 20 s a thing comes near and new over
    20 s: the bout changes as (second, behaviour taken, satiation of the bout that was left)."""
    if old_logic:
        monkeypatch.setattr(S, "UNFINISHED_OK", frozenset(C.BEHAVIOURS))
    ctx = S.Context(sitting=True, standing=False, asleep=False, can_walk=True, wakefulness=1.0)
    sel = S.Selector(_genome("lazy_dreamer"), seed=1)
    feat = S.empty_features()
    feat[S.F["sleepy"]] = 0.4
    out: list[tuple[float, str, float]] = []
    t = 0.0
    while t < 60.0:
        _rival(0.0 if t < 20.0 else min(1.0, (t - 20.0) / 20.0), feat)
        left = sel.sat[sel.current]
        name, new = sel.step(DT, feat, ctx, False, 0.0)
        if new:
            out.append((round(t, 2), name, left))
        t += DT
    return out


def test_sleep_is_not_held_to_the_half_satiation_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nap takes 900 s to use up half of itself: held to that, a nap would last for ten minutes
    against whatever creeps up (the longest nap of two hours went from 88 s to 762 s) and a duck
    that sleeps on through the day walks too little. The acts of sleep keep the old logic."""
    assert all(C.BEHAVIOURS[b].sleep for b in ("nap", "doze", "sleep_talk"))
    old = _sleep_after(True, monkeypatch)
    monkeypatch.undo()
    new = _sleep_after(False, monkeypatch)
    assert old[0][1] == new[0][1] == "nap", (old, new)
    assert old[1][1] == "nibble" and old[1][2] < S.FINISH_SAT, old
    assert new[1] == old[1], (old, new)  # the creeping rival takes the nap exactly as it did


def _bouts(brain: NeuroBrain, world: SynthWorld, seconds: float) -> list[tuple[str, float, float]]:
    """Every bout of a run but the last, as (behaviour, seconds, satiation at its last tick)."""
    out: list[tuple[str, float, float]] = []
    last: tuple[int, str, float, float] | None = None
    for cmd, tel in iterate(brain, world, seconds, hz=10.0):
        if last is not None and cmd.bout != last[0]:
            out.append((last[1], last[3], last[2]))
        if last is None or cmd.bout != last[0]:
            last = (cmd.bout, cmd.behaviour, tel.satiation, 0.0)
        last = (last[0], last[1], tel.satiation, last[3] + 0.1)
    return out


def test_the_bouts_of_a_life_mostly_end_satisfied_and_none_flicker() -> None:
    """The M5 check on a whole run (the L1 world: nobody comes): most bouts end with their satiation at
    half or more (before the rule 7–36 % did, by temperament), and no behaviour is begun three times
    in ten seconds."""
    done = total = 0
    flicker: list[str] = []
    for name in TEMPERAMENTS:
        for seed in (1, 2):
            starts: dict[str, list[float]] = defaultdict(list)
            t = 0.0
            for behaviour, seconds, satiation in _bouts(
                NeuroBrain(_genome(name), seed=seed), SynthWorld(seed=seed, family=False), 240.0
            ):
                starts[behaviour].append(t)
                t += seconds
                if seconds > 0.3:  # (a reflex's label is not a bout of the selector's own)
                    total += 1
                    done += satiation >= S.FINISH_SAT
            for behaviour, times in starts.items():
                for a in range(len(times) - 2):
                    if times[a + 2] - times[a] <= 10.0:
                        flicker.append(f"{name}/{seed}/{behaviour}@{times[a]:.0f}")
    assert total > 100
    assert done / total >= 0.6, (done, total)
    assert len(flicker) <= 2, flicker


def test_a_walk_ends_when_the_urge_to_walk_is_used_up() -> None:
    """Walking drains the urge to walk, and the walk is used up as it is: in an empty room a duck whose
    urge is full walks until the urge is spent (and not for ever) and the bout ends satisfied."""
    assert B.NEED_OF["wander"] == "restless"
    brain = NeuroBrain(_genome("bold_explorer"), seed=3)
    world = SynthWorld(seed=3, family=False)
    world.toys = []
    brain.drives.restless = 1.0
    lengths: list[float] = []
    satisfied: list[float] = []
    for behaviour, seconds, satiation in _bouts(brain, world, 400.0):
        if behaviour == "wander":
            lengths.append(seconds)
            satisfied.append(satiation)
    assert lengths, "it never walked"
    assert max(lengths) < 120.0, lengths  # the urge (about 40 s of walking) is what ends it
    assert sum(1 for s in satisfied if s >= S.FINISH_SAT) >= len(satisfied) * 0.6, satisfied
