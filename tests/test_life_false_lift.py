"""Growth is told a lift only when the duck was really lifted (owner principle 2026-10-10).

The body's ``lifted`` claim is made from the legs alone (they went light, or they work hard with the
trunk still), so a duck that stands still on the floor with light legs, or one that stands on a sill
with its legs working, is "lifted" too. A duck in a hand keeps its legs light (seated in the hand
≈ 0.4 of a quiet stand), a duck seated on the floor or on a sill does not (≈ 0.7 and more): the
experience observer tells ``lift`` once the legs have been light for half a second after the claim,
and counts the claim it did not tell as ``lift_false``. The body's own reflexes are not touched.

The cases in ``tests/fixtures/lift/traces.json`` are frames cut out of recorded runs of the real home
with nobody around (what the live server and the play rig really read) and of page pick-ups on the
same rig.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from duckmind.duckcore import bridge
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C
from duckmind.neuro.liftcheck import LIFT_LIGHT_MS, LIFT_LOAD

T0 = 1_773_000_000
FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "lift" / "traces.json").read_text(encoding="utf-8")
)["cases"]


def body_frame(
    t_ms: int,
    load: float | None,
    *,
    posture: str = "stand",
    events: tuple[str, ...] = (),
    fallen: bool = False,
    touch: tuple[str | None, str] = (None, ""),
) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture=posture, fallen=fallen, leg_load=load),
        events=events,
        touch=C.TouchSense(
            site=touch[0], pressure=0.3 if touch[0] else 0.0, stroke=0.6, by=touch[1]
        ),
    )


def replay(exp: Experiences, case: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t_ms, load, posture, events, fallen in case["frames"]:
        frame = body_frame(
            t_ms,
            load,
            posture=posture,
            events=tuple(e for e in events.split(",") if e),
            fallen=bool(fallen),
        )
        out += exp.observe(T0 + t_ms // 1000, frame)
    return out


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [e["k"] for e in events]


def edges_of(case: dict[str, Any]) -> list[int]:
    return [f[0] for f in case["frames"] if "lifted" in f[3].split(",")]


# ── the recorded false claims: a lift the legs do not bear out is not told ──────────────────────

FALSE_CLAIMS = [
    "live_stand_light",
    "live_stand_light_2",
    "rig_flail_on_sill",
    "rig_stand_light_a",
    "rig_stand_light_b",
]


@pytest.mark.parametrize("name", FALSE_CLAIMS)
def test_a_claim_the_legs_do_not_bear_out_is_not_told(name: str) -> None:
    case = FIXTURES[name]
    assert len(edges_of(case)) == 1  # (the body did claim a lift here)
    exp = Experiences()
    out = replay(exp, case)
    assert "lift" not in kinds(out)
    assert exp.ignored == {"lift_false": 1}
    assert exp.sent.get("lift", 0) == 0


def test_a_fall_and_its_righting_are_falls_and_no_lift() -> None:
    case = FIXTURES["rig_fall_righting"]
    exp = Experiences()
    out = replay(exp, case)
    assert kinds(out) == ["fall", "fall"]
    assert [e["outcome"] for e in out] == ["easy", "easy"]
    assert exp.ignored.get("lift_false", 0) == 0
    for event in out:
        bridge.check(event)


def test_a_claim_in_the_middle_of_a_righting_is_no_lift() -> None:
    """Whatever the legs read while it lies and rights, a duck that is falling is not carried."""
    case = FIXTURES["rig_fall_righting"]
    frames = [list(f) for f in case["frames"]]
    fell = next(i for i, f in enumerate(frames) if f[3] == "fell")
    # the claim comes 0.4 s after it fell, and the legs read light (they do, lying on its side)
    claim = next(i for i in range(fell, len(frames)) if frames[i][0] >= frames[fell][0] + 400)
    for i in range(claim, claim + 40):
        frames[i][1] = 0.3
    frames[claim][3] = "lifted"
    exp = Experiences()
    out = replay(exp, {"frames": frames})
    assert kinds(out) == ["fall", "fall"]
    assert exp.ignored == {"lift_false": 1}


# ── synthetic: what the rule is ────────────────────────────────────────────────────────────


def run(frames: list[C.SensorFrame], exp: Experiences | None = None) -> list[dict[str, Any]]:
    exp = exp or Experiences()
    out: list[dict[str, Any]] = []
    for f in frames:
        out += exp.observe(T0 + f.t_ms // 1000, f)
    return out


def held(a: int, b: int, load: float = 0.4, step: int = 20) -> list[C.SensorFrame]:
    return [body_frame(t, load, posture="sit") for t in range(a, b, step)]


def claim(t: int, load: float = 0.4, posture: str = "sit") -> C.SensorFrame:
    return body_frame(t, load, posture=posture, events=("lifted",))


def test_a_held_duck_is_told_once_at_the_second_it_was_picked_up() -> None:
    frames = [body_frame(t, 1.0) for t in range(0, 1900, 20)]
    frames += [claim(1900, 1.36)] + held(1920, 2900, 1.5)  # (it sits in the hand: legs swing)
    frames += held(2900, 5000, 0.4)  # … then they hang
    exp = Experiences()
    out = run(frames, exp)
    assert [(e["k"], e["t"]) for e in out] == [("lift", T0 + 1)]  # (the second of the claim)
    assert exp.ignored == {}
    bridge.check(out[0])


def test_a_pick_up_is_told_as_soon_as_the_legs_have_hung_for_half_a_second() -> None:
    frames = [claim(0, 0.4)] + held(20, 2000)
    told_at = None
    exp = Experiences()
    for f in frames:
        if exp.observe(T0, f) and told_at is None:
            told_at = f.t_ms
    assert told_at is not None and LIFT_LIGHT_MS <= told_at <= LIFT_LIGHT_MS + 40


def test_legs_that_hang_a_moment_and_load_again_are_no_lift() -> None:
    frames = [claim(0, 0.4)] + held(20, 300) + held(300, 5000, 0.75)
    exp = Experiences()
    assert run(frames, exp) == [] and exp.ignored == {"lift_false": 1}


def test_a_duck_put_down_before_its_legs_have_hung_is_no_lift() -> None:
    frames = [claim(0, 0.4)] + held(20, 400) + [body_frame(400, 0.8, events=("set_down",))]
    frames += held(420, 4000, 0.8)
    exp = Experiences()
    assert run(frames, exp) == [] and exp.ignored == {"lift_false": 1}


def test_the_hang_must_be_unbroken() -> None:
    frames = [claim(0, 0.4)] + held(20, 400) + held(400, 440, 0.9) + held(440, 900, 0.4)
    frames += held(900, 1200, 0.4)
    out = run(frames)
    assert kinds(out) == ["lift"]  # (the second stretch of hanging is long enough)


def test_a_claim_nothing_bears_out_within_three_seconds_is_dropped() -> None:
    frames = [claim(0, 0.9)] + held(20, 3500, 0.9)
    exp = Experiences()
    assert run(frames, exp) == [] and exp.ignored == {"lift_false": 1}
    # … and it is the loaded legs, not the clock: hanging at the end of the window is too late
    late = [claim(0, 0.9)] + held(20, 3100, 0.9) + held(3100, 4000, 0.4)
    assert run(late) == []


def test_the_threshold_is_between_a_held_duck_and_a_seated_one() -> None:
    assert 0.4 < LIFT_LOAD < 0.7
    for load, told in ((0.38, True), (0.5, True), (0.6, False), (0.75, False), (1.0, False)):
        out = run([claim(0, load)] + held(20, 2000, load))
        assert (kinds(out) == ["lift"]) is told, load


def test_a_frame_without_a_leg_load_is_told_at_once_as_before() -> None:
    """The synthetic world and old recordings have no leg load: nothing to check the claim by."""
    out = run([body_frame(0, None), body_frame(100, None, events=("lifted",))])
    assert [(e["k"], e["t"]) for e in out] == [("lift", T0)]


def test_a_falling_duck_is_not_carried() -> None:
    frames = [claim(0, 0.4), body_frame(20, 0.4, events=("fell",))] + held(40, 2000, 0.3)
    exp = Experiences()
    assert kinds(run(frames, exp)) == []
    assert exp.ignored == {"lift_false": 1}
    # the body says it is fallen (robotd's flag): no lift either
    fallen = [claim(0, 0.4)] + [body_frame(t, 0.3, fallen=True) for t in range(20, 2000, 20)]
    assert run(fallen) == []


def test_a_bump_before_the_claim_does_not_make_it_true() -> None:
    frames = [body_frame(t, 1.0) for t in range(0, 900, 20)]
    frames += [body_frame(900, 1.0, events=("stumbled",)), body_frame(920, 0.9)]
    frames += [claim(1000, 0.7, "stand")] + held(1020, 4200, 0.74)
    exp = Experiences()
    assert run(frames, exp) == [] and exp.ignored == {"lift_false": 1}


def test_the_hand_that_lifts_a_duck_is_no_caress_but_a_hand_after_a_false_claim_is() -> None:
    """The contact of the hand that lifted it is the lift, not a touch (G2). A claim that was no
    lift has no such hand: a real touch right after it is a touch."""

    def touch_run(load_after: float) -> list[str]:
        frames: list[C.SensorFrame] = []
        for t in range(0, 600, 20):
            frames.append(body_frame(t, 1.0, touch=("head", "hand")))
        frames.append(claim(600, load_after))
        for t in range(620, 4000, 20):
            frames.append(
                body_frame(
                    t,
                    load_after,
                    posture="sit",
                    touch=("head", "hand") if t < 1400 else (None, ""),
                )
            )
        return kinds(run(frames))

    assert touch_run(0.4) == ["lift"]  # (held: the hand's contact is part of the lift)
    assert touch_run(0.8) == ["touch"]  # (not lifted: the hand only touched it)


def test_a_watch_that_ends_with_a_claim_open_tells_nothing_and_counts_nothing() -> None:
    exp = Experiences()
    out = run([claim(0, 0.4)] + held(20, 300), exp)
    assert out == []
    assert exp.flush(T0 + 5) == [] and exp.ignored == {}
    assert run(held(5000, 8000), exp) == []  # (and the next watch starts clean)


def test_a_second_claim_is_a_second_lift_when_the_legs_bear_it_out() -> None:
    first = [claim(0, 0.4)] + held(20, 1500)
    down = [body_frame(1500, 0.9, events=("set_down",))] + held(1520, 4000, 0.9)
    second = [claim(4000, 0.4)] + held(4020, 5500)
    out = run(first + down + second)
    assert kinds(out) == ["lift", "lift"]


# ── the recorded page pick-ups: still lifts ────────────────────────────────────────────────

TRUE_LIFTS = [name for name, case in FIXTURES.items() if case.get("told") == 1]


@pytest.mark.parametrize("name", TRUE_LIFTS)
def test_a_recorded_page_pick_up_is_still_a_lift(name: str) -> None:
    case = FIXTURES[name]
    (edge,) = edges_of(case)
    exp = Experiences()
    out = replay(exp, case)
    assert [(e["k"], e["t"]) for e in out] == [("lift", T0 + edge // 1000)]
    assert exp.ignored.get("lift_false", 0) == 0


def test_the_recorded_cases_are_there() -> None:
    assert len(TRUE_LIFTS) >= 2 and len(FALSE_CLAIMS) == 5


# ── the same in the physics: a page pick-up on the play rig ────────────────────────────────


def _rig(seed: int):  # type: ignore[no-untyped-def]
    import os

    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from duckmind.neuro.genome import preset
    from duckmind.world import neuro_play_rig as R

    if not os.path.isdir(R.mjcf_dir()):
        pytest.skip("the official duck model is not on this machine")
    return R.PlayNeuro(
        preset("explorer"),
        seed,
        x=0.3,
        y=-0.6,
        yaw=0.0,
        home="real_home",
        when=datetime(2026, 10, 11, 10, 0),
        variant="no_nest",
    )


@pytest.mark.parametrize(("seed", "pick_at"), [(4107, 3.0), (4112, 13.0)])
def test_a_page_pick_up_in_the_physics_is_one_lift_and_a_set_down_is_none(
    seed: int, pick_at: float
) -> None:
    """The world's hand lifts the duck (seed 4107 while it sits and looks around, 4112 while it
    stands): the body claims, the legs hang at ≈ 0.4, growth is told one lift; put down, it is
    told nothing more of it."""
    told: list[tuple[int, dict[str, Any]]] = []
    exp = Experiences()

    def each(rig: Any) -> bool:
        f = rig.frame
        if f is not None:
            for e in exp.observe(T0 + f.t_ms // 1000, f, rig.cmd, rig.tel):
                told.append((f.t_ms, e))
        return False

    with _rig(seed) as rig:
        rig.run(pick_at, each)
        assert rig.part.stimulate({"stim": "pick_up"})["ok"]
        t_pick = rig.frame.t_ms
        rig.run(9.0, each)
        assert rig.part.stimulate({"stim": "set_down"})["ok"]
        rig.run(5.0, each)
    lifts = [(t, e) for t, e in told if e["k"] == "lift"]
    assert len(lifts) == 1, told
    assert t_pick < lifts[0][0] < t_pick + 5000
    assert exp.ignored.get("lift_false", 0) == 0
    assert math.isfinite(lifts[0][0])


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_hard_push_a_stumble_or_a_fall_with_its_righting_is_no_lift(side: str) -> None:
    """A page push of full strength on the play rig (7 N): the duck stumbles, and from one side it
    falls and rights itself. Growth is told the fall (when there was one) and the poke, and no
    lift: nobody picked it up, whatever its legs read on the way down and up."""
    told: list[dict[str, Any]] = []
    seen: list[str] = []
    exp = Experiences()

    def each(rig: Any) -> bool:
        f = rig.frame
        if f is not None:
            seen.extend(f.events)
            told.extend(exp.observe(T0 + f.t_ms // 1000, f, rig.cmd, rig.tel))
        return False

    with _rig(4107) as rig:
        rig.run(8.0, each)
        assert rig.part.stimulate({"stim": "push", "strength": 1.0, "side": side})["ok"]
        rig.run(25.0, each)
    told += exp.flush(T0 + 40)
    assert "stumbled" in seen or "fell" in seen  # (it was knocked about)
    assert "lift" not in kinds(told)
    assert exp.ignored.get("lift_false", 0) == seen.count("lifted")
    if "fell" in seen:
        assert "righted" in seen
        assert kinds(told).count("fall") == 1
