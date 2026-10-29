"""a:sound (2026-09-28) — 「咻～」兜風 (the brain's side): a ride is felt from its own body — lifted
(proprio) and swung (the gyro) — not merely held; bold, playful ducks are thrilled (the wheee
hold), timid ones frightened (alarm); good rides are learned; a set-down after a ride leaves the
pull to shake."""

from __future__ import annotations

import math

from duckmind.neuro import skill_ride as SR
from duckmind.neuro.contract import BodySense, SensorFrame


def _frame(
    t: float,
    *,
    lift_at: float = 1.0,
    down_at: float = 8.0,
    swing: tuple[float, float] = (2.0, 7.0),
    fallen: bool = False,
) -> SensorFrame:
    events: tuple[str, ...] = ()
    if abs(t - lift_at) < 0.011:
        events = ("lifted",)
    if abs(t - down_at) < 0.011:
        events = ("set_down",)
    gyro = (0.0, 0.0, 0.0)
    if swing[0] < t < swing[1]:
        w = 2 * math.pi / 1.7
        gyro = (0.4 * 2.6 * math.cos(w * t), 0.1, 2.6 * math.cos(w * t))
    return SensorFrame(
        t_ms=int(t * 1000), body=BodySense(posture="sit", gyro=gyro, fallen=fallen), events=events
    )


def _ride(skill: SR.RideSkill, seconds: float = 12.0, **kw) -> list[SR.RideTick]:
    out = []
    for tick in range(int(seconds * 50)):
        t = tick / 50.0
        out.append(skill.step(int(t * 1000), _frame(t, **kw), play=0.4, fear=0.05, arousal=0.4))
    return out


def test_held_still_is_not_a_ride() -> None:
    skill = SR.RideSkill(boldness=0.9)
    trace = _ride(skill, swing=(99.0, 99.5))
    assert any(t.view.lifted for t in trace)
    assert not any(t.view.riding for t in trace)
    assert not any(t.wheee_hold for t in trace)


def test_a_bold_duck_is_thrilled_until_it_is_set_down() -> None:
    skill = SR.RideSkill(boldness=0.9, activity=0.8)
    trace = _ride(skill)
    riding = [i for i, t in enumerate(trace) if t.view.riding]
    assert riding and riding[0] / 50.0 < 2.6  # swung: a ride within half a second
    assert all(trace[i].wheee_hold for i in riding)  # the whole ride holds the wheee
    assert riding[-1] / 50.0 < 8.05  # … and it ends at the set-down
    assert not any(t.alarm for t in trace)
    assert trace[-1].after_ride > 0.3  # ruffled: the pull to shake


def test_a_long_still_hold_ends_the_ride() -> None:
    skill = SR.RideSkill(boldness=0.9)
    trace = _ride(skill, seconds=14.0, down_at=13.5, swing=(2.0, 4.0))
    ends = [i for i, t in enumerate(trace) if t.ended]
    assert ends and 4.0 + SR.STILL_END_S - 0.3 < ends[0] / 50.0 < 4.0 + SR.STILL_END_S + 1.0


def test_a_timid_duck_is_frightened() -> None:
    skill = SR.RideSkill(boldness=0.02, activity=0.2, young=0.6)
    trace = _ride(skill)
    assert any(t.view.riding for t in trace)
    assert not any(t.wheee_hold for t in trace)
    assert sum(t.alarm for t in trace) >= 2
    assert any(t.pose == SR.FEAR_POSE for t in trace)


def test_good_rides_are_learned() -> None:
    skill = SR.RideSkill(boldness=0.3, activity=0.3)
    joys = []
    for ride in range(6):
        t0 = ride * 20.0
        for tick in range(20 * 50):
            t = t0 + tick / 50.0
            frame = _frame(tick / 50.0)
            frame = SensorFrame(t_ms=int(t * 1000), body=frame.body, events=frame.events)
            out = skill.step(int(t * 1000), frame, play=0.35, fear=0.05, arousal=0.4)
        joys.append(out.joy)
    assert skill.value > 0.5
    assert joys[-1] > joys[0]
    assert joys[0] < SR.JOY_AT <= joys[-1]  # from frightened to thrilled, by its own rides


def test_a_fall_right_after_makes_rides_scary() -> None:
    skill = SR.RideSkill(boldness=0.9)
    _ride(skill, seconds=8.5)
    for tick in range(int(8.5 * 50), int(9.5 * 50)):
        t = tick / 50.0
        skill.step(int(t * 1000), _frame(t, fallen=True), play=0.4, fear=0.05, arousal=0.4)
    assert skill.value < 0.0
