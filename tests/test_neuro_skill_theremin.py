"""a:sound (2026-09-28) — 手的特雷門琴 (the brain's side): a hand hovering in front of the head is
sensed from the ToF summary (a return 10–70 cm away that stays and moves); a still return (a
wall, a chair leg) soon is not interesting; no ToF, no theremin; playing along raises the learned
value of a hovering hand, a fright lowers it."""

from __future__ import annotations

import math

from duckmind.neuro import skill_theremin as ST
from duckmind.neuro.contract import BodySense, CameraSense, SensorFrame, Sighting, TofSense


def _frame(
    t_ms: int,
    ahead_mm: int | None,
    *,
    present: bool = True,
    hand: bool = False,
    touch: bool = False,
) -> SensorFrame:
    sightings = (Sighting("hand", 0.0, range_m=0.3),) if hand else ()
    return SensorFrame(
        t_ms=t_ms,
        body=BodySense(posture="stand"),
        tof=TofSense(present=present, ahead_mm=ahead_mm, touch=touch),
        camera=CameraSense(usable=True, sightings=sightings),
    )


def _wave(t: float) -> int:
    return int(300 + 150 * math.sin(2 * math.pi * 0.4 * t))


def test_a_moving_hand_in_front_is_a_hover() -> None:
    skill = ST.ThereminSkill()
    out = None
    for tick in range(100):
        out = skill.step(
            tick * 20, _frame(tick * 20, _wave(tick / 50)), playing=False, fright=False
        )
    assert out is not None and out.view.present
    assert out.view.moving > 0.5
    assert out.feature > 0.6  # (0.75 × with nothing learned yet)
    assert 0.1 < (out.view.range_m or 0) < 0.7


def test_it_takes_a_moment_to_be_there() -> None:
    skill = ST.ThereminSkill()
    out = skill.step(0, _frame(0, 300), playing=False, fright=False)
    out = skill.step(100, _frame(100, 310), playing=False, fright=False)
    assert not out.view.present and out.feature == 0.0


def test_a_still_return_is_not_a_hand() -> None:
    skill = ST.ThereminSkill()
    feats = []
    for tick in range(600):  # 12 s of a wall 40 cm ahead
        feats.append(
            skill.step(tick * 20, _frame(tick * 20, 400), playing=False, fright=False).feature
        )
    assert max(feats) == 0.0  # something is there, but it never moves: not a hand


def _noisy_tof_run(skill: ST.ThereminSkill, mm_at, seconds: float, seed: int = 3) -> list[float]:
    """The brain at 50 Hz, the ToF summary at tofd's 15 Hz with the model's noise (σ = 3 mm +
    0.5 % of the range, the nearest of a few cells), its age counting up between frames."""
    import random

    rng = random.Random(seed)
    feats: list[float] = []
    reading, frame_t = None, -1.0
    for tick in range(int(seconds * 50)):
        t = tick / 50
        if t - frame_t >= 1 / 15:
            frame_t = t
            true_mm = mm_at(t)
            sigma = 3.0 + 0.005 * true_mm
            reading = int(min(true_mm + rng.gauss(0.0, sigma) for _ in range(3)))
        age = int(round((t - frame_t) * 1000))
        frame = SensorFrame(
            t_ms=tick * 20,
            body=BodySense(posture="stand"),
            tof=TofSense(present=True, ahead_mm=reading, age_ms=age),
        )
        feats.append(skill.step(tick * 20, frame, playing=False, fright=False).feature)
    return feats


def test_the_tof_s_own_noise_on_a_still_thing_is_not_a_hand() -> None:
    """a:sound 2026-09-28: v1 took ±5 mm of ToF jitter on a box 45 cm ahead for a moving hand."""
    for mm in (150, 300, 460, 650):
        feats = _noisy_tof_run(ST.ThereminSkill(), lambda t, mm=mm: mm, 30.0, seed=mm)
        assert max(feats) == 0.0, mm


def test_a_noisy_moving_hand_is_still_a_hand() -> None:
    feats = _noisy_tof_run(
        ST.ThereminSkill(), lambda t: 320 + 60 * math.sin(2 * math.pi * 0.6 * t), 10.0
    )
    assert max(feats[100:]) > 0.6
    feats = _noisy_tof_run(ST.ThereminSkill(), lambda t: max(200, 450 - 125 * t), 4.0)
    assert max(feats) > 0.6  # a hand coming in from 45 to 20 cm


def test_a_hand_that_stops_moving_fades() -> None:
    skill = ST.ThereminSkill()
    for tick in range(100):
        skill.step(tick * 20, _frame(tick * 20, _wave(tick / 50)), playing=False, fright=False)
    feats = [
        skill.step(tick * 20, _frame(tick * 20, 400), playing=False, fright=False).feature
        for tick in range(100, 700)
    ]
    assert feats[0] > 0.3 and feats[-1] < 0.05


def test_the_camera_s_hand_counts_as_lively() -> None:
    skill = ST.ThereminSkill()
    out = None
    for tick in range(300):
        out = skill.step(tick * 20, _frame(tick * 20, 400, hand=True), playing=False, fright=False)
    assert out is not None and out.feature >= 0.74


def test_out_of_the_band_is_nothing() -> None:
    skill = ST.ThereminSkill()
    for mm in (50, 900):
        for tick in range(60):
            out = skill.step(tick * 20, _frame(tick * 20, mm), playing=False, fright=False)
        assert not out.view.present and out.feature == 0.0


def test_no_tof_module_no_theremin() -> None:
    skill = ST.ThereminSkill()
    for tick in range(100):
        out = skill.step(
            tick * 20, _frame(tick * 20, None, present=False), playing=True, fright=False
        )
    assert out.feature == 0.0 and not out.up


def test_playing_along_is_learned_and_a_fright_unlearns_it() -> None:
    skill = ST.ThereminSkill()
    for tick in range(50 * 40):  # 40 s of play, the hand moving
        out = skill.step(tick * 20, _frame(tick * 20, _wave(tick / 50)), playing=True, fright=False)
    assert out.up and out.playing_s > 39.0
    learned = out.value
    assert learned > 0.25
    out = skill.step(50 * 40 * 20 + 20, _frame(50 * 40 * 20 + 20, 300), playing=True, fright=True)
    assert out.value < learned - 0.15


def test_the_gaze_follows_the_third_it_is_in() -> None:
    skill = ST.ThereminSkill()
    for tick in range(100):
        frame = SensorFrame(
            t_ms=tick * 20,
            body=BodySense(posture="stand"),
            tof=TofSense(present=True, left_mm=_wave(tick / 50)),
        )
        out = skill.step(tick * 20, frame, playing=False, fright=False)
    assert out.gaze_bearing > 0.2  # its left third


def test_its_own_head_motion_is_not_the_hand_moving() -> None:
    """A still thing 30 cm ahead seems to move while the duck moves its own head: not a hand."""
    skill = ST.ThereminSkill()
    feats = []
    for tick in range(300):
        t = tick / 50
        yaw = 0.4 * math.sin(2 * math.pi * 0.5 * t)  # its head sweeping left and right
        frame = SensorFrame(
            t_ms=tick * 20,
            body=BodySense(posture="stand", head=(0.349, 0.349, yaw, 0.0)),
            tof=TofSense(present=True, ahead_mm=int(300 + 120 * abs(math.sin(math.pi * t)))),
        )
        feats.append(skill.step(tick * 20, frame, playing=False, fright=False).feature)
    assert max(feats) == 0.0


def test_a_slow_drift_of_its_own_head_is_not_the_hand_moving() -> None:
    """A preen's slow head drift (0.15 rad/s, under the joint-speed check) swings the beam over a
    slanted surface: the range changes by centimetres, but that is its own doing."""
    skill = ST.ThereminSkill()
    feats = []
    for tick in range(500):
        t = tick / 50
        pitch = 0.1 + 0.15 * (t % 2.0) if (t % 4.0) < 2.0 else 0.4 - 0.15 * (t % 2.0)
        frame = SensorFrame(
            t_ms=tick * 20,
            body=BodySense(posture="stand", head=(0.0, pitch, 0.0, 0.0)),
            tof=TofSense(present=True, ahead_mm=int(380 + 400 * (pitch - 0.1))),
        )
        feats.append(skill.step(tick * 20, frame, playing=False, fright=False).feature)
    assert max(feats) == 0.0


def test_a_far_moving_return_needs_the_eye() -> None:
    skill = ST.ThereminSkill()
    for tick in range(200):
        out = skill.step(
            tick * 20,
            _frame(tick * 20, 600 + int(60 * math.sin(tick / 10))),
            playing=False,
            fright=False,
        )
    assert out.feature == 0.0  # 60 cm and moving, but no hand in view: a person passing
    for tick in range(200, 300):
        out = skill.step(tick * 20, _frame(tick * 20, 600, hand=True), playing=False, fright=False)
    assert out.feature > 0.6
