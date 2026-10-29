"""Growth is not told of a hand or a visitor that was only the duck's own motion (owner principle
2026-10-10, found by the live check of 2026-10-11).

A fast ``looming`` sighting with something at the face for the ToF used to be a hand, and a hand
made "someone is here", so a duck that walked up to a thing, turned and tumbled was told it had been
visited. A real robot's rule: a looming sighting is a hand only while the duck itself has been quiet
(no walking, turning, bump, stumble, fall, righting, a still trunk); only the camera's own ``hand``
(or ``person``) sighting means someone is here. Blocked ones are counted in ``ignored`` as
``hand_false`` and ``company_false``.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

import pytest
from test_life_experience import frame, kinds, run, ticks
from test_neuro_world import Home

from duckmind.life.experience import QUIET_MS, Experiences
from duckmind.neuro import contract as C

LOOM = C.Sighting("looming", 0.0, 0.1, None, 0.2, 0.8, 2.5, 0, "model")
HAND = C.Sighting("hand", 0.0, 0.1, 0.4, 0.2, 0.8, 2.5, 7, "model")
PERSON = C.Sighting("person", 0.1, 0.3, 1.2, 0.25, 0.0, 0.0, 1, "model")
NEAR = C.TofSense(present=True, ahead_mm=120)


def seen(
    t: int,
    *sightings: C.Sighting,
    tof: C.TofSense = NEAR,
    vx: float = 0.0,
    vyaw: float = 0.0,
    gyro: float = 0.0,
    posture: str = "stand",
    events: tuple[str, ...] = (),
) -> C.SensorFrame:
    f = frame(t, sightings=sightings, events=events, posture=posture)
    body = dataclasses.replace(f.body, vx=vx, vyaw=vyaw, gyro=(0.0, 0.0, gyro))
    return dataclasses.replace(f, body=body, tof=tof)


def episode(
    start: int, *sightings: C.Sighting, ms: int = 1000, **kw: Any
) -> list[tuple[C.SensorFrame, None]]:
    return [(seen(t, *sightings, **kw), None) for t in range(start, start + ms, 100)]


def quiet(a: int, b: int) -> list[tuple[C.SensorFrame, None]]:
    return [(seen(t, tof=C.TofSense()), None) for t in range(a, b, 100)]


def test_a_thing_looming_at_the_face_of_a_walking_duck_is_no_hand_and_no_visitor() -> None:
    exp = Experiences()
    out = run(exp, quiet(0, 3000) + episode(3000, LOOM, vx=0.3, posture="walk") + ticks(4000, 6000))
    assert out == []
    assert exp.ignored == {"hand_false": 1}


@pytest.mark.parametrize(
    "busy",
    [
        {"vyaw": 0.5},
        {"gyro": 1.5},
        {"posture": "unknown"},
        {"events": ("stumbled",)},
    ],
)
def test_turning_tumbling_or_stumbling_makes_a_looming_thing_no_hand(busy: dict[str, Any]) -> None:
    exp = Experiences()
    out = run(
        exp, quiet(0, 3000) + episode(3000, LOOM, ms=300, **busy) + episode(3300, LOOM, ms=700)
    )
    # (the first 0.3 s was its own motion; the rest is within QUIET_MS of it)
    assert out == [] and exp.ignored.get("hand_false", 0) >= 1


def test_the_stumble_before_the_startle_of_the_live_check() -> None:
    """Pressed against a thing, a turn, a stumble, 1 s later a startle with the thing at its face:
    what the live check of 2026-10-11 told as a fast hand and a visitor."""
    frames = quiet(0, 3000)
    frames += [(seen(3000, tof=C.TofSense(), events=("stumbled",)), None)]
    frames += episode(3100, LOOM, ms=1200) + ticks(4300, 7000)
    exp = Experiences()
    out = run(exp, frames)
    assert out == [] and exp.ignored == {"hand_false": 1}


def test_a_looming_thing_at_the_face_of_a_quiet_duck_is_still_a_fast_hand_but_no_visitor() -> None:
    exp = Experiences()
    out = run(exp, quiet(0, 3000) + episode(3000, LOOM) + ticks(4000, 75_000, step=500))
    assert kinds(out) == ["hand"] and out[0]["speed"] == "fast"
    assert exp.ignored == {"company_false": 1}  # (nobody was shown to be here: no company)


def test_the_quiet_window_is_a_moment() -> None:
    assert 500 <= QUIET_MS <= 3000
    frames = quiet(0, 500) + [(seen(500, tof=C.TofSense(), vx=0.3, posture="walk"), None)]
    frames += quiet(600, 600 + QUIET_MS) + episode(600 + QUIET_MS, LOOM)
    out = run(Experiences(), frames + ticks(5000, 8000))
    assert kinds(out) == ["hand"]


def test_a_camera_hand_is_a_hand_and_a_visitor() -> None:
    """The page's hand in front of the eye (and a person's): the camera's own class."""
    exp = Experiences()
    out = run(exp, episode(0, HAND, tof=C.TofSense()) + ticks(1000, 75_000, step=500))
    assert kinds(out)[0] == "hand" and "company" in kinds(out)
    assert exp.ignored == {}
    # … also when the duck is walking (it is a hand the eye saw, not the walk's looming)
    walking = Experiences()
    out = run(walking, episode(0, HAND, vx=0.3, posture="walk") + ticks(1000, 75_000, step=500))
    assert kinds(out)[0] == "hand" and "company" in kinds(out)


def test_a_person_in_view_is_company_and_a_visit() -> None:
    exp = Experiences()
    frames = ticks(0, 700_000, step=1000)
    frames += [(seen(700_000 + i * 100, PERSON, tof=C.TofSense()), None) for i in range(900)]
    frames += ticks(790_000, 800_000, step=1000)
    out = run(exp, frames)
    assert "visit" in kinds(out) and "company" in kinds(out)


# ── the same in the physics: the page's hand in front of the eye ─────────────────────────


def test_the_page_hand_in_front_of_the_duck_is_still_a_hand() -> None:
    home = Home(seed=3, when=datetime(2026, 12, 20, 12, 0))
    home.run(1.0)
    home.part.stimulate({"stim": "hand_loom"})
    home.run(3.0)
    assert home.sighted("hand") or home.sighted("looming")
    exp = Experiences()
    out: list[dict[str, Any]] = []
    for f in home.frames:
        out += exp.observe(1_773_000_000 + f.t_ms // 1000, f)
    out += exp.flush(1_773_000_010)
    assert "hand" in kinds(out)
    assert exp.ignored.get("hand_false", 0) == 0
