"""Duck Core's growth is fed only what really happened to the duck (owner principle 2026-10-10).

* A slipper, a shoe or a toy bumping the duck is no touch and no poke: only the kit hand (a page
  pat or poke, a person's hand) is. The world's contact record says who (``by``); the touch sense
  carries its kind; the experience observer tells Duck Core only a hand's contact.
* A page poke is a poke, also when a thing rests against the duck: the hand's contact is not merged
  with the thing's into one long "touch".
* A fall is a fall of the body that really happened: the settling of a body just put into the world
  (a boot, a layout Apply) is not one.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from test_life_experience import T0, frame, kinds, run, ticks
from test_neuro_world import Home

from duckmind.duckcore import bridge
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C
from duckmind.world import neuro_senses as ns


def by(
    who: str, a: int, b: int, *, site: str = "left", press: float = 1.0, stroke: float = 0.0
) -> list[tuple[C.SensorFrame, None]]:
    """Frames a to b ms (100 ms apart) with a contact at ``site`` made by ``who``."""
    out = []
    for t in range(a, b, 100):
        f = frame(t, touch=(site, press, stroke))
        out.append((dataclasses.replace(f, touch=dataclasses.replace(f.touch, by=who)), None))
    return out


# ── (a) things are not hands ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("who", ["thing", "person"])
@pytest.mark.parametrize("site", ["head", "back", "left", "right"])
def test_a_thing_or_a_leg_against_it_is_no_touch_and_no_poke(who: str, site: str) -> None:
    exp = Experiences()
    out = run(exp, by(who, 0, 3000, site=site, press=0.8) + ticks(3000, 5000))
    assert out == [] and list(exp.contacts) == []
    assert exp.ignored == {who: 1}  # (counted, as one contact, never told)


def test_the_hand_is_told_and_so_is_a_contact_nobody_labelled() -> None:
    hand = run(
        Experiences(), by("hand", 0, 800, site="head", press=0.3, stroke=0.6) + ticks(800, 1800)
    )
    legacy = run(
        Experiences(), by("", 0, 800, site="head", press=0.3, stroke=0.6) + ticks(800, 1800)
    )
    assert kinds(hand) == ["touch"] and kinds(legacy) == [
        "touch"
    ]  # (the synthetic world has no by)
    poke = run(Experiences(), by("hand", 0, 300, site="left") + ticks(300, 1300))
    assert kinds(poke) == ["poke"]


def test_a_thing_after_a_hand_does_not_make_the_hand_longer() -> None:
    exp = Experiences()
    out = run(
        exp,
        by("hand", 0, 300, site="left")
        + by("thing", 300, 4000, site="left", press=0.4)
        + ticks(4000, 5000),
    )
    assert [(e["k"], e["ms"]) for e in out] == [("poke", 200)]


# ── (b) a poke is a poke, also with a thing resting against the duck ──────────────────────


def test_a_poke_on_top_of_a_resting_thing_is_still_a_poke() -> None:
    """The live fault: a thing's contact (a ball that rolled against it, resting for seconds) and
    the hand's poke were one episode of the touch sense, so the poke came out 2.4 s long and
    was told as a touch. With who touched it known they are two contacts."""
    frames = by("thing", 0, 2000, site="left", press=0.5)
    frames += by("hand", 2000, 2300, site="left") + ticks(2300, 3300)
    now = run(Experiences(), frames)
    assert [(e["k"], e["ms"]) for e in now] == [("poke", 200)]
    # the same frames when nobody says who (before the fix): one long touch
    blind = [
        (dataclasses.replace(f, touch=dataclasses.replace(f.touch, by="")), c) for f, c in frames
    ]
    old = run(Experiences(), blind)
    assert [e["k"] for e in old] == ["touch"] and old[0]["ms"] >= 2000


def test_the_touch_sense_says_who_by_kind_and_never_an_id() -> None:
    def contact(key: str, who: str) -> dict[str, Any]:
        return {
            "key": key,
            "by": who,
            "site": "left",
            "t0": 0.0,
            "t1": 1.0,
            "active": True,
            "depth_mm": 4.0,
            "closing": 0.5,
            "slide": 0.0,
            "now_depth_mm": 4.0,
            "now_slide": 0.0,
        }

    seen: dict[str, str] = {}
    for who, kind in (
        ("hand", "hand"),
        ("thing:slipper_left", "thing"),
        ("person:owner", "person"),
    ):
        for seed in range(30):  # (a firm contact is felt 92 % of the time)
            model = ns.TouchModel(seed=seed)
            model.update([contact(f"{who}#1", who)])
            sense = model.step(0.5)
            if sense.site is not None:
                seen[kind] = sense.by
                break
    assert seen == {"hand": "hand", "thing": "thing", "person": "person"}
    assert "slipper" not in json.dumps(dataclasses.asdict(sense))


def test_an_old_frame_without_by_still_loads() -> None:
    data = frame(0, touch=("head", 0.3, 0.2)).to_json()
    del data["touch"]["by"]
    assert C.SensorFrame.from_json(data).touch.by == ""


# ── the same in a real home (MuJoCo): a rolled ball, a page poke, a page pat ──────────────


def _live_run(seed: int, wait: float) -> tuple[list[dict[str, Any]], Experiences, Home]:
    home = Home(seed=seed)
    home.part.stimulate({"stim": "roll_ball"})
    home.run(wait)  # (the ball hit the duck and rests against it)
    home.part.stimulate({"stim": "poke", "side": "left"})
    home.run(4.0)
    exp = Experiences()
    out = run_frames(exp, home.frames)
    return out, exp, home


def run_frames(exp: Experiences, frames: list[C.SensorFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in frames:
        out += exp.observe(T0 + f.t_ms // 1000, f, None, None)
    return out + exp.flush(T0 + frames[-1].t_ms // 1000 + 5)


def test_a_ball_resting_against_the_duck_is_felt_but_not_told(tmp_path: Any) -> None:
    out, exp, home = _live_run(seed=2, wait=3.2)
    by_ball = {c.by for c in home.part.ended} | {c.by for c in home.part.contacts.values()}
    assert "hand" in by_ball and any(b.startswith("thing:") for b in by_ball)
    felt = [f.touch.by for f in home.frames if f.touch.site]
    assert "thing" in felt and "hand" in felt  # (the reflexes still feel the ball)
    assert [(e["k"], e["ms"]) for e in out] == [("poke", 200)]
    assert exp.ignored == {"thing": 1}
    for event in out:
        bridge.check(event)
    blind = Experiences()
    old = run_frames(
        blind,
        [dataclasses.replace(f, touch=dataclasses.replace(f.touch, by="")) for f in home.frames],
    )
    assert "touch" in kinds(old)  # (what the growth used to hear of a ball)


def test_a_page_pat_is_still_a_touch_on_the_head() -> None:
    told = 0
    for seed in range(1, 5):  # (a light touch is felt three times in four)
        home = Home(seed=seed)
        home.part.stimulate({"stim": "pet_head"})
        home.run(5.0)
        out = run_frames(Experiences(), home.frames)
        assert {c.by for c in home.part.ended} == {"hand"}
        told += kinds(out).count("touch")
    assert told >= 4


# ── (c) falls: only a body that really fell ────────────────────────────────────────────────


def test_a_body_settling_after_a_boot_is_no_fall() -> None:
    """What the first 40 s of a boot said on :8993 (five boots, none fell): it unloads and loads
    its feet as it stands up, may stumble once, and sits or stands. None of it is a fall."""
    frames: list[tuple[C.SensorFrame, None]] = []
    for t in range(0, 40_000, 100):
        events = {
            2_600: ("unloaded",),
            3_000: ("unloaded_clear",),
            17_700: ("stumbled",),
            22_000: ("blocked",),
            23_000: ("brushed",),
        }.get(t, ())
        posture = "sit" if t < 15_000 else "stand"
        frames.append((frame(t, events=events, posture=posture), None))
    exp = Experiences()
    assert [e for e in run(exp, frames) if e["k"] == "fall"] == []


def test_a_real_fall_is_one_fall_with_the_spot_and_how_it_ended() -> None:
    """The live record of a fall (loops/20261010-225558: tilt 107 degrees after walking into
    something, righted by itself in 0.44 s): one 'easy' fall at the spot; the body saying 'fell'
    again while it lies there does not make a second."""
    frames = [(frame(t, odom=(0.409, 0.053, 0.0)), None) for t in range(0, 1000, 100)]
    frames += [(frame(1000, events=("fell",), odom=(0.409, 0.053, 0.0)), None)]
    frames += [(frame(1100, events=("fell",)), None)]
    frames += [(frame(1442, events=("righted",)), None)]
    frames += [(frame(t), None) for t in range(1500, 4000, 100)]
    out = run(Experiences(), frames)
    assert [(e["k"], e["outcome"], e["x_mm"], e["y_mm"]) for e in out] == [
        ("fall", "easy", 409, 53)
    ]
    bridge.check(out[0])
