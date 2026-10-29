"""A sound the duck makes itself is not a noise to Duck Core (owner decision 2026-10-11).

Knocking a thing over, bumping into the furniture, a thing hitting it: its reflexes still hear them
(startle, orient), but the growth input is filtered: ``Experiences`` does not tell Duck Core a
``noise`` for them and counts them in ``ignored["noise_self"]``. A person's sound, the page's bang or
clap, and a thing that fell without the duck touching it still count.
(docs/design/SELF-NOISE-2026-10-11.md)
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from test_life_experience import frame, kinds, run, ticks
from test_life_true_experiences import by as touch_by
from test_life_true_experiences import run_frames
from test_neuro_world import Home

from duckmind.duckcore import bridge
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C
from duckmind.world import neuro_senses as ns


def heard(
    a: int, b: int, who: str = "", kind: str = "bang", db: float = 80.0
) -> list[tuple[C.SensorFrame, None]]:
    """Frames a to b ms (100 ms apart) with ``kind`` heard, made by ``who`` (the world's label)."""
    out = []
    for t in range(a, b, 100):
        f = frame(t, sound=(kind, db))
        out.append((dataclasses.replace(f, sound=dataclasses.replace(f.sound, by=who)), None))
    return out


# ── the observer ───────────────────────────────────────────────────────────────────────────


def test_a_sound_the_duck_made_is_no_noise_and_is_counted_once() -> None:
    exp = Experiences()
    out = run(exp, ticks(0, 500) + heard(500, 900, "self") + ticks(900, 3000))
    assert out == []
    assert exp.ignored == {"noise_self": 1}  # (one sound held over several ticks is one)
    assert exp.sent.get("noise", 0) == 0


@pytest.mark.parametrize("who", ["page", "thing", "person", ""])
@pytest.mark.parametrize("kind", ["bang", "clap", "door"])
def test_a_sound_from_the_house_is_a_noise(who: str, kind: str) -> None:
    exp = Experiences()
    out = run(exp, ticks(0, 500) + heard(500, 700, who, kind) + ticks(700, 2000))
    assert [(e["k"], e["what"]) for e in out] == [("noise", kind)]
    bridge.check(out[0])
    assert exp.ignored == {}


def test_a_self_sound_does_not_hide_a_real_bang_a_moment_later() -> None:
    exp = Experiences()
    frames = heard(0, 200, "self") + ticks(200, 2000) + heard(2000, 2200, "page")
    out = run(exp, frames + ticks(2200, 3000))
    assert kinds(out) == ["noise"] and exp.ignored == {"noise_self": 1}


def test_the_label_wins_over_a_bump_a_moment_ago() -> None:
    """A page bang right after the duck bumped into a slipper is still a bang in the house."""
    frames = touch_by("thing", 0, 500, site="left") + heard(600, 800, "page")
    assert kinds(run(Experiences(), frames + ticks(800, 2000))) == ["noise"]


def test_without_a_label_a_sound_right_after_its_own_bump_is_its_own() -> None:
    """The rule a real duck can apply (a microphone says nothing of who made a sound): a loud sound
    within a second of a thing against it, or of walking into something, is its own."""
    exp = Experiences()
    out = run(exp, touch_by("thing", 0, 500, site="left") + heard(600, 800) + ticks(800, 2000))
    assert out == [] and exp.ignored == {"thing": 1, "noise_self": 1}
    exp = Experiences()
    bump = [(frame(500, events=("stumbled",)), None)]
    out = run(exp, ticks(0, 500) + bump + heard(700, 900) + ticks(900, 2000))
    assert out == [] and exp.ignored == {"noise_self": 1}
    # …but not one that comes seconds later, and not with nothing against it
    late = run(Experiences(), touch_by("thing", 0, 500, site="left") + ticks(500, 3000))
    assert late == []
    exp = Experiences()
    after = touch_by("thing", 0, 500, site="left") + ticks(500, 3000) + heard(3000, 3200)
    assert kinds(run(exp, after + ticks(3200, 4500))) == ["noise"]
    assert kinds(run(Experiences(), ticks(0, 500) + heard(500, 700) + ticks(700, 2000))) == [
        "noise"
    ]


def test_a_hands_touch_is_not_the_ducks_own_bump() -> None:
    frames = touch_by("hand", 0, 600, site="head") + heard(700, 900)
    assert "noise" in kinds(run(Experiences(), frames + ticks(900, 2200)))


def test_an_old_frame_without_a_sound_label_still_loads() -> None:
    data = frame(0, sound=("bang", 80.0)).to_json()
    del data["sound"]["by"]
    assert C.SensorFrame.from_json(data).sound.by == ""


# ── the microphone carries the world's label (a kind, never an id) ───────────────────────────


def test_the_hearing_model_carries_the_kind_of_whoever_made_a_sound() -> None:
    seen = {}
    for label in ("self", "thing:slipper_left", "person:owner", "page", ""):
        model = ns.HearingModel(seed=3)
        item = {"id": 1, "kind": "bang", "t0": 1.0, "t1": 1.05, "db": 85.0, "by": label}
        model.update([item])
        sense = C.SoundSense()
        for k in range(150):
            sense = model.step(1.0 + k * 0.02, -100.0)
            if sense.kind:
                break
        seen[label] = (sense.kind, sense.by)
    assert seen["self"] == ("bang", "self")
    assert seen["thing:slipper_left"][1] == "thing" and seen["person:owner"][1] == "person"
    assert seen["page"][1] == "page" and seen[""][1] == ""


# ── in a real home (MuJoCo): who made an impact sound ────────────────────────────────────────


def _knock(
    home: Home,
    body: str,
    other: str | None,
    impulse: int = 300,
    pos: tuple[int, int, int] = (0, 330, 40),
) -> None:
    """One impact in the props' own record, as the physics writes it."""
    props = home.part.props
    props.seq += 1
    props.impacts.append(
        {
            "seq": props.seq,
            "t": round(float(home.d.time), 4),
            "body": body,
            "other": other,
            "impulse_mns": impulse,
            "dv_mm_s": 900,
            "speed_mm_s": 900,
            "pos_mm": list(pos),
        }
    )


def _lived(home: Home) -> tuple[list[dict[str, Any]], Experiences]:
    exp = Experiences()
    return run_frames(exp, home.frames), exp


def _by(home: Home) -> list[str]:
    return [e.by for e in home.part.sounds if e.kind in ("bang", "knock")]


def test_the_duck_bumping_a_thing_is_its_own_sound() -> None:
    home = Home(seed=3)
    home.run(1.0)
    _knock(home, "duck", "paper_cup")
    home.run(2.0)
    assert _by(home) == ["self"]
    out, exp = _lived(home)
    assert [e for e in out if e["k"] == "noise"] == [] and exp.ignored == {"noise_self": 1}


def test_a_thing_hitting_the_duck_is_its_own_sound() -> None:
    home = Home(seed=3)
    home.run(1.0)
    _knock(home, "ball_hollow", "duck")
    home.run(2.0)
    assert _by(home) == ["self"]
    out, exp = _lived(home)
    assert kinds(out).count("noise") == 0 and exp.ignored == {"noise_self": 1}


def test_a_thing_the_duck_pushed_and_that_falls_is_its_own_sound() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = (
        float(home.d.time) - 1.0
    )  # (against it a second ago)
    _knock(home, "paper_cup", "floor")
    home.run(2.0)
    assert _by(home) == ["self"]
    out, exp = _lived(home)
    assert kinds(out).count("noise") == 0 and exp.ignored == {"noise_self": 1}


def test_a_thing_the_duck_touched_long_ago_that_falls_is_a_noise() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = float(home.d.time) - 30.0
    _knock(home, "paper_cup", "floor")
    home.run(2.0)
    assert _by(home) == ["thing"]
    out, exp = _lived(home)
    assert kinds(out).count("noise") == 1 and exp.ignored == {}


def test_a_thing_that_fell_without_the_duck_is_a_noise() -> None:
    home = Home(seed=3)
    home.run(1.0)
    _knock(home, "books", "floor", impulse=400)
    home.run(2.0)
    assert _by(home) == ["thing"]
    out, _ = _lived(home)
    assert kinds(out).count("noise") == 1


def test_a_thing_the_duck_pushed_hitting_another_thing_is_its_own_sound() -> None:
    """The grey case of the live check (2026-10-11): the duck's side against a slipper, a stumble,
    and 0.14 s later the slipper knocks something else. The duck pushed it: its own doing."""
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = float(home.d.time) - 0.2
    _knock(home, "books", "paper_cup")  # (the other thing's side of the hit: it is struck by it)
    home.run(2.0)
    assert _by(home) == ["self"]
    out, exp = _lived(home)
    assert kinds(out).count("noise") == 0 and exp.ignored == {"noise_self": 1}


def test_a_thing_hit_by_one_the_duck_moved_is_the_ducks_for_one_step_only() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = float(home.d.time) - 0.2
    labels = []
    # the cup (pushed) hits the books; the books hit the box (one step on, still the duck's);
    # the box hits a block (two steps on: not the duck's)
    for body, other in (
        ("paper_cup", "books"),
        ("books", "tissue_box"),
        ("tissue_box", "toy_block"),
    ):
        _knock(home, body, other)
        home.run(0.3)
        labels.append(_by(home)[-1])
    assert labels == ["self", "self", "thing"]


def test_two_things_the_duck_did_not_move_hitting_each_other_is_a_noise() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = float(home.d.time) - 30.0  # (long ago)
    _knock(home, "paper_cup", "books")
    home.run(2.0)
    assert _by(home) == ["thing"]
    out, _ = _lived(home)
    assert kinds(out).count("noise") == 1


def test_a_person_knocking_a_thing_the_duck_pushed_is_still_a_person() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part._duck_touched["thing:paper_cup"] = float(home.d.time) - 0.5
    _knock(home, "paper_cup", "person_owner_leg_l")
    home.run(2.0)
    assert _by(home) == ["person"]
    out, _ = _lived(home)
    assert kinds(out).count("noise") == 1


def test_a_thing_knocked_by_a_person_is_a_noise() -> None:
    home = Home(seed=3)
    home.run(1.0)
    _knock(home, "paper_cup", "person_owner_leg_l")
    home.run(2.0)
    assert _by(home) == ["person"]
    out, _ = _lived(home)
    assert kinds(out).count("noise") == 1


def test_a_thing_resting_against_the_duck_marks_it_for_the_next_two_seconds() -> None:
    """The world's own contact record (the one that tells touch) is what remembers the duck pushing
    a thing: nothing else is added to the physics."""
    home = Home(seed=2)
    home.part.stimulate({"stim": "roll_ball"})  # (a ball that rolls to it and rests against it)
    home.run(6.0)
    assert "thing:ball_hollow" in home.part._duck_touched


def test_a_page_bang_is_a_noise_and_a_startle_still_hears_it() -> None:
    home = Home(seed=3)
    home.run(1.0)
    home.part.stimulate({"stim": "bang"})
    home.run(2.0)
    assert "page" in _by(home) or [e.by for e in home.part.sounds if e.kind == "bang"] == ["page"]
    heard_by = {f.sound.by for f in home.frames if f.sound.kind == "bang"}
    assert heard_by == {"page"}  # (the sense still reports the sound: the reflexes read the kind)
    out, exp = _lived(home)
    assert [e["what"] for e in out if e["k"] == "noise"] == ["bang"] and exp.ignored == {}
