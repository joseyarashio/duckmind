"""neuro:motor — the repertoire: a parametric program for every behaviour in the contract.

Every behaviour plays; the same bout plays the same way (deterministic), another bout plays
differently (never the identical movement twice); the duck's style shows (its innate side, its
stage's amplitude, its branch's head carriage and look width, its quirk habit); reactive programs
follow the body (walking ↔ stopped) and their vocal and posture cues are legal.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from duckmind.neuro import repertoire as P
from duckmind.neuro.contract import (
    BEHAVIOURS,
    VOCAL_TAGS,
    BodySense,
    MotorCommand,
    SensorFrame,
    Target,
)


def _frame(t: int, posture: str = "sit", vx: float = 0.0) -> SensorFrame:
    return SensorFrame(t_ms=t, body=BodySense(posture=posture, vx=vx))


def _cmd(behaviour: str, t: int, variant: int = 1, **kw: object) -> MotorCommand:
    base = MotorCommand(
        t_ms=t,
        behaviour=behaviour,
        bout=1,
        variant=variant,
        intensity=0.6,
        tempo=1.0,
        amplitude=0.9,
        side=1,
        target=Target(kind="thing", bearing=0.3, range_m=0.5),
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def _play(
    behaviour: str,
    style: P.Style,
    *,
    variant: int = 1,
    seconds: float = 8.0,
    posture: str = "sit",
    walking: bool = False,
    **kw: object,
) -> list[P.ProgramOut]:
    state = None
    outs = []
    for k in range(int(seconds * 50)):
        t = k * 20
        vx = 0.45 if walking else 0.0
        cmd = _cmd(behaviour, t, variant, vx=vx, **kw)
        state, out = P.step(state, cmd, style, _frame(t, "walk" if walking else posture, vx), t)
        outs.append(out)
    return outs


def _trace(outs: list[P.ProgramOut]) -> list[tuple[float, ...]]:
    return [(o.pose.np, o.pose.hp, o.pose.hy, o.pose.hr, o.pose.mouth) for o in outs]


STYLE = P.style_from(
    {"boldness": 5000, "activity": 5000, "exploration": 5000}, stage="juvenile", side=1
)


def test_every_contract_behaviour_has_a_program() -> None:
    assert set(BEHAVIOURS) <= set(P.BUILDERS)


@pytest.mark.parametrize("behaviour", sorted(BEHAVIOURS))
def test_every_program_plays_legally(behaviour: str) -> None:
    spec = BEHAVIOURS[behaviour]
    walking = spec.locomotes
    outs = _play(
        behaviour, STYLE, posture="stand" if spec.posture == "stand" else "sit", walking=walking
    )
    for out in outs:
        pose = out.pose
        for value in (pose.np, pose.hp, pose.hy, pose.hr):
            assert math.isfinite(value) and abs(value) <= 1.35
        assert 0.0 <= pose.mouth <= 1.0
        assert 0.0 <= out.gaze <= 1.0 and 0.0 <= out.alive <= 1.0
        for tag, level in out.vocals:
            assert tag in VOCAL_TAGS and 0.0 < level <= 1.0
        assert out.posture in (None, "sit", "stand")
        if out.twist is not None:
            vx, vyaw = out.twist
            assert 0.30 <= vx <= 0.55 and abs(vyaw) <= 1.0
        assert out.label
    if spec.posture in ("sit", "stand"):
        assert any(out.posture == spec.posture for out in outs)


@pytest.mark.parametrize("behaviour", sorted(BEHAVIOURS))
def test_same_bout_same_movement_new_bout_new_movement(behaviour: str) -> None:
    walking = BEHAVIOURS[behaviour].locomotes
    first = _trace(_play(behaviour, STYLE, variant=11, seconds=4.0, walking=walking))
    again = _trace(_play(behaviour, STYLE, variant=11, seconds=4.0, walking=walking))
    other = _trace(_play(behaviour, STYLE, variant=12, seconds=4.0, walking=walking))
    assert first == again
    if behaviour == "still":  # a still hold: only its length differs
        a = P.start(_cmd("still", 0, 11), STYLE, _frame(0), 0).score.length
        b = P.start(_cmd("still", 0, 12), STYLE, _frame(0), 0).score.length
        assert a != b
        return
    assert first != other


def test_units_repeat_until_the_brain_switches_and_each_unit_differs() -> None:
    outs = _play("look_around", STYLE, seconds=30.0)
    units = sorted({o.unit for o in outs})
    assert len(units) >= 3  # the bout went on: new units, not a timer ending it
    starts = [i for i, o in enumerate(outs) if o.unit_done]
    segments = [_trace(outs[a : a + 40]) for a in starts[:3]]
    assert len({tuple(s) for s in segments}) == len(segments)


def test_perk_tilt_goes_to_the_innate_side() -> None:
    for side in (1, -1):
        style = replace(STYLE, side=side)
        hits = 0
        for variant in range(40):
            outs = _play("perk_tilt", style, variant=variant, seconds=2.0)
            peak = max(outs, key=lambda o: abs(o.pose.hr)).pose.hr
            hits += int(peak * side > 0)
        assert hits >= 30  # its innate side, almost always (Duck Core quirk tilt_side)


def _mean_abs_yaw(style: P.Style, behaviour: str = "look_around") -> float:
    total, n = 0.0, 0
    for variant in range(25):
        for out in _play(behaviour, style, variant=variant, seconds=4.0):
            total += abs(out.pose.hy)
            n += 1
    return total / n


def test_a_hatchling_moves_smaller_than_an_adult() -> None:
    small = _mean_abs_yaw(replace(STYLE, stage="hatchling"))
    big = _mean_abs_yaw(replace(STYLE, stage="adult"))
    assert small < 0.8 * big


def test_branches_show_in_the_head() -> None:
    explorer = P.style_from({"exploration": 7000}, stage="adult", branch="explorer")
    snuggler = P.style_from({"exploration": 7000}, stage="adult", branch="snuggler")
    songbird = P.style_from({}, stage="adult", branch="songbird")
    assert _mean_abs_yaw(explorer) > 1.1 * _mean_abs_yaw(snuggler)  # 轉頭幅度大 vs 低低的
    assert snuggler.carriage.np > 0.1 > songbird.carriage.np  # low vs upright carriage
    assert songbird.carriage.hp > 0.0
    assert snuggler.hold_scale > explorer.hold_scale


def test_quirk_double_look_looks_again() -> None:
    def turns(style: P.Style) -> int:
        count = 0
        for variant in range(30):
            outs = _play("look_around", style, variant=variant, seconds=6.0)
            signs = [1 if o.pose.hy > 0.15 else -1 if o.pose.hy < -0.15 else 0 for o in outs]
            signs = [s for s in signs if s]
            count += sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)
        return count

    plain = replace(STYLE, habit="")
    quirky = replace(STYLE, habit="double_look")
    assert turns(quirky) > turns(plain)


def test_reactive_walk_programs_follow_the_body() -> None:
    state = None
    labels = []
    for k in range(400):
        t = k * 20
        moving = (k // 100) % 2 == 0
        vx = 0.45 if moving else 0.0
        cmd = _cmd("forage", t, vx=vx)
        frame = _frame(t, "walk" if moving else "stand", vx)
        state, out = P.step(state, cmd, STYLE, frame, t)
        labels.append((moving, out.label))
    walking_labels = {label for moving, label in labels if moving}
    stopped_labels = {label for moving, label in labels if not moving}
    assert "邊走邊找" in walking_labels
    assert "啄一啄" in stopped_labels


def test_stand_up_and_sit_down_cue_the_posture_after_the_head() -> None:
    outs = _play("stand_up", STYLE, seconds=3.0)
    cue = next(i for i, o in enumerate(outs) if o.posture == "stand")
    assert 5 <= cue <= 40  # after its head's preparation (100–800 ms)
    outs = _play("sit_down", STYLE, seconds=3.0, posture="stand")
    assert any(o.posture == "sit" for o in outs)


def test_sleep_programs_stay_low_and_quiet() -> None:
    outs = _play("nap", STYLE, seconds=12.0)
    late = outs[-100:]
    assert all(o.pose.np > 0.3 for o in late)  # head low (the nap pose)
    assert all(o.gaze == 0.0 and o.alive <= 0.2 for o in outs)


def test_zoomies_shape_the_walk_with_dashes_and_arcs() -> None:
    outs = _play("zoomies", STYLE, seconds=8.0, posture="stand", walking=True)
    twists = [o.twist for o in outs if o.twist is not None]
    assert any(vx >= 0.5 and abs(vyaw) < 0.3 for vx, vyaw in twists)  # a dash
    assert any(abs(vyaw) >= 0.7 and vx <= 0.40 for vx, vyaw in twists)  # a tight arc
    assert any(tag == "wheee" for o in outs for tag, _ in o.vocals)


def test_style_from_duck_core_fixed_point_axes() -> None:
    style = P.style_from(
        {"boldness": 2000, "activity": 8000},
        stage="duckling",
        side=-1,
        nature={"songbird": 4000, "explorer": 2000, "snuggler": 2000, "dreamer": 2000},
    )
    # the genome's own scale (genome._unit): 2000 → 0, 8000 → 1 — one diversity mapping
    assert style.boldness == pytest.approx(0.0) and style.activity == pytest.approx(1.0)
    assert style.side == -1 and style.stage == "duckling"
    assert style.w("songbird") == pytest.approx(0.4)
    assert style.amp_cap == pytest.approx(0.75)


def test_fixed_point_axes_land_on_the_genomes_scale() -> None:
    """ONE diversity mapping: a Duck Core fixed-point axis means the same 0..1 value to the motor
    side as to the brain's genome."""
    from duckmind.neuro import signals as S
    from duckmind.neuro.genome import Genome

    genome = Genome.from_seed(2026, gd=90)
    for axis, raw in genome.axes_raw.items():
        assert S.norm_axis(raw) == pytest.approx(genome.axes[axis], abs=1e-9)
