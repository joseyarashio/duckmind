"""The last seat at the dock (docs/design/neural-duck/DOCK-FALL-2026-10-11.md §7.7).

A duck that reversed onto the dock's face used to walk out a fixed 0.7 s before its first sit. The legs
answer that command slowly and coast on after it, so the first sit began a median 107 mm out (74-144 mm,
headless, 107 approaches) where one sit seats it from 30-90 mm; each sit slides the rear ~72 mm back, each
stand pushes it ~48 mm forward, and three of those ended a few centimetres short of the contacts (the
live 1001 run: sits at +76 / +49 / +28 mm). Now it walks out by its odometry from the face stop, just far
enough, and goes on sitting while each sit-and-stand still takes the rear in.

The first group runs the docking program on the stand-in body of ``test_neuro_dock.Kin`` with the numbers
measured headless (a walk command of 0.7 s moves the rear ~100 mm, a sit slides it 72 mm back and never
past the face, a stand pushes it 48 mm forward, the contacts close for a seat 0-22 mm from the face that
began 28 mm or more out of it). The second runs the whole neural duck on the headless MuJoCo duck in
``real_home`` (skipped without MuJoCo and the official MJCF).
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from duckmind.neuro import dock as D
from tests.test_neuro_dock import GEO, Kin, _read_from

FACE_X = float(GEO.face_contact_mm)  # the trunk when its rear meets the face (mm from the wall)
WALK_GAIN = (
    0.42  # how far a forward walk command goes (0.7 s of 0.35 m/s: ~58 mm + ~42 mm of coast)
)


def _engaged(config: D.DockConfig, **kin_kw: float) -> tuple[D.DockRun, Kin, D.DockOut]:
    """The program engaged on the lane, 1.4 m out, facing the dock; the stand-in's walk as asked."""
    run = D.DockRun(config)
    start = D.DockPose(1400.0, 0.0, math.pi)
    kin_kw.setdefault("gain", WALK_GAIN)
    kin = Kin(start.x, start.y, start.h, (3000.0, -1200.0, 0.7), **kin_kw)
    din = replace(kin.din(None), tag=replace(_read_from(start, kin.odom()), t_ms=0))
    run.engage(din)
    return run, kin, run.step(din)


def _seat_run(
    config: D.DockConfig, *, seconds: float = 150.0, **kin_kw: float
) -> tuple[D.DockRun, Kin]:
    """The J-hook from the lane to the end of the sits (docked, or the try failed)."""
    run, kin, out = _engaged(config, **kin_kw)
    for _ in range(int(seconds * 50)):
        kin.step(out)
        out = run.step(kin.din(out))
        if run.mode in ("home", "idle"):
            break
    return run, kin


def test_the_first_sit_begins_close_to_where_the_reverse_ended() -> None:
    """It no longer walks a fixed 0.7 s out: by its odometry, just far enough for the slide back."""
    firsts: list[float] = []
    for gain in (0.3, 0.42, 0.7, 1.0):  # the legs answer a walk command more or less
        run, kin, out = _engaged(D.DockConfig(), gain=gain)
        for _ in range(7500):
            kin.step(out)
            if out.phase == "sit" and kin.sits == 0 and kin.change_at is not None:
                firsts.append(kin.out_mm())
                break
            out = run.step(kin.din(out))
    assert len(firsts) == 4
    assert all(f <= 90.0 for f in firsts), firsts  # one sit seats from 30-90 mm out


def test_it_seats_on_the_first_sit_where_the_fixed_walk_ended_a_hand_too_far_out() -> None:
    """The 1001 pattern: the fixed walk carries on, the three sits end a few cm short of the contacts."""
    legacy = D.DockConfig(bump_out_mm=1000.0, max_sits_progress=3)
    run, kin = _seat_run(legacy, gain=0.8)
    assert run.result == "failed:tries" and not kin.pg() and kin.sits == 3
    assert kin.out_mm() > 22.0  # the last sit ended a few cm short of the contacts (62 mm here)
    run, kin = _seat_run(D.DockConfig(), gain=0.8)
    assert run.mode == "home" and run.result == "docked" and kin.pg() and kin.sits == 1


@pytest.mark.parametrize("gain", [0.3, 0.42, 0.7, 1.0])
def test_it_seats_whatever_the_legs_answer_to_the_walk_out(gain: float) -> None:
    run, kin = _seat_run(D.DockConfig(), gain=gain)
    assert run.mode == "home" and run.result == "docked" and kin.pg()
    assert kin.sits <= 2


def test_it_goes_on_sitting_while_each_sit_takes_the_rear_in() -> None:
    """The old walk-out kept, only the ladder changed: a fourth and fifth sit are earned by progress."""
    old_walk = D.DockConfig(bump_out_mm=1000.0)
    fixed = replace(old_walk, max_sits_progress=3)
    run, kin = _seat_run(fixed, gain=0.8)
    assert run.result == "failed:tries" and kin.sits == 3
    run, kin = _seat_run(old_walk, gain=0.8)
    assert run.mode == "home" and run.result == "docked" and kin.pg()
    assert kin.sits >= 4 and "more_sits" in run.log


def test_a_rear_that_does_not_move_in_gets_no_more_sits() -> None:
    """A sit that gains nothing (the stand pushes out as far as the sit slid in) ends the try at three."""
    old_walk = D.DockConfig(bump_out_mm=1000.0)
    run, kin = _seat_run(old_walk, gain=0.8, push_mm=72.0)
    assert run.result == "failed:tries" and kin.sits == 3 and "more_sits" not in run.log


def test_the_extra_sits_are_bounded() -> None:
    """Whatever the progress, never more than max_sits_progress sits an approach."""
    cfg = D.DockConfig(bump_out_mm=1000.0, max_sits_progress=5, seat_reach_mm=-1000.0)
    run, kin = _seat_run(cfg, gain=1.0, push_mm=40.0, slide_mm=60.0)
    assert kin.sits <= 5


def test_a_re_seat_from_the_dock_keeps_the_plain_three() -> None:
    """No face stop in this try (it starts seated in the dock): no reference, the old budget."""
    run = D.DockRun()
    kin = Kin(FACE_X + 60.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    kin.posture = kin.want = "sit"
    run.step(D.DockIn(t_ms=0, odom=kin.odom(), posture="sit", pg=True))
    kin.pg = lambda: False  # type: ignore[method-assign]  # (the contacts stay open)
    run.reseat(kin.din(None))
    out = run.step(kin.din(None))
    for _ in range(50 * 90):
        kin.step(out)
        out = run.step(kin.din(out))
        if run.mode == "idle":
            break
    assert run.result == "failed:tries" and "more_sits" not in run.log and kin.sits == 3


# ── the whole duck on the headless physics ──────────────────────────────────────────────────────────

_MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")
AXIS = 1.571  # real_home's dock (700, -2940) faces +y; the seated duck is at (0.669, -2.78)
SEAT = (0.669, -2.78)


def _need_headless() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    if not (_MJCF and (Path(_MJCF) / "robot_allcollisions.xml").is_file()):
        pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)")


def _out_of_the_seat_mm(x: float, y: float) -> float:
    return ((x - SEAT[0]) * math.cos(AXIS) + (y - SEAT[1]) * math.sin(AXIS)) * 1000.0


def _headless_dock(x: float, y: float, yaw: float, seconds: float, **cfg: float):  # type: ignore[no-untyped-def]
    """The whole neural duck (senses, brain, motor, body link) docks itself from (x, y, yaw), 1.0-1.3 m
    out; it is told to go home and that is all. Truth only scores. Returns what the program did."""
    from datetime import datetime

    from duckmind.neuro.genome import preset
    from duckmind.world import neuro_play_rig as R

    seen: dict[str, object] = {"results": [], "first_sit_mm": None, "docked_at": None, "seat": None}

    def each(rig: R.PlayNeuro) -> bool:
        d = rig.brain.dock
        if d.result and (not seen["results"] or seen["results"][-1] != d.result):  # type: ignore[index]
            seen["results"].append(d.result)  # type: ignore[union-attr]
        tx, ty, _ = rig.pose()
        if seen["first_sit_mm"] is None and d.phase == "sit" and rig.frame.body.posture == "stand":
            seen["first_sit_mm"] = _out_of_the_seat_mm(tx, ty)
        if d.mode == "home":
            seen["docked_at"] = rig.frame.t_ms / 1000.0
            seen["seat"] = _out_of_the_seat_mm(tx, ty)
            return True
        return False

    with R.PlayNeuro(
        preset("explorer"), 206, x=x, y=y, yaw=yaw, home="real_home", variant="backin",
        when=datetime(2026, 10, 16, 10, 0),
    ) as rig:  # fmt: skip
        tx, ty, _ = rig.pose()
        rig.brain.learn.dock = (SEAT[0] - tx, SEAT[1] - ty)  # where it believes its dock is
        rig.brain.sel.set_extra({"return_dock": 40.0})
        if cfg:
            rig.brain.dock.cfg = replace(rig.brain.dock.cfg, **cfg)
        rig.run(seconds, each)
        seen["sits"] = rig.brain.dock.sits
        seen["log"] = list(rig.brain.dock.log)
    return seen


# the two lane starts of the 30-start batch whose first approach ended 'failed:tries' with the fixed walk-out
LANE_STARTS = [(0.384, -1.545, -1.012), (0.808, -1.523, -1.670)]


@pytest.mark.parametrize("start", LANE_STARTS)
def test_headless_it_seats_without_a_failed_try(
    start: tuple[float, float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    _need_headless()
    monkeypatch.setenv("DUCKMIND_BODY_COMMAND_DELAY_MS", "20")
    seen = _headless_dock(*start, 240.0)
    assert seen["docked_at"] is not None, seen
    assert "failed:tries" not in seen["results"], seen  # type: ignore[operator]
    assert seen["first_sit_mm"] is not None and seen["first_sit_mm"] <= 95.0, seen  # type: ignore[operator]
    assert seen["seat"] is not None and seen["seat"] <= 30.0, seen  # type: ignore[operator]
