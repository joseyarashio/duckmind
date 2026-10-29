"""dock-home (2026-10-17): a duck walking home by a dock frame it carried too far looks for its dock and docks.

The post-fix seed-1001 run of the charge-cycle control walked `return_dock` for 268 s in nine bouts and the
docking program never took the legs. The only fix it had was a tag read 52 m of walking ago (past
``remembered_max_m``), so the lane point it could have engaged at was not trusted, and the walk went straight at
the nest's place in its memory: beside the dock, along the wall, where the tag cannot be read (it is read from
within 60 degrees of its face). The head sweep that looks for the nest ran only with no goal at all. Now such a
frame points the walk (and the eyes) to the lane point, and at that point, with no nest in view, the duck looks for
it; the legs are still taken only by a tag read or by a frame it trusts.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import replace
from pathlib import Path

import pytest
from test_neuro_dock import _frame, _read_from

from duckmind.neuro import brain as B
from duckmind.neuro import dock as D
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import preset

NEST_AT_LANE_M = B.NEST_AT_LANE_M

# the frame: 1.5 m out from the wall, 0.3 m to the dock's right of the axis, facing the wall (heading pi), taken at odometry 0
FIX = D.DockPose(1500.0, -300.0, math.pi)
ODOM0 = (0.0, 0.0, 0.0)
TAG = D.DockGeometry().face_mm


def _stale() -> D.DockRun:
    run = D.DockRun()
    run.tracker.set(FIX, ODOM0)
    run.fix_src = "tag"
    run.walked_mm = 1000.0 * (
        run.cfg.remembered_max_m + 7.0
    )  # 32 m since the tag read (the 1001 run: 32-52 m)
    return run


# ── the docking program ─────────────────────────────────────────────────────────────────────────────


def test_a_frame_carried_too_far_does_not_take_the_legs() -> None:
    run = _stale()
    assert not run.trusted()
    assert run.rendezvous(ODOM0) is None and run.dock_bearing(ODOM0) is None
    # at the lane point itself, the very place the remembered route engages at: no engagement without a trusted frame or a tag
    assert run.can_engage(D.DockIn(t_ms=0, odom=ODOM0)) is None


def test_it_still_points_the_walk_by_that_frame() -> None:
    run = _stale()
    bearing, range_m = run.rendezvous(ODOM0, stale_ok=True) or (None, None)
    assert range_m == pytest.approx(math.hypot(1500.0 - FIX.x, run.cfg.lane_y() - FIX.y) / 1000.0)
    assert bearing is not None
    toward = run.dock_bearing(ODOM0, stale_ok=True)
    assert toward is not None and toward[1] == pytest.approx(
        math.hypot(TAG - FIX.x, FIX.y) / 1000.0
    )
    # but no frame at all is no frame (a pick-up drops it)
    gone = D.DockRun()
    assert (
        gone.rendezvous(ODOM0, stale_ok=True) is None
        and gone.dock_bearing(ODOM0, stale_ok=True) is None
    )


def test_a_tag_read_gives_the_frame_back() -> None:
    run = _stale()
    seen = D.DockPose(2000.0, 200.0, math.pi - 0.1)
    din = D.DockIn(t_ms=100, odom=ODOM0, tag=replace(_read_from(seen, ODOM0), t_ms=100))
    assert run.can_engage(din) == "tag"
    run.step(
        din
    )  # idle: a reading within remember_m is taken as the new frame (odometry carries it)
    assert run.trusted() and run.walked_mm == 0.0


# ── the brain ───────────────────────────────────────────────────────────────────────────────────────


def _home_brain(
    odom: tuple[float, float, float], nest_in_memory: tuple[float, float]
) -> NeuroBrain:
    brain = NeuroBrain(preset("snuggler"), seed=6)
    brain.dock.tracker.set(FIX, ODOM0)
    brain.dock.fix_src = "tag"
    brain.dock.walked_mm = 32_000.0
    brain.learn.dock = nest_in_memory  # where it last charged, in odometry (it is the dock's tag, 1.39 m on and 0.3 m across)
    brain.drives.hunger = 0.9
    brain.sel.current = brain.sel.index("return_dock")
    brain.behaviour = "return_dock"
    return brain


def _walk_home(brain: NeuroBrain, odom: tuple[float, float, float], steps: int) -> list:  # type: ignore[type-arg]
    out = []
    for k in range(steps):
        brain.sel.current = brain.sel.index("return_dock")
        brain.behaviour = "return_dock"
        cmd, _ = brain.step(20 * k, _frame(20 * k, battery=0.34, odom=odom))
        out.append(cmd)
    return out


def test_walking_home_by_a_frame_carried_too_far_it_goes_to_the_lane_point_not_beside_the_nest() -> (
    None
):
    # 2.3 m from the dock's face, out to its side: the nest's place (memory) is 1.9 m ahead-left, the lane point is elsewhere
    start = D.DockPose(600.0, 1800.0, math.pi)
    nest = (
        (start.x - TAG) / 1000.0,
        start.y / 1000.0,
    )  # the tag in odometry at fix = start, heading pi: odom = -(dock - fix)
    brain = _home_brain(ODOM0, nest)
    brain.dock.tracker.set(start, ODOM0)
    _walk_home(brain, ODOM0, 5)
    goal, goal_range, _ = brain._dbg_goal
    lane = brain.dock.rendezvous(ODOM0, stale_ok=True)
    assert lane is not None and goal is not None and goal_range is not None
    assert goal == pytest.approx(lane[0], abs=1e-6) and goal_range == pytest.approx(
        lane[1], abs=1e-6
    )
    straight_at_nest = math.atan2(nest[1], nest[0])
    assert (
        abs(goal - straight_at_nest) > 0.4
    )  # the old walk: straight at the nest's place from the side


def test_at_the_lane_point_with_no_nest_in_view_it_looks_for_its_nest() -> None:
    nest = (
        (FIX.x - TAG) / 1000.0,
        FIX.y / 1000.0,
    )  # the tag, 1.39 m ahead of the duck (odometry), 0.3 m to its side
    brain = _home_brain(ODOM0, nest)
    lane = brain.dock.rendezvous(ODOM0, stale_ok=True)
    assert (
        lane is not None and lane[1] < NEST_AT_LANE_M
    )  # it stands at the lane point (by its belief)
    cmds = _walk_home(
        brain, ODOM0, 120
    )  # 2.4 s: the first sweep (SEEK_SWEEP) is in the first 1.1 s
    assert brain._seek_t0 is not None
    assert any(abs(c.gaze.bearing - 0.95) < 0.05 for c in cmds), [
        round(c.gaze.bearing, 2) for c in cmds[::20]
    ]
    assert (
        brain.dock.mode == "idle" and brain.dock.tries == 0
    )  # looking is not docking: no tag, no legs


def test_far_from_the_lane_point_it_walks_and_does_not_sweep() -> None:
    far = D.DockPose(600.0, 1800.0, math.pi)
    brain = _home_brain(ODOM0, ((far.x - TAG) / 1000.0, far.y / 1000.0))
    brain.dock.tracker.set(far, ODOM0)
    _walk_home(brain, ODOM0, 120)
    assert brain._seek_t0 is None


# ── the whole duck, headless (needs MuJoCo and the official MJCF) ───────────────────────────────────

_MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")
AXIS = 1.571
DOCK_AT = (0.669, -2.78)


def _need_headless() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    if not (_MJCF and (Path(_MJCF) / "robot_allcollisions.xml").is_file()):
        pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)")


def _dock_frame(x: float, y: float, yaw: float) -> tuple[float, float, float]:
    c, s = math.cos(AXIS), math.sin(AXIS)
    ax, ay = x - DOCK_AT[0], y - DOCK_AT[1]
    return ax * c + ay * s, -ax * s + ay * c, (yaw - AXIS + math.pi) % (2 * math.pi) - math.pi


def _headless_home(seed: int, x: float, y: float, yaw: float, seconds: float) -> dict[str, object]:
    """The whole neural duck in real_home, 1.5-4 m from its dock, told to go home; its dock frame is the one a duck has
    after a long walk: a tag read 32 m of walking ago, carried by an odometry with a 0.3 m / 0.05 rad error, and the place of
    its nest in memory with 0.3 m of error. Truth only scores."""
    from datetime import datetime

    from duckmind.world import neuro_play_rig as R

    rng = random.Random(seed * 31 + 7)
    seen: dict[str, object] = {"engaged": None, "docked": None, "why": None}

    def each(rig: R.PlayNeuro) -> bool:
        d = rig.brain.dock
        if d.tries and seen["engaged"] is None:
            seen["engaged"] = rig.frame.t_ms / 1000.0
            seen["why"] = next((e for e in d.log if e.startswith("engage:")), None)
        if d.mode == "home":
            seen["docked"] = rig.frame.t_ms / 1000.0
            return True
        return False

    with R.PlayNeuro(
        preset("explorer"), seed, x=x, y=y, yaw=yaw, home="real_home", variant="backin",
        when=datetime(2026, 10, 16, 10, 0),
    ) as rig:  # fmt: skip
        rig.step()
        odom = rig.frame.body.odom
        tx, ty, tyaw = rig.pose()
        a, b, hr = _dock_frame(tx, ty, tyaw)
        brain = rig.brain
        brain.learn.dock = (
            DOCK_AT[0] - tx + odom[0] + rng.gauss(0, 0.3),
            DOCK_AT[1] - ty + odom[1] + rng.gauss(0, 0.3),
        )
        here = D.DockPose(
            a * 1000.0 + 172.0 + rng.gauss(0, 300.0),
            b * 1000.0 + rng.gauss(0, 300.0),
            D.wrap(hr + rng.gauss(0, 0.05)),
        )
        brain.dock.tracker.set(here, (odom[0] * 1000.0, odom[1] * 1000.0, odom[2]))
        brain.dock.fix_ms, brain.dock.fix_src, brain.dock.walked_mm = 0, "tag", 32_000.0
        brain.sel.set_extra({"return_dock": 40.0})
        rig.run(seconds, each)
    return seen


# two of the 120 starts of the batch that never engaged before the change (a tag read from the side, beside the dock)
HEADLESS_STARTS = [(4, 0.761, -1.006, 1.44), (71, 1.093, -0.519, 0.485)]


@pytest.mark.parametrize("start", HEADLESS_STARTS)
def test_headless_it_finds_its_dock_and_docks_from_a_frame_carried_too_far(
    start: tuple[int, float, float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    _need_headless()
    monkeypatch.setenv("DUCKMIND_BODY_COMMAND_DELAY_MS", "20")
    seen = _headless_home(*start, 150.0)
    assert seen["engaged"] is not None, seen  # the program took the legs (before: never, in 300 s)
    assert seen["why"] == "engage:tag", (
        seen
    )  # by a tag it read itself, not by the frame it could not trust
    assert seen["docked"] is not None, seen
