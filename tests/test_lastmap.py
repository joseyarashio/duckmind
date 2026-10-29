"""The lasting map (neuro/lastmap.py): what lays evidence down, what takes it away, how it is saved, what the plan reads from it."""

from __future__ import annotations

import math

from duckmind.neuro import clutter as CL
from duckmind.neuro import clutter_route as CR
from duckmind.neuro import lastmap as LM
from duckmind.neuro.contract import (
    BodySense,
    CameraSense,
    FreeSector,
    SensorFrame,
    TofSense,
)


def frame(
    t_ms: int,
    odom=(0.0, 0.0, 0.0),
    *,
    floor=None,
    tof=None,
    events=(),
    posture="stand",
) -> SensorFrame:
    cam = CameraSense(floor=tuple(floor or ()), floor_t_ms=t_ms if floor else None, usable=True)
    return SensorFrame(
        t_ms=t_ms,
        body=BodySense(posture=posture, odom=odom),
        camera=cam,
        tof=tof if tof is not None else TofSense(),
        events=tuple(events),
    )


def wall_look(free: float) -> list[FreeSector]:
    return [
        FreeSector(bearing=b, width=0.13, free_m=free, end="wall")
        for b in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]


def look_many(
    m: LM.LastMap, n: int, free: float, odom=(0.0, 0.0, 0.0), t0: int = 0, gap_ms: int = 250
) -> int:
    t = t0
    for _ in range(n):
        m.update(t, frame(t, odom, floor=wall_look(free)))
        t += gap_ms
    return t


def test_floor_line_makes_free_floor_and_a_wall_at_its_end():
    m = LM.LastMap()
    look_many(m, 6, 1.0)
    assert m.known_free(0.5, 0.0)
    assert m.occupied(1.0, 0.0)
    assert not m.occupied(0.5, 0.0)
    assert m.evidence_at(2.0, 0.0) == 0.0  # beyond: nothing known


def test_unknown_end_lays_no_wall():
    m = LM.LastMap()
    sectors = [FreeSector(bearing=0.0, width=0.13, free_m=0.9, end="unknown")]
    for k in range(8):
        m.update(k * 250, frame(k * 250, floor=sectors))
    assert m.known_free(0.5, 0.0)
    assert not m.occupied(0.9, 0.0)


def test_a_repeat_from_the_same_stand_counts_for_little_a_second_visit_counts():
    quick = LM.LastMap()
    look_many(quick, 3, 1.0, gap_ms=100)
    slow = LM.LastMap()
    look_many(slow, 3, 1.0, gap_ms=1500)
    assert quick.evidence_at(1.0, 0.0) < slow.evidence_at(1.0, 0.0)


def test_a_later_look_that_sees_clear_floor_takes_a_wall_away():
    m = LM.LastMap()
    t = look_many(m, 6, 1.0)
    assert m.occupied(1.0, 0.0)
    clear = [
        FreeSector(bearing=b, width=0.13, free_m=1.45, end="unknown")
        for b in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]
    for k in range(12):
        m.update(t + k * 250, frame(t + k * 250, floor=clear))
    assert not m.occupied(1.0, 0.0)  # it was a chair that moved


def test_tof_ahead_return_is_a_wall_and_free_floor_before_it():
    m = LM.LastMap(LM.LastMapConfig(tof_ahead_off_m=0.0))
    tof = TofSense(present=True, ahead_mm=600)
    for k in range(5):
        m.update(k * 70, frame(k * 70, tof=tof))
    assert m.occupied(0.6, 0.0)
    assert m.evidence_at(0.3, 0.0) < 0.0


def test_a_bump_is_a_thing_just_ahead_and_odometry_turns_it():
    m = LM.LastMap()
    m.update(0, frame(0, (1.0, 1.0, math.pi / 2), events=("blocked",)))
    m.update(20, frame(20, (1.0, 1.0, math.pi / 2), events=("blocked",)))
    assert m.occupied(1.0, 1.0 + LM.BUMP_AHEAD_M)


def test_nothing_is_written_while_lost_or_sitting_or_off():
    m = LM.LastMap()
    m.update(0, frame(0, posture="sit", floor=wall_look(1.0)))
    assert not m.cells
    m.update(10, frame(10, events=("lifted",)))
    assert m.lost
    m.update(20, frame(20, floor=wall_look(1.0)))
    assert not m.cells
    off = LM.LastMap(LM.LastMapConfig(enabled=False))
    off.update(0, frame(0, floor=wall_look(1.0)))
    assert not off.cells


def test_evidence_is_forgotten_over_hours_not_seconds():
    m = LM.LastMap()
    t = look_many(m, 10, 1.0, gap_ms=1200)  # (looks a second apart are separate witnesses)
    assert m.occupied(1.0, 0.0)
    assert t > 0
    # (the map's own clock: seconds since its first update) a minute later, an hour later: still a wall
    m._sweep(60.0)
    assert m.occupied(1.0, 0.0)
    m._sweep(3600.0)
    assert m.occupied(1.0, 0.0)
    # many hours later, never looked at again: forgotten
    m._sweep(30 * 3600.0)
    assert not m.occupied(1.0, 0.0)


def test_state_roundtrip_keeps_cells_and_ages_them_by_the_time_down():
    m = LM.LastMap()
    look_many(m, 8, 1.0)
    state = m.to_state()
    assert state["v"] == 1 and state["cells"]
    again = LM.LastMap()
    again.load_state(state, 0.0, down_s=10.0)
    assert again.occupied(1.0, 0.0)
    assert again.known_free(0.5, 0.0)
    assert len(again.cells) >= len(m.cells) * 0.9
    # another cell size is not taken
    other = LM.LastMap()
    bad = dict(state)
    bad["cell"] = 0.25
    other.load_state(bad)
    assert not other.cells


def test_the_correction_moves_where_a_look_is_laid_down():
    m = LM.LastMap()
    m.corr = (1.0, 0.0, 0.0)  # the odometry is 1 m behind the map's x
    look_many(m, 6, 1.0)
    assert m.occupied(1.0, 0.0)  # (odometry frame query: the wall is at odometry x = 1)
    mx, _my = m.to_map(1.0, 0.0)
    assert abs(mx - 2.0) < 1e-9
    assert any(
        abs(i - 20) <= 1 and j in (-1, 0) for i, j in m.cells
    )  # the map's own cell for x = 2.0


def wall_map(y: float = 0.5, gap: tuple[float, float] | None = None) -> LM.LastMap:
    """A map with a long wall along y = ``y`` (a gap in x if given) and clear floor below it."""
    m = LM.LastMap()
    x = -2.0
    while x <= 2.0:
        if gap is None or not (gap[0] <= x <= gap[1]):
            for dy in (0.0, 0.1):
                m._add(x, y + dy, 4.0, 0.0)
        x += 0.1
    for xi in range(-30, 31):
        for yi in range(-10, int(y * 10)):
            m._add(xi * 0.1 + 0.05, yi * 0.1 + 0.05, -2.0, 0.0)
    return m


def test_line_clear_sees_a_wall_across_the_way():
    m = wall_map()
    assert not m.line_clear((0.0, 0.0), (0.0, 1.5))
    assert m.line_clear((0.0, 0.0), (1.0, 0.0))
    assert m.line_clear(
        (0.0, 0.0), (0.0, 1.5), skip_end_m=1.2
    )  # (the last stretch is not looked at)
    gap = wall_map(gap=(0.8, 1.4))
    assert gap.line_clear((1.1, 0.0), (1.1, 1.5))


def test_maps_block_near_a_wall_cost_more_beside_it_and_leave_the_gap():
    m = wall_map(gap=(0.8, 1.4))
    mp = m.maps((0.0, 0.0), (0.0, 0.5), block_r=0.16)
    i, j = mp.cell(0.0, 0.5)
    assert mp.blocked[i * CR.N + j]
    i2, j2 = mp.cell(1.1, 0.5)
    assert not mp.blocked[i2 * CR.N + j2]  # the gap's middle
    i3, j3 = mp.cell(0.0, 0.0)
    assert mp.cost[i3 * CR.N + j3] <= 1.0 + 1e-9  # known clear, far from the wall
    i4, j4 = mp.cell(0.0, 0.35)
    assert mp.cost[i4 * CR.N + j4] > 1.5  # beside the wall: the ring costs
    i5, j5 = mp.cell(0.0, 1.5)
    assert mp.cost[i5 * CR.N + j5] == CR.UNKNOWN_COST  # never looked at


def test_a_lone_weak_cell_is_not_a_thing_for_the_plan():
    m = LM.LastMap()
    m._add(1.0, 0.0, 1.5, 0.0)  # one stray return
    assert m.occupied(1.0, 0.0)
    assert not m.strong_cells()
    assert m.line_clear((0.0, 0.0), (2.0, 0.0))
    m._add(1.0, 0.0, 1.5, 5.0)  # seen again, later (a second witness)
    assert m.strong_cells()
    assert not m.line_clear((0.0, 0.0), (2.0, 0.0))


def test_search_goes_through_the_gap_not_the_wall():
    m = wall_map(gap=(0.8, 1.4))
    mp = m.maps((0.0, 0.0), (0.5, 1.0), block_r=0.16, goal=(0.0, 1.5))
    route = CR.search(mp, (0.0, 0.0), (0.0, 0.0), 1.0, (0.0, 1.5))
    assert route is not None and route.complete
    xs = [p[0] for p in route.points]
    assert max(xs) > 0.8  # it went round by the gap


class NoBound:
    def ahead(self, odom):
        return 1.5

    def clearance(self, odom, h):
        return 1.5

    def clearances(self, odom, headings):
        return [1.5 for _ in headings]

    def snapshot(self, centre, radius=3.4):
        return [], []

    trail: list = []


def map_cfg(**kw):
    return CL.ClutterConfig(enabled=True, action="route", use_map=True, map_goal=True, **kw)


def test_a_goal_behind_a_wall_begins_a_map_route_after_a_moment_and_ends_when_the_way_is_open():
    m = wall_map(gap=(0.8, 1.4))
    c = CL.Clutter(map_cfg())
    bound = NoBound()
    odom = (0.0, 0.0, math.pi / 2)  # facing the wall (+y)
    goal = (0.0, 1.5)  # bearing 0, 1.5 m: straight through the wall
    t = 0
    began = None
    for _ in range(80):
        d = c.step(t, odom, active=True, want=True, bound=bound, goal=goal, lmap=m)
        if c.active and began is None:
            began = t
        t += 100
    assert began is not None and began >= 1500  # not at once: the line has to stay shut
    assert c.why == "mapgoal" and c.map_started == 1
    assert c.route is not None and c.route.complete
    assert d is not None  # the legs are the escape's
    # the straight way opens (a map with no wall): the escape ends after a moment
    clear = LM.LastMap()
    for _ in range(40):
        c.step(t, odom, active=True, want=True, bound=bound, goal=goal, lmap=clear)
        t += 100
    assert not c.active and c.finished == 1


def test_no_map_route_without_the_setting_or_without_a_goal():
    m = wall_map(gap=(0.8, 1.4))
    off = CL.Clutter(CL.ClutterConfig(enabled=True, action="route"))
    odom = (0.0, 0.0, math.pi / 2)
    for k in range(60):
        off.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=(0.0, 1.5), lmap=m)
    assert not off.active
    on = CL.Clutter(map_cfg())
    for k in range(60):
        on.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=None, lmap=m)
    assert not on.active
    near = CL.Clutter(map_cfg())
    for k in range(60):  # a goal closer than the beak's own business: not the map's
        near.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=(0.0, 0.5), lmap=m)
    assert not near.active


def test_a_unicycle_following_the_map_route_gets_through_the_gap_without_flipping():
    m = wall_map(gap=(0.8, 1.4))
    c = CL.Clutter(map_cfg())
    x, y, yaw = 0.0, 0.0, math.pi / 2
    t = 0
    for _ in range(1200):
        d = c.step(
            t,
            (x, y, yaw),
            active=True,
            want=True,
            bound=NoBound(),
            goal=_goal_of((x, y, yaw), (0.0, 2.0)),
            lmap=m,
        )
        if d is not None:
            dt = 0.1
            yaw += d.vyaw * dt
            x += d.vx * math.cos(yaw) * dt
            y += d.vx * math.sin(yaw) * dt
        t += 100
        if y > 0.7 and abs(x - 1.1) < 0.6:
            break
    assert y > 0.7, (x, y)
    assert c.replans <= 4
    assert c.map_started == 1


def _goal_of(odom, world):
    dx, dy = world[0] - odom[0], world[1] - odom[1]
    return (math.atan2(dy, dx) - odom[2] + math.pi) % (2 * math.pi) - math.pi, math.hypot(dx, dy)


def test_the_dock_fixes_the_odometry_it_drifted_from():
    m = LM.LastMap()
    # first sight of the dock: where it is in the map (the odometry is believed)
    assert m.anchor_dock((1.0, 0.0, 0.0), (2.0, 0.0, math.pi)) == "set"
    assert m.dock_pose is not None
    assert abs(m.dock_pose[0] - 3.0) < 1e-9
    # later the odometry says 0.2 m / 0.1 m / 0.05 rad off: the dock says where the duck really is
    assert m.anchor_dock((1.2, 0.1, 0.05), (2.0, 0.0, math.pi), gain=1.0) == "fix"
    x, y, th = m.pose((1.2, 0.1, 0.05))
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9 and abs(th) < 1e-9
    # a fix that would move it by metres is a bad read, not a drift
    assert m.anchor_dock((1.2, 0.1, 0.05), (5.0, 3.0, math.pi)) == "skip"


def test_a_map_laid_down_with_the_correction_stays_where_the_dock_says():
    m = LM.LastMap()
    m.anchor_dock((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    look_many(m, 6, 1.0, gap_ms=1200)  # a wall 1 m ahead of the (believed) origin
    m.anchor_dock(
        (0.3, 0.0, 0.0), (1.0, 0.0, 0.0), gain=1.0
    )  # the odometry crept 0.3 m: the correction takes it out
    assert abs(m.corr[0] + 0.3) < 1e-6
    look_many(
        m, 6, 1.0, odom=(0.3, 0.0, 0.0), t0=20000, gap_ms=1200
    )  # the same wall seen from the drifted odometry
    wall_x = [i for i, j in m.cells if m.cells[(i, j)][0] >= LM.OCC_AT and j in (-1, 0)]
    assert max(wall_x) - min(wall_x) <= 2  # one wall (not two 0.3 m apart)


def test_a_lost_duck_takes_its_place_back_from_the_dock():
    m = LM.LastMap()
    m.anchor_dock((0.0, 0.0, 0.0), (0.5, 0.0, math.pi))
    m.update(0, frame(0, events=("lifted",)))
    assert m.lost
    assert m.anchor_dock((5.0, 5.0, 1.0), (0.5, 0.0, math.pi)) == "relocated"
    assert not m.lost
    assert abs(m.pose((5.0, 5.0, 1.0))[0]) < 1e-6


def test_the_dock_pose_is_saved_and_a_snapshot_without_the_map_still_loads():
    m = LM.LastMap()
    m.anchor_dock((0.0, 0.0, 0.0), (0.5, 0.0, math.pi))
    look_many(m, 4, 1.0, gap_ms=1200)
    again = LM.LastMap()
    again.load_state(m.to_state())
    assert again.dock_pose == tuple(round(v, 5) for v in m.dock_pose)
    again.load_state({})  # (nothing saved: an empty map)
    assert not again.cells


class SnapBrain:
    """The parts of a NeuroBrain the snapshot touches (as tests/test_life_snapshot.py), with a lasting map."""

    def __init__(self) -> None:
        from duckmind.neuro.drives import Drives
        from duckmind.neuro.genome import Genome
        from duckmind.neuro.habituation import Habituation
        from duckmind.neuro.learn import Learner

        p = Genome.from_seed(4107, gd=0).p
        self.learn = Learner(p)
        self.hab = Habituation(p)
        self.drives = Drives()
        self.lastmap = LM.LastMap()
        self._t0, self._t_prev = 0, 1000


def test_the_map_goes_into_the_snapshot_and_comes_back_in_the_same_home_only():
    from datetime import datetime, timezone

    from duckmind.life import snapshot as snap

    brain = SnapBrain()
    look_many(brain.lastmap, 8, 1.0, gap_ms=1200)
    ctx = {"home": "real_home", "start": "dock"}
    data = snap.capture(
        brain,
        name="蛋黃",
        seed=4107,
        gd=0,
        calendar=datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc),
        context=ctx,
        why="test",
        wall=1000.0,
    )
    assert data["map"]["cells"] and data["version"] == snap.SNAPSHOT_VERSION
    fresh = SnapBrain()
    report = snap.apply(fresh, data, seed=4107, context=ctx, wall=1000.0)
    assert report["map_cells"] > 0 and fresh.lastmap.occupied(1.0, 0.0)
    other = SnapBrain()
    report = snap.apply(
        other, data, seed=4107, context={"home": "apartment", "start": "dock"}, wall=1000.0
    )
    assert report["map_cells"] == 0 and "map" in report["skipped"]


def test_a_snapshot_without_a_map_loads_and_a_duck_without_a_map_saves_none():
    from datetime import datetime, timezone

    from duckmind.life import snapshot as snap

    brain = SnapBrain()
    brain.lastmap = LM.LastMap(LM.LastMapConfig(enabled=False))
    ctx = {"home": "real_home", "start": "dock"}
    data = snap.capture(
        brain,
        name="蛋黃",
        seed=4107,
        gd=0,
        calendar=datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc),
        context=ctx,
        why="test",
        wall=1000.0,
    )
    assert "map" not in data
    fresh = SnapBrain()
    assert snap.apply(fresh, data, seed=4107, context=ctx, wall=1000.0)["map_cells"] == 0


def test_the_brain_builds_the_map_in_every_mode_and_a_dock_fix_reaches_it():
    from duckmind.neuro.brain import NeuroBrain
    from duckmind.neuro.dock import DockPose
    from duckmind.neuro.genome import preset

    genome = preset("explorer")
    assert NeuroBrain(genome, 1, lastmap="off").lastmap.cfg.enabled is False
    assert NeuroBrain(genome, 1).lastmap.cfg.enabled is (LM.mode_of(None) != "off")
    assert NeuroBrain(genome, 1, lastmap="obs").clutter.cfg.enabled is CL.CLUTTER_ENABLED
    assert NeuroBrain(genome, 1, lastmap="wp").clutter.cfg.action == "waypoint"
    assert NeuroBrain(genome, 1, lastmap="act").clutter.cfg.action == "route"
    assert (
        NeuroBrain(genome, 1, lastmap="act").clutter.cfg.map_goal is False
    )  # (out of a knot only)
    assert NeuroBrain(genome, 1, lastmap="actgoal").clutter.cfg.map_goal is True
    assert NeuroBrain(genome, 1, lastmap="actgoal").clutter.cfg.action == "route"
    for mode in ("obs", "act", "actgoal", "wp"):
        brain = NeuroBrain(genome, 1, lastmap=mode)
        t = 1000
        for k in range(120):
            fr = frame(t, (0.003 * k, 0.0, 0.0), floor=wall_look(1.0) if k % 10 == 0 else None)
            if k == 60:  # a tag read: the dock tracker holds a fix
                brain.dock.tracker.set(DockPose(800.0, 0.0, math.pi), (0.18, 0.0, 0.0))
            brain.step(t, fr)
            t += 20
        assert brain.lastmap.cells, mode
        assert brain.lastmap.dock_pose is not None and brain.lastmap.anchors == 1, mode


def test_building_the_map_changes_nothing_the_duck_does_or_learns():
    """obs mode = the map is built and saved, nothing reads it: the brain's commands, its learner and its drives are the same as with no map."""
    from duckmind.neuro.brain import NeuroBrain
    from duckmind.neuro.genome import preset

    genome = preset("explorer")
    a = NeuroBrain(genome, 7, lastmap="off")
    b = NeuroBrain(genome, 7, lastmap="obs")
    t = 1000
    for k in range(600):
        odom = (0.003 * k, 0.0005 * k, 0.001 * k)
        tof = TofSense(
            present=True, left_mm=500, ahead_mm=900 - (k % 300), right_mm=700, touch=(k % 97 == 0)
        )
        events = ("blocked",) if k % 211 == 5 else ()
        fr = frame(
            t,
            odom,
            floor=wall_look(0.8 + 0.001 * (k % 200)) if k % 12 == 0 else None,
            tof=tof,
            events=events,
        )
        ca, _ta = a.step(t, fr)
        cb, _tb = b.step(t, fr)
        assert ca == cb, k
        t += 20
    assert b.lastmap.cells and not a.lastmap.cells
    assert a.learn.to_state(10.0) == b.learn.to_state(10.0)
    assert a.drives.to_state() == b.drives.to_state()


def test_the_map_goal_trigger_can_wait_for_a_stall_or_for_contacts():
    m = wall_map(gap=(0.8, 1.4))
    odom = (0.0, 0.0, math.pi / 2)
    goal = (0.0, 1.5)
    # asked for a stall (20 s without coming nearer): the shut line alone does not begin it
    c = CL.Clutter(map_cfg(map_goal_stall_s=20.0))
    for k in range(150):
        c.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=goal, lmap=m)
    assert not c.active  # (20 s not yet gone)
    for k in range(150, 400):
        c.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=goal, lmap=m)
    assert c.why == "mapgoal"  # (not moving for 20 s: a stall)
    # a clear line clears the history (and nothing breaks without a stall setting)
    clear = LM.LastMap()
    c2 = CL.Clutter(map_cfg(map_goal_stall_s=20.0))
    for k in range(50):
        c2.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=goal, lmap=clear)
    assert not c2.active and not c2._rng_hist
    # asked for contacts: none, no route; two, a route
    c3 = CL.Clutter(map_cfg(map_goal_events=2))
    for k in range(60):
        c3.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=goal, lmap=m)
    assert not c3.active
    c3.note(6000, 0.0, 0.0, "bump")
    c3.note(8000, 0.0, 0.0, "bump")
    for k in range(60, 120):
        c3.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=goal, lmap=m)
    assert c3.why == "mapgoal"


def test_a_quiet_clutter_keeps_no_events_and_takes_no_legs():
    m = wall_map(gap=(0.8, 1.4))
    c = CL.Clutter(map_cfg())
    c.quiet = True
    for k in range(10):
        c.note(k * 2000, 0.0, 0.0, "bump")
    assert len(c._events) == 0
    odom = (0.0, 0.0, math.pi / 2)
    for k in range(120):
        c.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=(0.0, 1.5), lmap=m)
    assert not c.active and c.map_started == 0
    c.quiet = False  # (the walk is the duck's own again: the same wall now begins a route)
    for k in range(120, 260):
        c.step(k * 100, odom, active=True, want=True, bound=NoBound(), goal=(0.0, 1.5), lmap=m)
    assert c.map_started == 1


def test_the_brain_is_quiet_unless_it_is_walking():
    from duckmind.neuro.brain import ROUTE_ACTS

    assert "forage" not in ROUTE_ACTS and "play_ball" not in ROUTE_ACTS and "flee" not in ROUTE_ACTS
    assert "wander" in ROUTE_ACTS and "return_dock" in ROUTE_ACTS


def test_a_map_saved_in_another_frame_waits_for_the_dock_and_lands_where_it_was():
    # the first boot: a wall 1 m ahead of the origin, the dock seen at (0.5, 0, pi): the dock's frame origin is 0.5 m behind the wall's side
    a = LM.LastMap()
    a.anchor_dock(
        (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)
    )  # the dock is 0.5 m ahead of the duck, facing it
    look_many(a, 10, 1.0, gap_ms=1200)  # the wall is 1.0 m ahead
    assert a.occupied(1.0, 0.0)
    state = a.to_state()
    # another boot, in a frame turned a quarter and moved: the duck sits where it saw the dock from, but its odometry says (3, 2, pi/2)
    b = LM.LastMap()
    b.load_state(state, rebase=True)
    assert not b.cells and b._pending  # (nothing yet: this boot does not know where its dock is)
    b.update(0, frame(0, (3.0, 2.0, math.pi / 2)))
    assert b.anchor_dock((3.0, 2.0, math.pi / 2), (-0.5, 0.0, 0.0)) == "set"
    assert b.cells and not b._pending
    # the wall is 1 m ahead of where the duck stood then: in this frame ahead is +y
    assert any(
        b.occupied(3.0 + dx, 3.05 + dy) for dx in (-0.1, 0.0, 0.1) for dy in (-0.1, 0.0, 0.1)
    )
    assert not any(b.occupied(4.0 + dx, 2.0) for dx in (-0.1, 0.0, 0.1))
    # a map with no dock pose cannot be rebased
    c = LM.LastMap()
    look_many(c, 6, 1.0, gap_ms=1200)
    d = LM.LastMap()
    d.load_state(c.to_state(), rebase=True)
    assert not d.cells and not d._pending


def test_a_route_that_goes_nowhere_is_given_up_after_its_tries_and_the_older_layers_have_the_legs():
    m = wall_map(gap=(0.8, 1.4))
    c = CL.Clutter(map_cfg(max_tries=2))
    odom = (0.0, 0.0, math.pi / 2)  # (the legs are held: it never gets anywhere)
    for k in range(200):
        c.step(
            k * 100,
            odom,
            active=True,
            want=True,
            bound=NoBound(),
            goal=(0.0, 1.5),
            lmap=m,
            held=True,
        )
    assert c.map_started == 1 and c.gave_up == 1 and not c.active
    stubborn = CL.Clutter(map_cfg())  # (the setting is off by default: it tries and tries)
    for k in range(200):
        stubborn.step(
            k * 100,
            odom,
            active=True,
            want=True,
            bound=NoBound(),
            goal=(0.0, 1.5),
            lmap=m,
            held=True,
        )
    assert stubborn.gave_up == 0 and stubborn.active


def test_the_brain_ends_an_escape_nobody_steps_and_a_route_gives_up_after_two_tries():
    from duckmind.neuro.brain import NeuroBrain
    from duckmind.neuro.genome import preset

    brain = NeuroBrain(preset("explorer"), 1, lastmap="act")
    assert brain.clutter.cfg.max_tries == 2
    t = 1000
    for k in range(30):
        brain.step(t, frame(t, (0.0, 0.0, 0.0)))
        t += 20
    brain.clutter.mode = "escape"  # (an escape left on by a step that never came)
    brain._clutter_t = t - 2000
    brain.step(t, frame(t, (0.0, 0.0, 0.0)))
    assert brain.clutter.mode != "escape"


def test_a_soft_margin_costs_beside_a_thing_and_shuts_nothing_but_the_thing():
    hard = wall_map(
        gap=(0.9, 1.1)
    )  # a gap 0.3 m wide (cells 0.9-1.1 free; the wall cells either side)
    soft = wall_map(gap=(0.9, 1.1))
    soft.cfg = LM.LastMapConfig(soft=True, hard_r=0.06, margin_r=0.22, margin_cost=2.0)
    mh = hard.maps((0.0, 0.0), (0.0, 0.5), block_r=0.22)
    ms = soft.maps((0.0, 0.0), (0.0, 0.5), block_r=0.22)
    gi, gj = ms.cell(1.0, 0.55)
    assert mh.blocked[gi * CR_N() + gj]  # the old ring shuts the gap
    assert not ms.blocked[gi * CR_N() + gj]  # the soft margin leaves it open …
    assert ms.cost[gi * CR_N() + gj] > 1.5  # … and dear
    # the cost falls off with the distance from the wall and is nothing past margin_r
    wi, wj = ms.cell(0.0, 0.55)
    costs = [ms.cost[wi * CR_N() + (wj - k)] for k in range(0, 5)]
    assert costs[1] > costs[2] >= costs[3] >= costs[4]
    assert costs[4] <= 1.0 + 1e-6 or costs[4] == costs[3]
    # exponential decay as an option
    soft.cfg = LM.LastMapConfig(soft=True, margin_exp=8.0, margin_cost=2.0)
    me = soft.maps((0.0, 0.0), (0.0, 0.5), block_r=0.22)
    near = me.cost[wi * CR_N() + (wj - 1)]
    far = me.cost[wi * CR_N() + (wj - 2)]
    assert near > far >= 1.0


def CR_N() -> int:
    return CR.N


def test_the_soft_route_goes_through_the_narrow_gap_the_hard_one_cannot():
    def through(soft: bool, block_r: float) -> bool:
        m = wall_map(gap=(0.9, 1.1))
        if soft:
            m.cfg = LM.LastMapConfig(soft=True)
        mp = m.maps((0.2, 0.0), (0.6, 0.5), block_r=block_r, goal=(1.0, 1.4))
        route = CR.search(mp, (0.2, 0.0), (0.2, 0.0), 1.0, (1.0, 1.4), 3.0)
        return route is not None and any(0.9 <= x <= 1.1 and 0.6 <= y for x, y in route.points)

    assert not through(False, 0.22)
    assert through(True, 0.22)


def test_a_soft_follower_does_not_back_off_for_geometry_where_the_hard_one_does():
    from duckmind.neuro.brain import SOFT_FOLLOW

    def backs(soft: bool) -> int:
        m = wall_map(gap=(0.8, 1.4))
        if soft:
            m.cfg = LM.LastMapConfig(soft=True)
        c = CL.Clutter(map_cfg(**(SOFT_FOLLOW if soft else {})))
        x, y, yaw = 0.0, 0.0, -math.pi / 2  # facing away from the wall: the way is behind
        n = 0
        for k in range(150):
            d = c.step(
                k * 100,
                (x, y, yaw),
                active=True,
                want=True,
                bound=NoBound(),
                goal=_goal_of((x, y, yaw), (0.0, 1.5)),
                lmap=m,
            )
            if d is None:
                continue
            n += 1 if d.back else 0
            yaw += d.vyaw * 0.1
            x += d.vx * math.cos(yaw) * 0.1
            y += d.vx * math.sin(yaw) * 0.1
        return n

    assert backs(True) <= backs(False)


def test_the_soft_mode_is_a_mode_and_builds_the_soft_map_and_follower():
    from duckmind.neuro.brain import SOFT_FOLLOW, map_configs

    assert LM.mode_of("soft") == "soft"
    lc, cc = map_configs("soft")
    assert not lc.soft and cc.soft_follow and cc.use_map and cc.look_m == SOFT_FOLLOW["look_m"]
    assert cc.map_goal is False and cc.max_tries == 3
    lc, cc = map_configs("act")
    assert not lc.soft and not cc.soft_follow and cc.max_tries == 2
    lc, cc = map_configs("soft", lcfg={"soft": True, "margin_cost": 1.0}, ccfg={"look_m": 0.4})
    assert lc.soft and lc.margin_cost == 1.0 and cc.look_m == 0.4
    lc, cc = map_configs("off")
    assert not lc.enabled and not cc.use_map
