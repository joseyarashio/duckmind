"""dock:build (2026-09-28): the neural duck backs into its charging dock by itself.

The docking program (:mod:`duckmind.neuro.dock`, phase 3's measured J-hook ported onto the
neural duck's own senses), its place in the brain, the motor and the body link, the senses' tag
reader, and the live seam. The state machine runs here on a small kinematic stand-in for the
body (a unicycle in the dock's frame, the face stopping the rear, PG closing when the seated rear
sits in the plates' window) — the physics is measured headless and live (PROTOTYPE.md).
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from duckmind.neuro import contract as C
from duckmind.neuro import dock as D
from duckmind.neuro import motor as M
from duckmind.neuro import select as S
from duckmind.neuro.bodylink import BodyLink, BodyRequest, shape_twist
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome, preset

GEO = D.DockGeometry()


# ── geometry ────────────────────────────────────────────────────────────────


def _read_from(pose: D.DockPose, odom: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> D.TagRead:
    """The tag reading a duck at ``pose`` (dock frame) would get: range and bearing from the
    trunk to the tag, and its heading off square-on."""
    dx, dy = GEO.face_mm - pose.x, -pose.y
    return D.TagRead(
        range_m=math.hypot(dx, dy) / 1000.0,
        bearing=D.wrap(math.atan2(dy, dx) - pose.h),
        facing=D.wrap(pose.h - math.pi),
        t_ms=0,
        odom=odom,
    )


@pytest.mark.parametrize(
    "pose",
    [
        D.DockPose(1400.0, 250.0, -2.6),
        D.DockPose(1150.0, -375.0, math.pi),
        D.DockPose(900.0, 0.0, 3.0),
    ],
)
def test_a_tag_reading_gives_the_pose_back(pose: D.DockPose) -> None:
    got = D.pose_from_read(_read_from(pose), GEO)
    assert got is not None
    assert got.x == pytest.approx(pose.x, abs=1e-6) and got.y == pytest.approx(pose.y, abs=1e-6)
    assert abs(D.wrap(got.h - pose.h)) < 1e-9


def test_an_unsure_facing_needs_a_carried_heading() -> None:
    pose = D.DockPose(1300.0, -200.0, 2.9)
    read = replace(_read_from(pose), facing=None)
    assert D.pose_from_read(read, GEO) is None  # a direction and a range are not a pose
    got = D.pose_from_read(read, GEO, heading=pose.h)
    assert got is not None and got.x == pytest.approx(pose.x) and got.y == pytest.approx(pose.y)


def test_the_docking_twist_rules() -> None:
    assert C.dock_walk(0.30, -1.0) == (0.30, -1.0)  # the U-turn as measured (not raised to 0.35)
    assert C.dock_walk(0.2, 0.5) == (0.30, 0.5)
    assert C.dock_walk(0.6, 0.0) == (C.APPROACH_VX, 0.0)
    assert C.dock_walk(-0.9, 0.3) == (C.REVERSE_VX, 0.3)
    assert C.dock_walk(-0.2, 0.0) == (-C.MIN_STEP_VX, 0.0)
    assert C.dock_walk(0.05, 1.0) == (0.0, 0.0)  # no turning on the spot
    assert C.dock_walk(0.3, 3.0)[1] == C.VYAW_MAX


# ── the state machine on a kinematic stand-in ──────────────────────────────


class Kin:
    """A unicycle in the dock's frame (mm, rad). The twist moves it after a short lag (``gain``: how
    far a forward command carries it, the legs answer slowly and coast on); its own odometry is the
    same motion in another frame (turned and shifted: it never knows the dock's frame); the face
    stops the rear (and proprio says ``blocked``). Sitting and standing as measured headless
    (docs/design/neural-duck/DOCK-FALL-2026-10-11.md §7.7): a sit slides the trunk ~72 mm back, never
    through the face; a stand pushes it ~48 mm forward; PG closes for a seat that ended 0-22 mm from
    the face and began 28 mm or more out of it (at the face the sit does not reach the plates)."""

    def __init__(
        self,
        x: float,
        y: float,
        h: float,
        odom0: tuple[float, float, float],
        *,
        gain: float = 1.0,
        slide_mm: float = 72.0,
        push_mm: float = 48.0,
    ) -> None:
        self.x, self.y, self.h = x, y, h
        self.ox, self.oy, self.oh = odom0
        self.rot = odom0[2] - h
        self.posture = "stand"
        self.want = "stand"
        self.change_at: int | None = None
        self.applied = (0.0, 0.0)
        self.blocked = False
        self.t = 0
        self.gain, self.slide_mm, self.push_mm = gain, slide_mm, push_mm
        self.sat_from = 0.0  # how far out of the face the last sit began (mm)
        self.sits = 0

    def odom(self) -> tuple[float, float, float]:
        return (self.ox, self.oy, self.oh)

    def face_x(self) -> float:
        """Where the trunk is when its rear meets the face."""
        return GEO.face_mm + GEO.rear_mm * math.cos(self.h)

    def out_mm(self) -> float:
        return self.x - self.face_x()

    def pg(self) -> bool:
        return (
            self.posture == "sit"
            and 0.0 <= self.out_mm() <= 22.0
            and self.sat_from >= 28.0
            and abs(self.y) < 70
        )

    def _move(self, dx: float, dy: float) -> None:
        """The trunk moves by (dx, dy) along the dock's frame; the odometry sees the same."""
        self.x += dx
        self.y += dy
        c, s = math.cos(self.rot), math.sin(self.rot)
        self.ox += c * dx - s * dy
        self.oy += s * dx + c * dy

    def step(self, out: D.DockOut, dt: float = 0.02) -> None:
        self.t += 20
        vx, vyaw = (out.vx, out.vyaw) if out.drive else (0.0, 0.0)
        if vx > 0.0:
            vx *= self.gain
        if out.posture in ("sit", "stand") and out.posture != self.want:
            self.want, self.change_at = out.posture, self.t + 1200
        if self.change_at is not None and self.t >= self.change_at:
            if self.want == "sit" and self.posture != "sit":
                self.sat_from = self.out_mm()
                self.sits += 1
                back = min(self.slide_mm, max(0.0, self.out_mm()))  # (the face stops the slide)
                self._move(-back * math.cos(self.h), -back * math.sin(self.h))
            elif self.want == "stand" and self.posture == "sit":
                self._move(self.push_mm * math.cos(self.h), self.push_mm * math.sin(self.h))
            self.posture, self.change_at = self.want, None
        if self.posture == "sit" or self.change_at is not None:
            vx = vyaw = 0.0
        a0, a1 = self.applied
        self.applied = (a0 + 0.3 * (vx - a0), a1 + 0.3 * (vyaw - a1))
        ax, ayaw = self.applied
        step = ax * 1000.0 * dt
        nx = self.x + step * math.cos(self.h)
        ny = self.y + step * math.sin(self.h)
        self.blocked = False
        if nx - GEO.rear_mm * math.cos(self.h) < GEO.face_mm and ax < 0.0:
            nx = GEO.face_mm + GEO.rear_mm * math.cos(self.h)  # the face stops the rear
            self.blocked = True
        dxw, dyw = nx - self.x, ny - self.y
        self.x, self.y = nx, ny
        # the walk's own rightward turn (phase 3: −8.5°/s forward, −4.5°/s reversing), which the
        # choreography's feed-forward cancels
        drift = 0.15 * min(1.0, ax / 0.35) if ax > 0.0 else 0.08 * min(1.0, -ax / 0.4)
        turn = (ayaw - drift) * dt
        self.h = D.wrap(self.h + turn)
        c, s = math.cos(self.rot), math.sin(self.rot)
        self.ox += c * dxw - s * dyw
        self.oy += s * dxw + c * dyw
        self.oh = D.wrap(self.oh + turn)

    def din(self, out: D.DockOut | None) -> D.DockIn:
        read = None
        if (
            out is not None
            and out.look is not None
            and self.t % 500 == 0
            and abs(self.applied[0]) < 0.01
        ):
            read = _read_from(D.DockPose(self.x, self.y, self.h), self.odom())
            read = replace(read, t_ms=self.t)
            seen = abs(D.wrap(read.bearing - out.look[0])) < 0.33 and abs(read.facing or 0.0) < 1.05
            read = read if seen else None
        return D.DockIn(
            t_ms=self.t,
            odom=self.odom(),
            posture="walk"
            if abs(self.applied[0]) > 0.05 and self.posture == "stand"
            else self.posture,
            pg=self.pg(),
            applied=self.applied,
            blocked=self.blocked,
            tag=read,
            head_home=True,
        )


def _dock_run(
    side: str, start: D.DockPose, seconds: float = 150.0
) -> tuple[D.DockRun, Kin, list[D.DockOut]]:
    run = D.DockRun(D.DockConfig(turn_side=side))
    kin = Kin(start.x, start.y, start.h, (3000.0, -1200.0, 0.7))
    first = replace(_read_from(start, kin.odom()), t_ms=0)
    din = replace(kin.din(None), tag=first)
    assert run.can_engage(din) == "tag"
    run.engage(din)
    outs = []
    out = run.step(din)
    for _ in range(int(seconds * 50)):
        kin.step(out)
        out = run.step(kin.din(out))
        outs.append(out)
        if run.mode == "home":
            break
    return run, kin, outs


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize(
    "start",
    [
        D.DockPose(1500.0, 150.0, -2.9),
        D.DockPose(1800.0, -400.0, 2.7),
        D.DockPose(1100.0, 0.0, math.pi),
    ],
)
def test_it_backs_into_its_dock_on_either_hook(side: str, start: D.DockPose) -> None:
    run, kin, outs = _dock_run(side, start)
    assert run.mode == "home" and run.result == "docked" and kin.pg()
    assert abs(kin.y) < 70 and abs(D.wrap(kin.h)) < 0.35  # seated on the axis, facing the room
    phases = [o.phase for o in outs]
    for step in ("fix", "uturn", "reverse_in", "bump_out", "sit"):
        assert step in phases
    # reversing only while backing in (and never faster than −0.40)
    assert all(o.phase == "reverse_in" for o in outs if o.drive and o.vx < 0.0)
    assert min(o.vx for o in outs) >= C.REVERSE_VX
    # the U-turn goes the side's way round: right turns (vyaw < 0) on the right, left mirrored
    uturn = [o.vyaw for o in outs if o.phase == "uturn" and o.vx > 0.0]
    assert uturn and all((v < 0.0) if side == "right" else (v > 0.0) for v in uturn)
    # the head is at HOME, no moments, while it turns round and backs in
    assert all(
        o.head_still and o.quiet
        for o in outs
        if o.phase in ("uturn", "reverse_in") and o.drive and o.vx != 0.0
    )
    assert any(o.look is not None for o in outs)  # it looked at the tag


def test_the_face_stop_is_blocked_equals_arrived() -> None:
    run, kin, _ = _dock_run("right", D.DockPose(1400.0, 0.0, math.pi))
    assert run.face_stop in ("stall", "blocked", "pose")
    assert any(entry.startswith("face_stop") for entry in run.log)


def _drive_until(
    run: D.DockRun, kin: Kin, out: D.DockOut, done: object, ticks: int = 6000
) -> D.DockOut:
    for _ in range(ticks):
        if done():  # type: ignore[operator]
            break
        kin.step(out)
        out = run.step(kin.din(out))
    return out


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("y", [520.0, -520.0])
def test_beside_the_guides_it_backs_out_in_an_arc_first(side: str, y: float) -> None:
    """(headless 2026-09-28: beside the dock, facing across its guides, the first forward arc
    swept over a guide's end and it tripped) it backs in an arc — the nose swinging out to the
    room — before it walks to the lane; reversing only there and backing in."""
    start = D.DockPose(400.0, y, math.copysign(math.radians(95.0), -y))
    run, kin, outs = _dock_run(side, start)
    assert "beside" in run.log
    first = [o for o in outs if o.phase == "back_step" and o.drive and o.vx < 0.0]
    assert first and all(math.copysign(1.0, o.vyaw) == math.copysign(1.0, y) for o in first[:10])
    assert first[0].zh == D.PHASE_ZH["beside"]
    assert all(o.phase in ("reverse_in", "back_step") for o in outs if o.drive and o.vx < 0.0)
    assert run.mode == "home" and run.result == "docked" and kin.pg()


def test_an_approach_from_inside_the_guides_walks_straight_out_first() -> None:
    """(headless 2026-09-28: its re-seats failed, a new approach from inside the dock arced
    toward the lane and swept over a guide) straight out along the axis first, no arcs inside."""
    run = D.DockRun(D.DockConfig(turn_side="right"))
    kin = Kin(260.0, 30.0, 0.3, (0.0, 0.0, 0.0))
    din = replace(kin.din(None), tag=replace(_read_from(D.DockPose(260.0, 30.0, 0.3)), t_ms=0))
    run.engage(din)
    out = run.step(din)
    forward_inside: set[str] = set()
    for _ in range(150 * 50):
        kin.step(out)
        out = run.step(kin.din(out))
        if kin.x < GEO.guides_mm and out.drive and out.vx > 0.0:
            forward_inside.add(out.phase)
        if run.mode == "home":
            break
    assert "out_first" in run.log
    # inside the guides it walks forward only straight out (steering onto the axis) or in the
    # small step off the face before it sits — never the lane walk's turning arcs
    assert forward_inside <= {"out_first", "bump_out"}
    assert run.result == "docked" and kin.pg()


def test_sideways_in_its_dock_it_stays_in_its_nest() -> None:
    """(headless 2026-09-28: stand-ups in the dock swung it ~80° round; the lane walk's arc from
    there swept it over a guide) no walk between the guides unless it faces out."""
    run = D.DockRun()
    pose = D.DockPose(210.0, -20.0, 1.4)
    din = D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=replace(_read_from(pose), t_ms=0))
    run.step(din)
    run.engage(replace(din, t_ms=20))
    assert run.mode == "home" and run.result == "failed:wedged"
    out = run.step(D.DockIn(t_ms=40, odom=(0.0, 0.0, 0.0), posture="stand"))
    assert out.vx == 0.0 and out.vyaw == 0.0


def test_a_stall_mid_room_is_not_the_face() -> None:
    """(live check 2026-09-28: the body link held the legs ~1 m out, proprio said ``blocked``,
    and the duck sat down mid-room three times) blocked = arrived only where the face can be."""
    run = D.DockRun()
    kin = Kin(1100.0, 0.0, math.pi, (0.0, 0.0, 0.0))
    din = replace(kin.din(None), tag=replace(_read_from(D.DockPose(1100.0, 0.0, math.pi)), t_ms=0))
    run.engage(din)
    out = run.step(din)

    def backing_far() -> bool:
        leg = run.leg
        pose = run.pose(kin.odom())
        return (
            run.phase == "reverse_in"
            and leg is not None
            and leg.moved_from is not None
            and pose is not None
            and leg.moved_from.x - pose.x > 60.0
            and pose.x > 700.0
        )

    out = _drive_until(run, kin, out, backing_far)
    assert backing_far()
    kin.step(out)
    out = run.step(replace(kin.din(out), blocked=True))
    assert run.phase == "reverse_in" and run.face_stop is None and "stall_far" in run.log
    assert out.drive and out.vx < 0.0  # it reverses on
    out = _drive_until(run, kin, out, lambda: run.mode == "home")
    assert run.result == "docked" and run.face_stop in ("stall", "blocked", "pose")


def test_the_first_look_with_a_remembered_frame_goes_on_unread() -> None:
    """(live check 2026-09-28: four unread looks at 2.9 m ended the try) with a frame it trusts,
    one unread look is enough: it walks to the lane's far point and looks from there."""
    run = D.DockRun()
    seen = D.DockPose(2400.0, -900.0, math.pi / 2)
    run.step(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=replace(_read_from(seen), t_ms=0)))
    assert run.trusted()
    t = 20
    din = D.DockIn(t_ms=t, odom=(0.0, 0.0, 0.0))
    run.engage(din, "remembered")
    for _ in range(400):
        t += 20
        run.step(D.DockIn(t_ms=t, odom=(0.0, 0.0, 0.0)))
        if run.phase == "to_lane":
            break
    assert run.phase == "to_lane" and run.mode == "docking"
    assert "blind" in run.log and run.log.count("fix") == 1
    assert t < 3500  # one look of 2.5 s, not four


def test_walking_into_something_it_steps_back_and_tries_again() -> None:
    """(live check 2026-09-28: 85 s pushing into a guide's side) a bump on the J-hook: a short
    straight step back (the owner's cornered back step), the walk again; the third: it ends."""
    run = D.DockRun()
    start = D.DockPose(700.0, 700.0, -1.4)  # beside the dock, facing across its mouth
    kin = Kin(start.x, start.y, start.h, (0.0, 0.0, 0.0))
    din = replace(kin.din(None), tag=replace(_read_from(start), t_ms=0))
    run.engage(din)
    out = run.step(din)
    for bump in range(3):
        out = _drive_until(run, kin, out, lambda: run.phase == "to_lane" and run.sub == "goto")
        assert run.phase == "to_lane" and run.sub == "goto"
        kin.step(out)
        out = run.step(replace(kin.din(out), blocked=True))
        if bump < 2:
            backs = [out]
            for _ in range(200):
                kin.step(out)
                out = run.step(kin.din(out))
                backs.append(out)
                if run.phase != "back_step":
                    break
            stepped = [o for o in backs if o.phase == "back_step" and o.vx < 0.0]
            assert stepped and all(o.vx >= C.REVERSE_VX and o.head_still for o in stepped)
            assert run.phase == "to_lane" and run.mode == "docking"
    assert run.mode == "idle" and run.result == "failed:blocked"
    assert run.log.count("bump") == 3


def test_a_homing_duck_keeps_trying_after_a_failed_try() -> None:
    """A failed try satisfies the wish to go home a little ("it gives up for now") — not while
    its pack runs low (live check 2026-09-28: it gave up and was stranded below 30 %)."""
    brain = NeuroBrain(preset("explorer"), seed=4)
    brain.behaviour = "return_dock"
    brain._dock_out = D.DockOut(event="failed")
    low = C.SensorFrame(t_ms=0, power=C.PowerSense(battery=0.33))
    full = C.SensorFrame(t_ms=0, power=C.PowerSense(battery=0.80))
    assert brain._consumed(0.02, None, 0.0, low, t_s_now=100.0) == pytest.approx(0.0)
    assert brain._consumed(0.02, None, 0.0, full, t_s_now=100.0) == pytest.approx(0.35)


def test_going_home_does_not_tire_while_the_program_has_the_legs() -> None:
    """(live check 2026-09-28: the 240-s backstop gave a homing duck up mid-docking) while its
    docking program is engaged the clock does not satiate going home; idle, it does."""
    brain = NeuroBrain(preset("explorer"), seed=4)
    seen = D.DockPose(1400.0, 0.0, math.pi)
    tag = replace(_read_from(seen), t_ms=0)
    brain.dock.step(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=tag))
    brain.dock.engage(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=tag))
    assert brain.dock.mode == "docking"
    i = brain.sel.index("return_dock")
    brain.sel.current = i
    brain.behaviour = "return_dock"
    brain.sel.sat[i] = 0.2
    brain.step(20, _frame(20, battery=0.34))
    assert brain.behaviour == "return_dock" and brain.dock.mode == "docking"
    assert brain.sel.sat[i] <= 0.2 + 1e-9
    brain.dock._end("switched", fail=False)
    brain.sel.current = i
    brain.behaviour = "return_dock"
    brain.step(40, _frame(40, battery=0.34))
    assert brain.behaviour == "return_dock" and brain.sel.sat[i] > 0.2
    assert S.SATIATE_S["return_dock"] == pytest.approx(240.0)  # the clock backstop otherwise


def test_walking_home_it_keeps_its_eyes_on_its_dock() -> None:
    """With a remembered frame and no tag in view it walks to the lane's far point — and looks
    where it believes the dock is (the reading comes the first moment one can)."""
    brain = NeuroBrain(preset("snuggler"), seed=6)
    seen = D.DockPose(3000.0, 1200.0, 3.0)
    brain.dock.step(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=replace(_read_from(seen), t_ms=0)))
    assert brain.dock.trusted()
    want = brain.dock.dock_bearing((0.0, 0.0, 0.0))
    assert want is not None and abs(want[0]) < 1.0
    brain.drives.hunger = 0.9
    gazes = []
    for k in range(1, 60):
        brain.sel.current = brain.sel.index("return_dock")
        brain.behaviour = "return_dock"
        cmd, _ = brain.step(20 * k, _frame(20 * k, battery=0.34))
        if cmd.behaviour == "return_dock" and brain.dock.mode == "idle":
            gazes.append(cmd.gaze)
    assert gazes
    assert any(abs(g.bearing - want[0]) < 0.05 and g.elevation < 0.0 for g in gazes)


def test_a_reflex_pauses_it_and_it_goes_on() -> None:
    run = D.DockRun()
    kin = Kin(1400.0, -100.0, math.pi, (0.0, 0.0, 0.0))
    din = replace(
        kin.din(None), tag=replace(_read_from(D.DockPose(1400.0, -100.0, math.pi)), t_ms=0)
    )
    run.engage(din)
    out = run.step(din)
    for _ in range(400):  # into the J-hook
        kin.step(out)
        out = run.step(kin.din(out))
    phase = run.phase
    halted = replace(kin.din(out), t_ms=kin.t + 20, halted=True)
    held = run.step(halted)
    assert held.drive and held.vx == 0.0 and held.vyaw == 0.0 and run.mode == "docking"
    assert run.phase == phase  # it goes on from where it was


def test_a_pickup_forgets_where_the_dock_is() -> None:
    run = D.DockRun()
    kin = Kin(1400.0, 0.0, math.pi, (0.0, 0.0, 0.0))
    din = replace(kin.din(None), tag=replace(_read_from(D.DockPose(1400.0, 0.0, math.pi)), t_ms=0))
    run.engage(din)
    run.step(din)
    run.step(replace(kin.din(None), t_ms=40, upset=True))
    assert run.mode == "idle" and run.pose((0.0, 0.0, 0.0)) is None
    assert run.result == "failed:upset"


def test_seated_on_its_contacts_it_is_home_and_charging() -> None:
    run = D.DockRun()
    out = run.step(D.DockIn(t_ms=0, odom=(10.0, 20.0, 0.3), posture="sit", pg=True))
    assert run.mode == "home" and out.phase == "home" and not out.drive
    view = run.view((10.0, 20.0, 0.3), 0, True, True)
    assert view["charging"] and view["zh"] == "在窩裡充電" and view["pose"][0] == GEO.seat_mm
    assert run.in_dock((10.0, 20.0, 0.3))


def test_it_leaves_straight_out_along_the_axis() -> None:
    """A backed-in duck faces the room: it gets up and walks out forward (no reversing), steering
    along the dock's axis until it is past the guides."""
    run = D.DockRun()
    kin = Kin(GEO.seat_mm, 10.0, 0.25, (0.0, 0.0, 0.0))
    kin.posture = kin.want = "sit"
    run.step(D.DockIn(t_ms=0, odom=kin.odom(), posture="sit", pg=True))
    run.leave(kin.din(None))
    outs = []
    path = []
    out = run.step(kin.din(None))
    for _ in range(600):
        kin.step(out)
        out = run.step(kin.din(out))
        outs.append(out)
        path.append((kin.x, kin.y))
        if run.mode == "idle":
            break
    assert run.result == "left" and kin.x >= D.DockConfig().leave_mm - 30
    assert all(o.vx >= 0.0 for o in outs)  # never backwards out of a back-in dock

    def room(x: float) -> float:
        """Half the free width for the hips (±71 mm) between the guides: the channel (±130 mm)
        to 230 mm from the wall, flaring to the mouth (±230 mm) at 440 mm."""
        wall = 130.0 if x <= 230.0 else 130.0 + 100.0 * min(1.0, (x - 230.0) / 210.0)
        return wall - 71.0

    assert all(abs(y) < room(x) for x, y in path if x < 440.0)  # between the guides
    assert "out" in run.log


def test_pg_decides_even_during_a_re_seat() -> None:
    run = D.DockRun()
    kin = Kin(GEO.seat_mm + 40.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    kin.posture = kin.want = "sit"
    run.step(D.DockIn(t_ms=0, odom=kin.odom(), posture="sit", pg=True))  # home
    run.reseat(D.DockIn(t_ms=20, odom=kin.odom(), posture="sit", pg=False))
    out = run.step(D.DockIn(t_ms=40, odom=kin.odom(), posture="sit", pg=True))
    assert run.mode == "home" and out.event == "docked"


def test_it_remembers_its_dock_from_seeing_the_tag_and_walks_to_the_lane() -> None:
    run = D.DockRun()
    seen = D.DockPose(2600.0, 900.0, -2.8)
    run.step(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=replace(_read_from(seen), t_ms=0)))
    assert run.trusted() and run.mode == "idle"
    bearing, range_m = run.rendezvous((0.0, 0.0, 0.0)) or (None, None)
    assert range_m == pytest.approx(math.hypot(1100.0, 900.0 + 375.0) / 1000.0, rel=1e-6)
    too_far = D.DockPose(4200.0, 0.0, math.pi)  # beyond remember_m: not taken
    fresh = D.DockRun()
    fresh.step(D.DockIn(t_ms=0, odom=(0.0, 0.0, 0.0), tag=replace(_read_from(too_far), t_ms=0)))
    assert not fresh.trusted()


# ── the brain ──────────────────────────────────────────────────────────────


def _sight(pose: D.DockPose) -> C.Sighting:
    read = _read_from(pose)
    return C.Sighting(
        kind="nest", bearing=read.bearing, range_m=read.range_m, facing=read.facing, track=1
    )


def _frame(
    t_ms: int, *, pose: D.DockPose | None = None, battery: float = 0.36, **body: object
) -> C.SensorFrame:
    cam = C.CameraSense(
        usable=True, light="bright", sightings=(_sight(pose),) if pose is not None else (), age_ms=0
    )
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(**{"posture": "stand", **body}),  # type: ignore[arg-type]
        camera=cam,
        power=C.PowerSense(battery=battery),
        day_minute=660.0,
    )


def test_a_low_battery_sends_it_home_and_the_docking_program_takes_its_legs() -> None:
    brain = NeuroBrain(preset("snuggler"), seed=3)
    brain.drives.hunger = 0.8
    pose = D.DockPose(1500.0, 100.0, math.pi - 0.1)
    docking = []
    for k in range(200):
        cmd, tel = brain.step(20 * k, _frame(20 * k, pose=pose))
        docking.append((cmd.behaviour, cmd.docking, tel.dock["mode"] if tel.dock else None))
    assert any(beh == "return_dock" for beh, _, _ in docking)
    assert any(mode == "docking" for _, _, mode in docking)
    assert any(flag for _, flag, _ in docking)  # its twist is the choreography's own
    assert tel.dock is not None and tel.dock["version"] == D.DOCK_VERSION
    assert "zh" in tel.dock and tel.locomotion["docking"] in (0.0, 1.0)


def test_home_pull_and_battery_low_are_features_it_selects_on() -> None:
    for name in ("docking", "home_pull", "battery_low", "homing"):
        assert name in S.F and name in S.FEATURE_ZH
    assert S.W["return_dock"]["homing"] > 0.0 and S.W["return_dock"]["home_pull"] > 0.0
    assert S.W["wander"]["battery_low"] < 0.0  # a low pack damps walking away
    assert S.W["nap"]["home_pull"] < 0.0  # away from home a sleepy duck goes home first


def test_on_its_contacts_it_charges_and_the_telemetry_says_so() -> None:
    brain = NeuroBrain(preset("dreamer"), seed=5)
    frame = replace(
        _frame(0, battery=0.5),
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.5, on_dock=True, charging=True),
    )
    _, tel = brain.step(0, frame)
    assert tel.dock is not None and tel.dock["mode"] == "home" and tel.dock["charging"]
    assert tel.dock["zh"] == "在窩裡充電"


def test_it_walks_straight_out_of_its_nest_before_walking_away() -> None:
    """(dock:build: replaces the forward dock's back-out) A duck seated on its contacts that
    wants to walk gets up and walks out FORWARD along the dock's axis — a back-in dock has it
    facing the room — then the rhythm takes over. It never reverses out."""
    brain = NeuroBrain(_bold(), seed=1)
    brain.drives.restless = 1.0
    seated = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.95, on_dock=True, charging=True),
    )
    brain.step(0, seated)
    assert brain.dock.mode == "home"
    brain.sel.current = brain.sel.index("wander")
    brain.behaviour = "wander"
    cmds: list[tuple[float, bool, str]] = []
    x = 0.0
    for k in range(1, 400):
        posture = "stand" if k > 20 else "sit"
        cmd, tel = brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(
                    posture=posture, odom=(x, 0.0, 0.0), vx=cmds[-1][0] if cmds else 0.0
                ),
                power=C.PowerSense(battery=0.95, on_dock=k <= 20, charging=k <= 20),
            ),
        )
        assert tel.dock is not None
        cmds.append((cmd.vx, cmd.docking, tel.dock["mode"]))
        x += max(0.0, cmd.vx) * 0.02
    assert any(mode == "leaving" for _, _, mode in cmds)
    assert any(vx > 0.0 and docking for vx, docking, _ in cmds)  # out along the axis
    assert all(vx >= 0.0 for vx, _, _ in cmds)  # never backing out of a back-in dock


@pytest.mark.parametrize("routine", ["stand_up", "wander"])
def test_a_routine_cannot_break_charging_before_the_leave_level(routine: str) -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    brain.drives.restless = 1.0
    seated = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.4, on_dock=True, charging=True),
        day_minute=660.0,
    )
    brain.step(0, seated)
    for k in range(1, 51):
        if k == 20:
            seated = replace(seated, power=replace(seated.power, battery=0.7))
        brain.sel.commit_next(routine)
        cmd, _ = brain.step(20 * k, replace(seated, t_ms=20 * k))
        assert cmd.posture == "sit"
        assert brain.dock.mode == "home"
    charged = replace(seated, power=replace(seated.power, battery=0.99))
    brain.sel.commit_next("wander")
    cmd, _ = brain.step(1020, replace(charged, t_ms=1020))
    assert brain.dock.mode == "leaving"
    assert cmd.posture == "stand"


def test_a_well_charged_arrival_need_not_start_a_top_up_session() -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    frame = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.85, on_dock=True, charging=True),
        day_minute=660.0,
    )
    brain.step(0, frame)
    brain.sel.commit_next("wander")
    cmd, _ = brain.step(20, replace(frame, t_ms=20))
    assert brain.dock.mode == "leaving"
    assert cmd.posture == "stand"


def test_lost_charge_contact_is_reseated_during_a_seated_head_act() -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    frame = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.4, on_dock=True, charging=True),
        day_minute=660.0,
    )
    brain.step(0, frame)
    opened = replace(frame, power=replace(frame.power, on_dock=False, charging=False))
    for k in range(1, 201):
        brain.sel.commit_next("watch")
        brain.step(20 * k, replace(opened, t_ms=20 * k))
    assert brain.dock.mode == "docking"


def _seated_charging_run(
    name: str, seed: int, *, bang: bool, seconds: float = 16.0
) -> tuple[list[str], list[str]]:
    """A duck seated on its contacts at 40 % charge, a real 94 dB bang at 4 s (or none), for ``seconds`` after. The body is a stand-in that
    takes the posture the brain asks for one second later, and is off the contacts once it stands. Returns the behaviours it passed through
    and the dock's modes."""
    brain = NeuroBrain(preset(name), seed=seed)
    posture, asked_at, asked = "sit", None, "sit"
    behaviours: list[str] = []
    modes: list[str] = []
    loud = C.SoundSense(level_db=94.0, onset=1.0, kind="bang")
    t_ms = 0
    while t_ms < 4000 + int(seconds * 1000):
        seated = posture == "sit"
        frame = C.SensorFrame(
            t_ms=t_ms,
            body=C.BodySense(posture=posture),
            power=C.PowerSense(battery=0.4, on_dock=seated, charging=seated),
            sound=loud if bang and t_ms == 4000 else C.SoundSense(),
            day_minute=660.0,
        )
        cmd, tel = brain.step(t_ms, frame)
        if cmd.posture in ("stand", "sit") and cmd.posture != posture:
            if asked != cmd.posture or asked_at is None:
                asked, asked_at = cmd.posture, t_ms
            elif t_ms - asked_at >= 1000:
                posture, asked_at = cmd.posture, None
        else:
            asked_at = None
        if not behaviours or behaviours[-1] != tel.behaviour:
            behaviours.append(tel.behaviour)
        if not modes or modes[-1] != brain.dock.mode:
            modes.append(brain.dock.mode)
        t_ms += 50
    return behaviours, modes


@pytest.mark.parametrize("name", ["snuggler", "hatchling"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_charging_does_not_prevent_a_panic_flight(name: str, seed: int) -> None:
    """A real bang while seated and charging at 0.4 frightens these ducks into a flight, and the flight takes them off the contacts: the
    dock goes to ``leaving``. The same duck without the bang sits out its charge."""
    behaviours, modes = _seated_charging_run(name, seed, bang=True)
    assert "startle" in behaviours and "flee" in behaviours, behaviours
    assert "leaving" in modes, modes
    quiet_behaviours, quiet_modes = _seated_charging_run(name, seed, bang=False)
    assert "flee" not in quiet_behaviours and quiet_modes == ["home"], (
        quiet_behaviours,
        quiet_modes,
    )


@pytest.mark.parametrize("behaviour", ["flee", "retreat"])
def test_a_calm_defensive_channel_cannot_end_recharging(behaviour: str) -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    seated = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.4, on_dock=True, charging=True),
        day_minute=660,
    )
    brain.step(0, seated)
    brain.sel.commit_next(behaviour)
    cmd, _ = brain.step(20, replace(seated, t_ms=20))
    assert cmd.posture == "sit"
    assert brain.dock.mode == "home"


def test_a_previous_charge_arrival_does_not_satiate_a_low_battery_return() -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    brain.sel.commit_next("return_dock")
    brain.drives.hunger = 1.0
    charging = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.33, on_dock=True, charging=True),
        day_minute=660.0,
    )
    for k in range(1500):
        brain.step(20 * k, replace(charging, t_ms=20 * k))
    selected = []
    for k in range(1500, 1600):
        cmd, _ = brain.step(20 * k, _frame(20 * k, battery=0.34, odom=(1.0, 0.0, 0.0)))
        selected.append(cmd.behaviour)
    assert "return_dock" in selected


def test_a_grounded_pickup_claim_does_not_erase_the_known_dock() -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    brain.step(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="sit", leg_load=0.8),
            power=C.PowerSense(battery=0.95, on_dock=True, charging=True),
            day_minute=600,
        ),
    )
    assert brain.dock.trusted()
    for k in range(1, 51):
        events = ("lifted",) if k == 1 else (("set_down",) if k == 10 else ())
        brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="stand", leg_load=0.9),
                power=C.PowerSense(battery=0.8),
                events=events,
                day_minute=600,
            ),
        )
        assert brain.dock.trusted()


def test_a_confirmed_pickup_invalidates_the_dock_frame() -> None:
    brain = NeuroBrain(preset("explorer"), seed=1)
    brain.step(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="sit", leg_load=0.8),
            power=C.PowerSense(battery=0.95, on_dock=True, charging=True),
            day_minute=600,
        ),
    )
    assert brain.dock.trusted()
    for k in range(1, 31):
        brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="sit", leg_load=0.4),
                power=C.PowerSense(battery=0.8),
                events=("lifted",) if k == 1 else (),
                day_minute=600,
            ),
        )
    assert not brain.dock.trusted()


def _seated_on_dock_then(
    loads: list[float | None], events: dict[int, tuple[str, ...]]
) -> NeuroBrain:
    """A duck that knows its dock (seated on it, charging), then one frame per 20 ms with the given leg loads and events (tick -> events)."""
    brain = NeuroBrain(preset("explorer"), seed=1)
    brain.step(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="sit", leg_load=0.8),
            power=C.PowerSense(battery=0.95, on_dock=True, charging=True),
            day_minute=600,
        ),
    )
    assert brain.dock.trusted()
    for k, load in enumerate(loads, start=1):
        brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="sit", leg_load=load),
                power=C.PowerSense(battery=0.8),
                events=events.get(k, ()),
                day_minute=600,
            ),
        )
    return brain


def test_a_short_carry_that_ended_at_a_set_down_clears_the_dock_frame() -> None:
    # held 0.3 s (the legs light 15 ticks, not the half second a hold needs), then put down: the hand did move it
    loads: list[float | None] = [0.4] * 15 + [0.8] * 10
    brain = _seated_on_dock_then(loads, {1: ("lifted",), 16: ("set_down",)})
    assert not brain.dock.trusted()


def test_a_claim_whose_legs_never_hung_light_keeps_the_dock_frame() -> None:
    # light for 4 ticks (80 ms) only: a blip, not a carry
    loads: list[float | None] = [0.4] * 4 + [0.8] * 21
    brain = _seated_on_dock_then(loads, {1: ("lifted",), 6: ("set_down",)})
    assert brain.dock.trusted()


def test_a_leg_load_that_goes_unknown_mid_window_clears_the_dock_frame() -> None:
    # a claim made with a known load, the legs light, then no load at all: the same as a claim made with none
    loads: list[float | None] = [0.4] * 5 + [None] * 5 + [0.4] * 15
    brain = _seated_on_dock_then(loads, {1: ("lifted",)})
    assert not brain.dock.trusted()


def _bold() -> Genome:
    return preset("explorer")


def test_a_reflex_while_backing_in_is_covert_for_the_head() -> None:
    brain = NeuroBrain(preset("snuggler"), seed=2)
    brain.dock.mode = "docking"
    brain.dock.phase = "reverse_in"
    assert brain.dock.head_busy() and brain.dock.expects_contact() is False
    brain.dock.backed = True
    assert brain.dock.expects_contact()


# ── the motor and the body link ────────────────────────────────────────────


def _cmd(**kw: object) -> C.MotorCommand:
    base = {"t_ms": 0, "behaviour": "return_dock", "bout": 1, "variant": 1, "posture": "stand"}
    return C.MotorCommand(**{**base, **kw})  # type: ignore[arg-type]


def test_the_motor_passes_the_docking_twist_as_it_is() -> None:
    profile = M.profile_from_genome(Genome.from_seed(7))
    state = M.MotorState()
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="walk", vx=0.3, vyaw=-0.9))
    for k in range(20):  # (a learned course-keeping trim would be added to an ordinary walk)
        state = replace(state, trim=0.2)
        state, out = M.step(
            profile,
            state,
            _cmd(t_ms=20 * k, vx=0.30, vyaw=-1.0, docking=True),
            replace(frame, t_ms=20 * k),
        )
    assert out.vx == pytest.approx(0.30) and out.vyaw == pytest.approx(-1.0)
    state, out = M.step(
        profile, state, _cmd(t_ms=500, vx=-0.40, vyaw=0.2, docking=True), replace(frame, t_ms=500)
    )
    assert out.vx == pytest.approx(-0.40) and out.vyaw == pytest.approx(0.2)


def test_the_motor_sits_when_the_docking_program_sits() -> None:
    profile = M.profile_from_genome(Genome.from_seed(7))
    state = M.MotorState()
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    state, out = M.step(profile, state, _cmd(posture="sit", docking=True), frame)
    assert out.posture == "sit"  # over return_dock's default need to stand


def test_the_body_link_lets_the_docking_twist_through() -> None:
    assert shape_twist(0.30, -1.0, docking=True)[:2] == (0.30, -1.0)
    assert shape_twist(0.30, -1.0)[:2] == (0.35, -1.0)  # an ordinary walk: not below 0.35
    assert shape_twist(-0.40, 0.3, docking=True)[:2] == (-0.40, 0.3)
    assert shape_twist(-0.40, 0.3)[:2] == (0.0, 0.0)  # reversing only while docking


def test_the_live_seam_sends_the_docking_flag_and_the_side() -> None:
    from types import SimpleNamespace

    from duckmind.simweb import neuro_live as NL

    brain = NeuroBrain(preset("snuggler"), seed=1)
    lanes = SimpleNamespace(brain=brain, notes=[])
    NL._sync_dock_side(lanes, SimpleNamespace(home="default_home"))  # type: ignore[arg-type]
    assert brain.dock.cfg.turn_side == "right"
    sample = NL._dock_sample(
        SimpleNamespace(dock={"mode": "home", "charging": True, "zh": "在窩裡充電"})  # type: ignore[arg-type]
    )
    assert sample is not None and sample["charging"] and sample["mode"] == "home"
    brain.set_dock("left")
    assert brain.dock.cfg.turn_side == "left" and brain.dock.cfg.lane_y() > 0.0
    with pytest.raises(ValueError):
        brain.set_dock("up")


def test_the_act_is_named_for_what_the_body_now_does() -> None:
    spec = C.BEHAVIOURS["return_dock"]
    assert spec.zh == "自己回去充電" and "倒退" in spec.body_zh and "還不會" not in spec.body_zh


def test_a_stall_while_backing_in_is_the_face_not_a_reflex_stop() -> None:
    link = BodyLink()
    assert link.cfg.dock_min_vx == 0.30
    # (the link's reflex hold and its arcs-only turning away are measured in the headless and
    # live runs; here only that a docking request is never turned into an ordinary walk)
    request = BodyRequest(vx=-0.4, vyaw=0.1, docking=True)
    assert shape_twist(request.vx, request.vyaw, docking=True)[0] == pytest.approx(-0.4)
