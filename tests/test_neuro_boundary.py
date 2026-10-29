"""The boundary memory (duckmind.neuro.boundary): the duck sees the walls and furniture it faces
through the camera's floor line (duck-eye FL-v1's five free-floor sectors, MODEL in the
prototype) and steers away before touching them; what it saw stays where it was while it walks
and turns (its own odometry), fades, and is erased where it now sees clear floor; the thing it
walks to is not a wall. And in the synthetic room the floor line cuts the bumps."""

from __future__ import annotations

import math

from duckmind.neuro import boundary as B
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.contract import (
    BodySense,
    CameraSense,
    FreeSector,
    SensorFrame,
    TofSense,
)
from duckmind.neuro.genome import Genome
from duckmind.neuro.synth import SynthWorld, iterate


def _line(
    free: float, end: str = "wall", bearings: tuple[float, ...] = (0.26, 0.13, 0.0, -0.13, -0.26)
):
    return tuple(FreeSector(b, 0.13, free, end if free < 1.45 else "unknown") for b in bearings)


def _frame(
    t_ms: int,
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    floor: tuple[FreeSector, ...] = (),
    look_ms: int | None = None,
    posture: str = "walk",
    events: tuple[str, ...] = (),
    tof: TofSense | None = None,
) -> SensorFrame:
    return SensorFrame(
        t_ms=t_ms,
        body=BodySense(posture=posture, odom=odom),
        events=events,
        tof=tof or TofSense(),
        camera=CameraSense(usable=True, source="model", floor=floor, floor_t_ms=look_ms),
    )


def test_a_wall_seen_ahead_bends_the_walk_before_it_is_touched() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.7), look_ms=0))
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert abs(steer.free_ahead - 0.7) < 0.03
    assert steer.urgency > 0.3 and steer.veer != 0.0  # it bends away, well before touching
    assert steer.cap is None or steer.cap >= 0.35  # an arc, never a stop on the spot
    # nothing near: no bend at all
    clear = B.Boundary()
    clear.update(0, _frame(0, floor=_line(1.5), look_ms=0))
    assert clear.steer(0, (0.0, 0.0, 0.0)).veer == 0.0


def test_close_to_a_wall_only_the_slowest_tight_arc_goes_on() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.25), look_ms=0))
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert steer.cap == 0.35 and abs(steer.veer) >= 0.8 and steer.urgency == 1.0


def test_it_turns_toward_the_side_it_saw_free() -> None:
    b = B.Boundary()
    # a wall ahead and to the right, clear floor to the left
    line = (
        FreeSector(0.26, 0.13, 1.5, "unknown"),
        FreeSector(0.13, 0.13, 1.5, "unknown"),
        FreeSector(0.0, 0.13, 0.6, "wall"),
        FreeSector(-0.13, 0.13, 0.55, "wall"),
        FreeSector(-0.26, 0.13, 0.5, "wall"),
    )
    b.update(0, _frame(0, floor=line, look_ms=0))
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert steer.veer > 0.0  # + = left: away from the wall on the right


def test_what_it_saw_stays_put_while_it_turns_and_walks() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.8), look_ms=0))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) < 0.85
    # turned 90° to the left: the wall is now on its right, the way ahead is clear
    turned = (0.0, 0.0, math.pi / 2)
    assert b.clearance(turned, 0.0) >= 1.49
    assert b.clearance(turned, -math.pi / 2) < 0.85
    # walked 0.5 m toward it: it is nearer
    assert b.clearance((0.5, 0.0, 0.0), 0.0) < 0.35


def test_a_look_is_placed_where_the_duck_was_when_it_looked() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, (0.0, 0.0, 0.0)))
    b.update(100, _frame(100, (0.02, 0.0, 0.0)))
    # the look (captured at t 0, from odometry x 0) arrives at t 200, from x 0.04
    b.update(200, _frame(200, (0.04, 0.0, 0.0), floor=_line(0.8), look_ms=0))
    assert abs(b.clearance((0.04, 0.0, 0.0), 0.0) - 0.76) < 0.03


def test_the_memory_fades_and_clear_floor_erases_it() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.8), look_ms=0))
    assert b.points
    # a later look sees clear floor there (the thing moved away): the points go
    b.update(500, _frame(500, floor=_line(1.5), look_ms=500))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) >= 1.49
    # unseen for a long time, a wall fades from memory
    c = B.Boundary()
    c.update(0, _frame(0, floor=_line(0.8), look_ms=0))
    for k in range(1, 121):
        c.update(k * 1000, _frame(k * 1000))
    assert not c.points



def test_a_point_under_its_own_body_is_not_a_wall_ahead() -> None:
    """A floor-line look at its own feet (or a ToF point at 0.06 m) lays a point under the body:
    nothing solid can be ahead there, so the way ahead stays the floor's own range and no arc is
    forced (live 2026-10-23: such a point held free_ahead at 0.0–0.18 m and the duck circled a
    place 1 m away instead of walking to it). A real point at 0.15 m ahead still bends the walk."""
    under = B.Boundary()
    under.update(0, _frame(0, floor=_line(0.06), look_ms=0))
    steer = under.steer(0, (0.0, 0.0, 0.0))
    assert steer.free_ahead >= 1.4, steer
    assert steer.veer == 0.0 and steer.urgency == 0.0, steer
    # a real obstacle 0.15 m ahead still bends it into the slowest tight arc
    near = B.Boundary()
    near.update(0, _frame(0, floor=_line(0.15), look_ms=0))
    tight = near.steer(0, (0.0, 0.0, 0.0))
    assert tight.free_ahead < 0.2 and abs(tight.veer) >= 0.8, tight


def test_the_thing_it_walks_to_is_not_a_wall() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.5, "thing"), look_ms=0))
    blocked = b.steer(0, (0.0, 0.0, 0.0))
    assert blocked.urgency > 0.5
    b2 = B.Boundary()
    b2.update(0, _frame(0, floor=_line(0.5, "thing"), look_ms=0))
    toward = b2.steer(0, (0.0, 0.0, 0.0), goal_bearing=0.0, goal_range=0.5)
    assert toward.urgency == 0.0 and toward.veer == 0.0


def test_a_bump_and_the_tof_lay_walls_down_too() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, events=("blocked",)))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) < 0.2
    c = B.Boundary()
    c.update(0, _frame(0, tof=TofSense(True, None, 400, None)))
    assert 0.4 < c.clearance((0.0, 0.0, 0.0), 0.0) < 0.5


def test_a_brush_still_steers_the_walk_and_can_be_left_out_of_a_room_check() -> None:
    # (FlipRoom: the flip's room check leaves the side brushes' guesses out; the walk keeps steering by them as before)
    b = B.Boundary()
    kicked_right = SensorFrame(
        t_ms=0, body=BodySense(posture="walk", gyro=(0.0, 0.0, -0.9)), events=("brushed",),
        camera=CameraSense(usable=True, source="model"),
    )
    b.update(0, kicked_right)
    left = math.radians(80.0)
    assert b.clearance((0.0, 0.0, 0.0), left) < 0.2  # (it is on the left, ~0.13 m away)
    assert b.clearances((0.0, 0.0, 0.0), [left])[0] < 0.2
    assert b.clearances((0.0, 0.0, 0.0), [left], brushes=False)[0] >= 1.4
    b.update(100, _frame(100, events=("blocked",)))
    assert b.clearances((0.0, 0.0, 0.0), [0.0], brushes=False)[0] < 0.2  # (a bump stays in the way)


def test_put_down_by_a_hand_it_forgets_where_the_walls_were() -> None:
    """(real-simulator flip measurement 2026-10-26) the memory is in its odometry, which knows nothing of a carry: after a
    hand set it down elsewhere the walls it remembers are nowhere, and a flip on open floor was refused as "no room"."""
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.4), look_ms=0, posture="stand"))
    b.update(20, _frame(20, events=("blocked",), posture="stand"))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) < 0.45
    b.update(1000, _frame(1000, events=("lifted",), posture="stand"))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) < 0.45  # (held: nothing is known yet of where it will be)
    b.update(4000, _frame(4000, events=("set_down",), posture="stand"))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) > 1.0
    assert min(b.clearances((0.0, 0.0, 0.0), [0.0, 1.57, 3.14, -1.57])) > 1.0
    assert not b.blocks
    # what it sees after that is remembered as before
    b.update(4100, _frame(4100, floor=_line(0.6), look_ms=4100, posture="stand"))
    assert b.clearance((0.0, 0.0, 0.0), 0.0) < 0.65


def test_in_the_synthetic_room_seeing_the_walls_cuts_the_bumps() -> None:
    """The bold explorer (Duck Core seed 4107 at GD 60), no ToF, 20 simulated minutes at 20 Hz:
    with the camera's floor line it bumps into the walls far less often than blind. (Blind, it
    still walks more carefully where it has not seen the floor clear: fewer bumps than before
    that caution — 17, 9, 5 in 20 min for seeds 3–5 — and 0 seeing.)"""

    class Blind(SynthWorld):  # (no camera at all: no floor line, nothing to stop and look for)
        def _camera(self, t_ms: int, light: str) -> CameraSense:
            cam = super()._camera(t_ms, light)
            return CameraSense(
                False, cam.light, cam.brightness, cam.motion, cam.sightings, cam.age_ms
            )

    def bumps(world_cls: type[SynthWorld]) -> int:
        world = world_cls(seed=3, tof=False, family=False, day_minute0=600.0)
        brain = NeuroBrain(Genome.from_seed(4107, gd=60), 3)
        count = 0
        for _cmd, _tel in iterate(brain, world, 1200, hz=20):
            count += world._events.count("blocked")
        return count

    seeing, blind = bumps(SynthWorld), bumps(Blind)
    assert blind >= 4 and seeing <= blind // 4


def test_once_it_bends_away_it_does_not_dither_between_left_and_right() -> None:
    """Live 2026-09-28: near the south wall the veer flipped −1 → +0.5 → −1 within 2 s, the
    arc began late, ran along the wall into the dock's guide rail and the duck fell. Once it
    bends one way, the other way must be clearly freer (COMMIT_M) before it swings back."""
    b = B.Boundary()
    # a wall ahead, a little more room on the right: it bends right
    right_freer = (
        FreeSector(0.26, 0.13, 0.75, "wall"),
        FreeSector(0.13, 0.13, 0.75, "wall"),
        FreeSector(0.0, 0.13, 0.7, "wall"),
        FreeSector(-0.13, 0.13, 0.85, "wall"),
        FreeSector(-0.26, 0.13, 0.9, "wall"),
    )
    b.update(0, _frame(0, floor=right_freer, look_ms=0))
    first = b.steer(0, (0.0, 0.0, 0.0))
    assert first.veer < 0.0
    # the next look shows the left a little freer: it keeps bending right
    left_bit_freer = tuple(
        FreeSector(s.bearing, s.width, f, "wall")
        for s, f in zip(right_freer, (0.95, 0.9, 0.7, 0.8, 0.8), strict=True)
    )
    b.update(200, _frame(200, floor=left_bit_freer, look_ms=200))
    again = b.steer(200, (0.0, 0.0, 0.0))
    assert again.veer < 0.0, again
    # after the way ahead has been clear for a while the commitment is gone
    b2 = B.Boundary()
    b2.update(0, _frame(0, floor=right_freer, look_ms=0))
    b2.steer(0, (0.0, 0.0, 0.0))
    b2.update(300, _frame(300, floor=_line(1.5), look_ms=300))
    b2.steer(300, (0.0, 0.0, 0.0))
    assert b2.steer(4000, (0.0, 0.0, 0.0)).veer == 0.0 and b2._commit == 0


def test_close_to_a_wall_it_turns_the_way_its_arc_can_sweep() -> None:
    """Live 2026-09-28: 0.44 m from the east wall, facing north-east, it turned right toward the
    open floor it had seen in the south — and the arc (this body cannot pivot) swept through east
    into the wall. Close to a wall the way is the side whose next ~50° the walls leave clearer."""
    b = B.Boundary()
    yaw = 0.47  # facing north-east; the wall runs north–south 0.44 m to the east
    for k in range(-40, 41):
        b._add((0.0, 0.0, 0.0), math.atan2(k * 0.05, 0.44), math.hypot(0.44, k * 0.05), "wall", 1.0)
    # it has seen the floor to the south clear (a look from here, before it turned north-east)
    b.rays.append(B._Ray(0.0, 0.0, -1.57, 0.3, 1.5, 1.0))
    steer = b.steer(0, (0.0, 0.0, yaw))
    assert steer.free_ahead < B.NEAR_M
    assert steer.best > 0.0 and steer.veer > 0.0, steer  # left: north, away from the wall


def test_a_known_obstacle_ends_the_floor_look_corridor_and_keeps_wall_steering() -> None:
    b = B.Boundary()
    b.update(0, _frame(0, floor=_line(0.65), look_ms=0))
    steer = b.steer(0, (0.0, 0.0, 0.0))

    # Floor is known through the available corridor to the wall. Do not re-enter the look-stop
    # every time the duck resumes, but retain the obstacle's cautious speed and bend.
    assert steer.known_ahead >= steer.free_ahead > 0.6
    assert steer.look_ahead is False
    assert steer.cap == B.APPROACH_CAP
    assert steer.veer != 0.0
    assert steer.cornered is False


def test_unknown_floor_and_tof_only_obstacle_still_require_a_look() -> None:
    unseen = B.Boundary().steer(0, (0.0, 0.0, 0.0))
    assert unseen.known_ahead == 0.0
    assert unseen.look_ahead is True
    assert unseen.cap == B.APPROACH_CAP

    short_unknown = B.Boundary()
    short_unknown.update(
        0,
        _frame(0, floor=_line(0.4, end="unknown"), look_ms=0),
    )
    short = short_unknown.steer(0, (0.0, 0.0, 0.0))
    assert short.known_ahead < B.KNOWN_AHEAD_M
    assert short.look_ahead is True
    assert short.cap == B.APPROACH_CAP

    tof_only = B.Boundary()
    tof_only.update(0, _frame(0, tof=TofSense(True, None, 400, None)))
    steer = tof_only.steer(0, (0.0, 0.0, 0.0))
    assert steer.known_ahead == 0.0  # a range return is not a floor look
    assert steer.look_ahead is True
    assert steer.cap == B.APPROACH_CAP


def test_zero_clearance_keeps_look_gate_even_with_floor_evidence() -> None:
    class ZeroCorridor(B.Boundary):
        def ahead(self, odom, *, goal=None, rel=None):
            return 0.0

    b = ZeroCorridor()
    b.update(0, _frame(0, floor=_line(0.8), look_ms=0))
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert steer.known_ahead > 0.0
    assert steer.free_ahead == 0.0
    assert steer.look_ahead is True
    assert steer.cap == 0.35
