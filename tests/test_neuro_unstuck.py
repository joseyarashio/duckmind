"""The neural duck gets unstuck (owner bug 2026-10-04: 「頂著東西往前走」, 「走到角落就卡死」, and
it did not turn away from walls by itself).

Found by measuring first (tools/machine_verify/measure_unstuck.py, headless MuJoCo): a wall 0.1 m
ahead held the duck for a minute (the reflex arc turned one way, the wall memory the other, the two
summed to a straight walk the body link refuses within 30 cm of a wall); a low box under the ToF's
beam was pushed along, its touch read as 「靠過去」 (a petting hand: a reward and a liked place);
walks too short for proprio's stall detector; search arcs that ignored the wall. These tests drive
the whole brain and the real body link in a small kinematic room (tests/press_toy.py) and the new
pieces on their own.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from press_toy import DT, Box, PressToy, make_brain

import duckmind
from duckmind.duckbody import protocol as P
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro import reflex as R
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.neuro.boundary import Boundary
from duckmind.neuro.perceive import Perceiver

# ── the whole brain and the body link in a room ───────────────────────────────


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("seed", [1, 2])
def test_a_wall_close_ahead_is_not_a_straight_walk_into_it(side: int, seed: int) -> None:
    """The reflex arc and the wall memory turn away to ONE side: with a wall 0.2 m ahead the walk
    the brain asks for is an arc (|vyaw| ≥ 0.5) — never the two biases cancelling into a straight
    walk the body link then holds (the duck stood at the wall)."""
    toy = PressToy(make_brain(side, seed), x=3.7, y=2.0, yaw=0.0)
    toy.run(25.0)
    weak = [
        r
        for r in toy.rows
        if r.want_vx > 0.05
        and r.tof_ahead is not None
        and r.tof_ahead <= 300
        and abs(r.want_vyaw) < 0.5
    ]
    # (the base brain asks for ~6 s of them: the arc began on its innate side, the wall memory
    # bent the other way; a tick or two while the arc takes the side the memory says is allowed)
    assert len(weak) * DT <= 0.1, (
        f"{len(weak) * DT:.2f} s of straight walks asked for next to a wall"
    )


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("seed", [1, 2])
def test_a_duck_with_its_nose_on_a_wall_backs_off_and_gets_away(side: int, seed: int) -> None:
    """Nose 0.1 m from a wall, ToF 'touch' once it creeps up: it does not stand there. It backs off
    in a small arc and turns away — far from the wall within 30 s."""
    toy = PressToy(make_brain(side, seed), x=3.8, y=2.0, yaw=0.0)
    toy.run(30.0)
    assert max(toy.W - r.x for r in toy.rows[-500:]) > 0.5, "still at the wall after 30 s"
    assert toy.pushing_s() < 3.0


def test_a_back_off_is_a_reverse_arc_never_a_straight_reverse() -> None:
    toy = PressToy(make_brain(1, 1), x=3.8, y=2.0, yaw=0.0)
    toy.run(15.0)
    backing = [r for r in toy.rows if r.vx < -0.05]
    assert backing, "it never backed off"
    assert all(abs(r.vyaw) >= 0.5 for r in backing)
    assert max(abs(r.vx) for r in backing) <= 0.40 + 1e-9


def test_a_low_box_it_cannot_see_is_not_pushed_along() -> None:
    """A box under the ToF's beam: only a gentle touch and its own lack of progress tell it. Six
    walks into it (both innate sides, three seeds): little time pressing."""
    pushed: list[float] = []
    for side in (1, -1):
        for seed in (1, 2, 3):
            box = Box(2.5, 1.7, 2.8, 2.3, hidden=True)
            toy = PressToy(make_brain(side, seed), x=1.9, y=2.0, yaw=0.0, boxes=(box,))
            toy.run(60.0)
            pushed.append(toy.pushing_s())
    assert sum(pushed) / len(pushed) < 2.5, pushed
    assert max(pushed) < 5.0, pushed


def test_touching_a_box_it_walked_into_is_not_being_petted() -> None:
    """The owner's screenshot: 「走走（反射：靠過去）／腦：身上被碰到，靠過去」 while its body is against a
    box. A gentle touch with no hand in view while it walks is a thing it met: no leaning into it,
    no reward, no liked place."""
    brain = make_brain(1, 1)
    leaned = 0
    for k in range(int(40.0 / 0.05)):
        t = k * 50
        # a walk (2.2 s in 3) and a gentle unplaced touch (pet-detect) pulsing on the body as the
        # duck keeps meeting the box: on 0.5 s, off 1.4 s
        walking = (k * 0.05) % 3.0 < 2.2
        touching = (k * 0.05) % 1.9 < 0.5
        frame = C.SensorFrame(
            t_ms=t,
            body=C.BodySense(posture="walk" if walking else "stand", vx=0.35 if walking else 0.0),
            touch=C.TouchSense(site="body", pressure=0.2, source="model")
            if touching
            else C.TouchSense(),
            sources={"body": "model"},
        )
        cmd, _ = brain.step(t, frame)
        leaned += cmd.reflex == "lean_in"
    assert leaned == 0
    assert all(spot.family == 0.0 for spot in brain.learn.places.values())


# ── what a hand does still works ──────────────────────────────────────────────


def _frame(t: int, **kw: object) -> C.SensorFrame:
    body = kw.pop("body", C.BodySense(posture="stand"))
    return C.SensorFrame(t_ms=t, body=body, **kw)  # type: ignore[arg-type]


PARAMS = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)
WALKING = C.BodySense(posture="walk", vx=0.4)
GENTLE = C.TouchSense(site="body", pressure=0.2, source="model")


def _reflexes(frames: list[C.SensorFrame], **kw: object) -> tuple[list[str], list[R.ReflexOutput]]:
    state = R.initial(5)
    outs = []
    for fr in frames:
        state, out = R.step(PARAMS, state, fr, **kw)  # type: ignore[arg-type]
        outs.append(out)
    return [str(o.active) for o in outs], outs


def test_a_gentle_touch_while_it_walks_with_no_hand_in_view_is_its_own_doing() -> None:
    frames = [_frame(t, body=WALKING) for t in range(0, 200, 20)]
    frames += [_frame(t, body=WALKING, touch=GENTLE) for t in range(200, 1200, 20)]
    # an earlier touch made this one "expected" (0.85) — but no hand is in view
    active, outs = _reflexes(frames, walking=True, expect_touch=0.85, expect_hand=0.0)
    assert "lean_in" not in active
    assert any(e.stimulus == "body:brushed" for o in outs for e in o.events)  # a veer away


def test_a_hand_in_view_makes_the_same_touch_one_it_leans_into() -> None:
    frames = [_frame(t, body=WALKING) for t in range(0, 200, 20)]
    frames += [_frame(t, body=WALKING, touch=GENTLE) for t in range(200, 1200, 20)]
    active, _ = _reflexes(frames, walking=True, expect_touch=0.9, expect_hand=0.9)
    assert "lean_in" in active


def test_a_stroke_and_a_firm_poke_are_still_a_stroke_and_a_poke_while_walking() -> None:
    stroke = C.TouchSense(site="head", pressure=0.5, stroke=0.3)
    active, _ = _reflexes(
        [_frame(t, body=WALKING) for t in range(0, 200, 20)]
        + [_frame(t, body=WALKING, touch=stroke) for t in range(200, 1200, 20)],
        walking=True,
        expect_hand=0.0,
    )
    assert "lean_in" in active
    poke = C.TouchSense(site="left", pressure=0.8)
    active, _ = _reflexes(
        [_frame(t, body=WALKING) for t in range(0, 200, 20)]
        + [_frame(t, body=WALKING, touch=poke) for t in range(200, 1200, 20)],
        walking=True,
        expect_hand=0.0,
    )
    assert "withdraw" in active


def test_the_perceiver_does_not_expect_a_touch_after_one_it_walked_into() -> None:
    p = Perceiver()
    first = p.perceive(_frame(0, body=WALKING, touch=GENTLE), 0.0, False)
    assert first.touch_gentle == 0.0  # no one is touching it
    later = p.perceive(_frame(1000, body=C.BodySense(posture="stand")), 1.0, False)
    assert later.expect_touch < 0.5 and later.expect_hand == 0.0
    # …but a touch with nothing walking into anything is one it will expect again
    q = Perceiver()
    q.perceive(_frame(0, touch=GENTLE), 0.0, False)
    assert q.perceive(_frame(1000), 1.0, False).expect_touch >= 0.5


# ── the escape on its own ─────────────────────────────────────────────────────


class _Feeder:
    """A duck that wants to walk, fed to a fresh :class:`Unstick` second by second."""

    def __init__(self) -> None:
        from duckmind.neuro.unstick import Unstick

        self.u = Unstick()
        self.t = 0.0
        self.x = 0.0

    def walk(
        self,
        seconds: float,
        *,
        want: float = 0.35,
        speed: float = 0.0,
        cue: bool = False,
        contact_at: tuple[float, ...] = (),
    ):
        """``seconds`` more of wanting ``want`` and moving ``speed`` m/s (``contact_at``: seconds on
        this feeder's own clock a contact happens). Returns (the Escape or None, the time)."""
        for _ in range(int(round(seconds * 50))):
            self.t += DT
            self.x += speed * DT
            contact = any(abs(self.t - c) < DT / 2 for c in contact_at)
            esc = self.u.step(
                int(round(self.t * 1000)),
                (self.x, 0.0, 0.0),
                active=True,
                want_vx=want,
                cue=cue or contact,
                contact=contact,
            )
            if esc is not None:
                return esc, self.t
        return None, None


def test_a_walk_that_gets_somewhere_is_never_stuck() -> None:
    esc, _ = _Feeder().walk(60.0, speed=0.18, cue=True)
    assert esc is None


def test_a_walk_that_gets_nowhere_against_something_backs_off_within_three_seconds() -> None:
    esc, t = _Feeder().walk(10.0, speed=0.0, cue=True)
    assert esc is not None and esc.why == "pressing" and t is not None and t <= 3.0


def test_a_walk_that_gets_a_third_of_the_way_it_should_is_still_pressing() -> None:
    """Pushing a loose box along: it moves, slowly (~0.05 m/s of the 0.18 m/s the gait makes)."""
    esc, t = _Feeder().walk(10.0, speed=0.05, cue=True)
    assert esc is not None and t is not None and t <= 4.0


def test_with_nothing_there_standing_still_is_not_pressing() -> None:
    esc, _ = _Feeder().walk(30.0, speed=0.0, cue=False)
    assert esc is None


def test_the_first_contact_is_the_reflexes_the_second_at_one_place_is_again() -> None:
    f = _Feeder()
    assert f.walk(1.0, speed=0.18, contact_at=(0.5,))[0] is None
    esc, _t = f.walk(5.0, speed=0.18, contact_at=(2.0,))
    assert esc is not None and esc.why == "again" and esc.count == 1
    # long apart is not "again"
    f = _Feeder()
    assert f.walk(1.0, speed=0.18, contact_at=(0.5,))[0] is None
    assert f.walk(15.0, speed=0.18, contact_at=(14.0,))[0] is None


def test_escapes_grow_at_one_place_and_start_over_somewhere_else() -> None:
    f = _Feeder()
    sizes = []
    for _ in range(4):
        esc, _at = f.walk(8.0, speed=0.0, cue=True)
        assert esc is not None
        sizes.append((esc.back_ms, esc.count))
    assert [c for _b, c in sizes] == [1, 2, 3, 4]
    assert sizes[0][0] < sizes[1][0] <= sizes[2][0]
    assert max(b for b, _c in sizes) <= 1600  # never longer than the reverse arc was measured
    far = _Feeder()
    far.u._escapes.append((0, 5.0, 5.0))  # an earlier escape, 5 m away
    esc, _ = far.walk(8.0, speed=0.0, cue=True)
    assert esc is not None and esc.count == 1


def test_it_is_not_asked_twice_while_it_is_still_getting_out() -> None:
    f = _Feeder()
    esc, _t = f.walk(8.0, speed=0.0, cue=True)
    assert esc is not None
    esc2, _ = f.walk(1.5, speed=0.0, cue=True)
    assert esc2 is None


def _circle(stationary: bool, seconds: float, *, cue: bool = True, contact_every_s: float = 0.0):
    """A duck that wants to walk and goes round a 0.10 m circle at 0.18 m/s (it gets somewhere:
    the chord of every 0.6 s is most of its path, so it is not pressing). Returns the first
    escape and its time, or (None, None)."""
    from duckmind.neuro.unstick import Unstick

    u = Unstick()
    radius, speed = 0.10, 0.18
    t = 0.0
    last_contact = -99.0
    for _ in range(int(round(seconds * 50))):
        t += DT
        a = speed * t / radius
        contact = contact_every_s > 0.0 and t - last_contact >= contact_every_s
        if contact:
            last_contact = t
        esc = u.step(
            int(round(t * 1000)),
            (radius * math.cos(a), radius * math.sin(a), 0.0),
            active=True,
            want_vx=0.35,
            cue=cue or contact,
            contact=contact,
            stationary=stationary,
        )
        if esc is not None:
            return esc, t
    return None, None


def test_going_round_in_a_small_circle_while_something_is_there_is_confined() -> None:
    esc, t = _circle(False, 30.0)
    assert esc is not None and esc.why == "confined" and t is not None and t <= 12.0


def test_an_act_that_stays_on_its_spot_on_purpose_is_not_confined() -> None:
    esc, _ = _circle(True, 60.0)
    assert esc is None


def test_pushing_into_something_still_counts_in_an_act_that_stays_on_its_spot() -> None:
    f = _Feeder()
    for _ in range(int(round(10.0 * 50))):
        f.t += DT
        esc = f.u.step(
            int(round(f.t * 1000)),
            (0.0, 0.0, 0.0),
            active=True,
            want_vx=0.35,
            cue=True,
            contact=False,
            stationary=True,
        )
        if esc is not None:
            assert esc.why == "pressing" and f.t <= 3.0
            return
    raise AssertionError("a stationary act that pushes into a wall was never told to back off")


def test_a_second_contact_at_one_place_is_still_again_in_an_act_that_stays_on_its_spot() -> None:
    esc, t = _circle(True, 20.0, contact_every_s=2.0)
    assert esc is not None and esc.why == "again" and t is not None and t <= 6.0


@pytest.mark.skipif(
    not Path("tools/machine_verify/measure_unstuck.py").is_file(),
    reason="tools/machine_verify is not in this checkout",
)
def test_the_acts_that_turn_or_work_on_one_spot_say_so_on_their_spec() -> None:
    import sys

    sys.path.insert(0, "tools/machine_verify")
    import measure_unstuck as M

    assert M.stationary_acts() == frozenset({"sunbathe", "circle_settle", "carry_toy", "spin"})
    assert not C.BEHAVIOURS["wander"].stationary and not C.BEHAVIOURS["investigate"].stationary


# ── the wall memory keeps where it was blocked ────────────────────────────────


def test_where_it_was_blocked_stays_a_wall_a_floor_look_cannot_erase() -> None:
    b = Boundary()
    b.note_block(0, (0.0, 0.0, 0.0))
    sector = C.FreeSector(bearing=0.0, width=0.13, free_m=0.9, end="unknown")
    frame = C.SensorFrame(
        t_ms=1000,
        body=C.BodySense(posture="stand", odom=(-0.12, 0.0, 0.0)),  # it backed off 12 cm
        camera=C.CameraSense(usable=True, floor=(sector,), floor_t_ms=1000),
    )
    b.update(1000, frame)
    assert b.clearance((-0.12, 0.0, 0.0), 0.0) < 0.4  # still a wall just ahead
    b.update(40_000, replace(frame, t_ms=40_000, camera=C.CameraSense()))
    assert b.clearance((-0.12, 0.0, 0.0), 0.0) > 1.0  # forgotten after a while


# ── the body link lets a back arc through ────────────────────────────────────


def _link_state(t: float) -> RobotState:
    joints = [0.0] * 15
    for i, home in enumerate((0.3491, 0.3491, 0.0, 0.0)):
        joints[5 + i] = home
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = 4.5
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": "stand",
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
    )


def test_a_back_arc_is_not_held_by_the_tof_that_looks_ahead() -> None:
    """A ToF 'touch' (a face pressed on a wall) re-armed the 0.5 s stall hold every time it ended:
    a duck pressed on something could not back away at all. Backing, the ToF says nothing about
    where it goes (duck-body's verdict says so)."""
    link = BodyLink()

    def depth(t: float) -> P.DepthView:
        return P.DepthView(
            t_ns=10**10 + int(round(t * 1e9)),
            ahead_mm=30,
            left_mm=None,
            right_mm=None,
            drop_zones=0,
            touch=True,
            head_pitch_mrad=0,
        )

    twists = []
    for k in range(200):
        t = k * DT
        # forward for the first half second (the touch holds it), then a back arc is asked for
        req = (
            BodyRequest(vx=0.35, vyaw=0.0)
            if t < 0.5
            else BodyRequest(vx=-0.4, vyaw=1.0, back_arc=True)
        )
        out = link.tick(_link_state(t), req, depth=depth(t))
        twists.append((t, out.twist))
    backing = [t for t, tw in twists if tw[0] < -0.05 and t > 0.5]
    assert backing and backing[0] < 1.6, "the back arc never got through the ToF touch"
    assert all(tw == (0.0, 0.0) for t, tw in twists if t < 0.5)  # a forward walk still did not


# ── a search for the nest does not sweep into the wall ───────────────────────


@pytest.mark.parametrize("side", [1, -1])
def test_a_search_arc_does_not_turn_into_the_wall_beside_it(side: int) -> None:
    """Looking for a nest it neither sees nor remembers, the duck turns round in tight arcs between
    sweeps of its head. That arc ignored the ToF, the wall memory and every reflex veer: with a wall
    close on the side of its own arc it swept into it (2026-10-04)."""
    brain = make_brain(side, 1, "return_dock", strength=12.0)
    wall_side = side  # the wall is on the side the duck's own arc turns to
    into = 0
    seeking = 0
    for k in range(int(30.0 / 0.05)):
        t = k * 50
        near = C.TofSense(
            present=True,
            left_mm=250 if wall_side > 0 else None,
            right_mm=250 if wall_side < 0 else None,
            ahead_mm=None,
            age_ms=0,
        )
        frame = C.SensorFrame(
            t_ms=t,
            body=C.BodySense(posture="stand"),
            tof=near,
            sources={"body": "model"},
        )
        cmd, _ = brain.step(t, frame)
        seeking += brain._seek_t0 is not None
        if k * 0.05 > 4.0 and cmd.vx > 0.3 and cmd.vyaw * wall_side > 0.3:
            into += 1
    assert seeking > 100, "it never went looking for its nest"
    assert into == 0, f"{into * 0.05:.1f} s of an arc turning toward the wall beside it"


# ── the second round (measured with the real skills) ────────────────────────────


def test_a_walk_the_body_stands_still_under_is_pressing_after_a_second_of_asking() -> None:
    """The body link holds a straight walk within 30 cm of a wall; the skills' walk requests come
    in short pulses. Nothing applied under the request: no gait spin-up to wait for."""
    f = _Feeder()
    pulse = 0.0
    fired = None
    for k in range(int(10.0 * 50)):
        f.t += DT
        asking = (k // 25) % 2 == 0  # 0.5 s on, 0.5 s off
        esc = f.u.step(
            int(round(f.t * 1000)),
            (0.0, 0.0, 0.0),
            active=True,
            want_vx=0.4 if asking else 0.0,
            cue=True,
            contact=False,
            held=True,
        )
        if esc is not None:
            fired = (f.t, esc)
            break
        pulse += DT
    assert fired is not None and fired[0] <= 4.5 and fired[1].why == "pressing"


def test_where_it_was_blocked_does_not_make_the_duck_cornered() -> None:
    b = Boundary()
    for k in range(6):  # the same pose six times: one set of points, not six
        b.note_block(0, (0.0, 0.0, 0.0))
    assert len(b.blocks) == 3
    b.note_block(0, (0.0, 0.0, 0.0))
    steer = b.steer(1000, (0.0, 0.0, 0.0))
    assert steer.free_ahead < 0.2  # it steers by them …
    assert not steer.cornered  # … but they are not a wall it has seen: no stopping for good


def test_a_floor_look_that_sees_clear_past_a_block_far_from_the_lens_clears_it() -> None:
    b = Boundary()
    b.note_block(0, (0.6, 0.0, 0.0))  # blocked 0.74 m ahead of the origin
    sector = C.FreeSector(bearing=0.0, width=0.13, free_m=2.0, end="unknown")
    seen_from_origin = C.SensorFrame(
        t_ms=500,
        body=C.BodySense(posture="stand", odom=(0.0, 0.0, 0.0)),
        camera=C.CameraSense(usable=True, floor=(sector,), floor_t_ms=500),
    )
    b.update(500, seen_from_origin)
    assert not b.blocks  # 0.74 m from the lens is far enough to see: it is open after all


def test_an_escape_goes_on_through_a_skills_walk() -> None:
    brain = make_brain(1, 1)
    brain._esc_side = 1
    brain._esc_until = 5000
    cmd = C.MotorCommand(t_ms=1000, behaviour="play_ball", bout=1, variant=0, vx=0.4, vyaw=0.1)
    out = brain.escape_command(cmd)
    assert out.vx >= 0.35 and out.vyaw >= 0.8 and not out.approach  # an arc away, not straight on
    later = C.MotorCommand(t_ms=9000, behaviour="play_ball", bout=1, variant=0, vx=0.4, vyaw=0.1)
    assert brain.escape_command(later) == later  # …only while it is still turning away
    brain._back = (4000, 1.0)
    brain._back_goal = (0.0, 0.0, 0.1)
    back = brain.escape_command(cmd)
    assert back.vx < 0.0 and back.back_arc and abs(back.vyaw) >= 0.5  # and the reverse arc first


# ── the production brain (the skills installed): in a subprocess ───────────────
# install_default_skills is permanent for a process; in this one it would change the behaviour
# tables other tests read (tests/test_neuro_integration.py does the same)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(duckmind.__file__).resolve().parents[1]  # the code under test, wherever it is


def _in_subprocess(body: str) -> dict:
    code = (
        "import json, sys\n"
        "sys.path.insert(0, 'tests')\n"
        "from press_toy import PressToy, make_integrated_brain\n"
        + body
        + "\nprint('JSON' + json.dumps(out))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(x for x in reversed(done.stdout.splitlines()) if x.startswith("JSON"))
    return json.loads(line[4:])


def test_the_production_brain_gets_away_from_a_wall_and_does_not_walk_straight_into_it() -> None:
    out = _in_subprocess(
        """
out = {"weak": [], "far": []}
for side in (1, -1):
    for seed in (1, 2):
        toy = PressToy(make_integrated_brain(side, seed), x=3.7, y=2.0, yaw=0.0)
        toy.run(25.0)
        out["weak"].append(sum(1 for r in toy.rows if r.want_vx > 0.05 and r.tof_ahead is not None
                               and r.tof_ahead <= 300 and abs(r.want_vyaw) < 0.5))
        toy = PressToy(make_integrated_brain(side, seed), x=3.8, y=2.0, yaw=0.0)
        toy.run(30.0)
        out["far"].append(max(toy.W - r.x for r in toy.rows[-500:]))
"""
    )
    assert all(n * DT <= 0.1 for n in out["weak"]), out["weak"]  # (the base: ~6 s)
    assert all(f > 0.5 for f in out["far"]), out["far"]  # (the base: 0.1 m, stuck)


def test_the_production_brain_does_not_push_a_low_box_along() -> None:
    out = _in_subprocess(
        """
from press_toy import Box
out = {"pushed": []}
for side in (1, -1):
    for seed in (1, 2, 3):
        toy = PressToy(make_integrated_brain(side, seed), x=1.9, y=2.0, yaw=0.0,
                       boxes=(Box(2.5, 1.7, 2.8, 2.3, hidden=True),))
        toy.run(60.0)
        out["pushed"].append(toy.pushing_s())
"""
    )
    # (the base brain with the skills: 4.5 s of six runs' mean; one run now rubs along the box's
    # face for 8 s — the toy counts a slide along it as pushing)
    assert sum(out["pushed"]) / len(out["pushed"]) < 4.0, out["pushed"]
