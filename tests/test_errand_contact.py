"""A walk to a place somebody pointed at, among things (GOTHERE-CONTACT-2026-11-07): the reactions that ate an errand's
progress on the v8 walk are errand-aware where it is safe, and the real contacts keep today's answers.

A small closed loop: the whole neural brain (the bold explorer of ``press_toy``) with the quick turn (``can_pivot``), a body
that turns on the spot while the brain asks a pivot and walks / backs as the twist says, and walls the ToF sees. Truth (the
pose) is read only by the tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest
from press_toy import PressToy, make_brain

from duckmind.neuro import contract as C
from duckmind.neuro.brain import NeuroBrain

DT = 0.02
SENSOR = 0.08  # the ToF sits this far ahead of the trunk centre
NOSE = 0.10


@dataclass
class Wall:
    """A wall segment from (x0, y0) to (x1, y1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def hit(self, ox: float, oy: float, ang: float) -> float | None:
        dx, dy = math.cos(ang), math.sin(ang)
        ex, ey = self.x1 - self.x0, self.y1 - self.y0
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            return None
        t = ((self.x0 - ox) * ey - (self.y0 - oy) * ex) / den
        u = ((self.x0 - ox) * dy - (self.y0 - oy) * dx) / den
        return t if t > 0.0 and 0.0 <= u <= 1.0 else None


@dataclass
class Loop:
    """The duck at (0, 0) facing +x, walls around; ``turn_rate``: how fast its turn on the spot goes (rad/s; 0 = it is held)."""

    brain: NeuroBrain
    walls: list[Wall]
    turn_rate: float = 1.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    t: float = 0.0
    applied: tuple[float, float] = (0.0, 0.0)
    gz: float = 0.0
    cmds: list[C.MotorCommand] = field(default_factory=list)
    # what the world does to the duck this tick (a drop ahead, a lift, a fall, a bang): applied to the frame it senses
    inject: Callable[[C.SensorFrame], C.SensorFrame] | None = None

    def _ray(self, ang: float) -> int | None:
        ox, oy = self.x + SENSOR * math.cos(self.yaw), self.y + SENSOR * math.sin(self.yaw)
        hits = [h for w in self.walls if (h := w.hit(ox, oy, ang)) is not None]
        r = min(hits, default=9.0)
        return None if r >= 1.5 else int(r * 1000)

    def _free(self, x: float, y: float, yaw: float) -> bool:
        """The nose has room (truth: the body cannot go into a wall)."""
        nx, ny = x + NOSE * math.cos(yaw), y + NOSE * math.sin(yaw)
        for w in self.walls:
            h = w.hit(x, y, math.atan2(ny - y, nx - x))
            if h is not None and h <= NOSE + 0.01:
                return False
        return True

    def step(self) -> C.MotorCommand:
        ahead = self._ray(self.yaw)
        walking = abs(self.applied[0]) > 0.05
        frame = C.SensorFrame(
            t_ms=int(round(self.t * 1000)),
            body=C.BodySense(
                posture="walk" if walking or self.gz else "stand",
                vx=self.applied[0],
                vyaw=self.applied[1],
                gyro=(0.0, 0.0, self.gz),
                odom=(self.x, self.y, self.yaw),
            ),
            tof=C.TofSense(
                present=True,
                left_mm=self._ray(self.yaw + 0.5),
                ahead_mm=ahead,
                right_mm=self._ray(self.yaw - 0.5),
                touch=ahead is not None and ahead < 45,
                age_ms=0,
            ),
            sources={"body": "model"},
        )
        if self.inject is not None:
            frame = self.inject(frame)
        cmd, _tel = self.brain.step(frame.t_ms, frame)
        self.cmds.append(cmd)
        vx, vyaw = float(cmd.vx), float(cmd.vyaw)
        self.gz = 0.0
        if vx < 0.0:  # a reverse arc: ~0.1 m/s back, turning as asked
            self.applied = (vx, vyaw)
            self.gz = 0.6 * vyaw
            self.yaw += self.gz * DT
            self.x -= 0.1 * math.cos(self.yaw) * DT
            self.y -= 0.1 * math.sin(self.yaw) * DT
        elif cmd.pivot:  # the walk's own turn in place
            self.applied = (0.0, 0.0)
            self.gz = self.turn_rate * (1.0 if cmd.pivot > 0.0 else -1.0)
            self.yaw += self.gz * DT
        elif vx > 0.0:
            self.gz = 0.6 * vyaw
            nyaw = self.yaw + self.gz * DT
            nx = self.x + 0.31 * vx * math.cos(nyaw) * DT
            ny = self.y + 0.31 * vx * math.sin(nyaw) * DT
            if self._free(nx, ny, nyaw):
                self.x, self.y, self.yaw = nx, ny, nyaw
                self.applied = (vx, vyaw)
            else:  # pushing on the wall: no ground, the legs still asked
                self.applied = (0.0, 0.0)
                self.gz = 0.0
        else:
            self.applied = (0.0, 0.0)
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.t += DT
        return self.cmds[-1]

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / DT))):
            self.step()

    def bearing_to(self, px: float, py: float) -> float:
        b = math.atan2(py - self.y, px - self.x) - self.yaw
        return math.atan2(math.sin(b), math.cos(b))


def _corner_duck(turn_rate: float = 1.0) -> Loop:
    """Nose 0.2 m from a wall, a second wall close on its right: a corner (v8 errands: the place behind it)."""
    brain = make_brain(side=1, seed=1)
    brain.can_pivot = True
    walls = [Wall(0.3, -1.0, 0.3, 1.0), Wall(-0.1, -0.25, 0.3, -0.25)]
    loop = Loop(brain, walls, turn_rate=turn_rate)
    loop.run(0.2)
    return loop


def _send(loop: Loop, bearing: float, range_m: float) -> tuple[float, float]:
    """Point at a place ``bearing`` / ``range_m`` from the trunk (truth: where it is)."""
    loop.brain.go_to(bearing, range_m)
    a = loop.yaw + bearing
    return loop.x + range_m * math.cos(a), loop.y + range_m * math.sin(a)


def test_a_duck_turning_on_the_spot_to_face_the_place_is_not_stuck() -> None:
    """Nose in a corner and the place behind it: it turns on the spot to face the place (the quick turn; v8's own turn in
    place). The turn asks the walk's slowest arc speed and makes no ground with the walls in sight — before, the unstick sense
    took that for pressing on the wall and backed off in a reverse arc. A turn that turns is not stuck: it faces the place and
    no escape or reverse arc comes."""
    loop = _corner_duck()
    escapes = loop.brain._unstick.fired
    px, py = _send(loop, -2.6, 1.0)
    loop.run(4.0)
    assert loop.brain._unstick.fired == escapes, (
        "an escape fired while it turned on the spot to face the place"
    )
    assert not any(c.vx < 0.0 for c in loop.cmds[-200:]), (
        "a reverse arc while it turned to face the place"
    )
    assert abs(loop.bearing_to(px, py)) < 0.5, "it did not turn to face the place"


def test_a_turn_on_the_spot_that_does_not_turn_is_still_pressing() -> None:
    """The same corner, but the turn is held (the body does not turn: wedged): it is stuck, and it backs off as today."""
    loop = _corner_duck(turn_rate=0.0)
    escapes = loop.brain._unstick.fired
    _send(loop, -2.6, 1.0)
    loop.run(4.0)
    assert loop.brain._unstick.fired > escapes
    assert any(c.vx < 0.0 and c.back_arc for c in loop.cmds[-200:])


def test_walking_into_a_wall_on_an_errand_still_backs_off() -> None:
    """A real press on an errand keeps today's answer (``press_toy``: the real body link, its stall and ToF verdicts): the
    place is straight ahead beyond the wall, it walks at it, its nose meets the wall and gets nowhere — it backs off in a
    reverse arc and is not left pushing."""
    brain = make_brain(side=1, seed=1)
    brain.can_pivot = True
    toy = PressToy(brain, x=3.7, y=2.0, yaw=0.0)
    toy.run(0.2)
    brain.go_to(0.0, 1.0)
    toy.run(12.0)
    assert brain._unstick.fired >= 1
    assert any(r.want_vx < 0.0 for r in toy.rows)
    assert toy.pushing_s() < 3.0


def _drop(f: C.SensorFrame) -> C.SensorFrame:
    return replace(f, tof=replace(f.tof, drop_zones=6))


def _lift(f: C.SensorFrame) -> C.SensorFrame:
    return replace(f, events=("lifted",))


def _fall(f: C.SensorFrame) -> C.SensorFrame:
    return replace(
        f, events=("fell",), body=replace(f.body, fallen=True, tilt=1.4, posture="fallen")
    )


def _bang(f: C.SensorFrame) -> C.SensorFrame:
    return replace(f, sound=C.SoundSense(level_db=92.0, onset=1.0, kind="bang"))


@pytest.mark.parametrize(
    ("world", "reflex"),
    [(_drop, "drop_stop"), (_lift, "lifted"), (_fall, "righting"), (_bang, "startle")],
)
def test_on_an_errand_a_drop_a_lift_a_fall_and_a_fright_keep_todays_answer(
    world: Callable[[C.SensorFrame], C.SensorFrame], reflex: str
) -> None:
    """In the middle of an errand's turn on the spot in a corner (where the errand-aware answers act), the floor missing
    ahead, a lift, a fall and a bang are answered as always: that reflex takes the body and the legs stop."""
    loop = _corner_duck()
    _send(loop, -2.6, 1.0)
    loop.run(0.6)
    assert any(c.pivot for c in loop.cmds[-10:]), "not turning on the spot when it came"
    loop.inject = world
    loop.run(0.1 if world is _bang or world is _lift else 0.4)
    loop.inject = None
    loop.run(0.3)
    got = [c for c in loop.cmds[-25:] if c.reflex == reflex]
    assert got, f"no {reflex}"
    assert all(c.vx <= 0.0 and not c.pivot for c in got if reflex != "lifted")


@pytest.mark.parametrize("bearing", [-0.5, 0.5, -1.0])
def test_after_backing_off_on_an_errand_it_turns_away_and_not_back_into_the_thing(
    bearing: float,
) -> None:
    """The place beyond a wall: it walks at it, presses, backs off in a reverse arc and then keeps turning away to the
    escape's side for a moment (the way round). Before, the errand's own turn on the spot to face the place went the other
    way at once — back into the wall it had just backed off (on v8, 36 % of the escapes' turning away on errands). While an
    escape turns away, the errand never pivots against it."""
    brain = make_brain(side=1, seed=1)
    brain.can_pivot = True
    asked: list[tuple[float, int]] = []
    step = brain.step

    def logged(t_ms: int, frame: C.SensorFrame):
        cmd, tel = step(t_ms, frame)
        turning_away = brain._esc_side if t_ms < brain._esc_until and brain._back is None else 0
        asked.append((float(cmd.pivot), int(turning_away)))
        return cmd, tel

    brain.step = logged  # type: ignore[method-assign]
    toy = PressToy(brain, x=3.7, y=2.0, yaw=0.0)
    toy.run(0.2)
    brain.go_to(bearing, 1.0)
    toy.run(15.0)
    assert brain._unstick.fired >= 1
    away = [(p, s) for p, s in asked if s]
    assert away, "no escape turned away"
    against = sum(1 for p, s in away if p * s < 0.0)
    assert against == 0, (
        f"{against} ticks of the errand pivoting back against the escape's turn away"
    )


def _soft_brain() -> NeuroBrain:
    """The bold explorer with the live duck's lasting-map setting (``soft``: the clutter escape takes the legs on a route)."""
    import dataclasses

    from press_toy import _BOLD_EXPLORER

    from duckmind.neuro.genome import Genome

    genome = dataclasses.replace(
        Genome.from_axes(_BOLD_EXPLORER, primary="explorer", branch="explorer", gd=120),
        side=1,
        seed=1,
    )
    brain = NeuroBrain(genome, 1, lastmap="soft")
    brain.sel.set_extra({"wander": 2.0})
    brain.can_pivot = True
    return brain


def _knot(brain: NeuroBrain, toy: PressToy) -> None:
    """Four contacts at the place it stands, over the last few seconds: a knot (what two escapes leave behind)."""
    t = int(round(toy.t * 1000))
    x, y = toy.odom[0], toy.odom[1]
    for k in range(4):
        brain.clutter.note(t - 6000 + 1600 * k, x, y, "bump")


def test_a_clutter_escape_on_an_errand_ends_once_the_way_to_the_place_is_clear() -> None:
    """On a walk to a place, a knot of contacts behind it starts the clutter escape — and with the way straight to the place
    open it is out at once, back on the walk (before: the escape kept the legs until it was 1 m from the knot, a median 21 s
    of a 40 s errand on v8)."""
    brain = _soft_brain()
    toy = PressToy(brain, x=1.0, y=2.0, yaw=0.0)
    toy.run(0.5)
    brain.go_to(0.0, 1.5)
    toy.run(1.0)
    started = brain.clutter.started
    _knot(brain, toy)
    escape_s = 0.0
    for _ in range(int(3.0 / DT)):
        toy.step()
        escape_s += DT if brain.clutter.mode == "escape" else 0.0
    assert brain.clutter.started > started, "no clutter escape began"
    assert escape_s <= 0.2, (
        f"the clutter escape kept the legs {escape_s:.1f} s with the way to the place open"
    )


def test_a_clutter_escape_off_an_errand_is_todays() -> None:
    """The same knot on a plain wander (no errand): the escape takes the legs as before, until it is out of the knot."""
    brain = _soft_brain()
    toy = PressToy(brain, x=1.0, y=2.0, yaw=0.0)
    toy.run(1.5)
    _knot(brain, toy)
    escape_s = 0.0
    for _ in range(int(3.0 / DT)):
        toy.step()
        escape_s += DT if brain.clutter.mode == "escape" else 0.0
    assert escape_s >= 1.0
