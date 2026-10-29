"""睡前轉一圈 — it walks one small circle before it sits down to sleep (a:social, 2026-09-28).

The owner's A list, behaviour (9): *circles once before sitting down to sleep*, as an arc loop
sized to the body's tightest safe arc; emerging from the brain (sleepiness), physically performed.

This body cannot turn on the spot (owner: turns are arcs; pet:body — pure yaw from standing never
steps, 0/660), so "turning round before lying down" is a small loop walked at the slowest arc the
gait keeps (vx 0.35–0.40 m/s, |vyaw| 1.0). The loop's size is the body's own and differs by side:
this body drifts right and turns right faster (pet:body steering at vx 0.40: +28 °/s to the left,
−48 °/s to the right; pet:probe path speed ≈ 0.20 m/s), so a right loop is ~0.5 m across and a
left one ~0.8 m. :class:`ArcModel` starts from those measured numbers and keeps learning them from
its own gyro and odometry whenever it walks a full-rate arc (a cerebellum-like calibration: every
duck's body is a little different; a:social's headless physics run measured them again).

:class:`CircleRun` walks the loop *closed-loop on its own heading* (contact odometry's yaw, the
IMU's): it keeps the full-rate arc until it has turned once round (:data:`TURN_TARGET`), whatever
the gait's drift does to the rate — an open-loop "arc for N seconds" turned −74…+20° when the
review measured the old zoomies circle (n = 12). It picks the side whose loop has room (the
boundary memory's clearances, :func:`room_for`), preferring its innate side (Duck Core's quirk
side) when both have room. A fright or no progress ends the loop early (and it just sits); its
body touching something as the loop begins sends it to find room and try again
(:meth:`CircleRun.bumped`); a reflex's moment's hold pauses the loop (:meth:`CircleRun.pause`);
it never takes longer than :data:`TIME_K` × the loop's own time.

Pure; clock-injected; deterministic; standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import msg

CIRCLE_VERSION = "neuro-circle-2"

# The body's arcs as measured (the priors; ArcModel refines them from its own sensors). a:social
# (2026-09-28, headless physics: the official MJCF, robotd's control path, velstand; a constant
# twist from standing, n = 30 per cell, floor friction 0.6–1.2, trunk mass ×0.9–1.15; 0/120
# falls, tilt ≤ 5.0°): vx 0.35 — left (vyaw +1) 26.3 °/s at 0.173 m/s, radius 0.375 m (p10–p90
# 0.357–0.390), 343° in 13.0 s, the loop 0.59 m across; right (vyaw −1) 41.6 °/s at 0.171 m/s,
# radius 0.236 m (0.215–0.251), 343° in 8.4 s, 0.40 m across. vx 0.40: left 0.43 m, right 0.24 m.
# (pet:body's steering step at 0.40: +28.3 / −48.3 °/s.) The tightest safe arc: vx 0.35 to the right.
PRIOR_RATE = {1: math.radians(26.3), -1: math.radians(41.6)}  # rad/s, by side (+1 = left)
PRIOR_SPEED = 0.172  # m/s along the path at vx 0.35
ARC_VX = (0.35, 0.38)  # the loop's forward command: the gait's slowest sure step (the tightest
# arc) … a little more for a lively duck
ARC_VYAW = 1.0
TURN_TARGET = 2.0 * math.pi - 0.3  # once round (≈ 343°): the last few degrees as it settles
TIME_K = 1.9  # it gives up after this × the loop's expected time (+ TIME_EXTRA_S)
TIME_EXTRA_S = 4.0
STALL_S = 3.0  # no heading change for this long while commanded: something stops it …
START_GRACE_S = 3.0  # … (plus this at the start: getting up, the legs' first steps)
# No room here: it walks toward the most open floor it has seen (SEEK_FAN, ≥ SEEK_OPEN_M clear),
# SEEK_M or at most SEEK_MAX_S, then looks for room again — at most SEEK_TRIES times
SEEK_TRIES = 2
SEEK_M = 0.6
SEEK_MAX_S = 7.0
SEEK_OPEN_M = 1.0
SEEK_STEER = 1.2  # rad/s per rad of heading error (an arc toward it)
SEEK_TURN_COST = 0.25  # m of clear floor a radian of turning is worth
SEEK_FAN = (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, math.pi)
STALL_RAD = 0.15
# Its body touched something as the loop began (a wall beside it that the loop's first swing
# brushed; a:social's physics runs: 7 of the 10 loops that ended early ended 0.1 s after the start,
# on the touch reflex): early in the loop it steps away toward open floor and tries again — within
# SEEK_TRIES seeks in all; later in the loop (or out of tries) the loop ends where it is.
BUMP_RESEEK_RAD = 1.0
LEARN_TAU_S = 20.0  # the arc model's memory, in seconds of full-rate arc walking
LEARN_MIN_S = 1.0  # it learns only after the gait has settled into the arc
ROOM_MARGIN_M = 0.1  # the loop needs its own width plus this, clear of walls and furniture (a toy
# in the way is only nudged: a:social's physics runs, 0 falls — see the module doc)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(slots=True)
class ArcModel:
    """How fast this body really turns and walks in a full-rate arc, per side (learned)."""

    rate: dict[int, float] = field(default_factory=lambda: dict(PRIOR_RATE))
    speed: dict[int, float] = field(default_factory=lambda: {1: PRIOR_SPEED, -1: PRIOR_SPEED})
    learned_s: dict[int, float] = field(default_factory=lambda: {1: 0.0, -1: 0.0})
    _arc_side: int = 0
    _arc_since: float = 0.0
    _prev: tuple[float, float, float, float] | None = None  # t, x, y, yaw

    def observe(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        vx_cmd: float,
        vyaw_cmd: float,
        gyro_z: float | None,
        walking: bool,
    ) -> None:
        """One tick of its own motion while it walks a full-rate arc (|vyaw| ≥ 0.9): the yaw
        rate (gyro, else odometry) and the path speed (odometry) refine the side's numbers."""
        prev = self._prev
        self._prev = (t_s, odom[0], odom[1], odom[2])
        side = (1 if vyaw_cmd > 0.0 else -1) if abs(vyaw_cmd) >= 0.9 and vx_cmd >= 0.3 else 0
        if side == 0 or not walking:
            self._arc_side = 0
            return
        if side != self._arc_side:
            self._arc_side, self._arc_since = side, t_s
            return
        if prev is None or t_s - self._arc_since < LEARN_MIN_S:
            return
        dt = t_s - prev[0]
        if dt <= 0.0 or dt > 0.5:
            return
        rate = gyro_z if gyro_z is not None else _wrap(odom[2] - prev[3]) / dt
        speed = math.hypot(odom[0] - prev[1], odom[1] - prev[2]) / dt
        if side * rate <= 0.0 or speed > 0.8:
            return  # not turning that way at all (stuck, lifted): nothing to learn from
        k = 1.0 - math.exp(-dt / LEARN_TAU_S)
        self.rate[side] += k * (side * rate - self.rate[side])
        self.speed[side] += k * (speed - self.speed[side])
        self.learned_s[side] += dt

    def radius(self, side: int) -> float:
        return self.speed[side] / max(0.05, self.rate[side])

    def loop_s(self, side: int) -> float:
        return TURN_TARGET / max(0.05, self.rate[side])

    def view(self) -> dict[str, Any]:
        return {
            side_name: {
                "deg_s": round(math.degrees(self.rate[s]), 1),
                "speed": round(self.speed[s], 3),
                "radius_m": round(self.radius(s), 2),
                "learned_s": round(self.learned_s[s], 1),
            }
            for s, side_name in ((1, "left"), (-1, "right"))
        }


def loop_needs(side: int, radius: float) -> dict[float, float]:
    """The clear way a loop of ``radius`` to ``side`` needs along each heading it sweeps (rad from
    the trunk now → m). The loop is a circle of that radius beside the trunk (its centre ``radius``
    to that side): along a heading θ it reaches 2·r·sin|θ| (the chord), ahead and behind r."""
    out = {0.0: radius + ROOM_MARGIN_M, side * math.pi: radius + ROOM_MARGIN_M}
    for k in (1, 2, 3):
        theta = k * math.pi / 4
        out[side * theta] = 2.0 * radius * math.sin(theta) + ROOM_MARGIN_M
    return out


def room_for(
    side: int,
    radius: float,
    clearance: Callable[[float], float],
    known: Callable[[float], float] | None = None,
) -> tuple[bool, bool]:
    """(whether a loop of ``radius`` to ``side`` has room — no wall it remembers inside the
    footprint, :func:`loop_needs`; whether it has SEEN that side's floor clear — ``known``, the
    boundary memory's seen-clear floor, over the loop's widest chord). ``clearance(heading)``:
    the boundary memory's clear way (m; walls it has seen)."""
    need = loop_needs(side, radius)
    ok = all(clearance(heading) >= want for heading, want in need.items())
    seen = True
    if known is not None:
        wide = side * math.pi / 2
        seen = known(wide) >= 0.5 * need[wide] or known(side * math.pi / 4) >= 0.5 * need[wide]
    return ok, seen


@dataclass(slots=True)
class CircleRun:
    """One loop before sleep (see the module doc). ``phase``: idle → (seeking →) turning → done /
    stopped. With no room for a loop where it stands (a nook, beside the furniture) it first
    walks a little way toward the most open floor it has seen (``seeking``, at most
    :data:`SEEK_TRIES` times) — a duck picks a clear spot to lie down in — and then loops."""

    innate_side: int = 1
    tempo: float = 0.5  # 0..1: a lively duck walks its loop at the cap, a calm one at the floor
    phase: str = "idle"
    side: int = 1
    turned: float = 0.0
    started_s: float = 0.0
    budget_s: float = 0.0
    radius_m: float = 0.3
    seeks: int = 0
    seek_heading: float = 0.0  # odometry heading it walks toward while seeking room
    seek_from: tuple[float, float] = (0.0, 0.0)
    seek_until: float = 0.0
    _yaw: float | None = None
    _last_turn_s: float = 0.0
    _last_turned: float = 0.0
    log: list[str] = field(default_factory=list)

    def _note(self, text: str) -> None:
        self.log.append(text)
        del self.log[:-6]

    def start(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        arcs: ArcModel,
        clearance: Callable[[float], float] | None = None,
        known: Callable[[float], float] | None = None,
    ) -> bool:
        """Choose the side — a side whose loop has room (no wall it remembers in the way), a side
        it has SEEN clear before one it has not, its innate side before the other — and start.
        No room either way: it goes looking for a clearer spot (True, phase ``seeking``) unless it
        has tried enough (False, ``stopped``: it will just sit down where it is)."""
        self.seeks = 0
        return self._begin(t_s, odom, arcs, clearance, known)

    def _begin(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        arcs: ArcModel,
        clearance: Callable[[float], float] | None,
        known: Callable[[float], float] | None,
    ) -> bool:
        order = (self.innate_side, -self.innate_side)
        side = None
        if clearance is None:
            side = order[0]
        else:
            fits = [(s, *room_for(s, arcs.radius(s), clearance, known)) for s in order]
            seen_ok = [s for s, ok, seen in fits if ok and seen]
            any_ok = [s for s, ok, _seen in fits if ok]
            side = seen_ok[0] if seen_ok else any_ok[0] if any_ok else None
        if side is None:
            heading = None if clearance is None else best_open(clearance)
            if heading is None or self.seeks >= SEEK_TRIES:
                self.phase = "stopped"
                self._note("no room")
                return False
            self._seek(t_s, odom, heading)
            return True
        self.side = side
        self.phase = "turning"
        self.turned = 0.0
        self.started_s = t_s
        self.radius_m = arcs.radius(side)
        self.budget_s = TIME_K * arcs.loop_s(side) + TIME_EXTRA_S
        self._yaw = odom[2]
        self._last_turn_s = t_s
        self._last_turned = 0.0
        self._note(f"start {'left' if side > 0 else 'right'} r{self.radius_m:.2f}")
        return True

    def _seek(self, t_s: float, odom: tuple[float, float, float], heading: float) -> None:
        self.seeks += 1
        self.phase = "seeking"
        self.seek_heading = _wrap(odom[2] + heading)
        self.seek_from = (odom[0], odom[1])
        self.seek_until = t_s + SEEK_MAX_S
        self._note(f"seek room {math.degrees(heading):.0f}°")

    def bumped(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        clearance: Callable[[float], float] | None = None,
    ) -> str:
        """Something touched its body while it looped (see :data:`BUMP_RESEEK_RAD`): early in
        the loop it goes looking for room and tries again; else the loop ends here."""
        if self.phase != "turning":
            return self.phase
        heading = None if clearance is None else best_open(clearance)
        if self.turned < BUMP_RESEEK_RAD and self.seeks < SEEK_TRIES and heading is not None:
            self._note(f"bumped {math.degrees(self.turned):.0f}°")
            self._seek(t_s, odom, heading)
        else:
            self.phase = "stopped"
            self._note(f"stopped {math.degrees(self.turned):.0f}° (bumped)")
        return self.phase

    def pause(self, dt: float) -> None:
        """A moment's hold (a reflex has the legs): the loop's clocks wait with it."""
        if self.phase in ("turning", "seeking") and dt > 0.0:
            self.started_s += dt
            self._last_turn_s += dt
            self.seek_until += dt

    def command(self, odom: tuple[float, float, float] | None = None) -> tuple[float, float]:
        """The twist now (vx, vyaw): the loop's full-rate arc while turning; while seeking, a walk
        toward the open floor (an arc where it must turn: no turning on the spot); else (0, 0)."""
        vx = ARC_VX[0] + (ARC_VX[1] - ARC_VX[0]) * _clip(self.tempo)
        if self.phase == "turning":
            return vx, self.side * ARC_VYAW
        if self.phase == "seeking" and odom is not None:
            err = _wrap(self.seek_heading - odom[2])
            return vx, max(-ARC_VYAW, min(ARC_VYAW, SEEK_STEER * err))
        return 0.0, 0.0

    def step(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        *,
        stopped: bool = False,
        arcs: ArcModel | None = None,
        clearance: Callable[[float], float] | None = None,
        known: Callable[[float], float] | None = None,
    ) -> str:
        """Advance on its own heading; returns the phase. ``stopped``: a reflex or a bump holds
        the legs (the loop ends where it is). While seeking, ``arcs``/``clearance``/``known`` are
        asked again once it has walked SEEK_M (or SEEK_MAX_S passed)."""
        if self.phase == "seeking":
            walked = math.hypot(odom[0] - self.seek_from[0], odom[1] - self.seek_from[1])
            if stopped or walked >= SEEK_M or t_s >= self.seek_until:
                # there (or stopped on the way — pressed against something): look for room
                # again from here; after SEEK_TRIES it just sits down
                if stopped:
                    self._note("seek blocked")
                self._begin(t_s, odom, arcs or ArcModel(), clearance, known)
            return self.phase
        if self.phase != "turning":
            return self.phase
        if self._yaw is not None:
            self.turned += self.side * _wrap(odom[2] - self._yaw)
        self._yaw = odom[2]
        if self.turned - self._last_turned > STALL_RAD:
            self._last_turned = self.turned
            self._last_turn_s = t_s
        if self.turned >= TURN_TARGET:
            self.phase = "done"
        elif stopped or t_s - self.started_s > self.budget_s:
            self.phase = "stopped"
        elif t_s - self._last_turn_s > STALL_S + (
            START_GRACE_S if self.turned < STALL_RAD else 0.0
        ):
            self.phase = "stopped"  # (the legs need a moment to start: a grace at the start)
        if self.phase != "turning":
            self._note(
                msg(
                    "neuro.incoming.incoming_skill_circle.CircleRun.step",
                    phase=self.phase,
                    turned=math.degrees(self.turned),
                )
            )
        return self.phase

    def view(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "side": "left" if self.side > 0 else "right",
            "turned_deg": round(math.degrees(self.turned), 1),
            "radius_m": round(self.radius_m, 2),
            "seeks": self.seeks,
            "log": list(self.log[-3:]),
        }


def best_open(clearance: Callable[[float], float]) -> float | None:
    """The heading (rad from the trunk now) with the most open floor it has seen — preferring a
    small turn — or None when nothing is open enough to find room in."""
    best: tuple[float, float] | None = None
    for heading in SEEK_FAN:
        free = clearance(heading)
        if free < SEEK_OPEN_M:
            continue
        score = free - SEEK_TURN_COST * abs(heading)
        if best is None or score > best[1]:
            best = (heading, score)
    return None if best is None else best[0]


__all__ = [
    "ARC_VX",
    "ArcModel",
    "CIRCLE_VERSION",
    "CircleRun",
    "PRIOR_RATE",
    "best_open",
    "loop_needs",
    "room_for",
]
