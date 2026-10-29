"""Confirm a body's pickup claim for growth and localisation, using the same leg-load evidence - and re-test a claim that persists.

The body claims ``lifted`` from the legs alone (they went light, or they work hard with the trunk still), so a duck pressed against a
wall, whose load drops for a moment, is "lifted" too (docs/design/FALSE-LIFT-2026-10-11.md). A duck held in a hand keeps its legs
light (seated in the hand about 0.4 of a quiet stand); a duck that sits down on the floor or on a sill loads them again (about 0.7
and more). So a claim is borne out when the legs stay light for :data:`LIFT_LIGHT_MS` without a break within :data:`LIFT_CONFIRM_MS`
of it, and is false when the floor, a fall or a righting ends it first, or the time runs out.

:class:`FloorProbe` is for the claim that does not end. A duck seated with one side on a ledge (the dock's guide, a door sill) reads
light legs too (0.42-0.53 of a quiet stand, against 0.38-0.40 in a hand) and its legs never load to the floor band that sets it down
(docs/design/neural-duck/DOCK-FALL-2026-10-11.md §7.6). The level cannot tell the two; what differs is whether the legs carry the trunk
at all. Raise the head to the seated envelope's edge: the weight moves back, and the load of a duck that rests on something rises
(+15 to +31 % of its load at HOME at the lowest read of a 0.5 s window, 23 ledge and sill seats), the load of a duck in a hand does not
(-0.5 %, 3 holds).

Pure: the caller gives the clock (``t_ms``), the leg load (1.0 = a quiet stand; None = not known) and, for the probe, the measured head.
"""

from __future__ import annotations

import math

# the leg load (the unit of the body sense, 1.0 = a quiet stand) below which the legs carry (almost) nothing. proprio's own claim levels
# are 20/35 = 0.57 (standing) and 18.5/35 = 0.53 (seated); a duck held in the hand reads ≈ 0.4 (13/35), seated on a floor ≈ 0.7 (25/35):
# the middle of the two
LIFT_LOAD = 0.55
LIFT_LIGHT_MS = 500  # light this long without a break
# … within this long of the claim (the sit in the hand swings the legs for ~0.8 s first)
LIFT_CONFIRM_MS = 3000
# the body events that end a claim: the floor again, a fall, a righting
LIFT_DROP_EVENTS = ("set_down", "fell", "righted", "righted_help", "righting_failed")
# a claim that ended at a set-down after the legs had read light this long (but not the full LIFT_LIGHT_MS) was a short carry: a hand
# held the duck, only for a moment. The recorded false claims (tests/fixtures/lift/traces.json) read no light at all after the claim; the
# world's hand holding the duck for 0.2 s read 0.44 s (headless, full stack). A quarter second is 12 ticks, not a one-tick blip.
LIFT_SHORT_MS = 250


class LiftWitness:
    """One open claim: :meth:`step` every tick until it says "yes" (borne out) or "no" (false).

    ``light_max_ms`` is the longest run of light legs read since the claim (below :data:`LIFT_LIGHT_MS` unless the verdict is "yes");
    it is only for the caller to look at, it changes no verdict."""

    __slots__ = ("light_max_ms", "light_since", "t_ms")

    def __init__(self, t_ms: int) -> None:
        self.t_ms = t_ms
        self.light_since: int | None = None
        self.light_max_ms = 0

    def step(self, t_ms: int, load: float | None, *, ended: bool) -> str | None:
        """``ended``: the floor, a fall or a righting ended the claim this tick. None while it is still open."""
        if ended or t_ms - self.t_ms > LIFT_CONFIRM_MS:
            return "no"
        if load is None or load >= LIFT_LOAD:
            self.light_since = None
            return None
        if self.light_since is None:
            self.light_since = t_ms
        run = t_ms - self.light_since
        self.light_max_ms = max(self.light_max_ms, run)
        if run >= LIFT_LIGHT_MS:
            return "yes"
        return None


# -- the re-test of a claim that persists (FloorProbe) --
Offsets = tuple[float, float, float, float]
# robot.head offsets from HOME (neck_pitch, head_pitch, head_yaw, head_roll), rad: the head up at the seated envelope's edge
# (content.SEATED_ENVELOPE: neck_pitch 0.25, head_pitch 0.35)
PROBE_HOME: Offsets = (0.0, 0.0, 0.0, 0.0)
PROBE_NOD: Offsets = (-0.25, -0.35, 0.0, 0.0)
# the measured head counts as at HOME / at the nod within this (neck_pitch, head_pitch, head_yaw, head_roll; rad). The servos sag and the
# neck is loose (bodylink.LinkConfig.home_tol); the nod's two pitch joints are read the same way
PROBE_HOME_TOL = (0.25, 0.10, 0.10, 0.10)
PROBE_NOD_TOL = 0.10
# the head must stay at a pose this long (the servos and the trunk settle), then the load is read over the next LIFT_LIGHT_MS
PROBE_SETTLE_MS = 500
PROBE_WINDOW_MS = LIFT_LIGHT_MS
# a pose that is not reached within this ends the probe unjudged
PROBE_POSE_TIMEOUT_MS = 4000
# the legs have answered when EVERY load of the raised head's window is this much (of the load at HOME) above it. Headless, the lowest read
# of the window: 23 ledge and sill seats +14.6 to +31.5 %, 3 hand-held ducks -0.5 %, a floor seat -0.9 %. Between the two, nearer the hand
PROBE_RESPONSE = 0.08
# a window whose load moves more than this (of the load at HOME) is not read: a hand that shifts the duck answers nothing, either way
PROBE_STEADY = 0.15
# the mean gyro (rad/s) over the probe must stay under this: a trunk that is being carried is not read (proprio's set-down calm)
PROBE_CALM_GYRO = 0.3
# A TILTED duck (bodylink: the trunk past proprio's ``lift_max_tilt_deg``, where its own floor bands are off too) cannot be asked for a pose:
# lying on its beak at 62.5 degrees the trunk's weight pins the neck and the sit-stand network ignores the head command (headless: the neck
# rests at +0.71 rad from HOME whatever is asked, the HOME tolerance is 0.25), so the probe never judged it. The head is then not read by where
# it is but by whether it has come to rest: within PROBE_NOD_TOL of where it was for the settle and the window. The legs' answer to the same
# two head commands is read against the same PROBE_RESPONSE, and it is not enough alone: a duck tipped on its side in a hand also moves its
# small leg load with the head (headless: +10 to +50 % of a load of 7 to 15), because the legs hang. What a hand does not do is let the trunk
# answer: a duck that lies on something rocks on it when the head's weight moves (headless, lying on its beak: 0.17 to 0.38 degrees between
# the two windows' mean gravity directions), a duck welded to a hand moves 0.000 to 0.002. So a tilted duck is "supported" only when the legs
# answer AND the trunk answers by this much (the sine of the angle between the two windows' mean gravity directions, in degrees); where the
# trunk does not answer, the verdict is "held" (the claim stands: the safe side, a hand is never let go).
PROBE_TRUNK_ANSWER_DEG = 0.05


class FloorProbe:
    """One re-test of a claim that persists: does anything under the duck bear its weight?

    The caller commands :meth:`head` (HOME, then the raised head) and calls :meth:`step` every tick with the load, the MEASURED head and
    (tilted trunk only) the gravity direction in the trunk frame.
    The verdict: ``"supported"`` (every load of the raised head's window is :data:`PROBE_RESPONSE` above the load at HOME: a floor or a
    ledge bears the duck, no hand holds it; a tilted trunk also rocked by :data:`PROBE_TRUNK_ANSWER_DEG`), ``"held"`` (not: the legs carry
    nothing the head moves, or the trunk does not answer) or ``"unclear"`` (the claim ended, the load was not known, a pose was not reached,
    a window moved or the trunk swung: nothing is judged). None until then.
    The load is in any one unit (only ratios are read). ``tilted``: the trunk is tilted, see the note above (a head at rest is a head at the pose)."""

    __slots__ = (
        "base_load",
        "nod_low",
        "phase",
        "tilted",
        "trunk_shift_deg",
        "_anchor",
        "_down_home",
        "_down_sum",
        "_gyro",
        "_pose_since",
        "_pose_t0",
        "_win",
    )

    def __init__(self, t_ms: int, *, tilted: bool = False) -> None:
        self.phase = "home"  # home, then nod
        self.tilted = tilted
        self.base_load: float | None = None  # the load at HOME, once read
        self.nod_low: float | None = None  # the lowest load of the raised head's window, once read
        self._pose_t0 = t_ms  # when the pose being waited for was asked
        self._pose_since: int | None = None  # the head at the pose since
        self._anchor: Offsets | None = None  # tilted: where the head was when it last came to rest
        self._win: list[float] = []
        self._gyro: list[float] = []
        self._down_sum = [
            0.0,
            0.0,
            0.0,
        ]  # tilted: the gravity direction summed over the window being read
        self._down_home: tuple[float, float, float] | None = (
            None  # ... the HOME window's mean direction
        )
        self.trunk_shift_deg: float | None = (
            None  # ... the angle the trunk turned by between the two windows
        )

    def head(self) -> Offsets:
        """Where the head is to be (robot.head offsets from HOME)."""
        return PROBE_HOME if self.phase == "home" else PROBE_NOD

    def _at_pose(self, head: Offsets) -> bool:
        if self.tilted:
            # (it has to stay within the tolerance of where it came to rest: moving beyond it starts the settling again)
            if self._anchor is None or any(
                abs(v - a) > PROBE_NOD_TOL for v, a in zip(head, self._anchor, strict=True)
            ):
                self._anchor = head
                return False
            return True
        if self.phase == "home":
            return all(abs(v) <= tol for v, tol in zip(head, PROBE_HOME_TOL, strict=True))
        return all(abs(head[i] - PROBE_NOD[i]) <= PROBE_NOD_TOL for i in (0, 1))

    def step(
        self,
        t_ms: int,
        load: float | None,
        head: Offsets,
        gyro: float | None,
        *,
        ended: bool,
        down: tuple[float, float, float] | None = None,
    ) -> str | None:
        """``ended``: the claim ended this tick, or the duck is not seated and still any more."""
        if ended or load is None:
            return "unclear"
        if t_ms - self._pose_t0 > PROBE_POSE_TIMEOUT_MS + PROBE_SETTLE_MS + PROBE_WINDOW_MS:
            return "unclear"
        if not self._at_pose(head):
            self._pose_since, self._win = (
                None,
                [],
            )  # (it has to stay there: the settling starts again)
            return None
        if self._pose_since is None:
            self._pose_since = t_ms
        if gyro is not None:
            self._gyro.append(gyro)
        held_ms = t_ms - self._pose_since
        if held_ms >= PROBE_SETTLE_MS:
            self._win.append(load)
            if self.tilted and down is not None:
                for i in range(3):
                    self._down_sum[i] += down[i]
        if held_ms < PROBE_SETTLE_MS + PROBE_WINDOW_MS:
            return None
        win = self._win
        if self.phase == "home":
            mean = sum(win) / len(win)
            if mean <= 0.0 or max(win) - min(win) > PROBE_STEADY * mean:
                return "unclear"
            self.base_load = mean
            self.phase = "nod"
            self._pose_t0, self._pose_since, self._win, self._anchor = t_ms, None, [], None
            self._down_home, self._down_sum = _unit(self._down_sum), [0.0, 0.0, 0.0]
            return None
        base = self.base_load or 0.0
        self.nod_low = min(win)
        if max(win) - min(win) > PROBE_STEADY * base:
            return "unclear"
        if self._gyro and sum(self._gyro) / len(self._gyro) > PROBE_CALM_GYRO:
            return "unclear"
        if self.nod_low - base < PROBE_RESPONSE * base:
            return "held"
        if not self.tilted:
            return "supported"
        # (a tilted trunk: the legs hang in a hand too, so the trunk has to answer the head's weight as well)
        nod_dir = _unit(self._down_sum)
        if self._down_home is None or nod_dir is None:
            return "unclear"
        self.trunk_shift_deg = _angle_deg(self._down_home, nod_dir)
        return "supported" if self.trunk_shift_deg >= PROBE_TRUNK_ANSWER_DEG else "held"


def _unit(v: list[float]) -> tuple[float, float, float] | None:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return None if n <= 0.0 else (v[0] / n, v[1] / n, v[2] / n)


def _angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return math.degrees(
        math.atan2(
            math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2),
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        )
    )
