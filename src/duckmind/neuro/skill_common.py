"""What the play acts share (a:play, 2026-09-28): the act's output, the duck's own memory of the
things it plays with, and the fine approach that brings a thing into a verb's reach.

Honest senses only (owner 2026-09-26): everything here is built from the brain's
:class:`~duckmind.neuro.contract.SensorFrame` — the head camera's sightings (egocentric bearing,
the ground-plane range, the apparent width), its own contact odometry (which drifts), the mouth
joint robotd reports (joint 9: something in the beak holds the jaw open), the posture and
``move.applied``. A thing is known again by where it was (odometry) and the eye's track while
it stays tracked, never by a world id. What a thing *affords* — it rolls, it can be picked up,
it will not come — is learned from what happened when the duck tried (or saw it roll), never
given: a ball is "a thing that rolled", a toy "a thing that stayed in my beak".

The fine approach (:class:`Approach`) is curiosity P1's (``duckmind.curiosity.act_runner``: the
M2 controller and its measured step tables, :data:`duckmind.curiosity.calib.CALIB`) on the
neural duck's sightings: stop, let the legs' tail die out, look (the head aimed at the thing),
then one calibrated move — a creep steered toward where the trunk must stand, a reverse arc
when the thing is well off to the side or too close — and look again, until the thing sits in
the verb's window. A window nearer than the eye sees (the full-down look sees the floor from
~95 mm, spec §4.1) is confirmed a little farther out and closed by one blind creep of measured
length. Every move is inside the walking rules (vx 0.40, |vyaw| ≤ 1.0, reversing only as an
arc; ``prims.problems`` is checked on each).

Pure: standard library + the brain's contract and curiosity's pure calibration.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from duckmind.curiosity import prims
from duckmind.curiosity.calib import CALIB, Window, step_ms
from duckmind.neuro.contract import Gaze, HeadPose, SensorFrame, Sighting, Target, wrap_angle
from duckmind.neuro.perceive import lens_point, sighting_odom  # the lens model lives there

COMMON_VERSION = "neuro-play-common-1"

# ── what an act asks for this tick ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PlayOut:
    """One play act's wish this tick (the brain folds it into its MotorCommand; the rest goes to
    the motor and the body link — see the integration notes).

    ``drive``: the act has the legs (``vx``/``vyaw``, 0 = hold still); ``fine``: a calibrated
    short step, to be passed on as it is (no burst/pause holds, no course trim: the motor's
    ``MotorCommand.fine`` at integration); ``reverse``: a reverse arc. ``head``/``head_blend``:
    the act's own head aim over the program's; ``gaze``: where to look; ``mouth``: robot.mouth
    ``open`` over the jaw's FULL travel (a grab needs 1.0: the validated 0.5 opens 12.5°, the fin
    needs more — this rig, 1/6 grabs at 0.5, 6/6 at 1.0), None = the program's; ``skill``: a
    robotd one-shot (``ground_pick``, ``kick_left``, ``kick_right``) sent once per
    ``skill_seq``; ``posture``; ``vocal``; ``phase`` and ``zh`` (plain words for the page);
    ``event``: something that just happened (``kicked``, ``pushed``, ``rolled``, ``grabbed``,
    ``slipped``, ``dropped_home``, ``settled``…); ``quiet``: no small moments now; ``far``: the
    thing is far — the brain's own walk goes there (``target`` says where)."""

    drive: bool = False
    vx: float = 0.0
    vyaw: float = 0.0
    fine: bool = False
    reverse: bool = False
    head: HeadPose | None = None
    head_blend: float = 0.0
    gaze: Gaze | None = None
    mouth: float | None = None
    skill: str | None = None
    skill_seq: int = 0
    posture: str | None = None
    vocal: str | None = None
    phase: str = "idle"
    zh: str = ""
    event: str | None = None
    quiet: bool = False
    far: bool = False
    target: Target | None = None


IDLE = PlayOut()

# ── the duck's own odometry, looked up at a camera pass's capture time ─────────

CAPTURE_LAG_MS = 100  # a pass is captured ~70–140 ms before it is delivered (neuro_senses)


class OdomHistory:
    """Its odometry over the last ~2 s, to place a sighting where it was captured."""

    def __init__(self, keep: int = 100) -> None:
        self.hist: deque[tuple[int, tuple[float, float, float]]] = deque(maxlen=keep)

    def add(self, t_ms: int, odom: tuple[float, float, float]) -> None:
        self.hist.append((t_ms, odom))

    def at(self, t_ms: int) -> tuple[float, float, float]:
        if not self.hist:
            return (0.0, 0.0, 0.0)
        best = self.hist[-1][1]
        for when, where in reversed(self.hist):
            best = where
            if when <= t_ms:
                break
        return best


def to_odom(
    odom: tuple[float, float, float], bearing: float, range_m: float
) -> tuple[float, float]:
    ox, oy, yaw = odom
    return ox + range_m * math.cos(yaw + bearing), oy + range_m * math.sin(yaw + bearing)


AXIS_KEEP_MS = 20_000  # a shape look older than this is not averaged with a new one
CARRIED_MS = 2500  # a thing out of its beak this recently: its moves were its own doing


def rel_of(odom: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    """An odometry point in the trunk's level frame now: (ahead, left), m."""
    ox, oy, yaw = odom
    dx, dy = x - ox, y - oy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def bearing_range(odom: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    a, left = rel_of(odom, x, y)
    return math.atan2(left, a), math.hypot(a, left)


# ── the things it plays with (its own memory) ─────────────────────────────────

SAME_M = 0.35  # a sighting this near a remembered thing (odometry) is that thing …
SAME_MOVING_M = 0.7  # … further for a thing that is rolling
FORGET_S = 1800.0
ROLL_MIN_M = 0.10  # moved this far along the floor (not carried by it): it rolled
SMALL_ROLLER_M = 0.10  # a thing this small seen going along the floor: a roller
MAX_ROLL_M_S = 1.5  # faster than this between two looks is a jump (a bad match), not a roll
MAX_THING_M = 0.18  # wider than this (apparent width) is not a thing to play with


@dataclass(slots=True)
class PlayThing:
    """One thing as the duck knows it (its odometry frame)."""

    key: int
    x: float
    y: float
    seen_ms: int
    first_ms: int
    width: float = 0.07  # apparent width when last seen (m)
    widest: float = 0.07  # the widest it has looked (m): its long side
    vx: float = 0.0  # its motion along the floor (odometry frame, m/s), from successive looks
    vy: float = 0.0
    moving_ms: int = -1_000_000  # when it was last seen moving
    track: int = 0
    rolled: int = 0  # times it rolled (after a push, a kick, or pushed by someone)
    held: int = 0  # times it stayed in its beak
    grab_fails: int = 0  # grabs that came up empty
    nudges: int = 0  # times it pushed or kicked it
    stuck: int = 0  # pushes that moved it < 1.5 cm (a heavy thing)
    home: bool = False  # it lies beside its nest (it put it there)
    seen_n: int = 0
    fun: float = 0.0  # how good playing with it felt (a leaky value)
    roll_from: tuple[float, float] | None = None  # where it was when it began to move
    fixes: list[int] = field(default_factory=list)  # when it was seen, the last few times
    axis: float | None = None  # its long axis on the floor (odometry frame, mod π), if it has one
    axis_ms: int = -1_000_000  # when that was last seen
    axis_n: int = 0  # looks averaged into it
    carried_ms: int = -1_000_000  # when it was last in its own beak (its moves then are its own)
    missing_ms: int = -1_000_000  # it looked where it remembered it and it was not there

    @property
    def rolls(self) -> bool:
        return self.rolled >= 1

    @property
    def carriable(self) -> bool:
        return self.held >= 1 and self.rolled == 0

    def moving(self, t_ms: int) -> bool:
        return t_ms - self.moving_ms < 600


class ThingBook:
    """The small things it has seen, with what they did (see the module doc)."""

    def __init__(self) -> None:
        self.things: dict[int, PlayThing] = {}
        self._tracks: dict[int, int] = {}
        self._next = 1
        self.pass_ms: int | None = None

    def see(
        self,
        t_ms: int,
        frame: SensorFrame,
        hist: OdomHistory,
        *,
        dipped: bool = False,
    ) -> list[PlayThing]:
        """A camera pass that just arrived: each small thing on the floor it shows, matched to
        what it remembers (or new). Returns the things seen in this pass. ``dipped``: the whole
        body is in one of robotd's one-shots (a ground pick, a kick) — the trunk pitched and the
        camera far from where the level frame puts it: no pass is used then (this rig: a missed
        grab's looks put the toy 6 cm nearer than it lay)."""
        body = frame.body
        stepping = body.posture == "walk" or abs(body.vx) > 0.2 or abs(body.vyaw) > 0.3
        cam = frame.camera
        if not cam.usable or dipped:
            return []
        pass_ms = t_ms - int(cam.age_ms)
        if self.pass_ms is not None and abs(pass_ms - self.pass_ms) <= 5:
            return []
        self.pass_ms = pass_ms
        at = hist.at(pass_ms - CAPTURE_LAG_MS)
        lens = lens_point(frame)
        seen: list[PlayThing] = []
        for s in cam.sightings:
            # (a ball rolling straight at it has no image motion across, only growth: the eye
            # calls it looming — on the floor and small, it is a thing coming its way)
            if s.kind not in ("thing", "toy", "motion", "looming") or s.range_m is None:
                continue
            width = s.size * s.range_m
            if width > MAX_THING_M or s.range_m > 2.5:
                continue
            x, y = sighting_odom(at, lens, s.bearing, s.range_m)
            thing = self._match(s, x, y, pass_ms)
            # the eye calls a thing moving when it moves in the picture — walking toward a still
            # thing makes it grow (it "looms") and slide: only while its own legs are still is
            # that the thing's own motion; walking, the thing's shift in its own odometry frame
            # says so (≥ 0.2 m/s between two close looks)
            flagged = s.kind in ("motion", "looming") or s.motion > 0.05
            moving = flagged and not stepping
            own = thing is not None and pass_ms - thing.carried_ms < CARRIED_MS
            if own and thing is not None:
                # it was in its own beak a moment ago (carried, or it just fell out): it did not
                # roll by itself
                moving = False
                thing.roll_from = None
            if thing is None:
                if flagged and stepping:
                    continue  # (a new thing seen only while walking: wait for a clear look)
                if s.kind in ("motion", "looming") and s.motion < 0.05 and s.looming < 0.3:
                    continue
                thing = PlayThing(self._next, x, y, pass_ms, pass_ms, width, width)
                self.things[self._next] = thing
                self._next += 1
            else:
                dt = max(0.05, (pass_ms - thing.seen_ms) / 1000.0)
                vx, vy = (x - thing.x) / dt, (y - thing.y) / dt
                if dt < 0.7 and math.hypot(vx, vy) <= MAX_ROLL_M_S:
                    thing.vx = 0.6 * thing.vx + 0.4 * vx
                    thing.vy = 0.6 * thing.vy + 0.4 * vy
                    if stepping and flagged and math.hypot(thing.vx, thing.vy) >= 0.2:
                        moving = True  # it shifted in its own frame while it walked: it moves
                else:
                    thing.vx = thing.vy = 0.0  # too long between looks (or a jump): no speed
                if not moving and math.hypot(x - thing.x, y - thing.y) < 0.05 and dt < 3.0:
                    # a still thing seen again where it was: average the looks (the eye's
                    # range scatters ~8 %)
                    x, y = 0.5 * (x + thing.x), 0.5 * (y + thing.y)
                thing.x, thing.y = x, y
                thing.seen_ms = pass_ms
                thing.width = width
                thing.widest = max(thing.widest, width)
            if moving:
                if not thing.moving(pass_ms) and thing.roll_from is None:
                    thing.roll_from = (thing.x, thing.y)
                thing.moving_ms = pass_ms
                if thing.rolled == 0 and thing.widest <= SMALL_ROLLER_M:
                    thing.rolled = 1  # a small thing going along the floor by itself: it rolls
            elif thing.roll_from is not None and pass_ms - thing.moving_ms > 800:
                # it came to rest: if it went ≥ 10 cm along the floor, it rolls (whoever pushed it)
                if (
                    math.hypot(thing.x - thing.roll_from[0], thing.y - thing.roll_from[1])
                    >= ROLL_MIN_M
                ):
                    thing.rolled += 1
                thing.roll_from = None
            axis = getattr(s, "axis", None)
            if axis is not None:
                seen_axis = axis_mod_pi(float(axis) + at[2])
                if thing.axis is None or moving or pass_ms - thing.axis_ms > AXIS_KEEP_MS:
                    thing.axis = seen_axis
                    thing.axis_n = 1
                else:
                    # the looks of a still thing's shape, averaged (a line: the doubled angle's
                    # mean), the newest weighing no less than 1/4 (σ ≈ 8° a look → ~4°)
                    k = max(0.25, 1.0 / (thing.axis_n + 1))
                    c = (1.0 - k) * math.cos(2.0 * thing.axis) + k * math.cos(2.0 * seen_axis)
                    s2 = (1.0 - k) * math.sin(2.0 * thing.axis) + k * math.sin(2.0 * seen_axis)
                    thing.axis = axis_mod_pi(0.5 * math.atan2(s2, c))
                    thing.axis_n += 1
                thing.axis_ms = pass_ms
            thing.track = s.track
            thing.seen_n += 1
            thing.fixes.append(pass_ms)
            if len(thing.fixes) > 6:
                del thing.fixes[:-6]
            if s.track:
                self._tracks[s.track] = thing.key
            seen.append(thing)
        for key in [k for k, th in self.things.items() if t_ms - th.seen_ms > FORGET_S * 1000]:
            del self.things[key]
        return seen

    def _match(self, s: Sighting, x: float, y: float, t_ms: int) -> PlayThing | None:
        if s.track and s.track in self._tracks:
            known = self.things.get(self._tracks[s.track])
            if known is not None and t_ms - known.seen_ms < 3000:
                return known
        best, best_d = None, math.inf
        for thing in self.things.values():
            px, py = thing.x, thing.y
            radius = SAME_MOVING_M if thing.moving(t_ms) or s.kind == "motion" else SAME_M
            since = (t_ms - thing.moving_ms) / 1000.0
            if 0.0 <= since < 4.0 and (thing.vx or thing.vy):
                # it was rolling when last seen: look for it where it was going (≤ 1.5 s on)
                ahead = min(1.5, max(0.0, (t_ms - thing.seen_ms) / 1000.0))
                px, py = thing.x + thing.vx * ahead, thing.y + thing.vy * ahead
                radius = max(radius, 0.5)
            d = math.hypot(px - x, py - y)
            if d < radius and d < best_d:
                best, best_d = thing, d
        return best

    def get(self, key: int | None) -> PlayThing | None:
        return None if key is None else self.things.get(key)

    def view(self) -> list[dict[str, object]]:
        return [
            {
                "key": th.key,
                "x_mm": round(th.x * 1000),
                "y_mm": round(th.y * 1000),
                "width_mm": round(th.widest * 1000),
                "rolled": th.rolled,
                "held": th.held,
                "grab_fails": th.grab_fails,
                "nudges": th.nudges,
                "home": th.home,
                "fun": round(th.fun, 2),
            }
            for th in sorted(self.things.values(), key=lambda t: -t.seen_ms)[:8]
        ]


# ── the fine approach (curiosity's M2 controller on the neural duck's senses) ──

TAIL_MS = 450  # the legs keep stepping after a stop (spec §4.2)
LOOK_WAIT_MS = 3500  # a look that brings no sighting of it by then is a miss (a still thing is
# found in ~1 of 4 looks, ~2 looks a second: neuro_senses P_STATIC)
MAX_MISSES = 4
FRESH_MS = 1500  # a sighting this recent is good enough to move on
# How near the eye sees a ball on the floor, standing (this lane's rig, 2026-09-28: the ball's
# centre is in the picture from 0.25 m ahead of the trunk at HOME, 0.17 m with the head pitched
# down to the standing envelope's 0.35 rad, 0.09 m only with the walking envelope's neck −0.5 /
# head +0.7) — curiosity's 95 mm came from a deeper look than the neural motor allows standing
EYE_NEAR_MM = 170
CONFIRM_MARGIN_MM = 20
CONFIRM_BAND_MM = 80
EYE_NEAR_M = EYE_NEAR_MM / 1000.0
NEAR_M = 0.34  # nearer than this, the fine approach has the legs (the brain's walk before)
TURN_RAD = 0.55  # near and this far off: turn with a reverse arc
BLIND_MIN_M, BLIND_MAX_M = 0.030, 0.110
MAX_MOVES = 14
# The walk's own turn to the right (this rig, creeps at 0.40 from standing: 0.8–1.3 s straight
# turn 8–11° right; with +0.15 rad/s of yaw 0–3° and ±13 mm): fed forward on every creep, as the
# dock program does (neuro.dock DockConfig.ff)
FF_VYAW = 0.15
FF_VYAW_MRAD = int(FF_VYAW * 1000)
# Creeps with the head pitched down to look at the thing (0.35 rad, the standing envelope's
# deepest look) step shorter than with the head at HOME (this rig, 0.40 m/s + the feed-forward,
# from standing: ms → mm ahead; HOME: 500 → 41, 1000 → 115, 1500 → 197 — curiosity's live table
# is CALIB.fwd_table); the blind last step is made with the head at HOME and that table
HEAD_DOWN_TABLE: tuple[tuple[int, int], ...] = (
    (400, 1), (500, 18), (600, 32), (800, 61), (1000, 90), (1200, 121), (1500, 167)
)  # fmt: skip
# … and the blind last step, the head at HOME with the feed-forward (this rig; curiosity's live
# table without the feed-forward: 1000 → 125 mm)
BLIND_TABLE: tuple[tuple[int, int], ...] = (
    (400, 21), (500, 41), (600, 53), (800, 86), (1000, 115), (1200, 153), (1500, 197)
)  # fmt: skip


def head_look(ahead: float, left: float, cam_h: float = 0.16) -> tuple[Gaze, HeadPose]:
    """Aim the head at a floor point (trunk frame): the gaze (bearing, elevation) and, near, the
    low looking pose (the full-down look under 0.26 m, half-down under 0.48 m: curiosity's
    NEAR_LOOK_MM / MID_LOOK_MM)."""
    dist = math.hypot(ahead, left)
    bearing = math.atan2(left, ahead)
    elevation = -math.atan2(cam_h, max(dist, 0.05))
    hy = max(-1.0, min(1.0, bearing / 0.6))
    if dist < 0.26:
        pose = HeadPose(np=1.0, hp=-1.0, hy=hy)
    elif dist < 0.48:
        pose = HeadPose(np=0.5, hp=-0.5, hy=hy)
    else:
        pose = HeadPose(np=0.0, hp=-0.2, hy=hy)
    return Gaze(bearing, elevation, 1.0, 0.6), pose


@dataclass
class Approach:
    """One fine approach to a thing's verb window (see the module doc). ``window`` in the trunk
    frame (ahead, side), ``side_sign`` +1 plain / ±1 toward the kicking foot, ``blind``: the
    window is nearer than the eye sees (confirm farther out, then one blind creep)."""

    window: Window
    side_sign: int = 1
    tol_pct: int = 100
    phase: str = "look"
    phase_ms: int = 0
    move: tuple[float, float, int] | None = None  # vx, vyaw, until_ms
    move_name: str = ""
    moves: int = 0
    misses: int = 0
    last_move_end: int = -1_000_000
    seen_after: int = -1_000_000  # a sighting newer than this counts as a look after the move
    notes: list[str] = field(default_factory=list)
    blind_done: bool = False

    def reset(self, t_ms: int) -> None:
        self.phase, self.phase_ms = "look", t_ms
        self.move, self.moves, self.misses = None, 0, 0
        self.last_move_end = t_ms
        self.seen_after = t_ms
        self.blind_done = False

    @property
    def blind(self) -> bool:
        return self.window.ahead[0] < EYE_NEAR_MM

    def confirm_window(self) -> Window:
        """Where it must be seen before the blind last step: a little beyond what the eye sees
        (right at the picture's bottom edge a still thing is found even more rarely)."""
        if not self.blind:
            return self.window
        near = EYE_NEAR_MM + CONFIRM_MARGIN_MM
        return Window((near, near + CONFIRM_BAND_MM), self.window.side)

    def step(
        self,
        t_ms: int,
        rel: tuple[float, float] | None,
        seen_ms: int | None,
        legs_still: bool,
        fixes: list[int] | None = None,
        head_down: bool = False,
    ) -> tuple[str, float, float]:
        """One tick: ``rel`` the thing's believed trunk-frame position now (m), ``seen_ms`` when
        it was last seen, ``legs_still`` robotd's applied twist is zero. Returns (state, vx,
        vyaw): ``moving`` (the twist to send), ``looking`` (hold still, look), ``ready`` (inside
        the window: act now), ``far`` (beyond the fine range: the brain's walk), ``lost``,
        ``gave_up``."""
        if self.move is not None:
            vx, vyaw, until = self.move
            if t_ms < until:
                return "moving", vx, vyaw
            self.move = None
            self.last_move_end = t_ms
            self.seen_after = t_ms + TAIL_MS // 2
            self.phase, self.phase_ms = "look", t_ms
        if rel is None:
            if t_ms - self.phase_ms > LOOK_WAIT_MS:
                return "lost", 0.0, 0.0
            return "looking", 0.0, 0.0
        ahead, left = rel
        dist = math.hypot(ahead, left)
        if self.blind_done:
            return "ready", 0.0, 0.0
        if dist > NEAR_M + 0.06 and self.moves == 0:
            return "far", 0.0, 0.0
        settled = legs_still and t_ms - self.last_move_end >= TAIL_MS
        fresh = seen_ms is not None and seen_ms >= self.seen_after
        if settled and not fresh and ahead < EYE_NEAR_M - 0.02 and t_ms - self.phase_ms > 1000:
            # it is too close to see (under the picture's bottom edge): back off a little in an
            # arc, turning toward its side, to bring it into view again (curiosity: "too close
            # to see: step back")
            self._start(t_ms, prims.back_arc(left=left >= 0, ms=800), "back_arc")
            return "moving", self.move[0], self.move[1]  # type: ignore[index]
        if not (settled and fresh):
            if t_ms - self.phase_ms > LOOK_WAIT_MS + TAIL_MS:
                self.misses += 1
                self.phase_ms = t_ms
                if self.misses >= MAX_MISSES:
                    return "lost", 0.0, 0.0
            return "looking", 0.0, 0.0
        if self.moves >= MAX_MOVES:
            return "gave_up", 0.0, 0.0
        target = self.confirm_window()
        side = left * self.side_sign
        a_mm, s_mm = int(round(ahead * 1000)), int(round(side * 1000))
        if target.contains(a_mm, s_mm, self.tol_pct):
            looks = sum(1 for f in (fixes or ()) if f >= self.seen_after)
            if fixes is not None and looks < 2 and t_ms - self.phase_ms < LOOK_WAIT_MS:
                return "looking", 0.0, 0.0  # (curiosity's confirm: two looks before it acts)
            if self.blind:
                # the last step, blind: from where it was seen to the window's middle
                want = ahead - self.window.centre[0] / 1000.0
                table = HEAD_DOWN_TABLE if head_down else BLIND_TABLE
                ms = max(400, min(1500, step_ms(table, want * 1000.0)))
                self._start(t_ms, prims.creep(ms, FF_VYAW_MRAD), "blind")
                self.blind_done = True
                return "moving", self.move[0], self.move[1]  # type: ignore[index]
            return "ready", 0.0, 0.0
        ca, cs = (v / 1000.0 for v in target.centre)
        ex, ey = ahead - ca, side - cs
        off = abs(math.atan2(left, max(ahead, 1e-3)))
        if (off > TURN_RAD and ahead < 0.30) or (ex < 0.02 and abs(ey) > 0.06):
            # well off to the side (or beside it with no room ahead): a reverse arc turns the
            # nose toward it (1.0 s ≈ ±33°, backing 75–115 mm; phase-3 live)
            self._start(t_ms, prims.back_arc(left=left > 0, ms=CALIB.back_arc_ms), "back_arc")
        elif ex < -0.012:
            # too close: back off a little, turning toward its side (never straight back)
            self._start(t_ms, prims.back_arc(left=left >= 0, ms=800), "back_arc")
        else:
            aim = math.atan2(ey * self.side_sign, max(ex, 0.03))
            steer = int(round(prims.clip(2.0 * aim + FF_VYAW, 1.0) * 1000))
            ms = min(1000, step_ms(HEAD_DOWN_TABLE, max(ex, 0.012) * 1000.0))
            if dist > 0.25:
                ms = max(ms, 600)
            prim = prims.trim(steer) if ms <= 300 else prims.creep(ms, steer)
            self._start(t_ms, prim, prim.name)
        return "moving", self.move[0], self.move[1]  # type: ignore[index]

    def _start(self, t_ms: int, prim: prims.Prim, name: str) -> None:
        problems = prims.twist_problems(prim.vx_mm, prim.vyaw_mrad, reverse_ok=prim.reverse)
        if problems:
            raise ValueError(f"{name}: {problems}")
        self.move = (prim.vx_mm / 1000.0, prim.vyaw_mrad / 1000.0, t_ms + prim.ms)
        self.move_name = name
        self.moves += 1
        self.phase = "move"
        self.notes.append(name)
        if len(self.notes) > 30:
            del self.notes[:10]


SCAN_HOLD_MS = 900  # each look of a search is held this long (a moving camera sees nothing:
# the eye is blind while the head turns fast, and a still thing is found in ~1 of 4 still looks)
SCAN_STEPS = (0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5)


def scan_gaze(
    bearing: float, since_ms: int, *, elevation: float = -0.45, width: float = 0.5
) -> Gaze:
    """Looking for a thing that is not where it was believed: a bird's shift-and-hold search —
    a look, held ~0.9 s, then the next one to the other side, wider as it goes on."""
    k = int(since_ms // SCAN_HOLD_MS) % len(SCAN_STEPS)
    return Gaze(bearing + width * SCAN_STEPS[k], elevation, 1.0, 0.6)


def head_is_down(frame: SensorFrame) -> bool:
    """Its measured head pitch is well below HOME (the step table differs: a looking-down duck
    steps shorter)."""
    head = frame.body.head
    return len(head) > 1 and float(head[1]) - 0.349 > 0.2


def legs_still(frame: SensorFrame) -> bool:
    """robotd's applied twist is (about) zero: the legs are not being driven."""
    return abs(frame.body.vx) < 0.05 and abs(frame.body.vyaw) < 0.15


def mouth_deg(frame: SensorFrame, joint9: float | None) -> float | None:
    """The jaw above closed (°) from robotd's joint 9 (None: not reported)."""
    if joint9 is None:
        return None
    return math.degrees(joint9 + math.radians(5.0))


IN_BEAK_DEG = 3.0  # curiosity E10 / outcomes.IN_BEAK_MARGIN_MRAD: closed yet open ≥ 3°: held


def angle_between(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def axis_mod_pi(a: float) -> float:
    """An axis direction (a line has no front): in [−π/2, π/2)."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


__all__ = [
    "Approach",
    "EYE_NEAR_M",
    "IDLE",
    "IN_BEAK_DEG",
    "NEAR_M",
    "OdomHistory",
    "PlayOut",
    "PlayThing",
    "ThingBook",
    "angle_between",
    "axis_mod_pi",
    "bearing_range",
    "head_look",
    "legs_still",
    "head_is_down",
    "lens_point",
    "rel_of",
    "scan_gaze",
    "sighting_odom",
    "to_odom",
]
