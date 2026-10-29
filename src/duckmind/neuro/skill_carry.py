"""叼玩具回窩 — carrying a small toy home in the beak (a:play, 2026-09-28; owner item 6).

The act (``carry_toy``) grows from a nesting urge (``nest_urge``: it builds over the day, a
sociable, snuggly duck's faster, and is used up by bringing a toy home) and from a toy it knows
it can pick up (``toy``: a thing that stayed in its beak before) — or, the first times, a small
elongated thing it has not tried yet. While something is in its beak (``carrying``: robotd's
mouth joint says the jaw stays open on it) the act holds on.

What the body does, all physical (this lane's rig: :class:`~duckmind.world.neuro_play_rig.
PlayDuck` with the jaw of :mod:`duckmind.world.jaw` — provisional, owner decision A4 — and robotd's
ground pick):

1. **to the toy** — the brain's own walk;
2. **line up from the side** — the jaw closes only on a toy whose long side faces the beak (this
   rig, 36 grabs: the long axis within ~±10° of across the heading 4–6/6 per angle, ±20° 1–3/6,
   ±30° 0/6), so it looks at the toy's shape (the long axis of the blob in its picture — a MODEL
   of what duck-eye's things task can add) and comes in along the toy's perpendicular: the legs
   only turn in arcs (≥ 0.35 m/s, |yaw rate| ≤ 1 rad/s, no turning on the spot), so it plans the
   shortest path of arcs and straights (:mod:`~duckmind.neuro.skill_path`, Dubins' curves at the
   legs' turning radius, never nearer the toy than 0.26 m) to a point 0.6 m out on the toy's
   perpendicular, facing it; follows it with a look-ahead point (planning again when the legs
   drift off); walks in along the line to 0.34 m out; the last few centimetres are curiosity's
   calibrated creeps and one blind step (the window, 52–70 mm ahead, is nearer than the eye
   sees);
3. **pick it up** — robotd's ``ground_pick`` (the whole body dips, the beak 2 cm off the floor)
   with the beak open from 0.2 s and shut at 1.4 s (curiosity M5); after it the mouth joint
   says whether it holds anything (≥ 3° open while told to shut, curiosity E10) — never the
   world;
4. **carry it home** — walking toward the spot beside its dock (the dock's frame it remembers:
   0.32 m out from the face, 0.42 m to the side the J-hook does not use, so the toy never lies in
   its way in), the head held still at HOME, the beak shut, at ≤ 0.40 m/s; if the mouth joint
   says the toy slipped out, it stops, looks down, finds it and picks it up again (in the
   simulator the provisional jaw's soft contacts let the fin creep out ~4 s after the beak
   shuts: this rig, 4.0–4.5 s; with MuJoCo's no-slip solver it held 8 s and more — see the
   report);
5. **put it down** beside the dock (the beak opens 0.8 s) — the urge used up, a soft coo.

Honest failures (every one from its own senses): a grab that comes up empty (tried again from
the toy's other long side, three times at most), a toy that slips on the way (picked up again),
a toy it cannot find again (it gives up), a home it does not know (it never starts: no dock
frame, no carry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from duckmind.curiosity import prims
from duckmind.curiosity.calib import CALIB, Window, step_ms
from duckmind.i18n import msg
from duckmind.neuro import signals as N
from duckmind.neuro import skill_path as SP
from duckmind.neuro.contract import Gaze, HeadPose, SensorFrame, Target, wrap_angle
from duckmind.neuro.skill_common import (
    BLIND_TABLE,
    EYE_NEAR_M,
    FF_VYAW,
    HEAD_DOWN_TABLE,
    IN_BEAK_DEG,
    MAX_MISSES,
    TAIL_MS,
    Approach,
    PlayOut,
    PlayThing,
    ThingBook,
    bearing_range,
    head_look,
    legs_still,
    rel_of,
    scan_gaze,
)

CARRY_VERSION = "neuro-carry-1"
# The grab: curiosity's window ahead (52–70 mm: phase 3, 36/36 lifted), wider to the side for a
# toy lying across the heading — the beak closes on its long side anywhere along it (this rig,
# 2026-09-28, the fin across the heading, one grab per cell: 52–68 mm ahead lifted at 0 and
# ±30 mm to the side 15/15, 72 mm 1/3, 48 mm 0/3, 75–120 mm 0/16; ±45 mm at 60 mm 2/2)
GRAB_WINDOW = Window((CALIB.grab.ahead[0], CALIB.grab.ahead[1]), (-40, 40))
MOUTH_OPEN_MS = CALIB.mouth_open_ms  # 200: open from here …
MOUTH_CLOSE_MS = CALIB.mouth_close_ms  # 1400: … shut from here (curiosity M5)
PICK_HOLD_MS = 3300  # the ground pick (2.8 s) and back to standing
PICK_START_TIMEOUT_MS = 3000  # allow 1200 ms readiness + 1200 ms policy-start timeout + tick margin
PUT_DOWN_MS = 800  # the beak open this long lets it drop (curiosity's put_down)
STEP_BACK_MS = 900  # then a back arc away from the dock
STAND_FAR_M = 0.50  # the survey: it stops about this far from the toy to see how it lies
GOAL_M = 0.80  # the arcs end on the toy's perpendicular this far out, facing it (room for the
# walk in along the line to take out what the arcs left: 0.6 m left 8–13 cm off in 5 of 8) …
GOAL_FAR_M = 1.0  # … or this far, when the nearer end cannot keep clear of the toy
FIELD_CHI = math.radians(60.0)  # along the line: at most this much across it …
FIELD_K = 10.0  # … bent by the side error (1/m: ~30 % of it at 5 cm off)
# The reach (the last 0.3 m): creeps straight in (the walk's own drift fed forward, a little
# steer), each decided on a fresh still look; the last one blind (the window is nearer than the
# eye sees) by the measured step table, the head at HOME. A toy too far to the side or a heading
# off the toy's perpendicular after a look: it came in crooked (round again).
REACH_BLIND_MAX_M = 0.27  # the toy this near (and seen): the blind creep reaches the window
REACH_SEE_MIN_M = 0.10  # nearer than this there is no creep short enough: back off
REACH_SIDE_M = 0.045  # the beak closes on the long side within ±40 mm of its middle (measured)
REACH_HDG_RAD = 0.30  # the heading off the toy's perpendicular (±20° grabs 1–3/6: measured)
REACH_SETTLE_MS = 450  # the legs still this long after a creep before a look counts
REACH_LOOK_MS = 3500
REACH_MAX_MOVES = 10
REACH_SIDE_GAIN = 0.3  # the reach's pull back onto the line (a fraction of the line-follower's)
HEAD_YAW_RAD_PER_UNIT = 0.6  # the head's yaw per clip unit (neuro.motor.ENVELOPE[2])
STOP_S_M = 0.34  # out along the line: stop here, the fine approach takes over (≤ 0.40 m)
KEEP_OUT_M = 0.26  # off the line, never nearer the toy than this (a foot on it is a stumble)
HOME_ARRIVE_M = 0.20  # the drop spot this near: put it down
HOME_TOY_M = 0.45  # a toy this near the drop spot is home already
DROP_OUT_MM = 420.0  # the drop spot in the dock's frame: out from the wall …
DROP_SIDE_MM = 600.0  # … and to the side the J-hook does not turn on (the guides reach ±244 mm)
PRE_OUT_MM = 800.0  # the way in: from this far out (in front of the dock, never across it) …
CORRIDOR_MM = 360.0  # … along the drop side, this far or more from the dock's axis
MAX_FAILS = 3
MAX_SLIPS = 8
MAX_CROOKED = 3  # approaches that ended off the line, per try
SEARCH_MS = 6300  # seven held looks about where it fell
SEARCH_ARC_MS = 800  # a slipped toy under the beak: two back arcs this long (~15 cm) …
SEARCH_ARCS = 2  # … and, still unseen and near, at most this many more pairs
LEG_VX = 0.40
CARRY_VX = 0.35  # carrying: the slowest arc the legs make (C.ARC_VX_MIN)
# The re-grip (learned): the provisional jaw's soft contacts let the fin creep out of the beak
# ~1.7–3 s into a carry with MuJoCo's default solver (curiosity: a 1.7 s relay leg 9/9 dropped; this
# rig: 2.5–4 s). After it has felt the toy slip once (the mouth joint), the duck carries in legs of
# half the time it held, stops, lets go (the fin lands 89–109 mm ahead of the trunk, 9/9), steps in
# with a 650 ms head-down creep (57–63 mm, 3/3) and picks it up again (3/3; 600 ms 2/3, 550 ms 0/3).
REGRIP_STOP_MS = 600
REGRIP_OPEN_MS = 800
REGRIP_HEAD_MS = 600
REGRIP_CREEP_MS = 650
REGRIP_SETTLE_MS = 900
RELAY_MIN_MS, RELAY_MAX_MS = 600, 1500
TOY_MEMORY_S = 180.0  # a toy it saw this long ago still draws it (it knows where it left it)
UNTRIED_TOY = 0.75  # a small long thing it has never carried draws it this much of a known toy
AXIS_OK_RAD = math.radians(25.0)  # facing the toy's long side within this: line up
HEAD_REACH_RAD = 0.8  # a toy further round than this is out of the head's reach (±1.0 rad)
BEHIND_RAD = 1.6  # carrying: home further round than this — back round toward it first
PHASE_ZH = {
    "idle": "",
    "to_toy": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.37dde1"),
    "survey": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.78698c"),
    "around": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.008133"),
    "line_up": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.2c9c51"),
    "grab": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.5f2a3b"),
    "carry": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.6dbacf"),
    "search": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.095549"),
    "drop": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.ea54ba"),
    "backoff": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.b638b4"),
    "regrip": msg("neuro.incoming.incoming_skill_carry.PHASE_ZH.regrip"),
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _clip_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def axis_error(axis: float | None, heading: float) -> float:
    """How far the heading is off the toy's perpendicular (rad, −π/2..π/2; 0 without an axis)."""
    if axis is None:
        return 0.0
    return (wrap_angle(axis + math.pi / 2 - heading) + math.pi / 2) % math.pi - math.pi / 2


def line_frame(ahead: float, left: float, axis_rel: float | None) -> tuple[float, float]:
    """(ψ, e) in the trunk frame for a toy at (ahead, left) whose long axis lies ``axis_rel`` off
    the heading: ψ the heading of the toy's perpendicular (toward the toy) — how far the duck is
    turned off it — and e how far the trunk is to the left of that line (m)."""
    if axis_rel is None:
        return 0.0, -left
    qx, qy = math.cos(axis_rel + math.pi / 2), math.sin(axis_rel + math.pi / 2)
    if qx * ahead + qy * left < 0.0:
        qx, qy = -qx, -qy  # the perpendicular that runs from the duck's side toward the toy
    return math.atan2(qy, qx), ahead * qy - left * qx


@dataclass
class Reach:
    """The last 0.3 m to the toy (see :data:`REACH_BLIND_MAX_M`): ``step`` returns (state, vx,
    vyaw) — ``moving`` (send this twist), ``looking`` (hold still, look), ``ready`` (the blind
    creep is done: pick it up), ``crooked`` (off the toy's side or heading: round again),
    ``lost`` (no look of it), ``gave_up`` (too many creeps)."""

    phase: str = "settle"
    phase_ms: int = 0
    move: tuple[float, float, int] | None = None
    move_name: str = ""
    move_sent: int = 0  # when the move was asked for …
    move_t0: int | None = None  # … and when the legs took it (robotd's applied twist)
    move_ms: int = 0
    moves: int = 0
    misses: int = 0
    last_move_end: int = -1_000_000
    notes: list[str] = field(default_factory=list)

    def reset(self, t_ms: int) -> None:
        self.phase, self.phase_ms = "settle", t_ms
        self.move, self.move_name, self.moves, self.misses = None, "", 0, 0
        self.last_move_end = t_ms

    def _start(self, t_ms: int, prim: prims.Prim, name: str) -> tuple[str, float, float]:
        self.move = (prim.vx_mm / 1000.0, prim.vyaw_mrad / 1000.0, t_ms + prim.ms)
        self.move_name = name
        self.move_sent, self.move_t0, self.move_ms = t_ms, None, prim.ms
        self.moves += 1
        self.phase = name
        self.notes.append(f"{name}:{prim.ms}")
        del self.notes[:-20]
        return "moving", self.move[0], self.move[1]

    def _drop_move(self, t_ms: int, note: str) -> tuple[str, float, float]:
        self.move = None
        self.last_move_end = t_ms
        self.phase, self.phase_ms = "settle", t_ms
        self.notes.append(note)
        return "looking", 0.0, 0.0

    def step(
        self,
        t_ms: int,
        rel: tuple[float, float] | None,
        seen_ms: int | None,
        still: bool,
        axis_rel: float | None,
        distracted: bool = False,
    ) -> tuple[str, float, float]:
        if self.move is not None:
            vx, vyaw, until = self.move
            if distracted:
                # a reflex has the head (it looks at something that moved or made a sound): it
                # stops its creep, looks, and comes back to the toy after (a creep with the head
                # turned away walks off the line: this rig, 15° in 0.9 s; and an orienting head
                # outside the walking envelope holds the legs in the body link anyway)
                return self._drop_move(t_ms, "paused")
            if self.move_t0 is None:
                # the step tables count from when the legs take the move: until robotd applies
                # it (the body link may hold the legs a moment), the clock does not run
                if not still:
                    self.move_t0 = t_ms
                    self.move = (vx, vyaw, t_ms + self.move_ms)
                elif t_ms - self.move_sent > 2000:
                    return self._drop_move(t_ms, "held")  # never taken: look again
                return "moving", vx, vyaw
            if t_ms < until:
                return "moving", vx, vyaw
            self.move = None
            self.last_move_end = t_ms
            self.phase, self.phase_ms = (
                ("blind_done" if self.move_name == "blind" else "settle"),
                t_ms,
            )
        settled = still and t_ms - self.last_move_end >= REACH_SETTLE_MS and not distracted
        if self.phase == "blind_done":
            return ("ready", 0.0, 0.0) if settled else ("looking", 0.0, 0.0)
        if rel is None or not settled:
            if t_ms - self.phase_ms > REACH_LOOK_MS + REACH_SETTLE_MS:
                return "lost", 0.0, 0.0
            return "looking", 0.0, 0.0
        ahead, left = rel
        fresh = seen_ms is not None and seen_ms >= self.last_move_end + REACH_SETTLE_MS // 2
        if not fresh:
            if ahead < EYE_NEAR_M - 0.01 and t_ms - self.phase_ms > 1500:
                # under the picture's bottom edge: back off a little (an arc, the nose toward it)
                return self._start(t_ms, prims.back_arc(left=left >= 0, ms=800), "back_arc")
            if t_ms - self.phase_ms > REACH_LOOK_MS:
                self.misses += 1
                self.phase_ms = t_ms
                if self.misses >= MAX_MISSES:
                    return "lost", 0.0, 0.0
            return "looking", 0.0, 0.0
        if self.moves >= REACH_MAX_MOVES:
            return "gave_up", 0.0, 0.0
        psi, e = line_frame(ahead, left, axis_rel)
        if abs(psi) > 2.0 * REACH_HDG_RAD or abs(e) > 2.0 * REACH_SIDE_M:
            return "crooked", 0.0, 0.0  # too far off to put right with creeps
        # steer to keep facing along the toy's perpendicular (the heading is what the grab cannot
        # forgive: ±17°; the side can be ±40 mm) — only a little for the side error (this rig: the
        # full line-follower's pull turned 1-s creeps 17° off, 5 of 8 reaches crooked)
        chi = psi - REACH_SIDE_GAIN * FIELD_CHI * (2.0 / math.pi) * math.atan(FIELD_K * e)
        centre = GRAB_WINDOW.centre[0] / 1000.0
        if ahead > REACH_BLIND_MAX_M:
            want = ahead - (REACH_BLIND_MAX_M - 0.04)
            ms = max(400, min(1500, step_ms(HEAD_DOWN_TABLE, want * 1000.0)))
            steer = _clip(1.0 * chi + FF_VYAW, -0.4, 0.4)
            return self._start(t_ms, prims.creep(ms, int(round(steer * 1000))), "creep")
        if ahead >= REACH_SEE_MIN_M:
            if abs(psi) > REACH_HDG_RAD or abs(e) > REACH_SIDE_M:
                return "crooked", 0.0, 0.0  # no room left to put it right
            want = ahead - centre
            ms = max(400, min(1500, step_ms(BLIND_TABLE, want * 1000.0)))
            steer = _clip(0.8 * chi + FF_VYAW, -0.3, 0.3)
            return self._start(t_ms, prims.creep(ms, int(round(steer * 1000))), "blind")
        return self._start(t_ms, prims.back_arc(left=left >= 0, ms=800), "back_arc")


@dataclass
class CarryHome:
    """The carry act's state machine (one per duck)."""

    seed: int = 0
    turn_side: str = "right"  # the dock's J-hook side (the home's layout; the drop goes opposite)
    phase: str = "idle"
    phase_ms: int = 0
    target: int | None = None
    waypoints: list[tuple[float, float]] | None = None
    line_n: tuple[float, float] | None = None  # the toy's perpendicular it comes in along (unit)
    sub: str = "path"  # the way round: the planned arcs, then along the line
    follow: SP.Follower | None = None
    planned_ms: int = 0
    clear_ms: int = 0
    plans: int = 0
    approach: Approach | None = None
    reach: Reach | None = None
    distracted: bool = False  # a reflex has the head this tick (an orienting look)
    search_arcs: int = 0
    dock_pose: tuple[float, float, float] | None = None  # its belief in the dock's frame now
    via_done: bool = False  # this carry has come in by the way in
    relay_ms: int | None = None  # learned: carry legs this long (None: it has never felt a slip)
    leg_ms: int = 0  # when this carried leg began
    regrips: int = 0
    regripping: bool = False  # the grab under way is a re-grip (the way in is kept)
    regrip_t0: int | None = None  # when it let go for the re-grip
    turn_until: int = -1  # a back arc toward the toy runs until then
    turn_left: bool = True
    skill_seq: int = 0
    acted_ms: int = 0
    grab_started_ms: int | None = None  # measured start of robotd ground_pick policy
    closed_ms: int | None = None  # when the beak shut on it last
    holding: bool = False
    grabs: int = 0
    held_n: int = 0
    fails: int = 0
    slips: int = 0
    homes: int = 0
    gave_up: int = 0
    crooked: int = 0
    carried_m: float = 0.0
    _last_odom: tuple[float, float, float] | None = None
    rng: int = 0
    last_event: str | None = None

    def __post_init__(self) -> None:
        self.rng = N.seed_from("neuro.play.carry", self.seed)

    # ── where home is ───────────────────────────────────────────────────────

    def drop_side(self) -> float:
        """+1: the dock's left (facing the room) — the side its J-hook lane is not on."""
        return 1.0 if self.turn_side == "right" else -1.0

    def dock_point(
        self,
        dock_pose: tuple[float, float, float] | None,
        odom: tuple[float, float, float],
        gx: float,
        gy: float,
    ) -> tuple[float, float] | None:
        """A point of the dock's frame (mm) in its odometry frame, from its belief of where it
        is in the dock's frame (x out from the wall, y to the dock's left facing the room, mm;
        heading); None when it does not know where its dock is."""
        if dock_pose is None:
            return None
        px, py, ph = dock_pose
        # the point in the duck's trunk frame, then in its odometry
        dx, dy = (gx - px) / 1000.0, (gy - py) / 1000.0
        c, s = math.cos(-ph), math.sin(-ph)
        ahead, left = c * dx - s * dy, s * dx + c * dy
        ox, oy, yaw = odom
        return ox + ahead * math.cos(yaw) - left * math.sin(yaw), oy + ahead * math.sin(
            yaw
        ) + left * math.cos(yaw)

    def drop_spot(
        self, dock_pose: tuple[float, float, float] | None, odom: tuple[float, float, float]
    ) -> tuple[float, float] | None:
        """The spot beside the dock in its odometry frame (see :meth:`dock_point`)."""
        self.dock_pose = dock_pose
        return self.dock_point(dock_pose, odom, DROP_OUT_MM, self.drop_side() * DROP_SIDE_MM)

    def way_in(self, odom: tuple[float, float, float]) -> tuple[float, float] | None:
        """Where to walk first with the toy: the point in front of the dock on the drop side
        (so the last leg runs toward the wall beside the dock, never across its plate and
        guides) — or None when it is already in that corridor (or does not know its dock)."""
        pose = self.dock_pose
        if pose is None:
            return None
        px, py, _ = pose
        if px <= PRE_OUT_MM + 60.0 and self.drop_side() * py >= CORRIDOR_MM:
            return None
        return self.dock_point(pose, odom, PRE_OUT_MM, self.drop_side() * DROP_SIDE_MM)

    # ── the choice ──────────────────────────────────────────────────────────

    def pick(
        self,
        book: ThingBook,
        odom: tuple[float, float, float],
        t_ms: int,
        home: tuple[float, float] | None,
    ) -> PlayThing | None:
        best, best_score = None, 0.0
        for th in book.things.values():
            if (
                (t_ms - th.seen_ms) / 1000.0 > TOY_MEMORY_S
                or th.rolls
                or th.grab_fails >= MAX_FAILS
            ):
                continue
            if th.missing_ms > th.seen_ms:
                continue  # it looked where it was and it was gone
            if th.axis is None and th.held == 0:
                continue  # not a thing with a side to grip (a ball, a box)
            if home is not None and math.hypot(th.x - home[0], th.y - home[1]) < HOME_TOY_M:
                th.home = True
                continue
            th.home = False
            _, rng = bearing_range(odom, th.x, th.y)
            score = (1.0 if th.carriable else 0.5) / (0.5 + rng)
            if score > best_score:
                best, best_score = th, score
        return best

    def start(self, t_ms: int) -> None:
        self.phase, self.phase_ms = "idle", t_ms
        self.target = None
        self.approach = None
        self.waypoints = None
        self.line_n = None
        self.follow = None
        self.sub = "path"
        self.crooked = 0
        self.grab_started_ms = None

    def _go(self, phase: str, t_ms: int) -> None:
        self.phase, self.phase_ms = phase, t_ms

    def _draw(self) -> float:
        self.rng, u = N.between(self.rng, 0.0, 1.0)
        return u

    # ── the step ────────────────────────────────────────────────────────────

    def step(
        self,
        t_ms: int,
        frame: SensorFrame,
        book: ThingBook,
        odom: tuple[float, float, float],
        home: tuple[float, float] | None,
        mouth_deg: float | None,
        *,
        halted: bool,
        standing: bool,
        distracted: bool = False,
    ) -> PlayOut:
        self.last_event = None
        self.distracted = distracted
        held_now = mouth_deg is not None and mouth_deg >= IN_BEAK_DEG
        if self._last_odom is not None and self.holding and held_now:
            self.carried_m += math.hypot(odom[0] - self._last_odom[0], odom[1] - self._last_odom[1])
        self._last_odom = odom
        thing = book.get(self.target)
        if self.holding and thing is not None:
            # it is in the beak: where the beak is (the camera cannot see it there)
            thing.x = odom[0] + 0.08 * math.cos(odom[2])
            thing.y = odom[1] + 0.08 * math.sin(odom[2])
            thing.seen_ms = t_ms
            thing.carried_ms = t_ms
        if home is None:
            return PlayOut(
                phase="idle",
                zh=msg("neuro.incoming.incoming_skill_carry.CarryHome.step.396ef4"),
                event="no_home",
            )
        if thing is None or self.phase == "idle":
            thing = self.pick(book, odom, t_ms, home)
            if thing is None:
                return PlayOut(
                    phase="idle",
                    zh=msg("neuro.incoming.incoming_skill_carry.CarryHome.step.b43686"),
                )
            self.target = thing.key
            self._go("to_toy", t_ms)
        if self.phase in ("grab", "drop", "regrip") and not halted:
            # (robotd's one-shot runs: robotd's label is the skill's, not stand — go on)
            handler = getattr(self, f"_p_{self.phase}")
            return handler(t_ms, frame, book, thing, odom, home, held_now)  # type: ignore[no-any-return]
        if halted or not standing:
            if self.approach is not None:
                self.approach.move = None
            return PlayOut(
                phase=self.phase,
                zh=PHASE_ZH.get(self.phase, ""),
                posture="stand",
                mouth=0.0 if self.holding else None,
            )
        handler = getattr(self, f"_p_{self.phase}")
        return handler(t_ms, frame, book, thing, odom, home, held_now)  # type: ignore[no-any-return]

    # ── phases ──────────────────────────────────────────────────────────────

    def _p_to_toy(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        bearing, rng = bearing_range(odom, thing.x, thing.y)
        if rng > STAND_FAR_M + 0.15:
            gaze, _ = head_look(*rel_of(odom, thing.x, thing.y))
            return PlayOut(
                far=True,
                phase="to_toy",
                zh=PHASE_ZH["to_toy"],
                gaze=Gaze(bearing, max(gaze.elevation, -0.4), 0.9, 0.6),
                target=Target("thing", bearing, rng, thing.track),
            )
        if abs(bearing) > HEAD_REACH_RAD or t_ms < self.turn_until:
            # near, but beside or behind it (it walked past): the head cannot turn that far —
            # back arcs, the nose toward it (~33° a second), until it is in front
            if t_ms < self.turn_until:
                return PlayOut(
                    drive=True,
                    vx=-0.40,
                    vyaw=1.0 if self.turn_left else -1.0,
                    fine=True,
                    reverse=True,
                    phase="to_toy",
                    zh=PHASE_ZH["to_toy"],
                    quiet=True,
                )
            if legs_still(frame) and t_ms - self.turn_until >= TAIL_MS:
                self.turn_left = bearing > 0
                self.turn_until = t_ms + CALIB.back_arc_ms
            return PlayOut(drive=True, phase="to_toy", zh=PHASE_ZH["to_toy"], quiet=True)
        self._go("survey", t_ms)
        return self._p_survey(t_ms, frame, book, thing, odom, home, held_now)

    def _p_survey(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        """Stop and look at how it lies (its long axis) — then plan the way round to its side."""
        a, left = rel_of(odom, thing.x, thing.y)
        gaze, head = head_look(a, left)
        still = legs_still(frame) and t_ms - self.phase_ms > TAIL_MS
        fresh = thing.axis is not None and thing.axis_ms >= self.phase_ms + TAIL_MS // 2
        if still and fresh:
            self._plan(thing, odom, t_ms)
            self._go("around", t_ms)
            return self._p_around(t_ms, frame, book, thing, odom, home, held_now)
        if t_ms - self.phase_ms > 4500:
            if thing.seen_ms < self.phase_ms and not self.holding:
                thing.missing_ms = t_ms  # not where it remembered it: gone (until seen again)
                return self._give_up(t_ms, "lost")
            if thing.axis is None:
                return self._give_up(t_ms, "no_side")
            self._plan(thing, odom, t_ms)  # an older look will do
            self._go("around", t_ms)
        return PlayOut(
            drive=True,
            head=head,
            head_blend=0.7,
            gaze=gaze,
            phase="survey",
            zh=PHASE_ZH["survey"],
            quiet=True,
        )

    def _plan(self, thing: PlayThing, odom: tuple[float, float, float], t_ms: int = 0) -> None:
        """The toy's perpendicular it comes in along — on the side it is on (after a miss, the
        other long side) — and the shortest path of arcs there (:mod:`skill_path`: Dubins'
        curves at the legs' own turning radius) that keeps clear of the toy, ending
        :data:`GOAL_M` out on the line, facing the toy."""
        axis = thing.axis if thing.axis is not None else 0.0
        nx, ny = math.cos(axis + math.pi / 2), math.sin(axis + math.pi / 2)
        if (odom[0] - thing.x) * nx + (odom[1] - thing.y) * ny < 0.0:
            nx, ny = -nx, -ny
        if thing.grab_fails % 2 == 1:
            nx, ny = -nx, -ny  # it missed from this side: the other long side this time
        best = None
        for sign, goal_m in ((1.0, GOAL_M), (-1.0, GOAL_M), (1.0, GOAL_FAR_M), (-1.0, GOAL_FAR_M)):
            gx, gy = thing.x + sign * nx * goal_m, thing.y + sign * ny * goal_m
            goal = (gx, gy, math.atan2(-sign * ny, -sign * nx))
            got = SP.plan_clear(odom, goal, (thing.x, thing.y), KEEP_OUT_M)
            if got is not None:
                best = (sign, got)
                break
        if best is None:
            # too near to turn round it clear of the toy: back off first (the "clear" arc)
            self.line_n = (nx, ny)
            self.sub = "clear"
            self.clear_ms = t_ms
            self.follow = None
            self.plans += 1
            return
        sign, (_, points) = best
        self.line_n = (sign * nx, sign * ny)
        self.sub = "path"
        self.follow = SP.Follower(points, vx=LEG_VX)
        self.planned_ms = t_ms
        self.plans += 1
        self.waypoints = [(p[0], p[1]) for p in points[:: max(1, len(points) // 12)]]

    def along(self, thing: PlayThing, odom: tuple[float, float, float]) -> tuple[float, float]:
        """(s, e): how far out along the toy's perpendicular the trunk is, and how far to the left
        of the line (facing the toy along it), m."""
        nx, ny = self.line_n or (1.0, 0.0)
        px, py = odom[0] - thing.x, odom[1] - thing.y
        return px * nx + py * ny, px * ny - py * nx

    def _p_around(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        """The planned arcs out and round to the toy's perpendicular (re-planned when the legs
        drift off them), then along the line to 0.34 m out, then the fine approach."""
        if self.line_n is None or (self.follow is None and self.sub == "path"):
            self._plan(thing, odom, t_ms)
        if t_ms - self.phase_ms > 40_000:
            return self._give_up(t_ms, "around")
        nx, ny = self.line_n  # type: ignore[misc]
        s, e = self.along(thing, odom)
        a, left = rel_of(odom, thing.x, thing.y)
        bearing_toy = math.atan2(left, a)
        dist_toy = math.hypot(a, left)
        gaze, _ = head_look(a, left)
        look = Gaze(_clip(gaze.bearing, -0.5, 0.5), max(gaze.elevation, -0.4), 0.6, 0.5)
        if (
            self.sub != "line"
            and s <= GOAL_M
            and abs(e) <= REACH_SIDE_M
            and abs(bearing_toy) <= AXIS_OK_RAD
        ):
            self.sub = "line"  # already on the line, facing the toy: straight in
        if self.sub == "clear":
            # no clear path from this near: back off in an arc (nose toward the toy), plan again
            since = t_ms - self.clear_ms
            if since < CALIB.back_arc_ms:
                return PlayOut(
                    drive=True,
                    vx=-0.40,
                    vyaw=1.0 if bearing_toy > 0 else -1.0,
                    fine=True,
                    reverse=True,
                    phase="around",
                    zh=PHASE_ZH["around"],
                    quiet=True,
                )
            if since < CALIB.back_arc_ms + TAIL_MS or not legs_still(frame):
                return PlayOut(drive=True, phase="around", zh=PHASE_ZH["around"], quiet=True)
            self._plan(thing, odom, t_ms)
            return PlayOut(drive=True, phase="around", zh=PHASE_ZH["around"], quiet=True)
        if self.sub == "path":
            fol = self.follow
            assert fol is not None
            vx, vyaw = fol.step(odom)
            if fol.done:
                self.sub = "line"
            else:
                if fol.off > SP.OFF_PATH_M and t_ms - self.planned_ms >= SP.REPLAN_MS:
                    self._plan(thing, odom, t_ms)  # the legs drifted off: plan again from here
                    if self.follow is None:  # (no clear path from here: the back-off arc next)
                        return PlayOut(
                            drive=True, phase="around", zh=PHASE_ZH["around"], quiet=True
                        )
                    vx, vyaw = self.follow.step(odom)
                if dist_toy < KEEP_OUT_M and abs(bearing_toy) < 1.2:
                    vyaw = -1.0 if bearing_toy > 0 else 1.0  # the toy close ahead: turn away
                return PlayOut(
                    drive=True,
                    vx=vx,
                    vyaw=vyaw,
                    phase="around",
                    zh=PHASE_ZH["around"],
                    gaze=look,
                    quiet=True,
                )
        # along the line: the heading field (toward the toy, bent onto the line by the side error)
        chi_line = math.atan2(-ny, -nx)
        chi = chi_line - FIELD_CHI * (2.0 / math.pi) * math.atan(FIELD_K * e)
        err = wrap_angle(chi - odom[2])
        if s <= STOP_S_M or dist_toy <= STOP_S_M:
            # stop: the reach looks and decides (a crooked arrival goes round again from there)
            self.reach = Reach()
            self.reach.reset(t_ms)
            self._go("line_up", t_ms)
            return PlayOut(drive=True, phase="line_up", zh=PHASE_ZH["line_up"], quiet=True)
        if s < 0.2 or abs(e) > 0.25 or abs(err) > 1.4:
            self._plan(thing, odom, t_ms)  # off the line: plan the arcs again
            return PlayOut(
                drive=True,
                vx=LEG_VX,
                vyaw=FF_VYAW,
                phase="around",
                zh=PHASE_ZH["around"],
                gaze=look,
                quiet=True,
            )
        _, down = head_look(a, left)
        return PlayOut(
            drive=True,
            vx=LEG_VX,
            vyaw=_clip(2.0 * err + FF_VYAW, -1.0, 1.0),
            phase="around",
            zh=PHASE_ZH["around"],
            head=HeadPose(np=down.np, hp=down.hp),
            head_blend=1.0,
            quiet=True,
        )

    def _crooked(self, t_ms: int, thing: PlayThing) -> PlayOut:
        """It came in off the toy's side or heading: back out in an arc, round again."""
        self.crooked += 1
        self.follow = None
        self.line_n = None
        if self.crooked > MAX_CROOKED:
            return self._give_up(t_ms, "crooked")
        self._go("backoff", t_ms)
        return PlayOut(drive=True, phase="backoff", zh=PHASE_ZH["backoff"], event="crooked")

    def _p_line_up(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        rc = self.reach
        if rc is None:
            rc = self.reach = Reach()
            rc.reset(t_ms)
        rel = rel_of(odom, thing.x, thing.y)
        axis_rel = None if thing.axis is None else wrap_angle(thing.axis - odom[2])
        state, vx, vyaw = rc.step(
            t_ms, rel, thing.seen_ms, legs_still(frame), axis_rel, self.distracted
        )
        gaze, head = head_look(*rel)
        # creeping, the head looks straight down its line — no turn (a turned head walks the creep
        # off the line) — and at HOME for the blind creep, as the step tables were measured; still,
        # it turns to the toy when the toy is off to the side (the picture is ±18° wide)
        still_head = HeadPose(np=head.np, hp=head.hp)
        look_head = HeadPose(
            np=head.np, hp=head.hp, hy=_clip(gaze.bearing / HEAD_YAW_RAD_PER_UNIT, -1.0, 1.0)
        )
        if state == "moving":
            blind = rc.move_name == "blind"
            return PlayOut(
                drive=True,
                vx=vx,
                vyaw=vyaw,
                fine=True,
                reverse=vx < 0.0,
                head=HeadPose() if blind else still_head,
                head_blend=1.0,
                phase="line_up",
                zh=PHASE_ZH["line_up"],
                quiet=True,
            )
        if state == "ready":
            self.acted_ms = t_ms
            self.grab_started_ms = None
            self.skill_seq += 1
            self.grabs += 1
            self._go("grab", t_ms)
            return self._p_grab(t_ms, frame, book, thing, odom, home, held_now)
        if state == "crooked":
            return self._crooked(t_ms, thing)
        if state in ("lost", "gave_up"):
            thing.grab_fails += 1
            self.fails += 1
            if thing.grab_fails >= MAX_FAILS:
                return self._give_up(t_ms, "line_up")
            self._go("search", t_ms)
            return PlayOut(drive=True, phase="search", zh=PHASE_ZH["search"], event="missed")
        blind_done = rc.phase == "blind_done"
        return PlayOut(
            drive=True,
            head=HeadPose() if blind_done else look_head,
            head_blend=1.0,
            phase="line_up",
            zh=PHASE_ZH["line_up"],
            quiet=True,
        )

    def _p_grab(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        active = frame.body.active_skill == "ground_pick"
        if self.grab_started_ms is None:
            if not active:
                if t_ms - self.acted_ms > PICK_START_TIMEOUT_MS:
                    self.closed_ms = None
                    self.grab_started_ms = None
                    self.regripping = False
                    thing.grab_fails += 1
                    self.fails += 1
                    self.last_event = "missed"
                    if thing.grab_fails >= MAX_FAILS:
                        return self._give_up(t_ms, "grab")
                    thing.carried_ms = t_ms
                    self._go("search", t_ms)
                    return PlayOut(
                        drive=True,
                        mouth=0.0,
                        phase="search",
                        zh=PHASE_ZH["search"],
                        event="missed",
                    )
                return PlayOut(
                    drive=True,
                    skill="ground_pick",
                    skill_seq=self.skill_seq,
                    mouth=0.0,
                    head=HeadPose(),
                    head_blend=1.0,
                    phase="grab",
                    zh=PHASE_ZH["grab"],
                    quiet=True,
                    event="grab" if t_ms == self.acted_ms else None,
                )
            self.grab_started_ms = t_ms
        since = t_ms - self.grab_started_ms
        mouth = 1.0 if MOUTH_OPEN_MS <= since < MOUTH_CLOSE_MS else 0.0
        if since >= MOUTH_CLOSE_MS and self.closed_ms is None:
            self.closed_ms = t_ms
        if active or since < PICK_HOLD_MS or not legs_still(frame):
            return PlayOut(
                drive=True,
                skill="ground_pick",
                skill_seq=self.skill_seq,
                mouth=mouth,
                head=HeadPose(),
                head_blend=1.0,
                phase="grab",
                zh=PHASE_ZH["grab"],
                quiet=True,
                event="grab" if since == 0 and t_ms == self.acted_ms else None,
            )
        if held_now:
            self.grab_started_ms = None
            thing.held += 1
            self.held_n += 1
            self.holding = True
            if not self.regripping:
                self.via_done = False
            self.regripping = False
            self.leg_ms = t_ms
            thing.axis, thing.axis_n = None, 0  # (in the beak it lies however it lies now)
            self.last_event = "grabbed"
            self._go("carry", t_ms)
            return PlayOut(
                drive=True,
                mouth=0.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="carry",
                zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._p_grab.912146"),
                event="grabbed",
            )
        self.closed_ms = None
        self.grab_started_ms = None
        self.regripping = False
        thing.grab_fails += 1
        self.fails += 1
        self.last_event = "missed"
        if thing.grab_fails >= MAX_FAILS:
            return self._give_up(t_ms, "grab")
        # it lies right under the beak (nearer than the eye sees, maybe nudged): back off and look
        # for it as after a slip, then round to its (other) long side again
        thing.carried_ms = t_ms
        self._go("search", t_ms)
        return PlayOut(
            drive=True,
            mouth=0.0,
            phase="search",
            zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._p_grab.b3c218"),
            event="missed",
        )

    def _p_backoff(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        """After a miss: back away in a small arc and plan again (from the other long side)."""
        since = t_ms - self.phase_ms
        if since < CALIB.back_arc_ms:
            left = self._draw() < 0.5
            return PlayOut(
                drive=True,
                vx=-0.40,
                vyaw=1.0 if left else -1.0,
                fine=True,
                reverse=True,
                phase="backoff",
                zh=PHASE_ZH["backoff"],
                quiet=True,
            )
        if since < CALIB.back_arc_ms + TAIL_MS or not legs_still(frame):
            return PlayOut(drive=True, phase="backoff", zh=PHASE_ZH["backoff"], quiet=True)
        self._go("survey", t_ms)
        return PlayOut(drive=True, phase="survey", zh=PHASE_ZH["survey"], quiet=True)

    def _p_carry(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        if not held_now:
            # it slipped out (the mouth joint shut): stop, look down, find it, pick it up again —
            # and from now on carry it in shorter legs, re-gripping before it slips (learned)
            held_for = t_ms - self.leg_ms
            learned = _clip_int(held_for // 2, RELAY_MIN_MS, RELAY_MAX_MS)
            self.relay_ms = learned if self.relay_ms is None else min(self.relay_ms, learned)
            self.holding = False
            self.closed_ms = None
            self.slips += 1
            self.last_event = "slipped"
            thing.x = odom[0] + 0.10 * math.cos(odom[2])
            thing.y = odom[1] + 0.10 * math.sin(odom[2])
            thing.seen_ms = t_ms
            thing.carried_ms = t_ms
            thing.axis, thing.axis_n = None, 0
            if self.slips > MAX_SLIPS:
                return self._give_up(t_ms, "slips")
            self._go("search", t_ms)
            return PlayOut(drive=True, phase="search", zh=PHASE_ZH["search"], event="slipped")
        bearing, rng = bearing_range(odom, home[0], home[1])
        if rng < HOME_ARRIVE_M:
            self._go("drop", t_ms)
            return self._p_drop(t_ms, frame, book, thing, odom, home, held_now)
        if (
            self.relay_ms is not None
            and t_ms - self.leg_ms >= self.relay_ms
            and rng > HOME_ARRIVE_M + 0.12
        ):
            self._go("regrip", t_ms)  # (it knows the toy slips if it carries it longer)
            self.regrip_t0 = None
            return self._p_regrip(t_ms, frame, book, thing, odom, home, held_now)
        if not self.via_done:
            pre = self.way_in(odom)
            if pre is None or math.hypot(pre[0] - odom[0], pre[1] - odom[1]) < 0.20:
                self.via_done = True  # (in the corridor beside the dock: the last leg)
            else:
                bearing, _ = bearing_range(odom, pre[0], pre[1])
        # straight for the spot beside the dock, steering on its bearing (the dock program's own
        # goto: kp 2.5, the walk's drift fed forward) — not the brain's wandering walk, which
        # took a carried toy 1 m out of the way here — at the slowest arc (a steadier beak);
        # home behind it: a reverse arc turns the nose toward it (~33° a second) while the body
        # backs that way (a forward arc there walked the toy away: this rig, 4 relay legs, +0.5 m)
        if abs(bearing) > BEHIND_RAD:
            return PlayOut(
                drive=True,
                vx=-0.40,
                vyaw=1.0 if bearing > 0 else -1.0,
                fine=True,
                reverse=True,
                mouth=0.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="carry",
                zh=PHASE_ZH["carry"],
                quiet=True,
                target=Target("place", bearing, rng),
            )
        vyaw = _clip(2.5 * bearing + FF_VYAW, -1.0, 1.0)
        return PlayOut(
            drive=True,
            vx=CARRY_VX,
            vyaw=vyaw,
            mouth=0.0,
            head=HeadPose(),
            head_blend=1.0,
            phase="carry",
            zh=PHASE_ZH["carry"],
            quiet=True,
            target=Target("place", bearing, rng),
        )

    def _p_regrip(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        """Stop, let go, one head-down step in, pick it up again (see REGRIP_*)."""
        since = t_ms - self.phase_ms
        zh = PHASE_ZH["regrip"]
        if self.regrip_t0 is None:
            # stop first (the legs still, or 1.5 s at most), the beak still shut on it
            if since < REGRIP_STOP_MS or (not legs_still(frame) and since < REGRIP_STOP_MS + 1500):
                return PlayOut(
                    drive=True,
                    mouth=0.0,
                    head=HeadPose(),
                    head_blend=1.0,
                    phase="regrip",
                    zh=zh,
                    quiet=True,
                )
            self.regrip_t0 = t_ms
        k = t_ms - self.regrip_t0
        down = HeadPose(np=0.5, hp=-0.5)
        t_head = REGRIP_OPEN_MS
        t_creep = t_head + REGRIP_HEAD_MS
        t_settle = t_creep + REGRIP_CREEP_MS
        t_pick = t_settle + REGRIP_SETTLE_MS
        thing.carried_ms = t_ms  # (its moves now are its own doing)
        if k < t_head:
            if self.holding:
                self.holding = False  # let go on purpose (the fin lands ~0.1 m ahead)
                thing.x = odom[0] + 0.098 * math.cos(odom[2])
                thing.y = odom[1] + 0.098 * math.sin(odom[2])
                thing.seen_ms = t_ms
                self.regrips += 1
                self.last_event = "regrip"
            return PlayOut(
                drive=True,
                mouth=1.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="regrip",
                zh=zh,
                quiet=True,
            )
        if k < t_creep:
            return PlayOut(
                drive=True, mouth=0.0, head=down, head_blend=1.0, phase="regrip", zh=zh, quiet=True
            )
        if k < t_settle:
            return PlayOut(
                drive=True,
                vx=0.40,
                vyaw=FF_VYAW,
                fine=True,
                mouth=0.0,
                head=down,
                head_blend=1.0,
                phase="regrip",
                zh=zh,
                quiet=True,
            )
        if k < t_pick or (not legs_still(frame) and k < t_pick + 1500):
            return PlayOut(
                drive=True,
                mouth=0.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="regrip",
                zh=zh,
                quiet=True,
            )
        self.acted_ms = t_ms
        self.grab_started_ms = None
        self.skill_seq += 1
        self.grabs += 1
        self.regripping = True
        self._go("grab", t_ms)
        return self._p_grab(t_ms, frame, book, thing, odom, home, held_now)

    def _p_search(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        """It slipped out: the toy lies about under the beak, nearer than the eye sees — back off
        in an arc (the nose toward where it fell) until a look finds it, then survey it again."""
        since = t_ms - self.phase_ms
        thing.carried_ms = t_ms  # (where it fell is a guess: finding it elsewhere is no roll)
        a, left = rel_of(odom, thing.x, thing.y)
        gaze, head = head_look(max(a, 0.12), left)
        if thing.seen_ms > self.phase_ms + TAIL_MS and legs_still(frame):
            self.search_arcs = 0
            self._go("survey", t_ms)
            return PlayOut(
                drive=True,
                head=head,
                head_blend=0.8,
                gaze=gaze,
                phase="survey",
                zh=PHASE_ZH["search"],
                quiet=True,
            )
        if since < 2 * SEARCH_ARC_MS:
            # two back arcs, one each way: ~15 cm back, the heading about kept (where it fell
            # is nearer than the eye sees and a little to one side)
            first = since < SEARCH_ARC_MS
            return PlayOut(
                drive=True,
                vx=-0.40,
                vyaw=1.0 if first else -1.0,
                fine=True,
                reverse=True,
                phase="search",
                zh=PHASE_ZH["search"],
                quiet=True,
            )
        if (
            a < EYE_NEAR_M + 0.05
            and since >= 2 * SEARCH_ARC_MS + SEARCH_MS // 2
            and self.search_arcs < SEARCH_ARCS
        ):
            self.search_arcs += 1
            self.phase_ms = t_ms  # (still not seen and near: back off again)
        if since > SEARCH_MS + 2 * SEARCH_ARC_MS:
            self.search_arcs = 0
            return self._give_up(t_ms, "lost")
        # where it fell is a guess (the beak's side): look about it, a look held at a time
        look = scan_gaze(
            math.atan2(left, max(a, 0.12)),
            max(0, since - 2 * SEARCH_ARC_MS),
            elevation=-0.6,
            width=0.3,
        )
        return PlayOut(drive=True, gaze=look, phase="search", zh=PHASE_ZH["search"], quiet=True)

    def _p_drop(self, t_ms, frame, book, thing, odom, home, held_now):  # type: ignore[no-untyped-def]
        since = t_ms - self.phase_ms
        if since < TAIL_MS or not legs_still(frame) and since < 2000:
            return PlayOut(
                drive=True,
                mouth=0.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="drop",
                zh=PHASE_ZH["drop"],
                quiet=True,
            )
        if since < TAIL_MS + PUT_DOWN_MS:
            return PlayOut(
                drive=True,
                mouth=1.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="drop",
                zh=PHASE_ZH["drop"],
                quiet=True,
            )
        if self.holding:
            self.holding = False
            self.closed_ms = None
            thing.home = True
            thing.carried_ms = t_ms
            thing.x = odom[0] + 0.098 * math.cos(odom[2])  # (it lands ~0.1 m ahead, as measured)
            thing.y = odom[1] + 0.098 * math.sin(odom[2])
            self.homes += 1
            self.last_event = "dropped_home"
            return PlayOut(
                drive=True,
                mouth=0.0,
                phase="drop",
                zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._p_drop"),
                event="dropped_home",
                vocal="coo",
            )
        if since < TAIL_MS + PUT_DOWN_MS + STEP_BACK_MS:
            # a step back from the wall in an arc, the nose away from the dock (its guides are a
            # hand's width away: two of this rig's ducks walked onto them after a delivery)
            return PlayOut(
                drive=True,
                vx=-0.40,
                vyaw=-self.drop_side(),
                fine=True,
                reverse=True,
                mouth=0.0,
                head=HeadPose(),
                head_blend=1.0,
                phase="drop",
                zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._p_drop.backoff"),
                quiet=True,
            )
        self._go("idle", t_ms)
        self.target = None
        return PlayOut(
            drive=True,
            mouth=0.0,
            phase="done",
            zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._p_drop"),
        )

    def _give_up(self, t_ms: int, why: str) -> PlayOut:
        self.gave_up += 1
        self.holding = False
        self.closed_ms = None
        self.last_event = f"gave_up:{why}"
        self._go("idle", t_ms)
        self.target = None
        return PlayOut(
            phase="idle",
            zh=msg("neuro.incoming.incoming_skill_carry.CarryHome._give_up"),
            event=self.last_event,
            mouth=0.0,
        )

    def view(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "target": self.target,
            "holding": self.holding,
            "grabs": self.grabs,
            "held": self.held_n,
            "fails": self.fails,
            "slips": self.slips,
            "homes": self.homes,
            "gave_up": self.gave_up,
            "crooked": self.crooked,
            "plans": self.plans,
            "regrips": self.regrips,
            "relay_ms": self.relay_ms,
            "carried_m": round(self.carried_m, 2),
        }


def carry_features(
    book: ThingBook,
    odom: tuple[float, float, float],
    t_ms: int,
    home: tuple[float, float] | None,
    holding: bool,
) -> dict[str, float]:
    """``toy`` (a toy it can pick up — or has not tried yet — away from home, seen lately, near)
    and ``carrying`` (something is in its beak on the way home), 0..1."""
    toy = 0.0
    if home is not None:
        for th in book.things.values():
            age = (t_ms - th.seen_ms) / 1000.0
            if age > TOY_MEMORY_S or th.rolls or th.grab_fails >= MAX_FAILS:
                continue
            if th.missing_ms > th.seen_ms:
                continue
            if th.axis is None and th.held == 0:
                continue
            if math.hypot(th.x - home[0], th.y - home[1]) < HOME_TOY_M:
                continue
            _, rng = bearing_range(odom, th.x, th.y)
            # near, and remembered for a few minutes (it knows where it left it)
            near = _clip(1.2 - 0.3 * rng) * _clip(1.0 - age / TOY_MEMORY_S)
            toy = max(toy, near * (1.0 if th.carriable else UNTRIED_TOY))
    return {"toy": toy, "carrying": 1.0 if holding else 0.0}


__all__ = ["CARRY_VERSION", "CarryHome", "carry_features"]

# (the toy's long axis is kept on PlayThing: see skill_common.PlayThing.axis)
_ = wrap_angle
