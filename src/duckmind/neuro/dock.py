"""dock: the neural duck backs into its charging dock by itself (dock:build 2026-09-28).

The owner confirmed the back-in dock (2026-09-28 「6. 可以」) and reversing while docking and when
cornered (「1. 可以」「2. 可以」). Phase 3 measured the choreography in duck-body's ``dockexec``
(``backin``, dock:measure's J-hook; ``docs/validation/evidence/phase3/
dockexec_backin_owner0928.json`` in the main tree: docked 25/30 in default_home on the right-hand
J-hook, 26/30 in the apartment on the mirrored one). This module ports its steps onto the neural
duck — the same geometry, numbers and order — with three differences:

* **its own senses.** A *fix* is one reading of the dock's face tag by the head camera: a
  ``nest`` :class:`~duckmind.neuro.contract.Sighting` with ``range_m``, ``bearing`` and
  ``facing`` (duck-eye NB-v1's tag pose — modelled in the prototype with the reader's measured
  errors, :mod:`duckmind.world.neuro_senses`), turned into the duck's pose in the dock's frame
  (:func:`pose_from_read`); between fixes its own contact odometry carries the pose. A nest
  sighting without a range (the dock's light in the dark, the cyan panel) gives a direction only
  (owner C5): it turns the carried heading, never the position. The PG pin
  (``power.on_dock``, a real sensor op) says it is on the contacts. Where it charged last stays
  remembered in its odometry (the dock's frame when PG closed). Nothing here reads the world.
* **its own body path.** The twist, the head's aim and the posture go out through the brain's
  :class:`~duckmind.neuro.contract.MotorCommand` (``docking``: the choreography's own twist,
  :func:`~duckmind.neuro.contract.dock_walk` — forward 0.30–0.40 m/s, the U-turn and the tight
  arcs at 0.30, reversing at −0.40), the motor and the body link's safety (falls, pickup, the
  guard, the deadman), like every other walk of the neural duck.
* **its own brain.** The brain chooses ``return_dock`` (a low battery, sleepy, the wish for its
  nest) and walks toward the tag it sees or the place it remembers; it hands the legs to this
  program once it has a fix (:meth:`DockRun.engage`). A reflex (a startle, a freeze) or the body
  link holding the legs pauses the program and it goes on from where it is; the brain may drop
  the act for something more pressing (:meth:`DockRun.abort`). Where the manoeuvre allows it the
  brain keeps its glances and small moments (walking the J-hook, pausing, charging); while it
  looks at the tag, turns round or backs in the head stays aimed or at HOME (as phase 3 did:
  reversing with a moving head was never measured). Once on the contacts it sits and charges;
  when it wants to walk again it gets up and walks straight out along the dock's axis
  (:meth:`DockRun.leave`: a backed-in duck faces the room — no reversing out).

The J-hook (dockexec's ``DockConfig`` / ``GEOMETRY["backin"]``; mm in the dock's frame: x from the
wall along its axis into the room, y to the dock's left facing the room; heading 0 = facing the
room, π = facing the dock):

0. ``stand``; ``fix`` — stop, the legs' tail, look at the tag (with a remembered frame one
   look will do: unread, it walks to the lane on that frame and looks from there);
1. ``to_lane`` — unless already on the lane facing the dock, walk to the lane's far point (1.5 m
   out, 375 mm to the turning side; tight arcs vx 0.30 / vyaw ±1.0 to turn, never on the spot) and
   arc round to face the dock within 20°; ``fix``; ``to_turn`` — follow the lane in to 1.15 m
   (a carrot on the lane, PI + the gait's drift feed-forward), one look a second; walking into
   something on the way (proprio's ``blocked``): ``back_step`` (1 s straight back at −0.40) and
   the walk again, at most twice a try (starting beside the dock facing across its guides,
   where any forward arc sweeps into one, it first backs in an arc — vx −0.40, the full yaw rate
   — until its nose points out to the room);
2. ``pause`` 0.6 s; ``fix``; ``uturn`` — from standing toward the axis (vx 0.30, vyaw ∓1.0) until
   the odometry's heading has turned 178° (3107 mrad): it ends on the axis ~1.1 m out, facing the
   room; ``brief`` 0.4 s;
3. ``reverse_in`` at −0.40: the first 0.8 s straight (feed-forward only), then steered on
   odometry toward the axis; no 15 mm of progress after 2 s: one straight 1.2 s reverse;
4. the alignment gate: inside 0.6 m, the 0.5 s-averaged pose more than 80 mm or 20° off the axis
   → ``realign`` (walk out along the axis to 1 m, straightening) and reverse again, at most twice
   (a third time: ``walk_out``, a U-turn and the whole approach again, at most twice in all);
5. **the face stop** (blocked = arrived): reversing until blocked — after 30 mm, the body's own
   stall (proprio's ``blocked``) or less than 12 mm of progress in 1.5 s, with the rear believed
   within 0.35 m of the face (a stall mid-room is the body link holding the legs, not the face)
   — or odometry puts the rear at the face; the odometry there is the reference for what
   follows (how far out the rear is); ``bump_out`` — walk out just far enough that the sit's
   ~70 mm slide back ends on the plates (:attr:`DockConfig.bump_out_mm` by that odometry, the
   coast after the stop comes on top; 0.7 s at 0.35 is the cap); ``pause``;
6. ``sit``; ``pg`` — PG after 0.8 s; open → ``stand``, ``sit`` again (no walking), at most three
   sits — a fourth, fifth… only while the last sit-and-stand still took the rear in
   (:attr:`DockConfig.sit_progress_mm` by the odometry) and it is still believed out of reach
   (:attr:`DockConfig.seat_reach_mm`), at most :attr:`DockConfig.max_sits_progress` — then the
   try has failed.

The turning side is the home's (``right``: the lane on the dock's right, a right U-turn;
``left``: mirrored, where the room has a wall there — the apartment's corridor); it is how the
family set the dock up, given at start (:attr:`DockConfig.turn_side`), not sensed.

Pure and clock-injected (``DockIn.t_ms``); floats in the prototype (phase 5: the tracker and the
legs are the same arithmetic as duck-body's dockexec, integers there).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro.contract import REVERSE_VX, Sighting

DOCK_VERSION = "neuro-dock-1"
SIDES = ("right", "left")
MODES = ("idle", "docking", "home", "leaving")
BUMP_PHASES = frozenset({"to_lane", "to_turn"})  # walking the J-hook forward: a bump is a thing


@dataclass(frozen=True)
class DockGeometry:
    """Where things are along the dock's axis (mm from the wall; ``world.dock`` backin, with
    dock:measure's calibration — dockexec's ``GEOMETRY["backin"]``)."""

    face_mm: int = 111  # the front of the face panel: where the big tag is
    line_up_mm: int = 1000  # where the alignment gate walks out to
    stop_mm: int = 289  # the pose-stop fallback (dock:measure's calibration)
    seat_mm: int = 172  # where the seated trunk ends up (world.dock.seat_x_mm backin)
    turn_mm: int = 1150  # where the U-turn starts
    uturn_offset_mm: int = 375  # the lane's offset from the axis (a U-turn's width)
    far_mm: int = 1500  # the lane's far point, where the J-hook begins
    rear_mm: int = 47  # the trunk's rear behind the trunk origin (world.dock.REAR_MM)
    guides_mm: int = 441  # the guides' mouth (depth 110 + guide length 330): out of the dock
    half_width_mm: int = 242  # half the guides' mouth (460 / 2 + a guide)

    @property
    def face_contact_mm(self) -> int:
        """Where the trunk is when its rear meets the face."""
        return self.face_mm + self.rear_mm


@dataclass(frozen=True)
class DockConfig:
    """dockexec's numbers (``DockConfig`` backin) and the neural duck's own (its looks, the
    engagement, the way out)."""

    turn_side: str = "right"
    geometry: DockGeometry = field(default_factory=DockGeometry)
    max_sits: int = 3  # sits per approach — the first, then "stand, sit again" twice
    # (headless 2026-10-17) … and more while each one still takes the rear in: a sit slides it ~72 mm
    # back, a stand pushes it ~50 mm forward, so a sit-and-stand nets only ~20 mm; started 0.14 m out
    # (the 1001 run: 76 / 49 / 28 mm) three of them end a few cm short of the contacts.
    # At most this many sits an approach, however it goes …
    max_sits_progress: int = 6
    # … a sit past max_sits needs the rear (odometry) this much further in than after the sit before …
    sit_progress_mm: float = 8.0
    # … and believed further out than this from the face (the plates reach ~25 mm)
    seat_reach_mm: float = 25.0
    max_approaches: int = 2  # a second J-hook only when the alignment gate fails 3 times
    vx: float = 0.40  # approaching a specific thing (owner A3)
    reverse_vx: float = REVERSE_VX  # −0.40 (vx −0.3 steps on the spot)
    uturn_vx: float = 0.30  # dock:measure's U-turn: vx 0.30, vyaw ±1.0, ~6 s
    uturn_vyaw: float = 1.0  # its size; the side gives the sign (right: −, left: +)
    uturn_turn: float = 3.107  # rad, 178°
    uturn_timeout_ms: int = 12_000
    arc_vx: float = 0.30  # a tight arc to turn (vx 0.30, vyaw ±1.0): never turn on the spot
    arc_vyaw: float = 1.0
    arc_rad: float = 0.95  # the bearing this far off: turn with a tight arc
    arc_radius_mm: float = 230.0  # a tight arc's radius (0.17–0.25 m measured)
    goto_kp: float = 2.5
    lane_arrive_mm: float = 150.0
    face_tol: float = 0.35  # facing the dock within 20°
    goto_timeout_ms: int = 30_000
    face_timeout_ms: int = 10_000
    lane_total_ms: int = 40_000  # the whole walk to the lane, reflex holds included
    # (live check 2026-09-28) walking the J-hook into something (a guide's side, a chair leg:
    # proprio's ``blocked``): one short straight step back (vx −0.40 for 1 s ≈ 0.1 m — the
    # owner's cornered back step), then the walk is planned again; a third time: the try ends
    back_step_ms: int = 1000
    max_back_steps: int = 2
    # (headless 2026-09-28) beside the guides facing across them, a straight 1-s step back
    # (~50 mm on the real legs) left the first forward arc sweeping over a guide's end (it
    # tripped, 2 of 3): it backs in an arc instead — curiosity's back_arc, vx −0.40 with the full
    # yaw rate, the nose swinging out toward the room — until it faces within beside_face_rad of
    # the room's way (by its belief), at most beside_back_ms
    beside_face_rad: float = 0.6
    beside_back_ms: int = 3500
    # (headless 2026-09-28) a new approach begun from inside the guides (its re-seats failed,
    # standing in the dock facing the room): straight out along the axis first to this far from
    # the wall — the arcs toward the lane from inside swept it over a guide (4 falls in 57)
    out_first_mm: float = 800.0
    out_first_ms: int = 8000
    out_face_rad: float = 1.2  # … when it faces out within this; more askew it does not walk
    lane_step_ms: int = 1000  # following the lane in: one look a second
    final_blind_mm: float = 350.0  # the last stretch to a goal goes on odometry, without looks…
    max_blind_mm: float = 450.0  # …and when no tag reads this close to it on a fix …
    blind_fix_age_ms: int = 15_000  # …no older than this
    carrot_mm: float = 250.0  # steer at a point on the lane this far ahead
    kp: float = 2.5  # vyaw = kp · heading error + ki · ∫ + feed-forward
    ki: float = 0.8
    ff: float = 0.150  # the walk's own rightward turn (−8.5°/s at vx 0.35)
    ff_reverse: float = 0.080  # (−4.5°/s at vx −0.4)
    fix_misses: int = 4  # looks in a row without the tag before giving the try up
    sit_timeout_ms: int = 8000  # (the body link settles the head first: a little longer)
    stand_timeout_ms: int = 8000
    # after the sit, before reading PG (dockexec 0.8 s; + the live PG's 5 Hz read, ≤ 0.2 s old)
    pg_settle_ms: int = 1200
    tail_ms: int = 450  # the legs keep stepping ~0.45 s after a stop (lookplan.TAIL_MS)
    pause_ms: int = 600  # dock:measure's stops: before the U-turn and the sit …
    brief_ms: int = 400  # … and after the U-turn
    reverse_straight_ms: int = 800
    kick_after_ms: int = 2000
    kick_ms: int = 1200
    kick_min_mm: float = 15.0
    reverse_max_ms: int = 15_000
    gate_x_mm: float = 600.0
    gate_mm: float = 80.0
    gate_rad: float = 0.349  # 20°
    gate_window_ms: int = 500  # one gait cycle: the trunk sways ±20–25 mm with every step
    max_relines: int = 2
    realign_tol_mm: float = 30.0
    realign_tol_rad: float = 0.09
    realign_timeout_ms: int = 12_000
    blocked_min_mm: float = 30.0
    blocked_window_ms: int = 1500
    blocked_progress_mm: float = 12.0
    # (live check 2026-09-28) blocked = arrived only where the face can be: the rear believed
    # within this of it (the pose after the U-turn is good to ~0.1 m). Farther out a stall is
    # the body link holding the legs (a leg unloaded…): it reversed on, not sat mid-room
    face_zone_mm: float = 350.0
    bump_walk_vx: float = 0.35  # after the face stop: walk out this fast …
    bump_walk_ms: int = 700  # … at most this long (1.0 s overshoots; 2.0 s walked ~260 mm) …
    # (headless 2026-10-17) … and no further than this by odometry from the face stop. The old fixed
    # 0.7 s walked ~58 mm and coasted ~42 mm on, so the first sit began a median 107 mm out (74–144 mm,
    # 107 approaches) where one sit seats from ~30–90 mm (body-only, 4 starts each: PG closed 19/20 from
    # 30–90 mm out, 1/4 from 105, 0/12 from 120–150, 0/8 from 0–15 mm: at the face the sit does not
    # reach the plates). 20 mm of walking + the coast starts it at ~65 mm. 0: sit where it ended
    bump_out_mm: float = 20.0
    unsure_wait_ms: int = 3000  # the body link holds the legs (legs light …): wait this long
    # ── the neural duck's own ──
    fix_look_ms: int = 2500  # one look: the legs quiet, then this long for a tag reading
    look_elevation: float = -0.10  # rad: the tag's centre is ~11 cm up, the lens ~20 cm
    look_max_rad: float = 0.9  # the head's yaw reach for an aim (the envelope is ±1.0)
    engage_m: float = 2.8  # a trusted fix this near: the program takes the legs
    remember_m: float = 3.2  # a trusted reading this near, any time: where the dock is (odometry)
    # the remembered lane point this near: take the legs (live check 2026-09-28: 0.45 m let the
    # brain's own wide turns carry it beside the dock, against a guide; the program's tight arcs
    # walk the last 1.8 m to the lane's far point)
    rendezvous_mm: float = 1800.0
    halt_max_ms: int = 25_000  # a reflex holding the body longer than this ends the try
    held_ms: int = 350  # commanded but not moving (robotd's applied twist ~0) this long: held
    head_settle_ms: int = 300  # the head at HOME this long before a U-turn or a reverse
    leave_vx: float = 0.35  # the way out along the axis (the slowest sure step)
    leave_mm: float = 650.0  # … until the trunk is this far from the wall (past the guides)
    leave_timeout_ms: int = 6000
    # … steering hard back onto the axis: seated, the sit's slide leaves it up to ~20° and ~40 mm
    # off, and between the channel's walls the hips (±71 mm) have ±59 mm to spare
    leave_kp: float = 3.5
    leave_carrot_mm: float = 180.0
    reseat_after_ms: int = 3000  # seated in the dock, PG open this long: stand and sit again
    remembered_max_m: float = 25.0  # a remembered dock frame: trusted for this much walking
    contact_heading_sigma: float = 0.2  # rad: a docked duck faces the room within ~±20°

    def lane_y(self) -> float:
        """The lane: to the dock's right (y < 0) or, mirrored, its left."""
        offset = float(self.geometry.uturn_offset_mm)
        return offset if self.turn_side == "left" else -offset

    def turn_sign(self) -> int:
        """The U-turn's direction: a right turn (vyaw < 0) from the right lane, left mirrored."""
        return 1 if self.turn_side == "left" else -1


@dataclass(frozen=True, slots=True)
class DockPose:
    """The trunk in the dock's frame: x from the wall along the axis (mm), y to the dock's left
    facing the room (mm), heading (rad; 0 = facing the room, π = facing the dock)."""

    x: float
    y: float
    h: float


@dataclass(frozen=True, slots=True)
class TagRead:
    """One reading of the dock's face tag (a camera pass, still or not): range and bearing from
    the trunk origin (level frame; m, rad + left), the duck's heading off square-on (``facing``,
    rad + left; None: the mirror pose fits as well — the range and bearing still hold), the
    brain time the pass was delivered and the odometry at the capture (mm, mm, rad)."""

    range_m: float
    bearing: float
    facing: float | None
    t_ms: int
    odom: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DockIn:
    """What the program reads each tick — all from the duck's own SensorFrame (and the brain's
    own reflex state): the contact odometry (mm, mm, rad), the posture, the PG pin, robotd's
    applied twist (``move.applied``: what the legs were really given), proprio's ``blocked`` edge,
    a new tag reading or a direction-only sighting of the dock this tick, whether a reflex holds
    the body, whether the measured head is at HOME, whether the odometry frame broke (picked up,
    set down) and whether it fell."""

    t_ms: int
    odom: tuple[float, float, float]
    posture: str = "stand"
    pg: bool = False
    applied: tuple[float, float] = (0.0, 0.0)
    blocked: bool = False
    tag: TagRead | None = None
    cue: float | None = None
    halted: bool = False
    head_home: bool = True
    upset: bool = False
    fell: bool = False


@dataclass(frozen=True, slots=True)
class DockOut:
    """What the program wants this tick. ``drive``: it owns the legs (the twist below, 0 = hold
    still); ``reversing``; ``posture`` wanted (None: the brain's); ``look``: aim the head at
    (bearing, elevation) — a fix; ``head_still``: the head at HOME (turning round, backing in);
    ``quiet``: no small moments now; ``phase`` and ``zh`` (plain words); ``event``: something
    that just happened (``docked``, ``failed``, ``left``, ``face``, ``resit``…)."""

    drive: bool = False
    vx: float = 0.0
    vyaw: float = 0.0
    reversing: bool = False
    posture: str | None = None
    look: tuple[float, float] | None = None
    head_still: bool = False
    quiet: bool = False
    phase: str = "idle"
    zh: str = ""
    event: str | None = None


PHASE_ZH: dict[str, str] = {
    "idle": "",
    "stand": msg("neuro.dock.phase.stand"),
    "fix": msg("neuro.dock.phase.fix"),
    "to_lane": msg("neuro.dock.phase.to_lane"),
    "to_turn": msg("neuro.dock.phase.to_turn"),
    "pause": msg("neuro.dock.phase.pause"),
    "brief": msg("neuro.dock.phase.brief"),
    "uturn": msg("neuro.dock.phase.uturn"),
    "reverse_in": msg("neuro.dock.phase.reverse_in"),
    "realign": msg("neuro.dock.phase.realign"),
    "walk_out": msg("neuro.dock.phase.walk_out"),
    "bump_out": msg("neuro.dock.phase.bump_out"),
    "back_step": msg("neuro.dock.phase.back_step"),
    "beside": msg("neuro.dock.phase.beside"),  # (the back arc beside the guides: phase back_step)
    "out_first": msg("neuro.dock.phase.out_first"),
    "sit": msg("neuro.dock.phase.sit"),
    "pg": msg("neuro.dock.phase.pg"),
    "home": msg("neuro.dock.phase.home"),
    "leave": msg("neuro.dock.phase.leave"),
}
RESULT_ZH: dict[str, str] = {
    "no_tag": msg("neuro.dock.fail.no_tag"),
    "tries": msg("neuro.dock.fail.tries"),
    "gate": msg("neuro.dock.fail.gate"),
    "lost": msg("neuro.dock.fail.lost"),
    "lane": msg("neuro.dock.fail.lane"),
    "blocked": msg("neuro.dock.fail.blocked"),
    "face": msg("neuro.dock.fail.face"),
    "uturn": msg("neuro.dock.fail.uturn"),
    "reverse": msg("neuro.dock.fail.reverse"),
    "realign": msg("neuro.dock.fail.realign"),
    "walk_out": msg("neuro.dock.fail.walk_out"),
    "stand": msg("neuro.dock.fail.stand"),
    "sit": msg("neuro.dock.fail.sit"),
    "halt": msg("neuro.dock.fail.halt"),
    "held": msg("neuro.dock.fail.held"),
    "upset": msg("neuro.dock.pending.2d739f"),
    "fell": msg("neuro.dock.pending.022d9e"),
    "switched": msg("neuro.dock.fail.switched"),
    "wedged": msg("neuro.dock.fail.wedged"),
}


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose_from_read(
    read: TagRead, geometry: DockGeometry, heading: float | None = None
) -> DockPose | None:
    """The duck's pose from one tag reading (dockexec's ``pose_from_fix`` on the neural duck's
    Sighting): heading = π + facing (square on, facing the dock, is π); the carried ``heading``
    when the reader is unsure of the facing (the mirror pose); ``None`` without either."""
    if read.facing is not None:
        heading = wrap(math.pi + read.facing)
    if heading is None:
        return None
    toward = heading + read.bearing
    r = read.range_m * 1000.0
    return DockPose(geometry.face_mm - r * math.cos(toward), -r * math.sin(toward), heading)


@dataclass
class _Tracker:
    """The dock-frame pose carried by odometry from the last fix."""

    fix: DockPose | None = None
    odom0: tuple[float, float, float] | None = None  # (x mm, y mm, h rad) at the fix

    def set(self, pose: DockPose, odom: tuple[float, float, float]) -> None:
        self.fix, self.odom0 = pose, odom

    def now(self, odom: tuple[float, float, float] | None) -> DockPose | None:
        if self.fix is None:
            return None
        if odom is None or self.odom0 is None:
            return self.fix
        dx, dy = odom[0] - self.odom0[0], odom[1] - self.odom0[1]
        rot = self.fix.h - self.odom0[2]
        c, s = math.cos(rot), math.sin(rot)
        return DockPose(
            self.fix.x + c * dx - s * dy,
            self.fix.y + s * dx + c * dy,
            wrap(self.fix.h + (odom[2] - self.odom0[2])),
        )


@dataclass
class _Leg:
    """One driving leg: forward (``reverse`` False) or backward toward ``goal_x`` along the axis
    (steering onto ``lane_y``), for at most ``ms``."""

    goal_x: float
    reverse: bool
    ms: int
    lane_y: float = 0.0
    started_ms: int = 0
    integral: float = 0.0
    last_ms: int = 0
    settled: bool = False
    moved_from: DockPose | None = None
    moving_ms: int | None = None
    samples: deque[tuple[int, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    kicked: bool = False
    kick_until: int | None = None
    restart: bool = False
    first_ms: int | None = None


class DockRun:
    """The back-in choreography as a step machine (see the module doc). One per duck; it
    remembers the dock's frame between dockings."""

    def __init__(self, config: DockConfig | None = None) -> None:
        self.cfg = config or DockConfig()
        if self.cfg.turn_side not in SIDES:
            raise ValueError(f"dock side {self.cfg.turn_side!r} not in {SIDES}")
        self.mode = "idle"
        self.phase = "idle"
        self.phase_ms = 0
        self.plan: list[str] = []
        self.tracker = _Tracker()
        self.fix_ms: int | None = None
        self.fix_src = ""  # tag | contact | remembered
        self.leg: _Leg | None = None
        self.sub: str | None = None
        self.sits = 0
        self.approaches = 1
        self.relines = 0
        self.lane_loops = 0
        self.misses = 0
        self.looks = 0
        self.look_t0: int | None = None
        self.look_aim: float = 0.0
        self.cmd_ms: int | None = None  # the program last commanded a move
        self.applied_ms: int | None = None  # robotd last applied a move (the legs were given one)
        self.turned = 0.0
        self.last_h: float | None = None
        self.home_since: int | None = None
        self.head_home_since: int | None = None
        self.sat_ms: int | None = None
        self.halt_since: int | None = None
        self.held_since: int | None = None
        self.unsure_since: int | None = None
        self.open_since: int | None = None
        self.walked_mm = 0.0  # odometry path since the dock frame was last confirmed
        self._odom_prev: tuple[float, float, float] | None = None
        self.result: str | None = None
        self.face_stop: str | None = None
        # odometry's x (mm) where the rear met the face this approach, and x after each of its sits
        self.face_x: float | None = None
        self.sit_ends: list[float] = []
        self.backed = False  # it has begun backing in on this try (the dock's contacts expected)
        self.back_steps = 0  # short steps back this try (walked into something on the J-hook)
        self.beside_steps = 0  # … and to give the first arc room beside the guides
        self.beside_backing = False  # this step back is the one beside the guides
        self.bumped_ms: int | None = None  # proprio's blocked while walking the J-hook forward
        self.tries = 0  # dockings begun (for the evaluators)
        self.docked = 0
        self.log: list[str] = []
        self._event: str | None = None
        self._t = 0

    # ── what the brain asks ──────────────────────────────────────────────────

    def pose(self, odom: tuple[float, float, float]) -> DockPose | None:
        """Where the duck believes it is in the dock's frame (None: it does not know)."""
        return self.tracker.now(odom)

    def trusted(self) -> bool:
        """The dock's frame is one it can steer by: a fix or contact this outing, or remembered
        over less than :attr:`DockConfig.remembered_max_m` of walking."""
        return self.tracker.fix is not None and self.walked_mm <= 1000.0 * self.cfg.remembered_max_m

    def in_dock(self, odom: tuple[float, float, float]) -> bool:
        """It believes it is between the dock's guides (the way out is straight ahead)."""
        pose = self.pose(odom) if self.trusted() else None
        if pose is None:
            return False
        geo = self.cfg.geometry
        return pose.x < geo.guides_mm + 60.0 and abs(pose.y) < geo.half_width_mm

    def rendezvous(
        self, odom: tuple[float, float, float], *, stale_ok: bool = False
    ) -> tuple[float, float] | None:
        """(bearing rad, range m) from the duck to the lane's far point by the remembered frame
        (the brain walks there to get the tag in view from the front), or None. ``stale_ok``: also by
        a frame carried farther than :attr:`DockConfig.remembered_max_m`. It then points the walk (and
        the eyes) at the front of the dock, as the nest's place in the lasting memory does, but the
        legs are never taken by it: that needs a tag read (:meth:`can_engage`)."""
        if not (stale_ok and self.tracker.fix is not None) and not self.trusted():
            return None
        pose = self.pose(odom)
        if pose is None:
            return None
        dx = self.cfg.geometry.far_mm - pose.x
        dy = self.cfg.lane_y() - pose.y
        return wrap(math.atan2(dy, dx) - pose.h), math.hypot(dx, dy) / 1000.0

    def dock_bearing(
        self, odom: tuple[float, float, float], *, stale_ok: bool = False
    ) -> tuple[float, float] | None:
        """(bearing rad, range m) to the tag by the carried frame, or None (``stale_ok``: see
        :meth:`rendezvous`)."""
        if not (stale_ok and self.tracker.fix is not None) and not self.trusted():
            return None
        pose = self.pose(odom)
        if pose is None:
            return None
        face = self.cfg.geometry.face_mm
        return (
            wrap(math.atan2(-pose.y, face - pose.x) - pose.h),
            math.hypot(face - pose.x, pose.y) / 1000.0,
        )

    def head_busy(self) -> bool:
        """The manoeuvre needs the head now: a look at the tag, the U-turn, backing in (or a
        step back)."""
        return self.mode == "docking" and self.phase in (
            "fix",
            "uturn",
            "brief",
            "reverse_in",
            "back_step",
        )

    def expects_contact(self) -> bool:
        """Backing in, sitting in the dock, getting out of it: its body meets the dock's face and
        guides — its own doing (the brain's reafference: no bump reflex, no wall marked)."""
        return self.mode in ("home", "leaving") or (self.mode == "docking" and self.backed)

    def can_engage(self, din: DockIn) -> str | None:
        """Why the program could take the legs now (``tag``, ``remembered``) or None: a trusted
        tag reading this near, or the remembered lane point reached."""
        read = din.tag
        if read is not None and read.facing is not None and read.range_m <= self.cfg.engage_m:
            return "tag"
        rv = self.rendezvous(din.odom)
        if rv is not None and rv[1] * 1000.0 <= self.cfg.rendezvous_mm:
            return "remembered"
        return None

    def engage(self, din: DockIn, why: str = "tag") -> None:
        """Take the legs: stand, fix, the J-hook, the reverse in, the sit."""
        self._reset_try()
        self.backed = False
        self.mode = "docking"
        self.tries += 1
        self.result = None
        if din.tag is not None:
            self._take(din.tag, din)  # the reading that made it engage is a first fix
        self.log.append(f"engage:{why}")
        self.plan = ["stand", *self._approach_plan()]
        pose = self.pose(din.odom) if self.trusted() else None
        if pose is not None and self.in_dock(din.odom):
            if abs(wrap(pose.h)) >= self.cfg.out_face_rad:
                # (headless 2026-09-28) standing in its dock turned sideways (its stand-ups
                # there swing it round): no arc can leave between the guides — it stays in
                # its nest (the re-seat stands and sits it again, in place)
                self._end("wedged", fail=True)
                self.mode = "home"
                self.phase = "home"
                self.open_since = din.t_ms
                return
            self.plan = ["stand", "out_first", *self._approach_plan()]  # straight out first
        self._start_phase(self.plan.pop(0), din.t_ms)

    def reseat(self, din: DockIn) -> None:
        """In the dock without contact: stand and sit again (dockexec's retry)."""
        self._reset_try()
        self.backed = True
        self.mode = "docking"
        self.tries += 1
        self.log.append("reseat")
        self.plan = ["sit", "pg"]
        self._start_phase("stand", din.t_ms)

    def leave(self, din: DockIn) -> None:
        """It wants to walk off: up, and straight out along the dock's axis."""
        self._reset_try()
        self.mode = "leaving"
        self.log.append("leave")
        self.plan = ["leave"]
        self._start_phase("stand", din.t_ms)

    def abort(self, why: str) -> None:
        """The brain dropped the act (or the body said so): legs back to the brain."""
        if self.mode in ("docking", "leaving"):
            self.log.append(f"abort:{why}")
            self._end(why, fail=self.mode == "docking")

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, din: DockIn) -> DockOut:
        now = din.t_ms
        self._t = now
        self._event = None
        self._carry(din)
        if din.upset:
            # picked up or set down: the odometry frame no longer holds
            self.tracker = _Tracker()
            self.fix_ms = None
            self.fix_src = ""
            if self.mode in ("docking", "leaving"):
                self._end("upset", fail=self.mode == "docking")
            self.mode = "idle"
            self.phase = "idle"
        elif din.fell and self.mode in ("docking", "leaving"):
            self._end("fell", fail=self.mode == "docking")  # (the next try looks again first)
        self._moving(din, now)
        if din.blocked and self.mode == "docking" and self.phase in BUMP_PHASES:
            self.bumped_ms = now  # (a reflex may hold it first: the step back comes after)
        read = din.tag
        if (
            self.mode == "idle"
            and read is not None
            and read.facing is not None
            and read.range_m <= self.cfg.remember_m
        ):
            self._take(read, din)  # it saw its dock: where it is stays remembered (odometry)
        if din.pg and din.posture == "sit":
            self._contact(din)
            if self.mode == "idle":
                # it finds itself seated on its contacts (set down there, or it started there)
                self.mode = "home"
                self.phase = "home"
                self.home_since = now
                self.open_since = None
                self.log.append("home")
        if self.mode == "docking" and din.pg and din.posture == "sit" and self.phase != "pg":
            # PG decides: seated on the contacts is docked, whatever step it was at (the sit's
            # slide back can close it a moment after the check, or during a re-seat)
            return self._docked(din, now)
        if self.mode == "home":
            return self._home(din, now)
        if self.mode not in ("docking", "leaving"):
            return DockOut(phase="idle", event=self._pop_event())
        # a reflex holds the body: hold still, the legs start over after it
        if din.halted:
            if self.halt_since is None:
                self.halt_since = now
            if self.leg is not None:
                self.leg.settled = False
                self.leg.restart = True
            if now - self.halt_since > self.cfg.halt_max_ms:
                self._end("halt", fail=True)
                return self._idle_out()
            seated = self.phase in ("sit", "pg")
            return self._out(posture="sit" if seated else "stand", quiet=False)
        self.halt_since = None
        handler: Callable[[DockIn, int], DockOut] = getattr(self, f"_phase_{self.phase}")
        out = handler(din, now)
        return replace(out, event=out.event or self._pop_event())

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def _reset_try(self) -> None:
        self.sits = 0
        self.approaches = 1
        self.relines = 0
        self.lane_loops = 0
        self.misses = 0
        self.leg = None
        self.sub = None
        self.look_t0 = None
        self.last_h = None
        self.turned = 0.0
        self.sat_ms = None
        self.halt_since = None
        self.held_since = None
        self.unsure_since = None
        self.face_stop = None
        self.face_x = None
        self.sit_ends = []
        self.back_steps = 0
        self.beside_steps = 0
        self.beside_backing = False
        self.bumped_ms = None

    def _pop_event(self) -> str | None:
        event, self._event = self._event, None
        return event

    def _carry(self, din: DockIn) -> None:
        prev = self._odom_prev
        if prev is not None:
            self.walked_mm += math.hypot(din.odom[0] - prev[0], din.odom[1] - prev[1])
        self._odom_prev = din.odom

    def _moving(self, din: DockIn, now: int) -> None:
        if abs(din.applied[0]) > 0.05 or abs(din.applied[1]) > 0.15:
            self.applied_ms = now

    def _last_motion(self) -> int | None:
        times = [t for t in (self.cmd_ms, self.applied_ms) if t is not None]
        return max(times) if times else None

    def _legs_quiet(self, now: int) -> bool:
        """Rule 1 of the looks: no move commanded or applied for the legs' tail (0.45 s)."""
        last = self._last_motion()
        return last is None or now - last >= self.cfg.tail_ms

    def _quiet_since(self) -> int:
        last = self._last_motion()
        return -(10**9) if last is None else last + self.cfg.tail_ms

    def _contact(self, din: DockIn) -> None:
        """Seated on PG: it is in its dock. Without a good carried pose (no fix this outing, or
        a fix gone stale), the dock's frame comes from the contact itself — the seat, facing the
        room (a backed-in duck faces the room; the seat's own turn is within ~±20°)."""
        pose = self.pose(din.odom)
        geo = self.cfg.geometry
        good = (
            pose is not None
            and self.fix_src in ("tag", "contact")
            and abs(pose.x - geo.seat_mm) < 150.0
            and abs(pose.y) < 120.0
        )
        if not good:
            self.tracker.set(DockPose(float(geo.seat_mm), 0.0, 0.0), din.odom)
            self.fix_src = "contact"
            self.fix_ms = din.t_ms
        self.walked_mm = 0.0

    def _start_phase(self, phase: str, now: int) -> None:
        self.phase = phase
        self.phase_ms = now
        self.leg = None
        self.sub = None
        self.look_t0 = None
        self.sat_ms = None
        self.head_home_since = None
        self.log.append(phase)
        if len(self.log) > 80:
            del self.log[:20]

    def _next(self, din: DockIn, now: int) -> DockOut:
        if not self.plan:
            self._end("plan", fail=self.mode == "docking")
            return self._idle_out()
        self._start_phase(self.plan.pop(0), now)
        handler: Callable[[DockIn, int], DockOut] = getattr(self, f"_phase_{self.phase}")
        return handler(din, now)

    def _end(self, why: str, *, fail: bool) -> None:
        was = self.mode
        self.mode = "idle"
        self.phase = "idle"
        self.plan = []
        self.leg = None
        if was == "docking":
            self.result = f"failed:{why}" if fail else why
            self._event = "failed" if fail else self._event
        elif was == "leaving":
            self.result = "left" if not fail else f"leave:{why}"
            self._event = "left"

    def _idle_out(self) -> DockOut:
        return DockOut(phase="idle", event=self._pop_event())

    def _out(
        self,
        *,
        drive: bool = True,
        vx: float = 0.0,
        vyaw: float = 0.0,
        posture: str | None = "stand",
        look: tuple[float, float] | None = None,
        head_still: bool = False,
        quiet: bool = False,
    ) -> DockOut:
        if vx != 0.0 or vyaw != 0.0:
            self.cmd_ms = self._t
        return DockOut(
            drive=drive,
            vx=vx,
            vyaw=vyaw,
            reversing=vx < 0.0,
            posture=posture,
            look=look,
            head_still=head_still,
            quiet=quiet,
            phase=self.phase,
            zh=self._phase_zh(),
        )

    def _phase_zh(self) -> str:
        """The plain words of the step it is doing now."""
        if self.phase == "back_step" and self.beside_backing:
            return PHASE_ZH["beside"]
        return PHASE_ZH.get(self.phase, self.phase)

    def _take(self, read: TagRead, din: DockIn) -> bool:
        """A tag reading → the tracker (the carried heading when the facing is unsure)."""
        carried = self.tracker.now(read.odom)
        pose = pose_from_read(
            read, self.cfg.geometry, carried.h if carried is not None and self.trusted() else None
        )
        if pose is None:
            return False
        self.tracker.set(pose, read.odom)
        self.fix_ms = din.t_ms
        self.fix_src = "tag"
        self.walked_mm = 0.0
        return True

    # ── the plans ────────────────────────────────────────────────────────────

    def _approach_plan(self) -> list[str]:
        """dock:measure steps 1–6: to the lane, in along it, the U-turn, the reverse in with its
        face stop, the sit."""
        return [
            "fix",
            "to_lane",
            "fix",
            "to_turn",
            "pause",
            "fix",
            "uturn",
            "brief",
            "reverse_in",
            "bump_out",
            "pause",
            "sit",
            "pg",
        ]

    # ── posture ──────────────────────────────────────────────────────────────

    def _phase_stand(self, din: DockIn, now: int) -> DockOut:
        if din.posture in ("stand", "walk"):
            return self._next(din, now)
        if now - self.phase_ms > self.cfg.stand_timeout_ms:
            self._end("stand", fail=self.mode == "docking")
            return self._idle_out()
        return self._out(posture="stand", quiet=True)

    def _phase_sit(self, din: DockIn, now: int) -> DockOut:
        if din.posture == "sit":
            if self.sat_ms is None:
                self.sat_ms = now
            if now - self.sat_ms >= self.cfg.pg_settle_ms:
                return self._next(din, now)
            return self._out(posture="sit", quiet=False)
        self.sat_ms = None
        if now - self.phase_ms > self.cfg.sit_timeout_ms:
            self._end("sit", fail=True)
            return self._idle_out()
        return self._out(posture="sit", quiet=True)

    def _docked(self, din: DockIn, now: int) -> DockOut:
        self.mode = "home"
        self.phase = "home"
        self.phase_ms = now
        self.home_since = now
        self.open_since = None
        self.result = "docked"
        self.docked += 1
        self._event = "docked"
        self.log.append("docked")
        self._contact(din)
        return self._home(din, now)

    def _phase_pg(self, din: DockIn, now: int) -> DockOut:
        self.sits += 1
        if din.pg:
            return self._docked(din, now)
        pose = self.pose(din.odom)
        if pose is not None and self.face_x is not None:
            self.sit_ends.append(pose.x)
        if not self._another_sit():
            self._end("tries", fail=True)
            return self._idle_out()
        self.log.append("resit")
        self._event = "resit"
        self.plan = ["sit", "pg"]  # stand, sit again — no walking (dockexec step 6)
        self._start_phase("stand", now)
        return self._phase_stand(din, now)

    def _another_sit(self) -> bool:
        """The sits so far did not close PG. :attr:`DockConfig.max_sits` is the plain budget; past it a
        sit-and-stand that still took the rear in (by odometry, ``sit_progress_mm`` further in than
        after the one before) while it is believed out of reach (``seat_reach_mm`` from the face) earns
        one more, up to ``max_sits_progress``. A rear in reach with PG open, or one that no longer
        moves, is not a matter of distance: the try ends as it always did. A re-seat from the dock
        (no face stop this try) has no reference and keeps the plain budget."""
        cfg = self.cfg
        if self.sits < cfg.max_sits:
            return True
        if self.sits >= cfg.max_sits_progress or self.face_x is None or len(self.sit_ends) < 2:
            return False
        last, before = self.sit_ends[-1], self.sit_ends[-2]
        if before - last < cfg.sit_progress_mm or last - self.face_x <= cfg.seat_reach_mm:
            return False
        self.log.append("more_sits")
        return True

    def _phase_pause(self, din: DockIn, now: int) -> DockOut:
        return self._hold(din, now, self.cfg.pause_ms)

    def _phase_brief(self, din: DockIn, now: int) -> DockOut:
        return self._hold(din, now, self.cfg.brief_ms, still=True)

    def _phase_tail(self, din: DockIn, now: int) -> DockOut:
        return self._hold(din, now, self.cfg.tail_ms)

    def _hold(self, din: DockIn, now: int, ms: int, *, still: bool = False) -> DockOut:
        if now - self.phase_ms >= ms:
            return self._next(din, now)
        return self._out(head_still=still, quiet=still)

    # ── home: seated on the contacts, charging ───────────────────────────────

    def _home(self, din: DockIn, now: int) -> DockOut:
        """In the dock: the brain has the body (rest, nap, look about); PG open while seated a
        while → the brain may ask for :meth:`reseat`. Leaving it (walking off by any means)
        ends home."""
        pose = self.pose(din.odom)
        if pose is not None and pose.x > self.cfg.geometry.guides_mm + 150.0:
            self.mode = "idle"  # it is out (the brain's own walk, a hand): home is over
            self.phase = "idle"
            return self._idle_out()
        if din.pg:
            self.open_since = None
        elif self.open_since is None:
            self.open_since = now
        return DockOut(phase="home", zh=PHASE_ZH["home"], event=self._pop_event())

    def contact_lost(self, now: int) -> bool:
        """Home, seated, and PG open for :attr:`DockConfig.reseat_after_ms`."""
        return (
            self.mode == "home"
            and self.open_since is not None
            and now - self.open_since >= self.cfg.reseat_after_ms
        )

    # ── looking at the tag ───────────────────────────────────────────────────

    def _aim(self, din: DockIn) -> float:
        cfg = self.cfg
        pose = self.pose(din.odom)
        base = 0.0
        if pose is not None:
            base = wrap(math.atan2(-pose.y, cfg.geometry.face_mm - pose.x) - pose.h)
        if self.misses:  # search about the believed bearing
            base += (0.45, -0.45, 0.0)[(self.misses - 1) % 3]
        return max(-cfg.look_max_rad, min(cfg.look_max_rad, base))

    def _phase_fix(self, din: DockIn, now: int) -> DockOut:
        cfg = self.cfg
        pose = self.pose(din.odom)
        if pose is not None and self.trusted() and self.misses == 0:
            bearing = wrap(math.atan2(-pose.y, cfg.geometry.face_mm - pose.x) - pose.h)
            if abs(bearing) > cfg.look_max_rad + 0.35:
                # the dock is behind it (its head cannot turn so far): go on by the carried pose
                self.log.append("fix_behind")
                return self._next(din, now)
        if self.look_t0 is None:
            self.look_t0 = now
            self.look_aim = self._aim(din)
            self.looks += 1
        look = (self.look_aim, cfg.look_elevation)
        quiet = self._legs_quiet(now)
        if not quiet:
            self.look_t0 = now  # the look's clock starts when the legs are quiet
            return self._out(look=look, quiet=True)
        look_t0 = self.look_t0 if self.look_t0 is not None else now
        since = max(self._quiet_since(), look_t0)
        read = din.tag
        if read is not None and read.t_ms - 150 >= since and self._take(read, din):
            self.misses = 0
            return self._next(din, now)
        if din.cue is not None and self.tracker.fix is not None and self.trusted():
            # a direction only (the light, the panel: owner C5): keep the carried position,
            # turn the heading toward the dock
            carried = self.pose(din.odom)
            if carried is not None:
                want = math.atan2(-carried.y, cfg.geometry.face_mm - carried.x)
                heading = wrap(want - din.cue)
                self.tracker.set(DockPose(carried.x, carried.y, heading), din.odom)
                self.misses = 0
                self.log.append("cue")
                return self._next(din, now)
        if now - look_t0 < cfg.fix_look_ms:
            return self._out(look=look, quiet=True)
        # no tag: close enough to the goal on a recent fix, go on blind (the last stretch is
        # odometry anyway); else look again, then give up
        carried = self.pose(din.odom)
        upcoming = self.plan[0] if self.plan else ""
        goal = self._goal_x(upcoming)
        recent = self.fix_ms is not None and now - self.fix_ms <= cfg.blind_fix_age_ms
        if carried is not None and goal is not None and recent:
            if abs(carried.x - goal) <= cfg.max_blind_mm:
                self.misses = 0
                self.log.append("blind")
                return self._next(din, now)
        if carried is not None and upcoming == "to_lane" and self.trusted():
            # (live check 2026-09-28) the first look of an approach, the tag not read (far, at a
            # slant, behind a guide): the walk to the lane's far point goes on the frame it has —
            # the look from there, square on at 1.5 m, puts it right (four misses here ended the
            # try and the brain walked it beside the dock)
            self.misses = 0
            self.log.append("blind")
            return self._next(din, now)
        self.misses += 1
        self.look_t0 = None
        if self.misses >= cfg.fix_misses:
            self._end("no_tag", fail=True)
            return self._idle_out()
        return self._out(look=look, quiet=True)

    # ── driving legs ─────────────────────────────────────────────────────────

    def _held(self, din: DockIn, now: int, leg: _Leg | None) -> DockOut | None:
        """The body link holds the legs (robotd's applied twist stays ~0 while the program
        commands a move: legs light on the plate, the posture, a stop): hold still until the body
        lets go — the leg starts over — or give the try up after ``unsure_wait_ms``."""
        commanded = self.cmd_ms is not None and now - self.cmd_ms <= 60
        moving = abs(din.applied[0]) > 0.05 or abs(din.applied[1]) > 0.15
        if not commanded or moving or (leg is not None and not leg.settled):
            self.held_since = None
            return None
        if self.held_since is None:
            self.held_since = now
            return None
        if now - self.held_since < self.cfg.held_ms:
            return None
        if now - self.held_since > self.cfg.unsure_wait_ms + self.cfg.held_ms:
            self._end("held", fail=self.mode == "docking")
            return self._idle_out()
        if leg is not None:
            leg.restart = True
        if self.log[-1:] != ["held"]:
            self.log.append("held")
        return None  # keep commanding: the body link decides when the legs may go

    def _settled(self, din: DockIn, now: int, leg: _Leg, *, home: bool) -> bool:
        """``home``: the head at HOME and settled (a U-turn, a reverse: as phase 3); else at once
        (the neural duck walks forward with a live head inside the measured walking envelope)."""
        if leg.settled:
            return True
        if home:
            if not din.head_home:
                self.head_home_since = None
                leg.started_ms = leg.last_ms = now
                return False
            if self.head_home_since is None:
                self.head_home_since = now
            if now - self.head_home_since < self.cfg.head_settle_ms:
                leg.started_ms = leg.last_ms = now
                return False
        leg.settled = True
        leg.started_ms = leg.last_ms = now
        return True

    def _drive(
        self, din: DockIn, now: int, leg: _Leg, arrived: Callable[[DockPose], bool]
    ) -> DockOut:
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        if self.bumped_ms is not None and self.phase in BUMP_PHASES:
            return self._bumped(din, now)
        if not self._settled(din, now, leg, home=leg.reverse):
            return self._out(head_still=leg.reverse, quiet=leg.reverse)
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if leg.restart:
            leg.restart = False
            leg.started_ms = now
        if arrived(pose):
            return self._next(din, now)
        if now - leg.started_ms >= leg.ms:
            return self._segment_end(din, now)
        return self._steer(now, leg, pose)

    def _bumped(self, din: DockIn, now: int) -> DockOut:
        """Walked into something on the J-hook: a short step back, then this walk planned
        again from where it is — at most :attr:`DockConfig.max_back_steps` times a try."""
        self.bumped_ms = None
        self.back_steps += 1
        self.log.append("bump")
        if self.back_steps > self.cfg.max_back_steps:
            self._end("blocked", fail=True)
            return self._idle_out()
        self.plan[:0] = ["back_step", self.phase]
        return self._next(din, now)

    def _phase_back_step(self, din: DockIn, now: int) -> DockOut:
        """One short straight step back (the head at HOME, as every reverse): vx −0.40 for
        :attr:`DockConfig.back_step_ms` — beside the guides a back arc instead, the nose swinging
        out to the room (:attr:`DockConfig.beside_face_rad`)."""
        cfg = self.cfg
        beside = self.beside_backing
        if self.leg is None:
            ms = cfg.beside_back_ms if beside else cfg.back_step_ms
            self._start_leg(now, 0.0, reverse=True, ms=ms)
        leg = self.leg
        assert leg is not None
        if not self._settled(din, now, leg, home=True):
            return self._out(head_still=True, quiet=True)
        if leg.restart:
            leg.restart = False
            leg.started_ms = now
        pose = self.pose(din.odom)
        out_facing = beside and pose is not None and abs(wrap(pose.h)) <= cfg.beside_face_rad
        if out_facing or now - leg.started_ms >= leg.ms:
            self.bumped_ms = None  # (its own step back is no new bump)
            self.beside_backing = False
            return self._next(din, now)
        if beside and pose is not None:
            # the nose swings away from the axis, out toward the room: left beside the left
            # guide (y > 0), right beside the right one
            swing = math.copysign(cfg.arc_vyaw, pose.y)
            return self._out(vx=cfg.reverse_vx, vyaw=swing, head_still=True, quiet=True)
        return self._out(vx=cfg.reverse_vx, vyaw=cfg.ff_reverse, head_still=True, quiet=True)

    def _steer(self, now: int, leg: _Leg, pose: DockPose) -> DockOut:
        """The leg's twist: a carrot on the lane further along the direction of travel (more
        when far off the lane, 1.5 × the offset), PI on the heading error plus the gait's drift
        feed-forward (dockexec's ``_steer``)."""
        cfg = self.cfg
        dt = max(0.0, (now - leg.last_ms) / 1000.0)
        leg.last_ms = now
        toward_wall = leg.goal_x < pose.x
        reach = max(cfg.carrot_mm, 1.5 * abs(pose.y - leg.lane_y))
        carrot_x = pose.x - reach if toward_wall else pose.x + reach
        target = math.atan2(leg.lane_y - pose.y, carrot_x - pose.x)
        moving = wrap(pose.h + math.pi) if leg.reverse else pose.h
        err = wrap(target - moving)
        leg.integral = max(-0.3, min(0.3, leg.integral + err * dt))
        ff = cfg.ff_reverse if leg.reverse else cfg.ff
        vyaw = cfg.kp * err + cfg.ki * leg.integral + ff
        vyaw = max(-1.0, min(1.0, vyaw))
        vx = cfg.reverse_vx if leg.reverse else cfg.vx
        return self._out(vx=vx, vyaw=vyaw, head_still=leg.reverse, quiet=leg.reverse)

    def _segment_end(self, din: DockIn, now: int) -> DockOut:
        """A segment ran its time: stop, let the tail go, look again, carry on — except for the
        last stretch, which goes on odometry in one leg."""
        current = self.phase
        pose = self.pose(din.odom)
        leg = self.leg
        if (
            pose is not None
            and leg is not None
            and abs(pose.x - leg.goal_x) <= self.cfg.final_blind_mm
        ):
            leg.started_ms = now
            leg.ms = max(leg.ms, 4000)
            return self._drive(din, now, leg, self._arrival(current))
        self.plan[:0] = ["tail", "fix", current]
        return self._next(din, now)

    def _goal_x(self, phase: str) -> float | None:
        geo = self.cfg.geometry
        return {
            "reverse_in": float(geo.stop_mm),
            "to_turn": float(geo.turn_mm),
            "uturn": float(geo.turn_mm),  # the last fix before the U-turn: a recent one will do
        }.get(phase)

    def _arrival(self, phase: str) -> Callable[[DockPose], bool]:
        geo = self.cfg.geometry
        if phase == "to_turn":
            return lambda p: p.x <= geo.turn_mm
        return lambda p: p.x >= geo.turn_mm

    def _start_leg(
        self, now: int, goal_x: float, reverse: bool, ms: int, lane_y: float = 0.0
    ) -> _Leg:
        leg = _Leg(
            goal_x=goal_x, reverse=reverse, ms=ms, lane_y=lane_y, started_ms=now, last_ms=now
        )
        self.leg = leg
        self.head_home_since = None
        self.held_since = None
        return leg

    # ── to the lane, along it, the U-turn ────────────────────────────────────

    def _on_lane(self, pose: DockPose) -> bool:
        """Already on the pre-turn lane, in front of the turn point and facing the dock."""
        ahead = pose.x - self.cfg.geometry.turn_mm
        return (
            ahead >= 250.0
            and abs(pose.y - self.cfg.lane_y()) <= 0.6 * ahead
            and abs(wrap(pose.h - math.pi)) <= 1.05
        )

    def _phase_to_lane(self, din: DockIn, now: int) -> DockOut:
        """Step 1: unless on the lane already, walk to its far point (tight arcs to turn), then
        arc round to face the dock within 20° — on odometry from the last fix; at most twice."""
        cfg, geo = self.cfg, self.cfg.geometry
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        self.backed = False  # (walking the J-hook, away from the dock)
        if self.bumped_ms is not None:
            return self._bumped(din, now)
        if now - self.phase_ms > cfg.lane_total_ms:
            # (a reflex holding it now and then restarts the legs' own clocks, never this one)
            self._end("lane", fail=True)
            return self._idle_out()
        if self.sub is None:
            if self._on_lane(pose):
                if self.lane_loops == 0 and self.plan[:1] == ["fix"]:
                    self.plan.pop(0)  # it has just looked: follow the lane in
                return self._next(din, now)
            if self._in_mouth(pose) and "out_first" not in self.log[-12:]:
                # inside the guides facing out: straight out first (no arcs inside them)
                self.plan[:0] = ["out_first", "to_lane"]
                return self._next(din, now)
            if self._beside_guides(pose) and self.beside_steps < self.cfg.max_back_steps:
                # (headless 2026-09-28) beside the dock, facing across its guides: any forward
                # arc sweeps the body into a guide's side (it tripped over them twice) — a step
                # back first gives the arc its room
                self.beside_steps += 1
                self.beside_backing = True
                self.log.append("beside")
                self.plan[:0] = ["back_step", "to_lane"]
                return self._next(din, now)
            self.lane_loops += 1
            self.sub = "goto"
            self._start_leg(now, float(geo.far_mm), reverse=False, ms=cfg.goto_timeout_ms)
        leg = self.leg
        assert leg is not None
        if not self._settled(din, now, leg, home=False):
            return self._out()
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if leg.restart:
            leg.restart = False
            leg.started_ms = now  # (the sub-step's own clock: from when the legs may go again)
        if self.sub == "goto":
            lane = cfg.lane_y()
            d = math.hypot(geo.far_mm - pose.x, lane - pose.y)
            off = wrap(math.atan2(lane - pose.y, geo.far_mm - pose.x) - pose.h)
            passing = d < 2.0 * cfg.arc_radius_mm and abs(off) > math.pi / 2.0
            if d < cfg.lane_arrive_mm or passing:
                self.sub = "face"
                leg.started_ms = now
            elif now - leg.started_ms > cfg.goto_timeout_ms:
                self._end("lane", fail=True)
                return self._idle_out()
            else:
                return self._goto_twist(pose, float(geo.far_mm), lane)
        err = wrap(math.pi - pose.h)
        if abs(err) < cfg.face_tol:
            if self._on_lane(pose) or self.lane_loops >= 2:
                return self._next(din, now)  # facing the dock on the lane: look, then in
            self.sub = None  # the arc carried it off the lane or toward the dock: again
            return self._out()
        if now - leg.started_ms > cfg.face_timeout_ms:
            self._end("face", fail=True)
            return self._idle_out()
        return self._out(vx=cfg.arc_vx, vyaw=math.copysign(cfg.arc_vyaw, err))

    def _in_mouth(self, pose: DockPose) -> bool:
        """Between the guides (or just at their mouth) and facing out to the room."""
        geo = self.cfg.geometry
        return (
            pose.x < geo.guides_mm + 250.0
            and abs(pose.y) < geo.half_width_mm + 60.0
            and abs(wrap(pose.h)) < self.cfg.out_face_rad
        )

    def _phase_out_first(self, din: DockIn, now: int) -> DockOut:
        """Straight out along the axis, facing the room, to :attr:`DockConfig.out_first_mm`
        (then the walk to the lane goes on)."""
        cfg = self.cfg
        if self.leg is None:
            self._start_leg(now, cfg.out_first_mm, reverse=False, ms=cfg.out_first_ms)
        leg = self.leg
        assert leg is not None
        if leg.settled and now - leg.started_ms >= leg.ms:
            return self._next(din, now)  # (as far as it got: the lane walk plans from there)
        return self._drive_out(din, now, leg, lambda p: p.x >= cfg.out_first_mm)

    def _beside_guides(self, pose: DockPose) -> bool:
        """Just outside a guide (within ~0.45 m of its side, not past its mouth) and heading
        across toward the dock's axis: a forward arc would sweep the body into it."""
        geo = self.cfg.geometry
        side = 1.0 if pose.y >= 0.0 else -1.0
        across = -side * math.sin(pose.h)  # > 0: the nose points toward the axis
        return (
            pose.x < geo.guides_mm + 200.0
            and geo.half_width_mm - 50.0 < abs(pose.y) < geo.half_width_mm + 450.0
            and across > 0.3
        )

    def _goto_twist(self, pose: DockPose, px: float, py: float) -> DockOut:
        """Toward a point: bearing steering, a tight arc while the bearing is more than
        :attr:`DockConfig.arc_rad` off — unless the point is inside that arc's circle (it would
        circle it for ever): then straight on until it is not."""
        cfg = self.cfg
        err = wrap(math.atan2(py - pose.y, px - pose.x) - pose.h)
        if abs(err) > cfg.arc_rad:
            side = math.copysign(1.0, err)
            r = cfg.arc_radius_mm
            cx = pose.x - side * r * math.sin(pose.h)
            cy = pose.y + side * r * math.cos(pose.h)
            if math.hypot(px - cx, py - cy) < r * 1.15:
                return self._out(vx=cfg.vx, vyaw=cfg.ff)
            return self._out(vx=cfg.arc_vx, vyaw=side * cfg.arc_vyaw)
        vyaw = cfg.goto_kp * err + cfg.ff
        return self._out(vx=cfg.vx, vyaw=max(-1.0, min(1.0, vyaw)))

    def _phase_to_turn(self, din: DockIn, now: int) -> DockOut:
        """Follow the lane toward the dock to where the U-turn starts, one look a second."""
        geo = self.cfg.geometry
        if self.leg is None:
            self._start_leg(
                now, float(geo.turn_mm), reverse=False, ms=self.cfg.lane_step_ms,
                lane_y=self.cfg.lane_y(),
            )  # fmt: skip
        assert self.leg is not None
        return self._drive(din, now, self.leg, lambda p: p.x <= geo.turn_mm)

    def _phase_uturn(self, din: DockIn, now: int) -> DockOut:
        """A U-turn from standing toward the axis (the side's way round): vx 0.30, vyaw ∓1.0
        until the heading has turned 178° by odometry (the IMU's yaw). The head at HOME."""
        cfg = self.cfg
        if self.last_h is None:
            if not din.head_home:
                self.head_home_since = None
                self.phase_ms = now
                return self._out(head_still=True, quiet=True)
            if self.head_home_since is None:
                self.head_home_since = now
            if now - self.head_home_since < cfg.head_settle_ms:
                self.phase_ms = now
                return self._out(head_still=True, quiet=True)
            self.last_h, self.turned = din.odom[2], 0.0
        self.turned += wrap(din.odom[2] - self.last_h)
        self.last_h = din.odom[2]
        if abs(self.turned) >= cfg.uturn_turn:
            self.last_h = None
            return self._next(din, now)
        if now - self.phase_ms > cfg.uturn_timeout_ms:
            self._end("uturn", fail=True)
            return self._idle_out()
        return self._out(
            vx=cfg.uturn_vx, vyaw=cfg.turn_sign() * cfg.uturn_vyaw, head_still=True, quiet=True
        )

    # ── the reverse in and the face stop ─────────────────────────────────────

    def _phase_reverse_in(self, din: DockIn, now: int) -> DockOut:
        """Steps 3–5: reverse in along the axis — straight first, then steered, one kick if it
        will not start — with the alignment gate, until the face stops it (blocked = arrived)."""
        cfg, geo = self.cfg, self.cfg.geometry
        if self.leg is None:
            self._start_leg(now, float(geo.face_contact_mm), reverse=True, ms=cfg.reverse_max_ms)
            self.backed = True
        leg = self.leg
        assert leg is not None
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        if not self._settled(din, now, leg, home=True):
            return self._out(head_still=True, quiet=True)
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if leg.moved_from is None or leg.restart:
            # (re)start: the straight start, the kick and the progress window count from here
            leg.moved_from, leg.moving_ms, leg.restart = pose, now, False
            leg.samples.clear()
            leg.kick_until = None
            leg.first_ms = now if leg.first_ms is None else leg.first_ms
        leg.samples.append((now, pose.x, pose.y, pose.h))
        moving_ms = leg.moving_ms if leg.moving_ms is not None else now
        moved = math.hypot(pose.x - leg.moved_from.x, pose.y - leg.moved_from.y)
        near_face = pose.x <= geo.face_contact_mm + cfg.face_zone_mm
        if din.blocked and moved >= cfg.blocked_min_mm:
            if near_face:
                return self._at_face(din, now, "stall")
            if self.log[-1:] != ["stall_far"]:
                self.log.append("stall_far")  # (held mid-room: not the face — on it goes)
        if pose.x <= geo.face_contact_mm:
            return self._at_face(din, now, "pose")  # odometry puts the rear at the face
        reversing_now = din.applied[0] < -0.05
        if near_face and moved >= cfg.blocked_min_mm and now - moving_ms >= cfg.blocked_window_ms:
            old = [s for s in leg.samples if s[0] <= now - cfg.blocked_window_ms]
            if old and reversing_now:
                _, ox, oy, _ = old[-1]
                if math.hypot(pose.x - ox, pose.y - oy) < cfg.blocked_progress_mm:
                    return self._at_face(din, now, "blocked")
        if pose.x < cfg.gate_x_mm and self._misaligned(leg, now):
            return self._realign(din, now)
        first = leg.first_ms if leg.first_ms is not None else moving_ms
        if now - first > leg.ms:
            self._end("reverse", fail=True)
            return self._idle_out()
        if now - moving_ms < cfg.reverse_straight_ms:
            leg.last_ms = now
            return self._out(vx=cfg.reverse_vx, vyaw=cfg.ff_reverse, head_still=True, quiet=True)
        if leg.kick_until is not None and now < leg.kick_until:
            leg.last_ms = now
            return self._out(vx=cfg.reverse_vx, vyaw=0.0, head_still=True, quiet=True)
        if not leg.kicked and now - moving_ms >= cfg.kick_after_ms and moved < cfg.kick_min_mm:
            # velstand can stand still on a steered reverse start: one straight reverse
            leg.kicked = True
            leg.kick_until = now + cfg.kick_ms
            self.log.append("reverse_kick")
            leg.last_ms = now
            return self._out(vx=cfg.reverse_vx, vyaw=0.0, head_still=True, quiet=True)
        return self._steer(now, leg, pose)

    def _misaligned(self, leg: _Leg, now: int) -> bool:
        """The alignment gate on the pose averaged over one gait cycle."""
        cfg = self.cfg
        window = [s for s in leg.samples if s[0] >= now - cfg.gate_window_ms]
        if not window:
            return False
        y = sum(s[2] for s in window) / len(window)
        h = math.atan2(sum(math.sin(s[3]) for s in window), sum(math.cos(s[3]) for s in window))
        return abs(y) > cfg.gate_mm or abs(h) > cfg.gate_rad

    def _realign(self, din: DockIn, now: int) -> DockOut:
        """The gate failed: walk out along the axis and reverse again (at most twice); once more,
        the whole approach again."""
        self.relines += 1
        self.log.append("gate")
        if self.relines <= self.cfg.max_relines:
            self.plan[:0] = ["pause", "realign", "pause", "reverse_in"]
            return self._next(din, now)
        if self.approaches >= self.cfg.max_approaches:
            self._end("gate", fail=True)
            return self._idle_out()
        self.approaches += 1
        self.relines = 0
        self.lane_loops = 0
        self.sits = 0
        self.plan = ["pause", "walk_out", "pause", "uturn", "brief", *self._approach_plan()]
        return self._next(din, now)

    def _at_face(self, din: DockIn, now: int, why: str) -> DockOut:
        """Blocked = arrived: the rear is at the face (the next phase walks out a little)."""
        self.face_stop = why
        pose = self.pose(din.odom)
        self.face_x = None if pose is None else pose.x
        self.sit_ends = []
        self.log.append(f"face_stop:{why}")
        self._event = "face"
        return self._next(din, now)

    def _phase_bump_out(self, din: DockIn, now: int) -> DockOut:
        """After the face stop: walk out along the axis at 0.35 so the sit's ~70 mm slide back ends on
        the plates, not against the face — until the odometry says it is :attr:`DockConfig.bump_out_mm`
        out from the face stop (the legs coast on a few cm after the stop), or 0.7 s, whichever first."""
        cfg = self.cfg
        if cfg.bump_out_mm <= 0.0:
            return self._next(din, now)
        if self.leg is None:
            self._start_leg(now, float(cfg.geometry.turn_mm), reverse=False, ms=cfg.bump_walk_ms)
        leg = self.leg
        assert leg is not None
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        if not self._settled(din, now, leg, home=False):
            return self._out(quiet=True)
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if leg.restart:
            leg.restart = False
            leg.started_ms = now
        out_mm = 0.0 if self.face_x is None else pose.x - self.face_x
        if out_mm >= cfg.bump_out_mm or now - leg.started_ms >= leg.ms:
            return self._next(din, now)
        out = self._steer(now, leg, pose)
        return replace(out, vx=cfg.bump_walk_vx, quiet=True)

    def _phase_realign(self, din: DockIn, now: int) -> DockOut:
        """The gate's walk-out: facing the room, out along the axis to the realign point,
        straightening (≥ 1000 mm within 30 mm and 5°, or 1150 mm)."""
        cfg, geo = self.cfg, self.cfg.geometry
        if self.leg is None:
            self._start_leg(now, float(geo.turn_mm), reverse=False, ms=cfg.realign_timeout_ms)
        assert self.leg is not None

        def lined(p: DockPose) -> bool:
            straight = abs(p.y) < cfg.realign_tol_mm and abs(wrap(p.h)) < cfg.realign_tol_rad
            return (p.x >= geo.line_up_mm and straight) or p.x >= geo.turn_mm

        leg = self.leg
        if leg.settled and now - leg.started_ms >= leg.ms:
            self._end("realign", fail=True)
            return self._idle_out()
        return self._drive_out(din, now, leg, lined)

    def _phase_walk_out(self, din: DockIn, now: int) -> DockOut:
        """The whole approach again: facing the room, walk out to the turning point, then the
        U-turn back to face the tag."""
        cfg, geo = self.cfg, self.cfg.geometry
        if self.leg is None:
            self._start_leg(now, float(geo.turn_mm), reverse=False, ms=cfg.realign_timeout_ms)
        assert self.leg is not None
        leg = self.leg
        if leg.settled and now - leg.started_ms >= leg.ms:
            self._end("walk_out", fail=True)
            return self._idle_out()
        return self._drive_out(din, now, leg, lambda p: p.x >= geo.turn_mm)

    def _drive_out(
        self, din: DockIn, now: int, leg: _Leg, arrived: Callable[[DockPose], bool]
    ) -> DockOut:
        """A leg out along the axis, facing the room (no looks: the tag is behind)."""
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        if not self._settled(din, now, leg, home=False):
            return self._out()
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if arrived(pose):
            return self._next(din, now)
        return self._steer(now, leg, pose)

    # ── the way out ──────────────────────────────────────────────────────────

    def _phase_leave(self, din: DockIn, now: int) -> DockOut:
        """Up and straight out along the dock's axis to past the guides (no arcs inside them);
        then the legs go back to the brain."""
        cfg = self.cfg
        if self.leg is None:
            self._start_leg(now, cfg.leave_mm, reverse=False, ms=cfg.leave_timeout_ms)
        leg = self.leg
        assert leg is not None
        pose = self.pose(din.odom)
        if pose is None:
            self._end("lost", fail=True)
            return self._idle_out()
        if not self._settled(din, now, leg, home=False):
            return self._out()
        held = self._held(din, now, leg)
        if held is not None:
            return held
        if leg.restart:
            leg.restart = False
            leg.started_ms = now  # (held by the body: its time counts from when it may go)
        if leg.moved_from is None:
            leg.moved_from = pose
        moved = pose.x - leg.moved_from.x
        # its own dock's contacts at the start (the rear on the face as it stands) are no stop;
        # something in the way further out is
        blocked = din.blocked and moved >= 100.0
        if pose.x >= cfg.leave_mm or now - leg.started_ms >= leg.ms or blocked:
            self.log.append("out" if pose.x >= cfg.leave_mm else "out_short")
            self._end("left", fail=False)
            return self._idle_out()
        target = math.atan2(-pose.y, cfg.leave_carrot_mm)
        err = wrap(target - pose.h)
        vyaw = cfg.leave_kp * err + cfg.ff * cfg.leave_vx / 0.35
        return self._out(vx=cfg.leave_vx, vyaw=max(-1.0, min(1.0, vyaw)))

    # ── telemetry ────────────────────────────────────────────────────────────

    def view(
        self, odom: tuple[float, float, float], t_ms: int, pg: bool, charging: bool
    ) -> dict[str, Any]:
        """What the page and the evaluators may show (the duck's own belief, plain words)."""
        pose = self.pose(odom) if self.tracker.fix is not None else None
        zh = (self._phase_zh() if self.phase != "idle" else "") if self.mode != "home" else (
            msg("neuro.dock.home.charging") if charging else (msg("neuro.dock.home.contact") if pg else msg("neuro.dock.home.no_contact"))
        )  # fmt: skip
        why = (self.result or "").split(":", 1)[-1]
        return {
            "version": DOCK_VERSION,
            "mode": self.mode,
            "phase": self.phase,
            "zh": zh,
            "side": self.cfg.turn_side,
            "pg": bool(pg),
            "charging": bool(charging),
            "tries": self.tries,
            "docked": self.docked,
            "sits": self.sits,
            "approaches": self.approaches,
            "face_stop": self.face_stop,
            "result": self.result,
            "result_zh": RESULT_ZH.get(why, "") if (self.result or "").startswith("failed") else "",
            "fix": self.fix_src or None,
            "fix_age_s": None if self.fix_ms is None else round((t_ms - self.fix_ms) / 1000.0, 1),
            "pose": None
            if pose is None
            else [round(pose.x), round(pose.y), round(math.degrees(pose.h), 1)],
            "log": self.log[-12:],
        }


def sighting_read(sight: Sighting, t_ms: int, odom_m: tuple[float, float, float]) -> TagRead | None:
    """A ``nest`` sighting read as a tag (with a range) → a :class:`TagRead` (odometry in mm);
    None for a direction-only cue."""
    if sight.kind != "nest" or sight.range_m is None:
        return None
    facing = getattr(sight, "facing", None)
    return TagRead(
        range_m=float(sight.range_m),
        bearing=float(sight.bearing),
        facing=None if facing is None else float(facing),
        t_ms=int(t_ms),
        odom=(odom_m[0] * 1000.0, odom_m[1] * 1000.0, odom_m[2]),
    )


__all__ = [
    "DOCK_VERSION",
    "MODES",
    "PHASE_ZH",
    "RESULT_ZH",
    "SIDES",
    "DockConfig",
    "DockGeometry",
    "DockIn",
    "DockOut",
    "DockPose",
    "DockRun",
    "TagRead",
    "pose_from_read",
    "sighting_read",
    "wrap",
]
