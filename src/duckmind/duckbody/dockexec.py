"""dockexec: ``duck.dock``, the body's closed loop into the charging dock (physical-design spec §8.5
item 4–5, §10.5; owner decision 2026-09-27 「窩 做成充電座 樣式（類似掃地機器人充電座）」).

The dock stands against a wall (``duckmind.world.dock``, read-only here: duck-body imports nothing
of the world). Two docking **styles**, one choreography each; the default is ``backin`` (owner
2026-09-28 「6. 可以」, the dock workflow's measured recommendation:
``docs/validation/evidence/phase2/dock_styles.json`` → ``recommendation``):

* ``backin`` — dock:measure's J-hook (``dockexec_choreography_backin``), ported step by step:

  0. enter with a fix on the face tag; the turning side is the command's ``side`` or
     :attr:`DockConfig.turn_side` — ``right`` (the lane 375 mm to the dock's right, a right
     U-turn) or ``left`` (mirrored, where the room needs it: the apartment's dock has the
     corridor wall on its right);
  1. unless already on the lane facing the dock, walk to the lane's far point (1.5 m out; tight
     arcs vx 0.30 / vyaw ±1.0 to turn, never on the spot) and arc round to face the dock within
     20°; look; follow the lane in to 1.15 m (carrot on the lane, PI + the gait's drift
     feed-forward), one look a second; the last fix is there;
  2. stop 0.6 s; U-turn from standing toward the axis (vx 0.30, vyaw ∓1.0) until the IMU yaw has
     turned 178°: it ends on the axis ~1.1 m out, facing the room; stop 0.4 s;
  3. reverse in at −0.40: the first 0.8 s straight (feed-forward only), then steered on
     odometry toward the axis; no 15 mm of progress after 2 s: one straight 1.2 s reverse;
  4. the alignment gate: inside 0.6 m, the 0.5 s-averaged pose more than 80 mm or 20° off the
     axis → walk out along the axis to 1 m (straightening) and reverse again, at most twice (a
     third time: the whole approach again, at most :attr:`DockConfig.max_approaches` in all);
  5. **the face stop**: reversing until blocked — after 30 mm, less than 12 mm of progress in
     1.5 s by odometry, or the body's own stall (``blocked = arrived``) — the rear is at the
     face; walk out 0.7 s at 0.35 and stop 0.6 s (the sit's ~70 mm slide back needs the room).
     The fallback (:attr:`DockConfig.stop_by` ``pose``) stops by the pose at 289 mm from the
     wall; the face stop also ends when odometry puts the rear at the face;
  6. sit, then PG (``pg``, the sensor, never the simulator's truth); PG open → stand and sit
     again (no walking), at most :attr:`DockConfig.max_sits` sits, then ``body.dock {contact:
     false}`` (Duck Core calls for help).

  Leaving is Duck Core's ``duck.walk`` straight out (no reversing, no arcs inside the guides).
* ``forward`` — walk in facing the dock's tag, stop just short of the soft front bumper, sit (the
  thighs land on the two floor pads); PG open → stand, **reverse** straight out to the line-up
  point, look, walk in again; at most :attr:`DockConfig.max_tries` tries.

Both styles reverse (owner 2026-09-28 「1. 可以」: reversing inside ``duck.dock``, never faster
than −0.40 m/s); the body loop judges a backing move by the body's own reflexes only (the ToF
looks ahead, away from where the duck goes).

**How it knows where it is.** Only from what the duck senses: a *fix* is one still look at the
dock's tag (duck-eye ``nest`` task: range, bearing, the tag's relative yaw; within 0.5 m the small
flat tag at the plate's front edge, read looking down), turned into the duck's pose in the dock's
frame (:func:`pose_from_fix`); between fixes the pose is carried by odometry. Walking toward the
dock is in short segments ("one step, one look", spec §8.5), then stop, tail, look; backing in, the
camera faces the room and the last fix before the U-turn carries the pose. A lit status light or
the cyan panel gives a direction only (owner C5): the duck may turn toward it and creep, never stop
on it.

**Geometry** (:class:`DockGeometry`, mm in the dock's frame: x from the wall into the room along
the dock's axis, y to the left of +x): ``forward`` is ``duckmind.world.dock.approach()``; ``backin``
takes dock:measure's calibrated numbers (stop 289 not 309, turn 1150, lane 375; ``tests/
test_duckbody_dockexec.py`` checks both against ``world.dock``); a board's dock gives its own.

A pure step machine (:class:`DockExec`, an :class:`~duckmind.duckbody.protocol.ActExecutor`): the
body loop gives it the :class:`~duckmind.duckbody.protocol.ActContext` (PG as ``ctx.pg`` when the
loop has it) and sends its :class:`~duckmind.duckbody.protocol.ActOutput` through the reflexes and
the guard. Floats inside; the percept is integers.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.duckbody import lookplan as L
from duckmind.duckbody.protocol import DOCK_SIDES, ActContext, ActOutput, DockCmd
from duckmind.duckbody.reflex import HEAD_SETTLE_MS

RESULT_KIND = "body.dock"
STYLES = ("forward", "backin")
SIDES = DOCK_SIDES
DEFAULT_STYLE = "backin"  # owner 2026-09-28 「6. 可以」
# reflexes about what is ahead of the duck (the ToF's): they never end a backing leg
AHEAD_HALTS = ("reflex_near", "reflex_touch", "reflex_drop", "stop_near", "no_drop")


@dataclass(frozen=True)
class DockGeometry:
    """Where things are along the dock's axis (mm from the wall; see ``world.dock``)."""

    face_mm: int  # the front of the face panel: where the big tag is
    edge_tag_mm: int  # the small flat tag at the plate's front edge
    line_up_mm: int  # forward: where to line up; backin: where the alignment gate walks out to
    stop_mm: int  # where the trunk stops walking (forward: chest 15 mm short of the bumper;
    #               backin: the pose-stop fallback, dock:measure's calibration)
    seat_mm: int  # where the seated trunk ends up
    turn_mm: int = 800  # backin: where the U-turn starts
    uturn_offset_mm: int = 333  # backin: the lane's offset from the axis (a U-turn's width)
    far_mm: int = 1500  # backin: the lane's far point, where the J-hook begins
    rear_mm: int = 47  # the trunk's rear behind the trunk origin (world.dock.REAR_MM)

    @property
    def face_contact_mm(self) -> int:
        """Where the trunk is when its rear meets the face."""
        return self.face_mm + self.rear_mm


GEOMETRY: dict[str, DockGeometry] = {
    # world.dock.approach() with DockShape(): see the tests
    "forward": DockGeometry(face_mm=120, edge_tag_mm=315, line_up_mm=700, stop_mm=173, seat_mm=240),
    # DockShape(panel_mm=1) with dock:measure's calibration (dock_styles.json recommendation:
    # stop 309 → 289, turn 800 → 1150, lane 333 → 375; the realign point 1000)
    "backin": DockGeometry(
        face_mm=111,
        edge_tag_mm=315,
        line_up_mm=1000,
        stop_mm=289,
        seat_mm=172,
        turn_mm=1150,
        uturn_offset_mm=375,
        far_mm=1500,
    ),
}


@dataclass(frozen=True)
class DockConfig:
    style: str = DEFAULT_STYLE
    turn_side: str = "right"  # backin: right (the default J-hook) or left (mirrored)
    max_tries: int = 3  # forward: walk-ins that end in a sit
    max_sits: int = 3  # backin: sits per approach — the first, then "stand, sit again" twice
    max_approaches: int = 2  # backin: a second J-hook only when the alignment gate fails 3 times
    vx_mm: int = 400  # approaching a specific thing (owner A3)
    reverse_vx_mm: int = -400  # vx −0.3 steps on the spot; −0.4 moves 0.15 m/s (dock scripts)
    uturn_vx_mm: int = 300  # dock:measure's U-turn: vx 0.30, vyaw ±1.0, ~6 s
    uturn_vyaw_mrad: int = 1000  # its size; the side gives the sign (right: −, left: +)
    uturn_turn_mrad: int = 3107  # 178° (dock:measure; 0.95 π in phase 2's re-dock loop)
    uturn_timeout_ms: int = 12_000
    arc_vx_mm: int = 300  # a tight arc to turn (vx 0.30, vyaw ±1.0): never turn on the spot
    arc_vyaw_mrad: int = 1000
    arc_mrad: int = 950  # the bearing this far off: turn with a tight arc
    arc_radius_mm: int = 230  # a tight arc's radius (0.17–0.25 m measured)
    goto_kp_milli: int = 2500
    lane_arrive_mm: int = 150
    face_tol_mrad: int = 350  # facing the dock within 20°
    goto_timeout_ms: int = 30_000
    face_timeout_ms: int = 10_000
    lane_step_ms: int = 1000  # backin, following the lane in: one look a second
    far_step_ms: int = 1500
    near_step_ms: int = 600
    near_mm: int = 600  # inside this from the stop point: short steps
    edge_tag_within_mm: int = 500  # look down for the edge tag inside this from the face
    final_blind_mm: int = 350  # the last stretch to a goal goes on odometry, without looks…
    max_blind_mm: int = 450  # …and when no tag reads this close to it on a fix …
    blind_fix_age_ms: int = 15_000  # …no older than this
    carrot_mm: int = 250  # steer at a point on the axis this far ahead (dock scripts)
    # walking in along the axis, never head more than this off the tag's direction: the head can
    # still aim at it (±540 mrad) for the next fix. A bare 250 mm carrot from 0.36 m off the axis
    # turned the duck 55° away and every later look missed the tag (phase-3 fix, rig re-run).
    keep_tag_mrad: int = 450
    kp_milli: int = 2500  # vyaw = kp · heading error + ki · ∫ + feed-forward
    ki_milli: int = 800
    ff_mrad: int = 150  # the walk's own rightward turn (−8.5°/s at vx 0.35)
    ff_reverse_mrad: int = 80  # (−4.5°/s at vx −0.4)
    lateral_tol_mm: int = 60  # (forward) lined up when this close to the axis…
    heading_tol_mrad: int = 175  # …and within 10° of it
    fix_misses: int = 4  # looks in a row without the tag before giving up the try
    sit_timeout_ms: int = 6000
    stand_timeout_ms: int = 6000
    toggle_retry_ms: int = 3000
    pg_settle_ms: int = 800  # after the sit, before reading PG
    tail_ms: int = L.TAIL_MS
    pause_ms: int = 600  # dock:measure's stops: before the U-turn and the sit …
    brief_ms: int = 400  # … and after the U-turn
    # backin's reverse in (dock:measure's choreography, steps 3–5)
    reverse_straight_ms: int = 800
    kick_after_ms: int = 2000
    kick_ms: int = 1200
    kick_min_mm: int = 15
    reverse_max_ms: int = 15_000
    gate_x_mm: int = 600
    gate_mm: int = 80
    gate_mrad: int = 349  # 20°
    gate_window_ms: int = 500  # one gait cycle: the trunk sways ±20–25 mm with every step
    max_relines: int = 2
    realign_tol_mm: int = 30
    realign_tol_mrad: int = 90
    realign_timeout_ms: int = 12_000
    stop_by: str = "face"  # "pose": dock:measure's fallback, stop by the pose at stop_mm
    blocked_min_mm: int = 30
    blocked_window_ms: int = 1500
    blocked_progress_mm: int = 12
    bump_walk_vx_mm: int = 350  # after the face stop: walk out this fast …
    bump_walk_ms: int = 700  # … this long (1.0 s overshoots; 2.0 s walked ~260 mm)
    # A stop_near reflex this close to the stop point counts as arriving (forward). The dock's
    # face is a wall to the ToF: the body loop must relax stop_near toward the dock during this
    # act (the A1 rule with :meth:`DockExec.target` as the just-seen target), or no ToF duck can
    # dock forward.
    stop_near_arrives_mm: int = 60
    # owner B3: an unsure lift only stops. Stepping onto the dock's plate can make the legs go
    # light for a moment (phase-3 verify: 15 reversing acts ended reflex_unloaded): hold still
    # this long for the body to know — the floor explains it (walk on) or it is a lift (the body
    # loop ends the act) — before giving the try up.
    unsure_wait_ms: int = 3000
    geometry: DockGeometry | None = None

    def geo(self) -> DockGeometry:
        return self.geometry or GEOMETRY[self.style]


@dataclass(frozen=True)
class DockPose:
    """The trunk in the dock's frame: x from the wall along the axis (mm), y to the left of +x
    (the dock's own left, facing the room; mm), heading (rad; 0 = facing the room, π = facing
    the dock)."""

    x: float
    y: float
    h: float  # radians


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_from_fix(
    fix: Mapping[str, Any], geometry: DockGeometry, heading: float | None = None
) -> DockPose | None:
    """The duck's pose from one tag fix (:func:`duckmind.duckbody.lookplan.nest_fix`: ``r_mm``
    range from the trunk origin to the tag, ``b_mrad`` its bearing in the level trunk frame,
    left +, ``yaw_mrad`` the duck's heading relative to facing the tag square-on, left +; ``tag``
    17 = the face tag, 18 = the plate's edge tag). ``heading`` (radians, the dock frame) is the
    carried heading to use instead of the tag's facing when the eye says that facing is unsure
    (``yaw_sure`` False: the mirror pose). ``None`` without a range, or without any heading (a
    direction-only cue, owner C5)."""
    r = fix.get("r_mm")
    yaw = fix.get("yaw_mrad")
    if r is None:
        return None
    if yaw is not None and (fix.get("yaw_sure", True) or heading is None):
        heading = wrap(math.pi + int(yaw) / 1000.0)
    if heading is None:
        return None
    tag_x = geometry.edge_tag_mm if int(fix.get("tag", 17)) == 18 else geometry.face_mm
    toward = heading + int(fix.get("b_mrad", 0)) / 1000.0
    return DockPose(tag_x - float(r) * math.cos(toward), -float(r) * math.sin(toward), heading)


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


@dataclass(frozen=True)
class DockFrame:
    """Where the dock is in duck-body's odometry frame, kept when a docking closed PG: the
    tracker's last fix and the odometry at it (``epoch``: the posture tap's odometry epoch — a
    lift starts a new one, and the frame no longer holds). walkexec's ``leave_nest`` walk steers
    out along the axis with it (the recommendation's step 8)."""

    fix: DockPose
    odom0: tuple[float, float, float]  # (x mm, y mm, h rad) at the fix
    epoch: int
    geometry: DockGeometry

    def pose(self, odom: tuple[float, float, float]) -> DockPose:
        return _Tracker(self.fix, self.odom0).now(odom) or self.fix


@dataclass
class _Leg:
    """One driving leg: go forward (``reverse`` False) or backward toward ``goal_x`` along the
    axis (steering onto it), for at most ``ms``."""

    goal_x: float
    reverse: bool
    ms: int
    lane_y: float = 0.0
    started_ms: int = 0
    integral: float = 0.0
    last_ms: int = 0
    settled: bool = False
    moved_from: DockPose | None = None  # where the legs first moved (backin's reverse in)
    moving_ms: int | None = None  # when they did
    samples: deque[tuple[int, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    kicked: bool = False
    kick_until: int | None = None
    restart: bool = False  # held for an unsure body: the next move starts the leg over
    first_ms: int | None = None  # when the legs first moved (the leg's whole time budget)


PgSource = Callable[[], "bool | None"]


@dataclass
class DockExec:
    """One ``duck.dock`` act (see the module docstring)."""

    command: DockCmd
    start_ms: int
    config: DockConfig = DockConfig()
    pg_source: PgSource | None = None  # the body loop's PG reader when ctx carries none
    look_prefix: str = ""
    memory: Any = None  # walkexec's WalkMemory: a docking that closes PG leaves its DockFrame
    act_id: str = field(init=False, default="")
    percept: dict[str, Any] | None = None
    # state
    phase: str = "stand"
    phase_ms: int = 0
    plan: list[str] = field(default_factory=list)
    tries: int = 0
    sits: int = 0  # backin: sits in this approach
    approaches: int = 1
    relines: int = 0
    lane_loops: int = 0
    sub: str | None = None  # to_lane: "goto" or "face"
    side: str = "right"
    seen_tag: bool = False
    misses: int = 0
    looks: int = 0
    tracker: _Tracker = field(default_factory=_Tracker)
    leg: _Leg | None = None
    look: L.LookRun | None = None
    toggled_ms: int | None = None
    sat_ms: int | None = None
    last_move_ms: int | None = None
    turned: float = 0.0
    last_h: float | None = None
    fix_ms: int | None = None
    settle: L.HomeSettle = field(default_factory=L.HomeSettle)
    unsure_ms: int | None = None
    reason: str | None = None
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.act_id = self.command.act_id
        if self.command.style is not None and self.command.style != self.config.style:
            self.config = replace(self.config, style=self.command.style, geometry=None)
        if self.config.style not in STYLES:
            raise ValueError(f"dock style {self.config.style!r} not in {STYLES}")
        self.side = getattr(self.command, "side", None) or self.config.turn_side
        if self.side not in SIDES:
            raise ValueError(f"dock side {self.side!r} not in {SIDES}")
        self.phase_ms = self.start_ms
        self.plan = self._first_plan()[1:]  # the act starts in "stand"

    # ── the plans ────────────────────────────────────────────────────────────

    def _first_plan(self) -> list[str]:
        if self.config.style == "forward":
            return ["stand", "fix", "approach", "tail", "sit", "pg"]
        return ["stand", *self._approach_plan()]

    def _approach_plan(self) -> list[str]:
        """backin, dock:measure steps 1–6: to the lane, in along it, the U-turn, the reverse in
        with its face stop, the sit."""
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

    def _retry_plan(self) -> list[str]:
        if self.config.style == "forward":
            return ["stand", "reverse_out", "tail", "fix", "approach", "tail", "sit", "pg"]
        return ["stand", "sit", "pg"]  # backin: stand, sit again — no walking (step 6)

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, ctx: ActContext) -> ActOutput:
        now = ctx.t_ms
        if ctx.proprio.fallen or ctx.proprio.lifted:
            return self._done(ctx, contact=False, why="fell" if ctx.proprio.fallen else "lifted")
        if self.command.deadline_ms and now >= self.command.deadline_ms:
            return self._done(ctx, contact=False, why="deadline")
        handler = getattr(self, f"_phase_{self.phase}")
        out: ActOutput = handler(ctx, now)
        return out

    def preempt(self, why: str, ctx: ActContext) -> ActOutput:
        return self._done(ctx, contact=False, why=why)

    def _next(self, ctx: ActContext, now: int) -> ActOutput:
        if not self.plan:
            return self._done(ctx, contact=False, why="plan")
        self.phase = self.plan.pop(0)
        self.phase_ms = now
        self.leg = None
        self.look = None
        self.sub = None
        self.log.append(self.phase)
        handler = getattr(self, f"_phase_{self.phase}")
        out: ActOutput = handler(ctx, now)
        return out

    def _odom(self, ctx: ActContext) -> tuple[float, float, float] | None:
        if ctx.pose is None:
            return None
        x, y, h = ctx.pose.odom
        return (x * 1000.0, y * 1000.0, h)

    def pose(self, ctx: ActContext) -> DockPose | None:
        return self.tracker.now(self._odom(ctx))

    def target(self, ctx: ActContext) -> dict[str, int] | None:
        """The dock's face as the just-seen target, for the reflexes' A1 relaxation: bearing
        (trunk frame, mrad) and range (mm, trunk origin to the face) from the carried pose; the
        fix's age (ms). ``None`` before the first fix."""
        pose = self.pose(ctx)
        if pose is None or self.fix_ms is None:
            return None
        face = self.config.geo().face_mm
        bearing = wrap(math.atan2(-pose.y, face - pose.x) - pose.h)
        return {
            "b_mrad": int(round(bearing * 1000)),
            "r_mm": int(round(math.hypot(face - pose.x, pose.y))),
            "age_ms": ctx.t_ms - self.fix_ms,
        }

    def lane_y(self) -> float:
        """backin's lane: to the dock's right (y < 0) or, mirrored, its left."""
        offset = float(self.config.geo().uturn_offset_mm)
        return offset if self.side == "left" else -offset

    def turn_sign(self) -> int:
        """The U-turn's direction: a right turn (vyaw < 0) from the right lane, left mirrored."""
        return 1 if self.side == "left" else -1

    # ── posture ──────────────────────────────────────────────────────────────

    def _phase_stand(self, ctx: ActContext, now: int) -> ActOutput:
        if ctx.proprio.posture == "stand" and ctx.proprio.policy in ("walk", "stand", ""):
            self.toggled_ms = None
            return self._next(ctx, now)
        if now - self.phase_ms > self.config.stand_timeout_ms:
            return self._done(ctx, contact=False, why="stand")
        stale = self.toggled_ms is None or now - self.toggled_ms >= self.config.toggle_retry_ms
        if stale and ctx.proprio.posture == "sit":  # never toggle an unknown posture
            self.toggled_ms = now
            return ActOutput(do="sit_toggle")
        return ActOutput()

    def _phase_sit(self, ctx: ActContext, now: int) -> ActOutput:
        if ctx.proprio.posture == "sit":
            if self.sat_ms is None:
                self.sat_ms = now
            if now - self.sat_ms >= self.config.pg_settle_ms:
                self.toggled_ms = None
                self.sat_ms = None
                return self._next(ctx, now)
            return ActOutput()
        if now - self.phase_ms > self.config.sit_timeout_ms:
            return self._done(ctx, contact=False, why="sit")
        stale = self.toggled_ms is None or now - self.toggled_ms >= self.config.toggle_retry_ms
        if stale and ctx.proprio.posture == "stand":
            self.toggled_ms = now
            return ActOutput(do="sit_toggle")
        return ActOutput()

    def _phase_pg(self, ctx: ActContext, now: int) -> ActOutput:
        pg = getattr(ctx, "pg", None)
        if pg is None and self.pg_source is not None:
            pg = self.pg_source()
        self.tries += 1
        self.sits += 1
        if pg:
            return self._done(ctx, contact=True, why="pg")
        if self.config.style == "forward":
            if self.tries >= self.config.max_tries:
                return self._done(ctx, contact=False, why="tries")
        elif self.sits >= self.config.max_sits:
            return self._done(ctx, contact=False, why="tries")
        self.plan = self._retry_plan()[1:]
        self.phase = "stand"
        self.phase_ms = now
        self.toggled_ms = None
        self.sat_ms = None
        self.log.append("retry")
        return self._phase_stand(ctx, now)

    def _phase_tail(self, ctx: ActContext, now: int) -> ActOutput:
        return self._hold(ctx, now, self.config.tail_ms)

    def _phase_pause(self, ctx: ActContext, now: int) -> ActOutput:
        return self._hold(ctx, now, self.config.pause_ms)

    def _phase_brief(self, ctx: ActContext, now: int) -> ActOutput:
        return self._hold(ctx, now, self.config.brief_ms)

    def _hold(self, ctx: ActContext, now: int, ms: int) -> ActOutput:
        if now - self.phase_ms >= ms:
            return self._next(ctx, now)
        return ActOutput()

    # ── looking at the tag ───────────────────────────────────────────────────

    def _phase_fix(self, ctx: ActContext, now: int) -> ActOutput:
        cfg = self.config
        if self.look is None:
            pose = self.pose(ctx)
            near = pose is not None and pose.x - cfg.geo().face_mm < cfg.edge_tag_within_mm
            spec = L.CATALOGUE["glance_down"] if near else L.CATALOGUE["fixate"]
            yaw = 0
            if self.misses:  # search: turn the head across the validated range
                yaw = (L.SCAN_LEFT_MRAD, L.SCAN_RIGHT_MRAD, 0)[(self.misses - 1) % 3]
            elif pose is not None:
                bearing = wrap(math.atan2(-pose.y, cfg.geo().face_mm - pose.x) - pose.h)
                yaw = int(round(bearing * 1000))
            self.looks += 1
            look_id = f"{self.look_prefix}{self.act_id}.fix{self.looks}"
            self.look = L.LookRun(
                spec,
                look_id,
                now,
                yaw_mrad=yaw,
                mode="docking",
                hints={"want_tag": 18 if near else 17},
            )
        head, asks, finished = self.look.step(
            now,
            still=bool(ctx.pose and ctx.pose.still),
            legs_quiet=L.legs_quiet(now, self.last_move_ms),
            results=ctx.eye,
        )
        asks = [_with_nest(a) for a in asks]
        if not finished:
            return ActOutput(head=None if head == L.HOME else L.to_radians(head), looks=tuple(asks))
        result = self.look.results[-1] if self.look.results else None
        self.look = None
        fix = L.nest_fix(result)
        odom = self._odom(ctx)
        carried_now = self.tracker.now(odom)
        pose = (
            pose_from_fix(fix, cfg.geo(), carried_now.h if carried_now is not None else None)
            if fix
            else None
        )
        if pose is not None and odom is not None:
            self.seen_tag = True
            self.misses = 0
            self.fix_ms = now
            self.tracker.set(pose, odom)
            return self._next(ctx, now)
        if fix is not None and odom is not None and self.tracker.fix is not None:
            # a direction only (the status light, the cyan panel: owner C5): keep the carried
            # position, correct the heading toward the dock
            carried = self.tracker.now(odom)
            if carried is not None:
                want = math.atan2(-carried.y, cfg.geo().face_mm - carried.x)
                heading = wrap(want - int(fix.get("b_mrad", 0)) / 1000.0)
                self.tracker.set(DockPose(carried.x, carried.y, heading), odom)
                self.misses = 0
                return self._next(ctx, now)
        # no tag: close enough to the goal on a recent fix, go on blind (the last stretch is
        # odometry anyway); else look again, then give up
        carried = self.tracker.now(odom)
        upcoming = self.plan[0] if self.plan else ""
        goal = self._goal_x(upcoming)
        recent = self.fix_ms is not None and now - self.fix_ms <= cfg.blind_fix_age_ms
        if (
            carried is not None
            and goal is not None
            and recent
            and abs(carried.x - goal) <= cfg.max_blind_mm
        ):
            self.misses = 0
            return self._next(ctx, now)
        self.misses += 1
        if self.misses >= cfg.fix_misses:
            return self._done(ctx, contact=False, why="no_tag")
        return ActOutput()  # look again (next tick starts a new look)

    # ── driving legs ─────────────────────────────────────────────────────────

    def _unsure(self, ctx: ActContext, now: int, leg: _Leg | None) -> ActOutput | None:
        """Owner B3: unsure only stops — hold still until the body knows (the floor explains
        it: carry on after the head settles again; a lift: the body loop ends the act)."""
        if ctx.proprio.unloaded or ctx.halted_by == "reflex_unloaded":
            if self.unsure_ms is None:
                self.unsure_ms = now
            if now - self.unsure_ms <= self.config.unsure_wait_ms:
                if leg is not None:
                    leg.settled = False
                self.settle = L.HomeSettle()
                if self.log[-1:] != ["unsure_hold"]:
                    self.log.append("unsure_hold")
                return ActOutput()
            return self._done(ctx, contact=False, why="reflex:reflex_unloaded")
        self.unsure_ms = None
        return None

    def _settled(self, ctx: ActContext, now: int, leg: _Leg) -> bool:
        """Rule 3 (and the guard): the head at HOME and settled 0.3 s before the legs move — by
        the guard's own count when the body loop gives it (else a first move would be refused
        and zeroed)."""
        if leg.settled:
            return True
        if not self.settle.step(now, _head_home(ctx)) or not _guard_settled(ctx):
            leg.started_ms = now
            leg.last_ms = now
            return False
        leg.settled = True
        return True

    def _drive(
        self, ctx: ActContext, now: int, leg: _Leg, arrived: Callable[[DockPose], bool]
    ) -> ActOutput:
        cfg = self.config
        pose = self.pose(ctx)
        if pose is None:
            return self._done(ctx, contact=False, why="lost")
        held = self._unsure(ctx, now, leg)
        if held is not None:
            return held
        if not self._settled(ctx, now, leg):
            return ActOutput()
        halted = ctx.halted_by
        if halted and leg.reverse and halted in AHEAD_HALTS:
            halted = None  # backing away from what is ahead
        if halted:
            near_stop = abs(pose.x - cfg.geo().stop_mm) <= cfg.stop_near_arrives_mm
            if "near" in halted and near_stop and not leg.reverse:
                return self._next(ctx, now)  # the face is right there: that is arriving
            return self._done(ctx, contact=False, why=f"reflex:{halted}")
        if arrived(pose):
            return self._next(ctx, now)
        if now - leg.started_ms >= leg.ms:
            return self._segment_end(ctx, now)
        return self._steer(ctx, now, leg, pose)

    def _steer(self, ctx: ActContext, now: int, leg: _Leg, pose: DockPose) -> ActOutput:
        """The leg's twist: a carrot on the lane (y = lane_y) further along the direction of
        travel (:attr:`DockConfig.carrot_mm`; backin, as dock:measure steered: more when far
        off the lane, 1.5 × the offset), PI on the heading error plus the gait's drift
        feed-forward."""
        cfg = self.config
        dt = max(0.0, (now - (leg.last_ms or now)) / 1000.0)
        leg.last_ms = now
        toward_wall = leg.goal_x < pose.x
        reach = float(cfg.carrot_mm)
        if cfg.style == "backin":
            reach = max(reach, 1.5 * abs(pose.y - leg.lane_y))
        carrot_x = pose.x - reach if toward_wall else pose.x + reach
        target = math.atan2(leg.lane_y - pose.y, carrot_x - pose.x)
        if toward_wall and not leg.reverse and leg.lane_y == 0.0 and cfg.keep_tag_mrad > 0:
            to_tag = math.atan2(-pose.y, cfg.geo().face_mm - pose.x)
            off = wrap(target - to_tag)
            limit = cfg.keep_tag_mrad / 1000.0
            if abs(off) > limit:
                target = to_tag + math.copysign(limit, off)
        moving = wrap(pose.h + math.pi) if leg.reverse else pose.h
        err = wrap(target - moving)
        leg.integral = max(-0.3, min(0.3, leg.integral + err * dt))
        ff = (cfg.ff_reverse_mrad if leg.reverse else cfg.ff_mrad) / 1000.0
        vyaw = cfg.kp_milli / 1000.0 * err + cfg.ki_milli / 1000.0 * leg.integral + ff
        vyaw = max(-1.0, min(1.0, vyaw))
        vx = cfg.reverse_vx_mm if leg.reverse else cfg.vx_mm
        self.last_move_ms = now
        return ActOutput(twist=(vx, int(round(vyaw * 1000))))

    def _segment_end(self, ctx: ActContext, now: int) -> ActOutput:
        """A segment ran its time: stop, let the tail go, look again, carry on — except for the
        last stretch, which no tag can be read from (the face tag fills the picture inside
        ~0.5 m, the edge tag passes under the head): that goes on odometry in one leg."""
        current = self.phase
        pose = self.pose(ctx)
        leg = self.leg
        if (
            pose is not None
            and leg is not None
            and abs(pose.x - leg.goal_x) <= self.config.final_blind_mm
        ):
            leg.started_ms = now
            leg.ms = max(leg.ms, 4000)
            return self._drive(ctx, now, leg, self._arrival(current))
        self.plan[:0] = ["tail", "fix", current]
        return self._next(ctx, now)

    def _goal_x(self, phase: str) -> float | None:
        geo = self.config.geo()
        return {
            "approach": geo.stop_mm,
            "reverse_in": geo.stop_mm,
            "to_turn": geo.turn_mm,
            "uturn": geo.turn_mm,  # the last fix before the U-turn: a recent one will do
            "reverse_out": geo.line_up_mm,
        }.get(phase)

    def _arrival(self, phase: str) -> Callable[[DockPose], bool]:
        geo = self.config.geo()
        if phase == "approach":
            return lambda p: p.x <= geo.stop_mm
        if phase == "to_turn":
            return lambda p: p.x <= geo.turn_mm
        if phase == "reverse_out":
            return lambda p: p.x >= geo.line_up_mm
        return lambda p: p.x >= geo.turn_mm

    def _start_leg(
        self, now: int, goal_x: float, reverse: bool, ms: int, lane_y: float = 0.0
    ) -> _Leg:
        leg = _Leg(
            goal_x=goal_x, reverse=reverse, ms=ms, lane_y=lane_y, started_ms=now, last_ms=now
        )
        self.leg = leg
        self.settle = L.HomeSettle()
        return leg

    def _step_ms(self, pose: DockPose | None) -> int:
        cfg = self.config
        if pose is None:
            return cfg.near_step_ms
        return (
            cfg.near_step_ms if abs(pose.x - cfg.geo().stop_mm) < cfg.near_mm else cfg.far_step_ms
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def _phase_approach(self, ctx: ActContext, now: int) -> ActOutput:
        """forward: walk in along the axis until the trunk is at the stop point."""
        geo = self.config.geo()
        if self.leg is None:
            self._start_leg(now, geo.stop_mm, reverse=False, ms=self._step_ms(self.pose(ctx)))
        assert self.leg is not None
        return self._drive(ctx, now, self.leg, lambda p: p.x <= geo.stop_mm)

    def _phase_reverse_out(self, ctx: ActContext, now: int) -> ActOutput:
        """forward's retry: back straight out to the line-up point (facing the dock)."""
        geo = self.config.geo()
        if self.leg is None:
            self._start_leg(now, geo.line_up_mm, reverse=True, ms=8000)
        assert self.leg is not None
        return self._drive(ctx, now, self.leg, lambda p: p.x >= geo.line_up_mm)

    # ── backin: to the lane, along it, the U-turn ────────────────────────────

    def _on_lane(self, pose: DockPose) -> bool:
        """Already on the pre-turn lane, in front of the turn point and facing the dock
        (dock:measure's test)."""
        geo = self.config.geo()
        ahead = pose.x - geo.turn_mm
        return (
            ahead >= 250
            and abs(pose.y - self.lane_y()) <= 0.6 * ahead
            and abs(wrap(pose.h - math.pi)) <= 1.05
        )

    def _phase_to_lane(self, ctx: ActContext, now: int) -> ActOutput:
        """backin step 1: unless on the lane already, walk to its far point (tight arcs to
        turn), then arc round to face the dock within 20° — on odometry from the last fix (the
        tag is behind the duck most of the way); at most twice."""
        cfg, geo = self.config, self.config.geo()
        pose = self.pose(ctx)
        if pose is None:
            return self._done(ctx, contact=False, why="lost")
        if self.sub is None:
            if self._on_lane(pose):
                if self.lane_loops == 0 and self.plan[:1] == ["fix"]:
                    self.plan.pop(0)  # it has just looked: follow the lane in
                return self._next(ctx, now)
            self.lane_loops += 1
            self.sub = "goto"
            self._start_leg(now, geo.far_mm, reverse=False, ms=cfg.goto_timeout_ms)
        leg = self.leg
        assert leg is not None
        held = self._unsure(ctx, now, leg)
        if held is not None:
            return held
        if not self._settled(ctx, now, leg):
            return ActOutput()
        if ctx.halted_by:
            return self._done(ctx, contact=False, why=f"reflex:{ctx.halted_by}")
        if self.sub == "goto":
            lane = self.lane_y()
            d = math.hypot(geo.far_mm - pose.x, lane - pose.y)
            off = wrap(math.atan2(lane - pose.y, geo.far_mm - pose.x) - pose.h)
            passing = d < 2 * cfg.arc_radius_mm and abs(off) > math.pi / 2  # turning back circles
            if d < cfg.lane_arrive_mm or passing:
                self.sub = "face"
                leg.started_ms = now
            elif now - leg.started_ms > cfg.goto_timeout_ms:
                return self._done(ctx, contact=False, why="lane")
            else:
                return self._goto_twist(now, pose, geo.far_mm, lane)
        err = wrap(math.pi - pose.h)
        if abs(err) * 1000 < cfg.face_tol_mrad:
            if self._on_lane(pose) or self.lane_loops >= 2:
                return self._next(ctx, now)  # facing the dock on the lane: look, then in
            self.sub = None  # the arc carried it off the lane or toward the dock: again
            return ActOutput()
        if now - leg.started_ms > cfg.face_timeout_ms:
            return self._done(ctx, contact=False, why="face")
        self.last_move_ms = now
        return ActOutput(twist=(cfg.arc_vx_mm, int(math.copysign(cfg.arc_vyaw_mrad, err))))

    def _goto_twist(self, now: int, pose: DockPose, px: float, py: float) -> ActOutput:
        """Toward a point: bearing steering, a tight arc while the bearing is more than
        :attr:`DockConfig.arc_mrad` off — unless the point is inside that arc's circle (it would
        circle it for ever): then straight on until it is not."""
        cfg = self.config
        err = wrap(math.atan2(py - pose.y, px - pose.x) - pose.h)
        self.last_move_ms = now
        if abs(err) * 1000 > cfg.arc_mrad:
            side = math.copysign(1.0, err)
            r = float(cfg.arc_radius_mm)
            cx = pose.x - side * r * math.sin(pose.h)
            cy = pose.y + side * r * math.cos(pose.h)
            if math.hypot(px - cx, py - cy) < r * 1.15:
                return ActOutput(twist=(cfg.vx_mm, cfg.ff_mrad))
            return ActOutput(twist=(cfg.arc_vx_mm, int(side * cfg.arc_vyaw_mrad)))
        vyaw = cfg.goto_kp_milli / 1000.0 * err + cfg.ff_mrad / 1000.0
        return ActOutput(twist=(cfg.vx_mm, int(round(max(-1.0, min(1.0, vyaw)) * 1000))))

    def _phase_to_turn(self, ctx: ActContext, now: int) -> ActOutput:
        """backin: follow the lane toward the dock to where the U-turn starts, one look a
        second."""
        geo = self.config.geo()
        if self.leg is None:
            self._start_leg(
                now,
                geo.turn_mm,
                reverse=False,
                ms=self.config.lane_step_ms,
                lane_y=self.lane_y(),
            )
        assert self.leg is not None
        return self._drive(ctx, now, self.leg, lambda p: p.x <= geo.turn_mm)

    def _phase_uturn(self, ctx: ActContext, now: int) -> ActOutput:
        """A U-turn from standing toward the axis (backin; the side's way round): vx 0.30,
        vyaw ∓1.0 until the heading has turned 178° by odometry (the IMU's yaw)."""
        odom = self._odom(ctx)
        if odom is None:
            return self._done(ctx, contact=False, why="lost")
        if self.last_h is None:
            if not self.settle.step(now, _head_home(ctx)) or not _guard_settled(ctx):
                self.phase_ms = now
                return ActOutput()
            self.last_h, self.turned = odom[2], 0.0
        self.turned += wrap(odom[2] - self.last_h)
        self.last_h = odom[2]
        if abs(self.turned) * 1000 >= self.config.uturn_turn_mrad:
            self.last_h = None
            return self._next(ctx, now)
        if ctx.halted_by:
            return self._done(ctx, contact=False, why=f"reflex:{ctx.halted_by}")
        if now - self.phase_ms > self.config.uturn_timeout_ms:
            return self._done(ctx, contact=False, why="uturn")
        self.last_move_ms = now
        vyaw = self.turn_sign() * self.config.uturn_vyaw_mrad
        return ActOutput(twist=(self.config.uturn_vx_mm, vyaw))

    # ── backin: the reverse in and the face stop ─────────────────────────────

    def _phase_reverse_in(self, ctx: ActContext, now: int) -> ActOutput:
        """backin steps 3–5: reverse in along the axis — straight first, then steered, one kick
        if it will not start — with the alignment gate, until the face stops it (or, with
        ``stop_by`` ``pose``, the trunk reaches the fallback stop)."""
        cfg, geo = self.config, self.config.geo()
        if self.leg is None:
            self._start_leg(now, geo.face_contact_mm, reverse=True, ms=cfg.reverse_max_ms)
        leg = self.leg
        assert leg is not None
        pose = self.pose(ctx)
        if pose is None:
            return self._done(ctx, contact=False, why="lost")
        held = self._unsure(ctx, now, leg)
        if held is not None:
            leg.restart = True  # owner B3: a reverse start can dip the leg load under "unsure"
            return held
        if not self._settled(ctx, now, leg):
            return ActOutput()
        if leg.moved_from is None or leg.restart:
            # (re)start: the straight start, the kick and the progress window count from here
            leg.moved_from, leg.moving_ms, leg.restart = pose, now, False
            leg.samples.clear()
            leg.kick_until = None
            leg.first_ms = now if leg.first_ms is None else leg.first_ms
        leg.samples.append((now, pose.x, pose.y, pose.h))
        moving_ms = leg.moving_ms if leg.moving_ms is not None else now
        moved = math.hypot(pose.x - leg.moved_from.x, pose.y - leg.moved_from.y)
        halted = ctx.halted_by
        if halted in AHEAD_HALTS:
            halted = None  # backing away from what is ahead
        stalled = halted == "stall" or ctx.proprio.blocked
        if stalled and moved >= cfg.blocked_min_mm:
            return self._at_face(ctx, now, "stall")
        if stalled:
            halted = None  # still spinning up backwards (~1 s): not the face
        if halted:
            return self._done(ctx, contact=False, why=f"reflex:{halted}")
        if cfg.stop_by == "pose":
            if pose.x <= geo.stop_mm:
                self.log.append("pose_stop")
                if self.plan[:1] == ["bump_out"]:
                    self.plan.pop(0)  # the fallback sits where it stopped: no walk out
                return self._next(ctx, now)
        elif pose.x <= geo.face_contact_mm:
            return self._at_face(ctx, now, "pose")  # odometry puts the rear at the face
        if moved >= cfg.blocked_min_mm and now - moving_ms >= cfg.blocked_window_ms:
            old = [s for s in leg.samples if s[0] <= now - cfg.blocked_window_ms]
            if old:
                _, ox, oy, _ = old[-1]
                if math.hypot(pose.x - ox, pose.y - oy) < cfg.blocked_progress_mm:
                    return self._at_face(ctx, now, "blocked")
        if pose.x < cfg.gate_x_mm and self._misaligned(leg, now):
            return self._realign(ctx, now)
        if now - (leg.first_ms if leg.first_ms is not None else moving_ms) > leg.ms:
            return self._done(ctx, contact=False, why="reverse")
        # the twist
        self.last_move_ms = now
        if now - moving_ms < cfg.reverse_straight_ms:
            leg.last_ms = now
            return ActOutput(twist=(cfg.reverse_vx_mm, cfg.ff_reverse_mrad))  # straight start
        if leg.kick_until is not None and now < leg.kick_until:
            leg.last_ms = now
            return ActOutput(twist=(cfg.reverse_vx_mm, 0))
        if not leg.kicked and now - moving_ms >= cfg.kick_after_ms and moved < cfg.kick_min_mm:
            # velstand can stand still on a steered reverse start: one straight reverse
            leg.kicked = True
            leg.kick_until = now + cfg.kick_ms
            self.log.append("reverse_kick")
            leg.last_ms = now
            return ActOutput(twist=(cfg.reverse_vx_mm, 0))
        return self._steer(ctx, now, leg, pose)

    def _misaligned(self, leg: _Leg, now: int) -> bool:
        """The alignment gate on the pose averaged over one gait cycle (the trunk sways ±20–25
        mm and a few degrees with every step)."""
        cfg = self.config
        window = [s for s in leg.samples if s[0] >= now - cfg.gate_window_ms]
        if not window:
            return False
        y = sum(s[2] for s in window) / len(window)
        h = math.atan2(sum(math.sin(s[3]) for s in window), sum(math.cos(s[3]) for s in window))
        return abs(y) > cfg.gate_mm or abs(h) * 1000 > cfg.gate_mrad

    def _realign(self, ctx: ActContext, now: int) -> ActOutput:
        """The gate failed: walk out along the axis and reverse again (at most
        :attr:`DockConfig.max_relines` times); once more, the whole approach again."""
        self.relines += 1
        self.log.append("gate")
        if self.relines <= self.config.max_relines:
            self.plan[:0] = ["pause", "realign", "pause", "reverse_in"]
            return self._next(ctx, now)
        if self.approaches >= self.config.max_approaches:
            return self._done(ctx, contact=False, why="gate")
        self.approaches += 1
        self.relines = 0
        self.lane_loops = 0
        self.sits = 0
        self.plan = ["pause", "walk_out", "pause", "uturn", "brief", *self._approach_plan()]
        return self._next(ctx, now)

    def _at_face(self, ctx: ActContext, now: int, why: str) -> ActOutput:
        """Blocked = arrived: the rear is at the face (the next phase walks out a little)."""
        self.log.append(f"face_stop:{why}")
        return self._next(ctx, now)

    def _phase_bump_out(self, ctx: ActContext, now: int) -> ActOutput:
        """After the face stop: walk out along the axis at 0.35 for 0.7 s, so the sit's ~70 mm
        slide back ends on the plates, not against the face."""
        cfg = self.config
        if self.leg is None:
            self._start_leg(now, float(cfg.geo().turn_mm), reverse=False, ms=cfg.bump_walk_ms)
        leg = self.leg
        assert leg is not None
        pose = self.pose(ctx)
        if pose is None:
            return self._done(ctx, contact=False, why="lost")
        held = self._unsure(ctx, now, leg)
        if held is not None:
            return held
        if not self._settled(ctx, now, leg):
            return ActOutput()
        if ctx.halted_by and ctx.halted_by not in AHEAD_HALTS:
            return self._done(ctx, contact=False, why=f"reflex:{ctx.halted_by}")
        if now - leg.started_ms >= leg.ms:
            return self._next(ctx, now)
        out = self._steer(ctx, now, leg, pose)
        twist = out.twist or (cfg.bump_walk_vx_mm, 0)
        return ActOutput(twist=(cfg.bump_walk_vx_mm, twist[1]))

    def _phase_realign(self, ctx: ActContext, now: int) -> ActOutput:
        """The alignment gate's walk-out: facing the room, out along the axis to the realign
        point, straightening (dock:measure: ≥ 1000 mm within 30 mm and 5°, or 1150 mm)."""
        cfg, geo = self.config, self.config.geo()
        if self.leg is None:
            self._start_leg(now, float(geo.turn_mm), reverse=False, ms=cfg.realign_timeout_ms)
        assert self.leg is not None

        def lined(p: DockPose) -> bool:
            straight = (
                abs(p.y) < cfg.realign_tol_mm and abs(wrap(p.h)) * 1000 < cfg.realign_tol_mrad
            )
            return (p.x >= geo.line_up_mm and straight) or p.x >= geo.turn_mm

        leg = self.leg
        if leg.settled and now - leg.started_ms >= leg.ms:
            return self._done(ctx, contact=False, why="realign")
        return self._drive_out(ctx, now, leg, lined)

    def _phase_walk_out(self, ctx: ActContext, now: int) -> ActOutput:
        """The whole approach again: facing the room, walk out to the turning point, then the
        U-turn back to face the tag."""
        cfg, geo = self.config, self.config.geo()
        if self.leg is None:
            self._start_leg(now, float(geo.turn_mm), reverse=False, ms=cfg.realign_timeout_ms)
        assert self.leg is not None
        leg = self.leg
        if leg.settled and now - leg.started_ms >= leg.ms:
            return self._done(ctx, contact=False, why="walk_out")
        return self._drive_out(ctx, now, leg, lambda p: p.x >= geo.turn_mm)

    def _drive_out(
        self, ctx: ActContext, now: int, leg: _Leg, arrived: Callable[[DockPose], bool]
    ) -> ActOutput:
        """A leg out along the axis, facing the room (no looks: the tag is behind)."""
        pose = self.pose(ctx)
        if pose is None:
            return self._done(ctx, contact=False, why="lost")
        held = self._unsure(ctx, now, leg)
        if held is not None:
            return held
        if not self._settled(ctx, now, leg):
            return ActOutput()
        if ctx.halted_by:
            return self._done(ctx, contact=False, why=f"reflex:{ctx.halted_by}")
        if arrived(pose):
            return self._next(ctx, now)
        return self._steer(ctx, now, leg, pose)

    # ── the end ──────────────────────────────────────────────────────────────

    def _phase_done(self, ctx: ActContext, now: int) -> ActOutput:
        return ActOutput(done=True)

    def _done(self, ctx: ActContext, *, contact: bool, why: str) -> ActOutput:
        if self.percept is None and self.memory is not None and hasattr(self.memory, "dock"):
            frame = None
            if contact and self.tracker.fix is not None and self.tracker.odom0 is not None:
                epoch = ctx.pose.epoch if ctx.pose is not None else 0
                frame = DockFrame(self.tracker.fix, self.tracker.odom0, epoch, self.config.geo())
            self.memory.dock = frame
        if self.percept is None:
            self.reason = why
            self.phase = "done"
            self.percept = {
                "kind": RESULT_KIND,
                "t_ms": ctx.t_ms,
                "act_id": self.act_id,
                "contact": bool(contact),
                "via": "pg" if contact else "none",
                "tries": max(self.tries, 1 if contact else self.tries),
                "seen_tag": self.seen_tag,
                "style": self.config.style,
            }
            return ActOutput(percepts=(self.percept,), done=True)
        return ActOutput(done=True)


def _head_home(ctx: ActContext) -> bool:
    """The measured head is at HOME (no posture tap: trust it)."""
    if ctx.pose is None:
        return True
    return L.head_at(ctx.pose.head, L.V5_HOME_HEAD, L.HOME)


def _guard_settled(ctx: ActContext) -> bool:
    """The body loop's guard counts the head settled at HOME (its rule 3), when it says."""
    return ctx.head_home_ms is None or ctx.head_home_ms >= HEAD_SETTLE_MS


def _with_nest(look: L.PlannedLook) -> L.PlannedLook:
    """A fix look always runs the ``nest`` task (and ``light``)."""
    if "nest" in look.tasks:
        return look
    return replace(look, tasks=("light", "nest"))


def make_factory(
    config: DockConfig | None = None, pg_source: PgSource | None = None, memory: Any = None
) -> Callable[[Any, int, Any], DockExec]:
    """The body loop's executor factory for ``duck.dock`` (``BodyConfig.executors``). The style
    and the turning side are the command's when it names them, else ``config``'s (the owner's
    pick: backin; the home's side). ``memory``: walkexec's WalkMemory, where a docking that closed
    PG leaves the dock's frame for the walk out."""
    base = config or DockConfig()

    def build(cmd: Any, t_ms: int, loop: Any) -> DockExec:
        return DockExec(cmd, t_ms, config=base, pg_source=pg_source, memory=memory)

    return build
