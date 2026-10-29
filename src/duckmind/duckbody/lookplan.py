"""lookplan: the look catalogue, the look rules, ``look_veto`` and the head-down approach
(physical-design spec §4.2–§4.3, §8.4; owner decision A2, 2026-09-27).

**The catalogue** (:data:`CATALOGUE`, spec §4.2). Every pose is one the A100 sweep validated
(rule 5): ``look.glance_down`` (half down, aimed ±0.54 rad), ``look.up``, ``look.seated`` and
Duck Core's ``look_around`` for the scan (``expression_sweep/looks.py``, ``content.py``), written
here as ``robot.head`` offsets from HOME in mrad (neck_pitch, head_pitch, head_yaw, head_roll;
``tests/test_duckbody_lookplan.py`` checks them against the clips). ``robot.look`` is not used
(it takes absolute targets, ``robotd.py``); aimed looks turn the head yaw only, within ±0.54 rad.

**The rules** (spec §4.2, the same functions on the board, beside the simulator and in HomeWorld):

1. no frame while the legs step: a capture waits until the walk's tail is over (0.45 s after the
   last non-zero twist) — :func:`legs_quiet`;
2. no frame until the head and body are still (joints < 0.2 rad/s, gyro < 0.3 rad/s for 150 ms:
   the posture tap's ``still``) — at the A100's capture instants the head was still in only
   84/576 first frames, so a look *waits* for stillness instead of a fixed time;
3. before a walk the head is back at HOME and has stayed there ≥ 0.3 s — :class:`HomeSettle`;
4. ``look_veto``: the predicted route (motion model + 70 mm tail + 150 mm margin) should lie on
   cells seen free recently — bright 10 min (owner 2026-09-28 「2. 10分鐘」), dim 5 min — else the
   segment is shortened to the part that was seen. **Owner 2026-09-28 「3. 一律放寬，即便跌倒才
   是真實模擬」**: floor no look has seen free never forbids a walk — an eye that has not learned
   its floor claims at most 35 cm, all of it ``unknown`` at its end — so the walk still takes one
   short step at the approach speed into it (:data:`UNSEEN_STEP_MS`, :data:`UNSEEN_VX_MM`) and
   physics decides what happens there. What a recent look *saw* still counts (owner 2026-09-28
   ~12:30 「1. 照建議」): cells just past a wall or a thing that ended the floor are *blocked*;
   the step ends before them, and with no room at all the walk is refused (``seen_blocked``).
   **The dark (owner 2026-09-28 ~12:30 「2. 放寬」, C2 reversed)** refuses nothing any more: the
   eye simply sees less there — dim frames claim 20 % less floor, dark and black frames none
   (duck-eye's floor line) and mark nothing here — so a dark walk goes on what was seen free
   while the light allowed it (the same 10 / 5 min, whatever the light is now) and steps slowly
   into the rest like any unseen floor (:func:`look_veto`, :class:`VetoGrid`); since the
   phase-4 landing (「一律放寬」) the dark no longer caps a segment either (the 3 s cap was
   pacing only). The dock's light alone in the dark still reads 很暗 (dark), not black (owner
   C1, a light-level fact of duck-eye's);
5. only validated poses; 6. seated looks use the seated HOME head; 7. personality changes dwell
   only — but **attention** decides whether the duck glances before a walk at all (owner
   2026-09-28 ~12:30 「保留鴨子偶爾不注意撞到東西的呆萌感」): the caller says where its attention
   is (``WalkCmd.attention``); a duck whose attention is not on its way glances less or not at
   all (:func:`need_glance`) and, not knowing what it has not looked at, walks there as asked —
   no slow unseen step (``look_veto(careful=False)``) — so it may bump into what it did not see,
   while what it did see still stops it. duck-body never skips a glance by chance.

**Head-down approach** (owner A2): the last approach toward a just-seen thing may walk head down
(the A100's half-down walking pose: 0/512 falls, −11 % speed) — but in phase 1 15 % of ducks
barely walked head down, so :class:`HeadDownWatch` raises the head when the walk makes no
progress (the walk stops, raises the head, lets it settle 0.3 s and walks on head up).

Pure and integer: mm, mrad, ms; trigonometry from :mod:`duckmind.duckbody.fixedmath` (Duck
Core's, copied). Nothing here reads a clock or a socket.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.fixedmath import SCALE, cos_mrad, isqrt, signed_mrad, sin_mrad
from duckmind.duckbody.protocol import ActContext, ActOutput, LookCmd, LookRequest

Offsets = tuple[int, int, int, int]  # mrad from HOME: neck_pitch, head_pitch, head_yaw, head_roll

HOME: Offsets = (0, 0, 0, 0)
V5_HOME_HEAD = (0.3490658503988659, 0.3490658503988659, 0.0, 0.0)  # robotd's head HOME (rad)
# Clip semantics → robot.head offsets: HOME + sign · envelope · value (content.head_targets):
# neck_pitch sign −1 × 0.25, head_pitch −1 × 0.35, head_yaw +1 × 0.6.
HALF_DOWN: Offsets = (-125, 175, 0, 0)  # looks.HALF_DOWN {np +0.5, hp −0.5}
UP: Offsets = (0, -350, 0, 0)  # looks.UP {hp +1}
AIM_YAW_MRAD = 540  # looks.AIM 0.9 × 0.6 rad: the widest aimed head turn
SCAN_LEFT_MRAD = 480  # look_around {hy +0.8}
SCAN_RIGHT_MRAD = -420  # look_around {hy −0.7}
HEAD_DOWN_WALK: Offsets = HALF_DOWN  # the A100's head-down walking pose (vx 0.30, 0/512 falls)

TAIL_MS = 450  # the legs keep stepping ~0.45 s after a stop (spec §4.2)
HOME_SETTLE_MS = 300  # rule 3
# measured head within this of the commanded offsets: live the servos sit ~0.10–0.13 rad off the
# command under gravity (phase 3, robotd --sim), so 50 mrad was never met and walks never started;
# 160 still tells the half-down look (−125/+175 mrad) from HOME (the guard's HEAD_HOME_TOL)
HOME_TOL_MRAD = 160
CAPTURE_TIMEOUT_MS = 2500  # waiting for stillness at one capture point
RESULT_TIMEOUT_MS = 1500  # waiting for eye.result after asking

LIGHT_RANK = {"bright": 3, "dim": 2, "dark": 1, "black": 0, "none": 0}


def yaw(offset_mrad: int) -> Offsets:
    """HOME with the head turned (clamped to the validated ±0.54 rad)."""
    return (0, 0, max(-AIM_YAW_MRAD, min(AIM_YAW_MRAD, offset_mrad)), 0)


def with_yaw(pose: Offsets, offset_mrad: int) -> Offsets:
    return (pose[0], pose[1], max(-AIM_YAW_MRAD, min(AIM_YAW_MRAD, offset_mrad)), pose[3])


def to_radians(offsets: Offsets) -> tuple[float, float, float, float]:
    """``robot.head`` offsets in rad (what :class:`~duckmind.duckbody.protocol.ActOutput.head`
    carries)."""
    return (offsets[0] / 1000.0, offsets[1] / 1000.0, offsets[2] / 1000.0, offsets[3] / 1000.0)


# ── the catalogue ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Key:
    """One keyframe: reach ``pose`` by ``t_ms`` (robotd slews; the clip holds it after)."""

    t_ms: int
    pose: Offsets
    capture: bool = False  # take a frame here once the head is still


@dataclass(frozen=True)
class LookSpec:
    kind: str  # eye-v1 LOOK_KINDS
    keys: tuple[Key, ...]
    tasks: tuple[str, ...]
    safe: bool = False  # never skipped (spec §4.6)
    seated_ok: bool = True

    @property
    def captures(self) -> int:
        return sum(1 for k in self.keys if k.capture)


CATALOGUE: dict[str, LookSpec] = {
    # the segment's end: head already HOME, legs stopped, one frame
    "end_look": LookSpec(
        "end_look", (Key(0, HOME, capture=True),), ("light", "floor", "nest", "things"), safe=True
    ),
    # before walking: half down, aimed along the route's 0.3 m point, one frame (+ a ToF frame)
    "glance_down": LookSpec(
        "glance_down",
        (Key(0, HOME), Key(300, HALF_DOWN, capture=True), Key(600, HOME)),
        ("light", "floor", "things"),
        safe=True,
        seated_ok=False,
    ),
    # the validated look_around: left, right, three frames (start, each pause)
    "scan": LookSpec(
        "scan",
        (
            Key(0, HOME, capture=True),
            Key(700, yaw(SCAN_LEFT_MRAD), capture=True),
            Key(2300, yaw(SCAN_RIGHT_MRAD), capture=True),
            Key(3900, HOME),
        ),
        ("light", "floor", "nest", "things"),
    ),
    "look_up": LookSpec(
        "look_up", (Key(0, HOME), Key(300, UP, capture=True), Key(1100, HOME)), ("light",)
    ),
    # aimed at a bearing (the head yaw only); the executor fills the yaw
    "fixate": LookSpec(
        "fixate", (Key(0, HOME), Key(300, HOME, capture=True)), ("light", "nest", "things")
    ),
    "seek_nest": LookSpec(
        "seek_nest", (Key(0, HOME), Key(300, HOME, capture=True)), ("light", "nest")
    ),
    "idle_glance": LookSpec(
        "idle_glance",
        (Key(0, HOME), Key(300, HOME, capture=True), Key(900, HOME)),
        ("light", "floor"),
    ),
}


# ── rules 1–3 ────────────────────────────────────────────────────────────────


def legs_quiet(now_ms: int, last_move_ms: int | None) -> bool:
    """Rule 1: the walk's tail is over (no non-zero twist for :data:`TAIL_MS`)."""
    return last_move_ms is None or now_ms - last_move_ms >= TAIL_MS


def head_at(
    measured: Sequence[float], home: Sequence[float], target: Offsets, tol_mrad: int = HOME_TOL_MRAD
) -> bool:
    """The measured head joints (rad) are within ``tol_mrad`` of HOME + ``target``, pitch and
    yaw joints (roll is not commanded)."""
    for i in range(3):
        want = home[i] * 1000.0 + target[i]
        if abs(measured[i] * 1000.0 - want) > tol_mrad:
            return False
    return True


@dataclass
class HomeSettle:
    """Rule 3: the head has been at HOME for ≥ :data:`HOME_SETTLE_MS` (feed every tick)."""

    since_ms: int | None = None

    def step(self, now_ms: int, at_home: bool) -> bool:
        if not at_home:
            self.since_ms = None
            return False
        if self.since_ms is None:
            self.since_ms = now_ms
        return now_ms - self.since_ms >= HOME_SETTLE_MS


GLANCE_FRESH_MS = 60_000  # the near band counts as seen for 60 s (bright or dim)
GLANCE_TURN_MRAD = 349  # a heading change of more than 20° is a new direction


def need_glance(
    *,
    now_ms: int,
    heading_mrad: int,
    last_glance_ms: int | None,
    last_glance_heading_mrad: int | None,
    light: str,
    first_leg: bool = False,
    after_arc: bool = False,
    set_down: bool = False,
    after_contact: bool = False,
    after_unseen: bool = False,
    attention: str = "on",
) -> bool:
    """Spec §4.2: glance down before walking when the near band (0.2–0.4 m) of the new direction
    has not been seen within 60 s — first leg from the nest, after an arc, after being put down,
    after a bump or stumble, after a step into floor no look had seen (``after_unseen``: one
    step, one look — and each glanced walk teaches the eye its floor, owner D1), or the heading
    moved > 20°. Only when the camera can see (bright, dim) — or when the light is not known
    (``none``: no reading yet, or too old), because the glance is how it is learnt.

    ``attention`` is the caller's (owner 2026-09-28 ~12:30 「呆萌感」; ``WalkCmd.attention``):
    ``on`` (the default, Duck Core v4) — every rule above; ``low`` (excited or playing, sleepy,
    a hatchling, low wit, too bold to check) — only what would catch anyone's attention: the
    first leg from the nest, being put down, a bump or stumble; ``elsewhere`` (its gaze is on a
    toy, a person, a sound) — never. Nothing here is drawn by chance: an inattentive duck bumps
    into things only where it really did not look."""
    if attention == "elsewhere":
        return False
    if light != "none" and LIGHT_RANK.get(light, 0) < LIGHT_RANK["dim"]:
        return False
    if first_leg or set_down or after_contact:
        return True
    if attention == "low":
        return False
    if after_arc or after_unseen:
        return True
    if last_glance_ms is None or now_ms - last_glance_ms > GLANCE_FRESH_MS:
        return True
    if last_glance_heading_mrad is None:
        return True
    turn = (heading_mrad - last_glance_heading_mrad + 3141) % 6283 - 3141
    return abs(turn) > GLANCE_TURN_MRAD


# ── the motion model (route prediction for look_veto) ─────────────────────────

# Trunk path speed (mm/s) per commanded vx (mm/s): phase 1 gait (0.30/0.35/0.40, robotd --sim,
# 20 walks each) and phase 3 A3 headless medians, simulator servos (0.45/0.50/0.55;
# docs/validation/evidence/phase3/a3_walk_speed.json).
SPEED_TABLE: tuple[tuple[int, int], ...] = (
    (300, 163),
    (350, 178),
    (400, 195),
    (450, 215),
    (500, 239),
    (550, 317),
)
# Yaw rate (mrad/s) = bias + gain × vyaw (mrad/s): phase 1 vx 0.40 (−10.0 °/s, 34.2 °/s per rad/s).
YAW_BIAS_MRAD_S = -175
YAW_GAIN_MILLI = 597
ROUTE_TAIL_MM = 70
ROUTE_MARGIN_MM = 150
BODY_HALF_MM = 100


def speed_mm_s(vx_mm: int, table: Sequence[tuple[int, int]] = SPEED_TABLE) -> int:
    """Expected path speed for a forward command (piecewise linear, clamped at the ends)."""
    v = abs(vx_mm)
    if v <= table[0][0]:
        return table[0][1] * v // table[0][0] if v < 300 else table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:], strict=False):
        if v <= x1:
            return y0 + (y1 - y0) * (v - x0) // (x1 - x0)
    return table[-1][1]


def yaw_rate_mrad_s(vyaw_mrad: int) -> int:
    return YAW_BIAS_MRAD_S + YAW_GAIN_MILLI * vyaw_mrad // 1000


@dataclass(frozen=True)
class Pose2:
    """A pose in the odometry frame (mm, mrad)."""

    x_mm: int
    y_mm: int
    h_mrad: int


def predict_route(
    start: Pose2, vx_mm: int, vyaw_mrad: int, ms: int, step_ms: int = 100
) -> list[Pose2]:
    """Where a segment is expected to go (every ``step_ms``), from the motion model. The last
    pose includes the 70 mm tail and the 150 mm margin straight on."""
    v = speed_mm_s(vx_mm)
    w = yaw_rate_mrad_s(vyaw_mrad)
    x, y, h = start.x_mm * 1000, start.y_mm * 1000, start.h_mrad  # µm for the integration
    out = [start]
    t = 0
    while t < ms:
        dt = min(step_ms, ms - t)
        mid = h + w * dt // 2000
        x += v * dt * cos_mrad(mid) // SCALE
        y += v * dt * sin_mrad(mid) // SCALE
        h += w * dt // 1000
        t += dt
        out.append(Pose2(x // 1000, y // 1000, h))
    extra = ROUTE_TAIL_MM + ROUTE_MARGIN_MM
    last = out[-1]
    out.append(
        Pose2(
            last.x_mm + extra * cos_mrad(last.h_mrad) // SCALE,
            last.y_mm + extra * sin_mrad(last.h_mrad) // SCALE,
            last.h_mrad,
        )
    )
    return out


# ── the free-cell memory ──────────────────────────────────────────────────────

CELL_MM = 100
# rule 4's N: cells seen free in bright stay trusted 10 min (owner 2026-09-28 「2. 10分鐘」 —
# not the 5 min once proposed), in dim 5 min — whatever the light is when the duck walks there
# (owner 2026-09-28 ~12:30 「2. 放寬」: C2's "seen lit within 30 min" is gone with the dark rule)
FRESH_MS = {"bright": 600_000, "dim": 300_000}
MIN_SEGMENT_MS = 800  # a seen stretch shorter than this is not kept as the segment …
# … the walk then takes one short step into the floor no look has seen, at the approach speed
# (owner 2026-09-28 「3. 一律放寬」: unseen floor may shorten or slow a walk, never forbid it; and
# 「2. 放寬」 ~12:30: nor in the dark) — but never into cells a recent look saw end the floor
UNSEEN_STEP_MS = 2000
UNSEEN_VX_MM = 400
BLOCKING_ENDS = ("wall", "thing")  # what a look saw end the floor; "unknown" ends nothing
# The duck's own footprint and the band just ahead that no look can cover edge to edge: the
# camera's 37.4° view is only ±0.10 m wide at 0.3 m, narrower than the body corridor nearer in.
# That band belongs to the reflexes (ToF, touch) and the glance_down's near band.
SELF_MM = 300
REVOKE_MM = 300  # a newer look forgets free cells this far past where it now sees the floor end


@dataclass
class Cell:
    seen_ms: int = -(10**12)  # last seen free (any light that can see)
    bright_ms: int = -(10**12)  # last seen free in bright
    level: str = "none"  # the light at ``seen_ms``
    blocked_ms: int = -(10**12)  # last seen just past a wall or a thing that ended the floor
    blocked_level: str = "none"


@dataclass
class VetoGrid:
    """duck-body's short-term free-cell memory in its own odometry frame (spec §4.2 rule 4),
    separate from Duck Core's space memory. A cell is *free* where a look saw floor, *blocked*
    where a newer look saw a wall or a thing end the floor, and *unseen* otherwise (never seen,
    stale, or forgotten when a look saw the floor end there at nothing it could name). ``epoch``
    is the posture tap's odometry epoch: a new one (lifted, robotd restarted) forgets everything
    (:meth:`sync`)."""

    cell_mm: int = CELL_MM
    epoch: int = 0
    cells: dict[tuple[int, int], Cell] = field(default_factory=dict)

    def sync(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.cells.clear()
            self.epoch = epoch

    def key(self, x_mm: int, y_mm: int) -> tuple[int, int]:
        return (x_mm // self.cell_mm, y_mm // self.cell_mm)

    def mark(self, x_mm: int, y_mm: int, now_ms: int, level: str) -> None:
        if LIGHT_RANK.get(level, 0) < LIGHT_RANK["dim"]:
            return  # dark and black frames prove nothing about the floor
        cell = self.cells.setdefault(self.key(x_mm, y_mm), Cell())
        cell.seen_ms = now_ms
        cell.level = level
        if level == "bright":
            cell.bright_ms = now_ms

    def mark_blocked(self, x_mm: int, y_mm: int, now_ms: int, level: str) -> None:
        if LIGHT_RANK.get(level, 0) < LIGHT_RANK["dim"]:
            return
        cell = self.cells.setdefault(self.key(x_mm, y_mm), Cell())
        cell.blocked_ms = now_ms
        cell.blocked_level = level

    def mark_walked(self, poses: Iterable[Pose2], now_ms: int, level: str = "bright") -> None:
        """Ground the body actually covered without contact is free (the feet felt it),
        recorded as seen in ``level`` — so ground walked in the dark never counts as seen in
        bright, and never refreshes what was seen there before (a dark walk marks nothing)."""
        if LIGHT_RANK.get(level, 0) < LIGHT_RANK["dim"]:
            return
        for p in poses:
            for dy in (-BODY_HALF_MM, 0, BODY_HALF_MM):
                x = p.x_mm - dy * sin_mrad(p.h_mrad) // SCALE
                y = p.y_mm + dy * cos_mrad(p.h_mrad) // SCALE
                cell = self.cells.setdefault(self.key(x, y), Cell())
                cell.seen_ms = now_ms
                cell.level = max(cell.level, level, key=lambda v: LIGHT_RANK.get(v, 0))
                if level == "bright":
                    cell.bright_ms = now_ms

    def mark_view(
        self,
        origin: Pose2,
        sectors: Sequence[tuple[int, int, int] | tuple[int, int, int, str]],
        now_ms: int,
        level: str,
        near_mm: int = 200,
    ) -> None:
        """A look's free ground: ``sectors`` = (bearing mrad in the trunk frame, half-width mrad,
        free_mm[, end]) as FL-v1 reports them (:func:`view_marks`); every cell from ``near_mm``
        to ``free_mm`` along the sector is marked free (dim frames: the eye already took 20 %
        off the distance). A look that can see also takes back what it now finds *not* free:
        the cells just past where each sector's free ground ends (up to :data:`REVOKE_MM` beyond
        it) are marked *blocked* when the look saw a wall or a thing end it
        (:data:`BLOCKING_ENDS`), else forgotten (unseen) — unless this same look saw them free
        along another sector: a thing put down since an older look must not stay "seen free"."""
        if LIGHT_RANK.get(level, 0) < LIGHT_RANK["dim"]:
            return  # dark and black frames prove nothing about the floor, either way
        free_keys: set[tuple[int, int]] = set()
        ended: list[tuple[int, int, bool]] = []
        for sector in sectors:
            b, half, free = sector[0], sector[1], sector[2]
            blocking = len(sector) > 3 and sector[3] in BLOCKING_ENDS  # type: ignore[misc]
            for db in (-half, 0, half):
                h = origin.h_mrad + b + db * 2 // 3
                c, s = cos_mrad(h), sin_mrad(h)
                r = near_mm
                while r <= free:
                    x, y = origin.x_mm + r * c // SCALE, origin.y_mm + r * s // SCALE
                    self.mark(x, y, now_ms, level)
                    free_keys.add(self.key(x, y))
                    r += self.cell_mm // 2
                r = max(free, near_mm) + self.cell_mm // 2
                while r <= max(free, near_mm) + REVOKE_MM:
                    x, y = origin.x_mm + r * c // SCALE, origin.y_mm + r * s // SCALE
                    ended.append((x, y, blocking))
                    r += self.cell_mm // 2
        for x, y, blocking in ended:
            key = self.key(x, y)
            if key in free_keys:
                continue
            if blocking:
                self.mark_blocked(x, y, now_ms, level)
            else:
                self.cells.pop(key, None)

    def blocked(self, x_mm: int, y_mm: int, now_ms: int) -> bool:
        """A recent look saw a wall or a thing here, and no newer look saw it free."""
        cell = self.cells.get(self.key(x_mm, y_mm))
        if cell is None or cell.blocked_ms <= cell.seen_ms:
            return False
        return now_ms - cell.blocked_ms <= FRESH_MS.get(cell.blocked_level, 0)

    def fresh(self, x_mm: int, y_mm: int, now_ms: int, light: str = "bright") -> bool:
        """Seen free recently: within :data:`FRESH_MS` of the light it was seen in. ``light`` —
        the light now — no longer matters (owner 2026-09-28 ~12:30 「2. 放寬」): what the duck
        saw free while it could see stays known for as long in the dark; a dark frame adds
        nothing (:meth:`mark`)."""
        cell = self.cells.get(self.key(x_mm, y_mm))
        if cell is None or cell.blocked_ms > cell.seen_ms:
            return False
        return now_ms - cell.seen_ms <= FRESH_MS.get(cell.level, 0)


@dataclass(frozen=True)
class Veto:
    """``look_veto``'s answer: walk ``max_ms`` (maybe shortened) no faster than ``vx_cap_mm``
    (``None``: the walk's own speed), or not at all (``ok`` False). ``why``: ``clear``,
    ``shortened``; ``unseen`` — a short step into floor no look has seen free
    (in any light since owner 2026-09-28 ~12:30 「2. 放寬」); ``seen_blocked`` — a wall or a
    thing a recent look saw end the floor is right there (refused: turn another way), the only
    refusal left."""

    ok: bool
    max_ms: int
    why: str
    vx_cap_mm: int | None = None


def look_veto(
    grid: VetoGrid,
    start: Pose2,
    vx_mm: int,
    vyaw_mrad: int,
    max_ms: int,
    now_ms: int,
    light: str,
    *,
    step_ms: int = 100,
    seen_blocks: bool = True,
    careful: bool = True,
) -> Veto:
    """Rule 4. The whole predicted route (body width, tail and margin) fresh: the segment as
    asked. Otherwise the longest fresh prefix is kept when it is ≥ :data:`MIN_SEGMENT_MS`; when
    it is shorter, the walk still goes — one step of :data:`UNSEEN_STEP_MS` at
    :data:`UNSEEN_VX_MM` into the unseen floor (owner 2026-09-28 「3. 一律放寬，即便跌倒才是
    真實模擬」: an eye that has not learned its floor — it claims 35 cm at most — or floor no look
    has seen never stops the duck; the reflexes and physics decide what the step meets). What a
    recent look *saw* still counts (``seen_blocks``): the step ends before cells where the floor
    ended at a wall or a thing, and with no room before them the walk is refused
    (``seen_blocked``). Dark, black and an unknown light (owner 2026-09-28 ~12:30 「2. 放寬」,
    C2 reversed; the 3 s dark segment cap went at the phase-4 landing, 「一律放寬」) neither
    refuse nor shorten: the route counts as seen only where a look saw it free while it could
    see (the same :data:`FRESH_MS`) — the rest is an unseen step like anywhere else, and
    physics decides whether the duck bumps or falls. ``light`` is the light now (it no longer
    changes the answer: what was seen free stays known in the dark, :meth:`VetoGrid.fresh`).

    ``careful`` False — a duck whose attention is not on its way (``WalkCmd.attention`` ``low``
    or ``elsewhere``; owner 2026-09-28 ~12:30 「呆萌感」): it does not know which floor it has
    not looked at, so it neither stops where the seen floor ends nor slows into the unseen: it
    walks as asked, and only what a recent look *saw* (a wall, a thing) still stops it before
    it — what it did not look at, it may bump into (``unseen``, no speed cap)."""
    cap = max_ms  # (no light caps a segment: owner 「2. 放寬」 and 「一律放寬」)
    route = predict_route(start, vx_mm, vyaw_mrad, cap, step_ms)
    extra = ROUTE_TAIL_MM + ROUTE_MARGIN_MM
    last = len(route) - 2  # the last real pose (the final entry is tail + margin)
    fresh_until = clear_until = 0  # seen free / no seen obstacle up to here (ms)
    fresh_open = clear_open = True
    for i, p in enumerate(route[1:], start=1):
        t = cap if i >= last else min(cap, i * step_ms)
        # the body corridor at this pose, plus the tail + margin ahead of it; the ground the
        # duck stands on and its toes' reach (SELF_MM) is where it already is
        ahead = Pose2(
            p.x_mm + extra * cos_mrad(p.h_mrad) // SCALE,
            p.y_mm + extra * sin_mrad(p.h_mrad) // SCALE,
            p.h_mrad,
        )
        checks = [q for q in (p, ahead) if distance_mm(q, start) > SELF_MM]
        if clear_open and seen_blocks and any(_corridor_blocked(grid, q, now_ms) for q in checks):
            clear_open = False
        if fresh_open and not all(_corridor_fresh(grid, q, now_ms, light) for q in checks):
            fresh_open = False
        if fresh_open:
            fresh_until = t
        if clear_open:
            clear_until = t
        if i >= last or not (fresh_open or clear_open):
            break
    if fresh_until >= cap:
        return Veto(True, cap, "clear")
    if not careful:
        # as asked, up to what it remembers seeing in the way
        if clear_until >= MIN_SEGMENT_MS:
            return Veto(True, min(cap, clear_until), "unseen")
        return Veto(False, 0, "seen_blocked")
    if fresh_until >= MIN_SEGMENT_MS:
        return Veto(True, fresh_until, "shortened")
    step = min(cap, max(fresh_until, UNSEEN_STEP_MS), clear_until)
    if step >= MIN_SEGMENT_MS:
        return Veto(True, step, "unseen", UNSEEN_VX_MM)
    return Veto(False, 0, "seen_blocked")


def _corridor_fresh(grid: VetoGrid, p: Pose2, now_ms: int, light: str) -> bool:
    c, s = cos_mrad(p.h_mrad), sin_mrad(p.h_mrad)
    for dy in (-BODY_HALF_MM, 0, BODY_HALF_MM):
        x = p.x_mm - dy * s // SCALE
        y = p.y_mm + dy * c // SCALE
        if not grid.fresh(x, y, now_ms, light):
            return False
    return True


def _corridor_blocked(grid: VetoGrid, p: Pose2, now_ms: int) -> bool:
    c, s = cos_mrad(p.h_mrad), sin_mrad(p.h_mrad)
    for dy in (-BODY_HALF_MM, 0, BODY_HALF_MM):
        if grid.blocked(p.x_mm - dy * s // SCALE, p.y_mm + dy * c // SCALE, now_ms):
            return True
    return False


# ── the head-down approach (owner A2) ─────────────────────────────────────────


@dataclass(frozen=True)
class HeadDownConfig:
    """When a head-down walk counts as making no progress (phase 1: 15 % of ducks walked
    < 10 cm in 6 s head down; walking ducks cover ≥ 0.10 m/s)."""

    grace_ms: int = 1200  # the first steps after the start are slow for everyone
    window_ms: int = 1000
    min_progress_milli: int = 350  # < 35 % of the model's expected advance over the window…
    min_progress_mm: int = 40  # …or under 4 cm, whichever is larger


@dataclass
class HeadDownWatch:
    """Watches a head-down walk's progress (odometry) and says when to raise the head.

    Progress is the displacement over the last ``window_ms`` (not the advance along the start
    heading: every walk curves ~10°/s, and a stuck duck steps on the spot). Feed every tick with
    the odometry pose; :meth:`step` returns ``True`` once — the moment the walk should stop, raise
    the head, settle and go on head up."""

    vx_mm: int
    start_ms: int
    config: HeadDownConfig = HeadDownConfig()
    samples: list[tuple[int, int, int]] = field(default_factory=list)  # (t_ms, x_mm, y_mm)
    fired_ms: int | None = None
    last_advance_mm: int | None = None

    def step(self, now_ms: int, start: Pose2, pose: Pose2) -> bool:
        if self.fired_ms is not None:
            return False
        self.samples.append((now_ms, pose.x_mm, pose.y_mm))
        cfg = self.config
        if now_ms - self.start_ms < cfg.grace_ms + cfg.window_ms:
            return False
        old = [s for s in self.samples if s[0] <= now_ms - cfg.window_ms]
        if not old:
            return False
        _, ox, oy = old[-1]
        advanced = isqrt((pose.x_mm - ox) ** 2 + (pose.y_mm - oy) ** 2)
        self.last_advance_mm = advanced
        expected = speed_mm_s(self.vx_mm) * cfg.window_ms // 1000
        need = max(cfg.min_progress_mm, expected * cfg.min_progress_milli // 1000)
        if advanced < need:
            self.fired_ms = now_ms
            return True
        cut = now_ms - 2 * cfg.window_ms
        self.samples = [s for s in self.samples if s[0] >= cut]
        return False


# ── looks as requests, and running one look ───────────────────────────────────


@dataclass(frozen=True)
class PlannedLook(LookRequest):
    """A :class:`~duckmind.duckbody.protocol.LookRequest` that also says which catalogue look it
    is and what the eye may use (``eye.look``'s ``kind``, ``mode`` and integer ``hints``). The
    body server reads them with ``getattr`` so a plain LookRequest still works."""

    kind: str = "end_look"
    mode: str = "awake"
    hints: Mapping[str, Any] = field(default_factory=dict)


def request(
    look_id: str,
    spec: LookSpec,
    *,
    not_before_ns: int = 0,
    mode: str = "awake",
    hints: Mapping[str, Any] | None = None,
    tasks: Sequence[str] | None = None,
) -> PlannedLook:
    return PlannedLook(
        look_id=look_id,
        tasks=tuple(tasks) if tasks is not None else spec.tasks,
        not_before_ns=not_before_ns,
        kind=spec.kind,
        mode=mode,
        hints=dict(hints or {}),
    )


@dataclass
class LookRun:
    """Runs one catalogue look as a pure step machine: head keyframes, a capture at each capture
    key once the head is still (rule 2) and the legs quiet (rule 1), the eye's results (retakes
    as the eye asks), then the head home. :meth:`step` returns (head offsets to send, looks to
    request, finished)."""

    spec: LookSpec
    look_id: str
    start_ms: int
    yaw_mrad: int = 0  # an aimed look (glance_down along the route, fixate, seek_nest)
    mode: str = "awake"
    hints: Mapping[str, Any] = field(default_factory=dict)
    seated: bool = False
    key_index: int = 0
    key_since_ms: int | None = None
    asked_ms: int | None = None
    attempt: int = 0
    retake_at_ms: int | None = None
    results: list[Mapping[str, Any]] = field(default_factory=list)
    heads: list[int] = field(default_factory=list)  # head yaw (mrad) at each capture
    missed: int = 0
    finished: bool = False

    def _pose(self, key: Key) -> Offsets:
        pose = key.pose
        if self.seated:
            pose = (0, 0, pose[2], 0)  # rule 6: the seated HOME head, yaw only
        if self.yaw_mrad and pose[2] == 0 and key.capture:
            pose = with_yaw(pose, self.yaw_mrad)
        return pose

    def _capture_id(self) -> str:
        n = sum(1 for k in self.spec.keys[: self.key_index] if k.capture)
        return self.look_id if self.spec.captures == 1 else f"{self.look_id}.{n}"

    def step(
        self,
        now_ms: int,
        *,
        still: bool,
        legs_quiet: bool,
        results: Iterable[Mapping[str, Any]] = (),
    ) -> tuple[Offsets, list[PlannedLook], bool]:
        if self.finished:
            return HOME, [], True
        keys = self.spec.keys
        key = keys[self.key_index]
        if self.key_since_ms is None:
            self.key_since_ms = now_ms
        asks: list[PlannedLook] = []
        wanted = self._capture_id()
        for result in results:
            if result.get("look_id") != wanted:
                continue
            status = result.get("status")
            if status == "retake" and self.attempt < 2:
                self.attempt += 1
                self.asked_ms = None
                self.retake_at_ms = now_ms + int(result.get("retake_in_ms") or 150)
                continue
            self.results.append(result)
            self.heads.append(self._pose(key)[2])
            self.asked_ms = None
            self._advance(now_ms)
            return self._pose(keys[min(self.key_index, len(keys) - 1)]), [], self.finished
        if key.capture:
            reached = now_ms - self.start_ms >= key.t_ms
            ready = (
                reached
                and still
                and legs_quiet
                and (self.retake_at_ms is None or now_ms >= self.retake_at_ms)
            )
            if self.asked_ms is None and ready:
                self.asked_ms = now_ms
                look = request(wanted, self.spec, mode=self.mode, hints=self.hints)
                asks.append(look)
            elif self.asked_ms is not None and now_ms - self.asked_ms > RESULT_TIMEOUT_MS:
                self.missed += 1
                self.asked_ms = None
                self._advance(now_ms)
            elif (
                self.asked_ms is None
                and reached
                and now_ms - self.key_since_ms > CAPTURE_TIMEOUT_MS
            ):
                self.missed += 1  # never still enough: give the frame up
                self._advance(now_ms)
            return self._pose(key), asks, self.finished
        if now_ms - self.start_ms >= key.t_ms:
            self._advance(now_ms)
        return self._pose(key), asks, self.finished

    def _advance(self, now_ms: int) -> None:
        self.key_index += 1
        self.key_since_ms = now_ms
        self.attempt = 0
        self.retake_at_ms = None
        if self.key_index >= len(self.spec.keys):
            self.key_index = len(self.spec.keys) - 1
            self.finished = True


# ── reading an eye result ─────────────────────────────────────────────────────

SECTORS = 5
CAMERA_HALF_FOV_MRAD = 326  # 37.4° / 2 of the family camera (the eye's own K decides; §4.1)


def light_of(result: Mapping[str, Any] | None) -> str:
    """The light level a look saw (``none`` when the eye gave no frame: treated as black) — the
    darker of the eye's level and the frame's own (``raw``): walking, the fold and the free-cell
    memory never go by a level brighter than the picture itself shows (the eye's level lags a
    change by a look or two; phase-3 verify: 23/30 camera-only departures in pitch dark)."""
    if not result or result.get("status") not in ("ok", "moved"):
        return "none"
    light = (result.get("results") or {}).get("light") or {}
    level = light.get("level")
    if level not in LIGHT_RANK or level == "none":
        return "none"
    raw = light.get("raw")
    if raw in LIGHT_RANK and raw != "none" and LIGHT_RANK[raw] < LIGHT_RANK[level]:
        return str(raw)
    return str(level)


def floor_fold(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A look as percept-v1's folded look (``walk.done.pre`` / ``end``): ``look_id``, ``light``,
    ``free_mm[5]``, ``ends[5]`` (+ ``trip``, ``drop``). ``None`` when the eye gave no frame."""
    if not result or result.get("status") not in ("ok", "moved"):
        return None
    floor = (result.get("results") or {}).get("floor") or {}
    free = list(floor.get("free_mm") or [0] * SECTORS)[:SECTORS]
    ends = list(floor.get("ends") or ["unknown"] * SECTORS)[:SECTORS]
    free += [0] * (SECTORS - len(free))
    ends += ["unknown"] * (SECTORS - len(ends))
    fold: dict[str, Any] = {
        "look_id": str(result.get("look_id")),
        "light": light_of(result),
        "free_mm": [max(0, int(v)) for v in free],
        "ends": [
            e if e in ("wall", "thing", "drop", "trip", "unknown") else "unknown" for e in ends
        ],
    }
    trip = floor.get("trip")
    if isinstance(trip, Mapping) and "mm" in trip:
        fold["trip"] = {"mm": max(0, int(trip["mm"])), "mrad": int(trip.get("mrad", 0))}
    return fold


def view_sectors(
    result: Mapping[str, Any] | None, head_yaw_mrad: int = 0
) -> list[tuple[int, int, int]]:
    """(bearing, half-width, free_mm) per floor sector in the trunk frame: the eye's own
    ``sectors`` when it gives them, else five equal slices of the camera's view turned by the
    head yaw."""
    if not result:
        return []
    floor = (result.get("results") or {}).get("floor") or {}
    sectors = floor.get("sectors")
    if isinstance(sectors, list) and sectors:
        out = []
        for s in sectors:
            if isinstance(s, Mapping) and "free_mm" in s:
                out.append(
                    (int(s.get("b_mrad", 0)), int(s.get("w_mrad", 130)) // 2, int(s["free_mm"]))
                )
        return out
    free = floor.get("free_mm") or []
    width = 2 * CAMERA_HALF_FOV_MRAD // SECTORS
    out = []
    for i, value in enumerate(list(free)[:SECTORS]):
        # sector 0 is the image's left (+ bearing)
        b = head_yaw_mrad + CAMERA_HALF_FOV_MRAD - width // 2 - i * width
        out.append((b, width // 2, int(value)))
    return out


def view_marks(
    result: Mapping[str, Any] | None, head_yaw_mrad: int = 0
) -> list[tuple[int, int, int, str]]:
    """:func:`view_sectors` with what ended each sector's free floor (``wall``, ``thing`` or
    ``unknown``), for :meth:`VetoGrid.mark_view`: a wall or a thing blocks the cells past it, an
    unknown end (nothing named; every end of an eye that has not learned its floor) does not."""
    if not result:
        return []
    floor = (result.get("results") or {}).get("floor") or {}
    sectors = floor.get("sectors")
    if isinstance(sectors, list) and sectors:
        ends = [
            str(s.get("end", "unknown"))
            for s in sectors
            if isinstance(s, Mapping) and "free_mm" in s
        ]
    else:
        ends = [str(e) for e in list(floor.get("ends") or [])[:SECTORS]]
    marks = view_sectors(result, head_yaw_mrad)
    ends += ["unknown"] * (len(marks) - len(ends))
    return [(b, half, free, end) for (b, half, free), end in zip(marks, ends, strict=False)]


YAW_SURE_MRAD = 300  # duck-eye's yaw_amb_mrad over this: the tag's facing may be the mirror pose


def nest_fix(result: Mapping[str, Any] | None) -> dict[str, int | str | bool] | None:
    """The dock as a look saw it, from duck-eye's ``nest`` task (:mod:`duckmind.eye.nesttag`,
    ``results.nest = {ok, level, anchor, tags, cue, …}``); ``None`` when not seen.

    * the tag ``anchor`` → ``{seen, src: "tag", tag (17 the face, 18 the plate's edge), r_mm,
      b_mrad, e_mrad?, conf_ppm, yaw_mrad, yaw_sure}``. ``yaw_mrad`` is the duck's heading off
      square-on (0 facing the tag squarely, left +): the eye gives the heading of the tag's face
      normal in the duck's level frame (π when square), so yaw = π − normal. ``yaw_sure`` is
      False when the eye says the mirror pose fits almost as well (``yaw_amb_mrad`` over
      :data:`YAW_SURE_MRAD`: 38/79 looks at 2.5 m) — the range and bearing still hold;
    * else a ``cue`` (the status light in the dark, the cyan panel by day) → ``{seen, src: "cue",
      b_mrad, e_mrad?, conf_ppm}``: a direction only (owner C5)."""
    if not result or result.get("status") not in ("ok", "moved"):
        return None
    nest = (result.get("results") or {}).get("nest") or {}
    if not isinstance(nest, Mapping):
        return None
    anchor = nest.get("anchor")
    if (
        isinstance(anchor, Mapping)
        and isinstance(anchor.get("r_mm"), int)
        and isinstance(anchor.get("b_mrad"), int)
    ):
        out: dict[str, int | str | bool] = {
            "seen": True,
            "src": "tag",
            "tag": int(anchor.get("id", 17)),
            "r_mm": int(anchor["r_mm"]),
            "b_mrad": int(anchor["b_mrad"]),
            "conf_ppm": int(anchor.get("conf_ppm", 0) or 0),
        }
        if isinstance(anchor.get("e_mrad"), int):
            out["e_mrad"] = int(anchor["e_mrad"])
        normal = anchor.get("yaw_mrad")
        if isinstance(normal, int):
            out["yaw_mrad"] = signed_mrad(3142 - normal)
            out["yaw_sure"] = int(anchor.get("yaw_amb_mrad", 0) or 0) <= YAW_SURE_MRAD
        return out
    cue = nest.get("cue")
    if isinstance(cue, Mapping) and isinstance(cue.get("b_mrad"), int):
        out = {
            "seen": True,
            "src": "cue",
            "b_mrad": int(cue["b_mrad"]),
            "conf_ppm": int(cue.get("conf_ppm", 0) or 0),
        }
        if isinstance(cue.get("e_mrad"), int):
            out["e_mrad"] = int(cue["e_mrad"])
        return out
    return None


def distance_mm(a: Pose2, b: Pose2) -> int:
    return isqrt((a.x_mm - b.x_mm) ** 2 + (a.y_mm - b.y_mm) ** 2)


# ── duck.look: one look act (spec §7.1), seated or standing, only when still ────────

LOOK_OF_KIND = {
    "scan": "scan",
    "look_up": "look_up",
    "seek_nest": "seek_nest",
    "fixate": "fixate",
    "idle": "idle_glance",
    "glance_down": "glance_down",
}
TASKS_OF_WANT = {
    "nest": ("light", "nest"),
    "thing": ("light", "floor", "things"),
    "duck": ("light", "duck"),
}


@dataclass
class LookExec:
    """``duck.look`` → ``sense.scan``: the catalogue look for the command's ``kind`` (aimed at
    ``bearing_mrad`` when given), run only while the body is still (legs quiet, head and gyro
    still); seated, the seated HOME head (rule 6; ``glance_down`` becomes a seated look). The
    result folds every capture: the head yaw of each, the light, the floor sectors and what was
    seen (the nest; the things by the eye's ``things`` list)."""

    command: LookCmd
    start_ms: int
    grid: VetoGrid | None = None
    look_prefix: str = ""
    act_id: str = field(init=False, default="")
    run: LookRun | None = None
    percept: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.act_id = self.command.act_id

    def _start(self, ctx: ActContext) -> LookRun:
        cmd = self.command
        seated = ctx.proprio.posture == "sit"
        kind = LOOK_OF_KIND.get(cmd.kind, "fixate")
        if seated and not CATALOGUE[kind].seated_ok:
            kind = "fixate"
        spec = CATALOGUE[kind]
        tasks = TASKS_OF_WANT.get(cmd.want)
        if tasks is not None:
            spec = LookSpec(
                spec.kind,
                spec.keys,
                tuple(dict.fromkeys(spec.tasks + tasks)),
                spec.safe,
                spec.seated_ok,
            )
        hints: dict[str, Any] = {}
        if cmd.bearing_mrad is not None:
            hints["target_b_mrad"] = int(cmd.bearing_mrad)
            if cmd.want == "nest":
                hints["nest_b_mrad"] = int(cmd.bearing_mrad)
        return LookRun(
            spec,
            f"{self.look_prefix}{cmd.act_id}",
            ctx.t_ms,
            yaw_mrad=int(cmd.bearing_mrad or 0),
            mode="docking" if cmd.want == "nest" else "awake",
            hints=hints,
            seated=seated,
        )

    def step(self, ctx: ActContext) -> ActOutput:
        if self.run is None:
            self.run = self._start(ctx)
        still = bool(ctx.pose is not None and ctx.pose.still)
        head, asks, finished = self.run.step(
            ctx.t_ms, still=still, legs_quiet=still, results=ctx.eye
        )
        if finished:
            return self._finish(ctx)
        return ActOutput(head=None if head == HOME else to_radians(head), looks=tuple(asks))

    def preempt(self, why: str, ctx: ActContext) -> ActOutput:
        return self._finish(ctx)

    def _finish(self, ctx: ActContext) -> ActOutput:
        if self.percept is None:
            run = self.run
            results = list(run.results) if run is not None else []
            heads = list(run.heads) if run is not None else []
            light = "none"
            sectors: list[dict[str, Any]] = []
            sightings: list[dict[str, Any]] = []
            for result, yaw_mrad in zip(results, heads, strict=False):
                level = light_of(result)
                if LIGHT_RANK.get(level, 0) > LIGHT_RANK.get(light, 0) or light == "none":
                    light = level
                floor = (result.get("results") or {}).get("floor") or {}
                for b, half, free in view_sectors(result, yaw_mrad):
                    end = "unknown"
                    for s in floor.get("sectors") or []:
                        if isinstance(s, Mapping) and int(s.get("b_mrad", 0)) == b:
                            end = str(s.get("end", "unknown"))
                    sectors.append(
                        {
                            "b_mrad": b,
                            "w_mrad": 2 * half,
                            "free_mm": max(0, free),
                            "end": end
                            if end in ("wall", "thing", "drop", "trip", "unknown")
                            else "unknown",
                            "conf_ppm": 800_000 if level == "bright" else 500_000,
                            "src": "cam",
                        }
                    )
                fix = nest_fix(result)
                if fix is not None:
                    sighting: dict[str, Any] = {
                        "what": "nest",
                        "b_mrad": int(fix["b_mrad"]),
                        "conf_ppm": int(fix.get("conf_ppm", 500_000)),
                        "look_id": str(result.get("look_id")),
                        "light": level,
                    }
                    if "r_mm" in fix:
                        sighting["r_mm"] = int(fix["r_mm"])
                        sighting["r_src"] = "tag"
                        sighting["id"] = f"tag36h11:{int(fix.get('tag', 17))}"
                    else:
                        sighting["r_mm"] = None
                        sighting["r_src"] = "cue"  # a direction only (owner C5)
                    if "yaw_mrad" in fix and fix.get("yaw_sure", True):
                        sighting["yaw_mrad"] = int(fix["yaw_mrad"])
                    sightings.append(sighting)
                if self.grid is not None and ctx.pose is not None:
                    x, y, h = ctx.pose.odom
                    origin = Pose2(int(round(x * 1000)), int(round(y * 1000)), int(round(h * 1000)))
                    self.grid.mark_view(origin, view_marks(result, yaw_mrad), ctx.t_ms, level)
            self.percept = {
                "kind": "sense.scan",
                "t_ms": ctx.t_ms,
                "act_id": self.act_id,
                "head_mrad": [int(v) for v in heads][:16],
                "light": light,
                "sectors": sectors[:32],
                "sightings": sightings[:32],
            }
        return ActOutput(percepts=(self.percept,), done=True)


def make_look_factory(grid: VetoGrid | None = None) -> Any:
    """The body loop's executor factory for ``duck.look`` (``BodyConfig.executors``)."""

    def build(cmd: Any, t_ms: int, loop: Any) -> LookExec:
        return LookExec(cmd, t_ms, grid=grid)

    return build
