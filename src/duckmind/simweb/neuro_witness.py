"""What the physical body measurably did (fid:page 2026-09-28): the page's honesty.

The owner judges only what the body visibly does in the 3D view, which draws the physics truth.
A reflex row that says 「嚇一跳，頭猛一縮」 while the head moved 2° reads as fake. So every reflex
row on the page carries what the body did in the moments after it — measured, in plain words
(「頭往後縮 18°（0.3 秒內）、停下來」) — and 「身體做不到」 when the motion was not performed;
and 「現在在做」 gets a line of what the body is really doing now.

An observer only, beside the driver (``simweb.neuro_live``): it reads robotd's measured joints
(``robot.state``, 50 Hz), robotd's own head command (``robot.state``'s ``head``: the smoothed
command the balance and walking networks are given — what the body was asked to do), the twist
the body link sent, and the physics truth (the trunk, from the body server's samples). Nothing
here reaches the brain. Pure and clock-injected: :meth:`Witness.feed` takes one :class:`Tick`.

**The verdicts are fid:audit's** (fidelity/NOTES.md 2026-09-28 11:10, the same thresholds the
visibility audit judges with):

* the head gesture index ``V = max(beak-tip travel / 1.2 cm, head rotation / 10°)`` in the trunk
  frame (beak levers 6.5 cm head pitch, 8.4 cm neck, 6.0 cm yaw; roll counts by rotation):
  visible ``V ≥ 1``, weak 0.5–1, invisible under 0.5; while walking the head is judged on its
  stride average (0.42 s: the gait itself jostles the beak ~0.7 cm);
* startle / withdraw: ``V ≥ 1`` within 0.40 s of the reflex → done; within 0.40–0.70 s or a peak
  ``V`` of 0.5–1 → partial; else not done. Orient: within 0.60 s → done (1.0 s partial). Lean in:
  ``V ≥ 0.7`` within 1.5 s;
* a stop while walking (the halting reflexes): the trunk still (chord speed under 3 cm/s over
  0.3 s) within 0.8 s and held ≥ 0.8 s → done; still within 1.5 s or held 0.4–0.8 s → partial;
* a freeze's head: every head joint within 2° peak-to-peak over the hold → still; 2–5° partial.

The words also say what robotd was asked to do (「要牠動 20°」), from its own head command.

**fid:fix 2026-09-28** (the review judged the page's claims by their words and found green
「done」 under motions that were missing or reversed):

* a reflex row names its motions (``parts``: the brain planned them from the measured head —
  contract.REFLEX_PARTS); each named part is judged on its own, in its direction (the nose up /
  down ≥ 10°, the neck back ≥ 8°, the head turned ≥ 10° left / right), on the motion that starts
  after the reflex (the pose at its onset + ONSET_MS is the baseline; motion already under way is
  not counted) — another part moving is no longer "done";
* the legs' stop has hysteresis (still under 3 cm/s, walking again only above 5 cm/s held
  0.2 s): a stop hovering at the line is one stop;
* a fall or a lift inside the window is said as such, and the head is not graded;
* the body-timed holds (「愣住 1.2 秒」: from the body — the head within HOLD_STILL_DEG — ended at
  the first visible move), the moments (「小動作」 only when the head performed it) and the body
  events (「絆了一下」 only when the trunk really tipped, outside a sit↔stand; 「被抱起來」 only when
  the trunk really rose) come from here too (``kind`` "hold" / "moment" / "body").

Conventions (``robot.head`` offsets from the v5 HOME, radians, as duck-body sends them):
``neck_pitch`` + leans the neck back, ``head_pitch`` − looks up, ``head_yaw`` / ``head_roll`` +
go to the duck's left (duckcore.content.JOINT_SIGN). The nose's pitch relative to the trunk is
``Δneck − Δhead_pitch`` (+ up / back).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import join, lst, msg
from duckmind.skilltrain import live_judge as SJ

WITNESS_VERSION = "fidfix-witness-4"
KEEP_MS = 12_000  # ticks kept (a window looks back 0.7 s and forward up to 8 s)
AUDITS_KEEP = 120
BEFORE_MS = 700  # the baseline and "what it was doing anyway" come from this long before
BASE_MS = 200  # the pose it started from: the mean over the last 200 ms before the reflex
# How long after a reflex its motion is looked for (ms), per reflex (contract.REFLEXES): long
# enough for a walking stop's 0.8 s + a 0.8 s hold.
WINDOW_MS = {
    "startle": 2000,
    "orient": 1500,
    "freeze": 2500,
    "withdraw": 2000,
    "lean_in": 3000,
    "veer": 2000,
    "drop_stop": 2000,
    "bump_stop": 2000,
    "righting": 8000,
    "lifted": 2000,
    "stabilise": 1700,
}
DEFAULT_WINDOW_MS = 1500
# The reflexes that stop the legs (reflex.HALTS): a walking body must come to a stop.
HALTING = frozenset(
    {"startle", "freeze", "withdraw", "drop_stop", "bump_stop", "stabilise", "righting", "lifted"}
)
# fid:audit's head gesture index and timings
BEAK_CM = 1.2  # V = 1: the beak tip moved 1.2 cm …
ROT_DEG = 10.0  # … or the head turned 10° (in the trunk frame)
LEVER_CM = {"head_pitch": 6.5, "neck": 8.4, "yaw": 6.0}
STRIDE_MS = 420  # while walking the head is judged on its stride average
HEAD_TIMES = {  # reflex → (done within s, partial within s, the V it must reach)
    "startle": (0.40, 0.70, 1.0),
    "withdraw": (0.40, 0.70, 1.0),
    "orient": (0.60, 1.00, 1.0),
    "lean_in": (1.50, None, 0.7),
}
WEAK_V = 0.5
STILL_MPS = 0.03  # chord speed over CHORD_S under this: the trunk is still
CHORD_S = 0.3
STOP_BY_S = 0.8  # still within this …
STOP_HOLD_S = 0.8  # … and held this long: stopped
STOP_LATE_S = 1.5
STOP_SHORT_S = 0.4
FREEZE_STILL_DEG = 2.0  # every head joint within this peak-to-peak over the hold: still
FREEZE_PARTIAL_DEG = 5.0
FREEZE_SETTLE_MS = 300  # (the move into the freeze pose first) at least …
FREEZE_HELD_S = 0.5  # (fid:fix) a freeze must hold the head still this long to be 「定住」
# (fid:fix, live 怯怯: lights off while preening — the head rose 19° into the freeze pose over
# ~0.5 s, then held within 1° for 12 s, and the line said 「頭沒定住（晃了 14°）」) … and until the
# head has SETTLED into its pose: every joint within SETTLED_DEG over the last SETTLED_MS; judged
# from SETTLE_MAX_MS at the latest (a head that never settles is not still)
SETTLED_MS = 150
SETTLED_DEG = 1.5
SETTLE_MAX_MS = 1200
# (fid:fix, live 阿勇: one push → a fall said 「跌倒了」 three times and 「自己爬起來了」 twice) a fall
# is said once until the body is up again: not fallen and the trunk under UP_TILT_DEG for UP_HOLD_MS
UP_TILT_DEG = 20.0
UP_HOLD_MS = 500
VISIBLE_DEG = 3.0  # a head move this big is said in words
# A part of the move robotd was clearly asked for (≥ 8°) that the body did not make (under 30 %
# of it) is said, and the reflex is then at most "partial": 「頭往後縮沒有做出來（要牠動 25°）」
# even when another part of the head moved visibly in time.
MISSING_CMD_DEG = 8.0
MISSING_RATIO = 0.3
COMMANDED_V = 1.0  # robotd was asked for a visible move (its own command's V)
WALKING_MPS = 0.05  # the trunk moving at least this fast: walking
STEP_BACK_M = 0.02
TURN_DEG = 12.0
CROUCH_M = 0.01
# 「現在在做」: the last second of the body.
NOW_MS = 1000
MOVING_DEG_S = 25.0  # the head turning faster than this: moving
HALT_GRACE_MS = 800  # a stop from a walk takes ~0.6 s (A100: from 0.55 m/s in 0.62 s)
HOLD_POSE_DEG = 6.0  # an offset from HOME at least this big is said
# (fid:integrate 2026-09-28) A new duck's first moments (「換一隻小鴨」, a restart): the body is
# still moving into the new duck's pose, so a reflex in them would be credited with that motion
# (live: an orient at 0.06 s was judged 「頭往後仰 64°」). Such reflexes get no body line.
BORN_SETTLE_MS = 800
# ── fid:fix 2026-09-28 ────────────────────────────────────────────────────────
ONSET_MS = 60  # the baseline is the pose over the reflex's first 60 ms (neural latency)
CARRY_S = 0.15  # a head already moving that way at the onset: its own momentum is not counted
# the named parts (contract.REFLEX_PARTS): (the measured motion, its sign, the degrees it needs)
PARTS = {
    "up": ("pitch", 1.0, 10.0),
    "dip": ("pitch", -1.0, 10.0),
    "down": ("pitch", -1.0, 10.0),
    "neck": ("neck", 1.0, 8.0),
    "yaw_left": ("yaw", 1.0, 10.0),
    "yaw_right": ("yaw", -1.0, 10.0),
}
PART_ZH = {
    "up": msg("neuro.body.part.up"),
    "dip": msg("neuro.body.part.dip"),
    "down": msg("neuro.body.part.down"),
    "neck": msg("neuro.body.part.neck"),
    "yaw_left": msg("neuro.body.part.yaw_left"),
    "yaw_right": msg("neuro.body.part.yaw_right"),
}
PART_TIMES = {  # reflex → (done within s, partial within s)
    "startle": (0.40, 0.70),
    "withdraw": (0.40, 0.70),
    "orient": (0.60, 1.00),
    "lean_in": (1.50, 2.50),
}
RESTART_MPS = 0.05  # the stop's hysteresis: walking again only above this …
RESTART_HOLD_S = 0.2  # … held this long
FALL_TILT_DEG = 45.0
# lifted (the physics truth): the trunk this high above the floor — a standing duck's trunk is at
# ~0.117 m, a seated one's ~0.061 m, the world's hand lifts it ~0.30 m — a stand-up is not a lift
LIFT_Z_M = 0.155
LIFT_RISE_M = LIFT_Z_M - 0.117  # (said as how far above a standing duck's trunk)
# body-timed holds (the page's 「愣住 / 縮著 / 停住 X 秒」)
HOLD_ACTS = {
    "startle": msg("neuro.body.hold.startle"),
    "freeze": msg("neuro.body.hold.freeze"),
    "withdraw": msg("neuro.body.hold.withdraw"),
    "still": msg("neuro.body.hold.still"),
}
HOLD_SETTLE_MS = 400  # the move into the pose first
HOLD_STILL_DEG = 5.0  # the head held within this (peak-to-peak, every joint): still
HOLD_MIN_S = 0.3
# moments (the page's 「小動作：…」): what each must show, within its time
MOMENT_CHECK = {
    "glance": ("yaw_any", 10.0),
    "glance_away": ("yaw_any", 10.0),
    "double_take": ("yaw_any", 10.0),
    "tilt": ("roll_any", 8.0),
    "perk_tilt": ("up_or_roll", 8.0),
    "peek_down": ("down", 10.0),
    "proud_puff": ("up", 10.0),
    "head_shake": ("swings", 6.0),
    "still": ("still", 3.0),
}
MOMENT_MS = 2200
# body events confirmed on the body (the page's body rows)
STUMBLE_TILT_DEG = 5.0  # a trip: the trunk tipped this much more than it was
STUMBLE_WINDOW_MS = 600
POSTURE_QUIET_MS = 1500  # … and not inside a sit↔stand (a policy change this close)
LIFT_CONFIRM_MS = 1500

Quad = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Tick:
    """One driver tick as the observer sees it. ``head``: the measured head joints as offsets
    from HOME (rad); ``head_cmd``: robotd's own head command (its smoothed ``robot.head``, the
    same offsets), None when robotd does not report it; ``twist_cmd``: (vx, vyaw) the body link
    sent; ``trunk``: the physics truth (x, y, z in metres, heading in rad) of the newest body
    sample and ``trunk_t`` its simulated time (None without one); ``holding``: the reflex the
    brain says holds the body now; ``behaviour``: the brain's act."""

    t_ms: int
    head: Quad
    head_cmd: Quad | None = None
    twist_cmd: tuple[float, float] = (0.0, 0.0)
    policy: str = ""
    trunk: tuple[float, float, float, float] | None = None
    trunk_t: float | None = None
    fallen: bool = False
    lifted: bool = False
    holding: str | None = None
    behaviour: str = ""
    # (fid:fix, additive) the trunk's tilt (robotd's IMU, degrees), the moment playing (and its
    # number: a new one starts when it changes)
    tilt_deg: float | None = None
    moment: str | None = None
    moment_seq: int = 0


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _dofs(head: Quad, base: Quad) -> dict[str, float]:
    """The visible head motions relative to ``base`` (degrees): the nose's pitch (+ up/back),
    the neck leaning back (+), the yaw and roll (+ the duck's left)."""
    d_neck = head[0] - base[0]
    d_hp = head[1] - base[1]
    return {
        "pitch": _deg(d_neck - d_hp),
        "neck": _deg(d_neck),
        "yaw": _deg(head[2] - base[2]),
        "roll": _deg(head[3] - base[3]),
    }


def gesture_index(head: Quad, base: Quad) -> float:
    """fid:audit's head gesture index V: the beak tip's travel in the trunk frame over 1.2 cm, or
    the head's rotation over 10°, whichever is larger (levers: 6.5 cm head pitch, 8.4 cm neck,
    6.0 cm yaw; both pitch joints move the beak up/back the same way)."""
    d_neck = head[0] - base[0]
    d_hp = head[1] - base[1]
    d_yaw = head[2] - base[2]
    d_roll = head[3] - base[3]
    travel = math.hypot(
        LEVER_CM["neck"] * d_neck - LEVER_CM["head_pitch"] * d_hp, LEVER_CM["yaw"] * d_yaw
    )
    turn = max(abs(_deg(d_neck - d_hp)), abs(_deg(d_yaw)), abs(_deg(d_roll)))
    return max(travel / BEAK_CM, turn / ROT_DEG)


def _mean(quads: list[Quad]) -> Quad:
    n = max(1, len(quads))
    return (
        sum(q[0] for q in quads) / n,
        sum(q[1] for q in quads) / n,
        sum(q[2] for q in quads) / n,
        sum(q[3] for q in quads) / n,
    )


def _stride_average(series: list[tuple[int, Quad]], span_ms: int) -> list[tuple[int, Quad]]:
    """Each head sample averaged with its neighbours within ± span/2 (the gait's jostle out)."""
    half = span_ms // 2
    out: list[tuple[int, Quad]] = []
    lo = 0
    for index, (t, _) in enumerate(series):
        while series[lo][0] < t - half:
            lo += 1
        hi = index
        while hi + 1 < len(series) and series[hi + 1][0] <= t + half:
            hi += 1
        out.append((t, _mean([q for _, q in series[lo : hi + 1]])))
    return out


def _peak(values: list[tuple[int, float]], t0: int) -> tuple[float, float | None]:
    """The signed value of largest size and how long (s) after ``t0`` it first reached 90 % of
    it."""
    if not values:
        return 0.0, None
    peak = max(values, key=lambda item: abs(item[1]))[1]
    if abs(peak) < 1e-9:
        return 0.0, None
    for t, value in values:
        if value * peak > 0 and abs(value) >= 0.9 * abs(peak):
            return peak, max(0.0, (t - t0) / 1000.0)
    return peak, None


def _peak_time(values: list[tuple[int, float]], peak: float) -> int:
    """When ``values`` first took the value ``peak`` (ms)."""
    for t, value in values:
        if value == peak:
            return t
    return values[-1][0] if values else 0


def _toward(values: list[tuple[int, float]], sign: float) -> float:
    """The largest motion in the direction ``sign`` (≥ 0)."""
    return max(0.0, max((value * sign for _, value in values), default=0.0))


def _speed(points: list[tuple[float, float, float]]) -> float | None:
    """Mean trunk speed (m/s) along simulated-time samples (t, x, y); None under two samples or
    under 0.1 s."""
    if len(points) < 2:
        return None
    span = points[-1][0] - points[0][0]
    if span < 0.1:
        return None
    path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(points, points[1:]))
    return path / span


def _stop(
    points: list[tuple[float, float, float, float, float]], t0_s: float, end_s: float
) -> tuple[float | None, float]:
    """When the trunk first stood still after ``t0_s`` (s after it; chord speed over CHORD_S
    under STILL_MPS) and how long it then stayed still (s, up to the window's end). (fid:fix)
    With hysteresis: it walks on again only when the chord speed stays above RESTART_MPS for
    RESTART_HOLD_S — a stop hovering at 3.0–3.1 cm/s is one stop (the review: 「只停了一下（0.1
    秒）」 under a trunk still for 2.6 s)."""
    if len(points) < 2:
        return None, 0.0
    still_at: float | None = None
    held = 0.0
    moving_since: float | None = None
    j = 0
    for i, p in enumerate(points):
        if p[0] < t0_s:
            continue
        while j < i and points[j][0] < p[0] - CHORD_S:
            j += 1
        a = points[j]
        if p[0] - a[0] < CHORD_S * 0.8:
            continue
        chord = math.hypot(p[1] - a[1], p[2] - a[2]) / max(1e-6, p[0] - a[0])
        if still_at is None:
            if chord < STILL_MPS:
                still_at = p[0]
            continue
        if chord > RESTART_MPS:
            if moving_since is None:
                moving_since = p[0]
            if p[0] - moving_since >= RESTART_HOLD_S:
                break  # (it walked on)
        else:
            moving_since = None
            held = p[0] - still_at
    if still_at is None:
        return None, 0.0
    return max(0.0, still_at - t0_s), held


PITCH_UP_ZH = msg("neuro.body.pitch.up")
PITCH_BACK_ZH = msg("neuro.body.pitch.back")
PITCH_DOWN_ZH = msg("neuro.body.pitch.down")


def head_words(dof: str, value: float, neck: float = 0.0) -> str:
    """One head motion in plain words (value in degrees, its sign the direction)."""
    size = msg("neuro.body.deg", value=abs(value))
    if dof == "pitch":
        if value > 0:
            return join(PITCH_BACK_ZH if neck >= 4.0 else PITCH_UP_ZH, " ", size)
        return join(PITCH_DOWN_ZH, " ", size)
    if dof == "yaw":
        return msg(
            "neuro.body.head.yaw",
            side=msg("neuro.side.left_short") if value > 0 else msg("neuro.side.right_short"),
            size=size,
        )
    return msg(
        "neuro.body.head.roll",
        side=msg("neuro.side.left_short") if value > 0 else msg("neuro.side.right_short"),
        size=size,
    )


def _name(dof: str, value: float, neck: float = 0.0) -> str:
    """The motion's name without its size (「頭往後縮」), as a catalog message."""
    if dof == "pitch":
        if value > 0:
            return PITCH_BACK_ZH if neck >= 4.0 else PITCH_UP_ZH
        return PITCH_DOWN_ZH
    side = msg("neuro.side.left_short") if value > 0 else msg("neuro.side.right_short")
    return msg(
        "neuro.body.head.yaw_name" if dof == "yaw" else "neuro.body.head.roll_name", side=side
    )


class _Window:
    __slots__ = ("key", "reflex", "stimulus", "t0", "end", "zh", "named")

    def __init__(
        self,
        key: tuple[Any, ...],
        reflex: str,
        stimulus: str,
        t0: int,
        zh: str,
        named: tuple[str, ...] | None = None,
    ):
        self.key = key
        self.reflex = reflex
        self.stimulus = stimulus
        self.t0 = t0
        self.end = t0 + WINDOW_MS.get(reflex, DEFAULT_WINDOW_MS)
        self.zh = zh
        self.named = named  # (fid:fix) the parts the row names; None: an event without them


class Witness:
    """Feed it every tick (:meth:`feed`) with the brain's newest reflex events; it keeps an
    audit per reflex (:attr:`audits`, newest last, each with ``seq``) and says what the body is
    doing now (:meth:`now`). (fid:fix) The same list carries the body-timed holds (``kind``
    "hold"), the moments it saw performed (``kind`` "moment") and the body events it confirmed
    on the body (``kind`` "body"); a reflex audit has ``kind`` "reflex"."""

    def __init__(self) -> None:
        self._ticks: deque[Tick] = deque()
        self._open: list[_Window] = []
        self._seen: deque[tuple[Any, ...]] = deque(maxlen=400)
        self.audits: deque[dict[str, Any]] = deque(maxlen=AUDITS_KEEP)
        self.seq = 0
        self._walk_asked_since: int | None = None
        self._halt_since: int | None = None
        self._born_ms: int | None = None  # the first tick of this brain clock
        # (fid:fix) a hold in progress: [words, act, t0, still-from t, lo[4], hi[4], moved t]
        self._hold: list[Any] | None = None
        self._moment: tuple[int, str, int] | None = None  # (seq, name, t0) being watched
        self._moment_seq = 0
        self._moment_at: tuple[str, str] = ("", "")
        self._pending: list[dict[str, Any]] = []  # body events waiting to be confirmed
        self._policy_t: tuple[str, int] = ("", -1_000_000)  # the last policy change
        self._lifted = False  # a lift confirmed on the body, not yet put down
        self._down = False  # (fid:fix) a fall said, the body not up again yet
        self._up_since: int | None = None

    # -- input

    def feed(
        self,
        tick: Tick,
        reflexes: Iterable[Mapping[str, Any]] = (),
        body_events: Iterable[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """One tick, the brain's recent reflex events (each once: keyed by t_ms, reflex,
        stimulus) and (fid:fix) the body link's events this tick (proprio's edges, the in-the-air
        detector: ``{"kind", …}``). Returns the audits and lines that closed on this tick."""
        ticks = self._ticks
        if ticks and tick.t_ms < ticks[-1].t_ms:
            # (the brain's clock started over: another duck, or the driver reattached)
            ticks.clear()
            self._open.clear()
            self._born_ms = None
            self._hold = None
            self._moment = None
            self._pending.clear()
            self._lifted = False
            self._down = False
            self._up_since = None
        if self._born_ms is None:
            self._born_ms = tick.t_ms
        if ticks and tick.policy != ticks[-1].policy:
            self._policy_t = (tick.policy, tick.t_ms)
        ticks.append(tick)
        while ticks and tick.t_ms - ticks[0].t_ms > KEEP_MS:
            ticks.popleft()
        for event in reflexes:
            reflex = str(event.get("reflex") or "")
            if not reflex:
                continue
            key = (event.get("t_ms"), reflex, event.get("stimulus"))
            if key in self._seen:
                continue
            self._seen.append(key)
            try:
                t0 = int(event.get("t_ms") or tick.t_ms)
            except (TypeError, ValueError):
                t0 = tick.t_ms
            if t0 - self._born_ms < BORN_SETTLE_MS:
                continue  # (the body is still settling into a new duck's pose)
            named = event.get("parts")
            self._open.append(
                _Window(
                    key,
                    reflex,
                    str(event.get("stimulus") or ""),
                    t0,
                    str(event.get("zh") or ""),
                    tuple(str(p) for p in named) if isinstance(named, (list, tuple)) else None,
                )
            )
        halting = (
            tick.holding in HALTING and tick.holding not in ("righting", "lifted")
        ) or tick.behaviour in ("freeze", "startle")
        if halting:
            if self._halt_since is None:
                self._halt_since = tick.t_ms
        else:
            self._halt_since = None
        vx = tick.twist_cmd[0]
        if vx >= 0.25 and tick.policy in ("walk", "stand"):
            if self._walk_asked_since is None:
                self._walk_asked_since = tick.t_ms
        else:
            self._walk_asked_since = None
        closed: list[dict[str, Any]] = []
        still: list[_Window] = []
        for window in self._open:
            if tick.t_ms >= window.end:
                closed.append(self._audit(window))
            else:
                still.append(window)
        self._open = still
        closed += self._track_hold(tick)
        closed += self._track_moment(tick)
        closed += self._confirm_body(tick, body_events)
        for audit in closed:
            self.seq += 1
            audit["seq"] = self.seq
            audit.setdefault("kind", "reflex")
            self.audits.append(audit)
        return closed

    # -- one reflex

    def _audit(self, w: _Window) -> dict[str, Any]:
        ticks = list(self._ticks)
        before = [k for k in ticks if w.t0 - BEFORE_MS <= k.t_ms < w.t0]
        during = [k for k in ticks if w.t0 <= k.t_ms <= w.end]
        out: dict[str, Any] = {
            "key": list(w.key),
            "t_ms": w.t0,
            "reflex": w.reflex,
            "window_s": round((w.end - w.t0) / 1000.0, 2),
            "version": WITNESS_VERSION,
        }
        if not during:
            out.update({"verdict": None, "zh": "", "parts": []})
            return out
        # (fid:fix) a fall or a lift inside the window: said as such; the head is not graded
        truth_all = _truth_points(before + during)
        fell = any(k.fallen for k in during) or any(
            (k.tilt_deg or 0.0) >= FALL_TILT_DEG for k in during
        )
        rose = w.reflex != "lifted" and any(p[3] >= LIFT_Z_M for p in truth_all)
        if w.reflex == "righting":
            # (fid:fix) 「跌倒了，爬起來」 / 「自己爬起來了」: whether the body IS up, and when
            up_at = _up_at(during)
            if up_at is None:
                out.update(
                    {
                        "verdict": "not_done",
                        "parts": [],
                        "zh": msg("neuro.body.righting.still_down"),
                    }
                )
            else:
                seconds = max(0.0, (up_at - w.t0) / 1000.0)
                out.update(
                    {
                        "verdict": "done",
                        "parts": [],
                        "up_s": round(seconds, 1),
                        "zh": msg("neuro.body.righting.up_after", seconds=seconds)
                        if seconds >= 0.1
                        else msg("neuro.body.righting.up"),
                    }
                )
            return out
        if fell or rose:
            out.update(
                {
                    "verdict": None,
                    "parts": [],
                    "fell": fell,
                    "lifted_up": rose,
                    "zh": msg("neuro.body.window.fell")
                    if fell
                    else msg("neuro.body.window.lifted"),
                }
            )
            return out
        # the legs first: walking decides how the head is judged
        truth_before = _truth_points(before)
        truth_during = _truth_points(during)
        v_before = _speed([(t, x, y) for t, x, y, _, _ in truth_before])
        walking = v_before is not None and v_before >= WALKING_MPS

        # the head, measured (stride-averaged while walking) and as robotd was asked
        series = [(k.t_ms, k.head) for k in before + during]
        if walking:
            series = _stride_average(series, STRIDE_MS)
        if w.named is not None:
            # (fid:fix) the motion that starts with the reflex: the pose over its first ONSET_MS
            base_rows = [q for t, q in series if w.t0 <= t <= w.t0 + ONSET_MS]
        else:
            base_rows = [q for t, q in series if w.t0 - BASE_MS <= t < w.t0]
        base = _mean(base_rows or [during[0].head])
        meas: dict[str, list[tuple[int, float]]] = {d: [] for d in ("pitch", "neck", "yaw", "roll")}
        vs: list[tuple[int, float]] = []
        noise_v = 0.0
        for t, q in series:
            if t < w.t0:
                noise_v = max(noise_v, gesture_index(q, base))
                continue
            for dof, value in _dofs(q, base).items():
                meas[dof].append((t, value))
            vs.append((t, gesture_index(q, base)))
        cmds_before = [
            k.head_cmd for k in before if k.head_cmd is not None and k.t_ms >= w.t0 - BASE_MS
        ]
        base_cmd = _mean(cmds_before) if cmds_before else None
        cmd: dict[str, list[tuple[int, float]]] = {d: [] for d in meas}
        cmd_v = 0.0
        if base_cmd is not None:
            for k in during:
                if k.head_cmd is not None:
                    for dof, value in _dofs(k.head_cmd, base_cmd).items():
                        cmd[dof].append((k.t_ms, value))
                    cmd_v = max(cmd_v, gesture_index(k.head_cmd, base_cmd))
        head: dict[str, dict[str, Any]] = {}
        for dof in ("pitch", "yaw", "roll"):
            peak, reach = _peak(meas[dof], w.t0)
            cmd_peak, _ = _peak(cmd[dof], w.t0)
            item: dict[str, Any] = {
                "deg": round(peak, 1),
                "reach_s": None if reach is None else round(reach, 2),
                "cmd_deg": round(cmd_peak, 1),
            }
            if abs(cmd_peak) >= 1.0:
                item["got_deg"] = round(_toward(meas[dof], 1.0 if cmd_peak > 0 else -1.0), 1)
            if dof == "yaw" and meas["yaw"]:
                at = min(during, key=lambda k: abs(k.t_ms - _peak_time(meas["yaw"], peak)))
                item["abs_deg"] = round(_deg(at.head[2]), 1)
            head[dof] = item
        neck_peak, _ = _peak(meas["neck"], w.t0)
        cmd_neck, _ = _peak(cmd["neck"], w.t0)
        peak_v = max((v for _, v in vs), default=0.0)
        need_v = HEAD_TIMES.get(w.reflex, (0.0, None, 1.0))[2]
        t_v = next(((t - w.t0) / 1000.0 for t, v in vs if v >= need_v), None)
        out["head"] = head
        out["neck_deg"] = round(neck_peak, 1)
        out["cmd_neck_deg"] = round(cmd_neck, 1)
        out["v_peak"] = round(peak_v, 2)
        out["v_before"] = round(noise_v, 2)
        out["v_cmd"] = round(cmd_v, 2)
        out["v_reach_s"] = None if t_v is None else round(t_v, 2)
        out["walking"] = walking

        head_ok: str | None = None
        times = HEAD_TIMES.get(w.reflex)
        if w.named:
            # (fid:fix) each named part on its own, in its direction, in its time
            carry = _carry(series, w.t0)
            times_p = PART_TIMES.get(w.reflex, (0.6, 1.0))
            judged = [
                _judge_part(part, meas, w.t0, times_p, carry) for part in w.named if part in PARTS
            ]
            if w.reflex == "orient":
                # a look toward something is also where the head POINTS in the room: the head's
                # turn plus the trunk's own (a look the body turns into — the brain's body-follows-
                # the-head arc, a walk's bend — is still a look that way); the better of the two
                gaze = _gaze_series(meas["yaw"], before + during, w.t0)
                if gaze:
                    room = {**meas, "yaw": gaze}
                    rank = {"done": 2, "partial": 1, "not_done": 0}
                    for i, item in enumerate(judged):
                        if item["part"] not in ("yaw_left", "yaw_right"):
                            continue
                        other = _judge_part(item["part"], room, w.t0, times_p, {})
                        if (rank[other["ok"]], other["deg"]) > (rank[item["ok"]], item["deg"]):
                            judged[i] = {**other, "room": True}
            out["named"] = judged
            oks = [item["ok"] for item in judged]
            if oks:
                head_ok = (
                    "done"
                    if all(ok == "done" for ok in oks)
                    else "not_done"
                    if all(ok == "not_done" for ok in oks)
                    else "partial"
                )
        elif times is not None:
            # the head's verdict (fid:audit), for the reflexes that claim a head motion;
            # orienting 「看正前方」 claims none when robotd was not asked for a visible move
            straight = "正前方" in w.zh and cmd_v < COMMANDED_V
            if not straight:
                done_by, partial_by, _ = times
                if t_v is not None and t_v <= done_by:
                    head_ok = "done"
                elif (t_v is not None and partial_by is not None and t_v <= partial_by) or (
                    peak_v >= WEAK_V
                ):
                    head_ok = "partial"
                else:
                    head_ok = "not_done"
            if w.named is None:
                # (the events of old: the parts robotd was asked for and did not make)
                missing = [
                    dof
                    for dof, item in head.items()
                    if abs(float(item["cmd_deg"])) >= MISSING_CMD_DEG
                    and float(item.get("got_deg") or 0.0)
                    < MISSING_RATIO * abs(float(item["cmd_deg"]))
                ]
                if head_ok is not None:
                    out["missing"] = missing
                    if head_ok == "done" and missing:
                        head_ok = "partial"
        out["head_ok"] = head_ok

        # a freeze's head: still over the hold
        freeze_ok: str | None = None
        if w.reflex == "freeze":
            # (fid:fix) the freeze's OWN hold: from when the head settled into the pose until the
            # command moves on (its release back to the act: headless, a 0.9-s freeze was judged
            # over the 2.5-s window and 「頭沒定住（晃了 30°）」 was the peck after it)
            from_ms = _settled_at(during, w.t0 + FREEZE_SETTLE_MS, w.t0 + SETTLE_MAX_MS)
            out["freeze_from_s"] = round((from_ms - w.t0) / 1000.0, 2)
            hold = _held_part(during, from_ms)
            posed = next((k for k in during if k.t_ms >= from_ms), None)
            if posed is not None:
                # the move into the pose (not graded: said, measured)
                moved = _dofs(posed.head, base)
                out["freeze_into"] = {dof: round(v, 1) for dof, v in moved.items()}
            if len(hold) >= 5:
                pp = max(
                    _deg(max(k.head[i] for k in hold) - min(k.head[i] for k in hold))
                    for i in range(4)
                )
                held_s = (hold[-1].t_ms - hold[0].t_ms) / 1000.0
                out["freeze_pp_deg"] = round(pp, 1)
                out["freeze_held_s"] = round(held_s, 1)
                freeze_ok = (
                    "done"
                    if pp <= FREEZE_STILL_DEG
                    else "partial"
                    if pp <= FREEZE_PARTIAL_DEG
                    else "not_done"
                )
                if freeze_ok == "done" and held_s < FREEZE_HELD_S:
                    freeze_ok = "partial"
                    out["freeze_short"] = True
            elif posed is not None:
                freeze_ok = "partial"  # it moved on before it held still
                out["freeze_short"] = True
        out["freeze_ok"] = freeze_ok

        # the legs (physics truth): a halting reflex while walking must stop the trunk
        legs: dict[str, Any] = {
            "before_mps": None if v_before is None else round(v_before, 3),
            "walking": walking,
        }
        late = [
            p for p in truth_during if p[0] >= (truth_during[0][0] if truth_during else 0) + 0.7
        ]
        v_after = _speed([(t, x, y) for t, x, y, _, _ in late])
        legs["after_mps"] = None if v_after is None else round(v_after, 3)
        legs["stop_cmd"] = any(abs(k.twist_cmd[0]) < 1e-6 for k in during if k.t_ms <= w.t0 + 800)
        # (fid:integrate) a withdrawal whose words say it steps away (「…，走開一步」 /
        # 「…，往後退一小步」: reflex.WITHDRAW escape) moves the legs on purpose: no stop is judged
        # (live: a poke while slowly moving was called 「身體做不到：腳沒停下來」 as it stepped away)
        steps_away = w.reflex == "withdraw" and ("走開" in w.zh or "退" in w.zh)
        legs["steps_away"] = steps_away
        if w.reflex in HALTING and walking and truth_before and truth_during and not steps_away:
            t0_s = truth_during[0][0]
            still_at, held = _stop(truth_before + truth_during, t0_s, truth_during[-1][0])
            legs["still_s"] = None if still_at is None else round(still_at, 2)
            legs["held_s"] = round(held, 2)
            if still_at is not None and still_at <= STOP_BY_S and held >= STOP_HOLD_S:
                legs["ok"] = "done"
            elif still_at is not None and (still_at <= STOP_LATE_S or held >= STOP_SHORT_S):
                legs["ok"] = "partial"
            else:
                legs["ok"] = "not_done"
        start = truth_during[0] if truth_during else (truth_before[-1] if truth_before else None)
        if start is not None and truth_during:
            end = truth_during[-1]
            heading = start[4]
            forward = (end[1] - start[1]) * math.cos(heading) + (end[2] - start[2]) * math.sin(
                heading
            )
            legs["forward_m"] = round(forward, 3)
            legs["turn_deg"] = round(_deg(_wrap(end[4] - heading)), 1)
            legs["drop_m"] = round(min(p[3] for p in truth_during) - start[3], 3)
            legs["rise_m"] = round(max(p[3] for p in truth_during) - start[3], 3)
        out["legs"] = legs
        if w.reflex == "lifted":
            # (fid:fix) 「覺得被抱起來，安靜不動」: did the trunk really rise (above where it stood
            # on the floor before: the lift may have begun before the claim came), and keep quiet
            top = max((p[3] for p in truth_during), default=0.0)
            rise = max(0.0, float(top) - 0.117)  # above a standing duck's trunk
            settled = [k for k in during if k.t_ms >= w.t0 + 1000]
            pp = (
                max(
                    _deg(max(k.head[i] for k in settled) - min(k.head[i] for k in settled))
                    for i in range(4)
                )
                if len(settled) >= 5
                else 0.0
            )
            out["lift_rise_m"] = round(rise, 3)
            out["quiet_pp_deg"] = round(pp, 1)
            out["parts"] = []
            if top >= LIFT_Z_M:
                out["verdict"] = "done" if pp <= 10.0 else "partial"
                if pp <= 10.0:
                    out["zh"] = join(
                        msg("neuro.body.lifted.rose_prefix", rise=rise * 100),
                        msg("neuro.body.lifted.quiet"),
                    )
                else:  # (the same words, behind the "only part of it" prefix)
                    out["zh"] = join(
                        msg("neuro.body.verdict.partial.prefix"),
                        msg("neuro.body.lifted.rose_part", rise=rise * 100),
                        msg("neuro.body.lifted.head_moving", pp=pp),
                    )
            else:
                out["verdict"] = None  # (its belief was wrong: nothing for the body to do)
                out["zh"] = msg("neuro.body.lifted.not_really")
            return out

        # (s1b:fix) a trained skill ran in this window: robotd ignored the head command and the
        # skill network moved the whole body — judged by the skill's own detectors (the Mac
        # gates' trunk dip, neck pull-back, one crouch, stop), not by the head the motor asked
        # for (live, tr:gate: 4 of 6 real 16–27 mm crouches read 「只做到一部分」)
        skilled = SJ.judge_window(before + during, w.reflex, w.t0, named=w.named)
        if skilled is not None:
            out["skill"] = skilled
            out["head_ok_without_skill"] = head_ok
            out["verdict"] = skilled["verdict"]
            out["parts"] = skilled["parts"]
            out["zh"] = skilled["zh"]
            return out
        verdicts = [v for v in (head_ok, freeze_ok, legs.get("ok")) if v is not None]
        verdict = (
            None
            if not verdicts
            else "not_done"
            if "not_done" in verdicts
            else "partial"
            if "partial" in verdicts
            else "done"
        )
        parts, fails, partial = _words(out, w)
        out["verdict"] = verdict
        out["parts"] = parts
        out["zh"] = _sentence(verdict, parts, fails, partial)
        return out

    # -- (fid:fix) the body-timed holds

    def _track_hold(self, tick: Tick) -> list[dict[str, Any]]:
        """「愣住 X 秒」 from the body: from the start of a holding act (startle, freeze,
        withdraw, still) the head is watched; X runs until the head moves (every joint within
        HOLD_STILL_DEG after it settled into the pose) or the act ends. Said only when it held
        still ≥ HOLD_MIN_S."""
        words = HOLD_ACTS.get(tick.behaviour)
        out: list[dict[str, Any]] = []
        hold = self._hold
        if hold is not None and words != hold[0]:
            out += self._close_hold(tick.t_ms)
            hold = self._hold = None
        elif hold is not None and tick.behaviour != hold[1]:
            # (fid:fix, live: 「愣住 1.1 秒」 then 「愣住 5.9 秒」 for one stillness) a startle going on
            # into its freeze is ONE 「愣住」 — unless the head already moved inside the startle:
            # then the freeze's own stillness is a new one
            if hold[6] is not None:
                out += self._close_hold(tick.t_ms)
                hold = self._hold = None
            else:
                hold[1] = tick.behaviour
        if words is None:
            return out
        if hold is None:
            self._hold = [words, tick.behaviour, tick.t_ms, None, None, None, None]
            return out
        t0 = int(hold[2])
        if tick.t_ms - t0 < HOLD_SETTLE_MS or hold[6] is not None:
            return out
        if hold[3] is None:
            # (fid:fix) the still part starts once the head has settled into the pose (a
            # startle's own snap and its return, a freeze's rise: live, 「愣住 0.5 秒，然後頭動了」
            # was the end of the snap itself), at the latest SETTLE_MAX_MS after the start
            if tick.t_ms - t0 < SETTLE_MAX_MS and not _settled(self._ticks, tick.t_ms):
                return out
            hold[3], hold[4], hold[5] = tick.t_ms, list(tick.head), list(tick.head)
            return out
        lo, hi = hold[4], hold[5]
        for i in range(4):
            lo[i] = min(lo[i], tick.head[i])
            hi[i] = max(hi[i], tick.head[i])
        if max(_deg(hi[i] - lo[i]) for i in range(4)) > HOLD_STILL_DEG:
            hold[6] = tick.t_ms  # the first visible move: the still part ends here
        return out

    def _close_hold(self, t_ms: int) -> list[dict[str, Any]]:
        hold = self._hold
        if hold is None or hold[3] is None:
            return []
        moved = hold[6]
        until = int(moved) if moved is not None else t_ms
        still_s = (until - int(hold[2])) / 1000.0
        if still_s < HOLD_MIN_S:
            return []
        zh = join(
            msg("neuro.body.hold.row", hold_word=hold[0], still_s=still_s),
            msg("neuro.body.hold.then_moved_suffix") if moved is not None else None,
        )
        return [
            {
                "kind": "hold",
                "act": hold[1],
                "t_ms": int(hold[2]),
                "still_s": round(still_s, 1),
                "still_from_ms": int(hold[3]),  # (the head settled into the pose here)
                "moved": moved is not None,
                "zh": zh,
                "version": WITNESS_VERSION,
            }
        ]

    # -- (fid:fix) the moments

    def _track_moment(self, tick: Tick) -> list[dict[str, Any]]:
        """A moment (「小動作」) is shown on the page only when the head performed it: its named
        motion within MOMENT_MS of its start (MOMENT_CHECK)."""
        out: list[dict[str, Any]] = []
        if tick.moment and tick.moment_seq != self._moment_seq:
            self._moment_seq = tick.moment_seq
            self._moment = (tick.moment_seq, str(tick.moment), tick.t_ms)
            self._moment_at = (tick.policy, tick.behaviour)
        watching = self._moment
        if watching is None:
            return out
        seq, name, t0 = watching
        span = [k for k in self._ticks if t0 <= k.t_ms <= t0 + MOMENT_MS]
        done_by = tick.t_ms - t0 >= MOMENT_MS
        ok, value = _moment_ok(name, span)
        if ok or done_by:
            self._moment = None
            out.append(
                {
                    "kind": "moment",
                    "moment_seq": seq,
                    "moment": name,
                    "t_ms": t0,
                    "ok": ok,
                    "value": round(value, 1),
                    "policy": self._moment_at[0],
                    "behaviour": self._moment_at[1],
                    "version": WITNESS_VERSION,
                }
            )
        return out

    # -- (fid:fix) the body events, confirmed on the body

    def _confirm_body(
        self, tick: Tick, events: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """proprio's edges said only when the body shows them: a stumble when the trunk tipped
        ≥ STUMBLE_TILT_DEG more than it was, outside a sit↔stand (the review: 43 of 44
        「絆了一下」 were a sit-down or a rise, 2.8° median); a lift when the trunk rose
        ≥ LIFT_RISE_M (a push on a seated duck unloads its legs too: 「被抱起來了」 while it
        moved 0.6 cm); set down after a real lift; a fall as robotd says it."""
        out: list[dict[str, Any]] = []
        for event in events:
            kind = str(event.get("kind") or "")
            if kind in ("proprio.stumbled", "proprio.lifted", "body.airborne", "proprio.set_down"):
                self._pending.append({"kind": kind, "t_ms": tick.t_ms, "tilt0": self._tilt_base()})
            elif kind in ("proprio.fell", "body.fell"):
                if not self._down:  # (proprio and duck-body both say it: once per fall)
                    self._down = True
                    self._up_since = None
                    out.append(self._body_row("fell", tick.t_ms, msg("neuro.body.event.fell")))
        z = tick.trunk[2] if tick.trunk is not None else None
        keep: list[dict[str, Any]] = []
        for item in self._pending:
            age = tick.t_ms - int(item["t_ms"])
            kind = item["kind"]
            if kind == "proprio.stumbled":
                recent = [k for k in self._ticks if int(item["t_ms"]) - 100 <= k.t_ms]
                tilt = max((k.tilt_deg or 0.0 for k in recent), default=0.0)
                in_posture_change = abs(int(item["t_ms"]) - self._policy_t[1]) <= POSTURE_QUIET_MS
                if tilt - float(item["tilt0"]) >= STUMBLE_TILT_DEG and not in_posture_change:
                    out.append(
                        self._body_row(
                            "stumble", int(item["t_ms"]), msg("neuro.body.event.stumble")
                        )
                    )
                elif age < STUMBLE_WINDOW_MS:
                    keep.append(item)
            elif kind in ("proprio.lifted", "body.airborne"):
                if self._lifted:
                    continue
                if z is not None and z >= LIFT_Z_M:
                    self._lifted = True
                    out.append(
                        self._body_row("lifted", int(item["t_ms"]), msg("neuro.body.event.lifted"))
                    )
                elif age < LIFT_CONFIRM_MS:
                    keep.append(item)
            elif kind == "proprio.set_down":
                if self._lifted:
                    if z is not None and z < LIFT_Z_M:
                        self._lifted = False
                        out.append(
                            self._body_row(
                                "set_down", int(item["t_ms"]), msg("neuro.body.event.set_down")
                            )
                        )
                    elif age < LIFT_CONFIRM_MS:
                        keep.append(item)
        self._pending = keep
        if self._lifted and z is not None and z < LIFT_Z_M - 0.02 and not tick.lifted:
            self._lifted = False  # back on the floor (a set-down edge may never come)
        if self._down:
            if not tick.fallen and (tick.tilt_deg or 0.0) < UP_TILT_DEG:
                if self._up_since is None:
                    self._up_since = tick.t_ms
                elif tick.t_ms - self._up_since >= UP_HOLD_MS:
                    self._down = False
            else:
                self._up_since = None
        return out

    def _tilt_base(self) -> float:
        tilts = [k.tilt_deg for k in list(self._ticks)[-25:] if k.tilt_deg is not None]
        if not tilts:
            return 0.0
        tilts.sort()
        return tilts[len(tilts) // 2]

    @staticmethod
    def _body_row(what: str, t_ms: int, zh: str) -> dict[str, Any]:
        return {"kind": "body", "what": what, "t_ms": t_ms, "zh": zh, "version": WITNESS_VERSION}

    # -- now

    def now(self) -> dict[str, Any] | None:
        """What the body is doing right now, from its last second: ``zh`` in plain words,
        ``mismatch_zh`` when it is not what the brain says (a halt while the legs walk on, a walk
        the body is not making)."""
        ticks = list(self._ticks)
        if not ticks:
            return None
        last = ticks[-1]
        recent = [k for k in ticks if last.t_ms - k.t_ms <= NOW_MS]
        points = _truth_points(recent)
        speed = _speed([(t, x, y) for t, x, y, _, _ in points])
        turn_rate = None
        if len(points) >= 2 and points[-1][0] - points[0][0] >= 0.3:
            turn_rate = _wrap(points[-1][4] - points[0][4]) / (points[-1][0] - points[0][0])
        bits: list[str] = []
        posture = "stand"
        if last.fallen:
            posture = "fallen"
            bits.append(msg("neuro.body.now.fallen"))
        elif last.lifted:
            posture = "held"
            bits.append(msg("neuro.body.now.held"))
        elif last.policy == "sit":
            posture = "sit"
            bits.append(msg("neuro.body.now.sitting"))
        elif SJ.doing_zh(last.policy) is not None:
            posture = "skill"  # (s1b:fix) a trained whole-body skill, not 「姿勢在換」
            bits.append(SJ.doing_zh(last.policy))
        elif speed is not None and speed >= WALKING_MPS:
            posture = "walk"
            turning = ""
            if turn_rate is not None and abs(turn_rate) >= 0.25:
                turning = msg(
                    "neuro.body.now.turning_prefix",
                    side=msg("neuro.side.left_short")
                    if turn_rate > 0
                    else msg("neuro.side.right_short"),
                )
            bits.append(msg("neuro.body.now.walking", turning=turning, speed=speed))
        elif last.policy in ("walk", "stand"):
            bits.append(msg("neuro.body.now.standing"))
        elif last.policy:
            posture = last.policy
            bits.append(
                {"rise": msg("neuro.body.now.rising")}.get(
                    last.policy, msg("neuro.body.now.changing_posture")
                )
            )
        # the head: where it points (from HOME) and whether it moves
        pose = _dofs(last.head, (0.0, 0.0, 0.0, 0.0))
        head_bits = []
        if abs(pose["yaw"]) >= HOLD_POSE_DEG:
            head_bits.append(
                msg(
                    "neuro.body.now.head_yaw",
                    side=msg("neuro.side.left_short")
                    if pose["yaw"] > 0
                    else msg("neuro.side.right_short"),
                    yaw=abs(pose["yaw"]),
                )
            )
        if abs(pose["pitch"]) >= HOLD_POSE_DEG:
            head_bits.append(
                msg(
                    "neuro.body.now.head_pitch",
                    choice=msg("neuro.web.neuro_witness.Witness.now.4bac0e")
                    if pose["pitch"] > 0
                    else msg("neuro.body.pitch.down"),
                    pitch=abs(pose["pitch"]),
                )
            )
        if abs(pose["roll"]) >= HOLD_POSE_DEG:
            head_bits.append(
                msg(
                    "neuro.body.now.head_roll",
                    side=msg("neuro.side.left_short")
                    if pose["roll"] > 0
                    else msg("neuro.side.right_short"),
                    roll=abs(pose["roll"]),
                )
            )
        head_rate = _head_rate([k for k in recent if last.t_ms - k.t_ms <= 300])
        moving = head_rate is not None and head_rate >= MOVING_DEG_S
        if head_bits:
            bits.append(lst(head_bits))
        bits.append(
            msg("neuro.body.now.head_moving") if moving else msg("neuro.body.now.head_still")
        )
        mismatch = None
        # (a stop takes the legs ~0.6 s from a walk: only a halt older than HALT_GRACE_MS that the
        # legs still walk through is a mismatch)
        if (
            self._halt_since is not None
            and last.t_ms - self._halt_since >= HALT_GRACE_MS
            and posture == "walk"
        ):
            mismatch = msg("neuro.body.mismatch.halt_but_walking")
        elif (
            self._walk_asked_since is not None
            and last.t_ms - self._walk_asked_since >= 1500
            and (speed is None or speed < STILL_MPS)
            and posture not in ("fallen", "held", "sit")
        ):
            mismatch = msg("neuro.body.mismatch.walk_but_still")
        return {
            "zh": lst(bits, "fmt.sep.dot"),
            "posture": posture,
            "speed_mps": None if speed is None else round(speed, 3),
            "head_moving": moving,
            "head_deg_s": None if head_rate is None else round(head_rate, 1),
            "mismatch_zh": mismatch,
            "t_ms": last.t_ms,
        }

    def speed_now(self, window_ms: int = 500) -> float | None:
        """(fid:fix) The trunk's speed over the last ``window_ms`` (the physics truth, m/s)."""
        ticks = self._ticks
        if not ticks:
            return None
        last = ticks[-1].t_ms
        recent = [k for k in reversed(ticks) if last - k.t_ms <= window_ms]
        return _speed([(t, x, y) for t, x, y, _, _ in _truth_points(list(reversed(recent)))])

    def turn_now(self, window_ms: int = 500) -> float | None:
        """(fid:fix) The trunk's turning rate over the last ``window_ms`` (truth, rad/s, + left)."""
        ticks = self._ticks
        if not ticks:
            return None
        last = ticks[-1].t_ms
        points = _truth_points([k for k in ticks if last - k.t_ms <= window_ms])
        if len(points) < 2 or points[-1][0] - points[0][0] < 0.2:
            return None
        return _wrap(points[-1][4] - points[0][4]) / (points[-1][0] - points[0][0])

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [audit for audit in self.audits if audit["seq"] > seq]


def _truth_ticks(ticks: list[Tick]) -> list[Tick]:
    """The ticks that carry a new truth sample (each body sample once)."""
    out: list[Tick] = []
    last_t: float | None = None
    for k in ticks:
        if k.trunk is None or k.trunk_t is None or k.trunk_t == last_t:
            continue
        last_t = k.trunk_t
        out.append(k)
    return out


def _truth_points(ticks: list[Tick]) -> list[tuple[float, float, float, float, float]]:
    """(simulated t, x, y, z, heading) of each new truth sample."""
    out = []
    for k in _truth_ticks(ticks):
        trunk = k.trunk
        assert trunk is not None and k.trunk_t is not None
        out.append((float(k.trunk_t), trunk[0], trunk[1], trunk[2], trunk[3]))
    return out


def _head_rate(ticks: list[Tick]) -> float | None:
    """The head's fastest joint speed over these ticks (°/s): total travel over the time."""
    if len(ticks) < 2:
        return None
    span = (ticks[-1].t_ms - ticks[0].t_ms) / 1000.0
    if span <= 0:
        return None
    best = 0.0
    for index in range(4):
        travel = sum(abs(b.head[index] - a.head[index]) for a, b in zip(ticks, ticks[1:]))
        best = max(best, _deg(travel) / span)
    return best


def _turn_words(item: Mapping[str, Any], deg: float) -> tuple[str, str]:
    """A head turn by where the head ended up (「頭轉到右邊 17°」), and — when that was the
    other way — by how much it turned (「往左轉了 8°」): a head already turned past what it looks
    at turns back toward it. (main words, a detail for the parenthesis)"""
    where = item.get("abs_deg")
    if where is None:
        return head_words("yaw", deg), ""
    where = float(where)
    if abs(where) < 5.0:
        return msg("neuro.body.turn.back_to_front"), msg("neuro.body.turn.turned_by", deg=abs(deg))
    side = msg("neuro.side.left_short") if where > 0 else msg("neuro.side.right_short")
    main = msg("neuro.body.turn.ended_at", side=side, where=abs(where))
    if where * deg > 0:
        return main, ""
    return main, msg(
        "neuro.body.turn.turned_way",
        side=msg("neuro.side.left_short") if deg > 0 else msg("neuro.side.right_short"),
        deg=abs(deg),
    )


def _motion(
    dof: str, item: Mapping[str, Any], neck: float, when: float | None = None, extra: str = ""
) -> str:
    """One measured head motion in words, with when it got there and any note, in one
    parenthesis (「頭往後縮 18°（0.3 秒內）」, 「頭轉到右邊 17°（往左轉了 8°，動得很小）」)."""
    deg = float(item.get("deg") or 0.0)
    if dof == "yaw":
        main, detail = _turn_words(item, deg)
    else:
        main, detail = head_words(dof, deg, neck), ""
    bits = [
        b
        for b in (detail, msg("neuro.body.within_s", when=when) if when is not None else "", extra)
        if b
    ]
    return (
        msg("neuro.body.motion_with_details", part=main, bits=lst(bits, "fmt.sep.comma"))
        if bits
        else main
    )


def _carry(series: list[tuple[int, Quad]], t0: int) -> dict[str, float]:
    """(fid:fix) How far the head would have gone anyway (degrees per motion): its speed over
    the 120 ms before the reflex × CARRY_S — a turn already under way is not the reflex's."""
    pre = [(t, q) for t, q in series if t0 - 120 <= t <= t0]
    if len(pre) < 2 or pre[-1][0] - pre[0][0] < 40:
        return {}
    span = (pre[-1][0] - pre[0][0]) / 1000.0
    moved = _dofs(pre[-1][1], pre[0][1])
    return {dof: value / span * CARRY_S for dof, value in moved.items()}


def _gaze_series(
    yaw: list[tuple[int, float]], ticks: list[Tick], t0: int
) -> list[tuple[int, float]]:
    """(fid:fix) The head's yaw plus the trunk's turn since the reflex's onset (truth heading),
    degrees: where the head points in the room. Empty without the physics truth."""
    headings = [(k.t_ms, k.trunk[3]) for k in ticks if k.trunk is not None]
    if not headings:
        return []
    start = [h for t, h in headings if t0 <= t <= t0 + ONSET_MS] or [
        h for t, h in headings if t <= t0
    ][-1:]
    if not start:
        return []
    h0 = start[0]
    out: list[tuple[int, float]] = []
    j = 0
    for t, value in yaw:
        while j + 1 < len(headings) and headings[j + 1][0] <= t:
            j += 1
        out.append((t, value + _deg(_wrap(headings[j][1] - h0))))
    return out


def _up_at(ticks: list[Tick]) -> int | None:
    """(fid:fix) When the body was up again: the start of the first UP_HOLD_MS with it not
    fallen and the trunk under UP_TILT_DEG."""
    since: int | None = None
    for k in ticks:
        if not k.fallen and (k.tilt_deg or 0.0) < UP_TILT_DEG:
            if since is None:
                since = k.t_ms
            if k.t_ms - since >= UP_HOLD_MS:
                return since
        else:
            since = None
    return None


def _held_part(ticks: list[Tick], from_ms: int) -> list[Tick]:
    """(fid:fix) A freeze's own hold: the ticks from ``from_ms`` while robotd's head command
    stays where it was (within SETTLED_DEG) and the freeze still holds the body — its release
    back to the act, and the act after it, are not the freeze."""
    out: list[Tick] = []
    ref: Quad | None = None
    for k in ticks:
        if k.t_ms < from_ms:
            continue
        if k.holding is not None and k.holding != "freeze":
            break
        if k.head_cmd is not None:
            if ref is None:
                ref = k.head_cmd
            elif max(_deg(abs(k.head_cmd[i] - ref[i])) for i in range(4)) > SETTLED_DEG:
                break
        out.append(k)
    return out


def _settled(ticks: Iterable[Tick], t_ms: int) -> bool:
    """(fid:fix) Whether the head had settled at ``t_ms``: every head joint within SETTLED_DEG
    over the SETTLED_MS before it (at least three ticks, covering that span)."""
    span = [k for k in ticks if t_ms - SETTLED_MS <= k.t_ms <= t_ms]
    if len(span) < 3 or span[0].t_ms > t_ms - SETTLED_MS + 40:
        return False
    return (
        max(_deg(max(k.head[i] for k in span) - min(k.head[i] for k in span)) for i in range(4))
        <= SETTLED_DEG
    )


def _settled_at(ticks: list[Tick], t_min: int, t_max: int) -> int:
    """(fid:fix) When the head had settled into its pose: the first tick from ``t_min`` on
    that ends a settled span (:func:`_settled`); ``t_max`` at the latest."""
    for k in ticks:
        if k.t_ms < t_min:
            continue
        if k.t_ms > t_max:
            break
        if _settled(ticks, k.t_ms):
            return k.t_ms
    return t_max


def _judge_part(
    part: str,
    meas: Mapping[str, list[tuple[int, float]]],
    t0: int,
    times: tuple[float, float],
    carry: Mapping[str, float],
) -> dict[str, Any]:
    """(fid:fix) One named motion: in its own direction, from the reflex's onset pose, less the
    head's own momentum; done within the reflex's time, partial late or half-way."""
    dof, sign, need = PARTS[part]
    done_by, partial_by = times
    momentum = max(0.0, sign * float(carry.get(dof, 0.0)))
    best = 0.0
    reach: float | None = None
    for t, value in meas.get(dof, ()):
        s = (t - t0) / 1000.0
        if s > partial_by:
            break
        got = sign * value - momentum
        best = max(best, got)
        if reach is None and got >= need:
            reach = s
    if reach is not None and reach <= done_by:
        ok = "done"
    elif reach is not None or best >= 0.5 * need:
        ok = "partial"
    else:
        ok = "not_done"
    return {
        "part": part,
        "ok": ok,
        "deg": round(best, 1),
        "reach_s": None if reach is None else round(reach, 2),
        "need_deg": need,
    }


def _moment_ok(name: str, span: list[Tick]) -> tuple[bool, float]:
    """(fid:fix) Whether a moment's own motion showed (MOMENT_CHECK), and how big it was."""
    kind, need = MOMENT_CHECK.get(name, ("yaw_any", 10.0))
    if len(span) < 3:
        return False, 0.0
    base = span[0].head
    rows = [_dofs(k.head, base) for k in span]
    if kind == "yaw_any":
        value = max(abs(r["yaw"]) for r in rows)
    elif kind == "roll_any":
        value = max(abs(r["roll"]) for r in rows)
    elif kind == "up_or_roll":
        value = max(max(r["pitch"] for r in rows), max(abs(r["roll"]) for r in rows))
    elif kind == "down":
        value = max(-r["pitch"] for r in rows)
    elif kind == "up":
        value = max(r["pitch"] for r in rows)
    elif kind == "swings":
        # half-swings of at least ``need`` degrees (a turning point needs that much back)
        yaws = [r["yaw"] for r in rows]
        count, extreme, direction = 0, yaws[0], 0
        for y in yaws[1:]:
            if direction >= 0 and y > extreme and direction == 1:
                extreme = y
            elif direction <= 0 and y < extreme and direction == -1:
                extreme = y
            elif direction != 1 and y - extreme >= need:
                count, extreme, direction = count + 1, y, 1
            elif direction != -1 and extreme - y >= need:
                count, extreme, direction = count + 1, y, -1
        return count >= 2, float(count)
    else:  # still: nothing moves over its own first part (0.3–0.9 s: a 定格 lasts ≥ 0.6 s)
        settled = [k for k in span if span[0].t_ms + 300 <= k.t_ms <= span[0].t_ms + 900]
        if not settled or span[-1].t_ms - span[0].t_ms < 900:
            return False, 99.0
        pp = max(
            _deg(max(k.head[i] for k in settled) - min(k.head[i] for k in settled))
            for i in range(4)
        )
        return pp <= need, pp
    return value >= need, value


def _words(audit: Mapping[str, Any], w: _Window) -> tuple[list[str], list[str], list[str]]:
    """What the body did (parts), what it could not do (fails) and did only in part."""
    parts: list[str] = []
    fails: list[str] = []
    partial: list[str] = []
    head: Mapping[str, Mapping[str, Any]] = audit.get("head") or {}
    neck = float(audit.get("neck_deg") or 0.0)
    cmd_neck = float(audit.get("cmd_neck_deg") or 0.0)
    ranked = sorted(head.items(), key=lambda item: -abs(float(item[1].get("deg") or 0.0)))
    visible = [
        (dof, item) for dof, item in ranked if abs(float(item.get("deg") or 0.0)) >= VISIBLE_DEG
    ]
    # what robotd was asked to do most (for 「要牠動 20°」)
    asked = max(
        head.items(), key=lambda item: abs(float(item[1].get("cmd_deg") or 0.0)), default=None
    )
    asked_deg = abs(float(asked[1].get("cmd_deg") or 0.0)) if asked else 0.0
    head_ok = audit.get("head_ok")
    reach = audit.get("v_reach_s")
    missing = list(audit.get("missing") or [])
    named = audit.get("named")
    if named:
        # (fid:fix) the named parts, each in its own words: done / only in part / not made
        first = True
        body_turn = float((audit.get("legs") or {}).get("turn_deg") or 0.0)
        for item in named:
            name = PART_ZH.get(str(item["part"]), str(item["part"]))
            if item.get("room"):
                sign = PARTS[str(item["part"])][1]
                if body_turn * sign >= 5.0:
                    name = msg("neuro.body.part.with_body", part=name)
            deg = float(item.get("deg") or 0.0)
            when = item.get("reach_s")
            if item["ok"] == "done":
                timing = (
                    msg("neuro.body.part.timing", when=max(0.1, float(when)))
                    if first and when is not None
                    else ""
                )
                parts.append(msg("neuro.body.part.done", part=name, deg=deg, timing=timing))
                first = False
            elif item["ok"] == "partial":
                if when is not None:
                    partial.append(
                        msg("neuro.body.part.late", part=name, deg=deg, when=float(when))
                    )
                else:
                    partial.append(msg("neuro.body.part.only", part=name, deg=max(0.0, deg)))
            else:
                fails.append(
                    msg("neuro.body.part.missing", part=name)
                    if deg < 1.0
                    else msg("neuro.body.part.only", part=name, deg=deg)
                )
    elif head_ok == "partial" and missing:
        # the parts it was asked for and did not make, then what it did make
        for dof in missing:
            item = head[dof]
            got = float(item.get("got_deg") or 0.0)
            cmd = float(item.get("cmd_deg") or 0.0)
            name = _name(dof, cmd, cmd_neck)
            partial.append(
                msg(
                    "neuro.body.part.asked.cbad43",
                    part=name,
                    choice=msg("neuro.web.neuro_witness.words.0179ff")
                    if got < 1.0
                    else msg("neuro.web.neuro_witness.words.a63492", got=got),
                    cmd=abs(cmd),
                )
            )
        done_ones = [(dof, item) for dof, item in visible if dof not in missing]
        for index, (dof, item) in enumerate(done_ones[:2]):
            parts.append(_motion(dof, item, neck, reach if index == 0 else None))
    elif head_ok == "done":
        for index, (dof, item) in enumerate(visible[:2]):
            parts.append(_motion(dof, item, neck, reach if index == 0 else None))
    elif head_ok == "partial":
        if visible and reach is not None and float(audit.get("v_peak") or 0.0) >= 1.0:
            dof, item = visible[0]
            partial.append(
                _motion(dof, item, neck, extra=msg("neuro.body.detail.late", reach=reach))
            )
        elif visible:
            dof, item = visible[0]
            asked_note = (
                msg("neuro.body.detail.asked", asked_deg=asked_deg)
                if asked_deg >= VISIBLE_DEG
                else ""
            )
            partial.append(
                _motion(
                    dof,
                    item,
                    neck,
                    extra=lst(
                        (b for b in (msg("neuro.body.detail.small"), asked_note) if b),
                        "fmt.sep.comma",
                    ),
                )
            )
        else:
            partial.append(msg("neuro.body.head.tiny"))
    elif head_ok == "not_done":
        if asked is not None and asked_deg >= VISIBLE_DEG:
            dof, item = asked
            got = float(item.get("got_deg") or 0.0)
            name = _name(dof, float(item.get("cmd_deg") or 0.0), cmd_neck)
            fails.append(
                msg(
                    "neuro.body.part.asked.be7a57",
                    part=name,
                    choice=msg("neuro.web.neuro_witness.words.0179ff")
                    if got < 1.0
                    else msg("neuro.web.neuro_witness.words.a63492", got=got),
                    asked_deg=asked_deg,
                )
            )
        else:
            fails.append(msg("neuro.body.head.none_not_asked"))
    elif w.reflex == "freeze":
        # (fid:fix) how the head moved into the freeze pose, then (below) whether it held still
        into = audit.get("freeze_into") or {}
        up = float(into.get("pitch") or 0.0)
        turn = float(into.get("yaw") or 0.0)
        if up >= VISIBLE_DEG:
            parts.append(msg("neuro.body.freeze.rose", up=up))
        elif up <= -VISIBLE_DEG:
            parts.append(msg("neuro.body.freeze.lowered", up=-up))
        if abs(turn) >= TURN_DEG:
            parts.append(
                msg(
                    "neuro.body.freeze.turned",
                    side=msg("neuro.side.left_short")
                    if turn > 0
                    else msg("neuro.side.right_short"),
                    turn=abs(turn),
                )
            )
    else:
        for dof, item in visible[:2]:
            parts.append(_motion(dof, item, neck))
    freeze = audit.get("freeze_ok")
    pp = audit.get("freeze_pp_deg")
    if freeze == "done":
        parts.append(msg("neuro.body.freeze.held"))
    elif freeze == "partial" and audit.get("freeze_short"):
        held = audit.get("freeze_held_s")
        partial.append(
            msg("neuro.body.freeze.short_s", held=float(held))
            if held is not None
            else msg("neuro.body.freeze.short")
        )
    elif freeze == "partial":
        partial.append(msg("neuro.body.freeze.mostly", pp=float(pp or 0)))
    elif freeze == "not_done":
        fails.append(msg("neuro.body.freeze.not_held", pp=float(pp or 0)))
    legs: Mapping[str, Any] = audit.get("legs") or {}
    ok = legs.get("ok")
    still = legs.get("still_s")
    held = float(legs.get("held_s") or 0.0)
    if ok == "done":
        parts.append(msg("neuro.body.legs.stopped", still=float(still or 0)))
    elif ok == "partial":
        if still is not None and float(still) > STOP_BY_S:
            partial.append(msg("neuro.body.legs.stopped_late", still=float(still)))
        else:
            partial.append(msg("neuro.body.legs.stopped_briefly", held=held))
    elif ok == "not_done":
        fails.append(
            msg("neuro.body.legs.not_stopped")
            if legs.get("stop_cmd")
            else msg("neuro.body.legs.not_stopped_no_cmd")
        )
    forward = legs.get("forward_m")
    if forward is not None and forward <= -STEP_BACK_M:
        parts.append(msg("neuro.body.legs.back_cm", forward=abs(forward) * 100))
    elif forward is not None and legs.get("steps_away") and forward >= STEP_BACK_M:
        parts.append(msg("neuro.body.legs.away_cm", forward=forward * 100))
    turned = legs.get("turn_deg")
    if turned is not None and abs(turned) >= TURN_DEG:
        parts.append(
            msg(
                "neuro.body.legs.turned",
                side=msg("neuro.side.left_short") if turned > 0 else msg("neuro.side.right_short"),
                turned=abs(turned),
            )
        )
    drop = legs.get("drop_m")
    if drop is not None and drop <= -CROUCH_M:
        parts.append(msg("neuro.body.legs.crouched", drop=abs(drop) * 100))
    return parts, fails, partial


def _sentence(verdict: str | None, parts: list[str], fails: list[str], partial: list[str]) -> str:
    """The row's line: 「身體：…」, 「身體只做到一部分：…」 or 「身體做不到：…」."""
    if verdict == "not_done":
        return msg("neuro.body.verdict.not_done", fails=lst(fails + partial + parts))
    if verdict == "partial":
        return msg("neuro.body.verdict.partial", fails=lst(fails + partial + parts))
    if parts:
        return msg("neuro.body.verdict.done", parts=lst(parts))
    return msg("neuro.body.verdict.nothing_seen")


__all__ = [
    "HALTING",
    "Tick",
    "WINDOW_MS",
    "WITNESS_VERSION",
    "Witness",
    "gesture_index",
    "head_words",
]
