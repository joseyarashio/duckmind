"""duck-body's side of the Stage-1 skills: which entry, when it may start, how often.

Pure standard library (board-safe: no simulator, no physics, no Duck Core), so the neural duck's
body link and, later, duck-body on the real duck can run it as is.

* :func:`startle_entry` / :class:`SkillPicker` — the brain's act and its size → a table entry
  (training plan §2: "the brain picks the entry from the reflex strength").
* :class:`SkillGuard` — the start rule and the wear budget:

  - a skill starts only from a posture it was trained and gated for (startle, freeze,
    hop-back: standing or walking; shake and the standing stretch: standing; the seated
    stretch: seated) and only when upright (tilt < 20°), not held by a hand, not docked, not
    fallen or righting, and no other skill running. This is not caution for its own sake: the
    Mac gate showed robotd runs a skill network on a seated duck if asked (``start_skill`` does
    not look at the posture), and a network trained only from standing entry states flips a
    seated duck (the v5 stand-in fell 1/1 on both servo models, 2026-09-28). Outside its
    validated envelope a policy gets a hard limit (owner 2026-09-28: relax everything else);
  - **(s1b:fix) walking only as fast as it was gated**: a rule's ``max_walk_mps`` (from the
    table's verdict: startle / freeze from a walk only ≤ 0.40 m/s where only 0.40 passed) against
    the body's forward speed (``BodyView.vx``, the applied twist, or a measured trunk speed);
  - **(s1b:fix) a settle time**: comfort skills and the hop-back only once the posture has held
    ``settle_s`` (1.5 s standing — after a rise, a stop, another skill; 3 s seated — after the
    sit-down glide) with the trunk still (gyro under 0.35 rad/s for 0.5 s, when the body reports
    it); startles and freezes ``rise_settle_s`` (1.5 s) after a rise. The A100's comfort falls
    were all in "risen" entries, 0.1–1.0 s after velstand took over (6/384 against 0/896); the
    Mac gate (s1b:fix, XML servos + 20 ms, DR, n = 30 each) measured it for every family:
    0.1–1.0 s after the rise startle_4 fell 4/30, shake_2 2/29, the hop 2/29 and the stretch
    showed in 6/30 — 1.6–2.2 s after it all 28–30/30 with 0 falls;
  - **hourly caps** per skill and for all of them together, and a minimum gap between two runs
    of the same skill, against gearbox wear (owner decision D3, recommended; the XL330's
    gearbox is what breaks). Provisional numbers (:data:`DEFAULT_RULES`); :func:`caps_from_strain`
    lowers a cap when that skill's strain gate is marginal and removes the skill when it fails;
  - **(s1b:fix) held crouches**: a hold entry (``Rule.hold``) re-requested while it runs is a
    refresh — allowed, not counted; one may take over a running startle (``Rule.continues``: the
    freeze continues the startle's crouch instead of the duck rising and crouching again);
    ``startle_release`` ends a running startle-family crouch;
  - :meth:`SkillGuard.watch` — a running skill whose trunk tilts past 30° is handed straight
    back to velstand with ``skill_abort`` (robotd's own fall reflex is off while any skill
    runs; velstand owns righting). Once per run. It also keeps the posture history the settle
    rule reads (:meth:`SkillGuard.observe`): the body link calls it every tick.

  - **(startle-hop, 2026-10-10) a startle-linked hop-back** (``request(..., chain=True)``): the back-steps that
    follow a scare. They wait for the stop, not for the 1.5 s settle (posture stand, trunk speed under
    :data:`CHAIN_SPEED_MPS`, tilt under :data:`CHAIN_TILT_DEG`) and count at half against the hourly caps
    (:data:`CHAIN_WEIGHT`). Everything else (held, docked, fallen, righting, the posture, the gap) is as for any run.

Reasons come in plain words for the page (``reason_zh``) and as codes for logs.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import lst, msg
from duckmind.skilltrain.table import LIVE_FREEZE, LIVE_STARTLE

ABORT = "skill_abort"
RELEASE = "startle_release"
BOW_RELEASE = "bow_release"
# the entry that ends a held run, and the families it may end (skill_abort ends any)
RELEASES: dict[str, tuple[str, ...]] = {RELEASE: ("startle", "freeze"), BOW_RELEASE: ("lie",)}
WINDOW_S = 3600.0
MAX_TILT_DEG = 20.0
ABORT_TILT_DEG = 30.0
GLOBAL_PER_HOUR = 60
STRAIN_MARGIN_NM = 0.30  # pooled torque p95 above this (gate 0.35) halves a cap
STALL_MARGIN = 0.003  # stall share above this (gate 0.005) halves a cap
SPEED_TOL_MPS = 0.02  # a walk commanded at 0.40 runs at 0.40 ± a little
STILL_GYRO_RAD_S = 0.35  # the trunk still: |gyro| under this …
STILL_FOR_S = 0.5  # … for this long (the settle rule, when the body reports its gyro)
STILL_SPEED_MPS = 0.03  # a measured trunk speed under this: not coasting
TOP_GATED_MPS = 0.55  # the fastest walk the Mac gates tried a skill from
CHAIN_SPEED_MPS = 0.05  # a startle-linked hop-back starts once the trunk is slower than this …
CHAIN_TILT_DEG = 8.0  # … and upright to this …
RISE_SETTLE_S = 1.5  # (the s1b:fix rule) no hop-back within this long after a rise, scare or not
CHAIN_WEIGHT = 0.5  # … and it counts this much against the hourly caps (a hop that follows a scare)


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    postures: frozenset[str]
    per_hour: int | None  # None: uncapped (release / abort only end a skill)
    min_gap_s: float = 0.0
    note: str = ""
    # (s1b:fix, additive; the defaults keep a rule's old meaning)
    max_walk_mps: float | None = None  # from a walk only at |vx| ≤ this (None: no limit)
    settle_s: float = 0.0  # the current posture held this long first
    rise_settle_s: float = 0.0  # … and this long since the last rise
    hold: bool = False  # re-requested while it runs (a refresh, not a new run)
    continues: frozenset[str] = frozenset()  # families it may take over from while they run
    # the tilt past which a running skill is handed back to velstand (None: the guard's 30°). The
    # front-down skills tip the trunk 25–33° on purpose (the bow, lying low): they get 50°.
    abort_tilt_deg: float | None = None
    # (2026-10-09) the panic entries: one flight is one run however its entry changes (a straight run into a
    # left arc and out of it): a takeover inside the family is free — not counted, no gap, no cap
    free_continue: bool = False
    # (2026-10-09, live bug) a flight is not wear-budgeted: the brain's walk blinks (a look before it goes on, a reflex) and
    # the same flight is asked for again a few ticks after its abort; counting each ask locked it out (the 3 s gap)
    counted: bool = True


def _rule(
    name: str,
    family: str,
    postures: Iterable[str],
    per_hour: int | None,
    gap: float,
    note: str = "",
    **more: Any,
) -> Rule:
    if "continues" in more:
        more["continues"] = frozenset(more["continues"])
    return Rule(name, family, frozenset(postures), per_hour, gap, note, **more)


# Provisional caps (per rolling hour) and gaps: small startles are frequent and cheap; the
# deep crouch, the freezes, the hop-back and the shake load the gearboxes most (plan §7 risk 5).
# The in-between levels (``*b``, s1b:fix) share the cap of the level below them.
_WALKING = ("stand", "walk")
_STARTLE: dict[str, Any] = {"max_walk_mps": TOP_GATED_MPS, "rise_settle_s": 1.5}
_HOLD: dict[str, Any] = {
    "max_walk_mps": TOP_GATED_MPS,
    "rise_settle_s": 1.5,
    "hold": True,
    "continues": ("startle", "freeze"),
}
_COMFORT: dict[str, Any] = {"settle_s": 1.5}
PANIC_ENTRIES = ("panic_run", "panic_arc_left", "panic_arc_right")
# the quick turn on the spot (queue-2026-10-09, QUICK-TURN-2026-10-09.md): a chaining skill under a constant yaw command
# (+ left), held by the body link while the brain asks for a pure turn, handed back to v5 with skill_abort when it stops
QUICK_TURN_ENTRIES = ("quick_turn_left", "quick_turn_right")
QUICK_TURN_MIN_YAW = 0.8  # the brain asks for a pure turn of at least this rad/s: the quick turn; slower turns stay v5's
_QUICK_TURN: dict[str, Any] = {
    "max_walk_mps": 0.50,
    "rise_settle_s": 1.5,
    "hold": True,
    "free_continue": True,
    "counted": False,
}
_PANIC: dict[str, Any] = {
    "max_walk_mps": 0.60,
    "rise_settle_s": 1.5,
    "hold": True,
    "free_continue": True,
    "counted": False,
}
# 2026-10-05 (trained-skills wiring): the B skills (R1 step-b3 / bow-b3, skillsB mat) and the forward
# hop-over (R1 hop-h3), as their Mac gates allow them: turns from standing or a walk up to 0.40 m/s (the
# gate's walk entry), the rest standing after a settle; the hourly caps are halved where the strain
# gate is over or marginal (bow, lie_low, hop-over, the mat steps).
_B_STAND: dict[str, Any] = {"settle_s": 1.5, "rise_settle_s": 1.5}
_B_TURN: dict[str, Any] = {"max_walk_mps": 0.40, "rise_settle_s": 1.5, "hold": True}
_B_LOW: dict[str, Any] = {"abort_tilt_deg": 50.0}
# the official forward roll (2026-10-26, ROULADE-2026-10-22.md §2 and §5): standing only, settled, rarely (the roll is twice over the strain
# gate: caps halved again) — and a roll passes every tilt, so the 30 degree watch must never hand it back to velstand (360 never fires)
ROLL_ENTRY = "roulade"
_ROLL: dict[str, Any] = {"settle_s": 1.5, "rise_settle_s": 1.5, "abort_tilt_deg": 360.0}
# the forward hop-overs (sill-circling 2026-10-28) lean past the 30 degree watch when they cross cleanly: hop_over_2 to 36.6 degrees
# at most in 102 clean hops at real_home's 20 mm sill (the brain's own crossings, MuJoCo) and to 41.8 in 54 of 55 clean body-test
# hops (0 and 30 degrees off square; one recovered from 65), hop_over_4 to 66.4 at a 40 mm sill (44 clean); a fall goes on past
# 70–150. Each is handed back to velstand only past its own lean (the 30 degree watch cut 57 of 172 hops short; 31 of those fell).
HOP_OVER_ABORT_TILT_DEG = {"hop_over_2": 45.0, "hop_over_4": 70.0}
DEFAULT_RULES: dict[str, Rule] = {
    r.name: r
    for r in (
        _rule("startle_1", "startle", _WALKING, 30, 2.0, "flinch", **_STARTLE),
        _rule("startle_1b", "startle", _WALKING, 30, 2.0, **_STARTLE),
        _rule("startle_2", "startle", _WALKING, 30, 2.0, **_STARTLE),
        _rule("startle_2b", "startle", _WALKING, 30, 2.0, **_STARTLE),
        _rule("startle_3", "startle", _WALKING, 12, 5.0, **_STARTLE),
        _rule("startle_3b", "startle", _WALKING, 12, 5.0, **_STARTLE),
        _rule("startle_4", "startle", _WALKING, 8, 10.0, "deep crouch", **_STARTLE),
        _rule("freeze_1b", "freeze", _WALKING, 6, 20.0, "held crouch", **_HOLD),
        _rule("freeze_2", "freeze", _WALKING, 6, 20.0, "held crouch", **_HOLD),
        _rule("freeze_2b", "freeze", _WALKING, 6, 20.0, "held crouch", **_HOLD),
        _rule("freeze_3", "freeze", _WALKING, 6, 20.0, "held crouch", **_HOLD),
        _rule("freeze_3b", "freeze", _WALKING, 4, 30.0, "held crouch", **_HOLD),
        _rule("freeze_4", "freeze", _WALKING, 4, 30.0, "held deep crouch", **_HOLD),
        _rule(
            "hop_back",
            "hop_back",
            _WALKING,
            4,
            60.0,
            "quick back-steps (D5)",
            **_COMFORT,
            max_walk_mps=0.40,
        ),  # fmt: skip
        _rule("shake_1", "shake", ("stand",), 8, 60.0, **_COMFORT),
        _rule("shake_2", "shake", ("stand",), 4, 120.0, **_COMFORT),
        _rule("stretch_tall", "stretch_tall", ("stand",), 6, 120.0, **_COMFORT),
        _rule("stretch_sit", "stretch_sit", ("sit",), 6, 120.0, settle_s=3.0),
        _rule(RELEASE, "release", ("skill",), None, 0.0, "ends a running crouch"),
        _rule(ABORT, "abort", ("skill",), None, 0.0, "hands a running skill to velstand"),
        _rule("turn_left", "turn", _WALKING, 30, 2.0, "turn on the spot", **_B_TURN),
        _rule("turn_right", "turn", _WALKING, 30, 2.0, "turn on the spot", **_B_TURN),
        _rule(
            "bow",
            "bow",
            ("stand",),
            6,
            60.0,
            "play bow (strain over: halved)",
            **_B_STAND,
            **_B_LOW,
        ),
        _rule("lie_low", "lie", ("stand",), 4, 90.0, "lie low", hold=True, **_B_STAND, **_B_LOW),
        _rule(BOW_RELEASE, "release_bow", ("skill",), None, 0.0, "ends lie_low", **_B_LOW),
        _rule(
            ROLL_ENTRY, "roulade", ("stand",), 4, 120.0,
            "forward roll (the flip trick): standing only, a roll passes every tilt", **_ROLL,
        ),  # fmt: skip
        _rule(
            "hop_over_2",
            "hop_over",
            _WALKING,
            10,
            20.0,
            "step over a low strip",
            max_walk_mps=0.50,
            rise_settle_s=1.5,
            abort_tilt_deg=HOP_OVER_ABORT_TILT_DEG["hop_over_2"],
        ),
        _rule(
            "hop_over_4",
            "hop_over",
            _WALKING,
            6,
            30.0,
            "step over a higher strip",
            max_walk_mps=0.50,
            rise_settle_s=1.5,
            abort_tilt_deg=HOP_OVER_ABORT_TILT_DEG["hop_over_4"],
        ),
        # the panic network as a robotd skill (2026-10-09): a flight, stood or walked into at <= 0.55 m/s
        # (the hand-off gate H1: walking at 0.40), held while the brain flees. The cap is the owner's
        # fear-flight rate, not wear: flights are rare and a refusal means v5 runs it at 0.55.
        *(
            _rule(n, "panic", _WALKING, None, 0.0, "panic flight (the panic network)", **_PANIC)
            for n in PANIC_ENTRIES
        ),
        *(
            _rule(n, "quick_turn", _WALKING, None, 0.0, "quick turn on the spot", **_QUICK_TURN)
            for n in QUICK_TURN_ENTRIES
        ),
    )
}

REASONS_ZH = {
    "unknown": msg("neuro.skill.reason.unknown"),
    "posture": msg("neuro.skill.reason.posture_any"),
    "speed": msg("neuro.skill.reason.speed"),
    "settle": msg("neuro.skill.reason.settle"),
    "tilt": msg("neuro.skill.reason.tilt"),
    "held": msg("neuro.skill.reason.held"),
    "docked": msg("neuro.skill.reason.docked"),
    "fallen": msg("neuro.skill.reason.fallen"),
    "busy": msg("neuro.skill.reason.busy"),
    "gap": msg("neuro.skill.reason.gap"),
    "hourly": msg("neuro.skill.reason.hourly"),
    "total": msg("neuro.skill.reason.total"),
    "nothing": msg("neuro.skill.reason.nothing"),
    "ok": msg("neuro.skill.reason.ok"),
    "hold": msg("neuro.skill.reason.hold"),
    "continue": msg("neuro.skill.reason.continue"),
}
POSTURE_ZH = {
    "stand": msg("neuro.skill.posture.stand"),
    "walk": msg("neuro.skill.posture.walk"),
    "sit": msg("neuro.skill.posture.sit"),
}


def posture_reason(postures: Iterable[str], now: str) -> Any:
    """「現在的姿勢不行」 naming where the motion works and what the duck is doing now."""
    need = [POSTURE_ZH[p] for p in ("stand", "walk", "sit") if p in set(postures)]
    here = POSTURE_ZH.get(now)
    if need and here is not None:
        return msg("neuro.skill.reason.posture", need=lst(need), now=here)
    return REASONS_ZH["posture"]


@dataclass(frozen=True)
class BodyView:
    """What the guard needs from ``robot.state`` and the body link, nothing else.

    ``label``: robot.state's ``policy`` (``walk``, ``stand``, ``sit``, ``rise``, a skill name or
    ``name:unwind``); ``twist``: |the applied twist| (walking vs standing under velstand).
    (s1b:fix, optional) ``vx``: the applied forward speed (m/s); ``speed_mps``: a measured trunk
    speed; ``gyro_rad_s``: |the trunk's angular rate| (the IMU)."""

    label: str
    twist: float = 0.0
    tilt_deg: float = 0.0
    fallen: bool = False
    righting: bool = False
    held: bool = False
    docked: bool = False
    vx: float | None = None
    speed_mps: float | None = None
    gyro_rad_s: float | None = None


@dataclass(frozen=True)
class Decision:
    ok: bool
    skill: str | None
    reason: str
    reason_zh: str


def posture_of(view: BodyView, skills: Iterable[str] = ()) -> str:
    """stand | walk | sit | rise | skill | other, from robotd's own label."""
    label = view.label or ""
    base = label.split(":", 1)[0]
    if base in set(skills) or label.endswith(":unwind"):
        return "skill"
    if label in ("walk", "stand"):
        return "walk" if view.twist > 0.05 else "stand"
    if label == "sit":
        return "sit"
    if label == "rise":
        return "rise"
    return "other"


class SkillGuard:
    """Start rule + wear budget for the trained skills robotd actually loaded."""

    def __init__(
        self,
        available: Iterable[str],
        rules: Mapping[str, Rule] | None = None,
        *,
        global_per_hour: int = GLOBAL_PER_HOUR,
        max_tilt_deg: float = MAX_TILT_DEG,
        abort_tilt_deg: float = ABORT_TILT_DEG,
        window_s: float = WINDOW_S,
    ) -> None:
        base = dict(rules if rules is not None else DEFAULT_RULES)
        self.names = frozenset(n for n in available if n in base)
        self.rules = {n: base[n] for n in self.names}
        self.global_per_hour = int(global_per_hour)
        self.max_tilt_deg = float(max_tilt_deg)
        self.abort_tilt_deg = float(abort_tilt_deg)
        self.window_s = float(window_s)
        self.history: dict[str, deque[float]] = {n: deque() for n in self.names}
        self.all: deque[float] = deque()
        # (startle-hop) the runs in those lists that followed a scare count CHAIN_WEIGHT, not 1
        self.chained: dict[str, deque[float]] = {n: deque() for n in self.names}
        self.all_chained: deque[float] = deque()
        self.aborted_label: str | None = None
        self.refusals: dict[str, int] = {}
        # (s1b:fix) what the body did lately: the posture and since when, the last rise, the gyro
        self.posture: str | None = None
        self.posture_since: float | None = None
        self.last_rise: float | None = None
        self.gyro: deque[tuple[float, float]] = deque()
        self.last_seen: float | None = None

    def _trim(self, now_s: float) -> None:
        edge = now_s - self.window_s
        for runs in (*self.history.values(), self.all, *self.chained.values(), self.all_chained):
            while runs and runs[0] <= edge:
                runs.popleft()

    def _no(self, skill: str | None, code: str, words: Any = None) -> Decision:
        self.refusals[code] = self.refusals.get(code, 0) + 1
        return Decision(False, skill, code, REASONS_ZH[code] if words is None else words)

    def observe(self, view: BodyView, now_s: float) -> str:
        """Keep the posture history (the settle rule's input); returns the posture now."""
        posture = posture_of(view, self.names)
        if self.last_seen is not None and now_s < self.last_seen:
            # the clock went back (another duck, a restart): nothing we saw applies
            self.posture = self.posture_since = self.last_rise = None
            self.gyro.clear()
        if posture != self.posture:
            self.posture, self.posture_since = posture, now_s
        if posture == "rise":
            self.last_rise = now_s
        if view.gyro_rad_s is not None and math.isfinite(view.gyro_rad_s):
            self.gyro.append((now_s, float(view.gyro_rad_s)))
        while self.gyro and self.gyro[0][0] < now_s - 2.0 * STILL_FOR_S:
            self.gyro.popleft()
        self.last_seen = now_s
        return posture

    def settled(self, rule: Rule, view: BodyView, now_s: float, *, chain: bool = False) -> bool:
        """The rule's settle time is over: the posture has held ``settle_s`` (and the trunk is
        still, when the body reports its gyro / speed), and the last rise is ``rise_settle_s``
        ago. A posture first seen now counts as just begun. ``chain`` (a hop-back that follows a
        scare): no posture time, but the trunk must be slow (``speed_mps`` under
        :data:`CHAIN_SPEED_MPS`, which the caller must give) and upright (:data:`CHAIN_TILT_DEG`)."""
        since = self.posture_since
        if chain:
            if view.speed_mps is None or not math.isfinite(view.speed_mps):
                return False
            if view.speed_mps >= CHAIN_SPEED_MPS or not view.tilt_deg < CHAIN_TILT_DEG:
                return False
        elif rule.settle_s > 0.0:
            if since is None or now_s - since < rule.settle_s - 1e-9:
                return False
            recent = [g for t, g in self.gyro if t >= now_s - STILL_FOR_S - 1e-9]
            if recent and max(recent) >= STILL_GYRO_RAD_S:
                return False
            if view.speed_mps is not None and view.speed_mps >= STILL_SPEED_MPS:
                return False
        rise_s = max(rule.rise_settle_s, RISE_SETTLE_S) if chain else rule.rise_settle_s
        if rise_s > 0.0 and self.last_rise is not None:
            if now_s - self.last_rise < rise_s - 1e-9:
                return False
        return True

    def running(self, view: BodyView) -> str | None:
        """The table entry robotd runs now (None: none of ours)."""
        base = (view.label or "").split(":", 1)[0]
        return base if base in self.names else None

    def _speed(self, view: BodyView) -> float:
        speeds = [abs(view.vx) if view.vx is not None else float(view.twist)]
        if view.speed_mps is not None and math.isfinite(view.speed_mps):
            speeds.append(float(view.speed_mps))
        return max(speeds)

    def _count(self, skill: str, rule: Rule, now_s: float) -> Decision | None:
        """The wear budget (None: within it). The runs that followed a scare count CHAIN_WEIGHT."""
        if not rule.counted:
            return None
        self._trim(now_s)
        runs = self.history[skill]
        if runs and now_s - runs[-1] < rule.min_gap_s:
            return self._no(skill, "gap")
        used = len(runs) - (1.0 - CHAIN_WEIGHT) * len(self.chained[skill])
        if rule.per_hour is not None and used >= rule.per_hour - 1e-9:
            return self._no(skill, "hourly")
        if (
            len(self.all) - (1.0 - CHAIN_WEIGHT) * len(self.all_chained)
            >= self.global_per_hour - 1e-9
        ):
            return self._no(skill, "total")
        return None

    def _record(self, skill: str, now_s: float, chain: bool) -> None:
        self.history[skill].append(now_s)
        self.all.append(now_s)
        if chain:
            self.chained[skill].append(now_s)
            self.all_chained.append(now_s)

    def request(
        self, skill: str, view: BodyView, now_s: float, *, manual: bool = False, chain: bool = False
    ) -> Decision:
        """May ``skill`` start (or go on) now? Counts it when it starts. ``manual``: a person
        pressed the page's button — the wear caps (hourly, gap, total: they protect the real
        duck's gearboxes) do not apply and the run is not counted; every safety rule does.
        ``chain``: the hop-back that follows a scare (see :meth:`settled`; counted at half)."""
        posture = self.observe(view, now_s)
        rule = self.rules.get(skill)
        if rule is None:
            return self._no(skill, "unknown")
        running = self.running(view)
        if skill == ABORT:
            if posture != "skill":
                return self._no(skill, "nothing")
            return Decision(True, skill, "ok", REASONS_ZH["ok"])
        if skill in RELEASES:
            fam = self.rules[running].family if running else None
            if posture != "skill" or fam not in RELEASES[skill]:
                return self._no(skill, "nothing")
            return Decision(True, skill, "ok", REASONS_ZH["ok"])
        if view.held:
            return self._no(skill, "held")
        if view.docked:
            return self._no(skill, "docked")
        if view.fallen or view.righting:
            return self._no(skill, "fallen")
        if posture == "skill":
            if rule.hold and running == skill:
                return Decision(True, skill, "hold", REASONS_ZH["hold"])  # a refresh: not a run
            fam = self.rules[running].family if running else None
            if rule.free_continue and fam == rule.family:
                return Decision(True, skill, "hold", REASONS_ZH["hold"])  # inside one flight
            if not (rule.hold and fam in rule.continues):
                return self._no(skill, "busy")
            if not (math.isfinite(view.tilt_deg) and view.tilt_deg < self.max_tilt_deg):
                return self._no(skill, "tilt")
            capped = None if manual else self._count(skill, rule, now_s)
            if capped is not None:
                return capped
            if not manual and rule.counted:
                self._record(skill, now_s, False)
            return Decision(True, skill, "continue", REASONS_ZH["continue"])
        if posture not in rule.postures:
            return self._no(skill, "posture", posture_reason(rule.postures, posture))
        if (
            posture == "walk"
            and rule.max_walk_mps is not None
            and self._speed(view) > rule.max_walk_mps + SPEED_TOL_MPS
        ):
            return self._no(skill, "speed")
        if not (math.isfinite(view.tilt_deg) and view.tilt_deg < self.max_tilt_deg):
            return self._no(skill, "tilt")
        if not self.settled(rule, view, now_s, chain=chain):
            return self._no(skill, "settle")
        capped = None if manual else self._count(skill, rule, now_s)
        if capped is not None:
            return capped
        if not manual and rule.counted:
            self._record(skill, now_s, chain)
        return Decision(True, skill, "ok", REASONS_ZH["ok"])

    def watch(self, view: BodyView, now_s: float) -> str | None:
        """``skill_abort`` when a running skill tilts past ``abort_tilt_deg`` (once per run).
        Also keeps the posture history (:meth:`observe`): call it every tick."""
        posture = self.observe(view, now_s)
        if ABORT not in self.names:
            return None
        if posture != "skill":
            self.aborted_label = None
            return None
        run = view.label.split(":", 1)[0]
        rule = self.rules.get(run)
        limit = (
            rule.abort_tilt_deg if rule is not None and rule.abort_tilt_deg else self.abort_tilt_deg
        )
        if view.tilt_deg > limit and self.aborted_label != run:
            self.aborted_label = run
            return ABORT
        return None

    def counts(self, now_s: float) -> dict[str, Any]:
        self._trim(now_s)
        return {
            "per_skill": {n: len(r) for n, r in sorted(self.history.items())},
            "chained": {n: len(r) for n, r in sorted(self.chained.items()) if r},
            "all": len(self.all),
            "caps": {n: self.rules[n].per_hour for n in sorted(self.names)},
            "global_cap": self.global_per_hour,
            "refusals": dict(self.refusals),
        }


# ── the brain's act → a table entry ──────────────────────────────────────────────────────────


def panic_names(names: Iterable[str]) -> frozenset[str]:
    """The panic flight entries among ``names`` (the table's, as robotd loaded them)."""
    return frozenset(n for n in names if n in PANIC_ENTRIES)


def panic_running(names: Iterable[str], label: str | None) -> bool:
    """robotd's label says a panic entry runs (``panic_run``; ``panic_run:unwind`` too)."""
    return (label or "").split(":", 1)[0] in panic_names(names)


def quick_turn_names(names: Iterable[str]) -> frozenset[str]:
    """The quick-turn entries among ``names`` (the table's, as robotd loaded them)."""
    return frozenset(n for n in names if n in QUICK_TURN_ENTRIES)


def quick_turn_running(names: Iterable[str], label: str | None) -> bool:
    """robotd's label says a quick-turn entry runs."""
    return (label or "").split(":", 1)[0] in quick_turn_names(names)


def quick_turn_entry(vyaw: float, names: Iterable[str]) -> str | None:
    """The quick-turn entry for a turn of ``vyaw`` rad/s (+ = left), None when the table has none."""
    have = quick_turn_names(names)
    pick = "quick_turn_left" if vyaw > 0.0 else "quick_turn_right"
    return pick if pick in have else None


ARC_ENTER_RAD_S = 0.45  # an arc entry from this yaw rate on …
ARC_LEAVE_RAD_S = 0.25  # … and back to the straight run below this (no chatter)
# The arc entries stay in the table but the live path does not ask for them: on the Mac live stack a flight
# that started as an arc and was handed back to v5 mid-turn tipped the duck (2 of 6 fresh boots: righting / fall),
# while the straight run never did. Headless H1 passes for the arcs, so this is a live-path choice, not a gate change.
ARCS_ON = False


def panic_entry(vyaw: float, names: Iterable[str], running: str | None = None) -> str | None:
    """The panic entry for a flight turning at ``vyaw`` (rad/s, + = left): the straight run, or the arc of that
    side; with hysteresis around the entry that runs now (``running``: robotd's label); None when the table
    has no panic entry at all."""
    have = panic_names(names)
    if not have:
        return None
    now = (running or "").split(":", 1)[0]
    sign = "panic_arc_left" if vyaw > 0.0 else "panic_arc_right"
    if not ARCS_ON:
        return "panic_run" if "panic_run" in have else next(iter(sorted(have)))
    if now in ("panic_arc_left", "panic_arc_right"):
        same = (now == "panic_arc_left") == (vyaw > 0.0)
        wants_arc = abs(vyaw) >= (ARC_LEAVE_RAD_S if same else ARC_ENTER_RAD_S)
    else:
        wants_arc = abs(vyaw) >= ARC_ENTER_RAD_S
    pick = sign if wants_arc and sign in have else "panic_run"
    return pick if pick in have else next(iter(sorted(have)))


def startle_entry(strength: float, *, lasting: bool = False, minimum: float = 0.15) -> str | None:
    """The four-level startle family's entry for a reflex of ``strength`` (0..1): none below
    ``minimum`` (the head program alone shows a small start), then ``startle_1…4``; a lasting
    strong fear (≥ 0.6) holds as ``freeze_3`` / ``freeze_4``. (The picker's finer grid:
    :func:`level_entry`.)"""
    s = float(strength)
    if not math.isfinite(s) or s < minimum:
        return None
    if lasting and s >= 0.6:
        return "freeze_4" if s >= 0.85 else "freeze_3"
    if s < 0.35:
        return "startle_1"
    if s < 0.6:
        return "startle_2"
    if s < 0.85:
        return "startle_3"
    return "startle_4"


STARTLE_ORDER = tuple(n for n, _ in LIVE_STARTLE)
FREEZE_ORDER = tuple(n for n, _ in LIVE_FREEZE)
LEVEL = {**dict(LIVE_STARTLE), **dict(LIVE_FREEZE)}


def lower(entry: str) -> str | None:
    """The next smaller entry of the same family (a capped big startle shows a smaller one;
    a capped hold a shallower hold)."""
    for order in (STARTLE_ORDER, FREEZE_ORDER):
        if entry in order:
            i = order.index(entry)
            return order[i - 1] if i > 0 else None
    return {"shake_2": "shake_1"}.get(entry)


def level_entry(
    k: float, names: Iterable[str], *, at_least: bool = False
) -> tuple[str, float] | None:
    """The entry of ``names`` (with a known level) nearest to ``k`` — or, ``at_least``, the
    shallowest one at or above ``k`` (a crouch that goes on never gets shallower); None when
    none has a level."""
    have = sorted((LEVEL[n], n) for n in names if n in LEVEL)
    if not have:
        return None
    if at_least:
        deeper = [(lv, n) for lv, n in have if lv >= k - 1e-9]
        lv, n = deeper[0] if deeper else have[-1]
        return n, lv
    lv, n = min(have, key=lambda item: (abs(item[0] - k), -item[0]))
    return n, lv


# The brain's own startle → freeze rule (neuro/reflex.py: a startle's then = "freeze" when its
# strength is over 0.3 or fear over 0.15): such a startle is followed by a freeze, so its crouch
# is held from the start instead of crouching, rising and crouching again.
LASTING_STRENGTH = 0.3
FEAR_REFLEXES = frozenset({"startle", "freeze"})
LEVEL_CV = 0.15  # per-execution variation of the startle depth (ethology: CV 0.1–0.3)
HOLD_CV = 0.25  # … and of the hold's own minimum length
HOLD_MIN_S, HOLD_MAX_S = 0.8, 10.0  # a held crouch lasts 0.8–10 s (freeze: 1–10 s, median 3)
HOLD_TAIL_S = 0.4  # the crouch stays this long after the brain's fear ends, then the rise
CAP_MEDIAN_S = (2.5, 4.5)  # a crouch's own length: median at a weak / the strongest fear
CAP_CV = 0.3
CONFIRM_S = 0.3  # a hold not running by then was refused: the episode ends
NO_BOB_S = 3.0  # a freeze arriving this soon after a crouch ended starts no second crouch
ONESHOT_S = 1.3  # a one-shot startle's window (0.5 s hold + 0.8 s unwind)
EVENT_SAME_S = 0.6  # a startle onset this soon after its event is the same startle


def _lognormal(rng: random.Random, median: float, cv: float) -> float:
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    return median * math.exp(rng.gauss(0.0, sigma))


@dataclass
class Episode:
    """One held crouch: the entry re-requested every tick until ``until`` (extended while the
    brain's fear lasts), then ``startle_release``."""

    entry: str
    level: float
    t0: float
    until: float
    cap: float = math.inf  # this crouch's own drawn length: it rises by then at the latest
    confirmed: bool = False
    fallback: str | None = None  # a one-shot to show if the hold was refused at once


@dataclass
class Follow:
    """A startle whose answer is the back-steps (``hop_back``): asked once the duck stands and its trunk is slow."""

    t0: float  # the scare
    deadline: float  # it must have started by then
    s: float  # the scare's size (after the per-execution jitter) …
    k: float  # … and the level it maps to
    lasting: bool
    dip: str | None = None  # the startle dip that comes first (HOP_DIP_FIRST) …
    dip_seen: bool = False
    dip_done: bool = True
    manual: bool = False  # … a person's press: the wear caps do not apply
    asked_at: float | None = None
    confirmed: bool = False  # robotd's label showed it


@dataclass
class Script:
    """A timed run of one entry: a held entry (a chaining one: ``turn_*``, ``march``, ``lie_low``,
    the freezes) re-requested for ``hold_s`` seconds and then ended by ``end`` (``skill_abort`` /
    ``startle_release`` / ``bow_release``); a one-shot (``hold_s`` 0) requested once. ``who`` is
    ``page`` (a button) or ``brain``. ``confirmed`` once robotd's label shows the entry."""

    entry: str
    t0: float
    hold_s: float = 0.0
    end: str | None = None
    who: str = "brain"
    confirmed: bool = False
    asked: bool = False  # a one-shot has been requested
    ending: bool = False
    then: str | None = (
        None  # (startle-hop) a one-shot dip the back-steps follow (a person's press of a startle button)
    )
    patience_s: float = (
        0.0  # keep asking this long while the guard says wait (a settle after a rise)
    )


# the brain's acts that start a short held run of a B skill (standing only; the guard decides)
# behaviour → posture → (entry, hold seconds (lo, hi), end entry)
BRAIN_SCRIPTS: dict[str, dict[str, tuple[str, tuple[float, float], str | None]]] = {
    "greet": {"stand": ("bow", (0.0, 0.0), None)},
    "peekaboo": {"stand": ("lie_low", (1.4, 2.6), BOW_RELEASE)},
    # the tricks a person taught (neuro.tricks): a bow, and a lie-low held a little longer
    "bow": {"stand": ("bow", (0.0, 0.0), None)},
    "lie_low": {"stand": ("lie_low", (1.2, 1.8), BOW_RELEASE)},
    # the flip: one asked roll of the official roulade (the brain has already checked its own senses; the guard decides the rest)
    "flip": {"stand": (ROLL_ENTRY, (0.0, 0.0), None)},
}
# what the brain does right after an act ends (standing): the previous act → (entry, chance, patience s)
# 「shake after a nap or after getting up from a fall」 — the guard's settle time decides when
BRAIN_AFTER: dict[str, tuple[str, float, float]] = {
    "righting": ("shake_1", 0.6, 5.0),
    "nap": ("shake_1", 0.5, 5.0),
    "doze": ("shake_1", 0.3, 5.0),
}
CORNER_TURN_S = (1.2, 1.8)  # an on-the-spot turn out of a corner: ~60–90° at 52 °/s
CORNER_QUICK_TURN_S = (0.5, 0.8)  # … the quick turn (2.0 rad/s command, ~2.3 turned): ~65–105°
SCRIPT_CONFIRM_S = 0.4

# ── (startle-hop, 2026-10-10) a startle includes two small steps back ──────────────────────────────────────────────────────────────
# robotd's ``hop_back`` is "the largest startle plus 2–3 quick back-steps" (the table's note): the crouch with the neck pulled
# back is part of it. A scare of level 2 or more (``startle_2`` and up) is therefore answered by it, away from the scare, and what
# the brain does next is as before: the freeze (the legs held still by the reflex), the flight (the panic skill), or nothing.
HOP_ENTRY = "hop_back"
HOP_MIN_LEVEL = 0.5  # the entry's level (LIVE_STARTLE): startle_2 and up
HOP_WAIT_S = 1.0  # a scare that finds the duck walking: the back-steps start once it has stopped, within this long
HOP_DIP_WAIT_S = 3.0  # … with a dip first (HOP_DIP_FIRST): the dip's run (0.5 s hold + 0.8 s unwind) comes before
HOP_REASK_S = 0.12  # an ask not shown by robotd's label this soon is made again
HOP_DIP_FIRST = False  # standing: the startle dip at its level first, then the back-steps (False: the back-steps ARE the startle)
HOP_GIVE_UP_DIP_S = 1.5  # the back-steps did not happen: the plain startle follows, if the scare is no older than this
# refusals that end the wait (the others — the trunk not slow enough yet, something still running — are waited out)
HOP_HARD_REFUSALS = frozenset(
    {"unknown", "posture", "speed", "tilt", "held", "docked", "fallen", "gap", "hourly", "total"}
)


@dataclass
class SkillPicker:
    """Every tick: which trained skill (if any) the act runs now.

    Inputs are plain values the neural duck's ``MotorCommand`` already carries: the active
    ``reflex`` (a new one is an onset: a change of ``reflex``, or of ``reflex_seq`` when the
    caller counts them), the ``behaviour`` and its ``bout`` counter, the ``intensity`` 0..1 (or
    the reflex's own ``strength``), the body's posture (:func:`posture_of`), whether the fear is
    lasting, and whether there is room behind it (a hop-back backs up blind: the caller must know;
    without it, no hop-back).

    * reflex ``startle`` → a startle at a level near its strength (jittered per execution:
      CV :data:`LEVEL_CV`); a startle the brain will follow with a freeze (strength over 0.3,
      or ``lasting_fear``) **is held from the start** (``now_s`` given: a hold entry re-requested
      while the brain's startle / freeze lasts, at least a drawn 0.8–10 s, then
      ``startle_release`` — one crouch, one rise); ``hop_back`` for the very strongest (≥ 0.95)
      with room behind;
    * a freeze (reflex or bout) while a one-shot startle still holds its crouch **takes it over**
      (a hold at least as deep: no rise in between); once the one-shot unwinds, and for
      :data:`NO_BOB_S` after a crouch ended, a freeze starts none (the duck is already rising
      or has just stood up: no crouch → rise → crouch); otherwise a strong freeze (≥ 0.6) is a
      held crouch of its own;
    * ``flee`` (≥ 0.8, room behind) → ``hop_back``; ``body_shake`` → ``shake_1`` (``shake_2`` at
      ≥ 0.75); ``stretch_yawn`` / ``wake_ritual`` → ``stretch_tall`` standing, ``stretch_sit``
      seated (:attr:`mapping`).

    Without ``now_s`` it behaves as the Stage-1 picker did (one name per onset, the four-level
    grid, fixed freezes)."""

    mapping: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "body_shake": {"stand": "shake_1"},
            "stretch_yawn": {"stand": "stretch_tall", "sit": "stretch_sit"},
            "wake_ritual": {"stand": "stretch_tall", "sit": "stretch_sit"},
        }
    )
    last_bout: int | None = None
    last_reflex: str | None = None
    last_reflex_seq: int | None = None
    seed: int = 0
    available: frozenset[str] | None = None  # the entries robotd loaded (None: every live one)
    episode: Episode | None = None
    one_shot: tuple[str, float, float] | None = None  # (entry, level, t) of the last one-shot
    last_release: float | None = None
    last_now: float | None = None
    last_event: int | None = None  # the brain's newest startle event already acted on
    last_fear: float | None = None  # the last tick the brain showed fear (reflex or freeze bout)
    event_start: float | None = None  # when a startle was started from its event
    script: Script | None = None  # a timed run in progress (the page's button or a brain act)
    script_over: float | None = None  # when the last script ended
    last_turn: float | None = None
    prev_behaviour: str | None = None  # the act before the one now (the after-a-nap shake)
    follow: Follow | None = None  # a scare answered by the back-steps, in progress
    chain: bool = (
        False  # this tick's pick is that hop-back (the guard counts it at half, waits for the stop)
    )
    hold_still: bool = False  # the duck is waiting for a hop-back after a press: keep it standing
    hop_t: float | None = (
        None  # when the last startle-linked hop-back began (the flee bout asks for none right after)
    )
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ── helpers ──
    def _names(self, order: Iterable[str]) -> list[str]:
        have = self.available
        return [n for n in order if have is None or n in have]

    def _jitter(self, s: float) -> float:
        return min(1.0, max(0.0, s * _lognormal(self.rng, 1.0, LEVEL_CV)))

    def _hold_s(self, s: float) -> float:
        """The crouch's minimum: it stays low at least this long (the flinch's own hold)."""
        median = 1.2 + 1.5 * min(1.0, max(0.0, (s - LASTING_STRENGTH) / 0.7))
        return min(HOLD_MAX_S, max(HOLD_MIN_S, _lognormal(self.rng, median, HOLD_CV)))

    def _cap_s(self, s: float) -> float:
        """The crouch's own length, drawn per execution (ethology: a freeze lasts 1–10 s,
        median ≈ 3 s; stronger fear longer): it rises by then even while the brain's freeze
        goes on (live s1b:fix session 2: the brain's own freeze reflex stayed on ~10 s after a
        bang, so following it alone held every crouch to the 10-s cap)."""
        median = CAP_MEDIAN_S[0] + (CAP_MEDIAN_S[1] - CAP_MEDIAN_S[0]) * min(
            1.0, max(0.0, (s - LASTING_STRENGTH) / 0.7)
        )
        return min(HOLD_MAX_S, max(1.0, _lognormal(self.rng, median, CAP_CV)))

    def _start_hold(
        self, k: float, now_s: float, s: float, *, at_least: bool, fallback: str | None = None
    ) -> str | None:
        found = level_entry(k, self._names(FREEZE_ORDER), at_least=at_least)
        if found is None:
            return fallback
        entry, level = found
        low = self._hold_s(s)
        cap = max(low, self._cap_s(s))
        self.episode = Episode(entry, level, now_s, now_s + low, now_s + cap, fallback=fallback)
        return entry

    # ── a startle answered by the back-steps ──
    def _dip(
        self,
        s: float,
        k: float,
        now_s: float,
        lasting_fear: bool,
        one: tuple[str, float] | None = None,
    ) -> str | None:
        """The startle as it always was: a held crouch (a startle the brain follows with a freeze, or a strong one) or a
        one-shot dip at the scare's level."""
        one = one if one is not None else level_entry(k, self._names(STARTLE_ORDER))
        if lasting_fear or s > LASTING_STRENGTH:
            held = self._start_hold(k, now_s, s, at_least=False, fallback=one[0] if one else None)
            if held is not None:
                return held
        if one is None:
            return None
        self.one_shot = (one[0], one[1], now_s)
        return one[0]

    def _hop_here(self, one: tuple[str, float] | None) -> bool:
        return (
            one is not None
            and LEVEL.get(one[0], 0.0) >= HOP_MIN_LEVEL - 1e-9
            and HOP_ENTRY in set(self._names((HOP_ENTRY,)))
        )

    def _give_up(self, f: Follow, now_s: float, posture: str) -> str | None:
        """The back-steps will not happen: the scare gets the plain startle (if it is still the time for it)."""
        self.follow = None
        if f.manual or f.dip is not None or posture not in ("stand", "walk"):
            return None
        if now_s - f.t0 > HOP_GIVE_UP_DIP_S:
            return None
        return self._dip(f.s, f.k, now_s, f.lasting)

    def _follow_step(
        self,
        now_s: float,
        base: str,
        posture: str,
        still: bool,
        hop_ok: bool | None,
        decision: tuple[bool, str, float] | None,
    ) -> str | None | bool:
        """This tick's request while a scare waits for its back-steps: False when none does."""
        f = self.follow
        if f is None:
            return False
        if base == HOP_ENTRY:
            if not f.confirmed:
                f.confirmed = True
                self.hop_t = now_s
                self.one_shot = (HOP_ENTRY, 1.0, now_s)
            return None
        if f.confirmed:  # it ran, and has ended
            self.follow = None
            return False
        if f.dip is not None and not f.dip_done:
            if base == f.dip:
                f.dip_seen = True
                return None
            if f.dip_seen or now_s - f.t0 > CONFIRM_S:
                f.dip_done = True
            else:
                return None
        if now_s > f.deadline or hop_ok is False:
            return self._give_up(f, now_s, posture)
        refused = (
            decision is not None
            and f.asked_at is not None
            and decision[2] >= f.asked_at - 1e-6
            and not decision[0]
        )
        if refused and decision is not None and decision[1] in HOP_HARD_REFUSALS:
            return self._give_up(f, now_s, posture)
        self.hold_still = f.manual
        if posture != "stand" or not still:
            return None
        if f.asked_at is not None and not refused and now_s - f.asked_at < HOP_REASK_S:
            return None
        f.asked_at = now_s
        self.chain = True
        return HOP_ENTRY

    # ── timed runs (the page's 「動作試玩」 and a few brain acts) ──
    def play(
        self,
        entry: str,
        now_s: float,
        *,
        hold_s: float = 0.0,
        end: str | None = None,
        then: str | None = None,
    ) -> None:
        """Run ``entry`` now (the page's button): re-requested for ``hold_s`` seconds, then ``end``
        (a one-shot: once). It replaces a script and a fear episode in progress. ``then``: the
        back-steps (``hop_back``) that follow the one-shot once the duck has stopped."""
        self.episode = None
        self.follow = None
        self.script = Script(entry, now_s, float(hold_s), end, who="page", then=then)

    def cancel(self) -> None:
        self.script = None

    def _script_step(
        self, now_s: float, base: str, label: str, fear_onset: bool
    ) -> str | None | bool:
        """The script's request this tick: an entry name, ``None`` (nothing: it is over), or
        ``False`` (no script: the picker goes on)."""
        s = self.script
        if s is None:
            return False
        names = set(self._names((s.entry, *(() if s.end is None else (s.end,)))))
        if s.entry not in names:
            self.script = None  # (this robotd does not have it)
            return None
        running = base == s.entry
        if s.ending:
            self.script, self.script_over = None, now_s
            return None
        if fear_onset and s.who == "brain":
            # a fright takes the body from a brain-started bow or lie-low at once
            self.script, self.script_over = None, now_s
            return s.end if s.end in names and base == s.entry else None
        if running:
            s.confirmed = True
        if s.hold_s <= 0.0:
            # a one-shot: asked once (again while the guard says wait, up to ``patience_s``); over
            # when it ran and its window ended, or was not taken
            if not s.asked:
                s.asked = True
                return s.entry
            if s.confirmed and not (label.split(":", 1)[0] == s.entry):
                self.script, self.script_over = None, now_s
                if (
                    s.then == HOP_ENTRY
                ):  # the startle ran; the back-steps follow it (a person's press)
                    self.follow = Follow(t0=now_s, deadline=now_s + HOP_WAIT_S, s=0.0, k=0.0,
                                         lasting=False, manual=True)  # fmt: skip
                return None
            if not s.confirmed:
                if now_s - s.t0 <= s.patience_s:
                    return s.entry  # (the guard counts a run only when it starts)
                if now_s - s.t0 > SCRIPT_CONFIRM_S + 0.3:
                    self.script, self.script_over = None, now_s
            return None
        if not s.confirmed and now_s - s.t0 > SCRIPT_CONFIRM_S:
            self.script, self.script_over = None, now_s  # refused (the guard said why)
            return None
        if now_s < s.t0 + s.hold_s:
            return s.entry
        s.ending = True
        if s.end is not None and s.end in names and (running or s.confirmed):
            return s.end
        self.script, self.script_over = None, now_s
        return None

    def _brain_script(
        self, behaviour: str, new_bout: bool, posture: str, now_s: float, corner_turn: int
    ) -> str | None:
        """Start a brain act's timed run (a greeting bow, a peekaboo lie-low, an on-the-spot turn
        out of a corner); returns the first entry, or None."""
        if self.script is not None:
            return None
        if corner_turn and posture in ("stand", "walk"):
            entry = "turn_left" if corner_turn > 0 else "turn_right"
            quick = "quick_turn_left" if corner_turn > 0 else "quick_turn_right"
            hold_s = CORNER_TURN_S
            if quick in set(self._names((quick, ABORT))):
                entry, hold_s = (
                    quick,
                    CORNER_QUICK_TURN_S,
                )  # (2026-10-09) the quick turn gets it out of the corner
            if entry in set(self._names((entry, ABORT))) and (
                self.last_turn is None or now_s - self.last_turn > 6.0
            ):
                self.last_turn = now_s
                hold = self.rng.uniform(*hold_s)
                self.script = Script(entry, now_s, hold, ABORT)
                return entry
        if not new_bout:
            return None
        before, self.prev_behaviour = self.prev_behaviour, behaviour
        after = BRAIN_AFTER.get(before or "")
        if after is not None and posture != "sit" and before != behaviour:
            entry, chance, patience = after
            if entry in set(self._names((entry,))) and self.rng.random() < chance:
                self.script = Script(entry, now_s, 0.0, None, patience_s=patience)
                return entry
        spec = BRAIN_SCRIPTS.get(behaviour, {}).get(posture)
        if spec is None:
            return None
        entry, (lo, hi), end = spec
        if entry not in set(self._names((entry,))):
            return None
        hold = 0.0 if hi <= 0.0 else self.rng.uniform(lo, hi)
        self.script = Script(entry, now_s, hold, end)
        return entry

    # ── one tick ──
    def pick(
        self,
        *,
        reflex: str | None,
        behaviour: str,
        bout: int,
        intensity: float,
        posture: str,
        reflex_seq: int | None = None,
        lasting_fear: bool = False,
        room_behind: bool = False,
        now_s: float | None = None,
        strength: float | None = None,
        label: str | None = None,
        event: tuple[int, str, float] | None = None,
        corner_turn: int = 0,
        hop_ok: bool | None = None,
        still: bool = False,
        hop_decision: tuple[bool, str, float] | None = None,
    ) -> str | None:
        """``event``: the brain's newest reflex event ``(t_ms, reflex, strength)``. A startle is
        started from its event, the tick the brain perceives it, not ``reflex``'s onset (the
        brain's modeled 45–80 ms neural latency later): the trained skill's own motor latency
        (≈ 40–60 ms to the first motion, plus the servo delay) stands in for it, so the
        stimulus → motion time stays near the brain's intended one (ethology: 0.15–0.30 s).

        (startle-hop) ``hop_ok``: the scare may be answered by the back-steps — the way behind is not known to be blocked,
        the scare is not behind the duck, the battery is up (the caller's senses; None: the old rule, ``room_behind`` and
        a scare of 0.95); ``still``: the trunk is slow (under 0.05 m/s) and upright (under 8°); ``hop_decision``: the
        body link's last answer to a ``hop_back`` ask ``(ok, reason, clock)``. After the call ``chain`` says the pick is
        that hop-back, and ``hold_still`` that the duck should stay standing for it (a person's press)."""
        self.chain = False
        self.hold_still = False
        if reflex_seq is None:
            onset = reflex is not None and reflex != self.last_reflex
        else:
            onset = reflex is not None and reflex_seq != self.last_reflex_seq
            self.last_reflex_seq = reflex_seq
        self.last_reflex = reflex
        new_bout = bout != self.last_bout
        self.last_bout = bout
        if now_s is None:
            return self._pick_once(reflex, onset, behaviour, new_bout, intensity, posture,
                                   lasting_fear, room_behind)  # fmt: skip
        if self.last_now is not None and now_s < self.last_now:
            # the brain's clock started over (another duck, a restart): forget the old crouch
            self.episode, self.one_shot, self.last_release = None, None, None
            self.last_fear = None
        self.last_now = now_s
        s_in = float(strength if strength is not None else intensity)
        base = (label or "").split(":", 1)[0]
        if event is not None and event[1] == "startle" and event[0] != self.last_event:
            # the startle as the brain perceives it: act now (its reflex name follows later)
            self.last_event = int(event[0])
            if self.episode is None and base not in set(self._names(STARTLE_ORDER)):
                reflex, onset, s_in = "startle", True, float(event[2])
                self.event_start = now_s
        elif reflex == "startle" and onset and self.event_start is not None:
            if now_s - self.event_start < EVENT_SAME_S:
                onset = False  # the same startle, already started from its event
        # the crouch follows the brain's startle → freeze REFLEX (the brain draws the freeze's own
        # length, lognormal, ≤ 6 s), not the freeze behaviour bout that may follow it (live
        # s1b:fix session 1: following the bout held 8 of 10 crouches to the 10-s cap)
        fear = reflex in FEAR_REFLEXES
        freeze_onset = (reflex == "freeze" and onset) or (behaviour == "freeze" and new_bout)
        # the last time any fear showed (the reflex or a freeze bout) before this tick: a freeze
        # starts a NEW crouch only after the fear was quiet for NO_BOB_S (live session 2: the
        # brain's freeze went on after the crouch rose and a new onset 3.0 s later crouched again)
        quiet_before = self.last_fear
        if fear or behaviour == "freeze":
            self.last_fear = now_s

        # a timed run (the page's button, a greeting bow, a lie-low, a turn out of a corner)
        fear_onset = reflex in FEAR_REFLEXES and onset
        stepped = self._script_step(now_s, base, label or "", fear_onset)
        if stepped is not False:
            if stepped is not None:
                return stepped
            if self.script is not None:
                return None
        ep = self.episode
        if ep is not None:
            holds = set(self._names(FREEZE_ORDER))
            if base in holds:
                ep.confirmed = True
                if base != ep.entry:  # (the guard stepped down to a shallower hold: go on with it)
                    ep.entry, ep.level = base, LEVEL.get(base, ep.level)
            elif ep.confirmed or now_s - ep.t0 > CONFIRM_S:
                # the crouch stopped (aborted, fell, taken over) or never started (refused)
                self.episode = None
                self.last_release = now_s
                if not ep.confirmed and ep.fallback and posture in ("stand", "walk"):
                    self.one_shot = (ep.fallback, LEVEL.get(ep.fallback, ep.level), now_s)
                    return ep.fallback
                return None
            if fear:
                ep.until = max(ep.until, now_s + HOLD_TAIL_S)
            if now_s < min(ep.until, ep.cap, ep.t0 + HOLD_MAX_S):
                return ep.entry  # the hold goes on (the guard: a refresh)
            self.episode = None
            self.last_release = now_s
            return RELEASE

        upright = posture in ("stand", "walk")
        stepped_follow = self._follow_step(now_s, base, posture, still, hop_ok, hop_decision)
        if stepped_follow is not False:
            return stepped_follow
        if reflex == "startle" and onset:
            if not upright or not math.isfinite(s_in) or s_in < 0.15:
                return None
            s = self._jitter(s_in)
            k = min(1.0, max(0.25, s))
            one = level_entry(k, self._names(STARTLE_ORDER))
            if (
                hop_ok is None
            ):  # (the old rule: only the strongest scare, and only when the way back is known free)
                if room_behind and s >= 0.95 and HOP_ENTRY in set(self._names((HOP_ENTRY,))):
                    return HOP_ENTRY
            elif hop_ok and self._hop_here(one):
                dip_first = HOP_DIP_FIRST and posture == "stand" and one is not None
                f = Follow(t0=now_s, deadline=now_s + (HOP_DIP_WAIT_S if dip_first else HOP_WAIT_S),
                           s=s, k=k, lasting=lasting_fear)  # fmt: skip
                self.follow = f
                if dip_first and one is not None:
                    f.dip, f.dip_done = one[0], False
                    self.one_shot = (one[0], one[1], now_s)
                    return one[0]
                again = self._follow_step(now_s, base, posture, still, hop_ok, hop_decision)
                return None if again is False else again
            return self._dip(s, k, now_s, lasting_fear, one)

        if freeze_onset and self.follow is not None:
            return None  # the back-steps are its crouch
        if freeze_onset:
            s = self._jitter(s_in) if math.isfinite(s_in) else 0.0
            holding = (label or "") == base  # (not ``name:unwind``)
            if posture == "skill" and holding and base in set(self._names(STARTLE_ORDER)):
                # take over the running startle while it still holds its crouch: at least as
                # deep, no rise in between (s1b:fix Mac gate: a takeover 0.4 s into the unwind
                # still let it rise 26 mm first — so none once the unwind has begun)
                k = max(LEVEL.get(base, 0.25), min(1.0, s))
                return self._start_hold(k, now_s, max(s, LASTING_STRENGTH), at_least=True)
            if posture == "skill" and base in set(self._names(STARTLE_ORDER)):
                return None  # it is already rising: the freeze plays standing, no second crouch
            recent = [t for t in (self.last_release, quiet_before) if t is not None]
            if self.one_shot is not None:
                recent.append(self.one_shot[2] + ONESHOT_S)
            if any(now_s - t < NO_BOB_S for t in recent):
                return None  # it has just stood up, or the fear never went quiet: no new crouch
            if upright and s_in >= 0.6:
                return self._start_hold(min(1.0, s), now_s, s, at_least=False)
            return None

        if new_bout or corner_turn:
            started = self._brain_script(behaviour, new_bout, posture, now_s, corner_turn)
            if started is not None:
                return started
        if not new_bout:
            return None
        if behaviour == "flee":
            if hop_ok is not None:
                return None  # (startle-hop) the flight follows the startle's back-steps: it is the panic skill's, not another hop
            ok = upright and room_behind and s_in >= 0.8
            return "hop_back" if ok and (self.available is None or "hop_back" in self.available) \
                else None  # fmt: skip
        pick = self.mapping.get(behaviour, {}).get(posture)
        if pick == "shake_1" and s_in >= 0.75:
            pick = "shake_2"
        return pick

    def _pick_once(
        self,
        reflex: str | None,
        onset: bool,
        behaviour: str,
        new_bout: bool,
        intensity: float,
        posture: str,
        lasting_fear: bool,
        room_behind: bool,
    ) -> str | None:
        """The Stage-1 picker (no clock): one name per onset / bout."""
        upright = posture in ("stand", "walk")
        if reflex == "startle" and onset:
            if not upright:
                return None
            choice = startle_entry(intensity, lasting=lasting_fear)
            if choice == "startle_4" and room_behind and intensity >= 0.95:
                choice = "hop_back"
            return choice
        if not new_bout:
            return None
        if behaviour == "freeze":
            return startle_entry(intensity, lasting=True) if upright and intensity >= 0.6 else None
        if behaviour == "flee":
            return "hop_back" if upright and room_behind and intensity >= 0.8 else None
        pick = self.mapping.get(behaviour, {}).get(posture)
        if pick == "shake_1" and intensity >= 0.75:
            pick = "shake_2"
        return pick


def request_with_fallback(
    guard: SkillGuard,
    skill: str | None,
    view: BodyView,
    now_s: float,
    *,
    manual: bool = False,
    chain: bool = False,
) -> Decision:
    """:meth:`SkillGuard.request`, stepping down to a smaller entry of the same family when the
    wanted one is capped (``hourly`` / ``gap``)."""
    if skill is None:
        return Decision(False, None, "none", "")
    tried = skill
    while True:
        d = guard.request(tried, view, now_s, manual=manual, chain=chain)
        if d.ok or d.reason not in ("hourly", "gap"):
            return d if d.ok or tried == skill else replace(d, skill=skill)
        smaller = lower(tried)
        if smaller is None or smaller not in guard.names:
            return replace(d, skill=skill)
        tried = smaller


def caps_from_strain(
    rules: Mapping[str, Rule], strain: Mapping[str, Mapping[str, Any]]
) -> dict[str, Rule]:
    """Caps set from the strain gates. ``strain[name]``: that skill's gate summary on the **bam**
    servo model (a :func:`duckmind.skilltrain.metrics.summarize` condition, or just its
    ``strain`` block plus ``servo_hits_upright_k``). A failed strain gate (pooled torque p95 over
    0.35 N·m, stalls over 0.5 %, a housing hit while upright) removes the skill (cap 0); a
    marginal one (p95 over 0.30 N·m or stalls over 0.3 %) halves the cap; a clean one keeps it.
    Judge on bam: the MJCF's XML servos have no current limit and read higher (v5's own stop
    from a 0.55 walk is 0.43 N·m p95 there, 0.33 on bam; 2026-09-28)."""
    out = dict(rules)
    for name, s in strain.items():
        rule = out.get(name)
        if rule is None or rule.per_hour is None:
            continue
        nested = s.get("strain")
        block: Mapping[str, Any] = nested if isinstance(nested, Mapping) else s
        hits_block = s.get("servo_hits_upright")
        hits = (
            int(hits_block.get("k") or 0)
            if isinstance(hits_block, Mapping)
            else int(s.get("servo_hits_upright_k") or 0)
        )
        tau = max(float(block.get("tau_leg_p95") or 0.0), float(block.get("tau_head_p95") or 0.0))
        stall = float(block.get("stall_frac") or 0.0)
        if tau > 0.35 or stall > 0.005 or hits > 0:
            out[name] = replace(rule, per_hour=0, note=rule.note + " (strain gate failed)")
        elif tau > STRAIN_MARGIN_NM or stall > STALL_MARGIN:
            out[name] = replace(
                rule,
                per_hour=max(1, rule.per_hour // 2),
                note=rule.note + " (strain marginal: halved)",
            )
    return out


__all__ = [
    "ABORT",
    "BodyView",
    "DEFAULT_RULES",
    "Decision",
    "Episode",
    "LASTING_STRENGTH",
    "RELEASE",
    "Rule",
    "SkillGuard",
    "SkillPicker",
    "caps_from_strain",
    "level_entry",
    "lower",
    "posture_of",
    "request_with_fallback",
    "startle_entry",
]
