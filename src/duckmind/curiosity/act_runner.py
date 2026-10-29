"""The Act Runner: one curiosity step at a time, with the duck's own eyes and body (curiosity spec
§5; physical-design spec §15; owner decisions A1–A5, 2026-09-27).

A ``duck.curio`` step (:class:`~duckmind.duckbody.protocol.CurioCmd`, ``phase: "step"``) names a
thing and a verb; :class:`CurioStep` carries it out as an
:class:`~duckmind.duckbody.protocol.ActExecutor`: every tick the body loop hands it an
:class:`~duckmind.duckbody.protocol.ActContext` and sends what its
:class:`~duckmind.duckbody.protocol.ActOutput` asks for (twist, head offsets, mouth, one
``robot.do``, looks for duck-eye) through the reflexes and the forbidden-command guard. It reads
no clock, socket or truth: the same code runs on the board, beside the live simulator and in the
headless rig.

**The step** (curiosity spec §5.2), whatever the verb:

1. *look* — the head turns toward where the thing is believed to be *when the look comes up*
   (low for near things; retries change the pitch, then the yaw) and, once still, duck-eye's
   ``things`` task looks (:mod:`duckmind.curiosity.perceive`); the thing is remembered in the
   odometry frame between looks;
2. *approach* — legs at 0.40 m/s (owner A3) while it is far, then steps whose length comes from
   the live step table (:attr:`~duckmind.curiosity.calib.Calib.fwd_table`: 0.3 s ≈ 6 mm …
   0.7 s ≈ 80 mm), steering the trunk toward where it must stand (the M2 controller), a look
   after each; reverse arcs (1.0 s: ±33°) to turn beside a thing; too close to see: a step back.
   The final approach may walk **head down** (owner A2) right after a sight: a creep that makes
   under 30 % of its measured length with the head down raises the head for the rest of the step;
3. *confirm* — two looks at least 0.2 s apart that agree within 12 mm, the thing inside the
   verb's window × the patience's ``tol_pct``. A window nearer than the eye sees (~95 mm: a fin
   under the beak) is confirmed 3–11 cm farther out and the last step is **blind**, its length
   from the step table (live ±4 mm). A hasty duck that runs out of ``align_budget`` acts anyway
   when the thing is within 1.5 windows, a careful one gives up;
4. *settle* ``settle_ms``, then *act*: ``bump`` (a 0.6 s creep into it), ``peck`` (ground_pick,
   beak closed), ``nibble`` (beak open, working at the bottom), ``tap`` (1–3 nods), ``grab``
   (open at 0.2 s, close at 1.4 s, stand 1 s: the mouth joint says held or not), ``carry`` (one
   relay leg with the thing held, released when it will not come), ``put_down``, ``kick``
   (the kick cone first), ``dribble`` (a chest push);
5. *watch* — looks at where it went; outcomes from perception only
   (:mod:`duckmind.curiosity.outcomes`); the knock's sound is ``unknown`` (owner A5);
6. *report* — one ``body.curio_step`` (and a ``sense.thing``, a ``body.guard`` when a hard rule
   fired), integers only.

**A1.** While a step approaches a thing it saw in the last :data:`TOWARD_FRESH_MS`,
:attr:`CurioStep.toward` names it (look id, bearing, range, when): the body loop's ToF
``stop_near`` may then let the duck closer than 30 cm, toward that thing only; every other
reflex stays as it is.

**Never here**: turning on the spot, |vx| > 0.40, |wz| > 1.0, reversing outside ``back`` /
``back_arc``, a skill while walking, a skill outside ``AUTONOMOUS_SKILLS``
(:func:`duckmind.curiosity.prims.problems` checks every output; :attr:`CurioStep.violations`
records any, and the tests keep it empty).

**In the dark** (owner 2026-09-28 「一律放寬」, applied at the phase-4 landing; physical spec §15.2
item 6 relaxed): a curiosity act may start in dark, black or eye-down light when the duck's own
perception still has the thing — a sighting its tracker still keeps (:meth:`Tracker.fresh`: 10 s
for a thing that stood still, 3 s for one that moved; a step on the same thing right after
another one carries that step's memory over, :class:`ThingMemory`, in the same odometry frame) or
the thing in its beak (``carry``, ``put_down``: the mouth joint says so, not the camera).
Otherwise it ends ``dark`` at once: it simply cannot see the thing there (duck-eye's ``things``
task names nothing in dark or black frames — the camera stays honest), so there is nothing to act
on. ``look`` always runs. What happens next is the act's usual business: its looks see nothing in
the dark, so it goes on what it remembers until the memory ages out (``lost``), and physics
decides what the steps meet.

``CurioCmd.params`` (integers and short strings, from Duck Core or a script):

* ``verb`` — one of :data:`VERBS`; ``cls`` — the thing's class (``ball_slow``, ``handle_toy``,
  ``box``, ``cup``, ``tall_light``, ``heavy``, ``furniture`` …); ``label`` — what duck-eye calls
  it (default: the command's ``thing``; :data:`ANY_THING` ``"*"``: whatever is at ``b_mrad``/
  ``r_mm``);
* ``b_mrad``/``r_mm`` — where Duck Core believes it is (trunk frame now), optional;
* ``wits`` (1500–8500) or ``tier`` (``hasty``/``normal``/``careful``) — patience (§7.7);
* ``h_mm``/``d_mm`` — its registered height and depth (peck reach, bump gap);
* kick: ``foot`` (``left``/``right``/``auto``), ``leash`` ``{x_mm, y_mm, r_mm}`` and ``keep_out``
  ``[{x_mm, y_mm, r_mm}]`` in the odometry frame; carry: ``relay_ms``, ``dest_b_mrad``;
  tap: ``taps`` (1–3); bump: ``bold`` (straight in, ≤ 1 s).
"""

from __future__ import annotations

import math
import weakref
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.curiosity import cone as cone_mod
from duckmind.curiosity import outcomes as oc
from duckmind.curiosity import prims
from duckmind.curiosity.calib import CALIB, TIERS, Calib, Patience, Window, step_ms
from duckmind.curiosity.perceive import (
    Fix,
    Odom,
    Seen,
    Tracker,
    centre_of,
    fix_of,
    pick,
    result_light,
    sightings,
)
from duckmind.duckbody.protocol import (
    ActContext,
    ActOutput,
    Command,
    CurioCmd,
    LookRequest,
    Toward,
    WalkCmd,
)
from duckmind.percept import PerceptError, validate

VERBS = ("look", "bump", "peck", "nibble", "tap", "grab", "carry", "put_down", "kick", "dribble")
# ``label: "*"`` — the thing at the believed place (``b_mrad``/``r_mm``), whatever duck-eye names
# it (a demonstration's "the one in front of you"; a registered toy or a colour name otherwise)
ANY_THING = "*"
ACT_VERBS = frozenset(VERBS) - {"look"}
# light levels in which the eye names no things (duck-eye's ``things`` task; ``none``: the eye is
# down or its level is stale): an act starts there only on what its perception still has
DARK_LIGHTS = ("dark", "black", "none")
HELD_VERBS = frozenset({"carry", "put_down"})  # the thing is in the beak: the mouth joint has it
PERCEPTIONS = ("camera", "robot")  # what Duck Core may be told (physical spec §15.2 item 3)
# curiosity spec §6.1's truth-plus-noise stand-in: headless / live development runs only, never
# the body loop's factory, never a gate, never Duck Core (tests/test_curio_act_runner.py)
DEV_PERCEPTIONS = ("oracle_noise",)
# robot.head offsets from HOME (rad): the A100-validated half-down look (lookplan.HALF_DOWN),
# the full-down near look (spec §4.1 "低頭", camera −34°, floor from 0.095 m; curiosity §5.1
# "近距離要低頭") and the widest validated yaw.
HOME_HEAD = (0.0, 0.0, 0.0, 0.0)
HALF_DOWN = (-0.125, 0.175, 0.0, 0.0)
FULL_DOWN = (-0.25, 0.35, 0.0, 0.0)
AIM_YAW = 0.54
NEAR_LOOK_MM = 260  # nearer than this: the full-down look
MID_LOOK_MM = 480  # nearer than this: the half-down look
HEAD_MOVE_MS = 450  # head to a new pose (0.3 s) + settle (0.15 s)
# the measured head within this of the commanded pose counts as there: duck-body's guard's own
# tolerance (bodyloop.HEAD_HOME_TOL), wide enough for the servos' ~0.13 rad sag at HOME (live)
HEAD_TOL_RAD = 0.16
HEAD_SETTLE_MS = 360  # spec §4.2 rule 3: 0.3 s at HOME (the guard checks it) + one 25 Hz tick
HEAD_SETTLE_CAP_MS = 1500
TAIL_MS = 450  # the legs keep stepping after a stop (spec §4.2)
LOOK_TIMEOUT_MS = 2500
MAX_LOST_LOOKS = 5  # looks in a row without it: lost
SEARCH_AFTER_LOST = 2  # two misses in a row: stop moving on memory, search with the head
PICK_GATE_MM = 600  # a sighting this near the remembered place is the thing (odometry drifts)
TOWARD_FRESH_MS = 3000
A2_FRESH_MS = 1200  # walk head down only when the thing was seen this recently
FAR_MM = 250  # farther than this beyond the window's centre: legs, then creeps
LEG_STOP_SHORT_MM = 90  # a leg aims to stop this far before the window (then creeps)
MAX_TRIMS = 2
TURN_MRAD = 524  # near, and 30° off where it should be: turn with a reverse arc (M2)
BIG_SIDE_MM = 60  # near with no room ahead, this far off to the side: a reverse arc
LEG_TURN_EFFICIENCY = 0.7
OUT_OF_REACH_MRAD = 900  # beyond the head's reach (0.54 rad + half the view): turn the body
HEAD_DOWN_MIN_PROGRESS = 0.30  # A2 fallback: a head-down creep under 30 % of its length
TETHER_PROGRESS = 0.30  # carry: under 30 % of the expected advance in 1 s …
TETHER_YAW_MRAD = 349  # … or turned more than 20 °: let go (F2)
WATCH_MS = 700
KICK_WATCH_MS = 1200
GRAB_HOLD_MS = 1000
PUT_DOWN_MS = 800
NIBBLE_PULSES_MS = ((1100, 1300), (1500, 1700))  # beak closed during these, open around them
BOLD_BUMP_MS = 1000
NEAR_STEP_MAX_MS = 700  # near the thing, one step covers at most ~80 mm (fine control)
BLIND_MIN_MM = 30  # the blind last step covers 30…110 mm (0.4–1.0 s from the step table)
BLIND_MAX_MM = 110


@dataclass(frozen=True)
class ThingMemory:
    """What the duck's perception still has of a thing when a step ends: duck-body's working
    memory between two curiosity steps (the step's tracker, in the odometry frame of ``epoch``).
    A step on the same ``thing`` that starts in the dark picks it up (owner 「一律放寬」); in the
    light a step looks for itself, as it always did."""

    thing: str | None
    epoch: int | None
    tracker: Tracker

    def recall(self, thing: str | None, epoch: int | None, now_ms: int) -> Tracker | None:
        """The remembered sighting for a step on ``thing`` now, or ``None``: another thing, a
        broken odometry frame (lifted, robotd restarted) or a memory that has aged out."""
        if thing != self.thing or epoch != self.epoch or not self.tracker.fresh(now_ms):
            return None
        copy = Tracker(self.tracker.key)
        copy.fixes = list(self.tracker.fixes)
        copy.moving = self.tracker.moving
        return copy


@dataclass(frozen=True)
class ThingInfo:
    """What the registration says about a thing (curiosity §4.3), in integers."""

    cls: str = "unknown"
    h_mm: int | None = None
    d_mm: int | None = None
    carry_ok: bool = False


@dataclass
class _Op:
    kind: str  # head | wait | move | look | skill | mouth | call
    head: tuple[float, float, float, float] | None = None
    ms: int = 0
    still: bool = False
    prim: prims.Prim | None = None
    head_down: bool = False
    look_id: str | None = None
    kind_look: str = "fixate"
    skill: str | None = None
    mouth: float | None = None
    plan: tuple[tuple[int, int, float], ...] = ()  # skill mouth plan: (from_ms, to_ms, open)
    call: str | None = None
    target: tuple[int, int] | None = None  # aim: a fixed trunk-frame point (else the belief)
    attempt: int = 0
    # runtime
    started_ms: int | None = None
    sent: bool = False
    left_policy: bool = False
    start_odom: Odom | None = None
    near_since: int | None = None  # the measured head first within HEAD_TOL_RAD of HOME (ms)


@dataclass
class StepRecord:
    """What the step saw and did (becomes the body.curio_step)."""

    looks: int = 0
    looks_ok: int = 0
    lost_looks: int = 0
    trims: int = 0
    creeps: int = 0
    legs: int = 0
    turns: int = 0
    backs: int = 0
    bursts: int = 0
    head_fallback: int = 0
    halted: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


class CurioStep:
    """One ``duck.curio`` step (see the module docstring)."""

    def __init__(
        self,
        command: CurioCmd,
        *,
        src: str = "live",
        pv: str = "p:00000000",
        calib: Calib = CALIB,
        tracker: Tracker | None = None,
        info: ThingInfo | None = None,
        perception: str = "camera",
        memory: ThingMemory | None = None,
    ) -> None:
        if command.phase != "step":
            raise ValueError("a CurioStep runs duck.curio phase 'step'")
        params = dict(command.params)
        verb = str(params.get("verb", "look"))
        if verb not in VERBS:
            raise ValueError(f"verb {verb!r} not in {VERBS}")
        if perception not in PERCEPTIONS + DEV_PERCEPTIONS:
            raise ValueError(f"perception {perception!r} not in {PERCEPTIONS + DEV_PERCEPTIONS}")
        self.act_id = command.act_id
        self.command = command
        self.verb = verb
        self.src = src
        self.pv = pv
        self.calib = calib
        self.perception = perception
        self.params = params
        label = str(params.get("label") or command.thing or "")
        # ANY_THING: the thing at the believed place (b_mrad/r_mm), whatever duck-eye calls it
        self.any_thing = label == ANY_THING
        self.label = str(command.thing or "") if self.any_thing else label
        self.info = info or ThingInfo(
            cls=str(params.get("cls", "unknown")),
            h_mm=_opt_int(params.get("h_mm")),
            d_mm=_opt_int(params.get("d_mm")),
            carry_ok=bool(params.get("carry_ok", False)),
        )
        self.patience = _patience(params)
        self.tracker = tracker if tracker is not None else Tracker(self._key())
        self.record = StepRecord()
        self.queue: deque[_Op] = deque()
        self.op: _Op | None = None
        self.phase = "start"
        self.head: tuple[float, float, float, float] = HOME_HEAD
        self.mouth = 0.0
        self.mouth_held = False
        self.t0_ms: int | None = None
        self.now_ms = 0
        self.odom: Odom | None = None
        self.start_odom: Odom | None = None
        self.last_step_ms = -10_000
        self.look_seq = 0
        self.pending_look: str | None = None
        self.results: dict[str, Mapping[str, Any]] = {}
        self.last_seen: list[Seen] = []
        self.before: Fix | None = None
        self.after: Fix | None = None
        self.align_started_ms: int | None = None
        self.trims_in_row = 0
        self.head_down_allowed = True
        self.foot = str(params.get("foot", "auto"))
        self.cone: cone_mod.ConeCheck | None = None
        self.mouth_mrad: int | None = None
        self.outcome: str | None = None
        self.ended_by = "done"
        self.extra: dict[str, Any] = {}
        self.guards: list[dict[str, Any]] = []
        self.finished: ActOutput | None = None
        self.fell = False
        # the step before's memory of its thing (used only to start in the dark, 「一律放寬」)
        self.memory = memory
        self.epoch: int | None = None  # the odometry frame this step saw the thing in
        self._head_changed = False
        self.lost_streak = 0  # looks in a row that did not find it

    # ── the executor interface ──────────────────────────────────────────────

    @property
    def violations(self) -> list[str]:
        return self.record.violations

    @property
    def toward(self) -> Toward | None:
        """Owner A1: the just-seen target this step is approaching (``None`` when it is not
        approaching, or the last sight is older than :data:`TOWARD_FRESH_MS`)."""
        last = self.tracker.last
        if self.phase not in ("approach", "confirm", "act") or last is None or self.odom is None:
            return None
        if self.now_ms - last.t_ms > TOWARD_FRESH_MS:
            return None
        x, y = self.odom.to_trunk(last.x_mm, last.y_mm)
        return Toward(
            look_id=last.look_id,
            b_mrad=int(round(math.atan2(y, x) * 1000.0)),
            r_mm=int(round(math.hypot(x, y))),
            seen_t_ms=last.t_ms,
            thing=self.label or None,
        )

    @property
    def moving_op(self) -> _Op | None:
        op = self.op
        return op if op is not None and op.kind == "move" and self.finished is None else None

    @property
    def head_down_ok(self) -> bool:
        """Owner A2: the running move is the final approach, head down, toward the just-seen
        thing (the body loop's guard lets exactly this walk with the head off HOME)."""
        op = self.moving_op
        return op is not None and op.head_down and self.toward is not None

    @property
    def walk_view(self) -> WalkCmd | None:
        """The running move as the ``duck.walk`` the reflexes and the guard understand: purpose
        ``curio``, ``toward`` the just-seen target (owner A1: the ToF stop relaxed toward it,
        down to the verb's standoff), ``head`` ``down`` for an A2 move. ``None`` when still."""
        op = self.moving_op
        if op is None or op.prim is None:
            return None
        toward = self.toward
        return WalkCmd(
            act_id=self.act_id,
            deadline_ms=self.command.deadline_ms,
            vx_mm=op.prim.vx_mm,
            vyaw_mrad=op.prim.vyaw_mrad,
            max_ms=op.prim.ms,
            pre_look="none",
            stop_ahead_mm=self._standoff_mm(),
            purpose="curio",
            toward=toward,
            head="down" if op.head_down and toward is not None else "home",
            curio_id=self.act_id,
        )

    def _standoff_mm(self) -> int:
        """How near the thing the step means to get (ToF range to its near face, roughly)."""
        window, _ = self._window()
        half = (self.info.d_mm or 60) // 2
        return max(50, window.ahead[0] - half)

    def step(self, ctx: ActContext) -> ActOutput:
        if self.finished is not None:
            return self.finished
        self.now_ms = ctx.t_ms
        if self.t0_ms is None:
            self.t0_ms = ctx.t_ms
        if ctx.pose is not None:
            self.odom = Odom.from_pose(ctx.pose.odom)
            self.epoch = ctx.pose.epoch
            if self.start_odom is None:
                self.start_odom = self.odom
        for result in ctx.eye:
            look_id = result.get("look_id") if isinstance(result, Mapping) else None
            if isinstance(look_id, str) and look_id.startswith(self.act_id):
                self.results[look_id] = result
        # body senses end the step at once
        if ctx.proprio.fallen:
            self.fell = True
            return self._end(ctx, "fell")
        if ctx.proprio.lifted:
            return self._end(ctx, "lifted")
        if self.phase == "start":
            self._begin(ctx)
        for _ in range(8):  # ops that complete at once chain within the tick
            if self.finished is not None:
                return self.finished
            if self.op is None:
                if not self.queue:
                    self._decide(ctx)
                    if self.finished is not None:
                        return self.finished
                    if not self.queue:
                        return self._output(None)
                self.op = self.queue.popleft()
            out = self._run(self.op, ctx)
            if out is not None:
                return out
            self.op = None
        return self._output(None)

    def preempt(self, why: str, ctx: ActContext) -> ActOutput:
        self.now_ms = ctx.t_ms
        return self._end(ctx, why if why in oc.ENDS else "preempt")

    # ── output helpers ──────────────────────────────────────────────────────

    def _output(
        self,
        twist: tuple[int, int] | None,
        *,
        prim: str | None = None,
        do: str | None = None,
        looks: tuple[LookRequest, ...] = (),
        head_down: bool = False,
    ) -> ActOutput:
        head = None if prims.head_is_home(self.head) else self.head
        problems = prims.problems(
            twist=twist,
            do=do,
            head=self.head,
            prim=prim,
            ms_since_step=self.now_ms - self.last_step_ms,
            head_down_ok=head_down,
        )
        if problems:
            self.record.violations += problems
            # never send a broken output: hold still instead
            return ActOutput(twist=None, head=head, mouth=self.mouth, looks=looks)
        if twist is not None and twist != (0, 0):
            self.last_step_ms = self.now_ms
        return ActOutput(twist=twist, head=head, do=do, mouth=self.mouth, looks=looks)

    # ── the ops ─────────────────────────────────────────────────────────────

    def _run(self, op: _Op, ctx: ActContext) -> ActOutput | None:
        now = ctx.t_ms
        if op.started_ms is None:
            op.started_ms = now
            op.start_odom = self.odom
        elapsed = now - op.started_ms
        if op.kind == "head":
            assert op.head is not None
            self._head_changed = op.head != self.head
            self.head = op.head
            return None
        if op.kind == "aim":
            target = op.target if op.target is not None else self._believed()
            pose = self._look_pose(target, op.attempt)
            self._head_changed = pose != self.head
            self.head = pose
            return None
        if op.kind == "mouth":
            self.mouth = float(op.mouth or 0.0)
            return None
        if op.kind == "call":
            getattr(self, f"_call_{op.call}")(ctx)
            return None
        if op.kind == "wait":
            if op.call == "settle_head":
                return self._settle_head(op, ctx, elapsed)
            if op.call == "aim_settle":
                op.ms = HEAD_MOVE_MS if self._head_changed else 200
                op.call = None
            if elapsed < op.ms:
                return self._output(None)
            if op.still and ctx.pose is not None and not ctx.pose.still and elapsed < op.ms + 1500:
                return self._output(None)
            return None
        if op.kind == "move":
            assert op.prim is not None
            if ctx.halted_by and elapsed > 0:
                self.record.halted.append(ctx.halted_by)
                return self._end_move(op, ctx, halted=True)
            if elapsed < op.prim.ms:
                return self._output(
                    (op.prim.vx_mm, op.prim.vyaw_mrad), prim=op.prim.name, head_down=op.head_down
                )
            return self._end_move(op, ctx, halted=False)
        if op.kind == "look":
            return self._run_look(op, ctx, elapsed)
        if op.kind == "skill":
            return self._run_skill(op, ctx, elapsed)
        raise ValueError(f"unknown op {op.kind}")

    def _settle_head(self, op: _Op, ctx: ActContext, elapsed: int) -> ActOutput | None:
        """Before a move the head must have reached its pose and been still ≥ 0.3 s (spec §4.2
        rule 3). The measured head never sits exactly on the command (standing, it sags ~10°,
        A100 look.hold), so "there" is: at least 0.3 s since the command, and either the posture
        tap says still (head < 0.2 rad/s, gyro < 0.3 rad/s for 150 ms) or the measurement is
        within :data:`HEAD_TOL_RAD`; at most :data:`HEAD_SETTLE_CAP_MS`."""
        if not self._head_changed and elapsed == 0:
            return None
        near = False
        if ctx.state is not None:
            measured = ctx.state.head_offsets
            near = all(abs(m - c) <= HEAD_TOL_RAD for m, c in zip(measured, self.head, strict=True))
        home = not any(self.head)
        if home:
            # duck-body's guard counts HOME only once the measured head has stayed within its
            # tolerance for 0.3 s (bodyloop.HEAD_HOME_TOL): wait for exactly that, or a move
            # would be refused (and zeroed) without the step knowing
            op.near_since = (op.near_since or ctx.t_ms) if near else None
        if elapsed < HEAD_SETTLE_MS:
            return self._output(None)
        if home and near and op.near_since is not None:
            if ctx.t_ms - op.near_since >= HEAD_SETTLE_MS:
                return None
        elif not home and (near or (ctx.pose is not None and ctx.pose.still)):
            return None
        if elapsed >= HEAD_SETTLE_CAP_MS:
            self.extra["head_settle_timeouts"] = self.extra.get("head_settle_timeouts", 0) + 1
            return None
        return self._output(None)

    def _end_move(self, op: _Op, ctx: ActContext, *, halted: bool) -> ActOutput | None:
        assert op.prim is not None
        moved = 0
        if self.odom is not None and op.start_odom is not None:
            moved = self.odom.delta(op.start_odom)[0]
        expected = self.calib.step_mm.get(op.prim.name, 0)
        if op.head_down and expected > 0 and moved < HEAD_DOWN_MIN_PROGRESS * expected:
            # A2 fallback: a head-down creep that barely moved: raise the head from now on
            self.head_down_allowed = False
            self.record.head_fallback += 1
        self.extra.setdefault("moves", []).append(
            {"prim": op.prim.name, "ms": op.prim.ms, "fwd_mm": moved, "halted": halted}
        )
        return None

    def _new_look_id(self) -> str:
        self.look_seq += 1
        return f"{self.act_id}.L{self.look_seq}"

    def _run_look(self, op: _Op, ctx: ActContext, elapsed: int) -> ActOutput | None:
        if not op.sent:
            op.sent = True
            op.look_id = op.look_id or self._new_look_id()
            self.pending_look = op.look_id
            self.record.looks += 1
            request = _look_request(op.look_id, op.kind_look, ctx.t_ns, self.label)
            return self._output(None, looks=(request,))
        assert op.look_id is not None
        result = self.results.pop(op.look_id, None)
        if result is None:
            if elapsed < LOOK_TIMEOUT_MS:
                return self._output(None)
            self.pending_look = None
            self.last_seen = []
            self.record.lost_looks += 1
            self.lost_streak += 1
            self.extra["look_timeouts"] = self.extra.get("look_timeouts", 0) + 1
            return None
        self.pending_look = None
        self.last_seen = sightings(result, t_ms=ctx.t_ms)
        if result.get("status") == "ok":
            self.record.looks_ok += 1
        level = result_light(result)
        if level is not None:
            self.extra["light"] = level
        near = self.tracker.relative(self.odom) if self.odom is not None else None
        # after the act the thing may be anywhere: the label decides, the old place does not
        gate = 3000 if self.phase == "watch" else PICK_GATE_MM
        seen = pick(self.last_seen, self._key(), near, gate_mm=gate)
        if seen is not None:
            seen = centre_of(seen, self.info.d_mm)
        if seen is not None and self.odom is not None:
            self.tracker.update(fix_of(seen, self.odom))
            self.extra["last_size"] = [seen.w_mm, seen.h_mm, seen.d_mm]
            self.lost_streak = 0
        else:
            self.record.lost_looks += 1
            self.lost_streak += 1
        return None

    def _run_skill(self, op: _Op, ctx: ActContext, elapsed: int) -> ActOutput | None:
        for start, stop, value in op.plan:
            if start <= elapsed < stop:
                self.mouth = value
                break
        if not op.sent:
            op.sent = True
            self.record.bursts += 1
            return self._output(None, do=op.skill)
        policy = ctx.proprio.policy or (ctx.state.policy if ctx.state is not None else "")
        if policy and policy not in ("walk", "stand"):
            op.left_policy = True
        limit = self.calib.skill_ms.get(op.skill or "", 2000) + 1500
        done = (op.left_policy and policy in ("walk", "stand")) or elapsed >= limit
        if not done:
            return self._output(None)
        self.last_step_ms = self.now_ms  # a skill moves the feet too: settle before walking
        return None

    # ── deciding what comes next ───────────────────────────────────────────

    def _begin(self, ctx: ActContext) -> None:
        self.phase = "find"
        if self.verb in ACT_VERBS and ctx.light in DARK_LIGHTS and not self._still_has_it(ctx):
            # it cannot see the thing in the dark (the eye names no things there): nothing to do
            self._finish_now(ctx, "dark")
            return
        if self.verb in ("grab", "carry") and not self.info.carry_ok:
            self._finish_now(ctx, "not_allowed")
            return
        if self.verb in ("carry", "put_down"):
            # it is in the beak (the camera cannot see it there): the mouth joint says so
            self.phase = "act"
            self._call_read_mouth(ctx)
            if not oc.in_beak(self.mouth_mrad):
                self.outcome = "slipped" if self.verb == "carry" else "missed"
                self.extra["beak"] = "slipped" if self.verb == "carry" else None
                self._finish_now(ctx, "done")
                return
            self.mouth_held = True
            self._queue_act(ctx)
            return
        self._queue_look_at(self._believed())

    def _still_has_it(self, ctx: ActContext) -> bool:
        """Owner 「一律放寬」 (the phase-4 landing): does the duck's own perception still have the
        thing, so an act may start in the dark? A sighting its tracker still keeps (its own, or
        the step before's on the same thing, :class:`ThingMemory`), or — for ``carry`` /
        ``put_down`` — the thing in its beak (the mouth joint is read right after, in
        :meth:`_begin`; a beak found empty ends the step as it does in the light)."""
        if self.verb in HELD_VERBS:
            return True
        if not self.tracker.fresh(ctx.t_ms) and self.memory is not None:
            recalled = self.memory.recall(self.command.thing, self.epoch, ctx.t_ms)
            if recalled is not None:
                self.tracker = recalled
        return self.tracker.fresh(ctx.t_ms)

    def remembered(self) -> ThingMemory:
        """What this step's perception has of its thing, for the next step (duck-body keeps it
        between steps: :func:`executor`)."""
        return ThingMemory(self.command.thing, self.epoch, self.tracker)

    def _key(self) -> str | None:
        """The label a sighting must carry to be the target (``None``: any, by place)."""
        return None if self.any_thing else (self.label or None)

    def _believed(self) -> tuple[int, int] | None:
        if self.odom is not None and self.tracker.last is not None:
            return self.tracker.relative(self.odom)
        b, r = _opt_int(self.params.get("b_mrad")), _opt_int(self.params.get("r_mm"))
        if b is not None and r is not None:
            return int(round(r * math.cos(b / 1000.0))), int(round(r * math.sin(b / 1000.0)))
        return None

    def _look_pose(
        self, rel: tuple[int, int] | None, attempt: int = 0
    ) -> tuple[float, float, float, float]:
        """The head pose to look at a trunk-frame point: low for near things (full down under
        :data:`NEAR_LOOK_MM`, half down under :data:`MID_LOOK_MM`), turned toward it (±0.54 rad).
        Retries: 1 — the neighbouring pitch; 2, 3 — ±0.3 rad around the bearing; 4 — wide."""
        if rel is None:
            sweep = (0.0, AIM_YAW, -AIM_YAW, 0.0, AIM_YAW, -AIM_YAW)
            pitch = HOME_HEAD if attempt < 3 else HALF_DOWN
            return (pitch[0], pitch[1], sweep[attempt % len(sweep)], 0.0)
        x, y = rel
        r = math.hypot(x, y)
        base = FULL_DOWN if r < NEAR_LOOK_MM else HALF_DOWN if r < MID_LOOK_MM else HOME_HEAD
        yaw = math.atan2(y, x)
        step = attempt % 5
        if step == 1:
            if base is FULL_DOWN:
                base = HALF_DOWN if r > 180 else FULL_DOWN
            elif base is HALF_DOWN:
                base = FULL_DOWN if r < 370 else HOME_HEAD
            else:
                base = HALF_DOWN
        elif step in (2, 3):
            yaw += -0.3 if step == 2 else 0.3
        elif step == 4:
            base, yaw = HALF_DOWN, 0.0
        yaw = max(-AIM_YAW, min(AIM_YAW, yaw))
        return (base[0], base[1], round(yaw, 3), 0.0)

    def _queue_look_at(
        self, rel: tuple[int, int] | None = None, attempt: int = 0, *, fixed: bool = False
    ) -> None:
        """Turn the head to the thing and look. The pose is chosen when the look comes up (from
        where the thing is believed to be *then*, after any move queued before it), unless
        ``fixed`` gives a point to look at."""
        self.queue.append(_Op("aim", target=rel if fixed else None, attempt=attempt))
        self.queue.append(_Op("wait", ms=0, call="aim_settle", still=True))
        self.queue.append(_Op("look", kind_look="fixate"))

    def _decide(self, ctx: ActContext) -> None:
        phase = self.phase
        if phase == "find":
            if (
                self.tracker.last is None
                or not self.tracker.fresh(ctx.t_ms)
                or (self.lost_streak >= SEARCH_AFTER_LOST)
            ):
                if self.lost_streak >= MAX_LOST_LOOKS:
                    self._finish_now(ctx, "lost")
                    return
                believed = self._believed()
                if (
                    believed is not None
                    and self.lost_streak >= 1
                    and self.record.backs < 2
                    and math.hypot(*believed) < self.calib.eye_near_mm + 20
                ):
                    # too close to see (under the full-down look's near edge): step back
                    self._queue_move(prims.back(self.calib.back_ms), head_down=False)
                    self.record.backs += 1
                elif (
                    believed is not None
                    and self.lost_streak >= 1
                    and self.record.turns < 3
                    and abs(math.atan2(believed[1], believed[0])) * 1000 > OUT_OF_REACH_MRAD
                ):
                    # beside the duck, where the head cannot turn: turn the body toward it
                    self._queue_move(
                        prims.back_arc(left=believed[1] > 0, ms=self.calib.back_arc_ms),
                        head_down=False,
                    )
                    self.record.turns += 1
                self._queue_look_at(attempt=self.lost_streak)
                return
            self.before = self.tracker.last
            if self.verb == "look":
                self.outcome = "seen"
                self.phase = "report"
                self._queue_home_and_report()
                return
            self.phase = "approach"
        if self.phase == "approach":
            self._decide_approach(ctx)
            return
        if self.phase == "confirm":
            self._decide_confirm(ctx)
            return
        if self.phase == "act":
            self._after_act(ctx)
            return
        if self.phase == "watch":
            self._after_watch(ctx)
            return
        if self.phase == "report":
            self._finish_now(ctx, self.ended_by)
            return
        raise ValueError(f"phase {phase}")

    # the verb's window, in the trunk frame (ahead, left)
    def _window(self) -> tuple[Window, int]:
        c = self.calib
        if self.verb in ("peck", "nibble"):
            return c.peck, 1
        if self.verb == "grab":
            return c.grab, 1
        if self.verb == "kick":
            return c.kick_window, (1 if self._foot() == "left" else -1)
        if self.verb == "tap":
            half = (self.info.d_mm or 60) // 2
            gap = c.tap_gap_mm + half
            return Window((gap - 15, gap + 5), (-20, 20)), 1
        if self.verb == "dribble":
            return Window((105, 135), (-20, 20)), 1
        # bump: the near face bump_gap ahead; the centre half a depth beyond
        half = (self.info.d_mm or 60) // 2
        gap = c.bump_gap_mm + half
        return Window((gap - 15, gap + 15), (-25, 25)), 1

    def _approach_window(self) -> tuple[Window, int, bool]:
        """Where the thing must be at the final confirming look. A window nearer than the eye can
        see (``eye_near_mm``: the full-down look sees the floor from ~95 mm, spec §4.1; a fin
        under the beak is below the camera) is confirmed one creep farther out, and the last
        creep is made blind (odometry-measured, ~42 mm ± 4 live)."""
        window, sign = self._window()
        if window.ahead[0] >= self.calib.eye_near_mm:
            return window, sign, False
        near = max(self.calib.eye_near_mm, window.ahead[0] + BLIND_MIN_MM)
        pre = Window((near, window.ahead[1] + BLIND_MAX_MM), window.side)
        return pre, sign, True

    def _foot(self) -> str:
        if self.foot in ("left", "right"):
            return self.foot
        rel = self.tracker.relative(self.odom) if self.odom is not None else None
        self.foot = "left" if rel is None or rel[1] >= 0 else "right"
        return self.foot

    def _target_rel(self) -> tuple[int, int] | None:
        if self.odom is None:
            return None
        return self.tracker.relative(self.odom)

    def _decide_approach(self, ctx: ActContext) -> None:
        """One move toward the verb's window, then a look (the M2 controller, curiosity §5.7:
        steer the trunk toward where it must stand, not toward the thing)."""
        rel = self._target_rel()
        if rel is None or not self.tracker.fresh(ctx.t_ms) or self.lost_streak >= SEARCH_AFTER_LOST:
            self.phase = "find"
            self._decide(ctx)
            return
        if self.verb == "kick" and self.params.get("foot", "auto") == "auto" and self.cone is None:
            self.foot = "left" if rel[1] >= 0 else "right"  # the foot on the ball's side (M2)
        window, sign, _ = self._approach_window()
        x, y = rel
        tol = self.patience.tol_pct
        if window.contains(x, y * sign, tol):
            self.phase = "confirm"
            self.queue.append(_Op("wait", ms=200))
            self._queue_look_at(rel)
            return
        ca, cs = window.centre
        dist = math.hypot(x, y)
        near = dist - ca < FAR_MM
        if near and self.align_started_ms is None:
            self.align_started_ms = ctx.t_ms  # the patience budget counts the final alignment
        if (
            self.align_started_ms is not None
            and ctx.t_ms - self.align_started_ms > self.patience.align_budget_ms
        ):
            if self.patience.tier == "hasty" and window.contains(x, y * sign, tol * 3 // 2):
                self.extra["rushed"] = True
                self.phase = "confirm"
                self._decide_confirm(ctx, force=True)
                return
            self.outcome = "missed"
            self._report_soon("gave_up")
            return
        ex = x - ca  # how far the thing is beyond the window's centre (trunk frame)
        ey = y - cs * sign  # and to the side of it
        half_x = (window.ahead[1] - window.ahead[0]) * tol // 200
        aim = math.atan2(ey, max(ex, 30))  # toward where the trunk must stand
        off_b = math.atan2(y, x) - math.atan2(cs * sign, ca)
        steer = int(round(prims.clip(2.0 * aim, 1.0) * 1000))  # M2's gain
        # A2 only right after a sight: the guard lets a head-down walk go only toward a thing
        # seen in the last 3 s, and the move starts after the head settles
        last = self.tracker.last
        head_down = (
            self.head_down_allowed
            and near
            and last is not None
            and ctx.t_ms - last.t_ms <= A2_FRESH_MS
        )
        if not near:
            ms = self._leg_ms(dist - ca - LEG_STOP_SHORT_MM)
            if abs(aim) > 0.5:
                ms = min(ms, 1000)  # a big turn: a shorter leg, then look again
            # turn the bearing error over the leg (robotd's arcs turn ~0.5–0.9 of the command,
            # live), not 2·bearing for the whole leg: two 2 s legs at 2·bearing overturned by
            # 60° and left the thing beside the duck, out of the head's reach (phase 3 live)
            wz = prims.clip(aim / (LEG_TURN_EFFICIENCY * ms / 1000.0), 1.0)
            self._queue_move(
                prims.Prim("leg", prims.APPROACH_VX_MM, int(round(wz * 1000)), ms), head_down=False
            )
            self.record.legs += 1
        elif abs(off_b) > TURN_MRAD / 1000.0 or (ex < 30 and abs(ey) > BIG_SIDE_MM):
            # well off to the side: a reverse arc (1.0 s ≈ ±33° under robotd) turns toward it
            self._queue_move(
                prims.back_arc(left=ey > 0, ms=self.calib.back_arc_ms), head_down=False
            )
            self.record.turns += 1
        elif ex > half_x * 7 // 10 or (ex > 10 and abs(ey) > 0):
            # forward by what is missing (the step table), steering the trunk to its stand point
            ms = min(NEAR_STEP_MAX_MS, step_ms(self.calib.fwd_table, max(ex, 6)))
            if ms <= 300:
                self._queue_move(prims.trim(steer), head_down=head_down)
                self.record.trims += 1
            else:
                self._queue_move(prims.creep(ms, steer), head_down=head_down)
                self.record.creeps += 1
        else:
            # too near (or sideways with no room ahead): a straight step back makes room; the
            # next creep steers (a 1.0 s reverse arc would swing it 33°, too far for a few cm)
            ms = step_ms(self.calib.back_table, min(ex, -20))
            self._queue_move(prims.back(max(1000, ms)), head_down=False)
            self.record.backs += 1
        self._queue_look_at(rel)

    def _leg_ms(self, want_mm: float) -> int:
        """How long a leg must be to cover ``want_mm`` from standing, from the live step model
        (0.5 s → ``creep``, 1.0 s → ``leg1``, linear beyond), clamped to 0.5–2.0 s."""
        creep = self.calib.step_mm.get("creep", 42)
        leg1 = self.calib.step_mm.get("leg1", 125)
        per_ms = max(0.05, (leg1 - creep) / 500.0)
        return int(max(500, min(2000, 500 + (want_mm - creep) / per_ms)))

    def _queue_move(self, prim: prims.Prim, *, head_down: bool) -> None:
        """A move with the head at HOME (settled 0.3 s, spec §4.2 rule 3), or — owner A2, the
        final approach toward the just-seen thing — in the validated half-down walking pose."""
        pose = HALF_DOWN if head_down else HOME_HEAD
        # the head goes there first (the look pose may differ): wait for it unless unchanged
        self.queue.append(_Op("head", head=pose))
        self.queue.append(_Op("wait", ms=0, call="settle_head"))
        self.queue.append(_Op("move", prim=prim, head_down=head_down))
        self.queue.append(_Op("wait", ms=TAIL_MS))

    def _decide_confirm(self, ctx: ActContext, force: bool = False) -> None:
        rel = self._target_rel()
        window, sign, _ = self._approach_window()
        if rel is None:
            self.phase = "find"
            self._decide(ctx)
            return
        inside = window.contains(rel[0], rel[1] * sign, self.patience.tol_pct)
        if not force and not inside:
            self.phase = "approach"
            self._decide_approach(ctx)
            return
        if not force and not self.tracker.agreed():
            if self.record.looks > 40:
                self.outcome = "missed"
                self._report_soon("gave_up")
                return
            self.queue.append(_Op("wait", ms=200))
            self._queue_look_at(rel)
            return
        if self.verb == "kick":
            self.cone = self._cone_check()
            if not self.cone.ok:
                if self.cone.foot != self._foot():
                    self.foot = self.cone.foot  # the other foot's cone is clear: realign
                    self.phase = "approach"
                    self._decide_approach(ctx)
                    return
                self.outcome = "missed"
                self._report_soon("cone_blocked")
                return
            self.foot = self.cone.foot
        self.extra["err_mm"] = _err_mm(rel, window, sign)
        self.before = self.tracker.last
        self.phase = "act"
        self._queue_act(ctx)

    def _cone_check(self) -> cone_mod.ConeCheck:
        last = self.tracker.last
        assert last is not None and self.odom is not None
        leash = self.params.get("leash")
        circle = None
        if isinstance(leash, Mapping):
            circle = cone_mod.Circle(
                int(leash["x_mm"]), int(leash["y_mm"]), int(leash["r_mm"]), "leash"
            )
        keep = []
        for zone in self.params.get("keep_out") or []:
            if isinstance(zone, Mapping):
                keep.append(
                    cone_mod.Circle(
                        int(zone["x_mm"]),
                        int(zone["y_mm"]),
                        int(zone["r_mm"]),
                        str(zone.get("why", "keep_out")),
                    )
                )
        roll = self.calib.roll_mm.get(
            self.info.cls, self.calib.roll_mm.get("ball_slow", (500, 950))
        )
        return cone_mod.choose(
            (last.x_mm, last.y_mm),
            self.odom.yaw_mrad,
            self._foot(),
            roll,
            leash=circle,
            keep_out=keep,
            foot_offset_mrad=self.calib.kick_dir_mrad,
            half_mrad=self.calib.kick_cone_half_mrad,
        )

    def _queue_act(self, ctx: ActContext) -> None:
        c = self.calib
        self.queue.append(_Op("head", head=HOME_HEAD))
        self.queue.append(_Op("wait", ms=0, call="settle_head"))
        verb = self.verb
        if verb not in ("carry", "put_down") and self._approach_window()[2]:
            # the last step, blind: the window is nearer than the eye sees; its length from the
            # measured step table (live: ±4 mm)
            rel = self._target_rel()
            window, _ = self._window()
            want = (rel[0] if rel is not None else window.centre[0] + 42) - window.centre[0]
            ms = max(300, min(1000, step_ms(self.calib.fwd_table, want)))
            self.queue.append(_Op("move", prim=prims.creep(ms, 0)))
            self.queue.append(_Op("wait", ms=TAIL_MS))
            self.extra["blind_step"] = True
            self.extra["blind_mm"] = int(round(want))
        self.queue.append(_Op("wait", ms=max(self.patience.settle_ms, 150), still=True))
        if verb == "bump":
            bold = int(self.params.get("bold", 0) or 0) >= 6000 and self.info.cls in (
                "ball_slow",
                "ball_fast",
                "box",
                "cup",
            )
            self.queue.append(_Op("move", prim=prims.bump(BOLD_BUMP_MS if bold else 600)))
            self.queue.append(_Op("wait", ms=TAIL_MS + 600))
        elif verb in ("peck", "nibble"):
            if self.info.h_mm is not None and self.info.h_mm < oc.PECK_MIN_H_MM:
                self.outcome = "unreachable"
                self._report_soon("done")
                return
            plan: tuple[tuple[int, int, float], ...] = ()
            if verb == "nibble":
                (a0, a1), (b0, b1) = NIBBLE_PULSES_MS
                plan = (
                    (200, a0, 1.0),
                    (a0, a1, 0.0),
                    (a1, b0, 1.0),
                    (b0, b1, 0.0),
                    (b1, 2300, 1.0),
                    (2300, 99_000, 0.0),
                )
            self.queue.append(_Op("skill", skill="ground_pick", plan=plan))
            self.queue.append(_Op("mouth", mouth=0.0))
            self.queue.append(_Op("wait", ms=500))
            self.queue.append(_Op("call", call="read_mouth"))
        elif verb == "grab":
            plan = ((c.mouth_open_ms, c.mouth_close_ms, 1.0), (c.mouth_close_ms, 99_000, 0.0))
            self.queue.append(_Op("skill", skill="ground_pick", plan=plan))
            self.queue.append(_Op("mouth", mouth=0.0))
            self.queue.append(_Op("wait", ms=GRAB_HOLD_MS))
            self.queue.append(_Op("call", call="read_mouth"))
        elif verb == "carry":
            self.queue.append(_Op("call", call="carry_leg"))
            return
        elif verb == "put_down":
            self.queue.append(_Op("mouth", mouth=1.0))
            self.queue.append(_Op("wait", ms=PUT_DOWN_MS))
            self.queue.append(_Op("mouth", mouth=0.0))
            self.mouth_held = False
        elif verb == "tap":
            taps = max(1, min(3, int(self.params.get("taps", 1) or 1)))
            for _ in range(taps):
                self.queue.append(_Op("head", head=FULL_DOWN))
                self.queue.append(_Op("wait", ms=300))
                self.queue.append(_Op("head", head=HOME_HEAD))
                self.queue.append(_Op("wait", ms=300))
                self.record.bursts += 1
        elif verb == "kick":
            self.queue.append(_Op("skill", skill=f"kick_{self._foot()}"))
            self.queue.append(_Op("wait", ms=KICK_WATCH_MS))
        elif verb == "dribble":
            self.queue.append(_Op("move", prim=prims.creep(600)))
            self.queue.append(_Op("wait", ms=TAIL_MS + 400))

    def _call_read_mouth(self, ctx: ActContext) -> None:
        if ctx.state is not None and len(ctx.state.joints) > 9:
            self.mouth_mrad = int(round(ctx.state.joints[9] * 1000.0))
        held = oc.in_beak(self.mouth_mrad)
        if self.verb == "grab":
            self.mouth_held = held
            if not held:
                self.mouth = 0.0

    def _call_carry_leg(self, ctx: ActContext) -> None:
        relay_ms = int(self.params.get("relay_ms") or self.patience.relay_ms)
        # a greedy (hasty) duck's long leg is allowed: it will drop the thing (the spec's 呆萌)
        relay_ms = max(500, min(2000, relay_ms))
        self.extra["relay_ms"] = relay_ms
        dest_b = int(self.params.get("dest_b_mrad", 0) or 0)
        wz = int(round(prims.clip(2.0 * dest_b / 1000.0, 0.5) * 1000))
        self.mouth_held = True
        self.queue.append(_Op("move", prim=prims.Prim("leg", prims.APPROACH_VX_MM, wz, relay_ms)))
        self.queue.append(_Op("wait", ms=TAIL_MS))
        self.queue.append(_Op("call", call="read_mouth"))

    def _after_act(self, ctx: ActContext) -> None:
        verb = self.verb
        if verb == "carry":
            # the tether rule (F2): progress under 30 % in the leg, or turned > 20 °
            moves = self.extra.get("moves", [])
            leg = moves[-1] if moves else None
            if leg is not None:
                expected = self.calib.step_mm.get("leg1", 140) * leg["ms"] // 1000
                dyaw = (
                    abs(self.odom.delta(self.start_odom)[2]) if self.odom and self.start_odom else 0
                )
                if leg["fwd_mm"] < TETHER_PROGRESS * expected or dyaw > TETHER_YAW_MRAD:
                    self.mouth = 1.0
                    self.guards.append({"guard": "tethered"})
                    self.extra["beak"] = "tethered"
                    self.queue.append(_Op("wait", ms=300))
                    self.queue.append(_Op("mouth", mouth=0.0))
                    self.queue.append(_Op("move", prim=prims.back(600)))
                    self.queue.append(_Op("wait", ms=TAIL_MS))
                    self.phase = "watch"
                    return
            held = oc.in_beak(self.mouth_mrad)
            self.extra["beak"] = "held" if held else "slipped"
            self.extra["dropped_by"] = "chosen" if held else "slipped_walking"
            if leg is not None and held:
                self.extra["carried_mm"] = max(0, int(leg["fwd_mm"]))
            self.queue.append(_Op("mouth", mouth=1.0))
            self.queue.append(_Op("wait", ms=PUT_DOWN_MS))
            self.queue.append(_Op("mouth", mouth=0.0))
            self.mouth_held = False
        self.phase = "watch"
        self.tracker.moving = True
        rel = self._target_rel()
        fixed = False
        if verb in ("carry", "put_down", "grab"):
            rel = (int(self.calib.beak_tip_mm[0]) + 15, 0)  # set down under the beak
            fixed = True
        if verb in ("kick", "dribble") and self.odom is not None and self.before is not None:
            # first where it was (still there = a miss), then along the kick's direction
            x0, y0 = self.odom.to_trunk(self.before.x_mm, self.before.y_mm)
            heading = self.odom.yaw_mrad / 1000.0
            d = (self.cone.direction_mrad / 1000.0 - heading) if self.cone is not None else 0.0
            plan = [(x0, y0)] + [
                (int(x0 + r * math.cos(d)), int(y0 + r * math.sin(d))) for r in (600, 1000)
            ]
            self.extra["watch_plan"] = plan
            self.extra["watch_idx"] = 0
            rel = plan[0]
            fixed = True
        self._queue_look_at(rel, fixed=fixed)

    def _after_watch(self, ctx: ActContext) -> None:
        last = self.tracker.last
        since = self.before.t_ms if self.before is not None else -1
        seen_now = last is not None and last.t_ms > since
        plan = self.extra.get("watch_plan")
        if plan is not None:
            idx = int(self.extra.get("watch_idx", 0))
            if idx == 0 and not seen_now:
                self.extra["left_spot"] = True  # not where it was: it went somewhere
            if not seen_now and idx + 1 < len(plan):
                self.extra["watch_idx"] = idx + 1
                self._queue_look_at(plan[idx + 1], fixed=True)
                return
            self.after = last if seen_now else None
            self._classify()
            self._queue_home_and_report()
            return
        if not seen_now and self.extra.get("watch_tries", 0) < 3:
            self.extra["watch_tries"] = self.extra.get("watch_tries", 0) + 1
            rel = self._target_rel()
            self._queue_look_at(rel, attempt=self.extra["watch_tries"])
            return
        self.after = last if seen_now else None
        self._classify()
        self._queue_home_and_report()

    def _classify(self) -> None:
        verb = self.verb
        before = (self.before.x_mm, self.before.y_mm) if self.before else None
        after = (self.after.x_mm, self.after.y_mm) if self.after else None
        moved = oc.moved_mm(before, after)
        self.extra["moved_mm"] = moved
        if moved is not None and before is not None and after is not None and self.odom is not None:
            if moved >= oc.MOVED_MM:
                self.extra["dir_mrad"] = oc.direction_mrad(before, after, self.odom.yaw_mrad)
        h0 = self.before.h_mm if self.before else None
        h1 = self.after.h_mm if self.after else None
        if verb == "bump":
            moves = [m for m in self.extra.get("moves", []) if m["prim"] == "bump"]
            fwd = moves[-1]["fwd_mm"] if moves else 0
            expected = self.calib.step_mm.get("bump", 60)
            witnessed = bool(self.record.halted)
            gap = None
            if after is not None and self.odom is not None:
                x, _ = self.odom.to_trunk(*after)
                gap = x - (self.info.d_mm or 60) // 2 - self.calib.front_mm
            self.outcome = oc.classify_bump(
                moved=moved,
                h_before=h0,
                h_after=h1,
                progress_mm=fwd,
                expected_mm=expected,
                witnessed=witnessed,
                gap_after_mm=gap,
            )
            self.extra["stuck"] = self.outcome == "stuck"
        elif verb in ("peck", "nibble"):
            self.outcome = oc.classify_peck(mouth_mrad=self.mouth_mrad, moved=moved, reachable=True)
            self.extra["gap_bin"] = oc.gap_bin(self.mouth_mrad)
        elif verb == "grab":
            self.outcome = "held" if self.mouth_held else "missed"
            self.extra["beak"] = self.outcome
            self.extra["gap_bin"] = oc.gap_bin(self.mouth_mrad)
        elif verb == "carry":
            self.outcome = self.extra.get("beak", "held")
        elif verb == "put_down":
            self.outcome = "placed" if after is not None else "missed"
        elif verb == "tap":
            self.outcome = oc.classify_tap(moved=moved, h_before=h0, h_after=h1)
        elif verb in ("kick", "dribble"):
            self.outcome = oc.classify_kick(moved)
            if moved is not None and moved >= oc.ROLLED_MM:
                self.extra["rolled_mm"] = moved
            if moved is None and self.extra.get("left_spot"):
                self.outcome = "moved"  # gone from its spot, not found: where to is unknown
        if self.after is None and verb in ("kick", "dribble", "bump", "tap"):
            self.extra["lost"] = True

    def _queue_home_and_report(self) -> None:
        self.phase = "report"
        self.queue.append(_Op("head", head=HOME_HEAD))
        self.queue.append(_Op("wait", ms=HEAD_MOVE_MS if self.head != HOME_HEAD else 0))

    def _report_soon(self, ended_by: str) -> None:
        self.ended_by = ended_by
        self.queue.clear()
        self._queue_home_and_report()

    def _finish_now(self, ctx: ActContext, ended_by: str) -> None:
        self.ended_by = ended_by
        self.finished = self._report(ctx)

    def _end(self, ctx: ActContext, ended_by: str) -> ActOutput:
        if self.finished is not None:
            return self.finished
        self.queue.clear()
        self.op = None
        if self.mouth_held or self.mouth > 0.0:
            self.mouth = (
                0.0 if not self.mouth_held else 1.0
            )  # held: let go (never carry while falling)
        self.ended_by = ended_by
        self.finished = self._report(ctx)
        return self.finished

    # ── the report ──────────────────────────────────────────────────────────

    def _report(self, ctx: ActContext) -> ActOutput:
        percepts = [self._curio_step(ctx)]
        sense = self._sense_thing(ctx)
        if sense is not None:
            percepts.append(sense)
        for guard in self.guards:
            percepts.append(
                self._event("body.guard", ctx, {**guard, "act_id": self.act_id, "verb": self.verb})
            )
        head = None if prims.head_is_home(self.head) else self.head
        return ActOutput(
            twist=None, head=head, mouth=self.mouth, percepts=tuple(percepts), done=True
        )

    def _event(self, kind: str, ctx: ActContext, fields: Mapping[str, Any]) -> dict[str, Any]:
        event = {"kind": kind, "src": self.src, "t_ms": int(ctx.t_ms), "pv": self.pv, **fields}
        problems = validate(event)
        if problems:
            raise PerceptError(f"{kind}: " + "; ".join(problems))
        return event

    def _curio_step(self, ctx: ActContext) -> dict[str, Any]:
        rec = self.record
        outcome = self.outcome or self.ended_by
        ok = self.ended_by == "done" and outcome not in ("missed", "lost", "unreachable")
        odom = {"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0}
        if self.odom is not None and self.start_odom is not None:
            dx, dy, dyaw = self.odom.delta(self.start_odom)
            odom = {"dx_mm": dx, "dy_mm": dy, "dyaw_mrad": dyaw}
        after = self.after or (self.tracker.last if self.verb == "look" else None)
        thing_after: dict[str, Any] = {"seen": after is not None}
        if after is not None:
            thing_after.update({"x_mm": after.x_mm, "y_mm": after.y_mm, "err_mm": 12})
        fields: dict[str, Any] = {
            "id": self.act_id,
            "act_id": self.act_id,
            "step": int(self.command.step or 0),
            "verb": self.verb,
            "thing": self.label or None,
            "cls": self.info.cls,
            "ok": bool(ok),
            "outcome": outcome,
            "ended_by": self.ended_by,
            "ms": int(ctx.t_ms - (self.t0_ms if self.t0_ms is not None else ctx.t_ms)),
            "moved_mm": _opt_int(self.extra.get("moved_mm")),
            "dir_mrad": _opt_int(self.extra.get("dir_mrad")),
            "rolled_mm": _opt_int(self.extra.get("rolled_mm")),
            "tipped": outcome == "tipped",
            "beak": self.extra.get("beak"),
            "gap_bin": _opt_int(self.extra.get("gap_bin")),
            "mouth_mrad": self.mouth_mrad,
            "carried_mm": _opt_int(self.extra.get("carried_mm")),
            "dropped_by": self.extra.get("dropped_by"),
            "stuck": bool(self.extra.get("stuck", False)),
            "lost": bool(self.extra.get("lost", False) or self.ended_by == "lost"),
            "sound": oc.SOUND_UNKNOWN,
            "err_mm": _opt_int(self.extra.get("err_mm")),
            "settle_ms": self.patience.settle_ms,
            "tol_pct": self.patience.tol_pct,
            "tier": self.patience.tier,
            "bursts": rec.bursts,
            "looks": rec.looks,
            "looks_lost": rec.lost_looks,
            "legs": rec.legs,
            "creeps": rec.creeps,
            "trims": rec.trims,
            "turns": rec.turns,
            "backs": rec.backs,
            "head_fallback": rec.head_fallback,
            "rushed": bool(self.extra.get("rushed", False)),
            "blind_step": bool(self.extra.get("blind_step", False)),
            "blind_mm": _opt_int(self.extra.get("blind_mm")),
            "foot": self.foot if self.verb == "kick" else None,
            "cone": None
            if self.cone is None
            else {"ok": self.cone.ok, "why": self.cone.why, "dir_mrad": self.cone.direction_mrad},
            "deliberate": True,
            "fell": self.fell,
            "light": self.extra.get("light", ctx.light),
            "thing_after": thing_after,
            "odom": odom,
            "perception": self.perception,
            "source": "robot" if self.src == "robot" else "physics",
            "violations": len(rec.violations),
        }
        return self._event("body.curio_step", ctx, fields)

    def _sense_thing(self, ctx: ActContext) -> dict[str, Any] | None:
        last = self.tracker.last
        if last is None:
            if self.verb == "look" or self.ended_by == "lost":
                return self._event(
                    "sense.thing",
                    ctx,
                    {
                        "act_id": self.act_id,
                        "things": [{"id": self.label or None, "edge": "lost", "seen": False}],
                    },
                )
            return None
        edge = "step"
        if self.before is not None and self.after is not None:
            moved = oc.moved_mm(
                (self.before.x_mm, self.before.y_mm), (self.after.x_mm, self.after.y_mm)
            )
            if (moved or 0) >= 300:
                edge = "moved"
        return self._event(
            "sense.thing",
            ctx,
            {
                "act_id": self.act_id,
                "things": [
                    {
                        "id": self.label or None,
                        "cls": self.info.cls,
                        "x_mm": last.x_mm,
                        "y_mm": last.y_mm,
                        "look_id": last.look_id,
                        "seen_t_ms": last.t_ms,
                        "edge": edge,
                        "seen": True,
                    }
                ],
            },
        )


# ── helpers ──────────────────────────────────────────────────────────────────


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(round(value))


def _patience(params: Mapping[str, Any]) -> Patience:
    tier = params.get("tier")
    if isinstance(tier, str) and tier in TIERS:
        return TIERS[tier]
    wits = params.get("wits")
    if isinstance(wits, int) and not isinstance(wits, bool):
        return Patience.from_wits(wits)
    return TIERS["normal"]


def _err_mm(rel: tuple[int, int], window: Window, sign: int) -> int:
    ca, cs = window.centre
    return int(round(math.hypot(rel[0] - ca, rel[1] * sign - cs)))


@dataclass(frozen=True)
class CurioLook(LookRequest):
    """A :class:`~duckmind.duckbody.protocol.LookRequest` that also says which look it is for
    duck-eye (``eye.look``'s ``kind``, ``mode`` and integer/short-text ``hints``), the same shape
    as lookplan's PlannedLook (the body server reads them with ``getattr``)."""

    kind: str = "fixate"
    mode: str = "curio"
    hints: Mapping[str, Any] = field(default_factory=dict)


def _look_request(look_id: str, kind: str, t_ns: int, label: str) -> LookRequest:
    """A fixate look for the ``things`` task, in curiosity mode, with the target as a hint."""
    hints = {"target_id": label} if label else {}
    return CurioLook(
        look_id=look_id, tasks=("things",), not_before_ns=t_ns, kind=kind, mode="curio", hints=hints
    )


# the last curiosity step of each body loop (weak: it goes with the loop): what its perception
# still has of its thing is handed to the next step (used only to start in the dark)
_LAST_STEP: weakref.WeakKeyDictionary[Any, CurioStep] = weakref.WeakKeyDictionary()


def executor(command: Command, t_ms: int, loop: Any) -> CurioStep:
    """duck-body's executor factory for ``duck.curio`` (``BodyConfig.executors["duck.curio"]``):
    the step stamps its percepts with the body loop's ``src`` and ``pv``, and carries what the
    loop's last step still had of its thing (:class:`ThingMemory`: a step on the same thing may
    start in the dark on it, owner 「一律放寬」)."""
    del t_ms
    if not isinstance(command, CurioCmd):
        raise TypeError("duck.curio only")
    cfg = getattr(loop, "cfg", None)
    src = str(getattr(cfg, "src", "live"))
    pv = str(getattr(cfg, "pv", "p:0000000000000000"))
    try:
        last = _LAST_STEP.get(loop)
    except TypeError:  # (a stand-in loop that cannot be weakly referenced: no memory)
        last = None
    step = CurioStep(
        command,
        src=src,
        pv=pv,
        perception="robot" if src == "robot" else "camera",
        memory=last.remembered() if last is not None else None,
    )
    try:
        _LAST_STEP[loop] = step
    except TypeError:
        pass
    return step


def curio_step_problems(event: Mapping[str, Any]) -> list[str]:
    """The curiosity team's own check of a body.curio_step beyond percept-v1 (field list)."""
    out = validate(event)
    for name in (
        "verb",
        "outcome",
        "ended_by",
        "ok",
        "sound",
        "deliberate",
        "perception",
        "thing_after",
        "odom",
    ):
        if name not in event:
            out.append(f"body.curio_step.{name}: missing")
    if event.get("sound") != oc.SOUND_UNKNOWN:
        out.append("sound must be 'unknown' in phase 3 (owner A5)")
    if event.get("perception") not in PERCEPTIONS:
        out.append("perception must be camera or robot for Duck Core (physical spec §15.2)")
    after = event.get("thing_after")
    if (
        isinstance(after, Mapping)
        and not after.get("seen")
        and ("x_mm" in after or "y_mm" in after)
    ):
        out.append("thing_after has a position although it was not seen")
    return out


__all__ = [
    "ACT_VERBS",
    "CurioStep",
    "DARK_LIGHTS",
    "HELD_VERBS",
    "ThingMemory",
    "executor",
    "FULL_DOWN",
    "HALF_DOWN",
    "ThingInfo",
    "VERBS",
    "curio_step_problems",
]
