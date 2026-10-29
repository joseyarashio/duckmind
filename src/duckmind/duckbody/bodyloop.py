"""duck-body's brain, one ``robot.state`` tick at a time (pure; the server and the rig drive it).

:class:`BodyLoop` owns everything between the sensor doors and body.sock except I/O:

* the posture tap (:mod:`.posture`), body sense (:mod:`.proprio`), righting (:mod:`.righting`);
* the reflexes and the forbidden-command guard (:mod:`.reflex`): every intent it returns has
  passed :func:`~duckmind.duckbody.reflex.check_intent`; refused ones are counted, and a refused
  move becomes a zero move;
* one act at a time (spec §7.3): ``duck.walk``/``duck.dock``/``duck.look``/``duck.curio`` run
  the executors their factories build (walkexec, dockexec … — see :attr:`BodyConfig.executors`),
  ``duck.posture`` runs :class:`PostureAct`; ``duck.stop`` preempts; a new act while one runs is
  refused ``busy`` unless it says ``preempt``; an act past its ``deadline_ms`` ends ``deadline``;
* owner B4: when *it* senses the pickup (proprio's ``lifted``) the duck sits (``sit_toggle``
  when standing) and the running act ends ``lifted``; owner B3: an unsure lift only stops;
* where the duck finds its ways shut (:mod:`.cornered`, from the ToF, contacts, walk results and
  looks): the acts see ``ActContext.cornered``, and a ``duck.walk`` back step walkexec granted
  on it may reverse until it has backed :data:`~duckmind.duckbody.reflex.BACK_STEP_MAX_MM` by
  odometry (the guard's ``back_step_ok``; owner 2026-09-28 「2. 可以」);
* a move that backs away is judged by the body's own reflexes only (the ToF looks ahead):
  a dock's reverse leg or a back step is never stopped by what is in front of the duck;
* the percepts (:mod:`.aggregate`) at the moment they are sensed, as body.sock lines.

Time is the tick's ``t_ns`` (``CLOCK_MONOTONIC``); commands carry the server's receive time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.aggregate import Aggregator
from duckmind.duckbody.cornered import Cornered, Spot
from duckmind.duckbody.fixedmath import isqrt
from duckmind.duckbody.footfalls import Footfalls
from duckmind.duckbody.posture import PostureTap
from duckmind.duckbody.proprio import Proprio, ProprioConfig, ProprioOut
from duckmind.duckbody.reflex import (
    A2_WALK_MAX_MM,
    BACK_STEP_MAX_MM,
    FRESH_TARGET_MS,
    GuardContext,
    Limits,
    check_intent,
    verdict,
)
from duckmind.duckbody.righting import Righting, RightingConfig
from duckmind.duckbody.robotd_client import MoveStreamer, RobotState

MS = 1_000_000
# The head is "at HOME" (the guard's rule 3) when it was commanded HOME and every measured joint
# is within this of HOME. Live, the servos hold HOME about (−0.13, +0.10) rad off under gravity
# (phase 3, robotd --sim; A100 look.hold), so 0.05 never counted and the guard refused every live
# walk. 0.16 still tells the half-down look (−0.125/+0.175, more with the sag) from HOME; and a
# commanded look never counts, whatever the tolerance.
HEAD_HOME_TOL = 0.16
HEAD_RESEND_NS = 500_000_000
POSTURE_RETRY_NS = 3_000_000_000
POSTURE_TIMEOUT_NS = 6_000_000_000
POSTURE_HOLD_NS = 500_000_000
# A look asked for and not answered yet: the frame may be being captured (the guard's "no walking
# while capturing"); after this long without an answer the look is given up.
CAPTURE_PENDING_NS = 1_500_000_000
# The light level is worth this long for walking decisions; older, it counts as unknown ("none":
# the next walk glances first — spec §4.2's 60 s near-band rule).
LIGHT_FRESH_MS = 60_000

ExecutorFactory = Callable[[P.Command, int, "BodyLoop"], P.ActExecutor]
_RANK = {"black": 0, "dark": 1, "dim": 2, "bright": 3}


class PostureAct:
    """``duck.posture``: ``sit_toggle`` until robotd's policy says the wanted posture (retry every
    3 s), confirmed for 0.5 s, within 6 s (spec §5.2) → ``body.posture``."""

    def __init__(self, command: P.PostureCmd, start_ns: int) -> None:
        self.command: P.Command = command
        self.posture_cmd = command
        self.act_id = command.act_id
        self.start_ns = start_ns
        self.toggled_ns: int | None = None
        self.ok_since: int | None = None
        self.fell = False

    def _result(self, ctx: P.ActContext, got: str) -> P.ActOutput:
        body = {
            "kind": "body.posture",
            "want": self.posture_cmd.want,
            "got": got if got in ("stand", "sit") else "unknown",
            "ms": max(0, (ctx.t_ns - self.start_ns) // MS),
            "fell": self.fell,
            "act_id": self.act_id,
        }
        return P.ActOutput(percepts=(body,), done=True)

    def step(self, ctx: P.ActContext) -> P.ActOutput:
        want = self.posture_cmd.want
        got = ctx.proprio.posture
        if ctx.proprio.fallen:
            self.fell = True
        if got == want and ctx.proprio.policy in ("walk", "stand", "sit"):
            self.ok_since = ctx.t_ns if self.ok_since is None else self.ok_since
            if ctx.t_ns - self.ok_since >= POSTURE_HOLD_NS:
                return self._result(ctx, got)
        else:
            self.ok_since = None
        if ctx.t_ns - self.start_ns >= POSTURE_TIMEOUT_NS:
            return self._result(ctx, got)
        transitioning = ctx.proprio.policy in ("rise", "held", "")
        due = self.toggled_ns is None or ctx.t_ns - self.toggled_ns >= POSTURE_RETRY_NS
        if got != want and not transitioning and due and not ctx.proprio.fallen:
            self.toggled_ns = ctx.t_ns
            return P.ActOutput(do="sit_toggle")
        return P.ActOutput()

    def preempt(self, why: str, ctx: P.ActContext) -> P.ActOutput:
        return self._result(ctx, ctx.proprio.posture)


@dataclass
class BodyConfig:
    src: str = "live"
    pv: str = "p:0000000000000000"
    has_tof: bool = False
    limits: Limits = field(default_factory=Limits)
    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    righting: RightingConfig = field(default_factory=RightingConfig)
    executors: dict[str, ExecutorFactory] = field(default_factory=dict)
    sit_when_lifted: bool = True  # owner B4
    footfalls: Footfalls | None = None  # the step counter (needs the leg model)


@dataclass
class Tick:
    intents: list[P.Intent] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    pose: P.PoseSample | None = None
    eye_pose: bool = False
    looks: list[P.LookRequest] = field(default_factory=list)
    refused: list[tuple[P.Intent, str]] = field(default_factory=list)


class BodyLoop:
    def __init__(self, cfg: BodyConfig | None = None, tap: PostureTap | None = None) -> None:
        self.cfg = cfg or BodyConfig()
        self.tap = tap or PostureTap()
        self.proprio = Proprio(self.cfg.proprio, feet=self.cfg.footfalls)
        self.righting = Righting(self.cfg.righting)
        self.agg = Aggregator(self.cfg.src, self.cfg.pv)
        self.streamer = MoveStreamer()
        self.act: P.ActExecutor | None = None
        self.act_deadline_ms = 0
        self.pending_stop: str | None = None
        self.state: RobotState | None = None
        self.light = "none"
        self.light_ms: int | None = None  # when the eye last gave a level
        self.head_target: tuple[float, float, float, float] | None = None
        self.head_sent_ns: int | None = None
        self.head_home_since: int | None = None
        self.twist: tuple[int, int] | None = None
        self.mouth_sent: float | None = None
        self.refused_total = 0
        self.last_look_id: str | None = None
        self.last_prox: ProprioOut | None = None
        self.last_guard = GuardContext()
        self.pending_looks: dict[str, int] = {}  # look_id → t_ns asked (not answered yet)
        self.cornered = Cornered()  # where the duck found its ways shut (owner 「2. 可以」)
        self.spot: Spot | None = None  # the trunk by odometry this tick (mm, mrad)
        self.asked: tuple[int, int] | None = None  # the act's own twist last tick (before halts)
        self._back_from: tuple[str, int, int] | None = None  # a granted back step's start

    # ── body.sock commands ──

    def command(self, obj: Mapping[str, Any], t_ns: int) -> list[dict[str, Any]]:
        t_ms = t_ns // MS
        try:
            cmd = P.parse_command(obj)
        except P.ProtocolError as problem:
            act = obj.get("act_id") if isinstance(obj, Mapping) else None
            return [P.ack(act if isinstance(act, str) else None, False, t_ms, problem.reason)]
        if isinstance(cmd, P.StopCmd):
            self.pending_stop = "preempt"
            return [P.ack(cmd.act_id, True, t_ms)]
        if isinstance(cmd, P.ExpressCmd):
            return [P.ack(cmd.act_id, True, t_ms)]
        if cmd.deadline_ms <= t_ms:
            return [P.ack(cmd.act_id, False, t_ms, "past_deadline")]
        if self.righting.active or self.proprio.fall_open:
            return [P.ack(cmd.act_id, False, t_ms, "fallen")]
        if self.proprio.lift == "lifted":
            return [P.ack(cmd.act_id, False, t_ms, "lifted")]
        if self.act is not None and not cmd.preempt:
            return [P.ack(cmd.act_id, False, t_ms, "busy")]
        if self.act is not None:
            self.pending_stop = "preempt"
        executor = self._build(cmd, t_ns)
        if executor is None:
            return [P.ack(cmd.act_id, False, t_ms, "unavailable")]
        self._queued = (executor, cmd.deadline_ms)
        if self.act is None:
            self._start_queued()
        return [P.ack(cmd.act_id, True, t_ms)]

    _queued: tuple[P.ActExecutor, int] | None = None

    def _start_queued(self) -> None:
        if self._queued is not None:
            self.act, self.act_deadline_ms = self._queued
            self._queued = None

    def _build(self, cmd: P.Command, t_ns: int) -> P.ActExecutor | None:
        if isinstance(cmd, P.PostureCmd):
            return PostureAct(cmd, t_ns)
        factory = self.cfg.executors.get(cmd.cmd)
        if factory is None:
            return None
        return factory(cmd, t_ns // MS, self)

    # ── the tick ──

    def _during(self) -> str:
        if self.act is None:
            return "still"
        return {
            "duck.walk": "walk",
            "duck.posture": "posture",
            "duck.curio": "curio",
            "duck.dock": "dock",
            "duck.look": "look",
        }.get(self.act.command.cmd if hasattr(self.act, "command") else "", "still")

    def _eye(self, results: Sequence[Mapping[str, Any]], t_ms: int, out: Tick) -> None:
        """Light from each eye result (``results.light = {level, luma, noise_q}``)."""
        for result in results:
            inner = result.get("results") if isinstance(result, Mapping) else None
            light = inner.get("light") if isinstance(inner, Mapping) else None
            if isinstance(result, Mapping) and isinstance(result.get("look_id"), str):
                self.last_look_id = result["look_id"]
                self.pending_looks.pop(result["look_id"], None)
            if isinstance(light, Mapping) and light.get("level") in (
                "bright",
                "dim",
                "dark",
                "black",
            ):
                level = str(light["level"])
                raw = light.get("raw")
                if raw in _RANK and _RANK[str(raw)] < _RANK[level]:
                    level = str(raw)  # never go by a level brighter than the frame shows
                self.light = level
                self.light_ms = t_ms
                event = self.agg.light(
                    t_ms,
                    self.light,
                    int(light.get("luma", 0) or 0),
                    int(light.get("noise_q", 0) or 0),
                )
                if event is not None:
                    out.lines.append(P.percept_line(event))
            elif isinstance(result, Mapping) and result.get("status") == "none":
                self.light = "none"
                self.light_ms = t_ms

    def _walk_view(self) -> P.WalkCmd | None:
        """The walk the running act is making: its own ``WalkCmd``, or — for a curiosity step —
        the ``walk_view`` it exposes (purpose ``curio``, ``toward`` the just-seen thing), so the
        guard (owner A2/A3) and the reflexes (owner A1) judge its creeps as that walk."""
        cmd = getattr(self.act, "command", None) if self.act is not None else None
        if isinstance(cmd, P.WalkCmd):
            return cmd
        view = getattr(self.act, "walk_view", None) if isinstance(cmd, P.CurioCmd) else None
        return view if isinstance(view, P.WalkCmd) else None

    def _light_now(self, t_ms: int) -> str:
        """The level for walking decisions: ``none`` once it is older than LIGHT_FRESH_MS."""
        if self.light_ms is None or t_ms - self.light_ms > LIGHT_FRESH_MS:
            return "none"
        return self.light

    def _curio_running(self) -> bool:
        """Owner A1/A2 belong to curiosity acts: is one running?"""
        return isinstance(getattr(self.act, "command", None), P.CurioCmd)

    def _head_home_ms(self) -> int:
        if self.head_home_since is None or self.state is None:
            return 0
        return int((self.state.t_ns - self.head_home_since) // MS)

    def _guard_ctx(self, prox: ProprioOut) -> GuardContext:
        cmd = getattr(self.act, "command", None) if self.act is not None else None
        walk = self._walk_view()
        head_ms = self._head_home_ms()
        if isinstance(cmd, P.CurioCmd):
            head_down = walk is not None and bool(getattr(self.act, "head_down_ok", False))
        else:
            # a plain duck.walk: only a final approach (near, fresh at the walk's start)
            toward = walk.toward if walk is not None else None
            started = getattr(self.act, "start_ms", None)
            head_down = (
                walk is not None
                and walk.head == "down"
                and toward is not None
                and toward.r_mm <= A2_WALK_MAX_MM
                and isinstance(started, int)
                and 0 <= started - toward.seen_t_ms <= FRESH_TARGET_MS
            )
        offsets = self.state.head_offsets if self.state is not None else (0.0, 0.0, 0.0, 0.0)
        return GuardContext(
            head_home_ms=head_ms,
            moving=self.streamer.last != (0.0, 0.0),
            head_cmd_home=self.head_target is None or not any(self.head_target),
            head_off_mrad=int(round(max(abs(v) for v in offsets) * 1000.0)),
            act=None if cmd is None else cmd.cmd,
            approach=walk is not None and walk.approach,
            head_down_ok=head_down,
            back_arc_ok=isinstance(cmd, P.CurioCmd),
            back_step_ok=self._back_step_ok(),
            lifted=prox.view.lifted,
            unloaded=prox.view.unloaded,
            fallen=prox.view.fallen,
            righting=self.righting.active,
            capturing=bool(self.pending_looks),
        )

    def _back_step_ok(self) -> bool:
        """A ``duck.walk`` back step walkexec granted (the duck was cornered) may reverse until
        it has backed :data:`BACK_STEP_MAX_MM` from where it stood, by odometry (owner
        「2. 可以」: ≤ 0.2 m) — past that the guard refuses every further reverse twist of it."""
        cmd = getattr(self.act, "command", None) if self.act is not None else None
        granted = (
            isinstance(cmd, P.WalkCmd)
            and cmd.vx_mm < 0
            and bool(getattr(self.act, "back_granted", False))
        )
        if not granted or self.spot is None or not isinstance(cmd, P.WalkCmd):
            self._back_from = None
            return False
        if self._back_from is None or self._back_from[0] != cmd.act_id:
            self._back_from = (cmd.act_id, self.spot.x_mm, self.spot.y_mm)
        _, x0, y0 = self._back_from
        return isqrt((self.spot.x_mm - x0) ** 2 + (self.spot.y_mm - y0) ** 2) < BACK_STEP_MAX_MM

    def _send(self, intent: P.Intent | None, guard: GuardContext, out: Tick) -> None:
        if intent is None:
            return
        why = check_intent(intent, guard, self.cfg.limits)
        if why is None:
            out.intents.append(intent)
            return
        out.refused.append((intent, why))
        self.refused_total += 1
        if intent.method == "robot.move" and self.state is not None:
            out.intents.append(self.streamer.zero_now(self.state.t_ns))

    def _end_act(self, result: P.ActOutput, t_ms: int, out: Tick) -> None:
        act = self.act
        self.act = None
        if act is None:
            return
        kind = None
        for body in result.percepts:
            event = self.agg.emit_body(dict(body), t_ms)
            out.lines.append(P.percept_line(event))
            self._note_result(event, t_ms)
            kind = kind or event["kind"]
        extra = getattr(act, "percept", None)
        if kind is None and isinstance(extra, Mapping):
            event = self.agg.emit_body(dict(extra), t_ms)
            out.lines.append(P.percept_line(event))
            self._note_result(event, t_ms)
            kind = event["kind"]
        out.lines.append(P.done(act.act_id, kind or "none", t_ms))
        self._start_queued()

    def _note_result(self, event: Mapping[str, Any], t_ms: int) -> None:
        """What an act's result says about the ways from here (:mod:`.cornered`)."""
        if self.spot is None:
            return
        if event.get("kind") == "walk.done":
            self.cornered.walked(t_ms, self.spot, event)
        elif event.get("kind") == "sense.scan":
            self.cornered.scanned(t_ms, self.spot, event)

    def tick(
        self,
        state: RobotState,
        *,
        depth: P.DepthView | None = None,
        eye: Sequence[Mapping[str, Any]] = (),
    ) -> Tick:
        out = Tick()
        self.state = state
        t_ns, t_ms = state.t_ns, state.t_ns // MS
        for look_id, asked in list(self.pending_looks.items()):
            if t_ns - asked >= CAPTURE_PENDING_NS:
                del self.pending_looks[look_id]  # never answered: given up
        # the head at HOME (measured), for the guard's "settled 0.3 s"
        offsets = state.head_offsets
        commanded_home = self.head_target is None or not any(self.head_target)
        at_home = commanded_home and all(abs(v) <= HEAD_HOME_TOL for v in offsets)
        if at_home:
            self.head_home_since = t_ns if self.head_home_since is None else self.head_home_since
        else:
            self.head_home_since = None
        sent = self.streamer.last
        prox = self.proprio.step(
            state, cmd=(sent[0], sent[1]), touch=bool(depth is not None and depth.touch)
        )
        self.last_prox = prox
        pose = self.tap.step(state, epoch=self.proprio.epoch)
        out.pose, out.eye_pose = pose, PostureTap.for_eye(pose)
        spot = self.spot = Spot.of(pose.odom)
        self.cornered.sync(pose.epoch)
        self.cornered.sense(
            t_ms,
            spot,
            depth,
            tuple(e for e in prox.edges if e in ("blocked", "brushed", "stumbled")),
            prox.contact_side,
        )
        self._eye(eye, t_ms, out)
        # falls and righting
        during = self._during()
        rout = self.righting.step(state, prox, during=during, ctx={"look_id": self.last_look_id})
        for body in rout.events:
            out.lines.append(P.percept_line(self.agg.emit_body(dict(body), t_ms)))
        guard = self._guard_ctx(prox)
        self.last_guard = guard
        if rout.do is not None:
            self._send(P.Intent.do(rout.do), guard, out)
        # lifts (B3/B4) and contacts
        view = prox.view
        walking = during == "walk"
        for edge in prox.edges:
            if edge in ("lifted", "set_down"):
                kind = "sense.lifted" if edge == "lifted" else "sense.set_down"
                event = self.agg.emit(
                    kind,
                    t_ms,
                    conf_ppm=max(prox.lift_conf_ppm, 500_000),
                    posture=view.posture if view.posture in ("stand", "sit") else "unknown",
                    on_nest=None,
                )
                out.lines.append(P.percept_line(event))
                if edge == "lifted" and self.cfg.sit_when_lifted and view.posture == "stand":
                    self._send(P.Intent.do("sit_toggle"), guard, out)
            elif edge in ("blocked", "brushed", "stumbled"):
                touched = self.agg.contact(
                    t_ms, kind=edge, side=prox.contact_side, ev=list(prox.evidence), walking=walking
                )
                if touched is not None:
                    out.lines.append(P.percept_line(touched))
        if depth is not None:
            sensed = self.agg.depth(t_ms, depth, walking=walking)
            if sensed is not None:
                out.lines.append(P.percept_line(sensed))
        # the act; a move that backs away is judged by the body's own reflexes only
        walk_view = self._walk_view()
        relax = self._curio_running()
        v_ahead = verdict(proprio=view, depth=depth, cmd=walk_view, now_ms=t_ms, relax=relax)
        v_back = verdict(
            proprio=view, depth=depth, cmd=walk_view, now_ms=t_ms, relax=relax, reversing=True
        )
        v = v_back if self.asked is not None and self.asked[0] < 0 else v_ahead
        ctx = P.ActContext(
            t_ns=t_ns,
            state=state,
            pose=pose,
            depth=depth,
            proprio=view,
            light=self._light_now(t_ms),
            eye=tuple(eye),
            halted_by=v.reason if v.halt else None,
            head_home_ms=guard.head_home_ms,
            cornered=self.cornered.now(t_ms, spot),
        )
        twist: tuple[int, int] | None = None
        head = self.head_target
        if self.act is not None:
            why = None
            if self.pending_stop is not None:
                why = self.pending_stop
            elif self.righting.active or view.fallen:
                why = "fell"
            elif view.lifted:
                why = "lifted"
            elif t_ms > self.act_deadline_ms:
                why = "deadline"
            if why is not None:
                self.pending_stop = None
                result = self.act.preempt(why, ctx)
                self._end_act(result, t_ms, out)
                head = None
            else:
                result = self.act.step(ctx)
                twist = result.twist
                head = result.head
                if result.do is not None:
                    self._send(P.Intent.do(result.do), guard, out)
                if result.mouth is not None and result.mouth != self.mouth_sent:
                    # the beak (grab, nibble, put_down): robot.mouth {open 0…1}, sent on change
                    self.mouth_sent = result.mouth
                    self._send(P.Intent("robot.mouth", {"open": float(result.mouth)}), guard, out)
                out.looks.extend(result.looks)
                for look in result.looks:
                    self.pending_looks[look.look_id] = t_ns
                if result.done:
                    twist, head = None, None  # a finished act moves nothing more
                    self._end_act(result, t_ms, out)
        else:
            self.pending_stop = None
            head = None
        self.asked = twist
        v = v_back if twist is not None and twist[0] < 0 else v_ahead
        if v.halt or self.righting.active:
            twist = None
        # the head: robot.head offsets, sent on change and every 0.5 s while away from HOME
        if head != self.head_target or (
            head is not None
            and (self.head_sent_ns is None or t_ns - self.head_sent_ns >= HEAD_RESEND_NS)
        ):
            self.head_target = head
            self.head_sent_ns = t_ns
            self._send(P.Intent.head(head or (0.0, 0.0, 0.0, 0.0)), guard, out)
        # the twist, streamed at 50 Hz (mm/s → m/s)
        self.twist = twist
        want = None if twist is None else (twist[0] / 1000.0, twist[1] / 1000.0)
        self._send(self.streamer.step(t_ns, want), guard, out)
        return out

    def health(self, result: Mapping[str, Any], t_ns: int) -> list[dict[str, Any]]:
        """robot.health (1 Hz): battery 5 % steps, thermal tiers."""
        t_ms = t_ns // MS
        lines: list[dict[str, Any]] = []
        battery = result.get("battery")
        if isinstance(battery, Mapping):
            pct = battery.get("percent")
            if isinstance(pct, (int, float)) and math.isfinite(pct):
                trend = "charging" if battery.get("charging") else "draining"
                event = self.agg.battery(t_ms, int(round(pct)), trend)
                if event is not None:
                    lines.append(P.percept_line(event))
        temp = result.get("cpu_temp_c")
        if isinstance(temp, (int, float)) and math.isfinite(temp):
            throttle = result.get("cpu_throttle")
            throttled = isinstance(throttle, Mapping) and bool(throttle.get("throttled"))
            ceiling = throttle.get("max_mhz") if isinstance(throttle, Mapping) else None
            event = self.agg.hot(
                t_ms,
                int(round(temp)),
                int(ceiling) if isinstance(ceiling, (int, float)) else None,
                throttled,
            )
            if event is not None:
                lines.append(P.percept_line(event))
        return lines


__all__ = ["BodyConfig", "BodyLoop", "PostureAct", "Tick"]
