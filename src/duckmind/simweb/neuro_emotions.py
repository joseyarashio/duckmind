"""The neural duck's emotions on the live seam (2026-10-08): plays Pollen's ``microduck_emotions`` through the body link
and lets the brain's emotion layer choose them (:mod:`duckmind.neuro.emotions`).

:class:`EmotionHost` sits between the motor and the body link in :meth:`NeuroDriver._tick_locked`: while an emotion plays it
replaces the motor's request (no walking, the emotion's head, beak and bow, the seat for the sitting ones) and holds the
brain still (:meth:`NeuroBrain.perform`); a reflex, a fall, a lift, the sleep or a trained motion ends it at once and
nothing resumes. When none plays it lets the page's press in (:meth:`press`), or asks the picker (the brain's emotion
layer) whether the duck feels one. It senses only what the duck senses: its own drives and behaviour, the microphone's
voice, a person coming into view, a stall, a scare — never the world's truth.

NOTICE: Pollen Robotics' microduck_emotions (Apache-2.0), see :mod:`duckmind.neuro.emotions`.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import contract as C
from duckmind.neuro import emotions as EO
from duckmind.neuro.bodylink import BodyRequest, LinkOut
from duckmind.skilltrain import motion_result as MR

ANSWER_S = 2.4  # the page's press waits this long for the answer
SETTLE_S = 2.0  # a body that walks has this long to stand still before an emotion starts
ARRIVAL_GAP_S = 10.0  # a person seen again after this long out of view has arrived
SPEECH_WATCH_S = (
    5.0  # with the cloud on: no answer from it this long after a voice = not understood
)
SCARE_KEEP_S = 6.0
SCRIPTED_LABEL = "scripted"  # robotd's policy label while a robot.poseJoints pose holds the duck
# the reflexes that end an emotion at once (orient / veer / stabilise are head turns it can carry on through)
ENDING_REFLEXES = frozenset(
    {"startle", "freeze", "withdraw", "righting", "lifted", "drop_stop", "bump_stop"}
)
ENDING_BEHAVIOURS = frozenset(
    {"flee", "retreat", "startle", "withdraw", "righting", "held", "freeze"}
)
PROGRAMS = EO.PROGRAM_BEHAVIOURS
PLAY_PROGRAMS = frozenset({"play_ball", "dance", "wheee_ride", "hide", "circle_settle"})
SAMPLE_KEEP = 900


@dataclass
class _Press:
    """The page's press waiting for the tick thread's answer."""

    emotion: EO.Emotion
    done: threading.Event = field(default_factory=threading.Event)
    answer: dict[str, Any] | None = None


@dataclass
class _Dead:
    """Play dead lying down (:data:`duckmind.neuro.emotions.DEAD1_AT_S`…): where the scripted pose and the wake-up are."""

    wake_at_s: float  # seconds from the start of the play: robot.init (the legs have been up for the draw's hold)
    stage: int = 0  # 0 sitting, 1 legs straight sent (it lies down from here), 2 legs up sent
    stage_s: dict[int, float] = field(
        default_factory=dict
    )  # the duck's clock when each stage was sent
    forced: str | None = None  # why it wakes before wake_at_s (asleep, docked, battery, recover)
    init_s: float | None = None  # robot.init sent
    enable_s: float | None = None  # robot.enable sent (the walking policy is back)
    up_since_s: float | None = None
    sat_since_s: float | None = (
        None  # robotd's seat has held since (the legs go straight only on a seated duck)
    )
    result: str | None = None  # up | stuck
    up_s: float | None = None  # seconds from the press until it stood again
    tilt_max: float = 0.0  # the trunk's largest tilt while it lay (degrees)


@dataclass
class _Run:
    emotion: EO.Emotion
    source: str  # page | brain
    why: str
    seq: int
    started_s: float  # the duck's clock when it was asked for
    player: EO.Player | None = None  # None while it settles
    press: _Press | None = None
    samples: list[tuple[float, float, float, float, bool, float]] = field(default_factory=list)
    voiced_s: float | None = None
    aborted: str | None = None
    dead: _Dead | None = (
        None  # play dead with a robotd that can lie it down (the sit-only play has none)
    )


class EmotionHost:
    def __init__(self, driver: Any, *, brain_use: bool | None = None) -> None:
        self.driver = driver
        self.brain_use = (
            os.environ.get("DUCKMIND_EMOTIONS", "on").lower() not in ("off", "0", "no")
            if brain_use is None
            else bool(brain_use)
        )
        self.log = EO.EmotionLog()
        self.run: _Run | None = None
        self.last: dict[str, Any] | None = None
        self._press: _Press | None = None
        self._lock = threading.Lock()
        self.picker: EO.Picker | None = None
        self._picker_key: Any = None
        self._prev_beh = ""
        self._prev_bout = -1
        self._prev_kind: str | None = None
        self._person_seen_s = -1e9
        self._person_in_view = False
        self._scares: deque[tuple[float, float]] = deque(maxlen=20)
        self._speech_watch: tuple[float, int] | None = None
        self._blocked_seen = 0
        self.stats = {"started": 0, "interrupted": 0, "fell": 0, "refused": 0}

    @property
    def lying(self) -> bool:
        """The duck lies down on purpose now (a play dead from its first scripted pose until it is up)."""
        run = self.run
        return bool(run is not None and run.dead is not None and run.dead.stage >= 1)

    # ── the page's press ──

    def press(self, emotion_id: str) -> dict[str, Any]:
        name = EO.BY_ID.get(str(emotion_id))
        if name is None:
            raise ValueError(msg("neuro.motion.unknown"))
        driver = self.driver
        if driver.state != "running" or driver._link is None:
            raise ValueError(msg("neuro.motion.not_running"))
        press = _Press(EO.catalog()[name])
        with self._lock:
            if self._press is not None and not self._press.done.is_set():
                raise ValueError(msg("neuro.motion.pressed"))
            self._press = press
        if not press.done.wait(ANSWER_S):
            with self._lock:
                self._press = None
            return {"ok": False, "id": press.emotion.id, "refused": True, "reason": "late",
                    "reason_zh": msg("neuro.motion.late")}  # fmt: skip
        with self._lock:
            self._press = None
        return {"id": press.emotion.id, **dict(press.answer or {})}

    # ── what the duck senses (its own senses and its brain's state only) ──

    def _sense(
        self,
        frame: C.SensorFrame,
        brain: C.BrainTelemetry | None,
        cmd: C.MotorCommand,
        now_s: float,
    ) -> tuple[list[str], bool, bool, float]:
        """cues, blocked, person in view, the biggest scare lately."""
        cues: list[str] = []
        kind = frame.sound.kind
        if kind != self._prev_kind:
            if kind == "voice":
                cues.append("speech")
            elif kind in ("name", "hello"):
                cues.append("greeted")
        self._prev_kind = kind
        persons = [s for s in frame.camera.sightings if s.kind == "person"]
        seen = bool(persons)
        if seen:
            if now_s - self._person_seen_s >= ARRIVAL_GAP_S and not self._person_in_view:
                cues.append("arrival")
            self._person_seen_s = now_s
        self._person_in_view = seen
        blocked = "blocked" in frame.events or "stumbled" in frame.events
        if brain is not None and brain.reflexes:
            newest = brain.reflexes[-1]
            if newest.reflex in ("startle", "freeze", "withdraw") and (
                not self._scares or abs(self._scares[-1][0] - newest.t_ms / 1000.0) > 0.5
            ):
                self._scares.append((newest.t_ms / 1000.0, float(newest.strength)))
        scare = max((s for t, s in self._scares if now_s - t <= SCARE_KEEP_S), default=0.0)
        return cues, blocked, seen, scare

    def _picker_for(self, brain: Any) -> EO.Picker | None:
        genome = getattr(brain, "genome", None)
        key = (getattr(genome, "seed", None), getattr(genome, "gd", None))
        if self.picker is None or key != self._picker_key:
            seed = int(getattr(genome, "seed", 0) or 0)
            self.picker = EO.Picker(EO.Temperament.of(genome), seed)
            self._picker_key = key
        return self.picker

    # ── the tick ──

    def step(
        self,
        *,
        cmd: C.MotorCommand,
        brain_tel: C.BrainTelemetry | None,
        frame: C.SensorFrame,
        state: Any,
        request: BodyRequest,
        t_ms: int,
        last: LinkOut | None,
    ) -> tuple[BodyRequest, bool]:
        """The request the body link gets this tick, and whether an emotion has the body."""
        now_s = t_ms / 1000.0
        cues, blocked, person, scare = self._sense(frame, brain_tel, cmd, now_s)
        cues = self._speech(cues, now_s)
        beh = cmd.behaviour
        new_bout = (beh, cmd.bout) != (self._prev_beh, self._prev_bout)
        play_over = bool(self._prev_beh in PLAY_PROGRAMS and beh not in PLAY_PROGRAMS)
        self._prev_beh, self._prev_bout = beh, cmd.bout
        run = self.run
        if run is not None:
            self._sample(run, state)
            why = self._interrupt(run, cmd, state, last)
            if why is not None:
                self._end(run, now_s, aborted=why)
                run = None
        if run is None:
            self._maybe_start(cmd, brain_tel, frame, state, last, now_s, cues, blocked, person, scare,
                              new_bout, play_over)  # fmt: skip
            run = self.run
        if run is None and str(getattr(state, "policy", "") or "") == SCRIPTED_LABEL:
            run = self._recover(
                now_s
            )  # (robotd still holds a dead pose a former watch began: wake it)
        if run is None:
            return request, False
        # a body that still walks stands still first (the brain is held; the link zeroes the twist)
        if run.player is None:
            if self._still(state, last):
                run.player = EO.Player(run.emotion, now_s)
                self._started(run, now_s)
            elif now_s - run.started_s > SETTLE_S:
                self._end(run, now_s, aborted="walking", started=False)
                return request, False
            else:
                self._hold_brain(t_ms)
                return replace(request, vx=0.0, vyaw=0.0, approach=False, docking=False,
                               back_arc=False, skill=None, skill_manual=False, toward=None), True  # fmt: skip
        fr = run.player.frame(now_s)
        if fr.voice is not None:
            self._voice(run, fr.voice, now_s)
        dead = run.dead
        if dead is not None:
            if self._dead_tick(run, fr.t_s, state, now_s):
                self._end(run, now_s)
                return request, False
        elif fr.done:
            self._end(run, now_s)
            return request, False
        self._hold_brain(t_ms)
        posture: str | None = "sit" if fr.sit else "stand"
        lying = dead is not None and dead.stage >= 1
        if (
            dead is not None and lying
        ):  # on its back: no seat to hold or leave; once the policy is back it stands (the link toggles out of 'sit')
            posture = "stand" if dead.enable_s is not None else None
        return (
            replace(
                request,
                vx=0.0,
                vyaw=0.0,
                head=fr.head,
                mouth=fr.mouth,
                posture=posture,
                approach=False,
                docking=False,
                back_arc=False,
                skill=None,
                skill_manual=False,
                toward=None,
                emotion=True,
                pose=fr.pitch,
                lying=lying,
            ),  # fmt: skip
            True,
        )

    # ── starting ──

    def _quiet(
        self, cmd: C.MotorCommand, state: Any, last: LinkOut | None, playing: bool = False
    ) -> str | None:
        """Why an emotion may not start now (None: it may). The reasons are catalog words."""
        if cmd.asleep:
            return "asleep"
        if cmd.reflex in ENDING_REFLEXES or cmd.behaviour in ENDING_BEHAVIOURS:
            return "reflex"
        if cmd.docking:
            return "docked"
        if last is not None:
            view = last.prox.view
            if view.fallen or last.righting != "idle":
                return "fallen"
            if view.lifted or "airborne" in last.gate:
                return "lifted"
            if view.unloaded and not playing:
                return "unsure"  # (the legs went light: a start waits; one that plays on — a bow shifts the weight)
            if "battery" in last.gate:
                return "battery"
        label = str(getattr(state, "policy", "") or "")
        if MR.is_skill_label(label):
            return "skill"  # (a trained motion runs; the sit, rise and hold labels of a posture change are not)
        return None

    def _maybe_start(
        self,
        cmd,
        brain_tel,
        frame,
        state,
        last,
        now_s,
        cues,
        blocked,
        person,
        scare,
        new_bout,
        play_over,
    ) -> None:  # noqa: ANN001
        press = self._press
        if press is not None and not press.done.is_set():
            why = self._quiet(cmd, state, last)
            label = str(getattr(state, "policy", "") or "")
            if why is None and label == "sit":
                why = "posture"
            if why is not None:
                self.stats["refused"] += 1
                press.answer = {"ok": False, "refused": True, "reason": why,
                                "reason_zh": msg(f"neuro.emotion.reason.{why}")}  # fmt: skip
                press.done.set()
                return
            self._begin(press.emotion, "page", "page", now_s, press)
            return
        if not self.brain_use or self.driver.state != "running":
            return
        brain = getattr(self.driver.lanes, "brain", None)
        picker = self._picker_for(brain)
        if picker is None:
            return
        quiet = self._quiet(cmd, state, last) is None
        label = str(getattr(state, "policy", "") or "")
        drives = dict(brain_tel.drives) if brain_tel is not None else {}
        loco = dict(brain_tel.locomotion) if brain_tel is not None else {}
        tof_ahead = frame.tof.ahead_mm if frame.tof.present else None
        stuck = float(loco.get("stuck", 0.0) or 0.0)
        ctx = EO.Context(
            t_s=now_s,
            behaviour=cmd.behaviour,
            new_bout=new_bout,
            reflex=cmd.reflex,
            asleep=bool(cmd.asleep),
            wants_walk=abs(cmd.vx) >= EO.STILL_V,
            moving=math.hypot(float(state.applied[0]), float(state.applied[2])) >= EO.SETTLE_TWIST,
            standing=label in ("stand", "walk"),
            quiet=quiet,
            drives=drives,
            stuck=stuck,
            blocked=blocked,
            strong_scare=scare,
            cornered=stuck >= 0.66 or (tof_ahead is not None and tof_ahead < 250),
            person_in_view=person,
            bout_ms=int(getattr(brain_tel, "bout_ms", 0) or 0),
            play_over=play_over,
            greeted=bool(new_bout and cmd.behaviour == "greet"),
            cues=tuple(cues),
        )
        choice = picker.step(ctx)
        if choice is not None:
            picker.played(choice.name, now_s)
            self._begin(EO.catalog()[choice.name], "brain", choice.why, now_s, None)

    def _speech(self, cues: list[str], now_s: float) -> list[str]:
        """A voice heard. With the cloud off the duck cannot understand it: the cue stands. With the cloud on it only
        counts when the cloud did nothing with the words in :data:`SPEECH_WATCH_S` (no call came back: not understood)."""
        live = getattr(self.driver, "live_voice", None)
        on = bool(live is not None and live.cloud_on())
        out = [c for c in cues if c != "speech" or not on]
        if on and "speech" in cues and self._speech_watch is None:
            self._speech_watch = (now_s + SPEECH_WATCH_S, int(live.calls_seen()))
        if self._speech_watch is not None and now_s >= self._speech_watch[0]:
            baseline = self._speech_watch[1]
            self._speech_watch = None
            if live is not None and int(live.calls_seen()) == baseline:
                out.append("speech")
        return out

    def _begin(
        self, emotion: EO.Emotion, source: str, why: str, now_s: float, press: _Press | None
    ) -> None:
        self.run = _Run(emotion, source, why, 0, now_s, None, press)

    def _still(self, state: Any, last: LinkOut | None) -> bool:
        twist = math.hypot(float(state.applied[0]), float(state.applied[2]))
        sent = (0.0, 0.0) if last is None else last.twist
        return twist < EO.SETTLE_TWIST and sent == (0.0, 0.0)

    def _started(self, run: _Run, now_s: float) -> None:
        item = self.log.add({
            "state": "started", "id": run.emotion.id, "name": run.emotion.name, "source": run.source,
            "why": run.why, "t_s": round(now_s, 2),
        })  # fmt: skip
        run.seq = int(item["seq"])
        self.stats["started"] += 1
        if run.emotion.name == "play_dead" and self._can_lie() and run.dead is None:
            hold = EO.wake_hold_s((self.stats["started"], round(now_s, 1)))
            run.dead = _Dead(wake_at_s=EO.DEAD2_AT_S + EO.DEAD2_RAMP_S + hold)
        self._event("emotion.start", run, now_s, msg(
            "neuro.emotion.event.start",
            name=msg(f"neuro.emotion.name.{run.emotion.name}"),
            why=msg(f"neuro.emotion.why.{run.why}"),
        ))  # fmt: skip
        if run.press is not None:
            run.press.answer = {"ok": True, "started": True, "seq": run.seq,
                                "duration_s": run.emotion.duration_s}  # fmt: skip
            run.press.done.set()
            run.press = None

    # ── play dead lies down (docs/design/neural-duck/PLAY-DEAD-2026-10-11.md) ──

    def _body(self) -> Any:
        return getattr(self.driver, "_body", None)

    def _can_lie(self) -> bool:
        """This simulator's robotd answers robot.poseJoints (the simulator-only build); the official one does not."""
        return bool(getattr(self._body(), "pose_joints_ok", False))

    def _recover(self, now_s: float) -> _Run | None:
        """robotd holds a scripted pose nobody plays now (a former watch began it): a run that only wakes it."""
        if self.driver.state != "running" or not self._can_lie():
            return None
        emotion = EO.catalog()["play_dead"]
        run = _Run(emotion, "brain", "recover", 0, now_s)
        run.player = EO.Player(emotion, now_s - 60.0)
        run.player.voiced = True
        run.dead = _Dead(wake_at_s=0.0, stage=2, forced="recover")
        self.run = run
        self._started(run, now_s)
        return run

    def _dead_tick(self, run: _Run, t: float, state: Any, now_s: float) -> bool:
        """The scripted pose, the wake and the get-up of a play dead that lies down. True: it is over (it stands, or did not
        get up by itself in time: the ordinary fall and its righting take it from there)."""
        dead = run.dead
        body = self._body()
        if dead is None or body is None:
            return True
        if dead.stage == 0:
            if str(getattr(state, "policy", "") or "") == "sit":
                dead.sat_since_s = now_s if dead.sat_since_s is None else dead.sat_since_s
            else:
                dead.sat_since_s = None
        seated = dead.sat_since_s is not None and now_s - dead.sat_since_s >= EO.DEAD_SEAT_HOLD_S
        if dead.stage == 0 and dead.forced is None and t >= EO.DEAD_SEAT_GIVE_UP_S and not seated:
            run.dead = None  # (it never sat down: a straight leg on a duck that is not seated topples it forward)
            return False
        if dead.stage == 0 and dead.forced is None and t >= EO.DEAD1_AT_S and seated:
            if body.pose_joints(
                EO.dead_pose(1), EO.DEAD_OFF, gain=EO.DEAD_GAIN, ramp_s=EO.DEAD1_RAMP_S
            ):
                dead.stage, dead.stage_s[1] = 1, now_s
            else:  # this robotd cannot (an official build, or it refused): the sitting play dead goes on
                run.dead = None
                return False
        if dead.stage == 1 and dead.forced is None and t >= EO.DEAD2_AT_S:
            if body.pose_joints(
                EO.dead_pose(2), EO.DEAD_OFF, gain=EO.DEAD_GAIN, ramp_s=EO.DEAD2_RAMP_S
            ):
                dead.stage, dead.stage_s[2] = 2, now_s
            else:
                dead.forced = "refused"
        if dead.stage == 0 and dead.forced is not None:
            run.dead = None  # (woken before it lay down: the sitting one ends through the ordinary interrupt)
            return False
        tilt = getattr(state, "tilt_rad", None)
        if tilt is not None:
            dead.tilt_max = max(dead.tilt_max, math.degrees(float(tilt)))
        if dead.init_s is None and (t >= dead.wake_at_s or dead.forced is not None):
            if dead.stage_s.get(2) is None and dead.forced is None:
                return False
            body.operator_init()
            dead.init_s = now_s
        if (
            dead.init_s is not None
            and dead.enable_s is None
            and now_s - dead.init_s >= EO.WAKE_HOMING_S
        ):
            body.operator_enable(True)
            dead.enable_s = now_s
        if dead.enable_s is None:
            return False
        link = getattr(self.driver, "_link", None)
        down = bool(link is not None and getattr(link.proprio, "fall_open", False))
        up = (
            str(getattr(state, "policy", "") or "") in ("walk", "stand")
            and not bool(getattr(state, "fallen", False))
            and (tilt is None or math.degrees(float(tilt)) < EO.WAKE_UP_TILT_DEG)
            and not down
        )
        if not up:
            dead.up_since_s = None
        elif dead.up_since_s is None:
            dead.up_since_s = now_s
        if dead.up_since_s is not None and now_s - dead.up_since_s >= EO.WAKE_UP_HOLD_S:
            dead.result, dead.up_s = "up", round(t, 2)
            return True
        if now_s - dead.enable_s >= EO.WAKE_UP_TIMEOUT_S:
            dead.result, dead.up_s = "stuck", None
            return True
        return False

    # ── interrupting and ending ──

    def _interrupt(
        self, run: _Run, cmd: C.MotorCommand, state: Any, last: LinkOut | None
    ) -> str | None:
        dead = run.dead
        if dead is not None and dead.stage >= 1:
            # lying on its back on purpose: its own fall, righting, lifted and unloaded senses and the reflexes are not reasons
            # to stop (the scripted pose has to be taken back through robot.init); only what really ends a play wakes it early
            held: str | None = None
            if cmd.asleep:
                held = "asleep"
            elif cmd.docking:
                held = "docked"
            elif last is not None and "battery" in last.gate:
                held = "battery"
            if held is not None and dead.forced is None:
                dead.forced = held
            return None
        why = self._quiet(cmd, state, last, playing=True)
        label = str(getattr(state, "policy", "") or "")
        if why is None and run.player is not None and not run.emotion.sits and label == "sit":
            why = "posture"
        return why

    def _end(
        self, run: _Run, now_s: float, aborted: str | None = None, started: bool = True
    ) -> None:
        self.run = None
        press = run.press
        if press is not None and not press.done.is_set():
            press.answer = {"ok": False, "refused": True, "reason": aborted or "late",
                            "reason_zh": msg(f"neuro.emotion.reason.{aborted or 'late'}")}  # fmt: skip
            press.done.set()
        if not started or run.player is None:
            return
        if aborted:
            run.player.abort(aborted)
        fell = any(s[4] for s in run.samples)
        tilt = max((s[3] for s in run.samples), default=0.0)
        drift = 0.0
        if len(run.samples) >= 2:
            x0, y0 = run.samples[0][1], run.samples[0][2]
            drift = max(math.hypot(s[1] - x0, s[2] - y0) for s in run.samples)
        dead = run.dead
        stuck = dead is not None and dead.result == "stuck"
        result = msg("neuro.emotion.result.fell") if fell else (
            msg("neuro.emotion.result.interrupted", why=msg(f"neuro.emotion.reason.{aborted}"))
            if aborted else (msg("neuro.emotion.result.stuck") if stuck else msg("neuro.emotion.result.done"))
        )  # fmt: skip
        item: dict[str, Any] = {
            "state": "fell" if fell else ("interrupted" if aborted else "done"),
            "id": run.emotion.id, "name": run.emotion.name, "source": run.source, "why": run.why,
            "seq": run.seq, "t_s": round(now_s, 2), "played_s": round(run.player.frame(now_s).t_s, 2),
            "interrupted_by": aborted, "fell": fell, "tilt_deg": round(tilt, 1), "drift_m": round(drift, 3),
            "voiced_at_s": run.voiced_s,
        }  # fmt: skip
        if dead is not None:
            item["lay"] = {
                "stage": dead.stage, "result": dead.result, "up_s": dead.up_s, "forced": dead.forced,
                "tilt_max": round(dead.tilt_max, 1), "wake_at_s": round(dead.wake_at_s, 2),
            }  # fmt: skip
        self.log.add(item)
        self.last = {
            **item, "zh": result, "name_zh": msg(f"neuro.emotion.name.{run.emotion.name}"),
            "why_zh": msg(f"neuro.emotion.why.{run.why}"),
        }  # fmt: skip
        if fell:
            self.stats["fell"] += 1
        elif aborted:
            self.stats["interrupted"] += 1
        self._event("emotion.end", run, now_s, msg(
            "neuro.emotion.event.end",
            name=msg(f"neuro.emotion.name.{run.emotion.name}"),
            result=result,
        ))  # fmt: skip
        if self.picker is not None:
            self.picker.ended(now_s)

    def _sample(self, run: _Run, state: Any) -> None:
        recent = getattr(self.driver, "_recent", None)
        tick = recent[-1] if recent else None
        if tick is None or run.player is None or tick.trunk is None:
            return
        fell = bool(tick.fallen)
        run.samples.append((tick.t_ms / 1000.0, float(tick.trunk[0]), float(tick.trunk[1]),
                            float(tick.tilt_deg or 0.0), fell, 0.0))  # fmt: skip
        del run.samples[:-SAMPLE_KEEP]

    def _voice(self, run: _Run, tag: str, now_s: float) -> None:
        run.voiced_s = round(now_s - run.player.t0_s, 3) if run.player else None
        body = getattr(self.driver, "_body", None)
        voice = getattr(body, "voice", None)
        if callable(voice):
            voice(tag, 1.0)

    def _hold_brain(self, t_ms: int) -> None:
        brain = getattr(self.driver.lanes, "brain", None)
        perform = getattr(brain, "perform", None)
        if callable(perform):
            perform(t_ms + 600)

    def _event(self, kind: str, run: _Run, now_s: float, words: Any) -> None:
        self.driver.telemetry_store.event({
            "t_ms": int(now_s * 1000), "wall_ms": int(time.time() * 1000), "kind": kind,
            "emotion": run.emotion.name, "source": run.source, "zh": words,
        })  # fmt: skip

    # ── what the page reads ──

    def view(self) -> dict[str, Any]:
        run = self.run
        return {
            "running": None if run is None or run.player is None else {
                "id": run.emotion.id, "seq": run.seq, "source": run.source},
            "last": self.last,
            "brain_use": self.brain_use,
            "counts": self.log.counts(),
            "recent": [dict(it) for it in list(self.log.items)[-40:]],
            "stats": dict(self.stats),
        }  # fmt: skip


__all__ = ["EmotionHost", "BodyRequest"]
