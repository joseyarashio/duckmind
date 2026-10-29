"""Formal play adapter and guarded one-shot bridge for the live Neuro9 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.reflex import GuardContext, check_intent
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import HOME, BodyRequest, LinkOut
from duckmind.neuro.skill_play import PLAY_REFLEX_COEXIST

ONE_SHOT_SKILLS = frozenset({"ground_pick", "kick_left", "kick_right"})
STILL_MS = 300
START_TIMEOUT_MS = 1200
SKILL_BEHAVIOURS = {
    "follow": "follow_mom",
    "hide": "hide",
    "circle": "circle_settle",
    "theremin": "theremin",
    "ride": "wheee_ride",
    "dance": "dance",
    "ball": "play_ball",
    "carry": "carry_toy",
    "sun": "sunbathe",
}


def skill_status(
    lanes: Any,
    behaviour: str | None,
    updated_ms: int | None,
    *,
    reflex: str | None = None,
    actions: list[dict[str, Any]] | None = None,
    world_errors: tuple[str, ...] | list[str] | None = None,
    tof_ready: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Serialize the nine program states without inferring success from a sent command."""
    brain = getattr(lanes, "brain", None)
    names = getattr(lanes, "names", {}) or {}
    base_reason = next(
        (f"missing_{name}" for name in ("brain", "senses", "motor") if names.get(name) != "real"),
        None,
    )
    play = getattr(brain, "play", None)
    play_out = getattr(brain, "play_out", None)
    sound = getattr(brain, "sound_out", None)
    social = brain if hasattr(brain, "social_view") else None
    sound_lane = getattr(lanes, "sound", None)
    engine = getattr(sound_lane, "engine", None)
    engine_view = engine.view() if engine is not None else {}
    voice_ready = bool(engine_view.get("helper"))
    jaw = getattr(getattr(lanes, "senses", None), "capabilities", {}).get("jaw_present")
    entries: dict[str, tuple[str | None, str | None, str | None]] = {}

    if social is not None:
        entries["follow"] = ("following" if behaviour == "follow_mom" else "waiting", None, None)
        hide_phase = str(getattr(getattr(social, "hide_run", None), "phase", "unknown"))
        entries["hide"] = (
            hide_phase,
            "calm" if hide_phase == "out" else "gave_up" if hide_phase == "failed" else None,
            "hide_failed" if hide_phase == "failed" else None,
        )
        circle_phase = str(getattr(getattr(social, "circle", None), "phase", "unknown"))
        entries["circle"] = (
            circle_phase,
            "complete"
            if circle_phase == "done"
            else "partial"
            if circle_phase == "stopped"
            else None,
            "circle_stopped" if circle_phase == "stopped" else None,
        )
    else:
        entries.update({key: (None, None, None) for key in ("follow", "hide", "circle")})

    if sound is not None:
        th = sound.view().get("theremin", {})
        ride_view = sound.view().get("ride", {})
        dance = sound.view().get("dance", {})
        entries["theremin"] = ("playing" if th.get("up") else "listening", None, None)
        ride_phase = (
            "riding"
            if ride_view.get("riding")
            else "lifted"
            if ride_view.get("lifted")
            else "settling"
            if ride_view.get("after", 0) > 0
            else "idle"
        )
        outcome = ride_view.get("outcome")
        entries["ride"] = (ride_phase, outcome if outcome in ("fear", "joy") else None, None)
        entries["dance"] = ("dancing" if dance.get("on") else "listening", None, None)
    else:
        entries.update({key: (None, None, None) for key in ("theremin", "ride", "dance")})

    if play is not None and play_out is not None:
        ball = getattr(play, "ball", None)
        carry = getattr(play, "carry", None)
        sun = getattr(play, "bask_ctl", None)
        ball_event = getattr(ball, "last_event", None)
        entries["ball"] = (
            str(getattr(ball, "phase", "idle")),
            ball_event if ball_event in ("rolled", "lost", "missed", "unseen") else None,
            None,
        )
        carry_event = getattr(carry, "last_event", None)
        entries["carry"] = (
            str(getattr(carry, "phase", "idle")),
            carry_event if carry_event == "dropped_home" else None,
            None,
        )
        entries["sun"] = (str(getattr(sun, "phase", "idle")), None, None)
    else:
        entries.update({key: (None, None, None) for key in ("ball", "carry", "sun")})

    result: dict[str, dict[str, Any]] = {}
    last_actions: dict[str, dict[str, Any]] = {}
    current_skill = str(getattr(play_out, "skill", "") or "")
    current_seq = int(getattr(play_out, "skill_seq", 0) or 0)
    for action in actions or ():
        key = next(
            (name for name, act in SKILL_BEHAVIOURS.items() if action.get("behaviour") == act), None
        )
        matching_request = (
            action.get("skill") == current_skill and int(action.get("seq") or 0) == current_seq
        )
        ongoing = action.get("status") in (
            "waiting_policy",
            "waiting_safety",
            "dispatched",
            "policy_started",
            "interrupted_waiting_for_exit",
        )
        if key in ("ball", "carry") and (matching_request or ongoing):
            last_actions[key] = action
    for key, expected_behaviour in SKILL_BEHAVIOURS.items():
        phase, outcome, reason = entries.get(key, (None, None, None))
        if base_reason:
            available, reason, status = False, base_reason, "missing"
        elif key in ("follow", "hide", "circle", "ball", "carry", "sun") and world_errors:
            available, reason, status = False, "world_sensor_error", "missing"
        elif key == "carry" and jaw is not True:
            available, reason, status = (
                False,
                "no_jaw" if jaw is False else "jaw_unknown",
                "missing",
            )
        elif key == "theremin" and tof_ready is not True:
            available, reason, status = (
                False,
                "no_sensor" if tof_ready is False else "sensor_not_ready",
                "missing",
            )
        elif key in ("theremin", "ride", "dance") and not voice_ready:
            available, reason, status = False, "no_voice_helper", "missing"
        elif phase is None:
            available, reason, status = False, "skill_state_unknown", "unknown"
        else:
            available, reason = True, None
            active = behaviour == expected_behaviour or (key == "ride" and behaviour == "held")
            program_active = {
                "follow": active,
                "hide": active and phase in ("go", "in"),
                "circle": active and phase in ("turning", "seeking"),
                "theremin": active and phase == "playing",
                "ride": active and phase in ("lifted", "riding"),
                "dance": active and phase == "dancing",
                "ball": active and phase not in ("idle", "done", "settled"),
                "carry": active and phase not in ("idle", "done", "settled"),
                "sun": active and phase not in ("idle", "done", "settled"),
            }.get(key, False)
            if active and reflex is not None:
                status = "active" if key == "ride" and reflex == "lifted" else "interrupted"
            elif program_active:
                status = "active"
            elif key == "hide" and phase in ("out", "failed"):
                status = "done"
            elif key == "circle" and phase in ("done", "stopped"):
                status = "done"
            elif key == "ride" and phase == "settling":
                status = "done"
            elif key == "carry" and outcome == "dropped_home":
                status = "done"
            else:
                status = "waiting"
            action = last_actions.get(key)
            if action is not None:
                action_status = action.get("status")
                if action_status in ("waiting_policy", "waiting_safety", "dispatched"):
                    phase, status = str(action_status), "waiting"
                elif action_status == "policy_started":
                    phase, status = "policy_started", "active"
                elif action_status == "interrupted_waiting_for_exit":
                    phase, status = "interrupted_waiting_for_exit", "interrupted"
                elif action_status in ("refused", "readiness_timeout", "policy_start_timeout"):
                    phase, status = str(action_status), "refused"
                    reason = str(action.get("reason") or action_status)
                elif action_status == "interrupted_policy_exited":
                    phase, status = "interrupted_policy_exited", "interrupted"
                elif action_status == "selection_ended":
                    phase, status = "selection_ended", "interrupted"
                elif action_status == "policy_exited":
                    phase = str(action_status)
                    status = "unknown" if outcome is None else status
        result[key] = {
            "available": available,
            "reason": reason,
            "phase": phase,
            "status": status,
            "result": outcome,
            "updated_ms": updated_ms,
        }
    return result


class IntegratedMotor:
    """Consume PlayOut on the same request path as the integrated brain's MotorCommand."""

    def __init__(self, base: Any, brain: Any) -> None:
        self.base = base
        self.brain = brain
        self.last: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(
        self, t_ms: int, cmd: C.MotorCommand, frame: C.SensorFrame, state: RobotState
    ) -> Any:
        request, vocals = self.base(t_ms, cmd, frame, state)
        play = getattr(self.brain, "play_out", None)
        detail: dict[str, Any] = {"fine": False, "skill": None, "skill_seq": 0}
        reflex_coexists = cmd.reflex is None or cmd.reflex in PLAY_REFLEX_COEXIST
        if play is not None and reflex_coexists and not frame.body.fallen and not cmd.asleep:
            if play.drive and not play.far:
                request = replace(
                    request,
                    vx=float(cmd.vx),
                    vyaw=float(cmd.vyaw),
                    approach=bool(cmd.approach),
                    back_arc=bool(cmd.back_arc),
                )
            if cmd.reflex is None and play.mouth is not None:
                request = replace(request, mouth=min(1.0, max(0.0, float(play.mouth))))
            if cmd.reflex is None and play.posture in ("sit", "stand"):
                request = replace(request, posture=play.posture)
            skill = str(play.skill or "")
            seq = int(play.skill_seq or 0)
            body = frame.body
            safe_policy = str(getattr(state, "policy", "")) in ("sit", "stand", "walk")
            body_safe = (
                safe_policy
                and not bool(getattr(state, "fallen", False))
                and not bool(getattr(body, "fallen", False))
                and not bool(getattr(body, "limp", False))
                and (getattr(body, "leg_load", None) is None or float(body.leg_load) >= 0.45)
            )
            if (
                skill in ONE_SHOT_SKILLS
                and seq > 0
                and cmd.reflex is None
                and not bool(getattr(cmd, "asleep", False))
                and body_safe
                and OneShotBridge._active(str(getattr(state, "policy", ""))) is None
            ):
                # robotd's do policies require the measured head at HOME and 300 ms still.
                # Request HOME through the ordinary body link while the one-shot stays pending;
                # OneShotBridge still checks measured state, battery, body gates and policy.
                request = replace(
                    request,
                    vx=0.0,
                    vyaw=0.0,
                    head=None,
                    approach=False,
                    docking=False,
                    back_arc=False,
                )
                if skill == "ground_pick" and getattr(body, "active_skill", None) != "ground_pick":
                    # The carry mouth program starts on robotd's observed policy, not on skill
                    # selection while the head is still returning HOME.
                    request = replace(request, mouth=0.0)
            detail = {
                "fine": bool(play.fine),
                "skill": play.skill if cmd.reflex is None else None,
                "skill_seq": int(play.skill_seq) if cmd.reflex is None else 0,
                "phase": str(play.phase),
                "event": play.event,
            }
        self.last = detail
        return request, vocals


@dataclass
class ActionRecord:
    behaviour: str
    skill: str
    seq: int
    requested_ms: int
    status: str = "requested"
    dispatched_ms: int | None = None
    started_ms: int | None = None
    exited_ms: int | None = None
    reason: str | None = None
    reason_detail: str | None = None
    outcome: str | None = None
    interrupted_ms: int | None = None
    policy_exit_observed: bool = False

    def view(self) -> dict[str, Any]:
        return {
            "behaviour": self.behaviour,
            "skill": self.skill,
            "seq": self.seq,
            "requested_ms": self.requested_ms,
            "dispatched_ms": self.dispatched_ms,
            "started_ms": self.started_ms,
            "exited_ms": self.exited_ms,
            "status": self.status,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "outcome": self.outcome,
            "interrupted_ms": self.interrupted_ms,
            "policy_exit_observed": self.policy_exit_observed,
        }


@dataclass
class OneShotBridge:
    """Guard actions from measured body state and observe uncancelable policy lifetime."""

    still_since_ns: int | None = None
    home_since_ns: int | None = None
    pending: ActionRecord | None = None
    active: ActionRecord | None = None
    seen_keys: set[tuple[str, str, int]] = field(default_factory=set)
    records: list[ActionRecord] = field(default_factory=list)

    def step(
        self,
        t_ms: int,
        state: RobotState,
        request: BodyRequest,
        cmd: C.MotorCommand,
        brain: Any,
        bodylink: Any,
        out: LinkOut,
        *,
        battery_pct: int | None,
    ) -> LinkOut:
        policy = str(getattr(state, "policy", "") or "")
        active_policy = self._active(policy)
        now_ns = int(state.t_ns)
        applied = getattr(state, "applied", None) or (0.0, 0.0, 0.0)
        physically_still = (
            abs(float(applied[0])) < 0.05
            and abs(float(applied[2])) < 0.15
            and out.twist == (0.0, 0.0)
        )
        self.still_since_ns = (
            (now_ns if self.still_since_ns is None else self.still_since_ns)
            if physically_still
            else None
        )
        cfg = bodylink.cfg
        head = getattr(state, "head_offsets", ())
        tolerances = tuple(getattr(cfg, "home_tol", (0.25, 0.10, 0.10, 0.10)))
        at_home = len(head) == 4 and all(
            abs(float(value) - float(home)) <= tol
            for value, home, tol in zip(head, HOME, tolerances, strict=True)
        )
        self.home_since_ns = (
            now_ns if at_home and self.home_since_ns is None else self.home_since_ns
        )
        if not at_home:
            self.home_since_ns = None

        if self.active is not None:
            if active_policy == self.active.skill:
                if self.active.started_ms is None:
                    self.active.status = "policy_started"
                    self.active.started_ms = t_ms
                if cmd.reflex is not None and self.active.interrupted_ms is None:
                    self.active.status = "interrupted_waiting_for_exit"
                    self.active.reason = "active_robotd_skill_not_cancelable_by_stop"
                    self.active.interrupted_ms = t_ms
            elif self.active.started_ms is not None and not active_policy:
                self.active.policy_exit_observed = True
                self.active.status = (
                    "interrupted_policy_exited"
                    if self.active.interrupted_ms is not None
                    else "policy_exited"
                )
                self.active.exited_ms = t_ms
                self.active = None
            elif (
                self.active.dispatched_ms is not None
                and self.active.started_ms is None
                and t_ms - self.active.dispatched_ms > START_TIMEOUT_MS
            ):
                self.active.status = "policy_start_timeout"
                self.active.exited_ms = t_ms
                self.active.reason = "policy_start_timeout"
                self.active = None

        play = getattr(brain, "play_out", None)
        skill = str(getattr(play, "skill", "") or "")
        seq = int(getattr(play, "skill_seq", 0) or 0)
        behaviour = str(getattr(cmd, "behaviour", ""))
        key = (behaviour, skill, seq)
        if not skill or skill not in ONE_SHOT_SKILLS or seq <= 0:
            if self.pending is not None:
                self.pending.status = "selection_ended"
                self.pending.reason = "selection_changed"
                self.seen_keys.add((self.pending.behaviour, self.pending.skill, self.pending.seq))
                self.pending = None
            return out
        if key in self.seen_keys:
            return out
        if (
            self.pending is not None
            and (self.pending.behaviour, self.pending.skill, self.pending.seq) == key
        ):
            record = self.pending
        else:
            if self.pending is not None:
                self.pending.status = "selection_ended"
                self.pending.reason = "selection_changed"
                self.seen_keys.add((self.pending.behaviour, self.pending.skill, self.pending.seq))
            record = ActionRecord(behaviour=behaviour, skill=skill, seq=seq, requested_ms=t_ms)
            self.pending = record
            self.records.append(record)

        view = out.prox.view
        reason: str | None = None
        waiting = False
        if self.active is not None or active_policy:
            reason, waiting = "policy_busy", True
        elif policy not in ("stand", "walk"):
            reason, waiting = "policy_unsettled", True
        elif request.posture not in (None, "stand"):
            reason, waiting = "policy_unsettled", True
        elif cmd.reflex is not None:
            reason, waiting = "reflex_active", True
        elif battery_pct is None:
            reason = "battery_unknown"
        elif battery_pct < cfg.battery_min_pct:
            reason = "battery"
        elif view.unloaded:
            reason = "unloaded"
        elif view.lifted:
            reason = "lifted"
        elif view.fallen or bodylink.righting.active:
            reason = "fallen_or_righting"
        elif any(
            code in out.gate for code in ("battery", "fallen", "lifted", "unloaded", "airborne")
        ):
            reason = "body_gate"
        elif (
            not physically_still
            or self.still_since_ns is None
            or (now_ns - self.still_since_ns) // 1_000_000 < STILL_MS
        ):
            reason, waiting = "body_not_still", True
        elif self.home_since_ns is None or (now_ns - self.home_since_ns) // 1_000_000 < STILL_MS:
            reason, waiting = "head_not_home", True
        else:
            reason = check_intent(
                P.Intent.do(skill),
                GuardContext(
                    head_home_ms=STILL_MS,
                    act="duck.curio",
                    lifted=bool(view.lifted),
                    unloaded=bool(view.unloaded),
                    fallen=bool(view.fallen),
                    righting=bool(bodylink.righting.active),
                ),
            )
            if reason is not None:
                record.reason_detail = reason
                reason = self._reason_code(reason)
        if reason is not None:
            record.reason = reason
            if waiting and t_ms - record.requested_ms <= START_TIMEOUT_MS:
                record.status = "waiting_policy" if reason == "policy_busy" else "waiting_safety"
                return out
            record.status = "readiness_timeout" if waiting else "refused"
            self.seen_keys.add(key)
            self.pending = None
            return out
        record.status = "dispatched"
        record.dispatched_ms = t_ms
        record.reason = None
        record.reason_detail = None
        self.seen_keys.add(key)
        self.pending = None
        self.active = record
        return replace(out, intents=tuple(out.intents) + (P.Intent.do(skill),))

    @staticmethod
    def _active(policy: str) -> str | None:
        token = policy.lower().replace("-", "_")
        if token.endswith(":unwind"):
            token = token[: -len(":unwind")]
        return next(
            (
                skill
                for skill in ONE_SHOT_SKILLS
                if token == skill or token.endswith(f".{skill}") or token.endswith(f"_{skill}")
            ),
            None,
        )

    @staticmethod
    def _reason_code(detail: str) -> str:
        if "not allowed" in detail or "never sent" in detail:
            return "forbidden_intent"
        if "only inside" in detail:
            return "wrong_act"
        if "lifted/fallen/righting" in detail or "no walking while" in detail:
            return "body_unsafe"
        if "HOME" in detail:
            return "head_not_home"
        return "guard_refused"

    def view(self) -> list[dict[str, Any]]:
        return [record.view() for record in self.records[-64:]]


__all__ = ["ActionRecord", "IntegratedMotor", "OneShotBridge", "skill_status"]
