"""Embodied skill lifecycle tracking above robotd."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from duckmind.body import BodyAck, BodyIntent, BodyPort, RobotObservation


class SkillPhase(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    STARTED = "started"
    COMPLETED = "completed"
    REFUSED = "refused"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


TERMINAL_PHASES = {
    SkillPhase.COMPLETED,
    SkillPhase.REFUSED,
    SkillPhase.TIMED_OUT,
    SkillPhase.INTERRUPTED,
}

# Command-space guard for the first, stationary head-only game. These fields
# are robotd's requested/applied velocity intent, not an odometry estimate.
FRONT_GAZE_MAX_COMMAND_MOTION = 0.01


def front_gaze_observation_issue(observation: RobotObservation) -> str | None:
    """Reject body states that cannot support a stationary v5 head game."""

    if observation.safety.fallen or observation.safety.limp:
        return "robot_safety_state"
    if observation.policy != "walk":
        return f"unexpected_policy:{observation.policy}"
    if any(
        not math.isfinite(value) or abs(value) > FRONT_GAZE_MAX_COMMAND_MOTION
        for value in (*observation.movement_requested, *observation.movement_applied)
    ):
        return "motion_command_active"
    return None


@dataclass(frozen=True, slots=True)
class DispatchAck:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SkillStatus:
    execution_id: str
    phase: SkillPhase
    at: float
    reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES


class SkillTracker:
    """Correlates an uncorrelated robotd ACK with observed policy transitions.

    The official API does not publish a request id on ``robot.state``. This
    tracker therefore treats ACK, observed start, observed exit, and outcome as
    separate evidence. It never promotes ACK directly to completion.
    """

    def __init__(
        self,
        *,
        execution_id: str,
        expected_policy: str,
        created_at: float,
        start_timeout_s: float,
        completion_timeout_s: float,
    ) -> None:
        if start_timeout_s <= 0 or completion_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        self.execution_id = execution_id
        self.expected_policy = expected_policy
        self.created_at = created_at
        self.start_timeout_s = start_timeout_s
        self.completion_timeout_s = completion_timeout_s
        self._accepted_at: float | None = None
        self._started_at: float | None = None
        self._status = SkillStatus(execution_id, SkillPhase.PENDING, created_at)

    @property
    def status(self) -> SkillStatus:
        return self._status

    def on_dispatch(self, ack: DispatchAck, *, at: float) -> SkillStatus:
        if self._status.terminal:
            return self._status
        if not ack.accepted:
            self._status = SkillStatus(
                self.execution_id,
                SkillPhase.REFUSED,
                at,
                ack.reason or "robotd_refused",
            )
            return self._status
        self._accepted_at = at
        self._status = SkillStatus(self.execution_id, SkillPhase.ACCEPTED, at, ack.reason)
        return self._status

    def on_observation(self, observation: RobotObservation) -> SkillStatus:
        self.on_tick(at=observation.monotonic_s)
        if self._status.terminal:
            return self._status
        if observation.safety.fallen or observation.safety.limp:
            return self.interrupt(at=observation.monotonic_s, reason="robot_safety_state")
        if self._status.phase is SkillPhase.ACCEPTED:
            if self._matches(observation.policy):
                self._started_at = observation.monotonic_s
                self._status = SkillStatus(
                    self.execution_id, SkillPhase.STARTED, observation.monotonic_s
                )
            return self._status
        if self._status.phase is SkillPhase.STARTED and not self._matches(observation.policy):
            if observation.policy in {"walk", "stand"}:
                self._status = SkillStatus(
                    self.execution_id, SkillPhase.COMPLETED, observation.monotonic_s
                )
            else:
                return self.interrupt(
                    at=observation.monotonic_s,
                    reason=f"unexpected_policy:{observation.policy}",
                )
        return self._status

    def on_tick(self, *, at: float) -> SkillStatus:
        if self._status.terminal:
            return self._status
        if self._status.phase in {SkillPhase.PENDING, SkillPhase.ACCEPTED}:
            anchor = self._accepted_at if self._accepted_at is not None else self.created_at
            if at - anchor >= self.start_timeout_s:
                self._status = SkillStatus(
                    self.execution_id, SkillPhase.TIMED_OUT, at, "start_timeout"
                )
        elif self._status.phase is SkillPhase.STARTED:
            assert self._started_at is not None
            if at - self._started_at >= self.completion_timeout_s:
                self._status = SkillStatus(
                    self.execution_id, SkillPhase.TIMED_OUT, at, "completion_timeout"
                )
        return self._status

    def interrupt(self, *, at: float, reason: str) -> SkillStatus:
        if not self._status.terminal:
            self._status = SkillStatus(self.execution_id, SkillPhase.INTERRUPTED, at, reason)
        return self._status

    def _matches(self, policy: str) -> bool:
        return policy == self.expected_policy or policy == f"{self.expected_policy}:unwind"


@dataclass(frozen=True, slots=True)
class OneShotSkillIntent:
    """Semantic instruction. No robotd method, joint or motor parameter."""

    intent_id: str
    skill: str

    def __post_init__(self) -> None:
        if not self.intent_id.strip() or not self.skill.strip():
            raise ValueError("intent_id and skill are required")


@dataclass(slots=True)
class OneShotExecution:
    """Observed execution. ACK, policy transition and outcome remain distinct."""

    intent: OneShotSkillIntent
    tracker: SkillTracker
    ack: BodyAck
    dispatched_at: float
    _body: BodyPort
    _clock: Callable[[], float]

    @property
    def status(self) -> SkillStatus:
        return self.tracker.status

    def poll(self) -> tuple[RobotObservation, SkillStatus]:
        try:
            observation = self._body.observe()
        except (ConnectionError, TimeoutError):
            self.tracker.interrupt(at=self._clock(), reason="body_transport_lost")
            raise
        self.tracker.on_tick(at=self._clock())
        self.tracker.on_observation(observation)
        return observation, self.tracker.status

    def interrupt(self, reason: str) -> SkillStatus:
        self._body.stop()
        return self.tracker.interrupt(at=self._clock(), reason=reason)


class EmbodiedSkillLayer:
    """Only component translating semantic skills into robotd-level intents."""

    def __init__(self, body: BodyPort, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._body = body
        self._clock = clock
        self._active: OneShotExecution | None = None

    def start_one_shot(
        self,
        intent: OneShotSkillIntent,
        *,
        start_timeout_s: float = 2.0,
        completion_timeout_s: float = 8.0,
    ) -> OneShotExecution:
        if self._active is not None and not self._active.status.terminal:
            raise RuntimeError("another Duck Mind skill execution is still active")
        if intent.skill == "sit_toggle":
            raise ValueError("sit_toggle has a multi-phase posture contract, not a named policy")
        created_at = self._clock()
        tracker = SkillTracker(
            execution_id=str(uuid.uuid4()),
            expected_policy=intent.skill,
            created_at=created_at,
            start_timeout_s=start_timeout_s,
            completion_timeout_s=completion_timeout_s,
        )
        ack = self._body.dispatch_intent(BodyIntent("robot.do", {"skill": intent.skill}))
        dispatched_at = self._clock()
        tracker.on_dispatch(DispatchAck(ack.accepted, ack.reason), at=dispatched_at)
        execution = OneShotExecution(intent, tracker, ack, dispatched_at, self._body, self._clock)
        self._active = execution
        return execution

    def perform_front_gaze(
        self,
        *,
        cancelled: threading.Event | None = None,
        command_lock: threading.Lock | None = None,
        timeout_s: float = 3.0,
        tolerance_rad: float = 0.12,
    ) -> SkillStatus:
        """First bounded play primitive: small head command, measured verification.

        The target is fixed inside the skill adapter. Current official v5
        velstand observes head deltas from HOME, despite robot.look returning
        absolute angles. We therefore do not use robot.look for board play.
        Neither System 1 nor a caller may choose head angles or joint indices.
        """

        if timeout_s <= 0 or not 0 < tolerance_rad < 0.5:
            raise ValueError("invalid gaze timeout or tolerance")
        execution_id = str(uuid.uuid4())

        def dispatch() -> BodyAck | None:
            if cancelled is not None and cancelled.is_set():
                return None
            return self._body.dispatch_intent(
                BodyIntent(
                    "robot.head",
                    {
                        "neck_pitch": 0.0,
                        "head_pitch": 0.15,
                        "head_yaw": 0.0,
                        "head_roll": 0.0,
                    },
                )
            )

        if command_lock is None:
            ack = dispatch()
        else:
            with command_lock:
                ack = dispatch()
        at = self._clock()
        if ack is None:
            return SkillStatus(execution_id, SkillPhase.INTERRUPTED, at, "owner_stop")
        if not ack.accepted or ack.reason == "clamped":
            return SkillStatus(execution_id, SkillPhase.REFUSED, at, ack.reason or "robotd_refused")
        deadline = at + timeout_s
        stable = 0
        samples = 0
        min_error: float | None = None
        while self._clock() < deadline:
            if cancelled is not None and cancelled.is_set():
                return SkillStatus(
                    execution_id, SkillPhase.INTERRUPTED, self._clock(), "owner_stop"
                )
            try:
                observation = self._body.observe()
            except (ConnectionError, TimeoutError):
                return SkillStatus(
                    execution_id,
                    SkillPhase.INTERRUPTED,
                    self._clock(),
                    "body_transport_lost",
                )
            issue = front_gaze_observation_issue(observation)
            if issue is not None:
                return SkillStatus(
                    execution_id,
                    SkillPhase.INTERRUPTED,
                    self._clock(),
                    issue,
                )
            samples += 1
            if observation.head_tracking_error_rad is not None:
                min_error = (
                    observation.head_tracking_error_rad
                    if min_error is None
                    else min(min_error, observation.head_tracking_error_rad)
                )
            if (
                observation.head_tracking_error_rad is not None
                and observation.head_tracking_error_rad <= tolerance_rad
            ):
                stable += 1
                if stable >= 2:
                    return SkillStatus(execution_id, SkillPhase.COMPLETED, observation.monotonic_s)
            else:
                stable = 0
        detail = "missing_measurement" if min_error is None else f"min_error_rad={min_error:.3f}"
        return SkillStatus(
            execution_id,
            SkillPhase.TIMED_OUT,
            self._clock(),
            f"unverified_head_tracking:{detail}:samples={samples}",
        )
