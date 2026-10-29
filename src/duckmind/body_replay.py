"""Strict BodyPort replay for recorded, hardware-observable command streams.

This port never opens robotd and never exposes simulator-only oracle values to
the Embodied Skill Layer. A mismatch is a validation failure, not a fallback.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence

from duckmind.body import BodyAck, BodyIntent, RobotObservation
from duckmind.replay import ReplayValidator, ValidationReport
from duckmind.skills import EmbodiedSkillLayer, OneShotSkillIntent


class ReplayMismatch(RuntimeError):
    """The replayed skill made a different body call than the recorded run."""


@dataclass(frozen=True, slots=True)
class ReplayDispatch:
    intent: BodyIntent
    ack: BodyAck


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    observation: RobotObservation

    def __post_init__(self) -> None:
        metadata = self.observation.metadata
        if metadata is not None and set(metadata) - {"odom"}:
            raise ValueError("simulator-only metadata cannot enter a replay body frame")


@dataclass(frozen=True, slots=True)
class ReplayStop:
    ack: BodyAck


ReplayStep = ReplayDispatch | ReplayFrame | ReplayStop


class RecordedBodyPort:
    """BodyPort that consumes one exact, ordered command/observation script."""

    def __init__(self, steps: Sequence[ReplayStep], *, initial_time: float) -> None:
        if not math.isfinite(initial_time):
            raise ValueError("replay initial time must be finite")
        self._steps = tuple(steps)
        self._index = 0
        self._now = initial_time
        self._failed = False

    def clock(self) -> float:
        return self._now

    def _mismatch(self, reason: str) -> NoReturn:
        self._failed = True
        raise ReplayMismatch(reason)

    def _next(self, action: str) -> ReplayStep:
        if self._failed:
            raise ReplayMismatch("recorded body trace has an earlier mismatch")
        if self._index >= len(self._steps):
            self._mismatch("recorded body trace exhausted")
        step = self._steps[self._index]
        if isinstance(step, ReplayDispatch):
            expected = "dispatch"
        elif isinstance(step, ReplayFrame):
            expected = "observation"
        else:
            expected = "stop"
        if action != expected:
            self._mismatch(f"expected {expected}, got {action}")
        self._index += 1
        return step

    def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
        step = self._next("dispatch")
        assert isinstance(step, ReplayDispatch)
        if intent != step.intent:
            self._mismatch(
                f"intent mismatch: expected {step.intent.method} {dict(step.intent.params)}"
            )
        return step.ack

    def observe(self) -> RobotObservation:
        step = self._next("observation")
        assert isinstance(step, ReplayFrame)
        observed = step.observation
        if not math.isfinite(observed.monotonic_s) or observed.monotonic_s <= self._now:
            self._mismatch("recorded observation time is not increasing")
        self._now = observed.monotonic_s
        return observed

    def stop(self) -> BodyAck:
        step = self._next("stop")
        assert isinstance(step, ReplayStop)
        return step.ack

    def assert_consumed(self) -> None:
        if self._failed:
            raise ReplayMismatch("recorded body trace has an earlier mismatch")
        if self._index != len(self._steps):
            raise ReplayMismatch(f"unconsumed body steps: {len(self._steps) - self._index}")


def _one(records: list[dict], kind: str) -> dict:
    matches = [row for row in records if row.get("type") == kind]
    if len(matches) != 1:
        raise ValueError(f"expected one {kind} record")
    return matches[0]


def parse_recorded_ack(value: object) -> BodyAck:
    if not isinstance(value, dict):
        raise ValueError("recorded acknowledgement must be an object")
    accepted, confirmed, reason = (
        value.get("accepted"),
        value.get("confirmed"),
        value.get("reason"),
    )
    if (
        not isinstance(accepted, bool)
        or not isinstance(confirmed, bool)
        or (reason is not None and not isinstance(reason, str))
    ):
        raise ValueError("recorded acknowledgement is malformed")
    return BodyAck(accepted, reason, confirmed)


def replay_named_skill_trace(path: str | Path) -> ValidationReport:
    """Replay an official named-skill trace through the real skill adapter.

    The existing trace validator first checks wire/oracle provenance. This
    second pass checks the semantic intent -> BodyPort command and phase path.
    """

    validated = ReplayValidator().validate_skill_trace(path)
    if not validated.passed:
        return validated
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        run = _one(records, "run")
        dispatch = _one(records, "dispatch")
        outcome = _one(records, "outcome")
        samples = [row for row in records if row.get("type") == "sample"]
        dispatch_at = ReplayValidator._finite_number(dispatch["local_monotonic_s"], "dispatch time")
        body = RecordedBodyPort(
            (
                ReplayDispatch(
                    BodyIntent("robot.do", {"skill": dispatch["skill"]}),
                    parse_recorded_ack(dispatch["ack"]),
                ),
                *(
                    ReplayFrame(ReplayValidator.project_robot_observation(sample))
                    for sample in samples
                ),
            ),
            initial_time=dispatch_at,
        )
        execution = EmbodiedSkillLayer(body, clock=body.clock).start_one_shot(
            OneShotSkillIntent(dispatch["intent_id"], run["skill"]),
            start_timeout_s=ReplayValidator._finite_number(run["start_timeout_s"], "start timeout"),
            completion_timeout_s=ReplayValidator._finite_number(
                run["completion_timeout_s"], "completion timeout"
            ),
        )
        for index, sample in enumerate(samples):
            _, status = execution.poll()
            if status.phase.value != sample["phase"]:
                return ValidationReport(False, ("skill_layer_phase_diverged",), index)
        execution.tracker.on_tick(
            at=ReplayValidator._finite_number(outcome["local_monotonic_s"], "outcome time")
        )
        if execution.status.phase.value != outcome["phase"]:
            return ValidationReport(False, ("skill_layer_outcome_diverged",))
        body.assert_consumed()
        return ValidationReport(True, ())
    except (OSError, ValueError, TypeError, KeyError, IndexError, ReplayMismatch) as error:
        return ValidationReport(False, (f"body_replay_invalid:{type(error).__name__}",))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded body calls through Duck Mind")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    report = replay_named_skill_trace(args.trace)
    print(json.dumps({"passed": report.passed, "reasons": report.reasons}))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
