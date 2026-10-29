"""Simulator-only recording and replay of the first pet head-game body path."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from duckmind.body import BodyAck, BodyIntent, BodyPort, RobotObservation
from duckmind.body_replay import (
    RecordedBodyPort,
    ReplayDispatch,
    ReplayFrame,
    ReplayMismatch,
    ReplayStop,
    parse_recorded_ack,
)
from duckmind.replay import ReplayValidator, ValidationReport
from duckmind.skills import EmbodiedSkillLayer, SkillStatus

MAX_STEPS = 256
MAX_TRACE_BYTES = 262_144
SHA256 = re.compile(r"[0-9a-f]{64}")


class BodyTraceRecorder:
    """Record only BodyPort calls made by one simulator look_game execution."""

    def __init__(
        self,
        body: BodyPort,
        *,
        interaction_id: str,
        source_sha256: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not interaction_id or SHA256.fullmatch(source_sha256) is None:
            raise ValueError("body trace requires an interaction ID and source hash")
        self._body = body
        self._interaction_id = interaction_id
        self._source_sha256 = source_sha256
        self._initial_time = clock()
        if not math.isfinite(self._initial_time):
            raise ValueError("body trace initial time must be finite")
        self._steps: list[dict[str, object]] = []

    def _append(self, step: dict[str, object]) -> None:
        if len(self._steps) >= MAX_STEPS:
            raise RuntimeError("simulator body trace exceeded step limit")
        self._steps.append(step)

    def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
        ack = self._body.dispatch_intent(intent)
        self._append(
            {
                "type": "dispatch",
                "method": intent.method,
                "params": dict(intent.params),
                "ack": {
                    "accepted": ack.accepted,
                    "reason": ack.reason,
                    "confirmed": ack.confirmed,
                },
            }
        )
        return ack

    def observe(self) -> RobotObservation:
        observed = self._body.observe()
        if observed.metadata is not None and set(observed.metadata) - {"odom"}:
            raise ValueError("simulator-only metadata cannot enter the body trace")
        if observed.robot_time_s is None:
            raise ValueError("body trace requires robotd time")
        head_error = observed.head_tracking_error_rad
        if head_error is not None and (not math.isfinite(head_error) or head_error < 0):
            raise ValueError("body trace head tracking error is invalid")
        self._append(
            {
                "type": "observation",
                "local_monotonic_s": observed.monotonic_s,
                "robot_time_s": observed.robot_time_s,
                "policy": observed.policy,
                "move": {
                    "requested": list(observed.movement_requested),
                    "applied": list(observed.movement_applied),
                    "limited_by": list(observed.limited_by),
                },
                "safety": {
                    "fallen": observed.safety.fallen,
                    "limp": observed.safety.limp,
                },
                "head_tracking_error_rad": head_error,
            }
        )
        return observed

    def stop(self) -> BodyAck:
        ack = self._body.stop()
        self._append(
            {
                "type": "stop",
                "ack": {
                    "accepted": ack.accepted,
                    "reason": ack.reason,
                    "confirmed": ack.confirmed,
                },
            }
        )
        return ack

    def save(self, path: Path, *, status: SkillStatus) -> None:
        record = {
            "schema_version": 1,
            "scenario": "simulator-look-game-body-v1",
            "interaction_id": self._interaction_id,
            "duckmind_source_sha256": self._source_sha256,
            "initial_monotonic_s": self._initial_time,
            "steps": self._steps,
            "result": {"phase": status.phase.value, "reason": status.reason, "at": status.at},
        }
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(encoded) > MAX_TRACE_BYTES:
            raise RuntimeError("simulator body trace exceeded size limit")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(encoded)


def replay_front_gaze_trace(path: str | Path) -> ValidationReport:
    """Re-run the fixed head game against recorded production-observable frames."""

    try:
        trace_path = Path(path)
        if trace_path.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("body trace exceeded size limit")
        record = json.loads(trace_path.read_text())
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("scenario") != "simulator-look-game-body-v1"
            or not isinstance(record.get("interaction_id"), str)
            or SHA256.fullmatch(str(record.get("duckmind_source_sha256"))) is None
        ):
            raise ValueError("unsupported pet body trace header")
        raw_steps = record.get("steps")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_STEPS:
            raise ValueError("invalid pet body trace steps")
        steps: list[ReplayDispatch | ReplayFrame | ReplayStop] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                raise ValueError("body step must be an object")
            kind = raw.get("type")
            if kind == "dispatch" and set(raw) == {"type", "method", "params", "ack"}:
                steps.append(
                    ReplayDispatch(
                        BodyIntent(raw["method"], raw["params"]),
                        parse_recorded_ack(raw["ack"]),
                    )
                )
            elif kind == "observation" and set(raw) == {
                "type",
                "local_monotonic_s",
                "robot_time_s",
                "policy",
                "move",
                "safety",
                "head_tracking_error_rad",
            }:
                observed = ReplayValidator.project_robot_observation(raw)
                head_error = raw["head_tracking_error_rad"]
                if head_error is not None:
                    head_error = ReplayValidator._finite_number(head_error, "head tracking error")
                    if head_error < 0:
                        raise ValueError("negative head tracking error")
                steps.append(ReplayFrame(replace(observed, head_tracking_error_rad=head_error)))
            elif kind == "stop" and set(raw) == {"type", "ack"}:
                steps.append(ReplayStop(parse_recorded_ack(raw["ack"])))
            else:
                raise ValueError("unknown or malformed pet body step")
        body = RecordedBodyPort(
            steps,
            initial_time=ReplayValidator._finite_number(
                record["initial_monotonic_s"], "initial time"
            ),
        )
        status = EmbodiedSkillLayer(body, clock=body.clock).perform_front_gaze()
        result = record.get("result")
        if not isinstance(result, dict) or set(result) != {"phase", "reason", "at"}:
            raise ValueError("malformed pet body result")
        if (
            status.phase.value != result["phase"]
            or status.reason != result["reason"]
            or not math.isclose(
                status.at,
                ReplayValidator._finite_number(result["at"], "result time"),
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            return ValidationReport(False, ("front_gaze_result_diverged",))
        body.assert_consumed()
        return ValidationReport(True, ())
    except (OSError, ValueError, TypeError, KeyError, IndexError, ReplayMismatch) as error:
        return ValidationReport(False, (f"invalid_pet_body_trace:{type(error).__name__}",))
