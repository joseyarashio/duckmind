"""Deterministic replay and counterfactual validation primitives."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from duckmind.body import RobotObservation, SafetyObservation
from duckmind.skills import DispatchAck, SkillPhase, SkillTracker


@dataclass(frozen=True, slots=True)
class ReplayExpectation:
    policies: tuple[str, ...]
    max_abs_vx_error: float
    applied_vx: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: bool
    reasons: tuple[str, ...]
    first_divergence_index: int | None = None


class ReplayValidator:
    @staticmethod
    def _finite_number(value: object, field: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid {field}")
        return float(value)

    @classmethod
    def _vector3(cls, value: object, field: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"invalid {field}")
        return (
            cls._finite_number(value[0], field),
            cls._finite_number(value[1], field),
            cls._finite_number(value[2], field),
        )

    def validate_skill_trace(self, path: str | Path) -> ValidationReport:
        """Re-derive a live simulator verdict from recorded wire/oracle evidence.

        The oracle values are used only in the evaluator. They are never put
        into the RobotObservation passed to the skill lifecycle reducer.
        """

        try:
            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            run = self._one(records, "run")
            dispatch = self._one(records, "dispatch")
            outcome = self._one(records, "outcome")
            episode = self._one(records, "episode")
            samples = [record for record in records if record.get("type") == "sample"]
            if run["schema_version"] != 1:
                return ValidationReport(False, ("unsupported_trace_schema",))
            if run["trace_id"] != outcome["trace_id"] or run["episode_id"] != episode["episode_id"]:
                return ValidationReport(False, ("trace_or_episode_id_mismatch",))
            if run["skill"] != dispatch["skill"]:
                return ValidationReport(False, ("skill_mismatch",))
            dispatch_at = float(dispatch["local_monotonic_s"])
            tracker = SkillTracker(
                execution_id=outcome["execution_id"],
                expected_policy=run["skill"],
                created_at=dispatch_at,
                start_timeout_s=float(run["start_timeout_s"]),
                completion_timeout_s=float(run["completion_timeout_s"]),
            )
            tracker.on_dispatch(
                DispatchAck(
                    accepted=dispatch["ack"]["accepted"],
                    reason=dispatch["ack"].get("reason"),
                ),
                at=dispatch_at,
            )
            phases = {tracker.status.phase.value}
            evidence_ids = []
            previous_at = dispatch_at
            previous_robot_at = -math.inf
            min_trunk_z = float(self._one(records, "baseline")["trunk_z"])
            max_joint_delta = 0.0
            limit_reasons: set[str] = set()
            for index, sample in enumerate(samples, 1):
                expected_id = f"{run['trace_id']}:{index}"
                if sample["event_id"] != expected_id:
                    return ValidationReport(False, ("event_order_or_id_mismatch",), index - 1)
                if sample["episode_id"] != run["episode_id"]:
                    return ValidationReport(False, ("sample_episode_mismatch",), index - 1)
                at = float(sample["local_monotonic_s"])
                robot_at = float(sample["robot_time_s"])
                if not math.isfinite(at) or at <= previous_at:
                    return ValidationReport(False, ("local_time_not_monotonic",), index - 1)
                if not math.isfinite(robot_at) or robot_at <= previous_robot_at:
                    return ValidationReport(False, ("robot_time_not_monotonic",), index - 1)
                previous_at, previous_robot_at = at, robot_at
                observed = self.project_robot_observation(sample)
                limit_reasons.update(observed.limited_by)
                status = tracker.on_observation(observed)
                if status.phase.value != sample["phase"]:
                    return ValidationReport(False, ("phase_diverged",), index - 1)
                tracker.on_tick(at=at)
                phases.add(tracker.status.phase.value)
                evidence_ids.append(sample["event_id"])
                oracle = sample["oracle"]
                min_trunk_z = min(min_trunk_z, float(oracle["trunk_z"]))
                max_joint_delta = max(max_joint_delta, float(oracle["max_joint_delta_rad"]))
            tracker.on_tick(at=float(outcome["local_monotonic_s"]))
            if tracker.status.phase.value != outcome["phase"]:
                return ValidationReport(False, ("outcome_phase_diverged",))
            if (
                outcome["sample_count"] != len(samples)
                or outcome["evidence_event_ids"] != evidence_ids
            ):
                return ValidationReport(False, ("outcome_evidence_mismatch",))
            if episode["event_ids"] != evidence_ids or episode["outcome_ids"] != [
                outcome["outcome_id"]
            ]:
                return ValidationReport(False, ("episode_provenance_mismatch",))
            if sorted(phases) != outcome["phase_evidence"]:
                return ValidationReport(False, ("phase_evidence_mismatch",))
            if not math.isclose(min_trunk_z, outcome["min_trunk_z_m"], abs_tol=1e-9):
                return ValidationReport(False, ("trunk_height_metric_mismatch",))
            if not math.isclose(max_joint_delta, outcome["max_joint_delta_rad"], abs_tol=1e-9):
                return ValidationReport(False, ("joint_delta_metric_mismatch",))
            gate_passed = (
                tracker.status.phase is SkillPhase.COMPLETED
                and SkillPhase.STARTED.value in phases
                and max_joint_delta >= 0.02
            )
            if outcome["execution_gate"] != ("pass" if gate_passed else "fail"):
                return ValidationReport(False, ("execution_gate_mismatch",))
            warnings = []
            if min_trunk_z < 0.06:
                warnings.append("low_trunk_height_observed")
            if "joint_range" in limit_reasons:
                warnings.append("joint_range_limit_observed")
            if outcome["warnings"] != warnings:
                return ValidationReport(False, ("execution_warnings_mismatch",))
            expected_reason = (
                "no_task_goal_evidence"
                if gate_passed
                else tracker.status.reason or "execution_gate_failed"
            )
            if outcome.get("reason") != expected_reason:
                return ValidationReport(False, ("outcome_reason_diverged",))
            if outcome["verdict"] != "unknown":
                return ValidationReport(False, ("outcome_verdict_mismatch",))
            return ValidationReport(True, ())
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as error:
            return ValidationReport(False, (f"invalid_trace:{type(error).__name__}",))

    @staticmethod
    def _one(records: list[dict], kind: str) -> dict:
        matches = [record for record in records if record.get("type") == kind]
        if len(matches) != 1:
            raise ValueError(f"expected one {kind} record")
        return matches[0]

    @classmethod
    def project_robot_observation(cls, sample: Mapping[str, object]) -> RobotObservation:
        """Production-equivalent projection; deliberately ignores `oracle`."""

        movement = sample.get("move")
        safety = sample.get("safety")
        policy = sample.get("policy")
        if (
            not isinstance(movement, dict)
            or not isinstance(safety, dict)
            or not isinstance(policy, str)
        ):
            raise ValueError("invalid robot observation envelope")
        limited = movement.get("limited_by")
        if not isinstance(limited, list) or any(not isinstance(item, str) for item in limited):
            raise ValueError("invalid movement limits")
        fallen = safety.get("fallen")
        limp = safety.get("limp")
        if not isinstance(fallen, bool) or not isinstance(limp, bool):
            raise ValueError("invalid safety flags")
        return RobotObservation(
            monotonic_s=cls._finite_number(sample.get("local_monotonic_s"), "local time"),
            robot_time_s=cls._finite_number(sample.get("robot_time_s"), "robot time"),
            policy=policy,
            movement_requested=cls._vector3(movement.get("requested"), "requested movement"),
            movement_applied=cls._vector3(movement.get("applied"), "applied movement"),
            limited_by=tuple(limited),
            safety=SafetyObservation(fallen=fallen, limp=limp),
            backend="simulator",
        )

    def validate(
        self,
        expected: ReplayExpectation,
        actual: Sequence[RobotObservation],
    ) -> ValidationReport:
        for index, expected_policy in enumerate(expected.policies):
            if index >= len(actual):
                return ValidationReport(False, ("observation_missing",), index)
            observed = actual[index]
            if observed.policy != expected_policy:
                return ValidationReport(
                    False,
                    (f"policy:expected={expected_policy} actual={observed.policy}",),
                    index,
                )
            if expected.applied_vx:
                error = abs(observed.movement_applied[0] - expected.applied_vx[index])
                if error > expected.max_abs_vx_error:
                    return ValidationReport(
                        False,
                        (f"applied_vx_error:{error:.6f}",),
                        index,
                    )
        if len(actual) != len(expected.policies):
            return ValidationReport(
                False,
                (f"observation_count:expected={len(expected.policies)} actual={len(actual)}",),
                min(len(actual), len(expected.policies)),
            )
        return ValidationReport(True, ())

    def compare_counterfactual(
        self,
        *,
        baseline_decision: str,
        counterfactual_decision: str,
        changed_fields: Mapping[str, tuple[object, object]],
        allowed_influences: Mapping[str, frozenset[str]],
    ) -> ValidationReport:
        if baseline_decision == counterfactual_decision:
            return ValidationReport(True, ())
        permitted = allowed_influences.get(counterfactual_decision, frozenset())
        disallowed = tuple(field for field in changed_fields if field not in permitted)
        if disallowed:
            return ValidationReport(
                False,
                tuple(f"decision_changed_without_allowed_cause:{field}" for field in disallowed),
            )
        return ValidationReport(True, ())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a recorded Duck Mind skill trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    report = ReplayValidator().validate_skill_trace(args.trace)
    print(json.dumps({"passed": report.passed, "reasons": report.reasons}))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
