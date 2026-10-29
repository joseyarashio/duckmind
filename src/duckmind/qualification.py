"""Bind official physics evidence to one trained model without approving hardware.

The short four-seed probe is sufficient to reject an observed fall. A clean
four-seed probe is still insufficient to qualify a locomotion policy.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["assess_official_probe"]

PROBE_SEEDS = (10001, 10002, 10003, 10004)
PROBE_TASK = "Mjlab-Velocity-Flat-MicroDuck"
PROBE_HORIZON = 250


def assess_official_probe(
    report: dict[str, Any],
    *,
    checkpoint_sha256: str,
    onnx_sha256: str,
    script_sha256: str,
    upstream_commit: str,
    uv_lock_sha256: str,
) -> dict[str, Any]:
    """Classify source-bound, completed evidence; never return an approval.

    An invalid report raises ValueError so orchestration fails closed instead
    of labelling missing or mismatched measurements as a physical pass.
    """
    try:
        if type(report["schema_version"]) is not int or report["schema_version"] != 1:
            raise ValueError("unsupported official probe schema")
        if report["status"] != "completed":
            raise ValueError("official probe did not complete")
        if (
            report["task_id"] != PROBE_TASK
            or type(report["horizon"]) is not int
            or report["horizon"] != PROBE_HORIZON
            or report["device"] != "cpu"
            or report["play"] is not True
            or report.get("evaluation_profile", "play") != "play"
            or report.get("diagnostic_only", False) is not False
            or report.get("push_event_enabled", True) is not True
            or report.get("push_velocity_zeroed", False) is not False
            or report["auto_reset"] is not False
            or type(report["num_envs"]) is not int
            or report["num_envs"] != 1
            or report["physical_hardware_gate"] != "not_run"
        ):
            raise ValueError("unexpected simulator profile or hardware claim")
        if (
            report["artifacts"]["checkpoint"]["sha256"] != checkpoint_sha256
            or report["artifacts"]["onnx"]["sha256"] != onnx_sha256
            or report["probe_script_sha256"] != script_sha256
            or report["upstream"]["commit"] != upstream_commit
            or report["upstream"]["uv_lock_sha256"] != uv_lock_sha256
        ):
            raise ValueError("simulator evidence refers to different source or model bytes")
        parity = report["parity"]
        episodes = report["candidate_episodes"]
        summary = report["candidate_summary"]
        if (
            parity["passed"] is not True
            or parity["all_finite"] is not True
            or type(parity["failed_samples"]) is not int
            or parity["failed_samples"] != 0
            or type(parity["samples"]) is not int
            or parity["samples"] < 1
            or type(parity["max_abs_error"]) not in (float, int)
            or not math.isfinite(parity["max_abs_error"])
            or parity["max_abs_error"] < 0
        ):
            raise ValueError("checkpoint and ONNX parity was not proved")
        if (
            report["seeds"] != list(PROBE_SEEDS)
            or not isinstance(episodes, list)
            or len(episodes) != len(PROBE_SEEDS)
            or any(
                item["seed"] != seed
                or type(item["steps"]) is not int
                or not 1 <= item["steps"] <= 250
                or type(item["fell_over"]) is not bool
                or type(item["terminated"]) is not bool
                or type(item["truncated"]) is not bool
                or item["all_finite"] is not True
                or type(item["min_trunk_height_m"]) not in (float, int)
                or not math.isfinite(item["min_trunk_height_m"])
                for seed, item in zip(PROBE_SEEDS, episodes, strict=True)
            )
        ):
            raise ValueError("missing or invalid paired official physics episodes")
        falls = sum(item["fell_over"] for item in episodes)
        steps = sum(item["steps"] for item in episodes)
        if (
            summary["episodes"] != len(episodes)
            or summary["falls"] != falls
            or summary["total_steps"] != steps
            or parity["samples"] != steps
        ):
            raise ValueError("physics summary or inference sample count differs from episodes")
        if report.get("reference_episodes"):
            reference = report["reference_episodes"]
            paired = report.get("paired_initial_observations_match") is True
            same_seeds = [item["seed"] for item in reference] == list(PROBE_SEEDS)
            if not paired or not same_seeds:
                raise ValueError("reference trials were not paired")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("incomplete official physics evidence") from error

    reasons = []
    if falls:
        reasons.append("fall_observed")
    if any(item["terminated"] and not item["fell_over"] for item in episodes):
        reasons.append("nonfall_termination_observed")
    if any(item["min_trunk_height_m"] < 0 for item in episodes):
        reasons.append("trunk_below_world_origin")
    verdict = "rejected" if reasons else "inconclusive"
    if verdict == "inconclusive":
        reasons.append("coverage_too_small")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "fall_count": falls,
        "trial_count": len(episodes),
        "inference_samples": parity["samples"],
        "physical_hardware_gate": "not_run",
    }
