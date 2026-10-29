"""A finished PPO export is not a qualified robot policy."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from duckmind.qualification import assess_official_probe
from tests.checkout_parts import needs_evidence


def probe_report():
    episodes = [
        {
            "seed": seed,
            "steps": 50,
            "fell_over": seed == 10002,
            "terminated": seed == 10002,
            "truncated": False,
            "all_finite": True,
            "min_trunk_height_m": 0.09,
        }
        for seed in range(10001, 10005)
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "task_id": "Mjlab-Velocity-Flat-MicroDuck",
        "horizon": 250,
        "device": "cpu",
        "play": True,
        "auto_reset": False,
        "num_envs": 1,
        "physical_hardware_gate": "not_run",
        "upstream": {"commit": "reviewed", "uv_lock_sha256": "lock"},
        "artifacts": {
            "checkpoint": {"sha256": "c" * 64},
            "onnx": {"sha256": "o" * 64},
        },
        "probe_script_sha256": "p" * 64,
        "seeds": [item["seed"] for item in episodes],
        "candidate_episodes": episodes,
        "candidate_summary": {"episodes": 4, "falls": 1, "total_steps": 200},
        "parity": {
            "passed": True,
            "all_finite": True,
            "failed_samples": 0,
            "samples": 200,
            "max_abs_error": 0.0,
        },
    }


def assess(report):
    return assess_official_probe(
        report,
        checkpoint_sha256="c" * 64,
        onnx_sha256="o" * 64,
        script_sha256="p" * 64,
        upstream_commit="reviewed",
        uv_lock_sha256="lock",
    )


def test_observed_fall_rejects_compatible_model():
    result = assess(probe_report())
    assert result["verdict"] == "rejected"
    assert result["fall_count"] == 1
    assert result["physical_hardware_gate"] == "not_run"


def test_no_falls_in_four_short_trials_is_still_unqualified():
    report = probe_report()
    report["candidate_episodes"][1]["fell_over"] = False
    report["candidate_episodes"][1]["terminated"] = False
    report["candidate_summary"]["falls"] = 0
    result = assess(report)
    assert result["verdict"] == "inconclusive"
    assert "coverage_too_small" in result["reasons"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["artifacts"]["onnx"].update(sha256="wrong"),
        lambda r: r["artifacts"]["checkpoint"].update(sha256="wrong"),
        lambda r: r["parity"].update(passed=False),
        lambda r: r["parity"].update(failed_samples=1),
        lambda r: r["candidate_summary"].update(falls=0),
        lambda r: r["candidate_episodes"][0].update(seed=10002),
        lambda r: r.update(physical_hardware_gate="pass"),
        lambda r: r.update(probe_script_sha256="other"),
        lambda r: r.update(task_id="Other-Task"),
        lambda r: r.update(play=False),
        lambda r: r.update(evaluation_profile="nonplay", diagnostic_only=True),
        lambda r: r.update(evaluation_profile="play-no-push", diagnostic_only=True),
        lambda r: r.update(evaluation_profile="play-zero-push", diagnostic_only=True),
        lambda r: r.update(push_event_enabled=False),
        lambda r: r.update(push_velocity_zeroed=True),
        lambda r: r.update(horizon=0),
        lambda r: r["candidate_episodes"][0].update(all_finite=False),
    ],
)
def test_report_tampering_cannot_qualify_a_candidate(mutate):
    report = copy.deepcopy(probe_report())
    mutate(report)
    with pytest.raises(ValueError):
        assess(report)


@needs_evidence
@pytest.mark.parametrize("run", ["mac-managed-gate-001", "mac-managed-gate-002"])
def test_published_managed_mac_evidence_is_bound_and_rejected(run):
    evidence_root = Path(__file__).parents[1] / "docs/validation/evidence"
    job = json.loads((evidence_root / f"{run}-job.json").read_text())
    physics = json.loads((evidence_root / f"{run}-physics-summary.json").read_text())
    assert job["artifacts"]["simulation"]["sha256"] == physics["raw_report_sha256"]
    assert job["artifacts"]["checkpoint"]["sha256"] == physics["artifacts"]["checkpoint"]["sha256"]
    assert job["artifacts"]["onnx"]["sha256"] == physics["artifacts"]["onnx"]["sha256"]
    decision = assess_official_probe(
        physics,
        checkpoint_sha256=job["artifacts"]["checkpoint"]["sha256"],
        onnx_sha256=job["artifacts"]["onnx"]["sha256"],
        script_sha256=physics["probe_script_sha256"],
        upstream_commit=job["upstream_identity"]["commit"],
        uv_lock_sha256=job["upstream_identity"]["uv_lock_sha256"],
    )
    assert decision == job["qualification"]
    assert job["status"] == "candidate_rejected"
    assert job["physical_hardware_gate"] == "not_run"


@needs_evidence
def test_longer_cpu_training_summary_preserves_both_failed_profiles():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (root / "docs/validation/evidence/mac-official-1000-001-summary.json").read_text()
    )
    assert summary["status"] == "candidate_rejected"
    assert summary["physical_hardware_gate"] == "not_run"
    assert summary["training"]["max_iterations"] == 1000
    assert summary["profiles"]["play"]["qualification"] == "rejected"
    assert summary["profiles"]["nonplay"]["qualification"] == "not_eligible"
    for profile in summary["profiles"].values():
        assert profile["falls"] == len(summary["probe"]["seeds"]) == 4
        assert sum(profile["steps"]) == profile["parity_samples"]
        assert profile["minimum_trunk_height_m"] < 0
        raw = root / profile["raw_report_path"]
        if raw.exists():
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == profile["raw_report_sha256"]
    pair = summary["robotd_comparison"]
    assert pair["assessment"] == "rejected"
    assert pair["baseline_walk_reached"] == [True, True]
    assert pair["candidate_walk_reached"] == [False, False]
    raw_pair = root / pair["raw_report_path"]
    if raw_pair.exists():
        assert hashlib.sha256(raw_pair.read_bytes()).hexdigest() == pair["raw_report_sha256"]


@needs_evidence
def test_push_ablation_summary_is_hash_bound_and_never_qualifies():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (root / "docs/validation/evidence/mac-official-resume-3004-003-summary.json").read_text()
    )
    assert summary["status"] == "candidate_rejected"
    assert summary["physical_hardware_gate"] == "not_run"
    assert summary["formal_play_physics"]["falls"] == 4
    ablation = summary["play_zero_push_diagnostic"]
    assert ablation["diagnostic_only"] is True
    assert ablation["push_event_enabled"] is True
    assert ablation["push_velocity_zeroed"] is True
    assert ablation["qualification"] == "not_applicable"
    assert ablation["falls"] == 0
    assert ablation["initial_actor_matches_formal"] is True
    assert len(ablation["first_action_divergence_steps"]) == 4
    for label in ("play_no_push_diagnostic", "play_zero_push_diagnostic"):
        diagnostic = summary[label]
        raw = root / diagnostic["report_path"]
        if raw.exists():
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == diagnostic["report_sha256"]
            report = json.loads(raw.read_text())
            with pytest.raises(ValueError, match="unexpected simulator profile"):
                assess_official_probe(
                    report,
                    checkpoint_sha256=summary["model"]["checkpoint_sha256"],
                    onnx_sha256=summary["model"]["onnx_sha256"],
                    script_sha256=diagnostic["probe_script_sha256"],
                    upstream_commit=summary["training"]["upstream_commit"],
                    uv_lock_sha256=summary["training"]["upstream_uv_lock_sha256"],
                )
