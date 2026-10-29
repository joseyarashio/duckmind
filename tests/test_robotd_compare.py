"""The live robotd twin is a diagnostic gate, never hardware approval."""

import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from duckmind.body import BodyAck, BodyIntent
from duckmind.robotd_compare import (
    FixedForwardProbeSkill,
    _capture_first_walk_truth,
    _child_env,
    assess_robotd_pair,
    make_walk_config,
    verify_candidate_job,
)
from tests.checkout_parts import needs_evidence


def test_first_walk_truth_is_captured_once_and_stays_out_of_body_observation():
    result = {"first_walk_truth": None}
    calls = 0

    def read_truth():
        nonlocal calls
        calls += 1
        return {
            "sim_time": 2.08,
            "trunk_z": 0.10,
            "imu": {"gravity": [0.0, 0.1, -0.995]},
        }

    safety = SimpleNamespace(fallen=True)
    homing = SimpleNamespace(policy="homing", robot_time_s=2.0, safety=safety)
    walking = SimpleNamespace(policy="walk", robot_time_s=2.14, safety=safety)
    _capture_first_walk_truth(result, homing, read_truth)
    assert calls == 0
    assert result["first_walk_truth"] is None

    _capture_first_walk_truth(result, walking, read_truth)
    _capture_first_walk_truth(result, walking, read_truth)
    assert calls == 1
    assert result["first_walk_truth"] == {
        "robot_time_s": 2.14,
        "sim_time_s": 2.08,
        "trunk_z_m": 0.10,
        "projected_gravity": [0.0, 0.1, -0.995],
        "safety_fallen": True,
    }
    assert not hasattr(walking, "trunk_z_m")


@pytest.mark.parametrize(
    "truth",
    [
        {"sim_time": 2.0, "trunk_z": float("nan"), "imu": {"gravity": [0, 0, -1]}},
        {"sim_time": 2.0, "trunk_z": 0.1, "imu": {"gravity": [0, float("inf"), -1]}},
        {"sim_time": 2.0, "trunk_z": 0.1, "imu": {"gravity": [0, -1]}},
    ],
)
def test_first_walk_truth_rejects_nonfinite_or_incomplete_oracle_data(truth):
    result = {"first_walk_truth": None}
    walking = SimpleNamespace(policy="walk", robot_time_s=2.14, safety=SimpleNamespace(fallen=True))
    with pytest.raises(ValueError, match="first walk oracle"):
        _capture_first_walk_truth(result, walking, lambda: truth)
    assert result["first_walk_truth"] is None


class ProbeBody:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.intents: list[BodyIntent] = []
        self.stops = 0

    def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
        self.intents.append(intent)
        return BodyAck(self.accepted)

    def stop(self) -> BodyAck:
        self.stops += 1
        return BodyAck(self.accepted)


def test_fixed_probe_only_emits_robotd_semantic_commands():
    body = ProbeBody()
    probe = FixedForwardProbeSkill(body)
    probe.forward_tick()
    probe.stop()
    assert len(body.intents) == 1
    assert body.intents[0].method == "robot.move"
    assert dict(body.intents[0].params) == {"vx": 0.15, "vy": 0.0, "vyaw": 0.0}
    assert body.stops == 1


def test_fixed_probe_fails_closed_on_refused_command():
    body = ProbeBody(accepted=False)
    probe = FixedForwardProbeSkill(body)
    with pytest.raises(RuntimeError, match="forward"):
        probe.forward_tick()
    with pytest.raises(RuntimeError, match="stop"):
        probe.stop()


def test_child_process_environment_omits_optional_remote_tokens(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secret")
    child = _child_env(PYTHONPATH="upstream/src")
    assert child["PYTHONPATH"] == "upstream/src"
    assert "WANDB_API_KEY" not in child
    assert "HF_TOKEN" not in child
    assert "HUGGING_FACE_HUB_TOKEN" not in child


def candidate_job(tmp_path: Path) -> tuple[Path, Path]:
    job = tmp_path / "job"
    model = job / "artifacts/policy.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture ONNX bytes")
    report = {
        "status": "candidate_rejected",
        "duckmind_source_sha256": "a" * 64,
        "compatibility_gate": "pass",
        "simulator_gate": "failed",
        "physical_hardware_gate": "not_run",
        "qualification": {"verdict": "rejected"},
        "artifacts": {
            "onnx": {
                "path": "artifacts/policy.onnx",
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "upstream_identity": {"commit": "cb70b792312d559a4da09064d92009079671815f"},
    }
    (job / "job.json").write_text(json.dumps(report))
    return job, model


def test_candidate_job_is_bound_to_model_bytes_even_when_previously_rejected(tmp_path):
    job, model = candidate_job(tmp_path)
    identity = verify_candidate_job(job)
    assert identity["model"] == model
    assert identity["quality_status"] == "candidate_rejected"
    assert identity["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["bytes", "path", "hardware", "compatibility", "upstream"])
def test_candidate_job_rejects_mismatched_provenance_before_launch(tmp_path, fault):
    job, model = candidate_job(tmp_path)
    report = json.loads((job / "job.json").read_text())
    if fault == "bytes":
        model.write_bytes(b"changed")
    elif fault == "path":
        report["artifacts"]["onnx"]["path"] = "../policy.onnx"
    elif fault == "hardware":
        report["physical_hardware_gate"] = "pass"
    elif fault == "compatibility":
        report["compatibility_gate"] = "failed"
    else:
        report["upstream_identity"]["commit"] = "other"
    (job / "job.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_candidate_job(job)


def test_isolated_walk_override_preserves_official_other_policies(tmp_path):
    policies = tmp_path / "official"
    policies.mkdir()
    for name in (
        "velstand.onnx",
        "alpha_sitstand.onnx",
        "alpha_ground_pick.onnx",
        "ball_kick_left.onnx",
        "ball_kick_right.onnx",
        "roulade.onnx",
    ):
        (policies / name).write_bytes(b"policy")
    candidate = tmp_path / "isolated/dm-candidate-1234.onnx"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    config = tomllib.loads(make_walk_config(policies, candidate))
    assert config["policy"]["walk"] == str(candidate)
    assert config["policy"]["kick_left"] == str(policies / "ball_kick_left.onnx")
    assert config["policy"]["sitstand"] == str(policies / "alpha_sitstand.onnx")
    assert config["policy"]["enabled"] is True


def test_walk_override_escapes_toml_special_characters(tmp_path):
    policies = tmp_path / "official"
    policies.mkdir()
    for name in (
        "velstand.onnx",
        "alpha_sitstand.onnx",
        "alpha_ground_pick.onnx",
        "ball_kick_left.onnx",
        "ball_kick_right.onnx",
        "roulade.onnx",
    ):
        (policies / name).write_bytes(b"policy")
    candidate = tmp_path / 'candidate"quoted.onnx'
    candidate.write_bytes(b"candidate")
    assert tomllib.loads(make_walk_config(policies, candidate))["policy"]["walk"] == str(candidate)


def pair_result(*, fall: bool = False, rate: float = 1.0) -> dict:
    scenarios = ("idle", "forward_stop")
    return {
        "baseline": {
            name: {
                "status": "completed",
                "initial_pose_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "sim_realtime_factor": rate,
                "samples": 25,
                "walk_reached": True,
                "ready_fallen": True,
                "command_applied": name == "forward_stop",
                "stop_confirmed": name == "forward_stop",
                "fallen": False,
                "limp": False,
                "min_trunk_z_m": 0.09,
            }
            for name in scenarios
        },
        "candidate": {
            name: {
                "status": "completed",
                "initial_pose_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "sim_realtime_factor": rate,
                "samples": 25,
                "walk_reached": True,
                "ready_fallen": True,
                "command_applied": name == "forward_stop",
                "stop_confirmed": name == "forward_stop",
                "fallen": fall and name == "idle",
                "limp": False,
                "min_trunk_z_m": 0.04 if fall and name == "idle" else 0.09,
            }
            for name in scenarios
        },
    }


def test_observed_robotd_sim_fall_rejects_candidate_without_hardware_claim():
    result = assess_robotd_pair(pair_result(fall=True))
    assert result["verdict"] == "rejected"
    assert "candidate_fell:idle" in result["reasons"]
    assert result["physical_hardware_gate"] == "not_run"


def test_clean_short_robotd_sim_pair_is_still_not_policy_approval():
    result = assess_robotd_pair(pair_result())
    assert result["verdict"] == "inconclusive"
    assert "coverage_too_small" in result["reasons"]


@pytest.mark.parametrize(
    "fault",
    [
        "slow",
        "unpaired",
        "empty_pose",
        "missing",
        "no_samples",
        "baseline_fell",
        "baseline_no_walk",
    ],
)
def test_unreliable_or_incomplete_robotd_sim_is_invalid_not_a_quality_pass(fault):
    report = pair_result()
    if fault == "slow":
        report["candidate"]["idle"]["sim_realtime_factor"] = 0.5
    elif fault == "unpaired":
        report["candidate"]["idle"]["initial_pose_sha256"] = "different"
    elif fault == "empty_pose":
        report["baseline"]["idle"]["initial_pose_sha256"] = ""
        report["candidate"]["idle"]["initial_pose_sha256"] = ""
    elif fault == "missing":
        del report["candidate"]["forward_stop"]
    elif fault == "baseline_fell":
        report["baseline"]["idle"]["fallen"] = True
    elif fault == "baseline_no_walk":
        report["baseline"]["idle"]["walk_reached"] = False
    else:
        report["candidate"]["idle"]["samples"] = 0
    result = assess_robotd_pair(report)
    assert result["verdict"] == "invalid"
    assert result["physical_hardware_gate"] == "not_run"


def test_candidate_that_never_reaches_walk_is_rejected_only_with_stable_baseline():
    report = pair_result()
    report["candidate"]["idle"]["walk_reached"] = False
    result = assess_robotd_pair(report)
    assert result["verdict"] == "rejected"
    assert "candidate_walk_not_reached:idle" in result["reasons"]


@pytest.mark.parametrize("field", ["command_applied", "stop_confirmed"])
def test_baseline_must_prove_command_and_stop_before_comparison(field):
    report = pair_result()
    report["baseline"]["forward_stop"][field] = False
    result = assess_robotd_pair(report)
    assert result["verdict"] == "invalid"
    assert "baseline_command_path_unverified" in result["reasons"]


@pytest.mark.parametrize("field", ["command_applied", "stop_confirmed"])
def test_candidate_that_ignores_move_or_stop_is_rejected(field):
    report = pair_result()
    report["candidate"]["forward_stop"][field] = False
    result = assess_robotd_pair(report)
    assert result["verdict"] == "rejected"
    assert f"candidate_{field}_failed" in result["reasons"]


@needs_evidence
def test_recorded_historical_robotd_pair_recomputes_rejection():
    evidence = Path(__file__).parents[1] / "docs/validation/evidence/robotd-pair-007-summary.json"
    report = json.loads(evidence.read_text())
    assert report["assessment"] == assess_robotd_pair(report["runs"])
    assert report["assessment"]["verdict"] == "rejected"
    raw = Path(__file__).parents[1] / report["raw_report"]
    if raw.exists():
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == report["raw_report_sha256"]
    assert report["physical_hardware_gate"] == "not_run"
    assert report["policy_deployment"] == "not_run"
    assert report["baseline_onnx_sha256"] != report["candidate_onnx_sha256"]


@needs_evidence
def test_first_walk_oracle_evidence_is_hash_bound_and_recomputable():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (root / "docs/validation/evidence/robotd-first-walk-013-summary.json").read_text()
    )
    assert summary["status"] == "candidate_rejected"
    assert summary["assessment"]["verdict"] == "rejected"
    assert summary["physical_hardware_gate"] == "not_run"
    assert summary["policy_deployment"] == "not_run"
    assert (
        summary["ort_lib_sha256"]
        == "49422a138fff12d7fcbd544373aefbffd510e473e08e18af28a8ecc3581029bf"
    )
    assert summary["baseline_onnx_sha256"] != summary["candidate_onnx_sha256"]
    for scenario in ("idle", "forward_stop"):
        pair = summary["scenarios"][scenario]
        assert pair["initial_pose_sha256"]
        for label in ("baseline", "candidate"):
            item = pair[label]
            truth = item["first_walk_truth"]
            assert truth["safety_fallen"] is True
            assert 0.035 < truth["trunk_z_m"] < 0.06
            assert truth["projected_gravity"][0] > 0.9
        assert pair["baseline"]["healthy_walk_reached"] is True
        assert pair["candidate"]["healthy_walk_reached"] is False
        assert pair["baseline"]["final_trunk_z_m"] > 0.10
        assert pair["candidate"]["final_trunk_z_m"] < 0.06
    assert summary["scenarios"]["forward_stop"]["baseline"]["command_applied"] is True
    assert summary["scenarios"]["forward_stop"]["baseline"]["stop_confirmed"] is True

    raw = root / summary["raw_report"]
    if raw.exists():
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary["raw_report_sha256"]
        report = json.loads(raw.read_text())
        assert report["assessment"] == summary["assessment"] == assess_robotd_pair(report["runs"])
        for name in (
            "duckmind_source_sha256",
            "ort_lib_sha256",
            "robotd_binary_sha256",
            "scene_sha256",
            "body_server_sha256",
            "policy_manifest_sha256",
            "baseline_onnx_sha256",
            "candidate_onnx_sha256",
        ):
            assert report[name] == summary[name]
        for scenario in ("idle", "forward_stop"):
            pair = summary["scenarios"][scenario]
            assert (
                pair["initial_pose_sha256"]
                == report["runs"]["baseline"][scenario]["initial_pose_sha256"]
                == report["runs"]["candidate"][scenario]["initial_pose_sha256"]
            )
            for label in ("baseline", "candidate"):
                source = report["runs"][label][scenario]
                item = pair[label]
                assert item["first_walk_truth"] == source["first_walk_truth"]
                assert item["healthy_walk_reached"] == source["walk_reached"]
                assert item["final_trunk_z_m"] == source["final_trunk_z_m"]
                assert item["command_applied"] == source["command_applied"]
                assert item["stop_confirmed"] == source["stop_confirmed"]
