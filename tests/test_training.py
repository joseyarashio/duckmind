import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from duckmind.policy_check import CONTRACT_COMMIT
from duckmind.training import (
    PhaseResult,
    TrainingSpec,
    child_environment,
    environment_reasons,
    make_plan,
    probe_environment,
    run_job,
    run_phase,
    verify_upstream,
)
from tests.checkout_parts import needs_evidence


def ready_environment(upstream):
    return {
        "python": [3, 12, 13],
        "platform": "linux",
        "packages": {
            "mjlab": "1.3.0",
            "warp-lang": "1.12.0",
            "torch": "2.9.1+cu129",
            "rsl-rl-lib": "5.0.1",
            "mjlab-microduck": "0.1.0",
        },
        "task_entrypoint": True,
        "cuda_available": True,
        "cuda_devices": 1,
        "cpu_available": True,
        "module_path": str(upstream / "src/mjlab_microduck/__init__.py"),
    }


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    package = root / "src/mjlab_microduck"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (root / "uv.lock").write_text("fixture lock\n")
    (root / "scripts").mkdir()
    (root / "scripts/export.py").write_text("# fixture\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr("duckmind.training.UPSTREAM_COMMIT", commit)
    return root


def test_plan_is_explicit_offline_and_keeps_outputs_out_of_upstream(upstream, tmp_path):
    job = tmp_path / "new-job"
    plan = make_plan(upstream, Path(sys.executable), job, TrainingSpec())
    command = plan["train_command"]
    assert command[1:4] == ["-m", "mjlab.scripts.train", "Mjlab-Velocity-Flat-MicroDuck"]
    assert command[command.index("--gpu-ids") + 1] == "[0]"
    assert command[command.index("--agent.logger") + 1] == "tensorboard"
    assert command[command.index("--agent.upload-model") + 1] == "False"
    assert command[command.index("--agent.resume") + 1] == "False"
    assert command[command.index("--agent.experiment-name") + 1] == str(job / "checkpoints")
    assert "--hf-jobs" not in command
    assert plan["physical_hardware_gate"] == "not_run"
    assert not job.exists()


def test_cpu_plan_uses_official_none_gpu_selection_and_labels_experimental(upstream, tmp_path):
    spec = TrainingSpec(num_envs=2, max_iterations=1, device="cpu")
    plan = make_plan(upstream, Path(sys.executable), tmp_path / "cpu-job", spec)
    command = plan["train_command"]
    assert command[command.index("--gpu-ids") + 1] == "None"
    assert plan["spec"]["device"] == "cpu"
    assert plan["profile"] == "official-offline-velocity-cpu-experimental-v1"
    assert plan["experimental"] is True
    assert plan["cpu_thread_limit"] == 2
    assert plan["cuda_quality_equivalence"] == "not_claimed"
    assert plan["physical_hardware_gate"] == "not_run"


def test_push_interval_experiment_is_explicit_and_source_bound(upstream, tmp_path):
    spec = TrainingSpec(
        num_envs=32,
        max_iterations=500,
        seed=11,
        device="cpu",
        push_interval_s=(1.0, 2.0),
    )
    plan = make_plan(upstream, Path(sys.executable), tmp_path / "push-job", spec)
    command = plan["train_command"]
    option = "--env.events.push-robot.interval-range-s"

    assert command[command.index(option) + 1] == "(1.0, 2.0)"
    assert plan["spec"]["push_interval_s"] == (1.0, 2.0)
    assert plan["profile"] == "official-offline-velocity-cpu-push-interval-experimental-v1"
    assert plan["experimental"] is True
    assert plan["physical_hardware_gate"] == "not_run"


@pytest.mark.parametrize(
    "interval",
    [
        (0.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (True, 1.0),
        (10**1000, 10**1000),
    ],
)
def test_push_interval_experiment_rejects_invalid_schedule(interval):
    with pytest.raises(ValueError, match="push_interval_s"):
        TrainingSpec(push_interval_s=interval)


@needs_evidence
def test_recorded_push_continuation_remains_rejected_and_hash_bound():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (
            root / "docs/validation/evidence/mac-official-push-resume-3500-005-summary.json"
        ).read_text()
    )
    assert summary["status"] == "candidate_rejected"
    assert summary["physical_hardware_gate"] == "not_run"
    assert summary["policy_deployment"] == "not_run"
    assert summary["training"]["training_push_interval_s"] == [1.0, 2.0]
    assert summary["training"]["formal_play_push_interval_s"] == [0.5, 1.0]
    assert summary["formal_play_physics"]["trial_steps"] == [161, 99, 70, 65]
    assert summary["formal_play_physics"]["falls"] == 4
    assert summary["robotd_comparison"]["baseline_walk_reached"] == [True, True]
    assert summary["robotd_comparison"]["candidate_walk_reached"] == [False, False]
    for section, path_key, hash_key in (
        ("training", "job_path", "job_sha256"),
        ("training", "official_env_yaml_path", "official_env_yaml_sha256"),
        ("formal_play_physics", "report_path", "report_sha256"),
        ("robotd_comparison", "report_path", "report_sha256"),
    ):
        record = summary[section]
        raw = root / record[path_key]
        if raw.exists():
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[hash_key]
    raw_job = root / summary["training"]["job_path"]
    if raw_job.exists():
        job = json.loads(raw_job.read_text())
        assert job["status"] == "candidate_rejected"
        assert job["qualification"]["verdict"] == "rejected"
        assert job["artifacts"]["checkpoint"]["sha256"] == summary["model"]["checkpoint_sha256"]
        assert job["artifacts"]["onnx"]["sha256"] == summary["model"]["onnx_sha256"]
        option = "--env.events.push-robot.interval-range-s"
        command = job["train_command"]
        assert command[command.index(option) + 1] == "(1.0, 2.0)"
    raw_environment = root / summary["training"]["official_env_yaml_path"]
    if raw_environment.exists():
        push_config = (
            raw_environment.read_text()
            .split("  push_robot:\n", 1)[1]
            .split("  foot_friction:\n", 1)[0]
        )
        assert "interval_range_s: !!python/tuple\n    - 1.0\n    - 2.0" in push_config


@needs_evidence
def test_recorded_push_continuation_zero_push_pair_is_diagnostic_and_recomputable():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (
            root / "docs/validation/evidence/mac-official-push-resume-3500-005-summary.json"
        ).read_text()
    )
    diagnostic = summary["play_zero_push_diagnostic"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["push_event_enabled"] is True
    assert diagnostic["push_velocity_zeroed"] is True
    assert diagnostic["initial_actor_matches_formal"] is True
    assert diagnostic["compiled_model_matches_formal"] is True
    assert diagnostic["first_action_and_command_match_formal"] is True
    assert diagnostic["first_action_divergence_steps"] == [48, 33, 43, 39]
    assert diagnostic["parity_samples"] == 1000
    assert diagnostic["parity_failed_samples"] == 0
    assert diagnostic["falls"] == 0
    assert diagnostic["trial_steps"] == [250, 250, 250, 250]
    assert diagnostic["qualification"] == "not_applicable"
    assert summary["formal_play_physics"]["falls"] == 4

    for name, digest_name in (("report_path", "report_sha256"), ("log_path", "log_sha256")):
        raw = root / diagnostic[name]
        if raw.exists():
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == diagnostic[digest_name]

    formal_path = root / summary["formal_play_physics"]["report_path"]
    zero_path = root / diagnostic["report_path"]
    if formal_path.exists() and zero_path.exists():
        formal, zero = json.loads(formal_path.read_text()), json.loads(zero_path.read_text())
        assert formal["status"] == zero["status"] == "completed"
        assert formal["evaluation_profile"] == "play"
        assert zero["evaluation_profile"] == "play-zero-push"
        assert zero["physical_hardware_gate"] == "not_run"
        assert zero["probe_script_sha256"] == diagnostic["probe_script_sha256"]
        assert formal["upstream"] == zero["upstream"]
        for model in ("checkpoint", "onnx"):
            assert formal["artifacts"][model]["sha256"] == zero["artifacts"][model]["sha256"]
        assert zero["parity"]["samples"] == diagnostic["parity_samples"]
        assert zero["parity"]["failed_samples"] == diagnostic["parity_failed_samples"]
        assert zero["candidate_summary"]["falls"] == diagnostic["falls"]
        for index, (pushed, unpushed) in enumerate(
            zip(formal["candidate_episodes"], zero["candidate_episodes"], strict=True)
        ):
            assert pushed["seed"] == unpushed["seed"]
            assert pushed["initial_actor_sha256"] == unpushed["initial_actor_sha256"]
            assert pushed["compiled_model_sha256"] == unpushed["compiled_model_sha256"]
            assert pushed["trace"][0]["raw_actions"] == unpushed["trace"][0]["raw_actions"]
            assert (
                pushed["trace"][0]["commands_before_step"]
                == unpushed["trace"][0]["commands_before_step"]
            )
            divergence = next(
                (
                    step + 1
                    for step, (left, right) in enumerate(
                        zip(pushed["trace"], unpushed["trace"], strict=False)
                    )
                    if left["raw_actions"] != right["raw_actions"]
                ),
                None,
            )
            assert divergence == diagnostic["first_action_divergence_steps"][index]
            assert pushed["fell_over"] is True
            assert unpushed["fell_over"] is False
            assert unpushed["steps"] == diagnostic["trial_steps"][index]


@needs_evidence
def test_recorded_robotd_readiness_distinguishes_policy_entry_from_healthy_walk():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (
            root / "docs/validation/evidence/mac-official-push-resume-3500-005-summary.json"
        ).read_text()
    )
    readiness = summary["robotd_comparison"]["readiness"]
    for scenario in ("idle", "forward_stop"):
        baseline, candidate = readiness[scenario]["baseline"], readiness[scenario]["candidate"]
        assert baseline["first_healthy_walk_s"] is not None
        assert baseline["first_healthy_walk_s"] < 3.0
        assert baseline["last_ready_fallen"] is False
        assert 2.0 < candidate["first_policy_walk_s"] < 3.0
        assert candidate["first_policy_walk_fallen"] is True
        assert candidate["first_healthy_walk_s"] is None
        assert candidate["last_ready_time_s"] > 12.0
        assert candidate["last_ready_fallen"] is True

    raw = root / summary["robotd_comparison"]["report_path"]
    if raw.exists():
        report = json.loads(raw.read_text())
        assert report["assessment"]["verdict"] == "rejected"
        for scenario in ("idle", "forward_stop"):
            assert (
                report["runs"]["baseline"][scenario]["initial_pose_sha256"]
                == report["runs"]["candidate"][scenario]["initial_pose_sha256"]
            )
            for label in ("baseline", "candidate"):
                states = report["runs"][label][scenario]["ready_states"]
                first_walk = next(state for state in states if state["policy"] == "walk")
                first_healthy = next(
                    (
                        state["robot_time_s"]
                        for state in states
                        if state["policy"] == "walk" and not state["fallen"]
                    ),
                    None,
                )
                assert readiness[scenario][label] == {
                    "first_policy_walk_s": first_walk["robot_time_s"],
                    "first_policy_walk_fallen": first_walk["fallen"],
                    "first_healthy_walk_s": first_healthy,
                    "last_ready_time_s": states[-1]["robot_time_s"],
                    "last_ready_fallen": states[-1]["fallen"],
                }


@pytest.fixture
def prior_official_job(upstream, tmp_path):
    parent = tmp_path / "prior-job"
    checkpoint = parent / "checkpoints/run/model_1999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted local fixture checkpoint")
    (parent / "job.json").write_text(
        json.dumps(
            {
                "status": "candidate_rejected",
                "task_id": "Mjlab-Velocity-Flat-MicroDuck",
                "duckmind_source_sha256": "a" * 64,
                "spec": {"device": "cpu", "num_envs": 32, "seed": 11},
                "upstream_identity": verify_upstream(upstream),
                "training_gate": "checkpoint_produced",
                "export_gate": "exported",
                "compatibility_gate": "pass",
                "simulator_gate": "failed",
                "qualification": {"verdict": "rejected", "physical_hardware_gate": "not_run"},
                "artifacts": {
                    "checkpoint": {
                        "path": "checkpoints/run/model_1999.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "bytes": checkpoint.stat().st_size,
                    }
                },
            }
        )
    )
    return parent


def test_resume_plan_selects_only_exact_local_checkpoint(upstream, tmp_path, prior_official_job):
    spec = TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu")
    plan = make_plan(
        upstream, Path(sys.executable), tmp_path / "next-job", spec, resume_job=prior_official_job
    )
    command = plan["train_command"]
    assert command[command.index("--agent.resume") + 1] == "True"
    assert command[command.index("--agent.load-run") + 1] == "^resume-source$"
    assert command[command.index("--agent.load-checkpoint") + 1] == r"^model_1999\.pt$"
    assert command[command.index("--agent.max-iterations") + 1] == "2"
    assert (
        plan["resume_source"]["checkpoint_sha256"]
        == hashlib.sha256(
            (prior_official_job / "checkpoints/run/model_1999.pt").read_bytes()
        ).hexdigest()
    )
    assert not (tmp_path / "next-job").exists(), "planning must be read-only"


def test_resume_accepts_inconclusive_parent_for_more_training(
    upstream, tmp_path, prior_official_job
):
    report_path = prior_official_job / "job.json"
    report = json.loads(report_path.read_text())
    report.update(status="candidate_unqualified", simulator_gate="inconclusive")
    report["qualification"]["verdict"] = "inconclusive"
    report_path.write_text(json.dumps(report))

    plan = make_plan(
        upstream,
        Path(sys.executable),
        tmp_path / "next-job",
        TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
        resume_job=prior_official_job,
    )
    assert plan["resume_source"]["parent_status"] == "candidate_unqualified"


@pytest.mark.parametrize(
    "fault",
    ["changed_bytes", "changed_upstream", "unvalidated", "wrong_seed", "forged_simulator_gate"],
)
def test_resume_rejects_untrusted_or_incompatible_parent(
    upstream, tmp_path, prior_official_job, fault
):
    report_path = prior_official_job / "job.json"
    report = json.loads(report_path.read_text())
    if fault == "changed_bytes":
        (prior_official_job / "checkpoints/run/model_1999.pt").write_bytes(b"altered")
    elif fault == "changed_upstream":
        report["upstream_identity"]["uv_lock_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
    elif fault == "unvalidated":
        report["status"] = "training_failed"
        report_path.write_text(json.dumps(report))
    elif fault == "forged_simulator_gate":
        report["simulator_gate"] = "pass"
        report_path.write_text(json.dumps(report))
    else:
        report["spec"]["seed"] = 12
        report_path.write_text(json.dumps(report))
    next_job = tmp_path / "next-job"

    with pytest.raises(ValueError, match="resume"):
        make_plan(
            upstream,
            Path(sys.executable),
            next_job,
            TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
            resume_job=prior_official_job,
        )
    assert not next_job.exists()


def test_resumed_job_requires_new_checkpoint_not_copied_parent(
    upstream, tmp_path, prior_official_job, monkeypatch
):
    monkeypatch.setattr(
        "duckmind.training.probe_environment",
        lambda *_, **__: {
            **ready_environment(upstream),
            "platform": "darwin",
            "cuda_available": False,
        },
    )

    def fake_phase(command, *, cwd, env, log, timeout_s):
        log.write_text("fake training produced no new checkpoint\n")
        return PhaseResult(0, False, 0.01)

    monkeypatch.setattr("duckmind.training.run_phase", fake_phase)
    next_job = tmp_path / "next-job"
    result = run_job(
        upstream,
        Path(sys.executable),
        next_job,
        TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
        execute=True,
        resume_job=prior_official_job,
    )
    assert result["status"] == "training_failed"
    assert result["reasons"] == ["no_checkpoint_produced"]
    assert (
        result["resume_source"]["staged_checkpoint"]["sha256"]
        == result["resume_source"]["checkpoint_sha256"]
    )


def test_resumed_job_rejects_new_filename_with_unchanged_parent_bytes(
    upstream, tmp_path, prior_official_job, monkeypatch
):
    monkeypatch.setattr(
        "duckmind.training.probe_environment",
        lambda *_, **__: {
            **ready_environment(upstream),
            "platform": "darwin",
            "cuda_available": False,
        },
    )
    next_job = tmp_path / "next-job"

    def fake_phase(command, *, cwd, env, log, timeout_s):
        if log.stem != "train":
            pytest.fail("unchanged resumed checkpoint must not be exported")
        checkpoint = next_job / "checkpoints/new-run/model_2000.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes((prior_official_job / "checkpoints/run/model_1999.pt").read_bytes())
        log.write_text("copied parent bytes under a new filename\n")
        return PhaseResult(0, False, 0.01)

    monkeypatch.setattr("duckmind.training.run_phase", fake_phase)
    result = run_job(
        upstream,
        Path(sys.executable),
        next_job,
        TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
        execute=True,
        resume_job=prior_official_job,
    )
    assert result["status"] == "training_failed"
    assert result["reasons"] == ["resume_checkpoint_unchanged"]


def test_resumed_job_detects_parent_mutation_during_training(
    upstream, tmp_path, prior_official_job, monkeypatch
):
    monkeypatch.setattr(
        "duckmind.training.probe_environment",
        lambda *_, **__: {
            **ready_environment(upstream),
            "platform": "darwin",
            "cuda_available": False,
        },
    )

    def fake_phase(command, *, cwd, env, log, timeout_s):
        (prior_official_job / "checkpoints/run/model_1999.pt").write_bytes(b"changed in-flight")
        log.write_text("parent mutated during training\n")
        return PhaseResult(0, False, 0.01)

    monkeypatch.setattr("duckmind.training.run_phase", fake_phase)
    result = run_job(
        upstream,
        Path(sys.executable),
        tmp_path / "next-job",
        TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
        execute=True,
        resume_job=prior_official_job,
    )
    assert result["status"] == "failed"
    assert "resume source changed" in result["error"]
    assert result["training_gate"] != "checkpoint_produced"


def test_resumed_job_exports_new_checkpoint_and_preserves_parent(
    upstream, tmp_path, prior_official_job, monkeypatch
):
    monkeypatch.setattr(
        "duckmind.training.probe_environment",
        lambda *_, **__: {
            **ready_environment(upstream),
            "platform": "darwin",
            "cuda_available": False,
        },
    )
    next_job = tmp_path / "next-job"
    parent_checkpoint = prior_official_job / "checkpoints/run/model_1999.pt"
    original_bytes = parent_checkpoint.read_bytes()

    def fake_phase(command, *, cwd, env, log, timeout_s):
        if log.stem == "train":
            checkpoint = next_job / "checkpoints/new-run/model_2000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"new local training checkpoint")
            log.write_text("new checkpoint produced\n")
            return PhaseResult(0, False, 0.01)
        assert log.stem == "export"
        assert command[command.index("--checkpoint-file") + 1] == str(
            next_job / "checkpoints/new-run/model_2000.pt"
        )
        log.write_text("export deliberately stopped by fixture\n")
        return PhaseResult(1, False, 0.01)

    monkeypatch.setattr("duckmind.training.run_phase", fake_phase)
    result = run_job(
        upstream,
        Path(sys.executable),
        next_job,
        TrainingSpec(num_envs=32, max_iterations=2, seed=11, device="cpu"),
        execute=True,
        resume_job=prior_official_job,
    )
    assert result["status"] == "export_failed"
    assert result["artifacts"]["checkpoint"]["path"] == "checkpoints/new-run/model_2000.pt"
    assert parent_checkpoint.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "kwargs",
    [{"num_envs": 0}, {"max_iterations": True}, {"seed": -1}, {"timeout_s": 0}, {"device": "mps"}],
)
def test_invalid_resource_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingSpec(**kwargs)


def test_environment_blocks_cpu_and_wrong_package_without_claiming_training(upstream):
    report = ready_environment(upstream)
    report.update(platform="darwin", cuda_available=False, cuda_devices=0)
    report["packages"]["mjlab"] = "2.0.0"
    reasons = environment_reasons(report, upstream)
    assert "linux_cuda_profile_required" in reasons
    assert "cuda_unavailable" in reasons
    assert "version_mismatch:mjlab" in reasons


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_explicit_cpu_profile_accepts_mac_or_linux_without_cuda(upstream, platform):
    report = ready_environment(upstream)
    report.update(platform=platform, cuda_available=False, cuda_devices=0)
    assert environment_reasons(report, upstream, device="cpu") == []


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("cpu", "cpu_unavailable"),
        ("version", "version_mismatch:torch"),
        ("entrypoint", "missing_task_entrypoint"),
        ("source", "wrong_microduck_import"),
        ("platform", "macos_linux_cpu_profile_required"),
    ],
)
def test_cpu_profile_preserves_strict_environment_gates(upstream, fault, expected):
    report = ready_environment(upstream)
    report.update(platform="darwin", cuda_available=False, cuda_devices=0)
    if fault == "cpu":
        report["cpu_available"] = False
    elif fault == "version":
        report["packages"]["torch"] = "2.10.0"
    elif fault == "entrypoint":
        report["task_entrypoint"] = False
    elif fault == "source":
        report["module_path"] = "/unrelated/mjlab_microduck/__init__.py"
    else:
        report["platform"] = "win32"
    assert expected in environment_reasons(report, upstream, device="cpu")


def test_cpu_probe_is_device_explicit_and_disables_inherited_cuda(upstream, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(variable, "64")
    monkeypatch.setattr(
        "duckmind.training.ENVIRONMENT_PROBE",
        "import json,os,sys; print(json.dumps({'device': sys.argv[1], "
        "'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'threads': [os.environ.get(key) for key in "
        "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')]}))",
    )
    report = probe_environment(Path(sys.executable), upstream, device="cpu", timeout_s=3)
    assert report == {"device": "cpu", "cuda_visible": "", "threads": ["2", "2", "2"]}


def test_environment_rejects_wrong_imported_checkout(upstream):
    report = ready_environment(upstream)
    report["module_path"] = "/unrelated/mjlab_microduck/__init__.py"
    assert "wrong_microduck_import" in environment_reasons(report, upstream)
    assert not environment_reasons(ready_environment(upstream), upstream)


def test_preflight_requires_task_registration_and_reviewed_runner(upstream):
    report = ready_environment(upstream)
    report["task_entrypoint"] = False
    report["packages"]["rsl-rl-lib"] = "6.0.0"
    assert "missing_task_entrypoint" in environment_reasons(report, upstream)
    assert "version_mismatch:rsl-rl-lib" in environment_reasons(report, upstream)


def test_child_environment_disables_uploads_and_does_not_inherit_pythonpath(upstream, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    monkeypatch.setenv("WANDB_API_KEY", "test-value-not-a-real-key")
    monkeypatch.setenv("HF_TOKEN", "test-value-not-a-real-key")
    environment = child_environment(upstream)
    assert environment["PYTHONPATH"] == str(upstream / "src")
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert "WANDB_API_KEY" not in environment
    assert "HF_TOKEN" not in environment


def test_single_gpu_environment_removes_inherited_distributed_rendezvous(upstream, monkeypatch):
    inherited = {
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "example.invalid",
        "MASTER_PORT": "29500",
        "GROUP_RANK": "1",
        "ROLE_RANK": "1",
        "ROLE_WORLD_SIZE": "2",
        "TORCHELASTIC_RUN_ID": "parent-job",
        "TORCHELASTIC_RESTART_COUNT": "2",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    environment = child_environment(upstream)
    retained = sorted(inherited.keys() & environment.keys())
    assert retained == []
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"


def test_source_pin_and_dirty_worktree_are_fail_closed(upstream):
    assert verify_upstream(upstream)["commit"]
    (upstream / "scripts/export.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="dirty"):
        verify_upstream(upstream)


def test_new_job_needs_explicit_execution_and_never_overwrites(upstream, tmp_path):
    job = tmp_path / "job"
    with pytest.raises(ValueError, match="explicit"):
        run_job(upstream, Path(sys.executable), job, TrainingSpec(), execute=False)
    assert not job.exists()
    job.mkdir()
    (job / "owner-data").write_text("keep")
    with pytest.raises(ValueError, match="exists"):
        run_job(upstream, Path(sys.executable), job, TrainingSpec(), execute=True)
    assert (job / "owner-data").read_text() == "keep"


def test_blocked_preflight_persists_honest_report_without_launching(
    upstream, tmp_path, monkeypatch
):
    report = ready_environment(upstream)
    report.update(cuda_available=False, cuda_devices=0)
    monkeypatch.setattr("duckmind.training.probe_environment", lambda *_, **__: report)
    monkeypatch.setattr(
        "duckmind.training.run_phase", lambda *a, **k: pytest.fail("must not train")
    )
    job = tmp_path / "job"
    result = run_job(upstream, Path(sys.executable), job, TrainingSpec(), execute=True)
    assert result["status"] == "blocked_preflight"
    assert result["training_gate"] == "not_run"
    assert result["physical_hardware_gate"] == "not_run"
    assert json.loads((job / "job.json").read_text())["status"] == "blocked_preflight"


@pytest.mark.parametrize(
    "fault",
    [
        "train_exit",
        "no_checkpoint",
        "export_exit",
        "inspect_exit",
        "wrong_model_sha256",
        "wrong_contract_commit",
        "wrong_profile",
        "not_compatibility_only",
        "claims_hardware_pass",
        "simulation_exit",
        "simulation_missing_report",
        "simulation_wrong_model_sha256",
        "fall_observed",
        None,
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_pipeline_gates_and_artifact_provenance(upstream, tmp_path, monkeypatch, fault, device):
    from duckmind.training import PhaseResult

    environment = ready_environment(upstream)
    if device == "cpu":
        environment.update(platform="darwin", cuda_available=False, cuda_devices=0)

    def fake_probe(*_, **kwargs):
        assert kwargs["device"] == device
        return environment

    monkeypatch.setattr("duckmind.training.probe_environment", fake_probe)
    job = tmp_path / "job"
    phases = []
    inspection_commands = []

    def fake_phase(command, *, cwd, env, log, timeout_s):
        phase = log.stem
        phases.append(phase)
        log.write_text(f"isolated orchestration fixture: {phase}\n")
        if phase == "train":
            if fault not in {"train_exit", "no_checkpoint"}:
                checkpoints = job / "checkpoints/run"
                checkpoints.mkdir(parents=True)
                (checkpoints / "model_4.pt").write_bytes(b"test checkpoint, not a trained model")
            return PhaseResult(1 if fault == "train_exit" else 0, False, 0.01)
        if phase == "export":
            assert "--checkpoint-file" in command
            assert command[command.index("--checkpoint-file") + 1].endswith("model_4.pt")
            assert "--wandb-run-path" not in command
            assert command[command.index("--device") + 1] == (
                "cpu" if device == "cpu" else "cuda:0"
            )
            if fault != "export_exit":
                (job / "artifacts/policy.onnx").write_bytes(b"test artifact, not an ONNX graph")
            return PhaseResult(1 if fault == "export_exit" else 0, False, 0.01)
        if phase == "inspect":
            inspection_commands.append(command)
            inspection = {
                "passed": fault != "inspect_exit",
                "compatibility_only": True,
                "physical_hardware_gate": "not_run",
                "model_sha256": hashlib.sha256(
                    (job / "artifacts/policy.onnx").read_bytes()
                ).hexdigest(),
                "contract_commit": CONTRACT_COMMIT,
                "profile": "microduck-feedforward-api1",
            }
            inspection.update(
                {
                    "wrong_model_sha256": {"model_sha256": "0" * 64},
                    "wrong_contract_commit": {"contract_commit": "0" * 40},
                    "wrong_profile": {"profile": "unsupported-api2"},
                    "not_compatibility_only": {"compatibility_only": False},
                    "claims_hardware_pass": {"physical_hardware_gate": "pass"},
                }.get(fault, {})
            )
            (job / "artifacts/compatibility.json").write_text(json.dumps(inspection))
            return PhaseResult(1 if fault == "inspect_exit" else 0, False, 0.01)
        assert phase == "simulation"
        assert command[1].endswith("offline_eval.py")
        assert command[command.index("--checkpoint") + 1].endswith("model_4.pt")
        assert command[command.index("--onnx") + 1].endswith("policy.onnx")
        if fault == "simulation_exit":
            return PhaseResult(1, False, 0.01)
        if fault == "simulation_missing_report":
            return PhaseResult(0, False, 0.01)
        episodes = [
            {
                "seed": seed,
                "steps": 50,
                "fell_over": fault == "fall_observed" and seed == 10001,
                "terminated": fault == "fall_observed" and seed == 10001,
                "truncated": False,
                "all_finite": True,
                "min_trunk_height_m": 0.09,
            }
            for seed in range(10001, 10005)
        ]
        from duckmind.training import verify_upstream

        source = verify_upstream(upstream)
        report = {
            "schema_version": 1,
            "status": "completed",
            "task_id": "Mjlab-Velocity-Flat-MicroDuck",
            "horizon": 250,
            "device": "cpu",
            "play": True,
            "auto_reset": False,
            "num_envs": 1,
            "physical_hardware_gate": "not_run",
            "upstream": {
                "commit": source["commit"],
                "uv_lock_sha256": source["uv_lock_sha256"],
            },
            "artifacts": {
                "checkpoint": {
                    "sha256": hashlib.sha256(
                        (job / "checkpoints/run/model_4.pt").read_bytes()
                    ).hexdigest()
                },
                "onnx": {
                    "sha256": hashlib.sha256(
                        (job / "artifacts/policy.onnx").read_bytes()
                    ).hexdigest()
                },
            },
            "probe_script_sha256": hashlib.sha256(
                (Path(__file__).parents[1] / "src/duckmind/offline_eval.py").read_bytes()
            ).hexdigest(),
            "seeds": [item["seed"] for item in episodes],
            "candidate_episodes": episodes,
            "candidate_summary": {
                "episodes": 4,
                "falls": sum(item["fell_over"] for item in episodes),
                "total_steps": 200,
            },
            "parity": {
                "passed": True,
                "all_finite": True,
                "failed_samples": 0,
                "samples": 200,
                "max_abs_error": 0.0,
            },
        }
        if fault == "simulation_wrong_model_sha256":
            report["artifacts"]["onnx"]["sha256"] = "wrong"
        (job / "artifacts/simulation.json").write_text(json.dumps(report))
        return PhaseResult(0, False, 0.01)

    monkeypatch.setattr("duckmind.training.run_phase", fake_phase)
    result = run_job(
        upstream, Path(sys.executable), job, TrainingSpec(seed=17, device=device), execute=True
    )
    assert result["physical_hardware_gate"] == "not_run"
    if fault in {"simulation_exit", "simulation_missing_report", "simulation_wrong_model_sha256"}:
        assert result["simulator_gate"] == "failed"
    elif fault == "fall_observed":
        assert result["simulator_gate"] == "failed"
    elif fault is None:
        assert result["simulator_gate"] == "inconclusive"
    else:
        assert result["simulator_gate"] == "not_run"
    assert not (upstream / "logs").exists()
    if fault in {"train_exit", "no_checkpoint"}:
        assert phases == ["train"]
        assert result["status"] == "training_failed"
    elif fault == "export_exit":
        assert phases == ["train", "export"]
        assert result["status"] == "export_failed"
    elif fault in {"simulation_exit", "simulation_missing_report", "simulation_wrong_model_sha256"}:
        assert phases == ["train", "export", "inspect", "simulation"]
        assert result["status"] == "simulation_failed"
    elif fault == "fall_observed":
        assert phases == ["train", "export", "inspect", "simulation"]
        assert result["status"] == "candidate_rejected"
        assert result["qualification"]["fall_count"] == 1
    elif fault is not None:
        assert phases == ["train", "export", "inspect"]
        assert result["status"] == "compatibility_failed"
        assert result["compatibility_gate"] == "failed"
        assert "compatibility" not in result["artifacts"]
    else:
        assert result["status"] == "candidate_unqualified"
        assert phases == ["train", "export", "inspect", "simulation"]
        assert len(result["artifacts"]["checkpoint"]["sha256"]) == 64
        assert len(result["artifacts"]["onnx"]["sha256"]) == 64
        assert result["normalizer_provenance"] == "official_checkpoint_export"
        command = inspection_commands[0]
        assert command[command.index("--command-profile") + 1] == "locomotion"
        assert command[command.index("--steps") + 1] == "50"
        assert command[command.index("--seed") + 1] == "17"


def test_cli_plan_accepts_explicit_cpu_profile(upstream, tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "duckmind.training",
            "plan",
            "--upstream",
            str(upstream),
            "--python",
            sys.executable,
            "--job",
            str(tmp_path / "cli-job"),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["spec"]["device"] == "cpu"


def test_phase_timeout_reaps_owned_process_and_keeps_log(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import os,time; print(os.getpid(),flush=True); time.sleep(30)",
    ]
    log = tmp_path / "phase.log"
    result = run_phase(command, cwd=tmp_path, env=dict(os.environ), log=log, timeout_s=0.4)
    assert result.timed_out is True
    assert result.returncode != 0
    pid = int(log.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_sigterm_to_phase_owner_does_not_leave_training_child_running(tmp_path):
    log = tmp_path / "training.log"
    script = (
        "import os,sys; from pathlib import Path; from duckmind.training import run_phase; "
        "run_phase([sys.executable,'-c','import os,time; print(os.getpid(),flush=True); time.sleep(30)'],"
        "cwd=Path(sys.argv[1]),env=dict(os.environ),log=Path(sys.argv[2]),timeout_s=30)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(log)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and log.read_text().strip():
                child_pid = int(log.read_text().strip())
                break
            time.sleep(0.01)
        assert child_pid is not None
        parent.terminate()
        parent.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
