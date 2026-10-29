import hashlib
import json

import pytest

from duckmind.local_train import LocalTrainingSpec, compare_evaluations, run_local_job


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timesteps": 0},
        {"timesteps": 257},
        {"threads": 0},
        {"threads": True},
        {"timeout_s": 0},
        {"seed": -1},
    ],
)
def test_local_resource_bounds_are_explicit(kwargs):
    with pytest.raises(ValueError):
        LocalTrainingSpec(**kwargs)


def test_evaluation_keeps_survival_separate_from_reward():
    baseline = [{"seed": 10001, "steps": 250, "return": 20.0, "fell": False}]
    candidate = [{"seed": 10001, "steps": 50, "return": 30.0, "fell": True}]
    result = compare_evaluations(baseline, candidate)
    assert result["reward_delta"] == 10.0
    assert result["survival_steps_delta"] == -200
    assert result["candidate_falls"] == 1
    assert result["non_regression"] is False
    assert result["hardware_approved"] is False


def test_comparison_rejects_unpaired_seeds():
    with pytest.raises(ValueError, match="paired"):
        compare_evaluations(
            [{"seed": 1, "steps": 1, "return": 1, "fell": False}],
            [{"seed": 2, "steps": 1, "return": 1, "fell": False}],
        )


def test_local_job_refuses_overwrite_before_any_training(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "keep").write_text("owner data")
    with pytest.raises(ValueError, match="exists"):
        run_local_job(tmp_path, job, LocalTrainingSpec())
    assert (job / "keep").read_text() == "owner data"


def test_owned_worker_timeout_never_becomes_success(tmp_path, monkeypatch):
    from duckmind.training import PhaseResult

    monkeypatch.setattr("duckmind.local_train.verify_upstream", lambda _: {"commit": "fixture"})
    monkeypatch.setattr("duckmind.local_train.run_phase", lambda *a, **k: PhaseResult(-9, True, 1))
    job = tmp_path / "new"
    report = run_local_job(tmp_path / "upstream", job, LocalTrainingSpec())
    assert report["status"] == "timed_out"
    assert report["physical_hardware_gate"] == "not_run"
    assert json.loads((job / "report.json").read_text())["status"] == "timed_out"


def test_worker_success_needs_persisted_model_evidence(tmp_path, monkeypatch):
    from duckmind.training import PhaseResult

    monkeypatch.setattr("duckmind.local_train.verify_upstream", lambda _: {"commit": "fixture"})
    monkeypatch.setattr("duckmind.local_train.run_phase", lambda *a, **k: PhaseResult(0, False, 1))
    report = run_local_job(tmp_path / "upstream", tmp_path / "new", LocalTrainingSpec())
    assert report["status"] == "failed"


def test_worker_launch_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("duckmind.local_train.verify_upstream", lambda _: {"commit": "fixture"})

    def cannot_start(*args, **kwargs):
        raise OSError("test process launch failure")

    monkeypatch.setattr("duckmind.local_train.run_phase", cannot_start)
    job = tmp_path / "new"
    report = run_local_job(tmp_path / "upstream", job, LocalTrainingSpec())
    assert report["status"] == "failed"
    assert "OSError" in report["error"]
    assert json.loads((job / "report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("expected,actual", [([float("nan")], [0.0]), ([0.0], [float("inf")])])
def test_export_parity_refuses_nonfinite_on_either_side(expected, actual):
    from duckmind.local_train import parity_error

    with pytest.raises(ValueError, match="finite"):
        parity_error(expected, actual)


def test_cleanup_attempts_all_envs_even_if_one_close_fails():
    from duckmind.local_train import close_environments

    closed = []

    class Env:
        def __init__(self, fails):
            self.fails = fails

        def close(self):
            closed.append(self.fails)
            if self.fails:
                raise RuntimeError("test close failure")

    errors = close_environments((Env(True), Env(False)))
    assert closed == [True, False]
    assert len(errors) == 1


def test_export_clips_actions_and_matches_predict_when_optional_stack_installed(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from duckmind.local_train import export_actor

    class Actor(torch.nn.Module):
        def forward(self, obs, deterministic=True):
            actions = obs[:, :14] * 10
            return actions, obs[:, :1], obs[:, 0]

    path = tmp_path / "actor.onnx"
    export_actor(Actor(), path, action_limit=0.35)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.ones((1, 61), dtype=np.float32)
    output = session.run(None, {"obs": obs})
    assert len(output) == 1
    assert output[0].shape == (1, 14)
    np.testing.assert_allclose(output[0], 0.35)
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
