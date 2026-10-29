import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from duckmind import policy_check
from duckmind.policy_check import (
    MAX_PROBE_STEPS,
    TensorOutput,
    TensorSpec,
    inspect_policy,
)


class FakeProbe:
    def __init__(self, *, inputs=None, outputs=None, result=None):
        self.inputs = inputs or (TensorSpec("obs", "float32", (1, 61)),)
        self.outputs = outputs or (TensorSpec("actor_output", "float32", (1, 14)),)
        self.result = result
        self.observations = []
        self.returned = []
        self.closed = False

    def run(self, observation):
        self.observations.append(observation)
        output = (
            self.result(observation)
            if self.result is not None
            else TensorOutput(
                "float32", (1, 14), tuple(2.0 + observation[6 + i] for i in range(14))
            )
        )
        self.returned.append(output)
        return output

    def close(self):
        self.closed = True


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "candidate.onnx"
    path.write_bytes(b"fake-model-contents")
    return path


def test_feedforward_accepts_arbitrary_sole_output_name_and_reports_limited_scope(candidate):
    probe = FakeProbe(outputs=(TensorSpec("not_actions", "float32", ("batch", 14)),))
    received = []

    def factory(model):
        received.append(model)
        return probe

    result = inspect_policy(candidate, probe_factory=factory, steps=8, seed=17)
    assert result.passed
    assert result.reasons == ()
    assert result.compatibility_only is True
    assert result.physical_hardware_gate == "not_run"
    assert result.normalizer_provenance == "unverified"
    assert result.profile == "microduck-feedforward-api1"
    assert result.model_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert result.probe_steps == 8
    assert received == [candidate.read_bytes()]
    assert probe.closed
    assert result.output_spec.name == "not_actions"
    json.dumps(result.as_dict(), allow_nan=False)


def test_feedforward_sole_output_can_even_have_a_recurrent_state_name(candidate):
    probe = FakeProbe(outputs=(TensorSpec("h_out", "float32", (1, 14)),))
    assert inspect_policy(candidate, probe_factory=lambda _: probe).passed


def test_plausible_probes_match_official_layout_and_raw_previous_action(candidate):
    left, right = FakeProbe(), FakeProbe()
    assert inspect_policy(candidate, probe_factory=lambda _: left, steps=8, seed=9).passed
    assert inspect_policy(candidate, probe_factory=lambda _: right, steps=8, seed=9).passed
    assert left.observations == right.observations
    nominal = left.observations[0]
    assert nominal == (0.0,) * 5 + (-1.0,) + (0.0,) * 55
    for index, obs in enumerate(left.observations):
        assert len(obs) == 61
        assert all(math.isfinite(item) for item in obs)
        assert math.isclose(sum(item * item for item in obs[3:6]), 1.0)
        assert obs[48:61] == (0.0,) * 13
        if index:
            assert obs[34:48] == left.returned[index - 1].values
    assert max(left.observations[1][34:48]) > 1.0  # no smoke-only clipping


def test_locomotion_probes_keep_unbound_body_axes_zero(candidate):
    probe = FakeProbe()
    assert inspect_policy(
        candidate, probe_factory=lambda _: probe, command_profile="locomotion", steps=8
    ).passed
    assert any(any(value != 0 for value in obs[48:55]) for obs in probe.observations[1:])
    assert all(obs[55] == obs[56] == obs[60] == 0.0 for obs in probe.observations)


@pytest.mark.parametrize(
    "inputs,outputs,reason",
    [
        ((TensorSpec("input", "float32", (1, 61)),), None, "input_name"),
        ((TensorSpec("obs", "float32", (1, 51)),), None, "input_shape"),
        ((TensorSpec("obs", "float32", (2, 61)),), None, "input_shape"),
        ((TensorSpec("obs", "float32", (1, 1, 61)),), None, "input_shape"),
        ((TensorSpec("obs", "float64", (1, 61)),), None, "input_dtype"),
        (None, (TensorSpec("actor", "float32", (1, 15)),), "output_shape"),
        (None, (TensorSpec("actor", "float64", (1, 14)),), "output_dtype"),
        (
            None,
            (TensorSpec("actor", "float32", (1, 14)), TensorSpec("extra", "float32", (1, 14))),
            "tensor_count",
        ),
        (
            (
                TensorSpec("obs", "float32", (1, 61)),
                TensorSpec("h_in", "float32", (1, 1, 16)),
                TensorSpec("c_in", "float32", (1, 1, 16)),
            ),
            None,
            "unsupported_recurrent_profile",
        ),
    ],
)
def test_rejects_contract_mismatch_before_inference(candidate, inputs, outputs, reason):
    probe = FakeProbe(inputs=inputs, outputs=outputs)
    report = inspect_policy(candidate, probe_factory=lambda _: probe)
    assert not report.passed
    assert any(reason in item for item in report.reasons)
    assert not probe.observations
    assert probe.closed


@pytest.mark.parametrize("batch", [1, -1, None, "batch"])
def test_supports_only_rust_compatible_symbolic_or_single_batch(candidate, batch):
    probe = FakeProbe(inputs=(TensorSpec("obs", "float32", (batch, 61)),))
    assert inspect_policy(candidate, probe_factory=lambda _: probe).passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_actions(candidate, value):
    probe = FakeProbe(result=lambda _: TensorOutput("float32", (1, 14), (value,) * 14))
    report = inspect_policy(candidate, probe_factory=lambda _: probe)
    assert not report.passed
    assert "non_finite_output" in report.reasons
    assert probe.closed


def test_constant_nonuniform_action_vector_is_not_evidence_of_response(candidate):
    output = TensorOutput("float32", (1, 14), tuple(float(i) for i in range(14)))
    probe = FakeProbe(result=lambda _: output)
    report = inspect_policy(candidate, probe_factory=lambda _: probe)
    assert not report.passed
    assert "constant_output" in report.reasons


@pytest.mark.parametrize(
    "output,reason",
    [
        (TensorOutput("float32", (2, 7), (0.0,) * 14), "runtime_output_shape"),
        (TensorOutput("float64", (1, 14), (0.0,) * 14), "runtime_output_dtype"),
        (TensorOutput("float32", (1, 14), (0.0,) * 13), "runtime_output_length"),
    ],
)
def test_rejects_runtime_output_disagreeing_with_graph(candidate, output, reason):
    probe = FakeProbe(result=lambda _: output)
    report = inspect_policy(candidate, probe_factory=lambda _: probe)
    assert not report.passed
    assert reason in report.reasons


def test_weight_change_during_inference_invalidates_otherwise_passing_report(candidate):
    probe = FakeProbe()
    original_run = probe.run

    def mutate(obs):
        candidate.write_bytes(b"replacement-weights")
        return original_run(obs)

    probe.run = mutate
    report = inspect_policy(candidate, probe_factory=lambda _: probe)
    assert not report.passed
    assert "model_changed_during_check" in report.reasons
    assert probe.closed


@pytest.mark.parametrize("steps", [0, 1, MAX_PROBE_STEPS + 1])
def test_invalid_probe_budget_does_not_start_runtime(candidate, steps):
    def forbidden(_):
        pytest.fail("invalid inspection budget must not load a model")

    report = inspect_policy(candidate, probe_factory=forbidden, steps=steps)
    assert not report.passed
    assert "invalid_probe_steps" in report.reasons


def test_direct_script_cli_reports_missing_model_without_optional_dependencies(tmp_path):
    script = Path(__file__).parents[1] / "src" / "duckmind" / "policy_check.py"
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--onnx",
            str(tmp_path / "absent.onnx"),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(report_path.read_text())
    assert not report["passed"]
    assert report["physical_hardware_gate"] == "not_run"
    assert report["normalizer_provenance"] == "unverified"


def test_cli_timeout_is_failed_compatibility_evidence(candidate, tmp_path, monkeypatch, capsys):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(policy_check, "_run_worker", timeout)
    report_path = tmp_path / "timeout.json"
    status = policy_check.main(
        ["--onnx", str(candidate), "--report", str(report_path), "--timeout-s", "0.25"]
    )
    result = json.loads(capsys.readouterr().out)
    assert status == 1
    assert result == json.loads(report_path.read_text())
    assert result["reasons"] == ["probe_wall_timeout"]
    assert result["passed"] is False
    assert result["compatibility_only"] is True
    assert result["physical_hardware_gate"] == "not_run"


def test_oversized_model_is_rejected_before_loading_runtime(candidate, monkeypatch):
    monkeypatch.setattr(policy_check, "MAX_MODEL_BYTES", 5)

    def forbidden(_):
        pytest.fail("oversized model must not load a runtime")

    report = inspect_policy(candidate, probe_factory=forbidden)
    assert not report.passed
    assert report.reasons == ("model_too_large",)


@pytest.mark.parametrize("interrupt", [signal.SIGTERM, signal.SIGINT, None])
def test_standalone_interruption_or_timeout_reaps_its_owned_worker(candidate, tmp_path, interrupt):
    pid_path = tmp_path / "worker.pid"
    fake_worker = tmp_path / "slow_worker.py"
    fake_worker.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    script = Path(__file__).parents[1] / "src" / "duckmind" / "policy_check.py"
    launcher = (
        "import runpy, sys; ns = runpy.run_path(sys.argv[1]); "
        "ns['main'].__globals__['__file__'] = sys.argv[2]; "
        "raise SystemExit(ns['main'](sys.argv[3:]))"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            launcher,
            str(script),
            str(fake_worker),
            "--onnx",
            str(candidate),
            "--timeout-s",
            "0.5" if interrupt is None else "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_path.exists(), "harmless worker did not start"
        worker_pid = int(pid_path.read_text())
        if interrupt is not None:
            parent.send_signal(interrupt)
        stdout, stderr = parent.communicate(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        assert parent.returncode == (1 if interrupt is None else 128 + interrupt), stderr
        result = json.loads(stdout)
        assert not result["passed"]
        assert result["reasons"] == [
            "probe_wall_timeout"
            if interrupt is None
            else f"probe_interrupted:{signal.Signals(interrupt).name}"
        ]
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
