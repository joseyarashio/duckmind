"""Bounded local compatibility checks for Microduck feed-forward ONNX exports.

Pinned contract: pollen-robotics/microduck a9ec4b2079ef8ee7904014089c885bb07d57d63c,
duck-control/src/policy.rs:478-524,537-620 and duck-control/src/obs.rs:12-42.
The Rust loader requires one float32 `obs` input [1 or dynamic,61] and one
float32 output [1 or dynamic,14]; the feed-forward output name is unrestricted.
LSTM/API 2 is intentionally unsupported by this API 1 inspection profile.

Unlike microduck_rl cb70b792312d559a4da09064d92009079671815f's publish smoke test,
these probes keep gravity normalized, feed back unclipped raw previous actions,
and check variation over time per action, not variance across different joints.
Passing proves neither baked-in observation normalization nor physical quality.

Standalone CLI (no duckmind installation or trainer PYTHONPATH override needed):
  python policy_check.py --onnx candidate.onnx --report report.json
The CLI runs inference in a child process with a wall timeout. The injectable
inspect_policy() API is synchronous; its caller owns process/time isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Callable, Protocol, Sequence

MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_PROBE_STEPS = 256
VARIATION_EPSILON = 1e-6
CONTRACT_COMMIT = "a9ec4b2079ef8ee7904014089c885bb07d57d63c"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str | None, ...]


@dataclass(frozen=True)
class TensorOutput:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[float, ...]


class PolicyProbe(Protocol):
    """Mock boundary: graph metadata plus one CPU inference, with no robot I/O."""

    @property
    def inputs(self) -> Sequence[TensorSpec]: ...

    @property
    def outputs(self) -> Sequence[TensorSpec]: ...

    def run(self, observation: tuple[float, ...]) -> TensorOutput: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PolicyInspection:
    model_path: str
    passed: bool = False
    reasons: tuple[str, ...] = ()
    schema_version: int = 1
    profile: str = "microduck-feedforward-api1"
    compatibility_only: bool = True
    physical_hardware_gate: str = "not_run"
    normalizer_provenance: str = "unverified"
    normalizer_note: str = "Shape and synthetic inference cannot prove a baked-in normalizer."
    contract_commit: str = CONTRACT_COMMIT
    model_sha256: str | None = None
    model_bytes: int = 0
    input_spec: TensorSpec | None = None
    output_spec: TensorSpec | None = None
    probe_steps: int = 0
    seed: int = 0
    command_profile: str = "zero"
    finite_output: bool = False
    non_constant_output: bool = False
    max_abs_action: float | None = None
    max_temporal_action_delta: float | None = None
    variation_epsilon: float = VARIATION_EPSILON

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PolicyCheckError(ValueError):
    pass


class ProbeInterrupted(PolicyCheckError):
    def __init__(self, signum: int) -> None:
        self.exit_code = 128 + signum
        super().__init__(f"probe_interrupted:{signal.Signals(signum).name}")


def _read_model(path: Path) -> bytes:
    if not path.is_file():
        raise PolicyCheckError("model_not_found")
    with path.open("rb") as stream:
        content = stream.read(MAX_MODEL_BYTES + 1)
    if not content:
        raise PolicyCheckError("empty_model")
    if len(content) > MAX_MODEL_BYTES:
        raise PolicyCheckError("model_too_large")
    return content


def _matrix(spec: TensorSpec, *, width: int, role: str) -> None:
    if spec.dtype != "float32":
        raise PolicyCheckError(f"{role}_dtype")
    if len(spec.shape) != 2:
        raise PolicyCheckError(f"{role}_shape")
    batch, columns = spec.shape
    valid_batch = (
        (isinstance(batch, int) and not isinstance(batch, bool) and batch in (1, -1))
        or batch is None
        or (isinstance(batch, str) and bool(batch))
    )
    if not valid_batch or type(columns) is not int or columns != width:
        raise PolicyCheckError(f"{role}_shape")


def _check_contract(probe: PolicyProbe) -> tuple[TensorSpec, TensorSpec]:
    if len(probe.inputs) != 1 or len(probe.outputs) != 1:
        if any(
            spec.name in {"h_in", "c_in", "h_out", "c_out"}
            for spec in (*probe.inputs, *probe.outputs)
        ):
            raise PolicyCheckError("unsupported_recurrent_profile")
        raise PolicyCheckError("tensor_count")
    input_spec, output_spec = probe.inputs[0], probe.outputs[0]
    if input_spec.name != "obs":
        raise PolicyCheckError("input_name")
    _matrix(input_spec, width=61, role="input")
    _matrix(output_spec, width=14, role="output")
    return input_spec, output_spec


def _observation(
    rng: random.Random,
    previous: tuple[float, ...],
    *,
    nominal: bool,
    command_profile: str,
) -> tuple[float, ...]:
    obs = [0.0] * 61
    obs[5] = -1.0
    obs[34:48] = previous
    if not nominal:
        obs[0:3] = [rng.uniform(-0.05, 0.05) for _ in range(3)]
        x, y = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
        obs[3:6] = [x, y, -math.sqrt(1.0 - x * x - y * y)]
        obs[6:20] = [rng.uniform(-0.025, 0.025) for _ in range(14)]
        obs[20:34] = [rng.uniform(-0.05, 0.05) for _ in range(14)]
        if command_profile == "locomotion":
            obs[48:51] = [
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.03, 0.03),
                rng.uniform(-0.1, 0.1),
            ]
            obs[51:55] = [rng.uniform(-0.03, 0.03) for _ in range(4)]
            obs[57:60] = [
                rng.uniform(-0.005, 0.005),
                rng.uniform(-0.02, 0.02),
                rng.uniform(-0.02, 0.02),
            ]
    # Body x/y/yaw (55,56,60) are unbound in training and always stay zero.
    return tuple(obs)


def _check_output(output: TensorOutput) -> tuple[float, ...]:
    if output.dtype != "float32":
        raise PolicyCheckError("runtime_output_dtype")
    if output.shape != (1, 14):
        raise PolicyCheckError("runtime_output_shape")
    if len(output.values) != 14:
        raise PolicyCheckError("runtime_output_length")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in output.values
    ):
        raise PolicyCheckError("non_finite_output")
    return tuple(float(value) for value in output.values)


class OnnxRuntimeProbe:
    """Optional production adapter, using immutable model bytes and CPU only."""

    def __init__(self, model: bytes) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )

        def spec(value) -> TensorSpec:
            return TensorSpec(
                value.name,
                "float32" if value.type == "tensor(float)" else value.type,
                tuple(value.shape),
            )

        self.inputs = tuple(spec(value) for value in self._session.get_inputs())
        self.outputs = tuple(spec(value) for value in self._session.get_outputs())

    def run(self, observation: tuple[float, ...]) -> TensorOutput:
        obs = self._np.asarray(observation, dtype=self._np.float32).reshape(1, 61)
        output = self._session.run([self.outputs[0].name], {"obs": obs})[0]
        return TensorOutput(
            str(output.dtype),
            tuple(int(value) for value in output.shape),
            tuple(float(value) for value in output.reshape(-1)),
        )

    def close(self) -> None:
        del self._session


def inspect_policy(
    path: str | Path,
    *,
    probe_factory: Callable[[bytes], PolicyProbe] | None = None,
    steps: int = 50,
    seed: int = 0,
    command_profile: str = "zero",
) -> PolicyInspection:
    """Check local model bytes; a passing report is compatibility evidence only.

    `probe_factory` receives the immutable, hashed model bytes. API callers that
    need a wall bound must isolate this synchronous method as the CLI does.
    """

    path = Path(path)
    report = PolicyInspection(
        model_path=str(path.resolve()), seed=seed, command_profile=command_profile
    )
    reasons: list[str] = []
    probe: PolicyProbe | None = None
    content: bytes | None = None
    try:
        if type(steps) is not int or not 2 <= steps <= MAX_PROBE_STEPS:
            raise PolicyCheckError("invalid_probe_steps")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise PolicyCheckError("invalid_seed")
        if command_profile not in {"zero", "locomotion"}:
            raise PolicyCheckError("invalid_command_profile")
        content = _read_model(path)
        digest = hashlib.sha256(content).hexdigest()
        report = replace(report, model_sha256=digest, model_bytes=len(content))
        probe = (probe_factory or OnnxRuntimeProbe)(content)
        input_spec, output_spec = _check_contract(probe)
        report = replace(report, input_spec=input_spec, output_spec=output_spec)
        rng = random.Random(seed)
        previous: tuple[float, ...] = (0.0,) * 14
        minimum: tuple[float, ...] | None = None
        maximum: tuple[float, ...] | None = None
        max_abs = 0.0
        for index in range(steps):
            obs = _observation(rng, previous, nominal=index == 0, command_profile=command_profile)
            previous = _check_output(probe.run(obs))
            minimum = previous if minimum is None else tuple(map(min, minimum, previous))
            maximum = previous if maximum is None else tuple(map(max, maximum, previous))
            max_abs = max(max_abs, *(abs(value) for value in previous))
            report = replace(report, probe_steps=index + 1)
        assert minimum is not None and maximum is not None
        temporal_delta = max(high - low for low, high in zip(minimum, maximum))
        report = replace(
            report,
            finite_output=True,
            non_constant_output=temporal_delta > VARIATION_EPSILON,
            max_abs_action=max_abs,
            max_temporal_action_delta=temporal_delta,
        )
        if not report.non_constant_output:
            raise PolicyCheckError("constant_output")
    except PolicyCheckError as error:
        reasons.append(str(error))
    except Exception as error:
        reasons.append(f"probe_error:{type(error).__name__}:{str(error)[:200]}")
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception as error:
                reasons.append(f"probe_close_error:{type(error).__name__}")
        if content is not None:
            try:
                if hashlib.sha256(_read_model(path)).hexdigest() != report.model_sha256:
                    reasons.append("model_changed_during_check")
            except (OSError, PolicyCheckError):
                reasons.append("model_unreadable_after_check")
    return replace(report, passed=not reasons, reasons=tuple(reasons))


def _run_worker(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Bound inference and reap only this invocation's child on every exit.

    Handlers only record cancellation: raising during Popen construction could
    abandon a child before the caller acquires its process handle.
    """

    interrupted: int | None = None
    process: subprocess.Popen[str] | None = None
    previous_handlers = {}

    def cancel(signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        if interrupted is None:
            interrupted = signum

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, cancel)
        deadline = time.monotonic() + timeout
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        while True:
            if interrupted is not None:
                raise ProbeInterrupted(interrupted)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(remaining, 0.1))
            except subprocess.TimeoutExpired:
                continue
            if interrupted is not None:
                raise ProbeInterrupted(interrupted)
            assert process.returncode is not None
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        try:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=1)
                finally:
                    for stream in (process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command-profile", choices=("zero", "locomotion"), default="zero")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.report is not None and args.report.resolve() == args.onnx.resolve():
        parser.error("report path must differ from the model path")
    if not math.isfinite(args.timeout_s) or not 0 < args.timeout_s <= 300:
        parser.error("timeout-s must be finite and in (0,300]")
    failure_exit_code = 1
    if args._worker:
        result = inspect_policy(
            args.onnx, steps=args.steps, seed=args.seed, command_profile=args.command_profile
        ).as_dict()
    else:
        try:
            worker = _run_worker(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_worker",
                    "--onnx",
                    str(args.onnx.resolve()),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(args.seed),
                    "--command-profile",
                    args.command_profile,
                ],
                timeout=args.timeout_s,
            )
            if worker.returncode not in (0, 1):
                raise PolicyCheckError(f"probe_process_exit:{worker.returncode}")
            result = json.loads(worker.stdout)
            if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
                raise PolicyCheckError("invalid_probe_process_report")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            if isinstance(error, ProbeInterrupted):
                failure_exit_code = error.exit_code
            reason = (
                "probe_wall_timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else str(error)[:200]
            )
            result = PolicyInspection(
                model_path=str(args.onnx.resolve()), reasons=(reason,)
            ).as_dict()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    sys.stdout.write(serialized)
    return 0 if result["passed"] else failure_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
