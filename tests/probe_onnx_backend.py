"""Optional real-ORT negative checks; deliberately outside pytest discovery.

Run from the repository root (no Duck Mind installation is required)::

    uv run --no-project --with onnx==1.20.1 --with onnxruntime==1.27.0 \
      --with numpy python tests/probe_onnx_backend.py \
      --report docs/validation/evidence/onnx-backend-negative.json

Only temporary, synthetic ONNX graphs are generated. This is compatibility
evidence, not training, simulator quality evaluation, or physical robot testing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def write_graph(path: Path, case: str) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_name = "wrong_input" if case == "wrong_input_name" else "obs"
    if case == "wrong_input_name":
        indices = helper.make_tensor("indices", TensorProto.INT64, [14], list(range(14)))
        node = helper.make_node("Gather", [input_name, "indices"], ["actions"], axis=1)
        initializers = [indices]
    else:
        values = [float(index) for index in range(14)]
        if case == "nan_output":
            values[3] = float("nan")
        tensor = helper.make_tensor("fixed_actions", TensorProto.FLOAT, [1, 14], values)
        node = helper.make_node("Constant", [], ["actions"], value=tensor)
        initializers = []
    graph = helper.make_graph(
        [node],
        f"synthetic_{case}",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="duckmind-synthetic-negative-probe",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    inspector = script.parents[1] / "src" / "duckmind" / "policy_check.py"
    cases: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="duckmind-onnx-negative-", dir="/tmp") as temporary:
        work = Path(temporary)
        for case, expected_reason in (
            ("constant_nonuniform_output", "constant_output"),
            ("wrong_input_name", "input_name"),
            ("nan_output", "non_finite_output"),
        ):
            model_path = work / f"{case}.onnx"
            report_path = work / f"{case}.json"
            write_graph(model_path, case)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(inspector),
                    "--onnx",
                    str(model_path),
                    "--report",
                    str(report_path),
                    "--steps",
                    "8",
                    "--seed",
                    "0",
                    "--timeout-s",
                    "15",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            result = json.loads(completed.stdout)
            expected_rejection = (
                completed.returncode == 1
                and result == json.loads(report_path.read_text())
                and result["passed"] is False
                and result["reasons"] == [expected_reason]
                and result["compatibility_only"] is True
                and result["physical_hardware_gate"] == "not_run"
            )
            cases.append(
                {
                    "case": case,
                    "expected_reason": expected_reason,
                    "rejected_as_expected": expected_rejection,
                    "inspector_exit_code": completed.returncode,
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "inspection": result,
                }
            )
    report = {
        "schema_version": 1,
        "synthetic": True,
        "trained_policy": False,
        "compatibility_only": True,
        "physical_hardware_gate": "not_run",
        "scope": "Three synthetic graphs; no training, simulator, Hub, or robot connection.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("onnx", "onnxruntime", "numpy")
        },
        "inspector_sha256": hashlib.sha256(inspector.read_bytes()).hexdigest(),
        "probe_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "temporary_models_removed": True,
        "all_rejected_as_expected": all(case["rejected_as_expected"] for case in cases),
        "cases": cases,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(serialized)
    sys.stdout.write(serialized)
    return 0 if report["all_rejected_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
