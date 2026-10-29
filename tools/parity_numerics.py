"""Why the ONNX round-trip parity is 1e-5 on the Mac and 1.5e-5 on the Spark (bug 6, 2026-10-09).

Imports the v5 policy into the mjlab actor, exports it, and compares three float32 evaluations of the exported
network against a float64 reference of the same weights:

  torch  = ``rl_onnx.torch_policy`` (the test's side), ort = onnxruntime's CPU provider (the other side),
  ref64  = numpy float64 on the ONNX's own weights (no accumulation-order noise at the 1e-5 scale).

If both float32 paths sit at the same distance from ref64 as they do from each other, the disagreement between them is
float32 accumulation order (BLAS / MLAS kernels differ per CPU), not a difference in the model.

    PYTHONPATH=src <mjlab python> tools/parity_numerics.py [--v5 DIR] [--samples 512] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

from duckmind import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--v5", default=str(paths.v5_policy_dir()))
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--out")
    args = ap.parse_args()
    import onnxruntime as ort
    import torch

    from duckmind.skilltrain import rl_onnx as O

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    report: dict = {
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "torch_threads": torch.get_num_threads(),
        "files": {},
    }
    for name in ("velstand.onnx", "alpha_sitstand.onnx"):
        src = Path(args.v5) / name
        actor = O.build_actor()
        O.load_into_actor(actor, src, init_std=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            out = O.export_actor(actor, Path(tmp) / name, {"tool": "parity_numerics"})
            w = O.read_onnx(out)
            rng = np.random.default_rng(0)
            x = np.concatenate(
                [
                    (
                        w["mean"]
                        + 3.0 * w["div"] * rng.standard_normal((args.samples, w["obs_dim"]))
                    ).astype(np.float32),
                    rng.standard_normal((max(8, args.samples // 8), w["obs_dim"])).astype(
                        np.float32
                    ),
                ]
            )
            t = O.torch_policy(src, "cpu")(torch.from_numpy(x)).numpy().astype(np.float64)
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            name_in = sess.get_inputs()[0].name  # (robotd's ONNX has batch 1: one row at a time)
            o = np.concatenate(
                [sess.run(None, {name_in: x[i : i + 1]})[0] for i in range(len(x))]
            ).astype(np.float64)
            r = O.reference64(out, x)
        row = {
            "output_scale": float(np.abs(r).max()),
            "torch_vs_ort": float(np.abs(t - o).max()),
            "torch_vs_ref64": float(np.abs(t - r).max()),
            "ort_vs_ref64": float(np.abs(o - r).max()),
        }
        # one row at a time: is the result independent of the batch (accumulation order per row)?
        one = np.stack(
            [
                O.torch_policy(src, "cpu")(torch.from_numpy(x[i : i + 1])).numpy()[0]
                for i in range(64)
            ]
        ).astype(np.float64)
        row["torch_batch_vs_single"] = float(np.abs(t[:64] - one).max())
        report["files"][name] = row
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
