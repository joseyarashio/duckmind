"""A B-skill training checkpoint → robotd's ONNX on the Mac (b:train, 2026-09-28).

The deterministic actor of ``model_<it>.pt`` / ``latest.pt`` (rsl_rl ``actor_state_dict``, the recipe's
61-512-256-128-14 MLP, normaliser baked) exported with the same metadata as ``bsk_train.export``, a parity
check (torch vs onnxruntime on 512 observations near the normaliser's mean) and robotd's format check, so an
intermediate checkpoint can go through the Mac gate (``bsk_probe run --onnx-dir``) while the A100 keeps training.
Stage 1b's ``ckpt_to_onnx`` for the B networks (that one reads Stage 1's variant table).

    PYTHONPATH=src .venv-official-cpu/bin/python -m duckmind.skilltrain.bsk_export \
        --net bskstep --ckpt latest.pt --out-dir onnx/step-a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B


def metadata(net: str, iteration: int, source: str) -> dict[str, Any]:
    """``bsk_train.export``'s ONNX metadata (+ the checkpoint it came from)."""
    es = B.entries(net)
    return {
        "duckmind_skill_net": net,
        "duckmind_stage": B.STAGE,
        "skills": [e.name for e in es],
        "iteration": int(iteration),
        "command_encoding": {e.name: list(e.command) for e in es},
        "action_scale": B.ACTION_SCALE,
        "gain_ratio": B.GAIN_RATIO,
        "trained_with_lowpass": [0.5, 0.7],
        "servo_delay_ms": [5 * B.SERVO_DELAY_STEPS[0], 5 * B.SERVO_DELAY_STEPS[1]],
        "from_checkpoint": source,
    }


def export_checkpoint(net: str, ckpt: str | Path, out_dir: str | Path) -> dict[str, Any]:
    import torch

    from duckmind.skilltrain import rl_onnx
    from duckmind.skilltrain.rl_eval import checkpoint_policy

    if net not in B.NETS:
        raise ValueError(f"net {net!r}; use one of {B.NETS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    actor = rl_onnx.build_actor()
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    actor.eval()
    it = int(ck.get("iter", -1))
    onnx_path = rl_onnx.export_actor(
        actor, out_dir / f"{net}.onnx", metadata(net, it, Path(ckpt).name)
    )
    fn, info = checkpoint_policy(ckpt, "cpu")
    par = rl_onnx.parity(onnx_path, lambda x: fn(x).cpu(), samples=512)
    res = {
        "net": net,
        "onnx": str(onnx_path),
        "sha256": rl_onnx.sha256(onnx_path),
        "iteration": it,
        "parity": par,
        "robotd_problems": rl_onnx.robotd_problems(onnx_path),
        "ckpt": info["ckpt"],
        "env_state": (ck.get("infos") or {}).get("env_state"),
    }
    (out_dir / f"{net}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_export")
    ap.add_argument("--net", choices=B.NETS, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res = export_checkpoint(a.net, a.ckpt, a.out_dir)
    print(json.dumps({k: v for k, v in res.items() if k not in ("ckpt",)}, default=str))
    bad = res["robotd_problems"] or res["parity"]["max_abs"] > 1e-4
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
