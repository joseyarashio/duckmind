#!/usr/bin/env python3
"""Build the neural duck's robotd skill table (skilltrain.motion_table): the trained motions of 2026-10-05 plus the
panic gait and the quick turn (2026-10-09), and the official forward roll for the flip trick (2026-10-26). Usage: install_motion_table.py
[OUT_DIR] [--panic PATH.onnx] [--qturn PATH.onnx] [--roulade PATH.onnx]  (default OUT_DIR: live/motions in the data folder, panic: the
gated R-x90 iteration 1000, roulade: the v5 set's own). The trained networks are read from the training runs in the data folder
(:func:`duckmind.paths.data_dir`) and this checkout's ``outputs/retrain-20260929`` (run it from the checkout that holds them)."""

import sys
from pathlib import Path

from duckmind import paths
from duckmind.skilltrain import motion_table as MT

TRAINING = paths.data_dir() / "training"
S1 = TRAINING / "stage1-2026-09-28"
SB = TRAINING / "skillsB-2026-09-28"
R1 = Path(__file__).resolve().parents[1] / "outputs" / "retrain-20260929" / "run" / "jobs"
PANIC = TRAINING / "panic-2026-10-08" / "best" / "R-x90_panic_999.onnx"
PANIC_SHA256 = "fede44de40e571b6bbabc0c400655d2f81001b0258c8291037fedaf4df7a346d"
# the quick-turn network (queue-2026-10-09, A100 run qt-turn-d iteration 1200: QUICK-TURN-2026-10-09.md)
QTURN = TRAINING / "queue-2026-10-09" / "best" / "turn_td1199.onnx"
QTURN_SHA256 = "c1c50bb1fb9a819c04b56ff5d2b98bb6e37c94ca33238545fd656b9bacb9f21f"

# the official forward roll of the v5 policy set (ROULADE-2026-10-22.md): the file the roll was measured with
ROULADE = paths.v5_policy_dir() / "roulade.onnx"

args = sys.argv[1:]
panic = PANIC
if "--panic" in args:
    i = args.index("--panic")
    panic = Path(args[i + 1])
    del args[i : i + 2]
qturn = QTURN
if "--qturn" in args:
    i = args.index("--qturn")
    qturn = Path(args[i + 1])
    del args[i : i + 2]
roulade = ROULADE
if "--roulade" in args:
    i = args.index("--roulade")
    roulade = Path(args[i + 1])
    del args[i : i + 2]
out = args[0] if args else str(MT.DEFAULT_DIR)
if roulade.is_symlink() or not roulade.is_file() or MT.sha256_of(roulade) != MT.ROULADE_SHA256:
    raise SystemExit(f"{roulade}: not the official roulade ({MT.ROULADE_SHA256[:12]}…)")
if MT.sha256_of(qturn) != QTURN_SHA256:
    raise SystemExit(f"{qturn}: sha256 is not the gated quick-turn network ({QTURN_SHA256[:12]}…)")
if MT.sha256_of(panic) != PANIC_SHA256:
    raise SystemExit(f"{panic}: sha256 is not the gated panic network ({PANIC_SHA256[:12]}…)")
sources = {
    "startle": S1 / "export/startle.onnx",
    "comfort": S1 / "export/comfort.onnx",
    "step": R1 / "step-b3/final/bskstep.onnx",
    "bow": R1 / "bow-b3/final/bskbow.onnx",
    "hop": R1 / "hop-h3/final/hopover.onnx",
    "panic": panic,
    "qturn": qturn,
    "roulade": roulade,
}
path = MT.install(out, sources, base_table=S1 / "export/skills.live.json")
print(path)
print(MT.check(path) or "pins ok")
