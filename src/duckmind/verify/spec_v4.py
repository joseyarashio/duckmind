"""The frozen numbers of MACHINE-VERIFY-v4.md (2026-10-02), beside the v1, v2 and v3 ones.

``spec.py``, ``spec_v2.py`` and ``spec_v3.py`` are untouched and stay what v1, v2 and v3 judged with. What v4 leaves as
it was is read from ``spec_v3`` (not copied); what v4 changes or adds is here, each naming the sentence of the spec it
comes from. The operational choices the spec leaves open are in
``docs/design/pet-behavior-v1/MACHINE-VERIFY-v4-CHOICES.md`` (committed before any v4 result).
"""

from __future__ import annotations

from duckmind.verify import spec_v3 as v3

# ── unchanged (v3's own) ─────────────────────────────────────────────────────────────
WINDOW_S = v3.WINDOW_S
M1_STIM_MIN = v3.M1_STIM_MIN  # 每格有刺激組 ≥ 70%
M1_DIFF_MIN = v3.M1_DIFF_MIN  # 有刺激組 − 對照組 ≥ 0.5
DIRECTIONAL = v3.DIRECTIONAL  # 滾球、放新東西
EVENT = v3.EVENT
STIMULI = v3.STIMULI
L1_CELL_PAIRS = v3.L1_CELL_PAIRS
L2_CELL_PAIRS = v3.L2_CELL_PAIRS
L2_SEEDS = v3.L2_SEEDS
L2_STIMULI = v3.L2_STIMULI
L2_ONSET_S = v3.L2_ONSET_S
PUSH = v3.PUSH
PHYSICS_NAME = v3.PHYSICS_NAME
EAR_LAYER = v3.EAR_LAYER
LOOM_SEE_S = v3.LOOM_SEE_S
OTHER_PHYSICS_STIMULI = v3.OTHER_PHYSICS_STIMULI

# ── M1「盯住」（改）：滾球、放新東西 ───────────────────────────────────────────────────
FIX_TOL_DEG = 15.0  # 頭的方向和真實物體方位的差距 ≤ 15°
FIX_HOLD_S = 0.5  # 連續 ≥ 0.5 秒
FIX_EPS = (
    1e-9  # numerical tolerance (rad for the gaps, s for the hold): a float error never decides
)

# ── 實驗擺放（物理層）────────────────────────────────────────────────────────────────
BAND_FRACTION = 0.6  # 畫面實際水平範圍的中間 60%
BAND_MARGIN = (1.0 - BAND_FRACTION) / 2.0  # 兩側各 20%
EXCLUDED_TARGET = 0.10  # 擺放上要讓排除的比例 ≤ 10%（設計目標，不是判定門檻）
SCAN_RANGE_DEG = (
    90.0  # the horizontal span is scanned from −90° to +90° of the head camera's heading …
)
SCAN_STEP_DEG = 0.05  # … in steps of 0.05°
SCAN_Z_M = 0.05  # at the height of a low thing's centre (the ball 0.034 m, the bottle about 0.1 m)
PLACE_DIST_M = {"roll_ball": 1.4, "novel_object": 1.2}  # v3's distances (a ball starts 1.4 m away)
DIST_STEPS_M = (0.0, 0.3, 0.6)  # then 0.4 m: v3's order of nearer spots
NEAREST_M = 0.4
LADDER_MIDDLE_STEP = 0.15  # the fall-back spots φ_mid ± 0.15·W

# ── 逼近的手（物理層）— v3's own numbers (neuro_world._stim_hand_loom), the geometry is new ──
HAND_REACH_M = 0.75  # the start is this far from the lens
HAND_STOP_M = 0.09  # the end is this far in front of the beak …
HAND_STOP_UP_M = 0.02  # … and this much higher
HAND_BACK_UP_M = 0.5  # the hand leaves up by this much
HAND_LEG_S = (0.25, 0.45, 0.35, 0.6)  # appear, loom, hold, leave

# the stimuli whose M1 rule v4 changed (the others keep v3's event rule)
FIXATION_STIMULI = DIRECTIONAL
