"""The frozen numbers of MACHINE-VERIFY-v5.md (2026-10-03), beside the v1-v4 ones.

``spec.py`` ... ``spec_v4.py`` are untouched and stay what v1-v4 judged with. Numbers v5 leaves as they were are read from
``spec_v4`` (not copied); what v5 changes or adds is here, each naming the sentence of the spec it comes from. The operational
choices the spec leaves open are in ``docs/design/pet-behavior-v1/MACHINE-VERIFY-v5-CHOICES.md`` (committed before any v5 judgement).

What v5 changes: M1 of a stimulus with a place (a rolled ball, a new object) is no longer the head's direction. The duck noticed
when its brain **selected the real object as its target** — the target column (``tgt_track``) pointing at a track the simulator
itself labelled with that object — for at least 0.3 s in a row, in a stretch that **starts after the stimulus tick**, inside the 3 s
window. The control arm (same seed, nothing put there) counts any thing whose true position is within 0.3 m of the empty spot (the
object's planned place at the same tick) being selected for at least 0.3 s in a row, in a stretch that starts after the stimulus tick.
The gates stay 70% and 0.5.
"""

from __future__ import annotations

from duckmind.verify import spec_v4 as v4

WINDOW_S = v4.WINDOW_S  # 窗口 3.0 秒（含兩端）
M1_STIM_MIN = v4.M1_STIM_MIN  # 每格有刺激組 ≥ 70%
M1_DIFF_MIN = v4.M1_DIFF_MIN  # 有刺激組 − 對照組 ≥ 0.5
DIRECTIONAL = v4.DIRECTIONAL  # 滾球、放新東西
EVENT = v4.EVENT
STIMULI = v4.STIMULI
L1_CELL_PAIRS = v4.L1_CELL_PAIRS
L2_CELL_PAIRS = v4.L2_CELL_PAIRS
EAR_LAYER = v4.EAR_LAYER
PUSH = v4.PUSH

TARGET_HOLD_S = 0.3  # 維持 ≥ 0.3 秒（從第一個到最後一個 tick 的時間）
SPOT_RADIUS_M = 0.3  # 半徑 0.3 公尺
EPS = 1e-9  # numerical tolerance (s for the hold, m for the radius): a float error never decides

# the known limit (reported, not a gate): a duck that is startled by the ball and runs has no target that names the ball
FLEE_BEHAVIOURS = frozenset({"flee", "retreat"})
INFO_STIMULUS = (
    "roll_ball"  # the physics layer's ball cell gets the extra "reflex or flee in 3 s" row
)
