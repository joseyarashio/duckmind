"""The numbers of MACHINE-VERIFY v7 (2026-10-03): v6's, with one change to what 'noticed' counts for a stimulus with a place.

``spec_v6.py`` and its predecessors are untouched and stay what v5, v5.1 and v6 judged with. The sentence (the planner's, verbatim in
``MACHINE-VERIFY-v7.md``): v7 = v6 with one change: for roll_ball and novel_object, 'noticed' also counts a visual reflex (startle, freeze,
orient, withdraw) whose onset in the window (at or after the stimulus tick) has a source track that maps (simulator labels) to the stimulus
object; the control arm counts a visual reflex whose source track maps to any thing within 0.3 m of the phantom spot. Thresholds unchanged
(≥70%, ≥0.5). The operational reading is in ``MACHINE-VERIFY-v7-CHOICES.md`` (committed before any v7 judgement).
"""

from __future__ import annotations

from duckmind.verify.spec_v6 import (
    CARRY_ONLY,
    DIRECTIONAL,
    EAR_LAYER,
    EPS,
    EVENT,
    FLEE_BEHAVIOURS,
    INFO_STIMULUS,
    KINDS,
    L1_CELL_PAIRS,
    L2_CELL_PAIRS,
    M1_DIFF_MIN,
    M1_STIM_MIN,
    PUSH,
    SPOT_RADIUS_M,
    STIMULI,
    TARGET_HOLD_S,
    WINDOW_S,
)

RULE = "v7"
VISUAL_REFLEXES = ("startle", "freeze", "orient", "withdraw")  # the reflexes the sentence lists

__all__ = [
    "CARRY_ONLY",
    "DIRECTIONAL",
    "EAR_LAYER",
    "EPS",
    "EVENT",
    "FLEE_BEHAVIOURS",
    "INFO_STIMULUS",
    "KINDS",
    "L1_CELL_PAIRS",
    "L2_CELL_PAIRS",
    "M1_DIFF_MIN",
    "M1_STIM_MIN",
    "PUSH",
    "RULE",
    "SPOT_RADIUS_M",
    "STIMULI",
    "TARGET_HOLD_S",
    "VISUAL_REFLEXES",
    "WINDOW_S",
]
