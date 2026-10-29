"""The numbers of MACHINE-VERIFY v5.1 (errata, 2026-10-03): v5's, with one change to the start rule.

``spec_v5.py`` is untouched and stays what v5 judged with. The errata (``MACHINE-VERIFY-v5.1-ERRATA.md``, the planner's): a stretch of the
brain's target on the object counts if it starts at or after the stimulus tick AND the selection was not present on the tick before
the stimulus (both arms); everything else is v5. So a selection that begins at the stimulus tick itself counts (in the brain layer the
brain answers in the tick the object appears), and only one that was already there on the tick before is a carry-over and is left out.
"""

from __future__ import annotations

from duckmind.verify.spec_v5 import (
    DIRECTIONAL,
    EAR_LAYER,
    EPS,
    EVENT,
    FLEE_BEHAVIOURS,
    INFO_STIMULUS,
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

RULE = "v5.1"
CARRY_ONLY = True  # a stretch is left out only when the selection was present on the tick before the stimulus

__all__ = [
    "CARRY_ONLY",
    "DIRECTIONAL",
    "EAR_LAYER",
    "EPS",
    "EVENT",
    "FLEE_BEHAVIOURS",
    "INFO_STIMULUS",
    "L1_CELL_PAIRS",
    "L2_CELL_PAIRS",
    "M1_DIFF_MIN",
    "M1_STIM_MIN",
    "PUSH",
    "RULE",
    "SPOT_RADIUS_M",
    "STIMULI",
    "TARGET_HOLD_S",
    "WINDOW_S",
]
