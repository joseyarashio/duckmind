"""The numbers of MACHINE-VERIFY v6 (2026-10-03): v5.1's, with one change to which target kinds select the object.

``spec_v5.py`` and ``spec_v51.py`` are untouched and stay what v5 and v5.1 judged with. The sentence (the planner's, verbatim in
``MACHINE-VERIFY-v6.md``): v6 = v5.1 with one change: for the directional object stimuli (roll_ball, novel_object), 'noticed' also counts
a stretch (≥0.3 s, starting at or after the stimulus tick, not present on the tick before) in which the brain's target is of kind
``threat`` or ``thing`` and its tgt_track maps (simulator labels) to the stimulus object. The control arm uses the same rule against any
thing within 0.3 m of the phantom spot. Thresholds unchanged (≥70%, ≥0.5). The operational reading is in
``MACHINE-VERIFY-v6-CHOICES.md`` (committed before any v6 judgement).
"""

from __future__ import annotations

from duckmind.verify.spec_v51 import (
    CARRY_ONLY,
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

RULE = "v6"
KINDS = ("thing", "threat")  # the target kinds that select the object (v5 and v5.1: "thing" only)

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
    "WINDOW_S",
]
