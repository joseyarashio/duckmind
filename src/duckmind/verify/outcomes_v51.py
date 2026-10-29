"""M1 of MACHINE-VERIFY v5.1 (the errata): v5's rule with the start rule read as "not already selected on the tick before the stimulus".

Beside ``outcomes_v5.py`` (untouched in what it judges: its default is v5's own reading). v5.1 is the same functions with
``carry_only`` on: a stretch counts if it starts at or after the stimulus tick and the selection was not present on the tick before the
stimulus, in both arms. Everything else — the label match, the control's spot, the window, the hold, the radius, the unknowns, the
reference "reflex or flee" reading — is v5's.
"""

from __future__ import annotations

from typing import Any

from duckmind.verify import outcomes_v5 as O5
from duckmind.verify.outcomes_v5 import Escape, PairOutcomeV5, Target, escape_in_window
from duckmind.verify.trace import Trace

__all__ = [
    "Escape",
    "PairOutcomeV5",
    "Target",
    "attended",
    "control_target",
    "escape_in_window",
    "pair_outcome",
    "stim_target",
]


def stim_target(trace: Trace, **kw: Any) -> Target:
    return O5.stim_target(trace, carry_only=True, **kw)


def control_target(trace: Trace, **kw: Any) -> Target:
    return O5.control_target(trace, carry_only=True, **kw)


def attended(trace: Trace) -> bool | None:
    return O5.attended(trace, carry_only=True)


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV5:
    return O5.pair_outcome(stim, sham, carry_only=True)
