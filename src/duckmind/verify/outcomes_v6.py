"""M1 of MACHINE-VERIFY v6: v5.1's rule with the brain's ``threat`` target also selecting the object.

Beside ``outcomes_v5.py`` and ``outcomes_v51.py`` (untouched in what they judge). v6 is the same functions with ``carry_only`` on (v5.1's
start rule) and ``kinds`` = thing or threat: a stretch counts when it is at least 0.3 s of consecutive ticks on which the brain's target is
of kind ``thing`` or ``threat`` and its ``tgt_track`` maps (the simulator's labels) to the stimulus object (control arm: to a thing within
0.3 m of the phantom spot), starts at or after the stimulus tick, and was not selected on the tick before it. Everything else — the window,
the hold, the radius, the spot, the unknowns, the reference "reflex or flee" reading — is v5's.

What v6 adds beyond the rule are numbers that only the report quotes (no verdict reads them): per arm, how many ticks of the window the
target was a ``threat``, how many of those were selected, and what v5.1's thing-only reading of the same arm says (``thing_only``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from duckmind.verify import outcomes_v5 as O5
from duckmind.verify import spec_v6 as S
from duckmind.verify.outcomes_v5 import Escape, PairOutcomeV5, Target, escape_in_window
from duckmind.verify.trace import Trace, window

__all__ = [
    "Escape",
    "PairOutcomeV5",
    "Target",
    "TargetV6",
    "attended",
    "control_target",
    "escape_in_window",
    "pair_outcome",
    "stim_target",
]

THREAT_ONLY: tuple[str, ...] = ("threat",)


@dataclass(frozen=True)
class TargetV6(Target):
    """v5's reading (``value``, the stretch numbers) under v6's kinds, plus what only the report uses."""

    thing_only: bool | None = None  # v5.1's reading of the same arm (target kind thing only)
    threat_ticks: int = 0  # ticks of the window on which the target is a threat (any track)
    threat_selected_ticks: int = (
        0  # …of those, the ones that select (the object / a thing near the spot)
    )


def _threat_ticks(trace: Trace) -> int:
    cols = trace["cols"]
    if "tgt_kind" not in cols:
        return 0
    return sum(1 for i in window(trace, S.WINDOW_S) if cols["tgt_kind"][i] == "threat")


def _extend(both: Target, trace: Trace, arm: str) -> TargetV6:
    read = O5.stim_target if arm == "stim" else O5.control_target
    thing = read(trace, carry_only=True)
    threat = read(trace, carry_only=True, kinds=THREAT_ONLY)
    return TargetV6(
        **{f.name: getattr(both, f.name) for f in dataclasses.fields(Target)},
        thing_only=thing.value,
        threat_ticks=_threat_ticks(trace),
        threat_selected_ticks=threat.matched_ticks,
    )


def stim_target(trace: Trace, **kw: Any) -> TargetV6:
    return _extend(O5.stim_target(trace, carry_only=True, kinds=S.KINDS, **kw), trace, "stim")


def control_target(trace: Trace, **kw: Any) -> TargetV6:
    both = O5.control_target(trace, carry_only=True, kinds=S.KINDS, **kw)
    return _extend(both, trace, "sham")


def attended(trace: Trace) -> bool | None:
    return O5.attended(trace, carry_only=True, kinds=S.KINDS)


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV5:
    out = O5.pair_outcome(stim, sham, carry_only=True, kinds=S.KINDS)
    if out.tgt_stim is not None and out.tgt_sham is not None:
        out.tgt_stim = _extend(out.tgt_stim, stim, "stim")
        out.tgt_sham = _extend(out.tgt_sham, sham, "sham")
    return out
