"""Per-trial and per-pair outcomes of MACHINE-VERIFY-v3.md, from recorded traces only.

Beside ``outcomes.py`` (v1) and ``outcomes_v2.py`` (v2), both untouched. What v3 changed is M1's
rule: only a thing that has a place (a rolled ball, a new object) is judged by the head turning toward
it; every other stimulus (clap, bang, name, hello, a hand coming up, a pat on the head, a poke) is an
event — any reflex in the 3 s window means the duck noticed. Everything else is v2's own function.
Booleans may be ``None`` = unknown; nothing here turns an unknown into a pass.
"""

from __future__ import annotations

import dataclasses

from duckmind.verify import outcomes as O1
from duckmind.verify import outcomes_v2 as O2
from duckmind.verify import spec_v3 as S
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.trace import Trace


def attended(trace: Trace) -> bool | None:
    """M1 for one arm. A stimulus with a place (``S.DIRECTIONAL``): the v2 head-turn rule (the head
    turned ≥ 10° toward the true direction and the error halved, inside the 3 s window; unknown when
    the rule cannot be met from the onset geometry or no direction was recorded). An event
    (``S.EVENT``): any reflex at all in the window — orient, lean_in, startle, freeze, withdraw …;
    a poke answered by a withdraw is noticed. The sham arm gets the same rule (for a directional
    stimulus its phantom direction is the stim arm's real one; for an event it is the reflexes its own
    duck shows in the same window)."""
    stimulus = trace["stimulus"]
    if stimulus in S.EVENT:
        return O1.reflex_in_window(trace)
    if stimulus in S.DIRECTIONAL:
        return O2.head_turn(trace)
    raise ValueError(f"stimulus {stimulus!r} is in neither list of the v3 spec")


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV2:
    """v2's pair outcome with ``attend_stim`` / ``attend_sham`` read by v3's M1 rule."""
    out = O2.pair_outcome(stim, sham)
    if not out.valid:
        return out
    return dataclasses.replace(out, attend_stim=attended(stim), attend_sham=attended(sham))
