"""M7's push trial of MACHINE-VERIFY-v3.md: push the duck over, then see whether the safety reflex took
over in the grace time and whether the brain stopped telling it to walk.

Pure functions over a recorded pair (the pushed arm and its same-seed arm that is not pushed). Truth
is the world's ``fallen`` flag (the body's own fall verdict or a trunk tilt over 40 degrees); what the
brain did is its own record (``reflex`` and ``vx``). The reading rules are in
``docs/design/pet-behavior-v1/MACHINE-VERIFY-v3-CHOICES.md`` §七.

A pair has one of these ``status`` values:

``tipped``          the pushed arm fell inside the window, stayed down at least the grace time, and
                    the other arm did not fall inside its window: the push tipped it over
``not_tipped``      the pushed arm did not fall inside the window (reported; neither pass nor fail)
``recovered_fast``  it fell but was up again before the grace time ran out (reported)
``both_tipped``     the arm that was not pushed fell in the same window too (not the push's doing)
``already_down``    the duck was down at the moment of the push
"""

from __future__ import annotations

from dataclasses import dataclass, field

from duckmind.verify import spec_v3 as S
from duckmind.verify.outcomes_v2 import validate_v2
from duckmind.verify.trace import Trace, n_ticks, window, window_complete

TIPPED = "tipped"
NOT_TIPPED = "not_tipped"
RECOVERED_FAST = "recovered_fast"
BOTH_TIPPED = "both_tipped"
ALREADY_DOWN = "already_down"
STATUSES = (TIPPED, NOT_TIPPED, RECOVERED_FAST, BOTH_TIPPED, ALREADY_DOWN)


@dataclass
class PushOutcome:
    pair: str
    preset: str
    seed: int
    valid: bool
    problems: list[str] = field(default_factory=list)
    status: str | None = None
    fall_idx: int | None = None
    down_s: float | None = None  # how long it stayed down (to the end of the trace if never up)
    took_over: bool | None = None  # a body-holding reflex inside the grace time
    first_reflex: str | None = None  # the first body-holding reflex from the fall on
    first_reflex_s: float | None = None  # …and how long after the fall it showed
    righting: bool | None = None  # ``righting`` itself showed inside the grace time
    walked: bool | None = None  # a walk command after the grace time while still down
    blocked: bool | None = None  # not took over, or walked

    @property
    def tid(self) -> str:
        return f"{self.preset}|seed{self.seed}|push"


def first_fall(trace: Trace) -> int | None:
    """The first tick inside the 3 s window where the truth ``fallen`` flag goes up (the tick before
    was not down), or None."""
    fallen = trace["cols"]["fallen"]
    for i in window(trace, S.PUSH_WINDOW_S):
        if fallen[i] and not fallen[i - 1]:
            return i
    return None


def _down_ticks(trace: Trace, i: int) -> int:
    fallen = trace["cols"]["fallen"]
    n = n_ticks(trace)
    j = i
    while j < n and fallen[j]:
        j += 1
    return j - i


def push_outcome(pushed: Trace, control: Trace) -> PushOutcome:
    out = PushOutcome(
        pair=pushed.get("pair", ""),
        preset=pushed.get("preset", ""),
        seed=int(pushed.get("seed", -1)),
        valid=True,
    )
    for label, trace in (("pushed", pushed), ("control", control)):
        problems = validate_v2(trace)
        if not problems and not window_complete(trace, S.PUSH_WINDOW_S):
            problems.append("trace ends inside the window")
        out.problems += [f"{label}: {p}" for p in problems]
        check = trace.get("pair_check")
        if check is not None and check.get("pre_onset_equal") is False:
            out.problems.append(f"{label}: the two arms differ before the onset ({check})")
    if not out.problems and not (pushed.get("stim") or {}).get("delivered"):
        out.problems.append("the push was not delivered by the world")
    if out.problems:
        out.valid = False
        return out
    cols = pushed["cols"]
    dt = float(pushed["dt_s"])
    onset = pushed["onset_idx"]
    if cols["fallen"][onset - 1]:
        out.status = ALREADY_DOWN
        return out
    i = first_fall(pushed)
    if i is None:
        out.status = NOT_TIPPED
        return out
    out.fall_idx = i
    ticks = _down_ticks(pushed, i)
    out.down_s = ticks * dt
    grace_ticks = round(S.FALL_GRACE_S / dt)
    if first_fall(control) is not None:
        out.status = BOTH_TIPPED
        return out
    if ticks < grace_ticks:
        out.status = RECOVERED_FAST
        return out
    out.status = TIPPED
    reflex, vx, fallen = cols["reflex"], cols["vx"], cols["fallen"]
    grace = range(i, min(n_ticks(pushed), i + grace_ticks + 1))
    holds = [j for j in grace if reflex[j] in S.HOLD_REFLEXES]
    out.took_over = bool(holds)
    out.righting = any(reflex[j] == "righting" for j in grace)
    if holds:
        out.first_reflex = str(reflex[holds[0]])
        out.first_reflex_s = (holds[0] - i) * dt
    out.walked = any(fallen[j] and abs(vx[j]) > 1e-9 for j in range(i + grace_ticks + 1, i + ticks))
    out.blocked = (not out.took_over) or out.walked
    return out
