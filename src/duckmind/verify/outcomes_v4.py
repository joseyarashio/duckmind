"""Per-trial and per-pair outcomes of MACHINE-VERIFY-v4.md, from recorded traces only.

Beside ``outcomes.py`` (v1), ``outcomes_v2.py`` (v2) and ``outcomes_v3.py`` (v3), all untouched. What v4 changed is
M1's rule for a stimulus that has a place (a rolled ball, a new object): not a head turn of at least 10° but a
**fixation** — inside the 3 s window the gap between the head's direction and the true direction of the object stays
within 15° for at least 0.5 s in a row, and at the beginning of that stretch the gap is smaller than it was at the
stimulus moment (the head turned there; it was not already looking). The control arm gets the same rule on its own
head towards the object's place. Every other stimulus keeps v3's event rule (any reflex in the window). The operational
details the spec leaves open are in ``MACHINE-VERIFY-v4-CHOICES.md`` §二.

Booleans may be ``None`` = unknown; nothing here turns an unknown into a pass.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from duckmind.verify import outcomes as O1
from duckmind.verify import outcomes_v3 as O3
from duckmind.verify import spec_v4 as S
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.trace import Trace, bearing_to, window, wrap

TOL_RAD = math.radians(S.FIX_TOL_DEG)


@dataclass(frozen=True)
class Fixation:
    """One arm's fixation reading. ``value``: True = a qualifying stretch exists; False = none; None = the truth
    needed is missing (``why`` says what). The rest are the numbers the rule used (also read by the report)."""

    value: bool | None
    why: str = ""
    g0_deg: float | None = None  # the gap at the stimulus moment
    already_inside: bool | None = (
        None  # g0 ≤ 15°: the head was already looking at the stimulus moment
    )
    start_gap_deg: float | None = None  # the gap at the beginning of the qualifying stretch
    start_s: float | None = None  # seconds after the stimulus at which that stretch began
    run_s: float | None = None  # its length
    longest_run_s: float = 0.0  # the longest stretch within 15° of any kind (qualifying or not)
    exist: bool | None = None  # the report-only reading: any start tick (CHOICES §二.9 b)


UNKNOWN_NO_PATH = "no_path"
UNKNOWN_NOT_AT_ONSET = "not_at_onset"


def gaps(trace: Trace) -> tuple[range, list[float | None]] | None:
    """The window's ticks and, per tick, the gap (rad, ≥ 0) between the head camera's heading and the true bearing from
    the trunk to the stimulus object; ``None`` for a tick at which the object is not in the world. ``None`` overall
    when the trace has no object path at all or the object is not in the world at the stimulus tick."""
    path = (trace.get("objects") or {}).get("stim")
    if path is None:
        return None
    onset = trace["onset_idx"]
    if onset >= len(path) or path[onset] is None:
        return None
    cols = trace["cols"]
    win = window(trace, S.WINDOW_S)
    out: list[float | None] = []
    for i in win:
        point = path[i]
        out.append(
            None
            if point is None
            else abs(
                wrap(bearing_to(cols["x"][i], cols["y"][i], point[0], point[1]) - cols["gaze"][i])
            )
        )
    return win, out


def fixation(trace: Trace) -> Fixation:
    """The v4 M1 rule for a stimulus with a place, read from one arm's trace (CHOICES §二). A run is a maximal stretch of
    consecutive ticks of the window whose gap is ≤ 15°; it counts when it lasts ≥ 0.5 s (first tick to last tick, by the
    recorded times) and the gap at its first tick is smaller than the gap at the stimulus tick."""
    got = gaps(trace)
    if got is None:
        path = (trace.get("objects") or {}).get("stim")
        return Fixation(None, why=UNKNOWN_NO_PATH if path is None else UNKNOWN_NOT_AT_ONSET)
    win, g = got
    t = trace["cols"]["t"]
    g0 = g[0]
    assert g0 is not None  # ``gaps`` checked the stimulus tick
    tol = TOL_RAD + S.FIX_EPS
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for k, gap in enumerate(g):
        inside = gap is not None and gap <= tol
        if inside and start is None:
            start = k
        elif not inside and start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, len(g) - 1))

    def length(a: int, b: int) -> float:
        return float(t[win[b]] - t[win[a]])

    longest = max((length(a, b) for a, b in runs), default=0.0)
    hold = S.FIX_HOLD_S - S.FIX_EPS
    found: tuple[int, int] | None = None
    exist = False
    for a, b in runs:
        long_enough = length(a, b) >= hold
        first = g[a]
        assert first is not None
        if long_enough and found is None and first < g0 - S.FIX_EPS:
            found = (a, b)
        if (
            long_enough
        ):  # any start tick of the stretch that still has 0.5 s of it ahead (CHOICES §二.9 b)
            for k in range(a, b + 1):
                gk = g[k]
                if gk is not None and gk < g0 - S.FIX_EPS and float(t[win[b]] - t[win[k]]) >= hold:
                    exist = True
                    break
    return Fixation(
        value=found is not None,
        g0_deg=math.degrees(g0),
        already_inside=g0 <= tol,
        start_gap_deg=None if found is None else math.degrees(g[found[0]] or 0.0),
        start_s=None if found is None else float(t[win[found[0]]] - t[win[0]]),
        run_s=None if found is None else length(*found),
        longest_run_s=longest,
        exist=exist,
    )


def fixated(trace: Trace) -> bool | None:
    return fixation(trace).value


def attended(trace: Trace) -> bool | None:
    """M1 for one arm. A stimulus with a place (``S.DIRECTIONAL``): the fixation rule; an event (``S.EVENT``): any
    reflex at all in the 3 s window (v3's rule). The control arm gets the same rule (for a stimulus with a place, its
    own head towards the place the object has in its pair's stimulated arm)."""
    stimulus = trace["stimulus"]
    if stimulus in S.EVENT:
        return O1.reflex_in_window(trace)
    if stimulus in S.DIRECTIONAL:
        return fixated(trace)
    raise ValueError(f"stimulus {stimulus!r} is in neither list of the v4 spec")


@dataclass
class PairOutcomeV4(PairOutcomeV2):
    """v3's pair outcome (v2's fields) with the v4 M1 reading, and for a stimulus with a place the numbers of the
    fixation rule and of v3's turn rule on the same recording (reported, never judged)."""

    fix_stim: Fixation | None = None
    fix_sham: Fixation | None = None
    turn_stim: bool | None = (
        None  # v3's rule (head turned ≥ 10° towards, error halved) on the same traces
    )
    turn_sham: bool | None = None


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV4:
    """v3's pair outcome with ``attend_stim`` / ``attend_sham`` read by v4's M1 rule."""
    base = O3.pair_outcome(stim, sham)
    out = PairOutcomeV4(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
    if not out.valid:
        return out
    if stim["stimulus"] in S.DIRECTIONAL:
        out.turn_stim, out.turn_sham = base.attend_stim, base.attend_sham  # v3's reading
        out.fix_stim, out.fix_sham = fixation(stim), fixation(sham)
        out.attend_stim, out.attend_sham = out.fix_stim.value, out.fix_sham.value
    return out
