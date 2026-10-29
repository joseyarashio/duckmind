"""Per-trial and per-pair outcomes of MACHINE-VERIFY-v2.md, from recorded traces only.

Beside ``outcomes.py`` (v1, untouched). What v2 did not change is that module's own functions,
called here; what v2 changed or added is written here: M1's rules (a phantom direction for the
sham, reflex-only for the sounds), M4's arrival at a remembered place (through the start pose),
M5's episode and its natural end as the spec now words them. Booleans may be ``None`` = unknown;
nothing here turns an unknown into a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from duckmind.verify import outcomes as O1
from duckmind.verify import spec
from duckmind.verify import spec_v2 as S
from duckmind.verify.outcomes import Claims, Episode, PairOutcome
from duckmind.verify.trace import (
    Trace,
    bearing_to,
    dist_to,
    n_ticks,
    validate,
    window,
    window_complete,
    wrap,
)

SCHEMAS = ("mv-trace-1", "mv-trace-2")


def as_v1(trace: Trace) -> Trace:
    """A shallow copy the v1 functions accept (they only look at the schema in ``validate``)."""
    return trace if trace.get("schema") == "mv-trace-1" else {**trace, "schema": "mv-trace-1"}


def validate_v2(trace: Trace) -> list[str]:
    if trace.get("schema") not in SCHEMAS:
        return ["schema"]
    return validate(as_v1(trace))


# ── M1 ─────────────────────────────────────────────────────────────────────────────


def head_turn(trace: Trace) -> bool | None:
    """M1 for a directional stimulus: the v1 head-turn rule (≥ 10° toward the true direction and
    the error halved, inside the window), or ``None`` when the trial has no true direction
    recorded, or the rule cannot be met from the onset geometry (the stimulus is at the gaze, so
    no turn of ≥ 10° leaves at most half the error): unknown, not "did not notice"."""
    path = (trace.get("objects") or {}).get("stim")
    if path is None:
        return None
    cols = trace["cols"]
    onset = trace["onset_idx"]
    if path[onset] is None:
        return None
    ox, oy = path[onset]
    err0 = wrap(bearing_to(cols["x"][onset], cols["y"][onset], ox, oy) - cols["gaze"][onset - 1])
    if abs(err0) < S.M1_MIN_ONSET_ERR_RAD:
        return None
    return O1.head_turned_toward(trace)


def attended(trace: Trace) -> bool | None:
    """M1 for one arm: a directional stimulus — the head turned toward the true direction; a
    stimulus without direction — any reflex inside the window. The sham arm gets the same rule
    (its phantom direction is the stim arm's real one)."""
    stimulus = trace["stimulus"]
    if stimulus in S.NON_DIRECTIONAL:
        return O1.reflex_in_window(trace)
    if stimulus in S.DIRECTIONAL:
        return head_turn(trace)
    raise ValueError(f"stimulus {stimulus!r} is in neither list of the spec")


# ── M5 ─────────────────────────────────────────────────────────────────────────────


def response_episode(trace: Trace) -> Episode | None:
    """「刺激引起的那段」: a chain of body-holding reflexes that begins inside the window (a startle
    that goes on into a freeze is one chain); else the first bout that begins inside the window;
    else none. ``category``: ``natural`` (the chain released itself; or the bout's own satiation
    was ≥ 0.5 at its last tick), ``displaced`` (satiation < 0.5: taken over), ``timeout`` (the
    trace marks it), ``censored`` (the trace ends before it did: unknown)."""
    cols = trace["cols"]
    n = n_ticks(trace)
    reflex = cols["reflex"]
    win = window(trace, S.WINDOW_S)
    for i in win:
        if reflex[i] in S.HOLD_REFLEXES and reflex[i - 1] not in S.HOLD_REFLEXES:
            j = i
            while j < n and reflex[j] in S.HOLD_REFLEXES:
                j += 1
            return Episode("hold", str(reflex[i]), i, "censored" if j >= n else "natural")
    for i in win:
        if cols["bout"][i] != cols["bout"][i - 1]:
            bout = cols["bout"][i]
            end = i
            while end + 1 < n and cols["bout"][end + 1] == bout:
                end += 1
            forced = cols.get("end_reason")
            if end >= n - 1:
                category = "censored"
            elif forced is not None and forced[end] == "timeout":
                category = "timeout"
            elif cols["phase"][end] >= S.NATURAL_SATIATION:
                category = "natural"
            else:
                category = "displaced"
            return Episode("bout", cols["behaviour"][i], i, category)
    return None


# ── M4: a remembered place, through the start pose ─────────────────────────────────


def odom_to_world(transform: dict[str, Any], px: float, py: float) -> tuple[float, float]:
    """An odometry-frame point in the world frame: the transform is the start pose — the true pose
    and the brain's odometry reading at the first tick."""
    tx, ty, tyaw = transform["start_truth"]
    ox, oy, oyaw = transform["start_odom"]
    rot = wrap(tyaw - oyaw)
    dx, dy = px - ox, py - oy
    return (
        tx + math.cos(rot) * dx - math.sin(rot) * dy,
        ty + math.sin(rot) * dx + math.cos(rot) * dy,
    )


def place_arrival(trace: Trace, info: dict[str, Any]) -> dict[str, Any]:
    """One 到了 whose target is a remembered place: the true distance from the trunk to the place
    (the brain's odometry point carried to the world by the start pose) against the touchable
    distance + 0.1 m. Unknown when no place point or transform was recorded."""
    cols = trace["cols"]
    i = info["tick"]
    base = {k: v for k, v in info.items() if k != "why"}
    px = (cols.get("place_x") or [None] * n_ticks(trace))[i]
    py = (cols.get("place_y") or [None] * n_ticks(trace))[i]
    transform = trace.get("transform")
    if px is None or py is None or not transform:
        return {**base, "ok": None, "why": "no place point or start pose was recorded"}
    wx, wy = odom_to_world(transform, float(px), float(py))
    true_d = dist_to(cols["x"][i], cols["y"][i], wx, wy)
    bound = float(trace.get("touch_reach_m", spec.BEAK_REACH_M)) + S.M4_ARRIVE_MARGIN_M
    return {
        **base,
        "object": f"place({wx:.2f},{wy:.2f})",
        "place_world": [round(wx, 3), round(wy, 3)],
        "true_dist_m": round(true_d, 3),
        "bound_m": round(bound, 3),
        "ok": true_d <= bound,
    }


def claims(trace: Trace) -> Claims:
    """v1's claims (a thing's bearing, range and 到了) with each 到了 at a remembered place
    judged through the start pose instead of left unknown."""
    out = O1.claims(trace)
    out.arrivals = [
        place_arrival(trace, a) if a.get("ok") is None and a.get("target_kind") == "place" else a
        for a in out.arrivals
    ]
    return out


# ── a pair ─────────────────────────────────────────────────────────────────────────


@dataclass
class PairOutcomeV2(PairOutcome):
    """v1's pair outcome with the v2 reading of ``attend_*`` and ``episode``, and the numbers
    the report lists next to the verdicts."""

    level_db_head: float | None = None
    reflex_sham: bool = False
    reflex_or_change_stim: bool = False  # v1's M1 "attention", kept for the comparison
    reflex_or_change_sham: bool = False
    head_turn_stim: bool | None = None  # an object stimulus: the v1 "head turned" part
    aimed_stim: bool | None = None  # …and the v1 "a bout aimed at it" part (v2 does not count it)


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV2:
    out = PairOutcomeV2(
        pair=stim.get("pair", ""),
        preset=stim.get("preset", ""),
        seed=int(stim.get("seed", -1)),
        stimulus=stim.get("stimulus", ""),
        valid=True,
    )
    for label, trace in (("stim", stim), ("sham", sham)):
        problems = validate_v2(trace)
        if not problems and not window_complete(trace, S.WINDOW_S):
            problems.append("trace ends inside the window")
        out.problems += [f"{label}: {p}" for p in problems]
    for label, trace in (("stim", stim), ("sham", sham)):
        check = trace.get("pair_check")
        if check is not None and check.get("pre_onset_equal") is False:
            out.problems.append(f"{label}: the two arms differ before the onset ({check})")
    if out.problems:
        out.valid = False
        return out
    if sham.get("stimulus") != stim.get("stimulus") or sham.get("seed") != stim.get("seed"):
        out.valid = False
        out.problems.append("sham is not the same stimulus and seed")
        return out
    s = stim.get("stim") or {}
    if stim["layer"] == "L2" and stim["stimulus"] == "bang":
        out.level_db_head = s.get("level_db_head")
        if out.level_db_head is None or out.level_db_head < S.LOUD_HEAD_DB_MIN:
            out.problems.append(f"the bang was {out.level_db_head} dB at the head: under 90 dB")
            out.valid = False
            return out
    out.perceivable = s.get("perceivable") if s.get("delivered") else False
    if not s.get("delivered"):
        out.problems.append("stimulus not delivered by the world")
        out.valid = False
        return out
    out.attend_stim = attended(stim)
    out.attend_sham = attended(sham)
    out.changed_stim = O1.changed(stim)
    out.changed_sham = O1.changed(sham)
    out.reflex_stim = O1.reflex_in_window(stim)
    out.reflex_sham = O1.reflex_in_window(sham)
    out.reflex_or_change_stim = out.reflex_stim or out.changed_stim
    out.reflex_or_change_sham = out.reflex_sham or out.changed_sham
    out.startled_stim = O1.startled(stim)
    out.startled_sham = O1.startled(sham)
    out.episode = response_episode(stim)
    out.caused = out.changed_stim and not out.changed_sham
    out.reopen_stim = O1.reopened(stim)
    out.falls_stim, out.blocked_stim = O1.falls(stim)
    out.falls_sham, out.blocked_sham = O1.falls(sham)
    if stim["layer"] == "L2":
        out.claims = [claims(stim), claims(sham)]
    out.approach = O1.approach(stim)
    out.behaviours_seen = frozenset(stim["cols"]["behaviour"]) | frozenset(
        sham["cols"]["behaviour"]
    )
    if S.NON_DIRECTIONAL.isdisjoint({stim["stimulus"]}):
        out.head_turn_stim = O1.head_turned_toward(stim)
        out.aimed_stim = O1.bout_aimed_at_stimulus(stim)
    return out
