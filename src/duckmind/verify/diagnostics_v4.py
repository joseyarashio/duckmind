"""Facts that explain the verdicts of MACHINE-VERIFY-v4.md but are not part of them (the spec gives them no
threshold): how the objects were placed and how many were out of view, how many pairs were already looking at the
object at the stimulus moment, what the other readings of the rule say on the same recordings, and what the hand that
comes up did. Pure functions over recorded traces and pair outcomes; the report quotes them, no verdict reads them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import outcomes_v4 as O4
from duckmind.verify import spec_v4 as S
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.trace import Trace

ALL = "ALL"


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def object_cells(
    outcomes: Iterable[PairOutcomeV2], *, merge_presets: bool
) -> dict[str, dict[str, Any]]:
    """For every cell of a stimulus with a place (key ``scope|stimulus``), over its pairs that were in view and read
    on both arms: the notice counts under the judged rule (the fixation), and — as reported readings, never judged —
    how many pairs were already looking (the gap at the stimulus moment ≤ 15°), the same counts on the pairs that were
    not, v3's turn rule on the same recordings, and the 'any start tick' reading of the fixation."""
    cells: dict[str, dict[str, Any]] = {}
    gaps: dict[str, list[float]] = {}
    longest: dict[str, list[float]] = {}
    for o in outcomes:
        if not isinstance(o, O4.PairOutcomeV4) or not o.valid or o.stimulus not in S.DIRECTIONAL:
            continue
        key = f"{ALL if merge_presets else o.preset}|{o.stimulus}"
        c = cells.setdefault(
            key,
            {
                "valid": 0,
                "in_view": 0,
                "out_of_view": 0,
                "view_unknown": 0,
                "read": 0,
                "stim": 0,
                "sham": 0,
                "already_inside": 0,
                "out_n": 0,
                "out_stim": 0,
                "out_sham": 0,
                "in_stim": 0,
                "in_sham": 0,
                "turn_n": 0,
                "turn_stim": 0,
                "turn_sham": 0,
                "exist_stim": 0,
                "exist_sham": 0,
                "unknown": 0,
            },
        )
        c["valid"] += 1
        if o.perceivable is False:
            c["out_of_view"] += 1
            continue
        if o.perceivable is None:
            c["view_unknown"] += 1
            continue
        c["in_view"] += 1
        fs, fh = o.fix_stim, o.fix_sham
        if fs is None or fh is None or fs.value is None or fh.value is None:
            c["unknown"] += 1
            continue
        c["read"] += 1
        c["stim"] += bool(fs.value)
        c["sham"] += bool(fh.value)
        if fs.already_inside:
            c["already_inside"] += 1
            c["in_stim"] += bool(fs.value)
            c["in_sham"] += bool(fh.value)
        else:
            c["out_n"] += 1
            c["out_stim"] += bool(fs.value)
            c["out_sham"] += bool(fh.value)
        if o.turn_stim is not None and o.turn_sham is not None:
            c["turn_n"] += 1
            c["turn_stim"] += bool(o.turn_stim)
            c["turn_sham"] += bool(o.turn_sham)
        c["exist_stim"] += bool(fs.exist)
        c["exist_sham"] += bool(fh.exist)
        if fs.g0_deg is not None:
            gaps.setdefault(key, []).append(fs.g0_deg)
        longest.setdefault(key, []).append(fs.longest_run_s)
    for key, c in cells.items():
        c["g0_deg"] = _spread(gaps.get(key, []))
        c["longest_run_s_stim"] = _spread(longest.get(key, []))
    return dict(sorted(cells.items()))


def exclusions(
    outcomes: Iterable[PairOutcomeV2], plan: A.Plan, stimuli: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Per stimulus (every temperament together): of the planned pairs, how many were valid, in view, out of view
    (excluded), unknown in view, invalid or missing. ``rate`` ＝ out of view / planned; ``target`` is the spec's
    ≤ 10% for the physics layer's placement (a design target, no verdict)."""
    planned = len(plan.presets) * len(plan.seeds)
    by: dict[str, Counter[str]] = {s: Counter() for s in stimuli}
    for o in outcomes:
        if o.stimulus not in by:
            continue
        c = by[o.stimulus]
        if not o.valid:
            c["invalid"] += 1
        elif o.perceivable is True:
            c["in_view"] += 1
        elif o.perceivable is False:
            c["out_of_view"] += 1
        else:
            c["view_unknown"] += 1
    out: dict[str, dict[str, Any]] = {}
    for stimulus, c in by.items():
        seen = c["in_view"] + c["out_of_view"] + c["view_unknown"] + c["invalid"]
        out[stimulus] = {
            "planned": planned,
            "in_view": c["in_view"],
            "out_of_view": c["out_of_view"],
            "view_unknown": c["view_unknown"],
            "invalid": c["invalid"],
            "missing": max(0, planned - seen),
            "rate": round(c["out_of_view"] / planned, 4) if planned else None,
            "target": S.EXCLUDED_TARGET,
        }
    return out


def placement_summary(traces: Iterable[Trace]) -> dict[str, dict[str, Any]]:
    """The physics layer's objects (stimulated arms): where each was placed — which rung of the ladder, the frame's range
    and the band as read, the bearing used — and the gap at the stimulus moment."""
    by: dict[str, list[Trace]] = {}
    for tr in traces:
        if tr.get("arm") == "stim" and tr.get("stimulus") in S.DIRECTIONAL:
            by.setdefault(tr["stimulus"], []).append(tr)
    out: dict[str, dict[str, Any]] = {}
    for stimulus, trs in sorted(by.items()):
        placements = [t["stim"].get("placement") for t in trs]
        have = [p for p in placements if p]
        spans = [p["span_deg"] for p in have if p.get("span_deg")]
        g0 = [
            g for g in (O4.fixation(t).g0_deg for t in trs if not t.get("error")) if g is not None
        ]
        out[stimulus] = {
            "trials": len(trs),
            "with_placement": len(have),
            "mode": dict(sorted(Counter(p["mode"] for p in have).items())),
            "rung": dict(sorted(Counter(p["rung"] for p in have).items())),
            "side": dict(sorted(Counter(p["side"] for p in have).items())),
            "dist_m": dict(sorted(Counter(p["dist_m"] for p in have).items())),
            "span_lo_deg": _spread([s[0] for s in spans]),
            "span_hi_deg": _spread([s[1] for s in spans]),
            "phi_deg": _spread([p["phi_deg"] for p in have]),
            "g0_deg": _spread(g0),
            "g0_within_15": sum(1 for g in g0 if g <= S.FIX_TOL_DEG + 1e-6),
            "g0_n": len(g0),
        }
    return out


def loom_summary(traces: Iterable[Trace], *, presets: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The hand that comes up (physics layer), per temperament and all together: pairs, delivered, the hand seen by the
    camera within 1.5 s (the truth), any reflex in the window (stimulated arm / control arm), startles, and the hand's
    own numbers (ticks driven, the nearest it came to the beak)."""
    arms: dict[str, dict[str, Trace]] = {}
    for tr in traces:
        if tr.get("stimulus") == "loom":
            arms.setdefault(tr["pair"], {})[tr["arm"]] = tr
    out: dict[str, dict[str, Any]] = {}
    for scope in (*presets, ALL):
        c: Counter[str] = Counter()
        gaps: list[float] = []
        ticks: list[float] = []
        lens: list[float] = []
        for pair in arms.values():
            stim, sham = pair.get("stim"), pair.get("sham")
            if stim is None or sham is None or (scope != ALL and stim["preset"] != scope):
                continue
            c["pairs"] += 1
            s = stim["stim"]
            c["delivered"] += bool(s.get("delivered"))
            c["seen"] += s.get("perceivable") is True
            c["not_seen"] += s.get("perceivable") is False
            c["seen_unknown"] += s.get("perceivable") is None
            if stim.get("error") or sham.get("error"):
                c["error"] += 1
            for arm, trace in (("stim", stim), ("sham", sham)):
                cols = trace["cols"]
                i0 = trace["onset_idx"]
                n = len(cols["t"])
                end = i0
                while end + 1 < n and cols["t"][end + 1] - cols["t"][i0] <= S.WINDOW_S + 1e-9:
                    end += 1
                reflexes = {cols["reflex"][i] for i in range(i0, end + 1) if cols["reflex"][i]}
                c[f"reflex_{arm}"] += bool(reflexes)
                c[f"startle_{arm}"] += bool(reflexes & {"startle", "freeze"})
            hand = s.get("hand") or {}
            if hand.get("min_gap_to_beak_m") is not None:
                gaps.append(float(hand["min_gap_to_beak_m"]))
            if hand.get("ticks") is not None:
                ticks.append(float(hand["ticks"]))
            first = hand.get("first") or {}
            if first.get("lens_m") is not None:
                lens.append(float(first["lens_m"]))
                c["start_lifted"] += bool(first.get("clamped"))
        out[scope] = {
            **{k: c[k] for k in sorted(c)},
            "min_gap_to_beak_m": _spread(gaps),
            "hand_ticks": _spread(ticks),
            "start_lens_m": _spread(lens),
        }
    return out


def rule_effect(
    v3_rows: Sequence[Mapping[str, Any]], v4_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per M1 gate row of a stimulus with a place: the same recording read by v3's rule and by v4's rule."""
    old = {(r["scope"], r["part"]): r for r in v3_rows if r["metric"] == "M1"}
    out: list[dict[str, Any]] = []
    for r in v4_rows:
        if r["metric"] != "M1" or r["part"].split("：", 1)[0] not in S.DIRECTIONAL:
            continue
        o = old.get((r["scope"], r["part"]))
        if o is None:
            continue
        out.append(
            {
                "scope": r["scope"],
                "part": r["part"],
                "v3_rule": {"value": o["value"], "n": o["n"], "verdict": o["verdict"]},
                "v4_rule": {"value": r["value"], "n": r["n"], "verdict": r["verdict"]},
            }
        )
    return out
