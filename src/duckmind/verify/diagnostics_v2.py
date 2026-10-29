"""Facts that explain the verdicts of MACHINE-VERIFY-v2.md but are not part of them (the spec gives
them no threshold): what the duck did when a directional rule was never met, how the time shares and
the reactions of a life separate the temperaments, what ended the stimulus-caused episodes, which gentle
stimuli startled. Pure functions over recorded traces, lives and pair outcomes; the report quotes them,
no verdict reads them."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import lives as LV
from duckmind.verify import spec
from duckmind.verify import spec_v2 as S
from duckmind.verify.aggregate import Outcomes
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.trace import Trace, bearing_to, window, wrap


def directional_diagnostics(traces: Iterable[Trace]) -> dict[str, dict[str, Any]]:
    """For each directional stimulus that has a true direction in the traces (stim arms, rule
    defined): how many trials, in how many of them a reflex of each name showed up in the 3 s window,
    the median of the largest head turn toward the true direction (deg; 0 = never toward it), and the
    median of the smallest error / onset error reached (the rule needs <= 0.5)."""
    per: dict[str, dict[str, Any]] = {}
    for tr in traces:
        kind = tr["stimulus"]
        path = (tr.get("objects") or {}).get("stim")
        if tr["arm"] != "stim" or kind not in S.DIRECTIONAL or path is None:
            continue
        cols = tr["cols"]
        k = tr["onset_idx"]
        if path[k] is None:
            continue
        err0 = wrap(bearing_to(cols["x"][k], cols["y"][k], *path[k]) - cols["gaze"][k - 1])
        if abs(err0) < S.M1_MIN_ONSET_ERR_RAD:
            continue
        sign = 1.0 if err0 > 0 else -1.0
        best_turn, best_ratio = 0.0, math.inf
        names: set[str] = set()
        for i in window(tr, S.WINDOW_S):
            best_turn = max(best_turn, wrap(cols["gaze"][i] - cols["gaze"][k - 1]) * sign)
            if path[i] is not None:
                err = wrap(bearing_to(cols["x"][i], cols["y"][i], *path[i]) - cols["gaze"][i])
                best_ratio = min(best_ratio, abs(err) / abs(err0))
            if cols["reflex"][i] is not None:
                names.add(str(cols["reflex"][i]))
        d = per.setdefault(kind, {"n": 0, "turns": [], "ratios": [], "reflex_trials": Counter()})
        d["n"] += 1
        d["turns"].append(math.degrees(best_turn))
        d["ratios"].append(best_ratio)
        d["reflex_trials"].update(names)
    out: dict[str, dict[str, Any]] = {}
    for kind, d in per.items():
        out[kind] = {
            "n": d["n"],
            "reflex_trials": dict(sorted(d["reflex_trials"].items(), key=lambda kv: -kv[1])),
            "median_max_turn_deg": round(statistics.median(d["turns"]), 1),
            "median_best_err_ratio": round(statistics.median(d["ratios"]), 2),
            "best_err_ratio": round(min(d["ratios"]), 2),
        }
    return out


def episode_counts(data: Outcomes) -> dict[str, dict[str, int]]:
    """How many stimulus-caused episodes of each kind and end (``bout_displaced`` …), per
    temperament and in all (key ``ALL``)."""
    out: dict[str, Counter[str]] = {p: Counter() for p in data.plan.presets}
    for o in data.by_id.values():
        if isinstance(o, PairOutcomeV2) and o.valid and o.episode is not None:
            out[o.preset][f"{o.episode.kind}_{o.episode.category}"] += 1
    total: Counter[str] = Counter()
    for c in out.values():
        total.update(c)
    return {**{p: dict(c) for p, c in out.items()}, "ALL": dict(total)}


def gentle_startles(data: Outcomes) -> dict[str, dict[str, list[int]]]:
    """Per temperament and gentle stimulus: [times startled or froze, trials]."""
    out: dict[str, dict[str, list[int]]] = {}
    for o in data.by_id.values():
        if not isinstance(o, PairOutcomeV2) or not o.valid or o.stimulus not in S.LIFE_STIM_ORDER:
            continue
        if o.stimulus not in spec.GENTLE_STIMULI:
            continue
        cell = out.setdefault(o.preset, {}).setdefault(o.stimulus, [0, 0])
        cell[0] += int(o.startled_stim)
        cell[1] += 1
    return out


def life_spread(
    lives: Mapping[LV.Key, Sequence[LV.Life]], presets: Sequence[str]
) -> dict[str, dict[str, float]]:
    """How far apart the temperaments' profiles are (mean distance between their mean profiles) and
    how far a seed is from its own temperament's mean (mean distance), for the time-share part and
    the reaction part of the profile separately — what decides the nearest-centroid guess."""
    usable = {k: v for k, v in lives.items() if len(v) == len(S.RUNS)}
    if not usable:
        return {}
    names = LV.behaviour_names(seg for segs in usable.values() for seg in segs)
    profiles = {k: LV.profile(v, names) for k, v in usable.items()}
    n = len(names)
    out: dict[str, dict[str, float]] = {}
    for label, lo, hi in (
        ("time_shares", 0, n),
        ("reactions", n, n + 3 * len(S.LIFE_STIM_ORDER)),
    ):
        cents = {
            p: LV.centroid([v[lo:hi] for k, v in profiles.items() if k[0] == p]) for p in presets
        }
        within = [LV.euclid(v[lo:hi], cents[k[0]]) for k, v in profiles.items() if k[0] in cents]
        between = [
            LV.euclid(cents[a], cents[b]) for i, a in enumerate(presets) for b in presets[i + 1 :]
        ]
        out[label] = {
            "dims": hi - lo,
            "within_mean": round(statistics.mean(within), 3),
            "between_mean": round(statistics.mean(between), 3),
        }
    out["behaviours"] = {"dims": n}
    return out
