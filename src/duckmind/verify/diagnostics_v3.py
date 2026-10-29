"""Facts that explain the verdicts of MACHINE-VERIFY-v3.md but are not part of them (the spec gives
them no threshold): which reflexes the duck showed when an M1 cell failed, what the push trials looked
like, how far apart the z-scored profiles of the temperaments are, what the delayed ear changed. Pure
functions over recorded traces, lives and pair outcomes; the report quotes them, no verdict reads them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import lives as LV
from duckmind.verify import lives_v3 as LZ
from duckmind.verify import push_v3 as P
from duckmind.verify import spec_v3 as S
from duckmind.verify.outcomes_v2 import validate_v2
from duckmind.verify.trace import Trace, window


def cell_reflexes(traces: Iterable[Trace], *, merge_presets: bool) -> dict[str, dict[str, Any]]:
    """For every M1 cell (key ``scope|stimulus``): the number of trials whose 3 s window held each
    reflex, in the stimulated arm and in the control arm, and how many trials of each kind there were
    (the arms with a recorded error are left out)."""
    out: dict[str, dict[str, Any]] = {}
    for tr in traces:
        if tr["stimulus"] == S.PUSH or validate_v2(tr):
            continue
        scope = A3_ALL if merge_presets else tr["preset"]
        cell = out.setdefault(
            f"{scope}|{tr['stimulus']}",
            {"stim": Counter(), "sham": Counter(), "n": {"stim": 0, "sham": 0}},
        )
        arm = tr["arm"]
        cell["n"][arm] += 1
        names = {
            str(tr["cols"]["reflex"][i])
            for i in window(tr, S.WINDOW_S)
            if tr["cols"]["reflex"][i] is not None
        }
        cell[arm].update(names or {"（沒有反射）"})

    def by_count(c: Counter[str]) -> dict[str, int]:
        """Most trials first; equal counts in name order (the output must not depend on set order)."""
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        k: {"stim": by_count(v["stim"]), "sham": by_count(v["sham"]), "n": v["n"]}
        for k, v in sorted(out.items())
    }


A3_ALL = "ALL"


def push_summary(pushes: Mapping[str, P.PushOutcome], presets: Sequence[str]) -> dict[str, Any]:
    """Per temperament: how the pushes came out, and for the ones that tipped the duck how long it was
    down, which reflex took over first and how soon."""
    out: dict[str, Any] = {}
    for preset in (*presets, "ALL"):
        mine = [o for o in pushes.values() if o.valid and (preset == "ALL" or o.preset == preset)]
        tipped = [o for o in mine if o.status == P.TIPPED]
        delays = [o.first_reflex_s for o in tipped if o.first_reflex_s is not None]
        down = [o.down_s for o in tipped if o.down_s is not None]
        out[preset] = {
            "pushes": len(mine),
            "statuses": dict(Counter(o.status for o in mine)),
            "invalid": sum(
                1
                for o in pushes.values()
                if not o.valid and (preset == "ALL" or o.preset == preset)
            ),
            "first_reflex": dict(Counter(o.first_reflex for o in tipped)),
            "median_first_reflex_s": round(statistics.median(delays), 2) if delays else None,
            "median_down_s": round(statistics.median(down), 2) if down else None,
            "max_down_s": round(max(down), 2) if down else None,
            "righting_in_grace": sum(1 for o in tipped if o.righting),
            "walked_after_grace": sum(1 for o in tipped if o.walked),
        }
    return out


def push_rows(pushes: Mapping[str, P.PushOutcome]) -> list[dict[str, Any]]:
    """One line per push (every temperament and seed, in order): what happened, for the report's table."""
    out: list[dict[str, Any]] = []
    for pid, o in sorted(pushes.items(), key=lambda kv: (kv[1].preset, kv[1].seed)):
        out.append(
            {
                "pair": pid,
                "preset": o.preset,
                "seed": o.seed,
                "side": "left" if o.seed % 2 == 0 else "right",
                "valid": o.valid,
                "status": o.status,
                "down_s": None if o.down_s is None else round(o.down_s, 2),
                "first_reflex": o.first_reflex,
                "first_reflex_s": None if o.first_reflex_s is None else round(o.first_reflex_s, 2),
                "righting": o.righting,
                "walked": o.walked,
                "blocked": o.blocked,
                "problems": o.problems,
            }
        )
    return out


def life_spread_z(
    lives: Mapping[LV.Key, Sequence[LV.Life]], presets: Sequence[str]
) -> dict[str, dict[str, float]]:
    """How far apart the temperaments are in z-score space (the statistics of ALL the seeds here: this is a
    description of the data, not a guess): the mean distance from a seed to its own temperament's mean
    and the mean distance between the temperaments' means, for the time-share part and the reaction part
    of the profile separately."""
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
        part = {k: v[lo:hi] for k, v in profiles.items()}
        fit = LZ.fit_z(list(part.values()))
        z = {k: LZ.apply_z(v, fit) for k, v in part.items()}
        cents = {p: LV.centroid([v for k, v in z.items() if k[0] == p]) for p in presets}
        within = [LV.euclid(v, cents[k[0]]) for k, v in z.items() if k[0] in cents]
        between = [
            LV.euclid(cents[a], cents[b]) for i, a in enumerate(presets) for b in presets[i + 1 :]
        ]
        out[label] = {
            "dims": hi - lo,
            "kept_dims": len(fit.keep),
            "within_mean": round(statistics.mean(within), 3),
            "between_mean": round(statistics.mean(between), 3),
        }
    out["behaviours"] = {"dims": n}
    return out


def ear_changes(
    standard: Sequence[A.Row], delayed: Sequence[A.Row], metrics: Sequence[str] = ("M1", "M2", "M3")
) -> list[dict[str, Any]]:
    """Rows (of the metrics) whose verdict or numbers differ between the standard brain-level run and the
    delayed-ear run: the same row (metric, scope, what is measured) in both."""
    old = {(r.metric, r.scope, r.part): r for r in standard if r.metric in metrics}
    out: list[dict[str, Any]] = []
    for r in delayed:
        if r.metric not in metrics:
            continue
        o = old.get((r.metric, r.scope, r.part))
        if o is None:
            continue
        if (o.n, o.ok, o.fail, o.unknown, o.na, o.verdict) == (
            r.n,
            r.ok,
            r.fail,
            r.unknown,
            r.na,
            r.verdict,
        ) and o.value == r.value:
            continue
        out.append(
            {
                "metric": r.metric,
                "scope": r.scope,
                "part": r.part,
                "standard": {"value": o.value, "n": o.n, "ok": o.ok, "verdict": o.verdict},
                "delayed": {"value": r.value, "n": r.n, "ok": r.ok, "verdict": r.verdict},
            }
        )
    return out
