"""Metric rows and verdicts of MACHINE-VERIFY-v4.md.

Beside ``aggregate_v3.py`` (untouched). Everything is v3's own function run on the v4 outcomes; the one thing v4
writes is M1's rule label: for a stimulus with a place, the cell's rule is now the fixation (CHOICES §二) instead of
v3's head turn. The thresholds, the cells, the unknown handling and every other metric are v3's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import lives as LV
from duckmind.verify import push_v3 as P
from duckmind.verify.aggregate import FAIL, INCOMPLETE, PASS, REPORT, Outcomes, Plan, Row
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.trace import Trace

SCOPE_ALL = A3.SCOPE_ALL
V3_RULE = "頭轉向真實方向"
V4_RULE = "盯住真實方向（連續 0.5 秒差距 ≤ 15°，開始時比刺激當下近）"


def _relabel(row: Row) -> Row:
    """v3 names the rule of a stimulus with a place 'head turned towards the true direction'; v4's is the fixation."""
    extra = dict(row.extra)
    if extra.get("rule") == V3_RULE:
        extra["rule"] = V4_RULE
    note = row.note.replace(f"規則：{V3_RULE}", f"規則：{V4_RULE}")
    return dataclasses.replace(row, note=note, extra=extra)


def m1_cell_rows(
    layer: str,
    scope: str,
    stimulus: str,
    good: Sequence[PairOutcomeV2],
    bad: Sequence[str],
) -> list[Row]:
    """v3's three rows of one M1 cell, with the v4 rule's name."""
    return [_relabel(r) for r in A3.m1_cell_rows(layer, scope, stimulus, good, bad)]


def m1_cells(data: Outcomes) -> list[Row]:
    """M1 per cell (v3's cells and gates; the pairs carry v4's ``attend_*``)."""
    layer = data.plan.layer
    rows: list[Row] = []
    merge = layer == "L2"
    scopes: Iterable[str | None] = [None] if merge else data.plan.presets
    for preset in scopes:
        for stimulus in data.plan.stimuli:
            good, bad = data.split(data.select(preset=preset, stimuli=[stimulus]))
            rows += m1_cell_rows(
                layer,
                SCOPE_ALL if preset is None else preset,
                stimulus,
                [o for o in good if isinstance(o, PairOutcomeV2)],
                bad,
            )
    return rows


def all_rows(
    data: Outcomes,
    *,
    l2_traces: Iterable[Trace] | None = None,
    lives: Mapping[LV.Key, Sequence[LV.Life]] | None = None,
    life_problems: Mapping[LV.Key, list[str]] | None = None,
    pushes: Mapping[str, P.PushOutcome] | None = None,
    planned_pushes: Mapping[str, str] | None = None,
) -> list[Row]:
    """v3's rows with M1 per cell under v4's rule label (M2–M7 are v3's functions, unchanged)."""
    rows = A3.all_rows(
        data,
        l2_traces=l2_traces,
        lives=lives,
        life_problems=life_problems,
        pushes=pushes,
        planned_pushes=planned_pushes,
    )
    mine = m1_cells(data)
    out: list[Row] = []
    placed = False
    for r in rows:
        if r.metric == "M1":
            if not placed:  # v4's M1 rows take the place of v3's, in the same position
                out += mine
                placed = True
            continue
        out.append(r)
    if not placed:
        out += mine
    return out


cell_verdicts = A3.cell_verdicts
m2_verdicts = A3.m2_verdicts

__all__ = [
    "FAIL",
    "INCOMPLETE",
    "PASS",
    "REPORT",
    "SCOPE_ALL",
    "Plan",
    "Row",
    "all_rows",
    "cell_verdicts",
    "m1_cell_rows",
    "m1_cells",
]
