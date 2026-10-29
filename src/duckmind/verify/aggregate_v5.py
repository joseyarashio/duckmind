"""Metric rows and verdicts of MACHINE-VERIFY-v5.md.

Beside ``aggregate_v3.py`` and ``aggregate_v4.py`` (untouched). Everything is v3's own function run on the v5 outcomes (M1's gates,
the cells, the unknown handling and every other metric are v3's); what v5 writes is M1's rule label for a stimulus with a place, and
the reference row of the known limit: for the physics layer's ball, how many pairs show a reflex or a flee in the 3 s window (not a
gate, verdict ``報告``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import lives as LV
from duckmind.verify import push_v3 as P
from duckmind.verify import spec_v5 as S
from duckmind.verify.aggregate import FAIL, INCOMPLETE, PASS, REPORT, Outcomes, Plan, Row
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.outcomes_v5 import PairOutcomeV5
from duckmind.verify.trace import Trace

SCOPE_ALL = A3.SCOPE_ALL
V3_RULE = "頭轉向真實方向"
V5_RULE = "追蹤標籤配對：大腦把真實物體選為目標（連續 0.3 秒，刺激之後才開始）"
V51_RULE = "追蹤標籤配對：大腦把真實物體選為目標（連續 0.3 秒，刺激前一個 tick 沒有選著）"
V6_RULE = "追蹤標籤配對：大腦把真實物體選為目標（目標種類 thing 或 threat，連續 0.3 秒，刺激前一個 tick 沒有選著）"
V7_RULE = (
    V6_RULE
    + "，或視覺反射（startle／freeze／orient／withdraw）的來源追蹤對到真實物體（反射起頭在刺激 tick 或之後）"
)
RULES = {"v5": V5_RULE, "v5.1": V51_RULE, "v6": V6_RULE, "v7": V7_RULE}
INFO_PART = "（報告）{stimulus}：{arm}3 秒內出現反射或逃開的對數（參考，不當門檻）"


def _relabel(row: Row, rule: str = "v5") -> Row:
    """v3 names the rule of a stimulus with a place 'head turned towards the true direction'; v5's (v5.1's, v6's) is the target selection."""
    label = RULES[rule]
    extra = dict(row.extra)
    if extra.get("rule") == V3_RULE:
        extra["rule"] = label
    note = row.note.replace(f"規則：{V3_RULE}", f"規則：{label}")
    return dataclasses.replace(row, note=note, extra=extra)


def m1_cell_rows(
    layer: str,
    scope: str,
    stimulus: str,
    good: Sequence[PairOutcomeV2],
    bad: Sequence[str],
    rule: str = "v5",
) -> list[Row]:
    """v3's three rows of one M1 cell, with the v5 (or v5.1) rule's name."""
    return [_relabel(r, rule) for r in A3.m1_cell_rows(layer, scope, stimulus, good, bad)]


def m1_cells(data: Outcomes, rule: str = "v5") -> list[Row]:
    """M1 per cell (v3's cells and gates; the pairs carry v5's ``attend_*``)."""
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
                rule,
            )
    return rows


def info_rows(data: Outcomes) -> list[Row]:
    """The known-limit reference of the physics layer's ball cell: of the pairs that were in view, how many show a reflex or a flee
    in the 3 s window, in the stimulated arm and in the control arm (their reflexes and flees come from the brain's own record).
    Verdict ``報告``: it is not a gate."""
    if data.plan.layer != "L2" or S.INFO_STIMULUS not in data.plan.stimuli:
        return []
    good, bad = data.split(data.select(preset=None, stimuli=[S.INFO_STIMULUS]))
    seen = [
        o
        for o in good
        if isinstance(o, PairOutcomeV5) and o.perceivable is True and o.esc_stim is not None
    ]
    n = len(seen)
    rows: list[Row] = []
    for arm, label in (("stim", "有刺激組"), ("sham", "對照組")):
        escapes = [o.esc_stim if arm == "stim" else o.esc_sham for o in seen]
        reflex = sum(1 for e in escapes if e is not None and e.reflex)
        flee = sum(1 for e in escapes if e is not None and e.flee)
        either = sum(1 for e in escapes if e is not None and e.either)
        rows.append(
            Row(
                data.plan.layer,
                "M1",
                INFO_PART.format(stimulus=S.INFO_STIMULUS, arm=label),
                SCOPE_ALL,
                either / n if n else None,
                n,
                either,
                n - either,
                len(bad),
                sum(1 for o in good if o.perceivable is False),
                "（不當門檻）",
                REPORT,
                note=(
                    "已知限制：大腦的「威脅」「地點」目標沒有物體身分，被球嚇到逃開歸不到那顆球；這一列只數次數。"
                    f"其中有反射 {reflex} 對、有逃開（目標種類 threat，或行為 flee／retreat）{flee} 對"
                ),
                extra={
                    "arm": arm,
                    "reflex": reflex,
                    "flee": flee,
                    "either": either,
                    "n": n,
                    "stimulus": S.INFO_STIMULUS,
                },
            )
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
    rule: str = "v5",
) -> list[Row]:
    """v3's rows with M1 per cell under v5's (or v5.1's) rule label, and the reference row after it (M2–M7 are v3's functions, unchanged)."""
    rows = A3.all_rows(
        data,
        l2_traces=l2_traces,
        lives=lives,
        life_problems=life_problems,
        pushes=pushes,
        planned_pushes=planned_pushes,
    )
    mine = m1_cells(data, rule) + info_rows(data)
    out: list[Row] = []
    placed = False
    for r in rows:
        if r.metric == "M1":
            if not placed:  # v5's M1 rows take the place of v3's, in the same position
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
    "V51_RULE",
    "V5_RULE",
    "V6_RULE",
    "V7_RULE",
    "Plan",
    "Row",
    "all_rows",
    "cell_verdicts",
    "info_rows",
    "m1_cell_rows",
    "m1_cells",
]
