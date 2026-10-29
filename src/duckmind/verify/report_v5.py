"""From recorded traces, lives and push trials to v5 rows, the verdict matrix and plain tables (pure; no file access).

Beside ``report_v3.py`` and ``report_v4.py`` (untouched): the pairing reads M1 of a stimulus with a place with v5's rule
(``outcomes_v5``), the rows come from ``aggregate_v5``; the tables and the text helpers are v3's, except the M1 cell table, which
names the v5 rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import aggregate_v5 as A5
from duckmind.verify import lives as LV
from duckmind.verify import outcomes_v6 as O6
from duckmind.verify import outcomes_v7 as O7
from duckmind.verify import push_v3 as P
from duckmind.verify import report_v3 as R3
from duckmind.verify import spec_v5 as S
from duckmind.verify.outcomes_v5 import PairOutcomeV5, pair_outcome
from duckmind.verify.trace import Trace

group_lives = R3.group_lives
pct = R3.pct
push_pairs = R3.push_pairs
fmt_value = R3.fmt_value
rows_markdown = R3.rows_markdown
matrix_markdown = R3.matrix_markdown
to_json = R3.to_json
m1_cells = R3.m1_cells
m1_roll_up = R3.m1_roll_up
cell_bits = R3.cell_bits
key_numbers_markdown = R3.key_numbers_markdown


def pair_up(traces: Iterable[Trace], rule: str = "v5") -> tuple[list[PairOutcomeV5], list[str]]:
    """Pair each trial's stim arm with its same-seed sham by the pair id and read M1 with v5's rule (``rule`` ``v5.1``: the errata's;
    ``v6``: the errata's with a threat target selecting the object too; ``v7``: v6's, or a visual reflex whose source track is the object)."""
    arms: dict[str, dict[str, Trace]] = {}
    for tr in traces:
        arms.setdefault(tr["pair"], {})[tr["arm"]] = tr
    outcomes: list[PairOutcomeV5] = []
    broken: list[str] = []
    for pid, pair in sorted(arms.items()):
        if "stim" not in pair or "sham" not in pair:
            broken.append(pid)
            continue
        if rule == "v7":
            outcomes.append(O7.pair_outcome(pair["stim"], pair["sham"]))
        elif rule == "v6":
            outcomes.append(O6.pair_outcome(pair["stim"], pair["sham"]))
        else:
            outcomes.append(pair_outcome(pair["stim"], pair["sham"], carry_only=rule == "v5.1"))
    return outcomes, broken


def evaluate(
    traces: Iterable[Trace],
    plan: A.Plan,
    *,
    lives: Iterable[LV.Life] | None = None,
    push_planned: bool = False,
    rule: str = "v5",
) -> tuple[A.Outcomes, list[A.Row], dict[str, P.PushOutcome]]:
    """Rows of one layer, as ``report_v3.evaluate`` with v5's M1 (``rule`` ``v5.1``: the errata's start rule; ``v6``: and threat targets)."""
    traces = list(traces)
    ordinary = [t for t in traces if t["stimulus"] != S.PUSH]
    pushes = push_pairs(t for t in traces if t["stimulus"] == S.PUSH)
    outcomes, _broken = pair_up(ordinary, rule)
    data = A.Outcomes(plan, outcomes)
    grouped, problems = (None, None) if lives is None else group_lives(lives)
    planned = (
        {f"{plan.layer}|{p}|{s}|{S.PUSH}": p for p in plan.presets for s in plan.seeds}
        if push_planned
        else None
    )
    rows = A5.all_rows(
        data,
        l2_traces=ordinary if plan.layer == "L2" else None,
        lives=grouped,
        life_problems=problems,
        pushes=pushes if push_planned else None,
        planned_pushes=planned,
        rule=rule,
    )
    return data, rows, pushes


def _num(v: float | None) -> str:
    return "—" if v is None else pct(v, 0)


def m1_tables_markdown(
    rows_by_layer: Mapping[str, Sequence[A.Row]], presets: Sequence[str], rule: str = "v5"
) -> str:
    """The M1 cells of every layer as tables (stimuli down, temperaments across; the physics layer one column) — v3's table with the
    v5 rule named."""
    out: list[str] = []
    for layer in sorted(rows_by_layer):
        cells = R3.m1_cells(rows_by_layer[layer])
        if not cells:
            continue
        scopes = [s for s in (*presets, A3.SCOPE_ALL) if any(c["scope"] == s for c in cells)]
        out.append(f"### {layer}：M1 每一格（有刺激組注意率 ／ 對照組 ／ 配對差，格的判定）\n")
        out.append("| 刺激 | 規則 | " + " | ".join(scopes) + " |")
        out.append("|---|---|" + "---|" * len(scopes))
        for stimulus in S.STIMULI:
            line = [
                f"| {stimulus}",
                (
                    (
                        "選為目標（0.3 秒、刺激前一個 tick 沒選著）"
                        if rule == "v5.1"
                        else (
                            "選為目標（thing 或 threat、0.3 秒、刺激前一個 tick 沒選著）"
                            if rule == "v6"
                            else (
                                "選為目標（同第 6 版），或視覺反射的來源追蹤是它"
                                if rule == "v7"
                                else "選為目標（0.3 秒、刺激後才開始）"
                            )
                        )
                    )
                    if stimulus in S.DIRECTIONAL
                    else "3 秒內有反射"
                ),
            ]
            for scope in scopes:
                c = next(
                    (x for x in cells if x["scope"] == scope and x["stimulus"] == stimulus), None
                )
                if c is None:
                    line.append("—")
                    continue
                diff = "—" if c["diff"] is None else f"{c['diff']:+.2f}"
                unk = f"，未知 {c['unknown']}" if c["unknown"] else ""
                ex = f"，排除 {c['excluded']}" if c["excluded"] else ""
                line.append(
                    f"**{c['verdict']}**<br>{_num(c['p_stim'])} ／ {_num(c['p_sham'])} ／ {diff}"
                    f"<br>N={c['n_pairs']}{unk}{ex}"
                )
            out.append(" | ".join(line) + " |")
        out.append("")
    return "\n".join(out)
