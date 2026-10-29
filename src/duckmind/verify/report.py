"""From recorded traces to rows, the verdict matrix and plain tables (pure; no file access)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify.outcomes import PairOutcome, pair_outcome
from duckmind.verify.trace import Trace


def pair_up(traces: Iterable[Trace]) -> tuple[list[PairOutcome], list[str]]:
    """Pair each trial's stim arm with its same-seed sham. Returns (outcomes, ids of pairs
    with a missing arm). A pair is matched by its pair id (layer, temperament, seed, stimulus)
    — never by anything the brain wrote (track ids)."""
    arms: dict[str, dict[str, Trace]] = {}
    for tr in traces:
        arms.setdefault(tr["pair"], {})[tr["arm"]] = tr
    outcomes: list[PairOutcome] = []
    broken: list[str] = []
    for pid, pair in sorted(arms.items()):
        if "stim" not in pair or "sham" not in pair:
            broken.append(pid)
            continue
        outcomes.append(pair_outcome(pair["stim"], pair["sham"]))
    return outcomes, broken


def evaluate(traces: Iterable[Trace], plan: A.Plan) -> tuple[A.Outcomes, list[A.Row]]:
    outcomes, _broken = pair_up(traces)
    data = A.Outcomes(plan, outcomes)
    return data, A.all_rows(data)


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def fmt_value(row: A.Row) -> str:
    if row.value is None:
        return "—"
    if row.metric == "M7":
        return f"{row.value:+.0f}" if "跌倒" in row.part else f"{row.value:.0f}"
    if row.metric == "走過去":
        return f"{row.value:.2f} m"
    if row.metric == "M6" and "vs" in row.scope:
        return f"{row.value:.3f}"
    if row.metric == "M2":
        return f"{row.value:+.2f}"
    return pct(row.value, 1)


def matrix(rows: Sequence[A.Row]) -> dict[tuple[str, str, str], str]:
    return A.summary(rows)


def rows_markdown(rows: Sequence[A.Row], *, with_fails: bool = True) -> str:
    """One table of every row: layer, metric, scope, what, value, N, ok/fail/unknown/na, bar, verdict."""
    out = [
        "| 層 | 指標 | 對象 | 量什麼 | 值 | N | 成功 | 失敗 | 未知 | 排除 | 門檻 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.layer} | {r.metric} | {r.scope} | {r.part} | {fmt_value(r)} | {r.n} | {r.ok} | "
            f"{r.fail} | {r.unknown} | {r.na} | {r.threshold} | **{r.verdict}** |"
        )
    text = "\n".join(out)
    if with_fails:
        lines = []
        for r in rows:
            if r.fails and r.verdict in (A.FAIL, A.INCOMPLETE):
                shown = "；".join(r.fails[:40])
                more = f"（共 {len(r.fails)} 筆，只列前 40）" if len(r.fails) > 40 else ""
                lines.append(f"- {r.layer} {r.metric} {r.scope}「{r.part}」：{shown}{more}")
        if lines:
            text += "\n\n失敗的回合（種子）：\n" + "\n".join(lines)
    return text


def verdict_matrix_markdown(rows: Sequence[A.Row], presets: Sequence[str]) -> str:
    """layer x preset x metric: the combined verdict of each metric's parts."""
    summ = A.summary(rows)
    layers = sorted({k[0] for k in summ})
    out: list[str] = []
    for layer in layers:
        header = "| 指標 | " + " | ".join(presets) + " |"
        out.append(f"### {layer}\n\n{header}\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = [summ.get((layer, metric, s), "—") for s in presets]
            if all(c == "—" for c in cells):
                cells = [summ.get((layer, metric, "ALL"), "—")] + ["—"] * (len(presets) - 1)
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        m2 = {k[2]: v for k, v in summ.items() if k[0] == layer and k[1] == "M2"}
        if m2:
            out.append(
                "\nM2（每種刺激，合併所有個性）："
                + "、".join(f"{s} {v}" for s, v in sorted(m2.items()))
            )
        pairs = {
            k[2]: v for k, v in summ.items() if k[0] == layer and k[1] == "M6" and " vs " in k[2]
        }
        if pairs:
            out.append("\nM6 個性配對：" + "、".join(f"{s} {v}" for s, v in sorted(pairs.items())))
        out.append("")
    return "\n".join(out)


def to_json(rows: Sequence[A.Row]) -> list[dict[str, Any]]:
    return [r.to_json() for r in rows]
