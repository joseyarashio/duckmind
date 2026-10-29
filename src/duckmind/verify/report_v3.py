"""From recorded v3 traces, lives and push trials to rows, the verdict matrix and plain tables (pure; no
file access).

Beside ``report.py`` (v1) and ``report_v2.py`` (v2), both untouched."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import lives as LV
from duckmind.verify import push_v3 as P
from duckmind.verify import report_v2 as R2
from duckmind.verify import spec_v3 as S
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.outcomes_v3 import pair_outcome
from duckmind.verify.trace import Trace

group_lives = R2.group_lives
pct = R2.pct


def pair_up(traces: Iterable[Trace]) -> tuple[list[PairOutcomeV2], list[str]]:
    """Pair each trial's stim arm with its same-seed sham by the pair id (layer, temperament,
    seed, stimulus) and read M1 with v3's rule. Returns (outcomes, ids of pairs with a missing arm)."""
    arms: dict[str, dict[str, Trace]] = {}
    for tr in traces:
        arms.setdefault(tr["pair"], {})[tr["arm"]] = tr
    outcomes: list[PairOutcomeV2] = []
    broken: list[str] = []
    for pid, pair in sorted(arms.items()):
        if "stim" not in pair or "sham" not in pair:
            broken.append(pid)
            continue
        outcomes.append(pair_outcome(pair["stim"], pair["sham"]))
    return outcomes, broken


def push_pairs(traces: Iterable[Trace]) -> dict[str, P.PushOutcome]:
    """Each push trial (the pushed arm and the arm that is not) read by ``push_v3``."""
    arms: dict[str, dict[str, Trace]] = {}
    for tr in traces:
        arms.setdefault(tr["pair"], {})[tr["arm"]] = tr
    out: dict[str, P.PushOutcome] = {}
    for pid, pair in sorted(arms.items()):
        if "stim" in pair and "sham" in pair:
            out[pid] = P.push_outcome(pair["stim"], pair["sham"])
    return out


def evaluate(
    traces: Iterable[Trace],
    plan: A.Plan,
    *,
    lives: Iterable[LV.Life] | None = None,
    push_planned: bool = False,
) -> tuple[A.Outcomes, list[A.Row], dict[str, P.PushOutcome]]:
    """Rows of one layer. ``plan.stimuli`` must not hold ``push``: the push trials in ``traces`` (if
    any) are read apart and only reach M7."""
    traces = list(traces)
    ordinary = [t for t in traces if t["stimulus"] != S.PUSH]
    pushes = push_pairs(t for t in traces if t["stimulus"] == S.PUSH)
    outcomes, _broken = pair_up(ordinary)
    data = A.Outcomes(plan, outcomes)
    grouped, problems = (None, None) if lives is None else group_lives(lives)
    planned = (
        {f"{plan.layer}|{p}|{s}|{S.PUSH}": p for p in plan.presets for s in plan.seeds}
        if push_planned
        else None
    )
    rows = A3.all_rows(
        data,
        l2_traces=ordinary if plan.layer == "L2" else None,
        lives=grouped,
        life_problems=problems,
        pushes=pushes if push_planned else None,
        planned_pushes=planned,
    )
    return data, rows, pushes


# ── text ───────────────────────────────────────────────────────────────────────────


def fmt_value(row: A.Row) -> str:
    if row.value is None:
        return "—"
    if row.metric == "M7":
        return pct(row.value, 0) if "結果" in row.part else f"{row.value:.0f}"
    if row.metric == "走過去":
        return f"{row.value:.2f} m"
    if row.metric in ("M1", "M2") and ("差距" in row.part or "−" in row.part):
        return f"{row.value:+.2f}"
    return pct(row.value, 1)


def rows_markdown(rows: Sequence[A.Row], *, with_fails: bool = True) -> str:
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


def matrix_markdown(rows: Sequence[A.Row], presets: Sequence[str]) -> str:
    """layer x preset x metric: the combined verdict of each metric's rows (the temperament's own and the
    ones of everybody); a layer shows only the metrics it has rows for (the delayed-ear run has M1-M3)."""
    cells = A3.cell_verdicts(rows, presets)
    out: list[str] = []
    for layer in sorted({k[0] for k in cells}):
        have = {r.metric for r in rows if r.layer == layer}
        out.append(
            f"### {layer}\n\n| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets)
        )
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            if metric not in have:
                continue
            out.append(
                f"| {metric} | "
                + " | ".join(cells.get((layer, metric, p), "—") for p in presets)
                + " |"
            )
        m2 = {
            r.scope: r.verdict
            for r in rows
            if r.layer == layer and r.metric == "M2" and r.verdict != A.REPORT
        }
        if m2:
            out.append(
                "\nM2（每種刺激，合併所有個性）："
                + "、".join(f"{s} {v}" for s, v in sorted(m2.items()))
            )
        out.append("")
    return "\n".join(out)


to_json = R2.to_json


# ── M1 cells ───────────────────────────────────────────────────────────────────────


def m1_cells(rows: Sequence[A.Row]) -> list[dict[str, Any]]:
    """One dict per M1 cell: its two gates, its numbers and its verdict (the two gates together)."""
    out: list[dict[str, Any]] = []
    gated = [r for r in rows if r.metric == "M1" and r.verdict != A.REPORT]
    by_key: dict[tuple[str, str], dict[str, A.Row]] = {}
    for r in gated:
        stimulus = r.part.split("：", 1)[0]
        kind = "rate" if r.part.endswith("有刺激組注意率") else "diff"
        by_key.setdefault((r.scope, stimulus), {})[kind] = r
    for (scope, stimulus), pair in by_key.items():
        a, d = pair.get("rate"), pair.get("diff")
        if a is None or d is None:
            continue
        extra = d.extra
        out.append(
            {
                "layer": a.layer,
                "scope": scope,
                "stimulus": stimulus,
                "rule": extra.get("rule", ""),
                "pairs": extra.get("pairs", 0),
                "n": a.n,
                "ok": a.ok,
                "n_pairs": extra.get("n_pairs", 0),
                "p_stim": extra.get("p_stim"),
                "p_sham": extra.get("p_sham"),
                "diff": d.value,
                "unknown": a.unknown,
                "excluded": a.na,
                "verdict_rate": a.verdict,
                "verdict_diff": d.verdict,
                "verdict": A.combine([a.verdict, d.verdict]),
                "fails_rate": a.fails,
                "fails_diff": d.fails,
            }
        )
    return out


def _num(v: float | None) -> str:
    return "—" if v is None else pct(v, 0)


def m1_tables_markdown(rows_by_layer: Mapping[str, Sequence[A.Row]], presets: Sequence[str]) -> str:
    """The M1 cells of every layer as tables: stimuli down, temperaments across (the brain layers) or
    one column (the physics layer)."""
    out: list[str] = []
    for layer in sorted(rows_by_layer):
        cells = m1_cells(rows_by_layer[layer])
        if not cells:
            continue
        scopes = [s for s in (*presets, A3.SCOPE_ALL) if any(c["scope"] == s for c in cells)]
        out.append(f"### {layer}：M1 每一格（有刺激組注意率 ／ 對照組 ／ 配對差，格的判定）\n")
        out.append("| 刺激 | 規則 | " + " | ".join(scopes) + " |")
        out.append("|---|---|" + "---|" * len(scopes))
        for stimulus in S.STIMULI:
            line = [f"| {stimulus}", "頭轉向" if stimulus in S.DIRECTIONAL else "3 秒內有反射"]
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


def m1_roll_up(rows: Sequence[A.Row], scope: str) -> dict[str, int]:
    """How many of a temperament's (or ``ALL``'s) M1 cells pass, fail, are incomplete."""
    counts = {A.PASS: 0, A.FAIL: 0, A.INCOMPLETE: 0}
    for c in m1_cells(rows):
        if c["scope"] == scope:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    return counts


# ── the key numbers behind each cell of the verdict matrix ─────────────────────────────


def _find(rows: Sequence[A.Row], metric: str, scope: str, part: str) -> A.Row | None:
    for r in rows:
        if r.metric == metric and r.scope == scope and r.part.startswith(part):
            return r
    return None


def cell_bits(lr: Sequence[A.Row], metric: str, preset: str, verdict: str = "") -> list[str]:
    """The numbers one cell of the verdict matrix was judged on (rows of one layer)."""

    def f(part: str, scope: str = preset) -> A.Row | None:
        return _find(lr, metric, scope, part)

    bits: list[str] = []
    if metric == "M1":
        counts = {A.PASS: 0, A.FAIL: 0, A.INCOMPLETE: 0}
        for scope in (preset, A3.SCOPE_ALL):
            for k, v in m1_roll_up(lr, scope).items():
                counts[k] += v
        total = sum(counts.values())
        if total:
            bits = [
                f"{total} 格：過 {counts[A.PASS]}",
                f"沒過 {counts[A.FAIL]}",
                f"判不出 {counts[A.INCOMPLETE]}",
            ]
    elif metric == "M3":
        r = f("溫和刺激")
        bits = [f"溫和 {R2._num(r)}"]
        for kind in ("bang", "loom"):
            r = f(f"很大聲／逼近：{kind}")
            if r is not None and r.verdict != A.UNTESTED:
                bits.append(f"{kind} {R2._num(r)}")
            elif r is not None:
                bits.append(f"{kind} 未測")
    elif metric == "M4":
        r = f("腦說")
        bits = [f"方位 {R2._num(f('方位誤差'))}", f"距離 {R2._num(f('距離誤差'))}"]
        if r is not None:
            bits.append(f"到了 {r.ok}/{r.n + r.unknown}")
    elif metric == "M5":
        n, o = f("刺激引起的那段：自然"), f("10 秒內")
        if n is not None:
            bits = [f"自然 {R2._num(n)}（N={n.n}）", f"重開 {R2._num(o)}"]
    elif metric == "M6":
        ga = _find(lr, "M6", "ALL", "猜個性（z 分數）：") or _find(
            lr, "M6", "ALL", "（報告）猜個性（z 分數）："
        )
        own = f("這個個性自己被猜對") or f("（報告）這個個性自己被猜對")
        sig = f("招牌動作") or f("（報告）招牌動作")
        bits = [f"猜對 {R2._num(ga)}", f"自己 {R2._num(own)}", f"招牌 {R2._num(sig)}"]
    elif metric == "M7":
        fall = f("有刺激組跌倒數")
        push = f("推一下：安全反射被擋")
        if fall is not None and fall.value is not None:
            bits = [f"跌倒差 {fall.value:+.0f}（N={fall.n}）"]
        if push is not None:
            bits.append(f"推倒 {push.n}/{push.extra.get('pushes', '?')}，被擋 {push.fail}")
    if not bits and verdict == A.UNTESTED:
        bits = ["這一層沒量"]
    return bits


def key_numbers_markdown(rows: Sequence[A.Row], presets: Sequence[str]) -> str:
    """One table per layer: the verdict of each metric for each temperament with the numbers it was
    judged on (the same numbers as in the rows, only chosen and put in one cell)."""
    cells = A3.cell_verdicts(rows, presets)
    out: list[str] = []
    for layer in sorted({r.layer for r in rows}):
        lr = [r for r in rows if r.layer == layer]
        out.append(
            f"### {layer}\n\n| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets)
        )
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            if not any(r.metric == metric for r in lr):
                continue
            line = [f"| {metric}"]
            for p in presets:
                verdict = cells.get((layer, metric, p), "—")
                bits = cell_bits(lr, metric, p, verdict)
                line.append(f"**{verdict}**<br>" + "<br>".join(bits))
            out.append(" | ".join(line) + " |")
        out.append("")
    return "\n".join(out)


# ── v2 beside v3 (the same code) ───────────────────────────────────────────────────────


def v2_v3_markdown(
    v2_rows: Mapping[str, Sequence[A.Row]],
    v3_rows: Mapping[str, Sequence[A.Row]],
    presets: Sequence[str],
    label_a: str = "第 2 版",
    label_b: str = "第 3 版",
) -> str:
    """The verdict matrices of the two versions side by side and every row of M2–M5 and the unchanged
    parts of M3/M7 whose numbers moved. M1, M6 and M7's push have new rows (different names), so they are
    compared by the matrix, and by ``m1_compare_markdown``."""
    return R2.side_by_side_markdown(v2_rows, v3_rows, presets, label_a, label_b)


def m1_compare_markdown(
    v2_rows: Sequence[A.Row], v3_rows: Sequence[A.Row], presets: Sequence[str]
) -> str:
    """One layer's M1: v2's per-temperament numbers next to v3's cells of that temperament."""
    cells = m1_cells(v3_rows)
    out = [
        "| 個性 | 第 2 版：有刺激組注意率 | 第 2 版：配對差 | 第 2 版判定 | 第 3 版：9 格裡過／沒過／判不出 "
        "| 第 3 版：每格注意率最低–最高 | 第 3 版：每格配對差最低–最高 |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in presets:
        a = _find(v2_rows, "M1", p, "有刺激組注意率")
        d = _find(v2_rows, "M1", p, "有刺激組 − 對照組")
        mine = [c for c in cells if c["scope"] in (p, A3.SCOPE_ALL)]
        if not mine:
            continue
        counts = {
            k: sum(1 for c in mine if c["verdict"] == k) for k in (A.PASS, A.FAIL, A.INCOMPLETE)
        }
        rates = [c["p_stim"] for c in mine if c["p_stim"] is not None]
        diffs = [c["diff"] for c in mine if c["diff"] is not None]
        v2v = A.combine([x.verdict for x in (a, d) if x is not None]) if a and d else "—"
        out.append(
            f"| {p} | {_num(a.value) if a else '—'} | "
            f"{'—' if d is None or d.value is None else f'{d.value:+.2f}'} | {v2v} | "
            f"{counts[A.PASS]} ／ {counts[A.FAIL]} ／ {counts[A.INCOMPLETE]} | "
            f"{_num(min(rates)) + '–' + _num(max(rates)) if rates else '—'} | "
            f"{f'{min(diffs):+.2f} 到 {max(diffs):+.2f}' if diffs else '—'} |"
        )
    return "\n".join(out)
