"""From recorded v2 traces and lives to rows, the verdict matrix and plain tables (pure; no file access).

Beside ``report.py`` (v1, untouched)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v2 as A2
from duckmind.verify import lives as LV
from duckmind.verify.outcomes_v2 import PairOutcomeV2, pair_outcome
from duckmind.verify.trace import Trace


def pair_up(traces: Iterable[Trace]) -> tuple[list[PairOutcomeV2], list[str]]:
    """Pair each trial's stim arm with its same-seed sham by the pair id (layer, temperament,
    seed, stimulus). Returns (outcomes, ids of pairs with a missing arm)."""
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


def group_lives(
    records: Iterable[LV.Life],
) -> tuple[dict[LV.Key, list[LV.Life]], dict[LV.Key, list[str]]]:
    """Life records by (temperament, seed), day before night, with what is wrong with each."""
    order = {"day": 0, "night": 1}
    lives: dict[LV.Key, list[LV.Life]] = {}
    for rec in records:
        lives.setdefault((rec["preset"], int(rec["seed"])), []).append(rec)
    problems: dict[LV.Key, list[str]] = {}
    for key, segs in lives.items():
        segs.sort(key=lambda r: order.get(r.get("segment", ""), 9))
        names = [s.get("segment") for s in segs]
        issues = [f"{s.get('segment')}: {p}" for s in segs for p in LV.validate_life(s)]
        if names != ["day", "night"]:
            issues.append(f"segments {names}, planned ['day', 'night']")
        if issues:
            problems[key] = issues
    return lives, problems


def evaluate(
    traces: Iterable[Trace],
    plan: A.Plan,
    *,
    lives: Iterable[LV.Life] | None = None,
) -> tuple[A.Outcomes, list[A.Row]]:
    traces = list(traces)
    outcomes, _broken = pair_up(traces)
    data = A.Outcomes(plan, outcomes)
    grouped, problems = (None, None) if lives is None else group_lives(lives)
    rows = A2.all_rows(
        data,
        l2_traces=traces if plan.layer == "L2" else None,
        lives=grouped,
        life_problems=problems,
    )
    return data, rows


# ── text ───────────────────────────────────────────────────────────────────────────


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def fmt_value(row: A.Row) -> str:
    if row.value is None:
        return "—"
    if row.metric == "M7":
        return f"{row.value:+.0f}" if "跌倒" in row.part else f"{row.value:.0f}"
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
    """layer x preset x metric: the combined verdict of each metric's rows (the temperament's own
    and the ones of everybody)."""
    cells = A2.cell_verdicts(rows, presets)
    out: list[str] = []
    for layer in sorted({k[0] for k in cells}):
        out.append(
            f"### {layer}\n\n| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets)
        )
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
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


def to_json(rows: Sequence[A.Row]) -> list[dict[str, Any]]:
    return [r.to_json() for r in rows]


# ── v1 beside v2 ───────────────────────────────────────────────────────────────────


def _v1_cells(rows: Sequence[A.Row], presets: Sequence[str]) -> dict[tuple[str, str, str], str]:
    summ = A.summary(rows)
    out: dict[tuple[str, str, str], str] = {}
    for (layer, metric, scope), verdict in summ.items():
        if scope in presets:
            out[(layer, metric, scope)] = verdict
    for (layer, metric, scope), verdict in summ.items():
        if scope == "ALL":
            for p in presets:
                out.setdefault((layer, metric, p), verdict)
    return out


def compare_markdown(
    v1_rows: Mapping[str, Sequence[A.Row]],
    v2_rows: Mapping[str, Sequence[A.Row]],
    presets: Sequence[str],
) -> str:
    """The verdict of every layer x preset x metric in v1 and in v2, side by side; and the M2 /
    M3 rows, which v2 did not change, as a check that the two runs measured the same thing."""
    out: list[str] = []
    for layer in sorted(v2_rows):
        a = _v1_cells(v1_rows.get(layer, []), presets)
        b = A2.cell_verdicts(v2_rows[layer], presets)
        out.append(f"### {layer}：第 1 版 → 第 2 版\n")
        out.append("| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = []
            for p in presets:
                v1 = a.get((layer, metric, p), "—")
                v2 = b.get((layer, metric, p), "—")
                cells.append(v1 if v1 == v2 else f"{v1} → **{v2}**")
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
        same: list[str] = []
        different: list[str] = []
        old = {(r.metric, r.scope, r.part): r for r in v1_rows.get(layer, [])}
        for r in v2_rows[layer]:
            if r.metric not in ("M2", "M3", "M7"):
                continue
            o = old.get((r.metric, r.scope, r.part))
            if o is None:
                continue
            key = f"{r.metric} {r.scope}「{r.part}」"
            if o.value == r.value and (o.n, o.ok, o.fail) == (r.n, r.ok, r.fail):
                same.append(key)
            else:
                different.append(
                    f"{key}：第 1 版 {fmt_value(o)}（N={o.n}）→ 第 2 版 {fmt_value(r)}（N={r.n}）"
                )
        out.append(
            f"沒改判法的列（M2、M3、M7）：{len(same)} 列和第 1 版的數字完全一樣"
            + (f"，{len(different)} 列不同：" if different else "。")
        )
        out += [f"- {line}" for line in different]
        out.append("")
    return "\n".join(out)


# ── two v2 runs side by side (before and after a fix) ───────────────────────────────

_RANK = {A.PASS: 3, A.INCOMPLETE: 2, A.UNTESTED: 1, A.FAIL: 0, A.REPORT: 2, "—": 2}


def side_by_side_markdown(
    a_rows: Mapping[str, Sequence[A.Row]],
    b_rows: Mapping[str, Sequence[A.Row]],
    presets: Sequence[str],
    label_a: str = "前",
    label_b: str = "後",
) -> str:
    """Two v2 runs (the same spec, the code before and after a change): the verdict matrices, then every
    row whose verdict moved (worse or better), then the rows whose numbers moved with the verdict still
    the same. A row is the same when metric, scope and what-is-measured are the same."""
    out: list[str] = []
    for layer in sorted(set(a_rows) | set(b_rows)):
        ra, rb = list(a_rows.get(layer, [])), list(b_rows.get(layer, []))
        ca, cb = A2.cell_verdicts(ra, presets), A2.cell_verdicts(rb, presets)
        out.append(f"### {layer}：{label_a} → {label_b}\n")
        out.append("| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = []
            for p in presets:
                va, vb = ca.get((layer, metric, p), "—"), cb.get((layer, metric, p), "—")
                cells.append(va if va == vb else f"{va} → **{vb}**")
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
        old = {(r.metric, r.scope, r.part): r for r in ra}
        new = {(r.metric, r.scope, r.part): r for r in rb}
        worse: list[str] = []
        better: list[str] = []
        sideways: list[str] = []
        moved = 0
        for key, r in new.items():
            o = old.get(key)
            if o is None:
                continue
            same = (o.n, o.ok, o.fail, o.unknown, o.na) == (r.n, r.ok, r.fail, r.unknown, r.na)
            if o.verdict == r.verdict and same and o.value == r.value:
                continue
            line = (
                f"- {r.metric} {r.scope}「{r.part}」：{label_a} {fmt_value(o)}（N={o.n}，{o.verdict}）"
                f" → {label_b} {fmt_value(r)}（N={r.n}，{r.verdict}）"
            )
            if _RANK.get(r.verdict, 2) < _RANK.get(o.verdict, 2):
                worse.append(line)
            elif _RANK.get(r.verdict, 2) > _RANK.get(o.verdict, 2):
                better.append(line)
            else:
                sideways.append(line)
            moved += 1
        unchanged = len(new) - moved - sum(1 for k in new if k not in old)
        out.append(
            f"{layer} 共 {len(new)} 列：判定變差 {len(worse)} 列、變好 {len(better)} 列、"
            f"判定沒變但數字變了 {len(sideways)} 列、完全一樣 {unchanged} 列。\n"
        )
        for title, lines in (("變差", worse), ("變好", better), ("判定沒變、數字變了", sideways)):
            if lines:
                out.append(f"**{title}**\n")
                out += lines
                out.append("")
        gone = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        if gone or added:
            out.append(
                f"只在{label_a}有的列 {len(gone)} 列、只在{label_b}有的列 {len(added)} 列。\n"
            )
    return "\n".join(out)


# ── the key numbers behind each cell ────────────────────────────────────────────────


def _find(rows: Sequence[A.Row], metric: str, scope: str, part: str) -> A.Row | None:
    for r in rows:
        if r.metric == metric and r.scope == scope and r.part.startswith(part):
            return r
    return None


def _num(r: A.Row | None, kind: str = "pct") -> str:
    if r is None or r.value is None:
        return "—"
    if kind == "diff":
        return f"{r.value:+.2f}"
    return pct(r.value, 1)


def cell_bits(lr: Sequence[A.Row], metric: str, preset: str, verdict: str = "") -> list[str]:
    """The numbers one cell of the verdict matrix was judged on (rows of one layer)."""

    def f(part: str, scope: str = preset) -> A.Row | None:
        return _find(lr, metric, scope, part)

    bits: list[str] = []
    if metric == "M1":
        a, d = f("有刺激組注意率"), f("有刺激組 − 對照組")
        bits = [f"注意 {_num(a)}", f"配對差 {_num(d, 'diff')}"]
        if a is not None and a.unknown:
            bits.append(f"未知 {a.unknown - len(a.extra.get('missing_or_invalid', []))} 組")
    elif metric == "M3":
        bits = [f"溫和 {_num(f('溫和刺激'))}"]
        for kind in ("bang", "loom"):
            r = f(f"很大聲／逼近：{kind}")
            if r is not None and r.verdict != A.UNTESTED:
                bits.append(f"{kind} {_num(r)}")
            elif r is not None:
                bits.append(f"{kind} 未測")
    elif metric == "M4":
        r = f("腦說")
        bits = [f"方位 {_num(f('方位誤差'))}", f"距離 {_num(f('距離誤差'))}"]
        if r is not None:
            bits.append(f"到了 {r.ok}/{r.n + r.unknown}")
    elif metric == "M5":
        n, o = f("刺激引起的那段：自然"), f("10 秒內")
        if n is not None:
            bits = [f"自然 {_num(n)}（N={n.n}）", f"重開 {_num(o)}"]
    elif metric == "M6":
        ga = _find(lr, "M6", "ALL", "猜個性：") or _find(lr, "M6", "ALL", "（報告）猜個性：")
        own = f("這個個性自己被猜對") or f("（報告）這個個性自己被猜對")
        sig = f("招牌動作") or f("（報告）招牌動作")
        bits = [f"猜對 {_num(ga)}", f"自己 {_num(own)}", f"招牌 {_num(sig)}"]
    elif metric == "M7":
        fall = f("有刺激組跌倒數")
        if fall is not None and fall.value is not None:
            bits = [f"跌倒差 {fall.value:+.0f}（N={fall.n}）"]
    if not bits and verdict == A.UNTESTED:
        bits = ["這一層沒量"]
    return bits


def key_numbers_markdown(rows: Sequence[A.Row], presets: Sequence[str]) -> str:
    """One table per layer: the verdict of each metric for each temperament with the numbers it was
    judged on (the same numbers as in the rows, only chosen and put in one cell)."""
    cells = A2.cell_verdicts(rows, presets)
    out: list[str] = []
    for layer in sorted({r.layer for r in rows}):
        lr = [r for r in rows if r.layer == layer]
        out.append(
            f"### {layer}\n\n| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets)
        )
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            line = [f"| {metric}"]
            for p in presets:
                verdict = cells.get((layer, metric, p), "—")
                bits = cell_bits(lr, metric, p, verdict)
                line.append(f"**{verdict}**<br>" + "<br>".join(bits))
            out.append(" | ".join(line) + " |")
        out.append("")
    return "\n".join(out)
