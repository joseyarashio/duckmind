"""The owner-facing text of a v3 result, in plain Traditional Chinese, made by the program from the rows
(the same rows the verdicts are in), the diagnostics and the manifests: so that a rerun on other code gets
its own up-to-date report with one command. Nothing here judges anything: every sentence quotes a row or
a diagnostic number, and says what the metric measures.

The pieces that are plain tables (the verdict matrix, the numbers behind each cell, the M1 cells, every
row, the failing trials) are made by ``report_v3``; this module puts the words around them. M2, M4 and
M5 are v2's text (those metrics did not change).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import push_v3 as P
from duckmind.verify import report_text_v2 as RT2
from duckmind.verify import report_v3 as R
from duckmind.verify import spec_v3 as S

Rows = Mapping[str, Sequence[A.Row]]

METRIC_ZH = RT2.METRIC_ZH
GATED = RT2.GATED
EAR = S.EAR_LAYER
_p = RT2._p
_rows = RT2._rows
_one = RT2._one

STATUS_ZH = {
    P.TIPPED: "推倒了",
    P.NOT_TIPPED: "沒推倒",
    P.RECOVERED_FAST: "倒了但 0.5 秒內自己起來",
    P.BOTH_TIPPED: "對照組也倒（不算推倒的）",
    P.ALREADY_DOWN: "推的時候已經倒著",
}


def _scopes(layer: str, presets: Sequence[str]) -> list[str]:
    return [A3.SCOPE_ALL] if layer == "L2" else list(presets)


# ── 一、結論 ────────────────────────────────────────────────────────────────────────


def _conclusions(rows: Rows, presets: Sequence[str]) -> list[str]:
    groups: dict[str, list[str]] = {
        A.PASS: [],
        A.FAIL: [],
        A.INCOMPLETE: [],
        A.UNTESTED: [],
        A.REPORT: [],
    }
    all_pass: list[str] = []
    for layer in sorted(rows):
        lr = rows[layer]
        cells = A3.cell_verdicts(lr, presets)
        metrics = [m for m in GATED if any(r.metric == m for r in lr)]
        for p in presets:
            vs = [cells.get((layer, m, p), "—") for m in metrics]
            judged = [v for v in vs if v in (A.PASS, A.FAIL, A.INCOMPLETE)]
            if judged and all(v == A.PASS for v in judged) and layer != EAR:
                all_pass.append(f"{layer} {p}")
        for metric in metrics:
            by_verdict: dict[str, list[str]] = {}
            for p in presets:
                by_verdict.setdefault(cells.get((layer, metric, p), "—"), []).append(p)
            for verdict, ps in by_verdict.items():
                if verdict not in groups:
                    continue
                if verdict == A.UNTESTED:
                    note = next((r.note for r in _rows(lr, metric) if r.verdict == A.UNTESTED), "")
                    groups[verdict].append(
                        f"{layer} {metric}「{METRIC_ZH[metric]}」（{'、'.join(ps)}）：{note or '這一層沒量'}"
                    )
                    continue
                parts = []
                for p in ps:
                    bits = R.cell_bits(lr, metric, p, verdict)
                    parts.append(f"{p}（{'、'.join(bits)}）" if bits else p)
                groups[verdict].append(
                    f"{layer} {metric}「{METRIC_ZH[metric]}」：" + "；".join(parts)
                )
        by_stim = {r.scope: r for r in lr if r.metric == "M2" and r.verdict in (A.PASS, A.FAIL)}
        ok = [
            f"{s} {r.value:+.2f}"
            for s, r in by_stim.items()
            if r.verdict == A.PASS and r.value is not None
        ]
        no = [
            f"{s} {r.value:+.2f}"
            for s, r in by_stim.items()
            if r.verdict == A.FAIL and r.value is not None
        ]
        if ok:
            groups[A.PASS].append(f"{layer} M2「因果」：" + "、".join(ok))
        if no:
            groups[A.FAIL].append(f"{layer} M2「因果」：" + "、".join(no))
    out: list[str] = []
    if not all_pass:
        out.append("- **沒有任何一個個性在任何一層的所有量到的項目都過。** 照實報，沒有修門檻。")
    else:
        out.append(
            "- 所有量到的項目都過的："
            + "、".join(all_pass)
            + "。其餘的個性至少有一項沒過或判不出。"
        )
    titles = (
        (A.PASS, "過的"),
        (A.FAIL, "沒過的"),
        (
            A.INCOMPLETE,
            "判不出來（`incomplete`：缺資料、規則在那些回合沒有定義，或推倒的次數不夠；不當通過）",
        ),
        (A.UNTESTED, "這一層沒量（`未測`）"),
        (A.REPORT, "只報數字、不當門檻（`報告`）"),
    )
    for verdict, title in titles:
        if groups[verdict]:
            out.append(f"- **{title}：**")
            out += [f"  - {line}" for line in groups[verdict]]
    return out


# ── 四、第 2 版 → 第 3 版 ─────────────────────────────────────────────────────────────


def _matrix_compare(v2: Rows, v3: Rows, presets: Sequence[str]) -> list[str]:
    out: list[str] = []
    for layer in sorted(v2):
        if layer not in v3:
            continue
        a = A3.cell_verdicts(v2[layer], presets)
        b = A3.cell_verdicts(v3[layer], presets)
        out.append(f"**{layer}**：第 2 版 → 第 3 版\n")
        out.append("| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = []
            for p in presets:
                x, y = a.get((layer, metric, p), "—"), b.get((layer, metric, p), "—")
                cells.append(x if x == y else f"{x} → **{y}**")
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _row_value(r: A.Row | None) -> str:
    return "—" if r is None else f"{R.fmt_value(r)}（N={r.n}，`{r.verdict}`）"


def _same_numbers(r2: Sequence[A.Row], r3: Sequence[A.Row]) -> str:
    """Rows of the metrics v3 did not change (M2, M3, M4, M5) that have the same name in both versions:
    how many have exactly the same numbers."""
    old = {(r.metric, r.scope, r.part): r for r in r2}
    same = diff = 0
    for r in r3:
        if r.metric not in ("M2", "M3", "M4", "M5"):
            continue
        o = old.get((r.metric, r.scope, r.part))
        if o is None:
            continue
        if o.value == r.value and (o.n, o.ok, o.fail, o.unknown) == (r.n, r.ok, r.fail, r.unknown):
            same += 1
        else:
            diff += 1
    tail = "，其餘的多半是這次種子和刺激變多，分母變大（見 `compare-v2-v3.md`）" if diff else ""
    return f"第 2 版和第 3 版相同名字的列有 {same + diff} 列，{same} 列數字完全一樣、{diff} 列不同{tail}。"


def _versus_v2(v2: Rows | None, v3: Rows, presets: Sequence[str]) -> list[str]:
    if not v2:
        return ["（沒有給第 2 版在同一份程式上的結果，無法並排。）", ""]
    out = [
        "第 2 版那一側＝同一份被驗程式（見上面的 commit）用第 2 版規格跑出來的結果；第 3 版的 L1 對照回合和 L2 種子 1–4 的"
        "五種老刺激，重錄的紀錄和第 2 版逐位元組相同的比例列在第十一節的「錄製身分檢查」。",
        "",
        *_matrix_compare(v2, v3, presets),
        "每一個指標，規格改了什麼、數字怎麼變：",
        "",
    ]
    for layer in sorted(v2):
        if layer not in v3:
            continue
        r2, r3 = v2[layer], v3[layer]
        out.append(f"**{layer}**")
        out.append(
            "- M1：第 2 版把每個個性所有刺激混在一起算（逼近、摸頭、戳被當成有方向，腦層記未知），第 3 版每一格自己判。"
            "第 2 版對照第 3 版每個個性的 9 格："
            if layer == "L1"
            else "- M1：第 2 版的物理層只量 5 種刺激、每個個性所有刺激混在一起算（拍手、叫名字、巨響撐著比例），"
            "第 3 版 9 種刺激每種自己一格（4 個個性合併，每格 32 對）。第 2 版對照第 3 版（物理層的格是合併個性的，每個個性那一列的第 3 版數字相同）："
        )
        out.append("")
        out.append(R.m1_compare_markdown(r2, r3, presets))
        out.append("")
        if layer == "L1":
            s2 = _one(r2, "M6", "ALL", "猜個性：")
            s3 = _one(r3, "M6", "ALL", "猜個性（z 分數）：")
            out.append(
                f"- M6：第 2 版用原始單位猜個性，猜對 {_row_value(s2)}；第 3 版每一維先用訓練種子的平均和標準差換成 z 分數，"
                f"猜對 {_row_value(s3)}。"
            )
        else:
            out.append(
                "- M6：物理層規格沒改，照第 2 版只報數字（種子從 4 顆變 8 顆、刺激從 5 種變 9 種；用 z 分數猜）。"
            )
        m7 = [r for r in r3 if r.metric == "M7" and r.part == A3.PUSH_PART]
        if layer == "L2" and m7:
            old = [r for r in r2 if r.metric == "M7" and r.part.startswith("安全反射被擋")]
            out.append(
                "- M7：第 2 版「安全反射被擋」"
                + ("：" + "、".join(f"{r.scope} `{r.verdict}`" for r in old) if old else "")
                + "（整批沒有一次跌倒，量不到）；第 3 版加了「推一下」："
                + "、".join(f"{r.scope} 推倒 {r.n} 次、被擋 {r.fail}（`{r.verdict}`）" for r in m7)
                + "。"
            )
        n2 = RT2._range(r2, "M5", "刺激引起的那段：自然結束")
        n3 = RT2._range(r3, "M5", "刺激引起的那段：自然結束")
        out.append(
            f"- M2、M3、M4、M5：定義沒改。M5 自然結束 第 2 版 {n2}、第 3 版 {n3}"
            + (
                "（L2 的種子從 4 顆變 8 顆、刺激從 5 種變 9 種，所以 N 變大）"
                if layer == "L2"
                else ""
            )
            + "；"
            + _same_numbers(r2, r3)
        )
        out.append("")
    return out


# ── 五、真實耳朵延遲版 ────────────────────────────────────────────────────────────────


def _count(x: Sequence[int] | None) -> str:
    """``[startled, trials]`` as ``startled/trials``."""
    return "—" if x is None else f"{x[0]}/{x[1]}"


def _ear_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    if EAR not in rows:
        return ["（這次沒有跑真實耳朵延遲版。）", ""]
    out = [
        "腦層的世界原本在聲音一開始就告訴大腦這是什麼聲音；真實的耳朵要等一下才認得出來（拍手、巨響 80–160 毫秒，"
        "叫名字要等字說完，約 1 秒）。這一版在實驗程式這一層，用物理層聽覺模型同一套延遲（直接用它的常數和它的亂數函式）"
        "把聲音的類型晚一點送給大腦：起音那一刻有音量、有起音強度，但類型是空的，過了延遲才變成類型。被測的程式一個字沒改。"
        "九種刺激 × 4 個個性 × 種子 1–16 全部重錄一次，沒有聲音的五種刺激應該和標準的腦層紀錄完全一樣（第十一節的身分檢查）。",
        "",
        f"**{EAR}** 判定矩陣（只重判 M1、M2、M3）：",
        "",
        "| 指標 | " + " | ".join(presets) + " |",
        "|---|" + "---|" * len(presets),
    ]
    cells = A3.cell_verdicts(rows[EAR], presets)
    std = A3.cell_verdicts(rows.get("L1", []), presets)
    for metric in ("M1", "M3"):
        line = []
        for p in presets:
            a, b = std.get(("L1", metric, p), "—"), cells.get((EAR, metric, p), "—")
            line.append(a if a == b else f"{a} → **{b}**")
        out.append(f"| {metric}（標準 → 延遲耳朵） | " + " | ".join(line) + " |")
    out.append("")
    gentle = [
        (p, k)
        for p in presets
        for k in ("call_name", "hello")
        if k in (diag.get("L1", {}).get("gentle_startles", {}).get(p, {}))
    ]
    if gentle:
        out.append(
            "溫和的聲音刺激被嚇到或定格的次數（次數／回合；M3 要求溫和刺激的驚嚇率每個個性 ≤ 10%）："
        )
        out.append("")
        out.append("| 個性 | 叫名字：標準 → 延遲耳朵 | 打招呼：標準 → 延遲耳朵 |")
        out.append("|---|---|---|")
        for p in presets:
            bits = []
            for k in ("call_name", "hello"):
                a = diag.get("L1", {}).get("gentle_startles", {}).get(p, {}).get(k)
                b = diag.get(EAR, {}).get("gentle_startles", {}).get(p, {}).get(k)
                bits.append(f"{_count(a)} → **{_count(b)}**")
            out.append(f"| {p} | " + " | ".join(bits) + " |")
        out.append("")
    changes = diag.get("ear_changes") or []
    if not changes:
        out.append("延遲耳朵版和標準腦層版的 M1、M2、M3 每一列都一樣。")
        return out + [""]
    out.append(f"和標準腦層版不一樣的列（共 {len(changes)} 列）：")
    out.append("")
    out.append("| 指標 | 對象 | 量什麼 | 標準腦層 | 延遲耳朵 |")
    out.append("|---|---|---|---|---|")

    def number(x: Mapping[str, Any], diff: bool) -> str:
        v = x["value"]
        s = "—" if v is None else (f"{v:+.2f}" if diff else _p(v))
        return f"{s}（N={x['n']}，`{x['verdict']}`）"

    for c in changes:
        diff = "差距" in c["part"] or "−" in c["part"]
        out.append(
            f"| {c['metric']} | {c['scope']} | {c['part']} | {number(c['standard'], diff)} | {number(c['delayed'], diff)} |"
        )
    out.append("")
    return out


def _m2_text(rows: Rows) -> list[str]:
    """v2's M2 text (the metric did not change) and, for every stimulus that fell under +0.5, what the
    number says in plain words: whether the duck answered with a reflex only (the 'with reflexes' reference
    number reaches +0.5) or hardly at all."""
    out = RT2._m2_text(rows)
    for layer in sorted(rows):
        lr = rows[layer]
        ref = {r.scope: r for r in lr if r.metric == "M2" and r.verdict == A.REPORT}
        lines = []
        for r in lr:
            if r.metric != "M2" or r.verdict != A.FAIL or r.value is None:
                continue
            extra = ref.get(r.scope)
            if extra is not None and extra.value is not None and extra.value >= 0.5:
                lines.append(
                    f"{r.scope}（{r.value:+.2f}）：牠有反應，只是多半停在反射（看一下、定格），沒有換行為；"
                    f"把反射也算進去是 {extra.value:+.2f}"
                )
            else:
                lines.append(
                    f"{r.scope}（{r.value:+.2f}）：這種刺激來了，牠的行為和什麼都沒發生時差不多，連反射算進去也不到 +0.5"
                    + (
                        f"（{extra.value:+.2f}）"
                        if extra is not None and extra.value is not None
                        else ""
                    )
                )
        if lines:
            out.append(f"{layer} 沒過的白話：" + "；".join(lines) + "。")
            out.append("")
    return out


# ── 六～七、每個沒過或判不出的項目 ────────────────────────────────────────────────────


def _unsettled(rows: Rows, metric: str, presets: Sequence[str]) -> bool:
    for layer, lr in rows.items():
        cells = A3.cell_verdicts(lr, presets)
        if any(cells.get((layer, metric, p)) in (A.FAIL, A.INCOMPLETE) for p in presets):
            return True
    return False


def _reflex_phrase(counts: Mapping[str, int], n: int) -> str:
    if not counts:
        return "沒有紀錄"
    return "、".join(f"{k} {v}/{n}" for k, v in counts.items())


def _cell_meaning(c: Mapping[str, Any], refl: Mapping[str, Any] | None) -> str:
    """One failing or unsettled cell, in plain words, from its numbers."""
    rate, diff = c["p_stim"], c["diff"]
    bits = []
    if c["verdict_rate"] != A.PASS:
        bits.append(
            f"有刺激組只有 {_p(rate, 0)} 注意到（要 ≥ 70%）"
            if c["verdict_rate"] == A.FAIL
            else f"有刺激組注意率 {_p(rate, 0)}，但有未知的回合，判不出"
        )
    if c["verdict_diff"] != A.PASS:
        same = (
            f"對照組（同一個位置、什麼都沒放）的頭自己也有 {_p(c['p_sham'], 0)} 轉向了那個位置"
            if c["stimulus"] in S.DIRECTIONAL
            else f"對照組自己 3 秒內也有 {_p(c['p_sham'], 0)} 出現反射"
        )
        bits.append(
            f"比對照組只多 {'—' if diff is None else f'{diff:+.2f}'}（要 ≥ +0.5）：{same}"
            if c["verdict_diff"] == A.FAIL
            else "配對差距有未知的回合，判不出"
        )
    line = "；".join(bits)
    if refl:
        n_s, n_c = refl["n"]["stim"], refl["n"]["sham"]
        line += (
            f"。3 秒內出現的反射（出現的組數）：有刺激組 {_reflex_phrase(refl['stim'], n_s)}；"
            f"對照組 {_reflex_phrase(refl['sham'], n_c)}"
        )
    if c["stimulus"] in S.DIRECTIONAL:
        line += "。（這種刺激是看頭有沒有轉向它真實的位置，反射不算）"
    return line


def _seeds_of(items: Sequence[str], merged: bool) -> list[str]:
    """The seeds named in a row's failing-trial ids (``temperament|seedN|stimulus|...``): ``seedN`` for a
    one-temperament cell, ``temperament:N`` for a cell of all of them."""
    out: list[str] = []
    for item in items:
        head, _, rest = item.partition("|seed")
        if not rest:
            continue
        seed = rest.partition("|")[0]
        label = f"{head}:{seed}" if merged else f"seed{seed}"
        if label not in out:
            out.append(label)
    return out


def _seed_lists(c: Mapping[str, Any], scope: str) -> str:
    merged = scope == A3.SCOPE_ALL
    parts = []
    lost = _seeds_of(c["fails_rate"], merged)
    if lost:
        parts.append(f"有刺激組沒注意的種子：{'、'.join(lost[:40])}{'…' if len(lost) > 40 else ''}")
    no_win = _seeds_of(c["fails_diff"], merged)
    if no_win:
        parts.append(
            f"沒有「有刺激組注意而對照組沒有」的種子（{len(no_win)} 顆）：{'、'.join(no_win[:40])}{'…' if len(no_win) > 40 else ''}"
        )
    return ("。" + "；".join(parts)) if parts else ""


def _m1_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M1 注意（每一格自己判）",
        "",
        "量的是：刺激來了，鴨子是不是「真的注意到」，而且比什麼都沒發生的同一顆種子多多少。"
        "有位置的刺激（滾球、放新東西）：3 秒內頭轉 ≥ 10° 朝向真實方向、和真實方位的差距縮小一半以上；對照組＝同一個位置、同一條規則，"
        "看頭有沒有轉向那個空位置。其餘七種是「事件」（拍手、巨響、叫名字、打招呼、快速逼近、摸頭、戳）：3 秒內有沒有出現任何反射；"
        "對照組＝同一個窗內自己有沒有出現反射（所以戳之後腦做出躲開 withdraw 也算注意到）。"
        "每一格的門檻：有刺激組 ≥ 70%，而且有刺激組 − 對照組 ≥ 0.5。一格＝腦層每個個性 × 每種刺激（每格 16 對），"
        "物理層每種刺激、合併 4 個個性（每格 32 對）。看不到、聽不到的回合排除在分母外。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        cells = R.m1_cells(lr)
        if not cells:
            continue
        refl_all = diag.get(layer, {}).get("cell_reflexes", {})
        bad = [c for c in cells if c["verdict"] != A.PASS]
        good = len(cells) - len(bad)
        out.append(f"**{layer}**：{len(cells)} 格裡過 {good} 格，沒過或判不出 {len(bad)} 格。")
        if not bad:
            out.append("")
            continue
        for scope in _scopes(layer if layer != EAR else "L1", presets):
            mine = [c for c in bad if c["scope"] == scope]
            if not mine:
                continue
            out.append(f"- {scope}：")
            for c in sorted(mine, key=lambda x: S.STIMULI.index(x["stimulus"])):
                refl = refl_all.get(f"{scope}|{c['stimulus']}")
                out.append(
                    f"  - {c['stimulus']}（`{c['verdict']}`）：{_cell_meaning(c, refl)}"
                    f"{_seed_lists(c, scope)}"
                )
        out.append("")
    return out


def _m6_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M6 同一隻鴨",
        "",
        f"量的是：同一個個性的鴨子換種子，是不是還認得出是同一種個性。每顆種子跑一整段生活（腦層：白天 2 小時＋夜晚 2 小時，5 Hz，"
        f"每 10 分鐘一次刺激，9 種輪流），做一份個性檔案（各行為的時間比例，加上對每種刺激的反應：換行為的比例、驚嚇的比例、反應快慢）。"
        f"**第 3 版：每一個數字先換成同一把尺（z 分數），平均和標準差只用訓練用的種子算（被猜的那顆不參與），標準差為 0 的維度拿掉。**"
        f"猜個性：拿掉一顆種子，用其他種子算每個個性的平均檔案，把拿掉的那顆歸給最像的個性（歐氏距離）。"
        f"門檻：猜對 ≥ {S.M6_ACC_MIN:.0%}（亂猜 25%），而且每個個性自己被猜對 ≥ {S.M6_EACH_MIN:.0%}；"
        f"招牌動作：每個個性 ≥ {S.M6_SIG_MIN:.0%} 的種子，整段生活裡兩個招牌動作都要出現。物理層規格寫明只報數字。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        g = next(
            (
                r
                for r in lr
                if r.metric == "M6"
                and r.scope == "ALL"
                and r.part.lstrip("（報告）").startswith("猜個性（z 分數）")
            ),
            None,
        )
        if g is None:
            continue
        out.append(f"**{layer}**：猜對 {g.ok}/{g.n}（{_p(g.value)}，`{g.verdict}`）")
        for p in presets:
            own = next(
                (r for r in lr if r.metric == "M6" and r.scope == p and "自己被猜對" in r.part),
                None,
            )
            sig = next(
                (
                    r
                    for r in lr
                    if r.metric == "M6"
                    and r.scope == p
                    and r.part.startswith(("招牌動作", "（報告）招牌動作"))
                ),
                None,
            )
            if own is not None and sig is not None:
                out.append(
                    f"- {p}：自己被猜對 {own.ok}/{own.n}（`{own.verdict}`），招牌動作 {sig.ok}/{sig.n}（`{sig.verdict}`）"
                )
        conf = g.extra.get("confusion")
        if conf:
            out += ["", "猜錯成什麼（列＝真正的個性，欄＝猜成的個性）：", ""]
            cols = [*presets, "none"]
            out.append("| 真正的個性 | " + " | ".join(cols) + " |")
            out.append("|---|" + "---|" * len(cols))
            for p in presets:
                out.append(f"| {p} | " + " | ".join(str(conf[p].get(c, 0)) for c in cols) + " |")
        sens = [r for r in lr if r.metric == "M6" and r.part.startswith("（報告，不是判定）猜個性")]
        if sens:
            out += [
                "",
                "另外報告（不當門檻，回答「個性差在過日子還是差在反應」）："
                + "；".join(f"{r.part.split('：')[-1]} 猜對 {_p(r.value)}" for r in sens)
                + "。",
            ]
            by = {r.part.split("：")[-1]: r.value for r in sens}
            t = by.get("只用行為時間比例（z 分數）")
            x = by.get("只用對刺激的反應（z 分數）")
            if t is not None and x is not None:
                if t > x + 0.05:
                    out.append(
                        "意思是：個性主要差在「過日子」（行為的時間分配），對刺激的反應比較認不出個性。"
                    )
                elif x > t + 0.05:
                    out.append(
                        "意思是：個性主要差在「對刺激的反應」，行為的時間分配比較認不出個性。"
                    )
                else:
                    out.append("意思是：過日子的時間分配和對刺激的反應，認出個性的能力差不多。")
        sp = diag.get(layer, {}).get("life_spread")
        if sp:
            parts = [
                f"{'行為時間比例' if k == 'time_shares' else '對刺激的反應'}（{v['dims']} 個數字，標準差不為 0 的 {v['kept_dims']} 個）：同一個個性內種子到平均的平均距離 "
                f"{v['within_mean']}、個性平均之間的平均距離 {v['between_mean']}"
                for k, v in sp.items()
                if k != "behaviours"
            ]
            out.append("")
            out.append(
                "個性檔案的兩個部分，在 z 分數空間裡分開看（描述資料，用全部種子的平均和標準差，不是猜法）："
                + "；".join(parts)
                + "。"
            )
        out.append("")
    return out


def _push_table(detail: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every push, one line: the failing seeds are in it."""
    if not detail:
        return []
    out = [
        "每一次推：",
        "",
        "| 個性 | 種子 | 從哪側推 | 結果 | 倒著幾秒 | 第一個定住身體的反射（倒下後幾秒） | 0.5 秒內有扶正 | 0.5 秒後下走路指令 | 被擋 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    yes = {True: "有", False: "沒有", None: "—"}
    for d in detail:
        if not d["valid"]:
            why = "；".join(d["problems"][:1])
            out.append(
                f"| {d['preset']} | {d['seed']} | {d['side']} | 沒錄到或無效（{why}） | | | | | |"
            )
            continue
        reflex = (
            "—" if d["first_reflex"] is None else f"{d['first_reflex']}（{d['first_reflex_s']} 秒）"
        )
        out.append(
            f"| {d['preset']} | {d['seed']} | {d['side']} | {STATUS_ZH.get(d['status'], d['status'])} | "
            f"{'—' if d['down_s'] is None else d['down_s']} | {reflex} | {yes[d['righting']]} | "
            f"{yes[d['walked']]} | {yes[d['blocked']]} |"
        )
    out.append("")
    return out


def _m7_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M7 安全（只在物理層）",
        "",
        "量的是：鴨子被推倒以後，安全反射要接手，而且之後不能再叫牠走路。**第 3 版加了「推一下」**：每個個性 × 種子 1–8 各推一次"
        f"（世界原本的推力介面，力用它的最大值 7 N，第 {S.L2_ONSET_S[S.PUSH]:.0f} 秒，偶數種子從左、奇數種子從右），配同種子不推的對照組。"
        "「推倒」＝推了之後 3 秒內真相的 `fallen` 旗標升起、倒著至少 0.5 秒（寬限時間），而且對照組沒有倒。"
        "推倒的回合：倒下後 0.5 秒內，紀錄裡要有「定住身體」的反射（扶正 righting 等）接手，而且 0.5 秒之後、倒著的時候不能再有走路指令；"
        "不然就是「被擋」，被擋的次數必須是 0。沒推倒的照實報，不算通過也不算失敗；推倒的次數（每個個性自己 8 次裡）不到 4 次，這一項判「未完成」。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        pushes = diag.get(layer, {}).get("push") or {}
        if not pushes:
            continue
        out.append(
            "| 個性 | 推了幾次 | 推倒 | 沒推倒 | 很快自己起來 | 兩組都倒 | 已經倒著 | 被擋 | 判定 |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for p in (*presets, "ALL"):
            s = pushes.get(p)
            if s is None:
                continue
            st = s["statuses"]
            row = next(
                (r for r in lr if r.metric == "M7" and r.scope == p and r.part == A3.PUSH_PART),
                None,
            )
            out.append(
                f"| {p} | {s['pushes']}"
                + f"{'（另有 ' + str(s['invalid']) + ' 次沒錄到或無效）' if s['invalid'] else ''} | "
                + " | ".join(
                    str(st.get(k, 0))
                    for k in (
                        P.TIPPED,
                        P.NOT_TIPPED,
                        P.RECOVERED_FAST,
                        P.BOTH_TIPPED,
                        P.ALREADY_DOWN,
                    )
                )
                + f" | {row.fail if row else '—'} | {('`' + row.verdict + '`') if row else '—'} |"
            )
        out.append("")
        for p in (*presets, "ALL"):
            s = pushes.get(p)
            if not s or not s["statuses"].get(P.TIPPED):
                continue
            out.append(
                f"- {p}：推倒的 {s['statuses'][P.TIPPED]} 次，倒著的時間中位數 {s['median_down_s']} 秒（最長 {s['max_down_s']} 秒）；"
                f"第一個定住身體的反射：{'、'.join(f'{k} {v}' for k, v in s['first_reflex'].items())}"
                f"（倒下後中位數 {s['median_first_reflex_s']} 秒）；0.5 秒內有扶正（righting）本身的 {s['righting_in_grace']} 次；"
                f"0.5 秒之後倒著還下走路指令的 {s['walked_after_grace']} 次。"
            )
        for r in lr:
            if (
                r.metric == "M7"
                and r.part == A3.PUSH_PART
                and r.verdict in (A.FAIL, A.INCOMPLETE)
                and r.note
            ):
                out.append(f"- {r.scope}（`{r.verdict}`）：{r.note}")
        out.append("")
        out += _push_table(diag.get(layer, {}).get("push_rows") or [])
        for r in lr:
            if r.metric == "M7" and r.part.startswith("有刺激組跌倒數"):
                out.append(
                    f"九種刺激的回合（不含推一下）：{r.scope} 有刺激組跌倒數 − 對照組 {r.value:+.0f}（N={r.n}，`{r.verdict}`）"
                )
        out.append("")
    return out


# ── 八、物理世界的刺激介面 ──────────────────────────────────────────────────────────────


def _support_text(supported: Sequence[str]) -> list[str]:
    nine = {S.PHYSICS_NAME.get(name, name): name for name in S.STIMULI}
    out = [
        f"物理世界自己的刺激介面（`neuro_world.STIMULI`，這次被驗的程式）支援 {len(supported)} 種。規格的 9 種刺激它都支援，全部測了；"
        "推（`push`）用在 M7；其餘的規格沒有給它們 M1／M2／M3 的類別和門檻，**寫「未測」**。",
        "",
        "| 世界的名字 | 規格的名字 | 這次怎麼用 |",
        "|---|---|---|",
    ]
    uses = {
        "roll_ball": "世界的 `x_roll_ball`（從頭相機朝向左或右 15°、1.4 m 處以 900 mm/s 滾向鴨子）",
        "novel_object": "世界的 `x_place_prop`（放在頭相機朝向左或右 15°、1.2 m 處）",
        "clap": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "call_name": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "hello": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "bang": "`stimulate`，加 `db1m` 100（頭的位置要 ≥ 90 dB，否則該回合無效）",
        "pet_head": "`stimulate`（手跟著頭，預設摸 2 下）",
        "poke": "`stimulate`（手在軀幹側邊，偶數種子左側、奇數種子右側）",
        "hand_loom": "`stimulate`（手從相機前 0.75 m 靠近；看得到＝1.5 秒內頭相機取樣看得到手）",
    }
    for name in supported:
        if name in nine:
            out.append(f"| {name} | {nine[name]} | 測：M1、M2、M3；{uses[name]} |")
        elif name == S.PUSH:
            out.append(f"| {name} | — | 測：M7（推力 7 N，0.15 秒，偶數種子左、奇數種子右） |")
        else:
            out.append(f"| {name} | — | **未測**（規格的 9 種刺激以外） |")
    out.append("")
    return out


# ── 十、沒量到或沒判出來的 ──────────────────────────────────────────────────────────────


def _unmeasured(rows: Rows) -> list[str]:
    out: list[str] = []
    for layer in sorted(rows):
        seen: set[tuple[str, str]] = set()
        for r in rows[layer]:
            if r.verdict in (A.UNTESTED, A.INCOMPLETE) and r.metric in METRIC_ZH:
                key = (r.metric, r.note or r.part)
                if r.metric == "M1":  # the cells are listed above
                    continue
                if key in seen:
                    continue
                seen.add(key)
                scope = "、".join(
                    sorted(
                        {
                            x.scope
                            for x in rows[layer]
                            if x.metric == r.metric
                            and (x.note or x.part) == key[1]
                            and x.verdict == r.verdict
                        }
                    )
                )
                out.append(
                    f"- {layer} {r.metric} {r.part}（{scope}）：`{r.verdict}`"
                    + (f"。{r.note}" if r.note else "")
                )
    m1_open = []
    for layer in sorted(rows):
        for c in R.m1_cells(rows[layer]):
            if c["verdict"] == A.INCOMPLETE:
                m1_open.append(f"{layer} {c['scope']} {c['stimulus']}")
    if m1_open:
        out.append(
            f"- M1 判不出的格（`incomplete`：那條規則在一些回合沒有定義，未知可以讓判定走向任一邊）：{'、'.join(m1_open)}"
        )
    out.append(
        f"- 物理世界的刺激介面支援、但不在規格的 9 種刺激裡的 {len(S.OTHER_PHYSICS_STIMULI)} 種"
        f"（{'、'.join(S.OTHER_PHYSICS_STIMULI)}）：`未測`（第七節）。"
    )
    return out


# ── the report ──────────────────────────────────────────────────────────────────────


def build_report(
    *,
    rows: Rows,
    presets: Sequence[str],
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wall: Mapping[str, Any] | None,
    v2_rows: Rows | None,
    tables_md: str,
    failing_md: str,
    supplement_md: str,
    m1_cells_md: str,
    identity_md: str,
    reproduce_md: str,
    physics_supported: Sequence[str],
) -> str:
    code = manifest.get("code", {})
    commit = str(code.get("commit", ""))[:7]
    date = str(manifest.get("started", ""))[:10]
    stages = manifest.get("stages") or []
    out: list[str] = [
        f"# 機器驗證 第 3 版 結果（{date}，程式版本 {commit}）",
        "",
        "給 owner 的白話版，由程式從每一列的判定資料自動寫出（重跑就會得到新的一份）。規格是 `docs/design/pet-behavior-v1/MACHINE-VERIFY-v3.md`"
        "（已凍結），門檻、定義、樣本數沒有改。規格沒寫清楚的操作選擇，在看到任何結果之前就寫好並提交了"
        "（`docs/design/pet-behavior-v1/MACHINE-VERIFY-v3-CHOICES.md`）。判定只用世界真相；腦自己說的話只拿來和真相比（M4）。"
        "第 1、2 版的規格和結果原樣保留。",
        "",
        f"- 被驗的程式：`{code.get('code_root', '')}`，commit `{code.get('commit', '')}`（修好名字被嚇到的那一版）"
        + ("（有未提交的改動！）" if code.get("dirty") else "（乾淨）"),
        f"- 驗證程式（錄製者與判定者）：`{code.get('verifier_root', '')}`，錄製開始時 commit `{str(code.get('verifier_commit', ''))[:7]}`"
        + (
            "；之後的階段："
            + "、".join(
                f"{str(s.get('verifier_commit', ''))[:7]}（{', '.join(s.get('steps', []))}）"
                for s in stages
            )
            if len(stages) > 1
            else ""
        ),
        f"- 事前選擇文件的 commit `{str(manifest.get('choices_commit', ''))[:7]}`，提交時間 {manifest.get('choices_commit_time', '')}；"
        f"錄製開始時間 {manifest.get('started', '')}；**選擇文件早於錄製開始："
        + ("是（程式比對）" if manifest.get("choices_before_start") else "否！")
        + "**",
        "",
        "## 一、先講結論",
        "",
        *_conclusions(rows, presets),
        "",
        "## 二、判定矩陣（每一層 × 個性 × 指標），和每格背後的數字",
        "",
        "`PASS`＝過；`FAIL`＝沒過；`incomplete`＝沒辦法判（缺資料、規則在那些回合沒有定義，或推倒的次數不夠；**不當通過**）；"
        "`未測`＝這一層沒量；`報告`＝規格事先寫明只報數字。未知的處理：把所有未知都當成對鴨子最不利，門檻還是過才算 `PASS`；"
        "都當成最有利還是不過就是 `FAIL`；中間是 `incomplete`。M1 的格＝這個個性的 9 種刺激各一格（物理層是 9 格、4 個個性合併）。",
        "",
        R.key_numbers_markdown([r for lr in rows.values() for r in lr], presets),
        "",
        "## 三、M1 每一格",
        "",
        "每一格寫「判定／有刺激組注意率 ／ 對照組 ／ 配對差距／N（該格有效的配對數）」。",
        "",
        m1_cells_md.strip(),
        "",
        "## 四、第 2 版 → 第 3 版（同一份被驗程式）",
        "",
        *_versus_v2(v2_rows, rows, presets),
        "## 五、真實耳朵延遲版（腦層，新）",
        "",
        *_ear_text(rows, diagnostics, presets),
        "## 六、每一個沒過或判不出的項目，白話是什麼意思",
        "",
        "下面的句子都是從上面的數字和診斷數字直接寫出來的；診斷數字沒有門檻，不影響任何判定。",
        "",
    ]
    for metric, fn in (
        ("M1", lambda: _m1_text(rows, diagnostics, presets)),
        ("M2", lambda: _m2_text(rows)),
        ("M3", lambda: RT2._m3_text(rows, diagnostics, presets)),
        ("M4", lambda: RT2._m4_text(rows, presets)),
        ("M5", lambda: RT2._m5_text(rows, diagnostics, presets)),
        ("M6", lambda: _m6_text(rows, diagnostics, presets)),
        ("M7", lambda: _m7_text(rows, diagnostics, presets)),
    ):
        if metric in ("M2", "M6", "M7") or _unsettled(rows, metric, presets):
            out += fn()
    out += [
        "## 七、物理世界的刺激介面：測了哪些、哪些「未測」",
        "",
        *_support_text(physics_supported),
        "## 八、規格沒寫、事先定的操作選擇（要審查的）",
        "",
        "全文在 `MACHINE-VERIFY-v3-CHOICES.md`（在任何 v3 結果出現以前提交）。最會影響結果的幾條：",
        "",
        "1. **M1 的格**：腦層每個個性 × 每種刺激 16 對；物理層每種刺激合併 4 個個性 32 對。事件類的規則是「3 秒內 `reflex` 欄位有任何非空值」。"
        "每格的兩個門檻各自用第 2 版的未知三分法，格的判定是兩者合起來，一個個性的 M1 是它所有格合起來（任一格沒過就沒過）。",
        "2. **M6 的 z 分數**：平均和標準差只用訓練種子（拿掉的那顆以外的 63 顆；物理層 31 顆）算，母體標準差，標準差為 0 的維度拿掉；"
        "行為名單（維度）由整批生活出現過的行為決定，不涉及平均和標準差。",
        "3. **延遲耳朵**：直接用物理層聽覺模型的常數（`CLASS_LATENCY_MS`、`CLASS_HOLD_MS`）和它的亂數函式，聲音的包絡用它的規則（`IMPULSIVE`＝拍手、巨響…；"
        "說話類要等字說完）。只延遲類型，音量、起音、混淆機率都不動。",
        "4. **L2 的新刺激**：打招呼、摸頭、戳、快速逼近（世界叫 `hand_loom`）都走世界自己的 `stimulate`；逼近的「看得到」＝送出後 1.5 秒內頭相機取樣看得到手；"
        "摸頭、戳一律摸得到。種子 1–8，每種刺激各有自己的秒數。",
        "5. **M7 的推**：世界的 `push`，力用介面的最大值（`strength` 1.0＝7 N）；倒下＝真相 `fallen` 在推了之後 3 秒內升起且至少 0.5 秒；"
        "「接手」＝倒下後 0.5 秒內 `reflex` 欄位有定住身體的反射（`HOLD_REFLEXES`）；「被擋」＝沒接手或寬限時間後倒著還下走路指令。",
        "",
        "## 九、沒量到或沒判出來的（照規格寫 `未測` 或 `incomplete`）",
        "",
        *_unmeasured(rows),
        "",
        "## 十、第 1、2 版重判（舊結果還能重現）",
        "",
        reproduce_md.replace("# 第 1、2 版重判：和原來的 rows 逐位元組比", "").strip()
        or "（這次沒有跑重判。）",
        "",
        "## 十一、錄製身分檢查（同一份程式兩次錄製是不是同一份紀錄）",
        "",
        identity_md.replace("# 錄製身分檢查（不是判定）", "").strip() or "（沒有比。）",
        "",
        "## 十二、時間",
        "",
    ]
    if wall:
        names = {
            "l1": "L1 對照回合",
            "l1ear": "L1 延遲耳朵版",
            "lives": "L1 整段生活",
            "l2": "L2 物理（2 個行程）",
            "l2_wait": "L2 等另一件工作的物理做完",
            "reproduce": "第 1、2 版重判",
            "judge": "判定",
            "sum_of_steps": "各步驟加總",
            "total": "從開始到現在（牆上時間）",
        }
        out.append("、".join(f"{names.get(k, k)} {v:.0f} 秒" for k, v in wall.items()) + "。")
    out += [
        "",
        "## 十三、檔案與重跑",
        "",
        "一條指令（見同目錄 `MACHINE-VERIFY-v3-RERUN.md`）：`python tools/machine_verify/run_v3.py --code-root <被驗的程式> --out <輸出目錄>`。"
        "這個目錄裡：`rows-L1.json`、`rows-L1ear.json`、`rows-L2.json`（每一列的值、分母、成功、失敗、未知、排除、門檻、判定、失敗的回合）、"
        "`tables.md`、`m1-cells.md`、`failing-trials.md`、`supplement.md`、`key-numbers.md`、`compare-v2-v3.md`、`trace-identity.md`、"
        "`reproduce-v1-v2.md`、`diagnostics.json`、`L1/`、`L1_ear/`、`L1_lives/`、`L2/`（原始紀錄和 manifest）、`manifest.json`、`wall.json`、`logs/`。",
        "",
        "## 十四、每一列的數字",
        "",
        "成功／失敗的意思依列而定（M1 配對差那列，成功＝有刺激組注意而對照組沒有；M5，成功＝自然結束；M3 溫和那列，成功＝沒嚇到；"
        "M3 很大聲那列，成功＝嚇到；M7 第一列，成功欄是有刺激組的跌倒數、失敗欄是對照組的跌倒數；M7 推一下那列，N＝推倒的次數、成功＝沒被擋、失敗＝被擋）。",
        "",
        tables_md.strip(),
        "",
        "## 十五、補充數字（不是判定）",
        "",
        supplement_md.strip(),
        "",
        "## 附錄：失敗與未知的回合（全部）",
        "",
        failing_md.replace("# 失敗與未知的回合（全部列出）", "").strip(),
        "",
    ]
    return "\n".join(out)


__all__ = ["build_report", "METRIC_ZH"]
