"""The owner-facing text of a v2 result, in plain Traditional Chinese, made by the program from the
rows (the same rows the verdicts are in), the diagnostics and the manifests: so that a rerun on other
code gets its own up-to-date report with one command. Nothing here judges anything: every sentence
quotes a row or a diagnostic number, and says what the metric measures.

The pieces that are plain tables (the verdict matrix, the numbers behind each cell, every row, the
failing trials, the v1-v2 comparison) are made by ``report_v2``; this module puts the words around them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v2 as A2
from duckmind.verify import report_v2 as R
from duckmind.verify import spec_v2 as S

Rows = Mapping[str, Sequence[A.Row]]

METRIC_ZH = {
    "M1": "注意",
    "M2": "因果",
    "M3": "驚嚇合理",
    "M4": "說真話",
    "M5": "收尾",
    "M6": "同一隻鴨",
    "M7": "安全",
}
GATED = ("M1", "M3", "M4", "M5", "M6", "M7")


def _p(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _rows(
    rows: Sequence[A.Row], metric: str, scope: str | None = None, part: str = ""
) -> list[A.Row]:
    return [
        r
        for r in rows
        if r.metric == metric and (scope is None or r.scope == scope) and r.part.startswith(part)
    ]


def _one(rows: Sequence[A.Row], metric: str, scope: str, part: str) -> A.Row | None:
    found = _rows(rows, metric, scope, part)
    return found[0] if found else None


# ── 一、結論 ────────────────────────────────────────────────────────────────────────


def _conclusions(rows: Rows, presets: Sequence[str]) -> list[str]:
    groups: dict[str, list[str]] = {
        A.PASS: [],
        A.FAIL: [],
        A.INCOMPLETE: [],
        A.UNTESTED: [],
        A.REPORT: [],
    }
    never_all_pass: list[str] = []
    all_pass: list[str] = []
    for layer in sorted(rows):
        lr = rows[layer]
        cells = A2.cell_verdicts(lr, presets)
        for p in presets:
            vs = [cells.get((layer, m, p), "—") for m in GATED]
            judged = [v for v in vs if v in (A.PASS, A.FAIL, A.INCOMPLETE)]
            if judged and all(v == A.PASS for v in judged):
                all_pass.append(f"{layer} {p}")
            else:
                never_all_pass.append(f"{layer} {p}")
        for metric in GATED:
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
        (A.INCOMPLETE, "判不出來（`incomplete`：缺資料，或規則在那些回合沒有定義；不當通過）"),
        (A.UNTESTED, "這一層沒量（`未測`）"),
        (A.REPORT, "只報數字、不當門檻（`報告`）"),
    )
    for verdict, title in titles:
        if groups[verdict]:
            out.append(f"- **{title}：**")
            out += [f"  - {line}" for line in groups[verdict]]
    return out


# ── 三、和第 1 版比 ──────────────────────────────────────────────────────────────────


def _range(rows: Sequence[A.Row], metric: str, part: str) -> str:
    vals = [
        r.value
        for r in rows
        if r.metric == metric and r.part.startswith(part) and r.value is not None
    ]
    if not vals:
        return "—"
    return f"{_p(min(vals), 1)}–{_p(max(vals), 1)}" if len(vals) > 1 else _p(vals[0], 1)


def _versus_v1(rows: Rows, v1: Rows | None, presets: Sequence[str]) -> list[str]:
    out = [
        R.compare_markdown(v1 or {}, rows, presets)
        if v1
        else "（沒有給第 1 版的結果，無法並排。）",
        "",
        "每一項，規格改了什麼、數字怎麼變：",
        "",
    ]
    v1 = v1 or {}
    for layer in sorted(rows):
        r2, r1 = rows[layer], v1.get(layer, [])
        out.append(f"**{layer}**")
        out.append(
            "- M1：第 1 版把「3 秒內有反射或換行為」當注意，對照組也用同一條，所以對照組「無端注意」"
            f" {_range(r1, 'M1', '對照組無端注意')}（門檻 ≤ 10%），有刺激組 {_range(r1, 'M1', '有刺激組注意率')}。"
            "第 2 版：有方向的刺激看頭有沒有轉向真實方向、沒有方向的看有沒有反射，對照組用同一個位置、同一條規則，"
            f"門檻改成有刺激組 ≥ 70% 且配對差 ≥ 0.5：有刺激組 {_range(r2, 'M1', '有刺激組注意率')}，"
            f"配對差 {_range_diff(r2, 'M1', '有刺激組 − 對照組')}，對照組同規則下 {_range(r2, 'M1', '（報告）對照組同規則')}。"
        )
        out.append("- M2、M3、M7：判法沒改。" + _same_numbers(r1, r2))
        a1 = _one_any(r1, "M4", "腦說")
        a2 = _one_any(r2, "M4", "腦說")
        if a2 is not None:
            out.append(
                "- M4：「到了」第 1 版有 "
                + (f"{a1.n + a1.unknown} 次" if a1 else "—")
                + f"，判不出（`{a1.verdict if a1 else '—'}`）；第 2 版把目標是「記得的地方」的也換算成世界座標判："
                f"{a2.n + a2.unknown} 次，{a2.ok} 次在可碰範圍內（`{a2.verdict}`）。"
            )
        n1 = _range(r1, "M5", "刺激引起的行為：自然結束")
        n2 = _range(r2, "M5", "刺激引起的那段：自然結束")
        out.append(
            f"- M5：第 1 版自然結束 {n1}；第 2 版把「刺激引起的那段」和「自然結束」照規格寫的定義重做（反射一串自己放開"
            f"或滿足度 ≥ 0.5 才算自然；分母是有刺激組窗內開始的那一段，不再要求對照組沒換行為）：{n2}。"
            "第 1 版分母（有刺激組換了、對照組沒換）的數字還在每個個性的「（報告）」列。"
        )
        out.append(
            "- M6：第 1 版用「個性之間的中心點距離 vs 同一個性內的最遠兩點距離」；第 2 版改成每顆種子一整段生活"
            "（白天 2 小時＋夜晚 2 小時）、拿掉一顆種子猜它是哪個個性，招牌動作改在整段生活裡找。"
            if layer == "L1"
            else "- M6：物理層規格事先寫明只報數字（第 1 版當門檻判，全部沒過）。"
        )
        out.append("")
    return out


def _range_diff(rows: Sequence[A.Row], metric: str, part: str) -> str:
    vals = [
        r.value
        for r in rows
        if r.metric == metric and r.part.startswith(part) and r.value is not None
    ]
    if not vals:
        return "—"
    return f"{min(vals):+.2f} 到 {max(vals):+.2f}"


def _one_any(rows: Sequence[A.Row], metric: str, part: str) -> A.Row | None:
    found = [r for r in rows if r.metric == metric and r.part.startswith(part)]
    if not found:
        return None
    # one row per temperament: add them up
    first = found[0]
    total = A.Row(
        first.layer, metric, part, "ALL", None,
        sum(r.n for r in found), sum(r.ok for r in found), sum(r.fail for r in found),
        sum(r.unknown for r in found), 0, first.threshold,
        A.combine([r.verdict for r in found]),
    )  # fmt: skip
    return total


def _same_numbers(r1: Sequence[A.Row], r2: Sequence[A.Row]) -> str:
    old = {(r.metric, r.scope, r.part): r for r in r1}
    same = diff = 0
    for r in r2:
        if r.metric not in ("M2", "M3", "M7"):
            continue
        o = old.get((r.metric, r.scope, r.part))
        if o is None:
            continue
        if o.value == r.value and (o.n, o.ok, o.fail) == (r.n, r.ok, r.fail):
            same += 1
        else:
            diff += 1
    return f"和第 1 版相同名字的列有 {same + diff} 列，{same} 列數字完全一樣" + (
        f"、{diff} 列不同（見上面並排表下面的列表；多半是這次多了很大聲，分母變大）。"
        if diff
        else "。"
    )


# ── 四、每個沒過的是什麼意思 ─────────────────────────────────────────────────────────


def _unsettled(rows: Rows, metric: str, presets: Sequence[str]) -> bool:
    for layer, lr in rows.items():
        cells = A2.cell_verdicts(lr, presets)
        if any(cells.get((layer, metric, p)) in (A.FAIL, A.INCOMPLETE) for p in presets):
            return True
    return False


def _m1_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M1 注意",
        "",
        "量的是：刺激來了，鴨子是不是「真的注意到」，而且比什麼都沒發生的同一顆種子多多少。"
        "有方向的刺激（滾球、放新東西、快速逼近、戳、摸頭）：3 秒內頭轉 ≥ 10° 朝向真實方向，而且和真實方位的差距縮小一半以上；"
        "沒有方向的刺激（拍手、很大聲、叫名字、打招呼）：3 秒內有沒有反射。對照組：同一顆種子、同一個時間點，"
        "有方向的假裝刺激在同一個位置，用同一條規則看頭有沒有轉向那個空位置；沒有方向的看同一個窗內自己有沒有出現反射。"
        "門檻：有刺激組 ≥ 70%，而且有刺激組 − 對照組 ≥ 0.5。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        out.append(f"**{layer}**")
        for p in presets:
            a = _one(lr, "M1", p, "有刺激組注意率")
            d = _one(lr, "M1", p, "有刺激組 − 對照組")
            if a is None or d is None:
                continue
            total = a.n + (a.unknown - len(a.extra.get("missing_or_invalid", [])))
            line = (
                f"- {p}：量得到的 {a.n} 組，有刺激組注意 {_p(a.value)}（門檻 ≥ 70%：`{a.verdict}`），"
                f"配對差 {d.value:+.2f}（門檻 ≥ 0.5：`{d.verdict}`）"
                if d.value is not None
                else f"- {p}：沒有量到的組"
            )
            unk = total - a.n
            if unk > 0 and a.value is not None:
                low, high = a.ok / total, (a.ok + unk) / total
                kinds = a.extra.get("unknown_by_stimulus", {})
                line += (
                    f"；另有 {unk} 組規則沒有定義（{'、'.join(f'{k} {v}' for k, v in kinds.items())}）。"
                    f"這些都算沒注意的話注意率是 {_p(low)}，都算注意的話是 {_p(high)}，所以只能判 `{a.verdict}`"
                )
            out.append(line)
        out.append("")
        out.append(
            "| 刺激 | 用的規則 | 看得到的組數 | 有刺激組注意 | 對照組注意 | 差距 | 規則沒定義的組數 |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for r in _rows(lr, "M1"):
            if r.scope not in S.DIRECTIONAL | S.NON_DIRECTIONAL:
                continue
            rule = "頭轉向真實方向" if r.scope in S.DIRECTIONAL else "3 秒內有反射"
            out.append(
                f"| {r.scope} | {rule} | {r.extra.get('n_pairs', 0)} | {_p(r.extra.get('p_stim'))} | "
                f"{_p(r.extra.get('p_sham'))} | {'—' if r.value is None else f'{r.value:+.2f}'} | {r.unknown} |"
            )
        out.append("")
        for kind, d in (diag.get(layer, {}).get("directional") or {}).items():
            row = next((r for r in _rows(lr, "M1") if r.scope == kind), None)
            if row is None or (row.extra.get("p_stim") or 0) > 0.2:
                continue
            reflexes = (
                "、".join(f"{k} {v}/{d['n']}" for k, v in d["reflex_trials"].items()) or "沒有反射"
            )
            out.append(
                f"{kind} 的注意率很低，原因（診斷數字，不是判定）：{d['n']} 組裡 3 秒內出現的反射（出現的組數）：{reflexes}；"
                f"頭朝真實方向轉的最大角度，中位數 {d['median_max_turn_deg']}°（0＝從沒朝那邊轉）；"
                f"和真實方位的差距最小縮到刺激那刻的 {d['median_best_err_ratio']} 倍（中位數；規則要 ≤ 0.5）。"
                + (
                    "也就是說牠有反應，只是反應的方式不是「把頭轉向刺激」。"
                    if d["reflex_trials"]
                    else "也就是說這些回合裡牠既沒有反射、也沒有把頭轉向刺激。"
                )
            )
        out.append("")
    return out


def _m2_text(rows: Rows) -> list[str]:
    out = [
        "### M2 因果（判法沒改）",
        "",
        "量的是：同一顆種子，「有刺激」比「沒刺激」多多少比例在 3 秒內換了行為或回合（反射不算）；每種刺激合併所有個性，門檻 ≥ 0.5。"
        "「含反射」的數字（換行為或有反射）只當參考。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        gated = [r for r in lr if r.metric == "M2" and r.verdict in (A.PASS, A.FAIL)]
        ref = {r.scope: r for r in lr if r.metric == "M2" and r.verdict == A.REPORT}
        out.append(
            f"**{layer}**："
            + "；".join(
                f"{r.scope} {r.value:+.2f}（{r.verdict}"
                + (
                    f"；含反射 {ref[r.scope].value:+.2f}"
                    if r.scope in ref and ref[r.scope].value is not None
                    else ""
                )
                + "）"
                for r in gated
                if r.value is not None
            )
        )
        out.append("")
    return out


def _m3_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M3 驚嚇合理",
        "",
        "量的是：溫和刺激（叫名字、打招呼、摸頭、放新東西、正常速度滾球）不該嚇到（每個個性被嚇到／定格 ≤ 10%）；"
        "很大聲或快速逼近時，非大膽型要嚇到（≥ 50%）。物理層的很大聲用世界自己的刺激路徑（拍手、叫名字用的同一條）"
        f"做出來：`bang`，來源 {S.L2_BANG_DB1M:.0f} dB（1 m），每個回合量到的頭位置音量都必須 ≥ {S.LOUD_HEAD_DB_MIN:.0f} dB。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        gs = diag.get(layer, {}).get("gentle_startles", {})
        for p in presets:
            g = _one(lr, "M3", p, "溫和刺激")
            if g is None:
                continue
            by = gs.get(p, {})
            which = "、".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items()) if v[0])
            out.append(
                f"- {layer} {p}：溫和刺激被嚇到／定格 {g.fail}/{g.n}（{_p(g.value)}，`{g.verdict}`）"
                + (f"，哪幾種：{which}" if which else "")
            )
        for r in _rows(lr, "M3", None, "很大聲／逼近"):
            out.append(
                f"- {layer} {r.scope}：{r.part} {_p(r.value)}（{r.ok}/{r.n}，`{r.verdict}`）"
                if r.verdict != A.UNTESTED
                else f"- {layer} {r.scope}：{r.part}：未測（{r.note}）"
            )
    out.append("")
    return out


def _m4_text(rows: Rows, presets: Sequence[str]) -> list[str]:
    out = [
        "### M4 說真話（只在物理層）",
        "",
        "量的是：腦自己說的（目標的方位、距離、「到了」）和世界真相比。方位誤差 ≤ 20° 的時間比例要 ≥ 90%，距離誤差 ≤ 30% 要 ≥ 80%，"
        "腦說「到了」時，真實距離 ≤ 這個行為的可碰距離 + 0.1 m 必須 100%。目標是東西：用模擬器自己的標籤配對；"
        "目標是記得的地方：用起始姿態（第 0 個 tick 的真實姿態和腦讀到的里程計姿態）把腦記得的地方換成世界座標再比。"
        f"可碰距離用量好的嘴巴伸出量 {S.BEAK_REACH_M} m，所以「到了」的門檻是 {S.PLACE_ARRIVE_BOUND_M:.3f} m（東西再加半個物體寬）。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        arrivals = []
        for p in presets:
            r = _one(lr, "M4", p, "腦說")
            if r is None:
                continue
            arrivals += [(p, a) for a in r.extra.get("arrivals", [])]
            b = _one(lr, "M4", p, "方位誤差")
            d = _one(lr, "M4", p, "距離誤差")
            out.append(
                f"- {layer} {p}：方位 {_p(b.value if b else None)}（`{b.verdict if b else '—'}`），"
                f"距離 {_p(d.value if d else None)}（`{d.verdict if d else '—'}`），"
                f"到了 {r.ok}/{r.n + r.unknown} 次在範圍內（`{r.verdict}`）"
            )
        if arrivals:
            out += [
                "",
                f"{layer} 每一次「到了」：",
                "",
                "| 回合 | 時間 | 目標 | 對象 | 真實距離 | 門檻 | 結果 |",
                "|---|---|---|---|---|---|---|",
            ]
            for _p_, a in arrivals:
                ok = {True: "在範圍內", False: "超出", None: "未知"}[a.get("ok")]
                dist = a.get("true_dist_m")
                out.append(
                    f"| {a.get('trial')} | {a.get('t', 0):.1f} s | {a.get('target_kind')} | {a.get('object', '—')} | "
                    f"{'—' if dist is None else f'{dist:.3f} m'} | {a.get('bound_m', '—')} m | {ok} |"
                )
            miss = [a["true_dist_m"] - a["bound_m"] for _p_, a in arrivals if a.get("ok") is False]
            if miss:
                out.append(
                    f"\n超出的 {len(miss)} 次，超出門檻 {min(miss) * 1000:.0f}–{max(miss) * 1000:.0f} mm。"
                )
        out.append("")
    return out


def _m5_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M5 收尾",
        "",
        "量的是：刺激引起的那段行為要自己結束，不能被切掉、也不能反覆開關。「刺激引起的那段」：刺激後 3 秒內第一個新開始的回合；"
        "如果 3 秒內有驚嚇、定格這類「定住身體」的反射，那一串反射就是這段。「自然結束」：這段結束時牠自己的滿足度 ≥ 0.5，"
        "或反射自己放開；被下一個行為搶走（滿足度 < 0.5）不算。門檻：自然結束 ≥ 80%、逾時 ≤ 10%、10 秒內重開 3 次以上 ≤ 5%。"
        "腦沒有切斷行為的計時器，所以逾時天生是 0。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        eps = diag.get(layer, {}).get("episodes", {})
        for p in presets:
            n = _one(lr, "M5", p, "刺激引起的那段：自然")
            o = _one(lr, "M5", p, "10 秒內")
            v1 = _one(lr, "M5", p, "（報告）第 1 版的分母")
            if n is None:
                continue
            e = eps.get(p, {})
            out.append(
                f"- {layer} {p}：自然結束 {n.ok}/{n.n}（{_p(n.value)}，`{n.verdict}`），"
                f"不自然的 {n.fail} 次；這段的種類：{'、'.join(f'{k} {v}' for k, v in sorted(e.items())) or '—'}；"
                f"10 秒內重開 {_p(o.value if o else None)}（`{o.verdict if o else '—'}`）"
                + (f"；用第 1 版的分母算是 {_p(v1.value)}" if v1 and v1.value is not None else "")
            )
        tot = eps.get("ALL", {})
        if tot:
            out.append(
                f"- {layer} 合併："
                + "、".join(f"{k} {v}" for k, v in sorted(tot.items()))
                + "（`hold`＝定住身體的反射一串，`bout`＝新開始的回合；`displaced`＝還沒滿足就被搶走）"
            )
        out.append("")
    return out


def _m6_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M6 同一隻鴨",
        "",
        f"量的是：同一個個性的鴨子換種子，是不是還認得出是同一種個性。每顆種子跑一整段生活（腦層：白天 2 小時＋夜晚 2 小時，5 Hz，"
        f"每 10 分鐘一次刺激，9 種輪流），做一份個性檔案（各行為的時間比例，加上對每種刺激的反應：換行為的比例、驚嚇的比例、反應快慢）。"
        f"猜個性：拿掉一顆種子，用其他種子算每個個性的平均檔案，把拿掉的那顆歸給最像的個性（歐氏距離，原始單位、不標準化）。"
        f"門檻：猜對 ≥ {S.M6_ACC_MIN:.0%}（亂猜 25%），而且每個個性自己被猜對 ≥ {S.M6_EACH_MIN:.0%}；"
        f"招牌動作：每個個性 ≥ {S.M6_SIG_MIN:.0%} 的種子，整段生活裡要出現（兩個招牌動作都要）。物理層規格事先寫明只報數字。",
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
                and "猜個性：" in r.part
                and "補充" not in r.part
            ),
            None,
        ) or next(
            (
                r
                for r in lr
                if r.metric == "M6" and r.scope == "ALL" and r.part.startswith("（報告）猜個性：")
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
        sens = [r for r in lr if r.metric == "M6" and r.part.startswith("（補充，不是判定）猜個性")]
        if sens:
            out.append("")
            out.append(
                "補充（不是判定，判定用的是上面事先定好的做法）："
                + "；".join(f"{r.part.split('：')[-1]} 猜對 {_p(r.value)}" for r in sens)
            )
        sp = diag.get(layer, {}).get("life_spread")
        if sp:
            parts = [
                f"{'行為時間比例' if k == 'time_shares' else '對刺激的反應'}（{v['dims']} 個數字）：同一個個性內種子到平均的平均距離 "
                f"{v['within_mean']}、個性平均之間的平均距離 {v['between_mean']}"
                for k, v in sp.items()
                if k != "behaviours"
            ]
            line = "個性檔案的兩個部分，分開看：" + "；".join(parts) + "。"
            noisy = [
                k
                for k, v in sp.items()
                if k != "behaviours" and v["within_mean"] > v["between_mean"]
            ]
            if noisy:
                line += (
                    "有一部分的「同一個性內種子之間的差」比「個性之間的差」還大"
                    f"（{'、'.join('對刺激的反應' if k == 'reactions' else '行為時間比例' for k in noisy)}），"
                    "在原始單位、不標準化的距離裡會蓋掉另一部分；這是事先定好的檔案做法的性質，不是修門檻的理由。"
                )
            out.append(line)
        out.append("")
    return out


def _m7_text(rows: Rows, presets: Sequence[str]) -> list[str]:
    out = ["### M7 安全（只在物理層）", ""]
    for layer in sorted(rows):
        lr = rows[layer]
        for p in presets:
            f = _one(lr, "M7", p, "有刺激組跌倒數")
            b = _one(lr, "M7", p, "安全反射被擋")
            if f is None or b is None:
                continue
            out.append(
                f"- {layer} {p}：有刺激組跌倒數 − 對照組 {f.value:+.0f}（N={f.n}，`{f.verdict}`）；"
                f"安全反射被擋 `{b.verdict}`"
                + (f"（{b.note}）" if b.verdict == A.INCOMPLETE and b.note else "")
            )
    out.append("")
    return out


# ── the report ──────────────────────────────────────────────────────────────────────


def build_report(
    *,
    rows: Rows,
    presets: Sequence[str],
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wall: Mapping[str, Any] | None,
    v1_rows: Rows | None,
    tables_md: str,
    failing_md: str,
    supplement_md: str,
) -> str:
    code = manifest.get("code", {})
    dims = int(
        diagnostics.get("L1", {}).get("life_spread", {}).get("behaviours", {}).get("dims", 0)
    )
    commit = str(code.get("commit", ""))[:7]
    date = str(manifest.get("started", ""))[:10]
    out: list[str] = [
        f"# 機器驗證 第 2 版 結果（{date}，程式版本 {commit}）",
        "",
        "給 owner 的白話版，由程式從每一列的判定資料自動寫出（重跑就會得到新的一份）。規格是 `docs/design/pet-behavior-v1/MACHINE-VERIFY-v2.md`"
        "（已凍結），門檻、定義、樣本數沒有改。規格沒寫清楚的操作選擇，在看到任何結果之前就寫好並提交了"
        "（`docs/design/pet-behavior-v1/MACHINE-VERIFY-v2-CHOICES.md`）。判定只用世界真相；腦自己說的話只拿來和真相比（M4）。",
        "",
        f"- 被驗的程式：`{code.get('code_root', '')}`，commit `{code.get('commit', '')}`"
        + ("（有未提交的改動！）" if code.get("dirty") else "（乾淨）"),
        f"- 驗證程式（錄製者與判定者）：`{code.get('verifier_root', '')}`，錄製時 commit `{str(code.get('verifier_commit', ''))[:7]}`",
        "",
        "## 一、先講結論",
        "",
        *_conclusions(rows, presets),
        "",
        "## 二、判定矩陣（每一層 × 個性 × 指標），和每格背後的數字",
        "",
        "`PASS`＝過；`FAIL`＝沒過；`incomplete`＝沒辦法判（缺資料，或規則在那些回合沒有定義；**不當通過**）；"
        "`未測`＝這一層沒量；`報告`＝規格事先寫明只報數字。未知的處理：把所有未知都當成對鴨子最不利，門檻還是過才算 `PASS`；"
        "都當成最有利還是不過就是 `FAIL`；中間是 `incomplete`。",
        "",
        R.key_numbers_markdown([r for lr in rows.values() for r in lr], presets),
        "",
        "## 三、和第 1 版比",
        "",
        *_versus_v1(rows, v1_rows, presets),
        "## 四、每一個沒過或判不出的項目，白話是什麼意思",
        "",
        "下面的句子都是從上面的數字和「補充（診斷）數字」直接寫出來的；診斷數字沒有門檻，不影響任何判定。",
        "",
    ]
    for metric, fn in (
        ("M1", lambda: _m1_text(rows, diagnostics, presets)),
        ("M2", lambda: _m2_text(rows)),
        ("M3", lambda: _m3_text(rows, diagnostics, presets)),
        ("M4", lambda: _m4_text(rows, presets)),
        ("M5", lambda: _m5_text(rows, diagnostics, presets)),
        ("M6", lambda: _m6_text(rows, diagnostics, presets)),
        ("M7", lambda: _m7_text(rows, presets)),
    ):
        if metric == "M2" or _unsettled(rows, metric, presets):
            out += fn()
    out += [
        "## 五、規格沒寫、事先定的操作選擇（要審查的）",
        "",
        "全文在 `MACHINE-VERIFY-v2-CHOICES.md`。最會影響結果的幾條：",
        "",
        "1. **哪些刺激「有方向」**照規格列的五種；SynthWorld 的快速逼近（手在相機光軸上）和摸頭（只有部位）沒有真實方向可轉，規則在這些回合沒有定義，記未知，不當成「沒注意」。戳用「左側被戳」＝鴨子左邊 0.4 m 的固定點當真實方向。",
        "2. **M5 的分母**用規格寫的定義（有刺激組窗內有那一段就算），不再要求對照組沒換行為（第 1 版的分母留在補充列）。「定住身體的反射」用腦自己的 `reflex.HOLDS`，從 `reflex` 欄位讀。",
        "3. **M6 的一整段生活**＝白天一段（09:00 起）＋夜晚一段（22:30 起），各模擬 2 小時、5 Hz，有家人；個性檔案用原始單位、不標準化；"
        "行為名單由這批生活出現過的行為決定，所以維度數是資料決定的"
        + (
            f"（這次：{dims} 種行為的時間比例 ＋ {3 * len(S.LIFE_STIM_ORDER)} 個反應數字 ＝ {dims + 3 * len(S.LIFE_STIM_ORDER)} 個數字；"
            "事先寫的選擇文件裡的「82 個」是我當時預估的行為數，沒有改那份文件）"
            if dims
            else ""
        )
        + "。",
        "4. **M4 的可碰距離**每個行為都用同一個量好的嘴巴伸出量 0.096 m ＋ 0.1 m。",
        f"5. **L2 的很大聲**用世界自己的 `stimulate` 路徑和它的 `db1m` 參數（{S.L2_BANG_DB1M:.0f} dB），並檢查每個回合頭的位置 ≥ {S.LOUD_HEAD_DB_MIN:.0f} dB。",
        "",
        "## 六、沒量到或沒判出來的（照規格寫 `未測` 或 `incomplete`）",
        "",
    ]
    listed = False
    for layer in sorted(rows):
        seen: set[tuple[str, str]] = set()
        for r in rows[layer]:
            if r.verdict in (A.UNTESTED, A.INCOMPLETE) and r.metric in METRIC_ZH:
                key = (r.metric, r.note or r.part)
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
                listed = True
    if not listed:
        out.append("（沒有。）")
    out += [
        "",
        "## 七、時間",
        "",
    ]
    if wall:
        names = {
            "l1": "L1 對照回合",
            "lives": "L1 整段生活",
            "l2": "L2 物理（一個行程，和 L1 同時跑）",
            "judge": "判定",
            "total": "全部（牆上時間）",
        }
        out.append("、".join(f"{names.get(k, k)} {v:.0f} 秒" for k, v in wall.items()) + "。")
    out += [
        "",
        "## 八、檔案與重跑",
        "",
        "一條指令（見同目錄 `RERUN.md`）：`python tools/machine_verify/run_v2.py --code-root <被驗的程式> --out <輸出目錄>`。"
        "這個目錄裡：`rows-L1.json`、`rows-L2.json`（每一列的值、分母、成功、失敗、未知、排除、門檻、判定、失敗的回合）、"
        "`tables.md`、`failing-trials.md`、`supplement.md`、`key-numbers.md`、`compare-v1-v2.md`、`diagnostics.json`、"
        "`L1/`、`L1_lives/`、`L2/`（原始紀錄和 manifest）、`manifest.json`、`logs/`。",
        "",
        "## 九、每一列的數字",
        "",
        "成功／失敗的意思依列而定（M1 配對差那列，成功＝有刺激組注意而對照組沒有；M5，成功＝自然結束；M3 溫和那列，成功＝沒嚇到；"
        "M3 很大聲那列，成功＝嚇到；M7 第一列，成功欄是有刺激組的跌倒數、失敗欄是對照組的跌倒數）。",
        "",
        tables_md.strip(),
        "",
        "## 十、補充數字（不是判定）",
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
