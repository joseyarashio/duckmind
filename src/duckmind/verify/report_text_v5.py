"""The owner-facing text of a v5 result, in plain Traditional Chinese, made by the program from the rows (the same rows the verdicts
are in), the diagnostics and the manifests: so that a rerun gets its own up-to-date report with one command. Nothing here judges
anything: every sentence quotes a row or a diagnostic number, and says what the metric measures.

M2–M7, the delayed ear, the placement and hand summaries and the support list are v3's and v4's text (those did not change), reused
from ``report_text_v3`` and ``report_text_v4``; what is new here are the M1 words (the target selection), the structure numbers of each
cell, the reference row of the known limit, and the v4 → v5 comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import report_text_v2 as RT2
from duckmind.verify import report_text_v3 as RT3
from duckmind.verify import report_text_v4 as RT4
from duckmind.verify import report_v2 as R2
from duckmind.verify import report_v3 as R3
from duckmind.verify import report_v5 as R
from duckmind.verify import spec_v5 as S

Rows = Mapping[str, Sequence[A.Row]]
METRIC_ZH = RT2.METRIC_ZH
EAR = S.EAR_LAYER
_p = RT2._p
_frac = RT4._frac
REPRODUCE_TITLE = "# 第 1–5 版重判：和原來的 rows 逐位元組比"
OLD_REPRODUCE_TITLE = "# 第 1–4 版重判：和原來的 rows 逐位元組比"
CARRY_RULES = (
    "v5.1",
    "v6",
    "v7",
)  # the rules that leave out only a selection that was there on the tick before the stimulus
KIND_RULES = ("v6", "v7")  # the rules that also count a threat target
VERSION = {"v5": "第 5 版", "v5.1": "第 5.1 版", "v6": "第 6 版", "v7": "第 7 版"}


# ── M1 (v5 words) ───────────────────────────────────────────────────────────────────────


def _object_meaning(c: Mapping[str, Any], oc: Mapping[str, Any] | None) -> str:
    """One failing or unsettled cell of a stimulus with a place, in plain words."""
    rate, diff = c["p_stim"], c["diff"]
    bits = []
    if c["verdict_rate"] != A.PASS:
        bits.append(
            f"有刺激組只有 {_p(rate, 0)} 把那個真實物體選為目標（要 ≥ 70%）"
            if c["verdict_rate"] == A.FAIL
            else f"有刺激組 {_p(rate, 0)}，但有未知的回合，判不出"
        )
    if c["verdict_diff"] != A.PASS:
        bits.append(
            f"比對照組只多 {'—' if diff is None else f'{diff:+.2f}'}（要 ≥ +0.5）：對照組（同一個位置、什麼都沒放）有 "
            f"{_p(c['p_sham'], 0)} 把空位置 0.3 m 內的東西選為目標"
            if c["verdict_diff"] == A.FAIL
            else "配對差距有未知的回合，判不出"
        )
    line = "；".join(bits)
    if oc and oc.get("read"):
        n = oc["read"]
        line += (
            f"。有刺激組命中 {oc['stim']}/{n}；其中刺激那個 tick 就已經選著、整段被排除的有 {oc['stim_at_stimulus']} 對"
            f"（排除後仍有另一段命中的 {oc['stim_at_stimulus_then_hit']} 對）；"
            f"刺激物體整段紀錄裡出現過有標籤的追蹤的對 {_frac(oc['label_exists'], n)}（寬鬆上限）；"
            f"窗裡鴨子拿「某個東西」當目標的對 {_frac(oc['stim_thing_windows'], n)}"
        )
        if "stim_by_threat" in oc:  # (v6 read this cell)
            line += (
                f"；有刺激組命中裡靠「威脅」目標才命中（只認 thing 的第 5.1 版讀法會判沒有）的 {oc['stim_by_threat']} 對，"
                f"窗裡鴨子的目標是「威脅」的對 {_frac(oc['stim_threat_windows'], n)}"
            )
        elif oc["esc_stim_either"]:
            line += f"；窗裡有反射或逃開的對 {_frac(oc['esc_stim_either'], n)}（逃開歸不到物體，見已知限制）"
    return line


def _m1_text(
    rows: Rows, diag: Mapping[str, Any], presets: Sequence[str], rule: str = "v5"
) -> list[str]:
    start = (
        "而且這一段不是刺激前一個 tick 就已經選著的延續（從刺激那個 tick 或之後開始都算；第 5.1 版勘誤）"
        if rule in CARRY_RULES
        else "而且這一段從刺激之後才開始（刺激當下就已經選著的不算）"
    )
    version = VERSION[rule]
    kinds = (
        "（目標種類是 `thing` 或 `threat`：被球嚇到逃開、退開也算，只要那個威脅目標的追蹤對到那個物體）"
        if rule in KIND_RULES
        else ""
    ) + (
        "；第 7 版另外認視覺反射（startle／freeze／orient／withdraw）：反射起頭在窗裡（刺激 tick 或之後），啟動它的來源追蹤對到那個物體，也算（不需要維持；"
        "對照組：來源追蹤對到空位置 0.3 公尺以內的東西）"
        if rule == "v7"
        else ""
    )
    out = [
        "### M1 注意（每一格自己判）",
        "",
        "量的是：刺激來了，鴨子是不是「真的注意到」，而且比什麼都沒發生的同一顆種子多多少。"
        f"**{version}**：有位置的刺激（滾球、放新東西）改成「大腦把那個真實的物體選為目標」——刺激後 3 秒內，目標欄位指向模擬器自己標成那個物體的追蹤，"
        f"至少連續 0.3 秒{kinds}，{start}；對照組＝同一個種子、什麼都沒放，大腦把目標選為「空位置 0.3 公尺以內的任何東西」"
        "（滾球的空位置跟著球原本的路線走），同樣至少連續 0.3 秒、同一條開始規則。"
        "其餘七種是「事件」（拍手、巨響、叫名字、打招呼、快速逼近、摸頭、戳）：3 秒內有沒有出現任何反射，和第 3、4 版一樣。"
        "每一格的門檻不變：有刺激組 ≥ 70%，而且有刺激組 − 對照組 ≥ 0.5。看不到、聽不到的回合排除在分母外。",
        "",
    ]
    for layer in sorted(rows):
        lr = rows[layer]
        cells = R3.m1_cells(lr)
        if not cells:
            continue
        refl_all = diag.get(layer, {}).get("cell_reflexes", {})
        obj_all = diag.get(layer, {}).get("object_cells", {})
        bad = [c for c in cells if c["verdict"] != A.PASS]
        good = len(cells) - len(bad)
        out.append(f"**{layer}**：{len(cells)} 格裡過 {good} 格，沒過或判不出 {len(bad)} 格。")
        if not bad:
            out.append("")
            continue
        for scope in RT3._scopes(layer if layer != EAR else "L1", presets):
            mine = [c for c in bad if c["scope"] == scope]
            if not mine:
                continue
            out.append(f"- {scope}：")
            for c in sorted(mine, key=lambda x: S.STIMULI.index(x["stimulus"])):
                key = f"{scope}|{c['stimulus']}"
                if c["stimulus"] in S.DIRECTIONAL:
                    meaning = _object_meaning(c, obj_all.get(key))
                else:
                    meaning = RT3._cell_meaning(c, refl_all.get(key))
                out.append(
                    f"  - {c['stimulus']}（`{c['verdict']}`）：{meaning}{RT3._seed_lists(c, scope)}"
                )
        out.append("")
    return out


# ── 結構數字 ────────────────────────────────────────────────────────────────────────────


def _structure_text(diag: Mapping[str, Any], rule: str = "v5") -> list[str]:
    """Per cell of a stimulus with a place: the numbers behind the two rates (no threshold reads them)."""
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        cells = d.get("object_cells") if isinstance(d, Mapping) else None
        if not cells:
            continue
        out.append(f"**{layer}**（每一格只看「看得到、兩組都判得出」的配對）：")
        out.append("")
        out.append(
            "| 格 | N | 有刺激組命中 | 對照組命中 | 贏（有刺激有、對照沒有） | 兩邊都有 | 只有對照 | 都沒有 | 刺激 tick 就選著而被整段排除（有刺激組／對照組） | "
            "其中刺激前一個 tick 就選著的（有刺激組／對照組） | 若刺激 tick 起頭的一段也算（只排除刺激前就有的）：有刺激組／對照組／贏（只報告） | "
            "刺激物體出現過有標籤的追蹤（寬鬆上限） | 窗裡拿某個東西當目標（有刺激組／對照組） | 對照組命中的東西 ／ 離空位置最近（m） |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, c in cells.items():
            n = c["read"]
            objs = "、".join(f"{k}×{v}" for k, v in c["ctrl_hit_objects"].items()) or "—"
            near = "、".join(str(x) for x in c["ctrl_hit_nearest_m"]) or "—"
            out.append(
                f"| {key.replace('|', ' ／ ')} | {n} | {_frac(c['stim'], n)} | {_frac(c['ctrl'], n)} | {c['wins']} | {c['both']} | "
                f"{c['only_ctrl']} | {c['neither']} | {c['stim_at_stimulus']}／{c['ctrl_at_stimulus']} | "
                f"{c['stim_before']}／{c['ctrl_before']} | {c['stim_alt']}／{c['ctrl_alt']}／{c['wins_alt']} | {_frac(c['label_exists'], n)} | "
                f"{c['stim_thing_windows']}／{c['ctrl_thing_windows']} | {objs} ／ {near} |"
            )
        out.append("")
    if not out:
        return ["（沒有。）", ""]
    first = (
        "這幾欄只報告、不判定。「刺激 tick 就選著」＝刺激那個 tick 這一組已經選中；其中「刺激前一個 tick 就選著」的是延續（規格要排除的），"
        "「刺激 tick 才開始選」的是反應（腦層的新東西在刺激那個 tick 就進了世界，腦在同一個 tick 就能反應）。"
    )
    if rule in CARRY_RULES:
        second = (
            "第 5.1 版（勘誤）的判定只排除延續（前一個 tick 就選著的），刺激 tick 才開始的一段照常算，所以「被整段排除」那一欄就是延續，"
            "「若刺激 tick 起頭的一段也算」那一欄和判定的命中數一樣。"
            + (
                "第 6 版的判定另外認目標種類 `threat`，下面的第二張表只報告這一條帶來的差別。"
                if rule == "v6"
                else (
                    "第 6 版的判定另外認目標種類 `threat`，第 7 版再加視覺反射的來源追蹤；下面的第二、第三張表只報告這兩條帶來的差別。"
                    if rule == "v7"
                    else ""
                )
            )
        )
    else:
        second = "第 5 版的判定把凡是從刺激 tick 起頭的一段都整段排除（延續和反應都排除）；「若刺激 tick 起頭的一段也算」那一欄只報告，就是第 5.1 版勘誤的讀法。"
    out.append(
        first
        + second
        + "「出現過有標籤的追蹤」用的是整段紀錄的標籤表（不是 3 秒窗），所以是寬鬆的上限。"
    )
    out.append("")
    if rule in KIND_RULES:
        out += _threat_table(diag)
    if rule == "v7":
        out += _reflex_table(diag)
    return out


def _threat_table(diag: Mapping[str, Any]) -> list[str]:
    """v6's own numbers per cell: what the threat target adds to the thing-only reading (reported, never judged)."""
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        cells = d.get("object_cells") if isinstance(d, Mapping) else None
        if not cells or not any("stim_by_threat" in c for c in cells.values()):
            continue
        out.append(f"**{layer}：第 6 版比只認 thing 的讀法多出來的（每格；只報告、不判定）**")
        out.append("")
        out.append(
            "| 格 | N | 有刺激組：第 6 版／只認 thing／靠 threat 才命中 | 對照組：第 6 版／只認 thing／靠 threat 才命中 | 贏：第 6 版／只認 thing | "
            "窗裡目標是 threat 的對（有刺激組／對照組） | 窗裡 threat 目標被選中的 tick 數（有刺激組／對照組） |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for key, c in cells.items():
            n = c["read"]
            out.append(
                f"| {key.replace('|', ' ／ ')} | {n} | {c['stim']}／{c.get('stim_thing_only', 0)}／{c.get('stim_by_threat', 0)} | "
                f"{c['ctrl']}／{c.get('ctrl_thing_only', 0)}／{c.get('ctrl_by_threat', 0)} | {c['wins']}／{c.get('wins_thing_only', 0)} | "
                f"{c.get('stim_threat_windows', 0)}／{c.get('ctrl_threat_windows', 0)} | "
                f"{c.get('stim_threat_selected_ticks', 0)}／{c.get('ctrl_threat_selected_ticks', 0)} |"
            )
        out.append("")
    return out


def _reflex_table(diag: Mapping[str, Any]) -> list[str]:
    """v7's own numbers per cell: what the reflex reading adds to v6's (reported, never judged)."""
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        cells = d.get("object_cells") if isinstance(d, Mapping) else None
        if not cells or not any("stim_by_reflex" in c for c in cells.values()):
            continue
        out.append(f"**{layer}：第 7 版比第 6 版多出來的（每格；只報告、不判定）**")
        out.append("")
        out.append(
            "| 格 | N | 有刺激組：第 7 版／第 6 版／靠反射才命中 | 對照組：第 7 版／第 6 版／靠反射才命中 | 贏：第 7 版／第 6 版 | "
            "窗裡有視覺反射起頭的對（有刺激組／對照組） | 反射起頭數：對到物體的／全部（有刺激組） | 反射起頭數：對到空位置附近東西的／全部（對照組） | "
            "命中的反射（有刺激組｜對照組） | 有 `reflex_track` 欄的對 |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, c in cells.items():
            n = c["read"]
            sk = "、".join(f"{k}×{v}" for k, v in c.get("stim_reflex_kinds", {}).items()) or "—"
            ck = "、".join(f"{k}×{v}" for k, v in c.get("ctrl_reflex_kinds", {}).items()) or "—"
            out.append(
                f"| {key.replace('|', ' ／ ')} | {n} | {c['stim']}／{c.get('stim_v6', 0)}／{c.get('stim_by_reflex', 0)} | "
                f"{c['ctrl']}／{c.get('ctrl_v6', 0)}／{c.get('ctrl_by_reflex', 0)} | {c['wins']}／{c.get('wins_v6', 0)} | "
                f"{c.get('stim_reflex_windows', 0)}／{c.get('ctrl_reflex_windows', 0)} | "
                f"{c.get('stim_reflex_on_target_onsets', 0)}／{c.get('stim_reflex_onsets', 0)} | "
                f"{c.get('ctrl_reflex_on_target_onsets', 0)}／{c.get('ctrl_reflex_onsets', 0)} | {sk}｜{ck} | "
                f"{c.get('recordings_with_reflex_track', 0)} |"
            )
        out.append("")
    return out


def _known_limit_text(rows: Rows, diag: Mapping[str, Any], rule: str = "v5") -> list[str]:
    """The reference rows of the known limit (physics layer's ball)."""
    out: list[str] = []
    for layer in sorted(rows):
        info = [r for r in rows[layer] if r.metric == "M1" and "反射或逃開" in r.part]
        if not info:
            continue
        if rule in KIND_RULES:
            out.append(
                f"**{layer} {S.INFO_STIMULUS}**：第 5 版寫下的已知限制是大腦的「威脅」目標沒有物體身分（track 一律是 0）。被驗的程式現在讓威脅目標帶來源追蹤編號"
                "（`fix/threat-track`，只是遙測），第 6 版的規則認得它：威脅目標的追蹤對到那顆球，就算選為目標。（舊紀錄的威脅目標 track 是 0，對不到任何東西。）"
                + (
                    "被嚇到但沒有逃開的反射（startle、freeze、orient）原本還是歸不到球；第 7 版讓反射也帶來源追蹤編號，認得它（見上面「第 7 版多出來的」表）。"
                    if rule == "v7"
                    else ""
                )
                + "下面照舊另外數「刺激後 3 秒內出現反射或逃開」的對數當參考（`報告`，不當門檻、不算未完成；這兩列的字句和格式沿用第 5 版）："
            )
        else:
            out.append(
                f"**{layer} {S.INFO_STIMULUS}**：大腦的「威脅」「地點」目標沒有物體身分（track 一律是 0），被球嚇到逃開、或走向球所在的地方，"
                "歸不到那顆球，所以規則量不到。這裡另外數「刺激後 3 秒內出現反射或逃開」的對數當參考（`報告`，不當門檻、不算未完成）："
            )
        out.append("")
        out.append(
            "| 哪一組 | 看得到的對 | 反射或逃開 | 其中有反射 | 其中有逃開（目標 threat，或行為 flee／retreat） |"
        )
        out.append("|---|---|---|---|---|")
        for r in info:
            e = r.extra
            out.append(
                f"| {'有刺激組' if e.get('arm') == 'stim' else '對照組'} | {e['n']} | {_frac(e['either'], e['n'])} | "
                f"{e['reflex']} | {e['flee']} |"
            )
        cell = (diag.get(layer, {}).get("object_cells", {}) or {}).get(f"ALL|{S.INFO_STIMULUS}")
        if cell:
            out.append("")
            out.append(
                f"對照：同一批對，把球選為目標（規則量到的）有刺激組 {cell['stim']}/{cell['read']}、對照組 {cell['ctrl']}/{cell['read']}。"
            )
        out.append("")
    return out or ["（這次沒有物理層的滾球。）", ""]


# ── 第 4 版 → 第 5 版 ─────────────────────────────────────────────────────────────────────


def _matrix_compare(old: Rows, new: Rows, presets: Sequence[str], a: str, b: str) -> list[str]:
    out: list[str] = []
    for layer in sorted(old):
        if layer not in new:
            continue
        x = A3.cell_verdicts(old[layer], presets)
        y = A3.cell_verdicts(new[layer], presets)
        out.append(f"**{layer}**：{a} → {b}\n")
        out.append("| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = []
            for p in presets:
                u, v = x.get((layer, metric, p), "—"), y.get((layer, metric, p), "—")
                cells.append(u if u == v else f"{u} → **{v}**")
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _m1_changes(old: Rows, new: Rows, a: str, b: str) -> list[str]:
    """Every M1 cell whose verdict or numbers moved between the two versions."""
    out = [
        f"| 層 | 對象 | 刺激 | {a}（判定 ／ 有刺激 ／ 對照 ／ 差 ／ N） | {b}（判定 ／ 有刺激 ／ 對照 ／ 差 ／ N） |",
        "|---|---|---|---|---|",
    ]

    def show(c: Mapping[str, Any] | None) -> str:
        if c is None:
            return "—"
        diff = "—" if c["diff"] is None else f"{c['diff']:+.2f}"
        return f"**{c['verdict']}** ／ {R._num(c['p_stim'])} ／ {R._num(c['p_sham'])} ／ {diff} ／ {c['n_pairs']}"

    n = 0
    for layer in sorted(old):
        if layer not in new:
            continue
        before = {(c["scope"], c["stimulus"]): c for c in R3.m1_cells(old[layer])}
        after = {(c["scope"], c["stimulus"]): c for c in R3.m1_cells(new[layer])}
        for key in sorted(after, key=lambda k: (k[0], S.STIMULI.index(k[1]))):
            u, v = before.get(key), after[key]
            if u is not None and show(u) == show(v):
                continue
            n += 1
            out.append(f"| {layer} | {key[0]} | {key[1]} | {show(u)} | {show(v)} |")
    if not n:
        return [f"{a}和{b}的 M1 每一格數字都一樣。"]
    return [f"M1 數字不一樣的格（共 {n} 格；沒列出的格數字完全相同）：", "", *out]


def _same_numbers(old: Sequence[A.Row], new: Sequence[A.Row]) -> str:
    before = {(r.metric, r.scope, r.part): r for r in old}
    same = diff = 0
    for r in new:
        if r.metric not in ("M2", "M3", "M4", "M5", "M6", "M7"):
            continue
        o = before.get((r.metric, r.scope, r.part))
        if o is None:
            continue
        if o.value == r.value and (o.n, o.ok, o.fail, o.unknown) == (r.n, r.ok, r.fail, r.unknown):
            same += 1
        else:
            diff += 1
    return f"M2–M7 兩個版本同名的列有 {same + diff} 列，{same} 列數字完全一樣、{diff} 列不同。"


def _versus_v4(
    old: Rows | None,
    new: Rows,
    presets: Sequence[str],
    recorded: Mapping[str, Any] | None,
    version: str = "第 5 版",
) -> list[str]:
    if not old:
        return ["（沒有給第 4 版的結果，無法並排。）", ""]
    same_recording = bool(recorded)
    out = [
        (
            "這次是「修正前」：判的是第 4 版已經錄好的那批紀錄（同一份被驗程式、同一批錄製），**只換 M1 的物體規則**，其他全部是第 4 版的判法；"
            "所以兩邊 M2–M7 每一列應該完全一樣，M1 的改變全來自規則。"
            if same_recording
            else "這次的紀錄是重新錄的（被驗的程式和第 4 版那次不同），所以兩邊的差別包括程式改動本身，也包括 M1 規則的改變。"
        ),
        "",
        *_matrix_compare(old, new, presets, "第 4 版", version),
        *_m1_changes(old, new, "第 4 版", version),
        "",
    ]
    for layer in sorted(old):
        if layer in new:
            out.append(f"- {layer}：{_same_numbers(old[layer], new[layer])}")
    out.append("")
    return out


def _versus_v5(old: Rows | None, new: Rows, presets: Sequence[str]) -> list[str]:
    """The v5 result beside the v5.1 one (the same recordings; only the start rule differs)."""
    if not old:
        return ["（沒有給第 5 版的結果，無法並排。）", ""]
    out = [
        "第 5 版的結果（`--v5-dir`，原樣保留，一個位元組都沒動）和這次第 5.1 版的結果用的是**同一批紀錄**，只差一條開始規則：第 5 版刺激 tick 起頭的一段整段不算，"
        "第 5.1 版只排除「刺激前一個 tick 就已經選著」的延續。所以兩邊 M2–M7 每一列應該完全一樣，M1 的改變全來自這一條。",
        "",
        *_matrix_compare(old, new, presets, "第 5 版", "第 5.1 版"),
        *_m1_changes(old, new, "第 5 版", "第 5.1 版"),
        "",
    ]
    for layer in sorted(old):
        if layer in new:
            out.append(f"- {layer}：{_same_numbers(old[layer], new[layer])}")
    out.append("")
    return out


def _versus_prev(
    old: Rows | None, new: Rows, presets: Sequence[str], a: str, b: str, differ: str
) -> list[str]:
    """The result of the previous version on the same recordings beside this one (only ``differ`` is different)."""
    if not old:
        return [f"（沒有給{a}的結果，無法並排。）", ""]
    out = [
        f"{a}的結果（同一批紀錄、原樣保留）和這次{b}的結果只差一件事：{differ}。"
        "所以兩邊 M2–M7 每一列應該完全一樣，M1 的改變全來自這一條。",
        "",
        *_matrix_compare(old, new, presets, a, b),
        *_m1_changes(old, new, a, b),
        "",
    ]
    for layer in sorted(old):
        if layer in new:
            out.append(f"- {layer}：{_same_numbers(old[layer], new[layer])}")
    out.append("")
    return out


def _versus_v51(old: Rows | None, new: Rows, presets: Sequence[str]) -> list[str]:
    return _versus_prev(
        old,
        new,
        presets,
        "第 5.1 版",
        "第 6 版",
        "第 6 版的「選中」也認目標種類 `threat`；沒有威脅目標帶追蹤編號的舊紀錄，M1 也應該完全一樣",
    )


def _versus_v6(old: Rows | None, new: Rows, presets: Sequence[str]) -> list[str]:
    return _versus_prev(
        old,
        new,
        presets,
        "第 6 版",
        "第 7 版",
        "第 7 版的「注意到」也認起頭在窗裡、來源追蹤對到那個物體的視覺反射；沒有 `reflex_track` 的舊紀錄，M1 也應該完全一樣",
    )


SPEC_DOCS = {
    "v5": "（已凍結）",
    "v5.1": "（已凍結）加 `MACHINE-VERIFY-v5.1-ERRATA.md`（規劃者的勘誤）",
    "v6": "（已凍結）、`MACHINE-VERIFY-v5.1-ERRATA.md`（規劃者的勘誤）和 `MACHINE-VERIFY-v6.md`（規劃者凍結的一句話）",
    "v7": "（已凍結）、`MACHINE-VERIFY-v5.1-ERRATA.md`（規劃者的勘誤）、`MACHINE-VERIFY-v6.md` 和 `MACHINE-VERIFY-v7.md`（規劃者凍結的一句話）",
}
CHOICES_DOCS = {
    "v5": "（`MACHINE-VERIFY-v5-CHOICES.md`）",
    "v5.1": "（`MACHINE-VERIFY-v5-CHOICES.md` 和 `MACHINE-VERIFY-v5.1-CHOICES.md`）",
    "v6": "（`MACHINE-VERIFY-v5-CHOICES.md`、`MACHINE-VERIFY-v5.1-CHOICES.md` 和 `MACHINE-VERIFY-v6-CHOICES.md`）",
    "v7": "（`MACHINE-VERIFY-v5-CHOICES.md`、`MACHINE-VERIFY-v5.1-CHOICES.md`、`MACHINE-VERIFY-v6-CHOICES.md` 和 `MACHINE-VERIFY-v7-CHOICES.md`）",
}
PREVIOUS = {"v5": "4", "v5.1": "5", "v6": "5.1", "v7": "6"}
WHAT_CHANGED = {
    "v5": "第 5 版只換了有位置的刺激（滾球、放新東西）的 M1。",
    "v5.1": "第 5.1 版＝第 5 版加勘誤，只改一件事：「刺激當下就已經選著的不算」改成「刺激前一個 tick 就已經選著的不算」。",
    "v6": "第 6 版＝第 5.1 版，只改一件事：目標種類是 `threat`（被嚇到逃開、退開）而且追蹤對到那個物體，也算「注意到」。",
    "v7": "第 7 版＝第 6 版，只改一件事：起頭在窗裡（刺激 tick 或之後）、來源追蹤對到那個物體的視覺反射（startle、freeze、orient、withdraw），也算「注意到」。",
}


def _before_after_text(
    before: Rows,
    after: Rows,
    presets: Sequence[str],
    info: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
    rule: str = "v5.1",
) -> list[str]:
    """The result of the code before the fix beside this one (the same spec and the same judge; only the duck's code differs)."""
    info = info or {}
    judged = VERSION[rule]
    commit_before = str(info.get("commit", ""))[:7] or "（沒寫）"
    commit_after = str(manifest.get("code", {}).get("commit", ""))[:7] or "（沒寫）"
    return [
        "## 一之二、修正前 → 修正後（同一套規格、同一個判法，只有鴨子的程式不一樣）",
        "",
        f"修正前：`{info.get('dir', '')}`（被驗的程式 `{commit_before}`，判定規則{judged}）。"
        f"修正後：這一份（被驗的程式 `{commit_after}`，整批重新錄製，判定規則{judged}）。"
        "兩邊的規格、門檻、樣本數、判定程式都一樣，沒有為了這次比較改任何東西；變的只有被驗的程式。"
        "下面先是判定矩陣（有箭頭的格＝判定變了，粗體是修正後），再逐列寫出有變的：判定變差、判定變好、判定沒變但數字變了；"
        "沒寫到的列就是完全一樣。",
        "",
        R2.side_by_side_markdown(before, after, presets, "修正前", "修正後"),
        "",
    ]


def _rule_effect_text(diag: Mapping[str, Any], version: str = "第 5 版") -> list[str]:
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        eff = d.get("rule_effect") if isinstance(d, Mapping) else None
        if not eff:
            continue
        out.append(
            f"**{layer}**：同一批紀錄，只換 M1 的規則（有位置的刺激），第 4 版（頭的方向）對{version}（選為目標）："
        )
        out.append("")
        out.append(
            f"| 對象 | 量什麼 | 第 4 版的規則（值 ／ N ／ 判定） | {version}的規則（值 ／ N ／ 判定） |"
        )
        out.append("|---|---|---|---|")
        for e in eff:
            a, b = (
                e["v3_rule"],
                e["v4_rule"],
            )  # (diagnostics_v4.rule_effect names its two sides v3_rule / v4_rule)

            def f(x: Mapping[str, Any], part: str = e["part"]) -> str:
                v = x["value"]
                s = (
                    "—"
                    if v is None
                    else (f"{v:+.2f}" if "差距" in part or "−" in part else _p(v, 0))
                )
                return f"{s} ／ {x['n']} ／ `{x['verdict']}`"

            out.append(f"| {e['scope']} | {e['part']} | {f(a)} | {f(b)} |")
        out.append("")
    return out or ["（沒有。）", ""]


# ── the report ──────────────────────────────────────────────────────────────────────────


def _header_lines(manifest: Mapping[str, Any]) -> list[str]:
    code = manifest.get("code", {})
    recorded = manifest.get("recorded") or {}
    stages = manifest.get("stages") or []
    lines = [
        f"- 被驗的程式：`{code.get('code_root', '')}`，commit `{code.get('commit', '')}`"
        + ("（有未提交的改動！）" if code.get("dirty") else "（乾淨）"),
    ]
    if recorded:
        rc = recorded.get("code", {})
        lines.append(
            f"- **錄製（沒有重錄）**：直接用第 4 版的紀錄 `{recorded.get('dir', '')}`（被驗的程式 commit `{str(rc.get('commit', ''))[:7]}`，"
            f"錄製開始 {recorded.get('started', '')}，錄製時驗證程式 commit `{str(rc.get('verifier_commit', ''))[:7]}`）；"
            "L1、L1_ear、L1_lives、L2 目錄是指向它的符號連結（只讀）。"
        )
        lines.append(
            "- 判定者（第 5 版驗證程式）：`"
            + str(code.get("verifier_root", ""))
            + "`，commit "
            + "、".join(
                f"`{str(s.get('verifier_commit', ''))[:7]}`（{', '.join(s.get('steps', []))}）"
                for s in stages
            )
        )
    else:
        lines.append(
            f"- 驗證程式（錄製者與判定者）：`{code.get('verifier_root', '')}`，錄製開始時 commit `{str(code.get('verifier_commit', ''))[:7]}`"
            + (
                "；之後的階段："
                + "、".join(
                    f"{str(s.get('verifier_commit', ''))[:7]}（{', '.join(s.get('steps', []))}）"
                    for s in stages
                )
                if len(stages) > 1
                else ""
            )
        )
    if manifest.get("report_written_by"):
        lines.append(
            f"- 這份報告和每一列由判定程式 commit `{str(manifest['report_written_by'])[:7]}` 寫出"
            "（判定是純函式，重判同一批紀錄得到逐位元組相同的 `rows-*.json`）。"
        )
    if manifest.get("rule") == "v5.1":
        lines.append(
            "- **判定規則：第 5.1 版**（`MACHINE-VERIFY-v5.1-ERRATA.md`：一段選為目標，只要從刺激 tick 或之後開始、而且刺激前一個 tick 沒有選著就算數；其餘同第 5 版）。"
            "第 5 版的結果原樣保留，並排在第四節。"
        )
    if manifest.get("rule") == "v6":
        lines.append(
            "- **判定規則：第 6 版**（`MACHINE-VERIFY-v6.md`，規劃者凍結的一句話：第 5.1 版，加上目標種類 `threat` 也算「選為目標」；其餘同第 5.1 版）。"
            "第 5.1 版的判法原樣保留，同一批紀錄的第 5.1 版結果並排在第四節。"
        )
    if manifest.get("rule") == "v7":
        lines.append(
            "- **判定規則：第 7 版**（`MACHINE-VERIFY-v7.md`，規劃者凍結的一句話：第 6 版，加上起頭在窗裡、來源追蹤對到刺激物體的視覺反射也算「注意到」；其餘同第 6 版）。"
            "第 6 版的判法原樣保留，同一批紀錄的第 6 版結果並排在第四節。"
        )
    lines.append(
        f"- 事前選擇文件的 commit `{str(manifest.get('choices_commit', ''))[:7]}`，提交時間 {manifest.get('choices_commit_time', '')}；"
        f"這次判定開始時間 {manifest.get('started', '')}；**選擇文件早於判定開始："
        + ("是（程式比對）" if manifest.get("choices_before_start") else "否！")
        + "**"
    )
    return lines


def build_report(
    *,
    rows: Rows,
    presets: Sequence[str],
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wall: Mapping[str, Any] | None,
    v4_rows: Rows | None,
    v5_rows: Rows | None = None,
    v51_rows: Rows | None = None,
    v6_rows: Rows | None = None,
    before_rows: Rows | None = None,
    before_info: Mapping[str, Any] | None = None,
    rule: str = "v5",
    tables_md: str,
    failing_md: str,
    supplement_md: str,
    m1_cells_md: str,
    identity_md: str,
    reproduce_md: str,
    physics_supported: Sequence[str],
) -> str:
    commit = str(manifest.get("code", {}).get("commit", ""))[:7]
    date = str(manifest.get("started", ""))[:10]
    recorded = manifest.get("recorded") or None
    version = VERSION[rule]
    out: list[str] = [
        f"# 機器驗證 {version} 結果（{date}，被驗的程式 {commit}"
        + ("，修正前" if recorded else ("，修正後" if before_rows is not None else ""))
        + "）",
        "",
        "給 owner 的白話版，由程式從每一列的判定資料自動寫出（重跑就會得到新的一份）。規格是 `docs/design/pet-behavior-v1/MACHINE-VERIFY-v5.md`"
        + SPEC_DOCS[rule]
        + "，門檻、定義、樣本數沒有改。規格沒寫清楚的操作選擇，在判任何結果之前就寫好並提交了"
        + CHOICES_DOCS[rule]
        + "。判定只用世界真相；腦自己說的話只拿來和真相比（M4）。"
        + f"第 1–{PREVIOUS[rule]} 版的規格和結果原樣保留。"
        + WHAT_CHANGED[rule],
        "",
        *_header_lines(manifest),
        "",
        "## 一、先講結論",
        "",
        *RT3._conclusions(rows, presets),
        "",
        *(
            _before_after_text(before_rows, rows, presets, before_info, manifest, rule)
            if before_rows is not None
            else []
        ),
        "## 二、判定矩陣（每一層 × 個性 × 指標），和每格背後的數字",
        "",
        "`PASS`＝過；`FAIL`＝沒過；`incomplete`＝沒辦法判（缺資料、規則在那些回合沒有定義，或推倒的次數不夠；**不當通過**）；"
        "`未測`＝這一層沒量；`報告`＝規格事先寫明只報數字。未知的處理：把所有未知都當成對鴨子最不利，門檻還是過才算 `PASS`；"
        "都當成最有利還是不過就是 `FAIL`；中間是 `incomplete`。M1 的格＝這個個性的 9 種刺激各一格（物理層是 9 格、4 個個性合併）。",
        "",
        R3.key_numbers_markdown([r for lr in rows.values() for r in lr], presets),
        "",
        "## 三、M1 每一格",
        "",
        f"每一格寫「判定／有刺激組注意率 ／ 對照組 ／ 配對差距／N（該格有效的配對數）」。滾球、放新東西的規則是{version}的「選為目標」。",
        "",
        m1_cells_md.strip(),
        "",
        *(
            [
                f"## 四、第 4 版 → {version}",
                "",
                *_versus_v4(v4_rows, rows, presets, recorded, version),
                *(
                    [
                        "## 四之二、第 5 版 → 第 5.1 版（同一批紀錄）",
                        "",
                        *_versus_v5(v5_rows, rows, presets),
                    ]
                    if rule == "v5.1"
                    else []
                ),
                *(
                    [
                        "## 四之二、第 5.1 版 → 第 6 版（同一批紀錄）",
                        "",
                        *_versus_v51(v51_rows, rows, presets),
                    ]
                    if rule == "v6"
                    else []
                ),
                *(
                    [
                        "## 四之二、第 6 版 → 第 7 版（同一批紀錄）",
                        "",
                        *_versus_v6(v6_rows, rows, presets),
                    ]
                    if rule == "v7"
                    else []
                ),
            ]
            if before_rows is None
            else [
                "## 四、和第 4、第 5 版的並排",
                "",
                "這一份的紀錄是被驗的程式修正後整批重錄的，和第 4、第 5 版用的不是同一批紀錄，直接並排會把「鴨子變了」和「判法變了」混在一起，所以不並排；"
                "要看修正的效果，請看「一之二」節（修正前、修正後都是第 5.1 版判法）。第 4、第 5 版的結果原樣保留在各自的目錄。",
                "",
            ]
        ),
        "## 五、M1 的結構數字（每格；不當門檻）",
        "",
        *_structure_text(diagnostics, rule),
        "### 已知限制：物理層滾球的參考列",
        "",
        *_known_limit_text(rows, diagnostics, rule),
        f"### 同一批紀錄，第 4 版的規則 ／ {version}的規則",
        "",
        *_rule_effect_text(diagnostics, version),
        "## 六、排除、擺放與逼近的手（紀錄是第 4 版的）",
        "",
        *RT4._placement_text(diagnostics),
        *RT4._loom_text(diagnostics, presets),
        "## 七、真實耳朵延遲版（腦層）",
        "",
        *RT3._ear_text(rows, diagnostics, presets),
        "## 八、每一個沒過或判不出的項目，白話是什麼意思",
        "",
        "下面的句子都是從上面的數字和診斷數字直接寫出來的；診斷數字沒有門檻，不影響任何判定。",
        "",
    ]
    for metric, fn in (
        ("M1", lambda: _m1_text(rows, diagnostics, presets, rule)),
        ("M2", lambda: RT3._m2_text(rows)),
        ("M3", lambda: RT2._m3_text(rows, diagnostics, presets)),
        ("M4", lambda: RT2._m4_text(rows, presets)),
        ("M5", lambda: RT2._m5_text(rows, diagnostics, presets)),
        ("M6", lambda: RT3._m6_text(rows, diagnostics, presets)),
        ("M7", lambda: RT3._m7_text(rows, diagnostics, presets)),
    ):
        if metric in ("M1", "M2", "M6", "M7") or RT3._unsettled(rows, metric, presets):
            out += fn()
    out += [
        "## 九、物理世界的刺激介面：測了哪些、哪些「未測」",
        "",
        *RT4._support_text(physics_supported),
        "## 十、規格沒寫、事先定的操作選擇（要審查的）",
        "",
        "全文在 `MACHINE-VERIFY-v5-CHOICES.md`（在任何 v5 判定結果出現以前提交）"
        + (
            "、`MACHINE-VERIFY-v5.1-CHOICES.md`、`MACHINE-VERIFY-v6-CHOICES.md`"
            if rule == "v6"
            else (
                "、`MACHINE-VERIFY-v5.1-CHOICES.md`、`MACHINE-VERIFY-v6-CHOICES.md`、`MACHINE-VERIFY-v7-CHOICES.md`"
                if rule == "v7"
                else ""
            )
        )
        + "。最會影響結果的幾條：",
        "",
        (
            "1. **選中**：有刺激組＝目標種類是 `thing` **或 `threat`**，追蹤經標籤表對到刺激物體的 id（不用方位猜）；對照組＝目標種類是 `thing` 或 `threat`、追蹤有標籤、"
            "那個東西在這個 tick 的真實位置離空位置 ≤ 0.3 m。`place`、`person`、`nest` 的目標不算；追蹤是 0 或沒有的 tick 查不到標籤，不算。種類可以在同一段裡從 `thing` 換成 `threat`。"
            + (
                "第 7 版再加：視覺反射（startle／freeze／orient／withdraw，`reflex_track` 不是 0）的起頭在窗裡（刺激 tick 或之後），來源追蹤經標籤表對到刺激物體"
                "（對照組：對到空位置 0.3 m 內的東西，位置取反射起頭那個 tick）；反射不需要維持；一個反射的起頭＝同一個來源追蹤的連續視覺反射 tick 的第一個，"
                "刺激前一個 tick 就在作用的不算；紀錄沒有 `reflex_track` 欄就沒有反射來源可讀（讀成「沒有」，和第 6 版逐列相同）。"
                if rule == "v7"
                else ""
            )
            if rule in KIND_RULES
            else "1. **選中**：有刺激組＝目標種類是 `thing`，追蹤經標籤表對到刺激物體的 id（不用方位猜）；對照組＝目標種類是 `thing`、追蹤有標籤、那個東西在這個 tick 的真實位置離空位置 ≤ 0.3 m。"
            "`threat`、`place` 的目標沒有物體身分，不算。"
        ),
        (
            "2. **一段**：窗 3.0 秒含兩端；連續 tick 的第一個到最後一個的時間 ≥ 0.3 秒；**從刺激 tick 或之後開始，而且刺激前一個 tick 沒有選著**"
            "（刺激前就選著、延續到刺激 tick 的整段不算，兩組一樣；它不擋住之後另開的一段）。"
            if rule in CARRY_RULES
            else "2. **一段**：窗 3.0 秒含兩端；連續 tick 的第一個到最後一個的時間 ≥ 0.3 秒；**第一個 tick 必須晚於刺激 tick**（刺激當下就選著的整段不算，兩組一樣；它不擋住之後另開的一段）。"
        ),
        "3. **空位置**：放新東西＝那個位置；滾球＝有刺激組的球在同一個 tick 的位置（跟著球原本的路線動）。",
        "4. **未知**只有真相缺了才有（沒有標籤表、沒有東西位置、沒有空位置路線）；標籤表有、但沒有這個物體，是「沒有」。",
        "5. **滾球的參考列**（物理層）：窗裡有反射，或有逃開（目標種類 threat／行為 flee、retreat），不當門檻。",
        "6. 其餘（M1 的事件類、分格、M2–M7、擺放、朝臉的手、延遲耳朵、推一下）都是第 3、4 版的選擇。",
        "",
        "## 十一、沒量到或沒判出來的（照規格寫 `未測` 或 `incomplete`）",
        "",
        *RT3._unmeasured(rows),
        "",
        "## 十二、第 1–5 版重判（舊結果還能重現）",
        "",
        reproduce_md.replace(REPRODUCE_TITLE, "").replace(OLD_REPRODUCE_TITLE, "").strip()
        or "（這次沒有跑重判。）",
        "",
        "## 十三、錄製身分檢查（同一份程式兩次錄製是不是同一份紀錄）",
        "",
        identity_md.replace("# 錄製身分檢查（不是判定）", "").strip() or "（沒有比。）",
        "",
        "## 十四、時間",
        "",
    ]
    names = {
        "l1": "L1 對照回合",
        "l1ear": "L1 延遲耳朵版",
        "lives": "L1 整段生活",
        "l2": f"L2 物理（{manifest.get('l2_shards', 2)} 個行程）",
        "l2_wait": "L2 等另一件工作的物理做完",
        "reproduce": "第 1–5 版重判",
        "judge": "判定",
        "sum_of_steps": "各步驟加總",
        "total": "從開始到現在（牆上時間）",
    }
    if wall:
        out.append("、".join(f"{names.get(k, k)} {v:.0f} 秒" for k, v in wall.items()) + "。")
    if recorded and recorded.get("wall"):
        out.append(
            "（沿用的第 4 版錄製的時間："
            + "、".join(f"{names.get(k, k)} {v:.0f} 秒" for k, v in recorded["wall"].items())
            + "。）"
        )
    out += [
        "",
        "## 十五、檔案與重跑",
        "",
        "一條指令（見同目錄 `MACHINE-VERIFY-v5-RERUN.md`）：`python tools/machine_verify/run_v5.py --code-root <被驗的程式> --out <輸出目錄>`"
        "（修正前：加 `--from-recorded <第 4 版結果目錄>`，只判不錄；修正後：加 `--before-dir <修正前的第 5.1 版結果目錄>`，報告會把兩份並排）。"
        "這個目錄裡：`rows-L1.json`、`rows-L1ear.json`、`rows-L2.json`（每一列的值、分母、成功、失敗、未知、排除、門檻、判定、失敗的回合）、"
        "`tables.md`、`m1-cells.md`、`failing-trials.md`、`supplement.md`、`key-numbers.md`"
        + (
            (f"、`compare-v4-{rule}.md`" if rule in KIND_RULES else "、`compare-v4-v5.md`")
            if before_rows is None
            else ""
        )
        + ("、`compare-v5-v5.1.md`" if rule == "v5.1" and before_rows is None else "")
        + ("、`compare-v5.1-v6.md`" if rule == "v6" and before_rows is None else "")
        + ("、`compare-v6-v7.md`" if rule == "v7" and before_rows is None else "")
        + ("、`compare-before-after.md`" if before_rows is not None else "")
        + "、`trace-identity.md`、"
        "`reproduce-v1-v2-v3-v4-v5.md`、`diagnostics.json`、`L1/`、`L1_ear/`、`L1_lives/`、`L2/`、`manifest.json`、`wall.json`、`logs/`。",
        "",
        "## 十六、每一列的數字",
        "",
        "成功／失敗的意思依列而定（M1 配對差那列，成功＝有刺激組注意而對照組沒有；M5，成功＝自然結束；M3 溫和那列，成功＝沒嚇到；"
        "M3 很大聲那列，成功＝嚇到；M7 第一列，成功欄是有刺激組的跌倒數、失敗欄是對照組的跌倒數；M7 推一下那列，N＝推倒的次數、成功＝沒被擋、失敗＝被擋）。",
        "",
        tables_md.strip(),
        "",
        "## 十七、補充數字（不是判定）",
        "",
        supplement_md.strip(),
        "",
        "## 附錄：失敗與未知的回合（全部）",
        "",
        failing_md.replace("# 失敗與未知的回合（全部列出）", "").strip(),
        "",
    ]
    return "\n".join(out)


__all__ = ["METRIC_ZH", "REPRODUCE_TITLE", "build_report"]
