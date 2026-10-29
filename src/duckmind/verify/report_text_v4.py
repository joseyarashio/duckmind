"""The owner-facing text of a v4 result, in plain Traditional Chinese, made by the program from the rows (the same rows the
verdicts are in), the diagnostics and the manifests: so that a rerun on other code gets its own up-to-date report with
one command. Nothing here judges anything: every sentence quotes a row or a diagnostic number, and says what the metric
measures.

The pieces that are plain tables are made by ``report_v4`` / ``report_v3``; M2–M7, the delayed ear and the support list are
v3's text (those metrics did not change), reused from ``report_text_v3``; what is new here are the M1 words (the
fixation), the placement and exclusion section, the hand section, the same-recording-two-rules section and the v3 → v4
comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v3 as A3
from duckmind.verify import report_text_v2 as RT2
from duckmind.verify import report_text_v3 as RT3
from duckmind.verify import report_v3 as R3
from duckmind.verify import report_v4 as R
from duckmind.verify import spec_v4 as S

Rows = Mapping[str, Sequence[A.Row]]
METRIC_ZH = RT2.METRIC_ZH
EAR = S.EAR_LAYER
_p = RT2._p


def _frac(k: int | None, n: int | None) -> str:
    return "—" if k is None or not n else f"{k}/{n}（{k / n * 100:.0f}%）"


# ── M1 (v4 words) ───────────────────────────────────────────────────────────────────────


def _object_meaning(c: Mapping[str, Any], oc: Mapping[str, Any] | None) -> str:
    """One failing or unsettled cell of a stimulus with a place, in plain words."""
    rate, diff = c["p_stim"], c["diff"]
    bits = []
    if c["verdict_rate"] != A.PASS:
        bits.append(
            f"有刺激組只有 {_p(rate, 0)} 盯住了（要 ≥ 70%）"
            if c["verdict_rate"] == A.FAIL
            else f"有刺激組盯住率 {_p(rate, 0)}，但有未知的回合，判不出"
        )
    if c["verdict_diff"] != A.PASS:
        bits.append(
            f"比對照組只多 {'—' if diff is None else f'{diff:+.2f}'}（要 ≥ +0.5）：對照組（同一個位置、什麼都沒放）的頭自己也有 "
            f"{_p(c['p_sham'], 0)} 盯住了那個空位置"
            if c["verdict_diff"] == A.FAIL
            else "配對差距有未知的回合，判不出"
        )
    line = "；".join(bits)
    if oc and oc.get("read"):
        n = oc["read"]
        line += (
            f"。刺激那一刻頭就已經在看（差距 ≤ 15°）的有 {_frac(oc['already_inside'], n)} 對——"
            "這種回合規格說「不是轉過去的，不是本來就在看」，只有先離開 15° 再用更小的差距回來才算"
        )
        if oc["out_n"]:
            line += (
                f"；只看刺激那一刻差距 > 15° 的 {oc['out_n']} 對：有刺激組 {_frac(oc['out_stim'], oc['out_n'])}、"
                f"對照組 {_frac(oc['out_sham'], oc['out_n'])}"
            )
    return line


def _m1_text(rows: Rows, diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    out = [
        "### M1 注意（每一格自己判）",
        "",
        "量的是：刺激來了，鴨子是不是「真的注意到」，而且比什麼都沒發生的同一顆種子多多少。"
        "**第 4 版**：有位置的刺激（滾球、放新東西）改成「盯住」——刺激後 3 秒內，頭的方向和真實物體方位的差距**連續 ≥ 0.5 秒都 ≤ 15°**，"
        "而且這一段開始時的差距比刺激當下的差距小（是轉過去的，不是本來就在看）；對照組＝同一個位置、同一條規則，看頭有沒有盯住那個空位置。"
        "其餘七種是「事件」（拍手、巨響、叫名字、打招呼、快速逼近、摸頭、戳）：3 秒內有沒有出現任何反射，和第 3 版一樣。"
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


# ── 四、第 3 版 → 第 4 版 ─────────────────────────────────────────────────────────────────


def _m1_changes(v3: Rows, v4: Rows) -> list[str]:
    """Every M1 cell whose verdict or numbers moved between the two versions (the cells of the unchanged event rule too:
    they are re-recorded)."""
    out = [
        "| 層 | 對象 | 刺激 | 第 3 版（判定 ／ 有刺激 ／ 對照 ／ 差 ／ N） | 第 4 版（判定 ／ 有刺激 ／ 對照 ／ 差 ／ N） |",
        "|---|---|---|---|---|",
    ]

    def show(c: Mapping[str, Any] | None) -> str:
        if c is None:
            return "—"
        diff = "—" if c["diff"] is None else f"{c['diff']:+.2f}"
        return f"**{c['verdict']}** ／ {R._num(c['p_stim'])} ／ {R._num(c['p_sham'])} ／ {diff} ／ {c['n_pairs']}"

    n = 0
    for layer in sorted(v3):
        if layer not in v4:
            continue
        old = {(c["scope"], c["stimulus"]): c for c in R3.m1_cells(v3[layer])}
        new = {(c["scope"], c["stimulus"]): c for c in R3.m1_cells(v4[layer])}
        for key in sorted(new, key=lambda k: (k[0], S.STIMULI.index(k[1]))):
            a, b = old.get(key), new[key]
            if a is not None and show(a) == show(b):
                continue
            n += 1
            out.append(f"| {layer} | {key[0]} | {key[1]} | {show(a)} | {show(b)} |")
    if not n:
        return ["第 3 版和第 4 版的 M1 每一格數字都一樣。"]
    return [f"M1 數字不一樣的格（共 {n} 格；沒列出的格數字完全相同）：", "", *out]


def _same_numbers(r3: Sequence[A.Row], r4: Sequence[A.Row]) -> str:
    """Rows of the metrics v4 did not change (M2–M7) that have the same name in both versions: how many have exactly the
    same numbers."""
    old = {(r.metric, r.scope, r.part): r for r in r3}
    same = diff = 0
    for r in r4:
        if r.metric not in ("M2", "M3", "M4", "M5", "M6", "M7"):
            continue
        o = old.get((r.metric, r.scope, r.part))
        if o is None:
            continue
        if o.value == r.value and (o.n, o.ok, o.fail, o.unknown) == (r.n, r.ok, r.fail, r.unknown):
            same += 1
        else:
            diff += 1
    tail = (
        "；不一樣的列來自改了擺放的滾球、放新東西和改了路線的逼近（它們的回合進了 M2、M3、M5 的分母），細節在 `compare-v3-v4.md`"
        if diff
        else ""
    )
    return f"M2–M7 第 3 版和第 4 版同名的列有 {same + diff} 列，{same} 列數字完全一樣、{diff} 列不同{tail}。"


def _matrix_compare(v3: Rows, v4: Rows, presets: Sequence[str]) -> list[str]:
    """The verdict matrices of the two versions side by side: ``a → b`` where a cell moved."""
    out: list[str] = []
    for layer in sorted(v3):
        if layer not in v4:
            continue
        a = A3.cell_verdicts(v3[layer], presets)
        b = A3.cell_verdicts(v4[layer], presets)
        out.append(f"**{layer}**：第 3 版 → 第 4 版\n")
        out.append("| 指標 | " + " | ".join(presets) + " |\n|---|" + "---|" * len(presets))
        for metric in ("M1", "M3", "M4", "M5", "M6", "M7"):
            cells = []
            for p in presets:
                x, y = a.get((layer, metric, p), "—"), b.get((layer, metric, p), "—")
                cells.append(x if x == y else f"{x} → **{y}**")
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _versus_v3(v3: Rows | None, v4: Rows, presets: Sequence[str]) -> list[str]:
    if not v3:
        return ["（沒有給第 3 版的結果，無法並排。）", ""]
    out = [
        "第 3 版那一側＝第 3 版規格在 `eed2d6b` 上跑出來的結果；`eed2d6b` 和這次被驗的程式的鴨子程式一個位元組都沒差"
        "（兩個 commit 只差 `src/duckmind/verify/` 的驗證程式），所以那一側就是同一隻鴨在第 3 版規格下的結果。"
        "第 4 版改的只有：有方向刺激（滾球、放新東西）的 M1 規則、物理層物體的擺放、物理層逼近的手的路線。",
        "",
        *_matrix_compare(v3, v4, presets),
        *_m1_changes(v3, v4),
        "",
    ]
    for layer in sorted(v3):
        if layer in v4:
            out.append(f"- {layer}：{_same_numbers(v3[layer], v4[layer])}")
    out.append("")
    return out


# ── 五、第 4 版新增：擺放、排除、已經在看 ────────────────────────────────────────────────


def _spread(s: Mapping[str, float] | None, unit: str = "") -> str:
    return "—" if not s else f"{s['min']}／{s['median']}／{s['max']}{unit}"


def _placement_text(diag: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        ex = d.get("exclusions") if isinstance(d, Mapping) else None
        if not ex:
            continue
        out.append(
            f"**{layer}：每種刺激排除了幾對**（看不到的回合；擺放的設計目標是物理層 ≤ {S.EXCLUDED_TARGET:.0%}，不是判定門檻）"
        )
        out.append("")
        out.append(
            "| 刺激 | 計畫 | 看得到 | 排除（看不到） | 排除率 | 看得到還不知道 | 無效 | 缺 |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for stimulus, e in ex.items():
            rate = "—" if e["rate"] is None else f"{e['rate'] * 100:.1f}%"
            flag = ""
            if layer == "L2" and e["rate"] is not None and e["rate"] > e["target"] + 1e-9:
                flag = "（超過 10%）"
            out.append(
                f"| {stimulus} | {e['planned']} | {e['in_view']} | {e['out_of_view']} | {rate}{flag} | "
                f"{e['view_unknown']} | {e['invalid']} | {e['missing']} |"
            )
        out.append("")
    pl = diag.get("L2", {}).get("placement") if isinstance(diag.get("L2"), Mapping) else None
    if pl:
        out.append(
            "**L2 物體的擺放**（有刺激組；水平範圍＝世界的 `in_frame` 掃出來的、畫面裡的最長連續區間，逆時針為正＝鴨子的左邊；"
            "下面每個範圍是 最小／中位數／最大）："
        )
        out.append("")
        out.append(
            "| 刺激 | 紀錄 | 擺放方式 | 左／右 | 範圍下緣（°） | 範圍上緣（°） | 用的方位 φ（°） | 刺激那一刻的差距（°） | 差距 ≤ 15° 的 |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for stimulus, p in pl.items():
            modes = "、".join(f"{k} {v}" for k, v in p["mode"].items()) or "—"
            sides = "、".join(f"{k} {v}" for k, v in p["side"].items()) or "—"
            out.append(
                f"| {stimulus} | {p['with_placement']}/{p['trials']} | {modes} | {sides} | {_spread(p['span_lo_deg'])} | "
                f"{_spread(p['span_hi_deg'])} | {_spread(p['phi_deg'])} | {_spread(p['g0_deg'])} | "
                f"{p['g0_within_15']}/{p['g0_n']} |"
            )
        out.append("")
        out.append(
            "擺放方式：`band`＝第一個選到的位置就被世界接受；`ladder`＝世界拒絕了，用同一個範圍裡的下一個位置或較近的距離；`v3`＝讀不到範圍或全部被拒絕，"
            "退回第 3 版的位置（會被排除）。"
        )
        out.append("")
    return out


def _readings_text(diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    """For the cells of a stimulus with a place: the judged reading, and the reported ones, on the same pairs."""
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        cells = d.get("object_cells") if isinstance(d, Mapping) else None
        if not cells:
            continue
        out.append(
            f"**{layer}**（每一格都只看「看得到、兩組都判得出」的配對；括號是有刺激組／對照組的盯住數）："
        )
        out.append("")
        out.append(
            "| 格 | 配對數 | **盯住（判定）** | 刺激那一刻已經在看（≤15°） | 差距 > 15° 的配對：有刺激／對照 | "
            "同一批紀錄，第 3 版的轉頭規則 | 「任一起點」讀法 |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for key, c in cells.items():
            n = c["read"]
            out.append(
                f"| {key.replace('|', ' ／ ')} | {n} | **{_frac(c['stim'], n)} ／ {_frac(c['sham'], n)}** | {_frac(c['already_inside'], n)} | "
                f"{_frac(c['out_stim'], c['out_n'])} ／ {_frac(c['out_sham'], c['out_n'])} | "
                f"{_frac(c['turn_stim'], c['turn_n'])} ／ {_frac(c['turn_sham'], c['turn_n'])} | "
                f"{_frac(c['exist_stim'], n)} ／ {_frac(c['exist_sham'], n)} |"
            )
        out.append("")
    if out:
        out.append(
            "這幾欄**只報告、不判定**（CHOICES §二.9）：判定只用「盯住」那一欄。「已經在看」的意思：物體放在畫面中間，鴨子的頭本來就離它不到 15°，"
            "這樣的回合只有先離開再用更小的差距回來才算盯住；「任一起點」讀法是一段 0.5 秒都 ≤ 15° 的連續區間裡，只要有一個時間點的差距比刺激當下小就算。"
        )
        out.append("")
    return out


def _loom_text(diag: Mapping[str, Any], presets: Sequence[str]) -> list[str]:
    ls = diag.get("L2", {}).get("loom") if isinstance(diag.get("L2"), Mapping) else None
    if not ls:
        return ["（這次沒有物理層的逼近。）", ""]
    out = [
        "第 4 版的手：從鴨子畫面正中央、離鏡頭 0.75 m 出現（停 0.25 秒），0.45 秒內靠近到嘴尖前 0.09 m（停 0.35 秒），再 0.6 秒離開；"
        "每個 tick 用當下的相機和嘴尖重算路線，所以頭轉了手跟著轉。時間和距離都是第 3 版的。",
        "",
        "| 個性 | 配對 | 手有送出 | 看得到（1.5 秒內頭相機取樣裡有手） | 看不到 | 還不知道 | 有刺激組 3 秒內有反射 | 對照組有反射 | 有刺激組驚嚇／定格 | 對照組驚嚇／定格 | 手離嘴尖最近（m，最小／中位／最大） |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scope in (*presets, "ALL"):
        c = ls.get(scope)
        if not c:
            continue
        n = c.get("pairs", 0)
        out.append(
            f"| {scope} | {n} | {c.get('delivered', 0)} | {c.get('seen', 0)} | {c.get('not_seen', 0)} | {c.get('seen_unknown', 0)} | "
            f"{c.get('reflex_stim', 0)} | {c.get('reflex_sham', 0)} | {c.get('startle_stim', 0)} | {c.get('startle_sham', 0)} | "
            f"{_spread(c.get('min_gap_to_beak_m'))} |"
        )
    out.append("")
    al = ls.get("ALL") or {}
    if al.get("start_lens_m"):
        out.append(
            f"手一出現的位置離鏡頭（m，最小／中位／最大）：{_spread(al['start_lens_m'])}；起點被抬高到終點（嘴尖前）的高度的有 "
            f"{al.get('start_lifted', 0)}/{al.get('pairs', 0)} 組（鏡頭低頭時畫面中央的射線在 0.75 m 處比嘴尖低，起點就抬到嘴尖的高度，只有水平方向取畫面中央；"
            "沒被抬高的，起點離鏡頭剛好 0.75 m）。"
        )
        out.append("")
    return out


def _rule_effect_text(diag: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for layer in sorted(diag):
        d = diag[layer]
        eff = d.get("rule_effect") if isinstance(d, Mapping) else None
        if not eff:
            continue
        out.append(f"**{layer}**：同一批第 4 版的紀錄，只換 M1 的規則（有位置的刺激）：")
        out.append("")
        out.append(
            "| 對象 | 量什麼 | 第 3 版的規則（值 ／ N ／ 判定） | 第 4 版的規則（值 ／ N ／ 判定） |"
        )
        out.append("|---|---|---|---|")
        for e in eff:
            a, b = e["v3_rule"], e["v4_rule"]

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


# ── 七、物理世界的刺激介面（第 4 版的用法）─────────────────────────────────────────────────


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
        "roll_ball": "世界的 `x_roll_ball`（**第 4 版**：放在畫面實際水平範圍的中間 60%，左右隨機，1.4 m 處以 900 mm/s 滾向鴨子）",
        "novel_object": "世界的 `x_place_prop`（**第 4 版**：放在畫面實際水平範圍的中間 60%，左右隨機，1.2 m 處）",
        "clap": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "call_name": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "hello": "`stimulate`（來源在鴨子 1.5–2.5 m 外）",
        "bang": "`stimulate`，加 `db1m` 100（頭的位置要 ≥ 90 dB，否則該回合無效）",
        "pet_head": "`stimulate`（手跟著頭，預設摸 2 下）",
        "poke": "`stimulate`（手在軀幹側邊，偶數種子左側、奇數種子右側）",
        "hand_loom": "**第 4 版**：實驗程式每個 tick 移動世界的手（畫面中央起點、朝嘴尖、跟著頭，時間和距離同第 3 版；沒有呼叫 `stimulate`）；看得到＝1.5 秒內頭相機取樣看得到手",
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


# ── the report ──────────────────────────────────────────────────────────────────────────


def build_report(
    *,
    rows: Rows,
    presets: Sequence[str],
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wall: Mapping[str, Any] | None,
    v3_rows: Rows | None,
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
        f"# 機器驗證 第 4 版 結果（{date}，程式版本 {commit}）",
        "",
        "給 owner 的白話版，由程式從每一列的判定資料自動寫出（重跑就會得到新的一份）。規格是 `docs/design/pet-behavior-v1/MACHINE-VERIFY-v4.md`"
        "（已凍結），門檻、定義、樣本數沒有改。規格沒寫清楚的操作選擇，在看到任何結果之前就寫好並提交了"
        "（`docs/design/pet-behavior-v1/MACHINE-VERIFY-v4-CHOICES.md`）。判定只用世界真相；腦自己說的話只拿來和真相比（M4）。"
        "第 1、2、3 版的規格和結果原樣保留。",
        "",
        f"- 被驗的程式：`{code.get('code_root', '')}`，commit `{code.get('commit', '')}`"
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
        *RT3._conclusions(rows, presets),
        "",
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
        "每一格寫「判定／有刺激組注意率 ／ 對照組 ／ 配對差距／N（該格有效的配對數）」。滾球、放新東西的規則是第 4 版的「盯住」。",
        "",
        m1_cells_md.strip(),
        "",
        "## 四、第 3 版 → 第 4 版（同一份被驗程式的鴨子）",
        "",
        *_versus_v3(v3_rows, rows, presets),
        "## 五、第 4 版新增的數字：擺放、排除、「已經在看」",
        "",
        *_placement_text(diagnostics),
        "### 同一批配對，不同讀法（只報告）",
        "",
        *_readings_text(diagnostics, presets),
        "### 同一批紀錄，第 3 版的規則 ／ 第 4 版的規則",
        "",
        *_rule_effect_text(diagnostics),
        "## 六、逼近的手（物理層）",
        "",
        *_loom_text(diagnostics, presets),
        "## 七、真實耳朵延遲版（腦層）",
        "",
        *RT3._ear_text(rows, diagnostics, presets),
        "## 八、每一個沒過或判不出的項目，白話是什麼意思",
        "",
        "下面的句子都是從上面的數字和診斷數字直接寫出來的；診斷數字沒有門檻，不影響任何判定。",
        "",
    ]
    for metric, fn in (
        ("M1", lambda: _m1_text(rows, diagnostics, presets)),
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
        *_support_text(physics_supported),
        "## 十、規格沒寫、事先定的操作選擇（要審查的）",
        "",
        "全文在 `MACHINE-VERIFY-v4-CHOICES.md`（在任何 v4 結果出現以前提交）。最會影響結果的幾條：",
        "",
        "1. **盯住**：差距＝軀幹到物體真相位置的方向和頭相機朝向的夾角；「刺激當下的差距」＝刺激那個 tick 自己的差距；一段＝窗（3 秒）裡連續差距 ≤ 15° 的最長連續區間，"
        "長度＝第一個到最後一個 tick 的時間，要 ≥ 0.5 秒；「開始時比刺激當下小」＝這一段第一個 tick 的差距嚴格小於刺激當下的差距。"
        "所以刺激那一刻就已經在看（≤ 15°）又一直待著的回合不算注意到（規格的「不是本來就在看」）。未知只有真相缺了才有。",
        "2. **擺放**：刺激那一 tick 讀世界自己的頭相機，用 `in_frame` 掃 −90° 到 +90° 找畫面裡的最長連續水平區間，取中間 60%，左右隨機"
        "（每個個性×種子×刺激一個固定的亂數）；被世界拒絕就退到鏡像、中心、較近的距離。",
        "3. **逼近的手**：起點＝畫面正中央那個像素的射線上離鏡頭 0.75 m，終點＝嘴尖前 0.09 m 高 +2 cm，每個 tick 依當下的頭重算；時間同第 3 版。",
        "4. 其餘（M1 的事件類、分格、M2、M3、M4、M5、M6 z 分數、M7 推一下、延遲耳朵）都是第 3 版的選擇。",
        "",
        "## 十一、沒量到或沒判出來的（照規格寫 `未測` 或 `incomplete`）",
        "",
        *RT3._unmeasured(rows),
        "",
        "## 十二、第 1、2、3 版重判（舊結果還能重現）",
        "",
        reproduce_md.replace("# 第 1、2、3 版重判：和原來的 rows 逐位元組比", "").strip()
        or "（這次沒有跑重判。）",
        "",
        "## 十三、錄製身分檢查（同一份程式兩次錄製是不是同一份紀錄）",
        "",
        identity_md.replace("# 錄製身分檢查（不是判定）", "").strip() or "（沒有比。）",
        "",
        "## 十四、時間",
        "",
    ]
    if wall:
        names = {
            "l1": "L1 對照回合",
            "l1ear": "L1 延遲耳朵版",
            "lives": "L1 整段生活",
            "l2": "L2 物理（2 個行程）",
            "l2_wait": "L2 等另一件工作的物理做完",
            "reproduce": "第 1、2、3 版重判",
            "judge": "判定",
            "sum_of_steps": "各步驟加總",
            "total": "從開始到現在（牆上時間）",
        }
        out.append("、".join(f"{names.get(k, k)} {v:.0f} 秒" for k, v in wall.items()) + "。")
    out += [
        "",
        "## 十五、檔案與重跑",
        "",
        "一條指令（見同目錄 `MACHINE-VERIFY-v4-RERUN.md`）：`python tools/machine_verify/run_v4.py --code-root <被驗的程式> --out <輸出目錄>`。"
        "這個目錄裡：`rows-L1.json`、`rows-L1ear.json`、`rows-L2.json`（每一列的值、分母、成功、失敗、未知、排除、門檻、判定、失敗的回合）、"
        "`tables.md`、`m1-cells.md`、`failing-trials.md`、`supplement.md`、`key-numbers.md`、`compare-v3-v4.md`、`trace-identity.md`、"
        "`reproduce-v1-v2-v3.md`、`diagnostics.json`、`L1/`、`L1_ear/`、`L1_lives/`、`L2/`（原始紀錄和 manifest）、`manifest.json`、`wall.json`、`logs/`。",
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


__all__ = ["METRIC_ZH", "build_report"]
