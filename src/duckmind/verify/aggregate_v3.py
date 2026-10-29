"""Metric rows and verdicts of MACHINE-VERIFY-v3.md from per-pair outcomes, recorded lives and the push
trials.

Beside ``aggregate.py`` (v1) and ``aggregate_v2.py`` (v2), both untouched. M2, M3, M4 and M5 are v1's and
v2's own functions run on the v3 outcomes; what v3 writes is M1 per cell (``m1_cells``), M6 with z scores
(``m6_lives`` and ``m6_l2_report``) and M7 with the push trial (``m7``).

Verdict words are v1's (``PASS``, ``FAIL``, ``incomplete``, ``未測``, ``報告``); unknowns are handled as in
v2 (``gate_min`` and friends).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from duckmind.verify import aggregate as A
from duckmind.verify import aggregate_v2 as A2
from duckmind.verify import lives as LV
from duckmind.verify import lives_v3 as LZ
from duckmind.verify import push_v3 as P
from duckmind.verify import spec_v3 as S
from duckmind.verify.aggregate import FAIL, INCOMPLETE, PASS, REPORT, Outcomes, Plan, Row
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.temperaments import signature_acts
from duckmind.verify.trace import Trace

EPS = A2.EPS
SCOPE_ALL = "ALL"


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


# ── M1: one cell = one stimulus (and, in the brain layer, one temperament) ───────────


def m1_cell_rows(
    layer: str,
    scope: str,
    stimulus: str,
    good: Sequence[PairOutcomeV2],
    bad: Sequence[str],
) -> list[Row]:
    """The three rows of one M1 cell: the stimulated group's notice rate against 70%, the paired
    difference against 0.5, and (report only) the same-rule rate of the control group. ``good`` are the
    cell's valid pairs, ``bad`` the ids of its planned pairs that are missing or invalid."""
    eligible = [o for o in good if o.perceivable is True]
    unseen = [o for o in good if o.perceivable is False]
    pending = [o for o in good if o.perceivable is None]
    total = len(eligible) + len(pending)
    a_known = [o for o in eligible if o.attend_stim is not None]
    a_ok = sum(1 for o in a_known if o.attend_stim)
    u_a = total - len(a_known)
    both = [o for o in eligible if o.attend_stim is not None and o.attend_sham is not None]
    d_sum = sum(int(bool(o.attend_stim)) - int(bool(o.attend_sham)) for o in both)
    u_b = total - len(both)
    won = [o for o in both if o.attend_stim and not o.attend_sham]
    sham_hit = [o for o in both if o.attend_sham]
    v_a = INCOMPLETE if bad else A2.gate_min(a_ok, u_a, total, S.M1_STIM_MIN)
    v_b = INCOMPLETE if bad else A2.gate_diff_min(d_sum, u_b, total, S.M1_DIFF_MIN)
    rule = "頭轉向真實方向" if stimulus in S.DIRECTIONAL else "3 秒內有反射"
    cell = {
        "stimulus": stimulus,
        "rule": rule,
        "pairs": len(good) + len(bad),
        "n_pairs": len(both),
        "p_stim": _rate(sum(1 for o in both if o.attend_stim), len(both)),
        "p_sham": _rate(len(sham_hit), len(both)),
        "diff_sum": d_sum,
        "unknown": u_b,
        "excluded": len(unseen),
        "missing_or_invalid": list(bad),
    }
    return [
        Row(
            layer,
            "M1",
            f"{stimulus}：有刺激組注意率",
            scope,
            _rate(a_ok, len(a_known)),
            len(a_known),
            a_ok,
            len(a_known) - a_ok,
            u_a + len(bad),
            len(unseen),
            f">= {S.M1_STIM_MIN:.0%}",
            v_a,
            fails=[A2._tid(o) + "|stim無注意" for o in a_known if not o.attend_stim],
            note=f"規則：{rule}。未知＝那條規則在該回合沒有定義（或看不看得到還不知道）",
            extra=cell,
        ),
        Row(
            layer,
            "M1",
            f"{stimulus}：有刺激組 − 對照組（配對差距）",
            scope,
            _rate(d_sum, len(both)),
            len(both),
            len(won),
            len(both) - len(won),
            u_b + len(bad),
            len(unseen),
            f">= {S.M1_DIFF_MIN}",
            v_b,
            fails=[
                f"{A2._tid(o)}|stim{'注意' if o.attend_stim else '無注意'}"
                f"/sham{'注意' if o.attend_sham else '無注意'}"
                for o in both
                if not (o.attend_stim and not o.attend_sham)
            ],
            note="每一組（有刺激組注意 − 對照組注意）的平均；成功＝有刺激組注意、對照組沒有",
            extra=cell,
        ),
        Row(
            layer,
            "M1",
            f"（報告）{stimulus}：對照組同規則下的注意率",
            scope,
            _rate(len(sham_hit), len(both)),
            len(both),
            len(both) - len(sham_hit),
            len(sham_hit),
            u_b + len(bad),
            len(unseen),
            "（不當門檻；配對差距那列才是門檻）",
            REPORT,
            fails=[A2._tid(o) + "|sham注意" for o in sham_hit],
            extra=cell,
        ),
    ]


def m1_cells(data: Outcomes) -> list[Row]:
    """M1 per cell. Brain layer: every temperament × every stimulus (16 pairs a cell). Physics layer:
    every stimulus with the temperaments together (32 pairs a cell; scope ``ALL``)."""
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


# ── M6: z scores ──────────────────────────────────────────────────────────────────────

Z_NOTE = (
    "每一維先用「訓練種子」（被拿掉的那顆以外的種子）的平均和標準差換成 z 分數，標準差為 0 的維度拿掉；"
    "被拿掉的那顆不參與平均和標準差"
)


def _acc_row(layer: str, part: str, g: Mapping[LV.Key, str | None], presets: Sequence[str]) -> Row:
    r, t = LZ.accuracy(g)
    return Row(
        layer,
        "M6",
        part,
        SCOPE_ALL,
        _rate(r, t),
        t,
        r,
        t - r,
        0,
        0,
        "（不當門檻）",
        REPORT,
        extra={"confusion": LZ.confusion(g, presets)},
    )


def _m6_report_rows(
    layer: str,
    presets: Sequence[str],
    profiles: Mapping[LV.Key, Sequence[float]],
    n_time: int,
    n_react: int,
) -> list[Row]:
    """The three guesses that are only reported: the time shares alone and the reactions alone (both as
    z scores), and the whole profile in raw units (v2's way, to join the two versions)."""
    rows = []
    for label, g in (
        ("只用行為時間比例（z 分數）", LZ.guess_z(profiles, presets, columns=range(n_time))),
        (
            "只用對刺激的反應（z 分數）",
            LZ.guess_z(profiles, presets, columns=range(n_time, n_time + n_react)),
        ),
        ("整份檔案不標準化（第 2 版的做法）", LV.guess(profiles, presets)),
    ):
        rows.append(_acc_row(layer, f"（報告，不是判定）猜個性猜對比例：{label}", g, presets))
    return rows


def m6_lives(
    layer: str,
    presets: Sequence[str],
    seeds: Sequence[int],
    lives: Mapping[LV.Key, Sequence[LV.Life]],
    problems: Mapping[LV.Key, list[str]],
) -> list[Row]:
    """L1's M6: the guess of the temperament over a whole life per seed (z scores of the training
    seeds), the signature acts, and the report-only profiles."""
    expected = [(p, s) for p in presets for s in seeds]
    usable = {
        k: segs
        for k, segs in lives.items()
        if k in expected and not problems.get(k) and len(segs) == len(S.RUNS)
    }
    bad = [k for k in expected if k not in usable]
    why = "這些種子的生活缺或出錯：" + "、".join(f"{p}|seed{s}" for p, s in bad[:8])
    names = LV.behaviour_names(seg for segs in usable.values() for seg in segs)
    profiles = {k: LV.profile(segs, names) for k, segs in usable.items()}
    guesses = LZ.guess_z(profiles, presets)
    rows = A2._guess_rows(
        layer, presets, guesses, gated=True, n_expected=len(expected), why_missing=why
    )
    rows[0] = dataclasses.replace(
        rows[0],
        part=rows[0].part.replace("猜個性：", "猜個性（z 分數）：", 1),
        note=(rows[0].note + "；" if rows[0].note else "") + Z_NOTE,
    )
    n_react = 3 * len(S.LIFE_STIM_ORDER)
    rows += _m6_report_rows(layer, presets, profiles, len(names), n_react)
    rows += _signature_rows(layer, presets, seeds, usable)
    return rows


def _signature_rows(
    layer: str,
    presets: Sequence[str],
    seeds: Sequence[int],
    usable: Mapping[LV.Key, Sequence[LV.Life]],
) -> list[Row]:
    rows: list[Row] = []
    for preset in presets:
        acts = signature_acts(preset)
        both_n = any_n = 0
        missing_by_seed: list[str] = []
        incomplete = 0
        for seed in seeds:
            segs = usable.get((preset, seed))
            if segs is None:
                incomplete += 1
                continue
            seen, absent = LV.signature_seen(segs, acts)
            both_n += 0 if absent else 1
            any_n += 1 if seen else 0
            if absent:
                missing_by_seed.append(f"seed{seed} 缺 {'、'.join(absent)}")
        n = len(seeds)
        if incomplete:
            verdict = INCOMPLETE
        else:
            verdict = PASS if both_n >= S.M6_SIG_MIN * n - EPS else FAIL
        rows.append(
            Row(
                layer,
                "M6",
                "招牌動作（" + "、".join(acts) + "）在整段生活裡都出現的種子比例",
                preset,
                _rate(both_n, n),
                n,
                both_n,
                n - both_n - incomplete,
                incomplete,
                0,
                f">= {S.M6_SIG_MIN:.0%}",
                verdict,
                fails=missing_by_seed,
                note="每顆種子的生活＝白天 2 小時＋夜晚 2 小時；兩個招牌動作都要出現過",
                extra={"acts": list(acts), "any_one": any_n},
            )
        )
        rows.append(
            Row(
                layer,
                "M6",
                "（補充，不是判定）至少出現一個招牌動作的種子比例",
                preset,
                _rate(any_n, n),
                n,
                any_n,
                n - any_n - incomplete,
                incomplete,
                0,
                "（不當門檻）",
                REPORT,
            )
        )
    return rows


def m6_l2_report(
    layer: str, presets: Sequence[str], seeds: Sequence[int], traces: Iterable[Trace]
) -> list[Row]:
    """L2's M6: numbers only, no threshold (the spec leaves the physics layer's M6 as v2 had it)."""
    traces = [t for t in traces if t["stimulus"] != S.PUSH]
    profiles, seen = A2.l2_profiles(traces, presets)
    expected = len(presets) * len(seeds)
    names = sorted({b for tr in traces for b in tr["cols"]["behaviour"]})
    stimuli = sorted({tr["stimulus"] for tr in traces})
    guesses = LZ.guess_z(profiles, presets) if profiles else {}
    rows = A2._guess_rows(
        layer,
        presets,
        guesses,
        gated=False,
        n_expected=expected,
        why_missing="有些種子沒有紀錄",
    )
    rows[0] = dataclasses.replace(
        rows[0],
        part=rows[0].part.replace("猜個性：", "猜個性（z 分數）：", 1),
        note=(rows[0].note + "；" if rows[0].note else "") + Z_NOTE,
    )
    if profiles:
        rows += _m6_report_rows(layer, presets, profiles, len(names), 3 * len(stimuli))
    for preset in presets:
        acts = signature_acts(preset)
        got = [
            s for s in seeds if (preset, s) in seen and all(a in seen[(preset, s)] for a in acts)
        ]
        rows.append(
            Row(
                layer,
                "M6",
                "（報告）招牌動作（" + "、".join(acts) + "）都出現的種子比例",
                preset,
                _rate(len(got), len(seeds)),
                len(seeds),
                len(got),
                len(seeds) - len(got),
                0,
                0,
                "（不當門檻：物理層只報數字）",
                REPORT,
                fails=[f"seed{s}" for s in seeds if s not in got],
            )
        )
    return rows


# ── M7: falls in the nine stimuli (v1) and the push trial (v3) ─────────────────────────

PUSH_PART = "推一下：安全反射被擋次數 = 0（推倒的回合 ≥ 4 才量得到）"


def m7_push_rows(
    layer: str,
    presets: Sequence[str],
    pushes: Mapping[str, P.PushOutcome],
    planned: Mapping[str, str],
) -> list[Row]:
    """The push trial's rows per temperament. ``pushes``: pair id -> outcome; ``planned``: pair id ->
    temperament of every planned push (a missing or invalid push leaves its row at best
    ``incomplete``)."""
    rows: list[Row] = []
    for preset in presets:
        ids = [pid for pid, p in planned.items() if p == preset]
        got = [pushes[i] for i in ids if i in pushes and pushes[i].valid]
        bad = [i for i in ids if i not in pushes or not pushes[i].valid]
        by_status = Counter(o.status for o in got)
        tipped = [o for o in got if o.status == P.TIPPED]
        blocked = [o for o in tipped if o.blocked]
        n = len(tipped)
        if n >= S.PUSH_MIN_TIPPED and len(blocked) > S.PUSH_BLOCKED_MAX:
            verdict = FAIL  # one more push can never undo a blocked reflex
        elif n < S.PUSH_MIN_TIPPED or bad:
            verdict = INCOMPLETE
        else:
            verdict = PASS
        if n < S.PUSH_MIN_TIPPED:
            note = (
                f"{len(ids)} 次推裡只有 {n} 次推倒（不到 {S.PUSH_MIN_TIPPED} 次），照規格判未完成"
                + (f"；其中已有 {len(blocked)} 次被擋" if blocked else "")
            )
        else:
            note = (
                "被擋＝寬限時間（0.5 秒）內沒有任何定住身體的反射，或寬限時間之後倒著還下走路指令"
            )
        rows.append(
            Row(
                layer,
                "M7",
                PUSH_PART,
                preset,
                float(len(blocked)),
                n,
                n - len(blocked),
                len(blocked),
                len(bad),
                len(got) - n,
                "被擋 = 0；推倒 >= 4 次",
                verdict,
                fails=[o.tid for o in blocked],
                note=note,
                extra={
                    "statuses": dict(by_status),
                    "pushes": len(ids),
                    "missing_or_invalid": bad,
                    "took_over": sum(1 for o in tipped if o.took_over),
                    "righting": sum(1 for o in tipped if o.righting),
                    "walked": sum(1 for o in tipped if o.walked),
                    "first_reflex_s": [
                        round(o.first_reflex_s, 3) for o in tipped if o.first_reflex_s is not None
                    ],
                    "first_reflex": dict(Counter(o.first_reflex for o in tipped)),
                    "down_s": [round(o.down_s, 2) for o in tipped if o.down_s is not None],
                    "tipped_seeds": [o.seed for o in tipped],
                },
            )
        )
        rows.append(
            Row(
                layer,
                "M7",
                "（報告）推一下的結果：推倒／沒倒／很快自己起來／兩組都倒／已經倒著",
                preset,
                _rate(n, len(got)),
                len(got),
                n,
                len(got) - n,
                len(bad),
                0,
                "（不當門檻）",
                REPORT,
                note="、".join(f"{k} {by_status.get(k, 0)}" for k in P.STATUSES),
                extra={"statuses": dict(by_status)},
            )
        )
    return rows


def m7(
    data: Outcomes,
    pushes: Mapping[str, P.PushOutcome] | None = None,
    planned: Mapping[str, str] | None = None,
) -> list[Row]:
    """L1: ``未測`` (v1). L2: v1's falls row over the nine stimuli, v1's 'blocked' row turned into a
    report, and the push trial's rows."""
    layer = data.plan.layer
    if layer != "L2":
        return A.m7(data)
    base = A.m7(data)
    rows: list[Row] = []
    for r in base:
        if r.part.startswith("安全反射被擋"):
            rows.append(
                dataclasses.replace(
                    r,
                    part="（報告）九種刺激的回合裡，跌倒後安全反射被擋的次數",
                    threshold="（不當門檻；門檻改由推一下的回合量）",
                    verdict=REPORT,
                )
            )
        else:
            rows.append(r)
    if pushes is None or planned is None:
        rows += [
            A._untested(layer, "M7", PUSH_PART, p, "沒有推一下的紀錄") for p in data.plan.presets
        ]
    else:
        rows += m7_push_rows(layer, data.plan.presets, pushes, planned)
    return rows


# ── everything ──────────────────────────────────────────────────────────────────────────


def all_rows(
    data: Outcomes,
    *,
    l2_traces: Iterable[Trace] | None = None,
    lives: Mapping[LV.Key, Sequence[LV.Life]] | None = None,
    life_problems: Mapping[LV.Key, list[str]] | None = None,
    pushes: Mapping[str, P.PushOutcome] | None = None,
    planned_pushes: Mapping[str, str] | None = None,
) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    rows += m1_cells(data)
    rows += A.m2(data)
    rows += A2.m2_with_reflex(data)
    rows += A.m3(data)
    rows += A2.m4(data)
    rows += A2.m5(data)
    if layer == "L1":
        if lives is None:
            rows += [A._untested(layer, "M6", "同一隻鴨（整段生活猜個性）", "ALL", "沒有錄生活")]
        else:
            rows += m6_lives(layer, data.plan.presets, data.plan.seeds, lives, life_problems or {})
    else:
        rows += m6_l2_report(layer, data.plan.presets, data.plan.seeds, l2_traces or [])
    rows += m7(data, pushes, planned_pushes)
    rows += A.approach_rows(data)
    return rows


cell_verdicts = A2.cell_verdicts
m2_verdicts = A2.m2_verdicts


__all__ = [
    "FAIL",
    "INCOMPLETE",
    "PASS",
    "REPORT",
    "Plan",
    "Row",
    "all_rows",
    "cell_verdicts",
    "m1_cell_rows",
    "m1_cells",
    "m6_l2_report",
    "m6_lives",
    "m7",
    "m7_push_rows",
]
