"""Metric rows and verdicts of MACHINE-VERIFY-v2.md from per-pair outcomes and recorded lives.

Beside ``aggregate.py`` (v1, untouched): M2, M3, M7 and the walk-over numbers are v1's own functions
run on the v2 outcomes; M4 is v1's with the remembered places now judged; M1, M5 and M6 are v2's.

Verdict words are v1's (``PASS``, ``FAIL``, ``incomplete``, ``未測``, ``報告``). A trial whose rule is
undefined (an unknown) does not make a row incomplete by itself: the gate is tried with every unknown
going against the duck (still passes: ``PASS``) and with every unknown going for it (still fails:
``FAIL``); anything in between is ``incomplete``. A planned trial that is missing or invalid still
makes the row ``incomplete``.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import aggregate as A
from duckmind.verify import lives as LV
from duckmind.verify import spec as v1spec
from duckmind.verify import spec_v2 as S
from duckmind.verify.aggregate import FAIL, INCOMPLETE, PASS, REPORT, UNTESTED, Outcomes, Plan, Row
from duckmind.verify.outcomes_v2 import PairOutcomeV2
from duckmind.verify.temperaments import signature_acts
from duckmind.verify.trace import Trace

EPS = 1e-9


# ── a gate with unknowns ───────────────────────────────────────────────────────────


def gate_min(ok: int, unknown: int, total: int, threshold: float) -> str:
    """``ok / total >= threshold``, where ``unknown`` of the ``total`` could still go either way."""
    if total == 0:
        return INCOMPLETE
    if ok >= threshold * total - EPS:
        return PASS
    if ok + unknown < threshold * total - EPS:
        return FAIL
    return INCOMPLETE


def gate_max(events: int, unknown: int, total: int, threshold: float) -> str:
    """``events / total <= threshold``, where ``unknown`` of the ``total`` could still go either way."""
    if total == 0:
        return INCOMPLETE
    if events + unknown <= threshold * total + EPS:
        return PASS
    if events > threshold * total + EPS:
        return FAIL
    return INCOMPLETE


def gate_diff_min(diff_sum: int, unknown: int, total: int, threshold: float) -> str:
    """``diff_sum / total >= threshold`` where an unknown pair adds anything from −1 to +1."""
    if total == 0:
        return INCOMPLETE
    if diff_sum - unknown >= threshold * total - EPS:
        return PASS
    if diff_sum + unknown < threshold * total - EPS:
        return FAIL
    return INCOMPLETE


def _tid(o: PairOutcomeV2) -> str:
    return A._tid(o)


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _good(data: Outcomes, **kw: Any) -> tuple[list[PairOutcomeV2], list[str]]:
    good, bad = data.split(data.select(**kw))
    return [o for o in good if isinstance(o, PairOutcomeV2)], bad


# ── M1 ──────────────────────────────────────────────────────────────────────────────


def m1(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = _good(data, preset=preset)
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
        both_ids = {id(o) for o in both}
        unknown_kinds = Counter(o.stimulus for o in eligible if id(o) not in both_ids)
        extra_unknown = {"unknown_by_stimulus": dict(sorted(unknown_kinds.items()))}
        v_a = INCOMPLETE if bad else gate_min(a_ok, u_a, total, S.M1_STIM_MIN)
        v_b = INCOMPLETE if bad else gate_diff_min(d_sum, u_b, total, S.M1_DIFF_MIN)
        rows.append(
            Row(
                layer,
                "M1",
                "有刺激組注意率",
                preset,
                _rate(a_ok, len(a_known)),
                len(a_known),
                a_ok,
                len(a_known) - a_ok,
                u_a + len(bad),
                len(unseen),
                f">= {S.M1_STIM_MIN:.0%}",
                v_a,
                fails=[_tid(o) + "|stim無注意" for o in a_known if not o.attend_stim],
                note="有方向：頭轉 ≥ 10° 且差距縮小一半以上；沒有方向：3 秒內有反射。未知＝那條規則在該回合沒有定義",
                extra={"missing_or_invalid": bad, **extra_unknown},
            )
        )
        rows.append(
            Row(
                layer,
                "M1",
                "有刺激組 − 對照組（配對差距）",
                preset,
                _rate(d_sum, len(both)),
                len(both),
                len(won),
                len(both) - len(won),
                u_b + len(bad),
                len(unseen),
                f">= {S.M1_DIFF_MIN}",
                v_b,
                fails=[
                    f"{_tid(o)}|stim{'注意' if o.attend_stim else '無注意'}"
                    f"/sham{'注意' if o.attend_sham else '無注意'}"
                    for o in both
                    if not (o.attend_stim and not o.attend_sham)
                ],
                note="每一組（有刺激組注意 − 對照組注意）的平均；成功＝有刺激組注意、對照組沒有",
                extra={
                    "p_stim": _rate(sum(1 for o in both if o.attend_stim), len(both)),
                    "p_sham": _rate(len(sham_hit), len(both)),
                    "diff_sum": d_sum,
                    "missing_or_invalid": bad,
                    **extra_unknown,
                },
            )
        )
        rows.append(
            Row(
                layer,
                "M1",
                "（報告）對照組同規則下的注意率",
                preset,
                _rate(len(sham_hit), len(both)),
                len(both),
                len(both) - len(sham_hit),
                len(sham_hit),
                u_b + len(bad),
                len(unseen),
                "（不當門檻；配對差距那列才是門檻）",
                REPORT,
                fails=[_tid(o) + "|sham注意" for o in sham_hit],
            )
        )
        rc = eligible
        rows.append(
            Row(
                layer,
                "M1",
                "（報告）第 1 版的判法：3 秒內有反射或換行為（有刺激組；對照組在 extra）",
                preset,
                _rate(sum(o.reflex_or_change_stim for o in rc), len(rc)),
                len(rc),
                sum(o.reflex_or_change_stim for o in rc),
                len(rc) - sum(o.reflex_or_change_stim for o in rc),
                0,
                len(unseen),
                "（不當門檻）",
                REPORT,
                extra={"sham_rate": _rate(sum(o.reflex_or_change_sham for o in rc), len(rc))},
            )
        )
    return rows


def m1_by_stimulus(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    for stimulus in data.plan.stimuli:
        good, _bad = _good(data, stimuli=[stimulus])
        eligible = [o for o in good if o.perceivable is True]
        a_known = [o for o in eligible if o.attend_stim is not None]
        both = [o for o in a_known if o.attend_sham is not None]
        k = sum(1 for o in a_known if o.attend_stim)
        h = sum(1 for o in both if o.attend_sham)
        s_b = sum(1 for o in both if o.attend_stim)
        rule = "頭轉向" if stimulus in S.DIRECTIONAL else "反射"
        rows.append(
            Row(
                layer,
                "M1",
                f"（報告）{stimulus}（{rule}）：有刺激組 / 對照組 / 差距（合併個性）",
                stimulus,
                None if not both else (s_b - h) / len(both),
                len(a_known),
                k,
                len(a_known) - k,
                len(eligible) - len(both),
                len(good) - len(eligible),
                "（不當門檻）",
                REPORT,
                extra={
                    "p_stim": _rate(k, len(a_known)),
                    "p_sham": _rate(h, len(both)),
                    "n_pairs": len(both),
                },
            )
        )
    return rows


# ── M2: v1's own, plus the reference with reflexes ──────────────────────────────────


def m2_with_reflex(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    for stimulus in data.plan.stimuli:
        good, bad = _good(data, stimuli=[stimulus])
        n = len(good)
        k = sum(o.reflex_or_change_stim for o in good)
        h = sum(o.reflex_or_change_sham for o in good)
        rows.append(
            Row(
                data.plan.layer,
                "M2",
                "（參考，不當門檻）含反射：有刺激組 − 對照組（3 秒內換行為或有反射）",
                stimulus,
                None if not n else (k - h) / n,
                n,
                k,
                n - k,
                len(bad),
                0,
                "（不當門檻）",
                REPORT,
                extra={"p_stim": _rate(k, n), "p_sham": _rate(h, n)},
            )
        )
    return rows


# ── M4: v1's rows, the arrival note rewritten ───────────────────────────────────────

ARRIVAL_NOTE = (
    "目標是東西：用模擬器自己的標籤配對；目標是記得的地方：用起始姿態把腦記得的地方換成世界座標，"
    "真實距離 <= 可碰距離（0.096 m）+ 0.1 m。未知＝說「到了」那一刻腦的目標既不是東西也不是地方，"
    "或沒有錄到地方座標；未知不算通過，一次錯的「到了」就 FAIL"
)


def m4(data: Outcomes) -> list[Row]:
    rows = A.m4(data)
    return [
        dataclasses.replace(r, note=ARRIVAL_NOTE) if r.part.startswith("腦說「到了」") else r
        for r in rows
    ]


# ── M5 ──────────────────────────────────────────────────────────────────────────────


def m5(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = _good(data, preset=preset)
        episodes = [(o, o.episode) for o in good if o.episode is not None]
        n_ep = len(episodes)
        cats = Counter(ep.category for _o, ep in episodes)
        natural, timeout, censored = cats["natural"], cats["timeout"], cats["censored"]
        known = n_ep - censored
        no_episode = len(good) - n_ep
        common = {"episodes": dict(cats), "missing_or_invalid": bad}
        v_nat = INCOMPLETE if bad else gate_min(natural, censored, n_ep, v1spec.M5_NATURAL_MIN)
        v_to = INCOMPLETE if bad else gate_max(timeout, censored, n_ep, v1spec.M5_TIMEOUT_MAX)
        rows.append(
            Row(
                layer,
                "M5",
                "刺激引起的那段：自然結束比例",
                preset,
                _rate(natural, known),
                known,
                natural,
                known - natural,
                censored + len(bad),
                no_episode,
                f">= {v1spec.M5_NATURAL_MIN:.0%}",
                v_nat,
                fails=[
                    _tid(o) + f"|{ep.kind}:{ep.category}"
                    for o, ep in episodes
                    if ep.category in ("displaced", "timeout")
                ],
                note="分母＝有刺激組窗內有「引起的那段」的回合（定住身體的反射一串，或第一個新開始的 bout）；"
                "排除欄＝窗內沒有這一段的回合；未知＝到紀錄結束還沒結束",
                extra=common,
            )
        )
        rows.append(
            Row(
                layer,
                "M5",
                "刺激引起的那段：逾時結束比例",
                preset,
                _rate(timeout, known),
                known,
                known - timeout,
                timeout,
                censored + len(bad),
                no_episode,
                f"<= {v1spec.M5_TIMEOUT_MAX:.0%}",
                v_to,
                fails=[_tid(o) for o, ep in episodes if ep.category == "timeout"],
                note="腦沒有切斷行為的計時器，紀錄也沒有標逾時的欄位，所以逾時天生是 0",
                extra=common,
            )
        )
        m = len(good)
        reopen = sum(o.reopen_stim for o in good)
        rows.append(
            Row(
                layer,
                "M5",
                "10 秒內同一行為重開 3 次以上的回合比例",
                preset,
                _rate(reopen, m),
                m,
                m - reopen,
                reopen,
                len(bad),
                0,
                f"<= {v1spec.M5_REOPEN_MAX:.0%}",
                INCOMPLETE if bad else gate_max(reopen, 0, m, v1spec.M5_REOPEN_MAX),
                fails=[_tid(o) for o in good if o.reopen_stim],
            )
        )
        paired = [(o, ep) for o, ep in episodes if o.caused]
        p_known = [(o, ep) for o, ep in paired if ep.category != "censored"]
        p_nat = sum(1 for _o, ep in p_known if ep.category == "natural")
        rows.append(
            Row(
                layer,
                "M5",
                "（報告）第 1 版的分母：有刺激組換了行為而對照組沒換的回合，自然結束比例",
                preset,
                _rate(p_nat, len(p_known)),
                len(p_known),
                p_nat,
                len(p_known) - p_nat,
                len(paired) - len(p_known),
                len(good) - len(paired),
                "（不當門檻）",
                REPORT,
            )
        )
    return rows


# ── M6 ──────────────────────────────────────────────────────────────────────────────

M6_GUESS_PART = "猜個性：拿掉一顆種子，歸給最像的個性，猜對比例"


def _guess_rows(
    layer: str,
    presets: Sequence[str],
    guesses: Mapping[LV.Key, str | None],
    *,
    gated: bool,
    n_expected: int,
    why_missing: str,
) -> list[Row]:
    rows: list[Row] = []
    right, total = LV.accuracy(guesses)
    conf = LV.confusion(guesses, presets)
    missing = n_expected - total
    unit = "" if gated else "（報告）"
    v_all = REPORT
    if gated:
        v_all = INCOMPLETE if missing else (PASS if right >= S.M6_ACC_MIN * total - EPS else FAIL)
    rows.append(
        Row(
            layer,
            "M6",
            unit + M6_GUESS_PART + "（全部）",
            "ALL",
            _rate(right, total),
            total,
            right,
            total - right,
            missing,
            0,
            f">= {S.M6_ACC_MIN:.0%}" if gated else "（不當門檻）",
            v_all,
            fails=[f"{p}|seed{s} 猜成 {g}" for (p, s), g in sorted(guesses.items()) if g != p],
            note=why_missing if missing else "",
            extra={"confusion": conf},
        )
    )
    for preset in presets:
        mine = {k: g for k, g in guesses.items() if k[0] == preset}
        k_ok = sum(1 for g in mine.values() if g == preset)
        v = REPORT
        if gated:
            v = (
                INCOMPLETE
                if missing
                else (PASS if k_ok >= S.M6_EACH_MIN * len(mine) - EPS else FAIL)
            )
        rows.append(
            Row(
                layer,
                "M6",
                unit + "這個個性自己被猜對的比例",
                preset,
                _rate(k_ok, len(mine)),
                len(mine),
                k_ok,
                len(mine) - k_ok,
                missing,
                0,
                f">= {S.M6_EACH_MIN:.0%}" if gated else "（不當門檻）",
                v,
                fails=[f"seed{s} 猜成 {g}" for (p, s), g in sorted(mine.items()) if g != preset],
            )
        )
    return rows


def m6_lives(
    layer: str,
    presets: Sequence[str],
    seeds: Sequence[int],
    lives: Mapping[LV.Key, Sequence[LV.Life]],
    problems: Mapping[LV.Key, list[str]],
) -> list[Row]:
    """L1's M6: the guess of the temperament over a whole life per seed, and the signature acts."""
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
    guesses = LV.guess(profiles, presets)
    rows = _guess_rows(
        layer, presets, guesses, gated=True, n_expected=len(expected), why_missing=why
    )
    # sensitivity (not a verdict): other ways to measure "most alike"
    n_names = len(names)
    sens = {
        "標準化後": LV.guess(profiles, presets, standardised=True),
        "只用行為時間比例": LV.guess(profiles, presets, columns=range(n_names)),
        "只用對刺激的反應": LV.guess(
            profiles, presets, columns=range(n_names, n_names + 3 * len(S.LIFE_STIM_ORDER))
        ),
    }
    for label, g in sens.items():
        r, t = LV.accuracy(g)
        rows.append(
            Row(
                layer,
                "M6",
                f"（補充，不是判定）猜個性猜對比例：{label}",
                "ALL",
                _rate(r, t),
                t,
                r,
                t - r,
                0,
                0,
                "（不當門檻）",
                REPORT,
                extra={"confusion": LV.confusion(g, presets)},
            )
        )
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


def l2_profiles(
    traces: Iterable[Trace], presets: Sequence[str]
) -> tuple[dict[LV.Key, list[float]], dict[LV.Key, set[str]]]:
    """L2's profile of each (temperament, seed): the time share of every behaviour over all its
    trials (both arms), then the three reaction numbers of each stimulus it got (stim arm)."""
    by_key: dict[LV.Key, list[Trace]] = {}
    for tr in traces:
        by_key.setdefault((tr["preset"], int(tr["seed"])), []).append(tr)
    names = sorted({b for trs in by_key.values() for tr in trs for b in tr["cols"]["behaviour"]})
    stimuli = sorted({tr["stimulus"] for trs in by_key.values() for tr in trs})
    profiles: dict[LV.Key, list[float]] = {}
    seen: dict[LV.Key, set[str]] = {}
    for key, trs in by_key.items():
        counts: Counter[str] = Counter()
        total = 0
        for tr in trs:
            counts.update(tr["cols"]["behaviour"])
            total += len(tr["cols"]["behaviour"])
        vec = [counts.get(b, 0) / total if total else 0.0 for b in names]
        for stimulus in stimuli:
            arms = [t for t in trs if t["stimulus"] == stimulus and t["arm"] == "stim"]
            if not arms:
                vec += [0.0, 0.0, 1.0]
                continue
            changed, startled, latency = LV.reaction(
                arms[0]["cols"], arms[0]["onset_idx"], arms[0]["dt_s"]
            )
            vec += [float(changed), float(startled), latency / S.LATENCY_CAP_S]
        profiles[key] = vec
        seen[key] = set(counts)
    return profiles, seen


def m6_l2_report(
    layer: str, presets: Sequence[str], seeds: Sequence[int], traces: Iterable[Trace]
) -> list[Row]:
    """L2's M6: numbers only, no threshold (the spec says four seeds cannot measure a temperament)."""
    profiles, seen = l2_profiles(traces, presets)
    expected = len(presets) * len(seeds)
    rows = _guess_rows(
        layer,
        presets,
        LV.guess(profiles, presets) if profiles else {},
        gated=False,
        n_expected=expected,
        why_missing="有些種子沒有紀錄",
    )
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


# ── everything ──────────────────────────────────────────────────────────────────────


def all_rows(
    data: Outcomes,
    *,
    l2_traces: Iterable[Trace] | None = None,
    lives: Mapping[LV.Key, Sequence[LV.Life]] | None = None,
    life_problems: Mapping[LV.Key, list[str]] | None = None,
) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    rows += m1(data)
    rows += m1_by_stimulus(data)
    rows += A.m2(data)
    rows += m2_with_reflex(data)
    rows += A.m3(data)
    rows += m4(data)
    rows += m5(data)
    if layer == "L1":
        if lives is None:
            rows += [A._untested(layer, "M6", "同一隻鴨（整段生活猜個性）", "ALL", "沒有錄生活")]
        else:
            rows += m6_lives(layer, data.plan.presets, data.plan.seeds, lives, life_problems or {})
    else:
        rows += m6_l2_report(layer, data.plan.presets, data.plan.seeds, l2_traces or [])
    rows += A.m7(data)
    rows += A.approach_rows(data)
    return rows


def cell_verdicts(rows: Sequence[Row], presets: Sequence[str]) -> dict[tuple[str, str, str], str]:
    """(layer, metric, temperament) → the combined verdict of that temperament's rows and the rows
    of everybody (scope ``ALL``). A failure dominates; an untested part makes the whole at best
    ``incomplete``."""
    out: dict[tuple[str, str, str], str] = {}
    layers = sorted({r.layer for r in rows})
    for layer in layers:
        for metric in A.GATED:
            if metric == "M2":
                continue
            for preset in presets:
                vs = [
                    r.verdict
                    for r in rows
                    if r.layer == layer and r.metric == metric and r.scope in (preset, "ALL")
                ]
                out[(layer, metric, preset)] = A.combine(vs) if vs else "—"
    return out


def m2_verdicts(rows: Sequence[Row]) -> dict[tuple[str, str], str]:
    return {(r.layer, r.scope): r.verdict for r in rows if r.metric == "M2" and r.verdict != REPORT}


__all__ = [
    "FAIL",
    "INCOMPLETE",
    "PASS",
    "REPORT",
    "UNTESTED",
    "Plan",
    "all_rows",
    "cell_verdicts",
    "gate_diff_min",
    "gate_max",
    "gate_min",
    "m1",
    "m2_verdicts",
    "m4",
    "m5",
    "m6_l2_report",
    "m6_lives",
]
