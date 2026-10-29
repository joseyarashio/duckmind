"""Metric rows and verdicts of MACHINE-VERIFY.md from per-pair outcomes.

Verdict words: ``PASS``, ``FAIL``, ``incomplete`` (a planned trial is missing or invalid, a
truth needed was unknown, or nobody produced the thing to measure: never a pass), ``未測`` (the
layer cannot make that stimulus), ``報告`` (a number the spec asks for but gives no threshold).
Every row carries its denominator, successes, failures, unknowns and the failing trials.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from duckmind.verify import spec
from duckmind.verify.outcomes import PairOutcome
from duckmind.verify.temperaments import BOLD_PRESETS, signature_acts

PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "incomplete"
UNTESTED = "未測"
REPORT = "報告"


@dataclass
class Row:
    layer: str
    metric: str  # M1 … M7
    part: str  # what is measured inside the metric
    scope: str  # a temperament, a stimulus, "ALL" or a pair of temperaments
    value: float | None
    n: int  # the denominator
    ok: int
    fail: int
    unknown: int
    na: int  # planned trials left out of the denominator for a stated reason
    threshold: str
    verdict: str
    fails: list[str] = field(default_factory=list)
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Plan:
    """What the run was supposed to contain: layer, temperaments, seeds, stimuli."""

    layer: str
    presets: tuple[str, ...]
    seeds: tuple[int, ...]
    stimuli: tuple[str, ...]

    def pair_ids(self) -> list[str]:
        return [
            pair_id(self.layer, p, s, st)
            for p in self.presets
            for s in self.seeds
            for st in self.stimuli
        ]


def pair_id(layer: str, preset: str, seed: int, stimulus: str) -> str:
    return f"{layer}|{preset}|{seed}|{stimulus}"


class Outcomes:
    """The outcomes of one layer, with the plan: a planned pair that is absent counts as
    missing, an invalid one as invalid."""

    def __init__(self, plan: Plan, outcomes: Iterable[PairOutcome]) -> None:
        self.plan = plan
        self.by_id = {o.pair: o for o in outcomes}

    def select(self, *, preset: str | None = None, stimuli: Iterable[str] | None = None):
        wanted = set(stimuli) if stimuli is not None else set(self.plan.stimuli)
        ids: list[str] = []
        for p in self.plan.presets:
            if preset is not None and p != preset:
                continue
            for s in self.plan.seeds:
                for st in self.plan.stimuli:
                    if st in wanted:
                        ids.append(pair_id(self.plan.layer, p, s, st))
        return ids

    def split(self, ids: Sequence[str]) -> tuple[list[PairOutcome], list[str]]:
        """(valid outcomes, ids missing or invalid)."""
        good: list[PairOutcome] = []
        bad: list[str] = []
        for i in ids:
            o = self.by_id.get(i)
            if o is None or not o.valid:
                bad.append(i)
            else:
                good.append(o)
        return good, bad


def _verdict_rate(value: float | None, n: int, unknown: int, passed: bool | None, bad: int) -> str:
    if bad or unknown or n == 0 or value is None or passed is None:
        return INCOMPLETE
    return PASS if passed else FAIL


def _tid(o: PairOutcome) -> str:
    return f"{o.preset}|seed{o.seed}|{o.stimulus}"


# ── M1 ──────────────────────────────────────────────────────────────────────────────


def m1(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = data.split(data.select(preset=preset))
        eligible = [o for o in good if o.perceivable is True]
        unseen = [o for o in good if o.perceivable is False]
        unknown = [o for o in good if o.perceivable is None or o.attend_stim is None]
        known = [o for o in eligible if o.attend_stim is not None and o.attend_sham is not None]
        n = len(known)
        stim_ok = sum(1 for o in known if o.attend_stim)
        sham_bad = sum(1 for o in known if o.attend_sham)
        stim_rate = stim_ok / n if n else None
        sham_rate = sham_bad / n if n else None
        fails = [_tid(o) + "|stim無注意" for o in known if not o.attend_stim]
        fails += [_tid(o) + "|sham無端注意" for o in known if o.attend_sham]
        verdict = _verdict_rate(
            stim_rate,
            n,
            len(unknown),
            None if stim_rate is None else stim_rate >= spec.M1_STIM_MIN,
            len(bad),
        )
        rows.append(
            Row(
                data.plan.layer,
                "M1",
                "有刺激組注意率",
                preset,
                stim_rate,
                n,
                stim_ok,
                n - stim_ok,
                len(unknown) + len(bad),
                len(unseen),
                f">= {spec.M1_STIM_MIN:.0%}",
                verdict,
                fails=[f for f in fails if f.endswith("stim無注意")],
                extra={"missing_or_invalid": bad},
            )
        )
        rows.append(
            Row(
                data.plan.layer,
                "M1",
                "對照組無端注意率",
                preset,
                sham_rate,
                n,
                n - sham_bad,
                sham_bad,
                len(unknown) + len(bad),
                len(unseen),
                f"<= {spec.M1_SHAM_MAX:.0%}",
                _verdict_rate(
                    sham_rate,
                    n,
                    len(unknown),
                    None if sham_rate is None else sham_rate <= spec.M1_SHAM_MAX,
                    len(bad),
                ),
                fails=[f for f in fails if f.endswith("sham無端注意")],
            )
        )
    return rows


# ── M2 ──────────────────────────────────────────────────────────────────────────────


def m2(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    for stimulus in data.plan.stimuli:
        good, bad = data.split(data.select(stimuli=[stimulus]))
        n = len(good)
        p_stim = sum(o.changed_stim for o in good) / n if n else None
        p_sham = sum(o.changed_sham for o in good) / n if n else None
        diff = None if p_stim is None or p_sham is None else p_stim - p_sham
        fails = [_tid(o) + "|stim沒換行為" for o in good if not o.changed_stim]
        fails += [_tid(o) + "|sham自己換行為" for o in good if o.changed_sham]
        rows.append(
            Row(
                data.plan.layer,
                "M2",
                "有刺激組 − 對照組（3 秒內換行為比例）",
                stimulus,
                diff,
                n,
                sum(o.changed_stim for o in good),
                n - sum(o.changed_stim for o in good),
                len(bad),
                0,
                f">= {spec.M2_DIFF_MIN}",
                _verdict_rate(
                    diff, n, 0, None if diff is None else diff >= spec.M2_DIFF_MIN, len(bad)
                ),
                fails=fails,
                extra={"p_stim": p_stim, "p_sham": p_sham, "missing_or_invalid": bad},
            )
        )
    return rows


# ── M3 ──────────────────────────────────────────────────────────────────────────────


def m3(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    layer = data.plan.layer
    gentle = [s for s in data.plan.stimuli if s in spec.GENTLE_STIMULI]
    loud = [s for s in data.plan.stimuli if s in spec.LOUD_STIMULI]
    other = [s for s in data.plan.stimuli if s not in spec.GENTLE_STIMULI | spec.LOUD_STIMULI]
    for preset in data.plan.presets:
        if gentle:
            good, bad = data.split(data.select(preset=preset, stimuli=gentle))
            n = len(good)
            k = sum(o.startled_stim for o in good)
            rate = k / n if n else None
            rows.append(
                Row(
                    layer,
                    "M3",
                    "溫和刺激的驚嚇／定格率",
                    preset,
                    rate,
                    n,
                    n - k,
                    k,
                    len(bad),
                    0,
                    f"<= {spec.M3_GENTLE_MAX:.0%}",
                    _verdict_rate(
                        rate, n, 0, None if rate is None else rate <= spec.M3_GENTLE_MAX, len(bad)
                    ),
                    fails=[_tid(o) for o in good if o.startled_stim],
                    extra={"stimuli": gentle, "missing_or_invalid": bad},
                )
            )
        else:
            rows.append(
                _untested(layer, "M3", "溫和刺激的驚嚇／定格率", preset, "這一層沒有溫和刺激")
            )
        for stimulus in sorted(spec.LOUD_STIMULI):
            if preset in BOLD_PRESETS:
                continue
            if stimulus not in loud:
                rows.append(
                    _untested(
                        layer,
                        "M3",
                        f"很大聲／逼近：{stimulus} 驚嚇率",
                        preset,
                        f"這一層沒有 {stimulus}",
                    )
                )
                continue
            good, bad = data.split(data.select(preset=preset, stimuli=[stimulus]))
            n = len(good)
            k = sum(o.startled_stim for o in good)
            rate = k / n if n else None
            rows.append(
                Row(
                    layer,
                    "M3",
                    f"很大聲／逼近：{stimulus} 驚嚇率（非大膽型）",
                    preset,
                    rate,
                    n,
                    k,
                    n - k,
                    len(bad),
                    0,
                    f">= {spec.M3_LOUD_MIN:.0%}",
                    _verdict_rate(
                        rate, n, 0, None if rate is None else rate >= spec.M3_LOUD_MIN, len(bad)
                    ),
                    fails=[_tid(o) for o in good if not o.startled_stim],
                    extra={"missing_or_invalid": bad},
                )
            )
    for stimulus in other:  # clap, poke: in neither of the spec's lists
        for preset in data.plan.presets:
            good, bad = data.split(data.select(preset=preset, stimuli=[stimulus]))
            n = len(good)
            k = sum(o.startled_stim for o in good)
            rows.append(
                Row(
                    layer,
                    "M3",
                    f"{stimulus} 驚嚇率（規格沒歸類，只報數字）",
                    preset,
                    k / n if n else None,
                    n,
                    n - k,
                    k,
                    len(bad),
                    0,
                    "（無門檻）",
                    REPORT,
                )
            )
    return rows


def _untested(layer: str, metric: str, part: str, scope: str, why: str) -> Row:
    return Row(layer, metric, part, scope, None, 0, 0, 0, 0, 0, "", UNTESTED, note=why)


# ── M4 ──────────────────────────────────────────────────────────────────────────────


def m4(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    if layer != "L2":
        return [
            _untested(
                layer, "M4", "說真話", "ALL", "M4 只在物理層量（腦層的世界沒有真實的物體位置可比）"
            )
        ]
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = data.split(data.select(preset=preset))
        b_ok = b_n = r_ok = r_n = unmatched = other = 0
        arrivals: list[tuple[PairOutcome, dict[str, Any]]] = []
        for o in good:
            for c in o.claims:
                b_ok += c.bearing_ok
                b_n += c.bearing_n
                r_ok += c.range_ok
                r_n += c.range_n
                unmatched += c.unmatched_ticks
                other += c.other_ticks
                arrivals += [(o, a) for a in c.arrivals]
        unknown_arrivals = [x for x in arrivals if x[1]["ok"] is None]
        known_arrivals = [x for x in arrivals if x[1]["ok"] is not None]
        a_ok = sum(1 for _, a in known_arrivals if a["ok"])
        b_rate = b_ok / b_n if b_n else None
        r_rate = r_ok / r_n if r_n else None
        a_rate = a_ok / len(known_arrivals) if known_arrivals else None
        rows.append(
            Row(
                layer,
                "M4",
                "方位誤差 <= 20° 的時間比例",
                preset,
                b_rate,
                b_n,
                b_ok,
                b_n - b_ok,
                unmatched + len(bad),
                0,
                f">= {spec.M4_BEARING_MIN:.0%}",
                _verdict_rate(
                    b_rate,
                    b_n,
                    unmatched,
                    None if b_rate is None else b_rate >= spec.M4_BEARING_MIN,
                    len(bad),
                ),
                fails=sorted(
                    {
                        _tid(o)
                        for o in good
                        if o.claims and any(c.bearing_n > c.bearing_ok for c in o.claims)
                    }
                ),
                note=f"單位是 tick（每 tick 一次），兩個組都算；目標不是物體（人、窩、地點）的 tick 不比：{other}",
            )
        )
        rows.append(
            Row(
                layer,
                "M4",
                "距離誤差 <= 30% 的時間比例",
                preset,
                r_rate,
                r_n,
                r_ok,
                r_n - r_ok,
                unmatched + len(bad),
                0,
                f">= {spec.M4_RANGE_MIN:.0%}",
                _verdict_rate(
                    r_rate,
                    r_n,
                    unmatched,
                    None if r_rate is None else r_rate >= spec.M4_RANGE_MIN,
                    len(bad),
                ),
                fails=sorted(
                    {
                        _tid(o)
                        for o in good
                        if o.claims and any(c.range_n > c.range_ok for c in o.claims)
                    }
                ),
                note="真實距離＝軀幹中心到物體中心",
            )
        )
        verdict = (
            FAIL
            if a_ok < len(known_arrivals)  # one wrong 到了 is a failure whatever else is unknown
            else INCOMPLETE
            if unknown_arrivals or bad or not known_arrivals
            else PASS
        )
        rows.append(
            Row(
                layer,
                "M4",
                "腦說「到了」時真實距離 <= 可碰距離 + 0.1 m",
                preset,
                a_rate,
                len(known_arrivals),
                a_ok,
                len(known_arrivals) - a_ok,
                len(unknown_arrivals) + len(bad),
                0,
                f"= {spec.M4_ARRIVE_MIN:.0%}",
                verdict,
                fails=[
                    f"{_tid(o)}|t={a['t']:.1f}s 離 {a['object']} 真實距離 {a['true_dist_m']} m > {a['bound_m']} m"
                    for o, a in known_arrivals
                    if not a["ok"]
                ],
                note="未知＝說「到了」的那一刻腦的目標不是世界叫得出名字的東西（多半是記憶裡的「地點」），無從知道牠說的是哪個東西；"
                "未知的次數算在「未知」欄，不能算通過",
                extra={"arrivals": [{**a, "trial": _tid(o)} for o, a in arrivals]},
            )
        )
        near = [a for _, a in arrivals if "nearest_dist_m" in a]
        near_ok = sum(1 for a in near if a["nearest_dist_m"] <= a["nearest_bound_m"])
        rows.append(
            Row(
                layer,
                "M4",
                "（只報數字）說「到了」時，離最近的一個東西 <= 可碰距離 + 0.1 m 的次數",
                preset,
                near_ok / len(near) if near else None,
                len(near),
                near_ok,
                len(near) - near_ok,
                0,
                0,
                "（不當門檻）",
                REPORT,
                note="不問牠說的是哪個東西，只看身邊最近的東西有多遠",
            )
        )
    return rows


# ── M5 ──────────────────────────────────────────────────────────────────────────────


def m5(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = data.split(data.select(preset=preset))
        caused = [(o, o.episode.category) for o in good if o.caused and o.episode is not None]
        n = len(caused)
        cats: dict[str, int] = {}
        for _o, category in caused:
            cats[category] = cats.get(category, 0) + 1
        natural = cats.get("natural", 0)
        timeout = cats.get("timeout", 0)
        nat_rate = natural / n if n else None
        to_rate = timeout / n if n else None
        not_caused = len(good) - n
        common = {"episodes": cats, "missing_or_invalid": bad}
        rows.append(
            Row(
                layer,
                "M5",
                "刺激引起的行為：自然結束比例",
                preset,
                nat_rate,
                n,
                natural,
                n - natural,
                len(bad),
                not_caused,
                f">= {spec.M5_NATURAL_MIN:.0%}",
                _verdict_rate(
                    nat_rate,
                    n,
                    0,
                    None if nat_rate is None else nat_rate >= spec.M5_NATURAL_MIN,
                    len(bad),
                ),
                fails=[_tid(o) + f"|{category}" for o, category in caused if category != "natural"],
                note="分母＝有刺激組在 3 秒內換了行為、而同種子對照組沒換的回合",
                extra=common,
            )
        )
        rows.append(
            Row(
                layer,
                "M5",
                "刺激引起的行為：逾時結束比例",
                preset,
                to_rate,
                n,
                n - timeout,
                timeout,
                len(bad),
                not_caused,
                f"<= {spec.M5_TIMEOUT_MAX:.0%}",
                _verdict_rate(
                    to_rate,
                    n,
                    0,
                    None if to_rate is None else to_rate <= spec.M5_TIMEOUT_MAX,
                    len(bad),
                ),
                fails=[_tid(o) for o, category in caused if category == "timeout"],
                extra=common,
            )
        )
        m = len(good)
        reopen = sum(o.reopen_stim for o in good)
        re_rate = reopen / m if m else None
        rows.append(
            Row(
                layer,
                "M5",
                "10 秒內同一行為重開 3 次以上的回合比例",
                preset,
                re_rate,
                m,
                m - reopen,
                reopen,
                len(bad),
                0,
                f"<= {spec.M5_REOPEN_MAX:.0%}",
                _verdict_rate(
                    re_rate,
                    m,
                    0,
                    None if re_rate is None else re_rate <= spec.M5_REOPEN_MAX,
                    len(bad),
                ),
                fails=[_tid(o) for o in good if o.reopen_stim],
            )
        )
    return rows


# ── M6 ──────────────────────────────────────────────────────────────────────────────


def _profile(outs: list[PairOutcome], stimuli: Sequence[str]) -> list[float] | None:
    by_stim = {o.stimulus: o for o in outs}
    if any(s not in by_stim for s in stimuli):
        return None
    vec: list[float] = []
    for s in stimuli:
        o = by_stim[s]
        vec += [
            1.0 if o.changed_stim else 0.0,
            1.0 if o.startled_stim else 0.0,
            1.0 if o.attend_stim else 0.0,
        ]
    return vec


def _rms(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a))


def m6(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    stimuli = data.plan.stimuli
    profiles: dict[str, list[list[float]]] = {}
    seeds_missing: list[str] = []
    for preset in data.plan.presets:
        profiles[preset] = []
        for seed in data.plan.seeds:
            ids = [pair_id(layer, preset, seed, s) for s in stimuli]
            good, bad = data.split(ids)
            vec = None if bad else _profile(good, stimuli)
            if vec is None or any(o.attend_stim is None for o in good):
                seeds_missing.append(f"{preset}|seed{seed}")
            else:
                profiles[preset].append(vec)
    rows: list[Row] = []
    centroid = {
        p: [sum(c) / len(v) for c in zip(*v, strict=True)] for p, v in profiles.items() if v
    }
    diameter = {
        p: max((_rms(a, b) for a in v for b in v), default=0.0) for p, v in profiles.items()
    }
    radius = {
        p: max((_rms(a, centroid[p]) for a in v), default=0.0) for p, v in profiles.items() if v
    }
    names = list(data.plan.presets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            scope = f"{a} vs {b}"
            if a not in centroid or b not in centroid or seeds_missing:
                rows.append(
                    Row(
                        layer,
                        "M6",
                        "個性之間的距離 > 同一個性內的最大距離",
                        scope,
                        None,
                        0,
                        0,
                        0,
                        len(seeds_missing),
                        0,
                        "嚴格大於",
                        INCOMPLETE,
                        note="有種子的回合缺或無效：" + ", ".join(seeds_missing[:6]),
                    )
                )
                continue
            between = _rms(centroid[a], centroid[b])
            within = max(diameter[a], diameter[b])
            rows.append(
                Row(
                    layer,
                    "M6",
                    "個性之間的距離 > 同一個性內的最大距離",
                    scope,
                    between,
                    len(profiles[a]) + len(profiles[b]),
                    1 if between > within else 0,
                    0 if between > within else 1,
                    0,
                    0,
                    "嚴格大於",
                    PASS if between > within else FAIL,
                    fails=[] if between > within else [scope],
                    extra={
                        "between_centroids": between,
                        "within_max_pairwise": within,
                        "within_max_to_centroid": max(radius[a], radius[b]),
                    },
                )
            )
    for preset in data.plan.presets:
        acts = signature_acts(preset)
        missing_by_seed: dict[int, list[str]] = {}
        incomplete = 0
        for seed in data.plan.seeds:
            ids = data.select(preset=preset)
            ids = [i for i in ids if i.split("|")[2] == str(seed)]
            good, bad = data.split(ids)
            if bad:
                incomplete += 1
                continue
            seen: frozenset[str] = frozenset().union(*[o.behaviours_seen for o in good])
            miss = [x for x in acts if x not in seen]
            if miss:
                missing_by_seed[seed] = miss
        n = len(data.plan.seeds)
        ok = n - len(missing_by_seed) - incomplete
        verdict = INCOMPLETE if incomplete else (PASS if not missing_by_seed else FAIL)
        rows.append(
            Row(
                layer,
                "M6",
                "招牌動作（" + "、".join(acts) + "）每顆種子都出現",
                preset,
                ok / n if n else None,
                n,
                ok,
                len(missing_by_seed),
                incomplete,
                0,
                "= 100%",
                verdict,
                fails=[f"seed{s} 缺 {'、'.join(m)}" for s, m in sorted(missing_by_seed.items())],
                extra={"acts": list(acts)},
            )
        )
    return rows


# ── M7 ──────────────────────────────────────────────────────────────────────────────


def m7(data: Outcomes) -> list[Row]:
    layer = data.plan.layer
    if layer != "L2":
        return [_untested(layer, "M7", "安全", "ALL", "M7 只在物理層量")]
    rows: list[Row] = []
    for preset in data.plan.presets:
        good, bad = data.split(data.select(preset=preset))
        f_stim = sum(o.falls_stim for o in good)
        f_sham = sum(o.falls_sham for o in good)
        ok = f_stim <= f_sham + spec.M7_FALL_SLACK
        rows.append(
            Row(
                layer,
                "M7",
                "有刺激組跌倒數 <= 對照組 + 1",
                preset,
                float(f_stim - f_sham),
                len(good),
                f_stim,
                f_sham,
                len(bad),
                0,
                f"stim <= sham + {spec.M7_FALL_SLACK}",
                _verdict_rate(float(f_stim - f_sham), len(good), 0, ok, len(bad)),
                fails=[_tid(o) for o in good if o.falls_stim > 0],
                note="value＝stim 跌倒數 − sham 跌倒數；ok 欄＝stim 跌倒數，fail 欄＝sham 跌倒數",
            )
        )
        events = sum(o.falls_stim + o.falls_sham for o in good)
        blocked = sum(o.blocked_stim + o.blocked_sham for o in good)
        if events == 0:
            verdict = INCOMPLETE
            note = "整批沒有一次跌倒，安全反射沒有被觸發過，無從驗證沒被擋（不能算通過）"
        else:
            verdict = INCOMPLETE if bad else (PASS if blocked <= spec.M7_BLOCKED_MAX else FAIL)
            note = "擋住＝倒下超過 0.5 秒後腦還在下走路指令"
        rows.append(
            Row(
                layer,
                "M7",
                "安全反射被擋次數 = 0",
                preset,
                float(blocked),
                events,
                events - blocked,
                blocked,
                len(bad),
                0,
                "= 0（要有跌倒事件才算量到）",
                verdict,
                fails=[_tid(o) for o in good if o.blocked_stim or o.blocked_sham],
                note=note,
            )
        )
    return rows


# ── approach (reported only) ───────────────────────────────────────────────────────


def approach_rows(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    for stimulus in data.plan.stimuli:
        if spec.STIMULUS_KIND[stimulus] != "object":
            continue
        for preset in data.plan.presets:
            good, _ = data.split(data.select(preset=preset, stimuli=[stimulus]))
            vals = [o.approach for o in good if o.approach]
            if not vals:
                continue
            mins = sorted(v["min_m"] for v in vals)
            reached = sum(1 for v in vals if v["min_m"] <= v["touchable_m"])
            rows.append(
                Row(
                    data.plan.layer,
                    "走過去",
                    f"{stimulus}：後來最近距離的中位數（m）",
                    preset,
                    mins[len(mins) // 2],
                    len(vals),
                    reached,
                    len(vals) - reached,
                    0,
                    0,
                    "（不當門檻）",
                    REPORT,
                    note="ok 欄＝最近距離 <= 可碰距離 + 0.1 m 的回合數；起點距離中位數 "
                    + f"{sorted(v['start_m'] for v in vals)[len(vals) // 2]:.2f} m",
                )
            )
    return rows


def all_rows(data: Outcomes) -> list[Row]:
    rows: list[Row] = []
    for fn in (m1, m2, m3, m4, m5, m6, m7, approach_rows):
        rows += fn(data)
    return rows


# ── the verdict table ───────────────────────────────────────────────────────────────

GATED = ("M1", "M2", "M3", "M4", "M5", "M6", "M7")


def combine(verdicts: Iterable[str]) -> str:
    vs = [v for v in verdicts if v != REPORT]
    if not vs:
        return REPORT
    if FAIL in vs:
        return FAIL
    if INCOMPLETE in vs:
        return INCOMPLETE
    if UNTESTED in vs:
        return UNTESTED if all(v == UNTESTED for v in vs) else INCOMPLETE
    return PASS


def summary(rows: Sequence[Row]) -> dict[tuple[str, str, str], str]:
    """(layer, metric, scope) → combined verdict (a failure dominates; an untested part makes the
    whole at best ``incomplete``)."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for r in rows:
        if r.metric not in GATED:
            continue
        groups.setdefault((r.layer, r.metric, r.scope), []).append(r.verdict)
    return {k: combine(v) for k, v in groups.items()}
