"""s1b:fix: the page's witness judges a trained skill by the skill's own detectors.

While robotd runs a trained skill it **ignores the head command**: the skill network moves the
whole body (the crouch, the neck pull-back, the hop, the shake). The witness (fid:page,
``simweb.neuro_witness``) judged every reflex window against the head the motor commanded and the
parts the brain named (a turn toward the sound), so a real 16–27 mm trained crouch read
「身體只做到一部分：頭往右轉沒有做出來」 live (tr:gate, 4 of 6). Inside a skill window those checks
mean nothing; this module judges it with the Mac gates' definitions
(:mod:`duckmind.skilltrain.metrics`, the numbers repeated here — this module is pure standard
library, like the witness — and ``tests/test_skilltrain_live_judge.py`` checks they agree):

* **startle / freeze** — the neck pulled back ≥ 0.15–0.25 rad (by level k) within 0.25 s of the
  skill's first tick, the trunk down ≥ 20·k mm within 0.3 s (the live samples come at ~48/s:
  each window gets one sample of slack); from a walk also stopped (the trunk's centred 0.5-s
  chord under 0.02 m/s within 0.7 s); the crouch **one crouch**: no rise of ≥ 8 mm followed by a
  new dip of ≥ 8 mm (a startle and the freeze after it);
* **hop_back** — ≥ 6 cm back along the starting heading by 0.7 s, ≤ 15° of turn;
* **shake** — the head yaw swinging ≥ ±0.15 rad (the tilt's own swing said, not graded: the
  witness has no trunk roll);
* **stretch** — the trunk up ≥ 8 mm (standing) / ≥ 10 mm (seated).

It also says how long after the brain's reflex the body visibly moved (the stimulus → motion
latency the ethology target puts at 0.15–0.30 s) and, as information only, the parts the brain
named that a skill does not make (the head turn toward the sound): not graded, because robotd
ignores the head command while the skill runs.

Duck-typed on the witness's ``Tick``: ``t_ms``, ``policy``, ``head`` (4 offsets from HOME, rad),
``trunk`` (x, y, z, heading; the physics truth) with ``trunk_t``, ``fallen``, ``tilt_deg``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.i18n import lst, msg
from duckmind.skilltrain.table import LIVE_FREEZE, LIVE_STARTLE

# metrics.py's thresholds (tests/test_skilltrain_live_judge.py keeps them equal)
RETRACT_LO, RETRACT_HI = 0.15, 0.25
DIP_PER_K_MM = 20.0
RETRACT_S, DIP_S = 0.25, 0.30
STOP_S, STOP_CM, STILL_MPS = 0.7, 12.0, 0.02
HOP_CM, HOP_S, HOP_YAW_DEG = 6.0, 0.7, 15.0
SHAKE_YAW_RAD = 0.15
STRETCH_MM = {"stretch_tall": 8.0, "stretch_sit": 10.0}
HOLD_DEV_MM = 3.0
# live-only
SLACK_S = 0.05  # one live truth sample (~48/s)
BOB_MM = 8.0  # a rise and a new dip both this big: two crouches, not one
MOTION_MM, MOTION_RAD = 3.0, math.radians(3.0)  # the first visible motion
WALKING_MPS = 0.05
CHORD_S = 0.5

FAMILIES: dict[str, tuple[str, float]] = {
    **{name: ("startle", k) for name, k in LIVE_STARTLE},
    **{name: ("freeze", k) for name, k in LIVE_FREEZE},
    "hop_back": ("hop_back", 1.0),
    "shake_1": ("shake", 0.5),
    "shake_2": ("shake", 1.0),
    "stretch_tall": ("stretch_tall", 1.0),
    "stretch_sit": ("stretch_sit", 1.0),
    "startle_release": ("release", 0.0),
    "skill_abort": ("abort", 0.0),
    # (2026-10-05) the B skills and the hop-over: named in 「現在在做」, not judged as reflex answers
    "turn_left": ("turn", 1.0),
    "turn_right": ("turn", -1.0),
    "quick_turn_left": ("quick_turn", 1.0),
    "quick_turn_right": ("quick_turn", -1.0),
    "march": ("march", 0.0),
    "bow": ("bow", 1.0),
    "lie_low": ("lie", 1.0),
    "bow_release": ("release_bow", 0.0),
    "hop_over_1": ("hop_over", 0.01),
    "hop_over_2": ("hop_over", 0.02),
    "hop_over_3": ("hop_over", 0.03),
    "hop_over_4": ("hop_over", 0.04),
    "step_up": ("step_up", 1.0),
    "step_down": ("step_down", 1.0),
}
DOING_ZH = {
    "startle": msg("neuro.skill.doing.startle"),
    "freeze": msg("neuro.skill.doing.freeze"),
    "hop_back": msg("neuro.skill.doing.hop_back"),
    "shake": msg("neuro.skill.doing.shake"),
    "stretch_tall": msg("neuro.skill.doing.stretch_tall"),
    "stretch_sit": msg("neuro.skill.doing.stretch_sit"),
    "release": msg("neuro.skill.doing.release"),
    "abort": msg("neuro.skill.doing.abort"),
    "turn": msg("neuro.skill.doing.turn"),
    "quick_turn": msg("neuro.skill.doing.turn"),
    "march": msg("neuro.skill.doing.march"),
    "bow": msg("neuro.skill.doing.bow"),
    "lie": msg("neuro.skill.doing.lie"),
    "release_bow": msg("neuro.skill.doing.release_bow"),
    "hop_over": msg("neuro.skill.doing.hop_over"),
    "step_up": msg("neuro.skill.doing.step_up"),
    "step_down": msg("neuro.skill.doing.step_down"),
}
DOING_ANY = msg("neuro.skill.doing.any")
YAW_PARTS = {
    "yaw_left": msg("neuro.skill.judge.yaw_left"),
    "yaw_right": msg("neuro.skill.judge.yaw_right"),
}
# the reflexes a trained skill answers (the startle family's picker): judged by the skill
SKILL_REFLEXES = frozenset({"startle", "freeze"})
# the brain's acts a comfort skill answers (the picker's mapping), for a caller that judges acts
ACT_FAMILIES = {
    "body_shake": ("shake",),
    "stretch_yawn": ("stretch_tall", "stretch_sit"),
    "wake_ritual": ("stretch_tall", "stretch_sit"),
}


def skill_of(
    policy: str | None, families: Mapping[str, tuple[str, float]] | None = None
) -> tuple[str, str, float] | None:
    """(entry, family, level) when robotd's label is one of our trained skills."""
    table = FAMILIES if families is None else families
    base = (policy or "").split(":", 1)[0]
    if base in table:
        fam, level = table[base]
        return base, fam, level
    return None


def doing_zh(policy: str | None) -> Any:
    """「現在在做」 for a trained skill (None: not one)."""
    found = skill_of(policy)
    return None if found is None else DOING_ZH.get(found[1], DOING_ANY)


def _need_retract(k: float) -> float:
    kk = min(max(k, 0.25), 1.0)
    return RETRACT_LO + (kk - 0.25) * (RETRACT_HI - RETRACT_LO) / 0.75


def bob_m(z: Sequence[float]) -> float:
    """metrics.bob in pure Python: the biggest valley → peak → valley."""
    if len(z) < 3:
        return 0.0
    before, low = [], math.inf
    for v in z:
        low = min(low, v)
        before.append(low)
    after, low = [0.0] * len(z), math.inf
    for i in range(len(z) - 1, -1, -1):
        low = min(low, z[i])
        after[i] = low
    return max(min(v - b, v - a) for v, b, a in zip(z, before, after, strict=True))


def _truth(ticks: Iterable[Any]) -> list[tuple[int, float, float, float, float]]:
    """(t_ms, x, y, z, heading) of each new physics sample."""
    out: list[tuple[int, float, float, float, float]] = []
    last = None
    for k in ticks:
        trunk, tt = getattr(k, "trunk", None), getattr(k, "trunk_t", None)
        if trunk is None or tt is None or tt == last:
            continue
        last = tt
        out.append((int(k.t_ms), float(trunk[0]), float(trunk[1]), float(trunk[2]),
                    float(trunk[3])))  # fmt: skip
    return out


def _chord_speed(points: Sequence[tuple[int, float, float, float, float]], i: int) -> float | None:
    """The trunk's speed at sample ``i`` from a centred 0.5-s chord (m/s)."""
    t = points[i][0]
    lo = [p for p in points if p[0] >= t - CHORD_S * 500]
    hi = [p for p in points if p[0] <= t + CHORD_S * 500]
    if not lo or not hi:
        return None
    a, b = lo[0], hi[-1]
    span = (b[0] - a[0]) / 1000.0
    if span < 0.3:
        return None
    return math.hypot(b[1] - a[1], b[2] - a[2]) / span


def skill_span(ticks: Sequence[Any], t0_ms: int) -> tuple[str, str, float, int, int] | None:
    """The first trained skill robotd ran at or after ``t0_ms``: (entry, family, level, first
    tick, last tick of the run of our skills that follows — a startle, the freeze that took it
    over, the release)."""
    start = None
    for idx, k in enumerate(ticks):
        if int(k.t_ms) < t0_ms:
            continue
        found = skill_of(getattr(k, "policy", ""))
        if found is not None and found[1] not in ("release", "abort"):
            start = (idx, found)
            break
    if start is None:
        return None
    idx, (entry, fam, level) = start
    end = idx
    for j in range(idx, len(ticks)):
        if skill_of(getattr(ticks[j], "policy", "")) is not None:
            end = j
        elif int(ticks[j].t_ms) - int(ticks[end].t_ms) > 400:
            break  # (a gap of 0.4 s: the run is over)
    # the deepest level of the crouch family inside the run
    for j in range(idx, end + 1):
        f = skill_of(getattr(ticks[j], "policy", ""))
        if f is not None and f[1] in ("startle", "freeze") and fam in ("startle", "freeze"):
            level = max(level, f[2])
    return entry, fam, level, int(ticks[idx].t_ms), int(ticks[end].t_ms)


def judge_window(
    ticks: Sequence[Any],
    reflex: str,
    t0_ms: int,
    *,
    named: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """A reflex window in which a trained skill ran, judged by the skill's own detectors: a dict
    with ``verdict`` (done / partial / not_done), ``parts`` (plain words), ``zh`` (the page's
    line), ``skill`` (entry, family, level), ``checks``, the numbers and the latencies. None when
    no trained skill ran in the window (the witness judges it as before).

    Only what a trained skill answers is judged by the skill (:data:`SKILL_REFLEXES`: a
    startle or a freeze answered by the startle network; :data:`ACT_FAMILIES`: the brain's
    shake / stretch acts answered by the comfort network); any other reflex (a look toward a
    sound…) whose window a skill ran through is **not graded** (``verdict`` None): robotd
    ignored the head then, and the row says so in plain words instead of 「做不到」."""
    span = skill_span(ticks, t0_ms)
    if span is None:
        return None
    entry, fam, level, s_ms, e_ms = span
    answers = (reflex in SKILL_REFLEXES and fam in ("startle", "freeze", "hop_back")) or (
        fam in ACT_FAMILIES.get(reflex, ())
    )
    if not answers:
        doing = DOING_ZH.get(fam, DOING_ANY)
        return {
            "skill": {"entry": entry, "family": fam, "level": level},
            "skill_start_s": round((s_ms - t0_ms) / 1000.0, 2),
            "verdict": None,
            "checks": {},
            "parts": [],
            "reflex": reflex,
            "zh": msg("neuro.skill.judge.not_graded", doing=doing),
            "not_graded": True,
        }
    points = _truth(ticks)
    before = [p for p in points if s_ms - 250 <= p[0] < s_ms]
    ref_pt = before[-1] if before else next((p for p in points if p[0] >= s_ms), None)
    heads_before = [k.head for k in ticks if s_ms - 120 <= int(k.t_ms) < s_ms]
    neck0 = (
        sum(h[0] for h in heads_before) / len(heads_before)
        if heads_before
        else float(next((k.head[0] for k in ticks if int(k.t_ms) >= s_ms), 0.0))
    )
    out: dict[str, Any] = {
        "skill": {"entry": entry, "family": fam, "level": level},
        "skill_start_s": round((s_ms - t0_ms) / 1000.0, 2),
    }
    checks: dict[str, bool] = {}
    parts: list[str] = []
    fails: list[str] = []

    def in_window(a_s: float, b_s: float) -> list[tuple[int, float, float, float, float]]:
        return [p for p in points if s_ms + a_s * 1000 <= p[0] <= s_ms + b_s * 1000]

    # the first visible motion (trunk ≥ 3 mm, the neck ≥ 3°), from the skill and from the reflex
    first_ms = None
    if ref_pt is not None:
        for k in ticks:
            t = int(k.t_ms)
            if t < s_ms or t > e_ms:
                continue
            moved = abs(float(k.head[0]) - neck0) >= MOTION_RAD
            tr = getattr(k, "trunk", None)
            if tr is not None:
                moved = moved or math.dist((tr[0], tr[1], tr[2]), ref_pt[1:4]) * 1000 >= MOTION_MM
            if moved:
                first_ms = t
                break
    out["motion_after_reflex_s"] = None if first_ms is None else round((first_ms - t0_ms) / 1000, 2)

    walking = False
    if len(before) >= 2 and before[-1][0] - before[0][0] >= 150:
        d = math.hypot(before[-1][1] - before[0][1], before[-1][2] - before[0][2])
        walking = d / ((before[-1][0] - before[0][0]) / 1000.0) >= WALKING_MPS
    out["walking"] = walking

    if fam in ("startle", "freeze") and ref_pt is not None:
        z0 = ref_pt[3]
        dip_w = in_window(0.0, DIP_S + SLACK_S)
        dip = max(0.0, (z0 - min((p[3] for p in dip_w), default=z0)) * 1000)
        necks = [float(k.head[0]) - neck0 for k in ticks
                 if s_ms <= int(k.t_ms) <= s_ms + (RETRACT_S + SLACK_S) * 1000]  # fmt: skip
        retract = max([0.0, *necks])
        span_z = [p[3] for p in points if s_ms <= p[0] <= e_ms + 200]
        bob = bob_m(span_z) * 1000 if span_z else 0.0
        deepest = max(0.0, (z0 - min(span_z, default=z0)) * 1000)
        need_r, need_d = _need_retract(level), DIP_PER_K_MM * min(max(level, 0.25), 1.0)
        checks["neck_back"] = retract >= need_r
        checks["trunk_down"] = dip >= need_d
        checks["one_crouch"] = bob < BOB_MM
        out.update({"dip_mm": round(dip, 1), "deepest_mm": round(deepest, 1),
                    "neck_back_deg": round(math.degrees(retract), 1), "bob_mm": round(bob, 1),
                    "need": {"dip_mm": need_d, "neck_rad": round(need_r, 3)}})  # fmt: skip
        when = out["motion_after_reflex_s"]
        timing = (
            msg("neuro.skill.judge.timing", s=round(max(0.1, when), 1)) if when is not None else ""
        )
        if checks["trunk_down"]:
            parts.append(msg("neuro.skill.judge.crouch", cm=round(deepest / 10, 1), timing=timing))
            timing = ""
        else:
            fails.append(
                msg(
                    "neuro.skill.judge.trunk_short",
                    got=round(dip / 10, 1),
                    need=round(need_d / 10, 1),
                )
            )
        if checks["neck_back"]:
            parts.append(
                msg("neuro.skill.judge.neck_back", deg=round(math.degrees(retract)), timing=timing)
            )
        else:
            fails.append(msg("neuro.skill.judge.neck_short", deg=round(math.degrees(retract))))
        if not checks["one_crouch"]:
            fails.append(msg("neuro.skill.judge.two_crouches", cm=round(bob / 10, 1)))
        if walking:
            still_at = None
            for i, p in enumerate(points):
                if p[0] < s_ms:
                    continue
                if p[0] > s_ms + 2000:
                    break
                v = _chord_speed(points, i)
                if v is not None and v < STILL_MPS:
                    still_at = (p[0] - s_ms) / 1000.0
                    break
            travel = None
            later = [p for p in points if s_ms <= p[0] <= s_ms + 2000]
            if later and ref_pt is not None:
                travel = math.hypot(later[-1][1] - ref_pt[1], later[-1][2] - ref_pt[2]) * 100
            checks["stopped"] = still_at is not None and still_at <= STOP_S + SLACK_S
            checks["travel_le_12cm"] = travel is not None and travel <= STOP_CM
            out["still_s"] = None if still_at is None else round(still_at, 2)
            out["travel_cm"] = None if travel is None else round(travel, 1)
            if checks["stopped"]:
                parts.append(msg("neuro.skill.judge.stopped", s=round(still_at or 0.0, 1)))
            else:
                fails.append(msg("neuro.skill.judge.not_stopped"))
    elif fam == "hop_back" and ref_pt is not None:
        heading = ref_pt[4]
        w = in_window(0.0, HOP_S + SLACK_S)
        back = max([0.0, *(-((p[1] - ref_pt[1]) * math.cos(heading)
                            + (p[2] - ref_pt[2]) * math.sin(heading)) for p in w)]) * 100  # fmt: skip
        end_pts = [p for p in points if p[0] <= e_ms + 100]
        turn = abs(math.degrees(math.atan2(math.sin(end_pts[-1][4] - heading),
                                           math.cos(end_pts[-1][4] - heading)))) if end_pts else 0.0  # fmt: skip
        checks["back_ge_6cm"] = back >= HOP_CM
        checks["turn_le_15deg"] = turn <= HOP_YAW_DEG
        out.update({"back_cm": round(back, 1), "turn_deg": round(turn, 1)})
        (parts if checks["back_ge_6cm"] else fails).append(
            msg("neuro.skill.judge.back", cm=round(back))
        )
        if not checks["turn_le_15deg"]:
            fails.append(msg("neuro.skill.judge.back_turned", deg=round(turn)))
    elif fam == "shake":
        yaws = [float(k.head[2]) for k in ticks if s_ms <= int(k.t_ms) <= e_ms]
        amp = (max(yaws) - min(yaws)) / 2 if yaws else 0.0
        tilts = [float(k.tilt_deg) for k in ticks
                 if s_ms <= int(k.t_ms) <= e_ms and k.tilt_deg is not None]  # fmt: skip
        checks["head_swing"] = amp >= SHAKE_YAW_RAD
        out.update({"head_yaw_rad": round(amp, 3),
                    "tilt_swing_deg": round(max(tilts) - min(tilts), 1) if tilts else None})  # fmt: skip
        (parts if checks["head_swing"] else fails).append(
            msg("neuro.skill.judge.shake_body", deg=round(math.degrees(amp)),
                body=round(out["tilt_swing_deg"]))
            if tilts
            else msg("neuro.skill.judge.shake", deg=round(math.degrees(amp)))
        )  # fmt: skip
    elif fam in STRETCH_MM and ref_pt is not None:
        rise = max([0.0, *((p[3] - ref_pt[3]) * 1000 for p in points if s_ms <= p[0] <= e_ms)])
        checks["rise"] = rise >= STRETCH_MM[fam]
        out["rise_mm"] = round(rise, 1)
        (parts if checks["rise"] else fails).append(
            msg("neuro.skill.judge.stretch", cm=round(rise / 10, 1))
        )
    if not checks:
        return None
    # the motion itself decides done / not done; "one crouch" can only lower a done to partial
    motion = [v for c, v in checks.items() if c != "one_crouch"]
    if not any(motion):
        verdict = "not_done"
    elif all(motion) and checks.get("one_crouch", True):
        verdict = "done"
    else:
        verdict = "partial"
    # parts the brain named that a whole-body skill does not make: said, not graded
    unmade = [YAW_PARTS[p] for p in (named or ()) if p in YAW_PARTS]
    note = msg("neuro.skill.judge.unmade", parts=lst(unmade)) if unmade else ""
    doing = DOING_ZH.get(fam, DOING_ANY)
    if verdict == "done":
        zh = msg("neuro.skill.judge.done", doing=doing, parts=lst(parts), note=note)
    elif verdict == "partial":
        zh = msg("neuro.skill.judge.partial", doing=doing, parts=lst(fails + parts), note=note)
    else:
        zh = msg("neuro.skill.judge.not_done", doing=doing, parts=lst(fails + parts), note=note)
    out.update({"verdict": verdict, "checks": checks, "parts": parts, "zh": zh,
                "reflex": reflex, "unmade_named": unmade})  # fmt: skip
    return out


__all__ = [
    "DOING_ZH",
    "FAMILIES",
    "SKILL_REFLEXES",
    "bob_m",
    "doing_zh",
    "judge_window",
    "skill_of",
    "skill_span",
]
