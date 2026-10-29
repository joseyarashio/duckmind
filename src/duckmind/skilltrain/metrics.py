"""The Stage-1 gates (training plan §6), computed from recorded traces only.

Everything here reads a trace (the arrays of :class:`~duckmind.skilltrain.skillduck.Trace`,
50 Hz) and the trial's markers — never a simulator — so the A100 tier can feed the same functions
with the same fields, and the witness on the page can use the same thresholds ("performed" =
this module's visibility verdict).

**Markers.** ``t_start``: the first tick the skill (or the v5 program) drives; ``t_hold_end``:
the first tick of its unwind (for a v5 program, the end of its active part); ``t_end``: the first
tick after it that robotd hands back (a v5 program: ``t_hold_end`` + 0.8 s). The reference state
is the tick just before ``t_start``.

**Visibility** (plan §6, each doubling as the "performed" detector):

* startle, standing — first visible motion (trunk ≥ 3 mm or ≥ 3°, or the beak ≥ 3 mm) ≤ 0.12 s;
  neck pull-back (Δ neck_pitch, + = back, as pet:probe's calibration) ≥ 0.15 rad at k 0.25 up
  to ≥ 0.25 rad at k 1.0 within 0.25 s; trunk dip ≥ 20·k mm within 0.3 s; the hold within
  ±3 mm of its median with ≤ 1 footfall per 2 s;
* startle / freeze / hop-back from walking — trunk speed ≤ 0.02 m/s within 0.7 s and ≤ 12 cm
  travelled (the A100 run's definitions: speed from a centred 0.5-s chord, the coast as the
  displacement over 2 s — :func:`chord_speed`); **and** (startle / freeze, s1b:fix 2026-09-28)
  the same neck pull-back and trunk dip as standing (≥ 0.15–0.25 rad within 0.25 s, ≥ 20·k mm
  within 0.3 s): a plain stop passes the stop checks, so without these the walking gate could not
  fail against today's body (v5's own stop: neck 0.00–0.12 rad p90, dip ≤ 13 mm p90, 30/30
  "stopped");
* hop-back, standing — ≥ 6 cm backward (along the start heading) within 0.7 s, ≤ 15° of yaw,
  planted at the end (both feet down, ≤ 0.02 m/s);
* shake-off — ≥ 0.5 s of trunk roll oscillation with every half-cycle ≥ 4° and at 2.5–4 Hz;
  head yaw ≥ ±0.15 rad at ≥ 3 Hz; ≤ 1 footfall; drift ≤ 2 cm;
* stretch — tall: trunk ≥ +8 mm and the head camera pitched ≥ +15° (optical axis above the
  horizon) together for ≥ 0.8 s; seated: ≥ +10 mm over the seat, same pitch, and re-seated
  (within 5 mm of the seat, robotd ``sit``) ≤ 1.5 s after the unwind.

**Safety.** Falls by robotd's rule (tilt > 60° for 0.2 s); any tilt > 45°; XL330-housing,
head and trunk *impacts* with the floor (> 2 N, risen by > 2 N within 0.1 s: :func:`impacts`)
while upright (tilt < 45°: "outside falls");
per-tick peak actuator force per joint group (legs, head) — p95 ≤ 0.35 N·m, the share of
joint-ticks above 0.4 N·m (velstand's strain threshold) and at ≥ 95 % of the servo model's
ceiling; stalls (> 0.4 N·m at < 0.5 rad/s) ≤ 0.5 %; joint acceleration p99 ≤ 300 rad/s²;
righting (tilt < 25°, no fall verdict, walk/stand for 1 s) within 8 s of a fall; the hand-back
(no fall in the 5 s after ``t_end``; walking resumed within ±10 % of its speed 2 s after the
walk command comes back).

Rates are k/n with Wilson 95 % (:func:`wilson`); a rate gate is PASS only when the interval's
upper end is under the gate, FAIL when its lower end is over it, else UNRESOLVED at this n.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DT = 0.02
LEG_IDX = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
HEAD_IDX = (5, 6, 7, 8)
NECK, HEAD_PITCH, HEAD_YAW = 5, 6, 7
# gate thresholds (plan §6)
FIRST_MOTION_S = 0.12
MOTION_MM, MOTION_DEG = 3.0, 3.0
RETRACT_LO, RETRACT_HI = 0.15, 0.25  # at k 0.25 and k 1.0
DIP_PER_K_MM = 20.0  # 5/10/15/20 mm at k 0.25/0.5/0.75/1.0
HOLD_DEV_MM = 3.0
BOB_MM = 8.0  # (s1b:fix) a rise and a new dip both this big: two crouches, not one
FOOTFALLS_PER_2S = 1.0
STILL_MPS, STILL_HOLD_S = 0.02, 0.1
STOP_WINDOW_TICKS = 100  # the coast is measured over 2 s (A100 neuro validation)
STOP_S, STOP_CM = 0.7, 12.0
HOP_CM, HOP_S, HOP_YAW_DEG = 6.0, 0.7, 15.0
SHAKE_ROLL_DEG, SHAKE_HZ, SHAKE_S = 4.0, (2.5, 4.0), 0.5
SHAKE_YAW_RAD, SHAKE_YAW_HZ = 0.15, 3.0
SHAKE_FOOTFALLS, SHAKE_DRIFT_CM = 1, 2.0
STRETCH_MM = {"stretch_tall": 8.0, "stretch_sit": 10.0}
STRETCH_CAM_DEG, STRETCH_HOLD_S, RESEAT_S, RESEAT_MM = 15.0, 0.8, 1.5, 5.0
FALL_GATE = {"low": 0.002, "high": 0.005}  # ≤ 0.2 % (startle 1–3, shake, stretch), ≤ 0.5 %
RIGHTING_GATE, RIGHTING_S = 0.85, 8.0
TAU_P95_GATE, STRAIN_NM, STALL_GATE, STALL_VEL = 0.35, 0.4, 0.005, 0.5
ACC_P99_GATE = 300.0
IMPACT_N = 2.0
HANDBACK_S, RESUME_TOL, RESUME_AT_S = 5.0, 0.10, 2.0
UPRIGHT_DEG, UPRIGHT_HOLD_S = 25.0, 1.0


# ── small helpers ────────────────────────────────────────────────────────────────────────────


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": round(k / n, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
    }


def rate_verdict(k: int, n: int, gate: float, *, at_most: bool = True) -> str:
    """PASS / FAIL / UNRESOLVED for "rate ≤ gate" (``at_most``) or "rate ≥ gate"."""
    lo, hi = wilson(k, n)
    if at_most:
        return "PASS" if hi <= gate else ("FAIL" if lo > gate else "UNRESOLVED")
    return "PASS" if lo >= gate else ("FAIL" if hi < gate else "UNRESOLVED")


def n_for_zero(gate: float, z: float = 1.959964) -> int:
    """The smallest n at which 0/n has a Wilson upper bound ≤ ``gate`` (the bound is
    z²/(n + z²) at k = 0)."""
    n = max(1, math.ceil(z * z * (1.0 - gate) / gate - 1e-9))
    while wilson(0, n, z)[1] > gate:
        n += 1
    return n


def dist(values: Iterable[float | None]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 10, 50, 90, 95])
    return {
        "n": int(v.size),
        "p5": _r(q[0]),
        "p10": _r(q[1]),
        "p50": _r(q[2]),
        "p90": _r(q[3]),
        "p95": _r(q[4]),
        "min": _r(v.min()),
        "max": _r(v.max()),
        "mean": _r(v.mean()),
    }


def _r(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def rpy(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll, pitch, yaw (rad) of (w, x, y, z) quaternions, the A100 reader's formulas."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_angle_deg(q0: np.ndarray, q: np.ndarray) -> np.ndarray:
    dot = np.abs(np.clip(q @ q0, -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def wrap(a: np.ndarray | float) -> Any:
    return np.arctan2(np.sin(a), np.cos(a))


def chord_speed(pos: np.ndarray, half: int = 12) -> np.ndarray:
    """Horizontal trunk speed per tick from a centred 0.5-s chord (m/s), as the A100 neuro
    validation's ``tick_speed``: the waddle's sway averages out; < 0.02 m/s = < 1 cm in 0.5 s."""
    n = len(pos)
    i = np.arange(n)
    a = np.clip(i - half - 1, 0, n - 1)
    b = np.clip(i + half, 0, n - 1)
    dt = np.maximum((b - a) * DT, DT)
    return np.linalg.norm(pos[b, :2] - pos[a, :2], axis=1) / dt


def impacts(force: np.ndarray, threshold: float = IMPACT_N, window: int = 5) -> np.ndarray:
    """Ticks where a part hits the floor: its contact force is over ``threshold`` and rose by
    more than ``threshold`` within the last ``window`` ticks (0.1 s). A load it rests on (a
    seated duck sits on its hip housings at ~4 N) is not an impact; a slam is."""
    f = np.asarray(force, float)
    low = np.array([f[max(0, k - window) : k].min() if k else f[0] for k in range(len(f))])
    return (f > threshold) & (f - low > threshold)


def housing_impacts(
    each: np.ndarray, i_ref: int, threshold: float = IMPACT_N, window: int = 5
) -> np.ndarray:
    """Ticks where an XL330 housing hits the floor (``each``: ticks × housings, N). A housing
    free at the reference tick (< 0.5 N) hits when its force is over ``threshold`` and rose by
    more than ``threshold`` within 0.1 s — training's servo-impact threshold on a part that should
    not touch. A housing the duck already rests on (a seated duck sits on its hip housings at
    ~4 N) hits only when its force rises by more than twice that over its resting load."""
    each = np.asarray(each, float)
    if each.ndim != 2 or each.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    rest = each[i_ref]
    over = np.where(rest < 0.5, threshold, rest + 2 * threshold)
    hit = np.zeros(each.shape[0], dtype=bool)
    for j in range(each.shape[1]):
        hit |= impacts(each[:, j], threshold, window) & (each[:, j] > over[j])
    return hit


def bob(z: np.ndarray | Sequence[float]) -> float:
    """The biggest valley → peak → valley of a height series (same units): for every sample the
    smaller of its rise above the lowest point before it and above the lowest point after it,
    maximised. 0 for a crouch followed by one rise (nothing lower comes after the peak); the
    crouch–rise–crouch the review saw live (27 mm down, up, down again) scores about 27 mm."""
    h = np.asarray(z, dtype=float)
    if h.size < 3:
        return 0.0
    before = np.minimum.accumulate(h)
    after = np.minimum.accumulate(h[::-1])[::-1]
    return float(np.max(np.minimum(h - before, h - after)))


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where ``mask`` holds."""
    out: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _held_first(mask: np.ndarray, hold_ticks: int) -> int | None:
    for a, b in _runs(mask):
        if b - a >= hold_ticks:
            return a
    return None


# ── markers ──────────────────────────────────────────────────────────────────────────────────


def skill_markers(tr: Mapping[str, Any], skill: str, after: float = 0.0) -> dict[str, float | None]:
    """``t_start`` / ``t_hold_end`` / ``t_end`` of the first run of ``skill`` after ``after``,
    from robotd's labels (``skill`` while holding, ``skill:unwind`` while unwinding)."""
    t = tr["t"]
    labels = tr["label"]
    mine = np.array([lab == skill or lab == f"{skill}:unwind" for lab in labels])
    mine &= t >= after - 1e-9
    i0 = _first(mine)
    if i0 is None:
        return {"t_start": None, "t_hold_end": None, "t_end": None}
    j = i0
    while j < len(labels) and (labels[j] == skill or labels[j] == f"{skill}:unwind"):
        j += 1
    unwind = [i for i in range(i0, j) if labels[i] == f"{skill}:unwind"]
    t_end = float(t[j]) if j < len(t) else None
    t_hold_end = float(t[unwind[0]]) if unwind else t_end
    return {"t_start": float(t[i0]), "t_hold_end": t_hold_end, "t_end": t_end}


def any_skill_markers(
    tr: Mapping[str, Any], names: Sequence[str], after: float = 0.0
) -> dict[str, Any]:
    """Markers for a chain of entries (a freeze ended by ``startle_release`` …): from the first
    tick any of ``names`` drives to the first tick none does."""
    t, labels = tr["t"], tr["label"]
    base = {n: n for n in names} | {f"{n}:unwind": n for n in names}
    mine = np.array([lab in base for lab in labels]) & (t >= after - 1e-9)
    i0 = _first(mine)
    if i0 is None:
        return {"t_start": None, "t_hold_end": None, "t_end": None, "sequence": []}
    j = i0
    seq: list[str] = []
    while j < len(labels) and labels[j] in base:
        if not seq or seq[-1] != labels[j]:
            seq.append(str(labels[j]))
        j += 1
    unwinds = [
        i
        for i in range(i0, j)
        if str(labels[i]).endswith(":unwind") or base.get(str(labels[i])) == "startle_release"
    ]
    t_end = float(t[j]) if j < len(t) else None
    t_hold_end = float(t[unwinds[0]]) if unwinds else t_end
    return {"t_start": float(t[i0]), "t_hold_end": t_hold_end, "t_end": t_end, "sequence": seq}


def span_markers(tr: Mapping[str, Any], names: Sequence[str], after: float = 0.0) -> dict[str, Any]:
    """(s1b:fix) Like :func:`any_skill_markers`, but from the first tick any of ``names`` drives
    to the first tick after the **last** one, gaps included: a startle that hands back and a
    freeze started later are judged as one trial (the crouch → rise → crouch the review saw)."""
    t, labels = tr["t"], tr["label"]
    base = {n: n for n in names} | {f"{n}:unwind": n for n in names}
    mine = np.array([lab in base for lab in labels]) & (t >= after - 1e-9)
    idx = np.flatnonzero(mine)
    if not idx.size:
        return {"t_start": None, "t_hold_end": None, "t_end": None, "sequence": []}
    i0, i1 = int(idx[0]), int(idx[-1])
    seq: list[str] = []
    for lab in labels[i0 : i1 + 1]:
        if not seq or seq[-1] != str(lab):
            seq.append(str(lab))
    unwinds = [
        i
        for i in range(i0, i1 + 1)
        if str(labels[i]).endswith(":unwind") or base.get(str(labels[i])) == "startle_release"
    ]
    t_end = float(t[i1 + 1]) if i1 + 1 < len(t) else None
    t_hold_end = float(t[unwinds[0]]) if unwinds else t_end
    return {"t_start": float(t[i0]), "t_hold_end": t_hold_end, "t_end": t_end, "sequence": seq}


# ── per-trial metrics ────────────────────────────────────────────────────────────────────────


def trial_metrics(
    tr: Mapping[str, Any],
    *,
    t_start: float,
    t_hold_end: float,
    t_end: float,
    t_resume: float | None = None,
    ceiling_nm: float = 0.64,
    post_s: float = HANDBACK_S,
) -> dict[str, Any]:
    """Every visibility and safety metric of one trial (the gates pick theirs by family)."""
    t = np.asarray(tr["t"], float)
    n = len(t)
    i0 = int(np.searchsorted(t, t_start - 1e-9))
    i_ref = max(i0 - 1, 0)
    i_hold = int(np.searchsorted(t, t_hold_end - 1e-9))
    i_end = int(np.searchsorted(t, t_end - 1e-9))
    i_post = int(np.searchsorted(t, t_end + post_s - 1e-9))
    pos, quat = tr["pos"], tr["quat"]
    q, qd, qacc, tau = tr["q"], tr["qd"], tr["qacc"], tr["tau_pk"]
    roll, pitch, yaw = rpy(quat)
    out: dict[str, Any] = {
        "ticks": n,
        "t_start": _r(t_start),
        "t_hold_end": _r(t_hold_end),
        "t_end": _r(t_end),
        "skill_s": _r(t_end - t_start),
        "hold_s": _r(t_hold_end - t_start),
    }

    def window(a: float, b: float) -> slice:
        return slice(
            int(np.searchsorted(t, t_start + a - 1e-9)), int(np.searchsorted(t, t_start + b + 1e-9))
        )

    # ── visibility: timing and startle shape ──
    pos0, quat0 = pos[i_ref], quat[i_ref]
    beak_w = _beak_world(tr)
    moved = (
        (np.linalg.norm(pos - pos0, axis=1) * 1000 >= MOTION_MM)
        | (quat_angle_deg(quat0, quat) >= MOTION_DEG)
        | (np.linalg.norm(beak_w - beak_w[i_ref], axis=1) * 1000 >= MOTION_MM)
    )
    moved[:i0] = False
    moved[max(i_hold, i0 + 1) :] = False  # the startle's onset: within the hold, not a later drift
    moved = moved & np.concatenate([moved[1:], [False]])  # persistent: ≥ 2 ticks in a row
    first = _first(moved)
    out["first_motion_s"] = _r(t[first] - t_start) if first is not None else None
    w25, w30, w70 = window(0.0, 0.25), window(0.0, 0.30), window(0.0, 0.70)
    w40 = window(0.0, 0.40)
    d_neck = q[:, NECK] - q[i_ref, NECK]
    out["retract_025_rad"] = _r(max(0.0, float(np.max(d_neck[w25], initial=0.0))))
    out["retract_040_rad"] = _r(max(0.0, float(np.max(d_neck[w40], initial=0.0))))
    out["retract_max_rad"] = _r(max(0.0, float(np.max(d_neck[i0:i_end], initial=0.0))))
    d_head = np.linalg.norm(q[:, list(HEAD_IDX)] - q[i_ref, list(HEAD_IDX)], axis=1)
    out["head_move_025_rad"] = _r(float(np.max(d_head[w25], initial=0.0)))
    z = pos[:, 2]
    out["dip_030_mm"] = _r(max(0.0, (z[i_ref] - float(np.min(z[w30], initial=z[i_ref]))) * 1000))
    out["dip_040_mm"] = _r(max(0.0, (z[i_ref] - float(np.min(z[w40], initial=z[i_ref]))) * 1000))
    out["dip_max_mm"] = _r(
        max(0.0, (z[i_ref] - float(np.min(z[i0:i_end], initial=z[i_ref]))) * 1000)
    )
    # (s1b:fix) a crouch → rise → crouch inside one skill sequence: the biggest rise between two
    # dips (the continuity check of startle → freeze: one crouch, one rise)
    out["bob_mm"] = _r(bob(z[i0:i_end]) * 1000)
    out["pitch_change_deg"] = _r(
        math.degrees(float(np.max(np.abs(wrap(pitch[i0:i_end] - pitch[i_ref])), initial=0.0)))
    )
    hold = slice(int(np.searchsorted(t, t_start + 0.3 - 1e-9)), i_hold)
    if hold.stop - hold.start >= 3:
        zh = z[hold]
        out["hold_dev_mm"] = _r(float(np.max(np.abs(zh - np.median(zh)))) * 1000)
        dur = max((hold.stop - hold.start) * DT, DT)
        out["hold_footfalls_per_2s"] = _r(float(tr["td"][hold].sum()) / dur * 2.0)
    else:
        out["hold_dev_mm"] = None
        out["hold_footfalls_per_2s"] = None
    out["footfalls_skill"] = int(tr["td"][i0:i_end].sum())

    # ── walking stop (the A100 neuro validation's definitions: the trunk speed from a centred
    # 0.5-s chord, "still" at its first tick under 0.02 m/s, the coast = displacement in 2 s) ──
    chord = chord_speed(pos)
    # the coast window: 2 s, or until the walk command comes back (a skill shorter than 2 s
    # hands back and the client resumes: that is walking again, not coasting)
    i_resume = int(np.searchsorted(t, t_resume - 1e-9)) if t_resume is not None else n
    e2 = max(i0 + 1, min(n, i0 + STOP_WINDOW_TICKS, i_resume))
    still = chord[i0:e2] < STILL_MPS
    j = _first(still)
    out["still_s"] = _r((j + 1) * DT) if j is not None else None
    out["coast_window_s"] = _r((e2 - i0) * DT)
    out["travel_cm"] = _r(float(np.linalg.norm(pos[e2 - 1, :2] - pos[i0, :2])) * 100)
    out["speed_before_mps"] = _r(float(np.mean(chord[max(i0 - 50, 0) : i0])) if i0 > 0 else 0.0)

    # ── hop-back ──
    heading = np.array([math.cos(yaw[i_ref]), math.sin(yaw[i_ref])])
    fwd = (pos[:, :2] - pos0[:2]) @ heading
    out["back_070_cm"] = _r(max(0.0, -float(np.min(fwd[w70], initial=0.0))) * 100)
    out["back_max_cm"] = _r(max(0.0, -float(np.min(fwd[i0:i_end], initial=0.0))) * 100)
    out["yaw_change_deg"] = _r(abs(math.degrees(float(wrap(yaw[min(i_end, n - 1)] - yaw[i_ref])))))
    e = min(i_end, n - 1)
    out["planted_end"] = bool(tr["feet"][e].all() and chord[e] < STILL_MPS)

    # ── shake ──
    ws = slice(int(np.searchsorted(t, t_start + 0.2 - 1e-9)), i_hold)
    out.update(_shake(roll[ws], q[ws, HEAD_YAW]))
    out["drift_cm"] = _r(float(np.linalg.norm(pos[e, :2] - pos0[:2])) * 100)

    # ── stretch ──
    rise_mm = (z - z[i_ref]) * 1000
    cam = np.asarray(tr["cam_pitch"], float)
    out["rise_max_mm"] = _r(float(np.max(rise_mm[i0:i_end], initial=0.0)))
    out["cam_pitch_ref_deg"] = _r(float(cam[i_ref]))
    out["cam_pitch_max_deg"] = _r(float(np.max(cam[i0:i_end], initial=cam[i_ref])))
    for fam, mm in STRETCH_MM.items():
        ok = (rise_mm >= mm) & (cam >= STRETCH_CAM_DEG)
        ok[:i0] = False
        ok[i_end:] = False
        best = max((b - a for a, b in _runs(ok)), default=0)
        out[f"{fam}_hold_s"] = _r(best * DT)
    labels = tr["label"]
    tilt_all = np.asarray(tr["tilt"], float)
    seat = (
        (np.abs(z - z[i_ref]) * 1000 <= RESEAT_MM)
        & np.array([lab == "sit" for lab in labels])
        & (tilt_all < UPRIGHT_DEG)  # a duck on its back can sit as low as its seat
    )
    seat[:i_end] = False
    k = _held_first(seat, int(round(0.2 / DT)))
    out["reseat_s"] = _r(t[k] - t_end) if k is not None else None

    # ── safety ──
    fallen = np.asarray(tr["fallen"], float) > 0.5
    tilt = np.asarray(tr["tilt"], float)
    span = slice(i0, i_post)
    fall_i = _first(fallen[i0:])
    out["fell"] = bool(fall_i is not None and i0 + fall_i < i_post)
    out["fall_t"] = _r(t[i0 + fall_i] - t_start) if fall_i is not None else None
    out["fell_in_skill"] = bool(fallen[i0:i_end].any())
    out["fell_after"] = bool(fallen[i_end:i_post].any()) and not bool(fallen[i0:i_end].any())
    out["t45"] = bool((tilt[span] > 45.0).any())
    out["tilt_max_deg"] = _r(float(np.max(tilt[span], initial=0.0)))
    out["tilt_p95_deg"] = (
        _r(float(np.percentile(tilt[span], 95))) if span.stop > span.start else None
    )
    upright = tilt < 45.0
    each = tr.get("f_housing")
    for key, name in (("f_servo", "servo"), ("f_head", "head"), ("f_trunk", "trunk")):
        f = np.asarray(tr[key], float)
        if name == "servo" and each is not None and len(each):
            hits = housing_impacts(np.asarray(each, float), i_ref) & upright
        else:
            hits = impacts(f) & upright
        out[f"{name}_hits_upright"] = int(hits[span].sum())
        out[f"{name}_hits_all"] = int(
            (
                housing_impacts(np.asarray(each, float), i_ref)
                if name == "servo" and each is not None and len(each)
                else impacts(f)
            )[span].sum()
        )
        out[f"{name}_force_max_n"] = _r(float(np.max(f[span], initial=0.0)))
        out[f"{name}_rest_n"] = _r(float(f[i_ref]))
    sk = slice(i0, max(i_end, i0 + 1))
    tau_leg = tau[sk][:, list(LEG_IDX)].max(axis=1)
    tau_head = tau[sk][:, list(HEAD_IDX)].max(axis=1)
    out["tau_leg"] = [round(float(v), 3) for v in tau_leg]
    out["tau_head"] = [round(float(v), 3) for v in tau_head]
    out["tau_leg_p95"] = _r(float(np.percentile(tau_leg, 95)))
    out["tau_head_p95"] = _r(float(np.percentile(tau_head, 95)))
    out["tau_max"] = _r(float(tau[sk].max()))
    cells = tau[sk].size
    out["strain_ticks"] = int((tau[sk] > STRAIN_NM).sum())
    out["sat_ticks"] = int((tau[sk] >= 0.95 * ceiling_nm).sum())
    out["stall_ticks"] = int(((tau[sk] > STRAIN_NM) & (np.abs(qd[sk]) < STALL_VEL)).sum())
    out["joint_ticks"] = int(cells)
    acc = np.abs(qacc[sk]).max(axis=1)
    out["acc"] = [round(float(v), 1) for v in acc]
    out["acc_p99"] = _r(float(np.percentile(acc, 99)), 1)
    out["acc_over_300_ticks"] = int((np.abs(qacc[sk]) > ACC_P99_GATE).sum())
    # righting after a fall
    if fall_i is not None:
        f0 = i0 + fall_i
        ok = (tilt < UPRIGHT_DEG) & ~fallen & np.array([lab in ("walk", "stand") for lab in labels])
        ok[:f0] = False
        r = _held_first(ok, int(round(UPRIGHT_HOLD_S / DT)))
        out["righted_s"] = _r(t[r] - t[f0]) if r is not None else None
        out["righted_8s"] = bool(r is not None and t[r] - t[f0] <= RIGHTING_S)
    else:
        out["righted_s"] = None
        out["righted_8s"] = None
    # the hand-back
    out["handback_fall"] = bool(fallen[i_end:i_post].any())
    if t_resume is not None and out["speed_before_mps"]:
        a = int(np.searchsorted(t, t_resume + RESUME_AT_S - 0.5 - 1e-9))
        b = int(np.searchsorted(t, t_resume + RESUME_AT_S + 0.5 + 1e-9))
        if b > a:
            v = float(np.mean(chord[a:b]))
            out["resume_ratio"] = _r(v / out["speed_before_mps"])
        else:
            out["resume_ratio"] = None
    else:
        out["resume_ratio"] = None
    return out


def _beak_world(tr: Mapping[str, Any]) -> np.ndarray:
    """The beak tip in the world frame, rebuilt from the trunk pose and its trunk-frame track."""
    quat, pos, beak = tr["quat"], tr["pos"], tr["beak"]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r = np.empty((len(w), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return pos + np.einsum("nij,nj->ni", r, beak)


def _shake(roll: np.ndarray, head_yaw: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(roll) < 8:
        return {
            "roll_osc_s": 0.0,
            "roll_amp_deg": None,
            "roll_hz": None,
            "yaw_amp_rad": None,
            "yaw_hz": None,
        }
    rd = np.degrees(roll - np.mean(roll))
    halves = _half_cycles(rd)
    good = [
        (dur, peak)
        for dur, peak in halves
        if peak >= SHAKE_ROLL_DEG
        and 1 / (2 * SHAKE_HZ[1]) - DT <= dur <= 1 / (2 * SHAKE_HZ[0]) + DT
    ]
    best = run = 0.0
    for dur, peak in halves:
        ok = peak >= SHAKE_ROLL_DEG and (
            1 / (2 * SHAKE_HZ[1]) - DT <= dur <= 1 / (2 * SHAKE_HZ[0]) + DT
        )
        run = run + dur if ok else 0.0
        best = max(best, run)
    out["roll_osc_s"] = _r(best)
    out["roll_good_halves"] = len(good)
    out["roll_amp_deg"] = _r(float(np.percentile(np.abs(rd), 95)))
    out["roll_hz"] = _r(_zero_cross_hz(rd))
    yd = head_yaw - np.mean(head_yaw)
    out["yaw_amp_rad"] = _r(float((np.max(yd) - np.min(yd)) / 2))
    out["yaw_hz"] = _r(_zero_cross_hz(yd))
    return out


def _half_cycles(x: np.ndarray) -> list[tuple[float, float]]:
    """(duration s, peak |x|) of each half-cycle between sign changes."""
    sign = np.sign(x)
    sign[sign == 0] = 1
    edges = np.flatnonzero(np.diff(sign)) + 1
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        out.append(((b - a) * DT, float(np.max(np.abs(x[a:b])))))
    return out


def _zero_cross_hz(x: np.ndarray) -> float:
    sign = np.sign(x)
    sign[sign == 0] = 1
    crossings = int(np.count_nonzero(np.diff(sign)))
    dur = len(x) * DT
    return crossings / 2.0 / dur if dur > 0 else 0.0


# ── visibility verdicts ──────────────────────────────────────────────────────────────────────


def visibility(
    family: str, level: float | None, posture: str, m: Mapping[str, Any]
) -> dict[str, Any]:
    """Pass/fail of each visibility check for a trial (``performed`` = all of them)."""
    checks: dict[str, bool] = {}
    k = 1.0 if level is None else float(level)

    def ge(key: str, gate: float) -> bool:
        v = m.get(key)
        return v is not None and float(v) >= gate

    def le(key: str, gate: float) -> bool:
        v = m.get(key)
        return v is not None and float(v) <= gate

    if family in ("startle", "freeze", "hop_back") and posture == "walk":
        checks["stops_in_0.7s"] = le("still_s", STOP_S)
        checks["travel_le_12cm"] = le("travel_cm", STOP_CM)
        if family in ("startle", "freeze"):
            # (s1b:fix) a visible flinch while stopping, not just a stop (v5's stop passes the
            # two checks above 30/30): the standing thresholds, same windows
            kk = min(max(k, 0.25), 1.0)
            need = RETRACT_LO + (kk - 0.25) * (RETRACT_HI - RETRACT_LO) / 0.75
            checks[f"walk_neck_back_ge_{need:.3f}rad"] = ge("retract_025_rad", need)
            checks[f"walk_dip_ge_{DIP_PER_K_MM * kk:.0f}mm"] = ge("dip_030_mm", DIP_PER_K_MM * kk)
    elif family in ("startle", "freeze"):
        kk = min(max(k, 0.25), 1.0)
        need = RETRACT_LO + (kk - 0.25) * (RETRACT_HI - RETRACT_LO) / 0.75
        checks["first_motion_le_0.12s"] = le("first_motion_s", FIRST_MOTION_S)
        checks[f"retract_ge_{need:.3f}rad"] = ge("retract_025_rad", need)
        checks[f"dip_ge_{DIP_PER_K_MM * kk:.0f}mm"] = ge("dip_030_mm", DIP_PER_K_MM * kk)
        checks["hold_within_3mm"] = le("hold_dev_mm", HOLD_DEV_MM)
        checks["footfalls_le_1_per_2s"] = le("hold_footfalls_per_2s", FOOTFALLS_PER_2S)
    elif family == "continuation":
        # (s1b:fix) a startle a freeze took over: the first entry's flinch (its level) and ONE
        # crouch — no rise of ≥ 8 mm between two dips (the crouch deepening to the freeze's
        # level is not a hold failure)
        kk = min(max(k, 0.25), 1.0)
        need = RETRACT_LO + (kk - 0.25) * (RETRACT_HI - RETRACT_LO) / 0.75
        if posture == "walk":
            checks["stops_in_0.7s"] = le("still_s", STOP_S)
            checks["travel_le_12cm"] = le("travel_cm", STOP_CM)
        else:
            checks["first_motion_le_0.12s"] = le("first_motion_s", FIRST_MOTION_S)
        checks[f"retract_ge_{need:.3f}rad"] = ge("retract_025_rad", need)
        checks[f"dip_ge_{DIP_PER_K_MM * kk:.0f}mm"] = ge("dip_030_mm", DIP_PER_K_MM * kk)
        checks[f"one_crouch_bob_lt_{BOB_MM:g}mm"] = le("bob_mm", BOB_MM - 1e-9)
    elif family == "hop_back":
        checks["back_ge_6cm_in_0.7s"] = ge("back_070_cm", HOP_CM)
        checks["yaw_le_15deg"] = le("yaw_change_deg", HOP_YAW_DEG)
        checks["planted_end"] = bool(m.get("planted_end"))
    elif family == "shake":
        checks["roll_osc_ge_0.5s"] = ge("roll_osc_s", SHAKE_S)
        checks["head_yaw_ge_0.15rad"] = ge("yaw_amp_rad", SHAKE_YAW_RAD)
        checks["head_yaw_ge_3hz"] = ge("yaw_hz", SHAKE_YAW_HZ)
        checks["footfalls_le_1"] = le("footfalls_skill", SHAKE_FOOTFALLS)
        checks["drift_le_2cm"] = le("drift_cm", SHAKE_DRIFT_CM)
    elif family == "stretch_tall":
        checks["tall_hold_ge_0.8s"] = ge("stretch_tall_hold_s", STRETCH_HOLD_S)
    elif family == "stretch_sit":
        checks["seated_hold_ge_0.8s"] = ge("stretch_sit_hold_s", STRETCH_HOLD_S)
        checks["reseated_le_1.5s"] = le("reseat_s", RESEAT_S)
    if not checks:
        return {"checks": {}, "performed": None}  # no visibility gate for this family
    return {"checks": checks, "performed": all(checks.values())}


def gate_family(entry_family: str, posture: str) -> str:
    """Which visibility gate a condition is judged by (a stand-in or a v5 baseline is judged by
    the gate of the motion it stands for)."""
    if entry_family == "dummy":
        return {"stand": "startle", "walk": "startle", "sit": "stretch_sit"}[posture]
    if entry_family in ("release", "abort"):
        return "none"  # they only end a skill: safety gates, no visibility gate
    return entry_family


# ── condition summaries ──────────────────────────────────────────────────────────────────────


def summarize(
    rows: Sequence[Mapping[str, Any]], *, fall_gate: float = FALL_GATE["low"]
) -> dict[str, Any]:
    """One condition: rates with Wilson 95 %, distributions, pooled strain, gate verdicts."""
    ok = [r for r in rows if not r.get("excluded")]
    n = len(ok)
    out: dict[str, Any] = {"n": n, "excluded": len(rows) - n}
    if not n:
        return out
    ms = [r["metrics"] for r in ok]
    falls = sum(1 for m in ms if m["fell"])
    out["falls"] = rate(falls, n)
    out["falls_in_skill"] = rate(sum(1 for m in ms if m["fell_in_skill"]), n)
    out["t45"] = rate(sum(1 for m in ms if m["t45"]), n)
    out["handback_falls"] = rate(sum(1 for m in ms if m["handback_fall"]), n)
    fallen = [m for m in ms if m["fell"]]
    out["righted_8s"] = rate(sum(1 for m in fallen if m["righted_8s"]), len(fallen))
    out["servo_hits_upright"] = rate(sum(1 for m in ms if m["servo_hits_upright"] > 0), n)
    out["head_hits_upright"] = rate(sum(1 for m in ms if m["head_hits_upright"] > 0), n)
    out["trunk_hits_upright"] = rate(sum(1 for m in ms if m["trunk_hits_upright"] > 0), n)
    leg = np.concatenate([np.asarray(m["tau_leg"], float) for m in ms]) if ms else np.zeros(0)
    head = np.concatenate([np.asarray(m["tau_head"], float) for m in ms]) if ms else np.zeros(0)
    acc = np.concatenate([np.asarray(m["acc"], float) for m in ms]) if ms else np.zeros(0)
    cells = sum(m["joint_ticks"] for m in ms)
    stall = sum(m["stall_ticks"] for m in ms)
    strain = sum(m["strain_ticks"] for m in ms)
    sat = sum(m["sat_ticks"] for m in ms)
    over300 = sum(m["acc_over_300_ticks"] for m in ms)
    out["strain"] = {
        "tau_leg_p95": _r(np.percentile(leg, 95)) if leg.size else None,
        "tau_head_p95": _r(np.percentile(head, 95)) if head.size else None,
        "tau_leg_max": _r(leg.max()) if leg.size else None,
        "tau_head_max": _r(head.max()) if head.size else None,
        "over_0.4Nm_frac": _r(strain / cells, 5) if cells else None,
        "at_ceiling_frac": _r(sat / cells, 5) if cells else None,
        "stall_frac": _r(stall / cells, 5) if cells else None,
        "acc_p99": _r(np.percentile(acc, 99), 1) if acc.size else None,
        "acc_over_300_frac": _r(over300 / cells, 5) if cells else None,
        "joint_ticks": int(cells),
    }
    out["moved_in_hold"] = rate(sum(1 for m in ms if m.get("first_motion_s") is not None), n)
    vis = [r.get("visibility") or {} for r in ok]
    gated = [v for v in vis if v.get("performed") is not None]
    out["performed"] = rate(sum(1 for v in gated if v.get("performed")), len(gated))
    check_names = sorted({c for v in vis for c in (v.get("checks") or {})})
    out["checks"] = {
        c: rate(sum(1 for v in vis if (v.get("checks") or {}).get(c)), n) for c in check_names
    }
    keys = (
        "first_motion_s",
        "retract_025_rad",
        "head_move_025_rad",
        "dip_030_mm",
        "dip_040_mm",
        "dip_max_mm",
        "retract_040_rad",
        "bob_mm",
        "hold_dev_mm",
        "hold_footfalls_per_2s",
        "footfalls_skill",
        "still_s",
        "travel_cm",
        "back_070_cm",
        "yaw_change_deg",
        "roll_osc_s",
        "roll_amp_deg",
        "roll_hz",
        "yaw_amp_rad",
        "yaw_hz",
        "drift_cm",
        "rise_max_mm",
        "cam_pitch_max_deg",
        "stretch_tall_hold_s",
        "stretch_sit_hold_s",
        "reseat_s",
        "tilt_max_deg",
        "tilt_p95_deg",
        "resume_ratio",
        "righted_s",
        "skill_s",
        "speed_before_mps",
    )
    out["dist"] = {k: dist(m.get(k) for m in ms) for k in keys}
    resumed = [m["resume_ratio"] for m in ms if m.get("resume_ratio") is not None]
    out["resume_within_10pct"] = rate(
        sum(1 for v in resumed if abs(v - 1.0) <= RESUME_TOL), len(resumed)
    )
    s = out["strain"]
    out["verdicts"] = {
        "falls": rate_verdict(falls, n, fall_gate),
        "handback_falls": rate_verdict(out["handback_falls"]["k"], n, fall_gate),
        "righting": (
            rate_verdict(out["righted_8s"]["k"], len(fallen), RIGHTING_GATE, at_most=False)
            if fallen
            else "no falls"
        ),
        "servo_hits_upright": "PASS" if out["servo_hits_upright"]["k"] == 0 else "FAIL",
        "tau_p95": (
            "PASS" if max(s["tau_leg_p95"] or 0, s["tau_head_p95"] or 0) <= TAU_P95_GATE else "FAIL"
        ),
        "stall": "PASS" if (s["stall_frac"] or 0) <= STALL_GATE else "FAIL",
        "acc_p99": "PASS" if (s["acc_p99"] or 0) <= ACC_P99_GATE else "FAIL",
        "fall_gate": fall_gate,
        "n_for_fall_gate_with_0_falls": n_for_zero(fall_gate),
    }
    return out


__all__ = [
    "any_skill_markers",
    "bob",
    "dist",
    "gate_family",
    "n_for_zero",
    "rate",
    "rate_verdict",
    "rpy",
    "skill_markers",
    "span_markers",
    "summarize",
    "trial_metrics",
    "visibility",
    "wilson",
]
