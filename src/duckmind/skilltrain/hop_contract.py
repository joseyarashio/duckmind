"""The forward hop-over skill in numbers (hop:feasibility, 2026-09-28) — pure standard library.

Owner 2026-09-28 ~19:40: 「我希望還有加入小跳躍，遇到不高的障礙或是門檻 個性有一定程度勇敢的 會主動嘗試跳躍」.
Orchestrator ~20:10: a NEW forward network (``hopover.onnx``), never a retrain of Stage 1b's startle
network (which owns the backward hop-back); servo command delay randomised 0–40 ms in training.

**What the body allows** (the feasibility evidence: measurement records, not in the repo):

* the shipped v5 walk lifts its feet ~11 mm (phase 1) and, in the Mac probe (both servo models, DR,
  n = 90 per height and edge), crosses 0.5–1 cm strips 56–87 % of the time, trips and falls at 2 cm
  24–62 %, and falls or sticks nose-down at 3–5 cm 60–80 %;
* a true two-foot hop *over* a 4 cm-deep strip needs ≥ 313 ms of flight (≥ 12 cm of centre-of-mass
  rise) at the walk's fastest forward speed; the ideal-balance bound for this body is a 2–18 mm rise
  (BAM XL330 at 6.5–8.2 V; the knee's no-load speed binds first) — so the skill is a **high step
  over** (one foot at a time, lifted above the strip, placed beyond it), which may include a short
  both-feet-off moment when the trailing foot pushes: that is the "small jump" the body can do.

**The skill as robotd runs it**: a one-shot ``[[policy.skill]]`` entry per height class (the brain
judges the height from the camera and picks the entry), from standing or walking, with the strip
0.06–0.18 m ahead of the trunk centre (the brain walks up to it by camera + odometry and then asks);
``command`` = (0.30 + 0.05 · class, 0, 0) — a forward walk command whose speed carries the height class
(1–4 cm → 0.35–0.50), so the network can start from v5 velstand (warm start, the walk slot's action
scale 0.9) and learn to lift over what it finds by feel (robotd's 61-D observation: it is otherwise
blind to the strip); ``duration`` 3.0 s, ``unwind`` 0.5 s with the zero command, then v5 velstand takes
over (the hand-back). Training value = deployment value (plan §2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STAGE = "hop-2026-09-28"
NET = "hopover"
TASK_ID = "Mjlab-DuckHopOver-Threshold-MicroDuck"
PLACEHOLDER = "/skilltrain/hopover.onnx"

# ── the strip (the terrain's obstacle) ──────────────────────────────────────────────────────
DEPTH_M = 0.04  # a door threshold / a book on its side (the Mac probe's)
DEPTH_RANGE = (0.03, 0.06)  # training varies it
WIDTH_M = 1.6  # across the whole tile
EDGES = ("sharp", "round")
HEIGHT_RANGE = (0.005, 0.040)  # the curriculum's span (row 0 → last row)
TERRAIN_ROWS = 8  # difficulty levels
TILE_M = (2.0, 2.0)
FACE_X_IN_TILE = 1.0  # the near face's x from the tile corner (the env origin sits on it)
START_D = (0.06, 0.18)  # trunk centre → near face at the skill's first tick
AIM_DEG = 12.0  # entry heading vs the strip's normal
HEIGHT_EST_NOISE = 0.005  # the brain's camera estimate of the height (m, uniform ±)


def round_radius(height_m: float, depth_m: float) -> float:
    """The round edge's radius: min(height, depth/2) — the Mac probe's shape."""
    return min(height_m, depth_m / 2.0)


def height_of_row(row: int, rows: int = TERRAIN_ROWS) -> float:
    """The strip height of a curriculum row (linear over HEIGHT_RANGE; mjlab adds a random
    fraction of a row inside ``difficulty``)."""
    lo, hi = HEIGHT_RANGE
    return lo + (hi - lo) * (row / max(1, rows - 1))


# ── the table entries (one per height class) ────────────────────────────────────────────────
HOLD_S = 3.0
UNWIND_S = 0.5
ACTION_SCALE = 0.9  # velstand's walk-slot scale: the warm start's own
GAIN_RATIO = 1.0
CMD_BASE, CMD_STEP = 0.30, 0.05  # command[0] = 0.30 + 0.05 · class (1–4 cm → 0.35–0.50)


@dataclass(frozen=True)
class Variant:
    name: str
    height_m: float  # the class's nominal height (the command)
    command: tuple[float, float, float]
    unwind: tuple[float, float, float]
    hold_s: float
    unwind_s: float
    postures: tuple[str, ...]
    action_scale: float = ACTION_SCALE
    gain_ratio: float = GAIN_RATIO


def height_class(height_m: float) -> int:
    """The entry the brain picks: its estimate rounded up to whole cm, 1–4."""
    return int(min(4, max(1, math.ceil(round(height_m * 100.0, 6) - 1e-9))))


def command_of(height_m: float) -> tuple[float, float, float]:
    return (round(CMD_BASE + CMD_STEP * height_class(height_m), 4), 0.0, 0.0)


def variants() -> tuple[Variant, ...]:
    """hop_over_1 … hop_over_4: 1, 2, 3, 4 cm classes (the brain rounds its estimate up)."""
    out = []
    for cm in (1, 2, 3, 4):
        h = cm / 100.0
        out.append(
            Variant(
                name=f"hop_over_{cm}",
                height_m=h,
                command=command_of(h),
                unwind=(0.0, 0.0, 0.0),
                hold_s=HOLD_S,
                unwind_s=UNWIND_S,
                postures=("stand", "walk"),
            )
        )
    return tuple(out)


def skill_defs(onnx_path: str = PLACEHOLDER) -> list[dict]:
    """robotd ``[[policy.skill]]`` fields for every entry (``table.SkillDef`` keyword args)."""
    return [
        {"name": v.name, "path": onnx_path, "duration": v.hold_s, "command": v.command,
         "unwind": v.unwind, "unwind_s": v.unwind_s, "chain": False,
         "action_scale": v.action_scale, "gain_ratio": v.gain_ratio}
        for v in variants()
    ]  # fmt: skip


# ── episodes ────────────────────────────────────────────────────────────────────────────────
ENTRY_MIX = {"walk": 0.55, "stand": 0.35, "coast": 0.10}  # library kinds (rl_contract.ENTRY_KINDS)
ENTRY_MIX_EARLY = {"walk": 0.30, "stand": 0.60, "coast": 0.10}
HOLD_JITTER = (0.9, 1.15)
SERVO_DELAY_STEPS = (0, 8)  # 0–40 ms at the 5 ms physics step (Stage 1b's finding; orchestrator)
# hop:train (2026-09-28): the lag is piecewise constant, as Stage 1b's retrain draws it
# (rl_contract.SERVO_LAG_TRAIN): a fresh 0–8-step lag per env, re-drawn every 100 physics steps (0.5 s,
# staggered) with p 0.5 — so training sees a steady 0 ms and a steady 40 ms, which the gates test. The
# calibration's "jitter" mode re-drew the lag every physics step (a noisy ~20 ms mean).
SERVO_LAG_TRAIN = {"min_lag": 0, "max_lag": 8, "update_period": 100, "hold_prob": 0.5}
DELAY_MODES = ("train", "jitter")  # or an int: a fixed lag in physics steps (5 ms each)

# ── success, as the gates will judge it (and the training metrics count) ────────────────────
PAST_FOOT_M = 0.028  # both soles' centroids this far past the far face (a sole is 5.4 cm)
PAST_TRUNK_M = 0.03
GOAL_TRUNK_M = 0.10  # the progress potential's cap: trunk centre 10 cm past the far face
FALL_TILT_DEG = 60.0  # robotd's rule (0.2 s)
FALL_DEBOUNCE_S = 0.2
STUMBLE_DEG = 20.0

BODY_WEIGHT_N = 0.7372 * 9.81  # the official model's total mass (hop_ceiling.MASS) × g
LANDING_BW_TARGET = 2.5  # the proposed per-foot landing peak target (× body weight)

# the reward terms a run may re-weight mid-way (hop_train's OVERRIDE.json), with the parameters it may
# change; the curriculum-driven weights (action_rate_l2, joint_torque_rate_l2) are not among them
OVERRIDABLE: dict[str, tuple[str, ...]] = {
    "progress": ("max_mps",),
    "cross_bonus": (),
    "upright": (),
    "heading": (),
    "swing_clearance": (),
    "settle": (),
    "toe_stub": (),
    "fallen": (),
    "leg_torque": ("free_nm",),
    "after_cross_speed": ("free_mps",),
    "servo_impact": ("threshold",),
    "foot_impact": ("free_bw",),
    "housing_touch": ("threshold",),
    "dash_speed": ("free_mps",),
    "body_ang_vel": (),
}

GATES = {
    "cross_rate_min": {"1": 0.95, "2": 0.90, "3": 0.80, "4": 0.70},  # proposals, per class
    "fall_rate_max": 0.02,  # during the skill, per class (a fall is a real outcome: reported)
    "handback_fall_max": 0.005,
    "torque_p95_nm": 0.35,
    "housing_impact_rate_max": 0.02,
}

# ── the ideal-balance hop bound (hop_ceiling.py, recorded 2026-09-28; ideal coordination) ───
CEILING = {
    "bam_6v5": {"com_rise_mm": 1.6, "flight_ms": 36, "clear_with_tuck_mm": 3.9},
    "bam_7v4": {"com_rise_mm": 5.5, "flight_ms": 67, "clear_with_tuck_mm": 10.3},
    "bam_8v2": {"com_rise_mm": 17.7, "flight_ms": 120, "clear_with_tuck_mm": 27.3},
    "mjcf": {"com_rise_mm": 13.7, "flight_ms": 106, "clear_with_tuck_mm": 20.4},
}


def hop_over_needs(forward_mps: float, over_m: float = DEPTH_M + 0.054) -> dict[str, float]:
    """What a two-foot hop *over* a strip needs: every sole airborne over ``over_m`` of travel."""
    t_air = over_m / forward_mps
    v0 = 9.81 * t_air / 2
    return {
        "flight_ms": t_air * 1000,
        "takeoff_mps": v0,
        "com_rise_mm": v0 * v0 / (2 * 9.81) * 1000,
    }


def delay_fields(mode: str | int) -> dict[str, float | int]:
    """The BAM actuator's delay fields (mjlab ``ActuatorCfg``) for a delay mode: ``"train"``
    (piecewise constant 0–40 ms), ``"jitter"`` (re-drawn every physics step, 0–40 ms: the
    calibration's), or a fixed lag in physics steps (0–8 = 0–40 ms)."""
    if mode == "train":
        lag = SERVO_LAG_TRAIN
        return {"delay_min_lag": int(lag["min_lag"]), "delay_max_lag": int(lag["max_lag"]),
                "delay_update_period": int(lag["update_period"]),
                "delay_hold_prob": float(lag["hold_prob"])}  # fmt: skip
    if mode == "jitter":
        lo, hi = SERVO_DELAY_STEPS
        return {"delay_min_lag": lo, "delay_max_lag": hi, "delay_update_period": 0,
                "delay_hold_prob": 0.0}  # fmt: skip
    steps = int(mode)
    if not 0 <= steps <= 20:
        raise ValueError(f"servo lag {steps} physics steps outside 0–20 (0–100 ms)")
    return {"delay_min_lag": steps, "delay_max_lag": steps, "delay_update_period": 0,
            "delay_hold_prob": 0.0}  # fmt: skip


def parse_delay(text: str | int) -> str | int:
    """``"train"`` / ``"jitter"`` / ``"20ms"`` / ``"4"`` (physics steps) → a delay mode."""
    s = str(text).strip().lower()
    if s in DELAY_MODES:
        return s
    if s.endswith("ms"):
        ms = float(s[:-2])
        if abs(ms / 5.0 - round(ms / 5.0)) > 1e-9:
            raise ValueError(f"{text}: a fixed delay must be a multiple of the 5 ms physics step")
        return int(round(ms / 5.0))
    return int(s)


def delay_label(mode: str | int) -> str:
    return str(mode) if isinstance(mode, str) else f"d{5 * int(mode)}ms"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


__all__ = [
    "EDGES",
    "GATES",
    "HEIGHT_RANGE",
    "NET",
    "TASK_ID",
    "Variant",
    "command_of",
    "delay_fields",
    "delay_label",
    "height_of_row",
    "hop_over_needs",
    "parse_delay",
    "round_radius",
    "skill_defs",
    "variants",
    "wilson",
]
