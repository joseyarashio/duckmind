"""The B-list movement skills in numbers (b:feasibility, 2026-09-28) — pure standard library.

Owner 2026-09-28 「A B 全數」, B = four trained movement skills: (1) 原地轉身 turning on the spot, only
where needed (cornered, tight gaps, facing a wall: normal turns stay arcs, owner motion design
2026-09-25); (2) 原地踏步／跺腳 marching / stomping in place (excitement, impatience, music — the A-list
dance and the songbird use it); (3) stepping up onto a 3–5 cm mat / cushion and back down (with the
hop-over skill); (4) 鞠躬／趴低 a front-down bow (play-bow invitation) and lying low, back to standing.

**What the v5 set does today** (``bsk_probe``, both servo models, DR ducks, n = 30 per cell;
the ``skillsB-2026-09-28`` probe in the measurement records, not in the repo):

* no turn on the spot from standing (pure yaw: 8–10° in 4 s p50; ≥ 90° in 3–8/30); after a walk a
  pure yaw keeps stepping and turns 29–35 °/s but wanders 8–14 cm (radius ≈ 0.24–0.27 m); arcs 0.25–0.43 m;
* no marching in place (dead zone: vx 0.20 from standing steps 1–4/30; after a walk it keeps walking
  forward 7–8 cm/s, heading ±50°);
* a 3–5 cm mat edge: 0/30 clean, falls 21–26/30; stepping off it: falls 6–18/30 by height;
* robotd's ground pick IS a slow deep front-down bow (30/30, 0 falls, pitch 33–34°, beak 20–25 mm off
  the floor, back by 2.8 s) — usable today; the sit is the low rest (rear down).

**What the body allows** (``bsk_body``, the hop workflow's trajectory search):

* no static one-leg stance: with the hip roll at its limit the centre of mass still misses the stance
  sole by 17–19 mm (sole half width 7 mm) — every step is dynamic;
* a swing sole can clear ≈ 50 mm (58 mm with the stance leg straight; knee/ankle at their limits); an
  open-loop step reaches 50–59 mm at leg torque p95 0.56–0.80 N·m (strain gate 0.35) → a 3 cm mat has
  margin, 4 cm is tight, 5 cm is the body's ceiling;
* front-down postures to 30° pitch and 30 mm lower are inside the joint ranges at 0.10–0.15 N·m static
  leg torque (0.20 with the head down); the chest cannot reach the floor (the knees would first), so
  "lying low" = the deepest bow held with the beak near the floor.

**Networks and entries** (training value = deployment value; scale/gain pinned; warm-started from v5
velstand, so the pinned action scale is velstand's walking 0.9):

=========  =================  ===============  ==================================================
network    entry              command          robotd window
=========  =================  ===============  ==================================================
bskstep    turn_left          (0.5, 0, +1)     chain, 1.0 s re-requested; released by skill_abort
bskstep    turn_right         (0.5, 0, −1)     same
bskstep    march              (0.5, 0, 0)      same (stepping in place, heading held)
bskmat     step_up            (0.5, 0, 0)      one-shot 2.5 s + 0.5 s unwind (0, 0, 0)
bskmat     step_down          (0.3, 0, 0)      same
bskbow     bow                (0.2, 0, 0)      one-shot 1.2 s + 0.8 s unwind (0, 0, 0)
bskbow     lie_low            (0, 0.2, 0)      chain, 1.0 s re-requested; ended by bow_release
bskbow     bow_release        (0, 0, 0)        chain 0.8 s: the network's own rise
=========  =================  ===============  ==================================================

The stepping codes make velstand (the warm start) step vigorously (vx 0.5); the reward removes the
travel. The bow codes sit in velstand's standing dead zone (the warm start keeps the feet planted).
robotd allows 30 skills; v5 built-ins 3 + Stage 1b's live table 20 = 23 → 7 free (8 without the unused
roulade) — the hop-over and B tables must share them (see :data:`TABLE_BUDGET`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STAGE = "skillsB-2026-09-28"
NETS = ("bskstep", "bskmat", "bskbow")
TASK_IDS = {
    "bskstep": "Mjlab-DuckStepInPlace-Flat-MicroDuck",
    "bskmat": "Mjlab-DuckMatStep-Mat-MicroDuck",
    "bskbow": "Mjlab-DuckBow-Flat-MicroDuck",
}
PLACEHOLDER = {n: f"/skilltrain/{n}.onnx" for n in NETS}
ACTION_SCALE = 0.9  # velstand's walking scale (warm start = v5 behaviour at iteration 0)
GAIN_RATIO = 1.0
SERVO_DELAY_STEPS = (0, 8)  # 0–40 ms at 5 ms physics steps (Stage 1b; the orchestrator's rule)
DELAY_UPDATE_STEPS = (
    100  # redrawn every 0.5 s (staggered): fixed short and fixed long delays both seen
)
MAT_HEIGHT_RANGE = (0.02, 0.05)  # the curriculum's span (row 0 → last row)
MAT_ROWS = 7
MAT_DEPTH_M = 0.40
MAT_START_D = (0.07, 0.14)  # trunk centre → the edge at the skill's first tick (toe 3–10 cm off)
MAT_AIM_DEG = 8.0
SOFT_SHARE = 0.4  # tiles with a compliant top (7 mm sink under the standing duck, bsk_probe)

TABLE_BUDGET = {
    "robotd_max": 30,
    "v5_builtins": 3,
    "stage1b_live": 20,
    "free": 7,
    "free_without_roulade": 8,
    "b_minimum": ("turn_left", "turn_right", "march", "step_up", "step_down", "bow"),
    "b_optional": ("lie_low", "bow_release"),
    "hop_plan": "hop_over_1..4 (4 entries, hop_contract)",
}


@dataclass(frozen=True)
class Entry:
    name: str
    net: str
    behaviour: str  # turn | march | step_up | step_down | bow | lie | release
    command: tuple[float, float, float]
    unwind: tuple[float, float, float]
    duration: float
    unwind_s: float
    chain: bool
    postures: tuple[str, ...]
    level: float = 1.0  # turn: +1 left / −1 right; march 0
    note: str = ""


ZERO3 = (0.0, 0.0, 0.0)
ENTRIES: tuple[Entry, ...] = (
    Entry("turn_left", "bskstep", "turn", (0.5, 0.0, 1.0), ZERO3, 1.0, 0.0, True, ("stand", "walk"), 1.0,
          "turn on the spot to the left while re-requested (cornered only); skill_abort hands back"),
    Entry("turn_right", "bskstep", "turn", (0.5, 0.0, -1.0), ZERO3, 1.0, 0.0, True, ("stand", "walk"), -1.0,
          "turn on the spot to the right"),
    Entry("march", "bskstep", "march", (0.5, 0.0, 0.0), ZERO3, 1.0, 0.0, True, ("stand", "walk"), 0.0,
          "march / stomp in place while re-requested (dance, excitement, impatience)"),
    Entry("step_up", "bskmat", "step_up", (0.5, 0.0, 0.0), ZERO3, 2.5, 0.5, False, ("stand",), 1.0,
          "step up onto a 2–5 cm mat whose edge is 3–10 cm ahead of the toes"),
    Entry("step_down", "bskmat", "step_down", (0.3, 0.0, 0.0), ZERO3, 2.5, 0.5, False, ("stand",), 1.0,
          "step down off a 2–5 cm mat whose edge is 3–10 cm ahead of the toes"),
    Entry("bow", "bskbow", "bow", (0.2, 0.0, 0.0), ZERO3, 1.2, 0.8, False, ("stand",), 1.0,
          "play-bow: front down 25°, head low looking ahead, held 1.2 s, back up"),
    Entry("lie_low", "bskbow", "lie", (0.0, 0.2, 0.0), ZERO3, 1.0, 0.0, True, ("stand",), 1.0,
          "lie low front-down (30°, 30 mm, beak near the floor) while re-requested"),
    Entry("bow_release", "bskbow", "release", ZERO3, ZERO3, 0.8, 0.0, True, ("stand",), 0.0,
          "ends lie_low: the network's own rise to standing"),
)  # fmt: skip
BEHAVIOURS = ("turn", "march", "step_up", "step_down", "bow", "lie", "release")


def entries(net: str) -> tuple[Entry, ...]:
    if net not in NETS:
        raise ValueError(f"net {net!r}; use one of {NETS}")
    return tuple(e for e in ENTRIES if e.net == net)


# ── targets ─────────────────────────────────────────────────────────────────────────────────

HOME = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)  # fmt: skip
NECK_PITCH, HEAD_PITCH = 5, 6
STAND_Z = 0.115

# Front-down postures (bsk_body.bow: both soles flat and in place, the centre of mass over the soles'
# centre, trunk slid back; least squares on robot_allcollisions.xml, 2026-09-28): the left leg's
# (hip_pitch, knee, ankle) absolute; the right leg mirrors (negated).
BOW_LEGS = {  # (pitch_deg, drop_mm) → left (hip_pitch, knee, ankle)
    (20, 10): (-0.837, 0.389, 0.879),
    (20, 20): (-0.694, 0.724, 1.070),
    (30, 20): (-1.110, 0.602, 1.191),
    (30, 30): (-1.039, 0.888, 1.406),
}


@dataclass(frozen=True)
class Posture:
    pitch_deg: float  # trunk nose-down
    drop_m: float  # trunk origin below STAND_Z
    legs: tuple[float, float, float]  # left (hip_pitch, knee, ankle)
    neck: float  # robot.head-style offsets from HOME
    head: float
    ramp_s: float

    def pose(self) -> tuple[float, ...]:
        q = list(HOME)
        hp, kn, an = self.legs
        q[2], q[3], q[4] = hp, kn, an
        q[11], q[12], q[13] = -hp, -kn, -an
        q[NECK_PITCH] += self.neck
        q[HEAD_PITCH] += self.head
        return tuple(q)


def _mid(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))


# The postures (the joint angles the rewards track, measured angles — not servo targets):
# * BOW: re-solved WITH its head pose in the balance (the head is 38 % of the mass) and the centre of mass 6 mm
#   behind the soles' centre (bsk_body, 2026-09-28 20:45);
# * LIE: the upstream ground pick's own lowest pose, which v5's ground-pick network holds balanced (bsk_probe,
#   12 ducks, both servo models, 2026-09-28 20:50: trunk 33° nose-down, 25 mm lower, hip −1.345 / knee 0.554 /
#   ankle 1.374, neck Δ −1.38…−1.46 (mjcf) / −1.19…−1.45 (bam), head Δ −0.18…−0.48 → the beak 19–40 mm off the
#   floor), with a slightly less extreme neck.
# Open loop, these angles as servo targets tip the duck forward (bsk_pose, 20/20): under load the servos sag
# ≈ 0.4 rad at the hip (the ground pick commands hip −0.91 to hold −1.35) — balance needs the network's feedback.
BOW = Posture(25.0, 0.015, (-1.159, 0.401, 1.126), -0.25, -0.51, 0.35)
LIE = Posture(33.0, 0.025, (-1.345, 0.554, 1.374), -1.30, -0.50, 0.6)

TURN_RATE = 0.9  # rad/s (52 °/s: a 90° turn in ≈ 1.8 s); the gate asks ≥ 30 °/s
TURN_RAMP_S = 0.4
STEP_AIR_S = (0.08, 0.45)  # a real step: a foot airborne this long before its touchdown
STEP_WINDOW_S = 0.9  # rev b1: "stepping" = both feet made a real step within this window
CADENCE_S = (
    0.7  # rev b1: a foot without a real step for longer than this costs (≥ 1.4 steps/s per foot)
)
LIFT_TARGET_M = {"turn": 0.018, "march": 0.030}  # swing sole height the clearance reward aims at
STAY_FREE_M = 0.02  # travel from the entry point allowed for free
TORQUE_FREE_NM = 0.33  # Σ |τ| above this is a cost (the plan's p95 ≤ 0.35 strain gate)
IMPACT_BW = 2.5  # touchdown vertical force above this × body weight is a cost (a stomp, not a slam)
MASS_KG = 0.7372

# ── episodes ───────────────────────────────────────────────────────────────────────────────

ENTRY_KINDS = ("stand", "walk", "coast", "head", "seated", "risen", "crouched")  # rl_contract's
ENTRY_MIX = {
    "turn": {"stand": 0.55, "walk": 0.25, "coast": 0.10, "head": 0.10},
    "march": {"stand": 0.55, "walk": 0.25, "coast": 0.10, "head": 0.10},
    "step_up": {"stand": 1.0},
    "step_down": {"stand": 1.0},
    "bow": {"stand": 0.7, "head": 0.2, "risen": 0.1},
    "lie": {"stand": 0.7, "head": 0.2, "risen": 0.1},
    "release": {"stand": 1.0},
}
HOLD_RANGE_S = {"turn": (1.0, 5.0), "march": (1.0, 6.0), "lie": (1.0, 6.0)}  # held (chain) entries
HOLD_JITTER = (0.85, 1.25)
VARIANT_WEIGHTS = {
    "bskstep": {"turn_left": 0.35, "turn_right": 0.35, "march": 0.30},
    "bskmat": {"step_up": 0.6, "step_down": 0.4},
    "bskbow": {"bow": 0.55, "lie_low": 0.45},
}


def episode_s(net: str) -> float:
    es = entries(net)
    held = max((HOLD_RANGE_S.get(e.behaviour, (0, e.duration))[1] for e in es), default=0.0)
    one = max(e.duration * HOLD_JITTER[1] + e.unwind_s for e in es)
    release = 0.8 if net == "bskbow" else 0.0
    return max(held + release, one) + 0.3


def entry_mix(behaviour: str) -> dict[str, float]:
    mix = ENTRY_MIX[behaviour]
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


# ── gates (the Mac gate and the page's "performed" detectors) ────────────────────────────────

GATES: dict[str, dict[str, Any]] = {
    "turn": {"yaw_deg_in_4s": 90.0, "rate_dps": 30.0, "excursion_m": 0.05, "steps_per_s": 1.5},
    "march": {"footfalls_per_s": 2.0, "lift_mm": 20.0, "travel_m_4s": 0.05, "yaw_deg_4s": 15.0},
    "step_up": {"on_top_by_s": 3.0, "tilt_max_deg": 30.0},
    "step_down": {"down_by_s": 3.0, "tilt_max_deg": 30.0},
    "bow": {"pitch_deg": 15.0, "beak_drop_mm": 40.0, "by_s": 0.6, "hold_s": 0.8, "footfalls": 1},
    "lie": {"pitch_deg": 20.0, "trunk_drop_mm": 20.0, "hold_s": 2.0, "footfalls": 1},
    "safety": {
        "fall_rate": 0.005,
        "torque_p95_nm": 0.35,
        "handback_s": 5.0,
        "delays_ms": (0, 20, 40),
    },
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def skill_defs(
    paths: dict[str, str] | None = None, nets: Sequence[str] = NETS
) -> list[dict[str, Any]]:
    """The B entries as ``table.SkillDef`` keyword dicts (robotd's fields; pinned scale/gain)."""
    out = []
    for e in ENTRIES:
        if e.net not in nets:
            continue
        out.append({
            "name": e.name, "path": (paths or PLACEHOLDER)[e.net], "duration": e.duration,
            "command": e.command, "unwind": e.unwind, "unwind_s": e.unwind_s, "chain": e.chain,
            "action_scale": ACTION_SCALE, "gain_ratio": GAIN_RATIO,
        })  # fmt: skip
    return out


__all__ = [
    "BEHAVIOURS",
    "BOW",
    "ENTRIES",
    "GATES",
    "LIE",
    "NETS",
    "Posture",
    "TABLE_BUDGET",
    "entries",
    "entry_mix",
    "episode_s",
    "skill_defs",
    "wilson",
]
