"""Stage 2 of the reopened motion training, in numbers (s2:build, 2026-09-28).

Owner, 2026-09-28 ~19:30 (「這些是下一階段？ 可以開始做了嗎」): start Stage 2 now — the walk network
fine-tuned so the neural duck can **sway, shift its weight and lean while standing** (training plan
§3.3, R3: robot.pose z/roll/pitch at zero twist, plus a fine standing head) and **keep its gaze
steady while walking** (§3.4, R4: the head's world orientation held against the stride). Rules
the numbers below carry: 「我要的是100%完整呈現」 and 「一律放寬，即便跌倒才是真實模擬」 (falls are
reported as outcomes, never hidden); alpha_stand stays OFF (「先維持」) — the sway must come from the
walk network itself, with no stand-network handover (the cause of the A100 study's stop-go near-falls
0.15 % → 2.05 %); robot.pose may be used by the neural duck (plan D4); only the neural demo loads
our walk network (蛋黃 keeps the official v5).

Pure standard library: the main venv's tests import it and every ``s2_*`` module reads its numbers
here, so a number is written once.

**The walk slot through robotd** (``control.rs`` / ``main.rs``, read-only upstream): the walk network
runs at action scale 0.9 × voltage adaptation, gain ratio 1.0 (v5 has no stand network, so robotd
never applies "standing tuning" to it), robotd's low-pass (head 0.5, legs 0.7) — the same action path
as Stage 1 (:mod:`duckmind.skilltrain.rl_action`). The command block it sees: the twist smoothed at
``cmd_alpha`` 0.2 and **zeroed while pose mode is held**; the head smoothed at ``head_alpha`` (the
neural demo runs 1.0); the body block ``[x, y, z, roll, pitch, yaw]`` = ``[0, 0, z, roll, pitch, 0]``
where z/roll/pitch are ``robot.pose``'s target slewed at ``cmd_alpha`` while pose mode is held and
**snapped to 0** when it is released (``main.rs``: ``if snapshot.pose.active {slew} else {body_ema =
0}``; ``duck-control`` ``obs.rs``: slots 57/58/59). Sign conventions (MuJoCo, the A100 reader's
formulas): roll + = right side down; pitch + = **nose down** (a chest-out is a negative pitch);
z + = up.

**The v5 stance** the pose targets are relative to (the Stage-1 A100 entry library: 55,205 standing
states of v5 under the recipe's final DR, 2026-09-28): trunk 116.2 mm (p5–p95 115.4–116.9), roll
−0.07°, pitch +0.71° (nose down; p5–p95 −1.56…+2.71°); the neck droops 0.117 rad at zero command.
The head camera's optical axis is level at HOME with a level trunk, and its pitch moves by
``Δneck_pitch − Δhead_pitch − trunk pitch`` (radian for radian; computed on the official MJCF).

**Servo command delay** (the Stage-1 RUNBOOK lesson): the owner's simulator has none, training's
recipe has 15–30 ms per physics step; the owner has not decided (「先量」). The fine-tune therefore
draws a per-episode delay of 0–40 ms (whole 5-ms physics steps) held for the episode, and a quarter
of the episodes keep the recipe's per-tick 15–30 ms jitter, so the network works whatever is chosen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.skilltrain import rl_contract as C1

STAGE = "stage2-2026-09-28"
MICRODUCK_RL_COMMIT = C1.MICRODUCK_RL_COMMIT
MJLAB_VERSION = C1.MJLAB_VERSION
V5_SHA256 = C1.V5_SHA256
STAGES = ("pose", "gaze")  # R3, R4
TASK_IDS = {
    "pose": "Mjlab-DuckWalkPose-Flat-MicroDuck",
    "gaze": "Mjlab-DuckWalkGaze-Flat-MicroDuck",
}
PLAN_ITERATIONS = {"pose": 3000, "gaze": 3500}  # plan §3.3/§3.4 full-run budgets (from scratch-ish)
# A fine-tune of v5 needs far fewer: the walk exists at iteration 0 (upstream's velstand warm start
# collapsed every curriculum for the same reason). S0 re-estimates; these fit the 70-unit cap.
DEFAULT_ITERATIONS = {"pose": 1500, "gaze": 1500}
WALK_FILE = "velstand_duck.onnx"
STEPS_PER_ENV = 24  # PPO rollout length (the recipe's num_steps_per_env)

# ── the robot and robotd (shared with Stage 1) ──────────────────────────────────────────────
CONTROL_HZ, DT = C1.CONTROL_HZ, C1.DT
PHYSICS_DT = 0.005
OBS_LEN, ACTION_LEN = C1.OBS_LEN, C1.ACTION_LEN
JOINTS, HOME = C1.JOINTS, C1.HOME
HEAD_IDX, LEG_IDX = C1.HEAD_IDX, C1.LEG_IDX
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = (
    C1.NECK_PITCH,
    C1.HEAD_PITCH,
    C1.HEAD_YAW,
    C1.HEAD_ROLL,
)
OBS_SLICES = C1.OBS_SLICES
BODY_Z_SLOT, BODY_ROLL_SLOT, BODY_PITCH_SLOT = 57, 58, 59  # duck-control obs.rs (z, roll, pitch)
BODY_UNBOUND_SLOTS = (55, 56, 60)  # x, y, yaw: always zero on the robot
WALK_ACTION_SCALE = C1.WALK_ACTION_SCALE  # 0.9
GAIN_RATIO = 1.0
CMD_ALPHA = 0.2
HEAD_ALPHA_DEMO = 1.0  # the neural demo's robotd (fid:motor, owner 「加快」)
HEAD_ALPHA_MIX = ((1.0, 0.8), (0.2, 0.2))  # training draw per episode: robustness to both robotds
STANDING_THRESHOLD = 0.05  # robotd DEFAULT_STANDING_THRESHOLD (|twist|)
FALL_TILT_DEG = C1.FALL_TILT_DEG  # robotd: tilt > 60° for 0.2 s
NEAR_FALL_TILT_DEG = 45.0  # the A100 study's near-fall (every fall crossed it first)

# ── the v5 stance (see the module doc) ──────────────────────────────────────────────────────
NOMINAL_Z = 0.1162
NOMINAL_ROLL = 0.0
NOMINAL_PITCH = math.radians(0.71)
V5_NECK_DROOP = -0.117  # rad, standing at zero head command (A100 library p50)


# ── robot.pose: the envelope duck-body keeps (plan §2 change 5; D4) ─────────────────────────
@dataclass(frozen=True)
class PoseEnvelope:
    z: tuple[float, float] = (-0.025, 0.008)  # m below / above the stance (HOME reach is +10 mm)
    angle: float = 0.20  # rad, |roll| and |pitch|
    rate_rad_s: float = 0.5  # the client's target slew limit
    rate_m_s: float = 0.060
    enter_tilt_deg: float = 20.0  # pose mode only when upright
    abort_tilt_deg: float = 30.0  # released at once past this (duck-body's guard)
    settle_s: float = 1.0  # after a walk, the twist at zero this long before pose mode (duck-body)

    def clamp(self, z: float, roll: float, pitch: float) -> tuple[float, float, float]:
        a = self.angle
        return (
            min(max(z, self.z[0]), self.z[1]),
            min(max(roll, -a), a),
            min(max(pitch, -a), a),
        )


POSE = PoseEnvelope()


# ── what a training episode asks for (the command program; plan §3.3 / §3.4) ───────────────
# Per-episode program kinds. stand = zero twist, head quiet/lively/fine; pose = standing with
# robot.pose segments; stopgo = walk → stop → (0–1.5 s) → pose → release → walk (the alpha_stand
# failure mode, now with one network); walk_v5 = the recipe's own velocity-command distribution
# (the gait anchor); walk_neuro = the neural duck's walking (OU speed, arcs, bursts and pauses,
# the owner's 0.70 flee bursts, head saccades inside the A100 envelope).
PROGRAMS = ("stand", "pose", "stopgo", "walk_v5", "walk_neuro")
PROGRAM_MIX = {
    # R3: plan "30 % plain stand, 20 % stop-go" + pose holds/sways; the walk keeps v5's share
    "pose": {"stand": 0.25, "pose": 0.30, "stopgo": 0.20, "walk_v5": 0.15, "walk_neuro": 0.10},
    # R4: the neural duck's real walking; R3's standing/pose data kept so it is not forgotten
    "gaze": {"stand": 0.10, "pose": 0.10, "stopgo": 0.15, "walk_v5": 0.20, "walk_neuro": 0.45},
}
EARLY_POSE_SCALE = 0.5  # the pose envelope's share until the range curriculum opens it fully
POSE_SEGMENT_S = (2.0, 6.0)
STAND_GAP_S = (0.5, 2.0)
STOP_WAIT_S = (0.0, 1.5)  # after the stop command, before pose mode (0 = while still coasting)
RESUME_WAIT_S = (0.0, 1.0)
WALK_SEGMENT_S = (2.0, 5.0)
RELEASE_RAMP_PROB = 0.7  # duck-body ramps the pose to 0 first; the rest snap (robotd's B-button)
RELEASE_RAMP_S = (0.3, 1.0)
POSE_AXIS_PROB = {"z": 0.5, "roll": 0.6, "pitch": 0.6}  # each axis active in a segment
POSE_ZERO_PROB = 0.15  # pose mode held at the stance (the nominal "idle" command)
POSE_MODES = (("step", 0.4), ("sine", 0.35), ("ou", 0.25))
POSE_SINE_HZ = (0.05, 0.5)
POSE_OU_TAU_S = (0.5, 3.0)
POSE_STEP_HOLD_S = (0.8, 4.0)

# The head. v5's own command distribution (velocity recipe final ranges, resampled 2–5 s) keeps its
# tracking; the standing programs add the plan's fine head (neck pull-back, ±3° micro-motion); the
# walking programs stay inside the A100 walking envelope (2026-09-28: 0.07 % falls, like HOME).
HEAD_V5_RANGES = (1.10, 1.10, 1.40, 0.31)
HEAD_V5_RESAMPLE_S = (2.0, 5.0)
HEAD_PULLBACK = {"neck": (0.20, 0.35), "head_per_neck": -2.0 / 3.0, "hold_s": (0.8, 3.0)}
HEAD_MICRO = {"amp_rad": (0.03, 0.07), "hz": (0.5, 2.0)}  # ±3° is 0.052 rad
WALK_HEAD = {
    "yaw_max": 1.0,
    "look_up": (0.0, -0.35, 0.0, 0.0),
    "half_down": (-0.125, 0.175, 0.0, 0.0),
    "scan_amp": 0.4,
    "scan_hz": (0.3, 1.0),
    "sacc_amp": (0.3, 0.8),
    "sacc_dwell_s": (0.8, 3.0),
}
HEAD_MIX_STAND = (
    ("home", 0.25),
    ("v5", 0.25),
    ("pullback", 0.2),
    ("micro", 0.15),
    ("lively", 0.15),
)
HEAD_MIX_WALK = (("home", 0.30), ("static", 0.20), ("sacc", 0.20), ("scan", 0.15), ("v5", 0.15))

# Walking. v5's distribution (velocity recipe, final stage) and the neural duck's family.
V5_TWIST = {
    "vx": (-0.4, 0.4), "vy": (-0.3, 0.3), "wz": (-1.0, 1.0), "resample_s": (3.0, 8.0),
    "standing": 0.25, "turn_in_place": 0.15, "turn_in_place_wz": (0.4, 1.0),
}  # fmt: skip
NEURO_WALK = {
    "vx": (0.35, 0.55), "vx_tau_s": (1.0, 3.0), "vx_sd": 0.07, "wz_sd": 0.5, "wz_tau_s": 1.0,
    "wz_max": 1.0, "arc_vx_max": 0.40, "arc_wz": 0.3, "burst_s": (0.4, 6.0), "pause_s": (0.2, 2.0),
    "flee_vx": 0.70, "flee_prob": 0.05, "flee_s": (0.6, 2.0),
}  # fmt: skip

# ── domain randomisation beyond the recipe's final stage ────────────────────────────────────
DELAY_STEPS = (0, 8)  # 0–40 ms, per episode, held (whole 5-ms physics steps)
DELAY_JITTER_PROB = 0.25
DELAY_JITTER_STEPS = (3, 6)  # the recipe's 15–30 ms, redrawn every control tick
FOOT_MU = (0.5, 1.3)  # plan: "floor μ down to 0.5 (a weight shift on slippery tile)"
HEAD_MASS_SCALE = (0.95, 1.05)  # R4: head mass ±5 % on top of head CoM ±10 mm
GYRO_NOISE_SCALE = 1.5  # R4: gyro noise × 1.5
IMU_EXTRA_LAG_TICKS = (0, 1)  # R4: 0–20 ms extra IMU latency

# ── gaze (R4) ───────────────────────────────────────────────────────────────────────────────
CAM_FLOOR_DPS = 10.0  # only the escapable part of the camera's ⊥ rate is charged
HEADING_RATE_TAU_S = 0.4  # the slow heading frame: low-passed yaw rate …
HEADING_TAU_S = 0.8  # … plus a correction toward the trunk's yaw (zero lag in a steady arc)
GAZE_YAW_STD = 0.15  # rad
WALKING_TWIST = 0.1  # |twist| above this counts as walking (the BC anchor's gate too)


# ── behaviour cloning (plan §3.3: dual teachers; §3.4: legs only) ───────────────────────────
@dataclass(frozen=True)
class BcSpec:
    anchor: str  # "v5" or a path to the start network's ONNX
    anchor_dims: tuple[int, ...]  # action dims the anchor constrains
    anchor_coef: tuple[tuple[int, float], ...]  # (iteration, coef) stages
    anchor_tilt_deg: float
    anchor_min_twist: float
    fallen_coef: float
    fallen_tilt_deg: float
    lr: float = 3e-4
    epochs: int = 5
    mini_batches: int = 4
    min_mini_batch: int = 512
    min_samples: int = 64

    def coef_at(self, iteration: int) -> float:
        value = self.anchor_coef[0][1]
        for it, c in self.anchor_coef:
            if iteration >= it:
                value = c
        return value


ALL_DIMS = tuple(range(ACTION_LEN))
BC = {
    # the walk anchor on upright walking frames (all 14 dims), v5 rights the fallen frames;
    # standing and pose frames are left to PPO
    "pose": BcSpec("v5", ALL_DIMS, ((0, 1.0),), 25.0, WALKING_TWIST, 1.0, 35.0),
    # the ten leg dims anchored to the start network, 1.0 → 0.3 over 1,500 it; the neck is free
    "gaze": BcSpec("start", LEG_IDX, ((0, 1.0), (500, 0.6), (1000, 0.45), (1500, 0.3)), 25.0,
                   WALKING_TWIST, 1.0, 35.0),
}  # fmt: skip
CRITIC_WARMUP_ITERS = 100  # plan: a critic-only warm-up before the actor moves
INIT_STD = 0.25
LEARNING_RATE = 3e-4  # then adaptive (desired KL 0.01), as the recipe
# The imported normaliser's body slots were fitted to ±5 mm / ±0.05 rad (div 0.0129 / 0.0389): the
# pose commands would reach 5σ. They are re-scaled once at import (first-layer columns scaled to
# match, so the network's function is unchanged — parity-checked) to these stds.
RECONDITION_STD = {BODY_Z_SLOT: 0.010, BODY_ROLL_SLOT: 0.10, BODY_PITCH_SLOT: 0.10}


# ── gates (plan §6; pre-registered; every rate k/n with Wilson 95 %) ────────────────────────
@dataclass(frozen=True)
class Gate:
    key: str
    kind: str  # min | max | rate_max | rate_min | ratio_max | within | max_vs_v5 | report
    threshold: float | None
    v5: Any
    zh: str
    source: str = ""
    stage: str = "both"  # pose | gaze | both
    note: str = ""


# v5 baselines, A100 2026-09-28 (the run's summary.md: measurement records, not in the repo), n=512
# per condition unless stated; the Mac baselines are measured by this package's harness.
V5 = {
    "cam_perp_dps": {"0.40": 56.9, "0.55": 74.6},
    # the Mac P0 (n = 30, 2026-09-28): v5 head at HOME command → the head joints held at HOME by the
    # servo targets ("fixed") → the best oracle counter-rotation; camera ⊥ RMS p50 °/s
    "p0_cam_perp_dps": {
        "bam 0.40": (61.8, 30.2, 33.9),
        "bam 0.55": (78.9, 40.6, 52.5),
        "mjcf 0.40": (80.7, 44.9, 39.2),
        "mjcf 0.55": (110.8, 86.5, 65.2),
    },
    "path_speed_mps": {"0.40": 0.178, "0.55": 0.240},
    "walk_falls": (10, 17407),
    "stopgo_near_falls": (3, 2048),
    "push_righting_8s": (0.77, 0.84),
    "sitstand_falls": (15, 1024),
    "stop_s_p50_055": 0.62,
    "yaw_gain": (0.40, 0.42),
    "stand_leg_rms_nm": 0.047,
    "stand_stepped_quiet": (5, 512),
    "pose_gain": {"roll+0.2": 0.003, "pitch+0.2": 0.073, "z-0.02": 0.0},
    "neck_pullback_gain": 0.00,
    "yaw_micro_gain": 0.53,
    "head_motion_falls": (8, 11263),
}

GATES: tuple[Gate, ...] = (
    # ── visibility, R3 (standing) ──
    Gate("lean_roll_deg", "min", 7.0, V5["pose_gain"]["roll+0.2"], "站著左右傾：指令 ±0.2 rad 要真的傾 ≥ 7°",
         "plan §6 sway/lean", "pose"),
    Gate("lean_pitch_deg", "min", 7.0, V5["pose_gain"]["pitch+0.2"], "站著前後傾：指令 ±0.2 rad 要真的傾 ≥ 7°",
         "plan §6 sway/lean", "pose"),
    Gate("sway_amp_deg", "min", 7.0, None, "左右／前後搖擺 0.1–0.5 Hz：擺幅 ≥ ±7°", "plan §6", "pose"),
    Gate("sway_lag_ms", "max", 300.0, None, "搖擺跟得上：落後 ≤ 0.3 秒", "plan §6", "pose"),
    Gate("pose_stepped", "rate_max", 0.01, None, "傾身或搖擺時不踏步（20 秒內踏步 ≤ 1 %）", "plan §6", "pose"),
    Gate("release_rate_dps", "max", 45.0, (79.0, 93.0), "放開時彈回不能太猛（≤ 45°/秒；alpha_stand 79–93）",
         "plan §6", "pose"),
    Gate("weight_shift_ratio", "min", 0.65, None, "重心轉移：左傾 0.15 rad 時一腳承重 ≥ 65 %", "plan §6", "pose"),
    Gate("chest_out_deg", "min", 6.0, None, "抬頭挺胸：pitch −0.15 → 真的往後仰 ≥ 6°", "plan §6", "pose"),
    Gate("crouch_mm", "min", 12.0, 0.0, "蹲低：指令 −20 mm → 真的蹲 ≥ 12 mm（今天 0）", "plan §6", "pose"),
    Gate("neck_pullback_gain", "min", 0.6, V5["neck_pullback_gain"], "站著縮脖子：做到指令的 ≥ 6 成（今天 0）",
         "plan §6 standing head", "pose"),
    Gate("yaw_micro_gain", "min", 0.6, V5["yaw_micro_gain"], "站著頭的小動作（±3°）：做到 ≥ 6 成",
         "plan §6 standing head", "pose"),
    Gate("stand_leg_rms_ratio", "ratio_max", 1.10, V5["stand_leg_rms_nm"], "站著腳的出力 ≤ 今天 +10 %",
         "plan §6", "pose"),
    Gate("pose_falls", "rate_max", 0.002, None, "傾身／搖擺時跌倒 ≤ 0.2 %（跌倒照實記下）", "plan §6 common safety",
         "pose"),
    # ── visibility, R4 (walking) ──
    Gate("cam_perp_ratio", "ratio_max", 0.60, V5["cam_perp_dps"], "走路時鏡頭晃動 ≤ 今天的 6 成（0.40/0.55）",
         "plan §6 gaze-stable, lowered from 0.50 by the P0 probe (plan §3.4 'measure first')", "gaze",
         note="P0 (Mac, n = 30 per cell, 2026-09-28): the best any head control reached was 0.49–0.51 of v5 "
              "on BAM and 0.49–0.59 on the XML servos; 0.50 stays the stretch target"),
    Gate("head_yaw_stride_sd_ratio", "ratio_max", 0.50, None, "走路時頭的左右擺（步伐頻率）≤ 今天的一半",
         "plan §6 gaze-stable", "gaze"),
    Gate("head_yaw_delivered", "min", 0.85, 0.89, "走路時轉頭 ±1.0 仍做到 ≥ 85 %", "plan §6 gaze-stable", "gaze"),
    Gate("neck_travel_rad_min", "report", None, None, "每分鐘走路脖子動多少（磨損，只記錄）", "plan §6", "gaze"),
    # ── regression against v5 (both stages) ──
    Gate("walk_falls", "rate_max", 0.0010, V5["walk_falls"], "走路跌倒 ≤ 0.10 %（今天 0.057 %）", "plan §6 V2"),
    Gate("stopgo_near_falls", "rate_max", 0.003, V5["stopgo_near_falls"], "走走停停差點跌倒 ≤ 0.3 %（今天 0.15 %）",
         "plan §6 V2"),
    Gate("push_righting_8s", "rate_min", 0.75, V5["push_righting_8s"], "被推倒後 8 秒內自己爬起來 ≥ 75 %", "plan §6 V2"),
    Gate("righting_p50_extra_s", "max", 0.3, None, "爬起來的時間 ≤ 今天 +0.3 秒", "plan §6 V2"),
    Gate("sitstand_falls", "rate_max", 0.018, V5["sitstand_falls"], "坐下站起跌倒 ≤ 1.8 %（今天 1.46 %）", "plan §6 V2"),
    Gate("walk_speed_ratio", "within", 0.10, V5["path_speed_mps"], "走路速度和今天差 ≤ ±10 %", "plan §6 V2"),
    Gate("turn_gain_ratio", "within", 0.15, V5["yaw_gain"], "轉彎的比例和今天差 ≤ ±15 %", "plan §6 V2"),
    Gate("stop_s_p50", "max", 0.75, V5["stop_s_p50_055"], "從走路停下 ≤ 0.75 秒", "plan §6 V2"),
    Gate("head_motion_falls", "max_vs_v5", None, V5["head_motion_falls"], "邊走邊動頭不多跌倒（和今天比）",
         "plan §6 V2 (0 excess falls: the candidate's Wilson low end may not pass v5's high end)"),
    Gate("stand_stepped", "max_vs_v5", None, V5["stand_stepped_quiet"], "站著不動時不亂踏步 ≤ 今天（5/512）",
         "plan §6 V2"),
    Gate("walkoffs", "max_vs_v5", None, None, "站著不動時自己走掉（> 40 cm）≤ 今天", "plan §6 V2",
         note="v5 is measured in the same battery session (the A100 run kept no per-duck rows)"),
    Gate("cam_perp_regress_ratio", "ratio_max", 1.10, V5["cam_perp_dps"], "走路鏡頭晃動不能比今天多 10 % 以上",
         "s2 (R3 must not shake the head more)", "pose"),
)  # fmt: skip
GATE_BY_KEY = {g.key: g for g in GATES}


def gates_for(stage: str) -> tuple[Gate, ...]:
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r}; use one of {STAGES}")
    if stage == "gaze":
        # R4 must keep R3's standing abilities too (it warm-starts from R3)
        return tuple(g for g in GATES if g.key != "cam_perp_regress_ratio")
    return tuple(g for g in GATES if g.stage in ("pose", "both"))


# ── statistics and verdicts ─────────────────────────────────────────────────────────────────
wilson = C1.wilson
rate = C1.rate


def rate_verdict(k: int, n: int, gate: float, *, at_most: bool = True) -> str:
    """PASS / FAIL / UNRESOLVED for "rate ≤ gate" (``at_most``) or "rate ≥ gate" at Wilson 95 %."""
    lo, hi = wilson(k, n)
    if at_most:
        return "PASS" if hi <= gate else ("FAIL" if lo > gate else "UNRESOLVED")
    return "PASS" if lo >= gate else ("FAIL" if hi < gate else "UNRESOLVED")


def n_for_zero(gate: float) -> int:
    """The smallest n at which 0/n has a Wilson upper bound ≤ ``gate``."""
    n = 1
    while wilson(0, n)[1] > gate:
        n += 1
    return n


def verdict(gate: Gate, value: Any, v5: Any = None) -> str:
    """One gate's verdict from its measured value (rates as (k, n)). ``report`` gates never fail."""
    kind = gate.kind
    if value is None:
        return "NOT MEASURED"
    if kind == "report":
        return "REPORT"
    if kind == "min":
        return "PASS" if value >= gate.threshold else "FAIL"
    if kind == "max":
        return "PASS" if value <= gate.threshold else "FAIL"
    if kind == "rate_max":
        k, n = value
        return rate_verdict(int(k), int(n), float(gate.threshold))
    if kind == "rate_min":
        k, n = value
        return rate_verdict(int(k), int(n), float(gate.threshold), at_most=False)
    if kind == "ratio_max":
        return "PASS" if value <= gate.threshold else "FAIL"
    if kind == "within":
        return "PASS" if abs(value - 1.0) <= gate.threshold else "FAIL"
    if kind == "max_vs_v5":
        base = v5 if v5 is not None else gate.v5
        if base is None:
            return "NO BASELINE"
        k, n = value
        kb, nb = base
        # the candidate's rate may not be clearly above v5's (its Wilson low end above v5's high end)
        return "FAIL" if wilson(int(k), int(n))[0] > wilson(int(kb), int(nb))[1] else "PASS"
    raise ValueError(f"gate kind {kind!r}")


# ── budget (plan §5; the Stage-1 calibration's measured speed) ──────────────────────────────
UNITS_PER_HOUR = 6.77
STAGE_CAP_UNITS = 70.0
STAGE1_S_PER_IT = {"startle_4096": 3.44, "comfort_4096": 3.75, "startle_8192": 5.52}
SESSION_OVERHEAD_MIN = 6.0  # setup 1.2 min + uploads + export + evaluation start-up
ATTEMPT_FACTOR = (1.3, 1.5, 2.0)  # low / likely / high: killed attempts + one full run


def units(minutes: float) -> float:
    return round(minutes / 60.0 * UNITS_PER_HOUR, 2)


def estimate(
    s_per_it: Mapping[str, float] | None = None,
    iterations: Mapping[str, int] | None = None,
    battery_min: float = 20.0,
    calibration_min: float = 50.0,  # S0-2 incl. the v5 control battery at j3-6/d0/d20/d40 (n = 512)
    sessions: int = 5,
) -> dict[str, Any]:
    """Stage-2 units: calibration + R3 and R4 attempts + two V2 batteries + session overhead.

    ``s_per_it``: seconds per PPO iteration per stage (default: Stage 1's measured 3.6 s at 4,096
    envs, +10 % for the VelStand sensors; S0 measures the real ones)."""
    sp = dict(s_per_it or {"pose": 4.0, "gaze": 4.2})
    its = dict(iterations or DEFAULT_ITERATIONS)
    out: dict[str, Any] = {"s_per_it": sp, "iterations": its}
    train_min = {k: its[k] * sp[k] / 60.0 for k in STAGES}
    rows = []
    for label, factor in zip(("low", "likely", "high"), ATTEMPT_FACTOR, strict=True):
        minutes = (
            calibration_min
            + factor * sum(train_min.values())
            + 2 * battery_min
            + sessions * SESSION_OVERHEAD_MIN
        )
        rows.append((label, round(minutes, 1), units(minutes)))
    out["rows"] = [{"case": a, "minutes": b, "units": c} for a, b, c in rows]
    out["one_attempt_units"] = {k: units(v) for k, v in train_min.items()}
    out["fits_cap"] = rows[1][2] <= STAGE_CAP_UNITS
    out["cap"] = STAGE_CAP_UNITS
    return out


def summary_of(values: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:.3g}" for k, v in values.items())


def _check() -> list[str]:
    """Internal consistency (the tests call it)."""
    problems = []
    for stage, mix in PROGRAM_MIX.items():
        if abs(sum(mix.values()) - 1.0) > 1e-9 or set(mix) != set(PROGRAMS):
            problems.append(f"PROGRAM_MIX[{stage}] does not cover PROGRAMS with total 1")
    for name, mix in (("HEAD_MIX_STAND", HEAD_MIX_STAND), ("HEAD_MIX_WALK", HEAD_MIX_WALK),
                      ("POSE_MODES", POSE_MODES)):  # fmt: skip
        if abs(sum(w for _, w in mix) - 1.0) > 1e-9:
            problems.append(f"{name} weights do not sum to 1")
    if not POSE.z[0] < 0 < POSE.z[1] or POSE.z[1] > C1.HOME_REACH_M:
        problems.append("the pose z envelope must straddle 0 and stay under the leg's reach")
    if len({BODY_Z_SLOT, BODY_ROLL_SLOT, BODY_PITCH_SLOT, *BODY_UNBOUND_SLOTS}) != 6:
        problems.append("body slots overlap")
    lo, hi = OBS_SLICES["body"]
    if not all(lo <= s < hi for s in (BODY_Z_SLOT, BODY_ROLL_SLOT, BODY_PITCH_SLOT)):
        problems.append("body slots outside the observation's body block")
    for stage in STAGES:
        spec = BC[stage]
        if any(d not in ALL_DIMS for d in spec.anchor_dims):
            problems.append(f"BC[{stage}] anchors a dim outside the 14 actions")
    return problems


__all__ = [
    "BC",
    "GATES",
    "HOME",
    "JOINTS",
    "NOMINAL_PITCH",
    "NOMINAL_Z",
    "POSE",
    "PROGRAMS",
    "PROGRAM_MIX",
    "STAGES",
    "TASK_IDS",
    "V5",
    "estimate",
    "gates_for",
    "rate",
    "verdict",
    "wilson",
]


def label(values: Sequence[float]) -> str:
    return "/".join(f"{v:g}" for v in values)
