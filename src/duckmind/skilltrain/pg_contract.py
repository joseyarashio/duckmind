"""The panic-gait fine-tune of the v5 walk network (2026-10-08), in numbers.

Owner goal (2026-10-04): when fleeing, the duck runs panicked at full speed; strolling stays
ordinary. The retrain step 0 (``training/retrain-2026-10-05/STEP0.md``) measured that v5's ~40 % speed
is the policy itself (tracking reward ``exp(-err²/0.1)`` over a command range of only ±0.4 m/s) and that
robotd's 0.9 scale costs 5–15 % more. So the fine-tune widens the speed command to 1.0, sharpens the
tracking, pays forward progress up to the command, and keeps the ordinary walk anchored to v5.

Pure standard library (the main venv's tests import it).
"""

from __future__ import annotations

from duckmind.skilltrain import s2_contract as S2

STAGE = "panic-2026-10-08"
NET = "panicwalk"
TASK_ID = "Mjlab-DuckPanicWalk-Flat-MicroDuck"
WALK_FILE = "velstand_panic.onnx"
STEPS_PER_ENV = S2.STEPS_PER_ENV
V5_SHA256 = S2.V5_SHA256
CONTROL_HZ, DT = S2.CONTROL_HZ, S2.DT
OBS_LEN, ACTION_LEN = S2.OBS_LEN, S2.ACTION_LEN
WALK_ACTION_SCALE = S2.WALK_ACTION_SCALE
MICRODUCK_RL_COMMIT = S2.MICRODUCK_RL_COMMIT
DELAY_STEPS = S2.DELAY_STEPS
FOOT_MU = S2.FOOT_MU
TWIST_SLOTS = (48, 51)  # vx, vy, vyaw in the actor observation
GRAVITY_SLOTS = (3, 6)
VX_SLOT = 48

# ── the command distribution ────────────────────────────────────────────────────────────────
# Upstream's own draw (vx ±0.4, vy ±0.3, yaw ±1, 25 % standing, 20 % forward-only, 15 % turn-in-place)
# is kept for ``P_ORDINARY``; ``P_MID`` re-draws vx in the ordinary strolling band the neural duck
# uses (0.35–0.70: its min_vx 0.35, max_vx 0.55, flee 0.70); ``P_FAST`` draws vx up to the panic range.
P_MID = 0.20
MID_VX = (0.35, 0.70)
P_FAST = 0.30
FAST_VX_MIN = 0.60
FAST_STRAIGHT_PROB = 0.5  # the other half: arcs, yaw uniform over ±YAW_MAX
FAST_STRAIGHT_YAW = 0.10
FAST_VY = 0.10
YAW_MAX = 1.0
VMAX_STAGES = ((0, 0.70), (150, 0.85), (350, 1.00))  # (iteration, the fast band's upper vx)
CMD_ALPHA = S2.CMD_ALPHA  # robotd's twist EMA

# ── the speed the reward asks for (``--target-map``) ─────────────────────────────────────────
# v5 walks at ~0.4 × the command and the gate asks ordinary strolling to stay within v5 −10 % … +60 % (chord
# at 0.35 / 0.45 / 0.55: 0.13 / 0.17 / 0.23 on BAM). The reward therefore asks for ``TARGET_K0 × cmd`` up to the
# knee and then rises to ``TARGET_TOP_VX`` at the command ``TARGET_TOP_CMD`` (the panic speed the gate needs
# plus margin): ordinary speeds are shaped, not anchored to v5.
TARGET_K0 = 0.50  # (g11 sets 0.42 / 0.85 through the TARGET_* overrides below)
TARGET_KNEE = 0.55
TARGET_TOP_CMD = 1.0
TARGET_TOP_VX = 0.70
TARGET_OVERRIDE: dict = {}  # filled by pg_train --target-k0 / --target-top (a run's own map; recorded in its metrics)

# ── the policy start ────────────────────────────────────────────────────────────────────────
INIT_STD = 0.30
LEARNING_RATE = 3e-4
CRITIC_WARMUP_ITERS = 100
RECONDITION_STD = {VX_SLOT: 0.45}  # v5's vx normaliser std is 0.21: vx 1.0 would be 4.8 σ
ANCHOR_MAX_VX = (
    0.42  # frames commanded at most this fast are cloned toward v5 (strolling stays ordinary)
)
BC = {
    "lr": 3e-4,
    "epochs": 5,
    "mini_batches": 4,
    "min_mini_batch": 512,
    "min_samples": 64,
    "fallen_tilt_deg": 35.0,
    "fallen_coef": 1.0,
    "anchor_tilt_deg": 25.0,
    "anchor_coef": ((0, 1.0),),
    "anchor_max_vx": ANCHOR_MAX_VX,
    "anchor_dims": tuple(range(ACTION_LEN)),
}

# ── pose reward widening for fast gaits (upstream: std_running for |twist| > 1.5, i.e. never) ──
STD_RUNNING = {".*hip_yaw.*": 0.35, ".*hip_roll.*": 0.08, ".*hip_pitch.*": 0.55,
               ".*knee.*": 0.55, ".*ankle.*": 0.35}  # fmt: skip
RUNNING_THRESHOLD = 0.65
AIR_TIME_MIN = 0.09  # s; upstream 0.125 (a faster cadence swings shorter)

# Reward revisions (recorded in every run's metrics.json): name → {term: {"weight": w, "params": {...}}};
# a term not in the dict keeps the base value. Base ("g1") values live in pg_tasks.BASE_TERMS.
REWARD_REVS = {
    "g1": {},
    # g2: sharper and stronger — the panic speed is paid harder
    "g2": {
        "track_vel_sharp": {"params": {"std": 0.12}},
        "track_vel_forward": {"weight": 5.0},
        "track_yaw_sharp": {"weight": 2.0, "params": {"std": 0.2}},
    },
    # g3: straight lines — v5's steady speed at command 1.0 is ~0.48 m/s along its path but its heading drifts
    # (the gate's 10-s net speed 0.32); the yaw rate is tracked much tighter (σ 0.12 rad/s, weight 3)
    "g3": {"track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12}}},
    "g4": {
        "track_vel_sharp": {"params": {"std": 0.12}},
        "track_vel_forward": {"weight": 5.0},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12}},
    },
    # g5: g4 with the target map (pg_tasks passes ``mapped``)
    "g5": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True}},
        "track_vel_forward": {"weight": 4.0, "params": {"mapped": True}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12}},
        "track_vel_coarse": {"params": {"mapped": True}},
    },
    # g6 / g7: g4 / g5 with the yaw tolerance widening with the yaw command (g4's arcs fell to 0.15 of the command)
    "g6": {
        "track_vel_sharp": {"params": {"std": 0.12}},
        "track_vel_forward": {"weight": 5.0},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
    },
    "g7": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True}},
        "track_vel_forward": {"weight": 4.0, "params": {"mapped": True}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
        "track_vel_coarse": {"params": {"mapped": True}},
    },
    # g8: g7 + heading hold (a reward on the accumulated heading error while commanded straight; the critic sees it)
    "g8": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True}},
        "track_vel_forward": {"weight": 4.0, "params": {"mapped": True}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
        "track_vel_coarse": {"params": {"mapped": True}},
        "heading_hold": {"weight": 2.0},
    },
    # g9: g8 + arcs may slow down (the forward goal falls by arc_k × |yaw command|): J's arcs lost yaw authority
    # (G2 0.33 / 0.32 against v5's 0.47 / 0.54) while its straight speed rose
    "g9": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True, "arc_k": 0.5}},
        "track_vel_forward": {"weight": 4.0, "params": {"mapped": True, "arc_k": 0.5}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
        "track_vel_coarse": {"params": {"mapped": True, "arc_k": 0.5}},
        "heading_hold": {"weight": 2.0},
    },
    # g10: g8 + yaw_mean (the yaw rate averaged over ~0.8 s against the command, σ 0.07 rad/s). Diagnosis from the
    # checkpoint table: path speeds are 0.55-0.60 m/s but the 10-s net speed 0.43 — heading drift of a median
    # 0.25 rad/s; the per-step yaw reward (σ 0.12) cannot see a bias under the stride's ±0.4 rad/s yaw oscillation.
    "g10": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True}},
        "track_vel_forward": {"weight": 4.0, "params": {"mapped": True}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
        "track_vel_coarse": {"params": {"mapped": True}},
        "heading_hold": {"weight": 2.0},
        "yaw_mean": {"weight": 4.0},
    },
    # g12: g10 + arcs may slow down (arc_k 0.5): N-249 straightens and speeds up but its arcs slip (G2 0.43 / 0.47)
    "g12": {
        "track_vel_sharp": {"params": {"std": 0.12, "mapped": True, "arc_k": 0.5}},
        "track_vel_forward": {"weight": 6.0, "params": {"mapped": True, "arc_k": 0.5}},
        "track_yaw_sharp": {"weight": 3.0, "params": {"std": 0.12, "widen": 0.4}},
        "track_vel_coarse": {"params": {"mapped": True, "arc_k": 0.5}},
        "heading_hold": {"weight": 2.0},
        "yaw_mean": {"weight": 4.0},
    },
}
DEFAULT_REWARD_REV = "g1"
# a run's reward weights can be changed mid-run through OVERRIDE.json (terms listed here only)
OVERRIDABLE = {
    "track_vel_sharp": ("std",),
    "track_vel_forward": (),
    "track_vel_coarse": ("std",),
    "track_yaw_sharp": ("std", "widen"),
    "heading_hold": ("sigma",),
    "yaw_mean": ("sigma",),
    "fall_event": (),
    "action_rate_l2": (),
    "servo_stall": (),
    "servo_acc_spike": (),
}
UNITS_PER_HOUR = S2.UNITS_PER_HOUR
