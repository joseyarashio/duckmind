"""The 2026-10-09 training queue, in numbers: expressive walk, quick turn, calm head, personality gaits.

Owner 2026-10-09 (「同意重訓，同步進行」): train the approved queue on the DGX Spark, in the order
(1) expressive walk (``ex``), (2) quick turn in place (``turn``), (4) head-stabilised walking (``calm``),
(3) personality gaits (``style``). Every item is a fine-tune of a network that already walks (v5 velstand, or the
panic-gait round's recipe): the pipeline of ``pg_*`` (BAM + XML servo mix, 0–40 ms servo delay, robotd's action
path and twist EMA, the v5 teacher as walk anchor and fallen expert) with a different command mixture and rewards.

Pure standard library: the main venv's tests import it.

**The command block.** obs[48:51] twist, obs[51:55] head (neck_pitch, head_pitch, head_yaw, head_roll: deltas from
HOME), obs[55:61] body ``[x, y, z, roll, pitch, yaw]``. robotd fills the body block as ``[0, 0, z, roll, pitch, 0]``
(``robot.pose``), so x / y / yaw are free carrier slots: item ``style`` writes its style code into x and y.
Sign: pitch + = nose down (a bow), z + = up.
"""

from __future__ import annotations

import math

from duckmind.skilltrain import pg_contract as P
from duckmind.skilltrain import s2_contract as S2

STAGE = "queue-2026-10-09"
ITEMS = ("ex", "turn", "calm", "style")
TASK_IDS = {
    "ex": "Mjlab-DuckExpressWalk-Flat-MicroDuck",
    "turn": "Mjlab-DuckQuickTurn-Flat-MicroDuck",
    "calm": "Mjlab-DuckCalmWalk-Flat-MicroDuck",
    "style": "Mjlab-DuckStyleWalk-Flat-MicroDuck",
}
WALK_FILE = {
    "ex": "velstand_express.onnx",
    "turn": "velstand_turn.onnx",
    "calm": "velstand_calm.onnx",
    "style": "velstand_style.onnx",
}
STEPS_PER_ENV = P.STEPS_PER_ENV
CONTROL_HZ, DT = P.CONTROL_HZ, P.DT
OBS_LEN, ACTION_LEN = P.OBS_LEN, P.ACTION_LEN
V5_SHA256 = P.V5_SHA256
MICRODUCK_RL_COMMIT = P.MICRODUCK_RL_COMMIT
WALK_ACTION_SCALE = P.WALK_ACTION_SCALE
FOOT_MU = P.FOOT_MU

# obs slots
TWIST_SLOTS = (48, 51)
HEAD_SLOTS = (51, 55)
BODY_SLOTS = (55, 61)
GRAVITY_SLOTS = (3, 6)
SLOT_X, SLOT_Y, SLOT_Z, SLOT_ROLL, SLOT_PITCH, SLOT_YAW = range(55, 61)
SLOT_TWIST_VX, SLOT_TWIST_VY, SLOT_TWIST_YAW = 48, 49, 50

# the v5 stance (S2 contract: 55,205 standing states of v5 under the recipe's final DR)
NOMINAL_Z = S2.NOMINAL_Z
NOMINAL_PITCH = S2.NOMINAL_PITCH
WALK_LIFT_M = 0.004  # v5's trunk is this much HIGHER walking than standing (measured: standing 116.2, walking 117-122 mm)
HEAD_IDX, LEG_IDX = S2.HEAD_IDX, S2.LEG_IDX
STANDING_THRESHOLD = S2.STANDING_THRESHOLD

# ── item 1: expressive walk ──────────────────────────────────────────────────────────────────
# The ranges the 14 emotions use (``neuro/emotions_data.json``, min / max over every keyframe):
#   neck [-1.50, 0], head_pitch [-1.00, 1.00], head_yaw [-0.70, 1.00], head_roll [-0.44, 0.35], body_pitch [0, 0.14];
#   body_z is 0 in every file.  The training ranges pad them by 5-10 % and add the robot.pose envelope's z.
EMOTION_RANGES = {
    "neck": (-1.50, 0.0), "head_pitch": (-1.00, 1.00), "head_yaw": (-0.70, 1.00),
    "head_roll": (-0.44, 0.35), "body_pitch": (0.0, 0.14), "body_z": (0.0, 0.0),
}  # fmt: skip
EX_HEAD_RANGES = ((-1.60, 0.20), (-1.05, 1.05), (-1.05, 1.05), (-0.46, 0.46))
EX_BODY_RANGES = (  # x, y, z, roll, pitch, yaw
    (0.0, 0.0), (0.0, 0.0), (-0.020, 0.008), (-0.06, 0.06), (-0.04, 0.20), (0.0, 0.0),
)  # fmt: skip
JOINT_LIMIT_DELTA = (  # joint limit minus HOME: the head cannot be commanded past it
    (-1.5708 - 0.3491, 1.0472 - 0.3491), (-1.5708 - 0.3491, 1.5708 - 0.3491), (-2.9671, 2.9671), (-0.4363, 0.4363),
)  # fmt: skip
EX_TWIST_MIX = (  # (prob, vx lo, vx hi, vy lo, vy hi, yaw lo, yaw hi, symmetric yaw); "v5" = upstream's draw
    (0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),  # standing
    (0.35, 0.35, 0.55, 0.0, 0.0, 0.0, 0.30, True),  # the neural duck's walking band
    (
        0.35,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    ),  # upstream's own distribution (anchor + general ability)
)
EX_HEAD_ALPHA = (
    (1.0, 0.8),
    (0.2, 0.2),
)  # head_alpha mix: the simulator runs 1.0, robotd's default is 0.2
EX_HEAD_RATE = (
    1.0,
    8.0,
)  # rad/s, per episode: how fast the client moves the head target (emotions go to 8.8)
EX_HEAD_RESAMPLE_S = (0.6, 2.5)
EX_BODY_RESAMPLE_S = (1.0, 3.5)
EX_BODY_RATE = (0.3, 2.0)  # rad/s (m/s ×0.1 for z)
EX_P_HEAD_ZERO = 0.45  # share of resamples with every head command at 0 (the anchor's frames)
EX_P_BODY_ZERO = 0.50
EX_P_STAND_TUNE = 0.5  # share of episodes whose standing frames run at robotd's stand-slot tuning (scale 1.0, gain 0.8)
EX_P_AXIS_OFF = 0.25  # per-axis: this axis stays 0 in a nonzero resample
EX_RANGE_STAGES = ((0, 0.6), (120, 0.8), (250, 1.0))  # (iteration, share of the ranges)

# ── item 2: quick turn in place ──────────────────────────────────────────────────────────────
TURN_YAW = (0.5, 3.0)  # |yaw command| trained (the gate asks 0.5-2.0; a skill may use more)
TURN_TWIST_MIX = (
    (0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),  # standing
    (0.14, 0.30, 0.50, 0.0, 0.0, 0.0, 0.30, True),  # walking at ~0.4, then the next resample turns
    (
        0.30,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        1.0,
        True,
    ),  # turn in place, slow (0.5 rad/s from standing did not start at it 499)
    (0.14, 0.0, 0.0, 0.0, 0.0, 1.0, 1.7, True),  # turn in place, medium
    (0.18, 0.0, 0.0, 0.0, 0.0, 1.7, 3.0, True),  # turn in place, fast
    (0.10, 0.0, 0.25, 0.0, 0.0, 0.5, 2.0, True),  # turn while creeping (a flight's first beat)
    (0.06, None, None, None, None, None, None, False),  # upstream's distribution
)
TURN_P_INSTANT = 0.6  # share of episodes whose twist is not smoothed (a skill's constant command)
TURN_RESAMPLE_S = (1.5, 4.0)
TURN_GOAL_K = 1.25  # the reward pays yaw rate up to this × the command

# ── item 4: head-stabilised walking ──────────────────────────────────────────────────────────
CALM_TWIST_MIX = (
    (0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
    (0.45, 0.35, 0.55, 0.0, 0.0, 0.0, 0.30, True),
    (0.10, 0.55, 1.0, 0.0, 0.0, 0.0, 0.6, True),  # the panic band (only trained when --panic)
    (0.30, None, None, None, None, None, None, False),
)

# ── item 3: personality gaits ────────────────────────────────────────────────────────────────
STYLES = ("normal", "sneak", "drag", "strut")
STYLE_CODE = {"normal": (0.0, 0.0), "sneak": (1.0, 0.0), "drag": (0.0, 1.0), "strut": (1.0, 1.0)}
# targets per style: trunk z delta (m), trunk pitch (rad, nose down +), foot swing height (m), step frequency (Hz, per foot),
# forward speed band at the walking command (m/s). normal = v5 itself.
STYLE_TARGETS = {
    "normal": {"z": 0.0, "pitch": math.radians(0.7), "clear": 0.020, "freq": None, "speed": None},
    "sneak": {"z": -0.012, "pitch": math.radians(5.0), "clear": 0.010, "freq": 1.5, "speed": (0.15, 0.30)},
    "drag": {"z": -0.005, "pitch": math.radians(8.0), "clear": 0.004, "freq": 2.3, "speed": (0.15, 0.30)},
    "strut": {"z": 0.007, "pitch": math.radians(-3.0), "clear": 0.034, "freq": 2.3, "speed": (0.35, 0.55)},
}  # fmt: skip
STYLE_TWIST_VX = (
    0.30,
    1.0,
)  # the style network is trained over this command range; the speed target is the band

# ── shared training numbers ─────────────────────────────────────────────────────────────────
INIT_STD = 0.15
LEARNING_RATE = 3e-4
CRITIC_WARMUP_ITERS = 60
RECONDITION_STD = {
    "v5": {55: 0.010, 56: 0.010, 57: 0.010, 58: 0.10, 59: 0.10, 60: 0.10},
    # item style: the style code (0 or 1) rides in slots 55 / 56: at std 0.010 a code of 1 would be a 100-sigma input and the
    # imported v5 falls at once (style-a: 57 % of the steps fallen); std 0.5 makes it a 2-sigma input
    "v5-style": {55: 0.5, 56: 0.5, 57: 0.010, 58: 0.10, 59: 0.10, 60: 0.10},
}  # body slots: pose commands reach 5 sigma of v5's normaliser (S2 contract)
BC = {
    "lr": 3e-4, "epochs": 5, "mini_batches": 4, "min_mini_batch": 512, "min_samples": 64,
    "fallen_tilt_deg": 35.0, "fallen_coef": 1.0, "anchor_tilt_deg": 25.0,
    "anchor_coef": ((0, 1.0),), "anchor_dims": tuple(range(ACTION_LEN)),
}  # fmt: skip
# which frames the v5 walk anchor holds (qt_bc.QtPpo): see qt_bc for the rules
ANCHOR_RULE = {"ex": "zero_pose", "turn": "not_turning", "calm": "legs", "style": "normal_style"}

# reward revisions: name -> {term: {"weight": w, "params": {...}}}; qt_tasks.base_terms has the g1 values
REWARD_REVS: dict[str, dict[str, dict]] = {
    "ex": {"e1": {}},
    "turn": {"t1": {}},
    "calm": {"c1": {}},
    "style": {"s1": {}},
}
DEFAULT_REV = {"ex": "e1", "turn": "t1", "calm": "c1", "style": "s1"}
# terms a run's OVERRIDE.json may re-weight / re-parameterise mid-run
OVERRIDABLE = {
    "ex_head": ("std", "mid_std", "fine_std"), "ex_neck": ("std",), "ex_trunk": ("pitch_std", "z_std", "roll_std"),
    "turn_rate": (), "turn_low": (), "turn_track": ("std",), "turn_still": ("std",),
    "calm_cam": (), "calm_head_still": (),
    "style_z": ("std",), "style_pitch": ("std",), "style_clear": ("std",), "style_cadence": ("std",), "style_speed": (),
    "fall_event": (), "action_rate_l2": (), "servo_stall": (), "servo_acc_spike": (),
    "track_vel_sharp": ("std",), "track_vel_forward": (),
}  # fmt: skip
UNITS_PER_HOUR = P.UNITS_PER_HOUR
