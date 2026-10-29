"""What the two Stage-1 networks are trained to do, in numbers (tr:tasks, 2026-09-28).

Pure standard library: the main venv's tests import it, and every mjlab module of the training
side (``rl_*``) reads its numbers from here, so a number is written once.

**Single source of truth for the skill windows** — :mod:`duckmind.skilltrain.table` (the gate
lane's robotd ``[[policy.skill]]`` table). :func:`variants` reads ``table.stage1()`` and keeps the
entries that run one of our two networks (``startle.onnx``: ``startle_1..4``, ``freeze_3/4``,
``hop_back``, ``startle_release``; ``comfort.onnx``: ``shake_1/2``, ``stretch_tall``,
``stretch_sit``). Their ``command`` / ``unwind`` / ``duration`` / ``unwind_s`` and the pinned
``params.action_scale`` / ``gain_ratio`` are what training feeds and runs at (training value =
deployment value, training plan §2). If the table changes, training follows it and
``tests/test_skilltrain_rl_contract.py`` says what changed.

**robotd's action path** (``robotd/src/control.rs`` ``Controller::step`` and ``main.rs``, read-only
upstream): ``target = HOME + action_scale · scale_mult · action`` with ``scale_mult =
7.4 / clamp(battery, 6.0, 9.5)`` (``voltage_adapt``, on by default), then the first-order low-pass
against the previous *filtered* target (head joints 0.5, the ten leg joints 0.7), then the ±π
actuator clamp; the network sees its own raw previous output, shared across networks (a skill's
first observation carries the walk network's last output). While a skill runs robotd feeds the
entry's constant twist and zeroes the head and body slots. Upstream trains unfiltered; we train
with robotd's filters inside the action term (plan §2).

**The kinematic tables** (:data:`HOME_DIP_TABLE`, :data:`SIT_RISE_TABLE`) are the left leg's
(hip_pitch, knee, ankle) offsets that move the trunk straight down/up with both feet flat and in
place and the trunk level, solved by least squares on the ``left_foot`` site pose of the official
``robot_allcollisions.xml`` (microduck_rl cb70b79; kinematics only, computed on the Mac on
2026-09-28 by the stage-1 ``ik_tables.py`` in the measurement records, not in the repo; the 5° hip roll
leaves ≤ 0.2 mm of lateral foot drift per 25 mm; ``tests/test_skilltrain_rl_mjlab.py`` re-checks it). The right leg
mirrors them (negated). Rising above HOME straightens the knee: +10 mm is the leg's reach
(knee straight), so the standing stretch's +8 mm gate sits near the kinematic ceiling.

**Head sign conventions** (pet:probe ``calib.json`` and the MJCF): ``head_pitch`` + = beak down,
``neck_pitch`` + = head back and up. With the trunk level, the camera's pitch moves by
``Δneck_pitch − Δhead_pitch`` (1:1, radians), and the neck pull-back the brain's startle commands
(neck +0.3, head −0.2) moves the head ~45 mm back and tips the beak ~29° up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.skilltrain import table as T

STAGE = "stage1-2026-09-28"
MICRODUCK_RL_COMMIT = "cb70b792312d559a4da09064d92009079671815f"
MJLAB_VERSION = "1.3.0"
# The v5 set the neuro validation used (sha256; the walk slot's velstand is what hands back).
V5_SHA256 = {
    "velstand.onnx": "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45",
    "alpha_sitstand.onnx": "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
}
V5_HF_URL = "https://huggingface.co/pollen-robotics/microduck-policies/resolve/v5/{name}"

# ── the robot and robotd ─────────────────────────────────────────────────────────────────────
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
OBS_LEN, ACTION_LEN = 61, 14
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
HOME = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)  # fmt: skip
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8
HEAD_IDX = (NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL)
LEFT_LEG = (0, 1, 2, 3, 4)
RIGHT_LEG = (9, 10, 11, 12, 13)
LEG_IDX = LEFT_LEG + RIGHT_LEG
LEFT_PITCH_CHAIN = (2, 3, 4)  # hip_pitch, knee, ankle
RIGHT_PITCH_CHAIN = (11, 12, 13)
LOWPASS_HEAD = 0.5  # control.rs Tuning::default (the alpha policies' training value)
LOWPASS_LEGS = 0.7
NOMINAL_V = 7.4  # robotd-params nominal_voltage; voltage_adapt on since 2026-09-14
VIN_CLAMP = (6.0, 9.5)
ACTUATOR_CLAMP = math.pi  # duck-control safety.rs
WALK_ACTION_SCALE = 0.9  # velstand in the walk slot (no stand network in v5)
SITSTAND_ACTION_SCALE = 1.0
FALL_TILT_DEG = 60.0  # duck-control Safety::observe: gravity z > -0.5 for 0.2 s
FALL_DEBOUNCE_S = 0.2
CAMERA_SITE = "head_camera"
# The obs layout every microduck policy shares (duck-control obs.rs).
OBS_SLICES = {
    "gyro": (0, 3), "gravity": (3, 6), "joint_pos": (6, 20), "joint_vel": (20, 34),
    "last_action": (34, 48), "twist": (48, 51), "head": (51, 55), "body": (55, 61),
}  # fmt: skip

STAND_Z = 0.115  # measured standing equilibrium (sitstand / velstand)
SIT_Z = 0.060
# The sitstand SIT keyframe (servo index → angle); the neck/head stay at HOME.
SIT_KEYFRAME = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}

# ── kinematic tables (left leg; right = −left) ──────────────────────────────────────────────
# (trunk dip below HOME in m; +down) → Δ(hip_pitch, knee, ankle) from HOME.
HOME_DIP_TABLE: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (-0.010, (-0.3990, -0.7849, -0.3859)),  # +10 mm: the knee is straight (the leg's reach)
    (-0.008, (-0.2652, -0.5197, -0.2545)),
    (-0.005, (-0.1447, -0.2824, -0.1377)),
    (0.000, (0.0, 0.0, 0.0)),
    (0.005, (0.1158, 0.2236, 0.1078)),
    (0.008, (0.1777, 0.3421, 0.1644)),
    (0.010, (0.2168, 0.4164, 0.1996)),
    (0.015, (0.3089, 0.5898, 0.2809)),
    (0.020, (0.3952, 0.7496, 0.3544)),
    (0.025, (0.4779, 0.8994, 0.4216)),
    (0.030, (0.5585, 1.0416, 0.4831)),
)
# (trunk rise above the SIT keyframe in m) → Δ(hip_pitch, knee, ankle) from SIT.
SIT_RISE_TABLE: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.000, (0.0, 0.0, 0.0)),
    (0.005, (0.0196, -0.1119, -0.1315)),
    (0.010, (0.0190, -0.2313, -0.2503)),
    (0.015, (0.0029, -0.3579, -0.3608)),
    (0.020, (-0.0258, -0.4921, -0.4664)),
    (0.025, (-0.0656, -0.6353, -0.5697)),
)
HOME_REACH_M = 0.010  # above this the legs cannot lift the trunk (flat feet, level trunk)


def _interp(table: Sequence[tuple[float, tuple[float, float, float]]], x: float):
    xs = [row[0] for row in table]
    if x <= xs[0]:
        return table[0][1]
    if x >= xs[-1]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:], strict=False):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return tuple(a + t * (b - a) for a, b in zip(y0, y1, strict=True))
    raise AssertionError("unreachable")


def _mirror_legs(base: Sequence[float], left: Sequence[float]) -> tuple[float, ...]:
    q = list(base)
    for k, (li, ri) in enumerate(zip(LEFT_PITCH_CHAIN, RIGHT_PITCH_CHAIN, strict=True)):
        q[li] += left[k]
        q[ri] -= left[k]
    return tuple(q)


def home_dip_pose(dip_m: float) -> tuple[float, ...]:
    """HOME with the legs bent (dip > 0) or straightened (dip < 0) to move the trunk by
    ``-dip_m`` with the feet flat and in place (14 joints, the head at HOME)."""
    return _mirror_legs(HOME, _interp(HOME_DIP_TABLE, dip_m))


def sit_pose() -> tuple[float, ...]:
    q = list(HOME)
    for i, v in SIT_KEYFRAME.items():
        q[i] = v
    return tuple(q)


def sit_rise_pose(rise_m: float) -> tuple[float, ...]:
    """The SIT keyframe with the legs pushing the trunk ``rise_m`` above the seat."""
    return _mirror_legs(sit_pose(), _interp(SIT_RISE_TABLE, rise_m))


# ── what each behaviour aims at ─────────────────────────────────────────────────────────────


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, t))


@dataclass(frozen=True)
class CrouchTarget:
    """Startle / freeze / hop-back posture at full blend (plan §3.1)."""

    k: float
    dip_m: float  # trunk below STAND_Z
    neck_rad: float  # neck_pitch pull-back (+)
    head_rad: float  # head_pitch counter-rotation (−)
    pitch_back_deg: float  # slight trunk pitch-back (nose up, +)
    ramp_s: float  # slewed target: 0 → full in this time (arriving early pays nothing)

    def pose(self) -> tuple[float, ...]:
        q = list(home_dip_pose(self.dip_m))
        q[NECK_PITCH] += self.neck_rad
        q[HEAD_PITCH] += self.head_rad
        return tuple(q)


def crouch_target(k: float) -> CrouchTarget:
    """k ∈ [0.25, 1]: dip 8 → 25 mm, neck pull-back 0.15 → 0.35 rad (head −2/3 of it, the ratio
    of the brain's startle program), pitch-back 1.5 → 4°, ramp 0.15 → 0.25 s (plan §3.1)."""
    t = (min(1.0, max(0.25, k)) - 0.25) / 0.75
    neck = _lerp(0.15, 0.35, t)
    return CrouchTarget(
        k=k,
        dip_m=_lerp(0.008, 0.025, t),
        neck_rad=neck,
        head_rad=-neck * 2.0 / 3.0,
        pitch_back_deg=_lerp(1.5, 4.0, t),
        ramp_s=_lerp(0.15, 0.25, t),
    )


HOP_BACK_M = 0.10  # potential cap (plan §3.1: capped at 10 cm)
HOP_WINDOW_S = 0.8  # back-steps must happen in this window, then planted
HOP_K = 1.0  # hop_back crouches like startle_4 (command [1.0, 1, 0])
# rev a3 (tr:train, 2026-09-28): the hop's TRUNK DIP is startle_2's (≈ 14 mm) while its neck pull-back stays
# startle_4's — the a2 hop stepped back in the 25 mm crouch at leg torque p95 0.49–0.62 N·m (gate ≤ 0.35).
HOP_DIP_K = 0.5
# rev a4 (s1b:retrain, 2026-09-28): a real two-to-three-step hop-back, not rev a3's 7–8 shuffles of ≈ 1.2 cm
# (the review's minor). A back-step is a touchdown after ≥ HOP_STEP_AIR_S of air; it pays by its backward
# length (0 at ≤ 1.5 cm, fully at ≥ 3 cm) for the first HOP_MAX_STEPS steps of the hop window; every
# touchdown (≥ 40 ms of air) beyond the HOP_MAX_STEPS-th costs; both feet off the floor costs (plan D5).
HOP_STEP_M = (0.015, 0.03)
HOP_STEP_AIR_S = 0.06
HOP_MAX_STEPS = 4
MIN_STEP_AIR_S = 0.04  # a footfall = a re-contact after ≥ 40 ms of air (the Mac gate's definition)


@dataclass(frozen=True)
class ShakeTarget:
    """Shake-off (plan §3.2): trunk roll and head yaw oscillation in amplitude/frequency bands,
    feet planted, heading held. Phase-invariant (the network has no clock)."""

    a: float
    roll_deg: tuple[float, float]
    roll_hz: tuple[float, float]
    head_yaw_rad: tuple[float, float]
    head_hz: tuple[float, float]


def shake_target(a: float) -> ShakeTarget:
    t = (min(1.0, max(0.5, a)) - 0.5) / 0.5
    return ShakeTarget(
        a=a,
        roll_deg=(_lerp(4.0, 6.0, t), _lerp(6.0, 9.0, t)),
        roll_hz=(2.5, 4.0),
        head_yaw_rad=(_lerp(0.15, 0.22, t), _lerp(0.22, 0.30, t)),
        head_hz=(3.0, 5.0),
    )


SHAKE_WINDOW_S = 0.5  # the oscillation statistics' sliding window
SHAKE_VALID_AFTER_S = 0.3


@dataclass(frozen=True)
class StretchTarget:
    """Stretch (plan §3.2): the trunk up, the neck/head up, held still; the yawn's beak is
    duck-body's (the mouth is outside every policy)."""

    seated: bool
    rise_m: float  # above STAND_Z (standing) or SIT_Z (seated)
    camera_deg: float  # head-camera optical-axis pitch, world (+ up)
    neck_rad: float
    head_rad: float
    ramp_s: float

    def pose(self) -> tuple[float, ...]:
        q = list(sit_rise_pose(self.rise_m) if self.seated else home_dip_pose(-self.rise_m))
        q[NECK_PITCH] += self.neck_rad
        q[HEAD_PITCH] += self.head_rad
        return tuple(q)

    @property
    def base_z(self) -> float:
        return SIT_Z if self.seated else STAND_Z


STRETCH_TALL = StretchTarget(False, 0.010, 20.0, 0.0, -0.40, 0.7)
# rev c4 (s1b:retrain 2026-09-28): the seated stretch lifts the head MORE than a plain seated look-up (the c3
# stretch reached a camera pitch of 20.5–20.7° against v5's seated look-up at 25.9–27.2°): the neck pulls back
# and up (+0.12 rad) and the head tips up (−0.46 rad), Δneck − Δhead = 0.58 rad (33°) against c3's 0.40 (23°);
# the camera target 30° (c3 reached its 20° target within 0.3°); the +15 mm half-rise is kept.
STRETCH_SIT = StretchTarget(True, 0.015, 30.0, 0.12, -0.46, 0.7)
UNWIND_RAMP_S = 0.5  # the blend returns to the rest posture this fast (well inside unwind_s)
RISE_SPEED_CAP = 0.08  # m/s, sitstand's rise cap, during unwinds (plan §3.1)

# ── variants: robotd's table entries that run our two networks ─────────────────────────────

NETS = ("startle", "comfort")
PLACEHOLDER = {
    "startle": "/skilltrain/startle.onnx",
    "comfort": "/skilltrain/comfort.onnx",
    "walk": "/skilltrain/velstand.onnx",
}
NET_OF_FAMILY = {
    "startle": "startle", "freeze": "startle", "hop_back": "startle", "release": "startle",
    "shake": "comfort", "stretch_tall": "comfort", "stretch_sit": "comfort",
}  # fmt: skip
BEHAVIOURS = ("crouch", "hop", "rest", "shake", "stretch_tall", "stretch_sit")


@dataclass(frozen=True)
class Variant:
    """One table entry as the training episode sees it."""

    name: str
    family: str
    net: str
    behaviour: str
    command: tuple[float, float, float]
    unwind: tuple[float, float, float]
    hold_s: float
    unwind_s: float
    level: float
    postures: tuple[str, ...]
    chain: bool
    action_scale: float
    gain_ratio: float


def _behaviour(family: str) -> str:
    return {
        "startle": "crouch", "freeze": "crouch", "hop_back": "hop", "release": "rest",
        "shake": "shake", "stretch_tall": "stretch_tall", "stretch_sit": "stretch_sit",
    }[family]  # fmt: skip


def stage1_entries() -> list[T.Entry]:
    return T.stage1(PLACEHOLDER["startle"], PLACEHOLDER["comfort"], PLACEHOLDER["walk"])


def variants(net: str, entries: Sequence[T.Entry] | None = None) -> tuple[Variant, ...]:
    """The table entries whose ONNX is ``net``'s, in table order."""
    if net not in NETS:
        raise ValueError(f"net {net!r}; use one of {NETS}")
    out = []
    for e in stage1_entries() if entries is None else entries:
        if e.skill.path != PLACEHOLDER[net]:
            continue
        if NET_OF_FAMILY.get(e.family) != net:
            raise ValueError(f"table entry {e.name!r} (family {e.family}) runs {net}.onnx")
        s = e.skill
        if s.action_scale is None or s.gain_ratio is None:
            raise ValueError(f"table entry {e.name!r}: scale/gain not pinned")
        out.append(
            Variant(
                name=e.name,
                family=e.family,
                net=net,
                behaviour=_behaviour(e.family),
                command=tuple(float(v) for v in s.command),  # type: ignore[arg-type]
                unwind=tuple(float(v) for v in s.unwind),  # type: ignore[arg-type]
                hold_s=float(TRAIN_HOLD_S.get(e.family, s.duration)),
                unwind_s=float(s.unwind_s),
                level=float(e.level if e.level is not None else 0.0),
                postures=tuple(e.postures),
                chain=bool(s.chain),
                action_scale=float(s.action_scale),
                gain_ratio=float(s.gain_ratio),
            )
        )
    if not out:
        raise ValueError(f"the skill table has no entry running {net}.onnx")
    return tuple(out)


def pinned(net: str) -> tuple[float, float]:
    """(action_scale, gain_ratio) every entry of ``net`` pins; training runs at exactly these."""
    pairs = {(v.action_scale, v.gain_ratio) for v in variants(net)}
    if len(pairs) != 1:
        raise ValueError(f"{net}: entries pin different scale/gain {sorted(pairs)}")
    return next(iter(pairs))


def release_variant() -> Variant:
    rel = [v for v in variants("startle") if v.behaviour == "rest"]
    if len(rel) != 1:
        raise ValueError("the table needs exactly one startle release entry")
    return rel[0]


# ── episodes: which entries, which hold lengths ─────────────────────────────────────────────

ENTRY_KINDS = ("stand", "walk", "coast", "head", "seated", "risen", "crouched")
KIND_POSTURE = {
    "stand": "stand", "walk": "walk", "coast": "stand", "head": "stand",
    "seated": "sit", "risen": "stand", "crouched": "stand",
}  # fmt: skip
# Plan §3.1 (startle) and §3.2 (comfort) entry mixes; kinds a variant's postures exclude are
# dropped and the rest renormalised.
ENTRY_MIX = {
    "crouch": {"stand": 0.35, "walk": 0.35, "coast": 0.10, "crouched": 0.10, "head": 0.10},
    "hop": {"stand": 0.35, "walk": 0.35, "coast": 0.10, "crouched": 0.10, "head": 0.10},
    "rest": {"stand": 0.30, "walk": 0.30, "coast": 0.10, "crouched": 0.30},
    "shake": {"stand": 0.60, "head": 0.20, "risen": 0.20},
    "stretch_tall": {"stand": 0.60, "head": 0.20, "risen": 0.20},
    "stretch_sit": {"seated": 1.0},
}
# Early-training mix (phase-aligned: easy standing entries first, the plan's mix by ~600 it).
ENTRY_MIX_EARLY = {
    "crouch": {"stand": 0.60, "walk": 0.15, "coast": 0.05, "crouched": 0.10, "head": 0.10},
    "hop": {"stand": 0.60, "walk": 0.15, "coast": 0.05, "crouched": 0.10, "head": 0.10},
    "rest": {"stand": 0.50, "walk": 0.15, "coast": 0.05, "crouched": 0.30},
}
# How often a startle-network hold is cut short by ``startle_release`` (a chaining entry that
# pre-empts: the network sees the zero command for its duration instead of the entry's unwind).
RELEASE_CUT_PROB = {"crouch": 0.30, "hop": 0.15}
REST_ONLY_WEIGHT = 0.05  # release requested with nothing running
HOLD_JITTER = (0.8, 1.3)  # a fraction of episodes jitter the nominal hold (robustness)
HOLD_JITTER_PROB = 0.3
CONTINUOUS_LEVEL_PROB = 0.2  # sample k / a between the table's levels too

# ── Stage 1b retrain (s1b:retrain, 2026-09-28; the review's majors (4) and minors) ──────────
# Training windows that differ from the table (``table.stage1``, the robotd contract, is unchanged): the shake-off
# is trained 1.5 s long (4–5 cycles at ≈ 3 Hz; rev c3 ran the table's 1.0 s, ≈ 2 cycles in band). The simulator
# runs it through ``table.stage1_live(shake_s=1.5)`` once it gates; the evaluator and the manifest use these.
TRAIN_HOLD_S = {"shake": 1.5}
# Held crouches (the live table's freeze_* re-requested while the fear lasts): a share of crouch episodes holds
# 2–6 s (robustness; the Mac gate held 1/3/10 s on the rev-a3 network without falls on the XML servos).
HOLD_LONG_PROB = {"crouch": 0.05}
HOLD_LONG_S = (2.0, 6.0)
# The servo command delay training randomises (physics steps of 5 ms, ``BamActuatorCfg.delay_*``): every
# episode starts with a lag drawn from 0–8 steps (0–40 ms) (``rl_mdp.randomize_servo_lag``), re-drawn every
# 100 physics steps (0.5 s, staggered per env) with probability 0.5 — a piecewise-constant delay covering fixed
# short and fixed long delays. The s1b latency map (21,600 Mac trials): rev a3/c3 were fall-free on the XML
# servos only at 15–30 ms, the hop performed only at 20–25 ms. The recipe's own delay (3–6 steps redrawn every
# step) stays the evaluator's default ("recipe").
PHYSICS_DT = 0.005
SERVO_LAG_TRAIN = {"min_lag": 0, "max_lag": 8, "update_period": 100, "hold_prob": 0.5}
SERVO_LAG_RECIPE = (3, 6)
# rev a4: stopping from a walk at the servo limit (the Mac: startle_4 from 0.55 m/s leg |τ| p95 0.56–0.64 N·m
# against v5's own stop 0.33–0.40) and the heading while stopping (BAM spins > 20° in up to 30 % of the 0.55 m/s
# cells; v5's own stop 10–18° p90): Σ leg |τ| above WALK_STOP_FREE_NM in the first WALK_STOP_WINDOW_S of a skill
# entered from a walk or a coast, and |Δheading| above WALK_YAW_FREE_RAD through it.
WALK_STOP_FREE_NM = 0.35
WALK_STOP_WINDOW_S = 0.7
WALK_YAW_FREE_RAD = 0.10
# rev c4: the stretches' joint acceleration (the Mac: stretch_tall p99 316 rad/s², gate 300; MuJoCo's qacc) — Σ
# over the servos of |Δqd/dt| (control rate, a proxy: the A100's p99 of it was 70–85 rad/s² on c3's stretches)
# above STRETCH_ACC_FREE through a stretch's window, weighted mildly (only the spikes).
STRETCH_ACC_FREE = 80.0


def entry_mix(behaviour: str, postures: Sequence[str], early: bool = False) -> dict[str, float]:
    mix = dict((ENTRY_MIX_EARLY if early else ENTRY_MIX).get(behaviour) or ENTRY_MIX[behaviour])
    kept = {k: w for k, w in mix.items() if KIND_POSTURE[k] in postures and w > 0}
    if not kept:
        raise ValueError(f"{behaviour}: no entry kind fits postures {tuple(postures)}")
    total = sum(kept.values())
    return {k: w / total for k, w in kept.items()}


def variant_weights(net: str) -> dict[str, float]:
    """Training mixture over the table entries of ``net`` (hop-back gets extra weight: it is
    the hardest; the release entry is sampled through cuts plus a small rest-only share)."""
    vs = variants(net)
    if net == "startle":
        crouch = [v for v in vs if v.behaviour == "crouch"]
        hop = [v for v in vs if v.behaviour == "hop"]
        weights = {v.name: 0.70 / len(crouch) for v in crouch}
        weights.update({v.name: 0.25 / len(hop) for v in hop})
        weights.update({v.name: REST_ONLY_WEIGHT for v in vs if v.behaviour == "rest"})
    else:
        weights = {v.name: 1.0 / len(vs) for v in vs}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def max_episode_s(net: str) -> float:
    vs = variants(net)
    hold = max(v.hold_s for v in vs) * HOLD_JITTER[1]
    if any(HOLD_LONG_PROB.get(v.behaviour, 0.0) > 0 for v in vs):
        hold = max(hold, HOLD_LONG_S[1])
    unwind = max([v.unwind_s for v in vs] + [release_variant().hold_s if net == "startle" else 0])
    return hold + unwind + 0.2


# ── gates (plan §6; the performed detectors are the same thresholds) ────────────────────────

GATES: dict[str, dict[str, Any]] = {
    "crouch": {
        "first_motion_s": 0.12, "first_motion_mm": 3.0, "first_motion_deg": 3.0,
        "retract_rad": ((0.25, 0.15), (1.0, 0.25)), "retract_by_s": 0.25,
        "dip_mm": ((0.25, 5.0), (0.5, 10.0), (0.75, 15.0), (1.0, 20.0)), "dip_by_s": 0.30,
        "hold_band_mm": 3.0, "hold_footfalls_per_2s": 1.0,
        "walk_speed_mps": 0.02, "walk_stop_by_s": 0.7, "walk_travel_m": 0.12,
    },
    "hop": {"back_m": 0.06, "by_s": 0.7, "yaw_deg": 15.0},
    "shake": {
        "roll_deg": 4.0, "roll_hz": (2.5, 4.0), "for_s": 0.5, "head_yaw_rad": 0.15,
        "head_hz_min": 3.0, "footfalls": 1, "drift_m": 0.02,
    },
    "stretch_tall": {"rise_mm": 8.0, "camera_deg": 15.0, "hold_s": 0.8},
    "stretch_sit": {"rise_mm": 10.0, "camera_deg": 15.0, "hold_s": 0.8, "reseat_by_s": 1.5},
    "safety": {
        "fall_rate": {"low": 0.002, "high": 0.005},  # startle 1–3/shake/stretch vs 4/hop
        "torque_p95_nm": 0.35, "stall_frac": 0.005, "jacc_p99": 300.0, "handback_s": 5.0,
        "reseated_z_m": SIT_Z + 0.005,
    },
}  # fmt: skip


def gate_interp(pairs: Sequence[tuple[float, float]], level: float) -> float:
    """Linear interpolation of a (level, threshold) gate table."""
    return _interp([(x, (y, 0.0, 0.0)) for x, y in pairs], level)[0]


def fall_rate_gate(variant: Variant) -> float:
    high = variant.behaviour == "hop" or (variant.behaviour == "crouch" and variant.level >= 1.0)
    return float(GATES["safety"]["fall_rate"]["high" if high else "low"])


# ── statistics ──────────────────────────────────────────────────────────────────────────────


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % interval for k/n (as fractions)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> list[float]:
    """``[k, n, lo %, hi %]`` — the format of the A100 validation results."""
    lo, hi = wilson(k, n)
    return [int(k), int(n), round(100 * lo, 3), round(100 * hi, 3)]


def summary_of(values: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:.3g}" for k, v in values.items())


__all__ = [
    "ACTION_LEN",
    "BEHAVIOURS",
    "CONTROL_HZ",
    "CrouchTarget",
    "DT",
    "ENTRY_KINDS",
    "GATES",
    "HOME",
    "HOME_DIP_TABLE",
    "JOINTS",
    "OBS_LEN",
    "SIT_RISE_TABLE",
    "ShakeTarget",
    "StretchTarget",
    "Variant",
    "crouch_target",
    "entry_mix",
    "home_dip_pose",
    "pinned",
    "rate",
    "shake_target",
    "sit_pose",
    "sit_rise_pose",
    "variant_weights",
    "variants",
    "wilson",
]
