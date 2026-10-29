"""The duck's jaw: a beak that can hold a toy (curiosity spec §5.6, E9/E10, M5; owner decision
A4, 2026-09-27 — the pivot is a guess and every number that depends on it is provisional).

The official head has no moving jaw in the model: its beak is the convex hull of the head meshes,
a solid blob that can push a toy but never close on it, and the robot has no ``mouth`` actuator
(robotd drives the real mouth servo; ``body_server`` reports a ``mouth`` joint only if the model
has an actuator of that name, ``body_server.Body.__init__``). This world extension gives the
simulated duck the jaw the real one has, in memory, without editing any official file:

* the four beak hulls (``jaw``, ``jaw_soft``, ``soft_mouth_top``, ``bottom_head_shell``) stop
  colliding (their visuals stay);
* a thin box under the head keeps the head from sinking into things, and two plates are the
  beak: the upper one fixed to the head, the lower one on a ``mouth`` hinge (the mandible);
* a ``mouth`` position actuator (XL330-like: kp 0.55 N·m/rad, ±0.52 N·m), which the official
  body server finds by name and puts at robotd's wire index 9, so ``robot.mouth {open}`` moves
  it through robotd's own ``mouth_target`` (−5° closed … +30° open);
* the mandible's 16 g comes OUT of ``jaw_soft`` (M5: the head keeps its 188.8 g).

**Provisional (A4):** the hinge position :data:`PIVOT_M` is next to the mouth servo (xl330_10)
by eye, not from the CAD; the plates' size and the silicone friction come from curiosity E9/E10.
The grab standoff, the relay distance and the drop time measured with this jaw are marked
provisional until the pivot is measured on the real duck.

Load it FIRST among the world server's extensions, before ``duckmind.world.sensors`` (whose
eye keeps a copy of the spec when its own ``edit_spec`` runs: a copy taken before the jaw was
added would not fit the world's ``qpos``)::

    DUCKMIND_SIMHOME_EXT=duckmind.world.jaw,duckmind.world.worldkit,duckmind.world.sensors

``mouth_open_deg`` / ``in_beak`` read the jaw the way duck-body reads robotd's joint 9: a
mouth that a close command leaves at least :data:`IN_BEAK_MARGIN_DEG` above closed has
something in it (E10: holding 8–13°, empty −4.6…−5°).
"""

from __future__ import annotations

import math
from typing import Any

JOINT = "mouth"
ACTUATOR = "mouth"
MANDIBLE = "mandible"
HEAD = "jaw_soft"
MOUTH_CLOSED_RAD = math.radians(-5.0)  # robotd duck_control::model::MOUTH_CLOSED
MOUTH_OPEN_RAD = math.radians(30.0)  # robotd duck_control::model::MOUTH_OPEN
# Provisional (A4): the hinge next to the mouth servo, in the head (jaw_soft) frame.
PIVOT_M = (0.003, 0.0, -0.018)
BEAK_HULLS = frozenset({"jaw", "jaw_soft", "soft_mouth_top", "bottom_head_shell"})
SILICONE_FRICTION = (1.1, 0.01, 0.0005)
KP_NM_PER_RAD = 0.55
TORQUE_NM = 0.52
MANDIBLE_KG = 0.016  # beak plate 12 g + lever 4 g, taken out of jaw_soft (M5)
PLATE_KG = 0.012
LEVER_KG = 0.004
IN_BEAK_MARGIN_DEG = 3.0
GEOMS = ("jaw_head_under", "jaw_beak_upper", "jaw_beak_lower", "jaw_lever")


def mouth_target(open_fraction: float) -> float:
    """robotd's mouth target (radians) for an opening fraction, clamped (model.rs)."""
    if not math.isfinite(open_fraction):
        open_fraction = 0.0
    fraction = min(1.0, max(0.0, open_fraction))
    return MOUTH_CLOSED_RAD + fraction * (MOUTH_OPEN_RAD - MOUTH_CLOSED_RAD)


def in_beak(mouth_rad: float, *, commanded_open: float = 0.0) -> bool:
    """Something is in the beak: closed was asked for, yet the jaw stays open by the margin."""
    if commanded_open > 0.0:
        return False
    return math.degrees(mouth_rad - MOUTH_CLOSED_RAD) >= IN_BEAK_MARGIN_DEG


def add_jaw(spec: Any, home: Any = None) -> None:
    """Add the jaw to a composed home spec (a simhome ``edit_spec`` hook; ``home`` unused)."""
    del home
    import mujoco

    head = spec.body(HEAD)
    if head is None:
        raise ValueError(f"the robot has no {HEAD!r} body: no jaw")
    if any(joint.name == JOINT for joint in spec.joints):
        raise ValueError("the model already has a mouth joint")
    for geom in head.geoms:
        if geom.meshname in BEAK_HULLS and geom.contype != 0:
            geom.contype = 0
            geom.conaffinity = 0
    friction = list(SILICONE_FRICTION)
    head.add_geom(
        name="jaw_head_under",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.0085, 0.0, -0.0025],
        size=[0.0085, 0.044, 0.041],
        friction=friction,
        rgba=[0.9, 0.9, 0.2, 0.0],
        mass=0.0,
    )
    head.add_geom(
        name="jaw_beak_upper",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.007, 0.0, -0.061],
        size=[0.002, 0.044, 0.017],
        friction=friction,
        rgba=[1.0, 0.6, 0.1, 0.0],
        mass=0.0,
    )
    if head.mass and head.mass > MANDIBLE_KG:
        head.mass = head.mass - MANDIBLE_KG
    jaw = head.add_body(name=MANDIBLE, pos=list(PIVOT_M))
    jaw.add_joint(
        name=JOINT,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        range=[MOUTH_CLOSED_RAD - 0.05, MOUTH_OPEN_RAD + 0.05],
        damping=0.053,
        frictionloss=0.0048,
        armature=0.0018,
    )
    jaw.add_geom(
        name="jaw_beak_lower",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.013 - PIVOT_M[0], 0.0, -0.061 - PIVOT_M[2]],
        size=[0.004, 0.044, 0.017],
        friction=friction,
        rgba=[1.0, 0.5, 0.0, 0.0],  # unseen: the eye must not see an orange toy in its beak
        mass=PLATE_KG,
    )
    jaw.add_geom(
        name="jaw_lever",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.010, 0.0, -0.012],
        size=[0.004, 0.03, 0.012],
        contype=0,
        conaffinity=0,
        mass=LEVER_KG,
        rgba=[1.0, 0.5, 0.0, 0.0],
    )
    spec.add_actuator(
        name=ACTUATOR,
        target=JOINT,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[KP_NM_PER_RAD] + [0.0] * 9,
        biasprm=[0.0, -KP_NM_PER_RAD, 0.0] + [0.0] * 7,
        forcelimited=True,
        forcerange=[-TORQUE_NM, TORQUE_NM],
    )


def mouth_open_deg(model: Any, data: Any) -> float:
    """The jaw's angle above closed, degrees (0 closed … 35 fully open)."""
    import mujoco

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT)
    if joint < 0:
        raise ValueError("no jaw in this model")
    return math.degrees(float(data.qpos[model.jnt_qposadr[joint]]) - MOUTH_CLOSED_RAD)


def info(model: Any) -> dict[str, Any]:
    """The jaw's numbers in a compiled model (for tests and evidence)."""
    import mujoco

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HEAD)
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MANDIBLE)
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR)
    return {
        "head_g": round(float(model.body_mass[head]) * 1000.0, 2),
        "mandible_g": round(float(model.body_mass[jaw]) * 1000.0, 2) if jaw >= 0 else None,
        "actuator": int(actuator),
        "pivot_m": list(PIVOT_M),
        "provisional": True,
    }


def register(registry: Any) -> None:
    """simhome's entry point (load before duckmind.world.sensors)."""
    registry.edit_spec(add_jaw, name="jaw.add")
