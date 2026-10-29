"""The official duck on the official flat floor, driven by the robotd port with skills.

:class:`SkillDuck` is :class:`~duckmind.world.headless_duck.HeadlessDuck` (phase 1's in-process
duck, moved into the repo) on ``scene_allcollisions.xml`` — the official floor and keyframes,
the whole-body collision model velstand was trained on and the live simulator runs since phase
0 — with three things swapped:

* the controller is :class:`~duckmind.skilltrain.robotd_port.Robotd` (robotd's control path
  *including* the one-shot skills; HeadlessDuck's own port refuses every skill but sit_toggle);
* the servos are either model of :mod:`~duckmind.skilltrain.servo` (``mjcf`` / ``bam``), behind
  the draw's command delay;
* the draw of :mod:`~duckmind.skilltrain.dr` is applied to the model and to what robotd reads.

Each 50 Hz tick is recorded (:class:`Trace`): the trunk pose and velocity, joints, targets, the
peak actuator force over the tick's 4 physics steps, footfalls (a re-contact after ≥ 40 ms of
air, as phase 1 counts them), the net robot–floor contact force on the XL330 housings, the head
and the trunk (peak over the substeps), the head camera's pitch, the beak in the trunk frame,
robotd's label and fall verdict. Evaluators only: nothing here is the duck's own perception.
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import dr as DR
from duckmind.skilltrain import servo as SV
from duckmind.skilltrain.robotd_port import PolicySet, Robotd, Sensors, Step
from duckmind.skilltrain.table import SkillDef
from duckmind.world.headless_duck import (
    HOME14,
    LEGS14,
    SEATED_BOOT_RAD,
    STEPS_PER_TICK,
    TICK_S,
    TIMESTEP_S,
    Composed,
    HeadlessDuck,
    gravity_in_trunk,
    official_dir,
)

SCENE = "scene_allcollisions.xml"
FEET = ("left_foot_collision", "right_foot_collision")
HEAD_CONTACT_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
MIN_AIR_S = 0.04  # phase 1 touchdowns(): a re-contact after < 40 ms of air is not a step
HOME_RAMP_S = 2.0
WALK_FILE, SITSTAND_FILE = "velstand.onnx", "alpha_sitstand.onnx"


class Trace:
    """Per-tick samples, appended as lists and read back as arrays (:meth:`arrays`)."""

    SCALARS = (
        "t",
        "tilt",
        "gz",
        "cam_pitch",
        "f_servo",
        "f_head",
        "f_trunk",
        "f_other",
        "fallen",
        "gain",
        "vin",
        "busy",
    )
    VECTORS = {
        "pos": 3,
        "quat": 4,
        "vel": 3,
        "angvel": 3,
        "q": 14,
        "qd": 14,
        "qacc": 14,
        "target": 14,
        "f_housing": 15,
        "tau_pk": 14,
        "feet": 2,
        "td": 2,
        "beak": 3,
        "cam": 3,
        "twist_cmd": 3,
        "head_cmd": 4,
        "body_cmd": 3,
        "cam_w": 3,
        "foot_z": 2,
    }

    def __init__(self) -> None:
        self.data: dict[str, list[Any]] = {k: [] for k in (*self.SCALARS, *self.VECTORS)}
        self.labels: list[str] = []

    def append(self, sample: dict[str, Any], label: str) -> None:
        for key, values in self.data.items():
            values.append(sample[key])
        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def arrays(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: np.asarray(v, dtype=float) for k, v in self.data.items()}
        out["label"] = np.asarray(self.labels, dtype=object)
        return out


class SkillDuck(HeadlessDuck):
    """One simulated duck with robotd's skill path, a servo model and a DR draw."""

    def __init__(
        self,
        skills: Sequence[SkillDef],
        *,
        servo: str = "mjcf",
        draw: DR.Draw | None = None,
        head_alpha: float = 1.0,
        keyframe: str = "SIT",
        policies: str | os.PathLike[str] | None = None,
        mjcf_dir: str | os.PathLike[str] | None = None,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        pose_zeroes_twist: bool = True,
    ) -> None:
        import mujoco

        if servo not in ("mjcf", "bam"):
            raise ValueError(f"servo model {servo!r}: mjcf or bam")
        folder = official_dir(mjcf_dir)
        spec = mujoco.MjSpec.from_file(str(folder / SCENE))
        super().__init__(
            Composed(spec=spec, layout=None, kit=None),
            mjcf_dir=folder,
            keyframe=keyframe,
            x=x,
            y=y,
            yaw=yaw,
            policies=policies,
            household=False,
        )
        m, d = self.model, self.data
        self.servo_name = servo
        self.draw = draw if draw is not None else DR.Draw(seed=0, preset="none")
        self.skill_defs = list(skills)
        self._ids(mujoco)
        DR.apply_physical(self.draw, m, d, trunk=self.trunk, foot_geoms=self.foot_geoms)
        self.physics: dict[str, Any] = {
            "integrator": int(m.opt.integrator),
            "iterations": int(m.opt.iterations),
            "ls_iterations": int(m.opt.ls_iterations),
        }
        self.servo: SV.XmlServos | SV.BamServos
        if servo == "bam":
            self.physics = SV.training_physics(m, self.servo_geoms, self.foot_geoms)
            self.servo = SV.BamServos(
                m,
                d,
                self.act,
                self.jq,
                self.jv,
                vin=self.draw.vin,
                vin_drop_gain=self.draw.vin_drop,
                friction_scale=self.draw.bam_friction,
                armature_scale=self.draw.armature,
            )
        else:
            self.servo = SV.XmlServos(m, self.act)
        self.lag_steps = int(self.draw.lag_steps) if servo == "bam" else 0
        self.obs_path = DR.ObsPath(self.draw if servo == "bam" else DR.Draw(0, "none"))
        self.battery_v = self.draw.vin if servo == "bam" else 7.4
        folder_p = Path(self.policies)
        self.port = Robotd(
            PolicySet(folder_p / WALK_FILE, folder_p / SITSTAND_FILE, self.skill_defs),
            self.skill_defs,
            head_alpha=head_alpha,
            pose_zeroes_twist=pose_zeroes_twist,
        )
        self.head_alpha = float(head_alpha)
        self._lag: deque[np.ndarray] = deque(maxlen=self.lag_steps + 1)
        self._homing: tuple[np.ndarray, float] | None = None
        self._air = np.full(2, 1.0)
        self._feet_prev = np.zeros(2, dtype=bool)
        self.trace = Trace()
        self.last_step: Step | None = None
        self.skill_log = self.port.controller.events

    # ── ids ──
    def _ids(self, mujoco: Any) -> None:
        m = self.model
        robot = self.robot.bodies
        self.robot_geoms = frozenset(
            g
            for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) in robot
            and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
        )
        mesh_names = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(m.nmesh)}
        self.servo_geoms = np.array(
            sorted(
                g
                for g in self.robot_geoms
                if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and mesh_names.get(int(m.geom_dataid[g])) == SV.SERVO_MESH
            ),
            dtype=int,
        )
        self.foot_geoms = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FEET], dtype=int
        )
        if (self.foot_geoms < 0).any():
            raise ValueError("the model has no named foot collision geoms")
        head_bodies = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in HEAD_CONTACT_BODIES
        }
        self.head_geoms = frozenset(
            g for g in self.robot_geoms if int(m.geom_bodyid[g]) in head_bodies
        )
        self.trunk_geoms = frozenset(
            g for g in self.robot_geoms if int(m.geom_bodyid[g]) == self.trunk
        )
        self._servo_set = frozenset(int(g) for g in self.servo_geoms)
        self._servo_index = {int(g): i for i, g in enumerate(self.servo_geoms)}
        self._foot_index = {int(g): i for i, g in enumerate(self.foot_geoms)}
        self.cam_site = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_camera"))
        self._force6 = np.zeros(6)

    # ── the client's side (robotd's surface) ──
    def enable(self) -> dict[str, Any]:
        d = self.data
        self.released = True
        joints = np.array(d.qpos[self.jq])
        legs = list(LEGS14)
        seated = float(np.mean(np.abs(joints[legs] - np.array(HOME14)[legs]))) > SEATED_BOOT_RAD
        self._lag.clear()
        if self.servo_name == "mjcf":
            d.ctrl[self.act] = joints
        if seated:
            self.port.boot(seated=True)
            how = "seated_boot_rise"
        else:
            self._homing = (joints, self.now())
            how = "home_ramp"
        self._log("enable", bringup=how)
        return {"accepted": True, "bringup": how}

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> dict[str, Any]:
        self.port.move(vx, vy, vyaw, self.now())
        return {"accepted": True}

    def head(self, offsets: Sequence[float]) -> dict[str, Any]:
        self.port.head(offsets)
        return {"accepted": True}

    def pose(self, z: float, roll: float, pitch: float, active: bool = True) -> dict[str, Any]:
        self.port.pose(z, roll, pitch, active)
        return {"accepted": True}

    def do(self, skill: str) -> dict[str, Any]:
        ok, why = self.port.request(skill)
        self._log("do", skill=skill, accepted=ok, why=why, policy_before=self.policy)
        return {"accepted": ok, **({} if ok else {"reason": why})}

    def stop(self) -> dict[str, Any]:
        self.port.move(0.0, 0.0, 0.0, self.now())
        return {"accepted": True}

    # ── stepping ──
    def tick(self) -> None:
        import mujoco

        m, d = self.model, self.data
        step: Step | None = None
        targets = np.array(d.qpos[self.jq])
        if self.released:
            now = self.now()
            q = d.qpos[self.qadr + 3 : self.qadr + 7]
            self.safety.observe(gravity_in_trunk(q)[2], TICK_S)
            if self._homing is not None:
                start, since = self._homing
                frac = (now - since) / HOME_RAMP_S
                if frac < 1.0:
                    targets = start + (np.array(HOME14) - start) * frac
                    self.policy = "homing"
                else:
                    self._homing = None
                    self.port.boot(seated=False)
            if self._homing is None:
                gyro, grav, jq, jv = self.obs_path(
                    np.array(d.qvel[self.vadr + 3 : self.vadr + 6]),
                    np.array(gravity_in_trunk(q)),
                    np.array(d.qpos[self.jq]),
                    np.array(d.qvel[self.jv]),
                )
                step = self.port.tick(Sensors(gyro, grav, jq, jv), now, self.battery_v)
                self.servo.set_gain(m, step.gain)
                targets = step.targets
                self.policy = step.label
            self._targets = targets
            if not self._lag:
                for _ in range(self.lag_steps + 1):
                    self._lag.append(np.array(d.qpos[self.jq]))
        tau_pk = np.zeros(14)
        forces = np.zeros(4)  # servo, head, trunk, other (net force norm, peak over substeps)
        housings = np.zeros(len(self.servo_geoms))  # each housing's peak floor force
        td = np.zeros(2)
        for _ in range(STEPS_PER_TICK):
            if self.released:
                self._lag.append(targets)
                self.servo.before_step(m, d, self._delayed())
            mujoco.mj_step(m, d)
            if not self.released:
                self._restore()
            np.maximum(tau_pk, np.abs(d.actuator_force[self.act]), out=tau_pk)
            feet, sub, each = self._contacts(mujoco)
            np.maximum(forces, sub, out=forces)
            np.maximum(housings, each, out=housings)
            for i in range(2):
                if feet[i]:
                    if not self._feet_prev[i] and self._air[i] >= MIN_AIR_S - 1e-9:
                        td[i] += 1
                    self._air[i] = 0.0
                else:
                    self._air[i] += TIMESTEP_S
            self._feet_prev = feet
        self._ticks += 1
        self.last_step = step
        self._record(step, tau_pk, forces, td, housings)

    def _delayed(self) -> np.ndarray:
        """The joint targets the servos act on this physics step: the ones sent
        :attr:`lag_steps` steps ago (a fixed delay; a subclass may redraw it every step, as
        training's ``DelayBuffer`` does — :mod:`duckmind.skilltrain.gt_latency`)."""
        return self._lag[0]

    def _contacts(self, mujoco: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Feet in contact; the net floor force on the housings, head, trunk and anything else
        (N); and each XL330 housing's own floor force (N), in :attr:`servo_geoms` order."""
        m, d = self.model, self.data
        feet = np.zeros(2, dtype=bool)
        sums = np.zeros((4, 3))
        each = np.zeros((len(self.servo_geoms), 3))
        robot = self.robot_geoms
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            in1, in2 = g1 in robot, g2 in robot
            if in1 == in2:
                continue
            mine = g1 if in1 else g2
            foot = self._foot_index.get(mine)
            if foot is not None:
                feet[foot] = True
                continue
            mujoco.mj_contactForce(m, d, i, self._force6)
            frame = np.asarray(c.frame).reshape(3, 3)
            world = frame.T @ self._force6[:3]
            if mine == g1:
                world = -world  # the normal points from geom1 to geom2
            if mine in self._servo_set:
                sums[0] += world
                each[self._servo_index[mine]] += world
            if mine in self.head_geoms:
                sums[1] += world
            elif mine in self.trunk_geoms:
                sums[2] += world
            if (
                mine not in self._servo_set
                and mine not in self.head_geoms
                and mine not in self.trunk_geoms
            ):
                sums[3] += world
        return feet, np.linalg.norm(sums, axis=1), np.linalg.norm(each, axis=1)

    def _record(
        self,
        step: Step | None,
        tau_pk: np.ndarray,
        forces: np.ndarray,
        td: np.ndarray,
        housings: np.ndarray,
    ) -> None:
        import mujoco

        d = self.data
        quat = np.array(d.qpos[self.qadr + 3 : self.qadr + 7])
        grav = gravity_in_trunk(quat.tolist())
        trunk_rot = np.asarray(d.xmat[self.trunk]).reshape(3, 3)
        beak = (
            trunk_rot.T @ (np.asarray(d.site_xpos[self.beak_site]) - np.asarray(d.xpos[self.trunk]))
            if self.beak_site >= 0
            else np.zeros(3)
        )
        cam_m = np.asarray(d.site_xmat[self.cam_site]) if self.cam_site >= 0 else np.zeros(9)
        cam_pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(cam_m[6])))))
        port = self.port
        cmd = step.command if step is not None else None
        cam_vel = np.zeros(6)
        if self.cam_site >= 0:
            mujoco.mj_objectVelocity(
                self.model, d, mujoco.mjtObj.mjOBJ_SITE, self.cam_site, cam_vel, 0
            )
        foot_z = np.array([d.geom_xpos[g][2] for g in self.foot_geoms], dtype=float)
        sample = {
            "t": float(d.time),
            "tilt": math.degrees(math.acos(max(-1.0, min(1.0, -grav[2])))),
            "gz": float(grav[2]),
            "cam_pitch": cam_pitch,
            "f_servo": float(forces[0]),
            "f_head": float(forces[1]),
            "f_trunk": float(forces[2]),
            "f_other": float(forces[3]),
            "f_housing": housings.copy(),
            "fallen": float(self.safety.fallen),
            "gain": float(step.gain) if step is not None else float("nan"),
            "vin": float(getattr(self.servo, "vin_now", self.battery_v)),
            "busy": float(step.busy) if step is not None else 0.0,
            "pos": np.array(d.qpos[self.qadr : self.qadr + 3]),
            "quat": quat,
            "vel": np.array(d.qvel[self.vadr : self.vadr + 3]),
            "angvel": np.array(d.qvel[self.vadr + 3 : self.vadr + 6]),
            "q": np.array(d.qpos[self.jq]),
            "qd": np.array(d.qvel[self.jv]),
            "qacc": np.array(d.qacc[self.jv]),
            "target": np.array(self._targets, dtype=float),
            "tau_pk": tau_pk.copy(),
            "feet": self._feet_prev.astype(float),
            "td": td,
            "beak": beak,
            "cam": np.asarray(d.site_xpos[self.cam_site]).copy()
            if self.cam_site >= 0
            else np.zeros(3),
            "twist_cmd": np.array(cmd.twist if cmd is not None else (0.0, 0.0, 0.0)),
            "head_cmd": np.array(port.head_ema),
            "body_cmd": np.array(port.body_ema),
            "cam_w": cam_vel[:3].copy(),
            "foot_z": foot_z,
        }
        self.trace.append(sample, self.policy)

    def info(self) -> dict[str, Any]:
        return {
            "servo": self.servo.info(),
            "physics": self.physics,
            "lag_steps": self.lag_steps,
            "head_alpha": self.head_alpha,
            "battery_v": round(float(self.battery_v), 3),
            "servo_housings": int(len(self.servo_geoms)),
            "scene": SCENE,
            "skills": [s.name for s in self.skill_defs],
        }


__all__ = ["SkillDuck", "Trace"]
