"""The posture tap (spec §4.3 step 5): every ``robot.state`` tick becomes a pose sample.

duck-body keeps the samples at 50 Hz (a ring of :data:`RING_S` seconds) and marks every other
one for duck-eye (25 Hz, the ``eye.pose`` line). Each sample
(:class:`~duckmind.duckbody.protocol.PoseSample`) carries what the eye needs to put a picture on
the floor and what the body's look rules need:

* **trunk height from leg FK** (:class:`LegFk`): both feet's sole points in the trunk frame from
  the measured leg joints (the official MJCF's ``left_foot``/``right_foot`` sites, read by
  :class:`~duckmind.duckbody.tof_reproject.Tree`, no numpy), projected on gravity (robotd's
  ``safety.gravity`` or the IMU quaternion); the lower sole is the floor. Calibrated on the
  headless twin against the true trunk height (``SOLE_OFFSET_M``);
* **camera_in_trunk**: robotd's ``frames.camera`` when the daemon publishes it (v24+), else the
  official head FK (:class:`~duckmind.duckbody.tof_reproject.HeadFk`) at the measured head
  joints — never the commanded ones;
* **still** (spec §4.2 rule 2): head joints under 0.2 rad/s and trunk gyro under 0.3 rad/s,
  held 150 ms;
* the odometry pose and its **epoch** (bumped by the caller when the frame breaks: a lift, a
  robotd restart), the policy, the gyro and the IMU quaternion.

Pure and clock-free: the tick's own ``t_ns`` is the only time.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from duckmind.duckbody.protocol import PoseSample
from duckmind.duckbody.robotd_client import JOINT_NAMES, RobotState
from duckmind.duckbody.tof_reproject import HeadFk, Tree

RING_S = 10.0
STILL_HEAD = 0.2  # rad/s
STILL_GYRO = 0.3  # rad/s
STILL_HOLD_NS = 150_000_000
# Calibrated on the headless twin against the true trunk height (flat floor, 4 ducks; raw in
# the measurement records, not in the repo): the MJCF foot sites are on the soles —
# standing −0.3 mm (n 400), walking median 0.0 mm (p5 −1.4, p95 +2.1; n 1000). Seated, the duck
# rests on its seat and the soles are off the floor: the true height is 22.9 mm more (n 400).
SOLE_OFFSET_M = 0.0
SEATED_EXTRA_M = 0.0229
FOOT_SITES = ("left_foot", "right_foot")


class LegFk:
    """Trunk height above the floor from the measured leg joints and gravity."""

    def __init__(self, tree: Tree, sole_offset_m: float = SOLE_OFFSET_M) -> None:
        self.tree = tree
        self.sole_offset_m = sole_offset_m
        feet = [tree.site(name) for name in FOOT_SITES]
        if any(f is None for f in feet):
            raise ValueError("the model has no left_foot/right_foot sites")
        self.feet = [int(f) for f in feet if f is not None]
        # robotd wire index for every MJCF joint (the mouth is not in the tree)
        self.wire_of = []
        for name in tree.joint_names:
            if name not in JOINT_NAMES:
                raise ValueError(f"MJCF joint {name!r} is not a robotd joint")
            self.wire_of.append(JOINT_NAMES.index(name))

    @classmethod
    def from_mjcf(cls, source: Path | str, sole_offset_m: float = SOLE_OFFSET_M) -> LegFk:
        return cls(Tree.from_mjcf(source), sole_offset_m)

    def feet_in_trunk(self, joints: Sequence[float]) -> list[tuple[float, float, float]]:
        angles = [float(joints[i]) for i in self.wire_of]
        return [self.tree.site_pose(site, angles).pos for site in self.feet]

    def height_m(
        self, joints: Sequence[float], down: Sequence[float], *, seated: bool = False
    ) -> float:
        """How far the trunk origin is above the floor along ``down`` (unit, trunk frame): the
        lower sole, or the seat when ``seated``."""
        depths = [
            p[0] * down[0] + p[1] * down[1] + p[2] * down[2] for p in self.feet_in_trunk(joints)
        ]
        return max(depths) + self.sole_offset_m + (SEATED_EXTRA_M if seated else 0.0)


@dataclass
class _Still:
    since: int | None = None


class PostureTap:
    """Samples from ticks: ``step(state, epoch)`` → the new :class:`PoseSample`."""

    def __init__(self, legs: LegFk | None = None, head: HeadFk | None = None) -> None:
        self.legs = legs
        self.head_fk = head
        self.seq = 0
        self.ring: deque[PoseSample] = deque()
        self._still = _Still()

    @classmethod
    def from_mjcf(cls, source: Path | str) -> PostureTap:
        tree = Tree.from_mjcf(source)
        return cls(LegFk(tree), HeadFk(tree))

    def step(self, state: RobotState, epoch: int = 0) -> PoseSample:
        t = state.t_ns
        down = state.down
        height: int | None = None
        if self.legs is not None and down is not None:
            seated = state.policy == "sit"
            height = int(round(self.legs.height_m(state.joints, down, seated=seated) * 1000.0))
        elif state.odom_pos[2] > 0.0:
            height = int(round(state.odom_pos[2] * 1000.0))  # robotd's contact-odometry height
        camera_pos = camera_quat = tof_pos = tof_quat = None
        if state.camera is not None:
            camera_pos, camera_quat = state.camera
        elif self.head_fk is not None:
            pose = self.head_fk.camera_in_trunk_cv2(state.head)
            camera_pos = pose.pos
            camera_quat = (pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
        if state.tof is not None:
            tof_pos, tof_quat = state.tof
        elif self.head_fk is not None:
            pose = self.head_fk.tof_in_trunk(state.head)
            tof_pos = pose.pos
            tof_quat = (pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
        head_vel = state.head_speed
        gyro = state.gyro
        gyro_mag = None if gyro is None else math.sqrt(sum(g * g for g in gyro))
        calm = (
            head_vel is not None
            and head_vel < STILL_HEAD
            and gyro_mag is not None
            and gyro_mag < STILL_GYRO
        )
        if calm:
            if self._still.since is None:
                self._still.since = t
        else:
            self._still.since = None
        still = self._still.since is not None and t - self._still.since >= STILL_HOLD_NS
        sample = PoseSample(
            seq=self.seq,
            t_ns=t,
            trunk_height_mm=height,
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            tof_pos=tof_pos,
            tof_quat=tof_quat,
            imu_quat=state.imu_quat,
            gyro=gyro,
            head=state.head,
            head_vel=head_vel,
            odom=(state.odom_pos[0], state.odom_pos[1], state.odom_yaw),
            epoch=epoch,
            still=still,
            policy=state.policy,
        )
        self.seq += 1
        self.ring.append(sample)
        while self.ring and t - self.ring[0].t_ns > int(RING_S * 1e9):
            self.ring.popleft()
        return sample

    @staticmethod
    def for_eye(sample: PoseSample) -> bool:
        """Every other 50 Hz sample goes to duck-eye (25 Hz)."""
        return sample.seq % 2 == 0

    def at(self, t_ns: int) -> PoseSample | None:
        """The sample nearest ``t_ns`` in the ring (``None`` if the ring does not cover it)."""
        if not self.ring or t_ns < self.ring[0].t_ns - 20_000_000:
            return None
        return min(self.ring, key=lambda s: abs(s.t_ns - t_ns))


__all__ = ["LegFk", "PostureTap", "SEATED_EXTRA_M", "SOLE_OFFSET_M"]
