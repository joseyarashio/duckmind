"""A home seen through the world's head camera, for duck-eye's vision tests (not a test file).

The eye may not import the world; its tests may: they build a home (``homelayout`` + the dock's
tags through ``nestpad``), pose the duck kinematically, capture a frame through the full camera
model (lens, AE, white balance, IMX219 noise) and hand the eye exactly what it gets on the robot —
the UYVY bytes, the frame's size and rotation, and the camera pose the pose tap would give
(from the physics' own camera and trunk, never the commanded head).

Render tier: numpy, MuJoCo 3.10, an offscreen GL context and the official checkout
(``DUCKMIND_MICRODUCK_MJCF``); otherwise :func:`scene` skips.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

MOMENTS = {
    "day": datetime(2026, 12, 20, 12, 0),
    "dusk": datetime(2026, 12, 20, 16, 55),
    "lamps": datetime(2026, 12, 20, 20, 0),
    "moon": datetime(2026, 12, 21, 2, 0),
}
GLANCE = {"head_pitch": 0.646}  # spec §4.2 glance_down (−17°)
S90 = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))  # upright camera vectors → sensor (rotate 90)


def _mjcf() -> str:
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")
    if not folder or not Path(folder, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    return folder


def mat_quat(m: Any) -> tuple[float, float, float, float]:
    """A rotation matrix's scalar-first quaternion."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


class Scene:
    """One home, its lights and the head camera (in-process, kinematic poses)."""

    def __init__(self, home: str = "default_home", edit: Any = None, seed: int = 7) -> None:
        folder = _mjcf()
        import mujoco

        from duckmind.world import camera_model as cm
        from duckmind.world import homelayout, lights, nestpad, render_scene

        self.cm = cm
        self.lights = lights
        self.layout = homelayout.load(home, mjcf_dir=folder)
        spec = homelayout.compose(self.layout, mjcf_dir=folder)
        nestpad.NestPad().edit_spec(spec, self.layout)
        if edit is not None:
            edit(spec)
        homelayout.fix_keyframes(spec, self.layout)
        spec.compile()
        self.home = lights.HomeLights.from_layout(self.layout)
        self.eye = render_scene.HomeEye.open(
            spec, home=self.home, config=cm.CameraConfig(quality="720p30"), seed=seed
        )
        self.m = self.eye.eye.model
        self.d = self.eye.eye.data
        self.mujoco = mujoco
        self.trunk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.n = 0

    def state(self, moment: str) -> Any:
        return self.lights.home_light_state(MOMENTS[moment], self.home)

    def geom(self, name: str) -> int:
        gid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            pytest.skip(f"this home has no {name}")
        return int(gid)

    def pose(
        self, x: float, y: float, heading: float, joints: dict[str, float] | None = None
    ) -> None:
        """The trunk at (x, y) facing ``heading`` (its forward axis, world radians)."""
        base = self.cm.keyframe_pose(self.eye.eye, "STAND")
        w0, x0, y0, z0 = (float(v) for v in base["quat"])
        yaw0 = math.atan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0 * y0 + z0 * z0))
        turn = heading - yaw0
        frame = dict(base)
        frame["trunk"] = [x, y, base["trunk"][2]]
        frame["quat"] = self.cm._quat_mul(
            [math.cos(turn / 2), 0.0, 0.0, math.sin(turn / 2)], list(base["quat"])
        )
        if joints:
            j = dict(base["joints"])
            j.update(joints)
            frame["joints"] = j
        self.eye.set_pose(frame)
        self.mujoco.mj_forward(self.m, self.d)

    def camera_world(self) -> tuple[Any, Any]:
        """Camera position, and the upright OpenCV axes (right, down, forward) in the world."""
        import numpy as np

        cam = self.eye.eye.camera
        pos = np.array(self.d.cam_xpos[cam])
        r = np.array(self.d.cam_xmat[cam]).reshape(3, 3)
        return pos, np.stack([r[:, 1], r[:, 0], -r[:, 2]], axis=1)

    def pose_camera_at(self, cx: float, cy: float, heading: float, joints: Any = None) -> None:
        self.pose(cx, cy, heading, joints)
        pos, _ = self.camera_world()
        self.pose(cx + (cx - pos[0]), cy + (cy - pos[1]), heading, joints)

    def trunk_xy_yaw(self) -> tuple[float, float, float]:
        import numpy as np

        rt = np.array(self.d.xmat[self.trunk]).reshape(3, 3)
        tp = np.array(self.d.xpos[self.trunk])
        return float(tp[0]), float(tp[1]), math.atan2(rt[1, 0], rt[0, 0])

    def to_level(self, wx: float, wy: float) -> tuple[float, float]:
        x0, y0, yaw = self.trunk_xy_yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = wx - x0, wy - y0
        return c * dx + s * dy, -s * dx + c * dy

    def to_world(self, lx: float, ly: float) -> tuple[float, float]:
        x0, y0, yaw = self.trunk_xy_yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        return x0 + c * lx - s * ly, y0 + s * lx + c * ly

    def camera_pose(self) -> Any:
        """What the pose tap gives the eye: the camera (sensor frame) in the trunk, the trunk
        IMU and the trunk height — here from the physics' own bodies."""
        import numpy as np

        from duckmind.eye.camera import CameraPose

        pos, r_up = self.camera_world()
        rt = np.array(self.d.xmat[self.trunk]).reshape(3, 3)
        tp = np.array(self.d.xpos[self.trunk])
        r_ws = r_up @ np.array(S90, dtype=float).T
        return CameraPose(
            cam_pos_m=tuple(float(v) for v in rt.T @ (pos - tp)),
            cam_quat=mat_quat((rt.T @ r_ws).tolist()),
            imu_quat=mat_quat(rt.tolist()),
            trunk_height_m=float(tp[2]),
        )

    def capture(self, moment: str) -> Any:
        st = self.state(moment)
        self.eye.converge(st)
        cap = self.eye.capture(st, index=self.n)
        self.n += 1
        return cap

    def seg_upright(self) -> Any:
        import numpy as np

        w, h = self.eye.frame_size
        return np.ascontiguousarray(np.rot90(self.eye.camera.renderer.segment_ids(w, h), k=-1))

    def close(self) -> None:
        self.eye.close()


def eye_inputs(cap: Any) -> tuple[Any, Any, Any]:
    """(planes, camera, luma_of) from one capture, as duck-eye's server builds them."""
    from duckmind.eye import frames
    from duckmind.eye.camera import EyeCamera

    data, w, h, rot = cap.uyvy(), cap.width, cap.height, cap.rotate

    def luma_of(box: Any, step: int) -> Any:
        if box is None:
            return frames.upright_luma(data, w, h, rot, step=step)
        return frames.roi_luma(data, w, h, rot, box, step=step)

    return frames.sample_quarter(data, w, h, rot), EyeCamera.for_frame(w, h, rot), luma_of


def add_box(
    spec: Any, name: str = "eyetest_box", size_m: float = 0.10, height_m: float = 0.06
) -> None:
    """A mocap box (no collision: only the camera sees it), parked below the floor."""
    import mujoco

    body = spec.worldbody.add_body()
    body.name = name
    body.mocap = True
    body.pos = [0, 0, -2]
    g = body.add_geom()
    g.name = name + "_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [size_m / 2, size_m / 2, height_m / 2]
    g.pos = [0, 0, height_m / 2]
    g.rgba = [0.8, 0.15, 0.12, 1]
    g.contype = 0
    g.conaffinity = 0
    g.group = 1
