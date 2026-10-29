"""duck-eye's own camera model (spec §4.3 step 7, §11.2): K and the lens, scaled to the frame.

The intrinsics belong in duck-eye's configuration, not in the camera daemon: mediad hard-codes
none that a client can read, and the eye may not import the simulator (:mod:`duckmind.world`),
so this module carries its own copy of the family calibration — ``robotd-params``
``CameraIntrinsics::alpha()``, solved on the unrotated 1280×720 frame as mediad sends it — and a
test pins it to :mod:`duckmind.world.camera_model`'s numbers. Perception code never writes a
pixel constant: it asks :meth:`EyeCamera.for_frame` for the K of the frame it holds (the header's
width, height and rotate), as mediad scales a solve for another ``[media] quality``.

Frames and conventions (robotd's):

* **sensor frame** — the OpenCV camera frame of the sensor as mounted: +x right and +y down *in
  the unrotated frame*, +z along the optical axis. K and the distortion are stated in it, and so
  is ``RobotState.frames.camera`` (sensor → trunk).
* **upright picture** — the frame turned ``rotate`` degrees clockwise (``uyvy::Turn``; 90 on the
  duck). Pixel ``(u, v)``: u to the right, v down, integer pixel centres. Everything the eye
  samples (:mod:`duckmind.eye.frames`) is upright.
* **trunk frame** — x forward, y left, z up. **level frame** — the trunk frame with the tilt taken
  out: origin on the floor under the trunk origin, z up against gravity, x the trunk's forward
  direction on the floor. Ground points, bearings (positive to the left) and elevations
  (positive up) are stated in it.

:func:`ray` and :func:`project` are exact inverses up to the Newton undistortion (1e-9 px).
Integer outputs (mm, mrad) come from :meth:`EyeCamera.ground_mm` and :meth:`EyeCamera.bearing_mrad`.
Pure standard library: works on floats, no numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# robotd-params CameraIntrinsics::alpha(): unit graphite, 2026-09-08, ChArUco, 80 views,
# 0.79 px RMS, the 1280×720 frame as mediad sends it (unrotated). Pinned to
# duckmind.world.camera_model.FAMILY by tests/test_eye_camera.py.
FAMILY_SIZE = (1280, 720)
FAMILY_K = (1061.8060025020175, 1062.193713957031, 596.7776848217006, 474.52348043048636)
# k1 k2 p1 p2 k3, OpenCV's order, of the unrotated frame (owner decision F1).
FAMILY_DISTORTION = (
    -0.34695388691888,
    0.14615866153945595,
    -0.0007449157712424215,
    -0.0022167170528743837,
    -0.016437266141048814,
)
ROTATE = 90  # mediad --rotate default
PROVISIONAL = "family calibration; the duck's own lens is measured on the board (phase 8)"

Quat = tuple[float, float, float, float]
Vec = tuple[float, float, float]
IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


# ── rotations (scalar-first quaternions, robotd's order) ──────────────────────


def q_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def q_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def q_normalized(q: Sequence[float]) -> Quat:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-12:
        return IDENTITY
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def q_rotate(q: Quat, v: Sequence[float]) -> Vec:
    """``q`` applied to vector ``v``: from the quaternion's source frame to its target frame."""
    w, x, y, z = q
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # v + 2w(q×v) + 2 q×(q×v)
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    ccx = y * cz - z * cy
    ccy = z * cx - x * cz
    ccz = x * cy - y * cx
    return (vx + 2 * (w * cx + ccx), vy + 2 * (w * cy + ccy), vz + 2 * (w * cz + ccz))


def q_angle(a: Quat, b: Quat) -> float:
    """The rotation between two orientations, radians (0 … π)."""
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def q_slerp(a: Quat, b: Quat, f: float) -> Quat:
    """Spherical interpolation, ``f`` = 0 at ``a`` and 1 at ``b`` (shortest way round)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot
    if dot > 0.9995:
        return q_normalized([x + f * (y - x) for x, y in zip(a, b, strict=True)])
    theta = math.acos(dot)
    s = math.sin(theta)
    wa = math.sin((1.0 - f) * theta) / s
    wb = math.sin(f * theta) / s
    return q_normalized([wa * x + wb * y for x, y in zip(a, b, strict=True)])


def q_about_z(angle: float) -> Quat:
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def level_quat(imu_quat: Quat) -> Quat:
    """Trunk → level frame: the IMU's trunk → world with the world yaw taken out, so the trunk's
    forward direction lies along the level frame's +x (on the floor)."""
    forward = q_rotate(imu_quat, (1.0, 0.0, 0.0))
    if math.hypot(forward[0], forward[1]) < 1e-9:  # pointing straight up or down: keep as is
        return imu_quat
    yaw = math.atan2(forward[1], forward[0])
    return q_normalized(q_mul(q_about_z(-yaw), imu_quat))


# ── the lens: OpenCV's model ───────────────────────────────────────────────


def distort(x: float, y: float, dist: Sequence[float] = FAMILY_DISTORTION) -> tuple[float, float]:
    """Ideal normalised coordinates → where the lens puts them (OpenCV's model)."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return (
        x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    )


def undistort(
    xd: float, yd: float, dist: Sequence[float] = FAMILY_DISTORTION, iterations: int = 10
) -> tuple[float, float]:
    """Distorted normalised coordinates → ideal ones (Newton on :func:`distort`)."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        slope = k1 + r2 * (2 * k2 + r2 * 3 * k3)
        ex = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        ey = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        a = radial + 2 * x * x * slope + 2 * p1 * y + 6 * p2 * x
        b = 2 * x * y * slope + 2 * p1 * x + 2 * p2 * y
        d = radial + 2 * y * y * slope + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * b
        if det == 0:
            break
        x -= (d * ex - b * ey) / det
        y -= (a * ey - b * ex) / det
    return x, y


# ── the camera ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is at the capture: ``RobotState.frames.camera`` (sensor → trunk), the
    trunk IMU (trunk → world) and the trunk height from leg FK."""

    cam_pos_m: Vec
    cam_quat: Quat
    imu_quat: Quat = IDENTITY
    trunk_height_m: float = 0.0

    @property
    def level_from_trunk(self) -> Quat:
        return level_quat(self.imu_quat)

    @property
    def sensor_to_level(self) -> Quat:
        return q_normalized(q_mul(self.level_from_trunk, self.cam_quat))

    @property
    def position_level(self) -> Vec:
        """The camera's centre in the level frame (metres)."""
        x, y, z = q_rotate(self.level_from_trunk, self.cam_pos_m)
        return (x, y, z + self.trunk_height_m)


@dataclass(frozen=True)
class EyeCamera:
    """The family K and lens on one frame: ``width``×``height`` as delivered (unrotated),
    turned upright by ``rotate``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = FAMILY_DISTORTION
    rotate: int = ROTATE

    @classmethod
    def for_frame(cls, width: int, height: int, rotate: int = ROTATE) -> EyeCamera:
        """The family calibration scaled to a delivered frame (as ``mediad/src/camera.rs``
        scales a solve: each axis by its own ratio)."""
        if width <= 0 or height <= 0:
            raise ValueError(f"no frame is {width}x{height}")
        if rotate not in (0, 90, 180, 270):
            raise ValueError(f"rotate {rotate} is not a quarter turn")
        sx, sy = width / FAMILY_SIZE[0], height / FAMILY_SIZE[1]
        fx, fy, cx, cy = FAMILY_K
        return cls(width, height, fx * sx, fy * sy, cx * sx, cy * sy, FAMILY_DISTORTION, rotate)

    @classmethod
    def for_header(cls, header: object) -> EyeCamera:
        """From a ``MediaFrameHeader`` (anything with width, height and rotate)."""
        return cls.for_frame(
            int(getattr(header, "width")),  # noqa: B009
            int(getattr(header, "height")),  # noqa: B009
            int(getattr(header, "rotate", ROTATE)),
        )

    # ── the upright picture ─────────────────────────────────────────

    @property
    def upright_size(self) -> tuple[int, int]:
        if self.rotate in (90, 270):
            return self.height, self.width
        return self.width, self.height

    def upright_k(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) of the upright picture (what OpenCV wants for an upright crop)."""
        w, h = self.width, self.height
        if self.rotate == 0:
            return self.fx, self.fy, self.cx, self.cy
        if self.rotate == 90:
            return self.fy, self.fx, h - 1 - self.cy, self.cx
        if self.rotate == 180:
            return self.fx, self.fy, w - 1 - self.cx, h - 1 - self.cy
        return self.fy, self.fx, self.cy, w - 1 - self.cx

    def upright_distortion(self) -> tuple[float, float, float, float, float]:
        """The lens coefficients of the upright picture: radial terms stay, the tangential ones
        turn with the picture."""
        k1, k2, p1, p2, k3 = self.dist
        if self.rotate == 0:
            return (k1, k2, p1, p2, k3)
        if self.rotate == 90:
            return (k1, k2, p2, -p1, k3)
        if self.rotate == 180:
            return (k1, k2, -p1, -p2, k3)
        return (k1, k2, -p2, p1, k3)

    def to_sensor_pixel(self, u: float, v: float) -> tuple[float, float]:
        """Upright pixel → the delivered frame's pixel (the inverse of ``uyvy::Turn``)."""
        w, h = self.width, self.height
        if self.rotate == 0:
            return u, v
        if self.rotate == 90:
            return v, h - 1 - u
        if self.rotate == 180:
            return w - 1 - u, h - 1 - v
        return w - 1 - v, u

    def to_upright_pixel(self, x: float, y: float) -> tuple[float, float]:
        """The delivered frame's pixel → upright (``uyvy::Turn``)."""
        w, h = self.width, self.height
        if self.rotate == 0:
            return x, y
        if self.rotate == 90:
            return h - 1 - y, x
        if self.rotate == 180:
            return w - 1 - x, h - 1 - y
        return y, w - 1 - x

    # ── rays ────────────────────────────────────────────────────────

    def ray(self, u: float, v: float) -> Vec:
        """The ray through upright pixel (u, v), in the sensor frame, with z = 1 (undistorted)."""
        x, y = self.to_sensor_pixel(u, v)
        xn, yn = undistort((x - self.cx) / self.fx, (y - self.cy) / self.fy, self.dist)
        return (xn, yn, 1.0)

    def project(self, point: Sequence[float]) -> tuple[float, float] | None:
        """Where a sensor-frame point lands in the upright picture (``None`` behind the lens)."""
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        if pz <= 1e-9:
            return None
        xd, yd = distort(px / pz, py / pz, self.dist)
        return self.to_upright_pixel(self.cx + self.fx * xd, self.cy + self.fy * yd)

    def angles_rad(self, u: float, v: float) -> tuple[float, float]:
        """Angles of upright pixel (u, v) from the optical axis in the upright picture: (to the
        right, downwards), radians, distortion undone. Needs no pose."""
        xn, yn, _ = self.ray(u, v)
        if self.rotate == 0:
            right, down = xn, yn
        elif self.rotate == 90:
            right, down = -yn, xn
        elif self.rotate == 180:
            right, down = -xn, -yn
        else:
            right, down = yn, -xn
        return math.atan(right), math.atan(down)

    def fov_deg(self) -> dict[str, float]:
        """How far the upright picture reaches either side of the axis through the middle of
        each edge (degrees; distortion undone)."""
        w, h = self.upright_size
        _, _, cxu, cyu = self.upright_k()
        left = -math.degrees(self.angles_rad(-0.5, cyu)[0])
        right = math.degrees(self.angles_rad(w - 0.5, cyu)[0])
        top = -math.degrees(self.angles_rad(cxu, -0.5)[1])
        bottom = math.degrees(self.angles_rad(cxu, h - 0.5)[1])
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    # ── on the floor ────────────────────────────────────────────────

    def ray_level(self, u: float, v: float, pose: CameraPose) -> Vec:
        """The ray through (u, v) in the level frame (unit length)."""
        d = q_rotate(pose.sensor_to_level, self.ray(u, v))
        n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        return (d[0] / n, d[1] / n, d[2] / n)

    def ground_m(self, u: float, v: float, pose: CameraPose) -> tuple[float, float] | None:
        """Where the ray through (u, v) meets the floor, (x forward, y left) in the level frame,
        metres; ``None`` when it does not (at or above the horizon)."""
        px, py, pz = pose.position_level
        dx, dy, dz = self.ray_level(u, v, pose)
        if dz >= -1e-6 or pz <= 0:
            return None
        t = -pz / dz
        return (px + t * dx, py + t * dy)

    def ground_mm(self, u: float, v: float, pose: CameraPose) -> tuple[int, int] | None:
        """:meth:`ground_m` in integer millimetres."""
        g = self.ground_m(u, v, pose)
        return None if g is None else (round(g[0] * 1000), round(g[1] * 1000))

    def bearing_mrad(self, u: float, v: float, pose: CameraPose) -> tuple[int, int]:
        """(bearing, elevation) of the ray through (u, v) in the level frame: bearing from the
        trunk's forward direction, positive to the left; elevation positive up; mrad."""
        dx, dy, dz = self.ray_level(u, v, pose)
        return (
            round(math.atan2(dy, dx) * 1000),
            round(math.atan2(dz, math.hypot(dx, dy)) * 1000),
        )

    def pixel_of_ground(
        self, x_m: float, y_m: float, pose: CameraPose, z_m: float = 0.0
    ) -> tuple[float, float] | None:
        """Where a level-frame point (on the floor unless ``z_m``) lands in the upright picture."""
        px, py, pz = pose.position_level
        rel = (x_m - px, y_m - py, z_m - pz)
        return self.project(q_rotate(q_conj(pose.sensor_to_level), rel))

    def nearest_floor_m(self, pose: CameraPose) -> float | None:
        """How close to the trunk origin (level frame, straight ahead of the picture's bottom
        middle) the camera sees the floor."""
        w, h = self.upright_size
        _, _, cxu, _ = self.upright_k()
        g = self.ground_m(cxu, h - 0.5, pose)
        return None if g is None else math.hypot(g[0], g[1])

    def config_digest_items(self) -> list[str]:
        """What goes into the perception hash for the camera (spec §7.6: K + distortion)."""
        return [
            f"k={self.width}x{self.height}:{self.fx!r},{self.fy!r},{self.cx!r},{self.cy!r}",
            "dist=" + ",".join(repr(v) for v in self.dist),
            f"rotate={self.rotate}",
        ]
