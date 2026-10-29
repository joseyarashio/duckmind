"""Synthetic scenes for duck-eye's vision tests: a ray-cast floor, wall, boxes and an AprilTag,
seen through duck-eye's own camera model (K, lens, upright turn) from a known camera pose.

Everything is numpy and deterministic (fixed noise seeds), so a test knows the truth exactly:
where the floor ends, how far a box is, where the tag's corners land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from duckmind.eye import tag36h11
from duckmind.eye.camera import CameraPose, EyeCamera
from duckmind.eye.floorline import _plane_rays, level_from_upright, upright_to_sensor
from duckmind.eye.frames import QuarterPlanes

CAMERA = EyeCamera.for_frame(1280, 720, 90)

# colours as (Y, U, V)
WOOD = (95, 112, 140)
SEAM = (60, 114, 138)
WALL = (175, 127, 129)
RED = (70, 110, 190)
CYAN = (140, 150, 70)


def mat_quat(m: Any) -> tuple[float, float, float, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def level_pose(height: float = 0.25, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> CameraPose:
    """A camera ``height`` m above the floor, 6 cm ahead of the trunk origin, pitched down by
    ``pitch_deg`` and turned left by ``yaw_deg``; the trunk level (the IMU at identity)."""
    th, ps = math.radians(pitch_deg), math.radians(yaw_deg)
    forward = np.array([math.cos(th) * math.cos(ps), math.cos(th) * math.sin(ps), -math.sin(th)])
    right = np.array([math.sin(ps), -math.cos(ps), 0.0])
    down = np.cross(forward, right)
    r_level_upright = np.stack([right, down, forward], axis=1)
    r_level_sensor = r_level_upright @ upright_to_sensor(90).T
    trunk_h = 0.12
    return CameraPose(
        cam_pos_m=(0.06, 0.0, height - trunk_h),
        cam_quat=mat_quat(r_level_sensor),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        trunk_height_m=trunk_h,
    )


@dataclass
class Box:
    x: float  # the front face, level frame (m)
    y: float  # its middle
    w: float = 0.10
    d: float = 0.10
    h: float = 0.06
    colour: tuple[int, int, int] = RED


@dataclass
class Scene:
    wall_x: float = 1.2  # a wall across the view at this x (m)
    seams_every: float = 0.0  # dark plank seams across the floor every this far (0: none)
    boxes: list[Box] = field(default_factory=list)


def _cast(rays_level: Any, pos: Any, scene: Scene) -> tuple[Any, Any, Any]:
    """(colour index per ray, range to the first hit, floor x of floor hits) for unit-free rays
    in the level frame: 0 floor, 1 seam, 2 wall, 3+i box i."""
    dx, dy, dz = rays_level[..., 0], rays_level[..., 1], rays_level[..., 2]
    shape = dx.shape
    best_t = np.full(shape, np.inf)
    kind = np.full(shape, 2, dtype=np.int32)
    # the floor
    down = dz < -1e-9
    t_floor = np.where(down, -pos[2] / np.where(down, dz, -1.0), np.inf)
    fx = pos[0] + t_floor * dx
    best_t = np.where(down, t_floor, best_t)
    kind = np.where(down, 0, kind)
    if scene.seams_every > 0:
        phase = np.mod(fx, scene.seams_every)
        seam = down & (phase < 0.008)
        kind = np.where(seam, 1, kind)
    # the wall (a plane x = wall_x)
    fwd = dx > 1e-9
    t_wall = np.where(fwd, (scene.wall_x - pos[0]) / np.where(fwd, dx, 1.0), np.inf)
    hit = t_wall < best_t
    best_t = np.where(hit, t_wall, best_t)
    kind = np.where(hit, 2, kind)
    # boxes: front face, top
    for i, b in enumerate(scene.boxes):
        t_face = np.where(fwd, (b.x - pos[0]) / np.where(fwd, dx, 1.0), np.inf)
        z = pos[2] + t_face * dz
        y = pos[1] + t_face * dy
        on_face = (t_face > 0) & (z >= 0) & (z <= b.h) & (np.abs(y - b.y) <= b.w / 2)
        t_top = np.where(dz < -1e-9, (b.h - pos[2]) / np.where(dz < -1e-9, dz, -1.0), np.inf)
        xt = pos[0] + t_top * dx
        yt = pos[1] + t_top * dy
        on_top = (t_top > 0) & (xt >= b.x) & (xt <= b.x + b.d) & (np.abs(yt - b.y) <= b.w / 2)
        t_box = np.minimum(np.where(on_face, t_face, np.inf), np.where(on_top, t_top, np.inf))
        hit = t_box < best_t
        best_t = np.where(hit, t_box, best_t)
        kind = np.where(hit, 3 + i, kind)
    return kind, best_t, fx


def _palette(scene: Scene) -> Any:
    cols = [WOOD, SEAM, WALL] + [b.colour for b in scene.boxes]
    return np.array(cols, dtype=np.float64)


def render_planes(
    scene: Scene, pose: CameraPose, camera: EyeCamera = CAMERA, seed: int = 0, noise: float = 2.0
) -> QuarterPlanes:
    """The quarter-resolution planes (320×180 luma, 160×90 chroma) of the scene."""
    rng = np.random.default_rng(seed)
    rows, cols = camera.upright_size[1] // 4, camera.upright_size[0] // 4
    rays = _plane_rays(camera, 4, rows, cols)
    rot, pos = level_from_upright(camera, pose)
    ones = np.ones(rays.shape[:2])
    vec = np.stack([rays[..., 0], rays[..., 1], ones], axis=-1) @ rot.T
    kind, _, _ = _cast(vec, pos, scene)
    pal = _palette(scene)
    yuv = pal[kind]
    yuv[..., 0] += rng.normal(0.0, noise, kind.shape)
    y = np.clip(np.rint(yuv[..., 0]), 0, 255).astype(np.uint8)
    u = np.clip(np.rint(yuv[::2, ::2, 1] + rng.normal(0, noise / 2, kind[::2, ::2].shape)), 0, 255)
    v = np.clip(np.rint(yuv[::2, ::2, 2] + rng.normal(0, noise / 2, kind[::2, ::2].shape)), 0, 255)
    return QuarterPlanes(y=y, u=u.astype(np.uint8), v=v.astype(np.uint8))


# ── a tag on a wall ──────────────────────────────────────────────────────────


@dataclass
class Tag:
    tag_id: int = tag36h11.NEST_TAG_ID
    x: float = 1.5  # the tag's plane (facing −x, toward the camera), level frame
    y: float = 0.0
    z: float = 0.112  # its centre's height
    size: float = 0.100  # the black square
    flip_bit: int | None = None  # print it with one data bit wrong


def tag_cells(tag: Tag) -> Any:
    cells = np.array(tag36h11.grid(tag.tag_id), dtype=np.int32)  # 10×10 with the white ring
    if tag.flip_bit is not None:
        x, y = tag36h11.BIT_X[tag.flip_bit], tag36h11.BIT_Y[tag.flip_bit]
        cells[y + 1, x + 1] ^= 1
    return cells


def tag_corners(tag: Tag, pose: CameraPose, camera: EyeCamera = CAMERA) -> Any:
    """The black square's corners in the upright picture: the tag's bottom-left, bottom-right,
    top-right, top-left (AprilTag's order)."""
    s = tag.size / 2
    # the printed x axis is −y (the viewer's right), the printed up axis is +z
    pts = [(-s, -s), (s, -s), (s, s), (-s, s)]  # (printed x, printed up)
    out = []
    for px, up in pts:
        p = camera.pixel_of_ground(tag.x, tag.y - px, pose, z_m=tag.z + up)
        assert p is not None
        out.append(p)
    return np.array(out, dtype=np.float64)


def render_luma(
    tag: Tag,
    pose: CameraPose,
    camera: EyeCamera = CAMERA,
    background: int = 120,
    seed: int = 0,
    noise: float = 1.5,
    black: int = 30,
    white: int = 200,
) -> Any:
    """The full-resolution upright luma of a grey wall carrying the tag (2×2 supersampled)."""
    rng = np.random.default_rng(seed)
    width, height = camera.upright_size
    corners = tag_corners(tag, pose, camera)
    u0 = max(0, int(corners[:, 0].min()) - 20)
    u1 = min(width, int(corners[:, 0].max()) + 21)
    v0 = max(0, int(corners[:, 1].min()) - 20)
    v1 = min(height, int(corners[:, 1].max()) + 21)
    img = np.full((height, width), float(background))
    rot, pos = level_from_upright(camera, pose)
    fx, fy, cx, cy = camera.upright_k()
    from duckmind.eye.floorline import _undistort_np

    cells = tag_cells(tag)
    cell = tag.size / 8
    acc = np.zeros((v1 - v0, u1 - u0))
    vv, uu = np.mgrid[v0:v1, u0:u1].astype(np.float64)
    for du in (-0.25, 0.25):
        for dv in (-0.25, 0.25):
            xn, yn = _undistort_np(
                (uu + du - cx) / fx, (vv + dv - cy) / fy, camera.upright_distortion()
            )
            d = np.stack([xn, yn, np.ones_like(xn)], axis=-1) @ rot.T
            t = (tag.x - pos[0]) / d[..., 0]
            py = pos[1] + t * d[..., 1]
            pz = pos[2] + t * d[..., 2]
            gx = (tag.y - py) / cell + 5.0  # printed x in cells, 0…10 with the white ring
            gy = (tag.z - pz) / cell + 5.0  # printed y (down) in cells
            inside = (gx >= 0) & (gx < 10) & (gy >= 0) & (gy < 10)
            val = np.full(gx.shape, float(background))
            ci = np.clip(gx.astype(int), 0, 9)
            cj = np.clip(gy.astype(int), 0, 9)
            val = np.where(inside, np.where(cells[cj, ci] == 1, white, black), val)
            acc += val
    img[v0:v1, u0:u1] = acc / 4.0
    img += rng.normal(0.0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def luma_of(full: Any) -> Any:
    """The ``luma_of(box, step)`` a NestTask reads with (the whole picture or an ROI)."""

    def get(box: tuple[int, int, int, int] | None, step: int) -> Any:
        if box is None:
            return np.ascontiguousarray(full[::step, ::step])
        u0, v0, u1, v1 = box
        return np.ascontiguousarray(full[v0:v1:step, u0:u1:step])

    return get


def quarter_of(full_luma: Any, u: int = 128, v: int = 128) -> QuarterPlanes:
    """Quarter-resolution planes of a luma picture with flat chroma."""
    y = np.ascontiguousarray(full_luma[::4, ::4])
    rows, cols = y.shape
    return QuarterPlanes(
        y=y,
        u=np.full((rows // 2, cols // 2), u, dtype=np.uint8),
        v=np.full((rows // 2, cols // 2), v, dtype=np.uint8),
    )
