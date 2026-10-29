"""eye.camera: duck-eye's own K and lens equal the world's family numbers, scaled to the frame,
and its rays and ground points are right (spec §4.3 step 7, §11.2)."""

from __future__ import annotations

import math
import random

import pytest

from duckmind.eye import camera as ec
from duckmind.world import camera_model as cm


def test_the_eye_carries_the_family_calibration_of_the_world():
    assert ec.FAMILY_SIZE == (cm.FAMILY.width, cm.FAMILY.height)
    assert ec.FAMILY_K == (cm.FAMILY.fx, cm.FAMILY.fy, cm.FAMILY.cx, cm.FAMILY.cy)
    assert ec.FAMILY_DISTORTION == cm.FAMILY_DISTORTION
    assert ec.ROTATE == cm.ROTATE_DEG


@pytest.mark.parametrize("label", list(cm.QUALITIES))
def test_k_scales_with_the_frame_like_every_media_quality(label):
    q = cm.QUALITIES[label]
    cam = ec.EyeCamera.for_frame(q.width, q.height, 90)
    k = q.k
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == pytest.approx((k.fx, k.fy, k.cx, k.cy))
    up = k.upright()
    assert cam.upright_size == (up.width, up.height)
    assert cam.upright_k() == pytest.approx((up.fx, up.fy, up.cx, up.cy))
    assert cam.upright_distortion() == pytest.approx(cm.upright_distortion())


def test_rays_match_the_world_s_undistortion_and_invert_projection():
    rng = random.Random(3)
    cam = ec.EyeCamera.for_frame(1280, 720, 90)
    for _ in range(50):
        u, v = rng.uniform(0, 719), rng.uniform(0, 1279)
        x, y = cam.to_sensor_pixel(u, v)
        assert cam.to_upright_pixel(x, y) == pytest.approx((u, v))
        ray = cam.ray(u, v)
        assert ray[:2] == pytest.approx(cm.undistort_pixel(x, y), abs=1e-9)
        depth = rng.uniform(0.2, 4.0)
        assert cam.project((ray[0] * depth, ray[1] * depth, depth)) == pytest.approx(
            (u, v), abs=1e-6
        )
    assert cam.project((0.0, 0.0, -1.0)) is None


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_every_quarter_turn_is_consistent(rotate):
    cam = ec.EyeCamera.for_frame(640, 360, rotate)
    w, h = cam.upright_size
    fx, fy, cx, cy = cam.upright_k()
    dist = cam.upright_distortion()
    rng = random.Random(rotate)
    for _ in range(20):
        u, v = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        right, down = cam.angles_rad(u, v)
        # the upright K and lens give the same angles as the sensor's, turned
        xn, yn = ec.undistort((u - cx) / fx, (v - cy) / fy, dist)
        assert (math.atan(xn), math.atan(yn)) == pytest.approx((right, down), abs=1e-9)


def test_the_upright_field_of_view_is_the_lens_s():
    fov = ec.EyeCamera.for_frame(1280, 720).fov_deg()
    # spec §4.1 / camera_model: 25.7° and 13.2° either side on the upright horizontal (the
    # principal point sits 114 px off-centre), 32.3° and 37.5° along the long side
    assert fov["left"] == pytest.approx(13.2, abs=0.1)
    assert fov["right"] == pytest.approx(25.7, abs=0.1)
    assert fov["top"] == pytest.approx(32.4, abs=0.1)
    assert fov["bottom"] == pytest.approx(37.5, abs=0.1)


def _level_camera(height_m: float, pitch_down_rad: float = 0.0) -> ec.CameraPose:
    """A camera on the duck's axis looking forward (trunk +x), tipped down by ``pitch``.

    Sensor axes as mounted (rotate 90): sensor +x points down in the world, +y left (the upright
    picture's right is sensor −y), +z forward."""
    # columns: sensor x → trunk -z, sensor y → trunk +y, sensor z → trunk +x (level)
    level = _quat_from_matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    tip = (math.cos(pitch_down_rad / 2), 0.0, math.sin(pitch_down_rad / 2), 0.0)  # about +y
    return ec.CameraPose(
        cam_pos_m=(0.0, 0.0, 0.0),
        cam_quat=ec.q_normalized(ec.q_mul(tip, level)),
        imu_quat=ec.IDENTITY,
        trunk_height_m=height_m,
    )


def _quat_from_matrix(m):
    w = math.sqrt(max(0.0, 1 + m[0][0] + m[1][1] + m[2][2])) / 2
    if w > 1e-6:
        return (
            w,
            (m[2][1] - m[1][2]) / (4 * w),
            (m[0][2] - m[2][0]) / (4 * w),
            (m[1][0] - m[0][1]) / (4 * w),
        )
    x = math.sqrt(max(0.0, 1 + m[0][0] - m[1][1] - m[2][2])) / 2
    return (0.0, x, (m[0][1] + m[1][0]) / (4 * x), (m[0][2] + m[2][0]) / (4 * x))


def test_a_level_camera_sees_the_floor_where_the_geometry_says():
    cam = ec.EyeCamera.for_frame(1280, 720)
    pose = _level_camera(0.245)
    # the optical axis is straight ahead; the principal point is on the horizon
    _, _, cxu, cyu = cam.upright_k()
    assert cam.bearing_mrad(cxu, cyu, pose) == (0, 0)
    assert cam.ground_m(cxu, cyu - 5, pose) is None  # above the horizon
    # a pixel 20° below the axis meets the floor at h / tan(20°)
    for down_deg in (10.0, 20.0, 30.0):
        v = cyu + cam.fx * math.tan(math.radians(down_deg))  # pinhole row …
        # … through the lens: find the distorted row whose ray is down_deg below the axis
        lo, hi = cyu, 1279.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if cam.angles_rad(cxu, mid)[1] < math.radians(down_deg):
                lo = mid
            else:
                hi = mid
        v = (lo + hi) / 2
        g = cam.ground_m(cxu, v, pose)
        assert g is not None
        assert g[0] == pytest.approx(0.245 / math.tan(math.radians(down_deg)), rel=1e-6)
        assert g[1] == pytest.approx(0.0, abs=2e-4)  # the lens's tangential terms: < 0.2 mm
        assert cam.ground_mm(cxu, v, pose) == (round(g[0] * 1000), round(g[1] * 1000))
        assert cam.pixel_of_ground(g[0], g[1], pose) == pytest.approx((cxu, v), abs=1e-6)
    # to the right in the picture is to the right of the duck: negative bearing
    b, _ = cam.bearing_mrad(cxu + 200, cyu, pose)
    assert b < 0
    # spec §4.1: at HOME the lens sees the floor from 0.32–0.34 m (camera 0.245–0.258 m)
    assert 0.30 < cam.nearest_floor_m(pose) < 0.36


def test_tipping_the_head_and_the_trunk_moves_the_floor_closer():
    cam = ec.EyeCamera.for_frame(1280, 720)
    level = cam.nearest_floor_m(_level_camera(0.245))
    down = cam.nearest_floor_m(_level_camera(0.228, math.radians(17)))
    assert down < level and 0.15 < down < 0.26  # spec §4.1: half-down about 0.20 m
    # the same tip carried by the trunk (IMU) instead of the head gives the same floor
    tipped_trunk = ec.CameraPose(
        cam_pos_m=(0.0, 0.0, 0.0),
        cam_quat=_level_camera(0.228).cam_quat,
        imu_quat=(math.cos(math.radians(17) / 2), 0.0, math.sin(math.radians(17) / 2), 0.0),
        trunk_height_m=0.228,
    )
    assert cam.nearest_floor_m(tipped_trunk) == pytest.approx(down, rel=1e-9)
    # the world's yaw (the IMU's arbitrary heading) changes nothing in the level frame
    yawed = ec.CameraPose(
        cam_pos_m=(0.0, 0.0, 0.0),
        cam_quat=_level_camera(0.228).cam_quat,
        imu_quat=ec.q_mul(ec.q_about_z(1.3), tipped_trunk.imu_quat),
        trunk_height_m=0.228,
    )
    assert cam.nearest_floor_m(yawed) == pytest.approx(down, rel=1e-9)


def test_quaternion_helpers():
    q = ec.q_about_z(math.pi / 2)
    assert ec.q_rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert ec.q_angle(ec.IDENTITY, q) == pytest.approx(math.pi / 2)
    half = ec.q_slerp(ec.IDENTITY, q, 0.5)
    assert ec.q_angle(ec.IDENTITY, half) == pytest.approx(math.pi / 4)
    assert ec.q_rotate(ec.q_mul(q, ec.q_conj(q)), (1, 2, 3)) == pytest.approx((1, 2, 3))
    with pytest.raises(ValueError):
        ec.EyeCamera.for_frame(1280, 720, 45)


# ── against a render (render tier): the eye's geometry is the camera's ─────────────


def _pose_from_render(eye_scene):
    """What duck-body would send for this instant: robotd's ``frames.camera`` (the OpenCV
    sensor frame in the trunk frame), the trunk IMU quat and the trunk height — read here from
    our copy of the scene (MuJoCo's camera looks along −z with y up: OpenCV's is (x, −y, −z))."""
    import numpy as np

    d = eye_scene.data
    t = eye_scene.trunk_qpos
    trunk = np.array(d.qpos[t : t + 3], dtype=float)
    w, x, y, z = (float(v) for v in d.qpos[t + 3 : t + 7])
    r_trunk = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    r_cam = np.array(d.cam_xmat[eye_scene.camera], dtype=float).reshape(3, 3) @ np.diag(
        [1.0, -1.0, -1.0]
    )
    r = r_trunk.T @ r_cam  # sensor → trunk
    pos = r_trunk.T @ (np.array(d.cam_xpos[eye_scene.camera], dtype=float) - trunk)
    q = _quat_from_matrix(r.tolist())
    return ec.CameraPose(
        cam_pos_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        cam_quat=ec.q_normalized(q),
        imu_quat=(w, x, y, z),
        trunk_height_m=float(trunk[2]),
    )


def test_the_eye_s_ground_points_land_on_the_rendered_floor():
    """Cards on the official floor, the duck standing (head HOME, then half down): each card's
    centre seen in the delivered (lens-distorted) frame, taken through eye.camera with the pose
    duck-body would send, lands within 2 cm (level head) / 3 cm (half down) of where the card
    is; and the eye's projection of the card's centre falls on the card."""
    import os
    from pathlib import Path

    pytest.importorskip("numpy", reason="the eye's renders need numpy")
    pytest.importorskip("mujoco", reason="the eye's renders need mujoco")
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not (Path(folder) / "scene_allcollisions.xml").is_file():
        pytest.skip("official microduck_rl checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    import mujoco
    import numpy as np

    from duckmind.world import lights
    from duckmind.world import render_scene as rs

    cards = {
        "eyecard_a": (0.9, 0.0),
        "eyecard_b": (0.6, 0.25),
        "eyecard_c": (1.3, -0.3),
        "eyecard_d": (0.45, -0.08),
    }

    def edit(spec):
        for name, (x, y) in cards.items():
            g = spec.worldbody.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [0.03, 0.03, 0.001]
            g.pos = [x, y, 0.001]
            g.rgba = [0.9, 0.1, 0.1, 1.0]
            g.contype = 0
            g.conaffinity = 0

    try:
        eye = rs.HomeEye.open(
            Path(folder) / "scene_allcollisions.xml", home=lights.DEFAULT_HOME, edit=edit
        )
    except Exception as error:
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")
    try:
        width, height = eye.frame_size
        cam = ec.EyeCamera.for_frame(width, height, 90)
        # offsets from STAND's head (robotd's HOME): HOME, and duck-body's half-down look
        # (lookplan.HALF_DOWN: Δneck −0.125, Δhead +0.175 → camera 0.231 m, −17.2°; spec §4.1
        # says 0.228 m, −17°). On this MJCF the camera pitch is (Δneck − Δhead) × 57.3°.
        for head, limit_mm in (({}, 20), ({"neck_pitch": -0.125, "head_pitch": 0.175}, 30)):
            pose = cm.keyframe_pose(eye.eye, "STAND")
            pose["trunk"] = [0.0, 0.0, pose["trunk"][2]]
            for name, delta in head.items():
                pose["joints"][name] += delta
            eye.set_pose(pose)
            cpose = _pose_from_render(eye.eye)
            seg = eye.camera.renderer.segment_ids(width, height)
            seen = 0
            for name, (x, y) in cards.items():
                gid = mujoco.mj_name2id(eye.eye.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                rows, cols = np.nonzero(seg == gid)
                if rows.size < 20:
                    continue  # out of this view
                seen += 1
                u, v = cam.to_upright_pixel(float(cols.mean()), float(rows.mean()))
                ground = cam.ground_m(u, v, cpose)
                if head:
                    assert cpose.position_level[2] == pytest.approx(0.231, abs=0.004)
                assert ground is not None
                # the card in the level frame (the duck faces +x at the origin)
                t = eye.eye.trunk_qpos
                tx, ty = (float(eye.eye.data.qpos[t + i]) for i in (0, 1))
                err = math.hypot(ground[0] - (x - tx), ground[1] - (y - ty)) * 1000
                assert err < limit_mm, (name, head, err)
                px = cam.pixel_of_ground(x - tx, y - ty, cpose, z_m=0.002)
                assert px is not None
                sx, sy = cam.to_sensor_pixel(*px)
                assert seg[int(round(sy)), int(round(sx))] == gid, (name, head)
            assert seen >= 3, head
    finally:
        eye.close()
