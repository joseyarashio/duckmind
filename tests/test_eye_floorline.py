"""duck-eye's floor line, FL-v1 (spec §4.4; owner decision D1: the floor is learned from ground
walked without a contact).

Pure tier (numpy): the floor geometry against duck-eye's camera model, the learning (only the
walked corridor, only clear walks, the model survives a round trip), the column scan (a seam is
not a wall; a thing is), conservative until learned, no floor line in the dark, determinism (P2)
and integers only. Render tier: a learned floor in the default home and a red box on the path,
through the world's full camera model. (Ray-cast synthetic scenes with an exact truth are in
``test_eye_floorline_scenes.py``.)
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from duckmind.eye import floorline as fl  # noqa: E402
from duckmind.eye import protocol  # noqa: E402
from duckmind.eye.camera import CameraPose, EyeCamera  # noqa: E402

CAM = EyeCamera.for_frame(1280, 720, 90)
S90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])


def _quat(m):
    from eye_vision_scene import mat_quat

    return mat_quat(np.asarray(m).tolist())


def synthetic_pose(height: float = 0.25, down_deg: float = 17.0) -> CameraPose:
    """A level trunk, the camera ``height`` above the floor, pitched down ``down_deg``."""
    p = math.radians(down_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    r_trunk_up = np.stack([right, down, fwd], axis=1)
    return CameraPose(
        cam_pos_m=(0.06, 0.0, height - 0.12),
        cam_quat=_quat(r_trunk_up @ S90.T),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        trunk_height_m=0.12,
    )


def planes(y: int = 120, u: int = 110, v: int = 140) -> tuple:
    return (
        np.full((320, 180), y, np.uint8),
        np.full((160, 90), u, np.uint8),
        np.full((160, 90), v, np.uint8),
    )


def test_the_floor_geometry_is_the_cameras() -> None:
    pose = synthetic_pose()
    grid = fl.ground_grid(CAM, pose, 4, 320, 180)
    for i, j in ((319, 90), (250, 20), (200, 170), (300, 0)):
        want = CAM.ground_m(j * 4, i * 4, pose)
        assert want is not None
        assert grid.gx[i, j] == pytest.approx(want[0], abs=1e-9)
        assert grid.gy[i, j] == pytest.approx(want[1], abs=1e-9)
    assert not np.isfinite(grid.rng[0, 90])  # the top of the picture is above the horizon
    assert grid.pos[2] == pytest.approx(0.25)


def _learn_walks(model: fl.FloorModel, pose: CameraPose, n: int = 3) -> None:
    y, u, v = planes()
    for k in range(n):
        fl.analyse(y, u, v, CAM, pose, model, look_id=f"L{k}")
        added = model.learn(f"L{k}", "clear", [(100, 0), (900, 0)], 120)
        assert added > 0


def test_learning_takes_only_the_walked_corridor_of_clear_walks() -> None:
    pose = synthetic_pose()
    model = fl.FloorModel()
    y, u, v = planes()
    fl.analyse(y, u, v, CAM, pose, model, look_id="A")
    fl.analyse(y, u, v, CAM, pose, model, look_id="B")
    assert model.learn("A", "contact", [(100, 0), (900, 0)]) == 0
    assert model.learn("nope", "clear", [(100, 0), (900, 0)]) == 0
    narrow = model.learn("A", "clear", [(300, 0), (600, 0)], 60)
    wide = model.learn("B", "clear", [(300, 0), (600, 0)], 240)
    assert 0 < narrow < wide
    assert model.walks == 2 and model.samples == narrow + wide
    back = fl.FloorModel.from_json(model.to_json())
    assert back.walks == model.walks and back.samples == model.samples
    assert np.array_equal(back.counts, model.counts)
    assert protocol.integers_only(model.to_json()) == []


def test_not_learned_is_conservative() -> None:
    pose = synthetic_pose()
    y, u, v = planes()
    look = fl.analyse(y, u, v, CAM, pose, fl.FloorModel())
    res = look.result
    assert res["ok"] == 1 and res["learned"] == 0
    assert max(res["free_mm"]) <= fl.BOOT_CAP_MM
    assert set(res["ends"]) == {"unknown"}


def test_no_floor_line_in_the_dark_or_without_a_pose() -> None:
    y, u, v = planes()
    model = fl.FloorModel()
    assert fl.analyse(y, u, v, CAM, synthetic_pose(), model, light="dark").result == {
        "ok": 0,
        "why": "dark",
    }
    assert fl.analyse(y, u, v, CAM, None, model).result["why"] == "no_pose"


def _box_rows(pose: CameraPose, near_m: float, far_m: float, col: int) -> tuple[int, int]:
    grid = fl.ground_grid(CAM, pose, 4, 320, 180)
    rng = grid.rng[:, col]
    rows = [i for i in range(320) if np.isfinite(rng[i]) and near_m <= rng[i] <= far_m]
    return min(rows), max(rows)


def test_a_thing_ends_the_stretch_and_a_seam_does_not() -> None:
    pose = synthetic_pose()
    model = fl.FloorModel()
    _learn_walks(model, pose)
    assert model.ready
    y, u, v = planes()
    # a red thing 0.60–0.66 m ahead (its face and top), the floor again behind it
    top, bottom = _box_rows(pose, 0.60, 0.66, 90)
    y2, u2, v2 = y.copy(), u.copy(), v.copy()
    y2[top : bottom + 1, 60:120] = 70
    v2[top // 2 : bottom // 2 + 1, 30:60] = 200
    # a one-row dark seam across the picture at ~0.40 m
    seam = _box_rows(pose, 0.40, 0.41, 90)[0]
    y2[seam, :] = 60
    look = fl.analyse(y2, u2, v2, CAM, pose, model)
    res = look.result
    assert res["ok"] == 1 and res["learned"] == 1
    cols = res["cols_free_mm"]
    middle = [cols[c] for c in range(24) if 9 <= c <= 14]
    assert all(560 <= f <= 680 for f in middle), middle
    assert all(res["cols_end"][c] == "thing" for c in range(10, 14))
    # the sides see past the seam to the look's range
    assert cols[0] >= 1000 and cols[23] >= 1000
    assert res["things"] and 560 <= res["things"][0]["r_mm"] <= 680
    # the principal point is off-centre (upright cx ≈ 244 px): the block's middle column (90,
    # u = 360 px) is ~0.11 rad right of the axis
    want_b, _ = CAM.bearing_mrad(90 * 4, 200 * 4, pose)
    assert abs(res["things"][0]["b_mrad"] - want_b) < 40
    assert protocol.integers_only(res) == []


def test_dim_light_shortens_by_a_fifth() -> None:
    pose = synthetic_pose()
    model = fl.FloorModel()
    _learn_walks(model, pose)
    y, u, v = planes()
    top, bottom = _box_rows(pose, 0.80, 0.90, 90)
    y2 = y.copy()
    y2[top : bottom + 1, :] = 60
    bright = fl.analyse(y2, u, v, CAM, pose, model, light="bright").result
    dim = fl.analyse(y2, u, v, CAM, pose, model, light="dim").result
    for b, d in zip(bright["free_mm"], dim["free_mm"], strict=True):
        assert d == round(b * fl.DIM_FACTOR)


def test_the_same_planes_give_the_same_integers() -> None:
    pose = synthetic_pose()
    model = fl.FloorModel()
    _learn_walks(model, pose)
    rng = np.random.default_rng(3)
    y = rng.integers(100, 140, (320, 180), dtype=np.uint8)
    u = rng.integers(105, 115, (160, 90), dtype=np.uint8)
    v = rng.integers(135, 145, (160, 90), dtype=np.uint8)
    a = fl.analyse(y, u, v, CAM, pose, model).result
    b = fl.analyse(y.copy(), u.copy(), v.copy(), CAM, pose, model).result
    c = fl.analyse(y, u, v, CAM, pose, fl.FloorModel.from_json(model.to_json())).result
    assert a == b == c


# ── render tier ───────────────────────────────────────────────────────────


def test_a_learned_floor_sees_a_red_box_on_the_path() -> None:
    import eye_vision_scene as ev

    box = "eyetest_box"
    scene = ev.Scene("default_home", edit=lambda spec: ev.add_box(spec, box, 0.10, 0.06))
    try:
        body = scene.mujoco.mj_name2id(scene.m, scene.mujoco.mjtObj.mjOBJ_BODY, box)
        mocap = int(scene.m.body_mocapid[body])
        model = fl.FloorModel()
        # learn from walks over open floor near the middle of the room (the rug is at +x +y)
        spots = [(-1.5, -0.5, 0.0), (-1.2, 0.8, -1.2), (-0.8, -1.4, 2.4), (-1.8, 1.5, -0.4)]
        for k, (x, y, heading) in enumerate(spots):
            scene.pose(x, y, heading, ev.GLANCE)
            cap = scene.capture("day")
            planes_, cam, _ = ev.eye_inputs(cap)
            fl.analyse(
                planes_.y,
                planes_.u,
                planes_.v,
                cam,
                scene.camera_pose(),
                model,
                look_id=f"L{k}",
            )
            assert model.learn(f"L{k}", "clear", [(80, 0), (700, 0)], 120) > 0
        assert model.ready
        # a red 6 cm box 0.6 m ahead on the path
        scene.pose(-1.5, -0.5, 0.0, ev.GLANCE)
        wx, wy = scene.to_world(0.65, 0.0)
        scene.d.mocap_pos[mocap] = [wx, wy, 0.0]
        scene.mujoco.mj_forward(scene.m, scene.d)
        cap = scene.capture("day")
        planes_, cam, _ = ev.eye_inputs(cap)
        res = fl.analyse(planes_.y, planes_.u, planes_.v, cam, scene.camera_pose(), model).result
        seg = scene.seg_upright()[::4, ::4]
        gid = scene.geom(box + "_geom")
        edges = fl._column_edges(180)
        box_cols = [c for c, (c0, c1) in enumerate(edges) if (seg[:, c0:c1] == gid).any()]
        assert box_cols
        seen = [res["cols_free_mm"][c] for c in box_cols]
        assert any(abs(f - 600) <= 150 for f in seen), seen
        assert protocol.integers_only(res) == []
    finally:
        scene.close()
