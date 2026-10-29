"""duck-eye's floor line FL-v1 (spec §4.4; owner decision D1: the duck learns its floor from
ground it walked over without a contact), on synthetic ray-cast scenes with a known truth, plus
one golden from the phase-2 world camera (render tier).
"""

from __future__ import annotations

import math
import os
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import eye_scenes as es  # noqa: E402

from duckmind.eye import floorline as fl  # noqa: E402
from duckmind.eye import protocol as proto  # noqa: E402

CAM = es.CAMERA


def _learned(scene: es.Scene, pose, walks: int = 3) -> fl.FloorModel:
    """A floor model taught the way duck-body teaches it: a look, then a clear walk through it."""
    model = fl.FloorModel()
    for k in range(walks):
        planes = es.render_planes(scene, pose, seed=100 + k)
        fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model, look_id=f"L{k}")
        corridor = [(x, 0) for x in range(80, 701, 50)]
        assert model.learn(f"L{k}", "clear", corridor, 160) > 0
    assert model.ready
    return model


def test_the_ground_grid_is_the_cameras_geometry() -> None:
    pose = es.level_pose(0.25, 10.0)
    grid = fl.ground_grid(CAM, pose, 4, 320, 180)
    for i, j in ((319, 90), (250, 20), (200, 170)):
        want = CAM.ground_m(j * 4, i * 4, pose)
        assert want is not None
        assert grid.gx[i, j] == pytest.approx(want[0], abs=1e-6)
        assert grid.gy[i, j] == pytest.approx(want[1], abs=1e-6)
    level = es.level_pose(0.25, 0.0)
    top = fl.ground_grid(CAM, level, 4, 320, 180)
    assert not np.isfinite(top.rng[0]).any()  # above the horizon: no floor
    assert top.pos[2] == pytest.approx(0.25)


def test_colour_bins_keep_the_hue_of_a_shadow() -> None:
    y = np.array([[100, 50]], dtype=np.uint8)
    u = np.array([[110, 119]], dtype=np.uint8)  # the same hue at half the light
    v = np.array([[150, 139]], dtype=np.uint8)
    bins = fl.colour_bins(y, u, v)
    cu = (bins // fl.C_BINS) % fl.C_BINS
    cv = bins % fl.C_BINS
    assert abs(int(cu[0, 0]) - int(cu[0, 1])) <= 1 and abs(int(cv[0, 0]) - int(cv[0, 1])) <= 1
    assert int(bins[0, 0] // (fl.C_BINS * fl.C_BINS)) == 100 >> 3


def test_only_ground_walked_without_a_contact_is_learned() -> None:
    scene = es.Scene()
    pose = es.level_pose(0.25, 17.0)
    model = fl.FloorModel()
    planes = es.render_planes(scene, pose)
    fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model, look_id="L1")
    assert model.learn("L1", "contact", [(200, 0), (500, 0)], 120) == 0
    assert model.learn("L1", "fell", [(200, 0), (500, 0)], 120) == 0
    assert model.learn("nope", "clear", [(200, 0), (500, 0)], 120) == 0
    added = model.learn("L1", "clear", [(200, 0), (500, 0)], 120)
    assert added > 100 and model.walks == 1 and model.samples == added
    back = fl.FloorModel.from_json(model.to_json())
    assert back.walks == 1 and back.samples == added
    assert (back.counts == model.counts).all()
    assert not proto.integers_only(model.to_json(), "model")


def test_before_it_has_learned_the_floor_it_is_conservative() -> None:
    pose = es.level_pose(0.25, 17.0)
    planes = es.render_planes(es.Scene(wall_x=3.0), pose)
    look = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, fl.FloorModel())
    res = look.result
    assert res["ok"] == 1 and res["learned"] == 0
    assert max(res["free_mm"]) <= fl.BOOT_CAP_MM
    assert set(res["ends"]) == {"unknown"}
    assert res["trip"] is None and res["things"] == []


def test_a_low_box_and_the_wall_behind_it() -> None:
    pose = es.level_pose(0.25, 17.0)
    model = _learned(es.Scene(wall_x=1.2), pose)
    scene = es.Scene(wall_x=1.2, boxes=[es.Box(x=0.60, y=0.0, h=0.06)])
    planes = es.render_planes(scene, pose, seed=7)
    look = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model)
    res = look.result
    assert res["ok"] == 1 and res["learned"] == 1
    assert not proto.integers_only(res, "floor")
    bearings = res["cols_b_mrad"]
    box_cols = [i for i, b in enumerate(bearings) if abs(math.tan(b / 1000) * 0.6) < 0.04]
    wall_cols = [i for i, b in enumerate(bearings) if abs(math.tan(b / 1000) * 0.6) > 0.09]
    assert box_cols and wall_cols
    for i in box_cols:
        assert abs(res["cols_free_mm"][i] - 600) <= 60, (i, res["cols_free_mm"][i])
        assert res["cols_end"][i] == "thing"
    for i in wall_cols:
        want = 1200 / math.cos(bearings[i] / 1000)
        assert abs(res["cols_free_mm"][i] - want) <= 80, (i, res["cols_free_mm"][i], want)
        assert res["cols_end"][i] == "wall"
    (thing,) = res["things"]
    assert abs(thing["r_mm"] - 600) <= 60 and abs(thing["b_mrad"]) <= 40
    # a 6 cm box: the run's top row is the top face's far edge, so the height is an upper bound
    assert 30 <= thing["h_mm"] <= 120
    assert 60 <= thing["w_mm"] <= 180
    # the sector straight ahead ends at the box
    assert res["ends"][2] == "thing" and abs(res["free_mm"][2] - 600) <= 60


def test_plank_seams_do_not_end_the_floor() -> None:
    pose = es.level_pose(0.25, 17.0)
    scene = es.Scene(wall_x=1.4, seams_every=0.15)
    model = _learned(scene, pose)
    planes = es.render_planes(scene, pose, seed=9)
    res = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model).result
    middle = res["cols_free_mm"][6:18]
    assert min(middle) >= 1250, middle


def test_dim_light_shortens_and_the_dark_refuses() -> None:
    pose = es.level_pose(0.25, 17.0)
    scene = es.Scene(wall_x=1.2)
    model = _learned(scene, pose)
    planes = es.render_planes(scene, pose, seed=3)
    bright = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model, light="bright").result
    dim = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model, light="dim").result
    for b, d in zip(bright["cols_free_mm"], dim["cols_free_mm"], strict=True):
        assert d == round(b * fl.DIM_FACTOR)
    for level in ("dark", "black"):
        assert fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model, light=level).result == {
            "ok": 0,
            "why": "dark",
        }
    assert fl.analyse(planes.y, planes.u, planes.v, CAM, None, model).result["why"] == "no_pose"


def test_the_same_frame_gives_the_same_integers() -> None:
    """P2 (spec §11.7): same bytes, same model → the same features."""
    pose = es.level_pose(0.25, 17.0)
    model = _learned(es.Scene(wall_x=1.2), pose)
    twin = fl.FloorModel.from_json(model.to_json())
    planes = es.render_planes(es.Scene(wall_x=1.2, boxes=[es.Box(x=0.5, y=0.1)]), pose, seed=5)
    a = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, model).result
    b = fl.analyse(planes.y, planes.u, planes.v, CAM, pose, twin).result
    assert a == b


def _view(planes, pose, look_id: str, level: str = "bright") -> types.SimpleNamespace:
    look = proto.EyeLook(look_id=look_id, tasks=("floor",), pose_tap_ref=1, not_before_ns=0)
    return types.SimpleNamespace(
        look=look, planes=planes, camera=CAM, camera_pose=pose, light={"level": level}
    )


def test_the_task_learns_from_eye_learn() -> None:
    pose = es.level_pose(0.25, 17.0)
    task = fl.FloorTask()
    scene = es.Scene(wall_x=1.2)
    for k in range(fl.READY_WALKS):
        view = _view(es.render_planes(scene, pose, seed=k), pose, f"W{k}")
        res = task.run(view)
        assert res["ok"] == 1 and res["learned"] == 0 and task.last_id == f"W{k}"
        event = proto.EyeLearn(
            walk_id=f"walk{k}", look_id=f"W{k}", outcome="clear", floor=((150, 0), (650, 0))
        )
        assert task.learn(event) > 0
    res = task.run(_view(es.render_planes(scene, pose, seed=9), pose, "W9"))
    assert res["learned"] == 1 and res["walks"] == fl.READY_WALKS
    restored = fl.FloorTask.from_config({"model": task.model.to_json()})
    assert restored.model.ready
    assert all(isinstance(item, str) for item in task.digest_items())


# ── the golden: the phase-2 world camera (render tier) ────────────────────────


MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")


def _world_eye(edit):
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    from duckmind.world import camera_model as cm
    from duckmind.world import homelayout, lights, nestpad, render_scene

    layout = homelayout.load("default_home", mjcf_dir=MJCF)
    spec = homelayout.compose(layout, mjcf_dir=MJCF)
    nestpad.NestPad().edit_spec(spec, layout)
    edit(spec)
    homelayout.fix_keyframes(spec, layout)
    spec.compile()
    home = lights.HomeLights.from_layout(layout)
    eye = render_scene.HomeEye.open(
        spec, home=home, config=cm.CameraConfig(quality="720p30"), seed=7
    )
    return eye, home


def _pose_duck(eye, x, y, yaw, joints):
    from duckmind.world import camera_model as cm

    base = cm.keyframe_pose(eye.eye, "STAND")
    w0, x0, y0, z0 = (float(v) for v in base["quat"])
    yaw0 = math.atan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0 * y0 + z0 * z0))
    q = [math.cos((yaw - yaw0) / 2), 0.0, 0.0, math.sin((yaw - yaw0) / 2)]
    frame = dict(base)
    frame["trunk"] = [x, y, base["trunk"][2]]
    frame["quat"] = cm._quat_mul(q, list(base["quat"]))
    frame["joints"] = {**base["joints"], **joints}
    eye.set_pose(frame)


def world_camera_pose(eye):
    """What the pose tap gives (sensor → trunk, IMU, trunk height), from the physics truth."""
    import mujoco

    from duckmind.eye.camera import CameraPose

    m, d = eye.eye.model, eye.eye.data
    mujoco.mj_forward(m, d)
    cam = eye.eye.camera
    pos = np.array(d.cam_xpos[cam])
    rc = np.array(d.cam_xmat[cam]).reshape(3, 3)
    r_world_upright = np.stack([rc[:, 1], rc[:, 0], -rc[:, 2]], axis=1)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rt = np.array(d.xmat[body]).reshape(3, 3)
    tp = np.array(d.xpos[body])
    r_world_sensor = r_world_upright @ fl.upright_to_sensor(90).T
    return CameraPose(
        cam_pos_m=tuple(float(v) for v in rt.T @ (pos - tp)),
        cam_quat=es.mat_quat(rt.T @ r_world_sensor),
        imu_quat=es.mat_quat(rt),
        trunk_height_m=float(tp[2]),
    )


def test_golden_a_box_in_default_home() -> None:
    """The world camera (lens, AE, white balance, IMX219 noise) in default_home by day: the floor
    learned from three walked corridors, then a red 6 cm box 0.6 m ahead is seen at its place."""
    mujoco = pytest.importorskip("mujoco")  # (numpy is a dependency now; MuJoCo is not)

    from duckmind.eye import frames
    from duckmind.world import lights

    def add_box(spec):
        body = spec.worldbody.add_body()
        body.name = "fl_test_box"
        body.mocap = True
        body.pos = [0, 0, -2]
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.05, 0.05, 0.03]
        g.pos = [0, 0, 0.03]
        g.rgba = [0.8, 0.15, 0.12, 1]
        g.contype = 0
        g.conaffinity = 0

    eye, home = _world_eye(add_box)
    try:
        m, d = eye.eye.model, eye.eye.data
        mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fl_test_box")])
        x0, y0, yaw = -1.8, 0.4, 0.0  # on the open floor of default_home, facing +x
        _pose_duck(eye, x0, y0, yaw, {"head_pitch": 0.646})
        pose = world_camera_pose(eye)
        state = lights.home_light_state(__import__("datetime").datetime(2026, 12, 20, 12, 0), home)
        eye.converge(state)
        model = fl.FloorModel()
        for k in range(3):
            cap = eye.capture(state, index=k)
            pl = frames.sample_quarter(cap.uyvy(), cap.width, cap.height, cap.rotate)
            fl.analyse(pl.y, pl.u, pl.v, CAM, pose, model, look_id=f"G{k}")
            model.learn(f"G{k}", "clear", [(x, 0) for x in range(100, 801, 50)], 160)
        assert model.ready
        d.mocap_pos[mocap] = [x0 + 0.65, y0, 0.0]
        mujoco.mj_forward(m, d)
        cap = eye.capture(state, index=9)
        pl = frames.sample_quarter(cap.uyvy(), cap.width, cap.height, cap.rotate)
        res = fl.analyse(pl.y, pl.u, pl.v, CAM, pose, model).result
        assert res["ok"] == 1 and res["learned"] == 1
        middle = [res["cols_free_mm"][i] for i in range(24) if abs(res["cols_b_mrad"][i]) < 60]
        front = 0.65 - 0.05  # the box's front face from the trunk origin
        assert middle and min(abs(v - front * 1000) for v in middle) <= 100, middle
        assert any(abs(t["r_mm"] - front * 1000) <= 120 for t in res["things"]), res["things"]
    finally:
        eye.close()
