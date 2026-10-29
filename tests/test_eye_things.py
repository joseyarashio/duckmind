"""duck-eye's ``things`` task (spec §4.4; owner decision D2): what the floor line's low things
look like — their colour name, a hue × saturation histogram, and which registered toy each is.

Pure tier (numpy): colour names, histograms, the toy registry and its round trip, a synthetic
look; the task answers nothing in the dark. Render tier: a red box on the default home's floor.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from duckmind.eye import floorline as fl  # noqa: E402
from duckmind.eye import protocol, things  # noqa: E402


@pytest.mark.parametrize(
    ("rgb", "name"),
    [
        ((0.8, 0.15, 0.12), "red"),
        ((0.93, 0.93, 0.90), "white"),
        ((0.5, 0.5, 0.5), "grey"),
        ((0.1, 0.1, 0.1), "black"),
        ((0.2, 0.4, 0.9), "blue"),
        ((0.2, 0.7, 0.3), "green"),
        ((0.95, 0.85, 0.2), "yellow"),
        ((0.07, 0.71, 0.77), "cyan"),
        ((0.55, 0.40, 0.25), "brown"),
        ((0.95, 0.55, 0.1), "orange"),
    ],
)
def test_colour_names(rgb: tuple[float, float, float], name: str) -> None:
    hue, sat, val = things._hsv(np.array([rgb]))
    assert things.colour_name(float(hue[0]), float(sat[0]), float(val[0])) == name


def test_histograms_and_the_toy_registry() -> None:
    red = np.tile(np.array([[0.8, 0.15, 0.12]]), (50, 1))
    blue = np.tile(np.array([[0.2, 0.4, 0.9]]), (50, 1))
    h_red, h_blue = things.histogram(red), things.histogram(blue)
    assert len(h_red) == things.HIST_LEN and abs(sum(h_red) - 1000) <= things.HIST_LEN
    assert things.intersection(h_red, h_red) >= 990 and things.intersection(h_red, h_blue) == 0
    reg = things.ToyRegistry()
    reg.register("ball", h_red)
    assert reg.match(h_red)[0] == "ball"
    assert reg.match(h_blue)[0] is None
    back = things.ToyRegistry.from_json(reg.to_json())
    assert back.toys == reg.toys
    assert protocol.integers_only(reg.to_json()) == []
    with pytest.raises(ValueError):
        reg.register("bad", [1, 2, 3])


def test_a_synthetic_thing_gets_its_colour() -> None:
    from test_eye_floorline import CAM, _box_rows, _learn_walks, planes, synthetic_pose

    pose = synthetic_pose()
    model = fl.FloorModel()
    _learn_walks(model, pose)
    y, u, v = planes()
    top, bottom = _box_rows(pose, 0.60, 0.66, 90)
    y2, u2, v2 = y.copy(), u.copy(), v.copy()
    y2[top : bottom + 1, 60:120] = 80
    u2[top // 2 : bottom // 2 + 1, 30:60] = 100
    v2[top // 2 : bottom // 2 + 1, 30:60] = 200
    look = fl.analyse(y2, u2, v2, CAM, pose, model)
    reg = things.ToyRegistry()
    items = things.describe(look, y2, u2, v2, reg)
    assert len(items) == 1 and items[0]["colour"] == "red"
    reg.register("red_box", items[0]["hist"])
    again = things.describe(look, y2, u2, v2, reg)
    assert again[0]["id"] == "red_box" and again[0]["match_pm"] >= things.MATCH_MIN
    assert protocol.integers_only(again) == []
    s = things.sightings({"things": again})
    assert s[0]["what"] == "thing" and s[0]["r_src"] == "ground" and s[0]["id"] == "red_box"


def test_no_things_in_the_dark() -> None:
    class View:
        light = {"level": "dark"}

    assert things.ThingsTask().run(View()) == {"ok": 0, "why": "dark", "things": []}


def test_things_share_the_floor_task_when_bound() -> None:
    floor = fl.FloorTask()
    task = things.ThingsTask()
    task.bind({"floor": floor, "things": task})
    assert task.floor is floor


# ── render tier ───────────────────────────────────────────────────────────


def test_a_red_box_on_the_floor_is_a_red_thing() -> None:
    import eye_vision_scene as ev

    box = "eyetest_box"
    scene = ev.Scene("default_home", edit=lambda spec: ev.add_box(spec, box, 0.10, 0.08))
    try:
        body = scene.mujoco.mj_name2id(scene.m, scene.mujoco.mjtObj.mjOBJ_BODY, box)
        mocap = int(scene.m.body_mocapid[body])
        model = fl.FloorModel()
        spots = [(-1.5, -0.5, 0.0), (-1.2, 0.8, -1.2), (-0.8, -1.4, 2.4), (-1.8, 1.5, -0.4)]
        for k, (x, y, heading) in enumerate(spots):
            scene.pose(x, y, heading, ev.GLANCE)
            planes, cam, _ = ev.eye_inputs(scene.capture("day"))
            fl.analyse(
                planes.y, planes.u, planes.v, cam, scene.camera_pose(), model, look_id=f"L{k}"
            )
            model.learn(f"L{k}", "clear", [(80, 0), (700, 0)], 120)
        scene.pose(-1.5, -0.5, 0.0, ev.GLANCE)
        wx, wy = scene.to_world(0.55, 0.0)
        scene.d.mocap_pos[mocap] = [wx, wy, 0.0]
        scene.mujoco.mj_forward(scene.m, scene.d)
        planes, cam, _ = ev.eye_inputs(scene.capture("day"))
        look = fl.analyse(planes.y, planes.u, planes.v, cam, scene.camera_pose(), model)
        items = things.describe(look, planes.y, planes.u, planes.v)
        near = [t for t in items if abs(t["r_mm"] - 500) <= 150]
        assert near and near[0]["colour"] in ("red", "brown", "orange"), items
        assert near[0]["h_mm"] > 20
    finally:
        scene.close()
