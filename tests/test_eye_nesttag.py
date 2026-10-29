"""duck-eye's dock task, NB-v1 (spec §4.4; owner decisions D3 — the official AprilTag library,
OpenCV as the fallback — and C5 — the dock's light and cyan panel give a direction only).

Pure tier: the eye's own read of a tag's cells (against its own copy of the codes), the planar
pose, the ROI, the cues, what becomes a sighting, and that black light reads no tag. Library
tier (pupil_apriltags / cv2): both backends find a drawn tag with the same corner order. Render
tier: the dock's face tag across the default home through the world's camera, and no anchor in
the dark.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from duckmind.eye import nesttag, protocol, tag36h11  # noqa: E402
from duckmind.eye.camera import EyeCamera  # noqa: E402

CAM = EyeCamera.for_frame(1280, 720, 90)


def draw_tag(tag_id: int, corners: list[tuple[float, float]], size=(480, 480), flip=None):
    """A grey picture with the tag (10×10 cells with its white border) mapped so the 8×8 black
    square's corners (its bottom-left, bottom-right, top-right, top-left) land on ``corners``."""
    cells = [list(r) for r in tag36h11.grid(tag_id)]
    if flip is not None:
        x, y = flip
        cells[y + 1][x + 1] ^= 1
    h = nesttag._homography(((0.0, 8.0), (8.0, 8.0), (8.0, 0.0), (0.0, 0.0)), corners)
    inv = np.linalg.inv(h)
    vv, uu = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float64)
    pts = inv @ np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    cx = (pts[0] / pts[2]).reshape(uu.shape) + 1.0  # into the 10×10 grid
    cy = (pts[1] / pts[2]).reshape(uu.shape) + 1.0
    img = np.full(uu.shape, 128, np.uint8)
    inside = (cx >= 0) & (cx < 10) & (cy >= 0) & (cy < 10)
    grid = np.array(cells)
    vals = grid[np.clip(cy.astype(int), 0, 9), np.clip(cx.astype(int), 0, 9)]
    img[inside] = np.where(vals[inside] == 1, 225, 25)
    return img


CORNERS = [(150.0, 330.0), (330.0, 322.0), (322.0, 150.0), (158.0, 158.0)]


def test_the_eye_reads_the_cells_itself() -> None:
    img = draw_tag(17, CORNERS)
    read = nesttag.read_cells(img, np.array(CORNERS))
    assert read is not None
    cells, contrast = read
    assert contrast > 150
    assert tag36h11.decode(cells, max_errors=0) == (17, 0, 0)
    raw = nesttag.RawTag(17, np.array(CORNERS), 0, 50.0)
    assert nesttag.verify(img, raw, 1, 30) == (0, pytest.approx(contrast))
    # a library that calls it 18 is overruled by the eye's own decode
    assert nesttag.verify(img, nesttag.RawTag(18, np.array(CORNERS), 0, 50.0), 1, 30) is None


def test_one_wrong_bit_is_tolerated_in_the_light_not_in_the_dark() -> None:
    bx, by = tag36h11.BIT_X[4], tag36h11.BIT_Y[4]
    img = draw_tag(17, CORNERS, flip=(bx, by))
    raw = nesttag.RawTag(17, np.array(CORNERS), 1, 40.0)
    assert nesttag.verify(img, raw, nesttag.MAX_HAMMING, nesttag.MIN_CONTRAST)[0] == 1
    assert nesttag.verify(img, raw, nesttag.DARK_MAX_HAMMING, nesttag.DARK_MIN_CONTRAST) is None


def test_low_contrast_is_not_a_tag() -> None:
    img = draw_tag(17, CORNERS)
    faint = (128 + (img.astype(int) - 128) // 12).astype(np.uint8)
    raw = nesttag.RawTag(17, np.array(CORNERS), 0, 5.0)
    assert nesttag.verify(faint, raw, 1, nesttag.MIN_CONTRAST) is None


def _project(rot, t, size):
    obj = nesttag._object_points(size)
    p = obj @ rot.T + t
    return p[:, :2] / p[:, 2:3]


def test_the_planar_pose_comes_back() -> None:
    a = math.radians(30.0)
    # the tag faces the camera turned 30° about the vertical (camera y is down)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, -1, 0], [math.sin(a), 0, -math.cos(a)]])
    rot = rot @ np.diag([1, 1, 1])
    rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
    t = np.array([0.1, 0.12, 1.5])
    norm = _project(rot, t, nesttag.FACE_TAG_M)
    sols = nesttag.solve_pose(norm, nesttag.FACE_TAG_M)
    best_rot, best_t, err = sols[0]
    assert err < 1e-9
    assert np.allclose(best_t, t, atol=1e-6)
    assert float(best_rot[:, 2] @ rot[:, 2]) > 0.9999


def test_black_light_reads_no_tag_and_the_roi() -> None:
    from eye_vision_scene import mat_quat  # noqa: F401 - the helper module is importable

    calls = []

    def luma_of(box, step):
        calls.append(box)
        return np.zeros((10, 10), np.uint8)

    class Backend:
        name = "fake"

        def detect(self, gray, decimate):
            return []

    out = nesttag.find_tags(luma_of, CAM, None, Backend(), level="black", nest_b_mrad=0)
    assert out == {"tags": [], "rejected": 0, "modes": []} and calls == []
    nesttag.find_tags(luma_of, CAM, None, Backend(), level="bright", nest_b_mrad=0)
    assert calls == [None]  # no pose: no ROI, the whole frame


def test_roi_box_follows_the_bearing() -> None:
    from test_eye_floorline import synthetic_pose

    pose = synthetic_pose(0.25, 0.0)
    ahead = nesttag.roi_box(CAM, pose, 0, nesttag.ROI_HALF_MRAD)
    assert ahead is not None
    u0, v0, u1, v1 = ahead
    width, height = CAM.upright_size
    assert 0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height
    left = nesttag.roi_box(CAM, pose, 300, 120)
    assert left is not None and left[1] <= left[3] and left[2] <= u1
    assert nesttag.roi_box(CAM, pose, 3000, 262) is None  # behind the duck


def test_cues_give_a_direction_only() -> None:
    from test_eye_floorline import synthetic_pose

    from duckmind.eye.frames import QuarterPlanes

    pose = synthetic_pose(0.25, 0.0)
    y = np.full((320, 180), 120, np.uint8)
    u = np.full((160, 90), 128, np.uint8)
    v = np.full((160, 90), 128, np.uint8)
    # a cyan patch on the left of the picture (U high, V low)
    y[140:160, 10:30] = 134
    u[70:80, 5:15] = 163
    v[70:80, 5:15] = 45
    cue = nesttag.cyan_cue(QuarterPlanes(y, u, v), CAM, pose)
    assert cue is not None and cue["kind"] == "panel" and cue["b_mrad"] > 0
    # the dock's light in a dark picture, right of the axis
    dark = np.full((320, 180), 20, np.uint8)
    dark[150:153, 150:156] = 250
    light = nesttag.light_cue(QuarterPlanes(dark, u, v), CAM, pose)
    assert light is not None and light["kind"] == "light" and light["b_mrad"] < 0
    # any other small light is no direction to the dock (owner C1/C5: only the dock's strip)
    led = np.full((320, 180), 20, np.uint8)
    led[150:155, 150:155] = 250  # a round standby LED: 5 × 5
    assert nesttag.light_cue(QuarterPlanes(led, u, v), CAM, pose) is None
    result = {"anchor": None, "cue": light}
    s = nesttag.sighting(result)
    assert s is not None and s["r_src"] == "cue" and s["r_mm"] is None and s["yaw_mrad"] is None
    assert protocol.integers_only(s) == []
    anchor = {
        "id": 17,
        "b_mrad": 5,
        "e_mrad": -60,
        "r_mm": 1500,
        "yaw_mrad": 3100,
        "yaw_amb_mrad": 0,
        "conf_ppm": 950000,
    }
    s = nesttag.sighting({"anchor": anchor, "cue": light})
    assert s is not None and s["r_src"] == "tag" and s["id"] == "tag36h11:17"
    s = nesttag.sighting({"anchor": dict(anchor, yaw_amb_mrad=900), "cue": None})
    assert s is not None and s["yaw_mrad"] is None  # the facing is not trusted


@pytest.mark.parametrize("backend", ["apriltag", "opencv"])
def test_both_libraries_find_a_drawn_tag_with_the_same_corners(backend: str) -> None:
    pytest.importorskip("pupil_apriltags" if backend == "apriltag" else "cv2")
    img = draw_tag(17, CORNERS)
    found = nesttag.TagBackend(backend).detect(img, 1.0)
    tags = [t for t in found if t.tag_id == 17]
    assert tags, backend
    assert np.allclose(tags[0].corners, np.array(CORNERS), atol=2.0)


def test_the_detector_is_destroyed_before_its_families():
    """Regression (Spark, glibc, 2026-10-08): pupil-apriltags' ``Detector.__del__`` destroyed the
    tag families first; ``apriltag_detector_destroy`` then read and wrote the freed family
    (heap use-after-free), which corrupts glibc's heap and aborted a later ``malloc``
    (SIGABRT in test_the_dock_tag_across_the_room…). macOS does not notice. The order is
    checked on a spy in front of the C library."""
    pytest.importorskip("pupil_apriltags")
    backend = nesttag.TagBackend("apriltag")
    assert backend.name == "apriltag"
    backend.detect(np.full((64, 64), 128, np.uint8), 1.0)
    det = backend._pupil[1.0]
    calls: list[str] = []

    class Fn:
        def __init__(self, name: str, real) -> None:
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "real", real)

        def __setattr__(self, key, value) -> None:
            setattr(self.real, key, value)

        def __call__(self, *args):
            calls.append(self.name)
            return self.real(*args)

    class Spy:
        def __init__(self, real) -> None:
            self.real = real

        def __getattr__(self, name):
            return Fn(name, getattr(self.real, name))

    det.libc = Spy(det.libc)
    type(det).__del__(det)
    det.tag_detector_ptr = None  # the object is finished; no second destroy at collection
    assert calls.index("apriltag_detector_destroy") < calls.index("tag36h11_destroy"), calls


# ── render tier ───────────────────────────────────────────────────────────


def test_the_dock_tag_across_the_room_and_nothing_in_the_dark() -> None:
    pytest.importorskip("pupil_apriltags")
    import eye_vision_scene as ev

    scene = ev.Scene("default_home")
    try:
        gid = scene.geom("home_nest_tag")
        tag = np.array(scene.d.geom_xpos[gid])
        normal = np.array(scene.d.geom_xmat[gid]).reshape(3, 3)[:, 2]
        heading = math.atan2(normal[1], normal[0])
        task = nesttag.NestTask()
        for dist, off_deg in ((1.5, 0.0), (2.5, 25.0)):
            a = heading + math.radians(off_deg)
            cx, cy = tag[0] + dist * math.cos(a), tag[1] + dist * math.sin(a)
            scene.pose_camera_at(cx, cy, math.atan2(tag[1] - cy, tag[0] - cx))
            lx, ly = scene.to_level(tag[0], tag[1])
            planes, cam, luma_of = ev.eye_inputs(scene.capture("day"))
            res = task.read(luma_of, planes, cam, scene.camera_pose(), level="bright")
            anchor = res["anchor"]
            assert anchor is not None and anchor["id"] == 17, res
            assert abs(anchor["r_mm"] - math.hypot(lx, ly) * 1000) < 60
            assert abs(anchor["b_mrad"] - math.atan2(ly, lx) * 1000) < 20
            assert 60 <= anchor["z_mm"] <= 160
            assert protocol.integers_only(res) == []
            # P2: the same frame gives the same integers
            again = task.read(luma_of, planes, cam, scene.camera_pose(), level="bright")
            assert again == res
        # moonlight, facing the dock from 1 m: no tag, the dock's light as a direction
        cx, cy = tag[0] + math.cos(heading), tag[1] + math.sin(heading)
        scene.pose_camera_at(cx, cy, heading + math.pi)
        planes, cam, luma_of = ev.eye_inputs(scene.capture("moon"))
        res = task.read(luma_of, planes, cam, scene.camera_pose(), level="dark")
        assert res["anchor"] is None and res["tags"] == []
        black = task.read(luma_of, planes, cam, scene.camera_pose(), level="black")
        assert black["tags"] == [] and black["modes"] == []
    finally:
        scene.close()
