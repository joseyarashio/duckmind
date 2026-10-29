"""P5: the head camera model — family K and its lens, principal-point sign, image orientation
after the flip — and the rest of the full camera: quality rungs, UYVY, white balance, RAW10,
denoise, vignetting, the glow of emitters.

The geometry checks render with MuJoCo (``uv run --with mujoco==3.10.0 --with numpy``); they
skip with the reason when MuJoCo, numpy, an offscreen GL context or the official checkout
(``DUCKMIND_MICRODUCK_MJCF``) is missing. The arithmetic runs everywhere.
"""

import importlib.util
import math
import os
from datetime import datetime
from pathlib import Path

import pytest

from duckmind.world import camera_model as cm

K = cm.FAMILY


# ── arithmetic (no MuJoCo) ────────────────────────────────────────────────


def test_family_k_is_the_62_degree_field_seen_upright():
    fov = K.fov_deg()
    assert fov["horizontal"] == pytest.approx(62.1, abs=0.2)  # the sensor's long side (pinhole)
    assert fov["vertical"] == pytest.approx(37.1, abs=0.2)
    up = K.upright()
    assert (up.width, up.height) == (720, 1280)
    assert (up.fx, up.fy) == (K.fy, K.fx)
    assert (up.cx, up.cy) == pytest.approx((719 - K.cy, K.cx))
    # The principal point sits ~115 px off centre across the upright picture: the duck sees
    # about 24° to one side of its optical axis and 13° to the other.
    side = up.fov_deg()
    assert (side["left"], side["right"]) == pytest.approx((13.0, 24.1), abs=0.1)
    assert (side["top"], side["bottom"]) == pytest.approx((29.4, 32.7), abs=0.1)


def test_mujoco_principal_pixel_has_the_opposite_sign_of_opencv():
    px, py = cm.mujoco_principal_pixel(K)
    assert px == pytest.approx(640 - (K.cx + 0.5))
    assert py == pytest.approx(360 - (K.cy + 0.5))
    assert px > 0 > py  # OpenCV's cx − W/2 is negative and cy − H/2 positive


def test_bearing_and_projection_are_inverse():
    for right, down in [(0.0, 0.0), (0.3, -0.2), (-0.5, 0.25)]:
        u, v = K.project(right, down, 1.0)
        bx, by = K.bearing_deg(u, v)
        assert bx == pytest.approx(math.degrees(math.atan(right)))
        assert by == pytest.approx(math.degrees(math.atan(down)))
    small = K.scaled(160, 90)
    assert (small.fx, small.cx) == pytest.approx((K.fx / 8, K.cx / 8))


def test_quality_rungs_are_robotd_params_and_scale_k():
    assert list(cm.QUALITIES) == ["1080p30", "720p30", "720p15", "360p30"]
    assert cm.quality_rung().label == cm.DEFAULT_QUALITY == "720p30"
    sizes = {q.label: (q.width, q.height, q.fps) for q in cm.QUALITIES.values()}
    assert sizes == {
        "1080p30": (1920, 1080, 30),
        "720p30": (1280, 720, 30),
        "720p15": (1280, 720, 15),
        "360p30": (640, 360, 30),
    }
    q360 = cm.quality_rung("360p30")
    assert (q360.k.fx, q360.k.cx, q360.k.cy) == pytest.approx((K.fx / 2, K.cx / 2, K.cy / 2))
    assert cm.quality_rung("1080p30").k.fy == pytest.approx(K.fy * 1.5)
    assert q360.frame_bytes == 640 * 360 * 2
    assert cm.quality_rung("720p15").frame_period_s == pytest.approx(1 / 15)
    assert cm.quality_from_config('[media]\nquality = "360p30"\n').label == "360p30"
    assert cm.quality_from_config("[robot]\nname = 'x'\n").label == "720p30"
    assert cm.quality_from_config({"media": {"quality": "1080p30"}}).label == "1080p30"
    with pytest.raises(ValueError, match="1080p30, 720p30, 720p15, 360p30"):
        cm.quality_rung("4k")


def test_quality_labels_match_robotd_params_upstream():
    """The rung names and sizes as robotd-params spells them (read from the checkout)."""
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder:
        pytest.skip("official checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    # …/work/upstream/microduck_rl/src/mjlab_microduck/robot/microduck → …/work/upstream
    lib = Path(folder).parents[4] / "microduck" / "robotd-params" / "src" / "lib.rs"
    if not lib.is_file():
        pytest.skip(f"robotd-params not found at {lib}")
    text = lib.read_text()
    assert 'QUALITY_LABELS: &[&str] = &["1080p30", "720p30", "720p15", "360p30"]' in text
    for q in cm.QUALITIES.values():
        assert f"({q.width}, {q.height})" in text
    assert "Quality::Q720p15 => 15" in text


def test_distortion_round_trips_and_undistorts_the_frame():
    x, y = 0.3, -0.2
    xd, yd = cm.distort(x, y)
    assert cm.undistort(xd, yd) == pytest.approx((x, y), abs=1e-12)
    # barrel: a point far off axis lands closer to the middle than a pinhole puts it
    assert abs(xd) < abs(x) and abs(yd) < abs(y)
    u, v = cm.distort_point(0.3, -0.2, 1.0)
    assert cm.bearing_deg(u, v) == pytest.approx(
        (math.degrees(math.atan(0.3)), math.degrees(math.atan(-0.2))), abs=1e-9
    )
    assert cm.undistort_pixel(K.cx, K.cy) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_the_lens_frame_reaches_about_70_degrees():
    lens = cm.Lens()
    fov = lens.fov_deg()
    # the survey's numbers: 32.3° and 37.5° either side along the 1280-px side, 25.7° / 13.2°
    assert (fov["left"], fov["right"]) == pytest.approx((32.4, 37.5), abs=0.1)
    assert (fov["top"], fov["bottom"]) == pytest.approx((25.7, 13.2), abs=0.1)
    assert fov["horizontal"] == pytest.approx(69.9, abs=0.2)  # against 62.1 for the pinhole
    canvas = lens.canvas
    assert (canvas.width, canvas.height) == (1615, 913)  # the survey's 1611×910 plus margins
    assert (canvas.fx, canvas.fy) == (K.fx, K.fy)
    assert canvas.cx > K.cx and canvas.cy > K.cy  # the canvas reaches further up and left
    # upright at HOME (camera 0.245 m, level) the floor shows from about 0.32 m, not 0.38 m
    assert 0.245 / math.tan(math.radians(fov["right"])) == pytest.approx(0.32, abs=0.005)
    assert lens.canvas_size(640, 360) == (808, 456)
    assert lens.canvas_size(*cm.METER_SIZE) == (202, 114)


def test_the_upright_picture_has_its_own_distortion():
    """``uyvy::Turn::Right`` turns (x, y) into (−y, x): the tangential terms swap sides."""
    up_k, up_dist = K.upright(), cm.upright_distortion()
    for right, down in [(0.3, -0.2), (-0.45, 0.1), (0.05, 0.4)]:
        u, v = cm.distort_point(right, down, 1.0)
        turned = (K.height - 1 - v, u)  # where the pixel lands upright
        # the same ray in upright camera axes
        assert cm.distort_point(-down, right, 1.0, up_k, up_dist) == pytest.approx(turned, abs=1e-6)


def test_denoise_ramps_with_the_gain_instead_of_jumping():
    assert cm.denoise_strength(1.0) == 0.0 and cm.denoise_strength(4.0) == 0.0
    assert cm.denoise_strength(4.1) == cm.denoise_strength(24.0) == 0.5
    assert cm.denoise_strength(64.0) == cm.denoise_strength(176.0) == pytest.approx(0.8)
    steps = [cm.denoise_strength(g) for g in (24, 28, 31.9, 32.1, 40, 50, 64)]
    assert steps == sorted(steps)
    assert cm.denoise_strength(32.1) - cm.denoise_strength(31.9) < 0.01  # no cliff at 32×


def test_white_balance_takes_most_of_a_colour_cast_slowly():
    from duckmind.world import lights

    lamp = lights.unit_luminance(lights.linear_rgb(lights.LAMP_COLOR))
    target = cm.Awb.target_for(lamp)
    assert target is not None
    r, g, b = target
    assert g == 1.0 and r < 1.0 < b
    # 85 % of the cast in log terms: a lamp-lit white wall stays a little warm
    assert math.log(r) == pytest.approx(cm.AWB_STRENGTH * math.log(lamp[1] / lamp[0]))
    awb = cm.Awb()
    awb.update(illuminant=lamp)
    assert awb.gains[0] == pytest.approx(r**cm.AWB_RATE)
    for _ in range(20):
        awb.update(illuminant=lamp)
    assert awb.settled() and awb.gains == pytest.approx(target, rel=1e-3)
    white = cm.Awb.target_for((1.0, 1.0, 1.0))
    assert white == (1.0, 1.0, 1.0)


# ── the sensor (numpy) ─────────────────────────────────────────────────────


def test_sensor_model_is_deterministic_for_a_seed():
    np = pytest.importorskip("numpy")
    scene = np.full((90, 160, 3), 20.0, dtype=np.float32)
    controls = cm.Controls(1200, 2816, 700)
    a = cm.expose(scene, controls, np.random.default_rng([7, 1, 0]))
    b = cm.expose(scene, controls, np.random.default_rng([7, 1, 0]))
    c = cm.expose(scene, controls, np.random.default_rng([7, 1, 1]))
    assert a.dtype == np.uint8 and (a == b).all() and not (a == c).all()
    clean = cm.expose(scene, controls, None)
    assert len(np.unique(clean)) == 1
    # more gain, more noise, the same mean (the AE target is what the loop steers)
    quiet = cm.expose(scene * 16, cm.Controls(1200, 2816, 256), np.random.default_rng(0))
    loud = cm.expose(scene, cm.Controls(1200, 2816, 4096), np.random.default_rng(0))
    assert cm.luma(quiet).std() < cm.luma(loud).std()
    assert abs(float(cm.luma(quiet).mean()) - float(cm.luma(loud).mean())) < 5


def test_raw10_codes_band_a_dark_ramp_at_high_digital_gain():
    np = pytest.importorskip("numpy")
    ramp = np.linspace(0.0, 1.2, 1000, dtype=np.float32)[None, :, None].repeat(3, axis=2)
    high = cm.Controls(1200, 2816, 16 * 256)  # 11× analogue, 16× digital: the moonlit ceiling
    raw10 = cm.expose(ramp, high, None)
    smooth = cm.expose(ramp, high, None, raw_bits=None)
    assert len(np.unique(raw10)) < len(np.unique(smooth)) / 2  # one code is 16/959 of full
    # the black-level pedestal keeps the noise below black through the ADC: the mean of a
    # moonlit grey is the same with and without the RAW10 step (no rectified noise)
    grey = np.full((180, 320, 3), 0.25 * 0.4, dtype=np.float32)
    with_raw = cm.expose(grey, high, np.random.default_rng(3))
    without = cm.expose(grey, high, np.random.default_rng(3), raw_bits=None)
    assert float(cm.luma(with_raw).mean()) == pytest.approx(float(cm.luma(without).mean()), abs=1)


def test_white_balance_gains_and_the_full_well_knob_act_on_the_sensor():
    np = pytest.importorskip("numpy")
    scene = np.full((60, 80, 3), 300.0 * 0.4, dtype=np.float32)
    controls = cm.Controls(600, 400, 256)
    plain = cm.expose(scene, controls, None).astype(float).mean(axis=(0, 1))
    warm_fix = cm.expose(scene, controls, None, wb=(0.7, 1.0, 1.4)).astype(float)
    warm_fix = warm_fix.mean(axis=(0, 1))
    assert warm_fix[0] < plain[0] and warm_fix[1] == plain[1] and warm_fix[2] > plain[2]
    dusk = cm.Controls(1200, 2816, 768)
    grey = np.full((180, 320, 3), 5.0 * 0.4, dtype=np.float32)
    phase0 = cm.expose(grey, dusk, np.random.default_rng(1))
    deeper = cm.expose(grey, dusk, np.random.default_rng(1), full_well=4 * cm.FULL_WELL_E)
    assert cm.luma(deeper).std() < cm.luma(phase0).std()  # a deeper well, less shot noise


def test_uyvy_packs_as_the_official_simulator_does():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(36, 64, 3), dtype=np.uint8)
    data = cm.to_uyvy(rgb)
    assert isinstance(data, bytes) and len(data) == 36 * 64 * 2
    packed = np.frombuffer(data, dtype=np.uint8).reshape(36, 32, 4)
    y = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    assert (packed[:, :, 1] == np.clip(y[:, 0::2], 0, 255).astype(np.uint8)).all()
    pairs = np.repeat(rgb[:, 0::2], 2, axis=1)  # each pixel pair one colour: no chroma lost
    back = cm.uyvy_to_rgb(cm.to_uyvy(pairs), 64, 36)
    assert back.shape == rgb.shape and np.abs(back.astype(int) - pairs).mean() < 1.5
    flat = np.full((36, 64, 3), 128, dtype=np.uint8)
    assert np.abs(cm.uyvy_to_rgb(cm.to_uyvy(flat), 64, 36).astype(int) - 128).max() <= 1
    assert cm.mean_luma(cm.to_uyvy(flat)) == pytest.approx(127, abs=1)
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    official = Path(folder).parents[1] / "sim" / "camera.py" if folder else None
    if official is None or not official.is_file() or importlib.util.find_spec("mujoco") is None:
        pytest.skip("official sim/camera.py (and MuJoCo to import it) not here")
    spec = importlib.util.spec_from_file_location("official_sim_camera", official)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert data == module.to_uyvy(rgb), "byte for byte the official packing"
    big = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    assert cm.to_uyvy(big) == module.to_uyvy(big)


# ── rendering (MuJoCo) ────────────────────────────────────────────────────


def _mujoco():
    pytest.importorskip("numpy", reason="the camera model's renders need numpy")
    return pytest.importorskip("mujoco", reason="the camera model's renders need mujoco")


def _renderer(eye):
    try:
        return cm.HeadCameraRenderer(eye)
    except Exception as error:  # no offscreen GL on this machine
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")


def _official_scene() -> Path:
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    scene = cm.default_scene(folder) if folder else None
    if scene is None or not scene.is_file():
        pytest.skip("official microduck_rl checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    return scene


def _checkerboard(mujoco, *, depth, square, cols, rows, width, height):
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    cam = spec.worldbody.add_camera()
    cam.name = "head_camera"
    board = spec.worldbody.add_geom()
    board.type = mujoco.mjtGeom.mjGEOM_BOX
    board.size = [3.0, 3.0, 0.001]
    board.pos = [0.0, 0.0, -depth - 0.002]
    squares = {}
    for i in range(-cols, cols + 1):
        for j in range(-rows, rows + 1):
            if (i + j) % 2:
                continue
            g = spec.worldbody.add_geom()
            g.name = f"sq_{i}_{j}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [square / 2, square / 2, 0.0005]
            g.pos = [i * square, j * square, -depth]
            g.rgba = [0.0, 0.0, 0.0, 1.0]
            squares[g.name] = (i * square, j * square)
    return spec, cam, squares


def _centroids(mujoco, model, seg, squares, width, height):
    """Mean pixel of every square wholly inside the frame (pixel centres at integers)."""
    import numpy as np

    out = {}
    for name, xy in squares.items():
        ys, xs = np.nonzero(seg == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if len(xs) < 20 or xs.min() == 0 or ys.min() == 0:
            continue
        if xs.max() == width - 1 or ys.max() == height - 1:
            continue
        out[name] = (xy, float(xs.mean()), float(ys.mean()))
    return out


def _checkerboard_eye(mujoco, *, offsamples):
    """The P5 checkerboard (1 m, family K) as an EyeScene whose model has ``offsamples``."""
    width, height = cm.FRAME_SIZE
    spec, cam, _ = _checkerboard(
        mujoco, depth=1.0, square=0.05, cols=14, rows=10, width=width, height=height
    )
    cm.set_intrinsics(cam, K, (width, height))
    spec.visual.quality.offsamples = offsamples
    model = spec.compile()
    model.vis.map.znear = cm.NEAR_CLIP_M / model.stat.extent
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return cm.EyeScene(model, data, 0, {}, -1, {}, K, Path("checkerboard"))


def test_segmentation_ids_do_not_depend_on_the_models_multisampling():
    """Regression (Spark, NVIDIA EGL, 2026-10-08): EGL ignores MuJoCo's automatic switching-off of
    multisampling for a segmentation pass, so a model with ``offsamples=4`` (MuJoCo's default)
    returned blended ids on the edges of the board's squares, ids of other squares, and the P5
    fit came out squashed (fy 873 for K's 1062). The segmentation pass has its own context
    without multisampling, so the ids are the same for any ``offsamples``."""
    mujoco = _mujoco()

    width, height = cm.FRAME_SIZE
    segs = []
    for samples in (4, 0):
        renderer = _renderer(_checkerboard_eye(mujoco, offsamples=samples))
        try:
            segs.append(renderer.segment_ids(width, height))
        finally:
            renderer.close()
    assert int((segs[0] != segs[1]).sum()) == 0


def test_the_segmentation_pass_gets_a_context_without_multisampling(monkeypatch):
    """Where the GL driver honours MuJoCo's switch (macOS) the ids are clean either way, so this
    pins the mechanism: the colour pass keeps the model's ``offsamples``, the segmentation pass's
    ``MjrContext`` is made with 0, and the model's value is put back."""
    mujoco = _mujoco()
    made: list[int] = []
    real = mujoco.MjrContext

    def spy(model, scale):
        made.append(int(model.vis.quality.offsamples))
        return real(model, scale)

    eye = _checkerboard_eye(mujoco, offsamples=4)
    monkeypatch.setattr(mujoco, "MjrContext", spy)
    renderer = _renderer(eye)
    try:
        renderer.render_pinhole(64, 36)
        assert made == [4]
        renderer.segment_ids(64, 36)
        renderer.segment_ids(64, 36)
        assert made == [4, 0], "one extra context, made once"
        assert int(eye.model.vis.quality.offsamples) == 4
    finally:
        renderer.close()


def test_p5_checkerboard_render_matches_family_k():
    """A checkerboard 1 m in front of a camera carrying K (pinhole): every square's bearing
    within 0.5°, and K fitted back from the render (so the principal point's sign is K's)."""
    mujoco = _mujoco()
    import numpy as np

    width, height = cm.FRAME_SIZE
    depth = 1.0
    spec, cam, squares = _checkerboard(
        mujoco, depth=depth, square=0.05, cols=14, rows=10, width=width, height=height
    )
    cm.set_intrinsics(cam, K, (width, height))
    model = spec.compile()
    model.vis.map.znear = cm.NEAR_CLIP_M / model.stat.extent
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eye = cm.EyeScene(model, data, 0, {}, -1, {}, K, Path("checkerboard"))
    renderer = _renderer(eye)
    try:
        seg = renderer.segment_ids(width, height)
    finally:
        renderer.close()
    errors, rows = [], []
    for (x, y), u, v in _centroids(mujoco, model, seg, squares, width, height).values():
        measured = K.bearing_deg(u, v)
        truth = (math.degrees(math.atan(x / depth)), math.degrees(math.atan(-y / depth)))
        errors.append(max(abs(measured[0] - truth[0]), abs(measured[1] - truth[1])))
        rows.append((x / depth, -y / depth, u, v))
    assert len(errors) >= 100, "the board should fill the frame"
    assert max(errors) < 0.5
    a = np.array(rows)
    fx, cx = np.polyfit(a[:, 0], a[:, 2], 1)
    fy, cy = np.polyfit(a[:, 1], a[:, 3], 1)
    assert (fx, fy) == pytest.approx((K.fx, K.fy), abs=2.0)
    assert (cx, cy) == pytest.approx((K.cx, K.cy), abs=2.0)
    # the sign the survey's probe used would have put the axis at (683, 245)
    assert abs(cy - (720 - K.cy)) > 200


@pytest.mark.parametrize("quality", ["720p30", "360p30"])
def test_p5_checkerboard_through_the_lens_to_the_frame_edges(quality):
    """P5 with distortion: the checkerboard through the family lens (the pinhole canvas and
    the remap). Every square's bearing, undone with family K + distortion (as the duck's eye
    will undo it), within 0.5° of the truth — out to the frame's edges, beyond where a
    pinhole of K reaches — and the on-axis square on K's principal point (its sign)."""
    mujoco = _mujoco()
    rung = cm.quality_rung(quality)
    width, height = rung.size
    k = rung.k
    lens = cm.Lens(k)
    depth = 1.0
    spec, cam, squares = _checkerboard(
        mujoco, depth=depth, square=0.05, cols=19, rows=13, width=2000, height=1000
    )
    cm.set_intrinsics(cam, lens.canvas, (lens.canvas.width, lens.canvas.height))
    model = spec.compile()
    model.vis.map.znear = cm.NEAR_CLIP_M / model.stat.extent
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eye = cm.EyeScene(model, data, 0, {}, -1, {}, k, Path("checkerboard"), lens=lens)
    renderer = _renderer(eye)
    try:
        seg = renderer.segment_ids(width, height)
    finally:
        renderer.close()
    found = _centroids(mujoco, model, seg, squares, width, height)
    errors, rights, lefts = [], [], []
    for (x, y), u, v in found.values():
        measured = cm.bearing_deg(u, v, k)
        truth = (math.degrees(math.atan(x / depth)), math.degrees(math.atan(-y / depth)))
        errors.append(max(abs(measured[0] - truth[0]), abs(measured[1] - truth[1])))
        (rights if truth[0] > 0 else lefts).append(truth[0])
    assert len(errors) >= (150 if quality == "720p30" else 100)
    assert max(errors) < 0.5, f"worst bearing error {max(errors):.3f}°"
    pinhole = k.fov_deg()
    assert max(rights) > pinhole["right"] + 2.0, "squares past the pinhole's right edge"
    assert min(lefts) < -(pinhole["left"] + 1.0), "squares past the pinhole's left edge"
    _, u0, v0 = found["sq_0_0"]
    assert (u0, v0) == pytest.approx((k.cx, k.cy), abs=1.5)


def test_p5_orientation_after_the_flip_on_the_official_duck():
    """The official head camera with the 180° roll kept, K and the lens applied, turned upright
    like ``mediad --rotate 90``: up is up, the duck's left is left, the axis lands on K's
    point (distortion leaves the principal point where it is)."""
    mujoco = _mujoco()
    import numpy as np

    scene = _official_scene()
    base = cm.load_scene(scene)
    pose = cm.keyframe_pose(base, "STAND")
    cm.set_pose(base, pose)
    cam_pos = base.data.cam_xpos[base.camera].copy()
    axis = -base.data.cam_xmat[base.camera].reshape(3, 3)[:, 2]
    assert axis[0] > 0.99, "the flipped camera looks the way the duck faces (+x)"
    marks = {
        "m_axis": cam_pos + 2.0 * axis,
        "m_up": cam_pos + 2.0 * axis + np.array([0.0, 0.0, 0.5]),
        "m_left": cam_pos + 2.0 * axis + np.array([0.0, 0.4, 0.0]),  # the duck's left is +y
    }

    def add_marks(spec):
        for name, where in marks.items():
            g = spec.worldbody.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [0.03, 0.0, 0.0]
            g.pos = [float(v) for v in where]
            g.contype = 0
            g.conaffinity = 0

    eye = cm.load_scene(scene, edit=add_marks)
    cm.set_pose(eye, pose)
    width, height = cm.FRAME_SIZE
    renderer = _renderer(eye)
    try:
        seg = renderer.segment_ids(width, height)
    finally:
        renderer.close()
    upright = {}
    for name in marks:
        ys, xs = np.nonzero(seg == mujoco.mj_name2id(eye.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        assert len(xs) > 100, name
        u, v = float(xs.mean()), float(ys.mean())
        upright[name] = (height - 1 - v, u)  # uyvy::Turn::Right: (x, y) → (h-1-y, x)
    kup = K.upright()
    assert upright["m_axis"] == pytest.approx((kup.cx, kup.cy), abs=3.0)
    assert upright["m_up"][1] < upright["m_axis"][1] - 200, "up in the room is up in the picture"
    assert abs(upright["m_up"][0] - upright["m_axis"][0]) < 3
    assert upright["m_left"][0] < upright["m_axis"][0] - 150, "the duck's left is on the left"
    useg = cm.upright(seg)
    floor = mujoco.mj_name2id(eye.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ceiling = mujoco.mj_name2id(eye.model, mujoco.mjtObj.mjOBJ_GEOM, "dm_ceiling")
    floor_rows = np.nonzero((useg == floor).any(axis=1))[0]
    ceiling_rows = np.nonzero((useg == ceiling).any(axis=1))[0]
    assert floor_rows.min() > ceiling_rows.max() - 3, "floor below, ceiling above"


def test_the_scene_is_set_up_like_the_real_camera():
    mujoco = _mujoco()
    scene = _official_scene()
    eye = cm.load_scene(scene)
    m = eye.model
    assert m.vis.map.znear * m.stat.extent == pytest.approx(cm.NEAR_CLIP_M)
    assert list(m.vis.headlight.diffuse) == [0.0, 0.0, 0.0]
    assert list(m.vis.headlight.specular) == [0.0, 0.0, 0.0]
    ours = set(eye.lights.values())
    assert len(ours) == cm.LIGHT_SLOTS == 7 and all(i >= 0 for i in ours)
    assert all(m.light_active[i] == 0 for i in range(m.nlight))  # nothing lit until a frame
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "dm_ceiling") >= 0
    # with the lens the camera carries the canvas; the frame stays 1280×720
    assert eye.lens is not None and eye.frame_size == (1280, 720)
    assert (m.cam_resolution[eye.camera] == [1615, 913]).all()
    assert m.vis.global_.offwidth >= 1615 and m.vis.global_.offheight >= 913
    assert eye.quality is not None and eye.quality.label == "720p30"
    pinhole = cm.load_scene(scene, distortion=False)
    assert pinhole.lens is None
    assert pinhole.model.cam_fovy[pinhole.camera] == pytest.approx(37.45, abs=0.05)
    assert (pinhole.model.cam_resolution[pinhole.camera] == [1280, 720]).all()
    # only the duck starts at STAND: nothing else is moved by the official keyframe
    assert eye.trunk_qpos >= 0 and set(eye.joints) <= set(cm.keyframe_pose(eye)["joints"])


def test_lights_put_a_white_floor_at_render_white_at_every_hour():
    """world.lights → MuJoCo lights: a white level surface at the reference spot renders
    RENDER_WHITE (luminance) for day, dusk, lamps and moonlight, so scene_lux() gives back the
    lux of the light state."""
    mujoco = _mujoco()
    import numpy as np

    from duckmind.world import lights

    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    material = spec.add_material()
    material.name = "white"
    material.rgba = [1.0, 1.0, 1.0, 1.0]
    material.specular = 0.0
    material.shininess = 0.0
    material.reflectance = 0.0
    floor.material = "white"
    cam = spec.worldbody.add_camera()
    cam.name = "head_camera"
    cam.pos = [0.0, 0.0, 0.3]
    cam.quat = [0.0, 1.0, 0.0, 0.0]  # looks up; the camera model's 180° roll turns it down
    cm.apply_to_spec(spec, frame_size=cm.METER_SIZE)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lights_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, n) for n in cm.LIGHT_NAMES}
    eye = cm.EyeScene(model, data, 0, lights_ids, -1, {}, K, Path("white"), ceiling_m=cm.CEILING_M)
    renderer = _renderer(eye)
    try:
        for when in (
            datetime(2026, 9, 27, 12, 0),
            datetime(2026, 12, 1, 16, 50),
            datetime(2026, 9, 27, 20, 0),
            datetime(2026, 9, 28, 2, 0),
        ):
            state = lights.light_state(when)
            per_unit = cm.set_lights(eye, state)
            rgb = renderer.render(*cm.METER_SIZE)[40:50, 75:85].reshape(-1, 3).mean(axis=0)
            linear_luminance = float(np.dot(rgb / 255.0, [0.2126, 0.7152, 0.0722]))
            assert linear_luminance == pytest.approx(cm.RENDER_WHITE, rel=0.05), state.label
            assert linear_luminance * per_unit == pytest.approx(state.lux, rel=0.05)
    finally:
        renderer.close()


def _glow_scene(mujoco):
    """A white floor, a small panel that can glow, a camera looking down at both."""
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    white = spec.add_material()
    white.name = "white"
    white.rgba = [1.0, 1.0, 1.0, 1.0]
    white.specular = 0.0
    white.shininess = 0.0
    white.reflectance = 0.0
    floor.material = "white"
    panel = spec.worldbody.add_geom()
    panel.name = "test_backlight"
    panel.type = mujoco.mjtGeom.mjGEOM_BOX
    panel.size = [0.06, 0.06, 0.002]
    panel.pos = [0.0, 0.0, 0.003]
    panel.rgba = [0.2, 0.2, 0.2, 1.0]  # a dark panel when it does not glow
    cam = spec.worldbody.add_camera()
    cam.name = "head_camera"
    cam.pos = [0.0, 0.0, 0.4]
    cam.quat = [0.0, 1.0, 0.0, 0.0]  # looks up; the 180° roll turns it down
    return spec


def test_a_glow_is_as_bright_by_day_as_by_night_and_skipped_out_of_view():
    """The HDR path: an emitter adds its absolute brightness (lux × albedo) whatever the room's
    light, where the normalised render alone would have made it 1200× brighter by day."""
    mujoco = _mujoco()
    from duckmind.world import lights

    eye = cm.load_spec(
        _glow_scene(mujoco),
        quality="360p30",
        emitters=["test_backlight"],
        ceiling_m=cm.CEILING_M,
    )
    camera = cm.HeadCamera(eye)
    width, height = 640, 360
    middle = (slice(height // 2 - 8, height // 2 + 8), slice(width // 2 - 8, width // 2 + 8))
    glow_lux = 30.0
    added = {}
    try:
        seg = camera.renderer.segment_ids(width, height)
        panel = mujoco.mj_name2id(eye.model, mujoco.mjtObj.mjOBJ_GEOM, "test_backlight")
        assert (seg[middle] == panel).all(), "the panel fills the middle of the frame"
        for key, when in (("day", datetime(2026, 9, 27, 12)), ("moon", datetime(2026, 9, 28, 2))):
            state = lights.light_state(when)
            per_unit = cm.set_lights(eye, state)
            cm.set_emitters(eye, {})
            dark, glowed, _ = camera.scene(width, height, per_unit)
            assert not glowed
            cm.set_emitters(eye, {"test_backlight": ((1.0, 1.0, 1.0), glow_lux)})
            bright, glowed, _ = camera.scene(width, height, per_unit)
            assert glowed
            added[key] = float((bright[middle] - dark[middle]).mean())
        assert added["day"] == pytest.approx(glow_lux, rel=0.03)
        assert added["moon"] == pytest.approx(glow_lux, rel=0.03)
        # facing away (the camera turned to look up): no second pass
        eye.data.cam_xmat[eye.camera] = [1, 0, 0, 0, -1, 0, 0, 0, -1]
        assert cm.emitters_in_view(eye) == []
    finally:
        camera.close()


@pytest.mark.parametrize("quality", ["360p30", "1080p30"])
def test_a_capture_has_its_rungs_size_and_packs_to_media_frame_bytes(quality):
    _mujoco()
    from duckmind.world import lights

    rung = cm.quality_rung(quality)
    eye = cm.load_scene(_official_scene(), quality=quality)
    assert eye.frame_size == rung.size and eye.k == rung.k
    try:
        camera = cm.HeadCamera(eye, seed=3)
    except Exception as error:
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")
    try:
        cm.set_pose(eye, cm.keyframe_pose(eye, "STAND"))
        rig = cm.light_rig(eye, lights.light_state(datetime(2026, 9, 27, 12)))
        camera.converge(rig.lux_per_unit, illuminant=rig.color)
        capture = camera.capture(rig.lux_per_unit)
    finally:
        camera.close()
    assert capture.rgb.shape == (rung.height, rung.width, 3)
    assert (capture.width, capture.height, capture.rotate) == (rung.width, rung.height, 90)
    assert capture.format == "UYVY" and len(capture.uyvy()) == capture.bytes_len
    assert capture.upright().shape == (rung.width, rung.height, 3)
    assert abs(float(cm.luma(capture.rgb).mean()) - cm.TARGET_Y) < 0.13 * cm.TARGET_Y + 3
    assert capture.wb == pytest.approx((1.0, 1.0, 1.0), abs=0.02)  # noon light is white


def test_vignetting_is_a_knob_and_off_by_default():
    _mujoco()
    eye = cm.load_scene(_official_scene(), quality="360p30")
    try:
        plain = cm.HeadCamera(eye)
    except Exception as error:
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")
    try:
        assert plain._vignette_map(640, 360) is None
        shaded = cm.HeadCamera(eye, config=cm.CameraConfig(quality="360p30", vignetting=0.5))
        try:
            gain = shaded._vignette_map(640, 360)
        finally:
            shaded.close()
    finally:
        plain.close()
    k = eye.k
    assert gain[round(k.cy), round(k.cx)] == pytest.approx(1.0, abs=1e-3)
    assert 0.5 < gain[0, 0] < gain[round(k.cy), 0] < 1.0
