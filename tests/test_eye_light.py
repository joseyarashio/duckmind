"""eye.light: the light meter's two readings against the survey's light ladder."""

import pytest

from duckmind.eye import light
from duckmind.eye.light import Exposure

# survey_calibration.md §2.3: AE state, mean luma and luma noise σ per rung.
LADDER = [
    # rung, level, exposures (lines, analogue, digital) seen, mean luma, σ range
    ("day 300 lux", "bright", [(600, 1.3, 1.0), (600, 2.0, 1.0)], 86.0, (1.6, 1.9)),
    ("lamps ~120 lux", "bright", [(600, 8.0, 1.0), (745, 11.0, 1.0)], 82.0, (2.7, 3.3)),
    ("dusk 5 lux", "dim", [(1200, 11.0, 2.0), (1200, 11.0, 3.0)], 80.0, (4.8, 5.6)),
    ("twilight 1 lux", "dark", [(1200, 11.0, 11.0), (1200, 11.0, 16.0)], 80.0, (12.0, 18.0)),
    ("night 0.25 lux", "black", [(1200, 11.0, 16.0)], 35.0, (21.0, 21.0)),
]


@pytest.mark.parametrize("rung,level,exposures,luma,sigmas", LADDER)
def test_the_exposure_rule_reads_the_survey_ladder(rung, level, exposures, luma, sigmas):
    for lines, analogue, digital in exposures:
        assert light.level_from_exposure(luma, Exposure(lines, analogue, digital)) == level, rung


@pytest.mark.parametrize("rung,level,exposures,luma,sigmas", LADDER)
def test_the_frame_rule_reads_the_survey_ladder(rung, level, exposures, luma, sigmas):
    for sigma in sigmas:
        assert light.level_from_frame(luma, sigma) == level, rung


def test_a_room_keeps_its_level_whichever_way_the_exposure_came():
    # Measured on the eye renders: mediad's deadband lets a 5-lux room settle at 2.7x digital
    # coming from daylight and at 4.2x coming from moonlight; a lamp-lit one at 1.0x or 1.09x.
    for digital, luma in ((2.7, 81.0), (4.23, 99.7)):
        assert light.level_from_exposure(luma, Exposure(1200, 11.0, digital)) == "dim"
    for lines, digital in ((785, 1.0), (1200, 1.094)):
        assert light.level_from_exposure(95.0, Exposure(lines, 11.0, digital)) == "bright"
    # twilight (1.4 lux) at 9.6-16x is dark either way
    for digital in (9.6, 10.5, 16.0):
        assert light.level_from_exposure(82.0, Exposure(1200, 11.0, digital)) == "dark"


def test_black_needs_the_loop_at_its_ceiling():
    dark_room_bright_frame = Exposure(1200, 11.0, 16.0)
    assert light.level_from_exposure(44.0, dark_room_bright_frame) == "black"
    assert light.level_from_exposure(46.0, dark_room_bright_frame) == "dark"
    # A frame that is dark only because the loop has not caught up yet is not "black".
    assert light.level_from_exposure(20.0, Exposure(600, 4.0, 1.0)) == "bright"
    assert Exposure.from_registers(1200, 2816, 4096) == dark_room_bright_frame
    assert dark_room_bright_frame.at_ceiling and not Exposure(1199, 11.0, 1.0).at_ceiling


def test_an_exposure_still_catching_up_is_seen():
    lamps = Exposure(600, 4.0, 1.0)  # the lamp-lit room's exposure, a moment after lamps off
    assert light.adapting(7.8, lamps)  # nearly black at lamp exposure: not caught up
    assert not light.adapting(40.0, Exposure(1200, 11.0, 16.0))  # at its ceiling: that is black
    assert not light.adapting(84.0, lamps) and not light.adapting(120.0, lamps)
    assert light.adapting(230.0, Exposure(1200, 11.0, 16.0))  # lamps on after the dark
    assert not light.adapting(230.0, Exposure(4, 1.0, 1.0))  # at its floor: truly that bright
    assert Exposure(4, 1.0, 1.0).at_floor and not lamps.at_floor


def test_the_lux_estimate_is_the_survey_constant():
    # An 18 % grey card at 400 lux converges at budget 1125 (11.4 ms at 1.875x) at luma 90.
    assert light.lux_estimate(90.0, Exposure(600, 1.875, 1.0)) == pytest.approx(400.0)
    assert light.lux_estimate(90.0, Exposure(1200, 11.0, 16.0)) < 3.0


def test_measure_without_numpy_is_not_required_for_the_rules():
    reading = light.LightReading(
        level="dim",
        level_zh="昏暗",
        method="exposure",
        luma=80.0,
        noise=5.0,
        level_frame="dim",
        level_exposure="dim",
        lux_estimate=5.0,
        exposure=None,
    )
    assert reading.agree and reading.to_dict()["agree"] is True
    assert light.LEVEL_ZH == {"bright": "亮", "dim": "昏暗", "dark": "很暗", "black": "全黑"}


# ── on frames (numpy) ─────────────────────────────────────────────────────


def _noise_plane(sigma: float, *, seed: int = 0, structure: bool = False, denoise: float = 0.0):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    h, w = 1280, 720
    base = np.full((h, w), 100.0)
    if structure:
        yy, xx = np.mgrid[0:h, 0:w]
        base = 70.0 + 40.0 * (((yy // 90) + (xx // 90)) % 2) + 0.02 * yy  # squares and a slope
    noise = rng.normal(0.0, sigma, (h, w))
    if denoise:
        p = np.pad(noise, 1, mode="edge")
        box = sum(p[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)) / 9
        noise = (1 - denoise) * noise + denoise * box
    return base + noise, float(noise.std())


@pytest.mark.parametrize("sigma", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_noise_estimate_on_flat_frames(sigma):
    plane, truth = _noise_plane(sigma)
    assert light.noise_sigma(plane) == pytest.approx(truth, rel=0.10)


@pytest.mark.parametrize("sigma", [2.0, 5.0, 12.0])
def test_noise_estimate_ignores_edges_and_slopes(sigma):
    plane, truth = _noise_plane(sigma, structure=True, seed=3)
    assert light.noise_sigma(plane) == pytest.approx(truth, rel=0.15)


def test_noise_estimate_sees_through_the_isp_denoise():
    # The ISP's box blend ties neighbours together; every fourth pixel is independent again.
    plane, truth = _noise_plane(8.0, denoise=0.5, seed=5)
    assert light.noise_sigma(plane) == pytest.approx(truth, rel=0.15)


def test_measure_reads_both_ways():
    np = pytest.importorskip("numpy")
    plane, _ = _noise_plane(5.0, seed=9)
    plane = np.clip(plane - 20.0, 0, 255)  # mean 80, σ 5
    by_frame = light.measure(plane)
    assert (by_frame.method, by_frame.level, by_frame.level_exposure) == ("frame", "dim", None)
    assert by_frame.luma == pytest.approx(80.0, abs=0.5)
    by_exposure = light.measure(plane, Exposure(1200, 11.0, 2.5))
    assert (by_exposure.method, by_exposure.level, by_exposure.agree) == ("exposure", "dim", True)
    assert by_exposure.exposure is not None and by_exposure.exposure["shutter_ms"] == 22.86
    disagree = light.measure(plane, Exposure(600, 2.0, 1.0))
    assert disagree.level == "bright" and disagree.level_frame == "dim" and not disagree.agree
    assert not by_exposure.adapting and not disagree.adapting


def test_a_frame_the_exposure_has_not_caught_up_with_is_judged_by_the_frame():
    """Lamps off: the first frame still has the lamp-lit exposure (digital 1x) and is nearly
    black. The exposure alone would say 亮; the frame decides, and says it is adapting."""
    np = pytest.importorskip("numpy")
    plane, _ = _noise_plane(3.0, seed=4)
    dark = np.clip(plane - 92.0, 0, 255)  # mean ~8
    reading = light.measure(dark, Exposure(600, 4.0, 1.0))
    assert reading.adapting and reading.method == "frame"
    assert reading.level == reading.level_frame == "black"
    assert reading.level_exposure == "bright"  # (what the old rule said)
    assert reading.to_dict()["adapting"] is True


def test_mean_luma_samples_every_eighth_pixel():
    np = pytest.importorskip("numpy")
    plane = np.zeros((4, 16), dtype=np.uint8)
    plane.reshape(-1)[::8] = 80  # only the sampled pixels are lit
    assert light.frame_mean_luma(plane) == 80.0


# ── v2: the board path in integers, C1's dock light, hysteresis, settling, the C3 fit ──


def _plane(mean: float, sigma: float, *, seed: int = 0, shape=(320, 180)):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(np.full(shape, mean) + rng.normal(0.0, sigma, shape)), 0, 255).astype(
        np.uint8
    )


@pytest.mark.parametrize("sigma", [1.0, 3.0, 6.0, 12.0, 20.0])
def test_the_integer_noise_estimate_is_the_float_one(sigma):
    np = pytest.importorskip("numpy")
    full, _ = _noise_plane(sigma, seed=11)
    full = np.clip(np.rint(full), 0, 255).astype(np.uint8)
    milli = light.noise_milli(full[::4, ::4])
    assert isinstance(milli, int)
    assert milli / 1000 == pytest.approx(light.noise_sigma(full), rel=0.01)


def test_the_board_numbers_are_integers_and_the_same_every_time():
    plane = _plane(80.0, 5.0, seed=2)
    a = light.read_quarter(plane)
    b = light.read_quarter(plane.copy())
    assert a == b and all(isinstance(v, int) for v in (a.luma, a.sigma_milli, a.glow_px))
    assert a.luma == 80 and a.noise_q == 5 and a.glow_px == 0


def test_the_levels_on_the_refit_thresholds():
    t = light.THRESHOLDS
    fl = light.FrameLight
    assert light.level_q(fl(95, t.sigma_bright_milli - 1, 0)) == "bright"
    assert light.level_q(fl(95, t.sigma_bright_milli, 0)) == "dim"
    assert light.level_q(fl(95, t.sigma_dim_milli - 1, 0)) == "dim"
    assert light.level_q(fl(95, t.sigma_dim_milli, 0)) == "dark"
    assert light.level_q(fl(t.black_luma - 1, 21000, 0)) == "black"
    assert light.level_q(fl(t.black_luma, 21000, 0)) == "dark"
    # owner decision C1: black but the dock's light in view → very dark
    assert light.level_q(fl(30, 21000, 12)) == "dark"
    assert t.source.startswith("C3 refit") and light.SPEC_THRESHOLDS.black_luma == 45


def test_the_dock_light_is_a_compact_glow_and_noise_is_not():
    dark = _plane(35.0, 21.0, seed=4)
    assert light.glow_q(dark) == 0  # 9σ above the mean: never noise
    strip = dark.copy()
    strip[150, 60:74] = 250  # the dock's strip at ~3 m: 14 × 1 quarter-resolution pixels
    assert light.glow_q(strip) == 14
    assert light.level_q(light.read_quarter(strip)) == "dark"
    specks = dark.copy()
    specks[[5, 300, 10, 200], [3, 170, 160, 20]] = 255  # hot pixels far apart: no glow
    assert light.glow_q(specks) == 0
    tiny = dark.copy()
    tiny[100, 50:53] = 250  # fewer than glow_min_px
    assert light.glow_q(tiny) == 0
    sparkle = dark.copy()
    sparkle[::20, ::20] = 250  # bright points all over the picture: not one light
    assert light.glow_q(sparkle) == 0


def test_only_the_docks_strip_turns_black_into_very_dark():
    """Owner C1 counts the *dock's* light only (phase-3 review): any other small light in a
    black room — a standby LED, a phone, a router's dots — leaves the frame black, the level a
    camera-only duck never departs on."""
    dark = _plane(35.0, 21.0, seed=4)
    led = dark.copy()
    led[100:104, 60:64] = 250  # a round standby LED or a phone's screen corner: 4 × 4
    assert light.glow_q(led) == 0
    assert light.level_q(light.read_quarter(led)) == "black"
    tall = dark.copy()
    tall[90:100, 60:63] = 250  # an upright light (a phone held on end, a lamp's slit)
    assert light.glow_q(tall) == 0
    dots = dark.copy()
    dots[120, 40:70:6] = 250  # a router's row of dots: wide, but not a filled strip
    dots[121, 40:70:6] = 250
    assert light.glow_q(dots) == 0
    near = dark.copy()
    near[140:146, 40:90] = 250  # the dock's strip close up (50 × 6): still the dock
    assert light.glow_q(near) == 300
    assert light.level_q(light.read_quarter(near)) == "dark"
    assert light.strip_like(14, 14, 1) and not light.strip_like(16, 4, 4)


def test_darker_counts_at_once_brighter_after_two_looks_that_agree():
    tracker = light.LightTracker()
    bright = light.FrameLight(95, 1800, 0)
    dim = light.FrameLight(92, 4500, 0)
    dark = light.FrameLight(85, 9000, 0)
    s = 1_000_000_000
    assert tracker.update(bright, 0)["level"] == "bright"  # the first look sets it
    # getting darker takes effect on the first look that sees it (a duck must never walk on a
    # level brighter than its picture: phase-3 verify, 23/30 departures in pitch dark)
    one = tracker.update(dim, 10 * s)
    assert (one["level"], one["raw"], one["pending"]) == ("dim", "dim", 0)
    assert tracker.update(dark, 20 * s)["level"] == "dark"
    # getting brighter needs two looks that agree; a different one restarts the count
    up = tracker.update(bright, 30 * s)
    assert (up["level"], up["raw"], up["pending"]) == ("dark", "bright", 1)
    assert tracker.update(dim, 40 * s)["pending"] == 1  # dim: a new candidate
    assert tracker.update(bright, 50 * s)["pending"] == 1
    assert tracker.update(bright, 60 * s)["level"] == "bright"
    out = tracker.update(bright, 70 * s)
    assert set(out) >= {"level", "luma", "noise_q", "raw", "glow", "settle_ms", "pending"}
    assert all(isinstance(v, (int, str)) for v in out.values())
    assert light.darker("bright", "dark") == "dark" and light.darker("black", "dim") == "black"


def test_a_look_while_the_exposure_is_still_moving_waits_half_a_second():
    s = 1_000_000_000
    # lamps off a moment ago: dark frame, but the gain is still low
    assert light.settling(light.FrameLight(20, 1500, 0))
    # lamps on after the dark: glaring
    assert light.settling(light.FrameLight(200, 20000, 0))
    # inside mediad's deadband the loop does not move, whatever the look before said
    assert not light.settling(light.FrameLight(100, 1800, 0), light.FrameLight(80, 1800, 0), s)
    # spec §4.5's 12 %: just outside the deadband, 0.4 s after a look at 90
    assert light.settling(light.FrameLight(115, 1800, 0), light.FrameLight(90, 1800, 0), s // 2)
    assert not light.settling(light.FrameLight(115, 1800, 0), light.FrameLight(90, 1800, 0), 5 * s)
    # black is the light itself, not the loop
    assert not light.settling(light.FrameLight(40, 21000, 0), light.FrameLight(90, 5000, 0), s)
    tracker = light.LightTracker()
    tracker.update(light.FrameLight(95, 1800, 0), 0)
    wait = tracker.update(light.FrameLight(20, 1500, 0), s)
    # the frame does not count yet (look again in 0.5 s) — but the level answered is never
    # brighter than the frame itself: the first look after the lamps go out reads black
    assert wait["settle_ms"] == 500 and wait["pending"] == 0
    assert (wait["level"], wait["raw"]) == ("black", "black")
    assert tracker.level == "bright"  # the tracker's own level waits for a counted look
    again = tracker.update(light.FrameLight(22, 1500, 0), s + s // 2)
    assert again["settle_ms"] == 500 and again["level"] == "black"
    # after two waits in a row the frame is taken as it is: darker, so at once
    taken = tracker.update(light.FrameLight(22, 1500, 0), 2 * s)
    assert taken["settle_ms"] == 0 and taken["pending"] == 0 and tracker.level == "black"


def test_the_truth_bands_and_the_fit():
    assert [light.level_of_lux(x) for x in (300, 30, 29.9, 2.5, 2.4, 0.5, 0.49, 0.0)] == [
        "bright",
        "bright",
        "dim",
        "dim",
        "dark",
        "dark",
        "black",
        "black",
    ]
    pytest.importorskip("numpy")
    fl = light.FrameLight
    samples = (
        [(fl(95, 1500 + 10 * i, 0), "bright") for i in range(50)]
        + [(fl(92, 3600 + 20 * i, 0), "dim") for i in range(50)]
        + [(fl(85, 7000 + 100 * i, 0), "dark") for i in range(50)]
        + [(fl(30 + i // 2, 20000, 0), "black") for i in range(50)]
    )
    fitted, hits = light.fit_thresholds(samples)
    assert hits == 200
    assert 55 <= fitted.black_luma <= 85
    assert 2000 <= fitted.sigma_bright_milli <= 3600 and 4600 <= fitted.sigma_dim_milli <= 7000
    assert light.agreement(samples, fitted) == (200, 200)


# ── golden renders: the eye's board path on the world camera's frames (render tier) ───


def _home_eye(name: str):
    import os
    from pathlib import Path

    pytest.importorskip("numpy", reason="the eye's renders need numpy")
    pytest.importorskip("mujoco", reason="the eye's renders need mujoco")
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not (Path(folder) / "scene_allcollisions.xml").is_file():
        pytest.skip("official microduck_rl checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    from duckmind.world import homelayout, lights
    from duckmind.world import render_scene as rs

    layout = homelayout.load(name, mjcf_dir=folder)
    home = lights.HomeLights.from_layout(layout)
    try:
        eye = rs.HomeEye.open(homelayout.compose(layout), home=home, seed=7)
    except Exception as error:  # no offscreen GL context on this machine
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")
    return layout, home, eye


def _stand(eye, x: float, y: float, yaw: float) -> None:
    import math

    from duckmind.world import camera_model as cm

    pose = cm.keyframe_pose(eye.eye, "STAND")
    pose["trunk"] = [x, y, pose["trunk"][2]]
    pose["quat"] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    eye.set_pose(pose)


def test_the_eye_reads_a_rendered_home_day_and_night_and_the_dock_light():
    """default_home through the phase-2 camera: noon reads bright, a moonless night black, and
    the same night facing the lit dock reads very dark (owner decision C1)."""
    import math
    from datetime import datetime

    from duckmind.eye import frames
    from duckmind.world import lights

    layout, home, eye = _home_eye("default_home")
    try:
        nest = layout.nest
        assert nest is not None
        a = nest.heading_mrad / 1000.0
        x, y = nest.x + 1.4 * math.cos(a), nest.y + 1.4 * math.sin(a)
        toward = math.atan2(nest.y - y, nest.x - x)
        cases = [
            (datetime(2026, 9, 27, 12, 0), True, 1.0, toward, "bright"),
            (datetime(2026, 9, 28, 2, 0), False, 0.0, toward + math.pi, "black"),
            (datetime(2026, 9, 28, 2, 0), False, 0.0, toward, "dark"),  # C1
        ]
        for when, lamps_on, moon, yaw, want in cases:
            _stand(eye, x, y, yaw)
            state = lights.home_light_state(when, home, lamps_on=lamps_on, moon=moon)
            eye.converge(state)
            capture = eye.capture(state)
            data = capture.uyvy()
            planes = frames.sample_quarter(data, capture.width, capture.height, 90)
            reading = light.read_quarter(planes.y)
            assert light.level_q(reading) == want, (want, reading)
            if want == "dark":
                assert reading.glow_px > 0 and reading.luma < light.THRESHOLDS.black_luma
    finally:
        eye.close()


def test_the_same_rendered_bytes_give_the_same_result_p2():
    """P2: a rendered frame through the whole eye core twice gives identical integers."""
    import json
    import math
    from datetime import datetime

    from duckmind.eye import frames, server
    from duckmind.eye import protocol as proto
    from duckmind.world import lights

    layout, home, eye = _home_eye("default_home")
    try:
        _stand(eye, 0.6, 0.0, math.pi)
        state = lights.home_light_state(datetime(2026, 12, 1, 16, 50), home, lamps_on=False)
        eye.converge(state)
        capture = eye.capture(state)
        data = capture.uyvy()
    finally:
        eye.close()
    head = frames.MediaFrameHeader(
        capture.width, capture.height, "UYVY", len(data), 1_790_000_000_000_000, 90
    )
    cam = (0.5, -0.5, 0.5, -0.5)
    poses = [
        proto.PoseSample(1, 990_000_000, 212, (0.05, 0.0, 0.1), cam),
        proto.PoseSample(2, 1_030_000_000, 212, (0.05, 0.0, 0.1), cam),
    ]
    look = proto.EyeLook("L1", ("floor", "nest", "things"), 1, 0)

    def once() -> str:
        core = server.EyeCore(server.EyeConfig())
        result = core.run_frame(look, head, data, captured_at_ns=1_000_000_000, poses=poses)
        assert result.status == "ok" and result.problems() == []
        record = result.to_json()
        record.pop("cost_us")
        return json.dumps(record, sort_keys=True)

    first = once()
    assert first == once()
    assert json.loads(first)["results"]["light"]["level"] in light.LEVELS
