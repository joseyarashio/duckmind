"""simweb.eyecam: one duck's-eye frame from a fixed pose, the render thread, the JPEG.

Rendering needs MuJoCo, numpy, an offscreen GL context and the official checkout
(``DUCKMIND_MICRODUCK_MJCF``); each test says which one is missing when it skips.
"""

import importlib.util
import io
import os
import time
from datetime import datetime

import pytest

from duckmind.world import lights

NOON = datetime(2026, 9, 27, 12, 0)


def _needs_render():
    pytest.importorskip("numpy", reason="eyecam renders need numpy")
    pytest.importorskip("mujoco", reason="eyecam renders need mujoco")
    from duckmind.world import camera_model as cm

    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    scene = cm.default_scene(folder) if folder else None
    if scene is None or not scene.is_file():
        pytest.skip("official microduck_rl checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    return scene


def _renderer(scene, **kwargs):
    from duckmind.simweb.eyecam import EyeRenderer

    try:
        return EyeRenderer(scene, **kwargs)
    except Exception as error:
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")


def test_one_frame_from_a_standing_pose():
    from duckmind.world import camera_model as cm

    scene = _needs_render()
    renderer = _renderer(scene)
    try:
        pose = cm.keyframe_pose(renderer.eye, "STAND")
        state = lights.light_state(NOON)
        ticks = renderer.converge(pose, state)
        out = renderer.frame(pose, state)
    finally:
        renderer.close()
    assert ticks < 20
    data = out["jpeg"]
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    assert (out["width"], out["height"]) == (360, 640)
    assert out["frame_size"] == [720, 1280]
    assert out["label"] == "白天" and out["light_key"] == "day"
    assert out["light"]["level"] == "bright" and out["light"]["method"] == "exposure"
    assert out["light"]["level_frame"] == "bright"
    assert out["ae"]["settled"] and out["ae"]["digital"] == 1.0
    assert abs(out["light"]["luma"] - 90) < 0.13 * 90 + 2, "the AE holds the mean near 90"
    assert out["camera"]["height_m"] == pytest.approx(0.248, abs=0.005)
    assert out["error"] is None and out["render_ms"] > 0
    if importlib.util.find_spec("PIL") is not None:  # decode it when Pillow is around
        from PIL import Image

        assert Image.open(io.BytesIO(data)).size == (360, 640)


def test_frames_are_deterministic_for_a_seed():
    from duckmind.world import camera_model as cm

    scene = _needs_render()
    state = lights.light_state(datetime(2026, 9, 28, 2, 0))  # moonlight: all noise and gain
    frames = []
    for _ in range(2):
        renderer = _renderer(scene, seed=11)
        try:
            pose = cm.keyframe_pose(renderer.eye, "STAND")
            renderer.converge(pose, state)
            frames.append(renderer.frame(pose, state))
        finally:
            renderer.close()
    assert frames[0]["jpeg"] == frames[1]["jpeg"]
    assert frames[0]["light"] == frames[1]["light"]
    assert frames[0]["light"]["level"] == "black"


def test_the_thread_mirrors_a_pose_and_stops():
    from duckmind.simweb.eyecam import EyeCam
    from duckmind.world import camera_model as cm

    scene = _needs_render()
    probe = _renderer(scene)
    pose = cm.keyframe_pose(probe.eye, "STAND")
    probe.close()
    calls = []

    def read_pose():
        calls.append(time.monotonic())
        return pose

    cam = EyeCam(read_pose, scene=scene, clock=lambda: NOON, period_s=0.3, meter_s=0.1)
    cam.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (cam.latest() or {}).get("frames", 0) < 2:
            time.sleep(0.05)
        latest = cam.latest()
        status = cam.status()
    finally:
        cam.stop()
    assert latest is not None, status
    assert latest["frames"] >= 2 and latest["label"] == "白天"
    assert status["running"] and status["error"] is None and status["jpeg_bytes"] > 1000
    assert "jpeg" not in status
    assert len(calls) >= 3
    assert cam.status()["running"] is False


def test_problems_become_words_not_crashes(tmp_path):
    from duckmind.simweb.eyecam import EyeCam

    missing = EyeCam(lambda: {}, scene=tmp_path / "nope.xml")
    missing.start()
    missing.stop()
    assert "找不到場景檔" in missing.status()["error"]
    assert missing.latest() is None
    nowhere = EyeCam(lambda: {}, scene=None, mjcf_dir=None)
    if nowhere.scene is None:
        nowhere.start()
        nowhere.stop()
        assert "沒有指定場景" in nowhere.status()["error"]


def test_no_pose_yet_means_no_frame():
    from duckmind.simweb.eyecam import EyeCam

    scene = _needs_render()
    cam = EyeCam(lambda: None, scene=scene, period_s=0.2, meter_s=0.1)
    cam.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not cam.status().get("running"):
            time.sleep(0.05)
        time.sleep(0.5)
        status = cam.status()
    finally:
        cam.stop()
    if status.get("error"):
        pytest.skip(f"no offscreen GL context here: {status['error']}")
    assert cam.latest() is None and status["note"] == "等物理身體的姿勢"


# ── the JPEG encoder the page's picture uses (numpy, no Pillow needed) ─────────


def test_numpy_jpeg_is_a_valid_baseline_file():
    np = pytest.importorskip("numpy")
    from duckmind.simweb import jpeg

    y, x = np.mgrid[0:101, 0:67]
    rgb = np.dstack([x * 3, y * 2, np.full_like(x, 128)]).astype(np.uint8)
    data = jpeg.encode_numpy(rgb, 85)
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    sof = data.index(b"\xff\xc0")
    assert data[sof + 5 : sof + 9] == bytes([0, 101, 0, 67])  # height, width


def test_numpy_jpeg_decodes_like_pillow_would_encode():
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image", reason="decoding the check needs Pillow")
    from duckmind.simweb import jpeg

    y, x = np.mgrid[0:101, 0:67]
    rgb = np.dstack([x * 3, y * 2, np.full_like(x, 128)]).astype(np.uint8)
    decoded = np.asarray(image.open(io.BytesIO(jpeg.encode_numpy(rgb, 85))).convert("RGB"))
    mse = ((decoded.astype(float) - rgb.astype(float)) ** 2).mean()
    assert 10 * np.log10(255**2 / mse) > 35


def _decode_baseline(data: bytes):
    """A minimal baseline JPEG decoder (8-bit Huffman, one scan), written apart from the
    encoder: every table is read from the stream. Returns the RGB image and how many bits of
    the entropy-coded scan were left over (only the 1-bit padding may be)."""
    import numpy as np

    zigzag = [
        r * 8 + (s - r)
        for s in range(15)
        for r in (
            range(max(0, s - 7), min(s, 7) + 1)
            if s % 2
            else reversed(range(max(0, s - 7), min(s, 7) + 1))
        )
    ]
    assert data[:2] == b"\xff\xd8"
    pos, quant, huffman = 2, {}, {}
    frame: tuple = ()
    select: dict = {}
    while True:
        assert data[pos] == 0xFF, pos
        marker, length = data[pos + 1], int.from_bytes(data[pos + 2 : pos + 4], "big")
        segment, pos = data[pos + 4 : pos + 2 + length], pos + 2 + length
        if marker == 0xDB:
            for i in range(0, len(segment), 65):
                quant[segment[i] & 15] = list(segment[i + 1 : i + 65])
        elif marker == 0xC4:
            i = 0
            while i < len(segment):
                counts = segment[i + 1 : i + 17]
                values = segment[i + 17 : i + 17 + sum(counts)]
                table, code, k = {}, 0, 0
                for size in range(1, 17):
                    for _ in range(counts[size - 1]):
                        table[(size, code)] = values[k]
                        code, k = code + 1, k + 1
                    code <<= 1
                huffman[(segment[i] >> 4, segment[i] & 15)] = table
                i += 17 + sum(counts)
        elif marker == 0xC0:
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            parts = [segment[6 + 3 * c : 9 + 3 * c] for c in range(segment[5])]
            frame = (height, width, [(p[0], p[1] >> 4, p[1] & 15, p[2]) for p in parts])
        elif marker == 0xDA:
            select = {segment[1 + 2 * c]: segment[2 + 2 * c] for c in range(segment[0])}
            break
    stuffed = data[pos : data.rindex(b"\xff\xd9")]
    # Inside a scan every 0xFF byte of data is stuffed with a 0x00 (anything else is a marker).
    assert all(stuffed[i + 1] == 0 for i in range(len(stuffed)) if stuffed[i] == 0xFF)
    scan = stuffed.replace(b"\xff\x00", b"\xff")
    bits = np.unpackbits(np.frombuffer(scan, dtype=np.uint8)).tolist()
    at = 0

    def receive(count: int) -> int:
        nonlocal at
        value = 0
        for _ in range(count):
            value, at = (value << 1) | bits[at], at + 1
        return value

    def signed(value: int, count: int) -> int:
        return value - (1 << count) + 1 if count and value < (1 << (count - 1)) else value

    def symbol(table: dict) -> int:
        code = 0
        for size in range(1, 17):
            code = (code << 1) | receive(1)
            if (size, code) in table:
                return table[(size, code)]
        raise AssertionError("not a code of this Huffman table")

    basis = np.array(
        [
            [(np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8)) * np.cos((2 * x + 1) * u * np.pi / 16)
             for x in range(8)]
            for u in range(8)
        ]
    )  # fmt: skip
    height, width, comps = frame
    hmax, vmax = max(c[1] for c in comps), max(c[2] for c in comps)
    across, down = -(-width // (8 * hmax)), -(-height // (8 * vmax))
    planes = {c[0]: np.zeros((down * c[2] * 8, across * c[1] * 8)) for c in comps}
    previous = {c[0]: 0 for c in comps}
    for my in range(down):
        for mx in range(across):
            for cid, h, v, table in comps:
                dc, ac = huffman[(0, select[cid] >> 4)], huffman[(1, select[cid] & 15)]
                for by in range(v):
                    for bx in range(h):
                        zz = [0] * 64
                        size = symbol(dc)
                        previous[cid] += signed(receive(size), size)
                        zz[0], k = previous[cid], 1
                        while k < 64:
                            rs = symbol(ac)
                            if rs & 15 == 0:
                                if rs >> 4 == 15:
                                    k += 16
                                    continue
                                break
                            k += rs >> 4
                            zz[k] = signed(receive(rs & 15), rs & 15)
                            k += 1
                        coef = np.zeros(64)
                        for i in range(64):
                            coef[zigzag[i]] = zz[i] * quant[table][i]
                        block = basis.T @ coef.reshape(8, 8) @ basis + 128
                        row, col = (my * v + by) * 8, (mx * h + bx) * 8
                        planes[cid][row : row + 8, col : col + 8] = block
    full = []
    for cid, h, v, _ in comps:
        plane = planes[cid].repeat(vmax // v, axis=0).repeat(hmax // h, axis=1)
        full.append(plane[:height, :width])
    y, cb, cr = (full[0], full[1] - 128, full[2] - 128)
    rgb = np.dstack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb])
    left = bits[at:]
    return np.clip(np.round(rgb), 0, 255), left


def test_the_numpy_jpeg_decodes_without_pillow():
    """The encoder the owner's live simulator uses (no Pillow there), checked by a decoder of
    its own: the whole scan decodes, only the 1-bit padding is left, and the picture is back."""
    np = pytest.importorskip("numpy")
    from duckmind.simweb import jpeg

    y, x = np.mgrid[0:101, 0:67]
    gradient = np.dstack([x * 3, y * 2, np.full_like(x, 128)]).astype(np.uint8)
    flat = np.zeros((40, 56, 3), dtype=np.uint8) + np.array([90, 120, 200], dtype=np.uint8)
    # Only the highest frequency in each block: long runs of zeros (the encoder's ZRL codes).
    wave = np.cos((2 * np.arange(48) + 1) * 7 * np.pi / 16)
    grey = np.round(128 + 60 * np.outer(wave, wave[:40])).astype(np.uint8)
    fine = np.dstack([grey, grey, grey])
    # Noise: long codes, so the scan holds 0xFF bytes that must be stuffed.
    noise = np.random.default_rng(3).integers(0, 256, (48, 64, 3)).astype(np.uint8)
    for image, least_db in ((gradient, 40.0), (flat, 40.0), (fine, 40.0), (noise, 10.0)):
        data = jpeg.encode_numpy(image, 85)
        decoded, left = _decode_baseline(data)
        assert decoded.shape == image.shape
        assert len(left) < 8 and all(bit == 1 for bit in left)  # nothing but the padding
        mse = ((decoded - image.astype(float)) ** 2).mean()
        assert 10 * np.log10(255**2 / max(mse, 1e-9)) > least_db
    assert b"\xff\x00" in jpeg.encode_numpy(noise, 85)  # (the stuffing was exercised)


def test_a_stopped_world_is_not_painted_again():
    """The body frame's physics time stops (robotd or the body server stopped): the camera
    stops making pictures of it — the last one keeps its age — and starts again with the
    world."""
    from duckmind.simweb.eyecam import FROZEN_NOTE, FROZEN_S, EyeCam
    from duckmind.world import camera_model as cm

    scene = _needs_render()
    probe = _renderer(scene)
    pose = dict(cm.keyframe_pose(probe.eye, "STAND"), t=123.45)
    probe.close()
    now = {"t": 123.45}
    cam = EyeCam(
        lambda: {**pose, **now}, scene=scene, clock=lambda: NOON, period_s=0.2, meter_s=0.1
    )
    cam.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not cam.status().get("frozen"):
            time.sleep(0.05)
        stopped = cam.status()
        time.sleep(0.8)
        still = cam.status()
        now["t"] = 124.0  # the world moves again
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and cam.status().get("frames", 0) <= still["frames"]:
            time.sleep(0.05)
        moving = cam.status()
    finally:
        cam.stop()
    if stopped.get("error"):
        pytest.skip(f"no offscreen GL context here: {stopped['error']}")
    assert stopped["frozen"] and stopped["note"] == FROZEN_NOTE
    assert still["frames"] == stopped["frames"] and still["wall"] == stopped["wall"]
    assert time.time() - still["wall"] >= FROZEN_S  # its age grows: no fresh stamp
    assert moving["frames"] > still["frames"] and not moving["frozen"] and moving["t"] == 124.0
