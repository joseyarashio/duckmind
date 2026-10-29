"""P2 and the lockstep rig (spec §11.7 P2 "同一段畫面位元組、同一個模型檔 → 同樣的整數特徵 100%";
§17 期3).

* The rig (:mod:`duckmind.world.rig`) run twice with the same home, seed and script gives the same
  ``robot.state`` stream, the same ToF frames at tofd's cadence, the same frame bytes and the same
  intents, byte for byte (its digests).
* The same frame bytes give the same integer features from duck-eye's per-look code — the
  quarter-resolution sampler and the light meter here — twice in this process and once in a fresh
  subprocess (no hidden state, no thread-count or BLAS dependence). The gate script ``g_p2.py``
  runs the same check through the whole eye (every task) once its server-side step exists.
* The doors are the board's shapes: ``tof.frame`` has tofd's fields and 64 zones, a frame's header
  has mediad's six fields, and nothing of the world's truth is in either.

Needs numpy, MuJoCo 3.10, onnxruntime, the official MJCF and the shipped policies: a render-check
test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind.world import headless_duck as hd  # noqa: E402
from duckmind.world import rig as R  # noqa: E402

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
if not MJCF_DIR or not (Path(MJCF_DIR) / "robot_allcollisions.xml").is_file():
    pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)", allow_module_level=True)
if not (hd.policy_dir() / "velstand.onnx").is_file():
    pytest.skip(f"needs the shipped policies in {hd.policy_dir()}", allow_module_level=True)

QUALITY = "360p30"  # the smallest rung keeps this test quick; the gates use 720p30


def _run(light: str = "bright") -> tuple[R.Rig, list[R.Frame], list[dict]]:
    rig = R.Rig("default_home", light=light, seed=7, keyframe="SIT", x=1.2, y=0.3, yaw=0.0,
                quality=QUALITY)  # fmt: skip
    ok, _ = rig.duck.stand(timeout=12.0)
    assert ok
    brain = R.ScriptBrain(
        R.stream_moves(0.0, 2.0, 0.40, 0.18) + R.stream_moves(2.5, 3.5, 0.35, -0.5)
    )
    tofs: list[dict] = []
    frames: list[R.Frame] = []

    def until(rig_: R.Rig, doors: R.Doors, out: R.BrainOut) -> bool:
        if doors.tof is not None:
            tofs.append(doors.tof)
        if rig_.counts["ticks"] in (60, 180):
            frames.append(rig_.capture())
        return False

    rig.run(brain, 4.0, until=until)
    return rig, frames, tofs


@pytest.fixture(scope="module")
def two_runs() -> tuple[tuple[R.Rig, list[R.Frame], list[dict]], ...]:
    first, second = _run(), _run()
    yield first, second
    first[0].close()
    second[0].close()


def test_two_rigs_give_the_same_bytes(two_runs) -> None:
    (a, fa, _), (b, fb, _) = two_runs
    assert a.digests() == b.digests()
    assert [f.payload for f in fa] == [f.payload for f in fb]
    assert int(a.digests()["n_tof"]) > 0 and int(a.digests()["n_frames"]) == 2


def test_tof_frames_come_at_tofds_cadence_with_tofds_fields(two_runs) -> None:
    (rig, _, tofs), _ = two_runs
    assert set(tofs[0]) == {"seq", "at_us", "t_ns", "rows", "cols", "distance_mm", "status"}
    assert all(len(f["distance_mm"]) == 64 and len(f["status"]) == 64 for f in tofs)
    gaps = [(b["t_ns"] - a["t_ns"]) / 1e9 for a, b in zip(tofs, tofs[1:], strict=False)]
    mean = sum(gaps) / len(gaps)
    assert abs(mean - R.TOF_PERIOD_S) < 0.01, f"mean ToF period {mean:.4f} s"
    assert [f["seq"] for f in tofs] == list(range(tofs[0]["seq"], tofs[0]["seq"] + len(tofs)))


def test_a_frame_has_mediads_header_and_no_truth_in_it(two_runs) -> None:
    (_, frames, _), _ = two_runs
    header = frames[0].header
    assert set(header) == {"width", "height", "format", "bytes", "captured_at_unix_us", "rotate"}
    assert header["format"] == "UYVY" and header["rotate"] == 90
    assert header["bytes"] == header["width"] * header["height"] * 2 == len(frames[0].payload)
    assert header["captured_at_unix_us"] >= hd.WALL_EPOCH_US
    assert "lux" in frames[0].truth and "level" in frames[0].truth  # beside, not in, the frame


def _features(payload: bytes, width: int, height: int, rotate: int) -> dict:
    from duckmind.eye import frames, light

    planes = frames.sample_quarter(payload, width, height, rotate)
    reading = light.read_quarter(planes.y)
    out = {
        "y_sha": frames.frame_sha16(planes.y.tobytes()),
        "u_sha": frames.frame_sha16(planes.u.tobytes()),
        "v_sha": frames.frame_sha16(planes.v.tobytes()),
        "light": {k: v for k, v in sorted(vars(reading).items())},
        "level": light.level_q(reading),
    }
    json.dumps(out)  # integers and strings only
    return out


def test_same_bytes_same_integer_features_here_and_in_a_subprocess(two_runs, tmp_path) -> None:
    pytest.importorskip("duckmind.eye.light")
    (_, frames_a, _), _ = two_runs
    frame = frames_a[0]
    h = frame.header
    first = _features(frame.payload, h["width"], h["height"], h["rotate"])
    second = _features(bytes(frame.payload), h["width"], h["height"], h["rotate"])
    assert first == second
    blob = tmp_path / "frame.bin"
    blob.write_bytes(frame.payload)
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_rig_determinism import _features\n"
        f"data = open({str(blob)!r}, 'rb').read()\n"
        f"print(json.dumps(_features(data, {h['width']}, {h['height']}, {h['rotate']}), sort_keys=True))\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=120,
    )
    assert json.loads(child.stdout.strip().splitlines()[-1]) == json.loads(
        json.dumps(first, sort_keys=True)
    )


def test_truth_light_band_follows_owner_c3() -> None:
    assert [R.light_band(x) for x in (300.0, 30.0, 29.9, 2.5, 2.49, 0.5, 0.49, 0.0)] == [
        "bright", "bright", "dim", "dim", "dark", "dark", "black", "black"
    ]  # fmt: skip
    assert (
        R.truth_level(0.1, dock_light_in_view=True) == "dark"
    )  # C1: only the dock light: very dark
    assert R.truth_level(0.1, dock_light_in_view=False) == "black"
    assert R.truth_level(100.0, dock_light_in_view=True) == "bright"
