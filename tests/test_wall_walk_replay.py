"""Replay of phase 1's 18 live walks (``wall_walk.jsonl``) through duck-body's senses, with
``tof_replay.txt`` as the expected answers (spec §17 期3 "wall_walk.jsonl 重播").

The recording (official ``robotd --sim``, the official single-ray sim ToF at ~6.4 frames/s,
free / wall / 30° wall / pole / low box / threshold / cliff / apartment, head level and head down)
is committed trimmed: ``tests/data/wall_walk_trim.jsonl.gz`` (no camera PNGs, floats rounded;
its header carries the source's sha256 and its place in the measurement records, which are not in
the repo) with the expected answers in
``tests/data/tof_replay_expected.json``. The truth's trunk quaternion and head joints stand in for
the IMU and the measured joints (the recording has no ``frames``).

DS-v1 (:mod:`duckmind.duckbody.depth`) against the expected answers:

* **stop line** — where ``tof_replay`` saw the first return ≤ 0.30 m ahead, DS-v1's ``ahead_mm``
  crosses 300 mm no later than one frame after it and at most three frames before (DS-v1 measures
  forward of the trunk origin inside the body corridor, ``tof_replay`` horizontally in columns
  2–5: a little earlier is the conservative side); where it saw none, DS-v1 sees none — except
  the pole, which ``tof_replay``'s narrower columns never counted: there DS-v1 may stop only when
  the pole really is within 0.40 m inside the corridor (truth);
* **drop** — at the cliff DS-v1's drop fires on the same frame ``tof_replay`` first counted ≥ 4
  drop zones; in every other run no drop fires before the duck has been at the obstacle (its
  trunk within 0.20 m of the near face at some point: phase 1's after-contact false drops — the
  head-down threshold walk and a box that threw the duck back still have one — are not a step
  the duck would walk off);
* **touch** — fires in a run iff ``tof_replay`` counted ≥ 32 too-close zones in it.

proprio (:mod:`duckmind.duckbody.proprio`), when present: none of these walks is a pickup, so no
``lifted`` edge may fire (the pickup false-alarm check on real robotd frames), and a ``fell``
edge fires in a run iff robotd's own fallen flag rose after the walk started.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from duckmind import paths
from duckmind.duckbody import tof_reproject as tr

DATA = Path(__file__).with_name("data")
FIXTURE = DATA / "wall_walk_trim.jsonl.gz"
EXPECTED = DATA / "tof_replay_expected.json"
MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
FRAME_S = 1.0 / 6.4  # the recording's ToF cadence
POLE = (0.50, -0.10, 0.015)  # phase-1 scenes/pole.xml: the pole's x, y and radius (m)
# phase-1 scenes/*.xml: the x of each obstacle's near face (the duck walks +x from x ≈ 0)
NEAR_FACE_X = {"wall": 0.55, "wall30": 0.55, "pole": 0.485, "lowbox": 0.50, "threshold": 0.475}

needs_mjcf = pytest.mark.skipif(
    not MJCF_DIR or not (Path(str(MJCF_DIR)) / "robot_walk.xml").is_file(),
    reason="needs DUCKMIND_MICRODUCK_MJCF (the official robot_walk.xml)",
)


@cache
def fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


@cache
def expected() -> dict[tuple[str, int], dict[str, Any]]:
    data = json.loads(EXPECTED.read_text(encoding="utf-8"))
    return {(e["scene"], e["rep"]): e for e in data["runs"]}


def gravity_in_trunk(q: list[float]) -> tuple[float, float, float]:
    w, x, y, z = q
    return (-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y)))


def _nearest(samples: list[dict[str, Any]], t: float, start: int) -> int:
    j = start
    while j + 1 < len(samples) and abs(samples[j + 1]["t"] - t) <= abs(samples[j]["t"] - t):
        j += 1
    return j


@cache
def ds_v1(scene: str, rep: int) -> dict[str, Any]:
    """Run one recorded walk through DS-v1: the first crossing / drop / touch times, and every
    frame that read ≤ 300 mm with the trunk's truth pose at that frame."""
    depth = pytest.importorskip("duckmind.duckbody.depth")
    rp = tr.Reprojector.from_mjcf(Path(str(MJCF_DIR)) / "robot_walk.xml")
    run = next(r for r in fixture()[1] if (r["scene"], r["rep"]) == (scene, rep))
    tracker = depth.DepthTracker(standing_height_m=rp.trunk_height_m)
    start = run["walk_sent"][0]
    truth = run["truth"]
    out: dict[str, Any] = {"a300": None, "drop": None, "reach_x": None, "touch": None, "near": []}
    reach = -math.inf
    j = 0
    for f in run["tof"]:
        j = _nearest(truth, f["t"], j)
        sample = truth[j]
        if abs(sample["t"] - f["t"]) > 0.1:
            continue
        pose = depth.pose_from_head(rp, sample["head"], gravity_in_trunk(sample["quat"]))
        reading, _ = tracker.step(
            depth.ToFFrame.of(int(round(f["t"] * 1e9)), f["mm"], f["st"]), pose, walking=True
        )
        if f["t"] < start - 0.3:
            continue
        reach = max(reach, sample["xyz"][0])
        if reading.ahead_mm <= 300:
            out["near"].append((f["t"], sample["xyz"], sample["quat"]))
            if out["a300"] is None:
                out["a300"] = f["t"]
        if out["drop"] is None and reading.drop:
            out["drop"] = f["t"]
            out["reach_x"] = reach  # how far the trunk had come by then
        if out["touch"] is None and reading.touch:
            out["touch"] = f["t"]
    return out


def test_the_fixture_is_the_recording_it_says() -> None:
    header, runs = fixture()
    assert header["schema"] == "wall-walk-trim-v1"
    assert len(runs) == 18 and len(expected()) == 18
    assert {(r["scene"], r["rep"]) for r in runs} == set(expected())
    source = paths.data_dir() / header["source"]
    if source.is_file():  # on the machine that holds the raw recording, pin it
        assert hashlib.sha256(source.read_bytes()).hexdigest() == header["source_sha256"]
    answers = paths.data_dir() / header["expected"]
    if answers.is_file():
        assert hashlib.sha256(answers.read_bytes()).hexdigest() == header["expected_sha256"]
        assert json.loads(EXPECTED.read_text())["source_sha256"] == header["expected_sha256"]
    for run in runs:
        assert run["tof"] and run["truth"] and run["robot"]
        assert all(len(f["mm"]) == 64 and len(f["st"]) == 64 for f in run["tof"])


RUNS = sorted(expected())


@needs_mjcf
@pytest.mark.parametrize("scene,rep", RUNS)
def test_stop_line_against_tof_replay(scene: str, rep: int) -> None:
    got = ds_v1(scene, rep)
    want = (expected()[(scene, rep)].get("first_ahead_lt_0.3") or {}).get("t")
    if want is not None:
        assert got["a300"] is not None, f"{scene}/{rep}: tof_replay stopped at {want}, DS-v1 never"
        assert want - 3 * FRAME_S - 1e-6 <= got["a300"] <= want + FRAME_S + 1e-6, (got, want)
    elif scene == "pole":
        for _t, xyz, quat in got["near"]:  # only a true near pole may stop the duck
            yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1 - 2 * (quat[2] ** 2 + quat[3] ** 2))  # fmt: skip
            dx, dy = POLE[0] - xyz[0], POLE[1] - xyz[1]
            ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            assert 0.0 < ahead - POLE[2] <= 0.40 and abs(lateral) <= 0.10 + POLE[2], (
                f"pole/{rep}: DS-v1 stopped with the pole {ahead:.3f} m ahead, {lateral:.3f} aside"
            )
    else:
        assert got["a300"] is None, f"{scene}/{rep}: a stop line where tof_replay saw nothing"


@needs_mjcf
@pytest.mark.parametrize("scene,rep", RUNS)
def test_drop_against_tof_replay(scene: str, rep: int) -> None:
    got = ds_v1(scene, rep)
    want = (expected()[(scene, rep)].get("first_cliff_ge4") or {}).get("t")
    if scene == "cliff":
        assert want is not None and got["drop"] is not None
        assert abs(got["drop"] - want) <= FRAME_S + 1e-6, (got["drop"], want)
    elif got["drop"] is not None:
        face = NEAR_FACE_X.get(scene)
        assert face is not None, f"{scene}/{rep}: a drop at {got['drop']} with nothing to fall off"
        assert got["reach_x"] >= face - 0.20, (
            f"{scene}/{rep}: a drop at {got['drop']} before the duck reached the obstacle "
            f"(trunk got to x {got['reach_x']:.3f}, near face x {face})"
        )


@needs_mjcf
@pytest.mark.parametrize("scene,rep", RUNS)
def test_touch_against_tof_replay(scene: str, rep: int) -> None:
    got = ds_v1(scene, rep)
    want = expected()[(scene, rep)]["max_too_close_zones"] >= 32
    assert (got["touch"] is not None) is want, (scene, rep, got["touch"])


# ── proprio on the same frames (when duck-body's proprio exists) ─────────────────


def _robot_states(run: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """robot.state-shaped frames from the recording (joints: the head from truth, legs 0)."""
    truth = run["truth"]
    out = []
    j = 0
    for r in run["robot"]:
        j = _nearest(truth, r["t"], j)
        s = truth[j]
        head = s["head"]
        joints = [0.0] * 5 + list(head) + [0.0] + [0.0] * 5
        applied = r["move"]
        state = {
            "t": r["t"],
            "t_ns": int(round(r["t"] * 1e9)) + 1_000_000_000,
            "move": {"requested": list(applied), "applied": list(applied)},
            "head": [0.0, 0.0, 0.0, 0.0],
            "policy": s.get("policy") or "walk",
            "safety": {"fallen": bool(s["fallen"]), "limp": False,
                       "gravity": list(gravity_in_trunk(s["quat"]))},
            "loop": {"hz": 50.0, "missed": 0},
            "joints": joints,
            "currents_ma": [float(c) for c in r["cur"]],
            "odom": {"position": r["odom"][:3], "yaw": r["odom"][3]},
            "imu": {"gyro": r["gyro"], "quat": s["quat"]},
        }  # fmt: skip
        out.append((state, bool(s["fallen"])))
    return out


@needs_mjcf
@pytest.mark.parametrize("scene,rep", RUNS)
def test_proprio_no_pickup_and_falls_follow_robotd(scene: str, rep: int) -> None:
    proprio = pytest.importorskip("duckmind.duckbody.proprio")
    client = pytest.importorskip("duckmind.duckbody.robotd_client")
    run = next(r for r in fixture()[1] if (r["scene"], r["rep"]) == (scene, rep))
    start = run["walk_sent"][0]
    sense = proprio.Proprio()
    lifted, fell = [], []
    robotd_fell = False
    was_fallen = True  # the recording boots fallen (standing boot): only later rises count
    for state, fallen in _robot_states(run):
        out = sense.step(client.RobotState.from_wire(state))
        if state["t"] < start:
            was_fallen = fallen
            continue
        if fallen and not was_fallen:
            robotd_fell = True
        was_fallen = fallen
        if "lifted" in out.edges:
            lifted.append(state["t"])
        if "fell" in out.edges:
            fell.append(state["t"])
    assert not lifted, f"{scene}/{rep}: a pickup claimed at {lifted} on a floor walk"
    if robotd_fell:
        assert fell, f"{scene}/{rep}: robotd said fallen, proprio never did"
