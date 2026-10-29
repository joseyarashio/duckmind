"""DS-v1 (duck-body ``depth.py``, spec §5.1): the ToF summary on the Reprojector port.

Synthetic frames are ray-cast here against a flat floor, a wall and a step (pure Python, no
simulator); the replay case reads phase 1's live ``wall_walk.jsonl`` when it is on this machine
and checks DS-v1 against ``tof_replay.txt``'s expected moments.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from duckmind import paths, percept
from duckmind.duckbody import depth as D
from duckmind.duckbody import tof_reproject as tr

TRUNK_H = 0.125  # the standing trunk height the Reprojector uses when the tap gives none
SENSOR_POS = (0.075, 0.0, 0.110)  # about where the head puts the ToF, trunk frame (m)


def sensor(pitch_deg: float = -10.0, yaw_deg: float = 0.0) -> tr.Pose:
    """The ToF looking ``pitch_deg`` above the horizon (negative: down)."""
    q = tr.Quat.from_axis_angle((0.0, 0.0, 1.0), math.radians(yaw_deg)) * tr.Quat.from_axis_angle(
        (0.0, 1.0, 0.0), math.radians(-pitch_deg)
    )
    return tr.Pose(SENSOR_POS, q)


def cast(
    pose: tr.Pose,
    *,
    wall_x: float | None = None,
    step_x: float | None = None,
    step_drop: float = 0.10,
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> tuple[list[int], list[int]]:
    """64 zones: slant range (mm) to the first of a floor (dropping ``step_drop`` beyond
    ``step_x``) and a wall at ``wall_x``, in the levelled frame; status 255 beyond 4 m."""
    level = tr.level_from_gravity(gravity)
    s = level.rotate(pose.pos)
    above = s[2] + TRUNK_H
    mm, st = [], []
    for beam in tr.BEAMS:
        d = level.rotate(pose.quat.rotate(beam))
        best = math.inf
        if d[2] < 0:
            r = above / -d[2]
            if step_x is not None and s[0] + r * d[0] > step_x:
                r = (above + step_drop) / -d[2]
            best = min(best, r)
        if wall_x is not None and d[0] > 0:
            best = min(best, (wall_x - s[0]) / d[0])
        if best > 4.0:
            mm.append(0)
            st.append(255)
        else:
            mm.append(int(round(best * 1000)))
            st.append(5)
    return mm, st


def frame(t_ms: int, mm: list[int], st: list[int]) -> D.ToFFrame:
    return D.ToFFrame.of(t_ms * 1_000_000, mm, st)


def tracker() -> D.DepthTracker:
    return D.DepthTracker(standing_height_m=TRUNK_H)


def test_flat_floor_reads_clear_and_integer() -> None:
    pose = sensor(-10)
    reading, events = tracker().step(frame(0, *cast(pose)), D.SensorPose(pose), walking=True)
    assert reading.available and not reading.drop and not reading.touch
    assert reading.ahead_mm == D.FAR_MM and reading.left_mm == D.FAR_MM
    assert reading.floor > 0 and reading.hits == 0
    assert events == []
    assert reading.head_pitch_mrad == pytest.approx(-175, abs=2)
    fields = reading.sense_depth()
    assert fields["ahead_mm"] is None and fields["touch"] is False  # nothing seen, no touch
    event = percept.make("sense.depth", src="live", t_ms=0, pv="p:0123456789abcdef", **fields)
    assert percept.validate(event) == []
    view = reading.view()
    assert view is not None and view.ahead_mm is None and view.drop_zones == 0


def test_depth_view_for_the_reflexes() -> None:
    pose = sensor(-10, yaw_deg=5)
    reading, _ = tracker().step(frame(0, *cast(pose, wall_x=0.40)), D.SensorPose(pose))
    view = reading.view()
    assert view is not None and view.ahead_mm == reading.ahead_mm
    assert abs(view.ahead_b_mrad) < 260  # the nearest return is in the body corridor
    assert D.DepthTracker(standing_height_m=TRUNK_H).unavailable(0).view() is None


def test_wall_ahead_distance_from_the_trunk_origin() -> None:
    pose = sensor(-10)
    reading, _ = tracker().step(frame(0, *cast(pose, wall_x=0.40)), D.SensorPose(pose))
    assert reading.ahead_mm == pytest.approx(400, abs=3)
    assert not reading.drop


def test_side_hits_go_to_left_and_right() -> None:
    pose = sensor(-5, yaw_deg=30)  # looking 30° to the left at a wall 0.35 m ahead
    reading, _ = tracker().step(frame(0, *cast(pose, wall_x=0.35)), D.SensorPose(pose))
    assert reading.left_mm < D.FAR_MM
    assert reading.left_mm >= 350


def test_step_down_ahead_is_a_drop_after_two_frames() -> None:
    pose = sensor(-15)
    tr_ = tracker()
    mm, st = cast(pose, step_x=0.45)
    first, ev1 = tr_.step(frame(0, mm, st), D.SensorPose(pose), walking=True)
    assert first.drop_zones >= 4 and first.drop_checked
    assert not first.drop  # debounce: one frame is not enough
    second, ev2 = tr_.step(frame(70, mm, st), D.SensorPose(pose), walking=True)
    assert second.drop
    assert ev1 == [] and len(ev2) == 1 and "drop" in str(ev2[0]["why"])


def test_a_step_three_metres_away_is_not_a_drop() -> None:
    # phase 1's drop test counted every downward zone and fired ~3 m before a step
    pose = sensor(-10)
    tr_ = tracker()
    for k in range(3):
        reading, _ = tr_.step(frame(70 * k, *cast(pose, step_x=3.0)), D.SensorPose(pose))
    assert reading.drop_zones == 0 and not reading.drop


def test_no_return_where_the_floor_should_be_counts_as_drop() -> None:
    pose = sensor(-15)
    mm, st = cast(pose)
    for i in range(tr.ZONES):
        if mm[i] and mm[i] < 1100:
            mm[i], st[i] = 0, 255
    reading, _ = tracker().step(frame(0, mm, st), D.SensorPose(pose))
    assert reading.drop_zones >= 4


def test_no_drop_verdict_while_tilted() -> None:
    pose = sensor(-15)
    g = (math.sin(math.radians(25)), 0.0, -math.cos(math.radians(25)))
    mm, st = cast(pose, step_x=0.45, gravity=g)
    tr_ = tracker()
    for k in range(3):
        reading, _ = tr_.step(frame(70 * k, mm, st), D.SensorPose(pose, gravity=g))
    assert not reading.drop_checked and not reading.drop and reading.drop_zones == 0
    assert reading.tilt_mrad == pytest.approx(436, abs=2)


def test_step_off_to_the_side_is_not_in_the_direction_of_travel() -> None:
    pose = sensor(-15, yaw_deg=40)  # the head turned 40° left; the floor drops only there
    level_step = cast(pose, step_x=0.30)
    reading, _ = tracker().step(frame(0, *level_step), D.SensorPose(pose))
    straight = sensor(-15)
    ahead, _ = tracker().step(frame(0, *cast(straight, step_x=0.30)), D.SensorPose(straight))
    assert reading.drop_zones < ahead.drop_zones


def test_touch_when_the_face_is_on_a_wall() -> None:
    pose = sensor(0)
    reading, events = tracker().step(
        frame(0, [40] * 64, [5] * 64), D.SensorPose(pose), walking=True
    )
    assert reading.touch and reading.too_close == 64
    assert events and "touch" in str(events[0]["why"])


def test_weak_status_needs_the_previous_frame_within_3_cm() -> None:
    pose = sensor(-10)
    mm, st = cast(pose, wall_x=0.45)
    centre = 3 * 8 + 3
    tr_ = tracker()
    weak_st = list(st)
    weak_st[centre] = 4
    first, _ = tr_.step(frame(0, mm, weak_st), D.SensorPose(pose))
    assert first.weak_used == 0  # nothing to compare with yet
    near = list(mm)
    near[centre] += 20
    second, _ = tr_.step(frame(70, near, weak_st), D.SensorPose(pose))
    assert second.weak_used == 1
    far = list(near)
    far[centre] += 80
    third, _ = tr_.step(frame(140, far, weak_st), D.SensorPose(pose))
    assert third.weak_used == 0


def test_crossings_fire_once_each_while_walking_and_rearm() -> None:
    pose = sensor(-10)
    tr_ = tracker()
    fired: list[str] = []
    for k, wall in enumerate([1.2, 0.9, 0.79, 0.7, 0.49, 0.45, 0.29, 0.25, 0.149, 0.2]):
        _, events = tr_.step(
            frame(70 * k, *cast(pose, wall_x=wall)), D.SensorPose(pose), walking=True
        )
        fired += [str(e["why"]) for e in events]
    assert fired == ["ahead_800", "ahead_500", "ahead_300", "ahead_150,touch"]
    # back out past the hysteresis, then in again: the 150 line fires again
    for k, wall in enumerate([0.25, 0.14]):
        _, events = tr_.step(
            frame(1000 + 70 * k, *cast(pose, wall_x=wall)), D.SensorPose(pose), walking=True
        )
        fired += [str(e["why"]) for e in events]
    # (inside ~0.175 m every zone is in the Reprojector's too-close band: 150 comes with touch)
    assert fired[-1] == "ahead_150,touch"


def test_standing_still_sends_nothing() -> None:
    pose = sensor(-10)
    _, events = tracker().step(frame(0, *cast(pose, wall_x=0.2)), D.SensorPose(pose), walking=False)
    assert events == []


def test_unavailable_tof() -> None:
    tr_ = tracker()
    reading = tr_.unavailable(5)
    assert not reading.available and reading.t_ns == 5


def test_pose_from_frames_matches_a_pose() -> None:
    pose = sensor(-12)
    built = D.pose_from_frames(pose.pos, pose.quat.wxyz(), (0, 0, -1), 120)
    a = D.summarise(frame(0, *cast(pose, wall_x=0.5)), [None] * 64, D.SensorPose(pose), TRUNK_H)
    b = D.summarise(frame(0, *cast(pose, wall_x=0.5)), [None] * 64, built, TRUNK_H)
    assert a.head_pitch_mrad == b.head_pitch_mrad


# ── replay: phase 1's live walks (official robotd --sim, the official single-ray ToF) ──────

WALL_WALK = paths.data_dir() / "physical-design/wall_walk.jsonl"
MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _gravity_in_trunk(q: list[float]) -> tuple[float, float, float]:
    w, x, y, z = q
    # R^T (0, 0, -1): the third row of R, negated
    return (-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y)))


@pytest.mark.skipif(
    not WALL_WALK.is_file() or not MJCF_DIR, reason="needs phase 1's wall_walk.jsonl and the MJCF"
)
def test_replay_wall_walk_against_tof_replay() -> None:
    rp = tr.Reprojector.from_mjcf(Path(str(MJCF_DIR)) / "robot_walk.xml")
    runs = {}
    for line in WALL_WALK.open():
        try:
            run = json.loads(line)
        except json.JSONDecodeError:
            continue
        runs[(run["scene"], run["rep"])] = run
    found: dict[tuple[str, int], dict[str, float | None]] = {}
    for key in (("wall", 0), ("wall", 1), ("free", 0), ("cliff", 0), ("apartment", 0)):
        run = runs.get(key)
        if run is None:
            continue
        tracker_ = D.DepthTracker(standing_height_m=rp.trunk_height_m)
        start = run["walk_sent"][0] - 0.3
        first300 = first_drop = None
        truth = run["truth"]
        for f in run["tof"]:
            if f["t"] < start:
                continue
            near = min(truth, key=lambda r: abs(r["t"] - f["t"]))
            if abs(near["t"] - f["t"]) > 0.1:
                continue
            pose = D.pose_from_head(rp, near["head"], _gravity_in_trunk(near["quat"]))
            reading, _ = tracker_.step(
                D.ToFFrame.of(int(f["t"] * 1e9), f["mm"], f["st"]), pose, walking=True
            )
            if first300 is None and reading.ahead_mm <= 300:
                first300 = f["t"]
            if first_drop is None and reading.drop:
                first_drop = f["t"]
        found[key] = {"ahead300": first300, "drop": first_drop}
    if ("wall", 0) in found:  # tof_replay: first < 0.30 m at 3.896 s
        assert found[("wall", 0)]["ahead300"] == pytest.approx(3.896, abs=0.35)
        assert found[("wall", 0)]["drop"] is None
    if ("free", 0) in found:
        assert found[("free", 0)] == {"ahead300": None, "drop": None}
    if ("cliff", 0) in found:  # the cliff edge: phase 1 saw ≥ 4 drop zones at 1.27 s
        assert found[("cliff", 0)]["drop"] is not None
