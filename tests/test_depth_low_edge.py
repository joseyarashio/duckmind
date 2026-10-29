"""A low edge (a curb, a step: 45-130 mm) in DS-v1's depth summary (duckbody/depth.py) and the duck's memory of it.

The ToF sits ~0.24 m above the floor and looks down at most ~20 degrees, so inside ~0.4 m every beam passes over a 60 mm
curb: the nearest return jumps from the curb's face to whatever lies beyond (measured 2026-10-10 headless: 450 mm, then
1177 mm, and the duck walked into the curb). Synthetic frames are ray-cast here against a floor and a box (pure Python, no
simulator), world frame, the duck at a pose in its own odometry.
"""

from __future__ import annotations

import math

from duckmind.duckbody import depth as D
from duckmind.duckbody import tof_reproject as tr

TRUNK_H = 0.12
SENSOR_POS = (0.075, 0.0, 0.11)  # trunk frame (m)
PITCH_DEG = -10.0


def sensor() -> tr.Pose:
    q = tr.Quat.from_axis_angle((0.0, 1.0, 0.0), math.radians(-PITCH_DEG))
    return tr.Pose(SENSOR_POS, q)


def _box_hit(
    o: tuple[float, float, float],
    d: tuple[float, float, float],
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> float | None:
    t0, t1 = 0.0, math.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if not lo[i] <= o[i] <= hi[i]:
                return None
            continue
        a, b = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return t0


def frame(
    pose_xy: tuple[float, float],
    yaw: float,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    t_ns: int = 0,
) -> D.ToFFrame:
    """The 64 zones the duck at ``pose_xy`` / ``yaw`` (world, m) would read: the floor at z = -TRUNK_H (trunk origin at
    z = 0) and axis-aligned boxes ``(lo, hi)`` in the same frame; 4 m range."""
    c, s = math.cos(yaw), math.sin(yaw)
    sp = SENSOR_POS
    origin = (pose_xy[0] + sp[0] * c - sp[1] * s, pose_xy[1] + sp[0] * s + sp[1] * c, sp[2])
    pose = sensor()
    mm, st = [], []
    for beam in tr.BEAMS:
        b = pose.quat.rotate(beam)
        d = (b[0] * c - b[1] * s, b[0] * s + b[1] * c, b[2])
        best = math.inf
        if d[2] < 0:
            best = (-TRUNK_H - origin[2]) / d[2]
        for lo, hi in boxes:
            t = _box_hit(origin, d, lo, hi)
            if t is not None:
                best = min(best, t)
        if best > 4.0:
            mm.append(0)
            st.append(255)
        else:
            mm.append(int(round(best * 1000)))
            st.append(5)
    return D.ToFFrame.of(t_ns, mm, st)


def curb(h_mm: float, x: float = 1.0, depth: float = 0.08, half: float = 0.45):
    """A step across the way: its face at world x, ``depth`` deep, ``half`` either side of y = 0, ``h_mm`` tall."""
    return ((x, -half, -TRUNK_H), (x + depth, half, -TRUNK_H + h_mm / 1000.0))


def read(
    x: float, y: float, yaw: float, boxes, tracker=None, t_ns: int = 0, with_odom: bool = True
):
    tracker = tracker or D.DepthTracker(standing_height_m=TRUNK_H)
    fr = frame((x, y), yaw, boxes, t_ns)
    reading, _ = tracker.step(
        fr,
        D.SensorPose(sensor(), (0.0, 0.0, -1.0), None),
        walking=True,
        odom=(x, y, yaw) if with_odom else None,
    )
    return reading


def test_a_60_mm_curb_seen_from_afar_is_a_wide_low_edge():
    r = read(0.2, 0.0, 0.0, [curb(60)])  # the face 0.8 m ahead
    assert 400 < r.ahead_mm < 1000
    assert 45 <= r.ahead_top_mm <= 100
    assert r.low_edge_mm >= 45
    assert r.low_face is not None
    x1, y1, x2, y2 = r.low_face
    assert abs(y2 - y1) >= 300  # the edge reaches across the view
    assert not r.virtual


def test_a_wall_is_not_a_low_edge():
    r = read(0.2, 0.0, 0.0, [curb(1000)])
    assert r.low_edge_mm == 0
    assert r.ahead_top_mm > 130


def test_a_thin_sill_is_floor_to_the_tof():
    for h in (5, 10, 20):
        r = read(0.2, 0.0, 0.0, [curb(h, depth=0.05)])
        assert r.low_edge_mm == 0
        assert r.ahead_mm >= 3000  # nothing in the way as far as the ToF can tell (floor returns)


def test_a_narrow_low_thing_is_not_an_edge():
    bowl = ((1.0, -0.075, -TRUNK_H), (1.15, 0.075, -TRUNK_H + 0.06))
    r = read(0.3, 0.0, 0.0, [bowl])
    assert r.low_edge_mm == 0


def test_the_beams_pass_over_a_curb_close_up_and_the_memory_keeps_it():
    boxes = [curb(60)]
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    plain = D.DepthTracker(standing_height_m=TRUNK_H)
    t = 0
    faces = []
    x = 0.1
    while x < 0.9:  # the duck walks up to the face (x = 1.0), a frame every 3 cm
        t += 70_000_000
        with_memory = read(x, 0.0, 0.0, boxes, tracker, t)
        without = read(x, 0.0, 0.0, boxes, plain, t, with_odom=False)
        faces.append((x, with_memory, without))
        x += 0.03
    lost = [(x, a, b) for x, a, b in faces if b.ahead_mm > 1000 and 1.0 - x < 0.5]
    assert lost, "the geometry of this test must make the face disappear under the beams"
    for x, a, b in lost:
        true_mm = (1.0 - x) * 1000.0
        if true_mm < 120:
            continue
        assert a.virtual, f"x={x}: the face is no longer seen but must be remembered"
        assert abs(a.ahead_mm - true_mm) < 60, (x, a.ahead_mm, true_mm)
        assert a.low_edge_mm >= 45
    # … and without the odometry the old behaviour is unchanged
    assert all(not b.virtual for _, _, b in faces)


def _seen_edge(tracker, t=0):
    """The duck walks up to 0.74 m of the curb face (x = 1.0), looking at it for three frames."""
    for x in (0.2, 0.23, 0.26):
        t += 70_000_000
        r = read(x, 0.0, 0.0, [curb(60)], tracker, t)
    assert r.low_edge_mm >= 45
    return t


def test_turning_away_moves_the_remembered_edge_to_the_side():
    boxes = [curb(60)]
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    t = _seen_edge(tracker)
    t += 70_000_000
    # now 0.35 m from the face (every beam over it) and facing left: the edge is on its right
    r = read(0.65, 0.0, math.pi / 2, boxes, tracker, t)
    assert r.virtual and r.right_mm < 700
    assert r.ahead_mm >= 3000 or not r.virtual
    # facing the way it came: it is behind, and forgotten
    t += 70_000_000
    r = read(0.65, 0.0, math.pi, boxes, tracker, t)
    assert not r.virtual


def test_the_memory_is_kept_while_the_duck_is_too_close_to_look_again_and_fades_when_it_is_not():
    boxes = [curb(60)]
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    t = _seen_edge(tracker)
    t += int(
        20e9
    )  # 20 s later, 0.28 m from the face: the beams pass over it, nothing can have changed its mind
    r = read(0.72, 0.0, 0.0, boxes, tracker, t)
    assert read(0.72, 0.0, 0.0, boxes, None, t, with_odom=False).ahead_mm > 1000
    assert r.virtual and abs(r.ahead_mm - 280) < 60
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    t = _seen_edge(tracker)
    t += int(20e9)  # … but 0.9 m from it, in view of where it was, it is not remembered
    r = read(0.1, 0.0, 0.0, [], tracker, t)  # (nothing there any more)
    assert not r.virtual and r.low_edge_mm == 0
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    t = _seen_edge(tracker)
    t += int(130e9)  # and never for ever
    r = read(0.72, 0.0, 0.0, boxes, tracker, t)
    assert not r.virtual


def test_a_lowered_trunk_lets_no_new_edge_in_and_the_helper_says_when_it_is_lowered():
    # seated or crouched the standing-height reprojection reads the floor too high: no edge is remembered from such frames
    tracker = D.DepthTracker(standing_height_m=TRUNK_H)
    t = 0
    for x in (0.2, 0.23, 0.26, 0.29):
        t += 70_000_000
        fr = frame((x, 0.0), 0.0, [curb(60)], t)
        tracker.step(
            fr,
            D.SensorPose(sensor(), (0.0, 0.0, -1.0), None),
            walking=True,
            odom=(x, 0, 0),
            upright=False,
        )
    r = read(0.72, 0.0, 0.0, [curb(60)], tracker, t + 70_000_000)
    assert not r.virtual
    assert D.upright((0.0, 0.0, 0.117)) and D.upright((0.0, 0.0, 0.105))
    assert not D.upright((0.0, 0.0, 0.06)) and not D.upright((0.0, 0.0, 0.17))
    assert D.upright((0.0, 0.0, 0.0)) and D.upright(None) and D.upright((0.0, 0.0))
