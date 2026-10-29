"""DS-v1: duck-body's ToF depth summary (physical-design spec §5.1, phase 3).

The official ``kinematics::tof::Reprojector`` (ported in :mod:`duckmind.duckbody.tof_reproject`)
says what each of the 64 zones hit — the floor, something too close, something there, or nothing.
DS-v1 adds what the upstream runtime does not have and duck-body needs:

* **weak returns** — statuses 4 and 13 are used only when the same zone read within
  :attr:`DepthConfig.weak_window_mm` (3 cm) in the previous frame (the real sensor flickers
  past 30 cm); 5 and 9 are valid; anything else, 255 included, is no return;
* **ahead / left / right** — in the gravity-levelled frame that turns with the trunk's heading:
  ``ahead_mm`` is how far forward of the trunk origin the nearest hit in the body-width corridor
  is (the Reprojector's own reference point: the beak is ~97 mm ahead of it, the chest ~38 mm);
  ``left_mm`` / ``right_mm`` the nearest hit beside the corridor (horizontal distance);
  :data:`FAR_MM` when nothing is seen;
* **drop** (the spec's drop-off test, with the bounds phase 1 found it needs — it fired ~3 m
  before a step when every downward zone counted, and after contact while the duck was tilted):
  a zone whose beam should meet a flat floor **within** :attr:`DepthConfig.drop_reach_mm`
  (1 m) ahead, **in the direction of travel** (inside :attr:`DepthConfig.drop_corridor_half_mm`),
  while the trunk tilts less than :attr:`DepthConfig.drop_tilt_max_mrad`, and that returns
  nothing or more than 1.15 × the expected range (+30 mm) is a drop zone; ≥ 4 drop zones in
  :attr:`DepthConfig.drop_frames` consecutive frames is a drop;
* **touch** — ≥ 32 zones inside the too-close band (a face pressed on a wall reads all 64);
  ``ahead_mm`` is then the sensor's own forward offset (fewer too-close zones are taken for the
  cover glass's crosstalk, as the Reprojector does, and read as nothing);
* **low edge** (obstacles, 2026-10-10) — a curb or a step of about 30 mm and more across the way is seen only from afar (its face is
  inside the beams for ~0.45 m; closer every beam passes over it). ``ahead_top_mm`` is the height of the face the nearest ahead
  return is on, ``low_edge_mm`` the height estimate of a wide low edge in view (``low_face`` is its segment) and, given the odometry,
  the tracker remembers the edge and keeps reporting it as ahead / left / right (``virtual``) while the duck is near it
  (:class:`DepthConfig`, ``low_block_mm``; tests/test_depth_low_edge.py);
* **crossings** — while walking, ``ahead_mm`` falling through 800 / 500 / 300 / 150 mm, a drop
  starting and a touch starting are the moments ``sense.depth`` is sent (spec §5.1 output).

A duck without a ToF gets :meth:`DepthTracker.unavailable`: ``available`` False, no event, and
every rule that reads depth must still hold without it (spec §5.1).

Pure and clock-injected: :class:`DepthTracker` is a step function over (frame, pose) with the
frame's own ``t_ns``; nothing here reads a clock, a socket or the simulator. The Reprojector
works in floats; everything leaving this module is an integer (mm, mrad, counts).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody import tof_reproject as tr
from duckmind.duckbody.protocol import DepthView

STATUS_WEAK = (4, 13)
FAR_MM = 4000  # nothing seen: the sensor's 4 m range
EVENT = "sense.depth"


@dataclass(frozen=True)
class DepthConfig:
    """DS-v1's thresholds (mm, mrad, zone counts). Board numbers come in phase 8."""

    corridor_half_mm: int = 100  # the body-width corridor: hips ±71 mm, feet ±63 mm, + margin
    side_reach_mm: int = 700  # a hit beside the corridor counts for left/right within this
    weak_window_mm: int = 30
    touch_zones: int = 32
    drop_ratio_milli: int = 1150  # 1.15 × the expected range…
    drop_slack_mm: int = 30  # …plus this (phase 1 / tof_replay used the same)
    drop_reach_mm: int = 1000  # only floor points expected within 1 m ahead (spec §5.1)
    drop_min_reach_mm: int = 120  # …and beyond the duck's own toes
    drop_corridor_half_mm: int = 150  # the direction of travel: the corridor plus a step aside
    drop_zones: int = 4
    drop_frames: int = 2  # consecutive frames (the tofd sim publishes ~13–15 frames/s)
    drop_tilt_max_mrad: int = 262  # 15°: walking tilts ≤ ~5°; after a contact it is garbage
    min_downward_milli: int = 35  # beams at least ~2° below the horizon can meet the floor
    crossings_mm: tuple[int, ...] = (800, 500, 300, 150)
    crossing_hysteresis_mm: int = 40  # back above a line by this much before it can fire again
    # A LOW EDGE ahead (a curb, a high sill, a step of about 30 mm and more) is seen by the beams that hit its face, and only from afar:
    # the sensor sits ~0.24 m above the floor and its beams reach the floor 0.4-1 m ahead, so inside ~0.4 m every beam passes over a
    # 60 mm face and "ahead" jumps to what lies beyond it (measured 2026-10-10: 450 mm, then 1177 mm, and the duck walked into the
    # curb). The tracker therefore remembers the edge it saw as a segment in the duck's own odometry and keeps reporting it as the
    # return ahead (or beside) while it is still there (needs the odometry given to ``step``). It is an EDGE (wide), not a bowl or a toy.
    #
    # The height is the second highest hit point of the nearest face above the floor. It reads high by 20-30 mm at 0.75-0.95 m (a zone
    # that grazes the floor returns its nearest part), so ``low_block_mm`` 55 means a step of about 30 mm and more: a 25 mm sill reads
    # at most 57 mm (27 % of its frames above 55), a 30 mm one up to 82, a 60 mm curb 55-95.
    low_block_mm: int = 55  # a face this high (by that estimate) is a step the duck must not climb
    low_top_mm: int = 130  # … up to this high; taller is a wall (the beams above hit it too)
    low_group_mm: int = 120  # hit points this far behind the nearest one are on the same face
    edge_min_span_mm: int = (
        300  # it is an EDGE when low hit points this far apart across the view are on it
    )
    latch_max_mm: int = (
        850  # remembered only when seen nearer than this (farther the floor reads 40-80 mm high)
    )
    latch_frames: int = 3  # … in this many frames in a row
    latch_ttl_s: float = 15.0  # it fades this long after it was last seen …
    latch_blind_mm: int = 450  # … but not while the duck is nearer than this to it (the beams pass over it: it cannot look again) …
    latch_hold_s: float = (
        120.0  # … for this long at most (an odometry that drifts, a thing that was moved)
    )
    latch_far_mm: int = 1200  # forgotten when farther than this from the trunk (the duck went away)
    latch_corridor_mm: int = (
        100  # in the body-width corridor it is "ahead", beside it "left" / "right"
    )
    face_depth_mm: int = (
        70  # hit points this far behind the nearest are the face (the top of a step lies beyond it)
    )
    face_margin_mm: int = 80  # the face is taken this much (at most) nearer than the nearest low return (see _fit_face)
    face_extend_mm: int = (
        250  # the remembered edge runs this far past each end of what was seen of it
    )
    face_min_half_mm: int = (
        150  # an edge is taken to reach at least this far each side of where it was seen …
    )
    face_max_half_mm: int = 450  # … and never more than this


DEFAULT = DepthConfig()


@dataclass(frozen=True)
class ToFFrame:
    """One ``tof.frame`` from tofd: 64 distances (mm) and statuses, row 0 the top, column 0 the
    sensor's left, stamped with the robot's monotonic clock (``t_ns``)."""

    t_ns: int
    distance_mm: tuple[int, ...]
    status: tuple[int, ...]

    @classmethod
    def of(cls, t_ns: int, distance_mm: Sequence[int], status: Sequence[int]) -> ToFFrame:
        return cls(int(t_ns), tuple(int(v) for v in distance_mm), tuple(int(v) for v in status))


@dataclass(frozen=True)
class SensorPose:
    """Where the ToF was when the frame was taken, from the posture tap: its pose in the trunk
    (``robot.state.frames.tof``, or the head FK), the trunk-frame projected gravity (upright is
    about ``(0, 0, -1)``) and the trunk's height from the leg FK (``None``: the model's standing
    height)."""

    sensor: tr.Pose
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    trunk_height_mm: int | None = None


@dataclass(frozen=True)
class DepthReading:
    """DS-v1 for one frame (all integers)."""

    t_ns: int
    available: bool
    ahead_mm: int = FAR_MM
    left_mm: int = FAR_MM
    right_mm: int = FAR_MM
    drop_zones: int = 0  # this frame, inside the bounds
    drop: bool = False  # debounced verdict
    drop_checked: bool = False  # False while tilted beyond the bound (no verdict either way)
    touch: bool = False
    too_close: int = 0
    floor: int = 0
    hits: int = 0
    empty: int = 0
    weak_used: int = 0
    head_pitch_mrad: int = 0  # the sensor's optical axis above (+) / below (−) the horizon
    tilt_mrad: int = 0  # the trunk's tilt from upright

    ahead_b_mrad: int = 0  # the nearest ahead return's bearing (trunk frame, left +)
    ahead_top_mm: int = (
        0  # the height above the floor (mm) of the face the nearest ahead return is on (0: none)
    )
    virtual: bool = False  # the nearest ahead / side return is the remembered face of a low thing under the beams
    # the face of the low thing the nearest ahead return is on, as a segment in the levelled trunk frame (x1, y1, x2, y2; mm,
    # x forward, y left): fitted through the hit points of the zones across the view, else square to the view
    low_face: tuple[int, int, int, int] | None = None
    low_edge_mm: int = 0  # the height estimate of the wide low EDGE in view (a curb, a step; seen or remembered), else 0

    def sense_depth(self) -> dict[str, int | bool | None]:
        """The ``sense.depth`` fields of percept-v1 (``None``: nothing seen that way)."""
        return {
            "ahead_mm": _none_if_far(self.ahead_mm),
            "left_mm": _none_if_far(self.left_mm),
            "right_mm": _none_if_far(self.right_mm),
            "drop_zones": self.drop_zones,
            "touch": self.touch,
            "head_pitch_mrad": self.head_pitch_mrad,
        }

    def view(self) -> DepthView | None:
        """The body loop's :class:`~duckmind.duckbody.protocol.DepthView` (``None`` without a
        ToF: every rule must then hold on the camera and the body alone)."""
        if not self.available:
            return None
        return DepthView(
            t_ns=self.t_ns,
            ahead_mm=_none_if_far(self.ahead_mm),
            left_mm=_none_if_far(self.left_mm),
            right_mm=_none_if_far(self.right_mm),
            drop_zones=self.drop_zones,
            touch=self.touch,
            head_pitch_mrad=self.head_pitch_mrad,
            ahead_b_mrad=self.ahead_b_mrad,
            ahead_top_mm=self.ahead_top_mm,
            virtual=self.virtual,
            low_edge_mm=self.low_edge_mm,
        )


def _none_if_far(value: int) -> int | None:
    return None if value >= FAR_MM else value


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in v))


def tilt_mrad(gravity: Sequence[float]) -> int:
    """The trunk's tilt from upright (mrad) from the trunk-frame projected gravity."""
    n = _norm(gravity)
    if n < 1e-9:
        return 0
    c = max(-1.0, min(1.0, -float(gravity[2]) / n))
    return int(round(math.acos(c) * 1000))


def accept_weak(
    frame: ToFFrame, previous: Sequence[int | None] | None, window_mm: int
) -> tuple[list[float | None], list[int | None], int]:
    """Slant ranges (m) per zone with DS-v1's status rule, the raw distances to remember for the
    next frame (valid or weak readings), and how many weak readings were used."""
    ranges: list[float | None] = []
    keep: list[int | None] = []
    used = 0
    for i in range(tr.ZONES):
        mm = frame.distance_mm[i] if i < len(frame.distance_mm) else 0
        code = frame.status[i] if i < len(frame.status) else tr.STATUS_NO_TARGET
        if code in tr.STATUS_VALID and mm > 0:
            ranges.append(mm / 1000.0)
            keep.append(mm)
        elif code in STATUS_WEAK and mm > 0:
            before = previous[i] if previous is not None and i < len(previous) else None
            if before is not None and abs(mm - before) <= window_mm:
                ranges.append(mm / 1000.0)
                used += 1
            else:
                ranges.append(None)
            keep.append(mm)
        else:
            ranges.append(None)
            keep.append(None)
    return ranges, keep, used


def summarise(
    frame: ToFFrame,
    ranges_m: Sequence[float | None],
    pose: SensorPose,
    standing_height_m: float,
    config: DepthConfig = DEFAULT,
    beam_dirs: Sequence[Sequence[float]] = tr.BEAMS,
) -> DepthReading:
    """One frame's DS-v1 numbers (no debounce: ``drop`` is this frame's count ≥ the threshold)."""
    posture = tr.Posture(
        gravity=pose.gravity,
        trunk_height_m=None if pose.trunk_height_mm is None else pose.trunk_height_mm / 1000.0,
    )
    zones = tr.project(ranges_m, pose.sensor, posture, standing_height_m, beam_dirs)
    level = tr.level_from_gravity(pose.gravity)
    sensor_level = level.rotate(pose.sensor.pos)
    height = posture.trunk_height_m if posture.trunk_height_m is not None else standing_height_m
    above = sensor_level[2] + height
    tilt = tilt_mrad(pose.gravity)
    ahead = left = right = float("inf")
    ahead_b = 0.0
    counts = {"floor": 0, "too_close": 0, "hit": 0, "empty": 0}
    # (forward, height above the floor) mm of the corridor's hits, (forward, left, height) of every forward hit and (forward, left) of
    # every forward floor return: what the low-edge test reads
    corridor: list[tuple[float, float]] = []
    hits3: list[tuple[float, float, float]] = []
    floors: list[tuple[float, float]] = []
    for zone in zones:
        counts[zone.kind] += 1
        if zone.kind == "floor" and zone.point is not None:
            pf = level.rotate(zone.point)
            if pf[0] > 0.0:
                floors.append((pf[0] * 1000.0, pf[1] * 1000.0))
        if zone.kind != "hit" or zone.point is None:
            continue
        p = level.rotate(zone.point)
        lateral_mm = p[1] * 1000.0
        if p[0] > 0.0 and abs(lateral_mm) <= config.side_reach_mm:
            hits3.append((p[0] * 1000.0, lateral_mm, (p[2] + height) * 1000.0))
        if p[0] > 0.0 and abs(lateral_mm) <= config.corridor_half_mm:
            corridor.append((p[0] * 1000.0, (p[2] + height) * 1000.0))
            if p[0] * 1000.0 < ahead:
                ahead = p[0] * 1000.0
                ahead_b = math.atan2(p[1], p[0])
        elif p[0] > -0.05 and abs(lateral_mm) <= config.side_reach_mm:
            horizontal = math.hypot(p[0], p[1]) * 1000.0
            if lateral_mm > 0:
                left = min(left, horizontal)
            else:
                right = min(right, horizontal)
    touch = counts["too_close"] >= config.touch_zones
    if touch:  # the face is on something: it is as near as the sensor itself
        ahead = min(ahead, max(0.0, sensor_level[0]) * 1000.0)
    # the drop test, inside its range, heading and tilt bounds
    checked = tilt <= config.drop_tilt_max_mrad and above > 0.0
    drops = 0
    if checked:
        for i, r in enumerate(ranges_m):
            direction = level.rotate(pose.sensor.quat.rotate(beam_dirs[i]))
            downward = -direction[2]
            if downward * 1000.0 < config.min_downward_milli:
                continue
            expected = above / downward
            fx = (sensor_level[0] + expected * direction[0]) * 1000.0
            fy = (sensor_level[1] + expected * direction[1]) * 1000.0
            if not (config.drop_min_reach_mm <= fx <= config.drop_reach_mm):
                continue
            if abs(fy) > config.drop_corridor_half_mm:
                continue
            if r is None or r * 1000.0 > expected * config.drop_ratio_milli + config.drop_slack_mm:
                drops += 1
    ahead_top = 0.0
    low_top = 0.0
    if corridor and not touch:
        ahead_top = max(z for x, z in corridor if x <= ahead + config.low_group_mm)
    # the nearest forward hit anywhere in the view (beside the corridor too: a duck facing a curb at an angle sees it on one
    # side first, and turns into it): if what it is on is a low EDGE, the segment of it (:class:`DepthConfig`, ``low_block_mm``)
    low_face = None
    if hits3 and not touch:
        d0 = min(math.hypot(x, y) for x, y, _ in hits3)
        group = sorted(
            (z for x, y, z in hits3 if math.hypot(x, y) <= d0 + config.low_group_mm), reverse=True
        )
        # (the second highest: one zone's outlier is not a step; a real face is hit by a row of zones across the view)
        top = group[1] if len(group) >= 2 else 0.0
        if config.low_block_mm <= top <= config.low_top_mm and group[0] <= config.low_top_mm * 1.5:
            low_face = _fit_face(hits3, floors, d0, config)
            if low_face[1]:
                low_top = top
    axis = level.rotate(pose.sensor.quat.rotate((1.0, 0.0, 0.0)))
    pitch = math.atan2(axis[2], math.hypot(axis[0], axis[1]))
    return DepthReading(
        t_ns=frame.t_ns,
        available=True,
        ahead_mm=_mm(ahead),
        left_mm=_mm(left),
        right_mm=_mm(right),
        drop_zones=drops,
        drop=checked and drops >= config.drop_zones,
        drop_checked=checked,
        touch=touch,
        too_close=counts["too_close"],
        floor=counts["floor"],
        hits=counts["hit"],
        empty=counts["empty"],
        head_pitch_mrad=int(round(pitch * 1000)),
        tilt_mrad=tilt,
        ahead_b_mrad=int(round(ahead_b * 1000)),
        ahead_top_mm=int(round(max(0.0, ahead_top))),
        low_face=None if low_face is None else low_face[0],
        low_edge_mm=int(round(low_top)) if low_face is not None and low_face[1] else 0,
    )


def _fit_face(
    hits: Sequence[tuple[float, float, float]],
    floors: Sequence[tuple[float, float]],
    ahead: float,
    config: DepthConfig,
) -> tuple[tuple[int, int, int, int], bool]:
    """The edge of a low thing as a segment (x1, y1, x2, y2; mm, trunk frame): a line through the low hit points near the
    nearest ahead return when at least three zones across the view are on it, else a segment square to the view through
    that return. Never shorter than 2 x ``face_min_half_mm`` or longer than 2 x ``face_max_half_mm``."""
    # (the face itself: the top of a step lies behind it and at its height, and is not the edge)
    near = [
        (x, y)
        for x, y, z in hits
        if -50.0 <= math.hypot(x, y) - ahead <= config.face_depth_mm and z <= config.low_top_mm
    ]
    near.sort(key=lambda q: q[1])
    half_min, half_max = float(config.face_min_half_mm), float(config.face_max_half_mm)
    # Where the face is: the beams that pass over its edge land on its top, behind it, and the beam below hit the floor in
    # front of it, so it lies between the last floor return and the first low one (a row of beams is 0.1-0.3 m apart on the
    # floor at this range). The nearest low return is the far end of that gap; it is taken a little nearer, by half the gap
    # up to ``face_margin_mm``, and the next look from closer narrows it.
    margin = 0.0
    if near:
        y_lo, y_hi = near[0][1] - 100.0, near[-1][1] + 100.0
        x_low = min(q[0] for q in near)
        before = [fx for fx, fy in floors if y_lo <= fy <= y_hi and fx < x_low]
        gap = x_low - max(before) if before else float(config.face_margin_mm)
        margin = min(float(config.face_margin_mm), 0.5 * gap)
    wide = len(near) >= 3 and near[-1][1] - near[0][1] >= float(config.edge_min_span_mm)
    if len(near) >= 3 and near[-1][1] - near[0][1] >= 150.0:
        n = float(len(near))
        my = sum(q[1] for q in near) / n
        mx = sum(q[0] for q in near) / n
        syy = sum((q[1] - my) ** 2 for q in near)
        sxy = sum((q[1] - my) * (q[0] - mx) for q in near)
        slope = max(-1.0, min(1.0, sxy / syy)) if syy > 1e-9 else 0.0  # x = mx + slope * (y - my)
        y1 = max(near[0][1] - 40.0, my - half_max)
        y2 = min(near[-1][1] + 40.0, my + half_max)
        if y2 - y1 < 2.0 * half_min:
            y1, y2 = my - half_min, my + half_min
        seg = (
            int(round(mx - margin + slope * (y1 - my))),
            int(round(y1)),
            int(round(mx - margin + slope * (y2 - my))),
            int(round(y2)),
        )
        return seg, wide
    ys = [q[1] for q in near] or [0.0]
    y0 = sum(ys) / len(ys)
    x0 = min((q[0] for q in near), default=ahead)  # (square to the view through the nearest point)
    x0 -= margin
    return (
        (int(round(x0)), int(round(y0 - half_min)), int(round(x0)), int(round(y0 + half_min))),
        wide,
    )


def _mm(value: float) -> int:
    if not math.isfinite(value):
        return FAR_MM
    return max(0, min(FAR_MM, int(round(value))))


@dataclass
class DepthTracker:
    """The per-frame step: weak-status memory, the drop debounce and the ``sense.depth`` edges.

    ``standing_height_m`` is the model's standing trunk height (``Reprojector.trunk_height_m``;
    the board reads it from ``robot.model``)."""

    standing_height_m: float
    config: DepthConfig = DEFAULT
    beam_dirs: Sequence[Sequence[float]] = tr.BEAMS
    _previous: list[int | None] | None = field(default=None, repr=False)
    _drop_run: int = 0
    _armed: dict[int, bool] = field(default_factory=dict, repr=False)
    _was_drop: bool = False
    _was_touch: bool = False
    last: DepthReading | None = None
    # the remembered face of a low thing (:class:`DepthConfig`, ``low_block_mm``): (x, y) in the duck's odometry (m), its
    # height (mm) and the time it was last seen (the frame's t_ns); how many frames in a row have seen one
    _face: tuple[float, float, float, float, int, int] | None = field(default=None, repr=False)
    _face_run: int = 0

    def reset(self) -> None:
        """Forget the previous frame (tofd restarted, the duck was carried)."""
        self._previous = None
        self._drop_run = 0
        self._armed = {}
        self._was_drop = False
        self._was_touch = False
        self.last = None
        self._face = None
        self._face_run = 0

    def unavailable(self, t_ns: int) -> DepthReading:
        """A duck without a ToF (or tofd down): nothing to read, no events."""
        self.reset()
        self.last = DepthReading(t_ns=int(t_ns), available=False)
        return self.last

    def step(
        self,
        frame: ToFFrame,
        pose: SensorPose,
        *,
        walking: bool = False,
        odom: tuple[float, float, float] | None = None,
        upright: bool = True,
    ) -> tuple[DepthReading, list[dict[str, Any]]]:
        """Summarise a frame; return the reading and the ``sense.depth`` payloads it triggers
        (only while ``walking``: a crossing of an ahead line, a drop starting, a touch starting).
        ``odom`` (x, y, yaw of the duck's own odometry, m / rad): with it a low thing seen from afar stays
        in the reading after the beams have passed over it (:class:`DepthConfig`, ``low_block_mm``); ``upright`` False (the
        trunk is lowered: :func:`upright`) lets no new edge in."""
        cfg = self.config
        ranges, keep, used = accept_weak(frame, self._previous, cfg.weak_window_mm)
        self._previous = keep
        raw = summarise(frame, ranges, pose, self.standing_height_m, cfg, self.beam_dirs)
        self._drop_run = self._drop_run + 1 if raw.drop else 0
        reading = DepthReading(
            **{
                **raw.__dict__,
                "drop": self._drop_run >= cfg.drop_frames,
                "weak_used": used,
                **self._remember_face(raw, frame.t_ns, odom, upright),
            }
        )
        events: list[dict[str, Any]] = []
        why: list[str] = []
        for line in cfg.crossings_mm:
            armed = self._armed.get(line, True)
            if armed and reading.ahead_mm <= line:
                self._armed[line] = False
                why.append(f"ahead_{line}")
            elif not armed and reading.ahead_mm > line + cfg.crossing_hysteresis_mm:
                self._armed[line] = True
        if reading.drop and not self._was_drop:
            why.append("drop")
        if reading.touch and not self._was_touch:
            why.append("touch")
        self._was_drop = reading.drop
        self._was_touch = reading.touch
        if walking and why:
            events.append(
                {"kind": EVENT, "t_ns": reading.t_ns, "why": ",".join(why), **reading.sense_depth()}
            )
        self.last = reading
        return reading, events

    def _remember_face(
        self,
        raw: DepthReading,
        t_ns: int,
        odom: tuple[float, float, float] | None,
        upright: bool = True,
    ) -> dict[str, Any]:
        """The low-edge memory (:class:`DepthConfig`): the changes to this frame's reading, if any."""
        cfg = self.config
        if odom is None:
            return {}
        ox, oy, yaw = odom
        c, s = math.cos(yaw), math.sin(yaw)
        seen = (
            upright
            and not raw.touch
            and raw.low_face is not None
            and raw.low_edge_mm > 0
            and _nearest_dist(raw.low_face) <= cfg.latch_max_mm
        )
        self._face_run = self._face_run + 1 if seen else 0
        if seen and self._face_run >= cfg.latch_frames and raw.low_face is not None:
            x1, y1, x2, y2 = (v / 1000.0 for v in raw.low_face)  # trunk frame (m) -> the odometry's
            new = (
                ox + x1 * c - y1 * s,
                oy + x1 * s + y1 * c,
                ox + x2 * c - y2 * s,
                oy + x2 * s + y2 * c,
            )
            old = self._face
            if old is not None and (t_ns - old[5]) * 1e-9 <= cfg.latch_ttl_s:
                new = _merge_segments(old[:4], new)  # the same edge seen again: the longest reach
            self._face = (*new, raw.low_edge_mm, t_ns)
        face = self._face
        if face is None:
            return {}
        # the segment in the trunk frame now: (forward, left) in metres, a little longer at both ends than it was seen (a
        # low edge runs to the walls of its doorway; the view only showed part of it)
        fx1, fy1, fx2, fy2 = _extend(
            face[:4], cfg.face_extend_mm / 1000.0, cfg.face_max_half_mm * 2 / 1000.0
        )
        a = ((fx1 - ox) * c + (fy1 - oy) * s, -(fx1 - ox) * s + (fy1 - oy) * c)
        b = ((fx2 - ox) * c + (fy2 - oy) * s, -(fx2 - ox) * s + (fy2 - oy) * c)
        near = _segment_nearest(a, b, -1e9, 1e9)
        age = (t_ns - face[5]) * 1e-9
        close = near is not None and math.hypot(near[0], near[1]) * 1000.0 <= cfg.latch_blind_mm
        if (
            age > (cfg.latch_hold_s if close else cfg.latch_ttl_s)
            or t_ns < face[5]
            or near is None
            or math.hypot(near[0], near[1]) * 1000.0 > cfg.latch_far_mm
            or max(a[0], b[0]) < -0.15
        ):
            self._face = None
            return {}
        corridor = cfg.latch_corridor_mm / 1000.0
        out: dict[str, Any] = {}
        ahead_piece = _segment_clip(a, b, -corridor, corridor)
        if ahead_piece is not None:
            f = min(ahead_piece[0][0], ahead_piece[1][0])
            if f >= 0.0 and raw.ahead_mm > f * 1000.0 + 120.0:
                at = ahead_piece[0] if ahead_piece[0][0] <= ahead_piece[1][0] else ahead_piece[1]
                out.update(
                    ahead_mm=_mm(f * 1000.0),
                    ahead_b_mrad=int(round(math.atan2(at[1], max(at[0], 1e-6)) * 1000)),
                    ahead_top_mm=int(face[4]),
                    low_edge_mm=int(face[4]),
                    virtual=True,
                )
        reach = cfg.side_reach_mm / 1000.0
        for lo, hi, key, current in (
            (corridor, reach, "left_mm", raw.left_mm),
            (-reach, -corridor, "right_mm", raw.right_mm),
        ):
            piece = _segment_clip(a, b, lo, hi)
            if piece is None:
                continue
            nearest = _segment_nearest(piece[0], piece[1], 0.0, 1e9)
            if nearest is None:
                continue
            d = math.hypot(nearest[0], nearest[1]) * 1000.0
            if d < current and d <= cfg.side_reach_mm and nearest[0] > -0.05:
                out[key] = _mm(d)
                out["virtual"] = True
        return out


def _extend(
    seg: tuple[float, float, float, float], by: float, longest: float
) -> tuple[float, float, float, float]:
    """The segment lengthened by ``by`` at each end, but not beyond ``longest`` in all."""
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return seg
    by = max(0.0, min(by, 0.5 * (longest - length)))
    ux, uy = dx / length, dy / length
    return (seg[0] - ux * by, seg[1] - uy * by, seg[2] + ux * by, seg[3] + uy * by)


def _nearest_dist(seg: tuple[int, int, int, int]) -> float:
    """The distance (mm) from the trunk to the nearest point of a segment given in the trunk frame."""
    near = _segment_nearest(
        (float(seg[0]), float(seg[1])), (float(seg[2]), float(seg[3])), -1e9, 1e9
    )
    return 1e9 if near is None else math.hypot(near[0], near[1])


def _merge_segments(
    old: tuple[float, float, float, float], new: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """The same straight edge seen again (both ends of ``new`` within 8 cm of the line of ``old``, nearly parallel): the old
    line over the union of the two reaches (a view narrows as the duck comes close; the edge does not). Else ``new``."""
    ox1, oy1, ox2, oy2 = old
    dx, dy = ox2 - ox1, oy2 - oy1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return new
    ux, uy = dx / length, dy / length
    ndx, ndy = new[2] - new[0], new[3] - new[1]
    nlen = math.hypot(ndx, ndy)
    if nlen > 1e-6 and abs(ux * ndy / nlen - uy * ndx / nlen) > 0.55:
        return new
    ts, off = [], 0.0
    for px, py in ((ox1, oy1), (ox2, oy2), (new[0], new[1]), (new[2], new[3])):
        ts.append((px - ox1) * ux + (py - oy1) * uy)
        off = max(off, abs(-(px - ox1) * uy + (py - oy1) * ux))
    if off > 0.08:
        return new
    lo, hi = min(ts), max(ts)
    if hi - lo > 0.9:  # no edge in a home is longer than a doorway
        mid = 0.5 * (ts[2] + ts[3])
        lo, hi = max(lo, mid - 0.45), min(hi, mid + 0.45)
    return (ox1 + lo * ux, oy1 + lo * uy, ox1 + hi * ux, oy1 + hi * uy)


def _segment_clip(
    a: tuple[float, float], b: tuple[float, float], lo: float, hi: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The part of the segment a-b whose second coordinate (left) is within [lo, hi], or None."""
    dl = b[1] - a[1]
    if abs(dl) < 1e-9:
        return (a, b) if lo <= a[1] <= hi else None
    t0, t1 = (lo - a[1]) / dl, (hi - a[1]) / dl
    if t0 > t1:
        t0, t1 = t1, t0
    t0, t1 = max(0.0, t0), min(1.0, t1)
    if t0 > t1:
        return None
    return (
        (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * dl),
        (a[0] + t1 * (b[0] - a[0]), a[1] + t1 * dl),
    )


def _segment_nearest(
    a: tuple[float, float], b: tuple[float, float], x_lo: float, x_hi: float
) -> tuple[float, float] | None:
    """The point of the segment a-b nearest the origin among those whose first coordinate (forward) is in [x_lo, x_hi]."""
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    # clip to the forward range first
    dx = bx - ax
    if abs(dx) < 1e-9:
        if not (x_lo <= ax <= x_hi):
            return None
        t0, t1 = 0.0, 1.0
    else:
        t0, t1 = (x_lo - ax) / dx, (x_hi - ax) / dx
        if t0 > t1:
            t0, t1 = t1, t0
        t0, t1 = max(0.0, t0), min(1.0, t1)
        if t0 > t1:
            return None
    px, py = ax + t0 * dx, ay + t0 * (by - ay)
    qx, qy = ax + t1 * dx, ay + t1 * (by - ay)
    vx, vy = qx - px, qy - py
    denom = vx * vx + vy * vy
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, -(px * vx + py * vy) / denom))
    return (px + t * vx, py + t * vy)


def pose_from_head(
    reprojector: tr.Reprojector,
    head_joints: Sequence[float],
    gravity: Sequence[float] = (0.0, 0.0, -1.0),
    trunk_height_mm: int | None = None,
) -> SensorPose:
    """A :class:`SensorPose` from the four measured head joints (when the robot state carries no
    ``frames.tof``): the official head FK."""
    return SensorPose(
        sensor=reprojector.sensor_in_trunk(head_joints),
        gravity=(float(gravity[0]), float(gravity[1]), float(gravity[2])),
        trunk_height_mm=trunk_height_mm,
    )


def upright(
    odom_pos: Sequence[float] | None, standing_m: float = 0.12, tol_m: float = 0.02
) -> bool:
    """Whether the trunk is at about its standing height (robotd's contact odometry z), so that the reprojection's assumption
    of it holds. Seated or crouched (6-9 cm) the same assumption reads the floor 20-60 mm too high: a wide low "edge" ahead
    that is only floor (measured 2026-10-10). No z (0) or no odometry: assumed upright (the rigs that have none)."""
    if odom_pos is None or len(odom_pos) < 3 or float(odom_pos[2]) <= 0.0:
        return True
    return abs(float(odom_pos[2]) - standing_m) <= tol_m


def pose_from_frames(
    tof_pos_m: Sequence[float],
    tof_quat_wxyz: Sequence[float],
    gravity: Sequence[float],
    trunk_height_mm: int | None = None,
) -> SensorPose:
    """A :class:`SensorPose` from ``robot.state.frames.tof`` (position in the trunk, metres, and
    the quaternion taking a sensor-frame vector to the trunk)."""
    return SensorPose(
        sensor=tr.Pose(
            (float(tof_pos_m[0]), float(tof_pos_m[1]), float(tof_pos_m[2])),
            tr.Quat(*(float(v) for v in tof_quat_wxyz)).normalized(),
        ),
        gravity=(float(gravity[0]), float(gravity[1]), float(gravity[2])),
        trunk_height_mm=trunk_height_mm,
    )


def body_view(reading: DepthReading) -> DepthView | None:
    """The body loop's :class:`~duckmind.duckbody.protocol.DepthView` of one reading, the one glue
    duck-body's process (``server.py``) and the rig's adapter (``brain.py``) share: ranges at or
    past :data:`FAR_MM` are ``None``; the bounded, debounced drop verdict is carried as
    ``drop_zones`` ≥ 4 (so the reflexes' ``no_drop`` fires exactly when DS-v1 says drop) or ≤ 3;
    ``ahead_b_mrad`` is the nearest ahead return's bearing, which owner A1's "is this return the
    target" check needs. ``None`` when the ToF has nothing (the camera and body must do)."""
    if not reading.available:
        return None
    return DepthView(
        t_ns=reading.t_ns,
        ahead_mm=_none_if_far(reading.ahead_mm),
        left_mm=_none_if_far(reading.left_mm),
        right_mm=_none_if_far(reading.right_mm),
        drop_zones=max(reading.drop_zones, 4) if reading.drop else min(reading.drop_zones, 3),
        touch=reading.touch,
        head_pitch_mrad=reading.head_pitch_mrad,
        ahead_b_mrad=reading.ahead_b_mrad,
        ahead_top_mm=reading.ahead_top_mm,
        virtual=reading.virtual,
    )
