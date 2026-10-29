"""Footfalls and steps from ``robot.state`` (spec §16.2, owner decision D8; phase 1 item 5).

A pure-Python port of phase 1's validated counter (``footfalls.py`` in the measurement records,
method "lift", parameters frozen 2026-09-26 before its held-out validation: −0.1 % over 14,513
footfalls). Each tick the lowest sole point of each foot is put in the gravity frame (leg FK from
the measured joints via :class:`~duckmind.duckbody.posture.LegFk`'s tree, levelled by the IMU);
the left-minus-right height difference is averaged over 40 ms and corrected by a learned standing
offset. A foot is *up* once it is 1.5 mm above the other and *down* again within 0.5 mm; a
landing after ≥ 40 ms of air is a footfall. Counted only while the walking policy drives, the duck
has not fallen and robotd applies a walking twist (or did within 1 s).

A **step** (the owner's step) is a footfall while the body moves forward ≥ 2 cm/s along the
odometry heading over 0.5 s centred on it (so it is classified 0.25 s late); otherwise it is
``in_place`` or ``backward``. No odometry (the headless path) → ``unknown``.

Sole points: robotd's odometry anchors (``odometry/src/anchors.rs`` ALPHA16) are the reference;
the board's copy is compiled into robotd, so this module takes them as data
(:func:`load_sole_points` reads the official file by path for the simulator side) and falls back
to the foot site alone (the sole centre).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from duckmind.duckbody.robotd_client import RobotState
from duckmind.duckbody.tof_reproject import Tree

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FootParams:
    lift_m: float = 0.0015
    land_m: float = 0.0005
    offset_tau_s: float = 2.0
    offset_band_m: float = 0.003
    offset_max_m: float = 0.005
    smooth_s: float = 0.04
    min_air_s: float = 0.04
    command_gate_s: float = 1.0
    window_s: float = 0.5
    forward_mps: float = 0.02


def load_sole_points(path: Path | str, name: str = "ALPHA16") -> tuple[list[Vec3], list[Vec3]]:
    """robotd's odometry anchors (left, right) in each foot-site frame, read from the official
    ``anchors.rs`` by path (nothing copied)."""
    text = Path(path).read_text(encoding="utf-8")
    block = text[text.index(f"pub const {name}") :]
    block = block[: block.index("};")]
    left_at, right_at = block.index("left: &["), block.index("right: &[")
    feet: list[list[Vec3]] = []
    for part in (block[left_at:right_at], block[right_at:]):
        nums = re.findall(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", part)
        feet.append([(float(a), float(b), float(c)) for a, b, c in nums])
    if len(feet[0]) != len(feet[1]) or len(feet[0]) < 4:
        raise ValueError(f"could not read {name} from {path}")
    return feet[0], feet[1]


@dataclass
class _Foot:
    up: bool = False
    t_off: float = 0.0
    peak: float = 0.0


@dataclass
class Footfalls:
    tree: Tree
    soles: tuple[Sequence[Vec3], Sequence[Vec3]] = (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),))
    params: FootParams = field(default_factory=FootParams)

    def __post_init__(self) -> None:
        from duckmind.duckbody.posture import FOOT_SITES
        from duckmind.duckbody.robotd_client import JOINT_NAMES

        sites = [self.tree.site(name) for name in FOOT_SITES]
        if any(s is None for s in sites):
            raise ValueError("the model has no foot sites")
        self.sites = [int(s) for s in sites if s is not None]
        self.wire_of = [JOINT_NAMES.index(name) for name in self.tree.joint_names]
        self.footfalls = 0
        self.steps = 0
        self.in_place = 0
        self.backward = 0
        self.unknown = 0
        self._feet = [_Foot(), _Foot()]
        self._gated_in = False
        self._offset = 0.0
        self._recent: deque[tuple[float, float]] = deque()
        self._prev_t: float | None = None
        self._cmd_t: float | None = None
        self._odom: deque[tuple[float, float, float, float]] = deque()
        self._waiting: deque[float] = deque()
        self._odom_seen = False

    def sole_heights(self, state: RobotState) -> tuple[float, float]:
        """Lowest sole point of each foot along the world vertical, relative to the trunk (m)."""
        angles = [float(state.joints[i]) for i in self.wire_of]
        if state.imu_quat is not None:
            w, x, y, z = state.imu_quat
            row = (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
        else:
            down = state.down or (0.0, 0.0, -1.0)
            row = (-down[0], -down[1], -down[2])
        out = []
        for site, points in zip(self.sites, self.soles, strict=True):
            pose = self.tree.site_pose(site, angles)
            lowest = min(
                sum(r * c for r, c in zip(row, pose.transform_point(p), strict=True))
                for p in points
            )
            out.append(lowest)
        return out[0], out[1]

    def _gate(self, state: RobotState, t: float) -> bool:
        if state.policy != "walk" or state.fallen:
            return False
        if any(abs(v) > 1e-3 for v in state.applied):
            self._cmd_t = t
        return self._cmd_t is not None and t - self._cmd_t <= self.params.command_gate_s

    def update(self, state: RobotState) -> int:
        """Feed one tick; returns how many steps were classified in it."""
        p = self.params
        t = state.t_ns / 1e9
        x, y, _ = state.odom_pos
        if (x, y) != (0.0, 0.0):
            self._odom_seen = True
        self._odom.append((t, x, y, state.odom_yaw))
        while len(self._odom) > 2 and self._odom[1][0] < t - 4.0 * p.window_s:
            self._odom.popleft()
        before = self.steps
        if not self._gate(state, t):
            self._gated_in = False
            self._feet = [_Foot(), _Foot()]
            self._recent.clear()
            self._classify(t)
            return self.steps - before
        zl, zr = self.sole_heights(state)
        raw = zl - zr - self._offset
        self._recent.append((t, raw))
        while self._recent and self._recent[0][0] < t - p.smooth_s - 1e-9:
            self._recent.popleft()
        diff = sum(v for _, v in self._recent) / len(self._recent)
        height = (diff, -diff)
        dt = 0.0 if self._prev_t is None else t - self._prev_t
        self._prev_t = t
        if not self._gated_in:
            self._gated_in = True
            self._feet = [
                _Foot(up=h > p.land_m, t_off=t, peak=h if h > p.land_m else 0.0) for h in height
            ]
            self._classify(t)
            return self.steps - before
        for i, h in enumerate(height):
            f = self._feet[i]
            if not f.up:
                if h > p.land_m:
                    f.up, f.t_off, f.peak = True, t, h
            else:
                f.peak = max(f.peak, h)
                if h <= p.land_m:
                    f.up = False
                    if f.peak >= p.lift_m and t - f.t_off >= p.min_air_s - 1e-9:
                        self.footfalls += 1
                        self._waiting.append(t)
        if not self._feet[0].up and not self._feet[1].up and abs(diff) < p.offset_band_m and dt > 0:
            a = min(1.0, dt / p.offset_tau_s)
            self._offset = max(-p.offset_max_m, min(p.offset_max_m, self._offset + a * diff))
        self._classify(t)
        return self.steps - before

    def _odom_at(self, t: float) -> tuple[float, float, float] | None:
        if not self._odom:
            return None
        prev = self._odom[0]
        if t <= prev[0]:
            return prev[1:]
        for cur in self._odom:
            if cur[0] >= t:
                a = (t - prev[0]) / max(1e-9, cur[0] - prev[0])
                return (
                    prev[1] + a * (cur[1] - prev[1]),
                    prev[2] + a * (cur[2] - prev[2]),
                    prev[3] + a * (cur[3] - prev[3]),
                )
            prev = cur
        return self._odom[-1][1:]

    def _classify(self, t: float) -> None:
        half = self.params.window_s / 2.0
        while self._waiting and self._waiting[0] + half <= t:
            at = self._waiting.popleft()
            if not self._odom_seen:
                self.unknown += 1
                continue
            a, b = self._odom_at(at - half), self._odom_at(at + half)
            if a is None or b is None:
                self.unknown += 1
                continue
            heading = a[2]
            fwd = (b[0] - a[0]) * math.cos(heading) + (b[1] - a[1]) * math.sin(heading)
            speed = fwd / self.params.window_s
            if speed >= self.params.forward_mps:
                self.steps += 1
            elif speed <= -self.params.forward_mps:
                self.backward += 1
            else:
                self.in_place += 1


__all__ = ["FootParams", "Footfalls", "load_sole_points"]
