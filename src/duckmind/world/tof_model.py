"""The world's ToF: the head's VL53L5CX/L8CX in 8×8 mode, for the world server's ``tof`` op
(spec §11.4, owner decisions F2 and "ToF gate: tofd --hz 15").

The official ``tofd --sim`` polls ``{"op":"tof"}`` on the body port at its own rate and publishes
the answer verbatim as ``tof.frame``, so this is where the simulated sensor lives: the answer is
exactly the official body server's four fields ``{"rows": 8, "cols": 8, "distance_mm": [64 ×
i16], "status": [64 × u8]}`` (``tofd``'s ``SimDepth``; P1: ``tests/test_tof_op_protocol.py``),
and everything else about the frame — the truth, why each zone got its status — goes to the
evaluators' log (``x_tof_log``), never into the op.

What it keeps from the official ``sim/tof.py``: the ``tof`` site's frame (+x forward, +y left,
+z up), the 45° field in 8 × 8 zones with the same zone centres (row 0 the top, column 0 the
sensor's LEFT, as ``kinematics::tof`` reads the real buffer), 4 m of range, noise σ = 3 mm + 0.5 %
of the range, distances truncated to whole millimetres, self-hits reported (the duck's own shell
is in the cast), and the guard against a zero-length ray before the first forward pass.

What it adds (the spec's list; every number is a guess until phase 8 refits it on the board):

* **3 × 3 sub-rays per zone** (``mj_multiRay``, 576 rays), grouped into targets (sub-returns
  closer than :attr:`TofParams.cluster_gap_m` are one target). The sensor reports one target per
  zone, the *strongest* (the VL53L5CX's default target order; upstream never changes it): signal
  ∝ the target's share of the zone ÷ its range², so a 3 cm table leg close by wins over the wall
  behind it and loses to it from far away. Its distance is the mean of its sub-returns.
* **Only what an infrared sensor sees**: the home's geom groups 0 (seen by camera and ToF) and 4
  (ToF only), and the duck's own visual shell (group 2) — a rug, a mat, window glass and the
  camera-only landmarks are passed through (``homelayout``'s groups).
* **Statuses** (``tof/src/lib.rs``): 5 valid; 9 valid with a large pulse when a second target
  in the zone is at least half as strong (a merged target); 255 no target (distance 0); 4 or 13
  ("weak": still carrying a distance, trusted by duck-body only when it repeats) with a chance
  that grows past 0.30 m and at grazing incidence; a crosstalk phantom under 10 cm with status
  5 (the Reprojector's too-close band) — off by default (``phantom_p`` 0: an unfitted guess of
  0.2 % per zone per frame is 1.7 phantoms a second at 13 frames/s, which phase 3's stop reflex
  would take for an obstacle; the knob stays for evaluators until the phase-8 board fit); and a
  shorter range where daylight from a
  window reaches the target (``ambient``: lux at each target from the lights; none at night).
  Dark, low-reflectance materials are not modelled: the layouts carry no material classes yet.

Deterministic: frame ``seq`` draws from its own stream ``(seed, seq)``. The world is read from a
qpos/mocap copy taken under ``world.lock`` (microseconds) and cast outside it on a private
``MjData``, so robotd's lockstep writes never wait for the rays.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

ROWS = 8
COLS = 8
ZONES = ROWS * COLS
FOV_DEG = 45.0
MAX_RANGE_M = 4.0
STATUS_VALID = 5
STATUS_LARGE_PULSE = 9
STATUS_NO_TARGET = 255
STATUS_CONSISTENCY = 4  # "target consistency failed"
STATUS_INCONSISTENT = 13  # "target detected but inconsistent data"
WEAK_STATUSES = (STATUS_CONSISTENCY, STATUS_INCONSISTENT)
I16_MAX = 32767
# the home's camera+ToF (0) and ToF-only (4) geoms and the duck's visual shell (2); homelayout
# puts camera-only things (rugs, mats, glass, landmarks) in 1 and sensor helpers in 5
TOF_GEOMGROUP = (1, 0, 1, 0, 1, 0)
SITE = "tof"
# the reasons in the evaluators' log, one letter per zone
WHY = {
    "valid": "v",
    "merged": "m",
    "weak": "w",
    "phantom": "p",
    "none": "n",
    "sun": "s",
    "far": "f",
}


@dataclass(frozen=True)
class TofParams:
    """The sensor model's knobs (all to be refitted on the board in phase 8)."""

    sub: int = 3  # sub-rays per zone per axis
    max_range_m: float = MAX_RANGE_M
    sigma_m: float = 0.003  # noise: σ = sigma_m + sigma_share × range (official sim/tof.py)
    sigma_share: float = 0.005
    cluster_gap_m: float = 0.06  # sub-returns closer than this are one target
    merge_ratio: float = 0.5  # a second target this strong: status 9
    weak_from_m: float = 0.30  # "4/13 flicker past 30 cm"
    weak_per_m: float = 0.04  # chance per metre beyond weak_from_m
    weak_incidence: float = 0.15  # extra chance at grazing incidence, × (1 − cos θ)
    weak_max: float = 0.35
    phantom_p: float = 0.0  # per zone per frame (off until the board fit; see the docstring)
    phantom_mm: tuple[int, int] = (20, 90)
    sun_lux: float = 300.0  # ambient lux at a target that halves the range


def zone_centres() -> list[float]:
    """The zone centres across the field (radians), as official ``sim/tof.py`` computes them."""
    half = math.radians(FOV_DEG) / 2.0
    edges = [-half + 2.0 * half * i / COLS for i in range(COLS + 1)]
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(COLS)]


def _direction(elevation: float, azimuth: float) -> tuple[float, float, float]:
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def beams() -> list[tuple[float, float, float]]:
    """The 64 zone-centre directions in the sensor frame, row-major (official ``sim/tof.py``)."""
    centres = zone_centres()
    return [_direction(-centres[row], -centres[col]) for row in range(ROWS) for col in range(COLS)]


def sub_beams(sub: int = 3) -> tuple[list[tuple[float, float, float]], list[int]]:
    """``sub × sub`` directions per zone, the centres of equal sub-cells of the zone, and the
    zone of each (zone-major: zone 0's sub-rays first)."""
    half = math.radians(FOV_DEG) / 2.0
    width = 2.0 * half / COLS
    dirs: list[tuple[float, float, float]] = []
    zones: list[int] = []
    for row in range(ROWS):
        for col in range(COLS):
            for i in range(sub):
                for j in range(sub):
                    el = -(-half + width * (row + (i + 0.5) / sub))
                    az = -(-half + width * (col + (j + 0.5) / sub))
                    dirs.append(_direction(el, az))
                    zones.append(row * COLS + col)
    return dirs, zones


@dataclass
class Target:
    """One target in a zone: its sub-returns' mean range (m), share of the zone and signal."""

    distance: float
    share: float
    signal: float
    cos_incidence: float
    geom: int


def targets(hits: Sequence[tuple[float, float, int]], rays: int, gap_m: float) -> list[Target]:
    """Group a zone's sub-returns ``(distance m, |cos incidence|, geom)`` into targets, nearest
    first (``rays``: how many sub-rays the zone had, hits or not)."""
    ordered = sorted(hits)
    groups: list[list[tuple[float, float, int]]] = []
    for hit in ordered:
        if groups and hit[0] - groups[-1][-1][0] < gap_m:
            groups[-1].append(hit)
        else:
            groups.append([hit])
    out = []
    for group in groups:
        distance = sum(h[0] for h in group) / len(group)
        share = len(group) / rays
        out.append(
            Target(
                distance=distance,
                share=share,
                signal=share / max(distance, 0.1) ** 2,
                cos_incidence=sum(h[1] for h in group) / len(group),
                geom=group[0][2],
            )
        )
    return out


def weak_chance(distance: float, cos_incidence: float, params: TofParams) -> float:
    """How likely a real sensor marks this return 4 or 13 (grows with range and obliquity)."""
    far = max(0.0, distance - params.weak_from_m) * params.weak_per_m
    oblique = (1.0 - max(0.0, min(1.0, cos_incidence))) * params.weak_incidence
    return min(params.weak_max, far + oblique)


def range_limit(ambient_lux: float, params: TofParams) -> float:
    """The sensor's reach with this much daylight on the target."""
    return params.max_range_m / (1.0 + max(0.0, ambient_lux) / params.sun_lux)


class TofSensor:
    """The sensor on one duck: :meth:`capture` a frame from a world (its model's ``tof`` site).

    ``ambient``: a callable ``(points (n, 3) world m) -> lux per point`` for the daylight at the
    targets (None: no sunlight effect). Thread-safe (one frame at a time)."""

    def __init__(
        self,
        model: Any,
        *,
        site: str = SITE,
        seed: int = 7,
        params: TofParams | None = None,
        geomgroup: Sequence[int] = TOF_GEOMGROUP,
        ambient: Callable[[Any], Any] | None = None,
        log_size: int = 256,
    ) -> None:
        import mujoco
        import numpy as np

        self._mj = mujoco
        self._np = np
        self.model = model
        self.site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site))
        if self.site < 0:
            raise ValueError(f"the model has no {site!r} site")
        self.seed = int(seed)
        self.params = params or TofParams()
        self.geomgroup = np.array(list(geomgroup), dtype=np.uint8)
        self.ambient = ambient
        self.data = mujoco.MjData(model)
        dirs, zones = sub_beams(self.params.sub)
        self.sub_dirs = np.array(dirs, dtype=np.float64)
        self.sub_zone = zones
        self.rays = len(dirs)
        self.centre_dirs = np.array(beams(), dtype=np.float64)
        self._geomid = np.zeros(self.rays, dtype=np.int32)
        self._dist = np.zeros(self.rays, dtype=np.float64)
        self._normal = np.zeros(self.rays * 3, dtype=np.float64)
        self._lock = threading.Lock()
        self.seq = 0
        self.log: deque[dict[str, Any]] = deque(maxlen=log_size)
        self.costs: deque[float] = deque(maxlen=512)  # ms per frame
        self._names: dict[int, str] = {}

    # ── the world ─────────────────────────────────────────────────────

    def copy_state(self, world: Any) -> float:
        """Copy the world's positions into the private data (the caller holds no lock; this
        takes ``world.lock`` for the copy only) and return its physics time."""
        data = self.data
        with world.lock:
            data.qpos[:] = world.data.qpos
            if self.model.nmocap:
                data.mocap_pos[:] = world.data.mocap_pos
                data.mocap_quat[:] = world.data.mocap_quat
            sim_time = float(world.data.time)
        data.time = sim_time
        self._mj.mj_kinematics(self.model, data)
        return sim_time

    def geom_name(self, geom: int) -> str:
        if geom not in self._names:
            mj = self._mj
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom)
            if not name:
                body = int(self.model.geom_bodyid[geom])
                owner = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body) or f"body{body}"
                name = f"{owner}/geom{geom}"
            self._names[geom] = str(name)
        return self._names[geom]

    # ── one frame ─────────────────────────────────────────────────────

    def capture(self, world: Any) -> dict[str, Any]:
        """One frame of the world as it is now: the op's four fields (the log gets the rest)."""
        with self._lock:
            t0 = time.perf_counter()
            sim_time = self.copy_state(world)
            frame, truth = self._frame(sim_time)
            cost = (time.perf_counter() - t0) * 1000.0
            self.costs.append(cost)
            truth["cost_ms"] = round(cost, 3)
            self.log.append(truth)
            return frame

    def frame_of(self, data: Any, sim_time: float = 0.0) -> dict[str, Any]:
        """A frame of an ``MjData`` whose kinematics are done (tests, no world lock)."""
        with self._lock:
            self._mj.mj_copyData(self.data, self.model, data)
            frame, truth = self._frame(sim_time)
            self.log.append(truth)
            return frame

    def _frame(self, sim_time: float) -> tuple[dict[str, Any], dict[str, Any]]:
        np, mj, p = self._np, self._mj, self.params
        data = self.data
        seq = self.seq
        self.seq += 1
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3).copy()
        world_dirs = self.sub_dirs @ rotation.T
        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES
        why = ["n"] * ZONES
        truth_mm = [-1] * ZONES
        geoms: list[str | None] = [None] * ZONES
        record: dict[str, Any] = {
            "seq": seq,
            "sim_time": round(sim_time, 4),
            "wall_us": time.time_ns() // 1000,
            "pos_mm": [int(round(float(v) * 1000.0)) for v in origin],
            "forward": [round(float(v), 4) for v in rotation[:, 0]],
        }
        if not np.isfinite(world_dirs).all() or np.linalg.norm(rotation) < 1e-9:
            record.update(distance_mm=distance_mm, status=status, why="".join(why))
            return self._op(distance_mm, status), record
        mj.mj_multiRay(
            self.model,
            data,
            origin,
            np.ascontiguousarray(world_dirs).ravel(),
            self.geomgroup,
            1,
            -1,
            self._geomid,
            self._dist,
            self._normal,
            self.rays,
            p.max_range_m,
        )
        dist = self._dist.tolist()
        gid = self._geomid.tolist()
        normals = self._normal.reshape(-1, 3)
        cosines = np.abs(np.einsum("ij,ij->i", normals, world_dirs)).tolist()
        per_zone: list[list[tuple[float, float, int]]] = [[] for _ in range(ZONES)]
        for k in range(self.rays):
            if 0.0 <= dist[k] <= p.max_range_m and gid[k] >= 0:
                per_zone[self.sub_zone[k]].append((dist[k], cosines[k], gid[k]))
        rng = np.random.default_rng([self.seed, seq])
        noise = rng.normal(0.0, 1.0, ZONES)
        weak_draw = rng.random(ZONES)
        weak_kind = rng.random(ZONES)
        phantom_draw = rng.random(ZONES)
        phantom_mm = rng.integers(p.phantom_mm[0], p.phantom_mm[1] + 1, ZONES)
        rays_per_zone = p.sub * p.sub
        chosen: list[Target | None] = [None] * ZONES
        seconds: list[Target | None] = [None] * ZONES
        for zone in range(ZONES):
            hits = per_zone[zone]
            if not hits:
                continue
            found = targets(hits, rays_per_zone, p.cluster_gap_m)
            truth_mm[zone] = int(found[0].distance * 1000.0)
            ranked = sorted(found, key=lambda t: -t.signal)
            chosen[zone] = ranked[0]
            seconds[zone] = ranked[1] if len(ranked) > 1 else None
        limits = [p.max_range_m] * ZONES
        if self.ambient is not None:
            points = [
                origin + world_dirs[zone * rays_per_zone + rays_per_zone // 2] * t.distance
                if t is not None
                else origin
                for zone, t in enumerate(chosen)
            ]
            lux = self.ambient(np.array(points))
            limits = [range_limit(float(v), p) for v in lux]
        for zone in range(ZONES):
            target = chosen[zone]
            if phantom_draw[zone] < p.phantom_p:
                distance_mm[zone] = int(phantom_mm[zone])
                status[zone] = STATUS_VALID
                why[zone] = WHY["phantom"]
                continue
            if target is None:
                continue
            if target.distance > limits[zone]:
                why[zone] = WHY["sun"]
                continue
            sigma = p.sigma_m + p.sigma_share * target.distance
            measured = max(0.0, target.distance + float(noise[zone]) * sigma)
            distance_mm[zone] = min(I16_MAX, int(measured * 1000.0))
            geoms[zone] = self.geom_name(target.geom)
            second = seconds[zone]
            if weak_draw[zone] < weak_chance(target.distance, target.cos_incidence, p):
                status[zone] = WEAK_STATUSES[int(weak_kind[zone] * 2) % 2]
                why[zone] = WHY["weak"]
            elif second is not None and second.signal >= p.merge_ratio * target.signal:
                status[zone] = STATUS_LARGE_PULSE
                why[zone] = WHY["merged"]
            else:
                status[zone] = STATUS_VALID
                why[zone] = WHY["valid"]
        record.update(
            distance_mm=list(distance_mm),
            status=list(status),
            why="".join(why),
            truth_mm=truth_mm,
            geoms=geoms,
        )
        return self._op(distance_mm, status), record

    @staticmethod
    def _op(distance_mm: list[int], status: list[int]) -> dict[str, Any]:
        return {"rows": ROWS, "cols": COLS, "distance_mm": distance_mm, "status": status}

    # ── for evaluators ────────────────────────────────────────────────

    def entries(self, since: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self.log if since is None or e["seq"] > since]
        return items[-max(1, int(limit)) :]

    def cost(self) -> dict[str, Any]:
        with self._lock:
            values = sorted(self.costs)
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "p50_ms": round(values[len(values) // 2], 3),
            "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 3),
            "max_ms": round(values[-1], 3),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "rows": ROWS,
            "cols": COLS,
            "fov_deg": FOV_DEG,
            "rays": self.rays,
            "geomgroup": [int(v) for v in self.geomgroup],
            "seed": self.seed,
            "frames": self.seq,
            "params": asdict(self.params),
        }
