"""duck-eye's ``floor`` task: FL-v1, the ground line and the trip line (spec §4.4; owner decision
D1: the duck learns what its floor looks like from ground it walked over without a contact).

**What it answers.** Looking from the camera's measured pose (the pose tap's trunk height from leg
FK, the camera in the trunk and the IMU: never the commanded head), how far the floor is clear in
five sectors across the picture (``free_mm``, from the trunk origin, level frame), what ends each
clear stretch (``wall``, ``thing`` or ``unknown``; the eye never claims a ``drop``: one camera
cannot see a step down of the same floor, the ToF does, spec §4.4), a trip line if one crosses the
floor (``trip``: a thin band that is not floor with the same floor beyond it — a threshold or a
cable), and the low things standing on the floor (``things``: bearing, range, width, height; the
``things`` task names them, :mod:`duckmind.eye.things`).

**How** (per look, on the quarter-resolution planes of :mod:`duckmind.eye.frames`):

1. the planes are smoothed 3×3 (sensor noise), and every luma sample gets a colour bin — luma in
   32 steps and the chroma *divided by the luma* (hue and saturation, so a plank's dark seam and
   a shaded patch keep their colour) in 48 × 48 — and a place on the floor: the ray through it
   (:mod:`duckmind.eye.camera`'s K and lens) cut with the floor plane from the camera's measured
   height and tilt;
2. the look's own reference: the *kinds* of floor (chroma cells) in the band just in front of the
   feet (:data:`NEAR_BAND_MM` beyond the nearest floor the camera sees) whose colours the learned
   model holds, and each kind's brightness there, in this light;
3. a sample is *floor-like* when the learned model holds its colour bin (a 3×3×3 neighbourhood
   count over a threshold: only colours the duck has walked on) **and** it is one of those kinds
   at :data:`LUMA_LO`–:data:`LUMA_HI` × its brightness at the feet. (Another kind of floor ahead —
   the rug beyond the planks — is not floor-like until the duck stands next to it; the
   "walked onto from" table that would allow it, :data:`USE_PAIRS`, is learned but off: measured,
   it let cardboard boxes pass as light planks.)
4. 24 columns (each ~1.6° wide), from the bottom up: the first run of rows that are mostly not
   floor and reach :data:`MIN_RUN_MM` deep on the floor ends the clear stretch (a plank seam or a
   cable is thinner: it becomes a trip-band candidate when its colour is not a learned floor
   colour); its bottom edge on the floor is the distance. If floor comes back above the run below
   the horizon it is a ``thing`` (low, the floor continues behind it), else a ``wall``; nothing up
   to :data:`MAX_RANGE_MM` is ``unknown``;
5. five sectors take their columns' nearest end; ``dim`` light multiplies the distances by 0.8
   (spec §4.4); ``dark`` and ``black`` give no floor line (the eye cannot be trusted there).

Measured 2026-09-27 (docs/validation/evidence/phase3/eye_floorline.json): default_home, 270
looks, median |error| 19 mm (p90 174 mm), a false stop in 16% of clear columns; the 4–8 cm box
on the path seen by the glance-down look 92–97% per colour by day and lamp light. A thing's
``h_mm`` is an upper bound (the run's top row is the top face's far edge).

**Learning** (:class:`FloorModel`). Each look keeps, for a few seconds and in memory only, the
colour bins of the floor it saw with their places on the floor (a list of integers, not a
picture; spec §4.8). When duck-body reports a walk through that look's view (``eye.learn``,
:class:`duckmind.eye.protocol.EyeLearn`: the ground points the duck walked over, the body's swept
width, the outcome), the bins inside the walked corridor become floor samples — only when the
walk was ``clear`` or ``stopped`` (no contact). Until the model has :data:`READY_WALKS` walks and
:data:`READY_SAMPLES` samples it is *not learned*: the look then seeds a provisional model from
the floor right in front of the feet (the picture's bottom band), marks every end ``unknown``,
caps the clear distance at :data:`BOOT_CAP_MM` and says ``learned: 0`` — duck-body walks short
segments until the duck knows its floor (conservative until learned).

Pure and deterministic: same planes, pose and model → the same integers (spec §11.7 P2). numpy
only; no clocks, no sockets, no :mod:`duckmind.world`.
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.eye.camera import CameraPose, EyeCamera

NCOL = 24  # columns across the upright picture (spec §4.4: each ~1.56°)
NSECT = 5  # sectors in a look (percept-v1 free_mm[5])
RUN = 2  # rows (quarter resolution) of not-floor that can end the clear stretch …
MIN_RUN_MM = 35  # … when they are this deep on the floor (a seam, a cable: thinner)
MIN_RUN_M = MIN_RUN_MM / 1000.0
RESUME = 3  # rows of floor above a run that make it a low thing (the floor goes on behind it)
FLOOR_FRACTION = 0.5  # a row of a column is floor when at least this much of it is floor-like
MAX_RANGE_MM = 1500  # the floor line looks no further (the planks blur into one colour)
DIM_FACTOR = 0.8  # spec §4.4: dim light → distances ×0.8
BOOT_CAP_MM = 350  # not learned yet: never claim more clear floor than this
BOOT_BAND_MM = 80  # the provisional model's seed: the floor this far beyond the nearest seen
READY_WALKS = 3  # walks learned before the model is trusted
READY_SAMPLES = 4000  # and floor samples
LEARN_RANGE_MM = 1600  # a look remembers the floor it saw up to this range
KEEP_LOOKS = 8  # looks remembered for eye.learn (a few seconds' worth)
LEARN_OUTCOMES = ("clear", "stopped")  # the walks whose ground is floor
TRIP_MAX_ROWS = 2  # a trip band is at most this many rows (thinner than RUN)
TRIP_MIN_COLS = 8  # columns that must show the band
TRIP_FIT_MM = 60  # the band's points must lie this close to one straight line

# colour bins
Y_BINS = 32  # luma / 8
C_BINS = 48  # normalised chroma: (U-128)·48/(Y+24), and V likewise, offset by 24
C_SCALE = 48.0
C_OFFSET = 24
N_BINS = Y_BINS * C_BINS * C_BINS
THRESH_FRACTION = 1.5e-4  # a bin's neighbourhood must hold this share of the samples …
THRESH_MIN = 3  # … and at least this many
# the look's own floor: the kinds of floor right in front of the feet, and their brightness now
TYPE_DIV = 2  # chroma bins per floor kind (both axes)
N_TYPES = (C_BINS // TYPE_DIV) ** 2
NEAR_BAND_MM = 120  # the reference band: this far beyond the nearest floor the camera sees
NEAR_MAX_MM = 800
TYPE_MIN_SHARE = 0.02  # a kind must be this much of the band (and 20 samples) to count
LUMA_LO = 0.80  # a floor sample is this much darker …
LUMA_HI = 1.25  # … to this much brighter than its kind in the band (before any learning)
R_BINS = 30  # learned relative brightness: log2(luma / the reference kind's), 0.1 steps …
R_STEP = 0.1
R_LO = -1.5  # … from -1.5 to +1.5
USE_PAIRS = False  # judge other kinds of floor by what was walked onto from the feet's kind:
# off by default — measured 2026-09-27 it let cardboard boxes pass as light planks (seen 66/72 →
# 63/72 by day, 31/36 → 15/36 at dusk) for few fewer false stops (19% → 18% of clear columns);
# the table is still learned
PAIR_READY = 500  # samples next to a reference kind before its pair table is used
PAIR_FRACTION = 1e-4  # a (kind, ratio) cell's neighbourhood must hold this share …
PAIR_MIN = 3  # … and at least this many samples

ENDS = ("wall", "thing", "unknown")


# ── geometry on the planes ─────────────────────────────────────────────────


def _undistort_np(xd: Any, yd: Any, dist: Sequence[float], iterations: int = 10) -> Any:
    """:func:`duckmind.eye.camera.undistort` on numpy arrays."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        slope = k1 + r2 * (2 * k2 + r2 * 3 * k3)
        ex = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        ey = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        a = radial + 2 * x * x * slope + 2 * p1 * y + 6 * p2 * x
        b = 2 * x * y * slope + 2 * p1 * x + 2 * p2 * y
        d = radial + 2 * y * y * slope + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * b
        x = x - (d * ex - b * ey) / det
        y = y - (a * ey - b * ex) / det
    return x, y


@functools.lru_cache(maxsize=8)
def _plane_rays(camera: EyeCamera, step: int, rows: int, cols: int) -> Any:
    """Undistorted normalised (x right, y down) of every plane sample, upright camera frame:
    an array (rows, cols, 2). Sample (i, j) is upright pixel (j·step, i·step)."""
    import numpy as np

    fx, fy, cx, cy = camera.upright_k()
    vv, uu = np.mgrid[0:rows, 0:cols].astype(np.float64) * step
    x, y = _undistort_np((uu - cx) / fx, (vv - cy) / fy, camera.upright_distortion())
    out = np.stack([x, y], axis=-1)
    out.setflags(write=False)
    return out


def upright_to_sensor(rotate: int) -> Any:
    """The rotation taking upright-camera vectors (x right, y down, z along the axis of the
    upright picture) to the sensor frame the pose is stated in."""
    import numpy as np

    table = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        270: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    }
    return np.array(table[rotate], dtype=np.float64)


def quat_matrix(q: Sequence[float]) -> Any:
    import numpy as np

    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def level_from_upright(camera: EyeCamera, pose: CameraPose) -> tuple[Any, Any]:
    """(R, p): upright-camera vectors → level frame, and the camera's centre in the level frame
    (metres; z is its height above the floor)."""
    import numpy as np

    rot = quat_matrix(pose.sensor_to_level) @ upright_to_sensor(camera.rotate)
    return rot, np.array(pose.position_level, dtype=np.float64)


@dataclass(frozen=True)
class GroundGrid:
    """Where every plane sample's ray meets the floor (level frame, metres): ``gx``, ``gy``,
    ``rng`` (the range from the level origin; ``inf`` at or above the horizon) and ``dz`` (the
    ray's vertical component, < 0 looking down)."""

    gx: Any
    gy: Any
    rng: Any
    dx: Any
    dy: Any
    dz: Any
    rot: Any
    pos: Any


def ground_grid(camera: EyeCamera, pose: CameraPose, step: int, rows: int, cols: int) -> GroundGrid:
    import numpy as np

    rays = _plane_rays(camera, step, rows, cols)
    rot, pos = level_from_upright(camera, pose)
    x, y = rays[..., 0], rays[..., 1]
    dx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2]
    dy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2]
    dz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2]
    down = dz < -1e-6
    t = np.where(down, -pos[2] / np.where(down, dz, -1.0), np.inf)
    gx = np.where(down, pos[0] + t * dx, np.inf)
    gy = np.where(down, pos[1] + t * dy, np.inf)
    rng = np.where(down, np.hypot(gx, gy), np.inf)
    return GroundGrid(gx=gx, gy=gy, rng=rng, dx=dx, dy=dy, dz=dz, rot=rot, pos=pos)


# ── colour bins ────────────────────────────────────────────────────────────


def colour_bins(y: Any, u: Any, v: Any) -> Any:
    """The colour bin of every luma sample (uint32): luma/8, and the chroma divided by the luma.
    ``u`` and ``v`` may be at half the luma's resolution (sample (i, j) on luma (2i, 2j))."""
    import numpy as np

    rows, cols = y.shape
    if u.shape != y.shape:
        ry, rx = -(-rows // u.shape[0]), -(-cols // u.shape[1])
        u = np.repeat(np.repeat(u, ry, axis=0), rx, axis=1)[:rows, :cols]
        v = np.repeat(np.repeat(v, ry, axis=0), rx, axis=1)[:rows, :cols]
    yy = y.astype(np.int32)
    den = yy + 24
    cu = ((u.astype(np.int32) - 128) * int(C_SCALE)) // den + C_OFFSET
    cv = ((v.astype(np.int32) - 128) * int(C_SCALE)) // den + C_OFFSET
    np.clip(cu, 0, C_BINS - 1, out=cu)
    np.clip(cv, 0, C_BINS - 1, out=cv)
    return ((yy >> 3) * C_BINS + cu) * C_BINS + cv


def blur3(plane: Any) -> Any:
    """3×3 box mean (edges repeated), rounded back to uint8: the sensor noise down ~3×."""
    import numpy as np

    a = plane.astype(np.int32)
    p = np.pad(a, 1, mode="edge")
    total = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            total += p[i : i + a.shape[0], j : j + a.shape[1]]
    return ((total + 4) // 9).astype(np.uint8)


def floor_kind(bins: Any) -> Any:
    """The coarse chroma cell (a kind of floor) of each colour bin."""
    cu = (bins // C_BINS) % C_BINS
    cv = bins % C_BINS
    per = C_BINS // TYPE_DIV
    return (cu // TYPE_DIV) * per + cv // TYPE_DIV


@dataclass(frozen=True)
class Reference:
    """The kinds of floor right in front of the feet in this look, and each one's mean luma."""

    kinds: tuple[int, ...] = ()
    luma: tuple[float, ...] = ()


def local_reference(bins: Any, luma: Any, grid: GroundGrid, table: Any) -> Reference:
    """The floor kinds in the band right in front of the feet (learned floor colours only)."""
    import numpy as np

    rng = grid.rng
    finite = np.isfinite(rng)
    if not finite.any():
        return Reference()
    nearest = float(rng[finite].min())
    near = finite & (rng <= min(nearest + NEAR_BAND_MM / 1000.0, NEAR_MAX_MM / 1000.0))
    ok = near & table[bins]
    n = int(ok.sum())
    if n < 20:
        return Reference()
    kinds = floor_kind(bins[ok])
    counts = np.bincount(kinds, minlength=N_TYPES)
    sums = np.bincount(kinds, weights=luma[ok].astype(np.float64), minlength=N_TYPES)
    keep = np.flatnonzero(counts >= max(20, int(n * TYPE_MIN_SHARE)))
    return Reference(
        kinds=tuple(int(k) for k in keep),
        luma=tuple(float(sums[k] / counts[k]) for k in keep),
    )


def ratio_bins(luma: Any, ref_luma: float) -> Any:
    """log2 of a sample's luma over a reference kind's, in :data:`R_BINS` steps."""
    import numpy as np

    lr = np.log2((luma.astype(np.float64) + 1.0) / (ref_luma + 1.0))
    return np.clip(((lr - R_LO) / R_STEP).astype(np.int64), 0, R_BINS - 1)


def _neighbourhood(counts: Any) -> Any:
    """Each bin's count plus its 26 neighbours' (luma ±1, both chroma ±1)."""
    import numpy as np

    cube = counts.reshape(Y_BINS, C_BINS, C_BINS).astype(np.int64)
    padded = np.pad(cube, 1)
    out = np.zeros_like(cube)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += padded[a : a + Y_BINS, b : b + C_BINS, c : c + C_BINS]
    return out.reshape(-1)


def floor_table(counts: Any, total: int | None = None) -> Any:
    """The floor-like lookup (bool per bin) of a count table."""
    import numpy as np

    total = int(counts.sum()) if total is None else total
    thresh = max(THRESH_MIN, int(total * THRESH_FRACTION))
    return np.ascontiguousarray(_neighbourhood(counts) >= thresh)


# ── the learned floor ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Seen:
    """What one look remembers for eye.learn: floor places (mm, level frame), colour bins, luma
    and the look's own reference (the kinds right in front of the feet)."""

    x_mm: Any
    y_mm: Any
    bins: Any
    luma: Any
    ref: Reference


def _pair_neighbourhood(counts: Any) -> Any:
    """A kind-by-ratio count table summed over neighbouring ratios (±1)."""
    import numpy as np

    table = counts.reshape(N_TYPES, R_BINS).astype(np.int64)
    padded = np.pad(table, ((0, 0), (1, 1)))
    return padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]


def _far_kinds(kind: int) -> Any:
    """Kinds at least two chroma cells away from ``kind`` (a different floor, not the same one
    shifted by noise or shade)."""
    import numpy as np

    per = C_BINS // TYPE_DIV
    ku, kv = divmod(np.arange(N_TYPES), per)
    au, av = divmod(kind, per)
    return (np.abs(ku - au) >= 2) | (np.abs(kv - av) >= 2)


class FloorModel:
    """The duck's own idea of what its floor looks like (owner decision D1).

    Two tables, both learned only from walked-clear ground: the floor's colours (``counts``, a
    gate: a colour never walked on is never floor) and, for each kind of floor that stood right
    in front of the feet when a look was taken (``pairs``), which kinds of floor at which
    brightness *relative to it* the duck then walked over — plank shades and seams on a wood
    floor, the rug beyond it — so a look judges the floor ahead against the floor at its feet,
    in this light."""

    def __init__(self, use_pairs: bool = USE_PAIRS) -> None:
        import numpy as np

        self.use_pairs = use_pairs
        self.counts = np.zeros(N_BINS, dtype=np.int64)
        self.pairs: dict[int, Any] = {}
        self.walks = 0
        self.samples = 0
        self._table: Any = None
        self._pair_tables: dict[int, Any] = {}
        self._seen: OrderedDict[str, _Seen] = OrderedDict()

    @property
    def ready(self) -> bool:
        return self.walks >= READY_WALKS and self.samples >= READY_SAMPLES

    def table(self) -> Any:
        if self._table is None:
            self._table = floor_table(self.counts, self.samples)
        return self._table

    def pair_table(self, kind: int) -> Any:
        """Which (kind, ratio) cells are floor next to reference ``kind`` (None: too few)."""
        if kind in self._pair_tables:
            return self._pair_tables[kind]
        counts = self.pairs.get(kind)
        table = None
        if counts is not None:
            total = int(counts.sum())
            if total >= PAIR_READY:
                thresh = max(PAIR_MIN, int(total * PAIR_FRACTION))
                table = _pair_neighbourhood(counts) >= thresh
        self._pair_tables[kind] = table
        return table

    def classify(self, bins: Any, luma: Any, ref: Reference, table: Any | None = None) -> Any:
        """Floor-like samples: a learned floor colour that is either the same kind as a kind
        right in front of the feet at about its brightness (×0.8…1.25), or another kind of floor
        the duck has walked onto from that kind, at the brightness it had then."""
        import numpy as np

        glob = (self.table() if table is None else table)[bins]
        kinds = floor_kind(bins)
        ok = np.zeros(bins.shape, dtype=bool)
        y = luma.astype(np.float64)
        for kind, ref_luma in zip(ref.kinds, ref.luma, strict=True):
            ok |= (kinds == kind) & (y >= LUMA_LO * ref_luma) & (y <= LUMA_HI * ref_luma)
            tab = self.pair_table(kind) if table is None and self.use_pairs else None
            if tab is not None:
                far = _far_kinds(kind)
                ok |= far[kinds] & tab[kinds, ratio_bins(luma, ref_luma)]
        return glob & ok

    def remember(
        self, look_id: str, grid: GroundGrid, bins: Any, luma: Any, ref: Reference
    ) -> None:
        """Keep a look's floor places, colour bins and luma (in memory, a few looks) for
        eye.learn."""
        import numpy as np

        near = grid.rng <= LEARN_RANGE_MM / 1000.0
        self._seen[look_id] = _Seen(
            x_mm=np.rint(grid.gx[near] * 1000).astype(np.int32),
            y_mm=np.rint(grid.gy[near] * 1000).astype(np.int32),
            bins=bins[near].astype(np.int64),
            luma=luma[near].astype(np.uint8),
            ref=ref,
        )
        self._seen.move_to_end(look_id)
        while len(self._seen) > KEEP_LOOKS:
            self._seen.popitem(last=False)

    def forget(self) -> None:
        self._seen.clear()

    def learn(
        self,
        look_id: str,
        outcome: str,
        floor: Sequence[Sequence[int]],
        width_mm: int = 120,
    ) -> int:
        """Add the samples inside a walked corridor to the model; the number added. The corridor
        is the path through ``floor`` (level frame of that look, mm) widened to ``width_mm``;
        nothing is learned from a walk that was not clear."""
        import numpy as np

        seen = self._seen.get(look_id)
        if seen is None or outcome not in LEARN_OUTCOMES or len(floor) == 0:
            return 0
        pts = np.asarray(floor, dtype=np.float64).reshape(-1, 2)
        px = seen.x_mm.astype(np.float64)
        py = seen.y_mm.astype(np.float64)
        half = max(float(width_mm), 1.0) / 2.0
        inside = np.zeros(px.shape, dtype=bool)
        if len(pts) == 1:
            inside = np.hypot(px - pts[0, 0], py - pts[0, 1]) <= half
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:], strict=True):
            ex, ey = bx - ax, by - ay
            seg2 = ex * ex + ey * ey
            if seg2 <= 0:
                d = np.hypot(px - ax, py - ay)
            else:
                t = np.clip(((px - ax) * ex + (py - ay) * ey) / seg2, 0.0, 1.0)
                d = np.hypot(px - (ax + t * ex), py - (ay + t * ey))
            inside |= d <= half
        added = int(inside.sum())
        if added:
            bins = seen.bins[inside]
            np.add.at(self.counts, bins, 1)
            kinds = floor_kind(bins)
            luma = seen.luma[inside]
            for kind, ref_luma in zip(seen.ref.kinds, seen.ref.luma, strict=True):
                table = self.pairs.setdefault(kind, np.zeros(N_TYPES * R_BINS, dtype=np.int64))
                np.add.at(table, kinds * R_BINS + ratio_bins(luma, ref_luma), 1)
                self._pair_tables.pop(kind, None)
            self.samples += added
            self.walks += 1
            self._table = None
        return added

    def learn_event(self, event: Any) -> int:
        """:meth:`learn` from an :class:`duckmind.eye.protocol.EyeLearn`."""
        return self.learn(event.look_id, event.outcome, event.floor, event.width_mm)

    # persistence (the model is the duck's own; integers only)
    def to_json(self) -> dict[str, Any]:
        import numpy as np

        nz = np.flatnonzero(self.counts)
        pairs = {}
        for kind, table in sorted(self.pairs.items()):
            idx = np.flatnonzero(table)
            pairs[str(kind)] = [[int(i), int(table[i])] for i in idx]
        return {
            "v": 2,
            "walks": self.walks,
            "samples": self.samples,
            "bins": [int(i) for i in nz],
            "counts": [int(c) for c in self.counts[nz]],
            "pairs": pairs,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> FloorModel:
        import numpy as np

        model = cls()
        bins = list(value.get("bins", []))
        counts = list(value.get("counts", []))
        if len(bins) != len(counts):
            raise ValueError("floor model: bins and counts differ in length")
        for b, c in zip(bins, counts, strict=True):
            if not 0 <= int(b) < N_BINS or int(c) < 0:
                raise ValueError("floor model: a bin is out of range")
            model.counts[int(b)] = int(c)
        for kind, cells in (value.get("pairs") or {}).items():
            k = int(kind)
            if not 0 <= k < N_TYPES:
                raise ValueError("floor model: a floor kind is out of range")
            table = np.zeros(N_TYPES * R_BINS, dtype=np.int64)
            for i, c in cells:
                if not 0 <= int(i) < N_TYPES * R_BINS or int(c) < 0:
                    raise ValueError("floor model: a pair cell is out of range")
                table[int(i)] = int(c)
            model.pairs[k] = table
        model.walks = int(value.get("walks", 0))
        model.samples = int(value.get("samples", int(model.counts.sum())))
        return model


# ── one look ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Column:
    free_mm: int
    end: str  # wall | thing | unknown
    b_mrad: int
    row: int  # the first not-floor row (quarter resolution), -1 when none
    top: int  # the run's top row for a thing, -1 otherwise
    thin: tuple[int, int] | None  # a trip band (row, rows) crossed before the end


def _column_edges(cols: int) -> list[tuple[int, int]]:
    import numpy as np

    e = np.linspace(0, cols, NCOL + 1).round().astype(int)
    return [(int(e[i]), int(e[i + 1])) for i in range(NCOL)]


def _sector_columns() -> list[tuple[int, int]]:
    import numpy as np

    e = np.linspace(0, NCOL, NSECT + 1).round().astype(int)
    return [(int(e[i]), int(e[i + 1])) for i in range(NSECT)]


def _interp_range(rng: Any, col: int, row: float) -> float:
    """The floor range (m) at a fractional row of one plane column."""
    r0 = int(math.floor(row))
    r1 = min(r0 + 1, rng.shape[0] - 1)
    a, b = float(rng[r0, col]), float(rng[r1, col])
    if not math.isfinite(a) or not math.isfinite(b):
        return b if math.isfinite(b) else a
    f = row - r0
    return a + (b - a) * f


def scan_columns(
    floorlike: Any,
    grid: GroundGrid,
    max_range_mm: int = MAX_RANGE_MM,
    learned: Any = None,
) -> list[Column]:
    """The 24 columns' clear distances from a floor-like mask (rows × cols bool). A run of rows
    that are not floor ends the clear stretch once it is :data:`RUN` rows and :data:`MIN_RUN_MM`
    deep on the floor (a plank's seam or a cable is thinner: a trip band when ``learned`` — the
    learned floor colours — says it is not a floor colour)."""
    import numpy as np

    rows, cols = floorlike.shape
    out: list[Column] = []
    limit = max_range_mm / 1000.0
    for c0, c1 in _column_edges(cols):
        mid = (c0 + c1 - 1) // 2
        frac = floorlike[:, c0:c1].mean(axis=1)
        is_floor = frac >= FLOOR_FRACTION
        rng_col = grid.rng[:, mid]
        in_range = np.isfinite(rng_col) & (rng_col <= limit)
        end_row = -1
        top_row = -1
        thin: tuple[int, int] | None = None
        end = "unknown"
        run = 0
        start = -1
        r = rows - 1
        last_range = float("nan")
        while r >= 0 and in_range[r]:
            last_range = float(rng_col[r])
            if not is_floor[r]:
                if run == 0:
                    start = r
                run += 1
                depth = float(rng_col[r]) - float(rng_col[start])
                if run >= RUN and depth >= MIN_RUN_M:
                    end_row = start
                    break
            else:
                if run > 0 and thin is None and start < rows - 1:
                    band = slice(start - run + 1, start + 1)
                    seam = learned is not None and float(learned[band, c0:c1].mean()) >= 0.5
                    if not seam:
                        thin = (start, run)  # the band's lowest row, its height in rows
                run = 0
            r -= 1
        if end_row < 0 and run >= RUN:
            end_row = start  # still not floor where the look's range runs out
        if end_row >= 0:
            if end_row >= rows - 1:
                edge = float(rng_col[rows - 1]) if math.isfinite(rng_col[rows - 1]) else 0.0
                free = edge
            else:
                free = _interp_range(grid.rng, mid, end_row + 0.5)
            # what is above the run: floor again below the horizon → a thing; else a wall
            rr = r - 1
            floor_again = 0
            end = "wall"
            while rr >= 0 and np.isfinite(rng_col[rr]):
                if is_floor[rr]:
                    floor_again += 1
                    if floor_again >= RESUME:
                        end = "thing"
                        top_row = rr + RESUME
                        break
                else:
                    floor_again = 0
                rr -= 1
        else:
            free = min(limit, last_range) if math.isfinite(last_range) else 0.0
            if r >= 0 and not in_range[r] and np.isfinite(rng_col[r]):
                free = limit  # the floor reached the look's range
        out.append(
            Column(
                free_mm=int(round(max(free, 0.0) * 1000)),
                end=end,
                b_mrad=0,
                row=end_row,
                top=top_row,
                thin=thin,
            )
        )
    return out


def _column_bearings(
    camera: EyeCamera, grid: GroundGrid, step: int, rows: int, cols: int
) -> list[int]:
    """Bearing (level frame, + left, mrad) of each column's middle at the picture's middle row."""
    rays = _plane_rays(camera, step, rows, cols)
    out = []
    row = rows // 2
    for c0, c1 in _column_edges(cols):
        mid = (c0 + c1 - 1) // 2
        x, y = float(rays[row, mid, 0]), float(rays[row, mid, 1])
        rot = grid.rot
        dx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2]
        dy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2]
        out.append(int(round(math.atan2(dy, dx) * 1000)))
    return out


def smooth(mask: Any) -> Any:
    """A 3×3 majority of a bool mask (speckle from sensor noise out)."""
    import numpy as np

    m = mask.astype(np.int16)
    p = np.pad(m, 1, mode="edge")
    total = sum(p[a : a + m.shape[0], b : b + m.shape[1]] for a in range(3) for b in range(3))
    return total >= 5


@dataclass
class FloorLook:
    """One look's floor line: the result integers plus what :mod:`things` needs."""

    result: dict[str, Any]
    columns: list[Column]
    floorlike: Any
    grid: GroundGrid | None


def _boot_table(bins: Any, grid: GroundGrid) -> Any:
    """Not learned yet: a provisional model from the floor just in front of the feet."""
    import numpy as np

    rng = grid.rng
    finite = rng[np.isfinite(rng)]
    if finite.size == 0:
        return None
    nearest = float(finite.min())
    band = np.isfinite(rng) & (rng <= nearest + BOOT_BAND_MM / 1000.0)
    if int(band.sum()) < 50:
        return None
    counts = np.bincount(bins[band].ravel(), minlength=N_BINS).astype(np.int64)
    return floor_table(counts)


def analyse(
    y: Any,
    u: Any,
    v: Any,
    camera: EyeCamera,
    pose: CameraPose | None,
    model: FloorModel,
    *,
    light: str = "bright",
    step: int = 4,
    look_id: str | None = None,
) -> FloorLook:
    """FL-v1 on one look's planes (``y`` rows × cols, ``u``/``v`` at half that), with the camera
    pose at the capture. Remembers the look's floor for :meth:`FloorModel.learn` when a
    ``look_id`` is given (bright or dim light only)."""
    import numpy as np

    if light not in ("bright", "dim"):
        return FloorLook({"ok": 0, "why": "dark"}, [], None, None)
    if pose is None:
        return FloorLook({"ok": 0, "why": "no_pose"}, [], None, None)
    rows, cols = y.shape
    grid = ground_grid(camera, pose, step, rows, cols)
    ys = blur3(y)
    bins = colour_bins(ys, blur3(u), blur3(v))
    learned = model.ready
    table = model.table() if learned else _boot_table(bins, grid)
    if table is None:
        return FloorLook({"ok": 0, "why": "no_floor_in_view"}, [], None, grid)
    ref = local_reference(bins, ys, grid, table)
    floorlike = smooth(model.classify(bins, ys, ref, None if learned else table))
    columns = scan_columns(floorlike, grid, learned=smooth(table[bins]))
    bearings = _column_bearings(camera, grid, step, rows, cols)
    factor = DIM_FACTOR if light == "dim" else 1.0
    cols_out = []
    for col, b in zip(columns, bearings, strict=True):
        free = int(round(col.free_mm * factor))
        end = col.end
        if not learned:
            free = min(free, BOOT_CAP_MM)
            end = "unknown"
        cols_out.append(Column(free, end, b, col.row, col.top, col.thin))
    sectors = []
    for s0, s1 in _sector_columns():
        group = cols_out[s0:s1]
        nearest = min(group, key=lambda c: c.free_mm)
        b0, b1 = group[0].b_mrad, group[-1].b_mrad
        conf = 900_000 if learned else 200_000
        if light == "dim":
            conf = conf * 7 // 10
        sectors.append(
            {
                "b_mrad": (b0 + b1) // 2,
                "w_mrad": abs(b0 - b1) + abs(bearings[1] - bearings[0]),
                "free_mm": nearest.free_mm,
                "end": nearest.end,
                "conf_ppm": conf,
            }
        )
    result: dict[str, Any] = {
        "ok": 1,
        "learned": int(learned),
        "walks": model.walks,
        "free_mm": [s["free_mm"] for s in sectors],
        "ends": [s["end"] for s in sectors],
        "sectors": sectors,
        "cols_free_mm": [c.free_mm for c in cols_out],
        "cols_end": [c.end for c in cols_out],
        "cols_b_mrad": [c.b_mrad for c in cols_out],
        "trip": _trip_line(cols_out, grid, camera, step, rows, cols) if learned else None,
        "things": _things(cols_out, grid, rows, cols, factor) if learned else [],
        "near_mm": int(
            round(float(np.nanmin(np.where(np.isfinite(grid.rng), grid.rng, np.nan))) * 1000)
        )
        if np.isfinite(grid.rng).any()
        else 0,
    }
    if look_id is not None:
        model.remember(look_id, grid, bins, ys, ref)
    return FloorLook(result, cols_out, floorlike, grid)


def _things(
    columns: list[Column], grid: GroundGrid, rows: int, cols: int, factor: float
) -> list[dict[str, int]]:
    """Low things on the floor: neighbouring ``thing`` columns at about the same range."""
    edges = _column_edges(cols)
    groups: list[list[int]] = []
    for i, col in enumerate(columns):
        if col.end != "thing":
            continue
        if groups and groups[-1][-1] == i - 1 and abs(columns[i - 1].free_mm - col.free_mm) <= 150:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    pos = grid.pos
    for g in groups:
        r_mm = min(columns[i].free_mm for i in g)
        b_left, b_right = columns[g[0]].b_mrad, columns[g[-1]].b_mrad
        step_mrad = abs(columns[1].b_mrad - columns[0].b_mrad) if len(columns) > 1 else 27
        w_mrad = abs(b_left - b_right) + step_mrad
        # height: the run's top row, where its ray passes over the thing's front face
        heights = []
        for i in g:
            col = columns[i]
            if col.top < 0:
                continue
            c0, c1 = edges[i]
            mid = (c0 + c1 - 1) // 2
            gx, gy = float(grid.gx[col.row, mid]), float(grid.gy[col.row, mid])
            if not (math.isfinite(gx) and math.isfinite(gy)):
                continue
            horiz = math.hypot(gx - pos[0], gy - pos[1])
            # the ray through the run's top row, followed out to the front face's distance
            flat = math.hypot(float(grid.dx[col.top, mid]), float(grid.dy[col.top, mid]))
            if flat > 1e-9:
                heights.append(pos[2] + float(grid.dz[col.top, mid]) / flat * horiz)
        h_mm = int(round(max(heights) * 1000)) if heights else -1
        out.append(
            {
                "b_mrad": (b_left + b_right) // 2,
                "w_mrad": w_mrad,
                "r_mm": r_mm,
                "w_mm": int(round(r_mm * w_mrad / 1000.0)),
                "h_mm": h_mm,
                "col0": g[0],
                "col1": g[-1],
                "row": max(columns[i].row for i in g),
                "top": min((columns[i].top for i in g if columns[i].top >= 0), default=-1),
            }
        )
    return out


def _trip_line(
    columns: list[Column], grid: GroundGrid, camera: EyeCamera, step: int, rows: int, cols: int
) -> dict[str, int] | None:
    """A thin not-floor band with floor beyond it across enough columns, on one straight line."""
    import numpy as np

    edges = _column_edges(cols)
    pts = []
    for i, col in enumerate(columns):
        if col.thin is None:
            continue
        row, _height = col.thin
        c0, c1 = edges[i]
        mid = (c0 + c1 - 1) // 2
        gx, gy = float(grid.gx[row, mid]), float(grid.gy[row, mid])
        if math.isfinite(gx) and math.isfinite(gy) and math.hypot(gx, gy) * 1000 < col.free_mm:
            pts.append((gx, gy))
    if len(pts) < TRIP_MIN_COLS:
        return None
    a = np.asarray(pts)
    centre = a.mean(axis=0)
    _, _, vt = np.linalg.svd(a - centre)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    off = np.abs((a - centre) @ normal)
    if float(np.median(off)) * 1000 > TRIP_FIT_MM:
        return None
    dist = abs(float(centre @ normal))
    if float(normal @ centre) < 0:
        normal = -normal
    return {
        "mm": int(round(dist * 1000)),
        "mrad": int(round(math.atan2(normal[1], normal[0]) * 1000)),
        "cols": len(pts),
    }


# ── the task ─────────────────────────────────────────────────────────────


class FloorTask:
    """The ``floor`` task as duck-eye's server runs it: ``run(view)`` per look,
    ``learn(event)`` per ``eye.learn``. Holds the duck's floor model."""

    name = "floor"

    def __init__(self, model: FloorModel | None = None) -> None:
        self.model = model if model is not None else FloorModel()
        self.last: FloorLook | None = None
        self.last_id: str | None = None

    def run(self, view: Any) -> dict[str, Any]:
        """One look (:class:`duckmind.eye.frames.LookView`)."""
        planes = view.planes
        level = str(view.light.get("level", "bright")) if view.light else "bright"
        look = analyse(
            planes.y,
            planes.u,
            planes.v,
            view.camera,
            view.camera_pose,
            self.model,
            light=level,
            step=planes.luma_step,
            look_id=view.look.look_id,
        )
        self.last = look
        self.last_id = view.look.look_id
        return look.result

    def learn(self, event: Any) -> int:
        return self.model.learn_event(event)

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> FloorTask:
        """``{"model": <FloorModel.to_json()>}`` restores a learned floor."""
        model = options.get("model")
        return cls(FloorModel.from_json(model) if isinstance(model, Mapping) else None)

    def digest_items(self) -> list[str]:
        """What goes into the perception hash (the learned floor itself does not: it is the
        duck's memory, not the perception code)."""
        return [
            f"floor.v1 ncol={NCOL} run={RUN} min_run_mm={MIN_RUN_MM} resume={RESUME}",
            f"floor.bins y={Y_BINS} c={C_BINS} kinds={TYPE_DIV} luma={LUMA_LO},{LUMA_HI}",
            f"floor.learn walks={READY_WALKS} samples={READY_SAMPLES} pairs={int(USE_PAIRS)}",
            f"floor.range={MAX_RANGE_MM} dim={DIM_FACTOR} boot={BOOT_CAP_MM}",
        ]
