"""duck-eye's ``nest`` task: NB-v1, the charging dock's tags and its direction cues (spec §4.4;
owner decisions D3 — the official AprilTag library, OpenCV as the fallback — and C5 — the dock's
light and its cyan panel give a direction only; a position needs a tag).

**Tags.** The dock carries tag36h11 id 17 (100 mm, on its face, seen across the room) and id 18
(40 mm, flat at the front of its floor plate, read looking down on the way in). A look reads them:

1. **the whole frame at half resolution** (AprilTag's ``quad_decimate`` 2: quads found at half
   resolution, edges refined and bits read at full);
2. **the ROI at full resolution** when that found nothing and duck-body says where it believes the
   dock is (``hints.nest_b_mrad``, trunk frame): within ±15° of that bearing, the rows where a
   tag 0.4–5 m away can be.

The spec's order is the other way round (ROI first, for the board's budget); measured on this
Mac (2026-09-27, 450 looks) the half-resolution frame costs less than the full-resolution ROI
(p50 31 ms against 48 ms) and reads as much (bright 272/300 against 269/300), so it goes first
(``NestConfig.roi_first`` restores the spec's order). Both passes are the spec's.

A detection becomes a **dock anchor** only after the eye's own checks, so a library's mistake
cannot move the duck's idea of home: the id is 17 or 18; the eye reads the tag's 8×8 cells itself
through the detected corners and decodes them against its own copy of the codes
(:mod:`duckmind.eye.tag36h11`) — at most :data:`MAX_HAMMING` wrong bit (none in ``dark``), a black
border, enough black/white contrast; the pose explains the corners to within a pixel or two; the
range is 0.1–5 m; and, with a pose, the tag sits where such a tag can be (the face tag well above
the floor, the plate tag on it). In ``black`` light no tag is read at all (spec §4.5: trust
nothing but the dock's light).

**Pose** (tag frame: x right, y up, z out of the printed face): the corners, undistorted with
duck-eye's own K and lens (:mod:`duckmind.eye.camera`), give a homography; its pose is refined by
Gauss–Newton, and the other planar solution (the mirror about the line of sight) is refined too:
the better fit wins, and when both fit alike the one whose face stands as that tag must (face tag
upright, plate tag flat) wins. Results in the level frame (x forward, y left, z up; origin on the
floor under the trunk origin): the tag's centre ``x_mm y_mm z_mm``, its range ``r_mm`` and bearing
``b_mrad`` from the trunk origin (+ left), elevation ``e_mrad`` from the camera, and ``yaw_mrad``,
the heading of the tag's face normal (π when it faces the duck square on).

**Direction cues** (never an anchor, never a range: C5):

* ``panel`` — the dock's cyan face panel (C3) in ``bright`` or ``dim`` light: the largest compact
  cyan blob on the quarter-resolution chroma;
* ``light`` — the dock's status light (C1/C4) in ``dark`` or ``black``: a small saturated blob in a
  dark picture.

Pure step functions (the detector object is the only state; no clocks, no sockets, no
:mod:`duckmind.world`). numpy, and ``pupil_apriltags`` (or ``cv2``) for the quad finding.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.eye import tag36h11
from duckmind.eye.camera import CameraPose, EyeCamera
from duckmind.eye.light import strip_like

FACE_TAG_M = 0.100  # tag 17's black square
EDGE_TAG_M = 0.040  # tag 18's
TAG_SIZES = {tag36h11.NEST_TAG_ID: FACE_TAG_M, tag36h11.EDGE_TAG_ID: EDGE_TAG_M}
ROI_HALF_MRAD = 262  # ±15° around the believed bearing (spec §4.4)
ROI_DECIMATE = 1.0
ROI_NEAR_M, ROI_FAR_M = 0.4, 5.0  # the ROI covers a tag this near to this far
YAW_AMBIGUOUS = 2.0  # the mirror pose fits within this factor: the tag's facing is not known
FULL_DECIMATE = 2.0  # the whole frame "at half resolution"
MAX_HAMMING = 2  # bits the eye's own decode may correct (bright, dim; ids 17/18 only, the
# family's codes are ≥ 11 bits apart: a random pattern lands within 2 bits of either id in any
# of four turns with a chance of ~8e-8)
DARK_MAX_HAMMING = 0  # … in dark light
MIN_CONTRAST = 30  # luma between the tag's white and black cells
DARK_MIN_CONTRAST = 40
MAX_BORDER_ERRORS = 3  # of the 28 black-border cells
MIN_SIDE_PX = 12.0
MIN_RANGE_M, MAX_RANGE_M = 0.08, 5.0
REPROJ_PX = 2.0  # the pose must explain the corners this well (or 3% of the side)
# where the tags can be (level frame, the tag centre's height above the floor, metres)
HEIGHT_BAND = {tag36h11.NEST_TAG_ID: (0.04, 0.30), tag36h11.EDGE_TAG_ID: (-0.03, 0.05)}
# the cyan panel (C3): normalised chroma (x48/(Y+24)), measured on the renders; bright/dim only
CYAN_U = (3, 22)
CYAN_V_MAX = -12
CYAN_MIN_PX = 12  # chroma samples (90×160 at 720p)
LIGHT_LUMA = 235  # the status light in a dark picture (quarter resolution luma)
LIGHT_MIN_PX, LIGHT_MAX_PX = 2, 900
LIGHT_MAX_MEAN = 80  # a dark picture: the frame's mean luma at most this


@dataclass(frozen=True)
class RawTag:
    """One detection from a library: its id, corners (upright full-resolution pixels, the tag's
    own bottom-left, bottom-right, top-right, top-left) and the library's own quality numbers."""

    tag_id: int
    corners: Any  # (4, 2) float
    hamming: int
    margin: float


_SAFE_DETECTOR: dict[int, Any] = {}


def safe_detector_class(pupil_mod: Any) -> Any:
    """``pupil_apriltags.Detector`` with a ``__del__`` that frees things in a safe order.

    pupil-apriltags 1.0.4.post11 (the current release) frees the tag families first and the
    detector second, but ``apriltag_detector_destroy`` still walks the families to free their
    decode tables: a heap use-after-free (a read, then ``fam->impl = NULL`` written into the freed
    family). macOS's allocator shrugs it off; glibc corrupts its heap and a later ``malloc``
    aborts the process (``malloc(): mismatching next->prev_size``), as seen on the Spark. Here
    the detector goes first, then the families, which no longer reference each other."""
    cls = _SAFE_DETECTOR.get(id(pupil_mod))
    if cls is not None:
        return cls

    class SafeDetector(pupil_mod.Detector):  # type: ignore[misc, name-defined]
        def __del__(self) -> None:
            ptr = getattr(self, "tag_detector_ptr", None)
            libc = getattr(self, "libc", None)
            families = getattr(self, "tag_families", None) or {}
            self.tag_detector_ptr = None
            if ptr is None or libc is None:
                return
            libc.apriltag_detector_destroy.restype = None
            libc.apriltag_detector_destroy(ptr)  # also frees the families' decode tables
            for name, family in families.items():
                destroy = getattr(libc, f"{name}_destroy", None)
                if destroy is not None:
                    destroy.restype = None
                    destroy(family)

    _SAFE_DETECTOR[id(pupil_mod)] = SafeDetector
    return SafeDetector


class TagBackend:
    """AprilTag 3 through ``pupil_apriltags`` (BSD; owner decision D3), else OpenCV's aruco
    ``DICT_APRILTAG_36h11``. ``detect(gray, decimate)`` → :class:`RawTag` list."""

    def __init__(self, prefer: str = "apriltag") -> None:
        self.name = ""
        self._pupil: dict[float, Any] = {}
        self._cv: dict[float, Any] = {}
        self._pupil_mod: Any = None
        self._cv_mod: Any = None
        if prefer == "apriltag":
            try:
                import pupil_apriltags

                self._pupil_mod = pupil_apriltags
                self.name = "apriltag"
            except ImportError:
                pass
        if not self.name:
            try:
                import cv2

                self._cv_mod = cv2
                self.name = "opencv"
            except ImportError as exc:
                raise RuntimeError("no tag library: install pupil-apriltags (or opencv)") from exc

    def detect(self, gray: Any, decimate: float = 1.0) -> list[RawTag]:
        import numpy as np

        img = np.ascontiguousarray(gray, dtype=np.uint8)
        if img.size == 0:
            return []
        if self._pupil_mod is not None:
            det = self._pupil.get(decimate)
            if det is None:
                det = safe_detector_class(self._pupil_mod)(
                    families="tag36h11",
                    nthreads=1,
                    quad_decimate=float(decimate),
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                )
                self._pupil[decimate] = det
            return [
                RawTag(
                    int(d.tag_id),
                    np.asarray(d.corners, dtype=np.float64),
                    int(d.hamming),
                    float(d.decision_margin),
                )
                for d in det.detect(img)
            ]
        cv2 = self._cv_mod
        det = self._cv.get(decimate)
        if det is None:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.aprilTagQuadDecimate = float(decimate)
            det = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params
            )
            self._cv[decimate] = det
        corners, ids, _ = det.detectMarkers(img)
        out = []
        if ids is not None:
            for c, i in zip(corners, ids.ravel(), strict=True):
                q = np.asarray(c, dtype=np.float64).reshape(4, 2)
                # OpenCV draws this dictionary turned 180° from AprilTag's own images: its
                # (TL, TR, BR, BL) are the tag's (BR, BL, TL, TR)
                out.append(RawTag(int(i), q[[1, 0, 3, 2]], -1, 0.0))
        return out


# ── the eye's own read of a tag's cells ────────────────────────────────────


def _homography(src: Any, dst: Any) -> Any:
    """3×3 H with dst ~ H·src (four or more point pairs, DLT)."""
    import numpy as np

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


# the corners' cell coordinates on the 8×8 black square (x right, y down as printed):
# bottom-left (0, 8), bottom-right (8, 8), top-right (8, 0), top-left (0, 0)
_CELL_CORNERS = ((0.0, 8.0), (8.0, 8.0), (8.0, 0.0), (0.0, 0.0))


def _bilinear(g: Any, u: Any, v: Any) -> Any:
    import numpy as np

    hgt, wid = g.shape
    uu = np.clip(u, 0, wid - 1.001)
    vv = np.clip(v, 0, hgt - 1.001)
    u0 = np.floor(uu).astype(int)
    v0 = np.floor(vv).astype(int)
    fu, fv = uu - u0, vv - v0
    return (
        g[v0, u0] * (1 - fu) * (1 - fv)
        + g[v0, u0 + 1] * fu * (1 - fv)
        + g[v0 + 1, u0] * (1 - fu) * fv
        + g[v0 + 1, u0 + 1] * fu * fv
    )


def _plane_fit(x: Any, y: Any, val: Any) -> Any:
    """Least-squares a + b·x + c·y through samples (the light across the tag)."""
    import numpy as np

    a = np.stack([np.ones_like(x), x, y], axis=1)
    coef, *_ = np.linalg.lstsq(a, val, rcond=None)
    return coef


CELL_SPREAD = 0.2  # samples at the cell centre and ±0.2 cell (in the tag's own plane)
DECODE_SHARPENING = 0.25  # AprilTag 3's default


def read_cells(gray: Any, corners: Any) -> tuple[list[list[int]], float] | None:
    """The 8×8 black-bordered square's cells (1 white) read through the corners, and the
    contrast (white − black luma); ``None`` when the square leaves the picture.

    Each cell is the mean of five bilinear samples at its centre and ±:data:`CELL_SPREAD` of a
    cell along the tag's own axes (so a tag seen at a slant is sampled inside its cells), and it
    is compared with a threshold that varies across the tag: planes fitted to the white ring and
    to the black border, as AprilTag's own decoder does."""
    import numpy as np

    h = _homography(_CELL_CORNERS, corners)
    idx = np.arange(-1, 9) + 0.5  # cell centres of the 10×10 grid, white ring included
    cx, cy = np.meshgrid(idx, idx)
    hgt, wid = gray.shape
    g = gray.astype(np.float64)
    vals = np.zeros(cx.shape)
    offsets = (
        (0.0, 0.0),
        (CELL_SPREAD, 0.0),
        (-CELL_SPREAD, 0.0),
        (0.0, CELL_SPREAD),
        (0.0, -CELL_SPREAD),
    )
    for dx, dy in offsets:
        pts = np.stack([(cx + dx).ravel(), (cy + dy).ravel(), np.ones(cx.size)])
        img = h @ pts
        u = img[0] / img[2]
        v = img[1] / img[2]
        if u.min() < 0 or v.min() < 0 or u.max() > wid - 1 or v.max() > hgt - 1:
            return None
        vals += _bilinear(g, u, v).reshape(cx.shape)
    vals /= len(offsets)
    ring = np.zeros((10, 10), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    border = np.zeros((10, 10), dtype=bool)
    border[1, 1:-1] = border[-2, 1:-1] = border[1:-1, 1] = border[1:-1, -2] = True
    white = _plane_fit(cx[ring], cy[ring], vals[ring])
    black = _plane_fit(cx[border], cy[border], vals[border])
    mid = (white + black) / 2.0
    rel = vals - (mid[0] + mid[1] * cx + mid[2] * cy)
    # AprilTag's decode sharpening: a Laplacian on the sampled grid undoes some of the blur a
    # small, slanted tag spreads into its neighbour cells
    padded = np.pad(rel, 1)
    lap = 4 * rel - (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
    rel = rel + DECODE_SHARPENING * lap
    inner = (rel > 0)[1:-1, 1:-1]
    cells = [[int(inner[y, x]) for x in range(8)] for y in range(8)]
    contrast = float(np.median(vals[ring]) - np.median(vals[border]))
    return cells, contrast


def verify(
    gray: Any, raw: RawTag, max_hamming: int, min_contrast: float
) -> tuple[int, float] | None:
    """The eye's own decode of a detection: (bit errors, contrast), or ``None`` when the cells do
    not say this tag in this orientation."""
    read = read_cells(gray, raw.corners)
    if read is None:
        return None
    cells, contrast = read
    if contrast < min_contrast:
        return None
    if tag36h11.border_errors(cells) > MAX_BORDER_ERRORS:
        return None
    value = tag36h11.read_code(cells)
    errors = tag36h11.hamming(value, tag36h11.code(raw.tag_id))
    if errors > max_hamming:
        return None
    return errors, contrast


# ── pose ───────────────────────────────────────────────────────────────


def _rodrigues(w: Any) -> Any:
    import numpy as np

    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def _residual(rot: Any, t: Any, obj: Any, norm: Any) -> Any:
    p = obj @ rot.T + t
    return (p[:, :2] / p[:, 2:3] - norm).ravel()


def _refine(rot: Any, t: Any, obj: Any, norm: Any, iterations: int = 8) -> tuple[Any, Any, float]:
    """Gauss–Newton on the normalised reprojection error (rotation as a left increment)."""
    import numpy as np

    for _ in range(iterations):
        r0 = _residual(rot, t, obj, norm)
        jac = np.zeros((r0.size, 6))
        eps = 1e-6
        for k in range(3):
            dw = np.zeros(3)
            dw[k] = eps
            jac[:, k] = (_residual(_rodrigues(dw) @ rot, t, obj, norm) - r0) / eps
            dt = np.zeros(3)
            dt[k] = eps
            jac[:, 3 + k] = (_residual(rot, t + dt, obj, norm) - r0) / eps
        step, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        rot = _rodrigues(step[:3]) @ rot
        t = t + step[3:]
        if float(np.linalg.norm(step)) < 1e-10:
            break
    err = float(np.sqrt(np.mean(_residual(rot, t, obj, norm) ** 2)))
    return rot, t, err


def _object_points(size_m: float) -> Any:
    import numpy as np

    s = size_m / 2.0
    return np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])


def undistort_px(camera: EyeCamera, pts: Any) -> Any:
    """Upright pixels → undistorted normalised (x right, y down) of the upright picture."""
    import numpy as np

    from duckmind.eye.floorline import _undistort_np

    fx, fy, cx, cy = camera.upright_k()
    p = np.asarray(pts, dtype=np.float64)
    x, y = _undistort_np((p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, camera.upright_distortion())
    return np.stack([x, y], axis=1)


def solve_pose(norm: Any, size_m: float) -> list[tuple[Any, Any, float]]:
    """Both planar solutions (rotation tag→upright camera, translation, rms normalised error),
    best fit first."""
    import numpy as np

    obj = _object_points(size_m)
    h = _homography(obj[:, :2], norm)
    h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
    lam = 2.0 / (np.linalg.norm(h1) + np.linalg.norm(h2))
    r1, r2, t = lam * h1, lam * h2, lam * h3
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    rot = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1, 1, -1]) @ vt
    first = _refine(rot, t, obj, norm)
    # the mirror solution: the face normal reflected about the line of sight
    rot1, t1, _ = first
    n = rot1[:, 2]
    sight = -t1 / np.linalg.norm(t1)
    n2 = 2 * float(n @ sight) * sight - n
    axis = np.cross(n, n2)
    s = float(np.linalg.norm(axis))
    out = [first]
    if s > 1e-9:
        ang = math.atan2(s, float(n @ n2))
        second = _refine(_rodrigues(axis / s * ang) @ rot1, t1, obj, norm)
        out.append(second)
    out.sort(key=lambda sol: sol[2])
    return out


# ── the look ──────────────────────────────────────────────────────────


@dataclass
class NestConfig:
    roi_half_mrad: int = ROI_HALF_MRAD
    full_decimate: float = FULL_DECIMATE
    roi_decimate: float = ROI_DECIMATE
    use_roi: bool = True
    full_frame: bool = True
    roi_first: bool = False
    height_band: Mapping[int, tuple[float, float]] = field(
        default_factory=lambda: dict(HEIGHT_BAND)
    )


def _yaw_gap(r_lu: Any, rot_a: Any, rot_b: Any) -> int:
    """The difference (mrad, 0…π) between two poses' face headings in the level frame."""
    na, nb = r_lu @ rot_a[:, 2], r_lu @ rot_b[:, 2]
    d = math.atan2(na[1], na[0]) - math.atan2(nb[1], nb[0])
    return int(round(abs((d + math.pi) % (2 * math.pi) - math.pi) * 1000))


YAW_TRUST_MRAD = 300  # a sighting carries the tag's facing only when the two poses agree this well


def roi_box(
    camera: EyeCamera, pose: CameraPose | None, bearing_mrad: int, half_mrad: int
) -> tuple[int, int, int, int] | None:
    """The upright pixel box (u0, v0, u1, v1) where a dock tag can be seen at bearings
    ``bearing ± half`` (level frame) from :data:`ROI_NEAR_M` to :data:`ROI_FAR_M` away and
    0–0.3 m above the floor; ``None`` when that wedge is out of the picture."""
    if pose is None:
        return None
    width, height = camera.upright_size
    us, vs = [], []
    for k in range(-4, 5):
        b = (bearing_mrad + half_mrad * k / 4) / 1000.0
        for dist in (ROI_NEAR_M, 1.0, ROI_FAR_M):
            for z in (0.0, 0.3):
                px = camera.pixel_of_ground(dist * math.cos(b), dist * math.sin(b), pose, z_m=z)
                if px is not None and -width <= px[0] <= 2 * width:
                    us.append(px[0])
                    vs.append(px[1])
    if not us:
        return None
    u0, u1 = max(0, int(math.floor(min(us))) - 8), min(width, int(math.ceil(max(us))) + 9)
    v0, v1 = max(0, int(math.floor(min(vs))) - 24), min(height, int(math.ceil(max(vs))) + 25)
    return (u0, v0, u1, v1) if u1 - u0 >= 16 and v1 - v0 >= 16 else None


def _tag_result(
    raw: RawTag,
    corners_full: Any,
    errors: int,
    contrast: float,
    camera: EyeCamera,
    pose: CameraPose | None,
    config: NestConfig,
    mode: str,
) -> dict[str, Any] | None:
    import numpy as np

    from duckmind.eye.floorline import level_from_upright

    size = TAG_SIZES[raw.tag_id]
    side = float(np.mean(np.linalg.norm(np.roll(corners_full, 1, axis=0) - corners_full, axis=1)))
    if side < MIN_SIDE_PX:
        return None
    norm = undistort_px(camera, corners_full)
    sols = solve_pose(norm, size)
    fx = camera.upright_k()[0]
    level = level_from_upright(camera, pose) if pose is not None else None
    best = sols[0]
    if level is not None and len(sols) > 1 and sols[1][2] < 3.0 * max(sols[0][2], 1e-9):
        # both explain the corners: take the one standing as this tag must
        want_up = raw.tag_id == tag36h11.EDGE_TAG_ID

        def upright_score(sol: tuple[Any, Any, float]) -> float:
            nz = float((level[0] @ sol[0][:, 2])[2])
            return -abs(nz) if not want_up else nz

        best = max(sols[:2], key=upright_score)
    rot, t, err = best
    other = sols[1] if best is sols[0] else sols[0]
    ambiguous = (
        len(sols) > 1
        and max(best[2], other[2]) < YAW_AMBIGUOUS * max(min(best[2], other[2]), 1e-9)
        and float(best[0][:, 2] @ other[0][:, 2]) < math.cos(math.radians(5.0))
    )
    err_px = err * fx
    if err_px > max(REPROJ_PX, 0.03 * side):
        return None
    dist = float(np.linalg.norm(t))
    if not MIN_RANGE_M <= dist <= MAX_RANGE_M:
        return None
    out: dict[str, Any] = {
        "id": raw.tag_id,
        "mode": mode,
        "ham": errors,
        "side_px": int(round(side)),
        "contrast": int(round(contrast)),
        "d_mm": int(round(dist * 1000)),
        "cam_mm": [int(round(v * 1000)) for v in t],
        "err_cpx": int(round(err_px * 100)),
    }
    conf = 950_000 if errors == 0 else 800_000
    if side < 20:
        conf = conf * 8 // 10
    if level is not None:
        r_lu, p = level
        centre = p + r_lu @ t
        normal = r_lu @ rot[:, 2]
        lo, hi = config.height_band.get(raw.tag_id, (-1.0, 9.0))
        if not lo <= float(centre[2]) <= hi:
            return None
        out.update(
            {
                "x_mm": int(round(centre[0] * 1000)),
                "y_mm": int(round(centre[1] * 1000)),
                "z_mm": int(round(centre[2] * 1000)),
                "r_mm": int(round(math.hypot(centre[0], centre[1]) * 1000)),
                "b_mrad": int(round(math.atan2(centre[1], centre[0]) * 1000)),
                "e_mrad": int(
                    round(
                        math.atan2(centre[2] - p[2], math.hypot(centre[0] - p[0], centre[1] - p[1]))
                        * 1000
                    )
                ),
                "yaw_mrad": int(round(math.atan2(normal[1], normal[0]) * 1000)),
                # the mirror pose fits too: how far apart the two facings are (0 when not)
                "yaw_amb_mrad": _yaw_gap(level[0], rot, other[0]) if ambiguous else 0,
            }
        )
    out["conf_ppm"] = conf
    return out


def find_tags(
    luma_of: Callable[[tuple[int, int, int, int] | None, int], Any],
    camera: EyeCamera,
    pose: CameraPose | None,
    backend: TagBackend,
    *,
    level: str = "bright",
    nest_b_mrad: int | None = None,
    config: NestConfig | None = None,
) -> dict[str, Any]:
    """Read the dock's tags. ``luma_of(box, step)`` gives upright luma: the ROI ``box`` (u0, v0,
    u1, v1) at full resolution, or the whole picture (``None``) — :class:`LookView`'s ``roi_luma``
    and ``luma``."""
    config = config or NestConfig()
    if level == "black":
        return {"tags": [], "rejected": 0, "modes": []}
    max_ham = DARK_MAX_HAMMING if level == "dark" else MAX_HAMMING
    min_con = DARK_MIN_CONTRAST if level == "dark" else MIN_CONTRAST
    accepted: dict[int, dict[str, Any]] = {}
    rejected = 0
    modes: list[str] = []

    def run(box: tuple[int, int, int, int] | None, decimate: float, mode: str) -> None:
        nonlocal rejected
        import numpy as np

        gray = luma_of(box, 1)
        if gray is None or getattr(gray, "size", 0) == 0:
            return
        modes.append(mode)
        u0, v0 = (box[0], box[1]) if box is not None else (0, 0)
        for raw in backend.detect(gray, decimate):
            if raw.tag_id not in TAG_SIZES:
                continue
            checked = verify(gray, raw, max_ham, min_con)
            if checked is None:
                rejected += 1
                continue
            full = raw.corners + np.array([u0, v0], dtype=np.float64)
            res = _tag_result(raw, full, checked[0], checked[1], camera, pose, config, mode)
            if res is None:
                rejected += 1
                continue
            prev = accepted.get(raw.tag_id)
            if prev is None or res["side_px"] > prev["side_px"]:
                accepted[raw.tag_id] = res

    def roi() -> None:
        if config.use_roi and nest_b_mrad is not None:
            box = roi_box(camera, pose, int(nest_b_mrad), config.roi_half_mrad)
            if box is not None:
                run(box, config.roi_decimate, "roi")

    if config.roi_first:
        roi()
    if not accepted and config.full_frame:
        run(None, config.full_decimate, "full")
    if not accepted and not config.roi_first:
        roi()
    tags = sorted(accepted.values(), key=lambda r: r["id"])
    return {"tags": tags, "rejected": rejected, "modes": modes}


# ── direction cues ────────────────────────────────────────────────────


def _blob_direction(
    mask: Any, step: int, camera: EyeCamera, pose: CameraPose | None, whole: bool = False
) -> dict[str, Any] | None:
    """The biggest cluster of a mask on a plane (every ``step``-th upright pixel): its bearing
    and elevation (level frame with a pose, else from the optical axis), pixel count and box."""
    import numpy as np

    count = int(mask.sum())
    if count == 0:
        return None
    colsum = mask.sum(axis=0)
    peak = int(np.argmax(colsum))
    # the run of occupied columns around the peak (a gap of 2 ends it)
    lo = hi = peak
    while lo > 0 and colsum[lo - 1 : lo].sum() + (colsum[lo - 2] if lo >= 2 else 0) > 0:
        lo -= 1
    while (
        hi < mask.shape[1] - 1
        and colsum[hi + 1 : hi + 2].sum() + (colsum[hi + 2] if hi + 2 < mask.shape[1] else 0) > 0
    ):
        hi += 1
    if whole:  # every masked sample (the panel shows on both sides of the tag's white mount)
        cols_any = np.flatnonzero(colsum)
        lo, hi = int(cols_any[0]), int(cols_any[-1])
    part = mask[:, lo : hi + 1]
    n = int(part.sum())
    rows, cols = np.nonzero(part)
    u = (float(cols.mean()) + lo) * step
    v = float(rows.mean()) * step
    out: dict[str, Any] = {
        "px": n,
        "box": [
            int(lo * step),
            int(rows.min() * step),
            int((hi + 1) * step),
            int((rows.max() + 1) * step),
        ],
    }
    if pose is not None:
        b, e = camera.bearing_mrad(u, v, pose)
    else:
        right, down = camera.angles_rad(u, v)
        b, e = int(round(-right * 1000)), int(round(-down * 1000))
    out["b_mrad"], out["e_mrad"] = int(b), int(e)
    return out


def cyan_cue(planes: Any, camera: EyeCamera, pose: CameraPose | None) -> dict[str, Any] | None:
    """The dock's cyan panel as a direction (bright/dim looks only)."""
    import numpy as np

    ratio = planes.chroma_step // planes.luma_step
    y = planes.y[::ratio, ::ratio][: planes.u.shape[0], : planes.u.shape[1]].astype(np.int32)
    den = y + 24
    nu = (planes.u.astype(np.int32) - 128) * 48 // den
    nv = (planes.v.astype(np.int32) - 128) * 48 // den
    mask = (nu >= CYAN_U[0]) & (nu <= CYAN_U[1]) & (nv <= CYAN_V_MAX) & (-nv > nu)
    blob = _blob_direction(mask, planes.chroma_step, camera, pose, whole=True)
    if blob is None or blob["px"] < CYAN_MIN_PX:
        return None
    blob["kind"] = "panel"
    blob["conf_ppm"] = min(900_000, 400_000 + blob["px"] * 2_000)
    return blob


def light_cue(planes: Any, camera: EyeCamera, pose: CameraPose | None) -> dict[str, Any] | None:
    """The dock's status light as a direction in a dark picture (dark/black looks): a lit blob
    shaped like the dock's level strip (:func:`duckmind.eye.light.strip_like`) — a standby LED, a
    phone or a street lamp gives no direction to the dock."""
    y = planes.y
    if float(y.mean()) > LIGHT_MAX_MEAN:
        return None
    mask = y >= LIGHT_LUMA
    n = int(mask.sum())
    if n < LIGHT_MIN_PX or n > LIGHT_MAX_PX:
        return None
    blob = _blob_direction(mask, planes.luma_step, camera, pose)
    if blob is None or blob["px"] < LIGHT_MIN_PX:
        return None
    step = planes.luma_step
    box = blob["box"]
    if not strip_like(blob["px"], (box[2] - box[0]) // step, (box[3] - box[1]) // step):
        return None
    blob["kind"] = "light"
    blob["conf_ppm"] = 600_000
    return blob


# ── the task ─────────────────────────────────────────────────────────


class NestTask:
    """The ``nest`` task as duck-eye's server runs it: ``run(view)`` per look.

    Result (integers; ``anchor`` is the one a position may come from, ``cue`` a direction only):
    ``{ok, level, anchor: {id, x_mm, y_mm, z_mm, r_mm, b_mrad, e_mrad, yaw_mrad, conf_ppm, ham,
    side_px, d_mm, mode, …} | None, tags: [...], cue: {kind, b_mrad, e_mrad, px, conf_ppm} | None,
    rejected, modes, backend}``.
    """

    name = "nest"

    def __init__(self, backend: TagBackend | None = None, config: NestConfig | None = None) -> None:
        self.backend = backend if backend is not None else TagBackend()
        self.config = config or NestConfig()

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> NestTask:
        """``{"backend": "apriltag"|"opencv", "roi": bool, "full_decimate": float}``."""
        config = NestConfig(
            use_roi=bool(options.get("roi", True)),
            full_decimate=float(options.get("full_decimate", FULL_DECIMATE)),
        )
        return cls(TagBackend(str(options.get("backend", "apriltag"))), config)

    def digest_items(self) -> list[str]:
        c = self.config
        return [
            f"nest.backend={self.backend.name} roi={int(c.use_roi)}:{c.roi_half_mrad}"
            f" dec={c.roi_decimate},{c.full_decimate}",
            f"nest.check ham={MAX_HAMMING},{DARK_MAX_HAMMING} contrast={MIN_CONTRAST},"
            f"{DARK_MIN_CONTRAST} border={MAX_BORDER_ERRORS} reproj={REPROJ_PX}",
            f"nest.tags sizes={sorted(TAG_SIZES.items())} heights={sorted(c.height_band.items())}",
            f"nest.cues cyan={CYAN_U},{CYAN_V_MAX},{CYAN_MIN_PX} light={LIGHT_LUMA},{LIGHT_MAX_MEAN}",
        ]

    def read(
        self,
        luma_of: Callable[[tuple[int, int, int, int] | None, int], Any],
        planes: Any,
        camera: EyeCamera,
        pose: CameraPose | None,
        *,
        level: str,
        nest_b_mrad: int | None = None,
    ) -> dict[str, Any]:
        found = find_tags(
            luma_of,
            camera,
            pose,
            self.backend,
            level=level,
            nest_b_mrad=nest_b_mrad,
            config=self.config,
        )
        tags = found["tags"]
        anchor = None
        if tags:
            # the face tag anchors from afar; the plate tag is the better one up close
            by_id = {t["id"]: t for t in tags}
            anchor = by_id.get(tag36h11.NEST_TAG_ID) or by_id.get(tag36h11.EDGE_TAG_ID)
            if tag36h11.EDGE_TAG_ID in by_id and tag36h11.NEST_TAG_ID in by_id:
                if by_id[tag36h11.EDGE_TAG_ID]["side_px"] >= 40:
                    anchor = by_id[tag36h11.EDGE_TAG_ID]
        cue = None
        if level in ("bright", "dim"):
            cue = cyan_cue(planes, camera, pose)
        elif level in ("dark", "black"):
            cue = light_cue(planes, camera, pose)
        return {
            "ok": 1,
            "level": level,
            "anchor": anchor,
            "tags": tags,
            "cue": cue,
            "rejected": found["rejected"],
            "modes": found["modes"],
            "backend": self.backend.name,
        }

    def run(self, view: Any) -> dict[str, Any]:
        """One look (:class:`duckmind.eye.frames.LookView`)."""
        level = str(view.light.get("level", "bright")) if view.light else "bright"

        def luma_of(box: tuple[int, int, int, int] | None, step: int) -> Any:
            return view.luma(step) if box is None else view.roi_luma(box, step=step)

        hint = view.look.hints.get("nest_b_mrad") if view.look.hints else None
        return self.read(
            luma_of,
            view.planes,
            view.camera,
            view.camera_pose,
            level=level,
            nest_b_mrad=int(hint) if isinstance(hint, int) else None,
        )


def sighting(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """What duck-body folds into ``sense.sighting`` for the dock (percept-v1 fields): the anchor
    with ``r_src`` ``tag``, else the cue with ``r_src`` ``cue`` and no range (owner C5)."""
    anchor = result.get("anchor")
    if anchor and "b_mrad" in anchor:
        return {
            "what": "nest",
            "id": f"{tag36h11.FAMILY}:{anchor['id']}",
            "b_mrad": anchor["b_mrad"],
            "e_mrad": anchor["e_mrad"],
            "r_mm": anchor["r_mm"],
            "r_src": "tag",
            "yaw_mrad": anchor["yaw_mrad"]
            if anchor.get("yaw_amb_mrad", 0) <= YAW_TRUST_MRAD
            else None,
            "conf_ppm": anchor["conf_ppm"],
        }
    cue = result.get("cue")
    if cue:
        return {
            "what": "nest",
            "id": None,
            "b_mrad": cue["b_mrad"],
            "e_mrad": cue["e_mrad"],
            "r_mm": None,
            "r_src": "cue",
            "yaw_mrad": None,
            "conf_ppm": cue["conf_ppm"],
        }
    return None
