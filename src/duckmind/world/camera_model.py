"""The duck's head camera in the simulator, full version (spec §11.2; phase 2).

What the real Sony IMX219 behind the ~3.05 mm M12 lens gives the duck, applied at load time to
our own copy of the scene (no official file is edited, nothing is copied):

* **Geometry**: the family calibration K (``robotd-params`` ``CameraIntrinsics::alpha()``,
  1280×720 as mediad sends it, unrotated) instead of MuJoCo's default 45° field. The official
  180° roll of ``head_camera`` is kept (``sim/camera.py``: the model's camera faces backwards),
  so a rendered frame is on its side exactly like the real mount, and :func:`upright` turns it a
  quarter turn clockwise like ``mediad --rotate 90`` / ``uyvy::Turn::Right``. MuJoCo's
  principal-point sign is the opposite of OpenCV's on both axes (measured:
  :func:`mujoco_principal_pixel`; P5 checks it with a checkerboard render).
* **Lens distortion** (owner decision F1): the family's k1…k3, p1, p2 (:data:`FAMILY_DISTORTION`,
  OpenCV's model). MuJoCo renders a wider pinhole *canvas* holding every ray of the frame
  (:class:`Lens`; 1611×910 for 1280×720), and a precomputed index remaps it to the delivered
  frame (nearest by default, 1.9 ms at 720p; bilinear on request). With the barrel distortion
  the frame reaches 32.3° and 37.5° either side of the axis along its 1280-px side (about 70°,
  against 62° for the pinhole), so the upright duck sees the floor from about 0.32 m instead of
  0.39 m. The polynomial is an extrapolation at the frame's edges until a board is measured
  (phase 8). Segmentation renders go through the same remap, always nearest.
* **Resolution** follows ``[media] quality`` (:data:`QUALITIES`: 1080p30, 720p30 — the default —,
  720p15, 360p30; ``robotd-params`` ``Quality``), K and the canvas scaled uniformly with the frame
  as mediad scales a solve.
* **Near clip** 5 mm absolute (MuJoCo's ``znear`` is relative to the model's extent, so the
  official twin cuts away everything within 10–22 cm of the lens).
* **No torch on the head**: the headlight's diffuse and specular are off; its ambient carries
  only the room's bounced light (:func:`set_lights`).
* **A ceiling** (the official scenes are open to the sky), and the lights of
  :mod:`duckmind.world.lights` — the one light state the page also draws — in seven MuJoCo light
  slots (the classic renderer lights only the first 7 active lights on this Mac; a home's rooms
  are chosen by :mod:`duckmind.world.render_scene`).
* **Light that glows** (the nest's backlight): an emitter has an absolute brightness, so it is
  drawn in a second, emission-only pass and added to the scene in lux after the room light is
  scaled back (:meth:`HeadCameraRenderer.render_emission`) — otherwise the normalised render would
  make it brighter by day than by night.
* **Auto-exposure**: mediad's software loop (``mediad/src/exposure.rs``) ported line by line
  (:class:`Ae`, :func:`mean_luma`, :class:`AeLoop`), stepped twice a second on a 160×90 metering
  render. Its unit tests are P3 (``tests/test_ae.py``).
* **White balance** (:class:`Awb`): standing in for the rkaiq 3A engine's, stepped with the
  metering render: it moves slowly towards the room light's colour and removes most, not all,
  of a lamp's or the moon's cast (a lamp-lit room stays a bit warm).
* **Sensor**: IMX219-like, the survey's model (``cam_survey.py``): shot noise from a 4000 e⁻ full
  well (a knob, :attr:`CameraConfig.full_well_scale`, kept at phase 0's value by default: the
  scaled 3280→1920→1280 readout may have about twice that, but the light meter's σ thresholds
  were derived from this model), 2.5 e⁻ read noise, both amplified by the gain, RAW10
  quantisation before the ISP's digital gain (visible as banding at high digital gain), white
  balance, the ISP's light denoise above 4× (stronger with the gain, :func:`denoise_strength`),
  gamma 2.2. The absolute scale is one assumption: an 18 % grey card at 400 lux converges at
  budget 1125 (11.4 ms × 1.9×) — to be measured once on a board (spec phase 8). Deterministic for
  a seed. Vignetting is a knob, off by default (the rkaiq lens-shading correction removes most
  of it).
* **Delivered format**: packed UYVY, BT.601 with the chroma averaged across each pixel pair,
  byte for byte as the official ``sim/camera.py`` packs it (:func:`to_uyvy`).

Deliberately not modelled: rolling shutter and motion blur. The duck looks only while still
(head joints below 0.2 rad/s and gyro below 0.3 rad/s for 150 ms, spec §4.2), which bounds the
skew and blur of an 11–23 ms exposure to about 4–5 px; modelling them would mean rendering
several sub-frames per frame for an effect the look rules already keep small.

The parts that need numpy or MuJoCo import them when called, so the exposure port runs (and is
tested) without either.
"""

from __future__ import annotations

import math
import os
import platform
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duckmind.world import lights as world_lights

if TYPE_CHECKING:
    from duckmind.world.lights import LightSource, LightState

# ── geometry: the family calibration ──────────────────────────────────────


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole K in pixels, OpenCV conventions (pixel centres at integers, y down)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, width: int, height: int) -> Intrinsics:
        """For a uniformly resized frame, the way ``mediad/src/camera.rs`` scales a solve."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    def upright(self) -> Intrinsics:
        """K of the picture turned a quarter turn clockwise (``mediad --rotate 90``).

        ``uyvy::Turn::Right`` sends source (x, y) to upright (h-1-y, x): the axes swap, and the
        principal point's source y becomes the upright horizontal, counted from the other side.
        """
        return Intrinsics(
            self.height, self.width, self.fy, self.fx, self.height - 1 - self.cy, self.cx
        )

    def fov_deg(self) -> dict[str, float]:
        """How far the picture reaches on each side of the optical axis, in degrees (pinhole)."""
        left = math.degrees(math.atan((self.cx + 0.5) / self.fx))
        right = math.degrees(math.atan((self.width - self.cx - 0.5) / self.fx))
        top = math.degrees(math.atan((self.cy + 0.5) / self.fy))
        bottom = math.degrees(math.atan((self.height - self.cy - 0.5) / self.fy))
        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "horizontal": left + right,
            "vertical": top + bottom,
        }

    def bearing_deg(self, u: float, v: float) -> tuple[float, float]:
        """Angles of pixel (u, v) from the optical axis: (to the right, downwards), degrees
        (pinhole: use :func:`undistort_pixel` first on a distorted frame)."""
        return (
            math.degrees(math.atan((u - self.cx) / self.fx)),
            math.degrees(math.atan((v - self.cy) / self.fy)),
        )

    def project(self, right: float, down: float, depth: float) -> tuple[float, float]:
        """Where a point (camera axes: right, down, along the axis) lands, in pixels (pinhole)."""
        return self.cx + self.fx * right / depth, self.cy + self.fy * down / depth


# robotd-params CameraIntrinsics::alpha(): solved on unit graphite, 2026-09-08, ChArUco, 80
# views, 0.79 px RMS, on the 1280×720 frame as mediad sends it (unrotated).
FAMILY = Intrinsics(
    width=1280,
    height=720,
    fx=1061.8060025020175,
    fy=1062.193713957031,
    cx=596.7776848217006,
    cy=474.52348043048636,
)
# k1 k2 p1 p2 k3 (OpenCV order), of the unrotated frame; applied by the render (Lens).
FAMILY_DISTORTION = (
    -0.34695388691888,
    0.14615866153945595,
    -0.0007449157712424215,
    -0.0022167170528743837,
    -0.016437266141048814,
)
ROTATE_DEG = 90  # mediad --rotate default: clockwise degrees from the frame to upright
CAMERA = "head_camera"
FRAME_SIZE = (1280, 720)  # [media] quality 720p30, as the real duck delivers it
METER_SIZE = (160, 90)  # the auto-exposure's metering render
NEAR_CLIP_M = 0.005
FAR_CLIP_M = 5000.0  # far enough that the endless floor meets the ceiling at the horizon
CEILING_M = world_lights.DEFAULT_CEILING_M


def mujoco_principal_pixel(k: Intrinsics) -> tuple[float, float]:
    """MuJoCo ``principalpixel`` that puts K's principal point where K says.

    Measured on MuJoCo 3.10 (a marker on the optical axis): ``principalpixel`` (px, py) moves the
    axis to image (W/2 − px, H/2 − py), x right and y down with pixel edges at integers — the
    opposite sign of OpenCV's (cx − W/2, cy − H/2) on both axes. OpenCV's cx is a pixel centre,
    hence the half pixel.
    """
    return k.width / 2 - (k.cx + 0.5), k.height / 2 - (k.cy + 0.5)


def upright(image: Any) -> Any:
    """A frame as the duck sees it: a quarter turn clockwise (``np.rot90(k=-1)``)."""
    import numpy as np

    return np.ascontiguousarray(np.rot90(image, k=-1))


# ── [media] quality: the frame sizes a duck can deliver ──────────────────────


@dataclass(frozen=True)
class Quality:
    """One rung of ``robotd-params`` ``Quality``: every rung is 16:9, a multiple of 8."""

    label: str
    width: int
    height: int
    fps: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def frame_period_s(self) -> float:
        """How often the real camera captures (what "the next frame" waits for at most)."""
        return 1.0 / self.fps

    @property
    def k(self) -> Intrinsics:
        """The family K scaled to this frame, as mediad scales a solve."""
        return FAMILY.scaled(self.width, self.height)

    @property
    def frame_bytes(self) -> int:
        """UYVY: two bytes a pixel."""
        return self.width * self.height * 2


QUALITIES = {
    q.label: q
    for q in (
        Quality("1080p30", 1920, 1080, 30),
        Quality("720p30", 1280, 720, 30),
        Quality("720p15", 1280, 720, 15),
        Quality("360p30", 640, 360, 30),
    )
}
DEFAULT_QUALITY = "720p30"  # robotd-params Quality::default()


def quality_rung(label: str | None = None) -> Quality:
    """The rung called ``label`` (``None``: the default, 720p30)."""
    name = DEFAULT_QUALITY if label is None else str(label).strip()
    if name not in QUALITIES:
        raise ValueError(f"unknown [media] quality {name!r}: the duck has {', '.join(QUALITIES)}")
    return QUALITIES[name]


def quality_from_config(config: str | Mapping[str, Any] | None) -> Quality:
    """``[media] quality`` from robotd.toml (its text, or the parsed table); missing: 720p30."""
    if config is None:
        return quality_rung()
    table = tomllib.loads(config) if isinstance(config, str) else config
    media = table.get("media") or {}
    return quality_rung(media.get("quality") if isinstance(media, Mapping) else None)


# ── the lens: OpenCV's distortion model with the family coefficients ─────────


def distort(x: Any, y: Any, dist: Sequence[float] = FAMILY_DISTORTION) -> tuple[Any, Any]:
    """Ideal (pinhole) normalised coordinates → where the lens puts them (OpenCV's model).

    Works on floats and on numpy arrays alike."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def undistort(
    xd: Any, yd: Any, dist: Sequence[float] = FAMILY_DISTORTION, iterations: int = 8
) -> tuple[Any, Any]:
    """Distorted normalised coordinates → ideal ones: Newton's method on :func:`distort`
    (converges to 1e-12 in a few steps over the whole frame; floats or numpy arrays)."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        slope = k1 + r2 * (2 * k2 + r2 * 3 * k3)  # d(radial)/d(r²)
        ex = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        ey = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        a = radial + 2 * x * x * slope + 2 * p1 * y + 6 * p2 * x
        b = 2 * x * y * slope + 2 * p1 * x + 2 * p2 * y
        c = 2 * x * y * slope + 2 * p1 * x + 2 * p2 * y
        d = radial + 2 * y * y * slope + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * c
        x = x - (d * ex - b * ey) / det
        y = y - (a * ey - c * ex) / det
    return x, y


def undistort_pixel(
    u: float, v: float, k: Intrinsics = FAMILY, dist: Sequence[float] = FAMILY_DISTORTION
) -> tuple[float, float]:
    """A pixel of the delivered (distorted) frame → its ray's normalised (right, down)."""
    x, y = undistort((u - k.cx) / k.fx, (v - k.cy) / k.fy, dist)
    return float(x), float(y)


def distort_point(
    right: float,
    down: float,
    depth: float,
    k: Intrinsics = FAMILY,
    dist: Sequence[float] = FAMILY_DISTORTION,
) -> tuple[float, float]:
    """Where a point (camera axes) lands in the delivered, distorted frame, in pixels."""
    xd, yd = distort(right / depth, down / depth, dist)
    return float(k.cx + k.fx * xd), float(k.cy + k.fy * yd)


def bearing_deg(
    u: float, v: float, k: Intrinsics = FAMILY, dist: Sequence[float] = FAMILY_DISTORTION
) -> tuple[float, float]:
    """Angles of a delivered pixel from the optical axis (right, down), distortion undone."""
    x, y = undistort_pixel(u, v, k, dist)
    return math.degrees(math.atan(x)), math.degrees(math.atan(y))


def upright_distortion(dist: Sequence[float] = FAMILY_DISTORTION) -> tuple[float, ...]:
    """The coefficients of the upright picture (``uyvy::Turn::Right``): the radial terms stay,
    the tangential ones swap (p1′ = p2, p2′ = −p1) because (x, y) becomes (−y, x)."""
    k1, k2, p1, p2, k3 = (float(v) for v in dist)
    return (k1, k2, p2, -p1, k3)


class Lens:
    """The family lens on one rung's frame, and the pinhole canvas MuJoCo renders for it.

    The canvas has the frame's fx and fy, and reaches just past every ray of the frame
    (``extent``: normalised left, right, top, bottom at the canvas's pixel edges); a render of
    it at any uniformly scaled size maps to the delivered frame through :meth:`remap`.
    """

    def __init__(
        self,
        k: Intrinsics = FAMILY,
        dist: Sequence[float] = FAMILY_DISTORTION,
        *,
        margin_px: float = 2.0,
    ) -> None:
        self.k = k
        self.dist = tuple(float(v) for v in dist)
        xs: list[float] = []
        ys: list[float] = []
        # The frame's outline at its pixel edges; with barrel distortion the corners reach
        # furthest, so every pixel of the outline is taken.
        edge_u = [min(max(float(i), -0.5), k.width - 0.5) for i in range(-1, k.width + 1)]
        edge_v = [min(max(float(i), -0.5), k.height - 0.5) for i in range(-1, k.height + 1)]
        outline = [(u, v) for u in edge_u for v in (-0.5, k.height - 0.5)]
        outline += [(u, v) for v in edge_v for u in (-0.5, k.width - 0.5)]
        for u, v in outline:
            x, y = undistort_pixel(u, v, k, self.dist)
            xs.append(x)
            ys.append(y)
        x0 = min(xs) - margin_px / k.fx
        y0 = min(ys) - margin_px / k.fy
        width = math.ceil((max(xs) + margin_px / k.fx - x0) * k.fx)
        height = math.ceil((max(ys) + margin_px / k.fy - y0) * k.fy)
        self.extent = (x0, x0 + width / k.fx, y0, y0 + height / k.fy)
        self.canvas = Intrinsics(width, height, k.fx, k.fy, -x0 * k.fx - 0.5, -y0 * k.fy - 0.5)
        self._maps: dict[tuple[int, int, bool], Any] = {}
        self._rays: dict[tuple[int, int], tuple[Any, Any]] = {}

    def fov_deg(self) -> dict[str, float]:
        """How far the delivered frame reaches, distortion included (degrees): each side along
        the rows and columns through the principal point, and the canvas's full span."""
        k = self.k
        left, _ = undistort_pixel(-0.5, k.cy, k, self.dist)
        right, _ = undistort_pixel(k.width - 0.5, k.cy, k, self.dist)
        _, top = undistort_pixel(k.cx, -0.5, k, self.dist)
        _, bottom = undistort_pixel(k.cx, k.height - 0.5, k, self.dist)
        deg = [math.degrees(math.atan(abs(v))) for v in (left, right, top, bottom)]
        return {
            "left": deg[0],
            "right": deg[1],
            "top": deg[2],
            "bottom": deg[3],
            "horizontal": deg[0] + deg[1],
            "vertical": deg[2] + deg[3],
        }

    def canvas_size(self, width: int, height: int) -> tuple[int, int]:
        """The canvas to render for a delivered frame of ``width``×``height``."""
        s = width / self.k.width
        return (
            max(2, round(self.canvas.width * s)),
            max(2, round(self.canvas.height * height / self.k.height)),
        )

    def rays(self, width: int, height: int) -> tuple[Any, Any]:
        """Each delivered pixel's ideal normalised (right, down), as two (height, width) arrays."""
        key = (width, height)
        if key not in self._rays:
            import numpy as np

            k = self.k.scaled(width, height)
            u = (np.arange(width, dtype=np.float64) - k.cx) / k.fx
            v = (np.arange(height, dtype=np.float64) - k.cy) / k.fy
            xd, yd = np.meshgrid(u, v)
            self._rays[key] = undistort(xd, yd, self.dist)
        return self._rays[key]

    def remap(self, width: int, height: int, *, bilinear: bool = False) -> Any:
        """Indices into a canvas render as ``mjr_readPixels`` returns it (rows bottom-up, flat
        (h·w, 3)): nearest, one flat index per delivered pixel; bilinear, (indices (4, n),
        weights (4, n))."""
        key = (width, height, bilinear)
        if key in self._maps:
            return self._maps[key]
        import numpy as np

        cw, ch = self.canvas_size(width, height)
        x, y = self.rays(width, height)
        x0, x1, y0, y1 = self.extent
        col = (x - x0) / (x1 - x0) * cw - 0.5
        row = (y - y0) / (y1 - y0) * ch - 0.5
        if not bilinear:
            c = np.clip(np.rint(col), 0, cw - 1).astype(np.int64)
            r = np.clip(np.rint(row), 0, ch - 1).astype(np.int64)
            index = ((ch - 1 - r) * cw + c).ravel()
            self._maps[key] = index
            return index
        c0 = np.clip(np.floor(col), 0, cw - 2).astype(np.int64)
        r0 = np.clip(np.floor(row), 0, ch - 2).astype(np.int64)
        fc = np.clip(col - c0, 0.0, 1.0).astype(np.float32).ravel()
        fr = np.clip(row - r0, 0.0, 1.0).astype(np.float32).ravel()
        c0, r0 = c0.ravel(), r0.ravel()
        idx = np.stack(
            [
                (ch - 1 - r0) * cw + c0,
                (ch - 1 - r0) * cw + c0 + 1,
                (ch - 2 - r0) * cw + c0,
                (ch - 2 - r0) * cw + c0 + 1,
            ]
        )
        weights = np.stack([(1 - fc) * (1 - fr), fc * (1 - fr), (1 - fc) * fr, fc * fr])
        self._maps[key] = (idx, weights)
        return self._maps[key]

    def apply(self, canvas: Any, width: int, height: int, *, bilinear: bool = False) -> Any:
        """A canvas render (``mjr_readPixels`` order, (ch, cw, c)) → the delivered frame, row 0
        at the top."""
        import numpy as np

        flat = canvas.reshape(-1, canvas.shape[-1])
        if not bilinear:
            return flat[self.remap(width, height)].reshape(height, width, -1)
        idx, weights = self.remap(width, height, bilinear=True)
        acc = (flat[idx].astype(np.float32) * weights[..., None]).sum(axis=0)
        out = np.rint(acc) if canvas.dtype == np.uint8 else acc
        return out.astype(canvas.dtype).reshape(height, width, -1)

    def cos4(self, width: int, height: int) -> Any:
        """cos⁴ of each delivered pixel's angle off the axis (for vignetting)."""
        x, y = self.rays(width, height)
        return (1.0 / (1.0 + x * x + y * y)) ** 2


# ── mediad's auto-exposure (mediad/src/exposure.rs), line by line ──────────────

INTERVAL_S = 0.5  # INTERVAL: twice a second
REASSERT_TICKS = 20  # quiet ticks before the values are written again anyway
REPORT_TICKS = 20
TARGET_Y = 90.0  # mean-luma setpoint, 8-bit, after the ISP's gamma
DEADBAND = 0.12
SOFT_LINES = 600.0  # ≈ 11.4 ms
HARD_LINES = 1200.0  # ≈ 22.9 ms
MAX_ANALOGUE = 11.0
MAX_DIGITAL = 16.0
CAPTURE_FORMAT = "UYVY"
LINE_S = 19.05e-6  # one sensor line in the pinned 1920×1080 mode


def _clamp(x: float, lo: float, hi: float) -> float:
    """``f64::clamp``."""
    return lo if x < lo else hi if x > hi else x


def _u32(x: float) -> int:
    """``as u32``: truncation toward zero, saturating."""
    if x != x or x <= 0:
        return 0
    return min(int(x), 0xFFFFFFFF)


@dataclass(frozen=True)
class Controls:
    """What the loop writes to the sensor, in the sensor's own units."""

    exposure: int  # lines
    analogue_gain: int  # 256 is 1x
    gain: int  # ISP digital gain, 256 is 1x


@dataclass
class Ae:
    """The loop's state: one brightness budget, split across three controls."""

    exposure: float
    analogue: float
    digital: float
    written: Controls | None = None

    @classmethod
    def starting_at(cls, exposure_lines: int, analogue_gain_reg: int) -> Ae:
        return cls(
            exposure=_clamp(float(exposure_lines), 4.0, HARD_LINES),
            analogue=_clamp(analogue_gain_reg / 256.0, 1.0, MAX_ANALOGUE),
            digital=1.0,
            written=None,
        )

    def step(self, mean_luma: float) -> Controls | None:
        """One step towards the setpoint; ``None`` inside the deadband or when unchanged."""
        luma = mean_luma if mean_luma >= 1.0 else 1.0  # f64::max, NaN included
        ratio = _clamp(TARGET_Y / luma, 0.25, 4.0)
        if abs(1.0 - ratio) < DEADBAND:
            return None

        budget = _clamp(
            self.exposure * self.analogue * self.digital * ratio**0.6,
            4.0,
            HARD_LINES * MAX_ANALOGUE * MAX_DIGITAL,
        )

        self.exposure = min(budget, SOFT_LINES)
        self.analogue = _clamp(budget / self.exposure, 1.0, MAX_ANALOGUE)
        self.exposure = _clamp(budget / self.analogue, self.exposure, HARD_LINES)
        self.digital = _clamp(budget / (self.exposure * self.analogue), 1.0, MAX_DIGITAL)

        after = self.controls()
        if self.written == after:
            return None
        self.written = after
        return after

    def current(self) -> Controls:
        """The values as they stand, for the re-assert; they count as written."""
        controls = self.controls()
        self.written = controls
        return controls

    def controls(self) -> Controls:
        return Controls(
            exposure=_u32(self.exposure),
            analogue_gain=_u32(self.analogue * 256.0),
            gain=_u32(self.digital * 256.0),
        )


def mean_luma(data: bytes, fmt: str = CAPTURE_FORMAT) -> float | None:
    """Mean luma of one UYVY frame: every eighth pixel (every 16th byte from index 1).

    ``None`` for any other format rather than a wrong answer.
    """
    if fmt != CAPTURE_FORMAT:
        return None
    samples = data[1::16]
    return sum(samples) / len(samples) if samples else None


@dataclass(frozen=True)
class Digital:
    """What this run knows about the ISP's digital gain: unknown, told(value) or absent."""

    kind: str
    value: int = 0

    def forget(self) -> Digital:
        """The heartbeat drops what this run believes — except that the control is absent."""
        return self if self.kind == "absent" else Digital("unknown")


UNKNOWN = Digital("unknown")
ABSENT = Digital("absent")


def told(value: int) -> Digital:
    return Digital("told", value)


def worth_writing_gain(gain: int, digital: Digital) -> bool:
    if digital.kind == "absent":
        return False
    if digital.kind == "unknown":
        return True
    return gain > 256 or digital.value != gain


def worth_reading_back(proven: bool, asked: int, pinned: int) -> bool:
    return not proven and asked != pinned


class AeLoop:
    """The auto-exposure thread's tick (``exposure::spawn``) against a simulated sensor.

    The sensor holds what was last written — the pinned start values until the first write, and
    digital gain 1x because the pipeline pins none — and every write lands (the simulated node
    has all three controls). Logging and the read-back check have nothing to act on here.
    """

    def __init__(self, exposure_lines: int = 600, analogue_gain_reg: int = 1024) -> None:
        self.ae = Ae.starting_at(exposure_lines, analogue_gain_reg)
        self.sensor = Controls(exposure_lines, analogue_gain_reg, 256)
        self.quiet = 0
        self.ticks = 0
        self.writes = 0
        self.last_mean: float | None = None
        self.last_step: Controls | None = None

    def tick(self, mean: float | None) -> Controls | None:
        """One 500 ms tick with the metered mean luma (``None``: no frame yet)."""
        self.ticks += 1
        if mean is None:
            return None
        self.last_mean = mean
        stepped = self.ae.step(mean)
        if stepped is None:
            if self.quiet >= REASSERT_TICKS:
                stepped = self.ae.current()
            else:
                self.quiet += 1
                self.last_step = None
                return None
        self.quiet = 0
        self.sensor = stepped
        self.writes += 1
        self.last_step = stepped
        return stepped

    def settled(self) -> bool:
        """Inside the deadband, or pinned at a ceiling it cannot leave."""
        return self.last_mean is not None and self.last_step is None


# ── the sensor (IMX219-like; the survey's assumptions) ──────────────────────

LIN_TARGET = (TARGET_Y / 255.0) ** 2.2
# Sensor linear output (0..1 of full scale) per (lux × budget) for a white surface: an 18 %
# grey card at 400 lux reaches the AE target at budget 1125. An assumption until measured.
KAPPA = LIN_TARGET / (0.18 * 400.0 * 1125.0)
FULL_WELL_E = 4000.0  # electrons at gain 1 (1.12 µm pixel)
READ_E = 2.5
DENOISE_FROM_GAIN = 4.0  # the ISP's noise reduction above this total gain
# Its strength: the survey's 0.5, rising with the gain to the survey's 0.8. The survey jumped
# from 0.5 to 0.8 at 32×, right where a dusk room settles (30–34× total gain), so whether dusk
# read "dim" or "bright" from the noise alone hung on a few % of AE deadband; an ISP interpolates
# its denoise strength over the gain (rkaiq's tables do), so here it ramps from 24× to 64×.
DENOISE_RAMP = (24.0, 64.0, 0.5, 0.8)  # gain where it starts and ends rising, strength there


def denoise_strength(gain: float) -> float:
    """The share of the 3×3 smoothing the ISP mixes in at this total gain (0 up to 4×)."""
    if gain <= DENOISE_FROM_GAIN:
        return 0.0
    g0, g1, k0, k1 = DENOISE_RAMP
    if gain <= g0:
        return k0
    if gain >= g1:
        return k1
    return k0 + (k1 - k0) * math.log(gain / g0) / math.log(g1 / g0)


RAW_BITS = 10  # the IMX219's RAW10 readout: quantised before the ISP's digital gain
# RAW10 codes sit on a black-level pedestal (64 on the IMX219), so the noise below black is kept
# through the ADC and the ISP's denoise averages it out before anything is clipped at 0.
BLACK_LEVEL = 64


def budget(controls: Controls) -> float:
    return controls.exposure * (controls.analogue_gain / 256.0) * (controls.gain / 256.0)


def total_gain(controls: Controls) -> float:
    return (controls.analogue_gain / 256.0) * (controls.gain / 256.0)


def shutter_ms(controls: Controls) -> float:
    return controls.exposure * LINE_S * 1000.0


def at_ceiling(controls: Controls) -> bool:
    return controls.exposure >= HARD_LINES and controls.analogue_gain >= MAX_ANALOGUE * 256.0


def exposure_summary(controls: Controls) -> dict[str, Any]:
    return {
        "exposure_lines": controls.exposure,
        "shutter_ms": round(shutter_ms(controls), 2),
        "analogue": round(controls.analogue_gain / 256.0, 3),
        "digital": round(controls.gain / 256.0, 3),
        "budget": round(budget(controls), 1),
        "at_ceiling": at_ceiling(controls),
    }


_GAMMA_LUT: Any = None


def _gamma_lut() -> Any:
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        import numpy as np

        levels = np.arange(65536, dtype=np.float64) / 65535.0
        _GAMMA_LUT = np.round(255.0 * levels ** (1 / 2.2)).astype(np.uint8)
    return _GAMMA_LUT


def _box3(a: Any) -> Any:
    """3×3 mean with edge padding, separable (the ISP's light spatial smoothing)."""
    import numpy as np

    pad = ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode="edge")
    rows = p[:-2] + p[1:-1] + p[2:]
    return (rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]) * np.float32(1 / 9)


def expose(
    scene_lux: Any,
    controls: Controls,
    rng: Any = None,
    *,
    wb: Sequence[float] | None = None,
    full_well: float = FULL_WELL_E,
    raw_bits: int | None = RAW_BITS,
    vignette: Any = None,
) -> Any:
    """Scene (lux × albedo per channel, float32) → the 8-bit RGB the ISP delivers.

    The survey's ``sensor()``: the Poisson shot noise is drawn as a Gaussian of the same variance
    (one draw per sample, several times faster, same σ). Then, in the sensor's order: the RAW10
    code (``raw_bits``; quantised after the analogue gain, before the ISP's digital gain, on a
    black-level pedestal — at 16× digital gain one code is 16/959 of full scale), the
    white-balance gains ``wb`` (R, G, B),
    the ISP's light denoise above 4× gain and gamma 2.2. ``vignette`` is a per-pixel gain map
    (off unless given); ``full_well`` the electrons at gain 1 that set the shot noise.
    """
    import numpy as np

    lin = scene_lux * np.float32(budget(controls) * KAPPA)
    if vignette is not None:
        lin = lin * vignette.astype(np.float32)[..., None]
    gain = total_gain(controls)
    if rng is not None:
        electrons = np.clip(lin, 0.0, 1.0) * np.float32(full_well / gain)
        sigma = np.sqrt(electrons + np.float32(READ_E * READ_E))
        noisy = electrons + sigma * rng.standard_normal(lin.shape, dtype=np.float32)
        lin = noisy * np.float32(gain / full_well)
    if raw_bits:
        black = BLACK_LEVEL * 2 ** (raw_bits - 10)
        span = float(2**raw_bits - 1 - black)  # codes from black to the ADC's top
        digital = controls.gain / 256.0
        codes = np.rint(np.clip(lin * np.float32(span / digital), -black, span))
        lin = codes * np.float32(digital / span)
    if wb is not None:
        lin = lin * np.asarray(wb, dtype=np.float32)
    if rng is not None and gain > DENOISE_FROM_GAIN:
        k = np.float32(denoise_strength(gain))
        lin = (1 - k) * lin + k * _box3(lin)
    index = (np.clip(lin, 0.0, 1.0) * np.float32(65535.0) + np.float32(0.5)).astype(np.uint16)
    out: Any = _gamma_lut()[index]
    return out


def luma(rgb: Any) -> Any:
    """BT.601 luma, as ``to_uyvy`` and ``duck_detect`` compute it (float32)."""
    import numpy as np

    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ── white balance (the rkaiq 3A engine's job on the robot) ─────────────────────

AWB_RATE = 0.35  # of the way to the estimate per metering tick (2 Hz): about 3 s to settle
AWB_STRENGTH = 0.85  # how much of a colour cast it removes: a lamp-lit room stays a bit warm
AWB_LIMITS = (0.4, 3.0)  # red and blue gains (green stays 1)
AWB_BRIGHT_SHARE = 0.25  # white patch: the brightest quarter of the pixels


class Awb:
    """Slow auto white balance standing in for the rkaiq 3A engine's.

    A tuned AWB finds the colour of an ordinary room's light; so this one is told that colour
    (the light rig's lux-weighted mix at the reference spot, :attr:`LightRig.color`) and removes
    :data:`AWB_STRENGTH` of it, moving :data:`AWB_RATE` of the way each metering tick. Without it
    it falls back to a white patch (the brightest quarter of the pixels). Its failure modes on a
    scene of one colour are not modelled.
    """

    def __init__(self, gains: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        self.gains = (float(gains[0]), float(gains[1]), float(gains[2]))
        self.ticks = 0
        self.last_target: tuple[float, float, float] | None = None

    @staticmethod
    def target_for(color: Sequence[float]) -> tuple[float, float, float] | None:
        """The gains that take :data:`AWB_STRENGTH` of the colour cast of light ``color``."""
        r, g, b = (float(v) for v in list(color)[:3])
        if min(r, g, b) <= 0:
            return None
        lo, hi = AWB_LIMITS
        return (
            _clamp((g / r) ** AWB_STRENGTH, lo, hi),
            1.0,
            _clamp((g / b) ** AWB_STRENGTH, lo, hi),
        )

    def target_from_scene(self, scene: Any) -> tuple[float, float, float] | None:
        """White patch: the light's colour as the brightest quarter of ``scene`` shows it."""
        import numpy as np

        flat = scene.reshape(-1, 3).astype(np.float64)
        if len(flat) == 0:
            return None
        count = max(1, int(len(flat) * AWB_BRIGHT_SHARE))
        level = flat.sum(axis=1)
        top = flat[np.argpartition(-level, count - 1)[:count]]
        return self.target_for(top.mean(axis=0))

    def update(
        self, scene: Any = None, *, illuminant: Sequence[float] | None = None
    ) -> tuple[float, float, float]:
        """One tick: towards the light's colour (``illuminant``), else the metering frame's
        white patch (``scene``: lux × albedo, linear RGB)."""
        self.ticks += 1
        if illuminant is not None:
            target = self.target_for(illuminant)
        elif scene is not None:
            target = self.target_from_scene(scene)
        else:
            target = None
        if target is not None:
            self.last_target = target
            r = math.exp(math.log(self.gains[0]) * (1 - AWB_RATE) + math.log(target[0]) * AWB_RATE)
            b = math.exp(math.log(self.gains[2]) * (1 - AWB_RATE) + math.log(target[2]) * AWB_RATE)
            self.gains = (r, 1.0, b)
        return self.gains

    def settled(self, tolerance: float = 0.03) -> bool:
        if self.last_target is None:
            return False
        return all(
            abs(math.log(g / t)) < tolerance
            for g, t in zip(self.gains, self.last_target, strict=True)
        )


# ── the delivered format: UYVY (official sim/camera.py, byte for byte) ────────

_UYVY_Y = (0.299, 0.587, 0.114)
_UYVY_U = (-0.168736, -0.331264, 0.5)
_UYVY_V = (0.5, -0.418688, -0.081312)


def to_uyvy(rgb: Any) -> bytes:
    """RGB (height, width, 3) → packed UYVY, ``U Y0 V Y1`` per pixel pair, BT.601, the chroma
    averaged across the pair (not dropped): the official ``sim/camera.py`` ``to_uyvy``, which
    also truncates to 8 bits rather than rounding."""
    import numpy as np

    frame = rgb.astype(np.float32)
    y = frame @ np.array(_UYVY_Y)
    u = frame @ np.array(_UYVY_U) + 128.0
    v = frame @ np.array(_UYVY_V) + 128.0
    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((u[:, 0::2] + u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(y[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((v[:, 0::2] + v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(y[:, 1::2], 0, 255)
    return bytes(packed.tobytes())


def uyvy_to_rgb(data: bytes, width: int, height: int) -> Any:
    """Packed UYVY → RGB (height, width, 3) uint8, BT.601 full range (for checks and pictures;
    the duck's eye samples UYVY itself)."""
    import numpy as np

    packed = np.frombuffer(data, dtype=np.uint8).reshape(height, width // 2, 4).astype(np.float32)
    y = np.empty((height, width), dtype=np.float32)
    y[:, 0::2] = packed[:, :, 1]
    y[:, 1::2] = packed[:, :, 3]
    u = np.repeat(packed[:, :, 0], 2, axis=1) - 128.0
    v = np.repeat(packed[:, :, 2], 2, axis=1) - 128.0
    rgb = np.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ── the scene: our own copy of the live body's world, set up at load time ────────

RENDER_WHITE = 0.5  # render value of a white level surface at the reference spot
# A lamp is a MuJoCo spot with a 180° cutoff: MuJoCo 3.10 draws its own "point" type without
# attenuation. And it hangs just above the ceiling at the lamp's x, y: measured on this Mac
# (CGL), a surface that faces a positional light from below renders wrong — outside a spot's cone
# it loses every light, ambient included (a black ceiling), and inside a point light's reach it
# comes out brighter than the lighting equation, not in proportion. Lit only from above, the
# floor matches the equation to ±1.5 of 255 and the ceiling gets exactly its bounced light: in
# effect a ceiling fixture over the lamp, calibrated to the same lux at the reference spot
# (world.lights computes its lux with the same fixture, so the two agree all over the room).
# The classic renderer lights per vertex: small surfaces follow the equation, a big box face
# (the official floor tiles) is off by up to ~10 % right under a lamp (measured).
SPOT_CUTOFF_DEG = 180.0
LAMP_ABOVE_CEILING_M = world_lights.LAMP_ABOVE_CEILING_M
LAMP_ATTENUATION = world_lights.LAMP_ATTENUATION
# MuJoCo 3.10's classic renderer on this Mac lights only the first 7 active lights (probed in
# the phase-2 survey; the headlight does not free a slot): seven slots, filled per frame.
LIGHT_SLOTS = 7
LIGHT_NAMES = tuple(f"dm_light_{i}" for i in range(LIGHT_SLOTS))
EMITTER_MATERIAL = "dm_emit_"  # + geom name: the material an emitter's glow is switched on


def import_mujoco() -> Any:
    """MuJoCo with an offscreen GL that works off the main thread (CGL on macOS)."""
    if platform.system() == "Darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
    import mujoco

    return mujoco


def default_scene(mjcf_dir: Path | str) -> Path:
    """The official full-collision scene: robot_allcollisions.xml on the official floor."""
    return Path(mjcf_dir).expanduser() / "scene_allcollisions.xml"


@dataclass
class EyeScene:
    """A compiled copy of the scene with the camera model applied; one thread at a time."""

    model: Any
    data: Any
    camera: int
    lights: dict[str, int]
    trunk_qpos: int
    joints: dict[str, int]
    k: Intrinsics  # of the delivered frame
    path: Path
    reference: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ceiling_m: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    lens: Lens | None = None  # None: a pinhole frame (phase 0)
    quality: Quality | None = None
    emitters: dict[str, int] = field(default_factory=dict)  # geom name → its own material id
    emission: dict[str, tuple[tuple[float, float, float], float]] = field(default_factory=dict)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.k.width, self.k.height)


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def set_intrinsics(cam: Any, k: Intrinsics, frame_size: tuple[int, int]) -> None:
    """K on an ``MjsCamera``: MuJoCo then ignores ``fovy`` and renders K's frustum."""
    width, height = frame_size
    kk = k.scaled(width, height)
    cam.resolution = [width, height]
    cam.sensor_size = [width * 1e-3, height * 1e-3]  # 1 µm "pixels": only the ratios matter
    cam.focal_pixel = [kk.fx, kk.fy]
    cam.principal_pixel = list(mujoco_principal_pixel(kk))


def _has_ceiling(spec: Any) -> bool:
    return any("ceiling" in (geom.name or "") for geom in spec.geoms)


def apply_to_spec(
    spec: Any,
    *,
    k: Intrinsics = FAMILY,
    frame_size: tuple[int, int] = FRAME_SIZE,
    ceiling_m: float | None = CEILING_M,
    camera: str = CAMERA,
    lens: Lens | None = None,
    emitters: Iterable[str] = (),
) -> None:
    """Everything the camera model changes, done on an ``MjSpec`` before compiling.

    With a ``lens`` the camera carries the lens's pinhole canvas (for ``frame_size``) instead of
    K. ``emitters``: geoms that can glow (the nest's backlight) get a material of their own.
    A ceiling plane at ``ceiling_m`` is added unless the scene already has a geom named
    "…ceiling…" (a home layout's own ceilings)."""
    mujoco = import_mujoco()
    width, height = frame_size
    canvas = lens.canvas_size(width, height) if lens is not None else (width, height)
    glob = spec.visual.global_
    glob.offwidth = max(int(glob.offwidth), width, canvas[0])
    glob.offheight = max(int(glob.offheight), height, canvas[1])
    # No torch on the head: the headlight keeps only its ambient term, set per frame.
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.headlight.ambient = [0.0, 0.0, 0.0]
    for light in spec.lights:
        light.active = False  # the scene's own lights; ours come from world.lights
    cam = spec.camera(camera)
    if cam is None:
        raise ValueError(f"the scene has no camera {camera!r}")
    # sim/camera.py's fix: the model's head camera faces backwards; a turn of 180° about the
    # camera's own right axis faces it forward and leaves the frame on its side like the mount.
    cam.quat = _quat_mul(list(cam.quat), [0.0, 1.0, 0.0, 0.0])
    if lens is not None:
        set_intrinsics(cam, lens.canvas, (lens.canvas.width, lens.canvas.height))
    else:
        set_intrinsics(cam, k, frame_size)
    if ceiling_m is not None and not _has_ceiling(spec):
        # An endless plane facing down, like the official endless floor: visual only.
        ceiling = spec.worldbody.add_geom()
        ceiling.name = "dm_ceiling"
        ceiling.type = mujoco.mjtGeom.mjGEOM_PLANE
        ceiling.size = [0.0, 0.0, 0.05]
        ceiling.pos = [0.0, 0.0, ceiling_m]
        ceiling.quat = [0.0, 1.0, 0.0, 0.0]
        ceiling.rgba = [0.93, 0.93, 0.90, 1.0]
        ceiling.contype = 0
        ceiling.conaffinity = 0
    for name in LIGHT_NAMES:
        light = spec.worldbody.add_light()
        light.name = name
        light.castshadow = False
        light.active = False
        light.type = mujoco.mjtLightType.mjLIGHT_SPOT  # each frame sets its kind
        light.pos = [0.0, 0.0, max(3.0, (ceiling_m or CEILING_M) + 0.5)]
        light.dir = [0.0, 0.0, -1.0]
        light.cutoff = SPOT_CUTOFF_DEG
        light.exponent = 0.0
        light.attenuation = list(LAMP_ATTENUATION)
        light.diffuse = [0.0, 0.0, 0.0]
        light.ambient = [0.0, 0.0, 0.0]
        light.specular = [0.0, 0.0, 0.0]
    default_rgba = [0.5, 0.5, 0.5, 1.0]  # a geom's rgba overrides its material's unless default
    for name in emitters:
        # (MjSpec finds a geom added since the last compile only by walking its geoms)
        geom = spec.geom(name) or next((g for g in spec.geoms if g.name == name), None)
        if geom is None:
            raise ValueError(f"the scene has no emitter geom {name!r}")
        rgba = list(geom.rgba)
        if geom.material and [float(v) for v in rgba] == default_rgba:
            base = spec.material(geom.material)
            if base is not None:
                rgba = list(base.rgba)
        material = spec.add_material()
        material.name = EMITTER_MATERIAL + name
        material.rgba = rgba
        material.emission = 0.0
        material.specular = 0.0
        material.shininess = 0.0
        material.reflectance = 0.0
        geom.material = material.name
        geom.rgba = default_rgba


def _robot_joints(model: Any, mujoco: Any, free: int) -> list[int]:
    """The joints of the duck (the trunk's tree); every joint when there is no trunk."""
    if free < 0:
        return list(range(model.njnt))
    root = int(model.body_rootid[model.jnt_bodyid[free]])
    return [j for j in range(model.njnt) if int(model.body_rootid[model.jnt_bodyid[j]]) == root]


def load_spec(
    spec: Any,
    *,
    path: Path | str = "<spec>",
    k: Intrinsics = FAMILY,
    frame_size: tuple[int, int] | None = None,
    quality: str | None = None,
    distortion: bool = True,
    ceiling_m: float | None = CEILING_M,
    camera: str = CAMERA,
    emitters: Iterable[str] = (),
    edit: Callable[[Any], None] | None = None,
) -> EyeScene:
    """Our own compiled copy of an ``MjSpec`` (a home layout's render spec, or an official
    scene read by :func:`load_scene`) with the camera model applied.

    The frame is ``frame_size``, else ``quality``'s rung (default 720p30); ``distortion``
    (owner decision F1: on) renders through the family lens. ``edit`` gets the ``MjSpec`` after
    the camera model is applied (test markers, say). The duck starts at its STAND keyframe;
    everything else stays where the model puts it (official keyframes carry the free objects at
    the origin, so they are not used for them)."""
    mujoco = import_mujoco()
    rung = quality_rung(quality)
    size = tuple(frame_size) if frame_size is not None else rung.size
    frame_k = k.scaled(size[0], size[1])
    lens = Lens(frame_k) if distortion else None
    emitter_names = list(emitters)
    apply_to_spec(
        spec,
        k=k,
        frame_size=(size[0], size[1]),
        ceiling_m=ceiling_m,
        camera=camera,
        lens=lens,
        emitters=emitter_names,
    )
    if edit is not None:
        edit(spec)
    model = spec.compile()
    # MuJoCo's clip planes are relative to the model's extent; ours are absolute.
    model.vis.map.znear = NEAR_CLIP_M / model.stat.extent
    model.vis.map.zfar = FAR_CLIP_M / model.stat.extent
    data = mujoco.MjData(model)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    light_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name) for name in LIGHT_NAMES
    }
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    robot = _robot_joints(model, mujoco, free)
    joints = {}
    for j in robot:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and j != free and model.jnt_type[j] in (2, 3):  # slide or hinge
            joints[name] = int(model.jnt_qposadr[j])
    emitter_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, EMITTER_MATERIAL + name)
        for name in emitter_names
    }
    eye = EyeScene(
        model=model,
        data=data,
        camera=cam,
        lights=light_ids,
        trunk_qpos=int(model.jnt_qposadr[free]) if free >= 0 else -1,
        joints=joints,
        k=frame_k,
        path=Path(path),
        ceiling_m=ceiling_m,
        lens=lens,
        quality=rung if frame_size is None or tuple(frame_size) == rung.size else None,
        emitters=emitter_ids,
    )
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand >= 0 and free >= 0:
        key = model.key_qpos[stand]
        for j in robot:
            adr = int(model.jnt_qposadr[j])
            width = 7 if model.jnt_type[j] == 0 else 4 if model.jnt_type[j] == 1 else 1
            data.qpos[adr : adr + width] = key[adr : adr + width]
    mujoco.mj_forward(model, data)
    return eye


def load_scene(
    scene: Path | str,
    *,
    k: Intrinsics = FAMILY,
    frame_size: tuple[int, int] | None = None,
    quality: str | None = None,
    distortion: bool = True,
    ceiling_m: float | None = CEILING_M,
    camera: str = CAMERA,
    emitters: Iterable[str] = (),
    edit: Callable[[Any], None] | None = None,
) -> EyeScene:
    """Our own compiled copy of ``scene`` (an official scene file, read where it lies); see
    :func:`load_spec` for the options."""
    mujoco = import_mujoco()
    path = Path(scene).expanduser()
    spec = mujoco.MjSpec.from_file(str(path))
    return load_spec(
        spec,
        path=path,
        k=k,
        frame_size=frame_size,
        quality=quality,
        distortion=distortion,
        ceiling_m=ceiling_m,
        camera=camera,
        emitters=emitters,
        edit=edit,
    )


def set_pose(eye: EyeScene, frame: dict[str, Any]) -> None:
    """Put our copy of the duck where the physics has it (``LiveBody.frame()``): trunk position
    and quaternion (w, x, y, z) and the joint angles by name. Unknown joints are ignored."""
    mujoco = import_mujoco()
    qpos = eye.data.qpos
    if eye.trunk_qpos >= 0:
        trunk = frame.get("trunk")
        quat = frame.get("quat")
        if trunk is not None:
            qpos[eye.trunk_qpos : eye.trunk_qpos + 3] = [float(v) for v in trunk][:3]
        if quat is not None:
            q = [float(v) for v in quat][:4]
            n = math.sqrt(sum(v * v for v in q)) or 1.0
            qpos[eye.trunk_qpos + 3 : eye.trunk_qpos + 7] = [v / n for v in q]
    for name, value in (frame.get("joints") or {}).items():
        adr = eye.joints.get(name)
        if adr is not None:
            qpos[adr] = float(value)
    mujoco.mj_forward(eye.model, eye.data)


def set_state(
    eye: EyeScene,
    qpos: Any,
    *,
    mocap_pos: Any = None,
    mocap_quat: Any = None,
) -> None:
    """Copy a world snapshot into our copy: the whole ``qpos`` (duck, toys, …) and the mocap
    bodies (hand, people). The render model must have the world's joints and mocap bodies (it
    is the same layout with render extras only); positions, not dynamics, so kinematics only."""
    mujoco = import_mujoco()
    model, data = eye.model, eye.data
    if len(qpos) != model.nq:
        raise ValueError(f"qpos has {len(qpos)} values, the eye's model {model.nq}")
    data.qpos[:] = qpos
    if mocap_pos is not None:
        if len(mocap_pos) != model.nmocap:
            raise ValueError(f"{len(mocap_pos)} mocap bodies, the eye's model {model.nmocap}")
        data.mocap_pos[:] = mocap_pos
    if mocap_quat is not None:
        data.mocap_quat[:] = mocap_quat
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def keyframe_pose(eye: EyeScene, keyframe: str = "STAND") -> dict[str, Any]:
    """A pose in ``LiveBody.frame()`` form from one of the scene's keyframes (for tests)."""
    mujoco = import_mujoco()
    key = mujoco.mj_name2id(eye.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key < 0:
        raise ValueError(f"the scene has no keyframe {keyframe!r}")
    qpos = eye.model.key_qpos[key]
    t = eye.trunk_qpos
    return {
        "t": 0.0,
        "trunk": [float(v) for v in qpos[t : t + 3]] if t >= 0 else [0.0, 0.0, 0.0],
        "quat": [float(v) for v in qpos[t + 3 : t + 7]] if t >= 0 else [1.0, 0.0, 0.0, 0.0],
        "joints": {name: float(qpos[adr]) for name, adr in eye.joints.items()},
    }


def camera_position(eye: EyeScene) -> tuple[float, float, float]:
    """Where the head camera is in the world (m)."""
    p = eye.data.cam_xpos[eye.camera]
    return (float(p[0]), float(p[1]), float(p[2]))


# ── the lights ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LightRig:
    """What :func:`light_rig` set: the scale back to lux and where each source went."""

    lux_per_unit: float  # lux per unit of render value (a white level surface at the reference)
    reference: tuple[float, float, float]
    lux: float  # at the reference
    lit: tuple[str, ...]  # sources given one of the MuJoCo light slots
    folded: tuple[str, ...]  # sources beyond the slots, added to the bounced light instead
    ambient: tuple[float, float, float]  # the headlight's ambient (render units)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # the light's mix there, luminance 1


def _direct_at(source: LightSource, point: tuple[float, float, float], ceiling: float) -> float:
    if source.kind == "spot" and (source.fixture is None or source.intensity is None):
        # A spot from elsewhere without its fixture: hang it over its position as phase 0 did.
        if source.position is None:
            return 0.0
        where = world_lights.fixture(source.position, ceiling)
        return (
            source.lux
            * (1.0 - source.indirect)
            * world_lights.reach(where, point)
            / max(world_lights.reach(where, (0.0, 0.0, 0.0)), 1e-12)
        )
    return source.direct_at(point)


def light_rig(
    eye: EyeScene,
    state: LightState,
    *,
    reference: Sequence[float] | None = None,
    slots: int = LIGHT_SLOTS,
) -> LightRig:
    """Set our MuJoCo lights from one room's light state.

    MuJoCo's classic renderer is linear and clamps at 1, so the lights are normalised: a white
    level surface at ``reference`` (default: the eye's reference spot) renders
    :data:`RENDER_WHITE` whatever the hour, and :func:`scene_lux` scales the render back to lux
    with :attr:`LightRig.lux_per_unit`. Each source's direct part becomes a MuJoCo light whose
    strength gives such a surface exactly its share there (a lamp with the fall-off world.lights
    uses, so the render agrees elsewhere too); its bounced part joins the (headlight) ambient.
    Beyond ``slots`` lights the weakest direct parts are folded into the ambient, so the level
    at the reference stays right. Colours are linear RGB of luminance 1.
    """
    mujoco = import_mujoco()
    model = eye.model
    where = [float(v) for v in (reference if reference is not None else eye.reference)]
    rx, ry, rz = where[0], where[1], where[2]
    ref = (rx, ry, rz)
    ceiling = eye.ceiling_m if eye.ceiling_m is not None else CEILING_M
    directs = [_direct_at(s, (rx, ry, rz), ceiling) for s in state.sources]
    total = sum(directs) + sum(s.lux * s.indirect for s in state.sources)
    total = max(total, 1e-9)
    scale = RENDER_WHITE / total  # render value per lux
    ambient = [0.0, 0.0, 0.0]
    for source in state.sources:
        for c in range(3):
            ambient[c] += source.lux * source.indirect * scale * source.color[c]
    ranked = sorted(
        (
            (direct, i)
            for i, (direct, s) in enumerate(zip(directs, state.sources, strict=True))
            if direct > 0 and s.kind in ("directional", "spot")
        ),
        reverse=True,
    )
    free = [eye.lights[name] for name in LIGHT_NAMES if eye.lights.get(name, -1) >= 0][:slots]
    lit: list[str] = []
    folded: list[str] = []
    used: set[int] = set()
    for rank, (direct, i) in enumerate(ranked):
        source = state.sources[i]
        if rank >= len(free):
            for c in range(3):
                ambient[c] += direct * scale * source.color[c]
            folded.append(source.name)
            continue
        slot = free[rank]
        if source.kind == "directional" and source.toward is not None:
            tx, ty, tz = source.toward
            cosine = max(tz, 0.05)
            model.light_type[slot] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            model.light_dir[slot] = [-tx, -ty, -tz]
            model.light_pos[slot] = [rx + 3 * tx, ry + 3 * ty, rz + 3 * tz]
            strength = direct * scale / cosine
        else:
            if source.fixture is not None and source.intensity is not None:
                px, py, pz = source.fixture
                strength = source.intensity * scale
            else:
                assert source.position is not None
                px, py, pz = world_lights.fixture(source.position, ceiling)
                strength = direct * scale / max(world_lights.reach((px, py, pz), ref), 1e-12)
            model.light_type[slot] = mujoco.mjtLightType.mjLIGHT_SPOT
            model.light_pos[slot] = [px, py, pz]
            model.light_dir[slot] = [0.0, 0.0, -1.0]
            model.light_cutoff[slot] = SPOT_CUTOFF_DEG
            model.light_exponent[slot] = 0.0
            model.light_attenuation[slot] = list(LAMP_ATTENUATION)
        model.light_diffuse[slot] = [strength * source.color[c] for c in range(3)]
        model.light_active[slot] = 1
        used.add(slot)
        lit.append(source.name)
    for slot in free:
        if slot not in used:
            model.light_active[slot] = 0
            model.light_diffuse[slot] = [0.0, 0.0, 0.0]
    model.vis.headlight.ambient = ambient
    model.vis.headlight.diffuse = [0.0, 0.0, 0.0]
    model.vis.headlight.specular = [0.0, 0.0, 0.0]
    mujoco.mj_camlight(model, eye.data)
    mix = [0.0, 0.0, 0.0]
    for direct, source in zip(directs, state.sources, strict=True):
        weight = direct + source.lux * source.indirect
        for c in range(3):
            mix[c] += weight * source.color[c]
    y = world_lights.luminance((mix[0], mix[1], mix[2]))
    color = (mix[0] / y, mix[1] / y, mix[2] / y) if y > 0 else (1.0, 1.0, 1.0)
    return LightRig(
        lux_per_unit=1.0 / scale,
        reference=(rx, ry, rz),
        lux=total,
        lit=tuple(lit),
        folded=tuple(folded),
        ambient=(ambient[0], ambient[1], ambient[2]),
        color=color,
    )


def set_lights(
    eye: EyeScene, state: LightState, *, reference: Sequence[float] | None = None
) -> float:
    """:func:`light_rig`, returning only the lux per unit of render value (phase 0's call)."""
    return light_rig(eye, state, reference=reference).lux_per_unit


def set_emitters(
    eye: EyeScene, glow: Mapping[str, tuple[Sequence[float], float]] | None = None
) -> None:
    """Which emitter geoms glow, with what colour (linear RGB) and brightness (lux × albedo, the
    camera's scale: π × cd/m²). Geoms not named are dark."""
    eye.emission = {}
    for name, (color, lux) in (glow or {}).items():
        if name not in eye.emitters or eye.emitters[name] < 0:
            raise ValueError(f"{name!r} is not an emitter of this eye")
        if lux > 0:
            c = [float(v) for v in color][:3]
            eye.emission[name] = ((c[0], c[1], c[2]), float(lux))


def scene_lux(rgb: Any, lux_per_unit: float) -> Any:
    """A rendered 8-bit frame as scene light (lux × albedo per channel, float32)."""
    import numpy as np

    return rgb.astype(np.float32) * np.float32(lux_per_unit / 255.0)


def emitters_in_view(eye: EyeScene) -> list[str]:
    """The glowing emitters that may be on this frame (inside the canvas frustum, allowing for
    their size); an emission pass is only needed for these."""
    if not eye.emission:
        return []
    mujoco = import_mujoco()
    d, m = eye.data, eye.model
    cam = d.cam_xpos[eye.camera]
    rot = d.cam_xmat[eye.camera].reshape(3, 3)
    if eye.lens is not None:
        x0, x1, y0, y1 = eye.lens.extent
    else:
        k = eye.k
        x0, x1 = -(k.cx + 0.5) / k.fx, (k.width - k.cx - 0.5) / k.fx
        y0, y1 = -(k.cy + 0.5) / k.fy, (k.height - k.cy - 0.5) / k.fy
    seen = []
    for name in eye.emission:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            continue
        local = rot.T @ (d.geom_xpos[g] - cam)
        depth = -float(local[2])
        radius = float(m.geom_rbound[g]) or 0.5
        if depth <= radius:
            if depth > -radius:
                seen.append(name)
            continue
        x, y = float(local[0]) / depth, -float(local[1]) / depth
        margin = radius / depth
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            seen.append(name)
    return seen


class HeadCameraRenderer:
    """Offscreen renders of the head camera; create, use and close it on one thread.

    Its own GL context (CGL on macOS, EGL elsewhere), one ``MjrContext`` whose offscreen buffer
    holds the full frame (and the lens canvas), and a scene with the extras a camera does not
    see switched off: skybox, mirror reflections (the floor's ``reflectance`` is a look, not a
    fact about the floor), shadows. ``bilinear`` picks the lens remap (nearest by default).

    Segmentation renders (``segment_ids``) draw into a second ``MjrContext`` made with
    ``offsamples = 0``: MuJoCo switches multisampling off for a segmentation pass, but NVIDIA's
    EGL ignores that (MuJoCo's own docs for ``offsamples`` say some backends do), and a
    multisample resolve averages the colour-coded geom ids at the edge of a thin geom into the
    ids of other geoms. macOS CGL honours the switch, so the same render was clean there.
    """

    def __init__(self, eye: EyeScene, *, bilinear: bool = False) -> None:
        mujoco = import_mujoco()
        self._mujoco = mujoco
        self.eye = eye
        self.bilinear = bilinear
        model = eye.model
        width = int(model.vis.global_.offwidth)
        height = int(model.vis.global_.offheight)
        if not hasattr(mujoco, "GLContext"):
            raise RuntimeError("MuJoCo has no offscreen GL context here (MUJOCO_GL disabled?)")
        self._gl: Any = mujoco.GLContext(width, height)
        self._gl.make_current()
        self._con: Any = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
        self._scene: Any = mujoco.MjvScene(model, maxgeom=4000)
        self._option: Any = mujoco.MjvOption()
        self._camera: Any = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = eye.camera
        self.max_size = (width, height)
        self._seg_con: Any = None  # made on the first segmentation render

    def _segmentation_context(self) -> Any:
        """The ``MjrContext`` for segmentation passes: same GL context, no multisampling (see the
        class docstring). ``offsamples`` is read when a context is made, so the model's value is
        set to 0 for that call and put back."""
        if self._seg_con is None:
            mujoco = self._mujoco
            quality = self.eye.model.vis.quality
            samples = quality.offsamples
            quality.offsamples = 0
            try:
                self._gl.make_current()
                con = mujoco.MjrContext(self.eye.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            finally:
                quality.offsamples = samples
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, con)
            self._seg_con = con
        return self._seg_con

    def _raw(self, width: int, height: int, segmentation: bool) -> Any:
        """One render of the camera's frustum into a width×height viewport, as
        ``mjr_readPixels`` returns it (rows bottom-up)."""
        import numpy as np

        mujoco = self._mujoco
        if width > self.max_size[0] or height > self.max_size[1]:
            raise ValueError(f"{width}x{height} is larger than the offscreen buffer")
        self._gl.make_current()
        con = self._segmentation_context() if segmentation else self._con
        # each context has its own offscreen framebuffer; bind the one this pass draws into
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, con)
        mujoco.mjv_updateScene(
            self.eye.model,
            self.eye.data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        flags = self._scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1 if segmentation else 0
        flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1 if segmentation else 0
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(rect, self._scene, con)
        out = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(out, None, rect, con)
        return out

    def render_pinhole(self, width: int, height: int, *, segmentation: bool = False) -> Any:
        """The camera's pinhole frustum as it is set (the lens canvas when there is a lens),
        row 0 at the top."""
        import numpy as np

        return np.flipud(self._raw(width, height, segmentation))

    def render(self, width: int, height: int, *, segmentation: bool = False) -> Any:
        """One RGB frame (height, width, 3) uint8, row 0 at the top, landscape as delivered:
        through the lens when the eye has one (a segmentation render always remaps nearest)."""
        lens = self.eye.lens
        if lens is None:
            return self.render_pinhole(width, height, segmentation=segmentation)
        cw, ch = lens.canvas_size(width, height)
        canvas = self._raw(cw, ch, segmentation)
        return lens.apply(canvas, width, height, bilinear=self.bilinear and not segmentation)

    def render_emission(self, width: int, height: int) -> tuple[Any, float] | None:
        """The glow of the eye's emitters alone (``None`` when nothing glows in view): an
        emission-only render — every light off, no ambient, every other material's emission off,
        the emitters at their colour — and the lux (× albedo) that 255 stands for."""
        eye = self.eye
        glowing = emitters_in_view(eye)
        if not glowing:
            return None
        model = eye.model
        mats = [eye.emitters[name] for name in glowing]
        peak = max(
            lux / max(world_lights.luminance(color), 1e-6)
            for color, lux in (eye.emission[name] for name in glowing)
        )
        saved_active = model.light_active.copy()
        saved_ambient = model.vis.headlight.ambient.copy()
        saved_emission = model.mat_emission.copy()
        saved_rgba = model.mat_rgba[mats].copy()
        try:
            model.light_active[:] = 0
            model.vis.headlight.ambient = [0.0, 0.0, 0.0]
            model.mat_emission[:] = 0.0
            for name, mat in zip(glowing, mats, strict=True):
                color, lux = eye.emission[name]
                weight = lux / max(world_lights.luminance(color), 1e-6) / peak
                model.mat_rgba[mat, :3] = [min(1.0, c * weight) for c in color]
                model.mat_emission[mat] = 1.0
            glow = self.render(width, height)
        finally:
            model.light_active[:] = saved_active
            model.vis.headlight.ambient = saved_ambient
            model.mat_emission[:] = saved_emission
            model.mat_rgba[mats] = saved_rgba
        return glow, peak

    def segment_ids(self, width: int, height: int) -> Any:
        """Per-pixel geom id (-1 where nothing), from a segmentation render."""
        import numpy as np

        rgb = self.render(width, height, segmentation=True).astype(np.int64)
        seg = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536
        lookup = np.full(self._scene.ngeom + 2, -1, dtype=np.int64)
        for g in range(self._scene.ngeom):
            geom = self._scene.geoms[g]
            if geom.segid >= 0 and geom.objtype == self._mujoco.mjtObj.mjOBJ_GEOM:
                lookup[geom.segid + 1] = geom.objid
        seg = np.where(seg < lookup.size, seg, 0)
        return lookup[seg]

    def close(self) -> None:
        try:
            if self._con is not None:
                self._gl.make_current()
                if self._seg_con is not None:
                    self._seg_con.free()
                self._con.free()
        finally:
            self._seg_con = None
            self._con = None
            if self._gl is not None:
                self._gl.free()
                self._gl = None


# ── the whole camera: render, light back to lux, sensor, ISP ─────────────────


@dataclass(frozen=True)
class CameraConfig:
    """The camera model's knobs (defaults: the owner's phase-2 decisions)."""

    quality: str = DEFAULT_QUALITY  # [media] quality
    distortion: bool = True  # F1
    white_balance: bool = True
    raw_bits: int | None = RAW_BITS
    full_well_scale: float = 1.0  # phase 0's noise; the scaled readout may be ~4 (σ ÷ 2)
    vignetting: float = 0.0  # residual after lens shading correction, 0..1 of cos⁴ (off)
    bilinear: bool = False  # the lens remap; nearest by default (1.9 ms at 720p)


@dataclass
class Capture:
    """One delivered frame and how it was made (the frame is on its side, as the sensor reads
    it; ``rotate`` says how to turn it upright, like the media.frame header)."""

    rgb: Any  # (height, width, 3) uint8
    width: int
    height: int
    index: int
    controls: Controls  # the AE registers it was exposed with
    wb: tuple[float, float, float]
    lux_per_unit: float
    glow: bool  # an emitter was drawn in (HDR pass)
    timings: dict[str, float]
    rotate: int = ROTATE_DEG
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def format(self) -> str:
        return CAPTURE_FORMAT

    @property
    def bytes_len(self) -> int:
        return self.width * self.height * 2

    def uyvy(self) -> bytes:
        """The frame as media.frame sends it: packed UYVY, ``width × height × 2`` bytes."""
        return to_uyvy(self.rgb)

    def upright(self) -> Any:
        return upright(self.rgb)


class HeadCamera:
    """The head camera end to end, for one thread: render (through the lens), light back to
    lux (plus any glow), the AE and white balance stepped on metering renders, the sensor.

    Call :func:`light_rig` (or :mod:`render_scene`) and :func:`set_emitters` for the frame's
    light first, then :meth:`meter` twice a second and :meth:`capture` on demand. Deterministic
    for a seed: metering tick n and frame n draw from their own random streams.
    """

    def __init__(
        self,
        eye: EyeScene,
        *,
        config: CameraConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.eye = eye
        self.config = config or CameraConfig()
        self.renderer = HeadCameraRenderer(eye, bilinear=self.config.bilinear)
        self.ae = AeLoop()
        self.awb = Awb()
        self.seed = int(seed)
        self.meter_ticks = 0
        self.frames = 0
        self._vignette: dict[tuple[int, int], Any] = {}

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.eye.frame_size

    def _rng(self, stream: int, index: int) -> Any:
        import numpy as np

        return np.random.default_rng([self.seed, stream, index])

    def _vignette_map(self, width: int, height: int) -> Any:
        strength = float(self.config.vignetting)
        if strength <= 0:
            return None
        key = (width, height)
        if key not in self._vignette:
            import numpy as np

            lens = self.eye.lens
            if lens is not None:
                cos4 = lens.cos4(width, height)
            else:
                k = self.eye.k.scaled(width, height)
                u = (np.arange(width) - k.cx) / k.fx
                v = (np.arange(height) - k.cy) / k.fy
                x, y = np.meshgrid(u, v)
                cos4 = (1.0 / (1.0 + x * x + y * y)) ** 2
            self._vignette[key] = (1.0 - strength * (1.0 - cos4)).astype(np.float32)
        return self._vignette[key]

    def scene(self, width: int, height: int, lux_per_unit: float) -> tuple[Any, bool, float]:
        """The scene light (lux × albedo, float32) of one render, glow included; whether an
        emitter glowed; the render time (ms)."""
        import numpy as np

        t0 = time.perf_counter()
        raw = self.renderer.render(width, height)
        light = scene_lux(raw, lux_per_unit)
        glow = self.renderer.render_emission(width, height)
        if glow is not None:
            image, peak = glow
            light = light + image.astype(np.float32) * np.float32(peak / 255.0)
        return light, glow is not None, (time.perf_counter() - t0) * 1000

    def _develop(self, light: Any, controls: Controls, rng: Any) -> Any:
        cfg = self.config
        return expose(
            light,
            controls,
            rng,
            wb=self.awb.gains if cfg.white_balance else None,
            full_well=FULL_WELL_E * cfg.full_well_scale,
            raw_bits=cfg.raw_bits,
            vignette=self._vignette_map(light.shape[1], light.shape[0]),
        )

    def meter(self, lux_per_unit: float, *, illuminant: Sequence[float] | None = None) -> float:
        """One auto-exposure (and white balance) tick on a metering render; returns the metered
        mean luma. ``illuminant``: the light's colour (:attr:`LightRig.color`) for the AWB."""
        light, _, _ = self.scene(*METER_SIZE, lux_per_unit)
        if self.config.white_balance:
            self.awb.update(light, illuminant=illuminant)
        out = self._develop(light, self.ae.sensor, self._rng(0, self.meter_ticks))
        self.meter_ticks += 1
        mean = float(luma(out).mean())
        self.ae.tick(mean)
        return mean

    def converge(
        self,
        lux_per_unit: float,
        max_ticks: int = 40,
        *,
        illuminant: Sequence[float] | None = None,
    ) -> int:
        """Tick until the AE (and the white balance) settle — for a first frame or a test."""
        for tick in range(1, max_ticks + 1):
            self.meter(lux_per_unit, illuminant=illuminant)
            wb_ok = not self.config.white_balance or self.awb.settled()
            if self.ae.settled() and wb_ok and tick > 1:
                return tick
        return max_ticks

    def capture(self, lux_per_unit: float, *, index: int | None = None) -> Capture:
        """The full frame at the rung's size with the AE's current registers."""
        width, height = self.frame_size
        n = self.frames if index is None else int(index)
        t0 = time.perf_counter()
        light, glowed, render_ms = self.scene(width, height, lux_per_unit)
        controls = self.ae.sensor
        rgb = self._develop(light, controls, self._rng(1, n))
        self.frames += 1
        return Capture(
            rgb=rgb,
            width=width,
            height=height,
            index=n,
            controls=controls,
            wb=self.awb.gains,
            lux_per_unit=lux_per_unit,
            glow=glowed,
            timings={
                "render_ms": round(render_ms, 3),
                "frame_ms": round((time.perf_counter() - t0) * 1000, 3),
            },
        )

    def close(self) -> None:
        self.renderer.close()
