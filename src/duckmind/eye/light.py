"""The light meter, version 2: how bright the room is, told from the frames (spec §4.5).

The camera is the duck's only light sensor (spec §4.5). mediad's auto-exposure pulls the mean
luma back to 90 in any room it can, so the brightness of a settled frame alone cannot tell day
from dusk; what gives the room away is what the exposure had to spend to get there — and, since
no socket publishes the exposure, the noise the gain leaves in the picture.

**The board path** (duck-eye's; :func:`read_quarter`, :func:`level_q`, :class:`LightTracker`):
integers from the quarter-resolution luma plane (:func:`duckmind.eye.frames.sample_quarter`) —
the mean luma, the noise σ in thousandths (:func:`noise_milli`, Immerkær's estimator on flat
blocks, integer arithmetic so the same bytes give the same numbers on any machine: P2) and the
glow (:func:`glow_q`: a compact patch of saturated pixels, the dock's status light or the old
backlight). The level:

* ``black`` — mean luma under :attr:`Thresholds.black_luma`, unless a glow is in view: in the
  dark with the dock's light in sight the level is ``dark`` (owner decision C1, 2026-09-27);
* ``bright`` / ``dim`` / ``dark`` — σ against :attr:`Thresholds.sigma_bright_milli` and
  :attr:`Thresholds.sigma_dim_milli`.

:data:`THRESHOLDS` are refitted on renders of the three simulated homes against the lux at the
duck (owner decision C3: ≥ 30 lux bright, ≥ 2.5 dim, ≥ 0.5 very dark, else black) and are
**sim-provisional** until a board is measured (phase 8). :class:`LightTracker` adds spec §4.5's
two rules: the level changes only when two looks in a row agree, and a look whose luma jumped
more than 12 % from the look just before (the auto-exposure still moving) — or a frame that is
dark with the gain still low, or glaring — is not counted: look again in 0.5 s.

**The evaluator path** (v1, kept): :func:`measure` reads a full-resolution frame, and the exposure
when it is known (:func:`level_from_exposure`, the survey's rule, §2.3: the sensor's shutter,
analogue and digital gain — known exactly in the simulator; readable on the board only from the
sensor node, spec §4.5's second method). Bright while the ISP adds (almost) no digital gain (up
to 1.4×), dim up to 5.7×, dark beyond, black when the loop is at its ceiling and still cannot
lift the mean luma to :data:`BLACK_LUMA`. While the mean luma is far from mediad's setpoint and
the loop could still move (:func:`adapting`), the frame alone decides and ``adapting`` says so.

Nothing here imports the simulator: the frame and the sensor readings are the inputs, as on the
robot. Duck Core does not read it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from duckmind.i18n import msg

LEVELS = ("bright", "dim", "dark", "black")
LEVEL_ZH = {
    "bright": msg("eye.light.LEVEL_ZH.bright"),
    "dim": msg("eye.light.LEVEL_ZH.dim"),
    "dark": msg("eye.light.LEVEL_ZH.dark"),
    "black": msg("eye.light.LEVEL_ZH.black"),
}

# mediad's auto-exposure limits (exposure.rs): what "at the ceiling" means.
HARD_LINES = 1200.0
MAX_ANALOGUE = 11.0
MAX_DIGITAL = 16.0
LINE_S = 19.05e-6

BLACK_LUMA = 45.0  # below this, with the loop at its ceiling, the frame is mostly noise
# Far from mediad's setpoint (luma 90 ± 12 %) with room left to move, the auto-exposure is
# still catching up with a change of light: below BLACK_LUMA and not at its ceiling, or above
# ADAPT_HIGH_LUMA and not at its floor (4 lines, 1x gains).
ADAPT_HIGH_LUMA = 180.0
MIN_LINES = 4.0
# Digital-gain boundaries sit halfway (geometrically) between the survey's rungs, not on
# them: lamps settle at 1x, dusk at 2-3x, twilight at 11-16x. mediad's ±12 % luma deadband lets
# the loop rest anywhere from 0.75x to 1.29x of a room's exact budget, depending on which way
# the light changed: a 5-lux room settles at 2.7x digital coming from daylight and at 4.2x
# coming from moonlight (measured on these renders), so the survey's proposed 4x would call the
# same dusk dim or dark by history. sqrt(1*2) and sqrt(3*11):
BRIGHT_MAX_DIGITAL = 1.4
DIM_MAX_DIGITAL = 5.7
SIGMA_BRIGHT = 4.0  # spec §4.5, frame alone: σ < 4 bright
SIGMA_DIM = 9.0  # 4 ≤ σ < 9 dim, σ ≥ 9 dark

# The camera's light constant, an assumption from the survey (an 18 % grey card at 400 lux
# converges at budget 1125): sensor output per lux per budget for a white surface.
KAPPA = (90.0 / 255.0) ** 2.2 / (0.18 * 400.0 * 1125.0)
GREY_WORLD = 0.18  # a reflected-light meter's assumption about the average surface


@dataclass(frozen=True)
class Exposure:
    """The sensor's exposure as mediad set it: lines and gains as multiples."""

    lines: float
    analogue: float
    digital: float

    @property
    def budget(self) -> float:
        return self.lines * self.analogue * self.digital

    @property
    def at_ceiling(self) -> bool:
        return self.lines >= HARD_LINES and self.analogue >= MAX_ANALOGUE

    @property
    def at_floor(self) -> bool:
        return self.lines <= MIN_LINES and self.analogue <= 1.0 and self.digital <= 1.0

    @classmethod
    def from_registers(cls, exposure: int, analogue_gain: int, gain: int) -> Exposure:
        """From mediad's ``Controls`` (256 is 1x for both gains)."""
        return cls(float(exposure), analogue_gain / 256.0, gain / 256.0)


@dataclass(frozen=True)
class LightReading:
    level: str  # bright / dim / dark / black
    level_zh: str
    method: str  # "exposure" or "frame"
    luma: float  # mean luma, every eighth pixel, 0..255
    noise: float  # luma noise σ on flat parts of the picture
    level_frame: str  # spec §4.5 from σ and luma alone
    level_exposure: str | None  # from exposure + mean luma, when the exposure is known
    lux_estimate: float | None  # reflected-light estimate (grey world), when known
    exposure: dict[str, float] | None
    adapting: bool = False  # the exposure is still catching up: the frame alone decided

    @property
    def agree(self) -> bool:
        return self.level_exposure is None or self.level_exposure == self.level_frame

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["agree"] = self.agree
        return out


# ── the rules (plain numbers in, a level out) ─────────────────────────────


def level_from_exposure(luma: float, exposure: Exposure) -> str:
    if exposure.at_ceiling and luma < BLACK_LUMA:
        return "black"
    if exposure.digital <= BRIGHT_MAX_DIGITAL + 1e-6:
        return "bright"
    if exposure.digital <= DIM_MAX_DIGITAL + 1e-6:
        return "dim"
    return "dark"


def adapting(luma: float, exposure: Exposure) -> bool:
    """The auto-exposure has not caught up with the light yet: the frame is far too dark or too
    bright and the loop still has room to move (mediad would step it)."""
    return (luma < BLACK_LUMA and not exposure.at_ceiling) or (
        luma > ADAPT_HIGH_LUMA and not exposure.at_floor
    )


def level_from_frame(luma: float, sigma: float) -> str:
    if luma < BLACK_LUMA:
        return "black"
    if sigma < SIGMA_BRIGHT:
        return "bright"
    if sigma < SIGMA_DIM:
        return "dim"
    return "dark"


def lux_estimate(luma: float, exposure: Exposure) -> float:
    """What a reflected-light meter reads: the light that would put a grey-world scene at this
    luma with this exposure. Off by the room's real reflectance, like any such meter."""
    linear = (max(luma, 0.5) / 255.0) ** 2.2
    return linear / (exposure.budget * KAPPA * GREY_WORLD)


# ── measuring a frame ────────────────────────────────────────────────────


def frame_mean_luma(plane: Any) -> float:
    """Mean of every eighth pixel in raster order, as mediad meters a frame."""
    import numpy as np

    flat = np.asarray(plane).reshape(-1)
    return float(flat[::8].astype(np.float64).mean())


def noise_sigma(plane: Any, *, step: int = 4, block: int = 8) -> float:
    """Luma noise σ from flat parts of a frame (Immerkær's estimator, block median).

    Measured on every ``step``-th pixel both ways — duck-eye's quarter-resolution plane — so
    the ISP's denoise, which ties neighbouring pixels together, does not hide the noise. Each
    ``block``×``block`` patch of the Laplacian-like residual gives σ = √(π/2)·mean|r|/6;
    clipped patches are skipped, textured ones (above 1.5× the median) dropped, and the median
    of the rest returned.
    """
    import numpy as np

    p = np.asarray(plane, dtype=np.float32)[::step, ::step]
    if p.shape[0] < block + 2 or p.shape[1] < block + 2:
        return 0.0
    r = (
        p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        + p[:-2, 2:]
        - 2 * p[1:-1, :-2]
        + 4 * p[1:-1, 1:-1]
        - 2 * p[1:-1, 2:]
        + p[2:, :-2]
        - 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    h = (r.shape[0] // block) * block
    w = (r.shape[1] // block) * block
    if h == 0 or w == 0:
        return 0.0
    blocks = np.abs(r[:h, :w]).reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    means = p[1 : h + 1, 1 : w + 1].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    sigmas = blocks * np.float32(math.sqrt(math.pi / 2) / 6)
    usable = sigmas[(means > 12) & (means < 243)]
    if usable.size == 0:
        usable = sigmas.reshape(-1)
    centre = float(np.median(usable))
    for _ in range(2):
        kept = usable[usable <= 1.5 * centre] if centre > 0 else usable
        if kept.size == 0:
            break
        centre = float(np.median(kept))
    return centre


def measure(plane: Any, exposure: Exposure | None = None) -> LightReading:
    """The light level from one upright luma frame (full resolution), and the exposure if known."""
    luma = frame_mean_luma(plane)
    sigma = noise_sigma(plane)
    by_frame = level_from_frame(luma, sigma)
    by_exposure = level_from_exposure(luma, exposure) if exposure is not None else None
    catching_up = exposure is not None and adapting(luma, exposure)
    level = by_exposure if by_exposure is not None and not catching_up else by_frame
    return LightReading(
        level=level,
        level_zh=LEVEL_ZH[level],
        method="exposure" if by_exposure is not None and not catching_up else "frame",
        adapting=catching_up,
        luma=round(luma, 2),
        noise=round(sigma, 3),
        level_frame=by_frame,
        level_exposure=by_exposure,
        lux_estimate=round(lux_estimate(luma, exposure), 4) if exposure is not None else None,
        exposure=(
            {
                "lines": exposure.lines,
                "shutter_ms": round(exposure.lines * LINE_S * 1000, 2),
                "analogue": round(exposure.analogue, 3),
                "digital": round(exposure.digital, 3),
                "budget": round(exposure.budget, 1),
            }
            if exposure is not None
            else None
        ),
    )


# ── the board path: integers from the quarter-resolution plane (v2) ─────────────


@dataclass(frozen=True)
class Thresholds:
    """The frame-only level's thresholds, as integers (σ in thousandths)."""

    black_luma: int
    sigma_bright_milli: int
    sigma_dim_milli: int
    glow_luma: int = 235  # a pixel this bright in a black frame is a light, not noise
    glow_min_px: int = 4  # quarter-resolution pixels (the dock's strip at 3 m: ~14)
    glow_spread: int = 40  # the glow's bounding box holds at most this many pixels per lit one
    # Owner C1 counts only the *dock's* light: its status light is a level strip (160 × 10 mm),
    # so the glow must be at least this much wider than tall (thousandths) and fill this much of
    # its box — a standby LED, a phone or a row of router dots is not the dock.
    glow_aspect_milli: int = 2000
    glow_fill_milli: int = 350
    source: str = ""

    def digest_items(self) -> list[str]:
        return [
            f"light:{self.black_luma},{self.sigma_bright_milli},{self.sigma_dim_milli},"
            f"{self.glow_luma},{self.glow_min_px},{self.glow_spread},"
            f"{self.glow_aspect_milli},{self.glow_fill_milli}"
        ]


# Owner decision C3 (2026-09-27): refitted on 4,500 renders of the three simulated homes
# (default_home, apartment, hazard_home; noon to a moonless night; head HOME, half down, up and
# turned) against the lux at the duck: 97.2 % agree (spec §4.5's 45 / 4 / 9 read 78.8 %, the
# 55 prior 86.6 %). Black moved to just under mediad's deadband (79): below it the loop is at its
# ceiling and still short, which in these homes happens under ~0.7 lux. Fitted on two homes the
# third reads 91.8–98.2 % (the apartment is the hard one), so these are sim-provisional until a
# board is measured (phase 8); docs/validation/evidence/phase3/eye_light_refit.json.
THRESHOLDS = Thresholds(
    black_luma=75,
    sigma_bright_milli=2800,
    sigma_dim_milli=5500,
    source="C3 refit 2026-09-27 on 3 sim homes (sim-provisional)",
)
SPEC_THRESHOLDS = Thresholds(45, 4000, 9000, source="spec §4.5 (phase 0 survey ladder)")
# The truth the gate is scored against (owner decision C3): lux at the duck.
LUX_BANDS = ((30.0, "bright"), (2.5, "dim"), (0.5, "dark"))


def level_of_lux(lux: float) -> str:
    """The level a lux figure at the duck stands for (≥ 30 bright, ≥ 2.5 dim, ≥ 0.5 very dark,
    else black) — for evaluators; the duck never knows its lux."""
    for floor, level in LUX_BANDS:
        if lux >= floor:
            return level
    return "black"


_IMMERKAER_E6 = 208_886  # √(π/2)/6 × 10⁶: σ = that × mean |residual|
_CLIP_LOW, _CLIP_HIGH = 12, 243  # blocks whose mean is clipped say nothing of the noise


def mean_luma_q(plane: Any) -> int:
    """The mean of the plane, rounded (integers)."""
    import numpy as np

    flat = np.asarray(plane).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return 0
    total = int(flat.sum(dtype=np.int64))
    return (2 * total + n) // (2 * n)


def _lower_median(values: Any) -> int:
    import numpy as np

    ordered = np.sort(values)
    return int(ordered[(ordered.size - 1) // 2])


def noise_milli(plane: Any, *, block: int = 8) -> int:
    """Luma noise σ × 1000 on flat parts of a (quarter-resolution) plane, in integers.

    Immerkær's residual on every pixel of the plane (already every fourth of the frame: the
    ISP's denoise ties neighbours together, every fourth pixel is independent again), summed
    over ``block``×``block`` patches; patches with a clipped mean are skipped, textured ones
    (above 1.5× the median) dropped twice, and the lower median of the rest returned."""
    import numpy as np

    p = np.asarray(plane, dtype=np.int32)
    if p.ndim != 2 or p.shape[0] < block + 2 or p.shape[1] < block + 2:
        return 0
    r = (
        p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        + p[:-2, 2:]
        - 2 * p[1:-1, :-2]
        + 4 * p[1:-1, 1:-1]
        - 2 * p[1:-1, 2:]
        + p[2:, :-2]
        - 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    h = (r.shape[0] // block) * block
    w = (r.shape[1] // block) * block
    area = block * block
    sums = np.abs(r[:h, :w]).reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    means = p[1 : h + 1, 1 : w + 1].reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    milli = (sums.astype(np.int64) * _IMMERKAER_E6 + area * 500) // (area * 1000)
    usable = milli[(means > _CLIP_LOW * area) & (means < _CLIP_HIGH * area)]
    if usable.size == 0:
        usable = milli.reshape(-1)
    centre = _lower_median(usable)
    for _ in range(2):
        kept = usable[2 * usable <= 3 * centre] if centre > 0 else usable
        if kept.size == 0:
            break
        centre = _lower_median(kept)
    return centre


def strip_like(count: int, width: int, height: int, t: Thresholds | None = None) -> bool:
    """Does a lit patch of ``count`` pixels in a ``width`` × ``height`` box look like the dock's
    status light — a level strip, wider than tall, filling its box (owner C1: only the dock's
    light turns black into very dark)?"""
    t = t or THRESHOLDS
    if count <= 0 or width <= 0 or height <= 0:
        return False
    wide = width * 1000 >= t.glow_aspect_milli * height
    filled = count * 1000 >= t.glow_fill_milli * width * height
    return wide and filled


def glow_q(plane: Any, t: Thresholds | None = None) -> int:
    """How many pixels of the dock's light the plane holds (0: none): pixels at or above
    ``glow_luma``, at least ``glow_min_px`` of them, inside a bounding box of at most
    ``glow_spread`` pixels per lit one (one light, not a bright scene) that is shaped like the
    dock's status strip (:func:`strip_like`; any other small light is not the dock)."""
    import numpy as np

    t = t or THRESHOLDS
    lit = np.asarray(plane) >= t.glow_luma
    count = int(lit.sum())
    if count < t.glow_min_px:
        return 0
    rows = np.flatnonzero(lit.any(axis=1))
    cols = np.flatnonzero(lit.any(axis=0))
    height = int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)
    if width * height > t.glow_spread * count:
        return 0
    return count if strip_like(count, width, height, t) else 0


@dataclass(frozen=True)
class FrameLight:
    """One frame's light numbers (the board path): all integers."""

    luma: int  # mean luma, 0..255
    sigma_milli: int  # noise σ × 1000
    glow_px: int  # quarter-resolution pixels of a compact glow (0: none)

    @property
    def noise_q(self) -> int:
        """σ rounded to a whole level (percept-v1 ``sense.light.noise_q``)."""
        return (self.sigma_milli + 500) // 1000


def read_quarter(plane: Any, t: Thresholds | None = None) -> FrameLight:
    """The light numbers of a quarter-resolution upright luma plane."""
    return FrameLight(mean_luma_q(plane), noise_milli(plane), glow_q(plane, t))


def level_q(reading: FrameLight, t: Thresholds | None = None) -> str:
    """The level of one frame from its numbers alone (the board path)."""
    t = t or THRESHOLDS
    if reading.luma < t.black_luma:
        return "dark" if reading.glow_px else "black"
    if reading.sigma_milli < t.sigma_bright_milli:
        return "bright"
    if reading.sigma_milli < t.sigma_dim_milli:
        return "dim"
    return "dark"


# ── two looks: hysteresis and the auto-exposure's settling (spec §4.5) ──────────

HYSTERESIS_LOOKS = 2  # the level brightens when this many looks in a row agree
_RANK = {"black": 0, "dark": 1, "dim": 2, "bright": 3}


def darker(a: str, b: str) -> str:
    """The darker of two levels (unknown words count as black)."""
    return a if _RANK.get(a, 0) <= _RANK.get(b, 0) else b


SETTLE_MS = 500  # the auto-exposure is still moving: look again this much later
SETTLE_JUMP_PPM = 120_000  # 12 %: two adjacent looks' lumas further apart than this
ADJACENT_NS = 2_000_000_000  # looks this close are "adjacent" (a scan's stops, a retake)
AE_DEADBAND = (79, 101)  # mediad's setpoint 90 ± 12 %: inside it the loop does not move
AE_HIGH_LUMA = 160  # far above it: lights just came on, the gain has not come down yet
MAX_SETTLES = 2  # after this many waits in a row, take the frame as it is


def settling(
    reading: FrameLight,
    previous: FrameLight | None = None,
    gap_ns: int | None = None,
    t: Thresholds | None = None,
) -> bool:
    """Is the auto-exposure still moving (spec §4.5: wait 0.5 s and look again)?

    Never inside mediad's deadband (the loop rests anywhere in it). Outside it:

    * darker than the deadband while the noise says the gain is still low (a lamp-lit exposure
      a moment after the lamps went off);
    * far brighter than it (the gain of the dark a moment after the lamps came on);
    * or, spec §4.5's rule, the look just before (within :data:`ADJACENT_NS`) had a luma more
      than 12 % away — unless this frame is black (in the dark the luma is the light, not the
      loop moving)."""
    t = t or THRESHOLDS
    low, high = AE_DEADBAND
    if low <= reading.luma <= high:
        return False
    if reading.luma < low and reading.sigma_milli < t.sigma_bright_milli:
        return True
    if reading.luma > AE_HIGH_LUMA:
        return True
    if previous is None or gap_ns is None or gap_ns > ADJACENT_NS or gap_ns < 0:
        return False
    if reading.luma < t.black_luma:
        return False
    jump = abs(reading.luma - previous.luma) * 1_000_000
    return jump > SETTLE_JUMP_PPM * max(previous.luma, 1)


class LightTracker:
    """The eye's light level across looks (pure; the caller passes each look's time).

    :meth:`update` takes one frame's numbers and answers the ``light`` result of that look:
    ``level``, ``luma``, ``noise_q`` (percept-v1 ``sense.light``), ``raw`` (this frame's own
    level), ``glow`` (pixels), ``settle_ms`` (> 0: the auto-exposure is still moving, look again
    that much later; this frame did not count) and ``pending`` (looks so far agreeing on a
    brighter level). The first counted look sets the level.

    The hysteresis is one-sided: a counted look *darker* than the level takes effect at once, and
    only brightening needs :data:`HYSTERESIS_LOOKS` agreeing looks. The reported ``level`` is
    never brighter than this frame's own (``raw``), even on a look that waits for the exposure —
    the first look after the lamps go out must not read the old "bright" (a camera-only duck
    walked 5–43 cm in pitch dark on it, phase-3 verify: 23/30)."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.t = thresholds or THRESHOLDS
        self.level: str | None = None
        self.candidate: str | None = None
        self.count = 0
        self.previous: FrameLight | None = None
        self.previous_ns: int | None = None
        self.settles = 0

    def update(self, reading: FrameLight, t_ns: int) -> dict[str, Any]:
        raw = level_q(reading, self.t)
        gap = None if self.previous_ns is None else t_ns - self.previous_ns
        wait = settling(reading, self.previous, gap, self.t) and self.settles < MAX_SETTLES
        self.previous, self.previous_ns = reading, t_ns
        if wait:
            self.settles += 1
        else:
            self.settles = 0
            if self.level is None:
                self.level = raw
            elif raw == self.level:
                self.candidate, self.count = None, 0
            elif darker(raw, self.level) == raw:
                self.level, self.candidate, self.count = raw, None, 0  # darker: at once
            else:
                if raw == self.candidate:
                    self.count += 1
                else:
                    self.candidate, self.count = raw, 1
                if self.count >= HYSTERESIS_LOOKS:
                    self.level, self.candidate, self.count = raw, None, 0
        level = self.level if self.level is not None else raw
        return {
            "level": darker(level, raw),
            "luma": reading.luma,
            "noise_q": reading.noise_q,
            "noise_milli": reading.sigma_milli,
            "raw": raw,
            "glow": reading.glow_px,
            "settle_ms": SETTLE_MS if wait else 0,
            "pending": self.count,
        }


# ── the C3 refit (evaluators; pure) ──────────────────────────────────────────


def agreement(samples: list[tuple[FrameLight, str]], t: Thresholds) -> tuple[int, int]:
    """(hits, n): how many frames' levels match their truth under ``t``."""
    hits = sum(1 for reading, truth in samples if level_q(reading, t) == truth)
    return hits, len(samples)


def fit_thresholds(
    samples: list[tuple[FrameLight, str]],
    *,
    prior: Thresholds | None = None,
    black_grid: range = range(30, 91),
    bright_grid: range = range(2000, 8001, 100),
    dim_grid: range = range(5000, 20001, 250),
) -> tuple[Thresholds, int]:
    """The thresholds that read the most frames right, and how many (grid search). Among ties
    the middle of the best run is taken on each axis (the most margin), the black line first
    (it alone decides black against the rest), then σ."""
    import numpy as np

    prior = prior or THRESHOLDS
    luma = np.array([r.luma for r, _ in samples], dtype=np.int64)
    sigma = np.array([r.sigma_milli for r, _ in samples], dtype=np.int64)
    glow = np.array([r.glow_px > 0 for r, _ in samples])
    truth = np.array([s for _, s in samples])

    def score(black: int, bright: int, dim: int) -> int:
        level = np.where(
            luma < black,
            np.where(glow, "dark", "black"),
            np.where(sigma < bright, "bright", np.where(sigma < dim, "dim", "dark")),
        )
        return int((level == truth).sum())

    def middle_of_best(values: list[int], scores: list[int]) -> int:
        best = max(scores)
        runs: list[list[int]] = []
        current: list[int] = []
        for v, s in zip(values, scores, strict=True):
            if s == best:
                current.append(v)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        longest = max(runs, key=len)
        return longest[len(longest) // 2]

    blacks = list(black_grid)
    black = middle_of_best(
        blacks, [score(b, prior.sigma_bright_milli, prior.sigma_dim_milli) for b in blacks]
    )
    brights = list(bright_grid)
    bright = middle_of_best(brights, [score(black, s, prior.sigma_dim_milli) for s in brights])
    dims = [d for d in dim_grid if d > bright]
    dim = middle_of_best(dims, [score(black, bright, d) for d in dims])
    fitted = Thresholds(
        black_luma=black,
        sigma_bright_milli=bright,
        sigma_dim_milli=dim,
        glow_luma=prior.glow_luma,
        glow_min_px=prior.glow_min_px,
        glow_spread=prior.glow_spread,
        source="refit",
    )
    return fitted, score(black, bright, dim)
