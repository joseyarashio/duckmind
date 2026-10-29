"""duck-eye's ``duck`` task: other Microducks in the frame, with the official detector's recipe
(spec §4.4 row ``duck``, §4.7; owner decision D4: the framework only — no weights are downloaded
until their licence is settled).

What the official model expects, and so what this module does (read from the upstream crates
``duck-detect`` and ``mediad/src/detect.rs``; our own code, nothing copied):

* the frame the right way up (``mediad --rotate``, default 90°), **RGB**, letterboxed — not
  stretched — into a 320×320 square padded with grey :data:`PAD` (114), nearest-neighbour
  (:func:`letterbox_rgb`);
* the ``.onnx`` model on the CPU takes NCHW floats in 0…1 on its input ``images``; the ``.rknn``
  model on the NPU takes the NHWC bytes and normalises them itself;
* the head is ``[1, 5, N]`` **planar**: every cx, then every cy, w, h, score (:data:`STRIDE`);
* candidates over :data:`THRESHOLD` (robotd-params' 0.35), suppressed at IoU :data:`IOU_LIMIT`
  (mediad's 0.5), mapped back out of the letterbox into the frame (:func:`decode`).

The backend is chosen by the model file's extension, as mediad does: ``.rknn`` runs on the NPU
through ``rknnlite`` (on the board only), anything else on the CPU through ``onnxruntime``. No
model → :func:`look` answers ``None`` ("the task found nothing because it could not look"), and
duck-body treats that as no duck seen. When the task runs at all is duck-body's call (spec §4.4:
only when the BLE chorale says a duck is near; at most one NPU model per look).

Pure functions and one small class; no sockets, no clocks, no :mod:`duckmind.world`. numpy is
needed; onnxruntime only when an ``.onnx`` model is opened.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STRIDE = 5  # the head emits cx, cy, w, h, score per candidate (planar)
PAD = 114  # the letterbox's grey (uyvy::PAD)
INPUT_SIZE = 320
THRESHOLD = 0.35  # robotd-params [detect] threshold default
IOU_LIMIT = 0.5  # mediad's decode(…, threshold, 0.5)
INPUT_NAME = "images"


class DetectError(RuntimeError):
    """The model cannot be opened or run (missing runtime, wrong shape, bad file)."""


@dataclass(frozen=True)
class Letterbox:
    """How a frame was fitted into the square: ``square = frame * scale + pad``."""

    scale: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class Detection:
    """One duck, in the pixels of the frame that went in (not of the letterbox)."""

    score: float
    box: tuple[float, float, float, float]  # x0 y0 x1 y1

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0

    def bearing(self, frame_width: float) -> float:
        """Across the frame: −1 hard left, 0 the middle, 1 hard right (duck-detect's number)."""
        return (self.centre()[0] / frame_width) * 2.0 - 1.0


def letterbox_rgb(frame: Any, size: int = INPUT_SIZE) -> tuple[Any, Letterbox]:
    """Scale an upright RGB frame (H×W×3 uint8) to fit a ``size`` square and pad it grey:
    ``(square NHWC-less H×W×3 uint8, Letterbox)``. Nearest-neighbour from the fitted size, the
    same arithmetic as duck-detect's ``letterbox_rgb``, so rounding cannot walk off the frame."""
    import numpy as np

    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[2] != 3:
        raise DetectError(f"expected an H×W×3 RGB frame, got shape {img.shape}")
    height, width = int(img.shape[0]), int(img.shape[1])
    f32 = np.float32
    scale = min(f32(size) / f32(width), f32(size) / f32(height))  # f32, as the crate computes
    fitted_w = min(max(int(_round_half_away(float(f32(width) * scale))), 1), size)
    fitted_h = min(max(int(_round_half_away(float(f32(height) * scale))), 1), size)
    pad_x = (size - fitted_w) // 2
    pad_y = (size - fitted_h) // 2
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    rows = (np.arange(fitted_h) * height) // fitted_h
    cols = (np.arange(fitted_w) * width) // fitted_w
    out[pad_y : pad_y + fitted_h, pad_x : pad_x + fitted_w] = img[rows[:, None], cols[None, :]]
    return out, Letterbox(scale=float(np.float32(scale)), pad_x=float(pad_x), pad_y=float(pad_y))


def _round_half_away(value: float) -> float:
    """Rust's ``f32::round``: halves away from zero (Python's round() goes to even)."""
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``x0 y0 x1 y1`` boxes (0 when the union is empty)."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)

    def area(r: Sequence[float]) -> float:
        return max(r[2] - r[0], 0.0) * max(r[3] - r[1], 0.0)

    union = area(a) + area(b) - overlap
    return 0.0 if union <= 0 else overlap / union


def decode(
    raw: Any,
    letterbox: Letterbox,
    threshold: float = THRESHOLD,
    iou_limit: float = IOU_LIMIT,
) -> list[Detection]:
    """Candidates over ``threshold`` from a planar ``[1, 5, N]`` head (any shape holding 5·N
    values in that order), best first, suppressed at ``iou_limit``, mapped back into the frame.
    Ties keep the head's order (a stable sort, like duck-detect's)."""
    import numpy as np

    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    count = flat.size // STRIDE
    heads = flat[: count * STRIDE].reshape(STRIDE, count)
    found: list[Detection] = []
    for index in np.flatnonzero(heads[4] >= np.float32(threshold)):
        cx, cy, w, h, score = (float(v) for v in heads[:, index])
        half_w, half_h = w / 2.0, h / 2.0
        found.append(
            Detection(
                score=score,
                box=(
                    (cx - half_w - letterbox.pad_x) / letterbox.scale,
                    (cy - half_h - letterbox.pad_y) / letterbox.scale,
                    (cx + half_w - letterbox.pad_x) / letterbox.scale,
                    (cy + half_h - letterbox.pad_y) / letterbox.scale,
                ),
            )
        )
    found.sort(key=lambda d: -d.score)
    kept: list[Detection] = []
    for detection in found:
        if all(iou(detection.box, other.box) < iou_limit for other in kept):
            kept.append(detection)
    return kept


# ── the runtimes ────────────────────────────────────────────────────────────────


class _Onnx:
    """An ``.onnx`` head on the CPU (onnxruntime), NCHW floats 0…1 on ``images``."""

    def __init__(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - a required dependency (D5)
            raise DetectError("onnxruntime is not installed") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # beside a 50 Hz control loop: one core at most
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises its own types
            raise DetectError(f"cannot open {path.name}: {exc}") from exc
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise DetectError(f"{path.name}: expected one input, found {len(inputs)}")
        shape = list(inputs[0].shape)
        if len(shape) != 4 or shape[1] != 3:
            raise DetectError(f"{path.name}: expected a [1, 3, H, W] input, got {shape}")
        height, width = shape[2], shape[3]
        if not isinstance(height, int) or not isinstance(width, int) or height != width:
            raise DetectError(f"{path.name}: expected a fixed square input, got {shape}")
        self.name = inputs[0].name
        self.size = int(height)

    def run(self, square: Any) -> Any:
        import numpy as np

        planar = np.ascontiguousarray(square.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.session.run(None, {self.name: planar})[0]


class _Rknn:
    """An ``.rknn`` head on the RK3566 NPU (``rknnlite``; the board only)."""

    def __init__(self, path: Path) -> None:
        try:
            from rknnlite.api import RKNNLite  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DetectError("no NPU runtime (rknnlite) on this machine") from exc
        runtime = RKNNLite()
        if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime() != 0:
            raise DetectError(f"the NPU would not take {path.name}")
        self.runtime = runtime
        self.size = INPUT_SIZE

    def run(self, square: Any) -> Any:  # pragma: no cover - needs the board
        return self.runtime.inference(inputs=[square[None]])[0]


class DuckDetector:
    """One detector model, chosen by its extension (``.rknn`` → NPU, else ONNX on the CPU)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise DetectError(f"no model at {self.path}")
        self.backend: _Onnx | _Rknn = (
            _Rknn(self.path) if self.path.suffix == ".rknn" else _Onnx(self.path)
        )

    @property
    def input_size(self) -> int:
        return self.backend.size

    def detect(
        self, rgb: Any, threshold: float = THRESHOLD, iou_limit: float = IOU_LIMIT
    ) -> list[Detection]:
        """Ducks in one upright RGB frame (H×W×3 uint8), in that frame's pixels."""
        square, fit = letterbox_rgb(rgb, self.input_size)
        raw = self.backend.run(square)
        return decode(raw, fit, threshold, iou_limit)


def open_detector(path: str | Path | None) -> DuckDetector | None:
    """The detector at ``path``, or ``None`` when there is no model (D4: none downloaded) or its
    runtime is missing — the task then answers ``None``."""
    if path is None:
        return None
    try:
        return DuckDetector(path)
    except DetectError:
        return None


@dataclass(frozen=True)
class DuckSeen:
    """A duck in the frame as integers for ``eye.result`` (spec §7.2 ``sense.sighting``)."""

    b_mrad: int  # bearing: + to the left of the optical axis (counter-clockwise, like vyaw)
    e_mrad: int  # elevation: + up
    conf_ppm: int
    box_px: tuple[int, int, int, int]  # x0 y0 x1 y1 in the upright frame


def look(
    detector: DuckDetector | None,
    rgb: Any,
    angles: Callable[[float, float], tuple[float, float]],
    threshold: float = THRESHOLD,
) -> list[DuckSeen] | None:
    """The ``duck`` task on one look: ``None`` without a model, else the ducks found (best first).

    ``angles(u, v)`` gives a pixel's (right, down) angles in radians from the optical axis with
    the lens undone (:mod:`duckmind.eye.camera`)."""
    if detector is None:
        return None
    out: list[DuckSeen] = []
    for det in detector.detect(rgb, threshold):
        u, v = det.centre()
        right, down = angles(u, v)
        out.append(
            DuckSeen(
                b_mrad=int(round(-right * 1000.0)),
                e_mrad=int(round(-down * 1000.0)),
                conf_ppm=int(round(min(max(det.score, 0.0), 1.0) * 1_000_000)),
                box_px=tuple(int(round(c)) for c in det.box),  # type: ignore[arg-type]
            )
        )
    return out


class DuckTask:
    """The ``duck`` task as duck-eye's server runs it (an NPU task: at most one a look).
    ``from_config({"model": path})``; no model (D4: none downloaded) → ``{"ok": 0, "why":
    "no_model"}`` every look."""

    name = "duck"

    def __init__(self, model: str | Path | None = None, threshold: float = THRESHOLD) -> None:
        self.model_path = None if model is None else Path(model)
        self.detector = open_detector(self.model_path)
        self.threshold = threshold

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> DuckTask:
        return cls(options.get("model"), float(options.get("threshold", THRESHOLD)))

    def digest_items(self) -> list[str]:
        import hashlib

        items = [f"duck.threshold={self.threshold!r}", f"duck.iou={IOU_LIMIT!r}"]
        if self.detector is not None and self.model_path is not None:
            items.append("duck.model=" + hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        return items

    def run(self, view: Any) -> dict[str, Any]:
        if self.detector is None:
            return {"ok": 0, "why": "no_model", "ducks": []}
        from duckmind.eye.frames import upright_rgb

        h = view.header
        step = 2  # half resolution is plenty for a 320×320 letterbox
        rgb = upright_rgb(view.data, h.width, h.height, h.rotate, step=step)
        pose = view.camera_pose
        ducks = []
        for det in self.detector.detect(rgb, self.threshold):
            u, v = (c * step for c in det.centre())
            if pose is not None:
                b, e = view.camera.bearing_mrad(u, v, pose)
            else:
                right, down = view.camera.angles_rad(u, v)
                b, e = int(round(-right * 1000)), int(round(-down * 1000))
            ducks.append(
                {
                    "b_mrad": int(b),
                    "e_mrad": int(e),
                    "conf_ppm": int(round(min(max(det.score, 0.0), 1.0) * 1_000_000)),
                    "box_px": [int(round(c * step)) for c in det.box],
                }
            )
        return {"ok": 1, "ducks": ducks}
