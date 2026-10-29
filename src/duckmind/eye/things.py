"""duck-eye's ``things`` task: the low things standing on the floor, and which toy each one is
(spec §4.4 row ``things``; owner decision D2 — the 4–8 cm box gate, reported per colour).

It works on what the floor line found (:mod:`duckmind.eye.floorline`): every ``thing`` end — not
floor, with the floor coming back above it below the horizon — grouped across neighbouring
columns is a *candidate* with a bearing, a range (its foot on the floor), a width and a height.
This task adds what the candidate looks like:

* its **colour** — the mean hue, saturation and brightness of the candidate's pixels that are not
  floor-like, named coarsely (``red`` ``orange`` ``yellow`` ``green`` ``cyan`` ``blue`` ``purple``
  ``pink`` ``brown`` ``white`` ``grey`` ``black``) for reports and for duck-body's logs;
* a **colour histogram** (hue × saturation, 12 × 3, plus 3 grey levels; integers per mille), and
  the **toy** it matches among the registered ones (histogram intersection ≥
  :data:`MATCH_MIN`); unregistered things are ``id`` ``None``.

Toys are registered from a look (:meth:`ToyRegistry.register`): the histogram is a feature, kept
only on the duck (spec §4.8), never a picture. No things in ``dark`` or ``black`` light (spec
§15.2: no curiosity action in the dark).

Pure; numpy; no clocks, no sockets, no :mod:`duckmind.world`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.eye.floorline import FloorLook, _column_edges

HUE_BINS = 12
SAT_BINS = 3
GREY_BINS = 3
HIST_LEN = HUE_BINS * SAT_BINS + GREY_BINS
SAT_GREY = 0.12  # below this saturation a pixel counts as grey (by its brightness)
MATCH_MIN = 600  # histogram intersection (per mille) to call a thing a registered toy
MIN_PIXELS = 6  # candidate pixels (quarter resolution) needed to describe it

_NAMES = (
    (15, "red"),
    (40, "orange"),
    (70, "yellow"),
    (165, "green"),
    (200, "cyan"),
    (260, "blue"),
    (290, "purple"),
    (335, "pink"),
    (360, "red"),
)


def _rgb(y: Any, u: Any, v: Any) -> Any:
    """BT.601 full-range RGB (floats 0…1) of luma/chroma samples."""
    import numpy as np

    yy = y.astype(np.float64)
    uu = u.astype(np.float64) - 128.0
    vv = v.astype(np.float64) - 128.0
    r = yy + 1.402 * vv
    g = yy - 0.344136 * uu - 0.714136 * vv
    b = yy + 1.772 * uu
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255) / 255.0


def _hsv(rgb: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.maximum(d, 1e-6)
    hue = np.where(
        mx == r,
        ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(d > 1e-6, hue * 60.0, 0.0)
    return hue, sat, mx


def colour_name(hue: float, sat: float, val: float) -> str:
    """A coarse name for a mean colour (degrees, 0…1, 0…1)."""
    if sat < SAT_GREY or val < 0.12:
        if val >= 0.75:
            return "white"
        return "grey" if val >= 0.25 else "black"
    if 15 <= hue < 45 and val < 0.6:
        return "brown"
    for top, name in _NAMES:
        if hue < top:
            return name
    return "red"


def histogram(rgb: Any) -> list[int]:
    """Hue × saturation (12 × 3) + grey (3) histogram of RGB pixels, per mille."""
    import numpy as np

    if rgb.size == 0:
        return [0] * HIST_LEN
    hue, sat, val = _hsv(rgb.reshape(-1, 3))
    grey = sat < SAT_GREY
    hb = np.clip((hue / 360.0 * HUE_BINS).astype(int), 0, HUE_BINS - 1)
    sb = np.clip(((sat - SAT_GREY) / (1 - SAT_GREY) * SAT_BINS).astype(int), 0, SAT_BINS - 1)
    gb = np.clip((val * GREY_BINS).astype(int), 0, GREY_BINS - 1)
    idx = np.where(grey, HUE_BINS * SAT_BINS + gb, hb * SAT_BINS + sb)
    counts = np.bincount(idx, minlength=HIST_LEN)
    total = int(counts.sum())
    return [int(round(1000 * c / total)) for c in counts]


def intersection(a: Sequence[int], b: Sequence[int]) -> int:
    """Histogram intersection, per mille."""
    return int(sum(min(x, y) for x, y in zip(a, b, strict=True)))


@dataclass
class ToyRegistry:
    """The toys the duck knows by colour (features only; spec §4.8)."""

    toys: dict[str, list[int]] = field(default_factory=dict)

    def register(self, toy_id: str, hist: Sequence[int]) -> None:
        if len(hist) != HIST_LEN:
            raise ValueError(f"a toy's histogram has {HIST_LEN} bins")
        self.toys[str(toy_id)] = [int(v) for v in hist]

    def match(self, hist: Sequence[int]) -> tuple[str | None, int]:
        best: tuple[str | None, int] = (None, 0)
        for toy_id, ref in sorted(self.toys.items()):
            score = intersection(hist, ref)
            if score > best[1]:
                best = (toy_id, score)
        return best if best[1] >= MATCH_MIN else (None, best[1])

    def to_json(self) -> dict[str, Any]:
        return {"v": 1, "toys": {k: list(v) for k, v in sorted(self.toys.items())}}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ToyRegistry:
        reg = cls()
        for toy_id, hist in (value.get("toys") or {}).items():
            reg.register(toy_id, hist)
        return reg


def describe(
    look: FloorLook, y: Any, u: Any, v: Any, registry: ToyRegistry | None = None
) -> list[dict[str, Any]]:
    """The floor line's thing candidates, each with its colour, histogram and toy id."""
    import numpy as np

    if not look.result.get("ok") or look.floorlike is None:
        return []
    rows, cols = y.shape
    if u.shape != y.shape:
        ry, rx = -(-rows // u.shape[0]), -(-cols // u.shape[1])
        u = np.repeat(np.repeat(u, ry, axis=0), rx, axis=1)[:rows, :cols]
        v = np.repeat(np.repeat(v, ry, axis=0), rx, axis=1)[:rows, :cols]
    edges = _column_edges(cols)
    out = []
    for cand in look.result.get("things", []):
        c0 = edges[cand["col0"]][0]
        c1 = edges[cand["col1"]][1]
        bottom = int(cand["row"])
        top = int(cand["top"]) if cand.get("top", -1) >= 0 else max(0, bottom - 12)
        region = (slice(top, bottom + 1), slice(c0, c1))
        mask = ~look.floorlike[region]
        if int(mask.sum()) < MIN_PIXELS:
            continue
        rgb = _rgb(y[region][mask], u[region][mask], v[region][mask])
        hist = histogram(rgb)
        mean = rgb.reshape(-1, 3).mean(axis=0)
        hue, sat, val = _hsv(mean[None, :])
        name = colour_name(float(hue[0]), float(sat[0]), float(val[0]))
        toy, score = registry.match(hist) if registry is not None else (None, 0)
        item = {
            "b_mrad": int(cand["b_mrad"]),
            "r_mm": int(cand["r_mm"]),
            "w_mm": int(cand["w_mm"]),
            "h_mm": int(cand["h_mm"]),
            "colour": name,
            "hue_deg": int(round(float(hue[0]))),
            "sat_pm": int(round(float(sat[0]) * 1000)),
            "val_pm": int(round(float(val[0]) * 1000)),
            "hist": hist,
            "id": toy,
            "match_pm": score,
            "px": int(mask.sum()),
            "conf_ppm": 700_000 if toy is None else min(950_000, 400_000 + score * 550),
        }
        out.append(item)
    return out


class ThingsTask:
    """The ``things`` task as duck-eye's server runs it, on the same look's floor line. Bound to
    the server's ``floor`` task (:meth:`bind`) it shares the learned floor and reuses that task's
    result when ``floor`` already ran on this look; unbound it keeps a floor line of its own
    (which learns from the same ``eye.learn`` events through :meth:`learn`)."""

    name = "things"

    def __init__(self, floor: Any = None, registry: ToyRegistry | None = None) -> None:
        from duckmind.eye.floorline import FloorTask

        self._own = floor is None
        self.floor = FloorTask() if floor is None else floor
        self.registry = registry if registry is not None else ToyRegistry()

    def bind(self, tasks: Mapping[str, Any]) -> None:
        """Share the server's ``floor`` task (and its learned floor) when there is one."""
        floor = tasks.get("floor")
        if floor is not None and floor is not self:
            self.floor = floor
            self._own = False

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> ThingsTask:
        """``{"toys": <ToyRegistry.to_json()>}`` restores the registered toys."""
        toys = options.get("toys")
        return cls(registry=ToyRegistry.from_json(toys) if isinstance(toys, Mapping) else None)

    def digest_items(self) -> list[str]:
        return [
            f"things.hist hue={HUE_BINS} sat={SAT_BINS} grey={GREY_BINS} sat_grey={SAT_GREY}",
            f"things.match={MATCH_MIN} min_px={MIN_PIXELS}",
        ]

    def learn(self, event: Any) -> int:
        """Its own floor line learns too (the shared one learns through the ``floor`` task)."""
        return int(self.floor.learn(event)) if self._own else 0

    def run(self, view: Any) -> dict[str, Any]:
        level = str(view.light.get("level", "bright")) if view.light else "bright"
        if level not in ("bright", "dim"):
            return {"ok": 0, "why": "dark", "things": []}
        if getattr(self.floor, "last_id", None) != view.look.look_id:
            self.floor.run(view)
        look = self.floor.last
        if look is None or not look.result.get("ok"):
            why = look.result.get("why", "no_floor") if look is not None else "no_floor"
            return {"ok": 0, "why": why, "things": []}
        planes = view.planes
        return {"ok": 1, "things": describe(look, planes.y, planes.u, planes.v, self.registry)}


def sightings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What duck-body folds into ``sense.sighting`` for things (percept-v1 fields)."""
    out = []
    for item in result.get("things", []):
        out.append(
            {
                "what": "thing",
                "id": item.get("id"),
                "b_mrad": item["b_mrad"],
                "r_mm": item["r_mm"],
                "r_src": "ground",
                "conf_ppm": item["conf_ppm"],
            }
        )
    return out
