"""Hand-made traces for the machine-verification evaluator tests (no simulator involved)."""

from __future__ import annotations

import math
from typing import Any


class T:
    """A mutable trace: still body at the origin, looking along +x, sitting in bout 0."""

    def __init__(
        self,
        stimulus: str = "clap",
        *,
        arm: str = "stim",
        layer: str = "L1",
        preset: str = "bold_explorer",
        seed: int = 1,
        dt: float = 0.1,
        seconds: float = 40.0,
        onset: float = 10.0,
        perceivable: bool | None = True,
        delivered: bool | None = None,
        obj: tuple[float, float] | None = None,
    ) -> None:
        n = int(round(seconds / dt))
        self.dt = dt
        self.n = n
        self.onset = int(round(onset / dt))
        cols: dict[str, list[Any]] = {
            "t": [round(i * dt, 6) for i in range(n)],
            "x": [0.0] * n,
            "y": [0.0] * n,
            "yaw": [0.0] * n,
            "gaze": [0.0] * n,
            "fallen": [False] * n,
            "behaviour": ["rest_sit"] * n,
            "bout": [0] * n,
            "reflex": [None] * n,
            "phase": [0.0] * n,
            "vx": [0.0] * n,
            "arrived": [False] * n,
            "tgt_kind": [None] * n,
            "tgt_bearing": [None] * n,
            "tgt_range": [None] * n,
            "tgt_track": [None] * n,
        }
        stim_obj = None
        if obj is not None:
            stim_obj = [[obj[0], obj[1]] for _ in range(n)]
        self.trace: dict[str, Any] = {
            "schema": "mv-trace-1",
            "layer": layer,
            "trial": f"{layer}|{preset}|{seed}|{stimulus}|{arm}",
            "pair": f"{layer}|{preset}|{seed}|{stimulus}",
            "preset": preset,
            "seed": seed,
            "stimulus": stimulus,
            "arm": arm,
            "dt_s": dt,
            "onset_idx": self.onset,
            "stim": {
                "perceivable": perceivable,
                "basis": "visible",
                "delivered": (arm == "stim") if delivered is None else delivered,
                "obj": "ball" if obj is not None else None,
                "width_m": 0.07 if obj is not None else None,
                "level_db_head": None,
                "closing_speed_m_s": None,
            },
            "cols": cols,
            "objects": {
                "stim": stim_obj,
                "stim_virtual": arm == "sham" and obj is not None,
                "things": {},
                "widths": {},
            },
            "touch_reach_m": 0.096,
            "error": None,
        }

    @property
    def cols(self) -> dict[str, list[Any]]:
        return self.trace["cols"]

    def i(self, t: float) -> int:
        return int(round(t / self.dt))

    def behave(self, t0: float, t1: float, name: str, *, bout: int | None = None, phase=None) -> T:
        """Behaviour ``name`` from t0 to t1 (s). ``bout`` None keeps the bout id."""
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["behaviour"][k] = name
            if bout is not None:
                self.cols["bout"][k] = bout
            if phase is not None:
                self.cols["phase"][k] = phase(k) if callable(phase) else phase
        return self

    def bouts(self, *segments: tuple[float, float, str, int, float]) -> T:
        """(t0, t1, behaviour, bout id, satiation at the end) segments."""
        for t0, t1, name, bout, end_phase in segments:
            a, b = self.i(t0), min(self.n, self.i(t1))
            for k in range(a, b):
                self.cols["behaviour"][k] = name
                self.cols["bout"][k] = bout
                self.cols["phase"][k] = end_phase * (k - a + 1) / max(1, b - a)
        return self

    def reflex(self, t0: float, t1: float, name: str) -> T:
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["reflex"][k] = name
        return self

    def gaze(self, t0: float, t1: float, a0: float, a1: float) -> T:
        """The head's heading goes linearly from a0 to a1 (deg) between t0 and t1 and stays."""
        i0, i1 = self.i(t0), self.i(t1)
        for k in range(self.n):
            if k < i0:
                v = a0
            elif k >= i1:
                v = a1
            else:
                v = a0 + (a1 - a0) * (k - i0) / (i1 - i0)
            self.cols["gaze"][k] = math.radians(v)
        return self

    def target(self, t0: float, t1: float, bearing_deg: float, rng: float, kind="thing", track=7):
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["tgt_kind"][k] = kind
            self.cols["tgt_bearing"][k] = math.radians(bearing_deg)
            self.cols["tgt_range"][k] = rng
            self.cols["tgt_track"][k] = track
        return self

    def arrive(self, t0: float, t1: float):
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["arrived"][k] = True
        return self

    def fall(self, t0: float, t1: float):
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["fallen"][k] = True
        return self

    def walk(self, t0: float, t1: float, vx: float = 0.3):
        for k in range(self.i(t0), min(self.n, self.i(t1))):
            self.cols["vx"][k] = vx
        return self

    def thing(self, key: str, x: float, y: float, width: float = 0.1):
        self.trace["objects"]["things"][key] = [[x, y] for _ in range(self.n)]
        self.trace["objects"]["widths"][key] = width
        return self

    def place(self, x: float, y: float, yaw: float = 0.0):
        for k in range(self.n):
            self.cols["x"][k], self.cols["y"][k], self.cols["yaw"][k] = x, y, yaw
        return self

    def out(self) -> dict[str, Any]:
        return self.trace


def pair(stimulus: str = "clap", **kw: Any) -> tuple[T, T]:
    """A stim trace and its same-seed sham, both still."""
    return T(stimulus, arm="stim", **kw), T(stimulus, arm="sham", **kw)
