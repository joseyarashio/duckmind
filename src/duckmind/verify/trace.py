"""A recorded trial as the evaluator reads it (schema ``mv-trace-1``).

A trace is a plain dict (one JSON line). ``cols`` holds one list per tick; every list has the
same length. Positions, headings and the ``fallen`` flag are the WORLD's truth, written by the
runner from the simulator. ``behaviour`` ... ``arrived`` are the brain's own record (what it
said and did); the evaluator uses them only as events to time (a change of behaviour, a reflex)
or, for M4, as claims to compare with the truth.

::

    {"schema": "mv-trace-1", "layer": "L1" | "L2", "trial": str, "pair": str,
     "preset": str, "seed": int, "stimulus": str, "arm": "stim" | "sham",
     "dt_s": float, "onset_idx": int,      # first tick that can see the stimulus
     "stim": {"perceivable": bool | None,  # truth: in view / audible / touching at onset
              "basis": "visible" | "audible" | "contact",
              "delivered": bool,           # stim arm: the world took it; sham: False
              "obj": str | None,           # an object stimulus: its id in objects["stim"]
              "width_m": float | None,
              "level_db_head": float | None, "closing_speed_m_s": float | None},
     "cols": {"t": [s], "x": [m], "y": [m], "yaw": [rad], "gaze": [rad, head camera heading],
              "fallen": [bool], "behaviour": [str], "bout": [int], "reflex": [str | None],
              "phase": [0..1 satiation], "vx": [m/s], "arrived": [bool],
              "tgt_kind": [str | None], "tgt_bearing": [rad | None],
              "tgt_range": [m | None], "tgt_track": [int | None]},
     "objects": {"stim": [[x, y]] | None,   # the stimulus object's truth path (a sham carries
                                            # the planned path of its pair: "virtual")
                 "stim_virtual": bool,
                 "things": {id: [[x, y]]},  # every other thing in the room
                 "widths": {id: m}},
     "touch_reach_m": float, "error": str | None}
"""

from __future__ import annotations

import math
from typing import Any

Trace = dict[str, Any]

COLUMNS = (
    "t",
    "x",
    "y",
    "yaw",
    "gaze",
    "fallen",
    "behaviour",
    "bout",
    "reflex",
    "phase",
    "vx",
    "arrived",
    "tgt_kind",
    "tgt_bearing",
    "tgt_range",
    "tgt_track",
)


def wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def n_ticks(trace: Trace) -> int:
    return len(trace["cols"]["t"])


def validate(trace: Trace) -> list[str]:
    """Problems that make a trace unusable (empty list: fine)."""
    problems: list[str] = []
    if trace.get("schema") != "mv-trace-1":
        problems.append("schema")
    if trace.get("error"):
        problems.append(f"recorded error: {trace['error']}")
    cols = trace.get("cols") or {}
    lengths = {len(cols.get(name, ())) for name in COLUMNS}
    if len(lengths) != 1 or 0 in lengths:
        problems.append("columns missing or of different length")
        return problems
    n = lengths.pop()
    onset = trace.get("onset_idx")
    if not isinstance(onset, int) or not 1 <= onset < n:
        problems.append("onset_idx out of range")
    return problems


def window(trace: Trace, seconds: float) -> range:
    """Tick indices from the onset tick to ``seconds`` after it (inclusive)."""
    t = trace["cols"]["t"]
    onset = trace["onset_idx"]
    end = onset
    while end + 1 < len(t) and t[end + 1] - t[onset] <= seconds + 1e-9:
        end += 1
    return range(onset, end + 1)


def window_complete(trace: Trace, seconds: float) -> bool:
    """The trace runs at least ``seconds`` past the onset."""
    t = trace["cols"]["t"]
    return t[-1] - t[trace["onset_idx"]] >= seconds - 1e-9


def bearing_to(x: float, y: float, ox: float, oy: float) -> float:
    return math.atan2(oy - y, ox - x)


def dist_to(x: float, y: float, ox: float, oy: float) -> float:
    return math.hypot(ox - x, oy - y)
