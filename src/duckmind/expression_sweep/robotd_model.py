"""How robotd turns a ``robot.head`` intent into what the network sees and the servos get.

Values are the pinned upstream defaults (microduck ``robotd``, ``duck-control``,
``robotd-params``); the sweep reproduces this path so its results describe the
robot, not an idealized head:

1. the head intent (offsets from HOME, radians) never expires; last write wins;
2. robotd slews it every 50 Hz tick: ``ema += 0.2 · (intent − ema)`` (head_alpha);
3. ``obs[51..55]`` carries the slewed offsets raw — the networks were trained on
   these as deltas from HOME — and the network outputs all 14 joint actions;
4. ``targets = HOME + scale · action`` (0.9 velstand, 1.0 sitstand, both times
   7.4 V / battery voltage), then an action low-pass: head ``0.5``, legs ``0.7``;
5. the mouth is not part of any network: ``−5° + open · 35°`` is written after
   inference.

Seated, robotd runs ``alpha_sitstand`` every tick with twist ``[1, 0, 0]`` and the head
slots live; standing still is ``velstand`` at zero twist.
"""

from __future__ import annotations

import math

CONTROL_HZ = 50
HEAD_ALPHA = 0.2
ACTION_LOWPASS = {"head": 0.5, "legs": 0.7}
ACTION_SCALE = {"stand": 0.9, "sit": 1.0}
SIT_TWIST = (1.0, 0.0, 0.0)
STAND_TWIST = (0.0, 0.0, 0.0)
NETWORKS = {"sit": "alpha_sitstand.onnx", "stand": "velstand.onnx"}
# Head command ranges the networks were trained on (deltas from HOME, radians).
TRAINED_HEAD_RANGE = {"neck_pitch": 1.10, "head_pitch": 1.10, "head_yaw": 1.40, "head_roll": 0.31}
# neck_pitch can lean back only ~0.70 rad past HOME before its MJCF joint limit.
NECK_BACK_LIMIT = 1.0472 - 0.3491
HEAD_FORCE_LIMIT_NM = 0.96  # head actuator forcerange in the official MJCF
HEAD_STRAIN_NM = 0.4  # velstand's stall penalty threshold: sustained load above is strain
MOUTH_CLOSED = math.radians(-5.0)
MOUTH_OPEN = math.radians(30.0)


def slew(offsets: list[float], alpha: float = HEAD_ALPHA) -> list[float]:
    """robotd's head EMA over a 50 Hz stream, starting from rest (ema = 0)."""
    ema = 0.0
    out = []
    for value in offsets:
        ema += alpha * (value - ema)
        out.append(ema)
    return out


def mouth_angle(open_fraction: float) -> float:
    if not math.isfinite(open_fraction):
        return MOUTH_CLOSED
    return MOUTH_CLOSED + max(0.0, min(1.0, open_fraction)) * (MOUTH_OPEN - MOUTH_CLOSED)
