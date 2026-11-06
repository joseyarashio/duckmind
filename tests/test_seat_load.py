"""Policy set v8's sit-stand network sits lower (legs ≈ 23 against v5's ≈ 26) and lands its seat in a bounce: proprio's
seated lift numbers, set on v5's seat, claimed a pickup 11 times an hour of free roam while it sat on the floor (v5: 2).
They now follow the loaded seat's measurement (neuro/seatcal.py); a duck held in the hand is still claimed, and v5 keeps
exactly its numbers (UPSTREAM-0185-2026-11-05 §10)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from duckmind.duckbody.proprio import Proprio, ProprioConfig
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import seatcal

CASES = json.loads((Path(__file__).parent / "fixtures" / "lift" / "seat_v8.json").read_text())[
    "cases"
]
TABLE = seatcal.load()
V8 = next(s for s in TABLE.seats.values() if s.sitstand_sha256 != TABLE.reference)
POLICY = {"sit": "sit", "stand": "stand", "walk": "walk"}


def state(t_ms: int, load: float, posture: str, tilt: float) -> RobotState:
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = load * 35.0 / 10.0
    return RobotState.from_wire(
        {
            "t_ns": int(t_ms) * 1_000_000,
            "policy": POLICY.get(posture, "walk"),
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {
                "fallen": False,
                "limp": False,
                "gravity": [math.sin(tilt), 0.0, -math.cos(tilt)],
            },
            "joints": [0.0] * 15,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def claims(frames, cfg: ProprioConfig) -> int:
    p = Proprio(cfg)
    return sum("lifted" in p.step(state(*f)).edges for f in frames)


def v8_cfg() -> ProprioConfig:
    return TABLE.proprio_for(ProprioConfig(), V8.sitstand_sha256, V8.body, V8.delay_ms)


def test_the_v8_seat_on_the_floor_is_not_a_pickup() -> None:
    for name, case in CASES.items():
        assert claims(case["frames"], ProprioConfig()) >= 1, name  # the v5 numbers: a false claim
        assert claims(case["frames"], v8_cfg()) == 0, name


def test_a_seated_duck_held_in_the_hand_is_still_claimed_on_v8() -> None:
    # sitting on the floor 3 s, then lifted: the legs carry ≈ 0.40 (V8Compare's in-hand seated median 0.401)
    frames = [(k * 20, 0.65, "sit", 0.05) for k in range(150)]
    frames += [(3000 + k * 20, 0.40 + 0.04 * math.sin(k / 3.0), "sit", 0.1) for k in range(100)]
    assert claims(frames, v8_cfg()) == 1


def test_v5_and_unmeasured_seats_keep_proprio_s_numbers() -> None:
    base = ProprioConfig()
    ref = TABLE.seats[TABLE.reference]
    assert TABLE.proprio_for(base, ref.sitstand_sha256, ref.body, ref.delay_ms) is base
    assert TABLE.proprio_for(base, "0" * 64, None, 20.0) is base
    assert (
        TABLE.proprio_for(base, V8.sitstand_sha256, "allcollisions", 20.0) is base
    )  # another body
    assert TABLE.proprio_for(base, V8.sitstand_sha256, V8.body, 0.0) is base  # another delay
