"""Policy set v8's walk network, standing still on the floor, settles for a second or more into a stance whose legs read
15.6-19.7 (v5's never under 19 for 0.3 s): under proprio's standing claim level (20, set on v5), so it claimed a pickup and
sat the duck down — the V8Soak's five "seated" false lifts were these. The standing level now follows the loaded walk's
measured stance (neuro/seatcal.py: v8 14.0, between its floor 15.6 and a duck standing in the hand, ≈ 10); a duck held in
the hand is still claimed, standing or seated, and v5 keeps exactly its numbers (UPSTREAM-0185-2026-11-05 §13)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from duckmind.duckbody.proprio import Proprio, ProprioConfig
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import seatcal

CASES = json.loads((Path(__file__).parent / "fixtures" / "lift" / "stance_v8.json").read_text())[
    "cases"
]
TABLE = seatcal.load()
V8_WALK = next(iter(TABLE.stances.values()))
V8_SEAT = next(s for s in TABLE.seats.values() if s.sitstand_sha256 != TABLE.reference)
POLICY = {"sit": "sit", "stand": "stand", "walk": "walk"}


def state(
    t_ms: int, load: float, posture: str, tilt: float, gyro: float = 0.0, vx: float = 0.0
) -> RobotState:
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = load * 35.0 / 10.0
    return RobotState.from_wire(
        {
            "t_ns": int(t_ms) * 1_000_000,
            "policy": POLICY.get(posture, "walk"),
            "move": {"requested": [vx, 0, 0], "applied": [vx, 0, 0]},
            "safety": {
                "fallen": False,
                "limp": False,
                "gravity": [math.sin(tilt), 0.0, -math.cos(tilt)],
            },
            "joints": [0.0] * 15,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, gyro], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def claims(frames, cfg: ProprioConfig) -> int:
    p = Proprio(cfg)
    return sum("lifted" in p.step(state(*f)).edges for f in frames)


def v8_cfg() -> ProprioConfig:
    """Everything the v8 body gets: its seat's numbers and its stance's."""
    cfg = TABLE.proprio_for(
        ProprioConfig(), V8_SEAT.sitstand_sha256, V8_SEAT.body, V8_SEAT.delay_ms
    )
    return TABLE.stand_proprio_for(cfg, V8_WALK.walk_sha256, V8_WALK.body, V8_WALK.delay_ms)


def test_the_v8_stance_standing_on_the_floor_is_not_a_pickup() -> None:
    false = {n: c for n, c in CASES.items() if c["kind"] == "false"}
    assert len(false) >= 6
    for name, case in false.items():
        assert claims(case["frames"], ProprioConfig()) >= 1, (
            name
        )  # proprio's own numbers: the false claim
        assert claims(case["frames"], v8_cfg()) == 0, name


def test_a_v8_duck_standing_in_the_hand_is_still_claimed() -> None:
    real = {n: c for n, c in CASES.items() if c["kind"] == "real"}
    assert real
    for name, case in real.items():  # the world's hand lifting the standing duck (V8Fix lift rig)
        assert claims(case["frames"], v8_cfg()) == 1, name
    # standing on the floor 3 s (≈ 32, v8's stance), then held: the legs carry ≈ 10 with a swing up to 13
    frames = [(k * 20, 0.92, "stand", 0.02) for k in range(150)]
    frames += [
        (3000 + k * 20, 0.29 + 0.08 * max(0.0, math.sin(k / 4.0)), "stand", 0.02)
        for k in range(100)
    ]
    assert claims(frames, v8_cfg()) == 1


def test_a_seated_duck_held_in_the_hand_is_still_claimed_with_the_whole_v8_body() -> None:
    frames = [(k * 20, 0.65, "sit", 0.05) for k in range(150)]
    frames += [(3000 + k * 20, 0.40 + 0.04 * math.sin(k / 3.0), "sit", 0.1) for k in range(100)]
    assert claims(frames, v8_cfg()) == 1


def test_v5_and_unmeasured_walks_keep_proprio_s_standing_number() -> None:
    base = ProprioConfig()
    assert TABLE.stance_reference is not None
    assert (
        TABLE.stand_proprio_for(base, TABLE.stance_reference, "allcollisions", 20.0) is base
    )  # v5
    assert TABLE.stand_proprio_for(base, "0" * 64, None, 20.0) is base
    assert (
        TABLE.stand_proprio_for(base, V8_WALK.walk_sha256, "allcollisions", 20.0) is base
    )  # another body
    assert (
        TABLE.stand_proprio_for(base, V8_WALK.walk_sha256, V8_WALK.body, 0.0) is base
    )  # another delay
    v8 = TABLE.stand_proprio_for(base, V8_WALK.walk_sha256, V8_WALK.body, V8_WALK.delay_ms)
    assert v8.lift_load == V8_WALK.lift_load < base.lift_load
    assert {k: v for k, v in vars(v8).items() if k != "lift_load"} == {
        k: v for k, v in vars(base).items() if k != "lift_load"
    }  # only the standing claim's level moves


def test_a_v9_duck_sits_on_v8s_seat_and_stands_on_proprio_s_own_level() -> None:
    # policy set v9 = v7's walk (ce894d30…) with v8's sit-stand (a6f102d9…, the same file): the seat is v8's measured one;
    # v9's walk stood above 20 in every 0.3 s of its 1 h roam records (lowest 20.8 / 22.2; V9Check 2026-11-08), as v5's
    # did, so it has no stance entry and the standing claim keeps proprio's own level
    v9_walk = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"
    base = ProprioConfig()
    assert TABLE.seat_for(V8_SEAT.sitstand_sha256, "allcollisions_backlash", 20.0) == V8_SEAT
    assert TABLE.stance_for(v9_walk, "allcollisions_backlash", 20.0) is None
    assert TABLE.stand_proprio_for(base, v9_walk, "allcollisions_backlash", 20.0) is base
