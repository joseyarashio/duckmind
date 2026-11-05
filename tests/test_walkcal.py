"""The loaded walk policy's map (duckmind.neuro.walkcal, 2026-11-05): the brain was tuned on walk policy v5, which turns
about half of what it is told; v8 turns 1.1-1.3 times it. The body link sends the loaded walk the command that answers as
v5 answered the brain's twist (measured tables keyed by the walk file's sha256); v5 itself gets exactly today's commands."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import walkcal
from duckmind.neuro.bodylink import DEFAULT, BodyLink, BodyRequest
from duckmind.simweb import neuro_live

TABLE = walkcal.load()
V5 = TABLE.walks[TABLE.reference]
V8 = next(w for w in TABLE.walks.values() if w.name.startswith("v8"))
DT = 0.02

# what the brain asks for while walking: cruise, steering, arcs that turn away, the tightest arcs
REQUESTS = [
    (vx, vyaw)
    for vx in (0.35, 0.40, 0.45, 0.50, 0.55)
    for vyaw in (-1.0, -0.9, -0.5, -0.28, -0.1, 0.0, 0.1, 0.28, 0.5, 0.9, 1.0)
]


def _state(t: float, applied: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> RobotState:
    joints = [0.0] * 15
    for i, home in enumerate((0.3491, 0.3491, 0.0, 0.0)):
        joints[5 + i] = home
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = 4.5
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": "walk",
            "move": {"requested": list(applied), "applied": list(applied)},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
    )


def _sent(link: BodyLink, vx: float, vyaw: float, *, t0: float = 0.0) -> tuple[float, float]:
    """The twist the link streams for a steady walk request (after it settles)."""
    twist = None
    for k in range(10):
        out = link.tick(_state(t0 + k * DT), BodyRequest(vx=vx, vyaw=vyaw, posture="stand"))
        moves = [i.twist for i in out.intents if i.method == "robot.move"]
        twist = moves[-1] if moves else twist
    assert twist is not None
    return twist


def _link(walk: walkcal.WalkMap | None) -> BodyLink:
    link = BodyLink(replace(DEFAULT, command_delay_ms=20.0))
    link.walk = walk
    return link


def _v8_map() -> walkcal.WalkMap:
    walk = TABLE.map_for(V8.walk_sha256, V8.body, 20.0)
    assert walk is not None
    return walk


def test_the_table_is_measured_for_both_walks_on_their_own_bodies() -> None:
    assert V5.body == "allcollisions" and V8.body == "allcollisions_backlash"
    assert V5.walk_sha256 != V8.walk_sha256
    # what the tables say about the walks (the reason for the map): v5 turns about half of the command, v8 more than it
    assert V5.at(0.40, 1.0)[1] < 0.75 and V5.at(0.40, -1.0)[1] > -1.0
    assert V8.at(0.40, 1.0)[1] > 1.0 and V8.at(0.40, -1.0)[1] < -1.0


def test_v5_gets_exactly_the_commands_it_always_did(monkeypatch: pytest.MonkeyPatch) -> None:
    # the driver finds no map for the reference walk, on any body
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V5.walk_sha256)
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision="allcollisions")
    assert neuro_live._walk_map(body, 20.0) is None
    assert neuro_live._walk_map(replace_ns(body, collision="allcollisions_backlash"), 20.0) is None
    assert neuro_live._walk_map(None, 20.0) is None
    # … so a v5 link sends what a link without the map sends, and the brain reads the state as robotd sent it
    today, v5 = _link(None), _link(neuro_live._walk_map(body, 20.0))
    for k, (vx, vyaw) in enumerate(REQUESTS):
        assert _sent(v5, vx, vyaw, t0=k) == _sent(today, vx, vyaw, t0=k)
    state = _state(0.0, (0.31, 0.0, 0.42))
    assert v5.planned_state(state) is state


def replace_ns(ns: SimpleNamespace, **kw: object) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(ns), **kw})


def test_the_map_of_a_walk_onto_itself_changes_nothing() -> None:
    same = walkcal.WalkMap(V5, V5)
    for vx, vyaw in REQUESTS:
        assert same.command(vx, vyaw) == pytest.approx((vx, vyaw), abs=1e-3)


def test_a_v8_link_sends_the_turn_the_brain_wants_not_double() -> None:
    today, v8 = _link(None), _link(_v8_map())
    for k, (vx, vyaw) in enumerate(REQUESTS):
        designed = _sent(
            today, vx, vyaw, t0=k
        )  # what v5 is sent today (with the servo delay's compensation)
        sent = _sent(v8, vx, vyaw, t0=k)
        want_speed, want_rate = V5.at(*designed)  # what the brain's twist did on v5
        speed, rate = V8.at(*sent)  # what the sent command does on v8
        assert rate == pytest.approx(want_rate, abs=0.03), (vx, vyaw, designed, sent)
        if 0.30 < sent[0] < 0.55:  # (inside v8's reach the path speed matches too)
            assert speed == pytest.approx(want_speed, abs=0.006), (vx, vyaw, designed, sent)
        # proprio's own-turn model reads the twist the brain meant, not the one v8 was sent
        assert v8.walk_sent == (sent, designed)
    # the failure it fixes: today's command on v8 turns about twice as hard as on v5
    designed = _sent(today, 0.45, 0.5)
    assert V8.at(*designed)[1] > 1.8 * V5.at(*designed)[1]
    assert abs(V8.at(*_sent(v8, 0.45, 0.5, t0=99.0))[1]) < 1.15 * abs(V5.at(*designed)[1])


def test_the_brain_reads_the_body_twist_in_the_units_it_plans_in() -> None:
    walk = _v8_map()
    link = _link(walk)
    for vx, vyaw in REQUESTS:
        sent = walk.command(vx, vyaw)
        if not 0.30 < sent[0] < 0.55:
            continue  # (beyond v8's reach — v5 walks faster at the top — the body did less than the brain meant)
        state = _state(0.0, (sent[0], 0.0, sent[1]))
        seen = link.planned_state(state)
        assert (seen.applied[0], seen.applied[2]) == pytest.approx((vx, vyaw), abs=0.03)
        assert (seen.requested[0], seen.requested[2]) == pytest.approx((vx, vyaw), abs=0.03)
        assert seen.odom_yaw == state.odom_yaw and seen.policy == state.policy


def test_commands_outside_the_table_go_out_as_they_are() -> None:
    walk = _v8_map()
    for twist in ((0.0, 0.0), (0.2, 0.5), (-0.4, 1.0), (0.6, 0.3)):
        assert walk.command(*twist) == twist
    v8 = _link(walk)
    # a docking reverse and a pause are not mapped
    out = v8.tick(_state(0.0), BodyRequest(vx=-0.4, vyaw=0.3, posture="stand", docking=True))
    assert [i.twist for i in out.intents if i.method == "robot.move"] in ([], [(-0.4, 0.3)])


def test_the_driver_maps_only_a_measured_walk_on_its_own_body_and_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision=V8.body)
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V8.walk_sha256)
    walk = neuro_live._walk_map(body, 20.0)
    assert walk is not None and walk.loaded == V8 and walk.reference == V5
    assert neuro_live._walk_map(body, 0.0) is None  # (measured with the simulator's 20 ms delay)
    assert neuro_live._walk_map(replace_ns(body, collision="allcollisions"), 20.0) is None
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: "0" * 64)
    assert neuro_live._walk_map(body, 20.0) is None  # an unmeasured walk: as it is


def test_the_table_file_is_well_formed() -> None:
    for w in TABLE.walks.values():
        assert len(w.speed) == len(w.vxs) and all(len(r) == len(w.vyaws) for r in w.speed)
        assert all(math.isfinite(v) and v >= 0.0 for r in w.speed for v in r)
        assert all(math.isfinite(v) for r in w.yaw for v in r)
