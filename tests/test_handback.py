"""Bug B (2026-10-09): a flight that ends on its own (the brain's walk request drops to zero for a pause) was handed back to v5
under a ZERO command at 0.4-0.6 m/s with the trunk leaned: live, 4 of 24 fresh boots fell right after a bang. The link now keeps
asking v5 for HANDBACK_VX for HANDBACK_S after such a flight (the panic gate's own hand-off check walks on at 0.40)."""

from __future__ import annotations

import dataclasses

import pytest
from test_neuro_bodylink import DT, moves, state
from test_panic_wiring import _do, _link, _tick

from duckmind.neuro.bodylink import HANDBACK_S, HANDBACK_VX, LEG_JOINTS, MIN_FLIGHT_S, BodyRequest
from duckmind.skilltrain import guard as G


def _flying_link(ticks: int = 55):
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    t += DT
    _tick(link, t, "walk", BodyRequest(vx=0.55, panic=True))
    for _ in range(ticks):
        t += DT
        _tick(link, t, "panic_run", BodyRequest(vx=0.55, panic=True))
    return link, t


def test_a_flight_that_ends_on_its_own_is_handed_back_at_a_walking_command_not_at_zero() -> None:
    link, t = _flying_link()
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest())  # the brain's walk request is zero
    assert _do(out) == [G.ABORT]
    assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))]
    seen = 0
    for _ in range(int(HANDBACK_S / DT) - 2):  # v5 is driving: the command goes on for HANDBACK_S
        t += DT
        out = _tick(link, t, "walk", BodyRequest())
        assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))]
        seen += 1
    assert seen > 10
    for _ in range(int(0.4 / DT)):  # then it stops as any walk stops
        t += DT
        out = _tick(link, t, "walk", BodyRequest())
    assert not moves(out.intents) or moves(out.intents) == [(0.0, 0.0)]


def test_a_stop_rule_or_a_fall_still_stops_it_at_once() -> None:
    link, t = _flying_link()
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest(), fallen=True)
    assert _do(out) == [G.ABORT]
    assert not [m for m in moves(out.intents) if m[0] > 0.0]
    t += DT
    out = _tick(link, t, "walk", BodyRequest(), fallen=True)
    assert not [m for m in moves(out.intents) if m[0] > 0.0]


def test_a_plain_walk_that_pauses_is_stopped_as_before() -> None:
    link = _link()
    t = 0.0
    for _ in range(60):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    for _ in range(10):
        t += DT
        out = _tick(link, t, "walk", BodyRequest())
    assert not [m for m in moves(out.intents) if m[0] > 0.0]


def test_a_flight_that_the_request_leaves_in_the_run_up_is_held_for_the_minimum_flight_time() -> (
    None
):
    """Live (fresh boots after a bang, 54 flights): the two flights handed back after 0.5-0.6 s at 0.2 m/s, in the run-up, fell; the 51
    that ran 1.1-1.4 s did not. The brain's request ending early (its stabilise reflex, a look) no longer ends a flight under MIN_FLIGHT_S."""
    link, t = _flying_link(
        ticks=10
    )  # panic_run has been running for 0.2 s when the request drops to zero
    held = 0
    while True:
        t += DT
        out = _tick(link, t, "panic_run", BodyRequest())
        if G.ABORT in _do(out):
            break
        assert _do(out) == ["panic_run"], _do(out)  # (asked again each tick: the skill is held)
        assert [m for m in moves(out.intents) if m[0] > 0.0] or held  # (and it keeps its twist)
        held += 1
        assert held < 100
    assert (
        0.75 <= held * DT <= MIN_FLIGHT_S
    )  # (the flight is about a second old when it is handed back)
    assert moves(out.intents) == [
        (pytest.approx(HANDBACK_VX), pytest.approx(0.0))
    ]  # …and then the coast of the hand-back


def test_the_minimum_flight_does_not_hold_a_flight_against_a_stop_rule() -> None:
    link, t = _flying_link(ticks=10)
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest(), fallen=True)
    assert _do(out) == [G.ABORT]
    assert not [m for m in moves(out.intents) if m[0] > 0.0]


def _fast_legs(t: float, policy: str, req: BodyRequest, link):
    """One tick whose ten leg joints swing at 11 rad/s (norm): what the legs of a flight that has just ended still do."""
    st = state(t, policy=policy)
    vel = [0.0] * 15
    for i in LEG_JOINTS:
        vel[i] = 11.0 / len(LEG_JOINTS) ** 0.5
    return link.tick(dataclasses.replace(st, velocities=tuple(vel)), req)


def test_the_legs_of_a_flight_that_just_ended_are_not_read_as_a_pick_up() -> None:
    """Live, a fresh boot after a bang: the flight's tempo (10.8 rad/s) 0.5 s after the hand-back read as "legs swinging free", the
    stop rule sat the duck down at 0.54 m/s and it tipped. Right after a flight the leg speed is not read; a real pick-up later is."""
    link, t = _flying_link()
    t += DT
    _tick(link, t, "panic_run", BodyRequest())  # the flight ends: skill_abort
    for _ in range(int(0.9 / DT)):
        t += DT
        out = _fast_legs(t, "walk", BodyRequest(), link)
        assert "airborne" not in out.gate
    assert not [e for e in out.events if e.get("kind") == "body.airborne"]
    for _ in range(
        int(2.0 / DT)
    ):  # a second after the flight a pick-up (legs swinging free in a walk policy) is read as before
        t += DT
        out = _fast_legs(t, "walk", BodyRequest(vx=0.4), link)
    assert "airborne" in out.gate


def test_the_legs_swinging_free_without_a_flight_are_still_a_pick_up() -> None:
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    for _ in range(10):
        t += DT
        out = _fast_legs(t, "walk", BodyRequest(vx=0.4), link)
    assert "airborne" in out.gate


def test_a_stumble_in_a_flight_hands_it_back_at_a_walking_command_and_does_not_stop_dead() -> None:
    """Live, 2 of 20 fresh boots after a bang: a stumble (a footfall's jolt, on the flight's own 11-15 degree lean) stopped the legs
    dead at 0.3-0.5 m/s and the duck tipped over. A stumble in a flight ends it (skill_abort); v5 walks on at HANDBACK_VX for HANDBACK_S."""
    link, t = _flying_link(ticks=20)
    t += DT
    out = _tick(
        link, t, "panic_run", BodyRequest(vx=0.55, panic=True), gyro=(3.5, 0.0, 0.0), load=120.0
    )
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest(vx=0.55, panic=True), tilt=14.0)
    assert _do(out) == [G.ABORT]
    assert "stumble" in out.gate
    assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))]
    for _ in range(int(HANDBACK_S / DT) - 3):
        t += DT
        out = _tick(link, t, "walk", BodyRequest(vx=0.55, panic=True), tilt=12.0)
        assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))]
    for _ in range(
        int(0.6 / DT)
    ):  # then the brain's flight (it still flees) goes on as any request
        t += DT
        out = _tick(link, t, "walk", BodyRequest(vx=0.55, panic=True))
    assert _do(out) in ([], ["panic_run"])


def test_a_stumble_in_a_plain_walk_still_stops_it() -> None:
    link = _link()
    t = 0.0
    for _ in range(60):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    t += DT
    first = _tick(link, t, "walk", BodyRequest(vx=0.4), gyro=(3.5, 0.0, 0.0), load=120.0, tilt=14.0)
    t += DT
    out = _tick(link, t, "walk", BodyRequest(vx=0.4), tilt=12.0)
    assert "stumble" in first.gate or "stumble" in out.gate
    assert not [m for m in moves(out.intents) if m[0] > 0.0]


def test_a_flight_whose_head_swings_out_of_the_walking_envelope_keeps_its_move_command() -> None:
    """Round 4 (2026-10-09), found in live traces (21 + 23 refused moves in 20 fresh boots after a bang): the panic gait swings the head
    far outside the walking envelope, the link lets the flight go on, but the move guard's "head must be at HOME and settled" rule
    refused each flight move (the link never says it is already moving) and zeroed the twist mid-flight: the speed decayed 0.53 -> 0.03
    m/s over 0.5 s, the trunk leaned, and some flights ended in a fall. A flight's move is not refused for the head."""
    wild = (-0.62, 0.0, -1.1, 0.0)  # what the panic gait does to the head at 1 m/s (measured)
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    t += DT
    _tick(link, t, "walk", BodyRequest(vx=0.55, panic=True))
    for _ in range(int(MIN_FLIGHT_S / DT) + 5):
        t += DT
        out = _tick(link, t, "panic_run", BodyRequest(vx=0.55, panic=True), head=wild)
        assert "refused" not in out.gate, out.refused
        assert "head" not in out.gate
    assert out.twist[0] > 0.0  # (the held flight's twist is still going out)
    # and the second after it: the head is still wide, the hand-back walks on at HANDBACK_VX instead of stopping on the head rule
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest(), head=wild)
    assert _do(out) == [G.ABORT]
    for _ in range(int(HANDBACK_S / DT) - 3):
        t += DT
        out = _tick(link, t, "walk", BodyRequest(), head=wild)
        assert "head" not in out.gate and "refused" not in out.gate, (out.gate, out.refused)
        assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))]


def test_robotds_skill_abort_label_is_not_a_posture_change_and_the_hand_back_walks_straight_on() -> (
    None
):
    """Round 4, live: after the abort robotd's policy label is `skill_abort` for a tick or two; the link read it as "not standing" (the
    "posture" gate), stopped the legs and wiped the hand-back window, and the duck then walked on at the flight's own arc (0.35 m/s,
    0.99 rad/s) with the trunk leaned 13 degrees: tilt 28, 59 degrees, a fall (3 of 20 fresh boots after a bang)."""
    link, t = _flying_link(ticks=20)
    arc = BodyRequest(vx=0.35, vyaw=0.99, panic=True)
    t += DT
    _tick(link, t, "panic_run", arc, gyro=(3.5, 0.0, 0.0), load=120.0)
    t += DT
    out = _tick(link, t, "panic_run", arc, tilt=13.0)
    assert _do(out) == [G.ABORT] and moves(out.intents) == [
        (pytest.approx(HANDBACK_VX), pytest.approx(0.0))
    ]
    for label in ("skill_abort", "skill_abort", "walk", "walk", "walk"):
        t += DT
        out = _tick(link, t, label, arc, tilt=15.0)
        assert "posture" not in out.gate, out.gate
        assert moves(out.intents) == [(pytest.approx(HANDBACK_VX), pytest.approx(0.0))], (
            label,
            out.gate,
        )
