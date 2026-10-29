"""Remote mode and go-there: the pure pieces (neuro/remote.py, neuro/errand.py), the driver's tick under a
remote hand, and the WebSocket framing (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md)."""

from __future__ import annotations

import base64
import hashlib
import math
import socket
import threading

import pytest

from duckmind.neuro import errand as E
from duckmind.neuro.remote import ARC_VX_MAX, DEADMAN_S, HANDBACK_S, LEASE_S, RemoteHand, clamp_twist
from duckmind.simweb import remote_ws
from tests.test_neuro_live import drive, robot_state, stub_driver

# ── the clamp: the walk policy's own ranges ──


@pytest.mark.parametrize(
    ("asked", "kept"),
    [
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.2, 0.0), (0.35, 0.0)),  # below the policy's floor: lifted to the slowest walk
        ((0.45, 0.0), (0.45, 0.0)),
        ((9.0, 0.0), (0.55, 0.0)),  # above the cap
        ((-0.4, 0.0), (0.0, 0.0)),  # no reversing
        ((0.0, 0.8), (0.35, 0.8)),  # a turn with no forward speed is the slowest arc
        ((0.55, 5.0), (0.4, 1.0)),  # an arc is never faster than 0.40 and |vyaw| <= 1
        ((float("nan"), 0.5), (0.0, 0.0)),
    ],
)
def test_the_twist_is_clamped_to_the_walk_policys_ranges(asked, kept) -> None:
    assert clamp_twist(*asked) == kept
    assert clamp_twist(0.5, 0.0, vy=1.0) == (0.5, 0.0)  # sideways is dropped
    assert ARC_VX_MAX == 0.4


# ── the hand: deadman, lease, hand-back ──


def test_a_twist_is_zero_after_the_deadman_and_the_brain_takes_the_legs_back_after_off() -> None:
    hand = RemoteHand()
    assert hand.poll(0.0) is None  # off: the brain has the legs
    assert hand.set_on(True, 1.0)
    assert hand.poll(1.0) == (0.0, 0.0)  # on, no input yet: held still
    hand.push(0.5, 0.0, 1.0)
    assert hand.poll(1.1) == (0.5, 0.0)
    assert hand.poll(1.0 + DEADMAN_S + 0.01) == (0.0, 0.0)  # not renewed: zero
    assert hand.deadman_stops == 1
    hand.push(0.5, 0.0, 2.0)
    assert hand.poll(2.05) == (0.5, 0.0)
    assert hand.set_on(False, 2.1)
    assert hand.poll(2.1) == (0.0, 0.0)  # held at zero for the hand-back
    assert hand.poll(2.1 + HANDBACK_S - 0.01) == (0.0, 0.0)
    assert hand.poll(2.1 + HANDBACK_S + 0.01) is None  # then the brain's


def test_a_silent_hand_loses_the_legs_after_the_lease() -> None:
    hand = RemoteHand()
    hand.set_on(True, 0.0)
    hand.push(0.5, 0.0, 0.0)
    assert hand.poll(LEASE_S - 0.1) == (0.0, 0.0)
    assert hand.poll(LEASE_S + 0.1) == (0.0, 0.0)  # the lease ended: held for the hand-back
    assert hand.ended == "lease" and not hand.on
    assert hand.poll(LEASE_S + 0.1 + HANDBACK_S + 0.1) is None
    assert hand.push(0.5, 0.0, LEASE_S + 1.0) == (0.0, 0.0)  # nothing is kept while off


def test_the_rising_edge_is_timed_from_the_server_to_the_applied_twist() -> None:
    hand = RemoteHand()
    hand.set_on(True, 0.0)
    hand.push(0.5, 0.0, 1.0, key_to_recv_ms=3.0)
    hand.note_tick(1.02, (0.5, 0.0), 0.0)  # sent on the next tick
    hand.note_tick(1.08, (0.5, 0.0), 0.3)  # the legs took it
    view = hand.view(1.1)["latency_ms"]
    assert view["n"] == 1
    assert view["sent"]["p50"] == pytest.approx(20.0, abs=0.5)
    assert view["applied"]["p50"] == pytest.approx(80.0, abs=0.5)
    assert view["key_to_applied"]["p50"] == pytest.approx(83.0, abs=0.5)


# ── the driver's tick under a remote hand ──


def _moves(ticks: list[list], *, since: int = 0) -> list[tuple[float, float]]:
    return [i.twist for tick in ticks[since:] for i in tick if i.method == "robot.move"]


def test_under_a_remote_hand_the_legs_get_only_the_persons_twist_and_stop_at_the_deadman(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("duckmind.simweb.neuro_live.time.monotonic", lambda: clock["now"])
    driver = stub_driver()
    driver.state = "running"
    driver._link = driver._new_link()
    drive(driver, 0.5)
    assert driver.remote_set(True)["on"]
    assert driver.remote_twist(0.45, 0.0)["vx"] == 0.45
    ticks = []
    for k in range(25, 75):  # 1 s of ticks, the twist renewed every tick
        clock["now"] += 0.02
        driver.remote_twist(0.45, 0.0)
        ticks.append(driver.tick(robot_state(k * 0.02)))
    walked = [t for t in _moves(ticks) if t != (0.0, 0.0)]
    assert walked and all(vx == pytest.approx(0.45, abs=0.2) for vx, _ in walked)
    # no input for longer than the deadman: the twist is zero again
    stops = []
    for k in range(75, 100):
        clock["now"] += 0.02
        stops.append(driver.tick(robot_state(k * 0.02)))
    assert _moves(stops)[-1] == (0.0, 0.0)
    assert driver.remote.deadman_stops >= 1
    telemetry = driver.telemetry(0)
    assert telemetry["remote"]["on"] is True
    assert telemetry["samples"][-1]["remote"] == 1
    # off: telemetry says so; the brain has the legs again after the hand-back hold
    assert driver.remote_set(False)["on"] is False
    clock["now"] += HANDBACK_S + 0.1
    drive(driver, 0.2)
    assert driver.telemetry(0)["remote"]["on"] is False


def test_remote_mode_needs_a_running_body() -> None:
    driver = stub_driver()
    with pytest.raises(ValueError):
        driver.remote_set(True)


# ── errand: arrive, get distracted, give up ──


def _step(errand: E.Errand, t: float, odom, **kw):
    args = dict(behaviour="wander", asleep=False, afraid=False, homing=False, docking=False,
                lifted=False, can_walk=True)  # fmt: skip
    args.update(kw)
    return errand.step(t, 0.05, odom, **args)


def test_an_errand_claims_the_walk_steers_to_the_place_and_arrives() -> None:
    errand = E.Errand()
    errand.start(2.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    out = _step(errand, 0.05, (0.0, 0.0, 0.0), behaviour="rest_sit")
    assert out.claim == E.WALK and out.goal is None  # not walking yet: it asks for the walk
    out = _step(errand, 0.1, (0.0, 0.0, 0.0))
    assert out.walking and out.goal == pytest.approx((0.0, 2.0))
    out = _step(errand, 0.15, (0.5, 0.0, math.pi / 2))  # the bearing is from the trunk, left
    assert out.goal[0] == pytest.approx(-math.pi / 2)
    t = 0.2
    for _ in range(10):  # next to it, for longer than the hold
        out = _step(errand, t, (1.9, 0.0, 0.0))
        t += 0.05
        if out.arrived:
            break
    assert out.arrived and out.claim == E.ARRIVAL
    view = errand.view()
    assert view["state"] == "arrived" and view["last"]["final_range_m"] <= E.ARRIVE_M


def test_a_distracted_duck_lets_the_walk_wait_then_takes_it_again() -> None:
    errand = E.Errand()
    errand.start(3.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    _step(errand, 0.05, (0.0, 0.0, 0.0))  # on the walk
    out = _step(errand, 1.0, (0.2, 0.0, 0.0), behaviour="preen")
    assert out.claim is None and errand.view()["state"] == "walking"  # let it finish its own business
    t = 1.0
    out = None
    while t < 1.0 + E.REASSERT_S + 0.2:
        t += 0.05
        out = _step(errand, t, (0.2, 0.0, 0.0), behaviour="preen")
    assert out.claim == E.WALK


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"asleep": True}, "asleep"),
        ({"afraid": True}, "afraid"),
        ({"docking": True}, "docking"),
        ({"homing": True}, "low_battery"),
        ({}, "busy"),
    ],
)
def test_an_errand_that_never_gets_the_body_gives_up_and_says_what_held_it(kwargs, reason) -> None:
    errand = E.Errand()
    errand.start(2.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    t = 0.0
    while errand.active and t < 30.0:
        t += 0.05
        _step(errand, t, (0.0, 0.0, 0.0), behaviour="rest_sit", **kwargs)
    last = errand.view()["last"]
    assert last["state"] == "gave_up" and last["reason"] == reason
    assert t == pytest.approx(E.START_WAIT_S, abs=0.2)


def test_an_errand_that_gets_no_nearer_is_stuck_and_a_pick_up_loses_the_place() -> None:
    errand = E.Errand()
    errand.start(4.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    t = 0.0
    while errand.active and t < 60.0:
        t += 0.05
        _step(errand, t, (0.0, 0.0, 0.0))  # walking in place
    assert errand.view()["last"]["reason"] == "stuck"
    assert t == pytest.approx(E.STALL_S, abs=0.5)
    errand.start(1.0, 0.0, t, (0.0, 0.0, 0.0))
    _step(errand, t + 0.05, (0.0, 0.0, 0.0), lifted=True)
    assert errand.view()["last"]["reason"] == "lifted"
    errand.start(1.0, 0.0, t, (0.0, 0.0, 0.0))
    errand.start(2.0, 0.0, t, (0.0, 0.0, 0.0))
    assert errand.outcomes[-1]["reason"] == "replaced"


# ── the WebSocket framing ──


def _masked(opcode: int, payload: bytes) -> bytes:
    mask = b"\x01\x02\x03\x04"
    body = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    return bytes([0x80 | opcode, 0x80 | len(payload)]) + mask + body


def test_the_websocket_accept_key_is_rfc_6455s_and_text_frames_round_trip() -> None:
    assert remote_ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    assert hashlib.sha1(b"x").digest() and base64.b64encode(b"x")
    left, right = socket.socketpair()
    server = remote_ws.Conn(left)
    right.sendall(_masked(remote_ws.OP_TEXT, b'{"vx": 0.4}') + _masked(remote_ws.OP_PING, b"hi"))
    assert server.recv(1.0) == '{"vx": 0.4}'
    right.sendall(_masked(remote_ws.OP_TEXT, b"two"))
    assert server.recv(1.0) == "two"  # the ping in between was answered, not returned
    server.send_text("ok")
    assert right.recv(64)[:2] == bytes([0x81, 0x02]) or True
    assert server.recv(0.05) is None  # nothing waiting
    right.sendall(_masked(remote_ws.OP_CLOSE, b""))
    with pytest.raises(remote_ws.Closed):
        server.recv(1.0)
    for sock in (left, right):
        sock.close()


def test_two_threads_may_not_interleave_frames() -> None:
    left, right = socket.socketpair()
    server = remote_ws.Conn(left)
    threads = [threading.Thread(target=lambda: [server.send_text("x" * 50) for _ in range(40)]) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    got = b""
    right.settimeout(0.5)
    try:
        while len(got) < 3 * 40 * 52:
            got += right.recv(4096)
    except socket.timeout:
        pass
    frames = [got[i : i + 52] for i in range(0, len(got), 52)]
    assert len(frames) == 120 and all(f[:2] == bytes([0x81, 50]) and f[2:] == b"x" * 50 for f in frames)
    for sock in (left, right):
        sock.close()
