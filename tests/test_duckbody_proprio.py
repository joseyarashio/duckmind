"""Body sense (duckmind.duckbody.proprio): lifts (owner B3/B4), falls (B1), contacts.

The lift cases replay real signals from the phase-3 headless trials (the world's hand lifting a
standing, a walking and a seated duck; a duck pressing into a wall with no lift):
``tests/fixtures/phase3_body/lift_signals.json`` — robot.state-shaped signals only (policy, fall
flag, the left/right leg-load sums of the body server's |force|×100 currents, gravity, gyro,
the commanded twist) plus the hand's truth for the assertions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from duckmind.duckbody.proprio import Proprio
from duckmind.duckbody.robotd_client import RobotState

FIXTURE = Path(__file__).parent / "fixtures" / "phase3_body" / "lift_signals.json"
POLICY = {0: "held", 1: "rise", 2: "walk", 3: "stand", 4: "sit", 9: "other"}


def tick(
    t: float,
    *,
    policy: str = "walk",
    fallen: bool = False,
    load: tuple[float, float] = (18.0, 18.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0),
    gyro: tuple[float, float, float] = (0.3, 0.2, 0.1),
    cmd: tuple[float, float] = (0.0, 0.0),
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    picked_up: bool = False,
) -> RobotState:
    currents = [0.0] * 15
    currents[0], currents[10] = load
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [cmd[0], 0.0, cmd[1]], "applied": [0, 0, 0]},
            "safety": {"fallen": fallen, "limp": False, "gravity": list(gravity),
                       "picked_up": picked_up},  # fmt: skip
            "joints": [0.0] * 15,
            "velocities": [0.0] * 15,
            "currents_ma": currents,
            "odom": {"position": [odom[0], odom[1], 0.117], "yaw": odom[2]},
            "imu": {"gyro": list(gyro), "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def replay(name: str) -> tuple[list[tuple[float, str, bool]], dict]:
    data = json.loads(FIXTURE.read_text())[name]
    p = Proprio()
    edges = []
    for i, t in enumerate(data["t"]):
        state = tick(
            t,
            policy=POLICY[data["pol"][i]],
            fallen=bool(data["fal"][i]),
            load=tuple(data["lr"][i]),
            gravity=tuple(data["g"][i]),
            gyro=tuple(data["gy"][i]),
            cmd=(data["cmd"][i][0], data["cmd"][i][1]),
        )
        for edge in p.step(state).edges:
            edges.append((t, edge, bool(data["held"][i])))
    return edges, data


@pytest.mark.parametrize("name", ["stand", "walk", "sit"])
def test_a_real_lift_is_claimed_while_held_and_set_down_after(name: str) -> None:
    edges, data = replay(name)
    held_t = [t for t, h in zip(data["t"], data["held"], strict=True) if h]
    lifted = [t for t, e, held in edges if e == "lifted"]
    assert lifted, edges
    assert held_t[0] <= lifted[0] <= held_t[-1]
    assert lifted[0] - held_t[0] < 3.0  # stand ≈ 0.5 s, sit ≈ 0.7 s, walking ≈ 2.5 s
    assert all(held for _, e, held in edges if e == "lifted")  # never claimed on the floor
    downs = [t for t, e, _ in edges if e == "set_down" and t > lifted[0]]
    assert downs and downs[0] >= held_t[-1] - 1.0


def test_a_walking_lift_first_stops_without_claiming() -> None:
    edges, _ = replay("walk")
    names = [e for _, e, _ in edges if e in ("unloaded", "lifted")]
    assert names[:2] == ["unloaded", "lifted"]


def test_a_wall_press_stops_but_claims_nothing() -> None:
    edges, data = replay("press")
    assert not any(data["held"])
    assert any(e == "unloaded" for _, e, _ in edges)
    assert not any(e == "lifted" for _, e, _ in edges)


def test_a_standing_duck_on_the_floor_is_never_lifted() -> None:
    p = Proprio()
    for k in range(1500):  # 30 s standing still, load ≈ 36
        out = p.step(tick(k * 0.02, load=(18.2, 18.0), gyro=(0.01, 0.01, 0.0)))
        assert "lifted" not in out.edges and "unloaded" not in out.edges


# ── falls: trigger A, trigger B, one episode per fall ─────────────────────────


def tilted(deg: float) -> tuple[float, float, float]:
    r = math.radians(deg)
    return (math.sin(r), 0.0, -math.cos(r))


def test_trigger_a_and_flicker_is_one_fall() -> None:
    p = Proprio()
    edges = []
    for k in range(400):
        t = k * 0.02
        fallen = 1.0 <= t < 3.0 and int(t * 10) % 3 != 0  # the flag flickers
        tilt = 70.0 if 1.0 <= t < 3.0 else 5.0
        out = p.step(tick(t, fallen=fallen, gravity=tilted(tilt)))
        edges += [(round(t, 2), e, out.fell_trigger) for e in out.edges if e in ("fell", "up")]
    assert [e for _, e, _ in edges] == ["fell", "up"]
    assert edges[0][2] == "fallen"
    assert edges[1][0] == pytest.approx(4.0, abs=0.05)  # up at 3.0 s, held 1 s


def test_trigger_b_nose_down_at_43_degrees() -> None:
    p = Proprio()
    fired = None
    for k in range(200):
        t = k * 0.02
        out = p.step(tick(t, gravity=tilted(43.0 if t >= 1.0 else 3.0)))
        if "fell" in out.edges:
            fired = (t, out.fell_trigger)
            break
    assert fired is not None and fired[1] == "tilt40"
    assert fired[0] == pytest.approx(2.0, abs=0.03)


def test_a_short_lean_is_not_a_fall() -> None:
    p = Proprio()
    for k in range(300):
        t = k * 0.02
        out = p.step(tick(t, gravity=tilted(45.0 if 1.0 <= t < 1.9 else 3.0)))
        assert "fell" not in out.edges


# ── contacts ─────────────────────────────────────────────────────────────────


def walk_ticks(
    p: Proprio,
    seconds: float,
    speed: float,
    *,
    t0: float = 0.0,
    x0: float = 0.0,
    yaw: float = 0.0,
    gyro: tuple[float, float, float] = (0.6, 0.4, 0.3),
) -> list[str]:
    edges: list[str] = []
    for k in range(int(seconds / 0.02)):
        t = t0 + k * 0.02
        x = x0 + speed * (t - t0)
        out = p.step(tick(t, cmd=(0.4, 0.0), load=(50.0, 50.0), gyro=gyro, odom=(x, 0.0, yaw)))
        edges += list(out.edges)
    return edges


def test_blocked_on_odometry_alone_after_two_seconds() -> None:
    p = Proprio()
    assert "blocked" not in walk_ticks(p, 3.0, 0.2)  # walking freely
    edges = walk_ticks(p, 3.0, 0.0, t0=3.0, x0=0.6)  # the odometry stops: a wall
    assert "blocked" in edges


def test_no_odometry_no_blocked() -> None:
    p = Proprio()
    edges = walk_ticks(p, 5.0, 0.0)  # the headless path: odometry never moves
    assert "blocked" not in edges


def test_a_stumble_needs_a_fast_turn_and_doubled_load() -> None:
    p = Proprio()
    for k in range(100):
        p.step(tick(k * 0.02, cmd=(0.4, 0.0), load=(25.0, 25.0), gyro=(0.5, 0.3, 0.2)))
    out = p.step(tick(2.02, cmd=(0.4, 0.0), load=(70.0, 60.0), gyro=(3.5, 1.0, 0.5)))
    assert "stumbled" in out.edges and "load" in out.evidence
    p2 = Proprio()
    for k in range(100):
        p2.step(tick(k * 0.02, cmd=(0.4, 0.0), load=(25.0, 25.0), gyro=(0.5, 0.3, 0.2)))
    out = p2.step(tick(2.02, cmd=(0.4, 0.0), load=(26.0, 26.0), gyro=(3.5, 1.0, 0.5)))
    assert "stumbled" not in out.edges  # a fast turn alone is not a trip


def test_the_quick_turn_s_own_spin_is_not_a_stumble_but_a_tip_still_is() -> None:
    """Live :8796 (2026-10-24): 22 of 29 'stumbled' edges came at the end of a quick turn on the spot
    (robotd's quick_turn_* skill, its hand-back skill_abort): the spin and the loaded legs are the
    turn itself. A trunk that really tips is still a trip."""
    p = Proprio()
    for k in range(100):
        p.step(tick(k * 0.02, policy="walk", load=(25.0, 25.0), gyro=(0.5, 0.3, 0.2)))
    for k in range(20):
        p.step(tick(2.0 + k * 0.02, policy="quick_turn_right", load=(30.0, 30.0), gyro=(0.2, 0.1, -2.0)))
    out = p.step(tick(2.42, policy="skill_abort", load=(70.0, 60.0), gyro=(3.5, 1.0, -0.5)))
    assert "stumbled" not in out.edges
    tipped = p.step(
        tick(2.44, policy="walk", load=(70.0, 60.0), gyro=(3.5, 1.0, -0.5), gravity=(0.0, -0.5, -0.85))
    )
    assert "stumbled" in tipped.edges and "tilt" in tipped.evidence


def test_a_long_quick_turn_on_the_spot_is_not_blocked_but_a_walk_that_stops_still_is() -> None:
    """(FlipRoom, LEARN-TRICKS §4.5) a turn once round on the spot (robotd's quick_turn_* skill while the brain asks its tight
    arc's walk): the odometry does not go forward by design, and 1.5-1.9 s into it a 'blocked' edge came every time, a bump
    0.12 m ahead in the wall memory and "no room" for the flip on open floor. A walk that stops against a wall still is."""
    p = Proprio()
    walk_ticks(p, 2.0, 0.2)
    edges: list[str] = []
    for k in range(150):
        t = 2.0 + k * 0.02
        out = p.step(tick(t, policy="quick_turn_left", cmd=(0.37, 0.9), load=(50.0, 50.0), gyro=(0.3, 0.2, 2.0),
                          odom=(0.4, 0.0, 0.9 * (t - 2.0))))
        edges += list(out.edges)
    assert "blocked" not in edges and "brushed" not in edges
    assert "blocked" in walk_ticks(p, 3.0, 0.0, t0=6.0, x0=0.4)


def _quick_turn_then_walk(graze_at: float | None = None) -> list[tuple[float, str]]:
    """A duck walking at 0.31 (the brain's pivot: its tight arc's walk is what robotd last got), robotd's quick turn right
    for 0.9 s (the heading spins at -2.6 rad/s, the trunk stays on the spot), then the walk handed back going straight on
    (its heading answers the command as the gait does, 0.2 s later). ``graze_at`` (s after the turn): a wall on its left
    pushes it 25° right over 0.3 s while it keeps walking. The edges after the turn, (s after the turn, edge)."""
    p = Proprio()
    gait = p.cfg
    x = y = yaw = 0.0
    edges = []
    end = 2.9
    for k in range(int(6.0 / 0.02)):
        t = k * 0.02
        turning = 2.0 <= t < end
        cmd = (0.31, -0.59) if t < end else (0.31, -0.03)
        if turning:
            yaw -= 2.6 * 0.02
            x += 0.01 * 0.02
        else:
            yaw += gait.expected_yaw_rate(0.31, cmd[1]) * 0.02
            x += 0.15 * 0.02 * math.cos(yaw)
            y += 0.15 * 0.02 * math.sin(yaw)
        if graze_at is not None and end + graze_at <= t < end + graze_at + 0.3:
            yaw -= math.radians(25.0) / 0.3 * 0.02
        out = p.step(tick(t, policy="quick_turn_right" if turning else "walk", cmd=cmd, load=(50.0, 50.0),
                          gyro=(0.3, 0.2, -2.6 if turning else 0.1), odom=(x, y, yaw)))
        if t >= end:
            edges += [(round(t - end, 2), e) for e in out.edges]
    return edges


def test_the_quick_turns_own_spin_is_not_a_brush_once_the_walk_is_back_but_a_graze_after_it_still_is() -> None:
    """V9Follow (2026-11-09): on v9 the brain's pivots are robotd's quick turns; proprio's 1 s brush window still held the
    turn's spin 0.6-1.0 s after the hand-back (own_turn_s is 0.6 s) and judged it against the walk's own turn — a 'brushed'
    edge right after the turn (live v9 roam: 4 of 33 turns, 0.76-0.94 s after them). The spin is the duck's own turn; a
    wall that pushes it aside after the turn is still a graze."""
    assert [e for e in _quick_turn_then_walk() if e[1] == "brushed"] == []
    grazed = [e for e in _quick_turn_then_walk(graze_at=1.5) if e[1] == "brushed"]
    assert grazed and 1.5 <= grazed[0][0] <= 2.0


# the brain re-steers every few hundred ms (wall check, 2026-11-06: 82 of 111 forward 'brushed' came within 1 s of a turn-command
# change, at no wall); the walk answers each command 0.2 s later (v5, fitted on the live roam tick logs)
STEERING = (-0.48, 0.30, -0.20, 0.50, -0.60, 0.10)


def _steer(t: float) -> float:
    return STEERING[int(t / 0.6) % len(STEERING)]


def _steered_walk(graze_at: float | None = None, seconds: float = 8.0) -> list[tuple]:
    """A duck walking at 0.4 while the brain re-steers every 0.6 s; its heading answers each command as the gait does
    (bias + gain × vyaw) 0.2 s later. ``graze_at``: a wall on its right pushes it 25° further left over 0.3 s from then
    while it keeps walking. The edges (t, edge, side)."""
    p = Proprio()
    gait = p.cfg
    x = y = yaw = 0.0
    edges = []
    for k in range(int(seconds / 0.02)):
        t = k * 0.02
        yaw += gait.expected_yaw_rate(0.4, _steer(max(0.0, t - 0.2))) * 0.02
        if graze_at is not None and graze_at <= t < graze_at + 0.3:
            yaw += math.radians(25.0) / 0.3 * 0.02
        x += 0.15 * 0.02 * math.cos(yaw)
        y += 0.15 * 0.02 * math.sin(yaw)
        state = tick(
            t, cmd=(0.4, _steer(t)), load=(50.0, 50.0), gyro=(0.6, 0.4, 0.3), odom=(x, y, yaw)
        )
        out = p.step(state)
        edges += [(round(t, 2), e, out.contact_side) for e in out.edges]
    return edges


def test_a_walk_the_brain_re_steers_is_not_read_as_a_graze() -> None:
    # the latest command times the whole 1 s window read the heading the last commands turned as a swing the command
    # does not explain: 'brushed' after every big change of the turn command
    assert [e for e in _steered_walk() if e[1] in ("brushed", "blocked")] == []


@pytest.mark.parametrize("graze_at", [3.1, 3.5, 4.3])
def test_a_graze_while_the_brain_re_steers_is_still_caught(graze_at: float) -> None:
    # 0.1 s after a turn-command change, mid-command, and 0.1 s after another change
    hits = [e for e in _steered_walk(graze_at) if e[1] == "brushed"]
    assert hits and graze_at < hits[0][0] <= graze_at + 1.0, hits
    assert hits[0][2] == "right"  # pushed left: the touch was on its right
    assert len(hits) == 1


def _reverse_start(dip_load: float, dip_at: float, dip_s: float, cmd_vx: float = -0.40) -> list:
    """A duck standing, then a reverse commanded at t = 1 s whose leg load dips to ``dip_load``
    for ``dip_s`` starting ``dip_at`` s into the reverse (the 2026-09-28 rig traces: the load
    slides from ~46 to 20.4–23.9 around 0.45 s, then the first backward step loads it again)."""
    p = Proprio()
    edges = []
    for k in range(250):
        t = k * 0.02
        cmd = (cmd_vx, 0.0) if t >= 1.0 else (0.0, 0.0)
        into = t - 1.0
        load = dip_load if 0 <= into - dip_at < dip_s else 36.0
        state = tick(t, load=(load / 2, load / 2), cmd=cmd, gyro=(0.3, 0.2, 0.1))
        edges += [(round(t, 2), e) for e in p.step(state, cmd=cmd).edges]
    return edges


def test_a_reverse_start_is_not_a_lift_but_a_real_pickup_while_reversing_is() -> None:
    """Owner B3's unsure mark (24) with the reverse start's own dip: 9 of 24 cornered back steps
    read "unsure" and never backed (2026-09-28). During a reverse's first 1.2 s only a held
    duck's load (≈ 9–13, under 18) counts; after the spin-up, and for forward walks, 24 again."""
    assert not [e for _, e in _reverse_start(20.4, 0.40, 0.20) if e == "unloaded"]
    assert not [e for _, e in _reverse_start(23.0, 0.35, 0.30) if e == "unloaded"]
    held = [t for t, e in _reverse_start(11.0, 0.30, 0.60) if e == "unloaded"]
    assert held and held[0] <= 1.0 + 0.30 + 0.2  # a pickup mid-spin-up is still sensed at once
    late = [t for t, e in _reverse_start(21.0, 1.40, 0.40) if e == "unloaded"]
    assert late  # past the spin-up the usual mark holds
    forward = [t for t, e in _reverse_start(21.0, 0.40, 0.25, cmd_vx=0.40) if e == "unloaded"]
    assert forward  # forward walks keep the usual mark


def test_robotds_own_pickup_pause_is_a_lift_and_its_resume_a_set_down() -> None:
    # (real duck prep 2026-11-08) a real board runs robotd's pick-up detector by default (0.15.2+):
    # held, the policy reads "held" and the legs hold the pause pose, so the legs' load cannot claim
    # the lift; robotd's safety.picked_up does. Before: the claim stayed "ground" the whole hold.
    p = Proprio()
    edges: list[tuple[float, str]] = []
    still = (0.01, 0.01, 0.01)
    for i in range(100):  # standing on the floor 2 s
        edges += [(i * 0.02, e) for e in p.step(tick(i * 0.02, policy="stand", gyro=still)).edges]
    assert edges == []
    for i in range(100, 250):  # in a hand 3 s: robotd paused (a tilt there is no fall)
        state = tick(i * 0.02, policy="held", picked_up=True, load=(4.0, 4.0),
                     gravity=tilted(50.0), gyro=(0.8, 0.5, 0.2))  # fmt: skip
        out = p.step(state)
        edges += [(i * 0.02, e) for e in out.edges]
        assert out.view.lifted and not out.view.fallen
    assert [e for _, e in edges] == ["lifted"] and edges[0][0] == 2.0
    for i in range(250, 450):  # set down: robotd hands the policy back, the legs carry it
        edges += [(i * 0.02, e) for e in p.step(tick(i * 0.02, policy="stand", gyro=still)).edges]
    down = [t for t, e in edges if e == "set_down"]
    assert len(down) == 1 and 5.0 + 1.5 <= down[0] <= 5.0 + 3.0
