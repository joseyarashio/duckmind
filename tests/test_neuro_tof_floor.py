"""A neural duck with the ToF module on open floor and at a wall (2026-11-05,
docs/design/neural-duck/TOF-FLOOR-2026-11-05.md).

The brain's depth summary reprojects every ToF zone with the trunk's height above the floor. A
seated or rising duck's trunk is 6-10 cm up, not the 12 cm of a standing one, and its head looks
down: taken at 12 cm, the floor 0.27-0.30 m ahead read as a thing 4-6 cm high, the near stop held
and for 5 s only arcs were let through — a straight remote walk went 5 cm in 5 s.

The body here is scripted (seated, the rise, standing) with the sensor poses, heights and gravity
the simulator's robotd reported during a real rise (port 8784, 2026-11-05); each ToF frame is the
exact range of every zone's beam to the floor (and a wall) for that pose. The legs follow the
twist the body link sends: the trunk moves by it, the odometry with it.
"""

from __future__ import annotations

import pytest

from duckmind.duckbody import tof_reproject as tr
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro.livestub import StubBrain
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec

DT = 0.02  # robot.state at 50 Hz
FRAME_S = 0.07  # tofd's ~14 frames/s
HEAD_JOINTS = (0.3491, 0.3491, 0.0, 0.0)
# (policy, trunk height m, ToF position in the trunk m, ToF quaternion wxyz, projected gravity) as robotd reported them
SEATED = (
    "sit",
    0.061,
    (0.0788, 0.0245, 0.0692),
    (0.9135, -0.0132, 0.4062, 0.0201),
    (-0.066, -0.003, -0.998),
)
RISING = (
    (
        "rise",
        0.065,
        (0.0864, 0.0182, 0.1105),
        (0.9886, -0.0172, 0.1445, -0.0387),
        (0.084, 0.007, -0.996),
    ),
    (
        "rise",
        0.084,
        (0.0900, 0.0139, 0.1059),
        (0.9814, -0.0359, 0.1729, -0.0757),
        (0.098, 0.0, -0.995),
    ),
)
STANDING = (
    "stand",
    0.1164,
    (0.0668, 0.0234, 0.1257),
    (0.9998, 0.0069, 0.0170, 0.0100),
    (0.053, 0.011, -0.999),
)
SIT_S, RISE_S = 0.4, 0.6


def _pose_at(t: float, walking: bool) -> tuple[str, float, tuple, tuple, tuple]:
    if t < SIT_S:
        return SEATED
    if t < SIT_S + RISE_S:
        return RISING[0] if t < SIT_S + RISE_S / 2 else RISING[1]
    policy, *rest = STANDING
    return ("walk" if walking else policy, *rest)  # type: ignore[return-value]


def _frame(t_ns: int, height: float, pos, quat, down, wall_ahead_m: float | None) -> dict:
    """Every zone's exact range to the floor (z = −height in the levelled trunk frame) and to a
    wall ``wall_ahead_m`` forward of the trunk origin; past 4 m no return."""
    level = tr.level_from_gravity(down)
    q = tr.Quat(*quat).normalized()
    s = level.rotate(pos)
    dist, status = [], []
    for beam in tr.BEAMS:
        d = level.rotate(q.rotate(beam))
        hits = []
        if d[2] < -1e-6:
            hits.append((s[2] + height) / -d[2])
        if wall_ahead_m is not None and d[0] > 1e-6:
            hits.append((wall_ahead_m - s[0]) / d[0])
        r = min((h for h in hits if h > 0.0), default=None)
        if r is None or r > 4.0:
            dist.append(0)
            status.append(255)
        else:
            dist.append(int(round(r * 1000.0)))
            status.append(5)
    return {"t_ns": t_ns, "distance_mm": dist, "status": status}


def _state(t: float, x: float, pose, load: float = 45.0) -> RobotState:
    policy, height, pos, quat, down = pose
    joints = [0.0] * 15
    joints[5:9] = HEAD_JOINTS
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = load / 10.0
    return RobotState.from_wire(
        {
            "t_ns": 5 * 10**12 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": list(down)},
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [x, 0.0, height], "yaw": 0.0},
            "frames": {"tof": {"pos": list(pos), "quat": list(quat)}},
        }
    )


def _remote_walk(
    monkeypatch,
    *,
    seconds: float = 5.0,
    tof: bool = True,
    wall_m: float | None = None,
    seated: bool = True,
) -> tuple[float, list[float]]:
    """A straight remote twist (vx 0.3, renewed every tick) from a seated (or standing) duck: how far
    the legs were sent (m) and the trunk's x every tick."""
    clock = {"now": 100.0}
    monkeypatch.setattr("duckmind.simweb.neuro_live.time.monotonic", lambda: clock["now"])
    driver = NeuroDriver(
        NeuroSpec(seed=7),
        lanes=Lanes(
            brain=StubBrain(seed=7), names={"brain": "stub", "senses": "stub", "motor": "stub"}
        ),
    )
    driver.state = "running"
    driver._link = driver._new_link()
    assert driver.remote_set(True)["on"]
    x, vx, walked, xs = 0.0, 0.0, 0.0, []
    t0 = 0.0 if seated else SIT_S + RISE_S
    next_frame = t0
    for k in range(int(seconds / DT)):
        t = t0 + k * DT
        clock["now"] += DT
        pose = _pose_at(t, vx != 0.0)
        state = _state(t, x, pose)
        if tof and t >= next_frame:
            next_frame += FRAME_S
            wall = None if wall_m is None else wall_m - x
            driver._tof_frame = _frame(state.t_ns, pose[1], pose[2], pose[3], pose[4], wall)
        driver.remote_twist(0.3, 0.0)
        for intent in driver.tick(state):
            if intent.method == "robot.move":
                vx = intent.twist[0]
        x += vx * DT
        walked += abs(vx) * DT
        xs.append(x)
    return walked, xs


def test_a_tof_duck_on_open_floor_walks_a_straight_remote_twist_as_far_as_one_without(
    monkeypatch,
) -> None:
    """From seated, 5 s of a straight twist at 0.3 m/s. Without a ToF the legs are sent 0.69 m (the rise and the start
    rule take ~2.7 s of the 5); the floor ahead is no obstacle, so a duck with the ToF is sent as far, within 10 %
    (before the fix: 0 m — the near stop at the rise, then 5 s of arcs only; live 0.04-0.06 m)."""
    plain, _ = _remote_walk(monkeypatch, tof=False)
    fitted, _ = _remote_walk(monkeypatch, tof=True)
    assert plain > 0.6
    assert fitted >= 0.9 * plain


def test_a_tof_duck_walking_straight_at_a_wall_stops_before_its_beak_reaches_it(
    monkeypatch,
) -> None:
    """Standing, a wall 0.8 m ahead: the straight twist stops at the near line and stays stopped — the
    trunk never gets within 0.15 m of the wall (the beak is ~0.10 m ahead of the trunk origin)."""
    walked, xs = _remote_walk(monkeypatch, tof=True, wall_m=0.8, seated=False, seconds=6.0)
    assert walked > 0.3  # it did walk toward it
    assert max(xs) < 0.8 - 0.15


@pytest.mark.parametrize("wall_m", [0.25, 0.35])
def test_a_tof_duck_rising_in_front_of_a_wall_is_still_held(monkeypatch, wall_m: float) -> None:
    """The trunk's real height is used for the floor only: a wall 0.25-0.35 m ahead of a duck getting up
    is a thing, and the straight twist takes the trunk no nearer to it than about the near line (0.30 m)."""
    _, xs = _remote_walk(monkeypatch, tof=True, wall_m=wall_m, seconds=3.0)
    assert max(xs) < wall_m - 0.15


@pytest.mark.parametrize(("wall_m", "near"), [(None, False), (0.25, True)])
def test_the_worlds_senses_do_not_take_the_floor_before_a_seated_duck_for_a_thing(
    wall_m, near
) -> None:
    """The world's own ToF door (a raw frame, no body-link depth) reads the same way: seated, the floor ahead
    is nothing in the way (before: a thing 0.17 m ahead); a wall 0.25 m ahead still is one."""
    from types import SimpleNamespace

    from duckmind.world import neuro_senses as ns

    hub = ns.SenseHub(seed=1)
    state = _state(0.0, 0.0, SEATED)
    raw = _frame(state.t_ns, SEATED[1], SEATED[2], SEATED[3], SEATED[4], wall_m)
    doors = SimpleNamespace(
        t_ms=state.t_ns // 1_000_000, state=state, link=None, events=(), depth=None,
        tof=raw, pg=None, day_minute=600.0, world=None,
    )  # fmt: skip
    tof = hub(doors).tof
    assert tof.present
    assert (tof.ahead_mm is not None and tof.ahead_mm <= 300) is near
