"""Getting up after a fall (duckmind.duckbody.righting; spec §5.4, owner B1/B2)."""

from __future__ import annotations

import math

from duckmind.duckbody.proprio import Proprio
from duckmind.duckbody.righting import Righting, RightingConfig
from duckmind.duckbody.robotd_client import RobotState


def tick(
    t: float,
    *,
    tilt: float = 3.0,
    fallen: bool = False,
    policy: str = "walk",
    current: float = 18.0,
) -> RobotState:
    r = math.radians(tilt)
    currents = [0.0] * 15
    currents[0] = currents[10] = current
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {
                "fallen": fallen,
                "limp": False,
                "gravity": [math.sin(r), 0.0, -math.cos(r)],
            },
            "joints": [0.0] * 15,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def run(script, seconds: float, cfg: RightingConfig | None = None, current=None):
    """``script(t, sit_toggles)`` → (tilt, fallen, policy); returns events and robot.do calls.
    ``current(t)`` → the leg motors' current (default: a quiet 18)."""
    p, r = Proprio(), Righting(cfg or RightingConfig())
    events, dos = [], []
    for k in range(int(seconds / 0.02)):
        t = k * 0.02
        tilt, fallen, policy = script(t, dos)
        state = tick(
            t,
            tilt=tilt,
            fallen=fallen,
            policy=policy,
            current=18.0 if current is None else current(t),
        )
        out = r.step(state, p.step(state), during="walk", ctx={"look_id": "L1"})
        events += [(round(t, 2), e) for e in out.events]
        if out.do:
            dos.append((round(t, 2), out.do))
    return events, dos, r


def test_a_quick_self_righting() -> None:
    def script(t, dos):
        return (70.0, True, "walk") if 1.0 <= t < 2.2 else (3.0, False, "walk")

    events, dos, r = run(script, 6.0)
    kinds = [e["kind"] for _, e in events]
    assert kinds == ["body.fell", "body.righting"]
    fell, righted = events[0][1], events[1][1]
    assert (
        fell["trigger"] == "fallen"
        and fell["during"] == "walk"
        and fell["ctx"] == {"look_id": "L1"}
    )
    assert righted["ok"] and righted["via"] == "self" and righted["how"] == "wait"
    assert righted["tries"] == 1 and not righted["hard"]
    assert abs(righted["ms"] - 1200) <= 40
    assert dos == []  # velstand was already driving: nothing to send


def test_a_seated_fall_hands_over_to_velstand() -> None:
    def script(t, dos):
        if 1.0 <= t < 2.0:
            return 70.0, True, "sit"
        return (3.0, False, "walk") if t >= 2.0 else (3.0, False, "sit")

    _, dos, _ = run(script, 5.0)
    assert dos[0] == (1.0, "sit_toggle")


def test_stuck_nose_down_is_found_by_trigger_b_and_rescued_by_sit_then_stand() -> None:
    """Owner B1 + B2: a duck stuck at 50° (robotd never says fallen) gets the second try."""

    def script(t, dos):
        toggles = len(dos)
        if t < 1.0:
            return 3.0, False, "walk"
        if toggles == 0:
            return 50.0, False, "walk"  # stuck nose-down: tilt 50°, no fallen flag
        if toggles == 1:
            return 30.0, False, "sit"  # sat down
        return 3.0, False, "walk"  # stood up again

    events, dos, r = run(script, 16.0)
    fell = events[0][1]
    assert fell["kind"] == "body.fell" and fell["trigger"] == "tilt40"
    assert events[0][0] == 2.0  # 1 s over 40°
    assert [d for _, d in dos] == ["sit_toggle", "sit_toggle"]
    assert dos[0][0] == 10.0 and dos[1][0] == 12.0  # at 8 s, then 2 s seated
    righted = events[-1][1]
    assert righted["ok"] and righted["how"] == "sitstand" and righted["tries"] == 2
    assert righted["hard"] is True
    assert r.last["trigger"] == "tilt40"


def test_not_up_after_both_tries_then_helped() -> None:
    def script(t, dos):
        if t < 1.0 or t >= 25.0:
            return 3.0, False, "walk"  # a hand rights it at 25 s
        return 80.0, True, "walk"

    events, dos, _ = run(script, 28.0)
    results = [e for _, e in events if e["kind"] == "body.righting"]
    assert results[0]["ok"] is False and results[0]["tries"] == 2
    assert results[1]["ok"] is True and results[1]["via"] == "help" and results[1]["hard"]
    assert [t for t, _ in dos] == [9.0, 11.0]


def test_with_sitstand_off_the_second_try_is_the_spec_wait() -> None:
    def script(t, dos):
        return (3.0, False, "walk") if t < 1.0 or t >= 16.0 else (80.0, True, "walk")

    events, dos, _ = run(script, 20.0, RightingConfig(sitstand=False))
    assert dos == []
    righted = [e for _, e in events if e["kind"] == "body.righting"][-1]
    assert righted["ok"] and righted["tries"] == 2 and righted["how"] == "wait"


def test_a_flickering_flag_is_one_fall() -> None:
    def script(t, dos):
        if 1.0 <= t < 3.0:
            return 65.0, int(t * 10) % 2 == 0, "walk"
        return 3.0, False, "walk"

    events, _, _ = run(script, 6.0)
    assert [e["kind"] for _, e in events] == ["body.fell", "body.righting"]


# ── waiting for help, with patience (bug 2, 2026-10-09) ───────────────────────────


def _stuck_until(sit_stands: int):
    """A body that stays on its side until the ``sit_stands``-th sit-stand (two toggles each) is done."""

    def script(t, dos):
        if t < 1.0:
            return 3.0, False, "walk"
        if len(dos) >= 2 * sit_stands:
            return 3.0, False, "walk"
        return 80.0, True, "walk"

    return script


def test_from_waiting_help_the_sit_stand_is_tried_again_every_30_s_and_a_late_one_works() -> None:
    """The body fails the first two sit-stands (try 2 and the first retry) and gets up on the third.
    Before the fix the duck waited for a hand for ever after try 2."""
    events, dos, r = run(_stuck_until(3), 140.0)
    toggles = [t for t, _ in dos]
    # try 2: sit at 9 s, stand at 11 s; the wait begins at 20 s; retry 1 at 50 s (stands at 52, waits again at 61); retry 2 at 91 s
    assert toggles[:2] == [9.0, 11.0]
    assert 49.0 <= toggles[2] <= 52.0 and 90.0 <= toggles[4] <= 93.0
    assert len(toggles) == 6
    results = [e for _, e in events if e["kind"] == "body.righting"]
    assert [e["ok"] for e in results] == [False, True]  # (a failed retry tells nothing new)
    assert results[1]["via"] == "self" and results[1]["how"] == "sitstand"
    assert results[1]["tries"] == 4 and results[1]["hard"] is True
    assert r.phase == "idle"


def test_the_retries_stop_after_three_and_the_duck_waits_for_help_again() -> None:
    events, dos, r = run(_stuck_until(99), 400.0)
    assert len(dos) == 2 + 2 * 3  # try 2 and three retries, not a fourth
    assert r.phase == "waiting_help" and r.retries == 3
    results = [e for _, e in events if e["kind"] == "body.righting"]
    assert len(results) == 1 and results[0]["ok"] is False  # (one failure told, not four)

    # a hand still rights it in the end, and it says so
    def helped(t, dos):
        return (3.0, False, "walk") if t >= 300.0 else (80.0, True, "walk")

    events, _, _ = run(helped, 310.0)
    results = [e for _, e in events if e["kind"] == "body.righting"]
    assert results[-1]["ok"] is True and results[-1]["via"] == "help"


def test_after_an_overload_trip_there_is_no_retry() -> None:
    """The motors were past their stall: trying again would only heat them. The wait is for a hand."""

    def hot(t):
        return 90.0 if 12.0 <= t < 16.0 else 18.0  # in try 2's seated/rise phase

    events, dos, r = run(_stuck_until(99), 200.0, current=hot)
    results = [e for _, e in events if e["kind"] == "body.righting"]
    assert len(results) == 1 and results[0].get("overload") is True
    assert len(dos) == 2  # try 2 only
    assert r.phase == "waiting_help" and r.overloaded and r.retries == 0


def test_with_sitstand_off_or_retry_max_zero_it_waits_as_before() -> None:
    _, dos, r = run(_stuck_until(99), 200.0, RightingConfig(retry_max=0))
    assert len(dos) == 2 and r.phase == "waiting_help"
    _, dos, r = run(_stuck_until(99), 200.0, RightingConfig(sitstand=False))
    assert dos == [] and r.phase == "waiting_help"


def test_a_new_fall_while_waiting_starts_the_count_again() -> None:
    def script(t, dos):
        if t < 1.0 or 70.0 <= t < 71.0:
            return 3.0, False, "walk"
        return 80.0, True, "walk"

    _, dos, r = run(script, 200.0)
    # fall 1 at 1 s: try 2 at 9 s, retry at 50 s; up at 70 s; fall 2 at 71 s: try 2 at 79 s ...
    assert r.retries <= 3 and len(dos) >= 8
