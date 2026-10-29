"""The 示範 demonstrations (duckmind.simweb.demo): 撞撞看, 啄一下, 踢一下, truth-positioned, never
Duck Core's. Driven against a stand-in world whose duck walks as told (no MuJoCo)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from duckmind.simweb import demo
from duckmind.simweb.demo import DemoRunner


class FakeWorld:
    """x_world and x_place_prop for one duck and one thing; commands move them."""

    def __init__(
        self, thing_at: tuple[float, float] = (1.5, 0.8), *, place_ok: bool = True
    ) -> None:
        self.duck = [0.0, 0.0, 0.0]  # x, y, yaw (m, rad)
        self.thing = list(thing_at)
        self.policy = "sit"
        self.place_ok = place_ok
        self.placed: list[tuple[int, int]] = []
        self.skills: list[str] = []

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        if op == "x_world":
            return {
                "duck": {
                    "pos_mm": [int(self.duck[0] * 1000), int(self.duck[1] * 1000), 120],
                    "yaw_mrad": int(self.duck[2] * 1000),
                    "tilt_mrad": 20,
                },
                "things": [
                    {
                        "id": "ball_felt",
                        "pos_mm": [int(self.thing[0] * 1000), int(self.thing[1] * 1000), 0],
                    }
                ],
            }
        if op == "x_place_prop":
            if not self.place_ok:
                return {"ok": False, "why_zh": ["放下去會卡進 home_sofa_seat"]}
            self.placed.append((params["x_mm"], params["y_mm"]))
            self.thing = [params["x_mm"] / 1000, params["y_mm"] / 1000]
            return {"ok": True, "pos_mm": [params["x_mm"], params["y_mm"], 1]}
        raise RuntimeError(op)

    def apply(self, command: dict[str, Any], dt: float) -> None:
        if command.get("posture") == "stand":
            self.policy = "walk"
        if "move" in command:
            vx, _, vyaw = command["move"]
            self.duck[2] += vyaw * dt
            self.duck[0] += vx * dt * math.cos(self.duck[2])
            self.duck[1] += vx * dt * math.sin(self.duck[2])
            if vx and self.kind == "bump":
                ahead = math.dist(self.duck[:2], self.thing)
                if ahead < 0.15:  # walked into it: pushed along
                    self.thing[0] += vx * dt * math.cos(self.duck[2])
                    self.thing[1] += vx * dt * math.sin(self.duck[2])
        if "skill" in command:
            self.skills.append(command["skill"])
            self.thing[0] += 0.6 * math.cos(self.duck[2])
            self.thing[1] += 0.6 * math.sin(self.duck[2])

    kind = ""


def _drive(runner: DemoRunner, world: FakeWorld, limit_s: float = 60.0) -> dict[str, Any]:
    world.kind = runner.kind
    t = 0.0
    while t < limit_s:
        command = runner.tick(t, policy=world.policy)
        if "done" in command:
            return command["done"]
        world.apply(command, 0.1)
        t = round(t + 0.1, 3)
    raise AssertionError("the demo did not end")


def test_a_kick_stands_up_walks_over_puts_the_ball_at_the_kicking_foot_and_kicks():
    world = FakeWorld()
    runner = DemoRunner("kick", "ball_felt", world=world)
    result = _drive(runner, world)
    states = [entry["state"] for entry in result["log"]]
    assert states == ["stand", "approach", "settle", "act", "watch"]
    assert result["ok"] and result["demo"] and result["zh"] == "示範：踢一下"
    assert result["skill"] in ("kick_left", "kick_right") and world.skills == [result["skill"]]
    assert result["moved_mm"] >= 500 and result["outcome"] == "rolled" and result["duck_upright"]
    # the ball was set 9 cm ahead and 4.2 cm toward the kicking foot (truth positioning)
    x, y = world.placed[0]
    duck = runner.before["duck"]
    ahead, left = DemoRunner._local(duck, [x, y])
    sign = 1 if result["skill"] == "kick_left" else -1
    assert ahead == pytest.approx(0.09, abs=0.002) and left == pytest.approx(
        sign * 0.042, abs=0.002
    )


def test_a_peck_puts_the_thing_under_the_beak_and_runs_ground_pick():
    world = FakeWorld(thing_at=(0.3, 0.1))
    world.policy = "walk"
    runner = DemoRunner("peck", "ball_felt", world=world)
    result = _drive(runner, world)
    assert result["skill"] == "ground_pick" and world.skills == ["ground_pick"]
    assert [e["state"] for e in result["log"]][:2] == ["approach", "settle"]  # already near
    ahead, left = DemoRunner._local(runner.before["duck"], list(world.placed[0]))
    assert ahead == pytest.approx(demo.PECK_SPOT_M, abs=0.002) and abs(left) < 0.002


def test_a_bump_walks_slowly_into_the_thing():
    world = FakeWorld(thing_at=(0.2, 0.0))
    world.policy = "walk"
    result = _drive(DemoRunner("bump", "ball_felt", world=world), world)
    assert result["skill"] is None and world.skills == []
    assert (
        result["moved_mm"] and result["moved_mm"] > 100 and result["outcome"] in ("moved", "rolled")
    )


def test_a_demo_that_cannot_set_up_stops_and_says_why():
    world = FakeWorld(place_ok=False)
    world.policy = "walk"
    result = _drive(DemoRunner("kick", "ball_felt", world=world, approach=False), world)
    assert not result["ok"] and result["outcome"] == "stopped"
    assert "home_sofa_seat" in result["why_zh"][0]
    with pytest.raises(ValueError):
        DemoRunner("juggle", "ball_felt", world=world)
    with pytest.raises(ValueError):
        DemoRunner("kick", "ball_felt", world=world, side="middle")


def test_run_drives_a_runner_to_its_end():
    world = FakeWorld(thing_at=(0.2, 0.0))
    world.policy = "walk"
    clock = [0.0]
    sent: list[dict[str, Any]] = []

    def send(command: dict[str, Any]) -> None:
        sent.append(command)
        world.apply(command, 0.1)

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    result = demo.run(
        DemoRunner("kick", "ball_felt", world=world),
        policy=lambda: world.policy,
        send=send,
        clock=lambda: clock[0],
        sleep=sleep,
    )
    assert result["ok"] and sent[-1]["done"] == result
