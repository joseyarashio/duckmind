"""duck-body as the rig's lockstep brain (duckmind.duckbody.brain): the same BodyLoop, fed doors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.brain import BodyBrain
from duckmind.duckbody.protocol import ActOutput, LookRequest


@dataclass
class Doors:
    t_ns: int
    state: dict[str, Any]
    tof: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    health: dict[str, Any] | None = None


def state(t_ns: int, load: float = 18.0) -> dict[str, Any]:
    currents = [0.0] * 15
    currents[0] = currents[10] = load
    return {
        "t_ns": t_ns,
        "policy": "walk",
        "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
        "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
        "joints": [0.0] * 5 + [0.3491, 0.3491, 0.0, 0.0] + [0.0] * 6,
        "velocities": [0.0] * 15,
        "currents_ma": currents,
        "odom": {"position": [0.0, 0.0, 0.117], "yaw": 0.0},
        "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
    }


class Looker:
    def __init__(self, cmd: Any, t_ms: int) -> None:
        self.command, self.act_id, self.asked = cmd, cmd.act_id, False

    def step(self, ctx: Any) -> ActOutput:
        if not self.asked:
            self.asked = True
            return ActOutput(looks=(LookRequest("L1", ("light", "floor"), ctx.t_ns + 5),))
        if any(r.get("look_id") == "L1" for r in ctx.eye):
            body = {"kind": "sense.light", "level": "dim", "luma": 80, "noise_q": 5}
            return ActOutput(percepts=(body,), done=True)
        return ActOutput()

    def preempt(self, why: str, ctx: Any) -> ActOutput:
        return ActOutput(done=True)


def test_the_brain_runs_the_loop_and_orders_looks_with_live_poses() -> None:
    loop = BodyLoop(BodyConfig(executors={"duck.look": lambda c, t, lp: Looker(c, t)}))
    brain = BodyBrain(loop)
    t0 = 10**10
    for k in range(10):
        out = brain.tick(Doors(t0 + k * 20_000_000, state(t0 + k * 20_000_000)))
    acks = brain.command(
        {"type": "cmd", "cmd": "duck.look", "act_id": "K1", "kind": "scan", "deadline_ms": 99_999},
        t0 + 200_000_000,
    )
    assert acks[0]["ok"] is True
    out = brain.tick(Doors(t0 + 220_000_000, state(t0 + 220_000_000)))
    assert len(out.looks) == 1
    order = out.looks[0]
    assert order.look["type"] == "eye.look" and order.look["tasks"] == ["light", "floor"]
    assert order.look["pose_tap_ref"] == brain.loop.tap.ring[-1].seq
    before = len(order.poses)
    brain.tick(Doors(t0 + 240_000_000, state(t0 + 240_000_000)))
    brain.tick(Doors(t0 + 260_000_000, state(t0 + 260_000_000)))
    assert len(order.poses) > before  # read when the rig captures, not when ordered
    assert all(p["type"] == "eye.pose" for p in order.poses)
    result = {
        "type": "eye.result",
        "look_id": "L1",
        "status": "ok",
        "results": {"light": {"level": "dim", "luma": 80, "noise_q": 5}},
    }
    out = brain.tick(Doors(t0 + 280_000_000, state(t0 + 280_000_000), results=[result]))
    kinds = [p["kind"] for p in out.percepts]
    assert "sense.light" in kinds
    assert any(line["type"] == "done" and line["act_id"] == "K1" for line in out.lines)


def test_without_an_eye_a_look_is_none() -> None:
    brain = BodyBrain()
    got = brain.see({"look_id": "L9"}, {}, b"", [])
    assert got["status"] == "none" and got["look_id"] == "L9"


def test_health_becomes_percepts() -> None:
    brain = BodyBrain()
    out = brain.tick(
        Doors(10**10, state(10**10), health={"healthy": True, "battery": {"percent": 64.0}})
    )
    assert [p["kind"] for p in out.percepts] == ["body.battery"]
