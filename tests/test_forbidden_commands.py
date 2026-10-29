"""Forbidden commands (spec §5.3; owner A2, A3): property tests over random intents and states.

Seeded random draws (no new dependency) drive :func:`duckmind.duckbody.reflex.check_intent`
directly and through the whole :class:`~duckmind.duckbody.bodyloop.BodyLoop`, and a static
check keeps robotd intents flowing through the guard only.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.protocol import ActOutput, Intent
from duckmind.duckbody.reflex import GuardContext, Limits, check_intent, check_world_op
from duckmind.duckbody.robotd_client import RobotState

SRC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "duckbody"
N = 20_000


def random_context(rng: random.Random) -> GuardContext:
    return GuardContext(
        head_home_ms=rng.choice([0, 100, 299, 300, 5000]),
        act=rng.choice([None, "duck.walk", "duck.curio", "duck.dock", "duck.posture"]),
        approach=rng.random() < 0.3,
        head_down_ok=rng.random() < 0.3,
        back_arc_ok=rng.random() < 0.3,
        back_step_ok=rng.random() < 0.3,
        lifted=rng.random() < 0.1,
        unloaded=rng.random() < 0.1,
        fallen=rng.random() < 0.1,
        righting=rng.random() < 0.1,
        capturing=rng.random() < 0.1,
        moving=rng.random() < 0.5,
        head_cmd_home=rng.random() < 0.8,
        head_off_mrad=rng.choice([0, 160, 396, 450, 451, 900]),
    )


def random_intent(rng: random.Random) -> Intent:
    kind = rng.random()
    if kind < 0.7:
        vx = rng.choice(
            [0.0, rng.uniform(-0.8, 0.8), rng.choice([0.29, 0.3, 0.4, 0.401, 0.55, -0.35])]
        )
        vyaw = rng.choice([0.0, rng.uniform(-2.0, 2.0), rng.choice([1.0, -1.0, 1.001])])
        vy = 0.0 if rng.random() < 0.9 else rng.uniform(-0.2, 0.2)
        return Intent.move(vx, vyaw, vy)
    if kind < 0.85:
        return Intent.do(
            rng.choice(["sit_toggle", "ground_pick", "kick_left", "kick_right", "roulade", "x"])
        )
    return Intent(
        rng.choice(
            [
                "robot.enable",
                "robot.gains",
                "robot.look",
                "robot.stop",
                "robot.head",
                "robot.reboot_motors",
                "robot.servo",
                "robot.load_policy",
            ]
        ),
        {},
    )


@pytest.mark.parametrize("vx_max", [400, 550])
def test_no_forbidden_intent_ever_passes(vx_max: int) -> None:
    rng = random.Random(20260927 + vx_max)
    limits = Limits(vx_max_mm=vx_max)
    passed = 0
    for _ in range(N):
        ctx, intent = random_context(rng), random_intent(rng)
        why = check_intent(intent, ctx, limits)
        if why is not None:
            continue
        passed += 1
        method = intent.method
        assert method in {"robot.move", "robot.do", "robot.stop", "robot.head", "robot.mouth"}
        if method == "robot.do":
            skill = intent.params["skill"]
            assert skill in {"sit_toggle", "ground_pick", "kick_left", "kick_right"}
            assert skill == "sit_toggle" or ctx.act == "duck.curio"
            unsure = ctx.lifted or ctx.unloaded or ctx.fallen or ctx.righting
            assert skill == "sit_toggle" or not unsure
        if method != "robot.move":
            continue
        vx = round(intent.params["vx"] * 1000)
        vyaw = round(intent.params["vyaw"] * 1000)
        vy = round(intent.params["vy"] * 1000)
        if (vx, vyaw, vy) == (0, 0, 0):
            continue  # stopping is always allowed
        assert vy == 0
        assert vx != 0, "turning in place"
        assert abs(vyaw) <= 1000
        assert abs(vx) >= 300, "velstand does not step under 0.30"
        assert abs(vx) <= (400 if ctx.approach else vx_max)
        # reversing: curiosity's back_arc, a duck.dock act (owner 2026-09-28 「1. 可以」) or a
        # granted cornered back step (owner 「2. 可以」) — nothing else, ever
        assert (
            vx > 0
            or (ctx.back_arc_ok and ctx.act == "duck.curio")
            or ctx.act == "duck.dock"
            or (ctx.back_step_ok and ctx.act == "duck.walk")
        )
        assert vx >= -400, "reversing faster than the measured −0.40"
        assert not (ctx.lifted or ctx.unloaded or ctx.fallen or ctx.righting or ctx.capturing)
        a2 = ctx.head_down_ok and ctx.approach and abs(vx) <= 400  # owner A2's final approach
        if not ctx.moving and ctx.head_home_ms < 300:
            assert a2  # rule 3: a walk starts only with the head settled at HOME
        if ctx.moving and not (ctx.head_cmd_home and ctx.head_off_mrad <= 450):
            assert a2  # a walking head swings with the gait, but is never sent away
    assert passed > N // 10  # the property was exercised


def test_zero_moves_and_stop_always_pass() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        ctx = random_context(rng)
        assert check_intent(Intent.move(0.0, 0.0), ctx) is None
        assert check_intent(Intent.stop(), ctx) is None


def test_the_named_cases() -> None:
    home = GuardContext(head_home_ms=1000, act="duck.walk")
    assert check_intent(Intent.move(0.0, 0.5), home)  # turning in place
    assert check_intent(Intent.move(0.2, 0.0), home)  # does not step
    assert check_intent(Intent.move(0.55, 0.0), home) is None  # owner A3: normal walks at 0.55
    assert check_intent(Intent.move(0.6, 0.0), home)  # over the measured cap
    assert check_intent(Intent.move(0.45, 0.0), home, Limits(vx_max_mm=400))  # the spec's cap
    approach = GuardContext(head_home_ms=1000, act="duck.walk", approach=True)
    assert check_intent(Intent.move(0.45, 0.0), approach, Limits(vx_max_mm=550))  # A3: 0.40
    assert check_intent(Intent.move(0.4, 1.2), home)
    assert check_intent(Intent.move(-0.35, 0.3), home)  # reversing
    back = GuardContext(head_home_ms=1000, act="duck.curio", back_arc_ok=True)
    assert check_intent(Intent.move(-0.35, 0.3), back) is None
    head_away = GuardContext(head_home_ms=0, act="duck.walk")
    assert check_intent(Intent.move(0.4, 0.0), head_away)
    a2 = GuardContext(head_home_ms=0, act="duck.walk", approach=True, head_down_ok=True)
    assert check_intent(Intent.move(0.4, 0.0), a2) is None
    assert check_intent(Intent("robot.look", {"x": 1.0}), home)
    assert check_intent(Intent.do("kick_left"), home)
    assert check_intent(Intent.do("kick_left"), GuardContext(act="duck.curio")) is None


def test_reversing_inside_duck_dock_and_the_cornered_back_step() -> None:
    """Owner 2026-09-28 「1. 可以」: a ``duck.dock`` act may reverse (both dock styles do), at
    −0.30…−0.40 m/s only. Owner 「2. 可以」: a ``duck.walk`` may reverse only while the body loop
    says its granted back step is under 0.2 m (``back_step_ok``); no other act, ever."""
    dock = GuardContext(head_home_ms=1000, act="duck.dock", moving=True)
    assert check_intent(Intent.move(-0.40, 0.08), dock) is None
    assert check_intent(Intent.move(-0.30, 0.0), dock) is None
    assert check_intent(Intent.move(-0.45, 0.0), dock)  # faster than the measured −0.40
    assert check_intent(Intent.move(-0.25, 0.0), dock)  # velstand does not step under 0.30
    assert check_intent(Intent.move(0.0, -0.5), dock)  # still never turns in place
    walk = GuardContext(head_home_ms=1000, act="duck.walk")
    assert check_intent(Intent.move(-0.40, 0.0), walk)  # a plain walk never reverses …
    granted = GuardContext(head_home_ms=1000, act="duck.walk", back_step_ok=True)
    assert check_intent(Intent.move(-0.40, 1.0), granted) is None  # … a granted back arc may
    assert check_intent(Intent.move(-0.45, 0.0), granted)
    for act in ("duck.look", "duck.posture", None):  # the flag means nothing to another act
        assert check_intent(Intent.move(-0.40, 0.0), GuardContext(act=act, back_step_ok=True))
    assert check_intent(
        Intent.move(-0.40, 0.0), GuardContext(head_home_ms=1000, act="duck.curio")
    )  # a curiosity act without its back_arc
    lifted = GuardContext(head_home_ms=1000, act="duck.dock", lifted=True)
    assert check_intent(Intent.move(-0.40, 0.0), lifted)  # the body's states still hold
    unsettled = GuardContext(head_home_ms=100, act="duck.dock")
    assert check_intent(Intent.move(-0.40, 0.0), unsettled)  # rule 3 before any leg starts


def test_the_world_port_only_pg() -> None:
    assert check_world_op("pg") is None
    for op in ("x_world", "x_hand", "x_place", "tof", "x_push", None):
        assert check_world_op(op) is not None


# ── through the whole body loop ───────────────────────────────────────────────


class Chaos:
    """An executor that asks for random things every tick."""

    def __init__(self, cmd, rng: random.Random) -> None:
        self.command = cmd
        self.act_id = cmd.act_id
        self.rng = rng

    def step(self, ctx):
        r = self.rng
        twist = (r.choice([0, 100, 300, 400, 550, 900, -350]), r.choice([0, 500, 1000, 1500, -800]))
        head = None if r.random() < 0.5 else (r.uniform(-0.5, 0.5), 0.0, r.uniform(-0.5, 0.5), 0.0)
        do = r.choice([None, None, None, "sit_toggle", "kick_left", "roulade"])
        return ActOutput(twist=twist, head=head, do=do, done=r.random() < 0.01)

    def preempt(self, why, ctx):
        return ActOutput(done=True)


def state(t: float, regime: dict) -> RobotState:
    """One tick of a 2-second regime: normal standing/walking, lifted, fallen, head away."""
    currents = [0.0] * 15
    currents[0] = currents[10] = regime["load"]
    joints = [0.0] * 15
    joints[5:9] = [0.3491 + regime["head"], 0.3491, 0.0, 0.0]
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(t * 1e9),
            "policy": regime["policy"],
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": regime["fallen"], "limp": False, "gravity": regime["gravity"]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": currents,
            "imu": {"gyro": [regime["gyro"], 0.0, 0.0], "quat": [1, 0, 0, 0]},
        }
    )


def regime(rng: random.Random) -> dict:
    kind = rng.choice(["normal", "normal", "normal", "lifted", "fallen", "head_away", "sit"])
    return {
        "load": {"lifted": 5.0}.get(kind, rng.choice([30.0, 45.0, 60.0])),
        "head": 0.3 if kind == "head_away" else 0.0,
        "policy": {"sit": "sit"}.get(kind, "walk"),
        "fallen": kind == "fallen",
        "gravity": [0.8, 0.0, -0.6] if kind == "fallen" else [0.0, 0.0, -1.0],
        "gyro": rng.choice([0.0, 0.5, 1.0]),
    }


def test_the_body_loop_never_emits_a_forbidden_intent() -> None:
    rng = random.Random(99)
    cfg = BodyConfig(
        executors={
            "duck.walk": lambda cmd, t_ms, loop: Chaos(cmd, rng),
            "duck.curio": lambda cmd, t_ms, loop: Chaos(cmd, rng),
        }
    )
    loop = BodyLoop(cfg)
    emitted = 0
    current = regime(rng)
    for k in range(15000):
        t = k * 0.02
        if k % 100 == 0:
            current = regime(rng)
        if k % 50 == 0:
            name = rng.choice(["duck.walk", "duck.curio"])
            obj = {
                "type": "cmd",
                "cmd": name,
                "act_id": f"A{k}",
                "deadline_ms": 10**9,
                "vx_mm": 400,
                "phase": "step",
                "preempt": True,
            }
            loop.command(obj, 10**10 + int(t * 1e9))
        s = state(t, current)
        tick = loop.tick(s)
        guard = loop.last_guard  # the context the loop checked this tick's intents against
        for intent in tick.intents:
            emitted += 1
            vx = round(intent.params.get("vx", 0.0) * 1000) if intent.method == "robot.move" else 0
            vyaw = (
                round(intent.params.get("vyaw", 0.0) * 1000) if intent.method == "robot.move" else 0
            )
            if intent.method == "robot.move" and (vx, vyaw) != (0, 0):
                assert check_intent(intent, guard, cfg.limits) is None
                assert vx != 0 and 300 <= abs(vx) <= cfg.limits.vx_max_mm and abs(vyaw) <= 1000
                assert abs(vx) <= 400 or not guard.approach
            assert intent.method in {"robot.move", "robot.head", "robot.do"}
            if intent.method == "robot.do":
                assert intent.params["skill"] in {"sit_toggle", "kick_left"}
    assert emitted > 100 and loop.refused_total > 100


def test_only_the_server_holds_a_robotd_client_and_only_the_loop_sends() -> None:
    """Static: RobotdClient is built in server.py only; the intents it sends come out of
    BodyLoop.tick, whose every intent went through BodyLoop._send → check_intent."""
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name == "RobotdClient":
                    assert path.name == "server.py", f"{path.name} builds a robotd client"
    loop_src = (SRC / "bodyloop.py").read_text(encoding="utf-8")
    assert (
        loop_src.count("out.intents.append(") == 2
    )  # the guarded one and the zero after a refusal
    assert "check_intent(intent, guard, self.cfg.limits)" in loop_src
