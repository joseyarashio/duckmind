"""Cornered and the small back step (owner 2026-09-28 「2. 可以」; phase-3 verify finding (f):
within 30 cm of a small thing no move was allowed and the duck stayed stuck).

The tracker alone (:mod:`duckmind.duckbody.cornered`), then the whole body loop with walkexec on a
kinematic body: in front of a small thing the ToF stops every forward walk, straight or arc; once
its own senses have found all three ways shut, a ``duck.walk`` back step goes — at −0.40, never
past 0.2 m (the guard stops it there even if walkexec would go on) — and a walk forward goes again.
Nothing here is physics: the phase-3 gate harness is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from duckmind import percept
from duckmind.duckbody import walkexec as W
from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.cornered import HOLD_MS, Cornered, Spot
from duckmind.duckbody.protocol import DepthView
from duckmind.duckbody.reflex import BACK_STEP_MAX_MM
from duckmind.duckbody.robotd_client import RobotState

NS = 1_000_000_000
AT = Spot(0, 0, 0)


def tof(ahead: int | None = None, left: int | None = None, right: int | None = None) -> DepthView:
    return DepthView(0, ahead, left, right, 0, False, 0)


def walk_done(vx: int, vyaw: int, reason: str, path: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "walk.done",
        "cmd": {"vx": vx, "vyaw": vyaw, "max_ms": 3000},
        "odom": {"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": path},
        "reason": reason,
        **extra,
    }


# ── the tracker ─────────────────────────────────────────────────────────────


def test_the_tof_alone_shuts_ways_it_sees_near() -> None:
    c = Cornered()
    c.sense(0, AT, tof(ahead=250))
    assert c.shut(0, AT) == {"ahead": "tof"} and not c.now(0, AT)
    c.sense(20, AT, tof(ahead=250, left=300, right=340))
    assert c.now(20, AT)
    c2 = Cornered()
    c2.sense(0, AT, tof(ahead=310, left=300, right=300))  # 31 cm ahead: the near stop is 30
    assert not c2.now(0, AT)
    c2.sense(0, AT, DepthView(0, None, 300, 300, 0, True, 0))  # the ToF touch
    assert c2.now(0, AT)


def test_walks_stopped_at_once_shut_the_way_they_tried() -> None:
    """A small thing 25 cm ahead with open sides: every forward walk, straight or arc, is
    halted at once by the near stop — three tries and the duck is cornered."""
    c = Cornered()
    c.walked(0, AT, walk_done(400, 0, "reflex_near"))
    c.walked(10, AT, walk_done(400, 1000, "reflex_near", path=20))
    assert not c.now(10, AT)
    c.walked(20, AT, walk_done(400, -1000, "reflex_near"))
    assert c.now(20, AT) and set(c.shut(20, AT)) == {"ahead", "left", "right"}
    moved = Cornered()
    moved.walked(0, AT, walk_done(400, 1000, "reflex_near", path=400))  # went on, then stopped
    assert moved.shut(0, AT) == {"ahead": "walk:reflex_near"}
    clear = Cornered()
    clear.walked(0, AT, walk_done(400, 0, "time", path=30))  # a short clear walk shuts nothing
    clear.walked(0, AT, walk_done(400, 0, "veto_dark"))  # nor does a refusal
    clear.walked(0, AT, walk_done(-400, 0, "stall", contact={"kind": "blocked", "side": "unknown"}))
    assert clear.shut(0, AT) == {}


def test_contacts_and_looks_shut_their_side() -> None:
    c = Cornered()
    c.sense(0, AT, None, ("blocked",), "front")
    c.sense(0, AT, None, ("brushed",), "left")
    c.walked(
        0, AT, walk_done(400, 0, "time", path=500, contact={"kind": "brushed", "side": "right"})
    )
    assert c.now(0, AT)
    look = Cornered()
    fold = {"light": "bright", "free_mm": [300, 900, 280, 900, 320], "ends": ["wall"] * 5}
    look.looked(0, AT, fold)
    assert look.now(0, AT)
    unknown = Cornered()
    unknown.looked(0, AT, {**fold, "ends": ["unknown"] * 5})  # an unlearned eye ends nothing
    dark = Cornered()
    dark.looked(0, AT, {**fold, "light": "dark"})
    assert unknown.shut(0, AT) == {} and dark.shut(0, AT) == {}
    scan = Cornered()
    scan.scanned(
        0,
        AT,
        {
            "light": "dim",
            "sectors": [
                {"b_mrad": 0, "free_mm": 250, "end": "thing"},
                {"b_mrad": 480, "free_mm": 330, "end": "wall"},
                {"b_mrad": -420, "free_mm": 300, "end": "wall"},
            ],
        },
    )
    assert scan.now(0, AT)


def test_marks_hold_only_where_they_were_set() -> None:
    c = Cornered()
    c.sense(0, AT, tof(ahead=200, left=200, right=200))
    assert c.now(1000, Spot(60, 30, 200))  # swaying on the spot
    assert not c.now(1000, Spot(90, 0, 0))  # moved away
    assert not c.now(1000, Spot(0, 0, 400))  # turned more than 20°
    assert not c.now(HOLD_MS + 1, AT)  # a minute later: look again
    c.sync(1)  # a new odometry epoch (lifted, set down) forgets everything
    assert not c.now(0, AT) and not c.marks


# ── the body loop, walkexec and the guard on a kinematic body ─────────────────


@dataclass
class Kin:
    """A kinematic duck in front of a small thing ``thing_mm`` ahead of where it starts (on +x):
    it walks forward at 0.195 m/s for vx 0.40, backs at 0.15 m/s, turns as asked; the ToF sees
    the thing when the duck faces it."""

    thing_mm: int | None = 280
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vyaw: float = 0.0
    backed_max: float = 0.0
    reverse_sent: int = 0
    lines: list[dict[str, Any]] = field(default_factory=list)

    def state(self, t: float) -> RobotState:
        moving = self.vx != 0.0
        per = 11.0 if moving else 3.5
        currents = [per if i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14) else 0.0 for i in range(15)]
        joints = [0.0] * 15
        joints[5], joints[6] = 0.3491, 0.3491  # the head at HOME
        return RobotState.from_wire(
            {
                "t_ns": 10 * NS + int(round(t * NS)),
                "policy": "walk",
                "move": {"requested": [self.vx, 0.0, self.vyaw], "applied": [self.vx, 0, 0]},
                "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
                "joints": joints,
                "velocities": [0.0] * 15,
                "currents_ma": currents,
                "odom": {"position": [self.x, self.y, 0.0], "yaw": self.yaw},
                "imu": {
                    "gyro": [0.3, 0.2, 0.0] if moving else [0.01, 0.0, 0.0],
                    "quat": [1, 0, 0, 0],
                },
            }
        )

    def depth(self) -> DepthView:
        ahead = None
        if self.thing_mm is not None and abs(self.yaw) < 0.4:
            ahead = int(round(self.thing_mm - self.x * 1000.0))
        return DepthView(0, ahead, None, None, 0, False, 0)

    def step(self, intents: list[Any], dt: float) -> None:
        for intent in intents:
            if intent.method == "robot.move":
                self.vx = float(intent.params.get("vx", 0.0))
                self.vyaw = float(intent.params.get("vyaw", 0.0))
                if self.vx < 0:
                    self.reverse_sent += 1
        speed = 0.195 if self.vx > 0 else -0.15 if self.vx < 0 else 0.0
        self.x += speed * dt * math.cos(self.yaw)
        self.y += speed * dt * math.sin(self.yaw)
        self.yaw += 0.6 * self.vyaw * dt if self.vx != 0 else 0.0
        self.backed_max = max(self.backed_max, -self.x)


def body(config: W.WalkConfig | None = None) -> BodyLoop:
    cfg = config or W.WalkConfig(use_look_veto=False, end_look=False)
    walk = W.make_factory(cfg, W.WalkMemory())
    return BodyLoop(BodyConfig(has_tof=True, executors={"duck.walk": walk}))


def run_walk(loop: BodyLoop, kin: Kin, t: float, act: str, **kw: Any) -> tuple[dict, float]:
    obj = {"type": "cmd", "cmd": "duck.walk", "act_id": act, "vx_mm": 400, "pre_look": "none"}
    obj.update(kw)
    obj["deadline_ms"] = (10 * NS + int(t * NS)) // 1_000_000 + 30_000
    ack = loop.command(obj, 10 * NS + int(t * NS))
    assert ack[0]["ok"], ack
    for _ in range(1500):
        tick = loop.tick(kin.state(t), depth=kin.depth())
        for line in tick.lines:
            if line.get("type") == "percept" and line["event"].get("act_id") == act:
                kin.lines.append(line["event"])
                assert percept.validate(line["event"]) == []
                for _ in range(40):  # the legs come to rest before the next command
                    kin.step([], 0.02)
                    t += 0.02
                    loop.tick(kin.state(t), depth=kin.depth())
                kin.vx = kin.vyaw = 0.0
                return line["event"], t
        kin.step(tick.intents, 0.02)
        t += 0.02
    raise AssertionError(f"{act} did not finish")


def test_in_front_of_a_small_thing_the_duck_backs_off_and_walks_on() -> None:
    loop, kin = body(), Kin(thing_mm=280)
    t = 0.0
    refused_before = loop.refused_total
    # not cornered yet: the back step is refused, nothing reverses
    done, t = run_walk(loop, kin, t, "B0", vx_mm=-400)
    assert done["reason"] == "veto_look" and kin.reverse_sent == 0
    for act, vyaw in (("S", 0), ("L", 1000), ("R", -1000)):
        done, t = run_walk(loop, kin, t, act, vyaw_mrad=vyaw)
        assert done["reason"] == "reflex_near" and done["odom"]["path_mm"] < 60, done
    assert set(loop.cornered.shut(t_ms(t), loop.spot)) == {"ahead", "left", "right"}
    done, t = run_walk(loop, kin, t, "B1", vx_mm=-550, max_ms=5000)
    assert done["reason"] == "time" and kin.reverse_sent > 0, done
    assert done["cmd"]["vx"] == -400
    assert 100 <= kin.backed_max * 1000 <= BACK_STEP_MAX_MM
    assert loop.refused_total == refused_before  # the guard never had to refuse a twist
    assert not loop.cornered.now(t_ms(t), loop.spot)  # moved out of its marks
    done, t = run_walk(loop, kin, t, "F", vyaw_mrad=1000, max_ms=600)
    assert done["odom"]["path_mm"] > 50  # free to walk on (an arc away, not stopped at once)


def test_the_guard_stops_a_back_step_at_0_2_m_whatever_walkexec_asks() -> None:
    long = W.WalkConfig(use_look_veto=False, end_look=False, back_max_mm=2000, back_max_ms=6000)
    loop, kin = body(long), Kin(thing_mm=280)
    t = 0.0
    for act, vyaw in (("S", 0), ("L", 1000), ("R", -1000)):
        _, t = run_walk(loop, kin, t, act, vyaw_mrad=vyaw)
    done, t = run_walk(loop, kin, t, "B1", vx_mm=-400, max_ms=6000)
    assert kin.reverse_sent > 0 and loop.refused_total > 0  # refused past the bound …
    assert kin.backed_max * 1000 <= BACK_STEP_MAX_MM + 5  # … one tick's travel at most


def test_nothing_ahead_stops_a_back_step_but_the_back_step_is_never_granted_in_the_open() -> None:
    loop, kin = body(), Kin(thing_mm=None)
    done, _ = run_walk(loop, kin, 0.0, "B0", vx_mm=-400, vyaw_mrad=1000)
    assert done["reason"] == "veto_look" and kin.reverse_sent == 0 and kin.backed_max == 0.0


def t_ms(t: float) -> int:
    return (10 * NS + int(round(t * NS))) // 1_000_000
