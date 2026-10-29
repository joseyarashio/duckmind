"""示範: the curiosity demonstrations on the simulator page — 撞撞看, 啄一下, 踢一下 (curiosity spec
§12 and §17 P0: "the owner puts a ball down, presses 踢一下, and sees it really roll").

A demonstration shows the physics, not the duck's mind: Duck Core is not asked and not told,
and the page labels it 示範. It is **truth-positioned** (physical-design spec §15.2 item 7 allows
that only on demo buttons): the duck walks toward the thing if it is far (an arc, as the policy
walks), stops, and then the world puts the thing exactly where the official skill was trained to
meet it (``x_place_prop``), so the demonstration works every time instead of depending on
walking accuracy (a 1.5 cm aiming error already misses 12 % of kicks):

* ``kick`` — the ball 9 cm ahead of the trunk and 4.2 cm toward the kicking foot (the official
  ``ball_kick_left/right`` training spot, curiosity exp2), then ``kick_left`` or
  ``kick_right``;
* ``peck`` — the thing under the lowest point of the beak (10.8 cm ahead at 2 cm high, measured
  with ``alpha_ground_pick``), then ``ground_pick``;
* ``bump`` — the thing 35 cm straight ahead, then a slow walk into it (``robot.move`` 0.3 m/s for
  2 s: the neck usually meets it first).

It then watches for :data:`WATCH_S` and reports what happened, from the world's truth
(``x_world``): how far the thing moved, whether it rolled or tipped, whether the duck stayed up.

:class:`DemoRunner` only decides; the live bridge carries out its commands, a few times a
second::

    runner = DemoRunner("kick", "ball_felt", world=client)
    while True:
        command = runner.tick(time.monotonic(), policy=body_policy)
        # {"move": [vx, vy, vyaw]} | {"posture": "stand"} | {"skill": "kick_left"}
        # | {"wait": true} | {"done": {...}}
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from duckmind.i18n import msg

KINDS = ("bump", "peck", "kick")
LABEL_ZH = {
    "bump": msg("web.demo.LABEL_ZH.bump"),
    "peck": msg("web.demo.LABEL_ZH.peck"),
    "kick": msg("web.demo.LABEL_ZH.kick"),
}
SKILLS = {"peck": "ground_pick"}
KICK_SPOT_M = (0.09, 0.042)  # ahead, toward the kicking foot (ball_kick_* training reset)
PECK_SPOT_M = 0.115  # ahead: the beak's lowest point (0.108 m) on the thing's near half
BUMP_SPOT_M = 0.35
BUMP_TWIST = (0.3, 0.0, 0.0)
BUMP_S = 2.0
APPROACH_FROM_M = 0.6  # farther than this: walk toward it first
APPROACH_TO_M = 0.45
APPROACH_S = 12.0
STILL_S = 0.6  # standing still this long before the thing is set in front
STAND_S = 8.0
WATCH_S = 3.0
SKILL_S = {"ground_pick": 2.8, "kick_left": 1.0, "kick_right": 1.0}
STANDING = ("walk", "stand")


class WorldPort(Protocol):
    def call(self, op: str, **params: Any) -> dict[str, Any]: ...


@dataclass
class DemoRunner:
    """One demonstration, step by step (see the module docstring)."""

    kind: str
    thing: str
    world: WorldPort
    side: str | None = None  # kick: "left" | "right" (default: the side the ball is on)
    approach: bool = True
    state: str = "start"
    started_s: float | None = None
    phase_s: float = 0.0
    before: dict[str, Any] | None = None
    placed: dict[str, Any] | None = None
    skill: str | None = None
    result: dict[str, Any] | None = None
    why_zh: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    _still_since: float | None = None
    _last_pos: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"demo {self.kind!r}: one of {', '.join(KINDS)}")
        if self.side not in (None, "left", "right"):
            raise ValueError("kick side: left or right")

    # ── truth ─────────────────────────────────────────────────────────────

    def _truth(self) -> tuple[dict[str, Any], dict[str, Any]]:
        snap = self.world.call("x_world")
        duck = snap.get("duck") or {}
        for item in snap.get("things") or []:
            if item.get("id") == self.thing:
                return duck, item
        for item in snap.get("props") or []:
            if item.get("name") in (self.thing, f"thing_{self.thing}"):
                return duck, {"id": self.thing, **item}
        raise ValueError(msg("web.demo.DemoRunner.truth", thing=repr(self.thing)))

    @staticmethod
    def _local(duck: dict[str, Any], point_mm: list[int]) -> tuple[float, float]:
        """A world point in the duck's heading frame (m): x ahead, y to its left."""
        x, y = duck["pos_mm"][0] / 1000.0, duck["pos_mm"][1] / 1000.0
        yaw = duck.get("yaw_mrad", 0) / 1000.0
        dx, dy = point_mm[0] / 1000.0 - x, point_mm[1] / 1000.0 - y
        return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)

    @staticmethod
    def _world(duck: dict[str, Any], ahead: float, left: float) -> tuple[int, int]:
        x, y = duck["pos_mm"][0] / 1000.0, duck["pos_mm"][1] / 1000.0
        yaw = duck.get("yaw_mrad", 0) / 1000.0
        wx = x + ahead * math.cos(yaw) - left * math.sin(yaw)
        wy = y + ahead * math.sin(yaw) + left * math.cos(yaw)
        return int(round(wx * 1000)), int(round(wy * 1000))

    # ── the steps ─────────────────────────────────────────────────────────

    def tick(self, now: float, *, policy: str | None = None) -> dict[str, Any]:
        """The next command (see the module docstring)."""
        if self.started_s is None:
            self.started_s = now
            self.phase_s = now
        if self.state == "done":
            return {"done": self.result}
        try:
            return self._tick(now, policy)
        except Exception as error:  # the world went away or refused: stop, say why
            self.why_zh.append(msg("web.demo.DemoRunner.tick.3c50b7", error=str(error)))
            return self._finish(now, ok=False)

    def _go(self, state: str, now: float) -> None:
        self.state = state
        self.phase_s = now
        self.log.append({"state": state, "t": round(now - (self.started_s or now), 2)})

    def _tick(self, now: float, policy: str | None) -> dict[str, Any]:
        elapsed = now - self.phase_s
        if self.state == "start":
            duck, thing = self._truth()
            self.before = {"duck": duck, "thing": thing}
            self._go("stand" if policy not in STANDING else "approach", now)
            return {"move": [0.0, 0.0, 0.0]}
        if self.state == "stand":
            if policy in STANDING:
                self._go("approach", now)
                return {"move": [0.0, 0.0, 0.0]}
            if elapsed > STAND_S:
                self.why_zh.append(msg("web.demo.DemoRunner.tick.095d16"))
                return self._finish(now, ok=False)
            return {"posture": "stand"}
        if self.state == "approach":
            duck, thing = self._truth()
            ahead, left = self._local(duck, thing["pos_mm"])
            far = math.hypot(ahead, left)
            if not self.approach or far <= APPROACH_FROM_M or elapsed > APPROACH_S:
                if self.approach and far > APPROACH_FROM_M:
                    self.why_zh.append(msg("web.demo.DemoRunner.tick.e803ef"))
                self._go("settle", now)
                self._still_since = None
                return {"move": [0.0, 0.0, 0.0]}
            if far <= APPROACH_TO_M:
                self._go("settle", now)
                return {"move": [0.0, 0.0, 0.0]}
            bearing = math.atan2(left, ahead)
            # velstand walks forward only from about 0.3 m/s and turns in arcs
            return {"move": [0.3, 0.0, max(-1.0, min(1.0, 2.0 * bearing))]}
        if self.state == "settle":
            duck, _ = self._truth()
            pos = (duck["pos_mm"][0] / 1000.0, duck["pos_mm"][1] / 1000.0)
            moving = self._last_pos is not None and math.dist(pos, self._last_pos) > 0.004
            self._last_pos = pos
            if moving or policy not in STANDING:
                self._still_since = None
            elif self._still_since is None:
                self._still_since = now
            if self._still_since is not None and now - self._still_since >= STILL_S:
                return self._set_up(now, duck)
            if elapsed > STAND_S:
                return self._set_up(now, duck)
            return {"move": [0.0, 0.0, 0.0]}
        if self.state == "act":
            if self.kind == "bump":
                if elapsed < BUMP_S:
                    return {"move": list(BUMP_TWIST)}
                self._go("watch", now)
                return {"move": [0.0, 0.0, 0.0]}
            if elapsed >= SKILL_S.get(self.skill or "", 1.0):
                self._go("watch", now)
            return {"wait": True}
        if self.state == "watch":
            if elapsed >= WATCH_S:
                return self._finish(now, ok=True)
            return {"move": [0.0, 0.0, 0.0]}
        return {"wait": True}

    def _set_up(self, now: float, duck: dict[str, Any]) -> dict[str, Any]:
        """Put the thing where the skill meets it (truth positioning: 示範 only)."""
        _, thing = self._truth()
        if self.kind == "kick":
            ahead, left = self._local(duck, thing["pos_mm"])
            side = self.side or ("left" if left > 0 else "right")
            sign = 1.0 if side == "left" else -1.0
            spot = self._world(duck, KICK_SPOT_M[0], sign * KICK_SPOT_M[1])
            self.skill = f"kick_{side}"
        elif self.kind == "peck":
            spot = self._world(duck, PECK_SPOT_M, 0.0)
            self.skill = SKILLS["peck"]
        else:
            spot = self._world(duck, BUMP_SPOT_M, 0.0)
            self.skill = None
        placed = self.world.call(
            "x_place_prop", id=self.thing, x_mm=spot[0], y_mm=spot[1], on="floor"
        )
        self.placed = placed
        if not placed.get("ok"):
            self.why_zh += list(placed.get("why_zh", [])) or [msg("web.demo.DemoRunner.set_up")]
            return self._finish(now, ok=False)
        self.before = {"duck": duck, "thing": {**thing, "pos_mm": placed.get("pos_mm")}}
        self._go("act", now)
        if self.skill is not None:
            return {"skill": self.skill}
        return {"move": list(BUMP_TWIST)}

    def _finish(self, now: float, *, ok: bool) -> dict[str, Any]:
        after_duck: dict[str, Any] = {}
        after_thing: dict[str, Any] = {}
        try:
            after_duck, after_thing = self._truth()
        except Exception:
            pass
        moved = None
        if self.before and after_thing.get("pos_mm") and self.before["thing"].get("pos_mm"):
            a, b = self.before["thing"]["pos_mm"], after_thing["pos_mm"]
            moved = int(round(math.dist(a[:2], b[:2])))
        tilt = after_duck.get("tilt_mrad")
        upright = tilt is not None and tilt < 436  # 25°
        outcome = "nothing"
        if moved is not None:
            if moved >= 150:
                outcome = "rolled" if self.kind == "kick" or "ball" in self.thing else "moved"
            elif moved >= 15:
                outcome = "moved"
        self.result = {
            "ok": ok,
            "demo": True,  # 示範: truth-positioned, never an input for Duck Core
            "kind": self.kind,
            "zh": msg("web.demo.DemoRunner.finish", kind=LABEL_ZH[self.kind]),
            "thing": self.thing,
            "skill": self.skill,
            "moved_mm": moved,
            "outcome": outcome if ok else "stopped",
            "duck_upright": upright,
            "seconds": round(now - (self.started_s or now), 2),
            "placed": self.placed,
            "log": list(self.log),
            "why_zh": list(self.why_zh),
        }
        self.state = "done"
        return {"move": [0.0, 0.0, 0.0], "done": self.result}


def run(
    runner: DemoRunner,
    *,
    policy: Callable[[], str | None],
    send: Callable[[dict[str, Any]], None],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    every_s: float = 0.1,
    limit_s: float = 40.0,
) -> dict[str, Any]:
    """Drive a runner to the end (the live bridge's loop, for scripts and tests)."""
    began = clock()
    while True:
        command = runner.tick(clock(), policy=policy())
        send(command)
        if "done" in command:
            return command["done"] or {}
        if clock() - began > limit_s:
            runner.why_zh.append(msg("web.demo.run"))
            return runner._finish(clock(), ok=False)["done"]
        sleep(every_s)
