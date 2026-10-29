"""A visible hand that really moves the physics duck (physical-design spec §11.5 W03; owner
decisions D1, D2, A2, A3).

A mocap body (:data:`HAND`: palm, fingers, thumb and forearm; seen by the camera and the ToF,
colliding with nothing) and a weld to the duck's trunk (:data:`WELD`), switched with
``data.eq_active``. Its scripts run in the world server's per-step hook on physics time, so in
lockstep they keep pace with robotd whatever the wall clock does:

* ``pickup`` — reach down, take the trunk where it is (the weld starts at the trunk's own pose:
  no snap), lift it about 30 cm and hold it there, upright at the heading it had;
* ``place`` — (pick it up first if needed,) carry it over whatever is in the way to ``floor``
  (a spot, or straight down), ``mat`` (the family's mat), ``desk`` (= the mat: owner decision
  A3, no real table until phase 3) or ``nest`` (the charging place: into the dock, seated where
  a docked duck sits and facing the way its style docks — the dock in ``forward``, the room in
  ``backin`` — so its contact closes; or, for phase 2's open nest, on the pad with the belly pin
  over the middle, facing the room), lower it until it touches, let go, and leave. A spot that
  is taken (a toy, furniture, a person) or not level moves to the nearest free level spot
  within 0.8 m;
* ``upright`` — D2's everyday recovery after a fall: take the trunk where it lies, lift it a
  little, turn it upright at the heading it fell with, set it down, let go. Nothing does this by
  itself (D2: no automatic help);
* ``touch`` / ``hold_head`` / ``near`` — the hand goes to the head (or the beak), pats or rests
  there, or stops short of it, and leaves; nothing is pushed (the hand collides with nothing);
* ``cancel`` — set down straight below if holding, then leave.

A pickup-carry-set-down takes about 2–4 s (D1). The duck's legs are robotd's: the live bridge
asks for sitting while ``held`` is true ("held in the air it sits with its legs tucked"). When
no one has enabled the duck yet (the official server holds such a duck in place every step),
the hand takes it out of that hold while it carries it and holds it where it set it down.

Ops (evaluator/world, on the body port; never for Duck Core)::

    {"op": "x_hand", "do": "pickup" | "place" | "upright" | "touch" | "hold_head" | "near"
                           | "cancel" | "status",
     "to": "floor" | "mat" | "desk" | "nest" | "here",    (place)
     "x_mm": int, "y_mm": int, "yaw_mrad": int,            (place to="floor": a world spot)
     "where": "head" | "beak" | "back",                    (touch)
     "seconds": number, "speed": "slow" | "fast",          (touch, hold_head / near)
     "from_mm": [x, y, z],   (where the hand comes from and goes back to: a person's hand)
     "who": "owner" | ...}   (the same, from that person's body when it is here)
    → {"ok": true, "script": n, "do": ..., "eta_s": sim seconds to the set-down, "target": {...},
       "why_zh": [...]}

``x_world`` → ``hand``: ``{"state", "do", "script", "holding", "held", "pos_mm", "quat",
"target", "eta_s", "last", "why_zh"}``; ``last`` is the finished script's result (where and how
upright the duck was at release).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import lst, msg
from duckmind.world.worldkit import (
    DUCK_TOP_M,
    PARK_Z_M,
    SENSOR_GROUP,
    TRUNK,
    Prim,
    RobotIds,
    Slide,
    SpotCheck,
    add_prims,
    body_id,
    geom_name,
    heading_of,
    layout_of,
    min_jerk,
    mm,
    mrad,
    prims_json,
    quat_mul,
    register_part,
    rotate,
    site_id,
    slerp,
    static_supports,
    surface_at,
    tilt_of,
    wrap,
    yaw_quat,
)

HAND = "kit_hand"
WELD = "kit_hand_weld"
SKIN = (0.91, 0.73, 0.6)
SLEEVE = (0.35, 0.42, 0.55)
GROUP_SEEN = 0  # camera and ToF
# In the hand's frame = the trunk's frame at the grab: the palm over the back, four fingers
# down the duck's right side, the thumb down its left, the forearm up and back. Nothing
# collides (the weld carries the duck).
HAND_PARTS: tuple[Prim, ...] = (
    Prim(f"{HAND}_palm", "box", SKIN, (0.045, 0.04, 0.012), pos=(-0.01, 0.0, 0.085)),
    *(
        Prim(
            f"{HAND}_finger{k}", "capsule", SKIN, (0.009,), fromto=(x, -0.045, 0.08, x, -0.07, 0.02)
        )
        for k, x in enumerate((-0.04, -0.015, 0.01, 0.035))
    ),
    Prim(f"{HAND}_thumb", "capsule", SKIN, (0.011,), fromto=(0.0, 0.045, 0.08, 0.01, 0.068, 0.03)),
    Prim(f"{HAND}_forearm", "capsule", SLEEVE, (0.03,), fromto=(-0.05, 0.0, 0.1, -0.2, 0.0, 0.36)),
)

REACH_SPEED_M_S = 1.4
CARRY_SPEED_M_S = 1.2
CARRY_MAX_S = 2.2
LIFT_M = 0.30  # held this high above what it was on
HOVER_M = 0.10  # the hand comes down onto the trunk from this far above it
LOWER_FAST_TO_M = 0.125  # quickly down to this trunk height above the surface
LOWER_SPEED_M_S = 0.2  # then slowly until it touches
LOWER_FLOOR_M = 0.045  # never lower the trunk closer than this to the surface
CLEAR_M = 0.16  # the trunk passes this far above anything under the way
MAX_CARRY_Z_M = 1.7
SLIP_M = 0.25  # holding, but the trunk is this far from the hand: something moved it
HEAD_CLEAR_M = 0.02  # the duck's top passes this far under anything over it
# A lift or a set-down needs this much room over the spot (the trunk LIFT_M up, the duck's top
# DUCK_TOP_M over the trunk): a spot under a table top gets the duck slid out along the floor
# first, and is never a set-down spot (it would go through the top).
LIFT_ROOM_M = LIFT_M + DUCK_TOP_M + HEAD_CLEAR_M
SLIDE_SPEED_M_S = 0.3
SLIDE_UP_M = 0.02  # a slide lifts the duck this far off what it sits on (1 cm drags on a thick rug)
TOUCH_S = 3.0
BACK_TOUCH_PAT_M = 0.002  # back-only; head/beak retain the existing 12 mm pat
BACK_TOUCH_DESCEND_M_S = 0.10
PLACE_REACH_M = 0.8
MOVES = ("pickup", "place", "upright", "touch", "hold_head", "near", "cancel", "status")
TARGETS = ("floor", "here", "mat", "desk", "nest")
# Why the hand refused a script, as a number (``x_world`` → hand → ``refused.code``): the words are
# in the error that crosses the socket, but a number is what a watcher can count.
REFUSE_NO_WAY_OUT = 1  # lift: the duck is under something low and cannot be slid out from there
REFUSE_NO_SPOT = 2  # set down: no free level spot with room over it within reach
REFUSE_NO_SURFACE = 3  # set down: nothing to stand on at the spot
REFUSE_HOLDING = 4  # touch: the hand already has the duck
REFUSE_NO_ROOM = 5  # touch the back: not enough room over the duck
REFUSE_NO_NEST = 6  # set down on a nest the home does not have
REFUSE_OTHER = 9  # a request the hand does not understand


class HandRefused(ValueError):
    """The world's hand will not do what was asked: in plain words (the ValueError's text) and as
    a number (``code``: :data:`REFUSE_NO_WAY_OUT` …)."""

    def __init__(self, code: int, words: Any) -> None:
        super().__init__(words)
        self.code = code


@dataclass
class Leg:
    """One piece of a script: a timed move, a move that follows the duck, or a lowering that
    stops when the duck touches."""

    label: str  # what the snapshot says: reach, grab, lift, carry, lower, hold, release, …
    kind: str  # move | track | lower | wait
    seconds: float
    to: list[float] | None = None  # move: world position
    quat: list[float] | None = None  # move: world orientation (None: keep)
    offset: list[float] | None = None  # track: world offset from the followed point
    follow: str = "trunk"  # track: trunk | site:<name>
    local_offset: bool = False  # track only: rotate offset by followed body's full quaternion
    follow_live: bool = False  # track: hold the current followed point without interpolation
    relative_orientation: list[float] | None = (
        None  # track: orientation composed in followed-body frame
    )
    floor_z: float = 0.0  # lower: never below this trunk height
    event: str | None = None  # at the end: grab | release | park
    pat: float = 0.0  # wait: pat up and down by this much (m)
    t0: float | None = None
    p0: list[float] = field(default_factory=list)
    q0: list[float] = field(default_factory=list)


@dataclass
class Script:
    id: int
    do: str
    legs: list[Leg]
    target: dict[str, Any]
    home: list[float]  # where the hand goes back to
    started_s: float
    eta_s: float
    index: int = 0
    why_zh: list[str] = field(default_factory=list)


def _local_offset_world(quat: list[float], offset: list[float], local: bool) -> list[float]:
    """Preserve legacy world offsets unless a leg explicitly opts into body-local tracking."""
    return rotate(quat, offset) if local else list(offset)


def _projected_geom_top(
    trunk_pos: list[float],
    trunk_quat: list[float],
    geom_positions: list[list[float]],
    geom_radii: list[float],
) -> float:
    """Conservative local-z envelope from world-space geom centers and bounding radii."""
    if not geom_positions or len(geom_positions) != len(geom_radii):
        raise ValueError("collision geometry is unavailable")
    axis = rotate(trunk_quat, [0.0, 0.0, 1.0])
    return max(
        sum(a * (p - o) for a, p, o in zip(axis, pos, trunk_pos, strict=True)) + radius
        for pos, radius in zip(geom_positions, geom_radii, strict=True)
    )


def _trunk_local_point(
    trunk_pos: list[float], trunk_quat: list[float], point: list[float]
) -> list[float]:
    """Convert a world point into the trunk frame using the full unit quaternion."""
    delta = [p - o for p, o in zip(point, trunk_pos, strict=True)]
    conjugate = [trunk_quat[0], -trunk_quat[1], -trunk_quat[2], -trunk_quat[3]]
    return rotate(conjugate, delta)


class Hand:
    """The hand (see the module docstring). One per world."""

    name = "hand"

    def __init__(self, *, nest: Any = None, people: Any = None) -> None:
        self.nest_part = nest
        self.people = people
        self.layout: Any = None
        self.model: Any = None
        self.world: Any = None
        self.body: Any = None
        self.robot: RobotIds | None = None
        self.mocap = -1
        self.weld = -1
        self.spots: SpotCheck | None = None
        self.supports: list[int] = []
        self.script: Script | None = None
        self.holding = False
        self.last: dict[str, Any] | None = None
        self.refused: dict[str, Any] | None = None  # the last script the hand would not start
        self.count = 0
        self._stashed: Any = None  # the official hold of a not-yet-enabled duck, while carried
        self._settle_steps = 0

    # ── building ──────────────────────────────────────────────────────────

    def edit_spec(self, spec: Any, home: Any) -> None:
        import mujoco

        self.layout = layout_of(home)
        hand = spec.worldbody.add_body()
        hand.name = HAND
        hand.mocap = True
        hand.pos = [0.0, 0.0, PARK_Z_M]
        add_prims(hand, HAND_PARTS, group=GROUP_SEEN)
        weld = spec.add_equality()
        weld.name = WELD
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = HAND
        weld.name2 = TRUNK
        weld.active = False

    def ready(self, world: Any, body: Any = None) -> None:
        import mujoco

        model = world.model
        self.model, self.world, self.body = model, world, body
        self.robot = RobotIds.of(model)
        hand = body_id(model, HAND)
        self.mocap = int(model.body_mocapid[hand]) if hand >= 0 else -1
        self.weld = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD))
        if self.mocap < 0 or self.weld < 0:
            raise ValueError("the hand's mocap body or weld is missing from the compiled world")
        # the weld holds the trunk exactly at the hand's frame: anchor 0, relpose identity
        model.eq_data[self.weld][:11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        world.data.eq_active[self.weld] = 0
        self.supports = static_supports(model)
        # things and people's legs are in the way too; the layout keeps spots indoors
        self.spots = SpotCheck(model, self.robot, layout=self.layout)
        self._park(world.data)

    # ── the per-step script ───────────────────────────────────────────────

    def before_step(self, world: Any) -> None:
        script = self.script
        data = world.data
        if script is None:
            if self.holding and self._slipped(data):  # held, and something else moved it
                self._release(data)
                self._park(data)
                self.last = {
                    "script": None,
                    "do": "hold",
                    "ok": False,
                    "finished_s": round(float(data.time), 4),
                    "target": {},
                    "holding": False,
                    "why_zh": [msg("world.hand.Hand.before_step.d1018fab")],
                }
            return
        now = float(data.time)
        guard = 0
        while script is self.script and script.index < len(script.legs) and guard < 4:
            guard += 1
            leg = script.legs[script.index]
            if leg.t0 is None:
                leg.t0 = now
                leg.p0 = [float(v) for v in data.mocap_pos[self.mocap]]
                leg.q0 = [float(v) for v in data.mocap_quat[self.mocap]]
            done = self._drive(leg, data, now)
            if self.holding and self._slipped(data):
                self._release(data, why=msg("world.hand.Hand.before_step.a057b38b"))
                script.why_zh.append(msg("world.hand.Hand.before_step.d1018fab"))
                self._finish(data, ok=False)
                return
            if not done:
                return
            if leg.event == "grab":
                self._grab(data)
            elif leg.event == "release":
                self._release(data)
            elif leg.event == "park":
                self._park(data)
            script.index += 1
            if script.index >= len(script.legs):
                self._finish(data, ok=True)
                return

    def _point(self, follow: str, data: Any) -> tuple[list[float], list[float]]:
        assert self.robot is not None
        if follow.startswith("site:"):
            site = site_id(self.model, follow[5:])
            pos = [float(v) for v in data.site_xpos[site]]
            quat = [float(v) for v in data.xquat[self.robot.trunk]]
            return pos, quat
        return (
            [float(v) for v in data.xpos[self.robot.trunk]],
            [float(v) for v in data.xquat[self.robot.trunk]],
        )

    def _drive(self, leg: Leg, data: Any, now: float) -> bool:
        assert leg.t0 is not None
        elapsed = now - leg.t0
        tau = 1.0 if leg.seconds <= 0 else elapsed / leg.seconds
        s = min_jerk(tau)
        if leg.kind == "track":
            base, quat = self._point(leg.follow, data)
            target_quat = (
                quat_mul(quat, leg.relative_orientation)
                if leg.relative_orientation is not None
                else quat
            )
            offset = leg.offset or [0.0, 0.0, 0.0]
            offset = _local_offset_world(quat, offset, leg.local_offset)
            if leg.pat and leg.local_offset:
                # A back pat is a bounded excursion along the trunk-local normal while the
                # hand continues to follow the duck. Legacy wait-leg pats remain world-Z.
                pat = rotate(
                    quat, [0.0, 0.0, -leg.pat * 0.5 * (1.0 - math.cos(2 * math.pi * 1.5 * elapsed))]
                )
                offset = [a + b for a, b in zip(offset, pat, strict=True)]
            goal = [b + o for b, o in zip(base, offset, strict=True)]
            pos = (
                goal
                if leg.follow_live
                else [a + (b - a) * s for a, b in zip(leg.p0, goal, strict=True)]
            )
            self._set(data, pos, target_quat if leg.follow_live else slerp(leg.q0, target_quat, s))
            return tau >= 1.0
        if leg.kind == "move":
            goal = leg.to or leg.p0
            pos = [a + (b - a) * s for a, b in zip(leg.p0, goal, strict=True)]
            self._set(data, pos, slerp(leg.q0, leg.quat or leg.q0, s))
            return tau >= 1.0
        if leg.kind == "wait":
            pos = list(leg.p0)
            if leg.pat:
                pos[2] -= leg.pat * 0.5 * (1.0 - math.cos(2 * math.pi * 1.5 * elapsed))
            self._set(data, pos, leg.q0)
            return tau >= 1.0
        # lower: straight down until the duck touches something, or the floor limit, or time
        assert self.robot is not None
        z = leg.p0[2] - LOWER_SPEED_M_S * elapsed
        trunk_z = float(data.xpos[self.robot.trunk][2])
        mocap_z = float(data.mocap_pos[self.mocap][2])
        lowest = leg.floor_z + (mocap_z - trunk_z)  # the hand's height at the trunk's limit
        pos = [leg.p0[0], leg.p0[1], max(z, lowest)]
        self._set(data, pos, leg.q0)
        if self._touching(data):
            self._settle_steps += 1
        else:
            self._settle_steps = 0
        return self._settle_steps >= 4 or z <= lowest or tau >= 1.0

    def _set(self, data: Any, pos: list[float], quat: list[float]) -> None:
        data.mocap_pos[self.mocap] = pos
        data.mocap_quat[self.mocap] = quat

    def _touching(self, data: Any) -> bool:
        """The duck touches something that is not itself (the floor, the pad, a toy)."""
        assert self.robot is not None
        model, bodies = self.model, self.robot.bodies
        for contact in data.contact[: int(data.ncon)]:
            one, two = int(contact.geom1), int(contact.geom2)
            a = int(model.geom_bodyid[one]) in bodies
            b = int(model.geom_bodyid[two]) in bodies
            if a == b:
                continue
            other = two if a else one
            if int(model.geom_group[other]) == SENSOR_GROUP:
                continue
            if float(contact.dist) < 0.0005:
                return True
        return False

    def _slipped(self, data: Any) -> bool:
        """Held, but the trunk is far from the hand: something else moved it (an ``x_place``
        writes qpos between steps; the weld must let go before the next step, or it would fling
        the duck back across the room)."""
        assert self.robot is not None
        qadr = self.robot.qadr
        trunk = data.qpos[qadr : qadr + 3]
        hand = data.mocap_pos[self.mocap]
        return math.dist([float(v) for v in trunk], [float(v) for v in hand]) > SLIP_M

    def _grab(self, data: Any) -> None:
        assert self.robot is not None
        # the hand's frame onto the trunk's exactly, then the weld: nothing snaps
        data.mocap_pos[self.mocap] = data.xpos[self.robot.trunk]
        data.mocap_quat[self.mocap] = data.xquat[self.robot.trunk]
        body = self.body
        if body is not None and not getattr(body, "released", True) and self._stashed is None:
            self._stashed = getattr(body, "held", None)
            body.held = None  # the official server would put it back every step
        data.eq_active[self.weld] = 1
        self.holding = True

    def _release(self, data: Any, why: str | None = None) -> None:
        assert self.robot is not None
        data.eq_active[self.weld] = 0
        self.holding = False
        body = self.body
        if body is not None and self._stashed is not None:
            body.remember()  # a duck nobody enabled stays where it was put
        self._stashed = None
        script = self.script
        if script is not None:
            quat = [float(v) for v in data.xquat[self.robot.trunk]]
            pos = [float(v) for v in data.xpos[self.robot.trunk]]
            result: dict[str, Any] = {
                "released_s": round(float(data.time), 4),
                "trunk_mm": mm(pos),
                "tilt_mrad": mrad(tilt_of(quat)),
                "yaw_mrad": mrad(heading_of(quat)),
            }
            aim = script.target.get("trunk_mm")
            if aim:
                result["error_mm"] = int(round(math.dist(result["trunk_mm"][:2], aim[:2])))
            script.target["release"] = result
            if why:
                script.why_zh.append(why)

    def _park(self, data: Any) -> None:
        pos = [float(v) for v in data.mocap_pos[self.mocap]]
        data.mocap_pos[self.mocap] = [pos[0], pos[1], PARK_Z_M]
        data.mocap_quat[self.mocap] = [1.0, 0.0, 0.0, 0.0]

    def _finish(self, data: Any, *, ok: bool) -> None:
        script = self.script
        if script is None:
            return
        self.last = {
            "script": script.id,
            "do": script.do,
            "ok": ok,
            "finished_s": round(float(data.time), 4),
            "target": script.target,
            "holding": self.holding,
            "why_zh": list(script.why_zh),
        }
        self.script = None

    # ── planning a script (op thread, the world's lock held) ──────────────

    def command(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Start a script (or answer ``status``); call with the world's lock held. A request the
        hand will not do raises (``ValueError``: :class:`HandRefused` when it is a matter of the
        world, with a number) and is counted in the snapshot's ``refused``."""
        try:
            return self._start(world, request)
        except ValueError as error:
            code = error.code if isinstance(error, HandRefused) else REFUSE_OTHER
            self.refused = {
                "code": code,
                "count": int(self.refused["count"]) + 1 if self.refused else 1,
                "sim_s": round(float(world.data.time), 3),
            }
            print(f"hand: refused {request.get('do')}, reason {code}", flush=True)
            raise

    def _start(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        do = str(request.get("do", "status"))
        if do not in MOVES:
            raise ValueError(msg("world.hand.Hand.command.4bb689c5", do=do, list=lst(MOVES)))
        if self.robot is None or self.mocap < 0:
            raise ValueError(msg("world.hand.Hand.command.87aed84f"))
        data = world.data
        if do == "status":
            return self.snapshot(world)
        why: list[str] = []
        start = self._start_point(data, request)
        legs: list[Leg] = []
        target: dict[str, Any] = {}
        busy = self.script is not None
        if busy and self.script is not None:
            why.append(msg("world.hand.Hand.command.a3fa2ecd", do=self.script.do))
        if do == "pickup":
            origin = None
            if not self.holding:
                legs += self._take(data, start)
                origin, slide = self._clear_sky(data, LIFT_ROOM_M, why)
                legs += slide
            legs += self._lift(data, lift_to=None, origin=origin)
            target = {"to": "held"}
        elif do == "place":
            aim = self._aim(data, request, why)
            target = aim
            origin = None
            if not self.holding:
                legs += self._take(data, start)
                origin, slide = self._clear_sky(data, LIFT_ROOM_M, why)
                legs += slide
            legs += self._carry_and_set(data, aim, origin=origin)
            legs += self._leave(start)
        elif do == "upright":
            origin = None
            if not self.holding:
                legs += self._take(data, start)
                origin, slide = self._clear_sky(data, self._right_room(data), why)
                legs += slide
            legs += self._right(data, origin=origin)
            legs += self._leave(start)
            target = {"to": "upright"}
        elif do in ("touch", "hold_head", "near"):
            if self.holding:
                raise HandRefused(REFUSE_HOLDING, msg("world.hand.Hand.command.b30fa77a"))
            legs += self._touch(data, request, start, do)
            if not (do == "touch" and str(request.get("where", "head")) == "back"):
                legs += self._leave(start)
            target = {"to": str(request.get("where", "head"))}
        elif do == "cancel":
            if self.holding:
                assert self.robot is not None
                here = [float(v) for v in data.xpos[self.robot.trunk]]
                surface = self._surface(data, here[0], here[1], here[2]) or 0.0
                legs += [
                    Leg("lower", "lower", 3.0, floor_z=surface + LOWER_FLOOR_M),
                    Leg("hold", "wait", 0.05, event="release"),
                ]
            legs += self._leave(start)
            target = {"to": "cancel"}
        now = float(data.time)
        eta = now + sum(leg.seconds for leg in legs)
        self.count += 1
        self._settle_steps = 0
        script = Script(
            id=self.count,
            do=do,
            legs=legs,
            target=target,
            home=start,
            started_s=now,
            eta_s=eta,
            why_zh=why,
        )
        if data.mocap_pos[self.mocap][2] < PARK_Z_M / 2:
            # coming out of hiding: appear where the hand comes from
            data.mocap_pos[self.mocap] = start
            data.mocap_quat[self.mocap] = [float(v) for v in data.xquat[self.robot.trunk]]
        self.script = script
        return {
            "ok": True,
            "script": script.id,
            "do": do,
            "eta_s": round(eta - now, 2),
            "target": target,
            "why_zh": why,
        }

    def _start_point(self, data: Any, request: dict[str, Any]) -> list[float]:
        assert self.robot is not None
        given = request.get("from_mm")
        if given is not None:
            if not isinstance(given, (list, tuple)) or len(given) != 3:
                raise ValueError(msg("world.hand.Hand._start_point.7d3e220d"))
            return [float(v) / 1000.0 for v in given]
        who = request.get("who")
        if who and self.people is not None and hasattr(self.people, "hand_point"):
            point = self.people.hand_point(str(who))
            if point is not None:
                return list(point)
        current = [float(v) for v in data.mocap_pos[self.mocap]]
        if current[2] > PARK_Z_M / 2:
            return current
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        return [trunk[0] - 0.15, trunk[1] - 0.15, trunk[2] + 0.6]

    def _surface(self, data: Any, x: float, y: float, below: float) -> float | None:
        hit = surface_at(self.model, data, self.supports, x, y, below)
        return None if hit is None else hit[0]

    def _take(self, data: Any, start: list[float]) -> list[Leg]:
        assert self.robot is not None
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        hover = [trunk[0], trunk[1], trunk[2] + HOVER_M]
        reach = min(1.4, max(0.3, math.dist(start, hover) / REACH_SPEED_M_S))
        return [
            Leg("reach", "track", reach, offset=[0.0, 0.0, HOVER_M]),
            Leg("grab", "track", 0.2, offset=[0.0, 0.0, 0.0], event="grab"),
        ]

    def _upright_quat(self, data: Any, yaw: float | None = None) -> list[float]:
        assert self.robot is not None
        quat = [float(v) for v in data.xquat[self.robot.trunk]]
        return yaw_quat(heading_of(quat) if yaw is None else yaw)

    def _trunk(self, data: Any, origin: list[float] | None = None) -> list[float]:
        """Where the trunk is (or will be, after a slide: ``origin``)."""
        assert self.robot is not None
        if origin is not None:
            return list(origin)
        return [float(v) for v in data.xpos[self.robot.trunk]]

    def _clear_sky(
        self, data: Any, room_m: float, why: list[str]
    ) -> tuple[list[float] | None, list[Leg]]:
        """Before lifting a duck that has something low over it (under a table): the nearest
        spot with ``room_m`` of room above it that it can be slid to along the floor
        (:meth:`SpotCheck.slide_out`: round a table's leg, over a rug's edge), and the slide there
        (the trunk a centimetre up, higher over a step, its heading kept). (None, []) when there
        is room to lift it where it is; :class:`HandRefused` when it cannot be got out."""
        assert self.robot is not None and self.spots is not None
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        surface = self._surface(data, trunk[0], trunk[1], trunk[2]) or 0.0
        above = self.spots.headroom(data, trunk[0], trunk[1], surface)
        if above is None or above[0] >= room_m:
            return None, []
        over = self._plain(geom_name(self.model, above[1]))
        way: Slide | None = self.spots.slide_out(data, trunk[0], trunk[1], headroom_m=room_m)
        if way is None:
            raise HandRefused(
                REFUSE_NO_WAY_OUT, msg("world.hand.Hand._clear_sky.c1c4c759", over=over)
            )
        x, y = way.route[-1]
        z = trunk[2] + (way.surface - surface) + SLIDE_UP_M
        carry = max(z, trunk[2] + (way.top - surface) + SLIDE_UP_M)  # (over the highest step)
        why.append(msg("world.hand.Hand._clear_sky.b5be9508", over=over))
        slide = [Leg("slide", "move", 0.2, to=[trunk[0], trunk[1], carry])]
        for a, b in zip(way.route, way.route[1:], strict=False):
            seconds = max(0.3, math.dist(a, b) / SLIDE_SPEED_M_S)
            slide.append(Leg("slide", "move", seconds, to=[b[0], b[1], carry]))
        if carry > z + 0.001:
            slide.append(Leg("slide", "move", 0.2, to=[x, y, z]))
        return [x, y, z], slide

    def _plain(self, name: str) -> str:
        """A geom's name as the owner would say it (the layout's words), else 東西."""
        zh_of = getattr(self.layout, "zh_of", None)
        said = zh_of(name) if callable(zh_of) else None
        return str(said) if said else msg("world.hand.Hand._plain.1cd15bf8")

    def _lift(
        self, data: Any, lift_to: float | None, origin: list[float] | None = None
    ) -> list[Leg]:
        assert self.robot is not None
        trunk = self._trunk(data, origin)
        surface = self._surface(data, trunk[0], trunk[1], trunk[2]) or 0.0
        z = lift_to if lift_to is not None else surface + LIFT_M
        return [
            Leg(
                "lift",
                "move",
                0.5,
                to=[trunk[0], trunk[1], max(z, trunk[2])],
                quat=self._upright_quat(data),
            ),
            Leg("hold", "wait", 0.0),
        ]

    def _aim(self, data: Any, request: dict[str, Any], why: list[str]) -> dict[str, Any]:
        """Where a place goes: the trunk's world spot, heading and surface."""
        assert self.robot is not None and self.spots is not None
        to = str(request.get("to", "floor"))
        if to not in TARGETS:
            raise ValueError(msg("world.hand.Hand._aim.a6096308", to=to, list=lst(TARGETS)))
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        heading = heading_of([float(v) for v in data.xquat[self.robot.trunk]])
        layout = self.layout
        search = True
        if to == "nest":
            nest = getattr(layout, "nest", None) if layout is not None else None
            if nest is None:
                raise HandRefused(REFUSE_NO_NEST, msg("world.hand.Hand._aim.22c6d7dd"))
            if getattr(nest, "is_dock", False):
                # seated where a docked duck sits, facing the way its style docks
                seat = nest.seat
                x, y, yaw = seat.x, seat.y, seat.yaw
                facing = (
                    msg("world.hand.Hand._aim.53b3d717")
                    if nest.style == "forward"
                    else msg("world.hand.Hand._aim.bce75c1f")
                )
                why.append(msg("world.hand.Hand._aim.a6882ccf", choice=facing))
            else:
                yaw = nest.heading_mrad / 1000.0
                # the belly pin (trunk x −20 mm) over the pad's middle, facing the room
                x = nest.x_mm / 1000.0 + 0.020 * math.cos(yaw)
                y = nest.y_mm / 1000.0 + 0.020 * math.sin(yaw)
            search = False
        elif to in ("mat", "desk"):
            mat = getattr(layout, "mat", None) if layout is not None else None
            if mat is None:
                x, y, yaw = 0.0, 0.0, 0.0
            else:
                x, y, yaw = mat.x_mm / 1000.0, mat.y_mm / 1000.0, mat.heading_mrad / 1000.0
            if to == "desk":
                why.append(msg("world.hand.Hand._aim.8b812bcf"))
        elif to == "here" or ("x_mm" not in request and "y_mm" not in request):
            x, y, yaw = trunk[0], trunk[1], heading
        else:
            try:
                x = int(request["x_mm"]) / 1000.0
                y = int(request["y_mm"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                raise ValueError(msg("world.hand.Hand._aim.0b97b84d")) from None
            yaw = heading
        if "yaw_mrad" in request:
            yaw = int(request["yaw_mrad"]) / 1000.0
        asked = (x, y)
        surface: float | None
        if search:
            found = self.spots.free(data, x, y, reach_m=PLACE_REACH_M, headroom_m=LIFT_ROOM_M)
            if found is None:
                raise HandRefused(REFUSE_NO_SPOT, msg("world.hand.Hand._aim.ce689aa2"))
            x, y, surface, in_way = found
            if in_way:
                names = list(dict.fromkeys(self._plain(name) for name in in_way))
                why.append(msg("world.hand.Hand._aim.f18fb093", list=lst(names)))
            elif math.dist(asked, (x, y)) > 0.005:
                refused = self.spots.problem(asked[0], asked[1])
                why.append(
                    msg("world.hand.Hand._aim.86da3cf6", refused=refused)
                    if refused
                    else msg("world.hand.Hand._aim.c5dce6cb")
                )
        else:
            surface = self._surface(data, x, y, 0.25)
            if surface is None:
                raise HandRefused(REFUSE_NO_SURFACE, msg("world.hand.Hand._aim.583c6c58"))
        return {
            "to": to,
            "asked_mm": mm(asked),
            "trunk_mm": mm((x, y)),
            "yaw_mrad": mrad(wrap(yaw)),
            "surface_mm": int(round(surface * 1000.0)),
        }

    def _carry_height(self, data: Any, a: list[float], b: list[float]) -> float:
        """The trunk's height for a carry from a to b: over everything on the way."""
        top = 0.0
        steps = max(2, int(math.dist(a[:2], b[:2]) / 0.05))
        for k in range(steps + 1):
            f = k / steps
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            z = self._surface(data, x, y, MAX_CARRY_Z_M)
            if z is not None:
                top = max(top, z)
        return top + CLEAR_M

    def _carry_and_set(
        self, data: Any, aim: dict[str, Any], origin: list[float] | None = None
    ) -> list[Leg]:
        assert self.robot is not None
        trunk = self._trunk(data, origin)
        x, y = aim["trunk_mm"][0] / 1000.0, aim["trunk_mm"][1] / 1000.0
        surface = aim["surface_mm"] / 1000.0
        yaw = aim["yaw_mrad"] / 1000.0
        here = self._surface(data, trunk[0], trunk[1], trunk[2]) or 0.0
        carry_z = min(
            MAX_CARRY_Z_M,
            max(
                self._carry_height(data, trunk, [x, y]),
                max(here, surface) + LIFT_M,
                trunk[2] if self.holding else 0.0,
            ),
        )
        up = yaw_quat(yaw)
        horizontal = math.dist(trunk[:2], [x, y])
        legs: list[Leg] = []
        if not self.holding or carry_z > trunk[2] + 0.005:
            # straight up first (held in the air too), then across: never up and across at
            # once past the edge of whatever is on the way
            legs.append(
                Leg(
                    "lift",
                    "move",
                    max(0.35, (carry_z - trunk[2]) / 0.6),
                    to=[trunk[0], trunk[1], carry_z],
                    quat=self._upright_quat(data),
                )
            )
        carry = min(CARRY_MAX_S, max(0.5, horizontal / CARRY_SPEED_M_S))
        legs.append(Leg("carry", "move", carry, to=[x, y, carry_z], quat=up))
        fast_to = surface + LOWER_FAST_TO_M
        if carry_z > fast_to:
            legs.append(
                Leg("lower", "move", max(0.3, (carry_z - fast_to) / 0.7), to=[x, y, fast_to])
            )
        legs += [
            Leg("lower", "lower", 1.2, floor_z=surface + LOWER_FLOOR_M),
            Leg("hold", "wait", 0.05, event="release"),
        ]
        return legs

    def _right_room(self, data: Any) -> float:
        """The room over a fallen duck that righting it needs (lifted a little, upright)."""
        assert self.robot is not None
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        surface = self._surface(data, trunk[0], trunk[1], trunk[2] + 0.05) or 0.0
        z = max(trunk[2] + 0.12, surface + LOWER_FAST_TO_M + 0.02)
        return z - surface + DUCK_TOP_M + HEAD_CLEAR_M

    def _right(self, data: Any, origin: list[float] | None = None) -> list[Leg]:
        """Lift a little, turn upright at the heading it fell with, set it down."""
        assert self.robot is not None
        trunk = self._trunk(data, origin)
        surface = self._surface(data, trunk[0], trunk[1], trunk[2] + 0.05) or 0.0
        up = self._upright_quat(data)
        z = max(trunk[2] + 0.12, surface + LOWER_FAST_TO_M + 0.02)
        return [
            Leg("lift", "move", 0.8, to=[trunk[0], trunk[1], z], quat=up),
            Leg("lower", "move", 0.3, to=[trunk[0], trunk[1], surface + LOWER_FAST_TO_M]),
            Leg("lower", "lower", 1.2, floor_z=surface + LOWER_FLOOR_M),
            Leg("hold", "wait", 0.05, event="release"),
        ]

    def _touch(self, data: Any, request: dict[str, Any], start: list[float], do: str) -> list[Leg]:
        where = str(request.get("where", "head"))
        seconds = request.get("seconds", TOUCH_S if do != "near" else 1.5)
        try:
            dwell = max(0.2, min(20.0, float(seconds)))
        except (TypeError, ValueError):
            raise ValueError(msg("world.hand.Hand._touch.0c20245c")) from None
        if where == "beak":
            follow, offset = "site:mouth_tip", [0.0, 0.0, 0.035]
        elif where == "back" and do == "touch":
            return self._touch_back(data, start, dwell, do)
        elif where == "back":
            follow, offset = "trunk", [0.0, 0.0, 0.0]
        else:
            follow, offset = "site:head_imu", [0.0, 0.0, 0.06]
        speed = REACH_SPEED_M_S
        if do == "near":
            offset = [offset[0], offset[1], offset[2] + 0.08]
            speed = 0.25 if request.get("speed") == "slow" else 1.6
        reach = min(4.0, max(0.25, math.dist(start, self._point(follow, data)[0]) / speed))
        return [
            Leg("reach", "track", reach, offset=offset, follow=follow),
            Leg(
                "touch" if do != "near" else "near",
                "wait",
                dwell,
                pat=0.012 if do == "touch" else 0.0,
            ),
        ]

    def _touch_back(self, data: Any, start: list[float], dwell: float, do: str) -> list[Leg]:
        """Reach the trunk's top from above, keeping the complete hand envelope clear.

        This is deliberately back-only: other touch targets retain their current scripts.
        A conservative sphere bound protects the first orientation change; a tighter aligned
        height then uses every currently colliding duck geom, including articulated head geoms.
        """
        assert self.robot is not None
        trunk = [float(v) for v in data.xpos[self.robot.trunk]]
        quat = [float(v) for v in data.xquat[self.robot.trunk]]
        top = self._duck_top_local(data)
        clearance = 0.03
        hand_low = self._hand_low_local()
        hand_radius = self._hand_radius()
        stage_z = top + clearance - hand_low
        high_z = top + clearance + hand_radius
        home_local = _trunk_local_point(trunk, quat, start)
        start_z = max(home_local[2], high_z)
        safe_start_local = [home_local[0], home_local[1], start_z]
        safe_start = [a + b for a, b in zip(trunk, rotate(quat, safe_start_local), strict=True)]
        high_center = [a + b for a, b in zip(trunk, rotate(quat, [0.0, 0.0, high_z]), strict=True)]
        high_home = [
            a + b
            for a, b in zip(
                trunk, rotate(quat, [home_local[0], home_local[1], high_z]), strict=True
            )
        ]
        if max(safe_start[2], high_center[2], high_home[2]) > MAX_CARRY_Z_M:
            raise HandRefused(REFUSE_NO_ROOM, msg("world.hand.Hand._touch_back.no_room"))

        # Candidate 03 uses the thumb's distal capsule after a back-only local yaw. Its
        # center is at hand-local (+0.01,+0.068,+0.03), radius 11 mm; the anchor puts the
        # lowest point at the compiled trunk surface near (-32,0,40) mm. This is specific
        # to the shipped MJCF and remains an experimental path, not a general skin model.
        anchor = [-0.100, 0.010, 0.021186137105937256]
        rear_x, rear_y = -0.160, 0.010
        rear_stage = [rear_x, rear_y, stage_z]
        rear_anchor = [rear_x, rear_y, anchor[2]]
        rear_high = [rear_x, rear_y, high_z]
        wrist_yaw = yaw_quat(-math.pi / 2)
        trunk_conjugate = [quat[0], -quat[1], -quat[2], -quat[3]]
        hand_quat = [float(v) for v in data.mocap_quat[self.mocap]]
        home_orientation = quat_mul(trunk_conjugate, hand_quat)
        # Rise with the current hand pose. Only after the whole hand envelope clears the
        # duck do we turn the wrist and move to the rear staging column.
        lift = Leg(
            "back-clearance",
            "move",
            max(0.2, math.dist(start, safe_start) / 0.25),
            to=safe_start,
            quat=hand_quat,
        )
        high = Leg(
            "back-clearance",
            "track",
            0.5,
            offset=rear_high,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        aligned = Leg(
            "back-stage",
            "track",
            max(0.35, (high_z - stage_z) / 0.5),
            offset=rear_stage,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        descend = max(0.25, (stage_z - anchor[2]) / BACK_TOUCH_DESCEND_M_S)
        behind = Leg(
            "back-approach",
            "track",
            descend,
            offset=rear_anchor,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        approach = Leg(
            "back-approach",
            "track",
            max(0.25, (anchor[0] - rear_x) / BACK_TOUCH_DESCEND_M_S),
            offset=anchor,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        touch = Leg(
            "touch",
            "track",
            dwell,
            offset=anchor,
            local_offset=True,
            follow_live=True,
            relative_orientation=wrist_yaw,
            pat=BACK_TOUCH_PAT_M,
        )
        retreat = Leg(
            "back-retreat",
            "track",
            max(0.25, (anchor[0] - rear_x) / BACK_TOUCH_DESCEND_M_S),
            offset=rear_anchor,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        rise = Leg(
            "back-retreat",
            "track",
            max(0.35, (stage_z - anchor[2]) / 0.5),
            offset=rear_stage,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        exit_high = Leg(
            "back-clearance",
            "track",
            0.5,
            offset=rear_high,
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        # Traverse back to the start only at the full-envelope clearance height. Restore
        # the original hand orientation there, then lower in that unchanged pose.
        over_home = Leg(
            "back-clearance",
            "track",
            0.5,
            offset=[home_local[0], home_local[1], high_z],
            local_offset=True,
            relative_orientation=wrist_yaw,
        )
        restore_home = Leg(
            "back-clearance",
            "track",
            0.5,
            offset=[home_local[0], home_local[1], high_z],
            local_offset=True,
            relative_orientation=home_orientation,
        )
        lower_home = Leg(
            "retreat",
            "track",
            max(0.3, (high_z - home_local[2]) / 0.5),
            offset=home_local,
            local_offset=True,
            relative_orientation=home_orientation,
            event="park",
        )
        return [
            lift,
            high,
            aligned,
            behind,
            approach,
            touch,
            retreat,
            rise,
            exit_high,
            over_home,
            restore_home,
            lower_home,
        ]

    def _duck_top_local(self, data: Any) -> float:
        """Conservative trunk-local top of every colliding duck geom at this posture."""
        assert self.robot is not None
        origin = [float(v) for v in data.xpos[self.robot.trunk]]
        positions, radii = [], []
        for geom in range(int(self.model.ngeom)):
            body = int(self.model.geom_bodyid[geom])
            if body not in self.robot.bodies or (
                int(self.model.geom_contype[geom]) == 0
                and int(self.model.geom_conaffinity[geom]) == 0
            ):
                continue
            positions.append([float(v) for v in data.geom_xpos[geom]])
            radii.append(float(self.model.geom_rbound[geom]))
        if not positions:
            raise ValueError(msg("world.hand.Hand._duck_top_local.no_geom"))
        return _projected_geom_top(
            origin,
            [float(v) for v in data.xquat[self.robot.trunk]],
            positions,
            radii,
        )

    @staticmethod
    def _hand_low_local() -> float:
        """Lowest z extent of the palm/fingers/thumb in the hand frame."""
        lows = []
        for part in HAND_PARTS:
            if part.shape == "capsule" and part.fromto:
                lows.append(min(part.fromto[2], part.fromto[5]) - part.size[0])
            elif part.shape == "box":
                lows.append(part.pos[2] - part.size[2])
            elif part.shape in ("sphere", "ellipsoid"):
                lows.append(part.pos[2] - max(part.size))
        if not lows:
            raise ValueError(msg("world.hand.Hand._hand_low_local.no_geom"))
        return min(lows)

    @staticmethod
    def _hand_radius() -> float:
        """Bounding sphere around every primitive of the hand, for orientation changes."""
        radii = []
        for part in HAND_PARTS:
            if part.shape == "capsule" and part.fromto:
                endpoints = (part.fromto[:3], part.fromto[3:])
                radii.extend(math.sqrt(sum(v * v for v in p)) + part.size[0] for p in endpoints)
            elif part.shape == "box":
                radii.append(
                    math.sqrt(sum(v * v for v in part.pos))
                    + math.sqrt(sum(v * v for v in part.size))
                )
            elif part.shape in ("sphere", "ellipsoid"):
                radii.append(math.sqrt(sum(v * v for v in part.pos)) + max(part.size))
        if not radii:
            raise ValueError(msg("world.hand.Hand._hand_radius.no_geom"))
        return max(radii)

    def _leave(self, home: list[float]) -> list[Leg]:
        return [
            Leg("retreat", "track", 0.3, offset=[0.0, 0.0, 0.18]),
            Leg("retreat", "move", 0.45, to=list(home), event="park"),
        ]

    # ── ops and the snapshot ──────────────────────────────────────────────

    def ops(self) -> dict[str, Any]:
        return {"x_hand": self.hand_op}

    def hand_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        with body.world.lock:
            return self.command(body.world, request)

    def snapshot(self, world: Any) -> dict[str, Any]:
        data = world.data
        script = self.script
        pos = [float(v) for v in data.mocap_pos[self.mocap]] if self.mocap >= 0 else None
        parked = pos is None or pos[2] < PARK_Z_M / 2
        state = "idle"
        if script is not None and script.index < len(script.legs):
            state = script.legs[script.index].label
        elif self.holding:
            state = "hold"
        return {
            "state": state,
            "do": script.do if script is not None else None,
            "script": script.id if script is not None else None,
            "holding": self.holding,
            "held": self.holding,  # the live bridge asks robotd to sit while this is true
            "shown": not parked,
            "pos_mm": None if parked or pos is None else mm(pos),
            "quat": None if parked else [round(float(v), 5) for v in data.mocap_quat[self.mocap]],
            "target": script.target if script is not None else None,
            "eta_s": None if script is None else round(script.eta_s, 3),
            "last": self.last,
            "refused": self.refused,
            "why_zh": list(script.why_zh) if script is not None else [],
        }


def page_json() -> dict[str, Any]:
    """The hand's shapes for the page: draw them in the frame of the ``kit_hand`` mocap body
    (``x_world`` → ``mocap``; hidden while ``hand.shown`` is false)."""
    return {"body": HAND, "parts": prims_json(HAND_PARTS)}


def register(registry: Any) -> Hand:
    """The hand alone as a simhome extension (``--ext duckmind.world.hand``)."""
    part = Hand()
    register_part(registry, part, section="hand")
    return part
