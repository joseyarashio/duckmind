"""The things in the home: toys and everyday objects the duck can bump, peck and kick
(curiosity spec §4.3, §12, §17 P0; physical-design spec §11.5; owner decision B5).

Every thing is a free body the layout put there (``homelayout``: ``thing_<id>``, or an
official ``obj_*`` / ``ball_*`` in the apartment — B5 keeps the official 4 cubes and 3 balls as
toys, the two badly placed ones moved in memory). This part knows them by their layout id and
gives the page and the evaluators (never Duck Core):

* ``x_props`` — every thing: id, name, class, verbs, room, pose, speed, how far it has moved
  from where the layout put it, and what it is under (``under``: a static collider above it,
  e.g. the sofa);
* ``x_place_prop`` — put one somewhere (at rest on the highest surface under the spot, or on
  the floor); refused when it would sit inside furniture, the duck or a person (unless
  ``force``);
* ``x_tidy`` — 收玩具: the toys into the layout's keepsake zone (the toy corner), in a grid; or
  ``"to": "home"``: everything back where the layout put it;
* ``x_roll_ball`` — 滾球給牠: a ball set rolling (without slipping) toward the duck or a spot,
  from where it is or from a start spot, at a speed;
* ``x_impacts`` — the knocks since the last read, for a sound model: a thing (or the duck)
  whose velocity changes by more than :data:`DV_THING_M_S` (:data:`DV_DUCK_M_S` for the duck)
  in one 5 ms step, with the impulse, the speed change and what it hit.

People's feet push things physically (D4); nothing here moves a thing by itself.

``x_world`` → ``things``: ``[{"id", "body", "zh", "cls", "pos_mm", "quat", "centre_mm",
"yaw_mrad", "speed_mm_s", "moved_mm", "room"}]``. ``pos_mm``/``quat`` are the body frame (for
drawing the layout's parts); ``centre_mm``, ``speed_mm_s`` and ``moved_mm`` are its centre of
mass. A layout thing's body frame sits at its lowest point, so on a rolling ball it is a point
of the surface going round: simhome's own ``props`` list (frame pose, and the frame origin's
speed) draws right but its ``speed_mm_s`` is not the ball's.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import lst, msg
from duckmind.world.worldkit import (
    SENSOR_GROUP,
    RobotIds,
    body_id,
    geom_name,
    layout_of,
    mm,
    mrad,
    register_part,
    rotate,
    static_supports,
    surface_at,
    yaw_quat,
)

TOY_CLASSES = frozenset({"ball_slow", "ball_fast", "handle_toy", "light_toy"})
BALL_CLASSES = frozenset({"ball_slow", "ball_fast"})
DV_THING_M_S = 0.10
DV_DUCK_M_S = 0.25
IMPACTS_KEPT = 256
BLOCK_DEPTH_M = 0.002
PLACE_TOP_M = 1.5
TIDY_GAP_M = 0.05
ROLL_FROM_M = 1.2


@dataclass
class Tracked:
    id: str  # layout id (or "duck")
    body: int
    qadr: int
    dof: int
    mass: float
    low_m: float  # body origin → its lowest point (m), when upright
    radius_m: float  # a ball's radius (0 otherwise)
    home: tuple[float, float, float]  # where the layout put it (x, y, lowest z), m
    geoms: tuple[int, ...]
    thing: Any = None  # the layout Thing
    last_v: tuple[float, float, float] = (0.0, 0.0, 0.0)  # its centre of mass's velocity
    home_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)  # its centre of mass at the start
    shift: tuple[float, ...] = (0.0, 0.0, 0.0)  # the layout's frame (lowest point) in its own


class Props:
    """The home's things (see the module docstring). One per world."""

    name = "props"

    def __init__(self) -> None:
        self.layout: Any = None
        self.shifts: dict[str, tuple[float, ...]] = {}  # body → the layout frame, in its frame
        self.model: Any = None
        self.robot: RobotIds | None = None
        self.things: dict[str, Tracked] = {}
        self.duck: Tracked | None = None
        self.supports: list[int] = []
        self.impacts: deque[dict[str, Any]] = deque(maxlen=IMPACTS_KEPT)
        self.seq = 0
        self.read_seq = 0

    # ── building ──────────────────────────────────────────────────────────

    def edit_spec(self, spec: Any, home: Any) -> None:
        """Balls get their body frame at their centre. The layout puts every thing's frame at
        its lowest point, which on a ball is a point of its surface: rolled, that point goes
        round at 28 rad/s and a 5 ms step integrates it badly (a rolling hollow ball dipped
        6 mm into the floor and sped up 8 % in a phase-2 probe; framed at its centre it rolls
        smoothly). Same sphere, same mass, same place: only the frame moves. :meth:`view`
        still reports the layout's frame (lowest point), so the page draws the layout's parts
        from it as before."""
        self.layout = layout_of(home)
        self.shifts = {}
        for thing in getattr(self.layout, "things", ()) or ():
            if thing.official or len(thing.parts) != 1 or thing.parts[0].shape != "sphere":
                continue
            body = spec.body(thing.body)
            if body is None or len(body.geoms) != 1:
                continue
            part = thing.parts[0]
            centre = [
                part.offset_mm[0] / 1000.0,
                part.offset_mm[1] / 1000.0,
                part.offset_mm[2] / 1000.0 + part.radius_mm / 1000.0,
            ]
            quat = [float(v) for v in body.quat]
            moved = rotate(quat, centre)
            body.pos = [float(body.pos[k]) + moved[k] for k in range(3)]
            geom = body.geoms[0]
            geom.pos = [float(geom.pos[k]) - centre[k] for k in range(3)]
            if body.explicitinertial:
                body.ipos = [float(body.ipos[k]) - centre[k] for k in range(3)]
            self.shifts[thing.body] = tuple(-c for c in centre)

    def ready(self, world: Any, body: Any = None) -> None:
        model, data = world.model, world.data
        self.model = model
        self.robot = RobotIds.of(model)
        self.supports = static_supports(model)
        self.things = {}
        for thing in getattr(self.layout, "things", ()) or ():
            index = body_id(model, thing.body)
            if index < 0:
                continue
            joint = int(model.body_jntadr[index])  # official things' free joints have no name
            if joint < 0:
                continue
            geoms = tuple(g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == index)
            radius = 0.0
            if thing.cls in BALL_CLASSES or thing.id.startswith("ball"):
                spheres = [
                    float(model.geom_size[g][0]) for g in geoms if int(model.geom_type[g]) == 2
                ]
                radius = max(spheres) if spheres else 0.0
            self.things[thing.id] = Tracked(
                id=thing.id,
                body=index,
                qadr=int(model.jnt_qposadr[joint]),
                dof=int(model.jnt_dofadr[joint]),
                mass=float(model.body_subtreemass[index]),
                low_m=_lowest(model, geoms),
                radius_m=radius,
                home=(
                    thing.pos_mm[0] / 1000.0,
                    thing.pos_mm[1] / 1000.0,
                    thing.pos_mm[2] / 1000.0,
                ),
                geoms=geoms,
                thing=thing,
                shift=self.shifts.get(thing.body, (0.0, 0.0, 0.0)),
            )
        robot = self.robot
        self.duck = Tracked(
            id="duck",
            body=robot.trunk,
            qadr=robot.qadr,
            dof=robot.dof,
            mass=float(model.body_subtreemass[robot.trunk]),
            low_m=0.0,
            radius_m=0.0,
            home=(0.0, 0.0, 0.0),
            geoms=robot.colliders,
        )
        for tracked in self._all():
            tracked.last_v = self._velocity(data, tracked)
            centre = data.xipos[tracked.body]
            tracked.home_centre = (float(centre[0]), float(centre[1]), float(centre[2]))

    def _all(self) -> list[Tracked]:
        return [*self.things.values(), *([self.duck] if self.duck is not None else [])]

    def _velocity(self, data: Any, tracked: Tracked) -> tuple[float, float, float]:
        """Its centre of mass's velocity (world). Not the free joint's: a layout thing's body
        frame sits at its lowest point, a point on a rolling ball's surface that goes round
        and round."""
        import mujoco
        import numpy as np

        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, tracked.body, res, 0)
        return float(res[3]), float(res[4]), float(res[5])

    # ── impacts, every step ───────────────────────────────────────────────

    def after_step(self, world: Any) -> None:
        data = world.data
        for tracked in self._all():
            v = self._velocity(data, tracked)
            last = tracked.last_v
            tracked.last_v = v
            dv = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, last, strict=True)))
            limit = DV_DUCK_M_S if tracked is self.duck else DV_THING_M_S
            if dv <= limit:
                continue
            self.seq += 1
            pos = data.xipos[tracked.body]
            self.impacts.append(
                {
                    "seq": self.seq,
                    "t": round(float(data.time), 4),
                    "body": tracked.id,
                    "other": self._partner(data, tracked),
                    "impulse_mns": int(round(tracked.mass * dv * 1000.0)),
                    "dv_mm_s": int(round(dv * 1000.0)),
                    "speed_mm_s": int(round(math.sqrt(sum(c * c for c in v)) * 1000.0)),
                    "pos_mm": mm(pos),
                }
            )

    def _partner(self, data: Any, tracked: Tracked) -> str | None:
        """What it is touching: something that moves (a person, the duck, another thing)
        before the furniture or the floor under it, the deepest contact first."""
        model = self.model
        mine = set(tracked.geoms)
        ours = self.robot.bodies if tracked is self.duck and self.robot is not None else None
        best: tuple[int, float, int] | None = None
        for contact in data.contact[: int(data.ncon)]:
            one, two = int(contact.geom1), int(contact.geom2)
            if ours is not None:
                a = int(model.geom_bodyid[one]) in ours
                b = int(model.geom_bodyid[two]) in ours
                if a == b:
                    continue
                other = two if a else one
            elif one in mine and two not in mine:
                other = two
            elif two in mine and one not in mine:
                other = one
            else:
                continue
            if int(model.geom_group[other]) == SENSOR_GROUP:
                continue
            static = (
                int(model.body_weldid[int(model.geom_bodyid[other])]) == 0
                and int(model.body_mocapid[int(model.geom_bodyid[other])]) < 0
            )
            rank = (1 if static else 0, float(contact.dist), other)
            if best is None or rank < best:
                best = rank
        return None if best is None else self._label(best[2])

    def _label(self, geom: int) -> str:
        body = int(self.model.geom_bodyid[geom])
        for tracked in self.things.values():
            if tracked.body == body:
                return tracked.id
        if self.robot is not None and body in self.robot.bodies:
            return "duck"
        return geom_name(self.model, geom)

    # ── ops ───────────────────────────────────────────────────────────────

    def ops(self) -> dict[str, Any]:
        return {
            "x_props": self._locked(self.list_things),
            "x_place_prop": self._locked(self.place),
            "x_tidy": self._locked(self.tidy),
            "x_roll_ball": self._locked(self.roll),
            "x_impacts": self._locked(self.read_impacts),
        }

    @staticmethod
    def _locked(fn: Any) -> Any:
        def op(body: Any, request: dict[str, Any]) -> dict[str, Any]:
            with body.world.lock:
                return fn(body.world, request)

        op.__name__ = fn.__name__
        return op

    def _find(self, request: dict[str, Any]) -> Tracked:
        key = str(request.get("id", ""))
        if key in self.things:
            return self.things[key]
        for tracked in self.things.values():
            if tracked.thing is not None and tracked.thing.body == key:
                return tracked
        raise ValueError(
            msg("world.props.Props._find.a1c3f0ea", key=repr(key), list=", ".join(self.things))
        )

    def view(self, data: Any, tracked: Tracked, *, under: bool = False) -> dict[str, Any]:
        """One thing: ``pos_mm`` and ``quat`` are its body frame (draw the layout's parts from
        them), ``centre_mm`` where it really is (its centre of mass: a rolled ball's frame
        origin goes round its surface), ``speed_mm_s`` and ``moved_mm`` of that centre."""
        raw = data.qpos[tracked.qadr : tracked.qadr + 3]
        quat = data.qpos[tracked.qadr + 3 : tracked.qadr + 7]
        turned = rotate([float(c) for c in quat], tracked.shift)
        pos = [float(raw[k]) + turned[k] for k in range(3)]  # the layout's frame
        centre = data.xipos[tracked.body]
        v = self._velocity(data, tracked)
        thing = tracked.thing
        x, y = float(centre[0]), float(centre[1])
        out: dict[str, Any] = {
            "id": tracked.id,
            "body": thing.body if thing is not None else None,
            "zh": thing.zh if thing is not None else tracked.id,
            "cls": thing.cls if thing is not None else None,
            "pos_mm": mm(pos),
            "quat": [round(float(c), 5) for c in quat],
            "centre_mm": mm(centre),
            "yaw_mrad": mrad(2 * math.atan2(float(quat[3]), float(quat[0]))),
            "speed_mm_s": int(round(math.sqrt(sum(c * c for c in v)) * 1000.0)),
            "moved_mm": int(round(math.dist((x, y), tracked.home_centre[:2]) * 1000.0)),
            "room": self.layout.room_at(x * 1000.0, y * 1000.0) if self.layout else None,
        }
        if under:
            out["verbs"] = list(thing.verbs) if thing is not None else []
            out["registered"] = bool(thing.registered) if thing is not None else True
            out["under"] = self._under(data, tracked)
        return out

    def _under(self, data: Any, tracked: Tracked) -> str | None:
        import mujoco
        import numpy as np

        pos = data.xipos[tracked.body]
        start = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model,
            data,
            start,
            np.array([0.0, 0.0, 1.0]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1,
            tracked.body,
            geomid,
        )
        if dist < 0 or dist > 0.6:
            return None
        geom = int(geomid[0])
        if int(self.model.body_weldid[int(self.model.geom_bodyid[geom])]) != 0:
            return None  # a person, the duck or another thing, not furniture
        return geom_name(self.model, geom)

    def snapshot(self, world: Any) -> list[dict[str, Any]]:
        return [self.view(world.data, tracked) for tracked in self.things.values()]

    def list_things(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "things": [self.view(world.data, t, under=True) for t in self.things.values()],
            "sim_time": round(float(world.data.time), 4),
        }

    # placing

    def _put(
        self,
        world: Any,
        tracked: Tracked,
        x: float,
        y: float,
        yaw: float,
        *,
        top: float = PLACE_TOP_M,
        force: bool = False,
    ) -> dict[str, Any]:
        """Set one thing at rest at (x, y) on the highest surface below ``top``; undone when it
        would be inside something (unless ``force``)."""
        import mujoco

        model, data = world.model, world.data
        hit = surface_at(model, data, self.supports, x, y, top)
        if hit is None:
            return {
                "ok": False,
                "id": tracked.id,
                "why_zh": [msg("world.props.Props._put.79c7f157")],
            }
        surface = hit[0]
        saved = (
            data.qpos[tracked.qadr : tracked.qadr + 7].copy(),
            data.qvel[tracked.dof : tracked.dof + 6].copy(),
        )
        data.qpos[tracked.qadr : tracked.qadr + 3] = [x, y, surface + tracked.low_m + 0.001]
        data.qpos[tracked.qadr + 3 : tracked.qadr + 7] = yaw_quat(yaw)
        data.qvel[tracked.dof : tracked.dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        blocked: set[str] = set()
        mine = set(tracked.geoms)
        for contact in data.contact[: int(data.ncon)]:
            one, two = int(contact.geom1), int(contact.geom2)
            if (one in mine) == (two in mine) or float(contact.dist) >= -BLOCK_DEPTH_M:
                continue
            other = two if one in mine else one
            if int(model.geom_group[other]) == SENSOR_GROUP:
                continue
            blocked.add(self._label(other))
        if blocked and not force:
            data.qpos[tracked.qadr : tracked.qadr + 7] = saved[0]
            data.qvel[tracked.dof : tracked.dof + 6] = saved[1]
            mujoco.mj_forward(model, data)
            return {
                "ok": False,
                "id": tracked.id,
                "blocked_by": sorted(blocked),
                "why_zh": [msg("world.props.Props._put.1863bb7c", list=lst(sorted(blocked)))],
            }
        tracked.last_v = (0.0, 0.0, 0.0)
        return {
            "ok": True,
            "id": tracked.id,
            "pos_mm": mm(data.qpos[tracked.qadr : tracked.qadr + 3]),
            "surface_mm": int(round(surface * 1000.0)),
            "on": geom_name(model, hit[1]),
            "blocked_by": sorted(blocked),
            "why_zh": [],
        }

    def place(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        tracked = self._find(request)
        try:
            x = int(request["x_mm"]) / 1000.0
            y = int(request["y_mm"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            raise ValueError("x_place_prop needs whole x_mm and y_mm") from None
        yaw = int(request.get("yaw_mrad", 0)) / 1000.0
        on = str(request.get("on", "top"))
        if on not in ("top", "floor"):
            raise ValueError("x_place_prop on: top (the highest surface there) or floor")
        top = PLACE_TOP_M if on == "top" else 0.05
        force = request.get("force", False)
        if not isinstance(force, bool):
            raise ValueError("x_place_prop force must be true or false")
        return self._put(world, tracked, x, y, yaw, top=top, force=force)

    def tidy(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """收玩具: toys into the keepsake zone in a grid (or every thing back home)."""
        to = str(request.get("to", "toy_corner"))
        results = []
        if to == "home":
            for tracked in self.things.values():
                x, y, _ = tracked.home
                yaw = tracked.thing.yaw_mrad / 1000.0 if tracked.thing is not None else 0.0
                results.append(self._put(world, tracked, x, y, yaw, top=0.05 + tracked.home[2]))
        else:
            zone = None
            for item in getattr(self.layout, "zones", ()) or ():
                if item.id == to or (to == "toy_corner" and item.kind == "keepsake"):
                    zone = item
                    break
            if zone is None:
                raise ValueError(msg("world.props.Props.tidy.68357fcd", to=to))
            x0, y0, x1, y1 = (v / 1000.0 for v in zone.rect_mm)
            toys = [
                t
                for t in self.things.values()
                if t.thing is not None and (t.thing.cls in TOY_CLASSES or t.thing.official)
            ]
            spots = []
            size = max((_footprint(t) for t in toys), default=0.1) + TIDY_GAP_M
            y = y0 + size / 2
            while y <= y1 - size / 2 + 1e-9:
                x = x0 + size / 2
                while x <= x1 - size / 2 + 1e-9:
                    spots.append((x, y))
                    x += size
                y += size
            free = list(spots)
            for tracked in toys:
                placed = None
                while free and placed is None:
                    x, y = free.pop(0)
                    attempt = self._put(world, tracked, x, y, 0.0, top=0.3)
                    if attempt["ok"]:
                        placed = attempt
                results.append(
                    placed
                    or {
                        "ok": False,
                        "id": tracked.id,
                        "why_zh": [msg("world.props.Props.tidy.77637120")],
                    }
                )
        return {"ok": all(r["ok"] for r in results), "to": to, "things": results}

    def roll(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """滾球給牠: a ball rolling without slipping toward the duck or a spot."""
        import numpy as np

        data = world.data
        if "id" in request:
            ball = self._find(request)
        else:
            balls = [t for t in self.things.values() if t.radius_m > 0]
            if not balls:
                raise ValueError(msg("world.props.Props.roll.9190669c"))
            ball = balls[0]
        if ball.radius_m <= 0:
            raise ValueError(msg("world.props.Props.roll.d5839298", ball_id=ball.id))
        speed = int(request.get("speed_mm_s", 600)) / 1000.0
        if not 0.05 <= speed <= 3.0:
            raise ValueError("speed_mm_s must be 50–3000")
        toward = request.get("toward", "duck")
        assert self.robot is not None
        duck = data.xpos[self.robot.trunk]
        if toward == "duck":
            aim = (float(duck[0]), float(duck[1]))
        elif isinstance(toward, (list, tuple)) and len(toward) == 2:
            aim = (int(toward[0]) / 1000.0, int(toward[1]) / 1000.0)
        else:
            raise ValueError("x_roll_ball toward: 'duck' or [x_mm, y_mm]")
        start = request.get("from_mm")
        placed = None
        if start is None and toward == "duck" and request.get("from", "") != "here":
            # from in front of the duck, ROLL_FROM_M away (or the nearest place it fits)
            quat = [float(v) for v in data.xquat[self.robot.trunk]]
            facing = 2 * math.atan2(quat[3], quat[0])
            for turn in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi):
                sx = aim[0] + ROLL_FROM_M * math.cos(facing + turn)
                sy = aim[1] + ROLL_FROM_M * math.sin(facing + turn)
                attempt = self._put(world, ball, sx, sy, 0.0, top=0.05)
                if attempt["ok"]:
                    placed = attempt
                    break
            if placed is None:
                raise ValueError(msg("world.props.Props.roll.5910cd4a"))
        elif start is not None:
            if not isinstance(start, (list, tuple)) or len(start) != 2:
                raise ValueError("from_mm is [x_mm, y_mm]")
            placed = self._put(
                world, ball, int(start[0]) / 1000.0, int(start[1]) / 1000.0, 0.0, top=0.05
            )
            if not placed["ok"]:
                return placed
        centre = data.xipos[ball.body]
        dx, dy = aim[0] - float(centre[0]), aim[1] - float(centre[1])
        length = math.hypot(dx, dy)
        if length < 1e-6:
            raise ValueError(msg("world.props.Props.roll.ccc6c47c"))
        ux, uy = dx / length, dy / length
        # rolling without slipping: the centre at v, ω = (z × v) / r; the free joint's own
        # velocity is its frame origin's (v + ω × (origin − centre)) and ω in the body frame
        v = np.array([ux * speed, uy * speed, 0.0])
        w = np.array([-uy * speed, ux * speed, 0.0]) / ball.radius_m
        arm = np.array(data.xpos[ball.body]) - np.array(centre)
        rotation = np.array(data.xmat[ball.body]).reshape(3, 3)
        data.qvel[ball.dof : ball.dof + 3] = v + np.cross(w, arm)
        data.qvel[ball.dof + 3 : ball.dof + 6] = rotation.T @ w
        ball.last_v = (float(v[0]), float(v[1]), 0.0)
        return {
            "ok": True,
            "id": ball.id,
            "from_mm": mm(centre),
            "toward_mm": mm(aim),
            "velocity_mm_s": mm(v),
            "why_zh": [],
        }

    def read_impacts(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        since = request.get("since")
        if since is None:
            since = self.read_seq
        if isinstance(since, bool) or not isinstance(since, int):
            raise ValueError("x_impacts since must be a whole number")
        events = [e for e in self.impacts if e["seq"] > since]
        self.read_seq = self.seq
        oldest = self.impacts[0]["seq"] if self.impacts else self.seq + 1
        return {
            "impacts": events,
            "next": self.seq,
            "lost": max(0, oldest - since - 1) if since < oldest - 1 else 0,
            "sim_time": round(float(world.data.time), 4),
        }


def _lowest(model: Any, geoms: tuple[int, ...]) -> float:
    """How far below its body origin a thing reaches when upright (m)."""
    low = 0.0
    for g in geoms:
        kind = int(model.geom_type[g])
        size = [float(v) for v in model.geom_size[g]]
        z = float(model.geom_pos[g][2])
        if kind == 2:  # sphere
            reach = size[0]
        elif kind == 6:  # box (upright parts)
            reach = size[2]
        elif kind == 5:  # cylinder
            reach = size[1]
        elif kind == 3:  # capsule
            reach = size[1] + size[0]
        else:
            reach = float(model.geom_rbound[g])
        low = max(low, reach - z)
    return low


def _footprint(tracked: Tracked) -> float:
    thing = tracked.thing
    if thing is None:
        return 0.1
    x0, y0, x1, y1 = thing.footprint()
    return max(x1 - x0, y1 - y0) / 1000.0


def register(registry: Any) -> Props:
    """The things alone as a simhome extension (``--ext duckmind.world.props``)."""
    part = Props()
    register_part(registry, part, section="things")
    return part
