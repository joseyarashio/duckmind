"""The family's bodies in the physics world (physical-design spec §11.5, W01/W02; owner decisions
D3, D4).

Kinematic people made of our own primitives: legs and feet that collide (they block the duck
and push its toys: D4), a torso, arms and a head that the camera and the ToF see but nothing
bumps. Each person is three mocap bodies (the body at the hips, and each leg swinging from its
hip while walking), parked under the floor while away. One body per persona
(:data:`PERSONAS`: 主人, 伴侶, 小孩, 長輩, 訪客), each with its own height and clothes.

People only move when asked (D3: the owner presses a family member's button; no routines yet).
``come`` brings a person in from the layout's entry spot (``agents.spots``) and walks them
around the furniture to a spot facing the duck, about :data:`STAND_OFF_M` from it; ``leave``
walks them back out. ``stand`` puts a person who is not here at a place at once (facing the duck;
the nearest free place, :data:`APART_M` from anyone else here) and walks one who is here to it:
the family member who speaks from a place in the home (live voice) is there to be seen. While walking, a person keeps the duck :data:`KEEP_M` (40 cm) from their
feet (D4): paths are planned around it, a person whose way ahead comes too close stops and
re-plans, and one the duck walks up to steps back. A standing person does not move for a
toy; walking feet push toys like real feet do.

Paths are planned on a 10 cm grid from the layout (every colliding element below head height,
every drop, the nest), grown by a person's half-width, off the lock (the world's lock is taken
only to read where the duck is and to install a plan). The per-step hook only moves bodies
along the plan: microseconds.

Ops (evaluator/world; never for Duck Core)::

    {"op": "x_people", "do": "come", "who": "child", "to": "duck" | "spot:<id>" | [x_mm, y_mm],
     "from": "entry"}
    {"op": "x_people", "do": "leave" | "stop", "who": "child"}
    {"op": "x_people", "do": "stand", "who": "partner", "at": [x_mm, y_mm]}
    {"op": "x_people", "do": "status"}

``x_world`` → ``people``: a list of the people here, each ``{"who", "zh", "state", "pos_mm",
"heading_mrad", "goal_mm", "eta_s" (sim time of arrival), "eta_in_s", "hand_mm",
"clearance_mm", "min_clearance_mm", "height_mm", "why_zh"}``; ``clearance_mm`` is from the
nearest foot to the duck's trunk, ``min_clearance_mm`` the least while they walked.
"""

from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import msg
from duckmind.world.worldkit import (
    PARK_Z_M,
    Prim,
    RobotIds,
    add_prims,
    body_id,
    layout_of,
    mm,
    mrad,
    prims_json,
    register_part,
    wrap,
    yaw_quat,
)

GROUP_SEEN = 0
KEEP_M = 0.40  # D4: feet at least this far from the duck
FOOT_REACH_M = 0.38  # a toe's farthest reach from the hips' centre (swing + shoe)
KEEP_CENTRE_M = KEEP_M + FOOT_REACH_M  # the hips stay this far from the duck
WALK_MARGIN_M = 0.08  # paths keep this much more (the duck may be walking toward them)
PLAN_KEEP_M = KEEP_CENTRE_M + WALK_MARGIN_M  # the hips' keep-out around the duck, walking
STAND_OFF_M = PLAN_KEEP_M + 0.04  # where a person comes to stand, from the duck
BACK_OFF_M = KEEP_M + 0.08  # a standing person's feet nearer than this to the duck: step back
HALF_WIDTH_M = 0.25  # a person's half-width, for planning around furniture
CELL_M = 0.10
LOOK_AHEAD_M = 0.8
CHECK_EVERY_S = 0.05
STEP_BACK_M_S = 0.5  # faster than the duck walks (0.3 m/s)
LEGS_HOME_RAD_S = 6.0  # a person who stops brings the feet together
SWING_RAD = 0.22
STRIDE_M = 0.62
TURN_RAD_S = 4.0
BLOCK_BELOW_MM = 1500  # elements lower than this block a walking person
APART_M = 0.55  # two people stand at least this far apart (``stand``)
STAND_SEARCH_M = 2.0  # a place asked that is not free: the nearest free one within this (m) …
STAND_KEEP_M = 0.66  # a person who stands still is at least this far (hips) from the duck: feet 0.4 m away, :data:`KEEP_M`
STAND_STEP_M = 0.1  # … on rings this far apart …
STAND_RAYS = 24  # … with this many places on a ring …
STAND_SIDE_COST = (
    0.3  # … and a bearing (rad) off the way the duck sees the place asked costs this many metres
)
STAND_MOVED_M = 0.3  # a person here farther than this from the place asked is walked to it
# A person who was asked to ``stand`` shifts their weight a little all the time, as one talking does:
# sideways, so that a camera facing them sees an image motion (0.13 m/s at its peak: 0.04 rad/s at 3 m,
# above the eye's 0.035 rad/s "moving"). The duck's eye reports a person only when it moves
# (neuro_senses ``_label``): a body that stood perfectly still would never be seen, whatever its voice.
SWAY_M = 0.04
SWAY_PERIOD_S = 2.0


@dataclass(frozen=True)
class Persona:
    id: str
    zh: str
    height_m: float
    speed_m_s: float
    shirt: tuple[float, float, float]
    trousers: tuple[float, float, float]
    hair: tuple[float, float, float]


PERSONAS: tuple[Persona, ...] = (
    Persona(
        "owner",
        msg("world.people.PERSONAS.owner_zh"),
        1.72,
        1.0,
        (0.20, 0.36, 0.62),
        (0.18, 0.2, 0.26),
        (0.12, 0.1, 0.08),
    ),
    Persona(
        "partner",
        msg("world.people.PERSONAS.partner_zh"),
        1.62,
        1.0,
        (0.78, 0.3, 0.36),
        (0.3, 0.3, 0.34),
        (0.25, 0.15, 0.1),
    ),
    Persona(
        "child",
        msg("world.people.PERSONAS.child_zh"),
        1.20,
        0.9,
        (0.96, 0.7, 0.16),
        (0.2, 0.36, 0.6),
        (0.15, 0.1, 0.06),
    ),
    Persona(
        "elder",
        msg("world.people.PERSONAS.elder_zh"),
        1.58,
        0.6,
        (0.45, 0.55, 0.42),
        (0.35, 0.3, 0.26),
        (0.8, 0.8, 0.8),
    ),
    Persona(
        "guest",
        msg("world.people.PERSONAS.guest_zh"),
        1.68,
        1.0,
        (0.5, 0.42, 0.62),
        (0.22, 0.22, 0.22),
        (0.1, 0.08, 0.06),
    ),
)
SKIN = (0.91, 0.74, 0.62)
SHOES = (0.25, 0.2, 0.18)


def persona(who: str) -> Persona:
    for item in PERSONAS:
        if item.id == who:
            return item
    raise ValueError(f"no family member {who!r}: one of {', '.join(p.id for p in PERSONAS)}")


def hip_height(p: Persona) -> float:
    return 0.53 * p.height_m


def person_parts(p: Persona) -> dict[str, tuple[float, tuple[Prim, ...]]]:
    """A person's three bodies (suffix → (sideways offset of its frame, its primitives)): the
    body at the hips (torso, arms, hands, head, hair: seen, never bumped) and each leg from
    its hip (leg and shoe: they collide, D4)."""
    scale = p.height_m / 1.72
    hip = hip_height(p)
    top = p.height_m - hip  # from the hips to the top of the head
    head_r = 0.105 * scale
    torso_h = top - 2 * head_r - 0.06 * scale
    shoulder = 0.06 + torso_h - 0.04 * scale
    name = f"person_{p.id}"
    upper: list[Prim] = [
        Prim(
            f"{name}_hips",
            "box",
            p.trousers,
            (0.1 * scale, 0.17 * scale, 0.07 * scale),
            pos=(0, 0, 0.02),
        ),
        Prim(
            f"{name}_torso",
            "box",
            p.shirt,
            (0.1 * scale, 0.19 * scale, torso_h / 2),
            pos=(0, 0, 0.06 + torso_h / 2),
        ),
    ]
    for side, y in (("l", 0.24), ("r", -0.24)):
        hand_y = (y + (0.02 if y > 0 else -0.02)) * scale
        upper.append(
            Prim(
                f"{name}_arm_{side}",
                "capsule",
                p.shirt,
                (0.045 * scale,),
                fromto=(0, y * scale, shoulder, 0.03, hand_y, shoulder - 0.52 * scale),
            )
        )
        upper.append(
            Prim(
                f"{name}_hand_{side}",
                "sphere",
                SKIN,
                (0.05 * scale,),
                pos=(0.03, hand_y, shoulder - 0.58 * scale),
            )
        )
    upper.append(Prim(f"{name}_head", "sphere", SKIN, (head_r,), pos=(0, 0, top - head_r)))
    upper.append(
        Prim(
            f"{name}_hair",
            "ellipsoid",
            p.hair,
            (head_r * 1.05, head_r * 1.02, head_r * 0.6),
            pos=(-0.01, 0, top - head_r * 0.75),
        )
    )
    out: dict[str, tuple[float, tuple[Prim, ...]]] = {"": (0.0, tuple(upper))}
    for side, y in (("l", 0.1), ("r", -0.1)):
        leg = f"{name}_leg_{side}"
        out[f"_leg_{side}"] = (
            y * scale,
            (
                Prim(
                    f"{leg}_leg",
                    "capsule",
                    p.trousers,
                    (0.065 * scale,),
                    fromto=(0, 0, -0.02, 0, 0, -(hip - 0.12)),
                    collide=True,
                ),
                Prim(
                    f"{leg}_foot",
                    "box",
                    SHOES,
                    (0.125 * scale, 0.05 * scale, 0.035),
                    pos=(0.04 * scale, 0, -hip + 0.037),
                    collide=True,
                ),
            ),
        )
    return out


def page_json() -> dict[str, Any]:
    """Every persona's bodies for the page: draw each body's parts in the frame of its mocap
    body (``person_<id>``, ``person_<id>_leg_l``/``_r`` in ``x_world`` → ``mocap``), only for
    the people in ``x_world`` → ``people``."""
    return {
        p.id: {
            "zh": p.zh,
            "height_mm": int(round(p.height_m * 1000)),
            "bodies": {
                f"person_{p.id}{suffix}": prims_json(prims)
                for suffix, (_, prims) in person_parts(p).items()
            },
        }
        for p in PERSONAS
    }


# ── the floor plan for walking ──────────────────────────────────────────────────


class Floor:
    """Where a person can stand: a grid over the layout, grown by a person's half-width."""

    def __init__(self, layout: Any, *, cell_m: float = CELL_M) -> None:
        self.cell = cell_m
        x0, y0, x1, y1 = (v / 1000.0 for v in layout.bounds_mm)
        self.x0, self.y0 = x0, y0
        self.nx = max(1, int(math.ceil((x1 - x0) / cell_m)))
        self.ny = max(1, int(math.ceil((y1 - y0) / cell_m)))
        rects: list[tuple[float, float, float, float]] = []
        for element in layout.elements:
            if not element.collide or element.pos_mm[2] >= BLOCK_BELOW_MM:
                continue
            if element.kind in ("floor", "stairs", "pit_bottom", "platform", "rug", "mat"):
                continue
            if element.kind in ("threshold", "cable"):  # a person steps over a sill or a cord
                continue
            rects.append(tuple(v / 1000.0 for v in element.footprint()))  # type: ignore[arg-type]
        for drop in layout.drops:
            a, b, c, d = (v / 1000.0 for v in drop.rect_mm)
            rects.append((a, b, c, d))
        nest = getattr(layout, "nest", None)
        if nest is not None:  # the open nest's pad, or the whole dock (duckmind.world.dock)
            keep = getattr(nest, "keepout_mm", None)
            if keep is None:
                r = nest.pad_radius_mm + 20
                keep = (nest.x_mm - r, nest.y_mm - r, nest.x_mm + r, nest.y_mm + r)
            a, b, c, d = (v / 1000.0 for v in keep)
            rects.append((a, b, c, d))
        self.rects = rects
        grow = HALF_WIDTH_M
        self.blocked = bytearray(self.nx * self.ny)
        for a, b, c, d in rects:
            i0, j0 = self.cell_of(a - grow, b - grow)
            i1, j1 = self.cell_of(c + grow, d + grow)
            for i in range(max(0, i0), min(self.nx, i1 + 1)):
                for j in range(max(0, j0), min(self.ny, j1 + 1)):
                    self.blocked[j * self.nx + i] = 1
        for i in range(self.nx):  # the home's edge
            for j in (0, self.ny - 1):
                self.blocked[j * self.nx + i] = 1
        for j in range(self.ny):
            for i in (0, self.nx - 1):
                self.blocked[j * self.nx + i] = 1

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / self.cell), int((y - self.y0) / self.cell)

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def free(self, x: float, y: float, duck: tuple[float, float] | None = None) -> bool:
        i, j = self.cell_of(x, y)
        if not self.inside(i, j) or self.blocked[j * self.nx + i]:
            return False
        return duck is None or math.dist((x, y), duck) >= PLAN_KEEP_M

    def line_free(
        self, a: tuple[float, float], b: tuple[float, float], duck: tuple[float, float] | None
    ) -> bool:
        steps = max(1, int(math.dist(a, b) / (self.cell * 0.5)))
        return all(
            self.free(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps, duck)
            for k in range(steps + 1)
        )

    def costs(
        self, start: tuple[float, float], duck: tuple[float, float] | None
    ) -> tuple[dict[int, float], dict[int, int]]:
        """Dijkstra from ``start`` over free cells (8-connected): cost and parent per cell."""
        si, sj = self.cell_of(*start)
        if not self.inside(si, sj):
            return {}, {}
        begin = sj * self.nx + si
        cost = {begin: 0.0}
        parent: dict[int, int] = {}
        heap = [(0.0, begin)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        steps += [(1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while heap:
            c, k = heapq.heappop(heap)
            if c > cost.get(k, math.inf):
                continue
            i, j = k % self.nx, k // self.nx
            for di, dj, w in steps:
                ni, nj = i + di, j + dj
                if not self.inside(ni, nj):
                    continue
                n = nj * self.nx + ni
                x, y = self.centre(ni, nj)
                if self.blocked[n] or (duck is not None and math.dist((x, y), duck) < PLAN_KEEP_M):
                    continue
                if di and dj and (self.blocked[j * self.nx + ni] or self.blocked[nj * self.nx + i]):
                    continue
                nc = c + w * self.cell
                if nc < cost.get(n, math.inf):
                    cost[n] = nc
                    parent[n] = k
                    heapq.heappush(heap, (nc, n))
        return cost, parent

    def path(
        self,
        start: tuple[float, float],
        goals: list[tuple[float, float]],
        duck: tuple[float, float] | None,
    ) -> list[tuple[float, float]] | None:
        """The shortest free path from start to the first reachable of ``goals`` (in the
        order given, ties to the cheapest), smoothed to straight runs."""
        cost, parent = self.costs(start, duck)
        best: tuple[float, int, tuple[float, float]] | None = None
        for rank, goal in enumerate(goals):
            gi, gj = self.cell_of(*goal)
            if not self.inside(gi, gj):
                continue
            k = gj * self.nx + gi
            if k in cost:
                score = cost[k] + 0.3 * rank
                if best is None or score < best[0]:
                    best = (score, k, goal)
        if best is None:
            return None
        _, k, goal = best
        cells = [k]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [start] + [self.centre(c % self.nx, c // self.nx) for c in cells[1:-1]] + [goal]
        smooth = [points[0]]
        index = 0
        while index < len(points) - 1:
            nxt = len(points) - 1
            while nxt > index + 1 and not self.line_free(points[index], points[nxt], duck):
                nxt -= 1
            smooth.append(points[nxt])
            index = nxt
        return smooth


# ── one person ──────────────────────────────────────────────────────────────────


@dataclass
class Walker:
    persona: Persona
    body: int = -1  # mocap ids
    leg_l: int = -1
    leg_r: int = -1
    here: bool = False
    state: str = "away"  # away | walking | waiting | standing | leaving | stepping_back
    pos: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)
    along: float = 0.0  # metres along the path walked
    goal_kind: str = "duck"  # duck | point | exit
    goal: tuple[float, float] | None = None
    phase: float = 0.0  # gait phase (radians)
    needs_plan: bool = False
    leave_after: bool = False
    sway: bool = (
        False  # standing, shifting their weight (set by ``stand``, cleared by come and leave)
    )
    last_s: float | None = None
    next_check_s: float = 0.0
    min_clearance: float | None = None
    clearance: float | None = None
    eta_s: float | None = None
    why_zh: list[str] = field(default_factory=list)


class People:
    """The family's bodies (see the module docstring). One per world."""

    name = "people"

    def __init__(self) -> None:
        self.layout: Any = None
        self.floor: Floor | None = None
        self.walkers: dict[str, Walker] = {p.id: Walker(p) for p in PERSONAS}
        self.model: Any = None
        self.world: Any = None
        self.robot: RobotIds | None = None
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._plan_lock = threading.RLock()  # the op thread and the planner thread

    # ── building ──────────────────────────────────────────────────────────

    def edit_spec(self, spec: Any, home: Any) -> None:
        self.layout = layout_of(home)
        for p in PERSONAS:
            for suffix, (offset_y, prims) in person_parts(p).items():
                body = spec.worldbody.add_body()
                body.name = f"person_{p.id}{suffix}"
                body.mocap = True
                body.pos = [0.0, offset_y, PARK_Z_M]
                add_prims(body, prims, group=GROUP_SEEN)

    def ready(self, world: Any, body: Any = None) -> None:
        model = world.model
        self.model, self.world = model, world
        self.robot = RobotIds.of(model)
        if self.layout is not None:
            self.floor = Floor(self.layout)
        for walker in self.walkers.values():
            p = walker.persona.id
            ids = [
                body_id(model, name)
                for name in (f"person_{p}", f"person_{p}_leg_l", f"person_{p}_leg_r")
            ]
            if min(ids) < 0:
                raise ValueError(f"the person {p!r} is missing from the compiled world")
            walker.body, walker.leg_l, walker.leg_r = (int(model.body_mocapid[i]) for i in ids)
            self._pose(world.data, walker)

    def start_thread(self) -> None:
        """The planner thread (the world server's): plans off the lock when asked."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="people-planner", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            try:
                self.pump()
            except Exception:  # a failed plan leaves the person waiting, not the world broken
                pass

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def bodies(self) -> list[int]:
        """Every mocap body id a person uses (for other parts)."""
        out = []
        for walker in self.walkers.values():
            out += [walker.body, walker.leg_l, walker.leg_r]
        return out

    # ── where things are ──────────────────────────────────────────────────

    def duck_xy(self, data: Any) -> tuple[float, float] | None:
        if self.robot is None:
            return None
        pos = data.xpos[self.robot.trunk]
        return float(pos[0]), float(pos[1])

    def hand_point(self, who: str) -> list[float] | None:
        """Where this person's right hand is (m), if they are here — the hand comes from it."""
        walker = self.walkers.get(who)
        if walker is None or not walker.here:
            return None
        p = walker.persona
        _, upper = person_parts(p)[""]
        hand = next(prim for prim in upper if prim.name.endswith("_hand_r"))
        dx, dy, dz = hand.pos
        c, s = math.cos(walker.heading), math.sin(walker.heading)
        return [
            walker.pos[0] + c * dx - s * dy,
            walker.pos[1] + s * dx + c * dy,
            hip_height(p) + dz,
        ]

    # ── planning (off the lock) ───────────────────────────────────────────

    def _goals(self, walker: Walker, duck: tuple[float, float] | None) -> list[tuple[float, float]]:
        if walker.goal_kind == "duck" and duck is not None:
            facing = 0.0
            if self.robot is not None and self.world is not None:
                quat = [float(v) for v in self.world.data.xquat[self.robot.trunk]]
                facing = 2 * math.atan2(quat[3], quat[0])
            order = sorted(range(24), key=lambda k: abs(wrap(2 * math.pi * k / 24)))
            return [
                (
                    duck[0] + STAND_OFF_M * math.cos(facing + 2 * math.pi * k / 24),
                    duck[1] + STAND_OFF_M * math.sin(facing + 2 * math.pi * k / 24),
                )
                for k in order
            ]
        return [walker.goal] if walker.goal is not None else []

    def plan(self, walker: Walker) -> bool:
        """Plan this person's way now (takes the world's lock only to read and to install)."""
        with self._plan_lock:
            return self._plan(walker)

    def _plan(self, walker: Walker) -> bool:
        if self.floor is None or self.world is None:
            return False
        with self.world.lock:
            duck = self.duck_xy(self.world.data)
            start = walker.pos
            goals = self._goals(walker, duck)
        route = self.floor.path(start, goals, duck)
        if route is None and duck is not None:
            # boxed in by the duck's keep-out (it walked up close): plan without it, the
            # per-step check still stops the person short of it
            route = self.floor.path(start, goals, None)
        with self.world.lock:
            if route is None:
                walker.state = "waiting"
                walker.why_zh = ["找不到走過去的路：先等一下"]
                walker.needs_plan = False
                return False
            walker.path = route
            walker.along = 0.0
            walker.needs_plan = False
            walker.state = "leaving" if walker.goal_kind == "exit" else "walking"
            length = sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))
            walker.eta_s = float(self.world.data.time) + length / walker.persona.speed_m_s
            walker.why_zh = []
        return True

    def pump(self) -> None:
        with self._plan_lock:
            for walker in self.walkers.values():
                if walker.here and walker.needs_plan:
                    self.plan(walker)

    # ── the per-step hook (lock held) ─────────────────────────────────────

    def before_step(self, world: Any) -> None:
        data = world.data
        now = float(data.time)
        for walker in self.walkers.values():
            if not walker.here:
                continue
            dt = 0.0 if walker.last_s is None else max(0.0, now - walker.last_s)
            walker.last_s = now
            duck = self.duck_xy(data)
            if walker.state in ("walking", "leaving") and walker.path:
                self._walk(walker, dt, duck, now)
            else:
                self._feet_together(walker, dt)
                if duck is not None:
                    self._keep_away(walker, dt, duck)
            if duck is not None:
                feet = self._feet_clearance(walker, duck)
                walker.clearance = feet
                if walker.state in ("walking", "leaving", "stepping_back"):
                    walker.min_clearance = (
                        feet if walker.min_clearance is None else min(walker.min_clearance, feet)
                    )
            self._pose(data, walker)

    def _walk(
        self, walker: Walker, dt: float, duck: tuple[float, float] | None, now: float
    ) -> None:
        if duck is not None and now >= walker.next_check_s:
            walker.next_check_s = now + CHECK_EVERY_S
            if self._path_near(walker, duck, LOOK_AHEAD_M) < PLAN_KEEP_M - 0.01:
                walker.state = "waiting"
                walker.needs_plan = True
                self._wake.set()
                return
        step = walker.persona.speed_m_s * dt
        walker.along += step
        pos, direction, done = self._at(walker.path, walker.along)
        walker.pos = pos
        if direction is not None:
            error = wrap(direction - walker.heading)
            turn = max(-TURN_RAD_S * dt, min(TURN_RAD_S * dt, error))
            walker.heading = wrap(walker.heading + turn)
        walker.phase += 2 * math.pi * step / STRIDE_M
        if done:
            walker.phase = 0.0
            walker.path = []
            walker.eta_s = None
            if walker.goal_kind == "exit":
                walker.here = False
                walker.state = "away"
            else:
                walker.state = "standing"
                if duck is not None:
                    walker.heading = math.atan2(duck[1] - pos[1], duck[0] - pos[0])

    @staticmethod
    def _feet_together(walker: Walker, dt: float) -> None:
        """Not walking: the swinging legs come back under the hips."""
        home = round(walker.phase / math.pi) * math.pi
        step = LEGS_HOME_RAD_S * dt
        walker.phase = (
            home
            if abs(walker.phase - home) <= step
            else walker.phase + math.copysign(step, home - walker.phase)
        )

    def _keep_away(self, walker: Walker, dt: float, duck: tuple[float, float]) -> None:
        """The duck came close: shuffle straight back from it (feet together, faster than it
        walks) until the feet are clear again, unless furniture is in the way."""
        feet = self._feet_clearance(walker, duck)
        if feet >= BACK_OFF_M:
            if walker.state == "stepping_back":
                walker.state = "standing"
            return
        away = math.atan2(walker.pos[1] - duck[1], walker.pos[0] - duck[0])
        step = STEP_BACK_M_S * dt
        for turn in (0.0, 0.6, -0.6, 1.2, -1.2):  # straight back, else sideways-back
            nxt = (
                walker.pos[0] + step * math.cos(away + turn),
                walker.pos[1] + step * math.sin(away + turn),
            )
            if self.floor is None or self.floor.free(*nxt):
                break
        else:
            if walker.state == "stepping_back":
                walker.state = "standing"  # boxed in by furniture: the duck came to them
            return
        walker.pos = nxt
        if walker.state in ("standing", "stepping_back"):
            walker.state = "stepping_back"

    def _path_near(self, walker: Walker, duck: tuple[float, float], ahead: float) -> float:
        """The nearest the hips come to the duck along the next ``ahead`` metres."""
        best = math.inf
        samples = 8
        for k in range(samples + 1):
            pos, _, _ = self._at(walker.path, walker.along + ahead * k / samples)
            best = min(best, math.dist(pos, duck))
        return best

    @staticmethod
    def _at(
        path: list[tuple[float, float]], along: float
    ) -> tuple[tuple[float, float], float | None, bool]:
        left = along
        for a, b in zip(path, path[1:], strict=False):
            length = math.dist(a, b)
            if length <= 1e-9:
                continue
            if left <= length:
                f = left / length
                return (
                    (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f),
                    math.atan2(b[1] - a[1], b[0] - a[0]),
                    False,
                )
            left -= length
        end = path[-1] if path else (0.0, 0.0)
        return end, None, True

    def _feet_clearance(self, walker: Walker, duck: tuple[float, float]) -> float:
        """From the nearest point of either foot to the duck's trunk (m, on the floor)."""
        best = math.inf
        swing = SWING_RAD * math.sin(walker.phase)
        hip = hip_height(walker.persona)
        scale = walker.persona.height_m / 1.72
        c, s = math.cos(walker.heading), math.sin(walker.heading)
        for side, angle in ((0.1, swing), (-0.1, -swing)):
            forward = hip * math.sin(angle)
            for fx in (0.04 * scale - 0.125 * scale, 0.04 * scale + 0.125 * scale):
                for fy in (-0.05 * scale, 0.05 * scale):
                    lx, ly = forward + fx, side * scale + fy
                    x = walker.pos[0] + c * lx - s * ly
                    y = walker.pos[1] + s * lx + c * ly
                    best = min(best, math.dist((x, y), duck))
        return best

    def _pose(self, data: Any, walker: Walker) -> None:
        p = walker.persona
        if not walker.here:
            for index, y in ((walker.body, 0.0), (walker.leg_l, 0.1), (walker.leg_r, -0.1)):
                data.mocap_pos[index] = [0.0, y, PARK_Z_M]
                data.mocap_quat[index] = [1.0, 0.0, 0.0, 0.0]
            return
        hip = hip_height(p)
        scale = p.height_m / 1.72
        heading = yaw_quat(walker.heading)
        c, s = math.cos(walker.heading), math.sin(walker.heading)
        x, y = walker.pos
        if walker.sway and walker.state == "standing":
            shift = SWAY_M * math.sin(2 * math.pi * float(data.time) / SWAY_PERIOD_S)
            x, y = x - s * shift, y + c * shift  # (sideways to the way they face)
        data.mocap_pos[walker.body] = [x, y, hip]
        data.mocap_quat[walker.body] = heading
        swing = SWING_RAD * math.sin(walker.phase)
        for index, side, angle in ((walker.leg_l, 0.1, swing), (walker.leg_r, -0.1, -swing)):
            data.mocap_pos[index] = [x - s * side * scale, y + c * side * scale, hip]
            # swing about the hip's sideways axis: heading, then a pitch
            pitch = [math.cos(-angle / 2), 0.0, math.sin(-angle / 2), 0.0]
            data.mocap_quat[index] = _mul(heading, pitch)

    # ── ops ───────────────────────────────────────────────────────────────

    def ops(self) -> dict[str, Any]:
        return {"x_people": self.people_op}

    def command(
        self, world: Any, request: dict[str, Any], *, plan_now: bool = True
    ) -> dict[str, Any]:
        """Handle an ``x_people`` request; call WITHOUT the world's lock (it takes it)."""
        do = str(request.get("do", "status"))
        if do == "status":
            with world.lock:
                return {"people": self.snapshot(world)}
        who = str(request.get("who", ""))
        walker = self.walkers.get(who)
        if walker is None:
            raise ValueError(f"x_people who {who!r}: one of {', '.join(self.walkers)}")
        if self.layout is None or self.floor is None:
            raise ValueError(msg("world.people.People.command.5e044c61"))
        with world.lock:
            if do == "come":
                to = request.get("to", "duck")
                if to == "duck":
                    goal_kind, goal = "duck", None
                elif isinstance(to, str) and to.startswith("spot:"):
                    x, y, _ = self._spot(to[5:])
                    goal_kind, goal = "point", (x, y)
                elif isinstance(to, (list, tuple)) and len(to) == 2:
                    goal_kind, goal = "point", (int(to[0]) / 1000.0, int(to[1]) / 1000.0)
                else:
                    raise ValueError("x_people to: 'duck', 'spot:<id>' or [x_mm, y_mm]")
                walker.sway = False
                if not walker.here:
                    spot = self._spot(str(request.get("from", "entry")))
                    walker.pos = spot[:2]
                    walker.heading = spot[2]
                    walker.here = True
                    walker.min_clearance = None
                    walker.last_s = float(world.data.time)
                walker.goal_kind, walker.goal = goal_kind, goal
                walker.state = "waiting"
                walker.needs_plan = True
            elif do == "leave":
                if not walker.here:
                    return {"ok": True, "who": who, "state": "away"}
                walker.sway = False
                x, y, _ = self._spot(str(request.get("to", "entry")))
                walker.goal_kind, walker.goal = "exit", (x, y)
                walker.state = "waiting"
                walker.needs_plan = True
            elif do == "stop":
                walker.path = []
                walker.state = "standing" if walker.here else "away"
                walker.needs_plan = False
            elif do == "stand":
                place = self._stand_at(world, walker, request.get("at"))
                if place is None:
                    return {"ok": False, "who": who, "state": walker.state}
                walker.sway = True
                if not walker.here:
                    duck = self.duck_xy(world.data)
                    walker.pos = place
                    walker.heading = (
                        0.0 if duck is None else math.atan2(duck[1] - place[1], duck[0] - place[0])
                    )
                    walker.here = True
                    walker.path = []
                    walker.along = 0.0
                    walker.phase = 0.0
                    walker.goal_kind, walker.goal = "point", None
                    walker.state = "standing"
                    walker.needs_plan = False
                    walker.min_clearance = None
                    walker.last_s = float(world.data.time)
                elif walker.state == "leaving" or math.dist(walker.pos, place) > STAND_MOVED_M:
                    walker.goal_kind, walker.goal = "point", place
                    walker.state = "waiting"
                    walker.needs_plan = True
            else:
                raise ValueError(f"x_people do {do!r}: come, leave, stand, stop or status")
        if plan_now and walker.needs_plan:
            self.plan(walker)
        elif walker.needs_plan:
            self._wake.set()
        with world.lock:
            now = float(world.data.time)
            return {"ok": True, "who": who, "sim_time": round(now, 4), **self._view(walker, now)}

    def _stand_at(self, world: Any, walker: Walker, at: Any) -> tuple[float, float] | None:
        """The place ``walker`` stands at for a request to stand at ``at`` ([x_mm, y_mm]): that
        place if it is free (the floor's grid, the duck's keep-out, anyone else here
        :data:`APART_M` away), else the nearest free place within :data:`STAND_SEARCH_M` (the
        duck is facing a wall and the place 1 m ahead of it is outside the home: the person
        stands where there is room, along the wall), preferring the places on the duck's side
        of the one asked; None when there is none (or ``at`` is not a place). Called with the
        world's lock."""
        try:
            want = (float(at[0]) / 1000.0, float(at[1]) / 1000.0)
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not (math.isfinite(want[0]) and math.isfinite(want[1])):
            return None
        duck = self.duck_xy(world.data)
        others = [w.pos for w in self.walkers.values() if w.here and w is not walker]

        def ok(x: float, y: float) -> bool:
            if self.floor is not None and not self.floor.free(x, y):
                return False
            if duck is not None and math.dist((x, y), duck) < STAND_KEEP_M:
                return False  # (a person who stands still needs less room than one who walks)
            return all(math.dist((x, y), o) >= APART_M for o in others)

        if ok(*want):
            return want
        way = None if duck is None else math.atan2(want[1] - duck[1], want[0] - duck[0])
        best: tuple[float, tuple[float, float]] | None = None
        radius = STAND_STEP_M
        while radius <= STAND_SEARCH_M + 1e-9:
            for k in range(STAND_RAYS):
                angle = 2 * math.pi * k / STAND_RAYS
                cand = (want[0] + radius * math.cos(angle), want[1] + radius * math.sin(angle))
                if not ok(*cand):
                    continue
                cost = radius
                if way is not None and duck is not None:
                    seen = math.atan2(cand[1] - duck[1], cand[0] - duck[0])
                    cost += STAND_SIDE_COST * abs(wrap(seen - way))
                if best is None or cost < best[0]:
                    best = (cost, cand)
            radius += STAND_STEP_M
        return None if best is None else best[1]

    def people_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self.command(body.world, request)

    def _spot(self, name: str) -> tuple[float, float, float]:
        spots = (self.layout.agents or {}).get("spots", []) if self.layout is not None else []
        for spot in spots:
            if spot.get("id") == name:
                return (
                    int(spot["x_mm"]) / 1000.0,
                    int(spot["y_mm"]) / 1000.0,
                    int(spot.get("heading_mrad", 0)) / 1000.0,
                )
        if spots:
            first = spots[0]
            return int(first["x_mm"]) / 1000.0, int(first["y_mm"]) / 1000.0, 0.0
        x0, y0, _, _ = (v / 1000.0 for v in self.layout.bounds_mm)
        return x0 + 0.5, y0 + 0.5, 0.0

    def _view(self, walker: Walker, now: float | None = None) -> dict[str, Any]:
        p = walker.persona
        hand = self.hand_point(p.id)
        eta_in = None
        if walker.eta_s is not None and now is not None:
            eta_in = round(max(0.0, walker.eta_s - now), 3)
        return {
            "zh": p.zh,
            "state": walker.state,
            "pos_mm": mm(walker.pos),
            "heading_mrad": mrad(walker.heading),
            "goal_mm": None if not walker.path else mm(walker.path[-1]),
            "eta_s": None if walker.eta_s is None else round(walker.eta_s, 3),
            "eta_in_s": eta_in,
            "hand_mm": None if hand is None else mm(hand),
            "clearance_mm": None
            if walker.clearance is None
            else int(round(walker.clearance * 1000)),
            "min_clearance_mm": None
            if walker.min_clearance is None
            else int(round(walker.min_clearance * 1000)),
            "height_mm": int(round(p.height_m * 1000)),
            "why_zh": list(walker.why_zh),
        }

    def snapshot(self, world: Any) -> list[dict[str, Any]]:
        now = float(world.data.time)
        return [
            {"who": walker.persona.id, **self._view(walker, now)}
            for walker in self.walkers.values()
            if walker.here
        ]


def _mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def register(registry: Any) -> People:
    """The family's bodies alone as a simhome extension (``--ext duckmind.world.people``)."""
    part = People()
    register_part(registry, part, section="people")
    registry.on_ready(lambda world, body: part.start_thread(), name="people.planner")
    return part
