"""a:play's headless rig: the official duck in a real home with the jaw and robotd's one-shot
skills, and the whole neural duck on it (ball play, carrying a toy home, sunbathing).

:class:`PlayDuck` is :class:`~duckmind.world.headless_duck.HeadlessDuck` (phase 1's port of
robotd's control path, moved into the repo) with three things robotd has and that port refuses:

* **the jaw** (:mod:`duckmind.world.jaw`, curiosity E9/E10, owner decision A4 — the pivot is a
  guess, every number that depends on it is provisional): a ``mouth`` hinge under the head and an
  XL330-like position actuator on it, driven exactly as robotd drives the real mouth servo —
  ``targets[MOUTH] = mouth_target(intent.mouth)`` every tick while driving (``robotd/src/main.rs``:
  "The mouth is not part of any policy; the intent is the only thing that moves it"), and
  ``robot.state`` joint 9 is the measured jaw angle (what duck-body reads to tell something is in
  the beak). The official MJCF has no jaw joint: without the jaw a beak can push a toy but never
  hold one, so a carry is impossible — the rig says so (``jaw=False``) instead of faking a grip.
  :func:`compose_play` adds the jaw after the household and re-maps the keyframes by joint name
  (MjSpec pads a grown keyframe at its END: after ``add_jaw`` the right leg's keyframe values
  were shifted one joint over — measured on default_home, 2026-09-28).
* **the ground pick** (``control.rs``: ``start_ground_pick``; the twist slots carry the phase
  ``(cos 2πφ, sin 2πφ)``, head and body zero, φ advancing ``dt / 4.0`` per tick until 0.7:
  2.8 s; action scale 1.0) — the whole-body skill that brings the beak down to the floor
  (curiosity calib: the beak tip 78 mm ahead, 21 mm up, 1.16 s in). The head alone cannot: with
  every head joint at its limit the beak tip stays 0.165 m above the floor standing and 0.125 m
  seated (this rig, 2026-09-28).
* **the kicks** (``kick_left`` / ``kick_right``: ``ball_kick_*.onnx``, episodic 0.5 s, zero
  command, walking tuning 0.9 — ``control.rs`` ``start_skill``; a non-chaining skill is refused
  while another runs or a ground pick runs).

And robotd's ``[control] head_alpha`` is the neural duck's 1.0 (owner 2026-09-28 「加快」), per
instance (the parent reads the module constant 0.2).

Evaluators only read the truth (:meth:`HeadlessDuck.truth`, :meth:`PlayDuck.thing_pose`); the
duck's own software reads ``robot.state``.

The whole neural duck on it (:class:`PlayNeuro`): the world's truth for the sensor models
(:mod:`duckmind.world.neuro_world`) with the sun through the windows added
(:mod:`duckmind.world.neuro_sun`), the senses (:mod:`duckmind.world.neuro_senses` plus the light
sightings of :mod:`duckmind.world.neuro_sun`), the brain with the play acts
(:mod:`duckmind.neuro.skill_play`), the motor and the body link — the live driver's tick without
its sockets and threads, on the simulation clock (~10–30× real time).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from duckmind import paths
from duckmind.world import headless_duck as HD
from duckmind.world.headless_duck import (
    ACTION_SCALE_SITSTAND,
    ACTION_SCALE_WALK,
    CMD_ALPHA,
    DEADMAN_S,
    HEAD14,
    HOME_RAMP_S,
    LEGS14,
    LOWPASS_HEAD,
    LOWPASS_LEGS,
    MOUTH,
    SITSTAND_RISE_S,
    TICK_S,
    Composed,
    HeadlessDuck,
    gravity_in_trunk,
)

# robotd's one-shots on the v5 set (the shipped policy set's manifest.json; control.rs)
KICKS = {"kick_left": "ball_kick_left.onnx", "kick_right": "ball_kick_right.onnx"}
KICK_S = 0.5  # episodic, no unwind, not chaining
GROUND_PICK_FILE = "alpha_ground_pick.onnx"
GROUND_PICK_PERIOD_S = 4.0  # control.rs SkillTuning::default().ground_pick_period
GROUND_PICK_END_PHASE = 0.7  # robotd-params DEFAULT_GROUND_PICK_END_PHASE: 2.8 s
GROUND_PICK_SCALE = 1.0  # ground_pick_action_scale
NEURAL_HEAD_ALPHA = 1.0  # the neural duck's robotd [control] head_alpha (owner 2026-09-28)


def mjcf_dir() -> str:
    return str(paths.microduck_mjcf_dir())


# ── the home with the jaw ─────────────────────────────────────────────────────


def compose_play(
    home: str | Mapping[str, Any] = "default_home",
    *,
    variant: str | Sequence[str] | None = None,
    jaw: bool = True,
    mjcf_dir_: str | None = None,
) -> Composed:
    """The home composed as :func:`~duckmind.world.headless_duck.compose` does, then — ``jaw`` —
    the jaw added (:func:`duckmind.world.jaw.add_jaw`) and every keyframe re-mapped by joint
    name (the jaw's hinge sits in the middle of the robot's joints; the mouth starts closed)."""
    import mujoco

    from duckmind.world import jaw as J

    folder = mjcf_dir_ or mjcf_dir()
    built = HD.compose(home, variant=variant, mjcf_dir=folder)
    if not jaw:
        return built
    spec = built.spec
    before = spec.compile()
    named: list[dict[str, list[float]]] = []
    for k in range(before.nkey):
        table: dict[str, list[float]] = {}
        for j in range(before.njnt):
            name = mujoco.mj_id2name(before, mujoco.mjtObj.mjOBJ_JOINT, j) or f"#{j}"
            adr = int(before.jnt_qposadr[j])
            width = {0: 7, 1: 4}.get(int(before.jnt_type[j]), 1)
            table[name] = [float(v) for v in before.key_qpos[k][adr : adr + width]]
        named.append(table)
    J.add_jaw(spec)
    after = spec.compile()
    for k, key in enumerate(spec.keys):
        qpos = [float(v) for v in after.qpos0]
        for j in range(after.njnt):
            name = mujoco.mj_id2name(after, mujoco.mjtObj.mjOBJ_JOINT, j) or f"#{j}"
            adr = int(after.jnt_qposadr[j])
            if name == J.JOINT:
                qpos[adr] = J.MOUTH_CLOSED_RAD
            elif name in named[k]:
                values = named[k][name]
                qpos[adr : adr + len(values)] = values
        key.qpos = qpos
    return Composed(spec=spec, layout=built.layout, kit=built.kit)


# ── the duck with robotd's skills and the jaw ─────────────────────────────────


CHAIN_WINDOW_S = 0.15  # control.rs CHAIN_WINDOW


@dataclass
class _Skill:
    name: str
    file: str
    remaining: float
    # a table skill (``PlayDuck.install_table``): the constant command it drives, whether it chains
    # (re-requests inside the 0.15 s window restart it), and the action scale (control.rs ActiveSkill)
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    chain: bool = False
    chain_left: float = 0.0
    duration: float = 0.0
    scale: float = ACTION_SCALE_WALK


class PlayDuck(HeadlessDuck):
    """:class:`HeadlessDuck` with the jaw, the ground pick and the kicks (module doc)."""

    def __init__(
        self,
        home: str | Mapping[str, Any] | Composed = "default_home",
        *,
        variant: str | Sequence[str] | None = None,
        jaw: bool = True,
        head_alpha: float = NEURAL_HEAD_ALPHA,
        keyframe: str = "STAND",
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        policies: str | None = None,
    ) -> None:
        import mujoco

        built = home if isinstance(home, Composed) else compose_play(home, variant=variant, jaw=jaw)
        super().__init__(
            built, mjcf_dir=mjcf_dir(), keyframe=keyframe, x=x, y=y, yaw=yaw, policies=policies
        )
        if not 0.0 < head_alpha <= 1.0:
            raise ValueError("head_alpha must be in (0, 1]")
        self.head_alpha = float(head_alpha)
        m = self.model
        self.mouth_act = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "mouth"))
        joint = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "mouth"))
        self.mouth_adr = int(m.jnt_qposadr[joint]) if joint >= 0 else -1
        self.has_jaw = self.mouth_act >= 0 and self.mouth_adr >= 0
        self._skill: _Skill | None = None
        self._table: dict[
            str, Any
        ] = {}  # robotd's [[policy.skill]] table entries by name (install_table)
        self._pick: float | None = None  # the ground pick's phase, None when not running
        self._requests: list[str] = []
        self.skill_log: list[dict[str, Any]] = []
        if self.has_jaw:
            self.data.qpos[self.mouth_adr] = self._mouth_target(0.0)
            self.data.ctrl[self.mouth_act] = self._mouth_target(0.0)
            mujoco.mj_forward(m, self.data)
        self.state_now = self._state()

    # ── intents ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mouth_target(open_fraction: float) -> float:
        from duckmind.world import jaw as J

        return J.mouth_target(open_fraction)

    def mouth(self, angle: float) -> dict[str, Any]:
        """``robot.mouth {open}``: 0..1 of the jaw's travel (−5° … +30°), applied every tick
        while driving (robotd main.rs). Without the jaw it is kept and does nothing."""
        self.intent.mouth = max(0.0, min(1.0, float(angle)))
        return {"accepted": True, "jaw": self.has_jaw}

    def install_table(self, defs: Sequence[Any]) -> list[str]:
        """robotd's ``[[policy.skill]]`` entries (``table.SkillDef``: name, path, duration, command, chain,
        action_scale) as this rig's skills: a constant command under the entry's network for ``duration``
        seconds, a chaining entry restarted by a request inside the 0.15 s window (control.rs).
        Returns the names; the body link's skill guard is theirs."""
        for d in defs:
            if d.removed or not d.path:
                continue
            self._table[d.name] = d
        return sorted(self._table)

    def do(self, skill: str) -> dict[str, Any]:
        """``robot.do``: ``sit_toggle`` as the parent; ``ground_pick``, ``kick_left``,
        ``kick_right`` and the table's entries queued for the next tick (the controller arbitrates,
        as robotd)."""
        if skill == "sit_toggle":
            return super().do(skill)
        if skill != "ground_pick" and skill not in KICKS and skill not in self._table:
            return {"accepted": False, "reason": f"no skill named {skill!r}"}
        if not self.released or self._bringup != "ready":
            return {"accepted": False, "reason": "the policy is not driving"}
        self._requests.append(skill)
        self._log("do", skill=skill, policy_before=self.policy)
        return {"accepted": True}

    def busy(self) -> bool:
        """A one-shot is running (a skill or the ground pick)."""
        return self._skill is not None or self._pick is not None

    # ── robotd's tick with the skills (control.rs Controller::step) ──────────

    def _start_requests(self) -> None:
        for name in self._requests:
            if name == "ground_pick":
                if self._pick is not None:
                    why = "ground pick already running"
                else:
                    self._pick, why = 0.0, "started"
            elif name in self._table:
                d = self._table[name]
                cur = self._skill
                if cur is not None and cur.name == name and cur.chain:
                    cur.chain_left, why = CHAIN_WINDOW_S, "chain refreshed"
                elif cur is not None and not d.chain:
                    why = "a scripted move is already running"
                elif self._pick is not None:
                    why = "a ground pick is running"
                else:
                    self._skill = _Skill(
                        name, str(d.path), float(d.duration), tuple(d.command), bool(d.chain), 0.0,
                        float(d.duration), float(d.action_scale or ACTION_SCALE_WALK),
                    )  # fmt: skip
                    why = "started"
            elif self._skill is not None:
                why = "a scripted move is already running"
            elif self._pick is not None:
                why = "a ground pick is running"
            else:
                self._skill, why = _Skill(name, KICKS[name], KICK_S), "started"
            self.skill_log.append({"t": round(self.now(), 3), "skill": name, "why": why})
        self._requests = []

    def _controller(self) -> None:
        np, d = self._np, self.data
        t = self.now()
        stale = t - self.intent.twist_at > DEADMAN_S + 1e-9
        twist = np.zeros(3) if stale else np.array(self.intent.twist, dtype=float)
        self._twist_ema += CMD_ALPHA * (twist - self._twist_ema)
        self._head_ema += self.head_alpha * (
            np.array(self.intent.head, dtype=float) - self._head_ema
        )
        q = d.qpos[self.qadr + 3 : self.qadr + 7]
        self.safety.observe(gravity_in_trunk(q)[2], TICK_S)
        if self._bringup == "homing":
            frac = (t - self._home_since) / HOME_RAMP_S
            if frac < 1.0:
                d.ctrl[self.act] = self._home_from + (self.home - self._home_from) * frac
                self._targets = np.array(d.ctrl[self.act])
                self.policy = "homing"
                return
            self._bringup = "ready"
            self._prev_targets = None
            self._last_action = np.zeros(14, dtype=np.float32)
        while self.intent.toggles > 0:
            self.intent.toggles -= 1
            if self._sit == "up":
                self._sit = "sitting"
            elif self._sit == "sitting":
                self._sit, self._rise_left = "rising", SITSTAND_RISE_S
            self._log("sit_toggle_applied", now=self._sit)
        self._start_requests()
        # windows expire first (control.rs)
        if self._skill is not None and self._skill.remaining <= 0.0:
            if self._skill.chain and self._skill.chain_left > 0.0:  # held: the window restarts
                self._skill.remaining, self._skill.chain_left = self._skill.duration, 0.0
            else:
                self.skill_log.append({"t": round(t, 3), "skill": self._skill.name, "why": "ended"})
                self._skill = None
        if self._sit == "rising" and self._rise_left <= 1e-9:
            self._sit = "up"
        cmd13 = np.zeros(13, dtype=np.float32)
        if self._skill is not None:
            net, label, scale = self._skill.file, self._skill.name, self._skill.scale
            cmd13[0:3] = self._skill.command
        elif self._pick is not None:
            angle = 2.0 * math.pi * self._pick
            cmd13[0], cmd13[1] = math.cos(angle), math.sin(angle)
            net, label, scale = GROUND_PICK_FILE, "ground_pick", GROUND_PICK_SCALE
        else:
            cmd13[3:7] = self._head_ema
            if self._sit == "sitting":
                net, label, scale = "alpha_sitstand.onnx", "sit", ACTION_SCALE_SITSTAND
                cmd13[0:3] = (1.0, 0.0, 0.0)
            elif self._sit == "rising":
                net, label, scale = "alpha_sitstand.onnx", "rise", ACTION_SCALE_SITSTAND
            else:
                net, label, scale = "velstand.onnx", "walk", ACTION_SCALE_WALK
                cmd13[0:3] = self._twist_ema
        session = HD._session(self.policies / net)
        feed = {session.get_inputs()[0].name: self._obs(cmd13)}
        action = session.run(None, feed)[0].squeeze(0).astype(np.float32)
        self._last_action = action
        targets = self.home + scale * action.astype(float)
        if self._prev_targets is not None:
            head, legs = list(HEAD14), list(LEGS14)
            prev = self._prev_targets
            targets[head] = LOWPASS_HEAD * targets[head] + (1 - LOWPASS_HEAD) * prev[head]
            targets[legs] = LOWPASS_LEGS * targets[legs] + (1 - LOWPASS_LEGS) * prev[legs]
        self._prev_targets = targets.copy()
        targets = np.clip(targets, -math.pi, math.pi)
        d.ctrl[self.act] = targets
        self._targets = targets
        if self.has_jaw:
            d.ctrl[self.mouth_act] = self._mouth_target(self.intent.mouth)
        # windows advance after the tick that used them
        if self._pick is not None:
            self._pick += TICK_S / GROUND_PICK_PERIOD_S
            if self._pick >= GROUND_PICK_END_PHASE - 1e-9:
                self.skill_log.append({"t": round(t, 3), "skill": "ground_pick", "why": "ended"})
                self._pick = None
        if self._skill is not None:
            self._skill.remaining -= TICK_S
            self._skill.chain_left = max(0.0, self._skill.chain_left - TICK_S)
        if self._sit == "rising":
            self._rise_left -= TICK_S
        self.policy = label

    def _state(self) -> dict[str, Any]:
        state = super()._state()
        if getattr(self, "has_jaw", False):
            mouth = float(self.data.qpos[self.mouth_adr])
            joints = list(state["joints"])
            joints[MOUTH] = mouth
            state["joints"] = joints
            targets = list(state["targets"])
            targets[MOUTH] = self._mouth_target(self.intent.mouth)
            state["targets"] = targets
        return state

    # ── truth, for evaluators only ───────────────────────────────────────────

    def mouth_open_deg(self) -> float | None:
        """The jaw above closed (°), truth; None without the jaw."""
        if not self.has_jaw:
            return None
        from duckmind.world import jaw as J

        return math.degrees(float(self.data.qpos[self.mouth_adr]) - J.MOUTH_CLOSED_RAD)

    def thing_pose(self, thing_id: str) -> tuple[float, float, float] | None:
        """A layout thing's centre of mass (world, m), truth."""
        props = getattr(self.kit, "props", None)
        tracked = getattr(props, "things", {}).get(thing_id) if props is not None else None
        if tracked is None:
            return None
        pos = self.data.xipos[int(tracked.body)]
        return float(pos[0]), float(pos[1]), float(pos[2])

    def put_thing(self, thing_id: str, x: float, y: float, yaw: float = 0.0) -> dict[str, Any]:
        """Put a layout thing at rest at (x, y) (the page's hand; evaluator set-up only)."""
        props = self.kit.props
        tracked = props.things[thing_id]
        with self.world.lock:
            answer = props._put(self.world, tracked, x, y, yaw, top=0.05)
        return dict(answer)

    def beak_contacts(self, thing_id: str) -> bool:
        """Truth: the jaw's plates (or the head) touch this thing now."""
        props = getattr(self.kit, "props", None)
        tracked = getattr(props, "things", {}).get(thing_id) if props is not None else None
        if tracked is None:
            return False
        body = int(tracked.body)
        m, d = self.model, self.data
        for i in range(int(d.ncon)):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[int(c.geom1)]), int(m.geom_bodyid[int(c.geom2)])
            if body in (b1, b2):
                other = b2 if b1 == body else b1
                name = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_BODY, other) or ""
                if name in ("jaw_soft", "mandible"):
                    return True
        return False


def trunk_frame(duck: HeadlessDuck, point: Sequence[float]) -> tuple[float, float, float]:
    """Truth: a world point in the trunk's level heading frame (ahead, left, up from the trunk
    origin), m."""
    tr = duck.truth()
    tx, ty, tz = tr["trunk"]
    yaw = float(tr["yaw"])
    dx, dy = float(point[0]) - tx, float(point[1]) - ty
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy, float(point[2]) - tz)


def world_point(duck: HeadlessDuck, ahead: float, left: float) -> tuple[float, float]:
    """Truth: the world (x, y) of a point ``ahead``/``left`` of the trunk origin (level)."""
    tr = duck.truth()
    tx, ty, _ = tr["trunk"]
    yaw = float(tr["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * ahead - s * left, ty + s * ahead + c * left


# ═════════════════════════════════════════════════════════════════════════════════════════════
# the whole neural duck on it (world truth → senses → PlayBrain → motor → body link → PlayDuck)
# ═════════════════════════════════════════════════════════════════════════════════════════════

AXIS_SIGMA = math.radians(8.0)  # the blob's long axis from the picture (image moments)
AXIS_RANGE_M = 0.8  # … only this near (it needs enough pixels)
AXIS_ELONG = 1.4  # … and only for a thing this elongated
PART_ELONG = 1.4


def _footprint(
    model: Any, data: Any, geoms: Sequence[int]
) -> tuple[float, float, list[tuple[float, float]], float]:
    """A thing's floor footprint (world truth): its long axis (world yaw), how elongated its most
    elongated part is, the footprint's corner points (xy) and a round part's radius."""
    import numpy as np

    pts: list[tuple[float, float]] = []
    radius = 0.0
    best_elong, best_axis = 1.0, 0.0
    for g in geoms:
        gtype = int(model.geom_type[g])
        size = [float(v) for v in model.geom_size[g]]
        centre = data.geom_xpos[g]
        mat = np.array(data.geom_xmat[g]).reshape(3, 3)
        if gtype == 6:  # box
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    p = centre + mat @ np.array([sx * size[0], sy * size[1], 0.0])
                    pts.append((float(p[0]), float(p[1])))
            lx, ly = size[0], size[1]
            elong = max(lx, ly) / max(1e-4, min(lx, ly))
            col = 0 if lx >= ly else 1
            axis = math.atan2(float(mat[1, col]), float(mat[0, col]))
            if elong > best_elong:
                best_elong, best_axis = elong, axis
        else:  # sphere, capsule, cylinder, ellipsoid: round on the floor
            r = size[0]
            radius = max(radius, r)
            for k in range(8):
                a = k * math.pi / 4
                pts.append((float(centre[0]) + r * math.cos(a), float(centre[1]) + r * math.sin(a)))
    return best_axis, best_elong, pts, radius


def apparent_width(
    pts: Sequence[tuple[float, float]], cam: Sequence[float], centre: Sequence[float]
) -> float:
    """The footprint's extent across the line of sight (m): what the picture shows as its width."""
    dx, dy = float(centre[0]) - float(cam[0]), float(centre[1]) - float(cam[1])
    n = math.hypot(dx, dy) or 1.0
    ux, uy = -dy / n, dx / n
    vals = [p[0] * ux + p[1] * uy for p in pts]
    return max(vals) - min(vals) if vals else 0.0


class PlayWorld:
    """:class:`~duckmind.world.neuro_world.NeuroWorld` with the sun through the windows (the lux
    on the floor, the camera's view of the sunny floor) and, for the things in view, what the
    picture shows of their shape (the apparent width across the line of sight, the long axis):
    truth for the senses' models only. Built by :func:`play_world` (a subclass made at call time
    so importing this module needs no MuJoCo)."""


def play_world(seed: int, sunfield: Any, clock: Any) -> Any:
    from duckmind.world import neuro_sun as NS
    from duckmind.world import neuro_world as NW

    class _PlayWorld(NW.NeuroWorld, PlayWorld):
        def __init__(self) -> None:
            super().__init__(seed=seed)
            self.sunfield = sunfield
            self.clock = clock

        def ready(self, world: Any, body: Any = None) -> None:
            super().ready(world, body)
            skip = set(self.robot.bodies)
            for cand in self.candidates:
                if cand.kind in ("thing", "person", "hand"):
                    skip.update(int(b) for b in cand.bodies)
            self.sunfield.shade.ready(world.model, sorted(skip))
            self.sunfield.refresh(world.data, self.clock(), float(world.data.time), force=True)

        def _lux(self, state: Any, point: Sequence[float]) -> float:  # type: ignore[override]
            base = NW.NeuroWorld._lux(state, point)
            if float(point[2]) > 0.25:
                return base
            return base + self.sunfield.extra_lux(float(point[0]), float(point[1]))

        def sample(self, data: Any) -> dict[str, Any]:
            now = float(data.time)
            self.sunfield.refresh(data, self.clock(), now)
            out = super().sample(data)
            cam, axes = self.camera(data)
            yaw = float(out["duck"]["yaw"])
            by_id = {c.id: c for c in self.candidates}
            for entry in out["seen"]:
                cand = by_id.get(entry["id"])
                if cand is None or cand.kind != "thing":
                    continue
                body = int(cand.bodies[0])
                geoms = tuple(
                    g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == body
                )
                axis, elong, pts, radius = _footprint(self.model, data, geoms)
                centre = data.xipos[body]
                entry["w"] = round(apparent_width(pts, cam, centre), 4)
                entry["elong"] = round(elong, 2)
                entry["ax"] = round(NW._wrap(axis - yaw), 4) if elong >= PART_ELONG else None
            state = self._light_state()

            def room_lux(p: Sequence[float]) -> float:
                return NW.NeuroWorld._lux(state, p)

            def clear(p: Sequence[float]) -> bool:
                return self._floor_seen(data, cam, p)

            out["sun"] = NS.view_entries(
                self.sunfield, data, cam, axes, yaw, NW.in_frame, clear, room_lux
            )
            readings = [
                room_lux((mx, my, 0.0)) + self.sunfield.extra_lux(mx, my)
                for mx, my in NS.meter_points(cam, axes[2])
            ]
            out["cam"]["lux_meter"] = round(NS.log_mean(readings), 2)
            return out

        def _floor_seen(self, data: Any, cam: Sequence[float], p: Sequence[float]) -> bool:
            import mujoco
            import numpy as np

            vec = np.array([p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]], dtype=float)
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                return True
            self._ray_hit[0] = -1
            got = float(
                mujoco.mj_ray(
                    self.model,
                    data,
                    np.array(cam, dtype=float),
                    vec / dist,
                    self._ray_group,
                    1,
                    self.cam_body,
                    self._ray_hit,
                )
            )
            hit = int(self._ray_hit[0])
            if got < 0 or hit < 0:
                return True
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, hit) or ""
            return got >= dist - 0.03 or "floor" in name or "rug" in name

        def snapshot(self, world: Any) -> dict[str, Any]:
            snap = super().snapshot(world)
            snap["sun"] = self.sunfield.view()
            return snap

    return _PlayWorld()


def play_senses(seed: int) -> Any:
    """:class:`~duckmind.world.neuro_senses.SenseHub` with the play channels: the light meter's
    full range (``CameraSense.lux``), the sunny floor as ``light`` sightings, a thing's apparent
    width as its size and its long axis (``Sighting.axis``), and robotd's mouth joint
    (``BodySense.mouth``). A subclass made at call time."""
    import random as _random

    from duckmind.world import neuro_senses as ns
    from duckmind.world import neuro_sun as NS

    class PlayVision(ns.VisionModel):
        def step(
            self,
            t_ms: int,
            sample: Mapping[str, Any],
            *,
            stepping: bool,
            saccade: bool,
            ego_v: float = 0.0,
        ) -> list[Any]:
            got = super().step(t_ms, sample, stepping=stepping, saccade=saccade, ego_v=ego_v)
            by_tid = {tr.tid: key for key, tr in self.tracks.items()}
            entries = {str(e.get("id")): e for e in sample.get("seen") or ()}
            band = ns.light_band(float((sample.get("cam") or {}).get("lux", 300.0)))
            out: list[Any] = []
            for s in got:
                e = entries.get(by_tid.get(s.track, ""), None)
                if e is None or e.get("kind") != "thing" or e.get("w") is None:
                    out.append(s)
                    continue
                rng = _random.Random(f"shape:{self.seed}:{s.track}:{t_ms}")
                dist = max(float(e.get("dist", 1.0)), 0.05)
                width = float(e["w"]) * (1.0 + rng.gauss(0.0, ns.SIZE_SIGMA))
                size = 2.0 * math.atan(max(width, 0.005) / 2.0 / dist)
                axis = None
                floor_rng = e.get("rng")
                if (
                    e.get("ax") is not None
                    and floor_rng is not None
                    and float(floor_rng) <= AXIS_RANGE_M
                    and float(e.get("elong", 1.0)) >= AXIS_ELONG
                    and band in ("bright", "dim")
                    and not stepping
                    and not saccade
                ):
                    a = float(e["ax"]) + rng.gauss(0.0, AXIS_SIGMA)
                    axis = (a + math.pi / 2) % math.pi - math.pi / 2
                out.append(replace(s, size=size, axis=axis))
            lux_under = float((sample.get("cam") or {}).get("lux", 300.0))
            out += self.sun.step(  # type: ignore[attr-defined]
                t_ms,
                sample.get("sun") or (),
                lux_under=lux_under,
                stepping=stepping,
                saccade=saccade,
            )
            return out

    class PlayHub(ns.SenseHub):
        def __post_init__(self) -> None:
            super().__post_init__()
            vision = PlayVision(seed=self.seed)
            vision.sun = NS.SunSightModel(seed=self.seed)  # type: ignore[attr-defined]
            self.vision = vision
            self.meter_full = ns.LightMeter(seed=self.seed + 101)

        def _see(
            self, t_ms: int, section: Mapping[str, Any], *, stepping: bool, saccade: bool
        ) -> None:
            before = len(self.pending)
            last = None
            for sample in section.get("samples") or ():
                last = sample
            super()._see(t_ms, section, stepping=stepping, saccade=saccade)
            if len(self.pending) > before and last is not None:
                cam = last.get("cam") or {}
                self.meter_full.read(t_ms, float(cam.get("lux_meter", cam.get("lux", 300.0))))
                for k in range(before, len(self.pending)):
                    item = self.pending[k]
                    item.camera = replace(item.camera, lux=round(self.meter_full.lux, 1))

        def _camera(self, t_ms: int, ego: float) -> Any:
            base = super()._camera(t_ms, ego)
            return replace(base, lux=getattr(self.camera, "lux", None))

        def step(self, doors: Any) -> Any:
            frame = super().step(doors)
            state = getattr(doors, "state", None)
            joints = getattr(state, "joints", None) if state is not None else None
            mouth = float(joints[9]) if joints is not None and len(joints) > 9 else None
            policy = str(getattr(state, "policy", "") or "").lower().replace("-", "_")
            known_policies = {
                "ground_pick",
                "ground_pick:unwind",
                "kick_left",
                "kick_left:unwind",
                "kick_right",
                "kick_right:unwind",
            }
            active_skill = policy.split(":", 1)[0] if policy in known_policies else None
            return replace(frame, body=replace(frame.body, mouth=mouth, active_skill=active_skill))

    return PlayHub(seed=int(seed))


class PlayMotor:
    """neuro:motor with the play act's own legs and beak (the integration's ``MotorCommand.fine``
    and the play mouth): a fine step goes out as it is — no burst/pause holds, no course trim
    (the motor holds a burst ≥ 0.8 s, which would triple a 0.3-s trim); the beak opens over the
    jaw's full travel for a grab (the validated 0.5 opens 12.5°: the fin needs the full 30°)."""

    def __init__(self, system: Any) -> None:
        self.system = system

    def step(self, cmd: Any, frame: Any, play: Any) -> Any:
        out = self.system.step(cmd, frame)
        from duckmind.neuro.skill_play import PLAY_ACTS

        mine = cmd.behaviour in PLAY_ACTS
        if mine and play.drive and not play.far:
            vx, vyaw = float(cmd.vx), float(cmd.vyaw)  # (already the act's, from _merge)
            if frame.body.fallen or cmd.asleep:
                vx = vyaw = 0.0
            out = replace(out, vx=vx, vyaw=vyaw)
        if mine and play.mouth is not None:
            out = replace(out, mouth=max(0.0, min(1.0, float(play.mouth))))
        if mine and play.posture in ("sit", "stand"):
            out = replace(out, posture=play.posture)
        return out


class PlayLink:
    """duck-body's body link with the play act's robotd one-shots (``robot.do ground_pick`` /
    ``kick_left`` / ``kick_right``, through duck-body's guard as a curiosity act: never while
    walking, lifted, fallen or righting, and only after the legs have been still 0.3 s) and the
    beak's full travel for a grab (``mouth_max`` 1.0)."""

    STILL_MS = 300  # curiosity prims.SKILL_STILL_MS

    def __init__(self, cfg: Any) -> None:
        from duckmind.neuro.bodylink import BodyLink

        self.link = BodyLink(replace(cfg, mouth_max=1.0))
        self.sent_seq = 0
        self.still_since: int | None = None
        self.skills_sent: list[tuple[int, str]] = []
        self.skills_refused: list[tuple[int, str, str]] = []

    def tick(
        self, state: Any, request: Any, play: Any, *, depth: Any = None, battery_pct: Any = None
    ) -> Any:
        from duckmind.duckbody import protocol as Pr
        from duckmind.duckbody.reflex import GuardContext, check_intent

        lo = self.link.tick(state, request, depth=depth, battery_pct=battery_pct)
        t_ms = int(state.t_ns // 1_000_000)
        applied = state.applied if getattr(state, "applied", None) is not None else (0.0, 0.0, 0.0)
        still = (
            abs(float(applied[0])) < 0.05
            and abs(float(applied[2])) < 0.15
            and lo.twist == (0.0, 0.0)
        )
        if still:
            self.still_since = t_ms if self.still_since is None else self.still_since
        else:
            self.still_since = None
        if play.skill and play.skill_seq != self.sent_seq:
            view = lo.prox.view
            ctx = GuardContext(
                head_home_ms=10_000,
                act="duck.curio",
                lifted=view.lifted,
                unloaded=view.unloaded,
                fallen=view.fallen,
                righting=self.link.righting.active,
            )
            intent = Pr.Intent.do(play.skill)
            why = check_intent(intent, ctx)
            ready = self.still_since is not None and t_ms - self.still_since >= self.STILL_MS
            if why is None and ready:
                self.sent_seq = play.skill_seq
                self.skills_sent.append((t_ms, play.skill))
                return replace(lo, intents=tuple(lo.intents) + (intent,))
            if why is not None:
                self.skills_refused.append((t_ms, play.skill, why))
                self.sent_seq = play.skill_seq
        return lo


class PlayNeuro:
    """The whole neural duck with the play acts, headless, in a real home (see the module doc).
    Use it as a context manager (the play entries are installed while it lives)::

        with PlayNeuro(genome, seed, x=…, y=…, yaw=…, when=datetime(2026, 9, 28, 8, 30)) as rig:
            rig.run(60.0)
    """

    def __init__(
        self,
        genome: Any,
        seed: int,
        *,
        x: float,
        y: float,
        yaw: float,
        home: str = "default_home",
        when: datetime | None = None,
        weather: str = "sunny",
        jaw: bool = True,
        noslip: int = 0,
        lamps_on: bool = True,
        variant: str | None = None,
        keyframe: str = "STAND",
    ) -> None:
        self.genome = genome
        self.seed = int(seed)
        self.args = dict(
            x=x,
            y=y,
            yaw=yaw,
            home=home,
            when=when,
            weather=weather,
            jaw=jaw,
            noslip=noslip,
            lamps_on=lamps_on,
            variant=variant,
            keyframe=keyframe,
        )
        self._stack: Any = None

    def __enter__(self) -> PlayNeuro:
        import contextlib

        from duckmind.neuro.skill_play import play_installed

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(play_installed())
        try:
            self._build(**self.args)
        except Exception:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def _build(
        self, *, x, y, yaw, home, when, weather, jaw, noslip, lamps_on, variant, keyframe
    ) -> None:  # type: ignore[no-untyped-def]
        from duckmind.neuro import motor as M
        from duckmind.neuro.bodylink import config_for_motor
        from duckmind.neuro.skill_play import PlayBrain
        from duckmind.world import lights as L
        from duckmind.world import neuro_sun as NS

        self.duck = PlayDuck(home, variant=variant, jaw=jaw, keyframe=keyframe, x=x, y=y, yaw=yaw)
        d = self.duck
        if noslip:
            d.model.opt.noslip_iterations = int(noslip)
        self.when0 = when or datetime(2026, 9, 28, 8, 30)
        self.home_lights = L.HomeLights.from_layout(d.layout)
        self.lamps_on = lamps_on
        self._light_key: tuple[int, bool] | None = None
        self._light: Any = None
        self.sun = NS.SunModel.from_layout(d.layout, weather=weather)
        self.sunfield = NS.SunField(self.sun)
        self.part = play_world(self.seed, self.sunfield, self.now)
        self.part.bind(
            hand=d.kit.hand,
            people=d.kit.people,
            props=d.kit.props,
            nest=d.kit.nest,
            lights=self.light_state,
            switch_lamps=self._switch,
        )
        self.part.ready(d.world, None)
        d._before.append(self.part.before_step)
        d._after.append(self.part.after_step)
        if keyframe == "SIT":
            d.enable()  # it wakes up seated where it was (on its dock at the layout's spawn)
            d.run(2.0)
            self.stood = False
        else:
            ok, _ = d.stand(timeout=10.0, hold=1.0)
            self.stood = ok
            if ok and yaw is not None:
                # the headless body turns ~0.2–0.7 rad while it stands up (the base duck as
                # well): set it down where the scene asked (a hand's set-up, before the brain's
                # first tick; odometry restarts)
                tr = d.truth()
                d.teleport(
                    tr["trunk"][0] if x is None else float(x),
                    tr["trunk"][1] if y is None else float(y),
                    float(yaw),
                )
                d.run(0.4)
        self.pg: bool | None = None
        self.hub = play_senses(self.seed)
        self.brain = PlayBrain(self.genome, self.seed)
        side = str(getattr(getattr(d.layout, "nest", None), "side", "") or "right")
        if side in ("left", "right"):
            self.brain.set_dock(side)
        profile = M.profile_from_genome(self.genome, head_alpha=NEURAL_HEAD_ALPHA)
        self.motor = PlayMotor(M.MotorSystem(profile, self.seed))
        self.link = PlayLink(config_for_motor(M.COMMAND_MAX))
        self.last: Any = None
        self.section: Any = None
        self.ticks = 0
        self.t0_ns: int | None = None
        self.cmd: Any = None
        self.tel: Any = None
        self.out: Any = None
        self.frame: Any = None
        self.day_minute0 = self.when0.hour * 60 + self.when0.minute + self.when0.second / 60.0

    # the clocks
    def now(self) -> datetime:
        return self.when0 + timedelta(seconds=float(self.duck.now()))

    def light_state(self) -> Any:
        from duckmind.world import lights as L

        when = self.now()
        key = (int(self.duck.now() // 60), self.lamps_on)
        if key != self._light_key:
            self._light_key = key
            self._light = L.home_light_state(when, self.home_lights, lamps_on=self.lamps_on)
        return self._light

    def _switch(self, on: bool) -> None:
        self.lamps_on = bool(on)

    def step(self) -> None:
        from duckmind.duckbody.robotd_client import RobotState
        from duckmind.neuro.bodylink import BodyRequest
        from duckmind.simweb.neuro_live import _events_of

        d = self.duck
        d.tick()
        if self.section is None or self.ticks % 10 == 0:  # the world's truth at 5 Hz, as live
            self.part.pump()
            with d.world.lock:
                self.section = self.part.snapshot(d.world)
            nest = getattr(d.kit, "nest", None)
            # the PG pin (a real sensor op: the nest's charger through the contact), 5 Hz as live
            self.pg = (
                bool(nest.contact_closed())
                if nest is not None and hasattr(nest, "contact_closed")
                else None
            )
        state = RobotState.from_wire(d.state())
        if self.t0_ns is None:
            self.t0_ns = state.t_ns
        t_ms = (state.t_ns - self.t0_ns) // 1_000_000
        doors = SimpleNamespace(
            t_ms=t_ms,
            state=state,
            link=self.last,
            events=_events_of(self.last),
            depth=None,
            tof=None,
            pg=self.pg,
            day_minute=self.day_minute0 + d.now() / 60.0,
            world={"neuro": self.section},
        )
        frame = self.hub(doors)
        cmd, tel = self.brain.step(t_ms, frame)
        play = self.brain.play_out
        out = self.motor.step(cmd, frame, play)
        request = BodyRequest(
            vx=float(out.vx),
            vyaw=float(out.vyaw),
            head=(float(out.head[0]), float(out.head[1]), float(out.head[2]), float(out.head[3])),
            mouth=float(out.mouth),
            posture=str(out.posture),
            approach=bool(cmd.approach),
            docking=bool(cmd.docking),
            back_arc=bool(cmd.back_arc) or (float(out.vx) < 0.0 and play.reverse),
        )
        lo = self.link.tick(state, request, play, depth=None, battery_pct=None)
        self.last = lo
        for intent in lo.intents:
            method, p = intent.method, dict(intent.params or {})
            if method == "robot.move":
                d.move(float(p.get("vx", 0.0)), float(p.get("vy", 0.0)), float(p.get("vyaw", 0.0)))
            elif method == "robot.head":
                names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
                d.head([float(p.get(n, 0.0)) for n in names])
            elif method == "robot.mouth":
                d.mouth(float(p.get("open", 0.0)))
            elif method == "robot.do":
                d.do(str(p.get("skill", "")))
        self.cmd, self.tel, self.out, self.frame = cmd, tel, out, frame
        self.ticks += 1

    def run(self, seconds: float, each: Any = None) -> bool:
        for _ in range(int(round(seconds * 50))):
            self.step()
            if each is not None and each(self):
                return True
        return False

    # truth, for the evaluation only
    def pose(self) -> tuple[float, float, float]:
        tr = self.duck.truth()
        return float(tr["trunk"][0]), float(tr["trunk"][1]), float(tr["yaw"])

    def fallen(self) -> bool:
        return bool(self.duck.safety.fallen) or self.duck.tilt_deg() > 40.0

    def drop_spot_world(self) -> tuple[float, float] | None:
        """Truth: where the spot beside the dock is in the world (the brain's own rule)."""
        from duckmind.neuro.skill_carry import DROP_OUT_MM, DROP_SIDE_MM

        nest = getattr(self.duck.layout, "nest", None)
        if nest is None:
            return None
        nx, ny = float(nest.x_mm) / 1000.0, float(nest.y_mm) / 1000.0
        heading = float(nest.heading_mrad) / 1000.0
        side = 1.0 if str(getattr(nest, "side", "right")) == "right" else -1.0
        out, lat = DROP_OUT_MM / 1000.0, side * DROP_SIDE_MM / 1000.0
        c, s = math.cos(heading), math.sin(heading)
        return nx + c * out - s * lat, ny + s * out + c * lat

    def sun_share(self) -> float:
        """Truth: the share of the duck's footprint (9 points, ±6 cm) in direct sun."""
        x, y, yaw = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [
            (x + c * a - s * b, y + s * a + c * b)
            for a in (-0.06, 0.0, 0.06)
            for b in (-0.05, 0.0, 0.05)
        ]
        return self.sunfield.lit_share(pts)


__all__ = [
    "GROUND_PICK_END_PHASE",
    "GROUND_PICK_PERIOD_S",
    "KICKS",
    "KICK_S",
    "NEURAL_HEAD_ALPHA",
    "PlayDuck",
    "PlayLink",
    "PlayMotor",
    "PlayNeuro",
    "apparent_width",
    "play_senses",
    "play_world",
    "compose_play",
    "mjcf_dir",
    "trunk_frame",
    "world_point",
]
