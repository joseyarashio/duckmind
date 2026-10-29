"""The lockstep rig: a home, the headless duck and its real sensor doors, stepped on one clock
(physical-design spec §11.7 P2, §14.7, §17 期3 gates).

The world server runs robotd, tofd and the media shim as separate processes on the wall clock;
the rig runs the same physics and the same sensor models *in this process on the simulation
clock*, so a gate can run many times faster than real time and twice in a row gives the same
bytes. What the duck's software gets is exactly what its doors give on the board:

* ``robot.state`` every 50 Hz tick (:class:`~duckmind.world.headless_duck.HeadlessDuck`, robotd's
  ``RobotState`` shape, ``t_ns`` on a synthetic ``CLOCK_MONOTONIC``);
* ``tof.frame`` at tofd's delivered cadence (default :data:`TOF_PERIOD_S`, 13.8 frames/s: tofd
  sleeps a full period after each answer), from the world's multi-ray
  :class:`~duckmind.world.tof_model.TofSensor` — ``{seq, at_us, t_ns, rows, cols, distance_mm,
  status}``;
* ``media.frame`` on demand (:meth:`Rig.capture`): the world camera
  (:class:`~duckmind.world.render_scene.HomeEye`: lens, AE and white balance metered at 2 Hz of
  sim time, sensor noise), header ``{width, height, format, bytes, captured_at_unix_us, rotate}``
  and the UYVY payload, ``captured_at_unix_us`` on a synthetic wall clock (the eye bridges it);
* the world's ``pg`` sensor through the nest part, nothing else of the world.

Everything else — trunk pose, the beak, contacts, the hand's hold, the lux at the duck — is
**truth** (:meth:`Rig.truth`, :meth:`Rig.truth_light`, the ``world.*`` side stream) for the
evaluators; it is never handed to a :class:`Brain`.

**Brains.** duck-body and duck-eye run as pure, clock-injected step functions (the phase-3
architecture rule): a :class:`Brain` gets each tick's doors and returns robot intents and look
orders; the rig applies the intents at the next tick (as robotd would), and for a look order it
captures the next frame at or after ``not_before_ns`` and hands (look, header, payload, poses) to
the brain's eye function, whose ``eye.result`` comes back on a later tick. The same code the
processes' ``server.py`` wrap runs here; sockets, threads and clocks live only there.

Determinism (P2): :meth:`Rig.digests` hashes every stream the brain saw (robot.state, tof.frame,
frame bytes) and every intent it sent; two runs with the same seed and script give the same
digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from duckmind.world.headless_duck import (
    CONTROL_HZ,
    WALL_EPOCH_US,
    Composed,
    HeadlessDuck,
    compose,
)

TOF_PERIOD_S = 1.0 / 13.8  # tofd's delivered cadence in the sim (15 Hz asked, one period of sleep)
METER_PERIOD_S = 0.5  # the robot's AE/AWB meters twice a second (render_scene.HomeEye.meter)
TRUTH_PERIOD_S = 0.5  # the world.* evaluator stream (spec §7.5: every 0.5 s)
EYE_LATENCY_S = 0.1  # sim seconds between a capture and its eye.result (a Mac look: ~70 ms)


@dataclass(frozen=True)
class LightSetting:
    """One moment of the home's light (the ``x_lights`` op's knobs): the duck's local time, the
    lamps, the moon."""

    when: datetime
    lamps_on: bool = True
    moon: float = 1.0
    label: str = ""


# The light ladder the phase-2 and phase-3 probes used (default_home: ~300 lux day, ~120 lux
# lamps, ~5 lux dusk, ~1 lux twilight with the lamps off, ~0.25 lux moonlight, ~0 black).
LIGHTS: dict[str, LightSetting] = {
    "bright": LightSetting(datetime(2026, 12, 20, 12, 0), label="day"),
    "lamps": LightSetting(datetime(2026, 12, 20, 20, 0), label="lamps"),
    "dim": LightSetting(datetime(2026, 12, 20, 16, 55), label="dusk"),
    "twilight": LightSetting(datetime(2026, 12, 20, 17, 15), lamps_on=False, label="twilight"),
    "dark": LightSetting(datetime(2026, 12, 21, 2, 0), label="moonlight"),
    "black": LightSetting(datetime(2026, 12, 21, 2, 0), moon=0.0, label="black"),
}
# Owner C3 (2026-09-27): the light gate's ground truth is the lux at the duck — ≥ 30 bright,
# ≥ 2.5 dim, ≥ 0.5 dark (very dark), else black. C1: black with only the dock light in view
# is "dark", not black.
LUX_BANDS = ((30.0, "bright"), (2.5, "dim"), (0.5, "dark"))


def light_band(lux: float) -> str:
    """Owner C3's ground-truth level for a lux reading."""
    for floor, level in LUX_BANDS:
        if lux >= floor:
            return level
    return "black"


def truth_level(lux: float, dock_light_in_view: bool) -> str:
    """C3's band, with C1: in total dark a visible dock light makes it "dark" (very dark)."""
    band = light_band(lux)
    return "dark" if band == "black" and dock_light_in_view else band


# ── what a brain gets and gives ───────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """One ``media.frame``: the header line's fields and the payload (UYVY), plus the render's
    evaluator-only extras (lux, AE registers, emitters in view) under ``truth``."""

    header: dict[str, Any]
    payload: bytes
    truth: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LookOrder:
    """A brain's request for one look: the ``eye.look`` line (duckmind.eye.protocol.EyeLook's
    JSON), the pose-tap samples the eye may pair the frame with, and who answers it."""

    look: Mapping[str, Any]
    poses: Sequence[Mapping[str, Any]] = ()


@dataclass
class Doors:
    """One tick's sensor doors: ``state`` (robot.state), ``tof`` (a tof.frame, when one was
    published this tick), ``results`` (eye.result lines that came back), ``health``
    (robot.health, once a second)."""

    t_ns: int
    state: dict[str, Any]
    tof: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    health: dict[str, Any] | None = None


@dataclass
class BrainOut:
    """What a brain asks for this tick: robot intents (``{"method", "params"}`` or
    ``duckmind.duckbody.protocol.Intent``), look orders, percepts it emitted (for the record)."""

    intents: list[Any] = field(default_factory=list)
    looks: list[LookOrder] = field(default_factory=list)
    percepts: list[Mapping[str, Any]] = field(default_factory=list)


class Brain(Protocol):
    """duck-body (and, through ``see``, duck-eye) as step functions."""

    def tick(self, doors: Doors) -> BrainOut: ...

    def see(
        self,
        look: Mapping[str, Any],
        header: Mapping[str, Any],
        payload: bytes,
        poses: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]: ...


# ── the rig ─────────────────────────────────────────────────────────────────


class Rig:
    """A home with the headless duck and its sensor doors on the simulation clock.

    ``home``: a name, a path, a layout dict (e.g. :mod:`duckmind.world.rig_scenes`) or a
    :class:`~duckmind.world.headless_duck.Composed`. ``light``: a :data:`LIGHTS` name or a
    :class:`LightSetting`. ``tof`` False: a duck without the sensor (tof frames never come).
    ``camera`` False: no renders (``capture`` raises). ``seed``: the camera noise and the ToF
    noise; the physics is deterministic by itself.
    """

    def __init__(
        self,
        home: str | Mapping[str, Any] | Composed = "default_home",
        *,
        variant: str | Sequence[str] | None = None,
        light: str | LightSetting = "bright",
        seed: int = 7,
        tof: bool = True,
        camera: bool = True,
        keyframe: str = "SIT",
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        mjcf_dir: str | None = None,
        quality: str = "720p30",
        household: bool = True,
        tof_period_s: float = TOF_PERIOD_S,
        meter_period_s: float = METER_PERIOD_S,
        truth_period_s: float = TRUTH_PERIOD_S,
        eye_latency_s: float = EYE_LATENCY_S,
    ) -> None:
        from duckmind.world import lights as world_lights

        self._home_arg = home
        self._variant = variant
        self._mjcf_dir = mjcf_dir
        self._household = household
        self.seed = int(seed)
        self.quality = quality
        composed = (
            home
            if isinstance(home, Composed)
            else compose(home, variant=variant, mjcf_dir=mjcf_dir, household=household)
        )
        self.duck = HeadlessDuck(composed, keyframe=keyframe, x=x, y=y, yaw=yaw, mjcf_dir=mjcf_dir)
        self.layout = self.duck.layout
        self.home_lights = world_lights.HomeLights.from_layout(self.layout)
        self.light = LIGHTS[light] if isinstance(light, str) else light
        self.has_tof = bool(tof)
        self.has_camera = bool(camera)
        self.tof_sensor: Any = None
        if tof:
            from duckmind.world.tof_model import TofSensor

            self.tof_sensor = TofSensor(self.duck.model, seed=self.seed)
        self._eye: Any = None
        self.tof_period_s = float(tof_period_s)
        self.meter_period_s = float(meter_period_s)
        self.truth_period_s = float(truth_period_s)
        self.eye_latency_s = float(eye_latency_s)
        now = self.duck.now()
        self._next_tof = now + self.tof_period_s
        self._next_meter = now
        self._next_truth = now
        self._next_health = now
        self._frames = 0
        self._tof_seq = 0
        self._pending_looks: list[LookOrder] = []
        self._pending_results: list[tuple[float, dict[str, Any]]] = []
        self.truth_log: list[dict[str, Any]] = []
        self.intent_log: list[dict[str, Any]] = []
        self._hash = {name: hashlib.sha256() for name in ("state", "tof", "frames", "intents")}
        self.counts = {"ticks": 0, "tof": 0, "frames": 0, "intents": 0, "looks": 0}

    # ── clocks ──────────────────────────────────────────────────────────────
    def now(self) -> float:
        """Simulation seconds."""
        return self.duck.now()

    def t_ns(self) -> int:
        return self.duck.t_ns()

    def wall_us(self) -> int:
        return self.duck.wall_us()

    # ── light ───────────────────────────────────────────────────────────────
    def set_light(self, light: str | LightSetting) -> None:
        self.light = LIGHTS[light] if isinstance(light, str) else light

    def light_state(self) -> Any:
        """The home's light now (charging: the nest's contact, as the world server wires it)."""
        from duckmind.world import lights as world_lights

        charging = False
        kit = self.duck.kit
        if kit is not None:
            closed = getattr(kit.nest, "contact_closed", None)
            try:
                charging = bool(closed()) if callable(closed) else bool(closed)
            except Exception:  # noqa: BLE001 - a part that cannot say is "not charging"
                charging = False
        s = self.light
        return world_lights.home_light_state(
            s.when, self.home_lights, lamps_on=s.lamps_on, moon=s.moon, charging=charging
        )

    # ── stepping ───────────────────────────────────────────────────────────
    def tick(self) -> Doors:
        """One robotd tick of physics; the doors that opened on it."""
        self.duck.tick()
        now = self.now()
        state = self.duck.state()
        self._hash["state"].update(_canon(state))
        doors = Doors(t_ns=int(state["t_ns"]), state=state)
        if self.tof_sensor is not None and now + 1e-9 >= self._next_tof:
            doors.tof = self._tof_frame()
            self._next_tof = _next(self._next_tof, self.tof_period_s, now)
        if now + 1e-9 >= self._next_health:
            doors.health = self.duck.health()
            self._next_health = _next(self._next_health, 1.0, now)
        if self.has_camera and self._eye is not None and now + 1e-9 >= self._next_meter:
            self._meter()
        if now + 1e-9 >= self._next_truth:
            self.truth_log.append(self.truth_sample())
            self._next_truth = _next(self._next_truth, self.truth_period_s, now)
        ready = [r for when, r in self._pending_results if when <= now + 1e-9]
        if ready:
            self._pending_results = [(w, r) for w, r in self._pending_results if w > now + 1e-9]
            doors.results = ready
        self.counts["ticks"] += 1
        return doors

    def run(
        self,
        brain: Brain,
        seconds: float,
        *,
        until: Callable[[Rig, Doors, BrainOut], bool] | None = None,
    ) -> bool:
        """Lockstep for ``seconds`` of sim time: tick, hand the doors to the brain, apply its
        intents, serve its looks. ``until`` may end the run early (returns True then)."""
        for _ in range(max(0, int(round(seconds * CONTROL_HZ)))):
            doors = self.tick()
            out = brain.tick(doors)
            for intent in out.intents:
                self.apply(intent)
            self._pending_looks += list(out.looks)
            self._serve_looks(brain)
            if until is not None and until(self, doors, out):
                return True
        return False

    def _serve_looks(self, brain: Brain) -> None:
        if not self._pending_looks:
            return
        now_ns = self.t_ns()
        waiting: list[LookOrder] = []
        for order in self._pending_looks:
            if int(order.look.get("not_before_ns", 0) or 0) > now_ns:
                waiting.append(order)
                continue
            if not self.has_camera:
                result = _no_frame(order.look, "frame_error")
            else:
                frame = self.capture()
                result = brain.see(order.look, frame.header, frame.payload, order.poses)
            self.counts["looks"] += 1
            self._pending_results.append((self.now() + self.eye_latency_s, dict(result)))
        self._pending_looks = waiting

    # ── robot intents ───────────────────────────────────────────────────────
    def apply(self, intent: Any) -> dict[str, Any]:
        """One robotd request (``{"method", "params"}`` or a protocol ``Intent``), as robotd
        answers it. Unknown methods are refused, never guessed."""
        method = getattr(intent, "method", None)
        params = getattr(intent, "params", None)
        if method is None and isinstance(intent, Mapping):
            method, params = intent.get("method"), intent.get("params")
        p = dict(params or {})
        record = {"t_ns": self.t_ns(), "method": method, "params": p}
        self._hash["intents"].update(_canon(record))
        self.intent_log.append(record)
        self.counts["intents"] += 1
        duck = self.duck
        if method == "robot.move":
            return duck.move(
                float(p.get("vx", 0.0)), float(p.get("vy", 0.0)), float(p.get("vyaw", 0.0))
            )
        if method == "robot.head":
            names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
            return duck.head([float(p.get(n, 0.0)) for n in names])
        if method == "robot.look":
            return duck.look(
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
                float(p.get("neck_pitch", 0.0)),
            )
        if method == "robot.mouth":
            return duck.mouth(float(p.get("angle", p.get("mouth")) or 0.0))
        if method == "robot.do":
            return duck.do(str(p.get("skill", "")))
        if method in ("robot.stop", "safety.stop"):
            return duck.stop()
        if method == "world.pg":
            return self.pg()
        return {"accepted": False, "reason": f"unknown method {method!r}"}

    def pg(self) -> dict[str, Any]:
        """The world's ``pg`` sensor (the one world op duck-body may call, sim only)."""
        kit = self.duck.kit
        if kit is None:
            return {"ok": False, "why": "no nest"}
        ops = kit.nest.ops() if hasattr(kit.nest, "ops") else {}
        fn = ops.get("pg")
        if fn is None:
            return {"ok": False, "why": "no pg op"}
        body = _BodyPort(self.duck.world)  # the op reads ``body.world`` and takes its lock
        return dict(fn(body, {"op": "pg"}))

    # ── the sensor doors ────────────────────────────────────────────────────
    def _tof_frame(self) -> dict[str, Any]:
        frame = self.tof_sensor.capture(self.duck.world)
        seq = self._tof_seq
        self._tof_seq += 1
        out = {
            "seq": seq,
            "at_us": self.wall_us(),
            "t_ns": self.t_ns(),
            "rows": int(frame["rows"]),
            "cols": int(frame["cols"]),
            "distance_mm": [int(v) for v in frame["distance_mm"]],
            "status": [int(v) for v in frame["status"]],
        }
        self._hash["tof"].update(_canon(out))
        self.counts["tof"] += 1
        return out

    def tof_truth(self) -> dict[str, Any] | None:
        """The last ToF frame's truth record (per-zone true distance and geom; evaluators)."""
        if self.tof_sensor is None or not self.tof_sensor.log:
            return None
        return dict(self.tof_sensor.log[-1])

    def eye(self) -> Any:
        """The world camera (opened on first use: the render copy is composed again, since an
        ``MjSpec`` with a buffer texture cannot be copied)."""
        if not self.has_camera:
            raise RuntimeError("this rig has no camera")
        if self._eye is None:
            from duckmind.world import camera_model as cm
            from duckmind.world import render_scene

            home = self._home_arg
            if isinstance(home, Composed):
                raise RuntimeError("a rig built from a Composed home cannot open its camera copy")
            copy = compose(
                home,
                variant=self._variant,
                mjcf_dir=self._mjcf_dir,
                household=self._household,
            )
            self._eye = render_scene.HomeEye.open(
                copy.spec,
                home=self.home_lights,
                config=cm.CameraConfig(quality=self.quality),
                seed=self.seed,
            )
            self._sync_eye()
            self._eye.converge(self.light_state())
            self._next_meter = self.now() + self.meter_period_s
        return self._eye

    def _sync_eye(self) -> None:
        d = self.duck.data
        self._eye.set_state(d.qpos, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)

    def _meter(self) -> None:
        self._sync_eye()
        self._eye.meter(self.light_state())
        self._next_meter = _next(self._next_meter, self.meter_period_s, self.now())

    def capture(self) -> Frame:
        """The next ``media.frame`` (the world as it is now), with its evaluator-only truth."""
        from duckmind.world import camera_model as cm

        eye = self.eye()
        self._sync_eye()
        state = self.light_state()
        cap = eye.capture(state, index=self._frames)
        payload = cap.uyvy()
        header = {
            "width": int(cap.width),
            "height": int(cap.height),
            "format": "UYVY",
            "bytes": len(payload),
            "captured_at_unix_us": self.wall_us(),
            "rotate": int(cap.rotate),
        }
        self._frames += 1
        self._hash["frames"].update(_canon(header))
        self._hash["frames"].update(payload)
        self.counts["frames"] += 1
        in_view = [str(n) for n in cm.emitters_in_view(eye.eye)]
        visible = [n for n in in_view if self._unoccluded(n)]
        lux = self.truth_lux()
        glowing = bool(getattr(cap, "glow", False))
        truth = {
            "t_ns": self.t_ns(),
            "lux": lux,
            "emitters_in_view": visible,
            "glow": glowing,
            "level": truth_level(lux, bool(visible) and glowing),
            "light": self.light.label,
            "extras": {k: v for k, v in cap.extras.items() if _jsonable(v)},
        }
        return Frame(header=header, payload=payload, truth=truth)

    def _unoccluded(self, geom_name: str) -> bool:
        """Does a ray from the camera reach this emitter first (nothing in between)?"""
        import numpy as np

        eye = self.eye().eye
        mj = self.duck._mj
        m, d = eye.model, eye.data
        g = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, geom_name))
        if g < 0:
            return False
        cam = np.array(d.cam_xpos[eye.camera], dtype=float)
        target = np.array(d.geom_xpos[g], dtype=float)
        vec = target - cam
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return True
        hit = np.zeros(1, dtype=np.int32)
        body = int(m.geom_bodyid[g])
        # the duck's own head never hides what it looks at (camera sits in it): skip its body
        exclude = int(m.cam_bodyid[eye.camera])
        got = mj.mj_ray(m, d, cam, vec / dist, None, 1, exclude, hit)
        if got < 0 or int(hit[0]) < 0:
            return True
        if int(hit[0]) == g or int(m.geom_bodyid[int(hit[0])]) == body:
            return True
        return float(got) >= dist - float(m.geom_rbound[g])

    # ── truth (evaluators only) ─────────────────────────────────────────────
    def truth(self) -> dict[str, Any]:
        return self.duck.truth()

    def truth_lux(self) -> float:
        """Lux on a level surface on the floor under the duck's trunk (owner C3's truth)."""
        tr = self.duck.data.qpos[self.duck.qadr : self.duck.qadr + 3]
        state = self.light_state()
        return float(state.lux_at((float(tr[0]), float(tr[1]), 0.0)))

    def truth_sample(self) -> dict[str, Any]:
        """One ``world.*`` evaluator record (spec §7.5): pose, beak, contacts, hold, light."""
        t = self.duck.truth()
        return {
            "kind": "world.sample",
            "t_ns": self.t_ns(),
            "trunk_mm": [int(round(v * 1000)) for v in t["trunk"]],
            "yaw_mrad": int(round(t["yaw"] * 1000)),
            "tilt_mdeg": int(round(t["tilt_deg"] * 1000)),
            "beak_mm": [int(round(v * 1000)) for v in t["beak"]] if t["beak"] else None,
            "contacts": [list(c) for c in t["contacts"]],
            "held": bool(t["held"]),
            "policy": t["policy"],
            "light": self.light.label,
        }

    def digests(self) -> dict[str, str]:
        """sha256 (first 16 hex digits) of every stream the brain saw and every intent it sent,
        plus the counts: equal digests = the same run, byte for byte (P2)."""
        out = {name: h.hexdigest()[:16] for name, h in self._hash.items()}
        out.update({f"n_{k}": str(v) for k, v in self.counts.items()})
        return out

    def close(self) -> None:
        if self._eye is not None:
            self._eye.close()
            self._eye = None


# ── small helpers ──────────────────────────────────────────────────────────


def _next(due: float, period: float, now: float) -> float:
    """The next due time after a firing: one period on, or one period from now when the duck was
    stepped outside the rig meanwhile (never a burst of catch-up frames)."""
    nxt = due + period
    return nxt if nxt > now + 1e-9 else now + period


@dataclass(frozen=True)
class _BodyPort:
    """What a world op gets as its ``body``: the world."""

    world: Any


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default).encode()


def _default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _no_frame(look: Mapping[str, Any], why: str) -> dict[str, Any]:
    return {
        "type": "eye.result",
        "look_id": str(look.get("look_id", "L0")),
        "attempt": int(look.get("attempt", 0) or 0),
        "kind": str(look.get("kind", "end_look")),
        "status": "none",
        "why": why,
        "results": {},
        "pv": "p:0000000000000000",
    }


class ScriptBrain:
    """A brain that only plays back robot intents at given sim times (``[(t_s, intent), …]``)
    and never looks: the rig's own self-test and the P2 physics digest. It knows nothing of the
    world but the clock."""

    def __init__(self, script: Iterable[tuple[float, Any]], *, t0_ns: int | None = None) -> None:
        self.script = sorted(script, key=lambda item: item[0])
        self.t0_ns = t0_ns
        self.seen = 0

    def tick(self, doors: Doors) -> BrainOut:
        if self.t0_ns is None:
            self.t0_ns = doors.t_ns
        now = (doors.t_ns - self.t0_ns) / 1e9
        out = BrainOut()
        while self.script and self.script[0][0] <= now + 1e-9:
            out.intents.append(self.script.pop(0)[1])
        self.seen += 1
        return out

    def see(
        self,
        look: Mapping[str, Any],
        header: Mapping[str, Any],
        payload: bytes,
        poses: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return _no_frame(look, "bad_request")


def move(vx: float, vyaw: float = 0.0) -> dict[str, Any]:
    """A ``robot.move`` intent as a plain dict."""
    return {"method": "robot.move", "params": {"vx": float(vx), "vy": 0.0, "vyaw": float(vyaw)}}


def stream_moves(
    t0: float, t1: float, vx: float, vyaw: float = 0.0, hz: float = 50.0
) -> list[tuple[float, dict[str, Any]]]:
    """``robot.move`` streamed at ``hz`` from ``t0`` to ``t1`` then one zero (duck-body streams
    at 50 Hz: robotd's deadman zeroes a twist after 500 ms)."""
    n = max(1, int(round((t1 - t0) * hz)))
    out = [(t0 + i / hz, move(vx, vyaw)) for i in range(n)]
    out.append((t1, move(0.0, 0.0)))
    return out


__all__ = [
    "LIGHTS",
    "LUX_BANDS",
    "Brain",
    "BrainOut",
    "Doors",
    "Frame",
    "LightSetting",
    "LookOrder",
    "Rig",
    "ScriptBrain",
    "WALL_EPOCH_US",
    "light_band",
    "move",
    "stream_moves",
    "truth_level",
]
