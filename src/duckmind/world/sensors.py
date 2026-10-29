"""The world's sensors, as one simhome extension (spec §11.1–§11.4; phase 2).

Loaded by the world server (``python -m duckmind.world.simhome … --ext duckmind.world.sensors``
or ``DUCKMIND_SIMHOME_EXT=duckmind.world.sensors``), it gives the simulated duck the sensors of
the real one, through the same doors:

* **the head camera** — :class:`~duckmind.world.eye_render.EyeRender` on its own render thread
  (its own compiled copy of the home as the world server built it, hand and people included; the
  lens, AE, white balance and sensor of :mod:`~duckmind.world.camera_model`; the lights of the
  camera's room from :mod:`~duckmind.world.lights`), metering twice a second always and
  capturing only when asked, behind :class:`~duckmind.world.media_shim.MediaShim` — mediad's
  ``media.frame`` socket at ``<dir>/mediad/media.sock``;
* **the ToF** — :class:`~duckmind.world.tof_model.TofSensor` answering the body port's ``tof``
  op in place of the official single-ray one (``tofd --sim`` polls it); for a duck without the
  module (``DUCKMIND_SIM_TOF=0``, owner decision F2) the op refuses and ``tofd`` runs its
  hardware path and reports the sensor unavailable;
* **tofd itself** when ``DUCKMIND_TOFD`` names the binary (or ``auto``: next to robotd): the
  official daemon, launched by :mod:`~duckmind.world.tofd_launch` against this server's own port
  (``--hz 15``), its socket at ``<dir>/tofd/tof.sock``, stopped with the world.

Evaluator truth goes out only through ``x_`` ops on the body port (never to Duck Core, never in
a sensor's own reply):

``{"op": "x_sensors"}``
    Where everything is and how it runs: ``dir``, ``media`` (socket, quality, clock, the shim's
    counters), ``eye`` (running, errors, frames, AE, timings, CPU share, frozen), ``tof``
    (fitted, frames, cost), ``tofd`` (pid, socket, its own ``TofStreamResult``), ``lights``.
``{"op": "x_eye_log", "since": seq, "limit": 20}``
    The served frames' truth: physics time, AE registers, white balance, the light (room, key,
    lux, glow), the camera's place, the timings, frozen; plus the latest metering ticks.
``{"op": "x_tof_log", "since": seq, "limit": 20}``
    Each ToF frame's truth: the sensor's pose, the nearest true range per zone, why each zone got
    its status (``why``: one letter per zone), the geom it saw, the cost.
``{"op": "x_lights"}`` / ``{"op": "x_lights", "local": "2026-09-27T19:30:00", "rate": 1.0,
"lamps_on": true, "moon": 1.0, "charging": null}``
    The light clock: the duck's local time for the lights (default: the wall clock in Taipei;
    simweb sets Duck Core's clock, which may run fast), the lamps switch, the moon; ``charging``
    overrides the nest's (``null``: follow the nest module's ``x_world`` section). Answers the
    clock and the whole home's light state (``lights.HomeLightState.to_page``); the same state
    fills ``x_world``'s ``lights`` section.

Settings (environment, read when the world server loads the extension):

======================== =================================================================
``DUCKMIND_SENSORS``     ``eye,tof`` (default), ``eye``, ``tof`` or ``none``
``DUCKMIND_SIM_TOF``     ``1`` (default): the ToF module is fitted; ``0``: it is not (F2)
``DUCKMIND_SENSORS_DIR`` the sockets' folder (default ``/private/tmp/dm-sensors-<port>``)
``DUCKMIND_MEDIA_QUALITY`` ``720p30`` (default), ``1080p30``, ``720p15``, ``360p30``
``DUCKMIND_EYE_CLOCK``   ``1`` (default): a frame waits for the camera's next tick; ``0``: none
``DUCKMIND_TOFD``        the tofd binary, ``auto``, or empty (default: tofd is not started)
``DUCKMIND_TOFD_HZ``     15 (the gate is "tofd --hz 15")
``DUCKMIND_SENSORS_SEED`` 7
======================== =================================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from duckmind.tmpdir import short_tmp
from duckmind.world import media_shim

TAIPEI = timezone(timedelta(hours=8))
DEFAULT_QUALITY = "720p30"
DEFAULT_SEED = 7
STATUS_CACHE_S = 2.0


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "no", "off", "false")


def taipei_now() -> datetime:
    return datetime.now(TAIPEI).replace(tzinfo=None)


@dataclass
class Settings:
    eye: bool = True
    tof: bool = True
    fitted: bool = True  # the ToF module (F2)
    folder: Path | None = None
    quality: str = DEFAULT_QUALITY
    clock: bool = True
    tofd: str = ""  # binary, "auto", or "" (do not launch)
    tofd_hz: int = 15
    seed: int = DEFAULT_SEED

    @classmethod
    def from_env(cls) -> Settings:
        which = os.environ.get("DUCKMIND_SENSORS", "eye,tof").strip().lower()
        parts = {p.strip() for p in which.split(",") if p.strip()}
        if parts & {"none", "off", "0"}:
            parts = set()
        folder = os.environ.get("DUCKMIND_SENSORS_DIR", "").strip()
        return cls(
            eye="eye" in parts,
            tof="tof" in parts,
            fitted=_flag("DUCKMIND_SIM_TOF", True),
            folder=Path(folder).expanduser() if folder else None,
            quality=os.environ.get("DUCKMIND_MEDIA_QUALITY", "").strip() or DEFAULT_QUALITY,
            clock=_flag("DUCKMIND_EYE_CLOCK", True),
            tofd=os.environ.get("DUCKMIND_TOFD", "").strip(),
            tofd_hz=int(os.environ.get("DUCKMIND_TOFD_HZ", "15") or 15),
            seed=int(os.environ.get("DUCKMIND_SENSORS_SEED", str(DEFAULT_SEED)) or DEFAULT_SEED),
        )


def _socket_too_long(folder: Path) -> bool:
    """Whether the sockets under ``folder`` would not fit a unix socket path (sun_path)."""
    longest = max(
        (folder / "mediad" / "media.sock", folder / "tofd" / "tof.sock"),
        key=lambda path: len(os.fsencode(str(path))),
    )
    return len(os.fsencode(str(longest))) >= media_shim.SUN_PATH_MAX


@dataclass
class LightClock:
    """The duck's local time for the lights, and the switches the page may set."""

    base_local: datetime | None = None
    base_wall: float = 0.0
    rate: float = 1.0
    lamps_on: bool = True
    moon: float = 1.0
    charging: bool | None = None  # None: the nest module says
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def now(self) -> datetime:
        with self.lock:
            if self.base_local is None:
                return taipei_now()
            return self.base_local + timedelta(seconds=(time.time() - self.base_wall) * self.rate)

    def set(self, request: dict[str, Any]) -> None:
        """``local`` (ISO, the duck's local time now), ``rate``, ``lamps_on``, ``moon``,
        ``charging`` (true/false, or null to follow the nest); ``ValueError`` when wrong."""
        local = request.get("local")
        rate = request.get("rate")
        with self.lock:
            if local is not None:
                if local == "wall":
                    self.base_local = None
                else:
                    try:
                        when = datetime.fromisoformat(str(local))
                    except ValueError:
                        raise ValueError(f"x_lights local {local!r} is not an ISO time") from None
                    self.base_local = when.replace(tzinfo=None)
                    self.base_wall = time.time()
            if rate is not None:
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
                    raise ValueError("x_lights rate must be a number ≥ 0")
                if self.base_local is not None and local is None:
                    # keep "now" continuous when only the rate changes
                    now = self.base_local + timedelta(
                        seconds=(time.time() - self.base_wall) * self.rate
                    )
                    self.base_local, self.base_wall = now, time.time()
                self.rate = float(rate)
            if "lamps_on" in request:
                if not isinstance(request["lamps_on"], bool):
                    raise ValueError("x_lights lamps_on must be true or false")
                self.lamps_on = request["lamps_on"]
            if "moon" in request:
                moon = request["moon"]
                if isinstance(moon, bool) or not isinstance(moon, (int, float)) or moon < 0:
                    raise ValueError("x_lights moon must be a number ≥ 0")
                self.moon = float(moon)
            if "charging" in request:
                value = request["charging"]
                if value is not None and not isinstance(value, bool):
                    raise ValueError("x_lights charging must be true, false or null")
                self.charging = value

    def view(self) -> dict[str, Any]:
        now = self.now()
        with self.lock:
            return {
                "local": now.isoformat(timespec="seconds"),
                "source": "wall" if self.base_local is None else "set",
                "rate": self.rate,
                "lamps_on": self.lamps_on,
                "moon": self.moon,
                "charging": self.charging,
            }


def body_address(argv: list[str] | None = None) -> tuple[str, int]:
    """The official server's ``--host``/``--port`` from this process's arguments (simhome has
    handed them to ``body_server.main`` by the time the world is ready)."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--host", default="127.0.0.1")
    peek.add_argument("--port", type=int, default=7801)
    seen, _ = peek.parse_known_args(sys.argv[1:] if argv is None else argv)
    return str(seen.host), int(seen.port)


class SensorSuite:
    """Everything this extension runs for one world (see the module docstring)."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: Any = None,
        address: tuple[str, int] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.registry = registry
        self.spec: Any = None
        self.world: Any = None
        self.body: Any = None
        self.folder: Path | None = None
        self.owns_folder = False
        self.clock = LightClock()
        self.frames = media_shim.Frames()
        self.shim: media_shim.MediaShim | None = None
        self.eye: Any = None  # EyeRender
        self.tof: Any = None  # TofSensor
        self.tofd: Any = None  # TofdLaunch
        self.home_lights: Any = None
        self.quality: Any = None
        self.errors: list[str] = []
        self.address = address
        self._light_lock = threading.Lock()
        self._state: Any = None  # the last HomeLightState and when (monotonic)
        self._state_at = -1e9
        self._page: dict[str, Any] | None = None
        self._tofd_status: tuple[float, dict[str, Any]] | None = None
        self._nest: Any = None

    # ── registry hooks ──────────────────────────────────────────────────

    def keep_spec(self, spec: Any, home: Any) -> None:
        """The home's spec as the world server compiles it (later edits by other extensions
        land in the same object; the eye copies it once the world is ready)."""
        self.spec = spec

    def start(self, world: Any, body: Any) -> None:
        """Once, after the spawn, before the server accepts connections."""
        from duckmind.world import camera_model as cm

        self.world, self.body = world, body
        s = self.settings
        host, port = self.address or body_address()
        self.folder = s.folder or Path(short_tmp()) / f"dm-sensors-{port}"
        self.owns_folder = s.folder is None
        if _socket_too_long(self.folder):
            # (a unix socket path must fit in sun_path: a deep state folder falls back to a
            # short one of our own instead of the camera never starting)
            short = Path(short_tmp()) / f"dm-sensors-{port}"
            self.errors.append(f"感測器資料夾的路徑太長，放不下 socket：改用 {short}")
            self.folder, self.owns_folder = short, True
        self.folder.mkdir(parents=True, exist_ok=True)
        self.quality = cm.quality_rung(s.quality)
        self.home_lights = self._home_lights()
        sections = getattr(self.registry, "sections", {}) if self.registry is not None else {}
        self._nest = sections.get("nest")
        if s.tof and s.fitted:
            self._start_tof(world)
        if s.eye:
            self._start_eye(world)
        if s.tofd:
            self._start_tofd(host, port)

    def stop(self) -> None:
        for step in (self._stop_tofd, self._stop_eye, self._stop_shim):
            try:
                step()
            except Exception as error:
                sys.stderr.write(f"== sensors: stop failed: {error}\n")
        if self.owns_folder and self.folder is not None:
            shutil.rmtree(self.folder, ignore_errors=True)

    # ── the parts ───────────────────────────────────────────────────────

    def _home_lights(self) -> Any:
        from duckmind.world import lights

        home = getattr(self.registry, "home", None) if self.registry is not None else None
        layout = getattr(home, "layout", None)
        if layout is None:
            return lights.DEFAULT_HOME
        try:
            return lights.HomeLights.from_layout(layout)
        except Exception as error:
            self.errors.append(f"燈光從佈置讀不出來，先用預設房間：{type(error).__name__}: {error}")
            return lights.DEFAULT_HOME

    def _start_tof(self, world: Any) -> None:
        from duckmind.world import tof_model

        self.tof = tof_model.TofSensor(
            world.model, seed=self.settings.seed, ambient=self.ambient_lux
        )

    def _start_eye(self, world: Any) -> None:
        from duckmind.world import camera_model as cm
        from duckmind.world import eye_render, render_scene

        if self.spec is None:
            self.errors.append("鴨眼鏡頭：沒有拿到家的模型（edit_spec 沒有被呼叫）")
            return
        spec = self.spec.copy()  # the eye's own copy: the world's spec is never touched
        config = cm.CameraConfig(quality=self.settings.quality)
        home, seed = self.home_lights, self.settings.seed

        def open_eye() -> Any:
            return render_scene.HomeEye.open(spec, home=home, config=config, seed=seed)

        assert self.folder is not None
        self.shim = media_shim.MediaShim(self.folder / "mediad" / "media.sock", self.frames)
        self.shim.start()
        self.eye = eye_render.EyeRender(
            open_eye=open_eye,
            snapshot=self.snapshot,
            light=self.light_for,
            frames=self.frames,
            frame_period_s=self.quality.frame_period_s,
            clock=self.settings.clock,
        )
        self.eye.start()

    def _start_tofd(self, host: str, port: int) -> None:
        from duckmind.world import tofd_launch

        binary = tofd_launch.find_tofd(self.settings.tofd)
        if binary is None:
            self.errors.append(f"找不到 tofd（DUCKMIND_TOFD={self.settings.tofd!r}）：ToF 沒有啟動")
            return
        assert self.folder is not None
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"  # a server listening everywhere is reached on the loopback
        body = (host, port) if self.settings.fitted and self.settings.tof else None
        self.tofd = tofd_launch.TofdLaunch(
            binary, runtime_dir=self.folder, body=body, hz=self.settings.tofd_hz
        )
        self.tofd.start()

    def _stop_tofd(self) -> None:
        if self.tofd is not None:
            self.tofd.stop()

    def _stop_eye(self) -> None:
        if self.eye is not None:
            self.eye.stop()

    def _stop_shim(self) -> None:
        if self.shim is not None:
            self.shim.close()

    # ── the world, the light ──────────────────────────────────────────────

    def snapshot(self) -> Any:
        from duckmind.world.eye_render import Snapshot

        world = self.world
        with world.lock:
            data = world.data
            qpos = data.qpos.copy()
            mocap_pos = data.mocap_pos.copy()
            mocap_quat = data.mocap_quat.copy()
            sim_time = float(data.time)
            charging = self._charging_locked(world)
            wall_us = time.time_ns() // 1000
        return Snapshot(qpos, mocap_pos, mocap_quat, sim_time, wall_us, charging)

    def _charging_locked(self, world: Any) -> bool:
        """Whether the duck is charging: the page's override, else the nest module's
        ``x_world`` section (``charging``), else not. The caller holds ``world.lock``."""
        if self.clock.charging is not None:
            return self.clock.charging
        entry = self._nest
        if entry is None or getattr(entry, "disabled", None):
            return False
        try:
            section = entry.fn(world)
        except Exception:
            return False
        return bool(section.get("charging")) if isinstance(section, dict) else False

    def light_state(self, charging: bool = False) -> Any:
        from duckmind.world import lights

        clock = self.clock
        with clock.lock:
            lamps_on, moon = clock.lamps_on, clock.moon
        state = lights.home_light_state(
            clock.now(), self.home_lights, lamps_on=lamps_on, moon=moon, charging=charging
        )
        with self._light_lock:
            self._state, self._state_at, self._page = state, time.monotonic(), None
        return state

    def recent_state(self, max_age_s: float = STATUS_CACHE_S) -> Any:
        """The last light state (the camera computes one twice a second), or a new one."""
        with self._light_lock:
            state, at = self._state, self._state_at
        if state is None or time.monotonic() - at > max_age_s:
            state = self.light_state(bool(self.clock.charging))
        return state

    def light_for(self, snap: Any) -> Any:
        return self.light_state(bool(getattr(snap, "charging", False)))

    def ambient_lux(self, points: Any) -> Any:
        """Daylight at each ToF target (lux): the windows of the target's room, each its sky
        light falling off within about a metre of the window (sunlight shortens the range)."""
        import numpy as np

        home = self.home_lights
        out = np.zeros(len(points))
        if home is None:
            return out
        state = self.recent_state()
        for i, p in enumerate(points):
            room = home.room_at(float(p[0]), float(p[1]))
            room_state = state.rooms.get(room.id)
            if room_state is None:
                continue
            by_id = {s.id: s for s in room_state.sources if s.id is not None}
            for window in room.windows:
                source = by_id.get(window.id)
                if source is None or source.lux <= 0:
                    continue
                gap = float(np.linalg.norm(np.asarray(p) - np.asarray(window.position)))
                out[i] += source.lux / (1.0 + (gap / 0.7) ** 2)
        return out

    # ── ops ─────────────────────────────────────────────────────────────

    def tof_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.fitted:
            raise ValueError("這隻小鴨沒有裝 ToF（DUCKMIND_SIM_TOF=0）")
        if self.tof is None:
            raise ValueError("ToF 沒有啟動（DUCKMIND_SENSORS 沒有 tof）")
        return dict(self.tof.capture(body.world))

    def eye_log_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        since, limit = _since_limit(request)
        if self.eye is None:
            return {"frames": [], "meter": [], "status": None}
        return {
            "frames": self.eye.entries(since, limit),
            "meter": self.eye.meter_entries(4),
            "status": self.eye.status(),
        }

    def tof_log_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        since, limit = _since_limit(request)
        if self.tof is None:
            return {"frames": [], "fitted": self.settings.fitted}
        return {
            "frames": self.tof.entries(since, limit),
            "fitted": self.settings.fitted,
            "cost": self.tof.cost(),
            "sensor": self.tof.describe(),
        }

    def lights_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        changes = {k: v for k, v in request.items() if k != "op"}
        if changes:
            self.clock.set(changes)
        charging = self.clock.charging
        if charging is None and self.world is not None:
            with self.world.lock:
                charging = self._charging_locked(self.world)
        state = self.light_state(bool(charging))
        return {"clock": self.clock.view(), "state": state.to_page()}

    def sensors_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        s = self.settings
        out: dict[str, Any] = {
            "dir": str(self.folder) if self.folder else None,
            "errors": list(self.errors),
            "media": {
                "socket": str(self.shim.path) if self.shim else None,
                "quality": self.quality.label if self.quality else s.quality,
                "clock": s.clock,
                "stats": self.shim.stats() if self.shim else None,
            },
            "eye": self.eye.status() if self.eye is not None else None,
            "tof": {
                "on": self.tof is not None,
                "fitted": s.fitted,
                "frames": self.tof.seq if self.tof is not None else 0,
                "cost": self.tof.cost() if self.tof is not None else None,
            },
            "tofd": self._tofd_view(),
            "lights": {"clock": self.clock.view(), "home": getattr(self.home_lights, "id", None)},
        }
        return out

    def _tofd_view(self) -> dict[str, Any] | None:
        if self.tofd is None:
            return None
        cached = self._tofd_status
        if cached is not None and time.monotonic() - cached[0] < STATUS_CACHE_S:
            return cached[1]
        view = self.tofd.status()
        self._tofd_status = (time.monotonic(), view)
        return view

    def lights_section(self, world: Any) -> Any:
        """``x_world``'s ``lights`` (the world's lock is held): the last light state (the one
        the camera used), as the page draws it."""
        state = self.recent_state()
        with self._light_lock:
            if self._page is None or self._state is not state:
                self._page = state.to_page()
            return self._page


def _since_limit(request: dict[str, Any]) -> tuple[int | None, int]:
    since = request.get("since")
    limit = request.get("limit", 20)
    if since is not None and (isinstance(since, bool) or not isinstance(since, int)):
        raise ValueError("since must be a whole number (a seq) or null")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be a whole number from 1 to 500")
    return since, limit


def register(registry: Any) -> SensorSuite:
    """simhome's entry point: hook the suite into the world server."""
    suite = SensorSuite(Settings.from_env(), registry)
    registry.edit_spec(suite.keep_spec, name="sensors.keep_spec")
    registry.on_ready(suite.start, name="sensors.start")
    registry.on_stop(suite.stop, name="sensors.stop")
    if suite.settings.tof or not suite.settings.fitted:
        registry.op("tof", suite.tof_op, override=True)
    registry.op("x_sensors", suite.sensors_op)
    registry.op("x_eye_log", suite.eye_log_op)
    registry.op("x_tof_log", suite.tof_log_op)
    registry.op("x_lights", suite.lights_op)
    try:
        registry.snapshot("lights", suite.lights_section)
    except ValueError as error:  # another module fills it: ours stays in x_lights
        suite.errors.append(f"x_world 的 lights 已經有別的模組在填：{error}")
    return suite
