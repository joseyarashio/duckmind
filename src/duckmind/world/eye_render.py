"""The head camera's capture branch in the world server: a render thread (spec §11.2).

On the real duck ``mediad`` runs the camera from boot: its auto-exposure meters a frame twice a
second whether or not anyone looks, and a ``media.frame`` reader gets the *next* buffer the
camera delivers. :class:`EyeRender` is that branch for the simulated camera, on one thread of the
world-server process that owns its own GL context (CGL on macOS: created, used and freed on that
thread; ``open_eye`` is called there):

* **metering**, every ``meter_s`` (0.5 s): a world snapshot, the lights at the duck's local time,
  one 160×90 metering render stepping mediad's AE and the white balance
  (:meth:`duckmind.world.render_scene.HomeEye.meter`), always, like the robot;
* **captures on demand**: when a reader waits (:attr:`media_shim.Frames.demand`), the next tick
  of the camera's own clock (the ``[media] quality`` rung's frame period, 33.3 ms at 720p30;
  ``clock=False`` answers at once) takes the request, snapshots the world, renders the full frame
  through the lens and the sensor model, packs UYVY and delivers it to every waiting reader.
  ``captured_at_unix_us`` is the host's wall clock at the snapshot;
* **a log for evaluators only** (``x_eye_log``): per frame the physics time, the AE registers,
  the white balance, the light it had, where the camera was, the timings, and whether the world
  was frozen (its physics time stood still for :data:`FROZEN_S`: the camera keeps capturing a
  stopped world, as a real one would, and the log says so).

The snapshot is the caller's (:class:`Snapshot`: qpos, mocap bodies, physics time, charging),
taken under ``world.lock`` in microseconds; the render never holds the world's lock.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import msg
from duckmind.world import media_shim

METER_S = 0.5  # mediad's INTERVAL: twice a second
FROZEN_S = 1.0  # physics time unchanged this long (wall): the world stopped
LOG_SIZE = 512


@dataclass(frozen=True)
class Snapshot:
    """The world at one moment, as the camera needs it."""

    qpos: Any
    mocap_pos: Any
    mocap_quat: Any
    sim_time: float
    wall_us: int  # host wall clock, microseconds since the epoch
    charging: bool = False


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(math.ceil(share * len(ordered))) - 1))]


class EyeRender:
    """The capture branch: see the module docstring. ``open_eye`` builds the
    :class:`~duckmind.world.render_scene.HomeEye` (on this thread); ``snapshot`` reads the world;
    ``light`` gives the light state for a snapshot; ``frames`` is the media-shim's rendezvous."""

    def __init__(
        self,
        *,
        open_eye: Callable[[], Any],
        snapshot: Callable[[], Snapshot],
        light: Callable[[Snapshot], Any],
        frames: media_shim.Frames,
        frame_period_s: float = 1.0 / 30.0,
        meter_s: float = METER_S,
        clock: bool = True,
        settle: bool = True,
        log_size: int = LOG_SIZE,
    ) -> None:
        self.open_eye = open_eye
        self.snapshot = snapshot
        self.light = light
        self.frames = frames
        self.frame_period_s = float(frame_period_s)
        self.meter_s = float(meter_s)
        self.clock = bool(clock)
        self.settle = bool(settle)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.log: deque[dict[str, Any]] = deque(maxlen=log_size)
        self.meters: deque[dict[str, Any]] = deque(maxlen=64)
        self.error: str | None = None
        self.error_detail: str | None = None
        self.running = False
        self.seq = 0
        self.meter_ticks = 0
        self.frozen = False
        self.info: dict[str, Any] = {}
        self._phase = time.monotonic()  # the camera clock's zero
        self._sim_time: float | None = None
        self._sim_moved = time.monotonic()
        self._cpu: deque[tuple[float, float]] = deque(maxlen=128)
        self.last_light: Any = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> EyeRender:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="eye-render", daemon=True)
            self._thread.start()
        return self

    def wait_ready(self, timeout_s: float = 30.0) -> bool:
        """Until the eye is built (or failed): whether it runs."""
        self._ready.wait(timeout_s)
        return self.running

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self.frames.demand.set()  # wake the thread
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    # ── the thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            eye = self.open_eye()
        except Exception as error:  # no GL here, MuJoCo missing, a spec that does not compile
            with self._lock:
                self.error = msg("world.eye_render.EyeRender._run.fabc633b")
                self.error_detail = f"{type(error).__name__}: {error}"
            self._ready.set()
            return
        try:
            width, height = eye.frame_size
            self.info = {
                "frame": [int(width), int(height)],
                "frame_period_s": round(self.frame_period_s, 6),
                "clock": self.clock,
                "meter_s": self.meter_s,
            }
            if self.settle:  # a camera that has been on since boot: its AE has converged
                snap = self._snap()
                self._pose(eye, snap)
                eye.converge(self.light(snap))
            self.running = True
            self._ready.set()
            self._loop(eye)
        except Exception as error:
            with self._lock:
                self.error = msg("world.eye_render.EyeRender._run.26f3f12e")
                self.error_detail = f"{type(error).__name__}: {error}"
        finally:
            self.running = False
            self._ready.set()
            try:
                eye.close()
            except Exception:
                pass

    def _loop(self, eye: Any) -> None:
        frames = self.frames
        next_meter = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_meter + self.meter_s:  # readers all along: still meter twice a second
                self._guard(self._meter, eye)
                next_meter = max(next_meter + self.meter_s, time.monotonic())
                continue
            if frames.demand.is_set():
                tick = self.next_tick(now)
                if tick > now and self._stop.wait(tick - now):
                    return
                if frames.take_request():
                    self._guard(self._capture, eye, tick)
                continue
            if now >= next_meter:
                self._guard(self._meter, eye)
                next_meter = max(next_meter + self.meter_s, time.monotonic())
                continue
            frames.demand.wait(timeout=max(0.0, next_meter - now))

    def _guard(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
            if self.error is not None:
                with self._lock:
                    self.error = self.error_detail = None
        except Exception as error:  # one bad frame must not stop the camera
            with self._lock:
                self.error = msg("world.eye_render.EyeRender._guard.39ab7cc8")
                self.error_detail = f"{type(error).__name__}: {error}"
        self._cpu.append((time.monotonic(), time.thread_time()))

    def next_tick(self, now: float) -> float:
        """When the camera's next buffer lands (the clock's phase is when this was made)."""
        if not self.clock or self.frame_period_s <= 0:
            return now
        k = math.ceil((now - self._phase) / self.frame_period_s - 1e-9)
        return self._phase + k * self.frame_period_s

    # ── one snapshot, one metering, one frame ─────────────────────────

    def _snap(self) -> Snapshot:
        snap = self.snapshot()
        now = time.monotonic()
        if snap.sim_time != self._sim_time:
            self._sim_time, self._sim_moved = snap.sim_time, now
            self.frozen = False
        else:
            self.frozen = now - self._sim_moved >= FROZEN_S
        return snap

    @staticmethod
    def _pose(eye: Any, snap: Snapshot) -> None:
        eye.set_state(snap.qpos, mocap_pos=snap.mocap_pos, mocap_quat=snap.mocap_quat)

    def _meter(self, eye: Any) -> None:
        snap = self._snap()
        self._pose(eye, snap)
        state = self.light(snap)
        luma = eye.meter(state)
        self.meter_ticks += 1
        light = eye.last_light
        self.last_light = state
        ae = eye.ae
        with self._lock:
            self.meters.append(
                {
                    "tick": self.meter_ticks,
                    "sim_time": round(snap.sim_time, 4),
                    "wall_us": snap.wall_us,
                    "luma": round(float(luma), 2),
                    "ae": _ae(ae),
                    "wb": [round(float(v), 4) for v in eye.awb.gains],
                    "light": light.to_dict() if light is not None else None,
                    "frozen": self.frozen,
                }
            )

    def _capture(self, eye: Any, tick: float) -> None:
        taken = time.monotonic()
        snap = self._snap()
        self._pose(eye, snap)
        state = self.light(snap)
        seq = self.seq
        capture = eye.capture(state, index=seq)
        t_pack = time.perf_counter()
        data = capture.uyvy()
        pack_ms = (time.perf_counter() - t_pack) * 1000.0
        frame = media_shim.Frame(
            width=int(capture.width),
            height=int(capture.height),
            data=data,
            captured_at_unix_us=int(snap.wall_us),
        )
        self.seq += 1
        self.last_light = state
        entry = {
            "seq": seq,
            "captured_at_unix_us": int(snap.wall_us),
            "sim_time": round(snap.sim_time, 4),
            "frozen": self.frozen,
            "tick_late_ms": round((taken - tick) * 1000.0, 3),
            "render_ms": capture.timings.get("render_ms"),
            "frame_ms": capture.timings.get("frame_ms"),
            "pack_ms": round(pack_ms, 3),
            "deliver_ms": round((time.monotonic() - taken) * 1000.0, 3),
            "size": [int(capture.width), int(capture.height)],
            "rotate": int(capture.rotate),
            "ae": capture.extras.get("ae"),
            "wb": capture.extras.get("wb"),
            "glow": bool(capture.glow),
            "light": {
                k: capture.extras.get(k)
                for k in ("room", "key", "label", "lux", "lit", "folded", "glow")
            },
            "charging": snap.charging,
            "camera": _camera(eye),
        }
        with self._lock:  # logged (and any earlier error cleared) before a reader can look
            self.log.append(entry)
            self.error = self.error_detail = None
        self.frames.deliver(frame)

    # ── for evaluators ────────────────────────────────────────────────

    def entries(self, since: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self.log if since is None or e["seq"] > since]
        return items[-max(1, int(limit)) :]

    def meter_entries(self, limit: int = 4) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.meters)[-max(1, int(limit)) :]

    def cpu_share(self) -> float | None:
        if len(self._cpu) < 2:
            return None
        (wall0, cpu0), (wall1, cpu1) = self._cpu[0], self._cpu[-1]
        return round((cpu1 - cpu0) / max(1e-6, wall1 - wall0), 4)

    def status(self) -> dict[str, Any]:
        with self._lock:
            frames = [e["frame_ms"] for e in self.log if e.get("frame_ms") is not None]
            deliver = [e["deliver_ms"] for e in self.log]
            last_meter = self.meters[-1] if self.meters else None
            out = {
                "running": self.running,
                "error": self.error,
                "error_detail": self.error_detail,
                "frames": self.seq,
                "meter_ticks": self.meter_ticks,
                "frozen": self.frozen,
                **self.info,
                "frame_ms_p50": percentile(frames, 0.5),
                "frame_ms_p95": percentile(frames, 0.95),
                "deliver_ms_p50": percentile(deliver, 0.5),
                "deliver_ms_p95": percentile(deliver, 0.95),
                "last_meter": last_meter,
            }
        out["cpu_share"] = self.cpu_share()
        return out


def _ae(ae: Any) -> dict[str, Any]:
    from duckmind.world import camera_model as cm

    return {**cm.exposure_summary(ae.sensor), "settled": bool(ae.settled())}


def _camera(eye: Any) -> dict[str, Any]:
    """Where the head camera was and where it looked (world mm, unit forward)."""
    inner = eye.eye
    d = inner.data
    pos = d.cam_xpos[inner.camera]
    m = d.cam_xmat[inner.camera]  # row-major 3×3: the camera looks down its −z column
    forward = [-float(m[2]), -float(m[5]), -float(m[8])]
    return {
        "pos_mm": [int(round(float(v) * 1000.0)) for v in pos],
        "forward": [round(float(v), 4) for v in forward],
        "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(forward[2]))))), 2),
    }
