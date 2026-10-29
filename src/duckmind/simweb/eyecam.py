"""The duck's-eye picture for the simulator page: what the head camera would see right now.

:class:`EyeCam` mirrors the live body into its own copy of the scene the body uses (the official
``robot_allcollisions`` duck on the official floor, plus our ceiling and the lights of
:mod:`duckmind.world.lights`) and renders the head camera through
:mod:`duckmind.world.camera_model`. It only ever reads the latest pose from a callable — trunk
position and quaternion and the joint angles, as :meth:`duckmind.livebody.LiveBody.frame`
returns them — and never talks to the body server or robotd, so it cannot slow or change the
physics.

On one render thread, with its own GL context made there (CGL on macOS, so no main thread is
needed):

* every 0.5 s a 160×90 metering render steps mediad's auto-exposure (as on the robot, twice a
  second);
* every 2 s the full 1280×720 frame goes through the sensor model, is turned upright, measured
  by the light meter (:mod:`duckmind.eye.light`) and encoded as a JPEG (half size by default).

:meth:`EyeCam.latest` holds ``{jpeg, width, height, t, light, label, render_ms, error, …}``.
When rendering is impossible (no GL here, no scene, MuJoCo missing) the reason goes into
``error`` and nothing crashes. The picture is computed, not a camera recording (spec §4.8);
nothing is written to disk.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from duckmind.i18n import msg
from duckmind.world import lights as world_lights
from duckmind.world.lights import LightState

PERIOD_S = 2.0
METER_S = 0.5
JPEG_SCALE = 2  # the page's picture is half size: 360×640 upright
JPEG_QUALITY = 75
TAIPEI = timezone(timedelta(hours=8))
# The body frame's physics time has not moved for this long: the world stopped (robotd or the
# body server stopped or hangs). Nothing is rendered then — a new picture of a stopped world
# would look like "now" — and the last picture keeps its age.
FROZEN_S = 1.0
FROZEN_NOTE = msg("web.eyecam.FROZEN_NOTE")

Pose = dict[str, Any]


def taipei_now() -> datetime:
    """The wall clock at the duck's home, for when no Duck Core clock is given."""
    return datetime.now(TAIPEI).replace(tzinfo=None)


def resolve_scene(
    scene: str | Path | None = None, mjcf_dir: str | Path | None = None
) -> Path | None:
    """The scene file: given, or ``scene_allcollisions.xml`` in the official MJCF folder."""
    from duckmind.world import camera_model

    if scene:
        return Path(scene).expanduser()
    folder = mjcf_dir or os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    return camera_model.default_scene(folder) if folder else None


def _half(image: Any, factor: int) -> Any:
    import numpy as np

    if factor <= 1:
        return image
    h = image.shape[0] // factor * factor
    w = image.shape[1] // factor * factor
    small = image[:h, :w].reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))
    return np.round(small).astype(np.uint8)


class EyeRenderer:
    """Everything one duck's-eye frame needs. Create, use and close it on one thread."""

    def __init__(
        self,
        scene: str | Path,
        *,
        frame_size: tuple[int, int] | None = None,
        seed: int = 7,
        jpeg_scale: int = JPEG_SCALE,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        from duckmind.world import camera_model as cm

        self.cm = cm
        self.frame_size = frame_size or cm.FRAME_SIZE
        self.scene_path = Path(scene)
        self.eye = cm.load_scene(self.scene_path, frame_size=self.frame_size)
        self.renderer = cm.HeadCameraRenderer(self.eye)
        self.ae = cm.AeLoop()
        self.seed = int(seed)
        self.jpeg_scale = max(1, int(jpeg_scale))
        self.jpeg_quality = int(jpeg_quality)
        self.meter_ticks = 0
        self.frames = 0

    def _rng(self, stream: int, index: int) -> Any:
        import numpy as np

        return np.random.default_rng([self.seed, stream, index])

    def _prepare(self, pose: Pose, state: LightState) -> float:
        self.cm.set_pose(self.eye, pose)
        return float(self.cm.set_lights(self.eye, state))

    def meter(self, pose: Pose, state: LightState) -> float:
        """One auto-exposure tick on a metering render; returns the metered mean luma."""
        cm = self.cm
        lux_per_unit = self._prepare(pose, state)
        small = self.renderer.render(*cm.METER_SIZE)
        out = cm.expose(
            cm.scene_lux(small, lux_per_unit), self.ae.sensor, self._rng(0, self.meter_ticks)
        )
        self.meter_ticks += 1
        mean = float(cm.luma(out).mean())
        self.ae.tick(mean)
        return mean

    def converge(self, pose: Pose, state: LightState, max_ticks: int = 40) -> int:
        """Tick the auto-exposure until it settles (for a first frame or a test); ticks used."""
        for tick in range(1, max_ticks + 1):
            self.meter(pose, state)
            if self.ae.settled() and tick > 1:
                return tick
        return max_ticks

    def frame(self, pose: Pose, state: LightState) -> dict[str, Any]:
        """The full frame: sensor, upright, light meter, JPEG."""
        import math

        from duckmind.eye import light as eye_light
        from duckmind.simweb import jpeg

        cm = self.cm
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        lux_per_unit = self._prepare(pose, state)
        raw = self.renderer.render(*self.frame_size)
        render_ms = (time.perf_counter() - wall0) * 1000
        sensor = self.ae.sensor
        rgb = cm.expose(cm.scene_lux(raw, lux_per_unit), sensor, self._rng(1, self.frames))
        self.frames += 1
        picture = cm.upright(rgb)
        exposure = eye_light.Exposure.from_registers(
            sensor.exposure, sensor.analogue_gain, sensor.gain
        )
        reading = eye_light.measure(cm.luma(picture), exposure)
        shown = _half(picture, self.jpeg_scale)
        data, encoder = jpeg.encode(shown, self.jpeg_quality)
        d = self.eye.data
        cam_pos = d.cam_xpos[self.eye.camera]
        axis = -d.cam_xmat[self.eye.camera].reshape(3, 3)[:, 2]
        return {
            "jpeg": data,
            "width": int(shown.shape[1]),
            "height": int(shown.shape[0]),
            "frame_size": [int(picture.shape[1]), int(picture.shape[0])],
            "t": pose.get("t"),
            "wall": time.time(),
            "clock": state.when,
            "light": reading.to_dict(),
            "label": state.label,
            "light_key": state.key,
            "lux": round(state.lux, 4),
            "ae": {**cm.exposure_summary(sensor), "settled": self.ae.settled()},
            "camera": {
                "height_m": round(float(cam_pos[2]), 4),
                "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2]))))), 2),
            },
            "render_ms": round(render_ms, 2),
            "frame_ms": round((time.perf_counter() - wall0) * 1000, 2),
            "cpu_ms": round((time.thread_time() - cpu0) * 1000, 2),
            "encoder": encoder,
            "frames": self.frames,
            "error": None,
        }

    def close(self) -> None:
        self.renderer.close()


class EyeCam:
    """The head camera, mirrored from the live body every 2 s on its own thread.

    ``pose`` returns the latest body frame (``None`` while there is none); ``clock`` the duck's
    local date and time for the light (default: the wall clock in Taipei; pass Duck Core's clock
    when it runs fast). ``scene`` is the scene file the live body uses; by default
    ``scene_allcollisions.xml`` in ``mjcf_dir`` or ``$DUCKMIND_MICRODUCK_MJCF``.
    """

    def __init__(
        self,
        pose: Callable[[], Pose | None],
        *,
        scene: str | Path | None = None,
        mjcf_dir: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        light: Callable[[datetime], LightState] = world_lights.light_state,
        period_s: float = PERIOD_S,
        meter_s: float = METER_S,
        frame_size: tuple[int, int] | None = None,
        seed: int = 7,
        jpeg_scale: int = JPEG_SCALE,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self._pose = pose
        self._clock = clock or taipei_now
        self._light = light
        self.scene = resolve_scene(scene, mjcf_dir)
        self.period_s = float(period_s)
        self.meter_s = float(meter_s)
        self.frame_size = frame_size
        self.seed = seed
        self.jpeg_scale = jpeg_scale
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._state: dict[str, Any] = {
            "running": False,
            "error": None,
            "note": msg("web.eyecam.EyeCam.init"),
        }
        self._cpu: deque[tuple[float, float]] = deque(maxlen=64)
        # The physics time of the latest pose and when it last moved on (monotonic).
        self._pose_t: Any = None
        self._pose_t_at = 0.0
        self._metered_key: str | None = None  # the light the auto-exposure last metered

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="eyecam", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._thread = None

    # ── what the page reads ─────────────────────────────────────────────

    def latest(self) -> dict[str, Any] | None:
        """The last frame with its JPEG bytes, or ``None`` before the first one."""
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def status(self) -> dict[str, Any]:
        """Everything but the picture, for a status line."""
        with self._lock:
            out = dict(self._state)
            if self._latest is not None:
                out.update({k: v for k, v in self._latest.items() if k != "jpeg"})
                out["jpeg_bytes"] = len(self._latest["jpeg"])
                out["error"] = self._state.get("error")
            out["cpu_share"] = self._cpu_share()
        return out

    def _cpu_share(self) -> float | None:
        if len(self._cpu) < 2:
            return None
        (wall0, cpu0), (wall1, cpu1) = self._cpu[0], self._cpu[-1]
        return round((cpu1 - cpu0) / max(1e-6, wall1 - wall0), 4)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    # ── the render thread ───────────────────────────────────────────────

    def _run(self) -> None:
        if self.scene is None:
            self._set(running=False, error=msg("web.eyecam.EyeCam.run.e7e80e"))
            return
        if not self.scene.is_file():
            self._set(
                running=False, error=msg("web.eyecam.EyeCam.run.ca368d", scene=str(self.scene))
            )
            return
        try:
            renderer = EyeRenderer(
                self.scene,
                frame_size=self.frame_size,
                seed=self.seed,
                jpeg_scale=self.jpeg_scale,
                jpeg_quality=self.jpeg_quality,
            )
        except Exception as error:  # no GL, no MuJoCo, a scene that does not compile
            self._set(
                running=False,
                error=msg("web.eyecam.EyeCam.run.05f77d"),
                error_detail=f"{type(error).__name__}: {error}",
            )
            return
        self._set(
            running=True,
            error=None,
            note=msg("web.eyecam.EyeCam.run.f38523"),
            scene=str(self.scene),
        )
        try:
            self._loop(renderer)
        finally:
            try:
                renderer.close()
            except Exception:
                pass
            self._set(running=False)

    def _loop(self, renderer: EyeRenderer) -> None:
        now = time.monotonic()
        next_meter, next_frame = now, now
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_meter or now >= next_frame:
                pose = self._read_pose()
                if pose is not None and self._frozen(pose):
                    self._set(note=FROZEN_NOTE, frozen=True)
                elif pose is not None:
                    if self._state.get("frozen"):
                        self._set(frozen=False, note="")  # the world moves again
                    try:
                        state = self._light(self._clock())
                        # The first frame under a new light comes after the auto-exposure has
                        # metered that light at least once, as on the robot (it meters all along).
                        if now >= next_meter or (
                            now >= next_frame and state.key != self._metered_key
                        ):
                            renderer.meter(pose, state)
                            self._metered_key = state.key
                        if now >= next_frame:
                            result = renderer.frame(pose, state)
                            with self._lock:
                                self._latest = result
                                self._state.update(
                                    error=None, error_detail=None, note="", frozen=False
                                )
                    except Exception as error:
                        self._set(
                            error=msg("web.eyecam.EyeCam.loop.d6b584"),
                            error_detail=f"{type(error).__name__}: {error}",
                        )
                if now >= next_meter:
                    next_meter = max(next_meter + self.meter_s, now)
                if now >= next_frame:
                    next_frame = max(next_frame + self.period_s, now)
                self._cpu.append((time.monotonic(), time.thread_time()))
            self._stop.wait(max(0.0, min(next_meter, next_frame) - time.monotonic()))

    def _frozen(self, pose: Pose) -> bool:
        """Whether the physics time of the pose has stood still for FROZEN_S (a pose without
        a time — a keyframe in a test — is never frozen)."""
        t = pose.get("t")
        now = time.monotonic()
        if t is None or t != self._pose_t:
            self._pose_t, self._pose_t_at = t, now
            return False
        return now - self._pose_t_at >= FROZEN_S

    def _read_pose(self) -> Pose | None:
        try:
            pose = self._pose()
        except Exception as error:
            self._set(
                note=msg("web.eyecam.EyeCam.read_pose.12d7a4"),
                error_detail=f"{type(error).__name__}: {error}",
            )
            return None
        if not pose:
            self._set(note=msg("web.eyecam.EyeCam.read_pose.f38523"))
            return None
        return pose
