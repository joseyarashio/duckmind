"""Formal world and sensor composition for the default Neuro9 runtime.

This is the live extension counterpart of the older headless play rig.  Truth stays in
the ``neuro`` world section; :class:`IntegratedSenseHub` turns it into typed, noisy,
egocentric contract readings before the brain sees it.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from duckmind.neuro import contract as C
from duckmind.world import neuro_senses as NSenses
from duckmind.world import neuro_social_senses as Social
from duckmind.world import neuro_sun as Sun
from duckmind.world import neuro_world as NW
from duckmind.world import worldkit as WK

EXTENSIONS = (
    "duckmind.world.jaw,duckmind.world.worldkit,duckmind.world.sensors,"
    "duckmind.world.neuro_integration,duckmind.world.neuro_sound"
)
AXIS_SIGMA = math.radians(8.0)
AXIS_RANGE_M = 0.8
AXIS_ELONG = 1.4


def _footprint(
    model: Any, data: Any, geoms: tuple[int, ...]
) -> tuple[float, float, list[tuple[float, float]]]:
    import numpy as np

    points: list[tuple[float, float]] = []
    best_axis, best_elong = 0.0, 1.0
    for geom in geoms:
        kind = int(model.geom_type[geom])
        size = [float(v) for v in model.geom_size[geom]]
        centre = data.geom_xpos[geom]
        matrix = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        if kind == 6:  # MuJoCo box
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    p = centre + matrix @ np.array([sx * size[0], sy * size[1], 0.0])
                    points.append((float(p[0]), float(p[1])))
            elong = max(size[0], size[1]) / max(1e-4, min(size[0], size[1]))
            col = 0 if size[0] >= size[1] else 1
            if elong > best_elong:
                best_elong = elong
                best_axis = math.atan2(float(matrix[1, col]), float(matrix[0, col]))
        else:
            radius = size[0]
            for k in range(8):
                angle = k * math.pi / 4
                points.append(
                    (
                        float(centre[0]) + radius * math.cos(angle),
                        float(centre[1]) + radius * math.sin(angle),
                    )
                )
    return best_axis, best_elong, points


def _width(points: list[tuple[float, float]], cam: tuple[float, ...], centre: Any) -> float:
    dx, dy = float(centre[0]) - cam[0], float(centre[1]) - cam[1]
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = -dy / norm, dx / norm
    dots = [x * ux + y * uy for x, y in points]
    return max(dots) - min(dots) if dots else 0.0


class IntegratedWorld(NW.NeuroWorld):
    """The one ``neuro`` section, extended with sun, object shape and roof truth."""

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self.sunfield: Any = None
        self._clock: Any = None

    def edit_spec(self, spec: Any, home: Any) -> None:
        # Registry.home is intentionally read here: simhome assigns it immediately before
        # running ordered spec edits, not when extension modules are imported/registered.
        if home.layout is None:
            raise RuntimeError("Neuro9 world requires the composed home layout for sun sensing")
        import os

        weather = os.environ.get("DUCKMIND_NEURO_WEATHER", "sunny")
        self.sunfield = Sun.SunField(Sun.SunModel.from_layout(home.layout, weather=weather))

    def ready(self, world: Any, body: Any = None) -> None:
        super().ready(world, body)
        entry = (getattr(self.registry, "sections", {}) or {}).get("lights")
        suite = getattr(getattr(entry, "fn", None), "__self__", None)
        if suite is not None and getattr(suite, "clock", None) is not None:
            self._clock = suite.clock.now
        if self.sunfield is None or self._clock is None:
            raise RuntimeError("Neuro9 sun channel requires the simulator light clock")
        skip = set(self.robot.bodies)
        for candidate in self.candidates:
            if candidate.kind in ("thing", "person", "hand"):
                skip.update(int(item) for item in candidate.bodies)
        self.sunfield.shade.ready(world.model, sorted(skip))
        self.sunfield.refresh(world.data, self._clock(), float(world.data.time), force=True)

    def _lux(self, state: Any, point: Any) -> float:
        room_lux = NW.NeuroWorld._lux(state, point)
        if float(point[2]) > 0.25:
            return room_lux
        return room_lux + self.sunfield.extra_lux(float(point[0]), float(point[1]))

    def sample(self, data: Any) -> dict[str, Any]:
        self.sunfield.refresh(data, self._clock(), float(data.time))
        out = super().sample(data)
        cam, axes = self.camera(data)
        yaw = float(out["duck"]["yaw"])
        by_id = {item.id: item for item in self.candidates}
        for entry in out["seen"]:
            candidate = by_id.get(entry["id"])
            if candidate is None or candidate.kind != "thing":
                continue
            body = int(candidate.bodies[0])
            geoms = tuple(
                g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == body
            )
            axis, elong, points = _footprint(self.model, data, geoms)
            centre = data.xipos[body]
            entry["w"] = round(_width(points, cam, centre), 4)
            entry["elong"] = round(elong, 2)
            entry["ax"] = round(NW._wrap(axis - yaw), 4) if elong >= AXIS_ELONG else None

        state = self._light_state()

        def room_lux(point: Any) -> float:
            return NW.NeuroWorld._lux(state, point)

        def clear(point: Any) -> bool:
            return self._floor_seen(data, cam, point)

        out["sun"] = Sun.view_entries(
            self.sunfield, data, cam, axes, yaw, NW.in_frame, clear, room_lux
        )
        readings = [
            room_lux((x, y, 0.0)) + self.sunfield.extra_lux(x, y)
            for x, y in Sun.meter_points(cam, axes[2])
        ]
        out["cam"]["lux_meter"] = round(Sun.log_mean(readings), 2)
        roof, looked_up = Social.roof_truth(self.model, data, cam, axes, exclude_body=self.cam_body)
        out["roof_truth"] = {"height_m": roof, "looked_up": looked_up}
        return out

    def _floor_seen(self, data: Any, cam: Any, point: Any) -> bool:
        import mujoco
        import numpy as np

        vector = np.asarray(point, dtype=float) - np.asarray(cam, dtype=float)
        distance = float(np.linalg.norm(vector))
        if distance < 1e-6:
            return True
        hit = np.zeros(1, dtype=np.int32)
        result = float(
            mujoco.mj_ray(
                self.model,
                data,
                np.asarray(cam),
                vector / distance,
                self._ray_group,
                1,
                self.cam_body,
                hit,
            )
        )
        geom = int(hit[0])
        if result < 0 or geom < 0:
            return True
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        return result >= distance - 0.03 or "floor" in name or "rug" in name

    def snapshot(self, world: Any) -> dict[str, Any]:
        result = super().snapshot(world)
        result["sun"] = self.sunfield.view()
        # Robotd's state packet always reserves wire joint 9 for a mouth, even when the
        # composed model has no jaw. Publish capability from the compiled model, never from
        # the length/value of that fixed-width packet.
        try:
            jaw_info = __import__("duckmind.world.jaw", fromlist=["info"]).info(world.model)
            result["capabilities"] = {
                **(result.get("capabilities") or {}),
                "jaw_present": bool(
                    jaw_info.get("actuator", -1) >= 0 and jaw_info.get("mandible_g") is not None
                ),
            }
            result["jaw"] = jaw_info
        except (ImportError, ValueError, AttributeError, TypeError):
            result["capabilities"] = {**(result.get("capabilities") or {}), "jaw_present": False}
        return result


def register(registry: Any) -> IntegratedWorld:
    import os

    part = IntegratedWorld(seed=int(os.environ.get("DUCKMIND_NEURO_SEED", "7") or 7))
    part.registry = registry
    WK.register_part(registry, part, section=NW.SECTION)
    registry.on_ready(lambda world, body: part.start_thread(), name="neuro-integration.pump")
    return part


class IntegratedVision(NSenses.VisionModel):
    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self.sun = Sun.SunSightModel(seed=seed)
        self.social = Social.SocialSenses(seed=seed)
        self.full_meter = NSenses.LightMeter(seed=seed + 101)
        self.roof: float | None = None
        self.roof_t_ms: int | None = None
        self.camera_readings = {"lux": None, "roof": None, "roof_t_ms": None}

    def step(
        self, t_ms: int, sample: Any, *, stepping: bool, saccade: bool, ego_v: float = 0.0
    ) -> list[Any]:
        # This method runs only when the existing look governor admits a camera pass.
        self.camera_readings = {"lux": None, "roof": None, "roof_t_ms": None}
        got = super().step(t_ms, sample, stepping=stepping, saccade=saccade, ego_v=ego_v)
        by_tid = {track.tid: key for key, track in self.tracks.items()}
        entries = {str(entry.get("id")): entry for entry in sample.get("seen") or ()}
        band = NSenses.light_band(float((sample.get("cam") or {}).get("lux", 300.0)))
        camera = sample.get("cam") or {}
        self.full_meter.read(t_ms, float(camera.get("lux_meter", camera.get("lux", 300.0))))
        self.camera_readings["lux"] = round(self.full_meter.lux, 1)
        roof_truth = sample.get("roof_truth") or {}
        reading = Social.read_roof(
            roof_truth.get("height_m"),
            band,
            stepping,
            bool(roof_truth.get("looked_up")),
            self.social.rng,
        )
        if reading is not None:
            self.roof, self.roof_t_ms = reading, t_ms
        fresh_roof = (
            self.roof_t_ms is not None and 0 <= t_ms - self.roof_t_ms <= NSenses.CAMERA_STALE_MS
        )
        self.camera_readings["roof"] = self.roof if fresh_roof else None
        self.camera_readings["roof_t_ms"] = self.roof_t_ms if fresh_roof else None
        output = []
        for sighting in got:
            entry = entries.get(by_tid.get(sighting.track, ""))
            if entry is None:
                output.append(sighting)
                continue
            changed: dict[str, Any] = {}
            if entry.get("kind") == "thing" and entry.get("w") is not None:
                rng = random.Random(f"shape:{self.seed}:{sighting.track}:{t_ms}")
                distance = max(float(entry.get("dist", 1.0)), 0.05)
                width = float(entry["w"]) * (1.0 + rng.gauss(0.0, NSenses.SIZE_SIGMA))
                changed["size"] = 2.0 * math.atan(max(width, 0.005) / (2.0 * distance))
                floor_range = entry.get("rng")
                if (
                    entry.get("ax") is not None
                    and floor_range is not None
                    and float(floor_range) <= AXIS_RANGE_M
                    and float(entry.get("elong", 1.0)) >= AXIS_ELONG
                    and band in ("bright", "dim")
                    and not stepping
                    and not saccade
                ):
                    angle = float(entry["ax"]) + rng.gauss(0.0, AXIS_SIGMA)
                    changed["axis"] = (angle + math.pi / 2) % math.pi - math.pi / 2
            if sighting.kind == "person":
                key = by_tid.get(sighting.track, "")
                person = key.split(":", 1)[1] if key.startswith("person:") else ""
                look = Social.observe_look(
                    person,
                    band,
                    None if entry.get("rng") is None else float(entry["rng"]),
                    self.social.rng,
                )
                changed["look"] = look
            output.append(replace(sighting, **changed) if changed else sighting)
        output.extend(
            self.sun.step(
                t_ms,
                sample.get("sun") or (),
                lux_under=float((sample.get("cam") or {}).get("lux", 300.0)),
                stepping=stepping,
                saccade=saccade,
            )
        )
        return output


class IntegratedSenseHub(NSenses.SenseHub):
    """Neuro9 channels with freshness-aware speaker, look, roof, lux, sun and jaw readings."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vision = IntegratedVision(seed=self.seed)
        self.voice: tuple[float, float] | None = None
        self.voice_t_ms: int | None = None
        self.capabilities: dict[str, bool | None] = {"jaw_present": None}
        self.world_errors: tuple[str, ...] = ()

    def _see(self, t_ms: int, section: Any, *, stepping: bool, saccade: bool) -> None:
        if not isinstance(section, dict):
            super()._see(t_ms, section, stepping=stepping, saccade=saccade)
            return
        # Process one world frame at a time so delayed captures keep their own lux/roof truth.
        for sample in section.get("samples") or ():
            before = len(self.pending)
            super()._see(
                t_ms,
                {**section, "samples": [sample]},
                stepping=stepping,
                saccade=saccade,
            )
            if len(self.pending) <= before:
                continue
            sidecar = self.vision.camera_readings
            item = self.pending[-1]
            item.camera = replace(
                item.camera,
                lux=sidecar["lux"],
                roof=sidecar["roof"],
                roof_t_ms=sidecar["roof_t_ms"],
            )

    def _camera(self, t_ms: int, ego: float) -> Any:
        camera = super()._camera(t_ms, ego)
        if self.camera_ms is None or t_ms - self.camera_ms > NSenses.CAMERA_STALE_MS:
            return replace(camera, lux=None, roof=None, roof_t_ms=None)
        return replace(
            camera,
            lux=self.camera.lux,
            roof=self.camera.roof,
            roof_t_ms=self.camera.roof_t_ms,
        )

    def step(self, doors: Any) -> C.SensorFrame:
        section = None
        world = getattr(doors, "world", None)
        if isinstance(world, dict):
            section = world.get("neuro")
        capabilities = (section or {}).get("capabilities") if isinstance(section, dict) else None
        errors = (section or {}).get("errors") if isinstance(section, dict) else None
        self.world_errors = tuple(
            str(error) for error in (errors or ()) if str(error).startswith(("sample:", "floor:"))
        )
        self.capabilities = {
            "jaw_present": (capabilities or {}).get("jaw_present")
            if isinstance(capabilities, dict)
            else None
        }
        frame = super().step(doors)
        social = self.vision.social
        social.update_voice(section, self.sim_s(frame.t_ms))
        if social.voice is not None and frame.sound.kind in Social.SPEECH:
            self.voice, self.voice_t_ms = social.voice, frame.t_ms
        else:
            self.voice = None
            self.voice_t_ms = None
        sound = replace(frame.sound, voice=self.voice, voice_t_ms=self.voice_t_ms)
        body = frame.body
        state = getattr(doors, "state", None)
        jaw_present = self.capabilities["jaw_present"] is True
        joints = getattr(state, "joints", None) if state is not None else None
        mouth = float(joints[9]) if jaw_present and joints is not None and len(joints) > 9 else None
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
        body = replace(body, active_skill=active_skill)
        if mouth is not None:
            body = replace(body, mouth=mouth)
        sources = dict(frame.sources)
        if mouth is None:
            sources["body"] = "real" if state is not None else "none"
        # A missing section is an explicit unknown, never a sticky stale social signature.
        return replace(frame, body=body, sound=sound, sources=sources)


def live_senses(seed: int = 7, vision_hz: float = NSenses.VISION_HZ) -> IntegratedSenseHub:
    return IntegratedSenseHub(seed=int(seed), vision_hz=float(vision_hz))


__all__ = ["EXTENSIONS", "IntegratedSenseHub", "IntegratedWorld", "live_senses", "register"]
