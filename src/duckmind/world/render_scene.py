"""The head camera in a home: which lights its render gets, room by room (spec §11.2, §12.3).

MuJoCo's classic renderer on this Mac lights only the first 7 active lights (probed in the
phase-2 survey) and draws no shadows here, so a lamp would light the next room through the wall.
The approximation the owner accepted (F5): only the room the camera is in is lit by its own
windows and lamps — at most :data:`MAX_LIGHTS` MuJoCo lights, the weakest folded into the
bounced light when a room has more — and everything the camera sees of the rest of the home gets
that room's bounced light. The normalisation follows the camera: a white level surface on the
floor under the lens renders :data:`camera_model.RENDER_WHITE`, and its lux comes from the same
:class:`lights.LightState` the page and HomeWorld read, so the three agree.

The nest's backlight (C4) glows through the camera model's emission pass: an absolute brightness,
switched and coloured by :func:`lights.home_light_state`. Its geoms are those whose names contain
``backlight`` or ``status_light`` (the dock's status light, which has the backlight's meaning),
else the layout's nest backboard (``home_nest_board``), or the names passed in.

:class:`HomeEye` is what the world's eye service drives, on its render thread::

    eye = HomeEye.open(render_spec, home=HomeLights.from_layout(layout))  # on the thread
    eye.set_state(qpos, mocap_pos=..., mocap_quat=...)  # a world snapshot
    state = lights.home_light_state(duck_time, eye.home, charging=...)
    eye.meter(state)  # twice a second, whether or not anyone asks for frames
    capture = eye.capture(state, index=seq)  # on demand
    capture.uyvy()  # media.frame's bytes: capture.width × capture.height × 2, rotate 90
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckmind.world import camera_model as cm
from duckmind.world import lights as world_lights
from duckmind.world.lights import HomeLights, HomeLightState, LightState, Room

MAX_LIGHTS = cm.LIGHT_SLOTS
EMITTER_MARK = "backlight"
EMITTER_MARKS = (EMITTER_MARK, "status_light")  # the dock's status light (duckmind.world.dock)
NEST_BOARD = "home_nest_board"  # home-layout-v2's nest backboard (homelayout._nest_elements)


def emitter_names(spec: Any) -> list[str]:
    """The geoms of an ``MjSpec`` that can glow: named "…backlight…" or "…status_light…";
    without any, the nest's backboard (the tag is then seen against a glowing board)."""
    names = [
        g.name
        for g in spec.geoms
        if g.name and any(mark in g.name.lower() for mark in EMITTER_MARKS)
    ]
    if not names and spec.geom(NEST_BOARD) is not None:
        names = [NEST_BOARD]
    return names


def load_home_eye(
    scene: Any,
    *,
    home: HomeLights | None = None,
    config: cm.CameraConfig | None = None,
    emitters: Iterable[str] | None = None,
    camera: str = cm.CAMERA,
    edit: Callable[[Any], None] | None = None,
) -> cm.EyeScene:
    """The eye's own compiled copy of a home: ``scene`` is an ``MjSpec`` (a home layout's render
    spec: the world's bodies, joints and mocap bodies plus render-only extras, so a world
    snapshot's qpos fits) or a scene file read where it lies. The ceiling is the home's highest
    (a plane, unless the scene brings its own "…ceiling…" geoms)."""
    mujoco = cm.import_mujoco()
    if hasattr(scene, "compile"):
        spec, path = scene, Path(f"<{home.id if home else 'spec'}>")
    else:
        path = Path(scene).expanduser()
        spec = mujoco.MjSpec.from_file(str(path))
    cfg = config or cm.CameraConfig()
    names = list(emitters) if emitters is not None else emitter_names(spec)
    eye = cm.load_spec(
        spec,
        path=path,
        quality=cfg.quality,
        distortion=cfg.distortion,
        ceiling_m=home.ceiling if home is not None else cm.CEILING_M,
        camera=camera,
        emitters=names,
        edit=edit,
    )
    eye.extras["home"] = home.id if home is not None else None
    return eye


def camera_room(eye: cm.EyeScene, home: HomeLights) -> Room:
    """The room the head camera is in (the nearest one from a doorway gap)."""
    x, y, _ = cm.camera_position(eye)
    return home.room_at(x, y)


@dataclass(frozen=True)
class RoomLight:
    """The light one render got."""

    rig: cm.LightRig
    room: str | None
    key: str
    label: str
    glow: tuple[str, ...]  # emitters glowing

    @property
    def lux_per_unit(self) -> float:
        return self.rig.lux_per_unit

    def to_dict(self) -> dict[str, Any]:
        """For an evaluator's log (never for the duck's perception)."""
        return {
            "room": self.room,
            "key": self.key,
            "label": self.label,
            "lux": round(self.rig.lux, 4),
            "reference": [round(v, 4) for v in self.rig.reference],
            "lit": list(self.rig.lit),
            "folded": list(self.rig.folded),
            "glow": list(self.glow),
        }


def _glow(eye: cm.EyeScene, state: HomeLightState) -> tuple[str, ...]:
    backlight = state.backlight
    if backlight is None or not backlight.on or not eye.emitters:
        cm.set_emitters(eye, {})
        return ()
    names = tuple(name for name, mat in eye.emitters.items() if mat >= 0)
    cm.set_emitters(eye, {name: (backlight.color, backlight.scene_lux) for name in names})
    return names


def apply_lights(
    eye: cm.EyeScene,
    state: HomeLightState | LightState,
    *,
    home: HomeLights | None = None,
    reference: Sequence[float] | None = None,
) -> RoomLight:
    """Light our copy for one render.

    A :class:`HomeLightState`: the camera's room's lights (≤ 7, the rest as bounced light),
    normalised on the floor under the lens, and the backlight's glow. A single room's
    :class:`LightState` (phase 0's :func:`lights.light_state`): its lights, normalised at the
    eye's reference spot as in phase 0 (or under the lens for a home's room state)."""
    x, y, _ = cm.camera_position(eye)
    if isinstance(state, HomeLightState):
        lights_home = home or state.lights
        if lights_home is None:
            raise ValueError("a home light state without its HomeLights")
        room = lights_home.room_at(x, y)
        room_state = state.rooms[room.id]
        ref = tuple(reference) if reference is not None else (x, y, room.floor)
        rig = cm.light_rig(eye, room_state, reference=ref, slots=MAX_LIGHTS)
        return RoomLight(rig, room.id, room_state.key, room_state.label, _glow(eye, state))
    if reference is not None:
        ref = tuple(reference)
    elif state.room is None:
        ref = eye.reference
    else:
        ref = (x, y, state.reference[2])
    rig = cm.light_rig(eye, state, reference=ref, slots=MAX_LIGHTS)
    cm.set_emitters(eye, {})
    return RoomLight(rig, state.room, state.key, state.label, ())


class HomeEye:
    """The head camera in a home, for one render thread: our compiled copy, the camera (lens,
    AE, white balance, sensor) and the home's lights. Create, use and close it on one thread.
    """

    def __init__(
        self,
        eye: cm.EyeScene,
        *,
        home: HomeLights | None = None,
        config: cm.CameraConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.eye = eye
        self.home = home
        self.camera = cm.HeadCamera(eye, config=config, seed=seed)
        self.last_light: RoomLight | None = None

    @classmethod
    def open(
        cls,
        scene: Any,
        *,
        home: HomeLights | None = None,
        config: cm.CameraConfig | None = None,
        seed: int = 7,
        emitters: Iterable[str] | None = None,
        edit: Callable[[Any], None] | None = None,
    ) -> HomeEye:
        eye = load_home_eye(scene, home=home, config=config, emitters=emitters, edit=edit)
        return cls(eye, home=home, config=config, seed=seed)

    @property
    def ae(self) -> cm.AeLoop:
        return self.camera.ae

    @property
    def awb(self) -> cm.Awb:
        return self.camera.awb

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.eye.frame_size

    def set_pose(self, frame: dict[str, Any]) -> None:
        """The duck's pose in ``LiveBody.frame()`` form (trunk, quat, joints)."""
        cm.set_pose(self.eye, frame)

    def set_state(self, qpos: Any, *, mocap_pos: Any = None, mocap_quat: Any = None) -> None:
        """A world snapshot: every free object and joint, the mocap bodies."""
        cm.set_state(self.eye, qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    def light(self, state: HomeLightState | LightState) -> RoomLight:
        self.last_light = apply_lights(self.eye, state, home=self.home)
        return self.last_light

    def meter(self, state: HomeLightState | LightState) -> float:
        """One AE and white-balance tick (the robot meters all along, twice a second)."""
        light = self.light(state)
        return self.camera.meter(light.lux_per_unit, illuminant=light.rig.color)

    def converge(self, state: HomeLightState | LightState, max_ticks: int = 40) -> int:
        """Meter until the AE and white balance settle (a first frame, a test)."""
        light = self.light(state)
        return self.camera.converge(light.lux_per_unit, max_ticks, illuminant=light.rig.color)

    def capture(
        self, state: HomeLightState | LightState, *, index: int | None = None
    ) -> cm.Capture:
        """One delivered frame; ``extras`` carries the light it had, for evaluators only."""
        light = self.light(state)
        capture = self.camera.capture(light.lux_per_unit, index=index)
        capture.extras.update(light.to_dict())
        capture.extras["ae"] = cm.exposure_summary(capture.controls)
        capture.extras["wb"] = [round(v, 4) for v in capture.wb]
        return capture

    def close(self) -> None:
        self.camera.close()


def default_home_state(when: Any, **kwargs: Any) -> HomeLightState:
    """Phase 0's room as a home (for callers that have no layout yet)."""
    return world_lights.home_light_state(when, world_lights.DEFAULT_HOME, **kwargs)
