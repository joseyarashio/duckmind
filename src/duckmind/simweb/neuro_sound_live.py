"""Sound skills for the formal Neuro9 runtime and its legacy demo compatibility shim.

The default driver uses :mod:`duckmind.neuro.integration`'s single integrated brain, attaches a
VoiceEngine in ``load_lanes`` and calls :func:`after_tick` once from ``NeuroDriver.tick``. The
older :func:`patch` entry point remains for demo callers; it installs the formal registry first
and detects an already-native sound extension so it cannot double-wrap lanes or tick PCM twice.

The :class:`SoundHub` routes expose the streamed voice at ``/api/neuro/sound/status`` and
``/pcm`` and send world events through ``/api/neuro/sound/world``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import skill_sound as SS
from duckmind.simweb import neuro_mic as MIC
from duckmind.simweb import neuro_voice_stream as V

SOUND_EXTENSION = "duckmind.world.neuro_sound"
MIC_ENV = "DUCKMIND_LIVE_MIC"  # "0": no microphone stream (live voice, LV0; on by default)
MAX_BODY = 2048
WORLD_DOS = ("music", "music_off", "hover", "hover_end", "swing")


def _offsets(pose: Any) -> tuple[float, float, float, float]:
    from duckmind.neuro import motor as M

    return M.pose_to_offsets(pose)


def link_light(link: Any) -> bool:
    """duck-body's unsure pickup states in the body link's last output (see SoundDuck)."""
    if link is None:
        return False
    view = link.prox.view
    return bool(view.unloaded or view.lifted or "airborne" in link.gate)


class SoundMotor:
    """The motor lane with the sound skills: the dance's nod on the head, and robotd's one-PCM
    rule for one-shot calls (dropped, not queued, while a stream holds it)."""

    def __init__(self, base: Callable[..., Any], brain: Any, engine: Any) -> None:
        self.base = base
        self.brain = brain
        self.engine = engine
        self.dropped = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(self, t_ms: int, cmd: Any, frame: Any, state: Any) -> tuple[Any, list[Any]]:
        request, vocals = self.base(t_ms, cmd, frame, state)
        # IntegratedBrain already runs SoundSkills.pre/post in its single brain step.
        # Read that tick's facade output; `last` is only the legacy wrapper API.
        out = getattr(self.brain, "sound_out", None)
        if out is None:
            out = getattr(self.brain, "last", None)
        if out is not None and out.dancing:
            bob = _offsets(out.bob)
            head = tuple(float(a) + float(b) for a, b in zip(request.head, bob, strict=True))
            request = replace(request, head=head)
        engine = self.engine
        # A reflex's alarm belongs to the new owner. after_tick releases a held ride or
        # theremin stream on this same tick, so do not discard that first alarm merely because
        # the previous engine tick still owned PCM. Carry remains intentionally silent.
        reflex_takeover = getattr(cmd, "reflex", None) is not None
        if (
            vocals
            and not reflex_takeover
            and engine is not None
            and engine.last is not None
            and engine.last.owns_pcm
        ):
            self.dropped += len(vocals)
            engine.voice.skipped += len(vocals)
            vocals = []
        return request, vocals


@dataclass
class Sound:
    """What a:sound hangs on one duck's lanes."""

    brain: Any
    motor: SoundMotor | None
    engine: V.VoiceEngine | None
    log: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20000))
    tof_seen: Any = None
    tof_at_ns: int = 0
    mouth_sent: float | None = None
    wheee_owned: bool = False
    mic: MIC.MicStream | None = None  # the duck's microphone as PCM (live voice, LV0)


STREAM = V.SoundStream()  # the page's stream (one per server; a duck switch keeps it)


def attach_lanes(lanes: Any, *, microduck: Path | None = None) -> Sound:
    """Wrap one duck's lanes (brain, motor) and give it robotd's streamed voice."""
    current = getattr(lanes, "sound", None)
    if isinstance(current, Sound):
        return current
    brain = lanes.brain
    # IntegratedBrain exposes the explicit sound-lane protocol and must not be
    # wrapped: doing so executes pre/post twice and duplicates state transitions.
    integrated_sound = all(
        callable(getattr(getattr(brain, "sound_skills", None), name, None))
        for name in ("pre", "post")
    ) and hasattr(brain, "sound_out")
    if not integrated_sound and not isinstance(brain, SS.SkillBrain):
        brain = SS.SkillBrain(brain)
        lanes.brain = brain
    engine: V.VoiceEngine | None = None
    try:
        engine = V.VoiceEngine(
            duck_seed=int(getattr(brain.genome, "seed", 7)), microduck=microduck, stream=STREAM
        )
    except Exception as error:  # (no voice: the skills still run)
        lanes.notes.append(f"a:sound voice: {type(error).__name__}: {error}")
    motor = None
    if lanes.motor is not None:
        motor = SoundMotor(lanes.motor, brain, engine)
        lanes.motor = motor
    sound = Sound(brain=brain, motor=motor, engine=engine)
    if os.environ.get(MIC_ENV, "1") != "0":
        try:
            sound.mic = MIC.make_mic(seed=int(getattr(brain.genome, "seed", 7)), engine=engine)
        except Exception as error:  # (no microphone stream: the voice and the skills still run)
            lanes.notes.append(f"live mic: {type(error).__name__}: {error}")
    lanes.sound = sound
    return sound


def after_tick(driver: Any, state: Any, intents: list[Any], *, lanes: Any = None) -> list[Any]:
    """After the driver's tick: robotd's sound for this tick (the engine on the physics clock:
    the wheee hold, the theremin up/down and its note from tofd's raw frame, the room's music),
    and the theremin's beak (robotd drives the mouth target itself while the instrument is up)."""
    lanes = driver.lanes if lanes is None else lanes
    sound: Sound | None = getattr(lanes, "sound", None)
    if sound is None or sound.engine is None:
        return intents
    out = getattr(sound.brain, "sound_out", None)
    if out is None:
        out = getattr(sound.brain, "last", None)
    if out is None:
        return intents
    from duckmind.duckbody import protocol as P

    t_ms = int(driver._last_t_ms)
    senses = lanes.senses
    sim_s = float(senses.sim_s(t_ms)) if hasattr(senses, "sim_s") else t_ms / 1000.0
    frame = driver._tof_frame
    age: float | None = None
    if frame is not None:
        if frame is not sound.tof_seen:
            sound.tof_seen = frame
            sound.tof_at_ns = int(frame.get("t_ns") or state.t_ns)
        age = max(0.0, (int(state.t_ns) - sound.tof_at_ns) / 1e9)
    world = driver._world or {}
    room = (world.get("neuro_sound") or {}).get("music") if isinstance(world, Mapping) else None
    wall = time.time()
    cmd = driver._last_cmd
    interrupted = bool(
        (cmd is not None and cmd.reflex is not None)
        or (cmd is not None and cmd.behaviour == "carry_toy")
    )
    hold = bool(out.wheee_hold and not interrupted)
    theremin = bool(out.theremin_up and not interrupted)
    release = sound.wheee_owned and not hold
    tick = sound.engine.tick(
        t_s=t_ms / 1000.0,
        sim_s=sim_s,
        w=wall,
        wheee_hold=hold,
        theremin_up=theremin,
        tof=frame,
        tof_age_s=age,
        room=room,
        release=release,
    )
    sound.wheee_owned = hold
    _tick_mic(sound, driver, state, sim_s, wall, room, tick)
    body = driver._body
    if tick.mouth is not None:
        # robotd: while the instrument is up the beak is the note's (whatever else is asked)
        intents = [i for i in intents if getattr(i, "method", "") != "robot.mouth"]
        intents.append(P.Intent("robot.mouth", {"open": round(float(tick.mouth), 3)}))
        sound.mouth_sent = float(tick.mouth)
        if body is not None and hasattr(body, "mouth_cmd"):
            body.mouth_cmd = float(tick.mouth)
    elif sound.mouth_sent is not None:
        sound.mouth_sent = None
    sound.log.append(
        {
            "t_ms": t_ms,
            "w": round(wall, 4),
            "sim_s": round(sim_s, 4),
            "beh": driver._last_cmd.behaviour if driver._last_cmd is not None else None,
            "ride": tick.ride,
            "note_hz": None if tick.note_hz is None else round(tick.note_hz, 1),
            "hand_mm": tick.hand_mm,
            "wheee": out.wheee_hold,
            "up": out.theremin_up,
            "bpm": round(out.beat.bpm, 1) if out.beat.active else None,
            "lock": round(out.beat.lock, 2),
            "chirp": out.chirp,
            "timing_ms": round(out.timing_s * 1000.0),  # its nods' timing, learned by feel
            "felt": out.felt_nods,
            "hp_felt": _felt_pitch(sound.brain),  # its joint sense of its head pitch (rad)
            # (the listening check: this tick's frame and the song sample its music starts at)
            "seq": None if tick.chunk is None else tick.chunk.seq,
            "room_k": sound.engine.room.last_start,
            "song": None
            if sound.engine.room.spec is None
            else [sound.engine.room.spec[0], sound.engine.room.spec[1]],
        }
    )
    return intents


def _tick_mic(
    sound: Sound, driver: Any, state: Any, sim_s: float, wall: float, room: Any, tick: Any
) -> None:
    """One 20 ms frame of the duck's microphone (:mod:`duckmind.simweb.neuro_mic`): the world's
    sounds as the hearing model gets them, the song, the plane of the voice engine and the calls
    the duck voiced this tick. A fault here never stops the voice or the body."""
    mic = sound.mic
    if mic is None:
        return
    try:
        motor_db: float | None = None
        if state is not None:
            try:
                from duckmind.world.neuro_senses import self_noise_db

                motor_db = float(self_noise_db(state))
            except (AttributeError, TypeError, ValueError):
                motor_db = None  # (no motor state to hear: the room's noise alone)
        world = driver._world or {}
        section = world.get("neuro") if isinstance(world, Mapping) else None
        sounds = section.get("sounds") if isinstance(section, Mapping) else None
        chunk = tick.chunk
        mic.tick(
            sim_s,
            sounds if isinstance(sounds, (list, tuple)) else None,
            music=room if isinstance(room, Mapping) else None,
            duck_pcm48=None if chunk is None else chunk.duck,
            calls=tuple(getattr(driver, "_tick_vocals", ()) or ()),
            motor_db=motor_db,
            wall=wall,
        )
    except Exception as error:
        if len(mic.errors) < 20:
            mic.errors.append(f"tick: {type(error).__name__}: {error}")


def _felt_pitch(brain: Any) -> float | None:
    head = getattr(brain, "felt_head", None)
    return None if not head or len(head) < 2 else round(float(head[1]), 4)


_PATCHED = False


def patch(*, microduck: Path | None = None) -> None:
    """Compatibility entry point for the older sound-only demo server."""
    global _PATCHED
    if _PATCHED:
        return
    from duckmind.neuro.integration import install_default_skills

    # The formal factory owns registry setup; calling SS.install first makes its legacy
    # overlap guard reject the default registry and can silently downgrade a lane.
    install_default_skills()
    SS.install()
    from duckmind.simweb import neuro_live as NL

    if microduck is not None:
        os.environ["DUCKMIND_MICRODUCK"] = str(microduck)
    # The default runtime already attaches VoiceEngine from load_lanes, runs after_tick once in
    # NeuroDriver.tick and registers neuro_sound in its extension list. Keep the demo shim from
    # wrapping any of those paths a second time.
    if SOUND_EXTENSION in NL.world_extensions():
        _PATCHED = True
        return

    base_load = NL.load_lanes

    def load_lanes(spec: Any) -> Any:
        lanes = base_load(spec)
        attach_lanes(lanes, microduck=microduck)
        return lanes

    NL.load_lanes = load_lanes
    base_tick = NL.NeuroDriver.tick

    def tick(self: Any, state: Any) -> list[Any]:
        brain = getattr(self.lanes, "brain", None)
        if hasattr(brain, "light"):
            brain.light = link_light(self._last_out)
        intents = base_tick(self, state)
        try:
            return after_tick(self, state, intents)
        except Exception as error:  # (a sound fault never stops the body)
            self.stats.last_error = f"a:sound: {type(error).__name__}: {error}"
            return intents

    NL.NeuroDriver.tick = tick  # type: ignore[method-assign]
    base_ext = NL.world_extensions

    def world_extensions() -> str:
        ext = base_ext()
        return ext if SOUND_EXTENSION in ext else f"{ext},{SOUND_EXTENSION}"

    NL.world_extensions = world_extensions
    _PATCHED = True


# ── the page's routes ──────────────────────────────────────────────────────────────


class SoundHub:
    """``/api/neuro/sound/…``: the stream (GET ``pcm``), what streams now (GET ``status``), and
    the world events a:sound's buttons make (POST ``world``)."""

    def __init__(self, driver_of: Callable[[], Any]) -> None:
        self.driver_of = driver_of
        self.routes = V.SoundRoutes(STREAM, self.status)
        self._lock = threading.Lock()

    @staticmethod
    def handles(path: str) -> bool:
        return path.startswith("/api/neuro/sound")

    def _sound(self) -> Sound | None:
        driver = self.driver_of()
        lanes = getattr(driver, "lanes", None)
        return getattr(lanes, "sound", None)

    def status(self) -> dict[str, Any]:
        driver = self.driver_of()
        sound = self._sound()
        out: dict[str, Any] = {"running": driver is not None and sound is not None}
        if sound is None:
            out["note_zh"] = msg("neuro.web.neuro_sound_live.SoundHub.status")
            return out
        last = getattr(sound.brain, "sound_out", None)
        if last is not None:
            out["skills"] = last.view()
        else:
            legacy = getattr(sound.brain, "last", None)
            out["skills"] = None if legacy is None else legacy.view()
        out["voice"] = None if sound.engine is None else sound.engine.view()
        world = getattr(driver, "_world", None) or {}
        out["world"] = world.get("neuro_sound") if isinstance(world, Mapping) else None
        out["tof"] = bool(getattr(driver, "_tof_frame", None))
        out["stream"] = {"seq": STREAM.seq, "clients": STREAM.clients, "sent": STREAM.sent}
        out["dropped_calls"] = 0 if sound.motor is None else sound.motor.dropped
        out["mic"] = None if sound.mic is None else sound.mic.view()
        out["genome"] = {
            "seed": int(getattr(sound.brain.genome, "seed", -1)),
            "branch": getattr(sound.brain.genome, "branch", None),
        }
        return out

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        if path == "/api/neuro/sound/log":
            # every tick's sound (the evaluators): ?since=<t_ms>
            from urllib.parse import parse_qs

            raw = parse_qs(query).get("since", ["-1"])[0]
            since = int(raw) if raw.lstrip("-").isdigit() else -1
            sound = self._sound()
            rows = [] if sound is None else [r for r in list(sound.log) if r["t_ms"] > since]
            _json(handler, HTTPStatus.OK, {"ok": True, "rows": rows[-5000:]})
            return
        self.routes.get(handler, path, query)

    def post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if path != "/api/neuro/sound/world":
            _json(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        length = int(handler.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY:
            _json(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False})
            return
        try:
            request = json.loads(handler.rfile.read(length) or b"{}")
            do = str(request.get("do", ""))
            if do not in WORLD_DOS:
                raise ValueError(msg("neuro.web.neuro_sound_live.SoundHub.post.126a36", do=do))
            driver = self.driver_of()
            body = getattr(driver, "_body", None)
            call = getattr(body, "world_call", None)
            if not callable(call):
                raise ValueError(msg("neuro.web.neuro_sound_live.SoundHub.post.eaef46"))
            params = {
                k: v
                for k, v in request.items()
                if k in ("bpm", "tempo", "seed", "distance_mm", "play", "cycles", "set_down")
            }
            with self._lock:
                answer = call("x_neuro_sound", do=do, **params)
            wake = getattr(driver, "_aux_wake", None)
            if wake is not None:
                wake.set()  # (the world's truth is read again at once)
            _json(handler, HTTPStatus.OK, {"ok": True, **dict(answer)})
        except (ValueError, RuntimeError, ConnectionError) as error:
            # (a refusal in plain words, not an HTTP error: the page says it, nothing broke)
            _json(handler, HTTPStatus.OK, {"ok": False, "error": str(error)})


def _json(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def microduck_from_env() -> Path | None:
    raw = os.environ.get("DUCKMIND_MICRODUCK")
    return Path(raw) if raw else None


__all__ = [
    "STREAM",
    "Sound",
    "SoundHub",
    "SoundMotor",
    "after_tick",
    "attach_lanes",
    "patch",
]
