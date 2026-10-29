"""The world side of the sound skills: music with a tempo, a hand hovering in front of the
duck's head, and a ride through the air (a:sound 2026-09-28).

A simhome extension loaded AFTER :mod:`duckmind.world.neuro_world` (it drives that part's hand
and adds sounds through it)::

    DUCKMIND_SIMHOME_EXT=duckmind.world.worldkit,duckmind.world.sensors,
                         duckmind.world.neuro_world,duckmind.world.neuro_sound

(the same part runs on :class:`~duckmind.world.headless_duck.HeadlessDuck` through
:func:`attach`). Like ``neuro_world`` nothing here is a sense: it makes world events, and the
duck's sensor models decide what reaches it.

* **Music with a tempo** (``do: music``): :class:`~duckmind.world.neuro_music.Song` from a
  speaker in the room — its steady bed as one ``music`` sound event (as the page's old 放音樂
  button made), and every kick and snare as a short ``music`` event at the song's own time and
  its own measured rise over the bed (:meth:`Song.hits`), scheduled a little ahead. The
  microphone model (``neuro_senses.HearingModel``) turns those rises into onsets like any other
  sound; the brain must find the beat itself. The section says the song (bpm, seed, its start
  in physics seconds) so the page can play the very same samples.
* **A hand hovering in front of the head** (``do: hover``): the household's visible hand (the
  ``kit_hand`` mocap body — the camera and the ToF see it, it touches nothing) held up in front
  of the duck's face, palm toward it, at a distance (``distance_mm``, the page's slider), or the
  whole theremin gesture (``play``: it comes, hovers, comes closer, wobbles, backs off, leaves —
  the gesture upstream's ``sounds theremin`` auditions). Through ``neuro_world``'s own gesture
  machinery (its hand rules: never while the household's hand carries the duck).
* **A ride** (``do: swing``): the household's hand picks the duck up (if it is not held) and
  swings it through the air — side to side along an arc, turning and banking with the swing,
  like a child playing aeroplane — then holds it (the page's 放下來 sets it down, or
  ``set_down: true``). The duck feels it only through its own IMU and legs.

Ops: ``x_neuro_sound`` ``{"do": "music", "bpm": 104, "seed": 7} | {"do": "music_off"} |
{"do": "hover", "distance_mm": 300} | {"do": "hover", "play": true} | {"do": "hover_end"} |
{"do": "swing", "cycles": 3, "set_down": false} | {"do": "status"}``. Section ``neuro_sound``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import join, msg
from duckmind.world import worldkit as wk
from duckmind.world.neuro_music import TEMPOS, Song

SECTION = "neuro_sound"
VERSION = 1
MUSIC_DB1M = 66.0  # the song's steady bed at 1 m (neuro_world.SOUND_DB_1M["music"])
HIT_S = {"kick": 0.07, "snare": 0.05}  # how long a hit's loud part lasts
AHEAD_S = 0.8  # hits are put in the world this far ahead of their time
SPEAKER_H_M = 0.8
# the hover: the hand's own frame turned +90° about the duck's heading-frame y — its palm faces
# the duck, the forearm rises up and away (out of the ToF's beam). What the ToF ranges is the
# palm (the curled fingertips in front of it are too thin for its zones): its face is
# HOVER_NEAR_M beyond the hand's origin (measured with the ToF model, tof_model.TofSensor)
PALM_TO_DUCK = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
HOVER_NEAR_M = 0.065
TOF_SITE = "tof"
HOVER_MM = (100, 700)  # kinematics::hand's playable band (robotd's theremin)
HOVER_HOLD_S = 90.0  # a hover holds this long unless moved or ended
# A player's hand comes nearer gently: its looming (closing speed ÷ distance) stays under the
# camera model's "coming nearer" rate (neuro_senses.LOOM_PER_S 0.35/s) — a hand darting at the
# face is the page's other button (手突然靠近臉). With a min-jerk move the peak rate is 1.875 ×
# the mean: a move from d0 in to d1 takes at least 1.875 × ln(d0/d1) / HOVER_LOOM seconds.
HOVER_LOOM = 0.28
HOVER_MOVE_MIN_S = 0.6


def move_seconds(d0: float, d1: float) -> float:
    """How long a hand takes from ``d0`` to ``d1`` metres in front of the sensor."""
    if d1 >= d0:
        return max(HOVER_MOVE_MIN_S, 0.6 + 1.2 * (d1 - d0))
    return max(HOVER_MOVE_MIN_S, 1.875 * math.log(d0 / max(0.05, d1)) / HOVER_LOOM)


# the ride: side-to-side along an arc under the hand (a pendulum), turning and banking with it
SWING_PERIOD_S = 1.7
SWING_ANGLE = 0.85  # rad: the arc's half-angle
SWING_RADIUS_M = 0.28
SWING_UP_M = 0.12  # the hand first lifts it this much higher (room to swing)
SWING_BANK = 0.45  # roll per rad of swing
SWING_TURN = 1.0  # heading turned per rad of swing
SWING_EASE_S = 0.6  # the swing grows in / dies away over this


def _vec(a: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


def _to_world(yaw: float, v: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], float(v[2]))


def _axis_quat(axis: Sequence[float], angle: float) -> list[float]:
    n = math.sqrt(sum(float(a) * float(a) for a in axis)) or 1.0
    s = math.sin(angle / 2.0) / n
    return [math.cos(angle / 2.0), float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s]


@dataclass
class Music:
    song: Song
    t0: float  # physics seconds of the song's start
    pos: tuple[float, float, float]
    bed: Any  # the neuro part's SoundEvent for the steady bed
    scheduled_to: float  # song seconds: hits are in the world up to here
    db1m: float = MUSIC_DB1M
    hits: int = 0

    def view(self) -> dict[str, Any]:
        return {
            "on": True,
            "bpm": round(self.song.bpm, 2),
            "seed": self.song.seed,
            "t0": round(self.t0, 4),
            "db1m": self.db1m,
            "pos_mm": wk.mm(self.pos),
            "levels": self.song.levels(),
            "hits": self.hits,
        }


@dataclass
class Swing:
    t0: float  # physics seconds (when the hand began to swing; −1: waiting for the pickup)
    cycles: float
    set_down: bool
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # the hand where the swing began
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    yaw: float = 0.0
    done: bool = False

    @property
    def seconds(self) -> float:
        return self.cycles * SWING_PERIOD_S


class NeuroSound:
    """The part (see the module docstring). One per world."""

    name = "neuro_sound"

    def __init__(self, *, seed: int = 7) -> None:
        self.seed = int(seed)
        self.registry: Any = None
        self.neuro: Any = None  # neuro_world's part
        self.hand: Any = None
        self.world: Any = None
        self.music: Music | None = None
        self.swing: Swing | None = None
        self.hover: dict[str, Any] | None = None  # {"distance_mm", "play", "t0"}
        self.swings = 0
        self.errors: list[str] = []
        self._lock = threading.RLock()

    # ── wiring ──

    def bind(self, neuro: Any) -> NeuroSound:
        self.neuro = neuro
        self.hand = getattr(neuro, "hand", None)
        return self

    def ready(self, world: Any, body: Any = None) -> None:
        if self.neuro is None and self.registry is not None:
            sections = getattr(self.registry, "sections", {}) or {}
            entry = sections.get("neuro")
            self.neuro = getattr(getattr(entry, "fn", None), "__self__", None)
        if self.neuro is not None and self.hand is None:
            self.hand = getattr(self.neuro, "hand", None)
        self.world = world

    def _error(self, text: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(text)

    # ── the step hooks (the world's lock is held) ──

    def before_step(self, world: Any) -> None:
        swing = self.swing
        if swing is None or swing.done:
            return
        try:
            self._drive_swing(world, swing)
        except Exception as error:  # a broken swing never stops the world
            swing.done = True
            self._error(f"swing: {type(error).__name__}: {error}")

    def after_step(self, world: Any) -> None:
        music = self.music
        if music is None:
            return
        now = float(world.data.time)
        song_now = now - music.t0
        if music.scheduled_to < song_now + AHEAD_S:
            try:
                self._schedule(music, song_now + AHEAD_S + 0.4)
            except Exception as error:
                self._error(f"music: {type(error).__name__}: {error}")

    # ── music ──

    def _schedule(self, music: Music, until: float) -> None:
        neuro = self.neuro
        if neuro is None:
            return
        for t, kind, rise in music.song.hits(music.scheduled_to, until):
            neuro.add_sound(
                "music",
                music.pos,
                t0=music.t0 + t,
                seconds=HIT_S.get(kind, 0.06),
                db1m=music.db1m + rise,
                env="impulse",
                src=f"music:{kind}",
            )
            music.hits += 1
        music.scheduled_to = until

    def _music_on(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        neuro = self.neuro
        if neuro is None:
            raise ValueError(
                msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._music_on.4551a7")
            )
        tempo = request.get("tempo")
        bpm = float(TEMPOS.get(str(tempo), request.get("bpm") or TEMPOS["mid"]))
        seed = int(request.get("seed", self.seed))
        self._music_off(world)
        with world.lock:
            now = float(world.data.time)
            where = request.get("where_mm")
            if isinstance(where, (list, tuple)) and len(where) >= 2:
                pos = (int(where[0]) / 1000.0, int(where[1]) / 1000.0, SPEAKER_H_M)
            else:
                bounds = neuro._bounds() or (-3.0, -3.0, 3.0, 3.0)
                pos = (bounds[0] + 0.4, (bounds[1] + bounds[3]) / 2.0, SPEAKER_H_M)
            song = Song(bpm=bpm, seed=seed)
            t0 = now + 0.3  # the song starts a moment after the press (the page gets it first)
            bed = neuro.add_sound(
                "music", pos, t0=t0, seconds=3600.0, db1m=MUSIC_DB1M, src="music:bed"
            )
            neuro.music = bed  # (neuro_world's own flag and its page's music marker)
            music = Music(song=song, t0=t0, pos=pos, bed=bed, scheduled_to=0.0)
            self._schedule(music, AHEAD_S + 0.4)
            self.music = music
        return {
            "on": True,
            "bpm": round(song.bpm, 2),
            "where_mm": wk.mm(pos),
            "note_zh": msg(
                "neuro.incoming.incoming_world_neuro_sound.NeuroSound._music_on.7d15c3",
                bpm=song.bpm,
            ),
        }

    def _music_off(self, world: Any) -> dict[str, Any]:
        music = self.music
        if music is None:
            return {
                "on": False,
                "note_zh": msg(
                    "neuro.incoming.incoming_world_neuro_sound.NeuroSound._music_off.d7ced8"
                ),
            }
        with world.lock:
            now = float(world.data.time)
        music.bed.t1 = now
        neuro = self.neuro
        if neuro is not None:
            with neuro._lock:
                for event in neuro.sounds:
                    # the hits not yet sounded: they end before they begin (silent), and stay
                    # listed a moment so a microphone model that already has them hears the end
                    if event.src.startswith("music:") and event.t0 > now:
                        event.t1 = now
            if getattr(neuro, "music", None) is music.bed:
                neuro.music = None
        self.music = None
        return {
            "on": False,
            "note_zh": msg(
                "neuro.incoming.incoming_world_neuro_sound.NeuroSound._music_off.48bb36"
            ),
        }

    # ── the hover (theremin) ──

    def _hover(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        neuro = self.neuro
        if neuro is None:
            raise ValueError(
                msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._hover.4551a7")
            )
        from duckmind.world.neuro_world import GLeg

        play = bool(request.get("play"))
        distance = int(request.get("distance_mm", 300))
        distance = max(HOVER_MM[0] - 40, min(HOVER_MM[1] + 150, distance))
        with world.lock:
            data = world.data
            cam, yaw = neuro._point(data, "camera")
            q = wk.quat_mul(wk.yaw_quat(yaw), list(PALM_TO_DUCK))
            quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            # on the ToF's beam: the sensor sits beside the lens and looks a little down (the
            # head's pitch now: the duck holds its head still while it plays)
            side, tilt = self._beam(data, cam, yaw)

            def spot(d_m: float, dz: float = 0.0) -> tuple[float, float, float]:
                off = _to_world(yaw, (d_m, side, dz - d_m * math.tan(tilt)))
                return (cam[0] + off[0], cam[1] + off[1], cam[2] + off[2])

            def track(label: str, seconds: float, d_m: float) -> Any:
                # ``d_m``: the palm's face from the sensor; the hand comes to the duck (it
                # follows its head while it arrives) …
                origin = d_m - HOVER_NEAR_M
                offset = (origin, side, -origin * math.tan(tilt))
                return GLeg(label, "track", seconds, follow="camera", offset=offset, quat=quat)

            def still(label: str, seconds: float, d_m: float) -> Any:
                # … and then it is held still in the room (a person's hand does not follow a
                # duck that walks off)
                origin = d_m - HOVER_NEAR_M
                return GLeg(label, "move", seconds, to=spot(origin), follow="camera", quat=quat)

            current = self.hover
            hand_up = (
                current is not None
                and getattr(neuro, "gesture", None) is not None
                and not neuro.gesture.done
                and neuro.gesture.stim == "theremin_hand"
            )
            start = spot(0.75, 0.35)
            if play:
                forward = _to_world(yaw, (1.0, 0.0, 0.0))
                wobble_at = spot(0.32 - HOVER_NEAR_M)
                legs = [
                    track("come", 1.5, 0.55),
                    still("hover", 1.5, 0.45),
                    still("closer", move_seconds(0.45, 0.20), 0.20),
                    still("hold", 2.0, 0.20),
                    still("back", move_seconds(0.20, 0.32), 0.32),
                    GLeg(
                        "wobble",
                        "wave",
                        4.0,
                        to=wobble_at,
                        follow="camera",
                        amp=0.06,
                        hz=0.6,
                        axis=forward,
                        quat=quat,
                    ),
                    still("away", 2.5, 0.60),
                    GLeg("leave", "move", 0.8, to=spot(0.8, 0.5), follow="camera", quat=quat),
                ]
                self.hover = {"play": True, "distance_mm": None, "t0": float(data.time)}
                zh = msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._hover.20cm")
            else:
                d_m = distance / 1000.0
                was = (current or {}).get("distance_mm") if hand_up else None
                seconds = move_seconds(was / 1000.0, d_m) if was else 1.5
                legs = [
                    still("move", seconds, d_m) if hand_up else track("come", seconds, d_m),
                    still("hover", HOVER_HOLD_S, d_m),
                    GLeg("leave", "move", 0.8, to=spot(0.8, 0.5), follow="camera", quat=quat),
                ]
                self.hover = {"play": False, "distance_mm": distance, "t0": float(data.time)}
                zh = msg(
                    "neuro.incoming.incoming_world_neuro_sound.NeuroSound._hover.17e6ce",
                    distance=distance / 10,
                )
            neuro._start_gesture(data, "theremin_hand", legs, start)
        return {"hover": dict(self.hover), "note_zh": zh}

    def _beam(self, data: Any, cam: Sequence[float], yaw: float) -> tuple[float, float]:
        """The ToF sensor's side offset from the lens (m, + = the duck's left) and how far its
        beam looks down (rad), in the duck's heading frame now."""
        model = getattr(self.neuro, "model", None)
        if model is None:
            return 0.0, 0.0
        site = wk.site_id(model, TOF_SITE)
        if site < 0:
            return 0.0, 0.0
        pos = data.site_xpos[site]
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = float(pos[0]) - cam[0], float(pos[1]) - cam[1]
        side = -s * dx + c * dy
        forward = data.site_xmat[site].reshape(3, 3)[:, 0]
        tilt = math.asin(max(-0.6, min(0.6, -float(forward[2]))))
        return side, tilt

    def _hover_end(self, world: Any) -> dict[str, Any]:
        neuro = self.neuro
        self.hover = None
        if neuro is None:
            return {"ok": False}
        with world.lock:
            gesture = getattr(neuro, "gesture", None)
            if gesture is not None and not gesture.done and gesture.stim == "theremin_hand":
                # skip to its last leg (the hand leaves)
                gesture.index = max(gesture.index, len(gesture.legs) - 1)
                gesture.legs[gesture.index].t0 = None
        return {
            "ok": True,
            "note_zh": msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._hover_end"),
        }

    # ── the ride ──

    def _swing(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        hand = self.hand
        if hand is None:
            raise ValueError(
                msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._swing.3d2be4")
            )
        cycles = max(1.0, min(6.0, float(request.get("cycles", 3))))
        set_down = bool(request.get("set_down", False))
        with world.lock:
            neuro = self.neuro
            gesture = getattr(neuro, "gesture", None)
            if gesture is not None and not gesture.done:
                neuro._end_gesture(
                    world.data,
                    msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._swing.a7cf3f"),
                )
            if not getattr(hand, "holding", False) and getattr(hand, "script", None) is None:
                hand.command(world, {"do": "pickup"})
            self.swing = Swing(t0=-1.0, cycles=cycles, set_down=set_down)
            self.swings += 1
        return {
            "swing": {"cycles": cycles, "set_down": set_down},
            "note_zh": join(
                msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._swing.f9c496"),
                msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._swing.1aeabf")
                if set_down
                else msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound._swing.d29e22"),
            ),
        }

    def _drive_swing(self, world: Any, swing: Swing) -> None:
        hand = self.hand
        data = world.data
        now = float(data.time)
        if swing.t0 < 0.0:
            # waiting for the pickup to finish (the duck in the hand, the script over)
            if getattr(hand, "script", None) is None and getattr(hand, "holding", False):
                mocap = int(hand.mocap)
                swing.pos = _vec(data.mocap_pos[mocap])
                q = [float(v) for v in data.mocap_quat[mocap]]
                swing.quat = (q[0], q[1], q[2], q[3])
                swing.yaw = wk.heading_of(q)
                swing.t0 = now
            elif getattr(hand, "script", None) is None and not getattr(hand, "holding", False):
                swing.done = True  # the pickup failed (nothing to swing)
            return
        if not getattr(hand, "holding", False) or getattr(hand, "script", None) is not None:
            swing.done = True  # set down, or the hand took over: the ride is over
            return
        t = now - swing.t0
        total = swing.seconds + 2.0 * SWING_EASE_S
        if t >= total:
            swing.done = True
            if swing.set_down:
                hand.command(world, {"do": "place", "to": "here"})
            return
        # lift a little higher first, then swing side to side, easing in and out
        lift = SWING_UP_M * wk.min_jerk(t / SWING_EASE_S)
        ease = min(1.0, t / SWING_EASE_S, (total - t) / SWING_EASE_S)
        ease = wk.min_jerk(max(0.0, ease))
        a = (
            SWING_ANGLE
            * ease
            * math.sin(2.0 * math.pi * max(0.0, t - SWING_EASE_S * 0.5) / SWING_PERIOD_S)
        )
        # a pendulum under the hand's pivot: sideways (the duck's heading frame y), rising at
        # the ends of the arc
        off = _to_world(
            swing.yaw,
            (0.0, SWING_RADIUS_M * math.sin(a), SWING_RADIUS_M * (1.0 - math.cos(a)) + lift),
        )
        pos = [swing.pos[0] + off[0], swing.pos[1] + off[1], swing.pos[2] + off[2]]
        turn = wk.yaw_quat(SWING_TURN * a)
        bank = _axis_quat(_to_world(swing.yaw + SWING_TURN * a, (1.0, 0.0, 0.0)), -SWING_BANK * a)
        quat = wk.quat_mul(bank, wk.quat_mul(turn, list(swing.quat)))
        mocap = int(hand.mocap)
        data.mocap_pos[mocap] = pos
        data.mocap_quat[mocap] = quat

    # ── ops and the section ──

    def ops(self) -> dict[str, Any]:
        return {"x_neuro_sound": self.op}

    def op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        world = self.world
        if world is None:
            raise ValueError(msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound.op.a1a379"))
        do = str(request.get("do", "status"))
        with self._lock:
            if do == "music":
                return self._music_on(world, request)
            if do == "music_off":
                return self._music_off(world)
            if do == "hover":
                return self._hover(world, request)
            if do == "hover_end":
                return self._hover_end(world)
            if do == "swing":
                return self._swing(world, request)
            if do == "status":
                with world.lock:
                    return {"ready": True, "section": self.snapshot(world)}
        raise ValueError(
            msg("neuro.incoming.incoming_world_neuro_sound.NeuroSound.op.0cad5b", given_id=do)
        )

    def snapshot(self, world: Any) -> dict[str, Any]:
        data = world.data
        now = float(data.time)
        music = self.music
        swing = self.swing
        neuro = self.neuro
        gesture = getattr(neuro, "gesture", None) if neuro is not None else None
        hovering = gesture is not None and not gesture.done and gesture.stim == "theremin_hand"
        hover = dict(self.hover) if self.hover is not None and hovering else None
        if hover is not None and gesture is not None:
            hover["leg"] = gesture.legs[min(gesture.index, len(gesture.legs) - 1)].label
        return {
            "v": VERSION,
            "sim_time": round(now, 4),
            "music": music.view() if music is not None else {"on": False},
            "hover": hover,
            "swing": None
            if swing is None or swing.done
            else {
                "t": round(now - swing.t0, 3) if swing.t0 >= 0.0 else None,
                "waiting": swing.t0 < 0.0,
                "seconds": swing.seconds,
            },
            "swings": self.swings,
            "errors": list(self.errors[-5:]),
        }


def register(registry: Any) -> NeuroSound:
    """simhome's entry point (after ``neuro_world``): the part, its hooks, ops and section."""
    import os

    part = NeuroSound(seed=int(os.environ.get("DUCKMIND_NEURO_SEED", "7") or 7))
    part.registry = registry
    wk.register_part(registry, part, section=SECTION)
    return part


def attach(host: Any, neuro: Any, *, seed: int = 7) -> NeuroSound:
    """The part on an in-process world (a HeadlessDuck or worldkit.Headless) beside the
    ``neuro_world`` part ``neuro`` (:func:`duckmind.world.neuro_world.attach`)."""
    part = NeuroSound(seed=seed).bind(neuro)
    part.ready(host.world, None)
    if hasattr(host, "before"):
        before, after = host.before, host.after
    else:
        before, after = host._before, host._after
    before.append(part.before_step)
    after.append(part.after_step)
    return part


__all__ = ["NeuroSound", "SECTION", "attach", "register"]
