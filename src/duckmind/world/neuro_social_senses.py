"""a:social's three MODEL readings: a person's look, a voice's speaker signature, a roof overhead.

World side (it reads the simulator's truth for the sensor models; the brain never imports it),
like :mod:`duckmind.world.neuro_senses`. What the duck's sensors could honestly report — with
their limits, scatter and mix-ups — for three skills of the neural duck (a:social 2026-09-28):

* **a person's look** (for 跟著媽媽走: who is who, :mod:`duckmind.neuro.skill_imprint`) — MODEL
  of a person re-identification pass on the head camera (no validated detector exists, spec
  §8.8): a 20-cm-tall duck sees a person's legs and feet, so the look is the colour of the legs
  (chromaticity r, g of the trousers, its brightness — which the room's light shifts), how long
  the legs are (the hip height, from the foot's ground-plane range: the range's 8 % scatter), and
  only from :data:`SHIRT_FROM_M` away the shirt's colour. Only in bright or dim light, scatter by
  the light band, now and then a bad look (a partial view, a shadow).
* **a voice's speaker signature** (who spoke) — MODEL of a speaker embedding on the mono
  microphone's speech (pitch class, timbre), scattered more as the speech gets quieter over the
  room's noise.
* **the roof** (for 害怕時躲起來: a shelter's cover, :mod:`duckmind.neuro.skill_shelter`) — MODEL of
  a reading of the top of the picture: whether a surface covers it close above the lens, and how
  high (a roof-line pass, the floor line's mirror; duck-eye has none yet). Truth: what lies right
  over the lens (MuJoCo ``mj_ray`` straight up and 15° forward, :func:`roof_truth` — a wall or a
  sofa's face ahead is not a roof); read only in bright or dim light, from a sharp frame (walking:
  :data:`P_SHARP_STEPPING` of them), and only when the top of the picture looks up enough.

The people's clothes and heights are :data:`duckmind.world.people.PERSONAS`' own.
Deterministic for a seed. Floats in the prototype.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.world.people import PERSONAS, Persona, hip_height

VERSION = "neuro-social-senses-1"

SHIRT_FROM_M = 1.2  # the shirt is in the picture from about this far (the 62° tall frame)
LOOK_SIGMA = {"bright": 0.012, "dim": 0.025}  # chromaticity scatter
LUMA_LIGHT = 0.12  # the room's light shifts the legs' measured brightness by ~12 %
LUMA_SIGMA = 0.02
RANGE_SIGMA = 0.08  # the foot's ground-plane range (neuro_senses.RANGE_SIGMA): the legs' length
P_BAD_LOOK = 0.04  # a partial view, a shadow: the colours are off
BAD_SHIFT = 0.06
P_LOOK = {"bright": 0.95, "dim": 0.75}  # a look per pass for a person in view (else none)
# speaker signatures (pitch class, timbre), by persona: the child's high, the elder's low
VOICES = {
    "owner": (0.35, 0.40),
    "partner": (0.62, 0.55),
    "child": (0.86, 0.70),
    "elder": (0.30, 0.22),
    "guest": (0.46, 0.62),
}
VOICE_SIGMA = (0.025, 0.2)  # a + b × how far under VOICE_CLEAR_DB of SNR the speech is (÷ 20)
VOICE_CLEAR_DB = 20.0
ROOM_DB = 30.0
SPEECH = frozenset({"voice", "name", "hello"})
# the roof: rays through the top of the delivered frame (the upright frame is 62.2° tall)
ROOF_ROWS = (math.radians(22.0), math.radians(29.0))  # above the optical axis, in the picture
ROOF_RANGE_M = 1.2  # a surface within this along the ray can be read
ROOF_UP_MIN = 0.25  # a ray must climb at least this (sin of its elevation) to look "up"
ROOF_SIGMA = 0.15  # the height's scatter (× itself)
ROOF_OPEN = 9.9  # (skill_shelter.ROOF_OPEN) looked up and saw open space
P_SHARP_STEPPING = 0.35  # neuro_senses.P_SHARP_STEPPING
ROOF_HZ = 4.0


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _chroma(rgb: Sequence[float]) -> tuple[float, float, float]:
    r, g, b = (float(v) for v in rgb)
    s = max(1e-6, r + g + b)
    return r / s, g / s, 0.299 * r + 0.587 * g + 0.114 * b


def persona_look(p: Persona) -> tuple[float, float, float, float, float, float]:
    """A person's true look (skill_imprint.LOOK_DIMS): legs r, g, luma, legs m, shirt r, g."""
    lr, lg, luma = _chroma(p.trousers)
    sr, sg, _ = _chroma(p.shirt)
    return (lr, lg, luma, hip_height(p), sr, sg)


LOOKS = {p.id: persona_look(p) for p in PERSONAS}


def observe_look(
    who: str, band: str, range_m: float | None, rng: random.Random
) -> tuple[float, ...] | None:
    """One look at ``who`` as the eye would give it (None: too dark, or no look this pass)."""
    true = LOOKS.get(who)
    if true is None or band not in LOOK_SIGMA:
        return None
    if rng.random() >= P_LOOK[band]:
        return None
    sigma = LOOK_SIGMA[band]
    lr = true[0] + rng.gauss(0.0, sigma)
    lg = true[1] + rng.gauss(0.0, sigma)
    luma = true[2] * (1.0 + rng.gauss(0.0, LUMA_LIGHT)) + rng.gauss(0.0, LUMA_SIGMA)
    legs = true[3] * (1.0 + rng.gauss(0.0, RANGE_SIGMA)) if range_m is not None else math.nan
    if range_m is not None and range_m >= SHIRT_FROM_M:
        sr = true[4] + rng.gauss(0.0, sigma * 1.3)
        sg = true[5] + rng.gauss(0.0, sigma * 1.3)
    else:
        sr = sg = math.nan
    if rng.random() < P_BAD_LOOK:
        lr += rng.gauss(0.0, BAD_SHIFT)
        lg += rng.gauss(0.0, BAD_SHIFT)
    return (round(lr, 4), round(lg, 4), round(luma, 4), round(legs, 4), sr, sg)


def observe_voice(who: str, snr_db: float, rng: random.Random) -> tuple[float, float] | None:
    """The speaker signature of ``who``'s speech at this SNR (None: too quiet to tell)."""
    true = VOICES.get(who)
    if true is None or snr_db < 3.0:
        return None
    sigma = VOICE_SIGMA[0] + VOICE_SIGMA[1] * _clip((VOICE_CLEAR_DB - snr_db) / 20.0)
    return (round(true[0] + rng.gauss(0.0, sigma), 4), round(true[1] + rng.gauss(0.0, sigma), 4))


def read_roof(
    truth: float | None, band: str, stepping: bool, looked_up: bool, rng: random.Random
) -> float | None:
    """The roof reading from the truth (m above the lens, or None: nothing within range): None
    when it cannot tell (dark, a smeared frame, the picture not looking up enough)."""
    if band not in ("bright", "dim") or not looked_up:
        return None
    if stepping and rng.random() >= P_SHARP_STEPPING:
        return None
    if truth is None:
        return ROOF_OPEN
    return round(max(0.03, truth * (1.0 + rng.gauss(0.0, ROOF_SIGMA))), 3)


def roof_truth(
    model: Any,
    data: Any,
    cam: Sequence[float],
    axes: Sequence[Sequence[float]],
    *,
    exclude_body: int = -1,
    groups: Sequence[int] = (1, 1, 0, 0, 0, 0),
) -> tuple[float | None, bool]:
    """(the height above the lens of a surface right over it — straight up, and tilted 15° the way
    the picture looks —, None when none within ROOF_RANGE_M; whether the top of the picture climbs
    enough to look up at all). A wall or a sofa's face ahead is not a roof: only what is over the
    head counts (a:social's physics check: rays through the top rows alone took the sofa's front
    and arms for a roof). ``axes``: the camera's sensor axes (right, down, forward) in the world,
    as neuro_world.camera gives them (the picture is delivered turned upright: its up is −right);
    ``groups``: the geom groups a ray may hit (the home and the people; not the duck itself)."""
    import mujoco
    import numpy as np

    right, down, forward = (np.asarray(a, dtype=float) for a in axes)
    # the delivered picture is the sensor turned upright (neuro_world.sample's cx/cy): the
    # picture's up is the sensor's −right, the picture's right is the sensor's −down
    up = -right
    origin = np.asarray(cam, dtype=float)
    looked_up = False
    for e in ROOF_ROWS:
        d = math.cos(e) * forward + math.sin(e) * up
        d = d / max(1e-9, float(np.linalg.norm(d)))
        if float(d[2]) >= ROOF_UP_MIN:
            looked_up = True
    geomgroup = np.asarray(groups, dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    flat = np.array([forward[0], forward[1], 0.0])
    flat = flat / max(1e-9, float(np.linalg.norm(flat)))
    best: float | None = None
    for tilt in (0.0, math.radians(15.0)):
        d = math.cos(tilt) * np.array([0.0, 0.0, 1.0]) + math.sin(tilt) * flat
        start = origin + 0.02 * d
        dist = mujoco.mj_ray(model, data, start, d, geomgroup, 1, exclude_body, geomid)
        if dist < 0.0 or dist > ROOF_RANGE_M:
            continue
        height = (dist + 0.02) * float(d[2])
        if best is None or height < best:
            best = height
    return best, looked_up


@dataclass
class SocialSenses:
    """The three readings for one duck, from the live driver's or the headless rig's views: the
    senses hub (its eye tracks: which person each track is, never passed on), the world's
    ``neuro`` section (the latest camera sample: each person's light and range; the sounds: who
    is talking, how loud) and, for the roof, the physics (``roof_from``)."""

    seed: int = 7
    rng: random.Random = field(init=False)
    looks: dict[int, tuple[float, ...]] = field(default_factory=dict)
    voice: tuple[float, float] | None = None
    roof: float | None = None
    _passes: int = -1
    _voice_id: int | None = None
    _voice_sig: tuple[float, float] | None = None
    _roof_t: float = -1.0

    def __post_init__(self) -> None:
        self.rng = random.Random(f"neuro-social-senses:{self.seed}")

    def update_looks(self, hub: Any, section: Mapping[str, Any] | None) -> None:
        """After a new salience pass: a look for every person track the pass reported."""
        vision = getattr(hub, "vision", None)
        if vision is None or section is None:
            return
        passes = int(getattr(vision, "passes", 0))
        if passes == self._passes:
            return
        self._passes = passes
        samples = section.get("samples") or []
        if not samples:
            return
        entries = {str(e.get("id")): e for e in samples[-1].get("seen") or ()}
        from duckmind.world.neuro_senses import light_band

        looks: dict[int, tuple[float, ...]] = {}
        for key, track in getattr(vision, "tracks", {}).items():
            if not str(key).startswith("person:") or getattr(track, "kind", "") != "person":
                continue
            entry = entries.get(str(key))
            if entry is None:
                continue
            band = light_band(float(entry.get("lux", 0.0)))
            rng_m = entry.get("rng")
            look = observe_look(
                str(key).split(":", 1)[1],
                band,
                None if rng_m is None else float(rng_m),
                self.rng,
            )
            if look is not None:
                looks[int(track.tid)] = look
        self.looks = looks

    def update_voice(self, section: Mapping[str, Any] | None, sim_s: float | None) -> None:
        """The speaker signature of the speech sounding now (one per utterance)."""
        self.voice = None
        if section is None:
            return
        now = float(section.get("sim_time", sim_s or 0.0))
        for e in section.get("sounds") or ():
            if e.get("kind") not in SPEECH:
                continue
            if not (float(e.get("t0", 0.0)) <= now <= float(e.get("t1", 0.0)) + 0.3):
                continue
            src = str(e.get("src") or "")
            if not src.startswith("person:"):
                continue
            if self._voice_id != e.get("id"):
                self._voice_id = e.get("id")
                self._voice_sig = observe_voice(
                    src.split(":", 1)[1], float(e.get("db", 0.0)) - ROOM_DB, self.rng
                )
            self.voice = self._voice_sig
            return

    def update_roof(
        self,
        model: Any,
        data: Any,
        cam: Sequence[float],
        axes: Sequence[Sequence[float]],
        *,
        band: str,
        stepping: bool,
        sim_s: float,
        exclude_body: int = -1,
    ) -> None:
        """A roof reading at most ROOF_HZ a second (between them: none, "cannot tell")."""
        if sim_s - self._roof_t < 1.0 / ROOF_HZ:
            self.roof = None
            return
        self._roof_t = sim_s
        truth, looked_up = roof_truth(model, data, cam, axes, exclude_body=exclude_body)
        self.roof = read_roof(truth, band, stepping, looked_up, self.rng)

    def reading(self) -> dict[str, Any]:
        return {"looks": dict(self.looks), "voice": self.voice, "roof": self.roof}


__all__ = [
    "LOOKS",
    "ROOF_OPEN",
    "SocialSenses",
    "VOICES",
    "observe_look",
    "observe_voice",
    "persona_look",
    "read_roof",
    "roof_truth",
]
