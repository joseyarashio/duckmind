"""Drives and emotions: leaky integrators that rise and fall and modulate everything.

Each drive relaxes toward a target with its own time constant (exact exponential steps, so any
tick rate gives the same curve) and takes impulses from events (an onset counts once, whatever
the tick rate). Nothing here decides a behaviour; the drives are inputs to selection
(``select.py``), to the reflex gains and to the motor style (tempo, amplitude, speed).

* ``energy`` (fatigue = 1 − energy): walking and exertion drain it, sitting and sleep restore it.
* ``sleep_pressure`` (process S) rises awake, falls asleep; ``alertness`` (process C) is the
  circadian day from the duck's clock, shifted by its chronotype; ``sleepiness`` combines them.
* ``curiosity`` builds up in a familiar world and jumps at novelty; investigating consumes it.
* ``fear`` jumps at threats (startle, looming, falls) and decays (slowly in a timid duck;
  faster with family close by: social buffering).
* ``social`` (the need for the family) builds alone and falls with contact (Duck Core's
  ``SOCIAL.contact`` weights), never below ``SOCIAL.floor``.
* ``comfort``, ``boredom``, ``restless`` (the locomotor urge), ``hunger`` (battery), ``play``.
* ``arousal`` — the neuromodulator: tonic level × alertness, lifted by salient input and fear;
  it scales locomotor speed, motor amplitude and tempo, and reflex gain.
* ``valence`` — mood, −1..1, slow.
* ``stiffness`` — builds while asleep; a stretch after waking consumes it.
* the maintenance needs (integration 2026-09-28): ``groom`` (the itch to preen, shake and
  stretch: builds over minutes, faster after walking and after a scare — comfort acts follow
  disturbance, McKinney 1965), ``peckish`` (the urge to peck and forage at the floor) and
  ``vigil`` (walking builds the urge to stop and look around) — each consumed by doing it, so a
  duck interleaves its walks with preening, pecking and scanning instead of walking all day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from duckmind.duckcore import content
from duckmind.neuro.genome import NeuralParams

SOCIAL_FLOOR = int(content.SOCIAL["floor"]) / 10_000
HUNGER_BATTERY_AT = 0.58
# how many seconds of doing it use up a full need (preening an itch, pecking, looking around)
GROOM_USE_S = 40.0  # (neuro:fix 08:30: at 90 s a preen used a sixth of its itch and the next
# preen came within a minute — 48 preens in 30 min, 34–36 % of a calm duck's day; pet:etho's
# comfort budget is 10–20 %: now a 15–30-s preen scratches the itch away)
PECK_USE_S = 40.0
VIGIL_USE_S = 20.0


def _relax(x: float, target: float, dt_s: float, tau_s: float) -> float:
    return target + (x - target) * math.exp(-dt_s / max(1e-3, tau_s))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def circadian(day_minute: float, shift_min: float) -> float:
    """Alertness 0..1 over the day: up from ~07:00, down from ~21:30 (Duck Core's RHYTHM wake
    and wind-down), with a small after-lunch dip; the chronotype shifts it (early bird −)."""
    wake = float(content.RHYTHM["wake_min"]) + shift_min
    wind = float(content.RHYTHM["wind_down_min"]) + shift_min
    m = day_minute % 1440.0
    day = _sig((m - wake) / 25.0) * _sig((wind - m) / 35.0)
    dip = 0.12 * math.exp(-(((m - 840.0 - shift_min) / 80.0) ** 2))
    return _clip(0.04 + 0.92 * day - dip)


@dataclass(slots=True)
class DriveInputs:
    """What the brain feeds the drives this tick (levels 0..1; impulses once per event)."""

    dt_s: float
    day_minute: float
    asleep: bool = False
    sitting: bool = True
    walking: float = 0.0  # fraction of the tick spent stepping
    exertion: float = 0.0  # non-walking motor effort (dance, shake, flee…)
    novelty: float = 0.0  # impulse: something new was noticed
    investigating: float = 0.0  # level: attending to/approaching a thing
    stimulation: float = 0.0  # level: how much is going on around it
    presence: float = 0.0  # level: family close (seen near, touching, voice close)
    contact: float = 0.0  # impulse: Duck Core contact weight (voice, name, touch, hello…)
    touch_gentle: float = 0.0  # level
    threat: float = 0.0  # impulse: startle/looming/sharp touch strength
    pain: float = 0.0  # impulse: a fall, a hard bump
    positive: float = 0.0  # impulse: a good thing (play answered, praise, gentle touch)
    playing: float = 0.0  # level: doing a play behaviour
    stretching: float = 0.0  # level: stretching
    grooming: float = 0.0  # level: preening (1), shaking or stretching (a part)
    pecking: float = 0.0  # level: pecking, nibbling, dabbling (1), foraging (a part)
    scanning: float = 0.0  # level: looking around, surveying, glancing
    battery: float = 0.8
    charging: bool = False


@dataclass(slots=True)
class Drives:
    energy: float = 0.85
    sleep_pressure: float = 0.25
    alertness: float = 0.8
    curiosity: float = 0.45
    fear: float = 0.0
    social: float = 0.35
    comfort: float = 0.55
    boredom: float = 0.3
    restless: float = 0.45
    hunger: float = 0.0
    play: float = 0.35
    arousal: float = 0.35
    valence: float = 0.15
    stiffness: float = 0.0
    groom: float = 0.25
    peckish: float = 0.3
    vigil: float = 0.1

    def update(self, inp: DriveInputs, p: NeuralParams, stiff_gain: float = 1.0) -> None:
        dt = max(0.0, inp.dt_s)
        awake = not inp.asleep
        self.alertness = circadian(inp.day_minute, p.chrono_shift_min)
        # energy / fatigue
        drain = inp.walking * dt / p.fatigue_walk_s + inp.exertion * dt / (3.0 * p.fatigue_walk_s)
        self.energy = _clip(self.energy - drain)
        rest_rate = 2.2 if inp.asleep else 1.0 if inp.sitting else 0.35
        if inp.walking < 0.05:
            self.energy = _relax(self.energy, 1.0, dt * rest_rate, p.rest_recover_s)
        # sleep pressure (process S)
        if awake:
            self.sleep_pressure = _relax(self.sleep_pressure, 1.0, dt, p.sleep_rise_s)
        else:
            self.sleep_pressure = _relax(self.sleep_pressure, 0.0, dt, p.sleep_fall_s)
        # stiffness: builds asleep, eases awake, a stretch works it out
        if inp.asleep:
            self.stiffness = _relax(self.stiffness, min(1.0, 0.8 * stiff_gain), dt, 240.0)
        else:
            self.stiffness = _relax(self.stiffness, 0.0, dt, 90.0)
            self.stiffness = _clip(self.stiffness - inp.stretching * dt / 2.5)
        # hunger (battery)
        # Duck Core's nest hysteresis: off the nest it gets hungry below ~55 %; on the nest it
        # wants to stay until its stage's leave level (NEST.leave: 80–95 %).
        if inp.charging:
            want = _clip((p.leave_battery - inp.battery) / 0.25)
        else:
            want = _clip((HUNGER_BATTERY_AT - inp.battery) / 0.3)
        self.hunger = _relax(self.hunger, want, dt, 20.0)
        # curiosity: builds in a familiar world, jumps at novelty, consumed by investigating
        if awake:
            self.curiosity = _relax(self.curiosity, 1.0, dt, p.curiosity_tau_s)
        self.curiosity = _clip(
            self.curiosity + (1 - self.curiosity) * 0.5 * inp.novelty * p.novelty_gain
        )
        self.curiosity = _clip(self.curiosity - inp.investigating * dt / 25.0)
        # boredom
        stim = _clip(inp.stimulation + inp.playing + 0.6 * inp.investigating + inp.walking * 0.5)
        if awake:
            self.boredom = _relax(self.boredom, 1.0 - stim, dt, 240.0 if stim < 0.3 else 40.0)
        else:
            self.boredom = _relax(self.boredom, 0.0, dt, 300.0)
        # fear: jumps at threats, decays (faster with family close: social buffering)
        self.fear = _clip(self.fear + (1 - self.fear) * p.fear_gain * (inp.threat + 0.8 * inp.pain))
        tau_fear = (
            p.fear_tau_s * (1.0 - 0.45 * inp.presence) * (0.6 if inp.touch_gentle > 0.2 else 1.0)
        )
        self.fear = _relax(self.fear, 0.0, dt, tau_fear)
        # social need
        if awake and inp.presence < 0.2:
            self.social = _relax(self.social, 1.0, dt, p.social_tau_s)
        self.social = _relax(self.social, SOCIAL_FLOOR, dt * inp.presence, 240.0)
        self.social = max(SOCIAL_FLOOR, self.social - inp.contact)
        # comfort
        target = 0.5 + 0.3 * inp.touch_gentle + (0.12 if inp.sitting else 0.0)
        target += (0.1 if inp.charging else 0.0) + 0.08 * inp.presence - 0.35 * self.fear
        self.comfort = _relax(self.comfort, _clip(target), dt, 45.0)
        self.comfort = _clip(self.comfort - 0.5 * inp.pain)
        # restless: the locomotor urge; walking consumes it, tiredness damps its build-up
        if awake:
            gain = (0.35 + 0.65 * self.alertness) * (0.4 + 0.6 * self.energy)
            self.restless = _relax(self.restless, 1.0, dt * gain, p.restless_tau_s)
        else:
            self.restless = _relax(self.restless, 0.1, dt, 600.0)
        self.restless = _clip(self.restless - inp.walking * dt / p.walk_consume_s)
        # the maintenance needs: build while awake, consumed by doing them
        if awake:
            itch = 1.0 + 1.0 * inp.walking + 0.8 * self.fear  # walking, a scare: sooner
            self.groom = _relax(self.groom, 1.0, dt * itch, p.groom_tau_s)
            self.peckish = _relax(self.peckish, 1.0, dt, p.peck_tau_s)
            self.vigil = _clip(self.vigil + inp.walking * dt / p.vigil_walk_s)
        # (review 2026-09-28: a preen used its itch up in ~14 s and ended at ~2 s; pet:etho has
        # preening bouts of 10 s–5 min, pecking in runs, a look around of several seconds)
        self.groom = _clip(self.groom - inp.grooming * dt / GROOM_USE_S)
        self.peckish = _clip(self.peckish - inp.pecking * dt / PECK_USE_S)
        self.vigil = _clip(self.vigil - inp.scanning * dt / VIGIL_USE_S)
        self.vigil = _relax(self.vigil, 0.0, dt * (1.0 - inp.walking), 60.0)
        # play
        if awake:
            self.play = _relax(
                self.play, self.energy * (0.6 + 0.4 * max(0.0, self.valence)), dt, p.play_tau_s
            )
        self.play = _clip(self.play - inp.playing * dt / 45.0)
        # arousal: the neuromodulator
        base = p.tonic_arousal * (0.35 + 0.65 * self.alertness) * (0.35 if inp.asleep else 1.0)
        base += 0.35 * self.fear + 0.12 * self.restless * self.energy + 0.1 * inp.walking
        self.arousal = _relax(self.arousal, _clip(base), dt, p.arousal_tau_s)
        jolt = 0.55 * inp.threat + 0.25 * inp.novelty + 0.2 * inp.contact * 3 + 0.4 * inp.pain
        self.arousal = _clip(self.arousal + (1 - self.arousal) * _clip(jolt))
        # valence: mood
        rest = 0.15 + 0.25 * (self.comfort - 0.5) - 0.25 * max(0.0, self.social - 0.6)
        rest -= 0.2 * self.hunger + 0.1 * (1 - self.energy) + 0.3 * self.fear
        self.valence = _relax(self.valence, max(-1.0, min(1.0, rest)), dt, 300.0)
        self.valence = max(
            -1.0, min(1.0, self.valence + 0.25 * inp.positive - 0.35 * inp.pain - 0.1 * inp.threat)
        )

    @property
    def fatigue(self) -> float:
        return 1.0 - self.energy

    @property
    def sleepiness(self) -> float:
        """How much it wants to sleep: pressure against the circadian day."""
        return _clip(
            self.sleep_pressure + 0.62 * (1.0 - self.alertness) - 0.4 + 0.25 * (1 - self.energy)
        )

    # ── kept across a restart (duckmind.life.snapshot) ────────────────────────

    def to_state(self) -> dict[str, float]:
        """Every drive's level, plain JSON values."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def load_state(self, state: dict[str, Any], down_s: float = 0.0, tau_s: float = 900.0) -> None:
        """Take saved levels back in. They are minutes-scale quantities, so the time that went by
        while nothing ran relaxes each one toward where a freshly woken duck starts
        (``exp(-down / tau)``): a restart of a minute changes nothing, a night away starts a
        fresh morning."""
        fresh = Drives()
        keep = math.exp(-max(0.0, float(down_s)) / max(1.0, tau_s))
        for f in fields(self):
            if f.name in state:
                base = float(getattr(fresh, f.name))
                setattr(self, f.name, base + (float(state[f.name]) - base) * keep)

    def as_dict(self) -> dict[str, float]:
        return {
            "energy": self.energy,
            "sleep_pressure": self.sleep_pressure,
            "alertness": self.alertness,
            "curiosity": self.curiosity,
            "fear": self.fear,
            "social": self.social,
            "comfort": self.comfort,
            "boredom": self.boredom,
            "restless": self.restless,
            "hunger": self.hunger,
            "play": self.play,
            "groom": self.groom,
            "peckish": self.peckish,
            "vigil": self.vigil,
        }
