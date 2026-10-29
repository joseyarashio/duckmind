"""Habituation and sensitisation, per stimulus class ("sound:clap", "touch:back", …).

Thompson & Spencer's classic properties, as simple dynamics:

* each presentation lowers that class's response gain (the 10th clap gets a weaker startle);
  weaker stimuli habituate faster than strong ones;
* the gain recovers by itself with rest (spontaneous recovery, time constant from wit);
* repeated series habituate faster (a slow "familiarity" trace speeds the drop);
* a strong stimulus of *another* class partly restores the others (dishabituation);
* a frightening or painful event sensitises: for minutes every response in that sense (and a
  little in all) is stronger, the startle threshold lower — the duck is jumpier for a while.

Levels are kept as (value, time) pairs and relaxed lazily on read: exact for any tick rate.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

from duckmind.neuro.contract import STIMULUS_CLASSES
from duckmind.neuro.genome import NeuralParams

FAMILIARITY_TAU_S = 3600.0
# What the startle floor asks (2026-10-09): "has this duck got used to LOUD sudden sounds" is its own level, kept apart from the
# classes' (``LOUD_KEY``). The quiet knocks of a thing the duck pushes along (64 dB at the head, raw 0.6-0.8; the ear
# names a third of them "bang") took the level of the bang class down to 0.2, so the first page bang of a fresh boot was
# under the floor's bar and no startle (live, 4 of 8 fresh boots with the 20 ms servo delay; 0 of 8 without it). The
# loud level is lowered only by sharp sounds of at least LOUD_RAW (a clap reads 0.89 at 80 dB, a bang 1.0), with the
# class dynamics and the same recovery, and starts fresh (1.0).
LOUD_KEY = "sound:loud"
LOUD_RAW = 0.85
DISHABITUATE_AT = 0.7  # a new stimulus at least this strong refreshes the other kinds
SENS_MAX = 0.8
# × the per-repeat rate: a startle habituates by h ≈ 0.3–0.5 per identical repeat (pet:etho B1)
STARTLE_HAB = {"startle": 3.0, "withdraw": 1.4, "freeze": 1.4}
SENS_GLOBAL_MAX = 0.35
# Stimulus generalisation: habituating to one class habituates its group half as much.
GROUPS: dict[str, tuple[str, ...]] = {
    "sharp_sound": ("sound:clap", "sound:bang", "sound:noise", "sound:door", "sound:footsteps"),
    # (its own name is not habituated by the family's talk: a conditioned cue — review 2026-09-28)
    "voice": ("sound:voice", "sound:hello"),
    "media": ("sound:music", "sound:tv"),
    "touch": tuple(c for c in STIMULUS_CLASSES if c.startswith("touch:")),
    "sight": ("visual:motion", "visual:hand", "visual:person", "visual:thing"),
    # something coming straight at the face is not "a thing in view": the things it keeps seeing
    # in its home (an orienting look each, 0.4 strong) habituated it to 0.2 and the bold explorer
    # looked at a hand rushing at its beak (live check 2026-10-05, 6 of 6)
    "looming": ("visual:looming",),
    "light": ("light:on", "light:off"),
}
GROUP = {cls: group for group, members in GROUPS.items() for cls in members}
GROUP_MEMBERS = GROUPS


class Habituation:
    __slots__ = ("_p", "_h", "_f", "_s", "_g", "_l")

    def __init__(self, p: NeuralParams) -> None:
        self._p = p
        self._h: dict[str, tuple[float, float]] = {}  # class → (gain, t_s)
        self._f: dict[str, tuple[float, float]] = {}  # class → (familiarity, t_s)
        self._s: dict[str, tuple[float, float]] = {}  # modality → (sensitisation, t_s)
        self._g: tuple[float, float] = (0.0, 0.0)  # global sensitisation
        self._l: tuple[float, float] = (1.0, 0.0)  # the loud-sound level (LOUD_KEY): (gain, t_s)

    # ── reads ────────────────────────────────────────────────────────────────

    def level(self, cls: str, t_s: float) -> float:
        """The habituation gain of ``cls`` now (1 = fresh, → floor = habituated). ``LOUD_KEY``: the level of
        the loud sudden sounds alone (what the startle floor asks)."""
        if cls == LOUD_KEY:
            h0, t0 = self._l
            return 1.0 - (1.0 - h0) * math.exp(-max(0.0, t_s - t0) / self._p.hab_recover_tau_s)
        h0, t0 = self._h.get(cls, (1.0, t_s))
        return 1.0 - (1.0 - h0) * math.exp(-max(0.0, t_s - t0) / self._p.hab_recover_tau_s)

    def sensitisation(self, cls: str, t_s: float) -> float:
        modality = cls.split(":", 1)[0]
        s0, t0 = self._s.get(modality, (0.0, t_s))
        g0, gt = self._g
        decay = math.exp(-max(0.0, t_s - t0) / self._p.sens_tau_s)
        g_decay = math.exp(-max(0.0, t_s - gt) / self._p.sens_tau_s)
        return s0 * decay + g0 * g_decay

    def gain(self, cls: str, t_s: float) -> float:
        """What a response to ``cls`` is multiplied by now: habituation × (1 + sensitisation)."""
        return self.level(cls, t_s) * (1.0 + self.sensitisation(cls, t_s))

    def familiarity(self, cls: str, t_s: float) -> float:
        f0, t0 = self._f.get(cls, (0.0, t_s))
        return f0 * math.exp(-max(0.0, t_s - t0) / FAMILIARITY_TAU_S)

    # ── writes ───────────────────────────────────────────────────────────────

    def present(
        self,
        cls: str,
        strength: float,
        t_s: float,
        reflex: str | None = None,
        resist: float = 0.0,
    ) -> float:
        """A stimulus of ``cls`` arrived with ``strength`` (0..1). Returns the gain its
        response gets (the state *before* this presentation), then habituates the class — and,
        half as much, the classes it generalises to (a clap and a bang are both sharp sounds).
        ``reflex``: what it set off; a startle habituates faster than an orienting turn
        (ethology table: h ≈ 0.3–0.5 per identical repeat for startle, 0.15–0.3 for orienting).
        ``resist`` (0..1): how much the stimulus has come to mean (a learned cue's value, its own
        name) — such a cue habituates more slowly."""
        p = self._p
        strength = max(0.0, min(1.0, strength))
        gain = self.gain(cls, t_s)
        h_before = self.level(cls, t_s)
        fam = self.familiarity(cls, t_s)
        rate = p.hab_rate * (1.35 - 0.6 * strength) * (1.0 + 0.8 * fam)
        rate = min(0.8, rate * STARTLE_HAB.get(reflex or "", 1.0))
        # a cue that has come to mean something (its name → someone nice) resists habituation
        rate *= 1.0 - 0.7 * max(0.0, min(1.0, resist))
        self._h[cls] = (max(p.hab_floor, h_before - rate * (h_before - p.hab_floor)), t_s)
        self._f[cls] = (min(1.0, fam + 0.08), t_s)
        group = GROUP.get(cls)
        if group == "sharp_sound" and strength >= LOUD_RAW:
            loud = self.level(LOUD_KEY, t_s)
            self._l = (max(p.hab_floor, loud - rate * (loud - p.hab_floor)), t_s)
        for other in GROUP_MEMBERS.get(group or "", ()):
            if other != cls:
                h_other = self.level(other, t_s)
                drop = 0.5 * rate * (h_other - p.hab_floor)
                self._h[other] = (max(p.hab_floor, h_other - drop), t_s)
        if strength >= DISHABITUATE_AT and h_before >= 0.7:
            # something new and strong: the other kinds of things get interesting again
            for other in list(self._h):
                if other != cls and GROUP.get(other) != group:
                    h_other = self.level(other, t_s)
                    self._h[other] = (h_other + (1.0 - h_other) * 0.3 * strength, t_s)
        return gain

    def sensitise(self, cls: str, amount: float, t_s: float) -> None:
        """An aversive event in ``cls``'s sense: that sense gets jumpier, everything a little.
        Callers scale ``amount`` by how new the stimulus was: a familiar harmless clap barely
        sensitises, so habituation wins over repetitions."""
        p = self._p
        amount = max(0.0, amount) * p.sens_gain
        modality = cls.split(":", 1)[0]
        s_now = self.sensitisation(modality + ":", t_s) - self._global(t_s)
        self._s[modality] = (min(SENS_MAX, max(0.0, s_now) + amount), t_s)
        self._g = (min(SENS_GLOBAL_MAX, self._global(t_s) + 0.3 * amount), t_s)

    def _global(self, t_s: float) -> float:
        g0, gt = self._g
        return g0 * math.exp(-max(0.0, t_s - gt) / self._p.sens_tau_s)

    # ── views for the reflex arcs (reflex.step's ``hab`` / ``sens``) ─────────────

    def level_view(self, t_s: float) -> _LevelView:
        """A read-only mapping: stimulus class → habituation level now (1 = fresh)."""
        return _LevelView(self, t_s)

    def sens_view(self, t_s: float) -> _SensView:
        """A read-only mapping: stimulus class → extra gain from sensitisation now (≥ 0)."""
        return _SensView(self, t_s)

    # ── the slow state, kept across a restart (duckmind.life.snapshot) ───────────

    def to_state(self, t_s: float) -> dict[str, Any]:
        """Every level as (value, age at ``t_s``), plain JSON values."""

        def pair(entry: tuple[float, float]) -> list[float]:
            return [entry[0], round(max(0.0, t_s - entry[1]), 3)]

        return {
            "h": {cls: pair(e) for cls, e in sorted(self._h.items())},
            "f": {cls: pair(e) for cls, e in sorted(self._f.items())},
            "s": {m: pair(e) for m, e in sorted(self._s.items())},
            "g": pair(self._g),
            "l": pair(self._l),
        }

    def load_state(self, state: Mapping[str, Any], t_now: float, down_s: float = 0.0) -> None:
        """Take a saved ``to_state`` back in, ``down_s`` seconds older: the levels relax by
        themselves (they are kept as (value, time) pairs and relaxed on read), so time that went
        by while nothing ran is time they had to recover."""
        down = max(0.0, float(down_s))

        def entry(pair: list[float]) -> tuple[float, float]:
            return float(pair[0]), t_now - float(pair[1]) - down

        self._h = {str(k): entry(v) for k, v in dict(state.get("h", {})).items()}
        self._f = {str(k): entry(v) for k, v in dict(state.get("f", {})).items()}
        self._s = {str(k): entry(v) for k, v in dict(state.get("s", {})).items()}
        self._g = entry(state["g"]) if "g" in state else (0.0, 0.0)
        self._l = entry(state["l"]) if "l" in state else (1.0, 0.0)

    # ── telemetry ────────────────────────────────────────────────────────────

    def levels(self, t_s: float) -> dict[str, float]:
        """Every class that has been met, with its habituation gain now."""
        return {cls: round(self.level(cls, t_s), 3) for cls in STIMULUS_CLASSES if cls in self._h}

    def sensitisations(self, t_s: float) -> dict[str, float]:
        out = {m: round(self.sensitisation(m + ":", t_s), 3) for m in sorted(self._s)}
        out["all"] = round(self._global(t_s), 3)
        return out


class _LevelView(Mapping[str, float]):
    """Lazy: computes a class's level only when the reflex arcs ask for it."""

    __slots__ = ("_hab", "_t")

    def __init__(self, hab: Habituation, t_s: float) -> None:
        self._hab = hab
        self._t = t_s

    def __getitem__(self, cls: str) -> float:
        return self._hab.level(cls, self._t)

    def __iter__(self) -> Iterator[str]:
        return iter(self._hab._h)

    def __len__(self) -> int:
        return len(self._hab._h)


class _SensView(Mapping[str, float]):
    """Lazy: the sensitisation on a class (its sense's plus the general level)."""

    __slots__ = ("_hab", "_t")

    def __init__(self, hab: Habituation, t_s: float) -> None:
        self._hab = hab
        self._t = t_s

    def __getitem__(self, cls: str) -> float:
        return self._hab.sensitisation(cls, self._t)

    def __iter__(self) -> Iterator[str]:
        return iter(STIMULUS_CLASSES)

    def __len__(self) -> int:
        return len(STIMULUS_CLASSES)
