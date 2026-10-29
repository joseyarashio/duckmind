"""Tricks the duck is taught: a proficiency per trick, kept in the duck's own mind (phase 2 「學會聽話」).

Five tricks (:data:`~duckmind.neuro.command_words.TRICKS`): ``lie_down``, ``spin``, ``bow``, ``shake``,
``flip``. Each has a **proficiency** 0..1 in this duck's :class:`~duckmind.neuro.learn.Learner`
(``learner.tricks``), saved with its mind (``life.snapshot`` version 2). It starts at 0: a duck that was
never taught a trick only now and then does it when told (:data:`OBEY_BASE`, a lucky guess).

How it moves (all numbers are tuned to the phase-2 targets and said so in
``docs/design/neural-duck/LEARN-TRICKS-2026-10-25.md``):

* **praise** (the owner's 「乖」, a pat or a treat, after the duck did the trick) raises it by a step that
  gets smaller the more it was praised that growth day (:data:`STEPS`), and by no more than
  :data:`CAP_PER_GD` in all that day — so one afternoon cannot fill the bar;
* **shaping**: the duck did the trick's act by itself and the owner praised within three seconds: the
  same step, :data:`SHAPE_SHARE` of it;
* **scold** (「不對」) lowers it by :data:`SCOLD_SHARE` of a step (never below :data:`FLOOR` once it has
  been learned) and the try does not count as practice;
* **forgetting**: a growth day that passes with no practice pulls it toward :data:`FLOOR` with time
  constant :data:`DECAY_TAU_GD` days — rusty, never back to nothing. A trick the duck never practised stays at 0.

Whether the duck obeys a command for a trick is :func:`obey_probability`: the proficiency is the main
term; being scared or sleepy lowers it; how close it feels to the speaker and its temperament only nudge
the odds (the exponent), so a trick it knows well is done whoever asks. The caller draws with the brain's
own seeded stream.

Pure and clock-free: the growth day is given, never read. Deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from duckmind.neuro.command_words import TRICKS

# ── how proficiency moves ─────────────────────────────────────────────────────────────────
STEPS = (0.060, 0.040, 0.030, 0.020, 0.015, 0.010)  # the n-th praise of a growth day (the last repeats)
CAP_PER_GD = 0.120  # most a trick can rise in one growth day
SHAPE_SHARE = 0.5  # praise for what it did by itself: this share of a step
SCOLD_SHARE = 0.5  # 「不對」: this share of a step down
FLOOR = 0.15  # a learned trick fades toward this, never to nothing
DECAY_TAU_GD = 12.0  # growth days
LEARNED_AT = 0.8  # the page calls a trick learned from here

# ── whether it does the trick when told ───────────────────────────────────────────────────
OBEY_BASE = 0.15  # an untaught trick: the chance it happens to do it
OBEY_FULL_AT = 0.7  # from this proficiency, mood is all that is left
OBEY_CURVE = 1.2
FEAR_FROM, FEAR_SPAN, FEAR_BITE = 0.15, 0.35, 0.9  # a scared duck: at most this much less likely …
TIRED_FROM, TIRED_SPAN, TIRED_BITE = 0.45, 0.45, 0.8  # … and a sleepy one
TIRED_GATE = 0.9  # sleepiness from here: it will not (``too_tired``)
BOND_NUDGE = 0.2  # closeness 0..1 turns the odds' exponent by ±this/2
PERSONALITY_NUDGE = {"snuggler": 1.10, "explorer": 0.90, "dreamer": 0.95}  # the odds' exponent
NUDGE_RANGE = (0.8, 1.3)

# shaping and teaching windows (seconds): the owner's word counts for what the duck just did
SHAPE_WINDOW_S = 3.0  # Duck Core's PRAISE_WINDOW: praise within this of an act it did by itself
TEACH_WINDOW_S = 10.0  # praise within this of a trick it did because it was told


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(slots=True)
class TrickSkill:
    """One trick of one duck."""

    proficiency: float = 0.0
    praised_today: int = 0  # praises counted this growth day (picks the step)
    scolded_today: int = 0
    gained_today: float = 0.0  # how far it rose today (the cap)
    tries_today: int = 0  # counted tries today (a praised try)
    total: int = 0  # counted tries ever
    shaped: int = 0  # … of which the duck did the act by itself

    def to_state(self) -> dict[str, Any]:
        return {
            "p": round(self.proficiency, 6),
            "praised": self.praised_today,
            "scolded": self.scolded_today,
            "gained": round(self.gained_today, 6),
            "tries": self.tries_today,
            "total": self.total,
            "shaped": self.shaped,
        }

    @classmethod
    def from_state(cls, raw: Mapping[str, Any]) -> TrickSkill:
        return cls(
            proficiency=_clip(float(raw.get("p", 0.0))),
            praised_today=max(0, int(raw.get("praised", 0))),
            scolded_today=max(0, int(raw.get("scolded", 0))),
            gained_today=max(0.0, float(raw.get("gained", 0.0))),
            tries_today=max(0, int(raw.get("tries", 0))),
            total=max(0, int(raw.get("total", 0))),
            shaped=max(0, int(raw.get("shaped", 0))),
        )


@dataclass(frozen=True, slots=True)
class Feedback:
    """What one word from the owner did to one trick."""

    kind: str  # praise | scold
    trick: str
    delta: float  # proficiency change (signed)
    proficiency: float
    capped: bool  # a praise that the day's cap cut short (or filled the bar)
    shaped: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "trick": self.trick,
            "step": round(self.delta, 4),
            "proficiency": round(self.proficiency, 4),
            "capped": self.capped,
            "shaped": self.shaped,
        }


class TrickBook:
    """The proficiencies of one duck's five tricks and the growth day they were last settled to."""

    __slots__ = ("skills", "day")

    def __init__(self) -> None:
        self.skills: dict[str, TrickSkill] = {t: TrickSkill() for t in TRICKS}
        self.day: int | None = None

    # -- the growth day ---------------------------------------------------------------------
    def settle(self, gd: int) -> None:
        """Bring the book to growth day ``gd``: each day that passed since the last one pulls a practised
        trick toward :data:`FLOOR`, and a new day starts the daily counters afresh. A day not later than
        the last one changes nothing (a clock that went back never forgives a cap)."""
        gd = int(gd)
        if self.day is None:
            self.day = gd
            return
        if gd <= self.day:
            return
        days = gd - self.day
        keep = math.exp(-days / DECAY_TAU_GD)
        for skill in self.skills.values():
            if skill.total > 0 and skill.proficiency > FLOOR:
                skill.proficiency = FLOOR + (skill.proficiency - FLOOR) * keep
            skill.praised_today = skill.scolded_today = skill.tries_today = 0
            skill.gained_today = 0.0
        self.day = gd

    def proficiency(self, trick: str) -> float:
        return self.skills[trick].proficiency

    # -- the owner's word -------------------------------------------------------------------
    def praise(self, trick: str, gd: int, *, shaped: bool = False) -> Feedback:
        self.settle(gd)
        s = self.skills[trick]
        step = STEPS[min(s.praised_today, len(STEPS) - 1)] * (SHAPE_SHARE if shaped else 1.0)
        room = max(0.0, min(CAP_PER_GD - s.gained_today, 1.0 - s.proficiency))
        gain = min(step, room)
        s.proficiency = _clip(s.proficiency + gain)
        s.gained_today += gain
        s.praised_today += 1
        s.tries_today += 1
        s.total += 1
        if shaped:
            s.shaped += 1
        return Feedback("praise", trick, gain, s.proficiency, gain < step - 1e-9, shaped)

    def scold(self, trick: str, gd: int) -> Feedback:
        self.settle(gd)
        s = self.skills[trick]
        step = SCOLD_SHARE * STEPS[min(s.scolded_today, len(STEPS) - 1)]
        s.scolded_today += 1  # (the try does not count: tries and praises stay)
        before = s.proficiency
        if before > 0.0:
            s.proficiency = max(before - step, min(before, FLOOR))
        return Feedback("scold", trick, s.proficiency - before, s.proficiency, False, False)

    # -- what the page and the file keep ------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {"day": self.day, "skills": {t: s.to_state() for t, s in self.skills.items()}}

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> TrickBook:
        book = cls()
        if not state:
            return book
        day = state.get("day")
        book.day = None if day is None else int(day)
        for trick, raw in dict(state.get("skills") or {}).items():
            if trick in book.skills and isinstance(raw, Mapping):
                book.skills[trick] = TrickSkill.from_state(raw)
        return book

    def view(self, gd: int, locked: Collection[str] = ()) -> list[dict[str, Any]]:
        """The trick book as the page shows it: one row per trick (ids and numbers, never words)."""
        self.settle(gd)
        rows = []
        for trick in TRICKS:
            s = self.skills[trick]
            if trick in locked:
                state = "locked"
            elif s.total == 0 and s.proficiency <= 0.0:
                state = "none"
            elif s.proficiency >= LEARNED_AT:
                state = "learned"
            else:
                state = "half"
            rows.append(
                {
                    "id": trick,
                    "proficiency": round(s.proficiency, 3),
                    "proficiency_pct": int(round(100.0 * s.proficiency)),
                    "state": state,
                    "unlocked": trick not in locked,
                    "tries_today": s.tries_today,
                    "gained_today": round(s.gained_today, 3),
                    "capped_today": s.gained_today >= CAP_PER_GD - 1e-9,
                    "total": s.total,
                    "shaped": s.shaped,
                    "suggest": False,
                }
            )
        open_rows = [r for r in rows if r["unlocked"]]
        if open_rows:
            min(open_rows, key=lambda r: (r["proficiency"], TRICKS.index(r["id"])))["suggest"] = True
        return rows


# ── whether it obeys ──────────────────────────────────────────────────────────────────────


def mood_factor(fear: float, sleepiness: float) -> float:
    """1 for a calm, rested duck; less when scared or sleepy (graded: the hard gates — afraid past
    ``select.CLAIM_FEAR_MAX``, asleep, held — are the command brain's refusals before this)."""
    scared = 1.0 - FEAR_BITE * _clip((fear - FEAR_FROM) / FEAR_SPAN)
    sleepy = 1.0 - TIRED_BITE * _clip((sleepiness - TIRED_FROM) / TIRED_SPAN)
    return scared * sleepy


def nudge(bond: float, primary: str) -> float:
    """The odds' exponent from closeness and temperament (1 = neutral). Only moves chances that are
    in doubt: a certain trick stays certain (:func:`obey_probability`)."""
    k = (1.0 - BOND_NUDGE / 2.0 + BOND_NUDGE * _clip(bond)) * PERSONALITY_NUDGE.get(primary, 1.0)
    return _clip(k, *NUDGE_RANGE)


def base_probability(proficiency: float) -> float:
    """The chance that rests on proficiency alone: :data:`OBEY_BASE` untaught, 1 from :data:`OBEY_FULL_AT`."""
    x = _clip(proficiency / OBEY_FULL_AT)
    return OBEY_BASE + (1.0 - OBEY_BASE) * x**OBEY_CURVE


def obey_probability(
    proficiency: float, *, fear: float = 0.0, sleepiness: float = 0.0, bond: float = 0.0, primary: str = ""
) -> float:
    """The chance that it does a trick it was told, 0..1 (module doc). ``base ** (1 / nudge) × mood``."""
    p = base_probability(proficiency) ** (1.0 / nudge(bond, primary)) * mood_factor(fear, sleepiness)
    return _clip(p)


__all__ = [
    "CAP_PER_GD",
    "DECAY_TAU_GD",
    "FLOOR",
    "Feedback",
    "LEARNED_AT",
    "OBEY_BASE",
    "SHAPE_SHARE",
    "SHAPE_WINDOW_S",
    "STEPS",
    "TEACH_WINDOW_S",
    "TIRED_GATE",
    "TRICKS",
    "TrickBook",
    "TrickSkill",
    "base_probability",
    "mood_factor",
    "nudge",
    "obey_probability",
]
