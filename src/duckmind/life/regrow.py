"""Growth G3: the duck grows in its sleep (docs/design/GROWTH-G3-2026-10-11.md).

Duck Core says who the duck has become (its growth day, stage, branch, what the family taught it);
the neural duck's body parameters are a genome built from that. Once a growth day has passed, the
genome is built again from Duck Core's state of that day and put into the sleeping duck's brain
(:meth:`duckmind.neuro.integration.IntegratedBrain.regrow`): **only while it sleeps, never awake**.

This module is the pure part, shared by the live server (:class:`~duckmind.simweb.life_host.LifeHost`)
and the headless acceptance harness (``tools/growth/regrow_harness.py``):

* :class:`Basis` — the few numbers a genome is built from that change with growth (the growth day,
  the branch the body follows, the bond and the preferences). Everything else (temperament, nature,
  quirks, pace, stage days) is fixed at hatching. A basis is written into the neural snapshot, so a
  restart builds *exactly* the genome the duck had: there is no jump between last night's regrow and
  the boot of today.
* :class:`Static` — the hatching part of Duck Core's state, copied once.
* :func:`calm` — whether the sleeping duck can be swapped without anyone noticing.
* :class:`Regrower` — holds the offer (the basis to grow into) and swaps it in when calm.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from duckmind.neuro import select as S
from duckmind.neuro.genome import Care, Genome

# The version of the mapping from a basis to a genome (genome.py's Care gains and the rest): a
# stamp of another revision is not trusted at boot, the genome is built from Duck Core's state again.
GROWTH_REV = "growth-g3-1"

# A swap needs the duck to have slept this long without a reflex, a body event or a skill (brain
# seconds): a nap that just began, or one a noise is about to end, is not the moment.
CALM_HOLD_S = 20.0
# a startle, a touch, a lift this recently: not now
RECENT_S = 30.0
# body events that mean something is happening to the body
BUSY_EVENTS = frozenset(
    {"fell", "lifted", "set_down", "righted", "righting_failed", "bump", "bumped", "dropped"}
)


@dataclass(frozen=True)
class Basis:
    """What the genome of a growth day is built from (see the module doc)."""

    gd: int
    branch: str | None
    care: Care

    def to_json(self) -> dict[str, Any]:
        return {
            "rev": GROWTH_REV,
            "gd": self.gd,
            "branch": self.branch,
            "care": self.care.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Basis | None:
        """The basis written in a snapshot, or None when it is not one this build trusts."""
        if not data or data.get("rev") != GROWTH_REV:
            return None
        try:
            branch = data.get("branch")
            return cls(
                gd=int(data["gd"]),
                branch=None if branch is None else str(branch),
                care=Care.from_json(data.get("care")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def same_as(self, other: Basis) -> bool:
        return self.gd == other.gd and self.branch == other.branch and self.care == other.care


@dataclass(frozen=True)
class Static:
    """The part of Duck Core's state that hatching fixed. ``view(basis)`` is a stand-in
    ``CoreState`` for :meth:`Genome.from_core_state`."""

    seed: int
    temperament: dict[str, int]
    nature: dict[str, int]
    branch_detail: dict[str, Any]
    quirks: dict[str, str]
    stage_days: dict[str, int]

    @classmethod
    def of(cls, state: Any) -> Static:
        return cls(
            seed=int(state.seed),
            temperament={k: int(v) for k, v in state.temperament.items()},
            nature={k: int(v) for k, v in state.nature.items()},
            branch_detail={k: v for k, v in state.branch_detail.items() if k == "primary"},
            quirks={k: str(v) for k, v in state.quirks.items()},
            stage_days={k: int(v) for k, v in state.stage_days.items()},
        )

    def genome(self, basis: Basis) -> Genome:
        """The genome (care not yet applied: the integrated brain does that last) of ``basis``."""
        view = SimpleNamespace(
            seed=self.seed,
            temperament=self.temperament,
            nature=self.nature,
            branch_detail=self.branch_detail,
            quirks=self.quirks,
            stage_days=self.stage_days,
            gd=basis.gd,
            branch=basis.branch,
        )
        return Genome.from_core_state(view, branch=basis.branch, gd=basis.gd, care=basis.care)  # type: ignore[arg-type]


def basis_of(state: Any, *, branch: str | None) -> Basis:
    """The basis of Duck Core's state now. ``branch``: the branch the body follows (the state's
    own, or the one the record pins: 蛋黃 follows its nature)."""
    return Basis(gd=int(state.gd), branch=branch, care=Care.from_state(state))


def genome_numbers(genome: Any) -> dict[str, Any]:
    """The genome a body runs, in plain numbers (for ``/api/life`` and the growth page)."""
    return {
        "gd": genome.gd,
        "stage": genome.stage,
        "branch": genome.branch,
        "bond": genome.care.bond,
        "prefs": len(genome.care.prefs),
        "unlocked": len(genome.unlocked),
        "leash_m": round(genome.p.leash_m, 3),
        "social_tau_s": round(genome.p.social_tau_s, 1),
        "greet_reflex": round(genome.p.greet_reflex, 3),
        "hatchling_wobble": round(genome.p.hatchling_wobble, 3),
    }


def calm(brain: Any, frame: Any, t_ms: int, cmd: Any | None = None) -> str | None:
    """None when the duck is asleep and nothing is happening to its body or its mind's reflexes;
    otherwise a plain word for what is in the way (for the log and the status)."""
    if str(brain.behaviour) not in S.SLEEP:
        return "awake"
    body = frame.body
    if getattr(body, "fallen", False) or getattr(body, "limp", False):
        return "fallen"
    if getattr(body, "posture", "sit") not in ("sit", "lie"):
        return "posture"
    if getattr(body, "active_skill", None):
        return "skill"
    if BUSY_EVENTS & set(frame.events):
        return "event"
    if cmd is not None and (getattr(cmd, "reflex", None) or getattr(cmd, "docking", False)):
        return "reflex"
    if getattr(getattr(brain, "_reflex", None), "active", None) is not None:
        return "reflex"
    if getattr(brain, "_escape", None) is not None or getattr(brain, "_back", None) is not None:
        return "escape"
    if t_ms < int(getattr(brain, "_perform_until_ms", 0)):
        return "emotion"
    t0 = getattr(brain, "_t0", None)
    brain_s = (t_ms - (t_ms if t0 is None else t0)) / 1000.0  # (the brain's own seconds)
    for name in ("_startled_s", "_strong_scare_s", "_touch_s"):
        when = getattr(brain, name, None)
        if when is not None and brain_s - float(when) < RECENT_S:
            return "recent"
    # (the play act's wish of this tick: IDLE unless a play act holds the body; PlayMind.out keeps
    # the last act's one-shot after it has ended, which says nothing about now)
    if getattr(getattr(brain, "play_out", None), "skill", None):
        return "play"
    return None


@dataclass
class Report:
    """One regrow, as it happened (for the log and /api/life)."""

    at: str  # the duck's calendar
    why: str  # "night" | "boot" | "catch-up"
    change: dict[str, Any]
    waited_s: float  # how long it was due before the duck slept calmly
    basis: dict[str, Any]

    def view(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "why": self.why,
            "waited_s": round(self.waited_s, 1),
            "basis": self.basis,
            **self.change,
        }


class Regrower:
    """The offer of a new basis and the moment it is put in. ``applied``: the basis the brain on
    screen has now. :meth:`offer` is called from outside the body loop; :meth:`tick` from the
    brain's tick, before the brain steps (so a swap is always between two ticks)."""

    def __init__(self, static: Static, applied: Basis) -> None:
        self.static = static
        self.applied = applied
        self._lock = threading.Lock()  # (the watch offers, the brain's tick applies)
        self._offer: Basis | None = None
        self._due_s: float | None = None  # brain seconds when the offer was first seen
        self._calm_since: float | None = None
        self.refused: str | None = None  # why it has not happened yet (the last word)
        self.regrows = 0
        self.last: Report | None = None

    # ── from outside the body loop ───────────────────────────────────────────────────────

    def offer(self, basis: Basis) -> bool:
        """The basis Duck Core's state asks for now. True when it differs from what the brain has
        (a regrow is due)."""
        with self._lock:
            if basis.same_as(self.applied):
                self._offer = None
                return False
            if self._offer is None or not self._offer.same_as(basis):
                self._offer = basis
            return True

    @property
    def due(self) -> bool:
        return self._offer is not None

    @property
    def offered(self) -> Basis | None:
        return self._offer

    # ── in the brain's tick ──────────────────────────────────────────────────────────────

    def tick(
        self,
        brain: Any,
        frame: Any,
        t_ms: int,
        *,
        cmd: Any | None = None,
        also_calm: str | None = None,
        calendar: str = "",
        why: str = "night",
    ) -> Report | None:
        """(The cheap path first: nothing is offered, nothing happens.) The swap, once the duck has
        slept calmly for :data:`CALM_HOLD_S` brain seconds; None otherwise. ``also_calm``: a reason
        the driver gives for not now (a skill of the body link, a motion trial)."""
        offer = self._offer
        if offer is None:
            return None
        now_s = t_ms / 1000.0
        if self._due_s is None:
            self._due_s = now_s
        reason = also_calm or calm(brain, frame, t_ms, cmd)
        if reason is not None:
            self._calm_since = None
            self.refused = reason
            return None
        if self._calm_since is None:
            self._calm_since = now_s
        if now_s - self._calm_since < CALM_HOLD_S:
            self.refused = "settling"
            return None
        with self._lock:
            if self._offer is None or self._offer.same_as(self.applied):
                self._offer = None
                return None  # (the watch withdrew it)
            return self._apply(brain, self._offer, now_s=now_s, calendar=calendar, why=why)

    def apply(
        self, brain: Any, offer: Basis, *, now_s: float, calendar: str = "", why: str = "night"
    ) -> Report:
        """Put ``offer`` into the sleeping ``brain`` (raises ``RegrowRefused`` awake)."""
        with self._lock:
            return self._apply(brain, offer, now_s=now_s, calendar=calendar, why=why)

    def _apply(self, brain: Any, offer: Basis, *, now_s: float, calendar: str, why: str) -> Report:
        genome = self.static.genome(offer)
        change = brain.regrow(genome)
        waited = 0.0 if self._due_s is None else max(0.0, now_s - self._due_s)
        report = Report(at=calendar, why=why, change=change, waited_s=waited, basis=offer.to_json())
        self.applied = offer
        self._offer = None
        self._due_s = self._calm_since = None
        self.refused = None
        self.regrows += 1
        self.last = report
        return report
