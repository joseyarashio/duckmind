"""Curiosity (curiosity spec §7): the sockets Duck Core calls, inert until curiosity P2.

Phase 4 (phase-4 design §9) fits the hooks once, so ``engine.py`` and ``excursion.py`` are
changed only once; what curiosity actually does (``content.CURIO``, ``THING_CLASSES``,
``CURIO_CAPS``, its moments) arrives with curiosity P2, because that content changes the
content table hash and with it every draw. Until then every socket answers with its inert
value and does nothing else:

* no draw (never ``core._rand``: the counters of every other purpose stay where v4 has them),
* no event (never ``core._emit``),
* no state change.

So a duck with the sockets and nothing in its home lives exactly as it would without them
(phase-4 test T8). P2 puts its work after :func:`has_things` (or :func:`active` for a
programme under way), so "no things, no change" stays true by construction.

Who calls what (phase-4 design §9):

=====================  ======================================================================
hook                   caller
=====================  ======================================================================
``minute``             the engine's minute, after ``nest.errand_minute``, before ``excursion``
``at_pause``           ``excursion.next_leg``: the look round between two legs of an outing
``attractions``        ``excursion._choices``: extra pull of the cells where things are
``wants_toy``          ``excursion.pick_reason``: may set off for a toy (reason ``toy``)
``zone_block``         the engine's ``_blocker``: a family zone it may not walk into
``on_fall``            the engine's ``_fall_outcome``: a fall counted on the thing it was at
``on_sense``           ``on_percept("sense.thing")``
``on_result``          ``on_percept("body.curio_step")``
``on_guard``           ``on_percept("body.guard")``
``on_timeout``         an act deadline for a ``duck.curio`` step (acts → ``duck.curio timeout``)
``on_home``            the App inputs ``thing_mark``, ``home_zone``, ``family_play``,
                       ``answer_ask`` (after their input record)
``holds_body``         the engine's ``_motion_block`` (reason ``curio``)
``next_due``           the engine's ``next_due``: when curiosity next has to look again
=====================  ======================================================================

Integers only; draws only through the core's counters (from P2 on).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore
    from duckmind.provenance import Event

# Preference keys curiosity learns under (the engine's ``_learn_pref`` accepts them).
PREF_PREFIXES = ("thing:", "curio:")
# The reason an outing sets off for a thing it likes (``excursion.pick_reason``).
TOY_REASON = "toy"
NO_PULL: Mapping[int, int] = {}


def has_things(core: DuckCore) -> bool:
    """Something in its home it knows of (``CoreState.things``). Phase 4 has no way in for
    one (``on_sense`` and ``on_home`` keep nothing), so this is always False there."""
    return bool(core.s.things)


def active(core: DuckCore) -> bool:
    """A curiosity programme is under way (``CoreState.curio``)."""
    return core.s.curio is not None


# ---------------------------------------------------------------------------- the minute


def minute(core: DuckCore) -> bool:
    """The duck's minute (curio-engage-v1 from a seat, a programme's next phase). True when
    this minute is curiosity's (nothing else is decided); False with nothing to be curious
    about, without a draw."""
    del core
    return False  # curiosity P2 goes after ``if not has_things(core): return False``


def next_due(core: DuckCore) -> datetime | None:
    """The next moment curiosity has to look again by itself (None: only its inputs wake it)."""
    del core
    return None


def holds_body(core: DuckCore) -> bool:
    """A programme holds the body (the engine's ``_motion_block`` says ``curio``)."""
    return active(core)


# ---------------------------------------------------------------------------- outings


def at_pause(core: DuckCore) -> bool:
    """The look round between two legs of an outing: True when curiosity takes the pause
    over (it engaged a thing seen close by: its programme now holds the body and the outing
    goes on after it). False, without a draw, when there is nothing to be curious about."""
    del core
    return False  # curiosity P2 (curio-engage-v1 at a pause), after ``has_things``


def attractions(core: DuckCore) -> Mapping[int, int]:
    """The extra pull of each cell of the room grid where it remembers a thing it is keen on
    (``interest × exploration // 2``, curiosity spec §7.4), for ``excursion._choices``: an
    empty mapping (no pull anywhere) with nothing to be curious about."""
    del core
    return NO_PULL  # curiosity P2, after ``has_things``


def attraction(core: DuckCore, cell: int) -> int:
    """The extra pull of one cell (see :func:`attractions`)."""
    return int(attractions(core).get(cell, 0))


def wants_toy(core: DuckCore) -> bool:
    """Whether an outing sets off for a toy (``excursion.pick_reason`` → ``toy``). Never
    with nothing to be curious about, and never with a draw then."""
    del core
    return False  # curiosity P2, after ``has_things``


# ---------------------------------------------------------------------------- the engine


def zone_block(core: DuckCore, point: tuple[int, int]) -> str | None:
    """Why ``point`` (mm, Duck Core's frame) is a place it may not walk into (a family zone,
    ``home.zone``), or None."""
    del core, point
    return None  # curiosity P2 (``CoreState.zones``)


def on_fall(core: DuckCore, spot: Mapping[str, Any]) -> None:
    """A fall while it was busy with a thing: the thing counts it (``thing.hard``)."""
    del core, spot  # curiosity P2, while a programme is under way (``active``)


# ---------------------------------------------------------------------------- inputs


def on_sense(core: DuckCore, percept: Mapping[str, Any], record: Event) -> None:
    """``sense.thing`` (already recorded as the input ``record``): what it saw of things."""
    del core, percept, record  # curiosity P2 (phase 4 keeps nothing: no thing ever appears)


def on_result(core: DuckCore, percept: Mapping[str, Any], record: Event) -> None:
    """``body.curio_step`` (already recorded): how the step of a programme went."""
    del core, percept, record  # curiosity P2, while a programme is under way


def on_guard(core: DuckCore, percept: Mapping[str, Any], record: Event) -> None:
    """``body.guard`` (already recorded): a body reflex stopped a step (cliff, spit…)."""
    del core, percept, record  # curiosity P2, while a programme is under way


def on_timeout(core: DuckCore, act_id: str) -> None:
    """A ``duck.curio`` step whose result never came before its deadline."""
    del core, act_id  # curiosity P2, while a programme is under way


def on_home(core: DuckCore, kind: str, params: Mapping[str, Any], record: Event) -> str | None:
    """An App input about things (``home.thing``, ``home.zone``, ``home.play``,
    ``home.ask_answer``; already recorded as ``record``). The note for the family, or None."""
    del core, kind, params, record
    return None  # curiosity P2 (phase 4 keeps nothing: the home stays empty)
