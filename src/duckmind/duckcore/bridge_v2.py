"""``bridge-v2``: :mod:`bridge-v1 <duckmind.duckcore.bridge>`'s vocabulary and three words of the
trick lessons (phase 2 「學會聽話」, docs/design/neural-duck/LEARN-TRICKS-2026-10-25.md).

bridge-v1 is frozen with its name, so it is not edited: this module is bridge-v1's kinds, fields and
handlers, unchanged, plus three kinds (:data:`NEW_KINDS`):

* ``teach_trick`` — the duck was told to do a trick and the command ended: ``outcome`` ``done``
  (it did it), ``refused`` (it would not, or could not yet) or ``gave_up`` (it tried and stopped);
  ``ms`` from hearing it to the end. Duck Core writes its own record of it (``sense.teach_trick``)
  and counts a lesson it did as play care (``_care``: five of a kind a growth day count) — a
  lesson is time the family spent playing with it; a refusal or giving up is someone here, no care.
* ``praise`` — the owner's 「乖」 (or a treat) for a trick (``trick``; empty: for nothing in
  particular). The praise itself — someone here, the happy moment, the care, the bond — is Duck
  Core's own phone input (``owner_good`` / ``pad_good``), which the family's feedback button sends
  with it; this kind adds only what the praise was *for*: with a trick, Duck Core's record
  ``owner.trick_good`` and a household preference ``trick:<id>`` one step up (``_learn_pref``, its
  diminishing steps of a growth day).
* ``scold`` — the owner's 「不對」 for a trick: as ``owner_not_that`` does for what it did, the
  preference ``trick:<id>`` one step down, once a growth day (record ``owner.trick_not_that``).

How well the duck *does* a trick (its proficiency) is the neural duck's own (``neuro.tricks``): Duck
Core never sees or keeps it. Duck Core imports nothing of the neural duck: :data:`TRICKS` is the
same five ids as ``neuro.command_words.TRICKS`` (a test keeps them equal).

A duck fed only bridge-v1 kinds saves its inputs as ``bridge-v1`` (as before, so its file still
opens on a build that knows only bridge-v1); one that was given any kind of :data:`NEW_KINDS` saves as
``bridge-v2`` (:func:`api_of`), which an older build refuses rather than misreads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from duckmind.duckcore import bridge
from duckmind.duckcore.bridge import Record, _Field, _int, _one_of

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore
    from duckmind.provenance import Event

BRIDGE_API = "bridge-v2"
# every vocabulary this build reads: an attached duck may have been fed by either
APIS = (bridge.BRIDGE_API, BRIDGE_API)
TRICKS = ("lie_down", "spin", "bow", "shake", "flip")
TEACH_OUTCOMES = ("done", "refused", "gave_up")
MS_MAX = 600_000
PREF = "trick:"  # a household preference for a trick: ``trick:<id>`` (as ``verb:<id>``)

NEW_KINDS: dict[str, dict[str, _Field]] = {
    "teach_trick": {
        "trick": (_one_of(TRICKS), None),
        "ms": (_int(0, MS_MAX), 0),
        "outcome": (_one_of(TEACH_OUTCOMES), None),
    },
    "praise": {"trick": (_one_of(("", *TRICKS)), "")},
    "scold": {"trick": (_one_of(("", *TRICKS)), "")},
}
KINDS: dict[str, dict[str, _Field]] = {**bridge.KINDS, **NEW_KINDS}
assert not set(bridge.KINDS) & set(NEW_KINDS)

RECORD_KINDS = bridge.RECORD_KINDS | {
    "sense.teach_trick",
    "owner.trick_good",
    "owner.trick_not_that",
}


def check(event: Mapping[str, Any]) -> dict[str, Any]:
    return bridge.check(event, KINDS)


def check_all(events: Any) -> list[dict[str, Any]]:
    return bridge.check_all(events, KINDS)


def arrive(core: DuckCore, event: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    return bridge.arrive(core, event, KINDS)


def record_of(core: DuckCore, ev: dict[str, Any]) -> Record | None:
    """What Duck Core writes of this input (bridge-v1's own records for its kinds)."""
    kind = ev["k"]
    if kind == "teach_trick":
        payload = {"trick": ev["trick"], "seconds": ev["ms"] // 1000, "outcome": ev["outcome"]}
        return Record("sense.teach_trick", payload, "live")
    if kind == "praise":
        return Record("owner.trick_good", {"trick": ev["trick"]}, "owner") if ev["trick"] else None
    if kind == "scold":
        return (
            Record("owner.trick_not_that", {"trick": ev["trick"]}, "owner") if ev["trick"] else None
        )
    return bridge.record_of(core, ev)


def api_of(inputs: Iterable[Any]) -> str:
    """The vocabulary a duck's inputs need: ``bridge-v2`` once any of them is a :data:`NEW_KINDS`
    experience, else ``bridge-v1``. ``inputs``: ``(method, params)`` pairs, as a session keeps them."""
    for method, params in inputs:
        if method != "experience" or not isinstance(params, Mapping):
            continue
        for event in params.get("events") or ():
            if isinstance(event, Mapping) and event.get("k") in NEW_KINDS:
                return BRIDGE_API
    return bridge.BRIDGE_API


# ── the handlers ────────────────────────────────────────────────────────────────────────────


def _teach_trick(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._care("play"): someone is here and taught it a trick; a lesson it did is play care
    (the engine's cap: five of a kind a growth day count). One it refused or gave up on is only
    someone here."""
    assert record is not None
    core._presence("", record)
    if ev["outcome"] != "done":
        return ev["outcome"]
    core._care("play", record, why="teach")
    return "taught"


def _praise(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._learn_pref (as a praised verb): the family likes it doing this trick. The praise
    itself is the phone's ``owner_good`` / ``pad_good`` input, sent with this one."""
    if record is None:
        return "no trick"
    core._learn_pref(PREF + ev["trick"], +1, record)
    return "praised"


def _scold(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.owner_not_that (for this trick): the preference one step down, once a growth day."""
    if record is None:
        return "no trick"
    item = PREF + ev["trick"]
    if item in core.s.not_that_gd:
        return "said today"
    core.s.not_that_gd.add(item)
    core._learn_pref(item, -1, record)
    return "scolded"


HANDLERS: dict[str, Callable[[DuckCore, dict[str, Any], Event | None], str]] = {
    **bridge.HANDLERS,
    "teach_trick": _teach_trick,
    "praise": _praise,
    "scold": _scold,
}
assert set(HANDLERS) == set(KINDS)
