"""M1 of MACHINE-VERIFY-v5.md (frozen) for a stimulus with a place: did the brain select the real object as its target?

Beside ``outcomes.py`` ... ``outcomes_v4.py`` (all untouched). Operational details the spec leaves open: ``MACHINE-VERIFY-v5-CHOICES.md``.

Stimulated arm: inside the 3 s window (the stimulus tick to 3.0 s after it, both ends) the brain's target column says "a thing"
(``tgt_kind == "thing"``) and the track it names (``tgt_track``) is one the simulator labelled with the stimulus object
(``track_keys``: track id -> object id, written by the runner from the simulator's own vision model; never guessed from a
bearing). A track the simulator never labelled, or labelled with another object, is not the object. The target counts when it
stays on the object for at least 0.3 s in a row (consecutive ticks; length = first tick to last tick by the recorded times) **in a
stretch that starts after the stimulus tick**: a selection that is already there at the stimulus tick (or before it) is not the
stimulus's doing and its whole stretch is left out (it is reported as ``at_stimulus`` / ``carried_s``). Any track of the object
counts, so a track that is re-issued while the duck keeps its target on the object is one stretch.

Control arm (same seed, nothing was put there): the same, for **any** labelled thing whose true position (``objects.things``) is
within 0.3 m of the empty spot — the spot the stimulated arm's object has at the same tick (the pair's planned path: for a new
object that is the place it is put; for a ball it moves with the ball) — in a stretch that starts after the stimulus tick.

Unknown only when the truth needed is missing (no label map; for the control: no thing positions or no spot path). Everything
else is a yes or a no. Booleans may be ``None`` = unknown; nothing here turns an unknown into a pass.

The known limit (reported, not a gate): the brain's "threat" and "place" targets carry no track, so being scared off by the ball
cannot be put down to the ball. ``escape_in_window`` counts, for the reference row, a reflex or a flee in the 3 s window.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

from duckmind.verify import outcomes as O1
from duckmind.verify import outcomes_v4 as O4
from duckmind.verify import spec_v5 as S
from duckmind.verify.trace import Trace, window

UNKNOWN_NO_LABELS = "no_labels"
UNKNOWN_NO_PATH = "no_path"
UNKNOWN_NO_THINGS = "no_things"
UNKNOWN_NO_SPOT = "no_spot_at_onset"
UNKNOWN_NO_TARGET_COLUMNS = "no_target_columns"
THING_ONLY: tuple[str, ...] = ("thing",)  # the target kinds v5 and v5.1 read; v6 adds "threat"


@dataclass(frozen=True)
class Target:
    """One arm's reading. ``value``: True = a qualifying stretch exists; False = none; None = unknown (``why``). The rest are the
    numbers the rule used, for the report (nothing judges on them)."""

    value: bool | None
    why: str = ""
    longest_s: float = (
        0.0  # the longest stretch that starts after the stimulus tick, first tick to last tick
    )
    longest_cover_s: float = (
        0.0  # the same counted as ticks × dt (a reported variant of "in a row")
    )
    start_s: float | None = None  # seconds after the stimulus at which that stretch began
    matched_ticks: int = 0  # ticks of the window that match (every stretch)
    thing_ticks: int = 0  # ticks of the window on which the brain's target is a thing (any thing)
    unlabelled_ticks: int = 0  # …whose track the simulator never labelled
    before: bool = False  # the tick before the stimulus already matched
    at_stimulus: bool = False  # the stimulus tick itself matches: a selection that was there already (its stretch is left out)
    carried_s: float = (
        0.0  # the length of that left-out stretch, first tick to last tick (inside the window)
    )
    objects: tuple[str, ...] = field(
        default_factory=tuple
    )  # the labels matched (control: every thing near the spot)
    nearest_m: float | None = (
        None  # control: nearest the selected labelled things came to the spot, any tick of the window
    )
    labelled_object: bool = False  # stimulated arm: some track of the trial (any time) was labelled with the stimulus object


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for k, flag in enumerate(flags):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


@dataclass(frozen=True)
class Stretch:
    qualifies: bool  # a stretch of at least the hold, starting after the stimulus tick
    longest_s: (
        float  # the longest stretch that starts after the stimulus tick (first tick to last tick)
    )
    cover_s: float  # …counted as ticks × dt
    start_s: float | None  # …and when it began, after the stimulus tick
    carried_s: float  # the stretch present at the stimulus tick, left out (0 when there is none)


def _stretch(
    trace: Trace,
    win: range,
    flags: list[bool],
    hold_s: float,
    *,
    carry_only: bool = False,
    before: bool = False,
) -> Stretch:
    """The stretches of consecutive matching ticks. v5 (``carry_only`` false): one that starts at the stimulus tick (window index 0)
    was already there and is left out, whatever its length. v5.1 (``carry_only`` true): the one at index 0 is left out only when the
    selection was also there on the tick before the stimulus (``before``); a stretch that starts at the stimulus tick with nothing
    selected before it counts. Of the others, the longest is reported and one of at least ``hold_s`` qualifies."""
    t = trace["cols"]["t"]
    dt = float(trace["dt_s"])
    best: tuple[float, float, float | None] = (0.0, 0.0, None)
    qualifies = False
    carried = 0.0
    for a, b in _runs(flags):
        length = float(t[win[b]] - t[win[a]])
        if a == 0 and (not carry_only or before):
            carried = length
            continue
        if length >= hold_s - S.EPS:
            qualifies = True
        if length > best[0] or (length == best[0] and best[2] is None):
            best = (length, (b - a + 1) * dt, float(t[win[a]] - t[win[0]]))
    return Stretch(qualifies, best[0], best[1], best[2], carried)


def _label(trace: Trace, i: int, kinds: tuple[str, ...] = THING_ONLY) -> tuple[bool, str | None]:
    """(the brain's target at tick i is of one of ``kinds`` — v5, v5.1: a thing; v6: a thing or a threat —, the simulator's label of
    that track or None)."""
    cols = trace["cols"]
    if cols["tgt_kind"][i] not in kinds:
        return False, None
    tid = cols["tgt_track"][i]
    if tid is None:
        return True, None
    return True, (trace.get("track_keys") or {}).get(str(tid))


def _has_columns(trace: Trace) -> bool:
    cols = trace["cols"]
    return "tgt_kind" in cols and "tgt_track" in cols


def stim_target(
    trace: Trace,
    *,
    hold_s: float = S.TARGET_HOLD_S,
    window_s: float = S.WINDOW_S,
    carry_only: bool = False,
    kinds: tuple[str, ...] = THING_ONLY,
) -> Target:
    """The stimulated arm: the target on the label-matched track of the stimulus object, at least ``hold_s`` in a row, in a stretch that
    starts after the stimulus tick. ``kinds``: which target kinds select (v6: a thing or a threat); ``thing_ticks`` and
    ``unlabelled_ticks`` always count things only."""
    if not _has_columns(trace):
        return Target(None, why=UNKNOWN_NO_TARGET_COLUMNS)
    if trace.get("track_keys") is None:
        return Target(None, why=UNKNOWN_NO_LABELS)
    key = (trace.get("stim") or {}).get("obj") or "__stim__"
    win = window(trace, window_s)
    flags: list[bool] = []
    thing = unlabelled = 0
    for i in win:
        is_thing, label = _label(trace, i)
        thing += is_thing
        unlabelled += is_thing and label is None
        chosen, chosen_label = (is_thing, label) if kinds == THING_ONLY else _label(trace, i, kinds)
        flags.append(chosen and chosen_label == key)
    onset = trace["onset_idx"]
    before_kind, before_label = _label(trace, onset - 1, kinds)
    before = before_kind and before_label == key
    got = _stretch(trace, win, flags, hold_s, carry_only=carry_only, before=before)
    return Target(
        got.qualifies,
        longest_s=got.longest_s,
        longest_cover_s=got.cover_s,
        start_s=got.start_s,
        matched_ticks=sum(flags),
        thing_ticks=thing,
        unlabelled_ticks=unlabelled,
        before=before,
        at_stimulus=flags[0],
        carried_s=got.carried_s,
        objects=(key,) if any(flags) else (),
        labelled_object=key in set((trace.get("track_keys") or {}).values()),
    )


def control_target(
    trace: Trace,
    *,
    hold_s: float = S.TARGET_HOLD_S,
    radius_m: float = S.SPOT_RADIUS_M,
    spot: str = "path",
    window_s: float = S.WINDOW_S,
    carry_only: bool = False,
    kinds: tuple[str, ...] = THING_ONLY,
) -> Target:
    """The control arm: the target on any labelled thing whose true position is within ``radius_m`` of the empty spot, at least
    ``hold_s`` in a row, in a stretch that starts after the stimulus tick. ``spot``: ``path`` (the frozen reading) = the spot is where
    the stimulated arm's object is at the same tick (the pair's planned path); ``onset`` = where it is at the stimulus tick (a
    variant the dry run reported; no judgement uses it). ``kinds``: which target kinds select (v6: a thing or a threat)."""
    if not _has_columns(trace):
        return Target(None, why=UNKNOWN_NO_TARGET_COLUMNS)
    if trace.get("track_keys") is None:
        return Target(None, why=UNKNOWN_NO_LABELS)
    objects = trace.get("objects") or {}
    path = objects.get("stim")
    things = objects.get("things")
    if path is None:
        return Target(None, why=UNKNOWN_NO_PATH)
    if things is None:
        return Target(None, why=UNKNOWN_NO_THINGS)
    onset = trace["onset_idx"]
    if path[onset] is None:
        return Target(None, why=UNKNOWN_NO_SPOT)

    def spot_at(i: int) -> list[float] | None:
        if spot == "onset" or i < onset:
            return path[onset]
        return path[i]

    def distance(label: str | None, i: int) -> float | None:
        if label is None:
            return None
        track = things.get(label)
        where = None if track is None or i >= len(track) else track[i]
        target_spot = spot_at(i)
        if where is None or target_spot is None:
            return None
        return math.hypot(where[0] - target_spot[0], where[1] - target_spot[1])

    def near(i: int) -> tuple[bool, str | None, float | None]:
        is_chosen, label = _label(trace, i, kinds)
        if not is_chosen:
            return False, None, None
        d = distance(label, i)
        return d is not None and d <= radius_m + S.EPS, label, d

    win = window(trace, window_s)
    flags: list[bool] = []
    thing = unlabelled = 0
    matched: list[str] = []
    nearest: float | None = None
    for i in win:
        is_thing, label = _label(trace, i)
        thing += is_thing
        unlabelled += is_thing and label is None
        hit, hit_label, d = near(i)
        flags.append(hit)
        if hit and hit_label is not None and hit_label not in matched:
            matched.append(hit_label)
        if d is not None and (nearest is None or d < nearest):
            nearest = d
    before = near(onset - 1)[0]
    got = _stretch(trace, win, flags, hold_s, carry_only=carry_only, before=before)
    return Target(
        got.qualifies,
        longest_s=got.longest_s,
        longest_cover_s=got.cover_s,
        start_s=got.start_s,
        matched_ticks=sum(flags),
        thing_ticks=thing,
        unlabelled_ticks=unlabelled,
        before=before,
        at_stimulus=flags[0],
        carried_s=got.carried_s,
        objects=tuple(matched),
        nearest_m=nearest,
    )


def target(trace: Trace, **kw: Any) -> Target:
    """The arm's own reading: the stimulated arm by ``stim_target``, the control arm by ``control_target``."""
    if trace["arm"] == "stim":
        keep = ("hold_s", "window_s", "carry_only", "kinds")
        return stim_target(trace, **{k: v for k, v in kw.items() if k in keep})
    return control_target(trace, **kw)


def attended(
    trace: Trace, *, carry_only: bool = False, kinds: tuple[str, ...] = THING_ONLY
) -> bool | None:
    """M1 for one arm. A stimulus with a place: the brain's target on the real object (control: on any thing at the empty
    spot); an event: any reflex in the window (v3's rule). ``carry_only``: the v5.1 errata's reading of the start rule;
    ``kinds``: v6's target kinds."""
    stimulus = trace["stimulus"]
    if stimulus in S.EVENT:
        return O4.attended(trace)
    if stimulus in S.DIRECTIONAL:
        return target(trace, carry_only=carry_only, kinds=kinds).value
    raise ValueError(f"stimulus {stimulus!r} is in neither list of the spec")


@dataclass(frozen=True)
class Escape:
    """The reference numbers of the known limit: a reflex, or the duck running off, in the 3 s window."""

    reflex: bool
    flee: bool

    @property
    def either(self) -> bool:
        return self.reflex or self.flee


def escape_in_window(trace: Trace) -> Escape:
    """A reflex of any kind in the window (v3's event rule) and a flee: the brain's target kind is ``threat`` or its behaviour is
    ``flee`` / ``retreat`` on some tick of the window. Both come from the brain's own record; neither is judged."""
    cols = trace["cols"]
    flee = any(
        cols["tgt_kind"][i] == "threat" or cols["behaviour"][i] in S.FLEE_BEHAVIOURS
        for i in window(trace, S.WINDOW_S)
    )
    return Escape(O1.reflex_in_window(trace), flee)


@dataclass
class PairOutcomeV5(O4.PairOutcomeV4):
    """v4's pair outcome (its fixation and turn readings kept, reported) with v5's reading in ``attend_*``, its numbers, and the
    reference "reflex or flee" reading of each arm."""

    tgt_stim: Target | None = None
    tgt_sham: Target | None = None
    esc_stim: Escape | None = None
    esc_sham: Escape | None = None


def pair_outcome(
    stim: Trace, sham: Trace, *, carry_only: bool = False, kinds: tuple[str, ...] = THING_ONLY
) -> PairOutcomeV5:
    base = O4.pair_outcome(stim, sham)
    out = PairOutcomeV5(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
    if not out.valid or stim["stimulus"] not in S.DIRECTIONAL:
        return out
    out.tgt_stim = stim_target(stim, carry_only=carry_only, kinds=kinds)
    out.tgt_sham = control_target(sham, carry_only=carry_only, kinds=kinds)
    out.attend_stim, out.attend_sham = out.tgt_stim.value, out.tgt_sham.value
    out.esc_stim, out.esc_sham = escape_in_window(stim), escape_in_window(sham)
    return out
