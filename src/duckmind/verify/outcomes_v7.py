"""M1 of MACHINE-VERIFY v7: v6's reading, or a visual reflex whose source track is the object.

Beside ``outcomes_v5.py`` ... ``outcomes_v6.py`` (untouched in what they judge). For a stimulus with a place an arm 'noticed' when v6's
reading says so (the brain's target of kind thing or threat on the object's track, at least 0.3 s, started at or after the stimulus tick
and not selected on the tick before) OR when a visual reflex (startle, freeze, orient, withdraw) had its onset inside the 3 s window with
a source track (``reflex_track``) that the simulator's labels map to the stimulus object (control arm: to a thing whose true position is
within 0.3 m of the phantom spot, at the tick the reflex began). A reflex needs no hold.

A reflex's onset is the first tick of a run of consecutive visual-reflex ticks on the same source track (a startle that goes on into its
freeze is one run). A run that is already going on the tick before the stimulus has its onset before the window and does not count.

A recording without a ``reflex_track`` column (made before the duck's code said what started its reflexes), or one whose column is empty
(None on every tick), has no reflex source to read: the reflex reading is "no" there and v7 reads it exactly as v6 does.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

from duckmind.verify import outcomes_v6 as O6
from duckmind.verify import spec_v7 as S
from duckmind.verify.outcomes_v5 import Escape, PairOutcomeV5, escape_in_window
from duckmind.verify.trace import Trace, window

__all__ = [
    "Escape",
    "PairOutcomeV5",
    "TargetV7",
    "attended",
    "control_target",
    "escape_in_window",
    "pair_outcome",
    "stim_target",
]


@dataclass(frozen=True)
class TargetV7(O6.TargetV6):
    """``value``: v6's reading or the reflex reading (True if either; None if neither is True and one is unknown). The rest is for the
    report only."""

    v6_value: bool | None = None  # v6's reading of the same arm
    reflex_value: bool | None = None  # the reflex reading alone
    has_reflex_track: bool = False  # the recording says what started its reflexes
    reflex_onsets: int = 0  # visual-reflex onsets inside the window (any track)
    reflex_on_target_onsets: int = (
        0  # …of those, the ones whose track maps to the object / a thing at the spot
    )
    reflex_kinds: tuple[str, ...] = field(
        default_factory=tuple
    )  # the reflexes at the onsets that map, sorted, each once


def has_reflex_track(trace: Trace) -> bool:
    cols = trace["cols"]
    return "reflex_track" in cols and any(v is not None for v in cols["reflex_track"])


def _key(cols: dict[str, Any], i: int) -> int | None:
    """The source track of the visual reflex at tick i, or None (no reflex, another kind of reflex, no source track)."""
    track = cols["reflex_track"][i]
    if cols["reflex"][i] in S.VISUAL_REFLEXES and track:
        return int(track)
    return None


def onsets(trace: Trace) -> list[int]:
    """Tick indices inside the window at which a visual reflex begins (the tick before is not a visual reflex on the same track)."""
    cols = trace["cols"]
    out: list[int] = []
    for i in window(trace, S.WINDOW_S):
        k = _key(cols, i)
        if k is not None and (i == 0 or _key(cols, i - 1) != k):
            out.append(i)
    return out


def _combine(v6: bool | None, reflex: bool | None) -> bool | None:
    if v6 is True or reflex is True:
        return True
    if v6 is None or reflex is None:
        return None
    return False


def _stim_reflex(trace: Trace) -> tuple[bool | None, int, int, tuple[str, ...]]:
    if not has_reflex_track(trace):
        return False, 0, 0, ()
    cols = trace["cols"]
    starts = onsets(trace)
    keys = trace.get("track_keys")
    if keys is None:
        return None, len(starts), 0, ()
    obj = (trace.get("stim") or {}).get("obj") or "__stim__"
    hits = [i for i in starts if keys.get(str(cols["reflex_track"][i])) == obj]
    return bool(hits), len(starts), len(hits), tuple(sorted({cols["reflex"][i] for i in hits}))


def _control_reflex(trace: Trace) -> tuple[bool | None, int, int, tuple[str, ...]]:
    if not has_reflex_track(trace):
        return False, 0, 0, ()
    cols = trace["cols"]
    starts = onsets(trace)
    keys = trace.get("track_keys")
    objects = trace.get("objects") or {}
    path, things = objects.get("stim"), objects.get("things")
    if keys is None or path is None or things is None or path[trace["onset_idx"]] is None:
        return None, len(starts), 0, ()
    hits: list[int] = []
    for i in starts:
        label = keys.get(str(cols["reflex_track"][i]))
        track = None if label is None else things.get(label)
        where = None if track is None or i >= len(track) else track[i]
        spot = path[i] if i < len(path) else None
        if where is None or spot is None:
            continue
        if math.hypot(where[0] - spot[0], where[1] - spot[1]) <= S.SPOT_RADIUS_M + S.EPS:
            hits.append(i)
    return bool(hits), len(starts), len(hits), tuple(sorted({cols["reflex"][i] for i in hits}))


def _extend(base: O6.TargetV6, trace: Trace, arm: str) -> TargetV7:
    reflex, n_onsets, n_hits, kinds = (_stim_reflex if arm == "stim" else _control_reflex)(trace)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(O6.TargetV6)}
    fields["value"] = _combine(base.value, reflex)
    if fields["value"] is not None:
        fields["why"] = ""
    return TargetV7(
        **fields,
        v6_value=base.value,
        reflex_value=reflex,
        has_reflex_track=has_reflex_track(trace),
        reflex_onsets=n_onsets,
        reflex_on_target_onsets=n_hits,
        reflex_kinds=kinds,
    )


def stim_target(trace: Trace, **kw: Any) -> TargetV7:
    return _extend(O6.stim_target(trace, **kw), trace, "stim")


def control_target(trace: Trace, **kw: Any) -> TargetV7:
    return _extend(O6.control_target(trace, **kw), trace, "sham")


def attended(trace: Trace) -> bool | None:
    stimulus = trace["stimulus"]
    if stimulus in S.DIRECTIONAL:
        read = stim_target if trace["arm"] == "stim" else control_target
        return read(trace).value
    return O6.attended(
        trace
    )  # an event (any reflex in the window), or a stimulus the spec does not list


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcomeV5:
    out = O6.pair_outcome(stim, sham)
    if isinstance(out.tgt_stim, O6.TargetV6) and isinstance(out.tgt_sham, O6.TargetV6):
        out.tgt_stim = _extend(out.tgt_stim, stim, "stim")
        out.tgt_sham = _extend(out.tgt_sham, sham, "sham")
        out.attend_stim, out.attend_sham = out.tgt_stim.value, out.tgt_sham.value
    return out
