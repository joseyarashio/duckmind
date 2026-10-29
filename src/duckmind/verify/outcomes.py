"""Per-trial and per-pair outcomes of MACHINE-VERIFY.md, from recorded traces only.

Pure functions: a trace goes in, a fact about the world's truth (or about the brain's own
record, where the spec says so) comes out. Booleans may be ``None`` = unknown; nothing here turns
an unknown into a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from duckmind.verify import spec
from duckmind.verify.trace import (
    Trace,
    bearing_to,
    dist_to,
    n_ticks,
    validate,
    window,
    window_complete,
    wrap,
)

# ── what happened in the window ────────────────────────────────────────────────────


def _baseline(trace: Trace) -> tuple[str, int]:
    cols = trace["cols"]
    i = trace["onset_idx"] - 1
    return cols["behaviour"][i], cols["bout"][i]


def change_tick(trace: Trace) -> int | None:
    """The first tick in the window where the brain's behaviour or bout differs from just
    before the onset (M2's 換行為), or None."""
    cols = trace["cols"]
    base_behaviour, base_bout = _baseline(trace)
    for i in window(trace, spec.WINDOW_S):
        if cols["behaviour"][i] != base_behaviour or cols["bout"][i] != base_bout:
            return i
    return None


def changed(trace: Trace) -> bool:
    return change_tick(trace) is not None


def reflex_in_window(trace: Trace) -> bool:
    cols = trace["cols"]
    return any(cols["reflex"][i] is not None for i in window(trace, spec.WINDOW_S))


def startled(trace: Trace) -> bool:
    """M3's 驚嚇／定格: a startle or freeze reflex, or the behaviour showing one, in the window."""
    cols = trace["cols"]
    return any(
        cols["reflex"][i] in spec.STARTLE_NAMES or cols["behaviour"][i] in spec.STARTLE_NAMES
        for i in window(trace, spec.WINDOW_S)
    )


# ── truth objects ──────────────────────────────────────────────────────────────────


STIM_FALLBACK_KEY = "__stim__"


def _stim_key(trace: Trace) -> str:
    return (trace.get("stim") or {}).get("obj") or STIM_FALLBACK_KEY


def _object_paths(trace: Trace, *, with_stim: bool) -> dict[str, list[list[float] | None]]:
    """Every truth object's path by its own id. The stimulus object's path (stim arm: it is in
    the world; sham: its pair's planned path, when ``with_stim``) is under the stimulus object's id."""
    objects = trace.get("objects") or {}
    paths = dict(objects.get("things") or {})
    stim = objects.get("stim")
    if with_stim and stim is not None:
        paths[_stim_key(trace)] = stim
    return paths


def _object_width(trace: Trace, key: str) -> float:
    objects = trace.get("objects") or {}
    if key == _stim_key(trace):
        width = (trace.get("stim") or {}).get("width_m")
    else:
        width = (objects.get("widths") or {}).get(key)
    return float(width) if width else spec.DEFAULT_WIDTH_M


def _track_object(trace: Trace, track: Any) -> str | None:
    """The truth object a brain track came from, by the world's own label of that track
    (``track_keys``: track id -> object id, written by the runner from the simulator's vision
    model). Track ids are only ever read inside their own trial."""
    keys = trace.get("track_keys") or {}
    return keys.get(str(track)) if track is not None else None


def head_turned_toward(trace: Trace) -> bool | None:
    """M1 (a): inside the window the head's heading turned ≥ 10° toward the stimulus's true
    direction AND the error to that direction (from the trunk) fell to half or less."""
    path = (trace.get("objects") or {}).get("stim")
    if path is None:
        return None
    cols = trace["cols"]
    onset = trace["onset_idx"]
    base = onset - 1
    if path[onset] is None:
        return None
    ox, oy = path[onset]
    err0 = wrap(bearing_to(cols["x"][onset], cols["y"][onset], ox, oy) - cols["gaze"][base])
    if err0 == 0.0:
        return False
    sign = 1.0 if err0 > 0.0 else -1.0
    for i in window(trace, spec.WINDOW_S):
        if path[i] is None:
            continue
        ox, oy = path[i]
        err = wrap(bearing_to(cols["x"][i], cols["y"][i], ox, oy) - cols["gaze"][i])
        turn = wrap(cols["gaze"][i] - cols["gaze"][base]) * sign
        if turn >= math.radians(spec.M1_TURN_MIN_DEG) and abs(err) <= spec.M1_ERR_SHRINK * abs(
            err0
        ):
            return True
    return False


def _nearest_by_bearing(
    trace: Trace, i: int, claimed: float, *, with_stim: bool
) -> tuple[str | None, float]:
    """The truth object whose true bearing (trunk frame) is closest to ``claimed`` at tick i."""
    cols = trace["cols"]
    best: tuple[str | None, float] = (None, math.inf)
    for key, path in _object_paths(trace, with_stim=with_stim).items():
        point = path[i]
        if point is None:  # not in the world yet
            continue
        ox, oy = point
        true = wrap(bearing_to(cols["x"][i], cols["y"][i], ox, oy) - cols["yaw"][i])
        err = abs(wrap(claimed - true))
        if err < best[1]:
            best = (key, err)
    return best


def bout_aimed_at_stimulus(trace: Trace) -> bool | None:
    """M1 (b): a NEW bout starts inside the window whose named target is, by the truth, the
    stimulus's object. The truth pairing is the world's own label of the brain's track
    (``track_keys``); a hand-made trace without labels is paired by bearing (the object nearest
    in bearing, within 20°, among every truth object). Track ids are never compared across arms."""
    objects = trace.get("objects") or {}
    if objects.get("stim") is None:
        return None
    cols = trace["cols"]
    _, base_bout = _baseline(trace)
    labelled = trace.get("track_keys") is not None
    key = _stim_key(trace)
    for i in window(trace, spec.WINDOW_S):
        if cols["bout"][i] == base_bout or cols["tgt_kind"][i] != "thing":
            continue
        if labelled:
            if _track_object(trace, cols["tgt_track"][i]) == key:
                return True
            continue
        if cols["tgt_bearing"][i] is None:
            continue
        nearest, err = _nearest_by_bearing(trace, i, cols["tgt_bearing"][i], with_stim=True)
        if nearest == key and err <= math.radians(spec.MATCH_DEG):
            return True
    return False


def attended(trace: Trace) -> bool | None:
    """M1 for one arm. An object: the head turned toward its true place or a bout aimed at it.
    Anything else (a mono sound, the shell touched, a hand coming up): a reflex or a change of
    behaviour inside the window. None = the truth needed is missing."""
    kind = spec.STIMULUS_KIND[trace["stimulus"]]
    if kind == "event":
        return reflex_in_window(trace) or changed(trace)
    head = head_turned_toward(trace)
    aimed = bout_aimed_at_stimulus(trace)
    if head is None or aimed is None:
        return None
    return head or aimed


# ── M5 ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segment:
    bout: int
    behaviour: str
    start: int
    end: int  # last tick of the bout
    reason: str  # "selector" | "reflex_cut" | "timeout" | "censored"
    phase_end: float

    @property
    def category(self) -> str:
        if self.reason == "timeout":
            return "timeout"
        if self.reason == "censored":
            return "censored"
        if self.reason == "reflex_cut":
            return "reflex_cut"
        return "natural" if self.phase_end >= spec.NATURAL_SATIATION else "displaced"


def bout_segments(trace: Trace) -> list[Segment]:
    cols = trace["cols"]
    n = n_ticks(trace)
    dt = trace["dt_s"]
    cut_ticks = max(1, round(spec.REFLEX_CUT_S / dt))
    forced = cols.get("end_reason") or [None] * n
    out: list[Segment] = []
    start = 0
    for k in range(1, n + 1):
        if k < n and cols["bout"][k] == cols["bout"][start]:
            continue
        end = k - 1
        if k == n:
            reason = "censored"
        elif forced[end] == "timeout":
            reason = "timeout"
        elif any(
            cols["behaviour"][j] in spec.HOLD_BEHAVIOURS
            for j in range(max(start, k - cut_ticks), k + 1)
        ):
            reason = "reflex_cut"
        else:
            reason = "selector"
        out.append(
            Segment(
                int(cols["bout"][start]),
                cols["behaviour"][start],
                start,
                end,
                reason,
                float(cols["phase"][end]),
            )
        )
        start = k
    return out


@dataclass(frozen=True)
class Episode:
    """The stretch of behaviour the stimulus started (the first change in the window)."""

    kind: str  # "hold" (a reflex took the body) | "bout"
    behaviour: str
    start: int
    category: str  # natural | displaced | reflex_cut | timeout | censored


def response_episode(trace: Trace) -> Episode | None:
    """The stretch of behaviour the stimulus caused. A reflex that takes the body (startle,
    freeze, ...) beginning inside the window is the response: the whole run of such holds
    (a startle that goes on into a freeze is one) until the body is free again. Without one, the
    first bout that begins in the window. (A startle first makes the selector open a short bout
    of its own — a sit_down for 0.4 s — which is part of the cascade, not the response.)"""
    cols = trace["cols"]
    n = n_ticks(trace)
    holds = spec.HOLD_BEHAVIOURS
    for i in window(trace, spec.WINDOW_S):
        if cols["behaviour"][i] in holds and cols["behaviour"][i - 1] not in holds:
            j = i
            while j < n and cols["behaviour"][j] in holds:
                j += 1
            # a reflex's own program releases the body: its end is its natural end
            return Episode("hold", cols["behaviour"][i], i, "censored" if j >= n else "natural")
    first = change_tick(trace)
    if first is None:
        return None
    bout = cols["bout"][first]
    for seg in bout_segments(trace):
        if seg.bout == bout:
            return Episode("bout", cols["behaviour"][first], first, seg.category)
    return None


def reopened(trace: Trace) -> bool:
    """10 秒內同一行為重開 3 次以上: after the onset, some behaviour's bout started
    ``M5_REOPEN_COUNT`` times within ``M5_REOPEN_WINDOW_S`` seconds."""
    cols = trace["cols"]
    starts: dict[str, list[float]] = {}
    for i in range(trace["onset_idx"], n_ticks(trace)):
        if cols["bout"][i] != cols["bout"][i - 1]:
            starts.setdefault(cols["behaviour"][i], []).append(cols["t"][i])
    need = spec.M5_REOPEN_COUNT
    for times in starts.values():
        for a in range(len(times) - need + 1):
            if times[a + need - 1] - times[a] <= spec.M5_REOPEN_WINDOW_S + 1e-9:
                return True
    return False


# ── M7 ─────────────────────────────────────────────────────────────────────────────


def falls(trace: Trace) -> tuple[int, int]:
    """(number of falls, falls during which the brain kept commanding a walk). A fall is the
    truth ``fallen`` flag going up; blocked = walking commanded more than the grace period
    after it, while the body was still down."""
    cols = trace["cols"]
    n = n_ticks(trace)
    dt = trace["dt_s"]
    count = blocked = 0
    i = 0
    while i < n:
        if cols["fallen"][i] and (i == 0 or not cols["fallen"][i - 1]):
            count += 1
            j = i
            walk = False
            while j < n and cols["fallen"][j]:
                if (j - i) * dt > spec.FALL_GRACE_S and abs(cols["vx"][j]) > 1e-9:
                    walk = True
                j += 1
            blocked += 1 if walk else 0
            i = j
        else:
            i += 1
    return count, blocked


# ── M4 (physics) ───────────────────────────────────────────────────────────────────


@dataclass
class Claims:
    bearing_ok: int = 0
    bearing_n: int = 0
    range_ok: int = 0
    range_n: int = 0
    other_ticks: int = 0  # a target that is not a thing (a person, the nest, a place)
    unmatched_ticks: int = 0  # a named thing but no truth object to compare with
    arrivals: list[dict[str, Any]] = field(default_factory=list)


def _arrival(
    trace: Trace,
    paths: dict[str, list[list[float] | None]],
    i: int,
    labelled: bool,
    tie: dict[tuple[int, int | None], str | None],
    reach0: float,
) -> dict[str, Any]:
    """One 到了: the true distance to the thing the brain was naming at that tick, against the
    touchable distance + 0.1 m. When the brain's target at that tick is not a thing the world
    can name (a remembered *place* — most 到了 come then), what it arrived at is not known:
    ``ok`` is None, and the nearest object's true distance is reported next to it (information
    only, never a verdict)."""
    cols = trace["cols"]
    x, y = cols["x"][i], cols["y"][i]
    near: tuple[str, float] | None = None
    for key, path in paths.items():
        point = path[i]
        if point is not None:
            d = dist_to(x, y, point[0], point[1])
            if near is None or d < near[1]:
                near = (key, d)
    info: dict[str, Any] = {"tick": i, "t": cols["t"][i], "target_kind": cols["tgt_kind"][i]}
    if near is not None:
        info["nearest_object"] = near[0]
        info["nearest_dist_m"] = round(near[1], 3)
        info["nearest_bound_m"] = round(
            reach0 + _object_width(trace, near[0]) / 2.0 + spec.M4_ARRIVE_MARGIN_M, 3
        )
    obj = None
    if cols["tgt_kind"][i] == "thing":
        if labelled:
            obj = _track_object(trace, cols["tgt_track"][i])
        elif cols["tgt_bearing"][i] is not None:
            obj = tie.get((int(cols["bout"][i]), cols["tgt_track"][i]))
            if obj is None:
                obj = _nearest_by_bearing(trace, i, cols["tgt_bearing"][i], with_stim=bool(paths))[
                    0
                ]
    found = None if obj is None else paths.get(obj)
    point = None if found is None else found[i]
    if obj is None or point is None:
        return {**info, "ok": None, "why": "the target is not a thing the world can name"}
    true_d = dist_to(x, y, point[0], point[1])
    bound = reach0 + _object_width(trace, obj) / 2.0 + spec.M4_ARRIVE_MARGIN_M
    return {
        **info,
        "object": obj,
        "true_dist_m": round(true_d, 3),
        "bound_m": round(bound, 3),
        "ok": true_d <= bound,
    }


def claims(trace: Trace) -> Claims:
    """The brain's named targets and its 到了, against the truth. A target is tied to the truth
    object the world labelled its track with (``track_keys``; a hand-made trace without labels:
    the object nearest in bearing the first tick its (bout, track) is named). Every tick the
    claimed bearing and range are compared with that object's true ones. An arrival is judged
    against the thing the brain names at that tick (see ``_arrival``)."""
    out = Claims()
    cols = trace["cols"]
    n = n_ticks(trace)
    objects = trace.get("objects") or {}
    paths = _object_paths(trace, with_stim=not objects.get("stim_virtual"))
    labelled = trace.get("track_keys") is not None
    tie: dict[tuple[int, int | None], str | None] = {}
    reach0 = float(trace.get("touch_reach_m", spec.BEAK_REACH_M))
    for i in range(n):
        kind = cols["tgt_kind"][i]
        if kind is not None and kind != "thing":
            out.other_ticks += 1
        if kind == "thing" and cols["tgt_bearing"][i] is not None:
            if labelled:
                obj = _track_object(trace, cols["tgt_track"][i])
            else:
                key = (int(cols["bout"][i]), cols["tgt_track"][i])
                if tie.get(key) is None:
                    tie[key] = _nearest_by_bearing(
                        trace, i, cols["tgt_bearing"][i], with_stim=bool(paths)
                    )[0]
                obj = tie[key]
            found = None if obj is None else paths.get(obj)
            point = None if found is None else found[i]
            if point is None:
                out.unmatched_ticks += 1
            else:
                ox, oy = point
                x, y, yaw = cols["x"][i], cols["y"][i], cols["yaw"][i]
                true_b = wrap(bearing_to(x, y, ox, oy) - yaw)
                out.bearing_n += 1
                if abs(wrap(cols["tgt_bearing"][i] - true_b)) <= math.radians(spec.M4_BEARING_DEG):
                    out.bearing_ok += 1
                rng = cols["tgt_range"][i]
                if rng is not None:
                    true_d = dist_to(x, y, ox, oy)
                    out.range_n += 1
                    if true_d > 1e-6 and abs(rng - true_d) / true_d <= spec.M4_RANGE_REL:
                        out.range_ok += 1
        if cols["arrived"][i] and (i == 0 or not cols["arrived"][i - 1]):
            out.arrivals.append(_arrival(trace, paths, i, labelled, tie, reach0))
    return out


# ── approach (reported only) ───────────────────────────────────────────────────────


def approach(trace: Trace) -> dict[str, float] | None:
    """How close the trunk came to the stimulus object after the onset (a number, no verdict)."""
    path = (trace.get("objects") or {}).get("stim")
    if path is None or (trace.get("objects") or {}).get("stim_virtual"):
        return None
    cols = trace["cols"]
    onset = trace["onset_idx"]
    dists = [
        dist_to(cols["x"][i], cols["y"][i], path[i][0], path[i][1])
        for i in range(onset, n_ticks(trace))
        if path[i] is not None
    ]
    if not dists:
        return None
    reach = (
        float(trace.get("touch_reach_m", spec.BEAK_REACH_M))
        + _object_width(trace, _stim_key(trace)) / 2.0
    )
    return {
        "start_m": dists[0],
        "min_m": min(dists),
        "touchable_m": reach + spec.M4_ARRIVE_MARGIN_M,
    }


# ── a pair ─────────────────────────────────────────────────────────────────────────


@dataclass
class PairOutcome:
    pair: str
    preset: str
    seed: int
    stimulus: str
    valid: bool
    problems: list[str] = field(default_factory=list)
    perceivable: bool | None = None
    attend_stim: bool | None = None
    attend_sham: bool | None = None
    changed_stim: bool = False
    changed_sham: bool = False
    reflex_stim: bool = False
    startled_stim: bool = False
    startled_sham: bool = False
    episode: Episode | None = None
    caused: bool = False
    reopen_stim: bool = False
    falls_stim: int = 0
    falls_sham: int = 0
    blocked_stim: int = 0
    blocked_sham: int = 0
    claims: list[Claims] = field(default_factory=list)
    approach: dict[str, float] | None = None
    behaviours_seen: frozenset[str] = frozenset()


def pair_outcome(stim: Trace, sham: Trace) -> PairOutcome:
    out = PairOutcome(
        pair=stim.get("pair", ""),
        preset=stim.get("preset", ""),
        seed=int(stim.get("seed", -1)),
        stimulus=stim.get("stimulus", ""),
        valid=True,
    )
    for label, trace in (("stim", stim), ("sham", sham)):
        problems = validate(trace)
        if not problems and not window_complete(trace, spec.WINDOW_S):
            problems.append("trace ends inside the window")
        out.problems += [f"{label}: {p}" for p in problems]
    for label, trace in (("stim", stim), ("sham", sham)):
        check = trace.get("pair_check")
        if check is not None and check.get("pre_onset_equal") is False:
            out.problems.append(f"{label}: the two arms differ before the onset ({check})")
    if out.problems:
        out.valid = False
        return out
    if sham.get("stimulus") != stim.get("stimulus") or sham.get("seed") != stim.get("seed"):
        out.valid = False
        out.problems.append("sham is not the same stimulus and seed")
        return out
    s = stim.get("stim") or {}
    out.perceivable = s.get("perceivable") if s.get("delivered") else False
    if not s.get("delivered"):
        out.problems.append("stimulus not delivered by the world")
        out.valid = False
        return out
    out.attend_stim = attended(stim)
    out.attend_sham = attended(sham)
    out.changed_stim = changed(stim)
    out.changed_sham = changed(sham)
    out.reflex_stim = reflex_in_window(stim)
    out.startled_stim = startled(stim)
    out.startled_sham = startled(sham)
    out.episode = response_episode(stim)
    out.caused = out.changed_stim and not out.changed_sham
    out.reopen_stim = reopened(stim)
    out.falls_stim, out.blocked_stim = falls(stim)
    out.falls_sham, out.blocked_sham = falls(sham)
    if stim["layer"] == "L2":
        out.claims = [claims(stim), claims(sham)]
    out.approach = approach(stim)
    out.behaviours_seen = frozenset(stim["cols"]["behaviour"]) | frozenset(
        sham["cols"]["behaviour"]
    )
    return out
