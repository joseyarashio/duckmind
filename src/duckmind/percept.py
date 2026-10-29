"""``percept-v1``: the sensations duck-body hands Duck Core (physical-design spec §7.2, App. A).

Every percept is one JSON object. Four stamps are on all of them:

* ``kind`` — the event name (``walk.done``, ``sense.light``, ``body.fell`` …, see :data:`SCHEMAS`);
* ``src`` — who sensed it: ``robot`` (the board), ``live`` (duck-body beside the live simulator),
  ``surrogate`` (HomeWorld), ``legacy`` (Duck Core's old dice moved out), ``replay``;
* ``t_ms`` — when the sensation *happened*, on the sender's millisecond axis (duck-body: its
  ``CLOCK_MONOTONIC`` axis, the same one as ``robot.state.t_ns``; the body.sock ``hello`` pairs it
  with the wall clock, :mod:`duckmind.duckbody.protocol`). Duck Core takes it in at the whole
  second :func:`due_second` and keeps the raw ``t_ms`` in the content (§7.3 rule 4);
* ``pv`` — the perception version: ``p:<hex>`` (:func:`perception_hash`) from duck-body,
  ``c:<hex>`` (a calibration hash) from a surrogate, ``l:<hex>`` from the legacy world.

**Integers only** (§7.2 "全部整數"): distances in mm, angles in mrad, rates in mm/s and mrad/s,
probabilities in ppm, battery in %. A float anywhere is a validation error; ``true``/``false``,
text enums, ids and ``null`` for an absent optional are the only non-integers. Use :func:`mm`,
:func:`mrad`, :func:`ppm` to quantise.

Phase-3 additions to the spec's table (owner decisions 2026-09-27; all optional for old readers):

* ``body.fell.trigger`` — ``fallen`` (robotd's own flag, trigger A) or ``tilt40`` (tilt > 40° held
  1 s, trigger B, owner B1);
* ``body.righting.how`` — ``wait`` (velstand at zero command) or ``sitstand`` (the sit-2-s-then-
  stand second try, owner B2);
* ``walk.done.reason`` gains ``reflex_unloaded`` (the legs went light but not enough to claim a
  lift: the duck stopped without saying "picked up", owner B3), ``deadline`` (the act ran past its
  ``deadline_ms``), ``reflex_dark`` (a dark-rule segment cap: kept for readers, though duck-body
  caps no dark segment since the phase-4 landing, owner 「一律放寬」) and ``veto_head`` (the
  head-down approach made no progress, owner A2);
* ``walk.done`` optional ``purpose``, ``head`` (``home``/``down``: owner A2 lets the final
  approach toward a just-seen thing walk head down), ``stop_mm`` (the ToF ahead range at the stop),
  ``attention`` (``low``/``elsewhere``: the caller said its attention was not on its way, owner
  2026-09-28 「呆萌感」 — absent when it was, the default);
* ``act_id`` is optional on every kind that answers a command (``body.posture``, ``body.dock``,
  ``sense.sighting`` for a ``duck.look``), so a caller can match results without guessing;
* ``sense.sighting.r_src`` gains ``cue`` and ``r_mm`` may be ``null``: a lit backlight or the
  dock's cyan cue gives a direction only, never a range (owner C5);
* ``sense.light.level`` gains ``none`` (the eye is down: treated as black, spec §6).

Phase-4 additions (「邊界」, WP9; ``docs/design/phase4-boundary.md`` §4): the three kinds the
world hands Duck Core that phase 4 validated provisionally in ``duckmind.worldport`` until this
file was committed — the same names and fields as v4's records, so the page and the memory
readers keep working:

* ``sense.touch {where, seconds: 0–86,400, firmness, path: ear|joint, latency_ms,
  legacy_phantom?}`` — a hand on the duck (``seconds`` how long it lasted); ``legacy_phantom`` is
  a transitional truth field (design A10: LegacyWorld marks the TV's false touch with it;
  ``tests/legacy_truth_fields.json`` lists it, phase 5f removes it);
* ``sense.hold_head {path, moving}`` — a hand holding the head;
* ``body.play {policy, meters, skill?, minute?}`` — a minute of gamepad play.

The contact kinds keep the spec's names: ``blocked``, ``brushed`` (grazed), ``stumbled``
(tripped). Correction: the spec's ``sense.contact {kind, side, ev}`` would put two ``kind`` keys
in one flat object, so the contact kind is the field ``contact`` here.

Pure: standard library only, so Duck Core, duck-body, duck-eye, the world and the tests can all
import it (``tests/test_imports.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

PERCEPT_API = "percept-v1"

SOURCES = ("robot", "live", "surrogate", "legacy", "replay")
PV_PATTERN = re.compile(r"^[pcl]:[0-9a-f]{8,64}$")
ACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
PPM = 1_000_000
STAMPS = ("kind", "src", "t_ms", "pv")

LIGHT_LEVELS = ("bright", "dim", "dark", "black", "none")
FREE_ENDS = ("wall", "thing", "drop", "trip", "unknown")
WALK_REASONS = (
    "time",
    "reflex_near",
    "reflex_drop",
    "reflex_touch",
    "reflex_unloaded",
    "reflex_dark",
    "stall",
    "stumble",
    "fell",
    "lifted",
    "veto_look",
    "veto_dark",
    "veto_head",
    "preempt",
    "deadline",
)
CONTACT_KINDS = ("blocked", "brushed", "stumbled")
SIDES = ("left", "right", "front", "unknown")
POSTURES = ("stand", "sit", "unknown")
FELL_DURING = ("walk", "posture", "still", "curio", "dock", "look")
FELL_TRIGGERS = ("fallen", "tilt40")
RIGHTING_VIA = ("self", "help")
RIGHTING_HOW = ("wait", "sitstand")
SIGHTING_WHAT = ("nest", "duck", "thing", "person")
SIGHTING_EDGES = ("new", "moved", "lost")
RANGE_SOURCES = ("tag", "panel", "ground", "size", "tof", "cue")
SECTOR_SOURCES = ("cam", "tof", "both")
DOCK_VIA = ("pg", "trend", "none")
BATTERY_TRENDS = ("charging", "draining", "steady")
GUARD_KINDS = ("cliff", "spit", "tethered", "overload", "hot")
HEAD_MODES = ("home", "down")
ATTENTION_MODES = ("on", "low", "elsewhere")  # walk.done.attention (duck-body WalkCmd.attention)
TOUCH_PATHS = ("ear", "joint")  # sense.touch.path: the ear (a sound) or a joint (a push)
TOUCH_MAX_S = 86_400  # sense.touch.seconds: at most a day


class PerceptError(ValueError):
    """A percept that breaks the percept-v1 contract."""


# ── the schema language ───────────────────────────────────────────────────────
# A checker takes (value, path) and returns a list of problems ("" = none).

Checker = Callable[[Any, str], list[str]]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def integer(lo: int | None = None, hi: int | None = None) -> Checker:
    def check(value: Any, path: str) -> list[str]:
        if not _is_int(value):
            return [f"{path}: expected an integer, got {type(value).__name__}"]
        if lo is not None and value < lo:
            return [f"{path}: {value} < {lo}"]
        if hi is not None and value > hi:
            return [f"{path}: {value} > {hi}"]
        return []

    return check


def text(pattern: re.Pattern[str] | None = None) -> Checker:
    def check(value: Any, path: str) -> list[str]:
        if not isinstance(value, str) or not value:
            return [f"{path}: expected non-empty text"]
        if pattern is not None and not pattern.match(value):
            return [f"{path}: {value!r} does not match {pattern.pattern}"]
        return []

    return check


def boolean() -> Checker:
    def check(value: Any, path: str) -> list[str]:
        return [] if isinstance(value, bool) else [f"{path}: expected true/false"]

    return check


def one_of(options: Iterable[str]) -> Checker:
    allowed = tuple(options)

    def check(value: Any, path: str) -> list[str]:
        return [] if value in allowed else [f"{path}: {value!r} not in {allowed}"]

    return check


def nullable(inner: Checker) -> Checker:
    def check(value: Any, path: str) -> list[str]:
        return [] if value is None else inner(value, path)

    return check


def seq(inner: Checker, length: int | None = None, max_len: int = 256) -> Checker:
    def check(value: Any, path: str) -> list[str]:
        if not isinstance(value, list):
            return [f"{path}: expected a list"]
        if length is not None and len(value) != length:
            return [f"{path}: expected {length} items, got {len(value)}"]
        if len(value) > max_len:
            return [f"{path}: more than {max_len} items"]
        problems: list[str] = []
        for index, item in enumerate(value):
            problems += inner(item, f"{path}[{index}]")
        return problems

    return check


def record(
    required: Mapping[str, Checker],
    optional: Mapping[str, Checker] | None = None,
    *,
    open_ended: bool = False,
) -> Checker:
    """An object with these fields. ``open_ended`` lets unknown fields through (still checked
    for the integer rule by :func:`validate`)."""
    optional = dict(optional or {})

    def check(value: Any, path: str) -> list[str]:
        if not isinstance(value, dict):
            return [f"{path}: expected an object"]
        problems: list[str] = []
        for name, inner in required.items():
            if name not in value:
                problems.append(f"{path}.{name}: missing")
            else:
                problems += inner(value[name], f"{path}.{name}")
        for name, inner in optional.items():
            if name in value:
                problems += inner(value[name], f"{path}.{name}")
        if not open_ended:
            extra = sorted(set(value) - set(required) - set(optional))
            problems += [f"{path}.{name}: unknown field" for name in extra]
        return problems

    return check


def anything() -> Checker:
    return lambda value, path: []


# ── field shapes shared by several kinds ──────────────────────────────────────

MM = integer(0)
SIGNED = integer()
CONF = integer(0, PPM)
PCT = integer(0, 100)
ACT_ID = text(ACT_ID_PATTERN)
LOOK_ID = text(ACT_ID_PATTERN)

LOOK_FOLD = record(
    {
        "look_id": LOOK_ID,
        "light": one_of(LIGHT_LEVELS),
        "free_mm": seq(MM, 5),
        "ends": seq(one_of(FREE_ENDS), 5),
    },
    {
        "trip": nullable(record({"mm": MM, "mrad": SIGNED})),
        "drop": integer(0, 64),
    },
)

SIGHTING_BODY: dict[str, Checker] = {
    "what": one_of(SIGHTING_WHAT),
    "b_mrad": SIGNED,
    "conf_ppm": CONF,
}
SIGHTING_OPTIONAL: dict[str, Checker] = {
    "id": nullable(text()),
    "e_mrad": SIGNED,
    "r_mm": nullable(MM),
    "r_src": one_of(RANGE_SOURCES),
    "yaw_mrad": nullable(SIGNED),
    "light": one_of(LIGHT_LEVELS),
    "look_id": LOOK_ID,
}

SCHEMAS: dict[str, Checker] = {
    "walk.done": record(
        {
            "act_id": ACT_ID,
            "cmd": record({"vx": SIGNED, "vyaw": SIGNED, "max_ms": MM}),
            "odom": record({"dx_mm": SIGNED, "dy_mm": SIGNED, "dyaw_mrad": SIGNED, "path_mm": MM}),
            "steps": MM,
            "reason": one_of(WALK_REASONS),
        },
        {
            "pre": nullable(LOOK_FOLD),
            "end": nullable(LOOK_FOLD),
            "contact": nullable(record({"kind": one_of(CONTACT_KINDS), "side": one_of(SIDES)})),
            "battery_pct": nullable(PCT),
            "purpose": text(),
            "head": one_of(HEAD_MODES),
            "stop_mm": nullable(MM),
            "attention": one_of(ATTENTION_MODES),
        },
    ),
    "sense.scan": record(
        {
            "act_id": ACT_ID,
            "head_mrad": seq(SIGNED, max_len=16),
            "light": one_of(LIGHT_LEVELS),
            "sectors": seq(
                record(
                    {
                        "b_mrad": SIGNED,
                        "w_mrad": MM,
                        "free_mm": MM,
                        "end": one_of(FREE_ENDS),
                        "conf_ppm": CONF,
                        "src": one_of(SECTOR_SOURCES),
                    }
                ),
                max_len=32,
            ),
            "sightings": seq(record(SIGHTING_BODY, SIGHTING_OPTIONAL), max_len=32),
        }
    ),
    "sense.sighting": record(
        {**SIGHTING_BODY, "look_id": LOOK_ID, "edge": one_of(SIGHTING_EDGES)},
        {**{k: v for k, v in SIGHTING_OPTIONAL.items() if k != "look_id"}, "act_id": ACT_ID},
    ),
    "sense.light": record(
        {"level": one_of(LIGHT_LEVELS), "luma": integer(0, 255), "noise_q": MM},
        {"why": one_of(("change", "periodic"))},
    ),
    "sense.depth": record(
        {
            "ahead_mm": nullable(MM),
            "left_mm": nullable(MM),
            "right_mm": nullable(MM),
            "drop_zones": integer(0, 64),
            "touch": boolean(),
            "head_pitch_mrad": SIGNED,
        }
    ),
    "sense.contact": record(
        {
            "contact": one_of(CONTACT_KINDS),
            "side": one_of(SIDES),
            "ev": seq(text(), max_len=8),
        }
    ),
    "sense.lifted": record(
        {"conf_ppm": CONF, "posture": one_of(POSTURES)}, {"on_nest": nullable(boolean())}
    ),
    "sense.set_down": record(
        {"conf_ppm": CONF, "posture": one_of(POSTURES)}, {"on_nest": nullable(boolean())}
    ),
    "body.fell": record(
        {
            "during": one_of(FELL_DURING),
            "ctx": record(
                {},
                {
                    "look_id": nullable(LOOK_ID),
                    "act_id": nullable(ACT_ID),
                    "last_trip": nullable(MM),
                    "last_drop": nullable(MM),
                },
            ),
        },
        {"trigger": one_of(FELL_TRIGGERS)},
    ),
    "body.righting": record(
        {"ok": boolean(), "ms": MM, "tries": integer(0, 10)},
        {
            "via": one_of(RIGHTING_VIA),
            "how": one_of(RIGHTING_HOW),
            "hard": boolean(),
            "overload": boolean(),  # try 2 stopped: a leg motor past its stall too long
        },
    ),
    "body.posture": record(
        {"want": one_of(("stand", "sit")), "got": one_of(POSTURES), "ms": MM, "fell": boolean()},
        {"act_id": ACT_ID},
    ),
    "body.dock": record(
        {
            "contact": boolean(),
            "via": one_of(DOCK_VIA),
            "tries": integer(0, 20),
            "seen_tag": boolean(),
        },
        {"act_id": ACT_ID, "style": text()},
    ),
    "body.battery": record({"pct": PCT, "trend": one_of(BATTERY_TRENDS)}),
    "body.hot": record({"soc_c": SIGNED, "tier": integer(0, 3)}, {"ceiling_mhz": nullable(MM)}),
    # Curiosity (curiosity spec §5.8, §6.1, §7.2): the curiosity team owns the field lists;
    # percept-v1 pins the ids and the integer rule.
    "body.curio_step": record({"id": ACT_ID, "step": MM}, {"act_id": ACT_ID}, open_ended=True),
    "sense.thing": record({"things": seq(anything(), max_len=64)}, open_ended=True),
    # the curiosity spec names the guard's field ``kind``, which is the event's own kind: one
    # JSON object cannot carry both, so percept-v1 calls it ``guard`` (as sense.contact's is
    # ``contact``)
    "body.guard": record({"guard": one_of(GUARD_KINDS)}, {"act_id": ACT_ID}, open_ended=True),
    "env.cue": record({"cue": text()}, open_ended=True),
    # Phase 4 (WP9): the hands and the gamepad, as the world hands them to Duck Core
    # (``legacy_phantom``: LegacyWorld's transitional mark on the TV's false touch, design A10)
    "sense.touch": record(
        {
            "where": text(),
            "seconds": integer(0, TOUCH_MAX_S),
            "firmness": text(),
            "path": one_of(TOUCH_PATHS),
            "latency_ms": MM,
        },
        {"legacy_phantom": boolean()},
    ),
    "sense.hold_head": record({"path": text(), "moving": boolean()}),
    "body.play": record({"policy": text(), "meters": MM}, {"skill": text(), "minute": MM}),
}

KINDS = tuple(SCHEMAS)


def known_kind(kind: str) -> bool:
    """Whether ``kind`` is a percept-v1 kind."""
    return kind in SCHEMAS


def _integer_rule(value: Any, path: str) -> list[str]:
    """No floats anywhere, no NaN, only JSON-shaped containers."""
    if isinstance(value, float):
        return [f"{path}: floats are not allowed in percept-v1 ({value!r})"]
    if isinstance(value, dict):
        problems: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{path}: non-text key {key!r}")
            problems += _integer_rule(item, f"{path}.{key}")
        return problems
    if isinstance(value, (list, tuple)):
        problems = []
        for index, item in enumerate(value):
            problems += _integer_rule(item, f"{path}[{index}]")
        return problems
    if value is None or isinstance(value, (bool, int, str)):
        return []
    return [f"{path}: {type(value).__name__} is not JSON"]


def validate(event: Mapping[str, Any]) -> list[str]:
    """Every problem with ``event`` (empty when it is a valid percept-v1 event)."""
    if not isinstance(event, Mapping):
        return ["event: expected an object"]
    problems = _integer_rule(dict(event), "event")
    kind = event.get("kind")
    if kind not in SCHEMAS:
        return problems + [f"event.kind: unknown kind {kind!r}"]
    if event.get("src") not in SOURCES:
        problems.append(f"event.src: {event.get('src')!r} not in {SOURCES}")
    t_ms = event.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool) or t_ms < 0:
        problems.append("event.t_ms: expected a non-negative integer")
    pv = event.get("pv")
    if not isinstance(pv, str) or not PV_PATTERN.match(pv):
        problems.append(f"event.pv: {pv!r} is not p:/c:/l: plus 8-64 hex digits")
    body = {key: value for key, value in event.items() if key not in STAMPS}
    problems += SCHEMAS[str(kind)](body, "event")
    return problems


def check(event: Mapping[str, Any]) -> dict[str, Any]:
    """``event`` as a plain dict, or :class:`PerceptError` listing every problem."""
    problems = validate(event)
    if problems:
        raise PerceptError("; ".join(problems))
    return dict(event)


def make(kind: str, *, src: str, t_ms: int, pv: str, **fields: Any) -> dict[str, Any]:
    """Build and validate one event: ``make("sense.light", src="live", t_ms=…, pv=…, level=…)``."""
    return check({"kind": kind, "src": src, "t_ms": t_ms, "pv": pv, **fields})


def encode(event: Mapping[str, Any]) -> str:
    """The canonical one-line JSON (sorted keys, no spaces, UTF-8 text kept)."""
    return json.dumps(
        event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def decode(line: str | bytes) -> dict[str, Any]:
    """Parse one line and validate it."""
    value = json.loads(line)
    if not isinstance(value, dict):
        raise PerceptError("a percept line must be a JSON object")
    return check(value)


def due_second(t_ms: int) -> int:
    """The whole second Duck Core takes the percept in (``ceil(t_ms / 1000)``, §7.3 rule 4)."""
    return -(-int(t_ms) // 1000)


def perception_hash(parts: Mapping[str, str | bytes]) -> str:
    """``p:`` + 16 hex digits over the named parts (code, weights, thresholds, K, distortion):
    the same parts in any order give the same hash."""
    digest = hashlib.sha256()
    for name in sorted(parts):
        value = parts[name]
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        digest.update(name.encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0" + data)
    return "p:" + digest.hexdigest()[:16]


# ── quantisation (round half to even, like Python's round; non-finite is refused) ──


def _quantise(value: float, scale: float) -> int:
    if not math.isfinite(value):
        raise PerceptError(f"cannot quantise {value!r}")
    return int(round(value * scale))


def mm(metres: float) -> int:
    return _quantise(metres, 1000.0)


def mrad(radians: float) -> int:
    return _quantise(radians, 1000.0)


def ppm(probability: float) -> int:
    return max(0, min(PPM, _quantise(probability, float(PPM))))


def pct(fraction_or_percent: float, *, is_fraction: bool = False) -> int:
    value = fraction_or_percent * 100.0 if is_fraction else fraction_or_percent
    return max(0, min(100, _quantise(value, 1.0)))


__all__ = [
    "ACT_ID_PATTERN",
    "ATTENTION_MODES",
    "BATTERY_TRENDS",
    "CONTACT_KINDS",
    "FELL_DURING",
    "FELL_TRIGGERS",
    "FREE_ENDS",
    "KINDS",
    "LIGHT_LEVELS",
    "PERCEPT_API",
    "POSTURES",
    "PerceptError",
    "RIGHTING_HOW",
    "SCHEMAS",
    "SIDES",
    "SOURCES",
    "TOUCH_MAX_S",
    "TOUCH_PATHS",
    "WALK_REASONS",
    "check",
    "decode",
    "due_second",
    "encode",
    "known_kind",
    "make",
    "mm",
    "mrad",
    "perception_hash",
    "pct",
    "ppm",
    "validate",
]
