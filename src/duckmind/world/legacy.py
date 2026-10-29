"""LegacyWorld: v4's physics, outside the mind (physical-design spec §7.4, §17 期4; phase-4
design (measurement records, not in the repo) §6, decisions A1–A3, A10, A12; the contract is
``docs/design/phase4-boundary.md``).

Duck Core v4 rolled the world's dice itself: whether a walk ends in a fall and when, each
self-righting try, whether it stands up, whether it sits down onto the nest's contacts, whether
a touch or a hand on its head is detected, the TV heard as a touch, and what the gamepad makes
of a minute of play. Phase 4 moves those dice out here **unchanged**. The world draws with Duck
Core's own key — ``rng.draw(seed, table_hash, purpose, counter)``, one counter per purpose —
so every relocated draw returns exactly the value the v4 duck would have drawn, and a v4_1 duck
with a LegacyWorld behaves bit-for-bit like v4.

The truth here is the duck's belief (``bel_from``, ``expect`` and ``bel_pose`` are taken as
where it really is): LegacyWorld is the v4 physics, not a model of a room. Phase 5 replaces it.

**The port** (:class:`duckmind.worldport.WorldPort`; Duck Core *pulls*, A1):

* :meth:`LegacyWorld.command` takes Duck Core's command wire dicts. What takes no time is
  answered in the same call (A2): a stand-up (``body.posture``), a dock try (``body.dock``), the
  nest look (``sense.sighting`` / ``sense.scan``) and a stop (the stopped walk's ``walk.done
  {reason: lifted|preempt}``). A walk is answered when it ends (``walk.done {reason: time}``)
  or falls (``body.fell`` then ``walk.done {reason: fell}`` at the fall's second; a fall at the
  last second wins, as v4's ``if``/``elif``). After a fall the world tries to roll back up: the
  first try 2 s later, then every 4 s after a failed try, at most 3, one ``body.righting`` per
  try (A12).
* :meth:`next_at_ms` / :meth:`due` hand over what is scheduled, in time order (at one second:
  a fall, then a walk's end, then a righting try).
* :meth:`poll` answers the per-minute questions only this world has: ``tv_minute`` (the TV
  heard as a touch, ``legacy_phantom: true``, A10) and ``play_minute`` (the gamepad's metres
  and, every third minute, its skill).
* :meth:`adopt` is the migration seam (WP13): a fresh world takes over a v4 duck's outcome
  counters and any walk or righting in progress, so the duck goes on as v4 would have.
* The family's hands (the api's family proxy calls them, only while Duck Core is awake, as v4
  drew only then): :meth:`touch` and :meth:`hold_head` are the detectors (a miss is ``None``);
  :meth:`set_setting` keeps the detectors' settings (``conservative``, ``ear``); :meth:`tv`
  and :meth:`hand` are kept on the tape for the truth stream.

**Stopping.** Only Duck Core knows whether a v4 hand reached the body (it may be switched off,
or have no pickup detection); v4 cuts a walk or a righting exactly where it sets a posture. So
the schedules end by Duck Core's own ``duck.stop``, never by :meth:`hand`: a stop naming the
walk's ``act_id`` ends that walk at once; a stop naming no act ends whatever the body is doing
(the walk, and every righting try still to come).

**The tape.** Every call that changes the world (commands, ``due`` calls that delivered
something, polls, the family's hands and settings) is kept with its millisecond, and so is every
percept it sent. :func:`replay` feeds a tape to a fresh world built from the same header; it
must send the same percepts (``world_replay_check``).

**Header and hash.** ``{"world": "legacy", "v": 1, "seed", "table_hash", "rates": {…},
"marker", "sensors": {"ear", "conservative"}}``; every percept is stamped ``src: "legacy"``,
``pv: "l:" + worldport.calib_hash(header)``. :func:`header_from_v4_config` builds the header of
an old v4 ``Config`` (its rates, ``camera`` as the marker, ``ear``, ``conservative``).

**v1 is pinned.** A v4_1 duck is Duck Core *and* this world: its saved history replays only if
both stay as they are. So what ``"v": 1`` does never changes (``tests/golden_v4_1/`` replays
v4_1 ducks with their worlds on every run); a change of behaviour is a new header version (or
phase 5's LiveSimWorld) and a new engine revision, after freezing this one (boundary §11).

Imports only Duck Core's public maths (``fixed``, ``rng``) and :mod:`duckmind.worldport`
(``tests/test_imports.py``); registers itself with :func:`duckmind.worldport.register_world`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from duckmind import worldport
from duckmind.duckcore import rng
from duckmind.duckcore.fixed import (
    SCALE,
    atan2_mrad,
    clamp,
    cos_mrad,
    isqrt,
    signed_mrad,
    sin_mrad,
)

WORLD = "legacy"
HEADER_V = 1
SRC = "legacy"
PPM = 1_000_000

# The outcome purposes that left Duck Core (design §4); each keeps its own counter here.
PURPOSES = (
    "fall",
    "fall.at",
    "righting",
    "stand.try",
    "dock.try",
    "detect.touch",
    "detect.hold",
    "false_alarm.tv",
    "play.meters",
    "play.skill",
)
# v4's rates (HardwareConfig and api.Config defaults) for a header built from an old Config.
V4_RATES = {
    "fall_ppm_per_s": 50,
    "fall_ppm_per_start": 0,
    "righting_ppm": 600_000,
    "stand_ppm": 1_000_000,
    "dock_ppm": 850_000,
}
# v4 constants, verbatim (engine.py, nest.py).
RIGHTING_TRIES = 3
RIGHTING_FIRST_S = 2
RIGHTING_GAP_S = 4
TV_FALSE_TOUCH = 33  # of SCALE per TV minute heard by an ear (engine._tv_minute)
PLAY_SKILLS = ("walk", "kick_left", "roulade", "bow", "sit_toggle")
DOCK_ERR_BLIND_MM = 800  # the dock chance falls linearly to 0 at 800 mm of pose error
STEPS_PER_S_X10 = 46  # phase-1 footfalls per walking second (4.6), for walk.done.steps
CONF_CERTAIN = PPM
# A stop whose ``why`` means the body left the floor ends its walk ``lifted``, else ``preempt``.
LIFTED = frozenset({"lifted", "pickup", "picked_up", "placed"})
# Transitional truth (the only-shrink list; phase 5f empties it): the TV's touch says it was
# no hand (A10); one percept per righting try (A12); the nest look answers from the belief; the
# dock's chance follows the pose error the duck believes it has; a walk's odometry is the
# duck's own prediction (expect − bel_from).
LEGACY_TRUTH = (
    "sense.touch.legacy_phantom",
    "body.righting per try",
    "sense.sighting from bel_pose",
    "body.dock from bel_err_mm",
    "walk.done.odom from expect",
)

_NO_ARGS: dict[str, Any] = {}  # (shared, never changed)


class LegacyWorldError(ValueError):
    """A header or a call LegacyWorld cannot take."""


# ── header ────────────────────────────────────────────────────────────────────


def make_header(
    *,
    seed: int,
    table_hash: str,
    fall_ppm_per_s: int = V4_RATES["fall_ppm_per_s"],
    fall_ppm_per_start: int = V4_RATES["fall_ppm_per_start"],
    righting_ppm: int = V4_RATES["righting_ppm"],
    stand_ppm: int = V4_RATES["stand_ppm"],
    dock_ppm: int = V4_RATES["dock_ppm"],
    marker: bool = True,
    ear: bool = True,
    conservative: bool = True,
) -> dict[str, Any]:
    """A checked LegacyWorld header."""
    return check_header(
        {
            "world": WORLD,
            "v": HEADER_V,
            "seed": seed,
            "table_hash": table_hash,
            "rates": {
                "fall_ppm_per_s": fall_ppm_per_s,
                "fall_ppm_per_start": fall_ppm_per_start,
                "righting_ppm": righting_ppm,
                "stand_ppm": stand_ppm,
                "dock_ppm": dock_ppm,
            },
            "marker": marker,
            "sensors": {"ear": ear, "conservative": conservative},
        }
    )


def header_from_v4_config(config: Mapping[str, Any] | object, *, table_hash: str) -> dict[str, Any]:
    """The header of the world an old v4 duck lived in, from its ``Config`` (the dict a save
    keeps, or the dataclass): the rates it carried (v4's defaults for any it lacks), ``camera``
    as the marker (v4's api: the same head camera sees the nest's marker), ``ear`` and the
    ``conservative`` mode its detectors started in. ``table_hash``: the content table the duck
    hatched with (Duck Core's ``content.table_hash()``)."""
    if is_dataclass(config) and not isinstance(config, type):
        config = {f.name: getattr(config, f.name) for f in fields(config)}

    def get(name: str, default: Any) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    return make_header(
        seed=int(get("seed", 7)),
        table_hash=table_hash,
        marker=bool(get("camera", True)),
        ear=bool(get("ear", True)),
        conservative=bool(get("conservative", True)),
        **{name: int(get(name, default)) for name, default in V4_RATES.items()},
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_header(header: Mapping[str, Any]) -> dict[str, Any]:
    """A plain, checked copy of ``header`` (else :class:`LegacyWorldError`). ``sensors.
    conservative`` may be left out (the api then mirrors the setting): it starts ``true``."""
    if not isinstance(header, Mapping):
        raise LegacyWorldError("the header must be an object")
    out: dict[str, Any] = json.loads(worldport.canonical(dict(header)))
    problems = []
    if out.get("world") != WORLD:
        problems.append(f"world: {out.get('world')!r} is not {WORLD!r}")
    if out.get("v") != HEADER_V:
        problems.append(f"v: {out.get('v')!r} is not {HEADER_V}")
    if not _is_int(out.get("seed")):
        problems.append("seed: expected an integer")
    if not isinstance(out.get("table_hash"), str) or not out.get("table_hash"):
        problems.append("table_hash: expected text")
    rates = out.get("rates")
    if not isinstance(rates, dict) or set(rates) != set(V4_RATES):
        problems.append(f"rates: expected exactly {sorted(V4_RATES)}")
    else:  # (any integer, as v4 took any: a chance is compared with a draw, never clamped)
        problems += [f"rates.{k}: expected an integer" for k, v in rates.items() if not _is_int(v)]
    if not isinstance(out.get("marker"), bool):
        problems.append("marker: expected true/false")
    sensors = out.get("sensors")
    if (
        not isinstance(sensors, dict)
        or "ear" not in sensors
        or not set(sensors) <= {"ear", "conservative"}
        or not all(isinstance(value, bool) for value in sensors.values())
    ):
        problems.append("sensors: expected {ear: bool, conservative?: bool}")
    extra = sorted(set(out) - {"world", "v", "seed", "table_hash", "rates", "marker", "sensors"})
    if extra:
        problems.append(f"unknown header fields {extra}")
    if problems:
        raise LegacyWorldError("; ".join(problems))
    return out


# ── commands ──────────────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """A frozen event payload (mapping proxies, tuples) as plain JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _milli(value: Any) -> int:
    """A length or angle: an integer is mm/mrad already, a float is m/rad (v4's page form)."""
    if isinstance(value, float):
        return int(round(value * 1000))
    return int(value)


_POSE_KEYS = (
    ("x_mm", "y_mm", "yaw_mrad"),
    ("x_mm", "y_mm", "heading_mrad"),
    ("x", "y", "heading"),
    ("x", "y", "yaw"),
)


def _pose(value: Any) -> list[int] | None:
    """[x mm, y mm, heading mrad] from ``{x_mm, y_mm, yaw_mrad}`` (the contract), or ``[x, y,
    h]`` (floats: v4's ``_pose_list`` in m and rad; integers: mm and mrad)."""
    if value is None:
        return None
    if type(value) is dict and "x_mm" in value and "yaw_mrad" in value:  # the contract's form
        return [int(value["x_mm"]), int(value["y_mm"]), int(value["yaw_mrad"])]
    if isinstance(value, Mapping):
        value = _plain(value)
        for keys in _POSE_KEYS:
            if all(key in value for key in keys):
                return [_milli(value[key]) for key in keys]
        raise LegacyWorldError(f"a pose needs x, y and a heading: {dict(value)!r}")
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 3:
        return [_milli(item) for item in value]
    raise LegacyWorldError(f"not a pose: {value!r}")


def normalize_command(command: Any) -> dict[str, Any] | None:
    """The fields of one command this world reads, as a plain dict — or ``None`` for anything
    that is not a command to the body. Takes the contract's wire dict (``{"cmd": …, **payload}``),
    and also a Duck Core event (``.kind`` and ``.payload``) or ``{"kind", "payload"}``."""
    kind: str
    payload: Any
    if type(command) is dict:  # the contract's wire dict (the hot path: a walk every few s)
        if command.get("cmd") == "duck.walk":
            fast = _wire_walk(command)
            if fast is not None:
                return fast
    if type(command) is dict and "cmd" in command:
        kind, payload = "wire", command
    elif hasattr(command, "kind") and hasattr(command, "payload"):
        kind, payload = str(command.kind), command.payload
    elif isinstance(command, Mapping) and "cmd" in command:
        kind, payload = "wire", command
    elif isinstance(command, Mapping) and "kind" in command and "payload" in command:
        kind, payload = str(command["kind"]), command["payload"]
    else:
        raise LegacyWorldError(f"not a command: {command!r}")
    if not isinstance(payload, Mapping):
        return None
    name = payload.get("cmd")
    if kind == "duck.walk":
        name = "duck.walk"
    elif kind not in ("duck.act", "duck.curio", "wire") or not isinstance(name, str):
        return None  # a record (v4's duck.posture and duck.stop records are not commands)
    act_id = payload.get("act_id")
    if name != "duck.stop" and not isinstance(act_id, str):
        return None  # a command is an event with an act_id (A7)
    if name == "duck.walk":
        max_ms = payload.get("max_ms")
        vx = payload.get("vx_mm")
        vyaw = payload.get("vyaw_mrad")
        bel_from = payload.get("bel_from")
        expect = payload.get("expect")
        walk: dict[str, Any] = {
            "cmd": name,
            "act_id": act_id,
            "seconds": int(max_ms) // 1000 if max_ms is not None else int(payload["seconds"]),
            "vx_mm": int(vx) if vx is not None else _milli(payload.get("vx", 0)),
            "vyaw_mrad": int(vyaw) if vyaw is not None else _milli(payload.get("vyaw", 0)),
            "bel_from": _pose(bel_from if bel_from is not None else payload.get("from")),
            "expect": _pose(expect if expect is not None else payload.get("to")),
        }
        cm = payload.get("cm")
        if cm is not None:
            walk["path_mm"] = int(cm) * 10
        purpose = payload.get("purpose")
        if isinstance(purpose, str) and purpose:
            walk["purpose"] = purpose
        return walk
    out: dict[str, Any] = {"cmd": name, "act_id": act_id if isinstance(act_id, str) else None}
    if name == "duck.posture":
        out["want"] = "sit" if payload.get("want") == "sit" else "stand"
    elif name == "duck.dock":
        out["bel_err_mm"] = int(payload.get("bel_err_mm", 0))
    elif name == "duck.look":
        out["kind"] = str(payload.get("kind", "scan"))
        out["want"] = str(payload.get("want", "none"))
        out["bel_pose"] = _pose(payload.get("bel_pose"))
    elif name == "duck.stop":
        out["why"] = str(payload.get("why", "stop"))
    return out


def _wire_walk(wire: dict[str, Any]) -> dict[str, Any] | None:
    """:func:`normalize_command` for the contract's ``duck.walk`` wire dict (integers, poses as
    ``{x_mm, y_mm, yaw_mrad}``); ``None`` sends anything else down the general path."""
    act_id = wire.get("act_id")
    max_ms, vx, vyaw = wire.get("max_ms"), wire.get("vx_mm"), wire.get("vyaw_mrad")
    start, end = wire.get("bel_from"), wire.get("expect")
    if (
        type(act_id) is not str
        or type(max_ms) is not int
        or type(vx) is not int
        or type(vyaw) is not int
        or type(start) is not dict
        or type(end) is not dict
    ):
        return None
    try:
        bel_from = [start["x_mm"], start["y_mm"], start["yaw_mrad"]]
        expect = [end["x_mm"], end["y_mm"], end["yaw_mrad"]]
    except KeyError:
        return None
    walk: dict[str, Any] = {
        "cmd": "duck.walk",
        "act_id": act_id,
        "seconds": max_ms // 1000,
        "vx_mm": vx,
        "vyaw_mrad": vyaw,
        "bel_from": bel_from,
        "expect": expect,
    }
    cm = wire.get("cm")
    if cm is not None:
        walk["path_mm"] = int(cm) * 10
    purpose = wire.get("purpose")
    if purpose and type(purpose) is str:
        walk["purpose"] = purpose
    return walk


def _odom(
    bel_from: Sequence[int] | None, expect: Sequence[int] | None, part: int, whole: int
) -> dict[str, int]:
    """``expect − bel_from`` in the frame the walk started in, scaled by ``part / whole``."""
    if bel_from is None or expect is None or whole <= 0:
        return {"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0}
    fx, fy, fh = bel_from
    ex, ey, eh = expect
    dx, dy = ex - fx, ey - fy
    c, s = cos_mrad(fh), sin_mrad(fh)
    ahead = (c * dx + s * dy + SCALE // 2) // SCALE  # (rounded: cos 0 is 9999/10000 here)
    left = (c * dy - s * dx + SCALE // 2) // SCALE
    return {
        "dx_mm": ahead * part // whole,
        "dy_mm": left * part // whole,
        "dyaw_mrad": signed_mrad(eh - fh) * part // whole,
    }


# ── the world ─────────────────────────────────────────────────────────────────


class LegacyWorld:
    """v4's dice and timed outcomes (see the module docstring). One per duck.

    ``keep_tape=False`` keeps no tape and no sent list (a long-lived page duck that will not
    be world-replayed: about 1 KB per percept otherwise). ``validate_first=n`` checks only the
    first ``n`` percepts against percept-v1, ``None`` checks every one; by default (``-1``) the
    class's :attr:`VALIDATE_FIRST` (design A13: 1,000 per duck in population runs; tests set it
    to ``None`` to check all).
    """

    src = SRC
    VALIDATE_FIRST: int | None = 1_000
    KEEP_TAPE = True

    def __init__(
        self,
        header: Mapping[str, Any],
        *,
        keep_tape: bool | None = None,
        validate_first: int | None = -1,
    ) -> None:
        self._header = check_header(header)
        self.calib_hash = worldport.calib_hash(self._header)
        self.pv = "l:" + self.calib_hash
        self.seed = int(self._header["seed"])
        self.table = str(self._header["table_hash"])
        rates = self._header["rates"]
        self.fall_ppm_per_s = int(rates["fall_ppm_per_s"])
        self.fall_ppm_per_start = int(rates["fall_ppm_per_start"])
        self.righting_ppm = int(rates["righting_ppm"])
        self.stand_ppm = int(rates["stand_ppm"])
        self.dock_ppm = int(rates["dock_ppm"])
        self.marker = bool(self._header["marker"])
        sensors = self._header["sensors"]
        self.ear = bool(sensors["ear"])
        self.conservative = bool(sensors.get("conservative", True))
        self.tv_on = False
        self.counters: dict[str, int] = {}
        # The schedule: at most one walk (its end, and its fall if it falls) and one righting
        # (its next try) are ever pending; at one second a fall comes first, then a walk's end,
        # then a righting try.
        self._walk: dict[str, Any] | None = None  # the walk in progress (its command)
        self._walk_start = 0
        self._walk_end_ms = 0
        self._fall_ms: int | None = None
        self._righting: dict[str, Any] | None = None  # the righting in progress
        self._righting_ms = 0
        self.keep_tape = self.KEEP_TAPE if keep_tape is None else keep_tape
        self.tape: list[tuple[int, str, dict[str, Any]]] = []
        self.sent: list[dict[str, Any]] = []
        self._validate_left = self.VALIDATE_FIRST if validate_first == -1 else validate_first

    @classmethod
    def from_header(cls, header: Mapping[str, Any], **options: Any) -> LegacyWorld:
        return cls(header, **options)

    def header(self) -> dict[str, Any]:
        return json.loads(worldport.canonical(self._header))  # type: ignore[no-any-return]

    # ---------------------------------------------------------------- plumbing

    def _rand(self, purpose: str) -> int:
        counter = self.counters.get(purpose, 0) + 1
        self.counters[purpose] = counter
        return rng.draw(self.seed, self.table, purpose, counter)

    def _hit(self, purpose: str, chance_ppm: int) -> bool:
        """One draw of ``purpose`` that succeeds with ``chance_ppm`` (v4's ppm comparison)."""
        return (self._rand(purpose) * PPM) >> 64 < chance_ppm

    def _record(self, t_ms: int, call: str, args: dict[str, Any]) -> None:
        if self.keep_tape:
            self.tape.append((t_ms, call, args))

    def _sent(self, out: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.keep_tape and out:
            self.sent.extend(out)
        return out

    def _percept(self, kind: str, t_ms: int, body: dict[str, Any]) -> dict[str, Any]:
        event = {"kind": kind, "src": SRC, "t_ms": t_ms, "pv": self.pv}
        event.update(body)
        if self._validate_left is None:
            worldport.check(event)
        elif self._validate_left > 0:
            self._validate_left -= 1
            worldport.check(event)
        return event

    # ---------------------------------------------------------------- WorldPort

    def command(self, commands: Iterable[Any], now_ms: int) -> list[dict[str, Any]]:
        """Duck Core's commands, issued at ``now_ms`` in order. Returns what the world senses
        in this same call (answers that take no time); a walk's end or fall comes later."""
        now_ms = int(now_ms)
        out: list[dict[str, Any]] = []
        for command in commands:
            cmd = normalize_command(command)
            if cmd is not None:
                answer = self._take(cmd, now_ms)
                if answer:
                    out += answer
        if out and self.keep_tape:
            self.sent.extend(out)
        return out

    def _take(self, cmd: dict[str, Any], now_ms: int) -> list[dict[str, Any]]:
        """One normalized command (:func:`normalize_command`)."""
        if self.keep_tape:
            self.tape.append((now_ms, "command", cmd))
        name = cmd["cmd"]
        if name == "duck.walk":
            return self._start_walk(cmd, now_ms)
        if name == "duck.stop":
            return self._stop(cmd, now_ms)
        if name == "duck.posture":
            return [self._percept("body.posture", now_ms, self._posture(cmd))]
        if name == "duck.dock":
            return [self._percept("body.dock", now_ms, self._dock(cmd))]
        if name == "duck.look":
            return [self._look(cmd, now_ms)]
        return []  # not a v4 act (duck.curio …): no answer; Duck Core's deadline says so

    def next_at_ms(self) -> int | None:
        """When the next scheduled percept falls due (``None``: nothing is scheduled)."""
        if self._walk is not None:
            first = self._walk_end_ms if self._fall_ms is None else self._fall_ms
            if self._righting is not None and self._righting_ms < first:
                return self._righting_ms
            return first
        return self._righting_ms if self._righting is not None else None

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        """Every scheduled percept due by ``now_ms``, earliest first (outcomes are drawn now,
        as v4 drew them when their moment came)."""
        now_ms = int(now_ms)
        out: list[dict[str, Any]] = []
        while True:
            walk_at = righting_at = None
            if self._walk is not None:
                walk_at = self._walk_end_ms if self._fall_ms is None else self._fall_ms
            if self._righting is not None:
                righting_at = self._righting_ms
            if (
                walk_at is not None
                and walk_at <= now_ms
                and (righting_at is None or walk_at <= righting_at)
            ):
                out += self._walk_end(walk_at) if self._fall_ms is None else self._fall(walk_at)
            elif righting_at is not None and righting_at <= now_ms:
                out += self._righting_try(righting_at)
            else:
                break
        if out and self.keep_tape:
            self.tape.append((now_ms, "due", _NO_ARGS))
            self.sent.extend(out)
        return out

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        """``tv_minute``: a TV minute heard by the ear — mistaken for a touch? ``play_minute``
        (``minute=m``): a minute of gamepad play. Any other kind: ``[]``."""
        now_ms = int(now_ms)
        if kind == worldport.POLL_TV_MINUTE:
            self._record(now_ms, "poll", {"kind": kind})
            return self._sent(self._tv_minute(now_ms))
        if kind == worldport.POLL_PLAY_MINUTE:
            minute = int(params["minute"])
            self._record(now_ms, "poll", {"kind": kind, "minute": minute})
            return self._sent(self._play_minute(now_ms, minute))
        return []

    # ---------------------------------------------------------------- walks and falls

    def _start_walk(self, cmd: dict[str, Any], now_ms: int) -> list[dict[str, Any]]:
        # (v4 never starts a walk over a walk; if one came, the old walk ends here)
        out = self._end_walk(now_ms, "preempt") if self._walk is not None else []
        seconds = int(cmd["seconds"])
        chance = min(PPM, self.fall_ppm_per_s * seconds + self.fall_ppm_per_start * 2)
        counter = self.counters.get("fall", 0) + 1  # (the "fall" draw, inline: every walk)
        self.counters["fall"] = counter
        self._fall_ms = None
        if (rng.draw(self.seed, self.table, "fall", counter) * PPM) >> 64 < chance:
            # (1 … seconds: a fall at the last second comes before the end, v4's if/elif)
            self._fall_ms = now_ms + (1 + rng.pick_index(self._rand("fall.at"), seconds)) * 1000
        self._walk = cmd
        self._walk_start = now_ms
        self._walk_end_ms = now_ms + seconds * 1000
        return out

    def _walk_done(self, walk: dict[str, Any], t_ms: int, reason: str) -> dict[str, Any]:
        whole = int(walk["seconds"]) * 1000
        part = max(0, min(whole, t_ms - self._walk_start))
        bel_from, expect = walk["bel_from"], walk["expect"]
        path_mm = walk.get("path_mm")
        if path_mm is None:
            path_mm = (
                isqrt((expect[0] - bel_from[0]) ** 2 + (expect[1] - bel_from[1]) ** 2)
                if bel_from and expect
                else 0
            )
        odom = _odom(bel_from, expect, part, whole)
        odom["path_mm"] = int(path_mm) * part // whole if whole else 0
        body: dict[str, Any] = {
            "act_id": walk["act_id"],
            "cmd": {"vx": walk["vx_mm"], "vyaw": walk["vyaw_mrad"], "max_ms": whole},
            "odom": odom,
            "steps": part * STEPS_PER_S_X10 // 10_000,
            "reason": reason,
        }
        if "purpose" in walk:
            body["purpose"] = walk["purpose"]
        return self._percept("walk.done", t_ms, body)

    def _end_walk(self, t_ms: int, reason: str) -> list[dict[str, Any]]:
        """The walk in progress ends now (stopped): its ``walk.done``."""
        walk, self._walk = self._walk, None
        if walk is None:
            return []
        return [self._walk_done(walk, t_ms, reason)]

    def _walk_end(self, t_ms: int) -> list[dict[str, Any]]:
        walk, self._walk = self._walk, None
        if walk is None:
            return []
        return [self._walk_done(walk, t_ms, "time")]

    def _fall(self, t_ms: int) -> list[dict[str, Any]]:
        walk, self._walk = self._walk, None
        if walk is None:
            return []
        fell = self._percept(
            "body.fell",
            t_ms,
            {"during": "walk", "ctx": {"act_id": walk["act_id"]}, "trigger": "fallen"},
        )
        done = self._walk_done(walk, t_ms, "fell")
        self._righting = {"fell_ms": t_ms, "tries": 0}  # (a new fall starts afresh)
        self._righting_ms = t_ms + RIGHTING_FIRST_S * 1000
        return [fell, done]

    def _righting_try(self, t_ms: int) -> list[dict[str, Any]]:
        state = self._righting
        if state is None:
            return []
        state["tries"] += 1
        tries = int(state["tries"])
        ok = self._hit("righting", self.righting_ppm)
        body: dict[str, Any] = {
            "ok": ok,
            "ms": t_ms - int(state["fell_ms"]),  # how long it has lain there
            "tries": tries,
            "how": "wait",
        }
        if ok:
            body["via"] = "self"
            body["hard"] = tries > 1  # (v4's outcome: easy on the first try, else hard)
        if ok or tries >= RIGHTING_TRIES:
            self._righting = None  # up, or stuck until a hand
        else:
            self._righting_ms = t_ms + RIGHTING_GAP_S * 1000
        return [self._percept("body.righting", t_ms, body)]

    def _stop(self, cmd: dict[str, Any], now_ms: int) -> list[dict[str, Any]]:
        """``duck.stop``: naming the walk's act, that walk ends now; naming no act, whatever
        the body does ends (the walk, and every righting try still to come)."""
        target = cmd.get("act_id")
        out: list[dict[str, Any]] = []
        walk = self._walk
        if walk is not None and target in (None, walk["act_id"]):
            reason = "lifted" if str(cmd.get("why")) in LIFTED else "preempt"
            out = self._end_walk(now_ms, reason)
        if target is None:
            self._righting = None
        return out

    # ---------------------------------------------------------------- answers that take no time

    def _posture(self, cmd: dict[str, Any]) -> dict[str, Any]:
        want = str(cmd["want"])
        got = want
        # v4 draws only below certain (engine._stand_fails); sitting down has no dice.
        if want == "stand" and self.stand_ppm < PPM and not self._hit("stand.try", self.stand_ppm):
            got = "sit"
        return {"want": want, "got": got, "ms": 0, "fell": False, "act_id": cmd["act_id"]}

    def _dock(self, cmd: dict[str, Any]) -> dict[str, Any]:
        err = 0 if self.marker else int(cmd.get("bel_err_mm", 0))
        chance = self.dock_ppm * clamp(SCALE - err * SCALE // DOCK_ERR_BLIND_MM) // SCALE
        return {
            "contact": self._hit("dock.try", chance),
            "via": "pg",
            "tries": 1,
            "seen_tag": self.marker,
            "act_id": cmd["act_id"],
        }

    def _look(self, cmd: dict[str, Any], now_ms: int) -> dict[str, Any]:
        act_id = cmd["act_id"]
        pose = cmd.get("bel_pose")
        if cmd.get("want") == "nest" and self.marker and pose is not None:
            x, y, heading = pose
            dist = isqrt(x * x + y * y)
            bearing = atan2_mrad(-y, -x) if dist else heading
            return self._percept(
                "sense.sighting",
                now_ms,
                {
                    "what": "nest",
                    "b_mrad": signed_mrad(bearing - heading),
                    "conf_ppm": CONF_CERTAIN,
                    "look_id": act_id,
                    "edge": "new",
                    "r_mm": dist,
                    "r_src": "tag",
                    "act_id": act_id,
                },
            )
        return self._percept(
            "sense.scan",
            now_ms,
            {"act_id": act_id, "head_mrad": [], "light": "bright", "sectors": [], "sightings": []},
        )

    # ---------------------------------------------------------------- polls

    def _tv_minute(self, now_ms: int) -> list[dict[str, Any]]:
        if not self.ear or not rng.below(self._rand("false_alarm.tv"), TV_FALSE_TOUCH):
            return []
        return [
            self._percept(
                "sense.touch",
                now_ms,
                {
                    "where": "crown",
                    "seconds": 1,
                    "firmness": "light",
                    "path": "ear",
                    "latency_ms": 1100,
                    "legacy_phantom": True,
                },
            )
        ]

    def _play_minute(self, now_ms: int, minute: int) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "policy": "walk",
            "meters": 3 + rng.pick_index(self._rand("play.meters"), 6),
            "minute": minute,
        }
        if minute % 3 == 0:
            body["skill"] = PLAY_SKILLS[rng.pick_index(self._rand("play.skill"), 5)]
        return [self._percept("body.play", now_ms, body)]

    # ---------------------------------------------------------------- the family's hands

    def touch(
        self,
        persona: str | None,
        *,
        seconds: int = 5,
        where: str = "crown",
        firmness: str = "normal",
        now_ms: int,
    ) -> dict[str, Any] | None:
        """A hand on the duck (ask only while it is on, as v4 drew only then): the detector's
        ``sense.touch``, or ``None`` when it missed (engine.touch's chances, verbatim). A touch
        the percept could not carry is refused before the world changes (no draw, no tape)."""
        t_ms = int(now_ms)
        held = int(round(seconds))
        if not 0 <= held <= 86_400:
            raise LegacyWorldError(f"a touch of {seconds!r} seconds")
        where_text, firmness_text = str(where), str(firmness)
        self._record(
            t_ms,
            "touch",
            {"persona": persona, "seconds": seconds, "where": where, "firmness": firmness},
        )
        if where == "crown":
            chance = {"scratch": 9000, "normal": 8500, "light": 7000}.get(firmness, 8500)
            path = "ear" if self.ear else "joint"
            if not self.ear:
                chance = 5000
        else:
            chance, path = 9000, "joint"
        if self.conservative:
            chance -= 500
        if not rng.below(self._rand("detect.touch"), chance):
            return None
        event = self._percept(
            "sense.touch",
            t_ms,
            {
                "where": where_text,
                "seconds": held,
                "firmness": firmness_text,
                "path": path,
                "latency_ms": 1100 if path == "ear" else 400,
            },
        )
        self._sent([event])
        return event

    def hold_head(
        self, persona: str | None, *, moving: bool = True, now_ms: int
    ) -> dict[str, Any] | None:
        """A hand holding the head still (ask only while it is on). ``moving``: the head was
        moving under the hand (v4: it was not napping). ``sense.hold_head``, or ``None``."""
        t_ms = int(now_ms)
        self._record(t_ms, "hold_head", {"persona": persona, "moving": bool(moving)})
        if not rng.below(self._rand("detect.hold"), 9000 if persona == "child" else 9500):
            return None
        event = self._percept(
            "sense.hold_head", t_ms, {"path": "joint+probe", "moving": bool(moving)}
        )
        self._sent([event])
        return event

    def set_setting(self, key: str, value: Any, *, now_ms: int) -> None:
        """A setting the detectors follow (``conservative``, ``ear``); any other is ignored."""
        self._record(int(now_ms), "set_setting", {"key": str(key), "value": value})
        if key == "conservative":
            self.conservative = bool(value)
        elif key == "ear":
            self.ear = bool(value)

    def tv(self, on: bool, *, now_ms: int) -> None:
        self._record(int(now_ms), "tv", {"on": bool(on)})
        self.tv_on = bool(on)

    def hand(self, kind: str, *, persona: str | None = None, now_ms: int) -> None:
        """A hand that moved the duck (``pickup``, ``set_upright``, ``place_on_nest``,
        ``power_off``, ``pad_play`` …), for the truth stream only: Duck Core's own
        ``duck.stop`` ends a walk or a righting (module docstring)."""
        self._record(int(now_ms), "hand", {"kind": str(kind), "persona": persona})

    # ---------------------------------------------------------------- the migration seam

    def adopt(
        self,
        *,
        now_ms: int,
        counters: Mapping[str, int],
        walk: Mapping[str, Any] | None = None,
        righting: Mapping[str, Any] | None = None,
        conservative: bool | None = None,
        tv_on: bool | None = None,
    ) -> None:
        """Take over from a v4 duck mid-life (the migration seam, design WP13): a fresh world
        continues exactly where v4's own dice and schedule stood.

        * ``counters`` — how many times the v4 Duck Core had drawn each of the ten outcome
          purposes (its ``counters``): the world's next draw of a purpose is the one v4 would
          have made next.
        * ``walk`` — a walk in progress at the seam: its command (as :func:`normalize_command`
          reads it: ``act_id``, ``seconds``, ``vx_mm``, ``vyaw_mrad``, ``bel_from``, ``expect``
          …) plus ``start_ms``, ``end_ms`` and ``fall_ms`` (v4 already drew whether and when it
          falls: ``walk_plan["fall_at"]``, or ``None``).
        * ``righting`` — a righting in progress: ``fell_ms``, ``tries`` so far, ``next_ms``.
        * ``conservative``, ``tv_on`` — the detectors' setting and the TV at the seam.
        """
        if self.counters or self._walk is not None or self._righting is not None:
            raise LegacyWorldError("adopt: only a fresh world can take over from a v4 duck")
        unknown = sorted(set(counters) - set(PURPOSES))
        if unknown:
            raise LegacyWorldError(f"adopt: not an outcome purpose: {unknown}")
        t_ms = int(now_ms)
        args: dict[str, Any] = {
            "counters": {k: int(v) for k, v in sorted(counters.items()) if int(v)}
        }
        self.counters = dict(args["counters"])
        if walk is not None:
            args["walk"] = dict(walk)
            cmd = {k: v for k, v in walk.items() if k not in ("start_ms", "end_ms", "fall_ms")}
            cmd.setdefault("cmd", "duck.walk")
            self._walk = cmd
            self._walk_start = int(walk["start_ms"])
            self._walk_end_ms = int(walk["end_ms"])
            fall_ms = walk.get("fall_ms")
            self._fall_ms = None if fall_ms is None else int(fall_ms)
        if righting is not None:
            args["righting"] = dict(righting)
            self._righting = {"fell_ms": int(righting["fell_ms"]), "tries": int(righting["tries"])}
            self._righting_ms = int(righting["next_ms"])
        if conservative is not None:
            args["conservative"] = self.conservative = bool(conservative)
        if tv_on is not None:
            args["tv_on"] = self.tv_on = bool(tv_on)
        self._record(t_ms, "adopt", args)

    # ---------------------------------------------------------------- records

    def tape_json(self) -> list[dict[str, Any]]:
        """The calls this world took, as JSON records (what :func:`replay` reads)."""
        return [{"t_ms": t_ms, "call": call, "args": args} for t_ms, call, args in self.tape]

    def state(self) -> dict[str, Any]:
        """What the world has in hand (for pages and tests): counters and schedules."""
        return {
            "calib_hash": self.calib_hash,
            "counters": dict(sorted(self.counters.items())),
            "walk": None if self._walk is None else self._walk["act_id"],
            "righting_tries": None if self._righting is None else int(self._righting["tries"]),
            "next_at_ms": self.next_at_ms(),
            "conservative": self.conservative,
            "ear": self.ear,
            "tv_on": self.tv_on,
        }


def replay(header: Mapping[str, Any], tape: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Feed a recorded tape to a fresh world built from ``header``: every percept it sends, in
    order (``world_replay_check``: it must equal the recorded :attr:`LegacyWorld.sent`)."""
    world = LegacyWorld(header, keep_tape=False)
    out: list[dict[str, Any]] = []
    for record in tape:
        t_ms, call, args = int(record["t_ms"]), str(record["call"]), dict(record["args"])
        if call == "command":
            out += world._take(args, t_ms)  # (the tape keeps normalized commands)
        elif call == "due":
            out += world.due(t_ms)
        elif call == "poll":
            out += world.poll(str(args.pop("kind")), t_ms, **args)
        elif call == "touch":
            event = world.touch(args.pop("persona"), now_ms=t_ms, **args)
            out += [event] if event is not None else []
        elif call == "hold_head":
            event = world.hold_head(args.pop("persona"), now_ms=t_ms, **args)
            out += [event] if event is not None else []
        elif call == "set_setting":
            world.set_setting(args["key"], args["value"], now_ms=t_ms)
        elif call == "tv":
            world.tv(args["on"], now_ms=t_ms)
        elif call == "hand":
            world.hand(args["kind"], persona=args.get("persona"), now_ms=t_ms)
        elif call == "adopt":
            world.adopt(now_ms=t_ms, **args)
        else:
            raise LegacyWorldError(f"unknown tape call {call!r}")
    return out


def world_replay_check(
    header: Mapping[str, Any], tape: Iterable[Mapping[str, Any]], sent: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """``{"identical", "percepts", "first_difference"}``: does a fresh world, fed the tape,
    send exactly ``sent``?"""
    again = replay(header, tape)
    first = next(
        (i for i, (a, b) in enumerate(zip(again, sent, strict=False)) if a != dict(b)), None
    )
    if first is None and len(again) != len(sent):
        first = min(len(again), len(sent))
    return {"identical": first is None, "percepts": len(sent), "first_difference": first}


worldport.register_world(WORLD, LegacyWorld.from_header)
