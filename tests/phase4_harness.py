"""Phase-4 test harness: v4 and v4_1 ducks side by side (not a test module).

* :func:`open_v40` — a duck on the frozen v4 engine (``duckmind.engines.v4_0``), from a v4
  ``Config`` dict; :func:`open_v41` — the same duck on today's Duck Core with LegacyWorld
  attached (``api.open_session`` moves the rates into the world header, design A5), or with
  any port.
* :func:`send` — ``dispatch`` for either kind of session.
* Lives: :func:`walking_age` (to the growth day it walks), :func:`lively` (a busy scripted day:
  ~40 hands, voices, lifts, the gamepad, the TV, at random seconds, the time between them
  advanced in one piece, in random pieces, or by ``advance_to(stop_on_command=True)``),
  :func:`live_v2` / :func:`live_legacy` (the population's household-v2 and legacy families,
  with popsim's own draws, so index *i* is run A's duck *i*).
* :class:`Watch` and :func:`projection` — the *behaviour projection* (design §2.2, G1): every
  decision, expression, walk, day's numbers, dock, outing, power, nap, bid, moment, family
  belief and growth record, with its second; every posture and position change as it happens
  (watched on Duck Core's state, so it does not depend on how a record is shaped); and at the
  end the draw counters (Duck Core's and the world's together). Event ids are left out: new
  records renumber them.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

RATE_KEYS = ("fall_ppm_per_s", "fall_ppm_per_start", "righting_ppm", "stand_ppm", "dock_ppm")
STRESS = {
    "fall_ppm_per_s": 30_000,
    "righting_ppm": 200_000,
    "stand_ppm": 700_000,
    "fall_ppm_per_start": 20_000,
    "conservative": False,
}
# The phase-4 outcome purposes (design §4): Duck Core's counters in v4, the world's in v4_1.
WORLD_PURPOSES = frozenset(
    {
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
    }
)
FROZEN_MODULE = "duckmind.engines.v4_0"


# ---------------------------------------------------------------------------- opening ducks


def open_v40(config: Mapping[str, Any]) -> Any:
    from duckmind.engines.v4_0.duckcore import api

    return api.Session(api.Config(**dict(config)))


def open_v41(config: Mapping[str, Any], *, port: Any = None) -> Any:
    """The same duck on engine v4_1, opened the way a saved duck is (``api.open_session``: the
    v4 rate fields become its LegacyWorld header). With ``port``, Duck Core's commands and
    percepts go through that port instead (the family's hands still reach the LegacyWorld)."""
    import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld, as conftest will)
    from duckmind.duckcore import api

    session = api.open_session({**dict(config), "engine_rev": "v4_1"})
    if port is not None:
        session.core.attach_port(port)
    return session


def is_frozen(session: Any) -> bool:
    return type(session).__module__.startswith(FROZEN_MODULE)


def send(session: Any, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """``dispatch`` for either a frozen v4 session or a current one."""
    if is_frozen(session):
        from duckmind.engines.v4_0.duckcore import api as frozen

        result: dict[str, Any] = frozen.dispatch(session, method, dict(params))
        return result
    from duckmind.duckcore import api

    result = api.dispatch(session, method, dict(params))
    return result


def act(session: Any, action: str, persona: str | None = "owner", **params: Any) -> None:
    send(session, "act", {"action": action, "params": params, "persona": persona})


def world_of(session: Any) -> Any:
    """The world port of a v4_1 session (None for a frozen one)."""
    if is_frozen(session):
        return None
    return session.core.acts.port


# ---------------------------------------------------------------------------- lives


def walking_age(session: Any) -> None:
    """Grow it to the day it walks (the page's 「過到第 5 個成長日」 button), then morning,
    switched on, on the floor."""
    if bool(session.config.nest):
        send(session, "live", {"family": "elder_home", "days": 5})
        send(session, "advance", {"until": "07:00"})
    else:
        send(session, "live", {"routine": "floor", "days": 5})
        send(session, "advance", {"until": "07:00"})
        act(session, "power_on")
        send(session, "set_setting", {"key": "floor", "value": True})


# (action, persona, params): the inputs of the phase-4 step-invariance prototype
ACTS: tuple[tuple[str, str | None, dict[str, Any]], ...] = (
    ("touch", "child", {"where": "crown", "seconds": 6}),
    ("say", "owner", {"kind": "voice"}),
    ("pickup", "owner", {}),
    ("set_upright", "owner", {}),
    ("call_name", "owner", {}),
    ("hear_home", "owner", {"kind": "voice"}),
    ("hold_head", "owner", {}),
    ("pad_play", "child", {"minutes": 4}),
    ("place_on_nest", "owner", {}),
    ("tv", None, {"on": True}),
    ("tv", None, {"on": False}),
)
PIECES = (1, 2, 3, 7, 30, 59, 60, 61, 120, 600)
MODES = ("one", "pieces", "stop")


@dataclass
class Drive:
    """How the time between inputs passes: in one piece, in random pieces (seeded), or by
    ``advance_to(stop_on_command=True)`` until the input's second, draining the outbox."""

    mode: str = "one"
    seed: int = 0
    drained: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(self.mode)
        self.cut = random.Random(self.seed * 7 + 1)

    def wait(self, session: Any, seconds: int) -> None:
        if self.mode == "one":
            send(session, "advance", {"seconds": seconds})
        elif self.mode == "pieces":
            left = seconds
            while left > 0:
                piece = min(left, self.cut.choice(PIECES))
                send(session, "advance", {"seconds": piece})
                left -= piece
        else:
            core = session.core
            target = core.now_ms() + seconds * 1000
            while core.now_ms() < target:
                core.advance_to(target, stop_on_command=True)
                self.drained += core.drain_outbox()


def lively(session: Any, seed: int, days: int = 1, drive: Drive | None = None) -> None:
    """``days`` × 40 inputs at random seconds (20–1,500 s apart), the same for every drive."""
    drive = drive or Drive()
    plan = random.Random(seed)
    for _ in range(days * 40):
        drive.wait(session, plan.randint(20, 1500))
        action, persona, params = ACTS[plan.randrange(len(ACTS))]
        act(session, action, persona, **params)


def lively_config(seed: int, nest: bool, stress: bool) -> dict[str, Any]:
    config: dict[str, Any] = {"seed": seed, "nest": nest}
    if stress:
        config.update(STRESS)
    return config


def lively_v41(seed: int, nest: bool, stress: bool = False, days: int = 1) -> Any:
    session = open_v41(lively_config(seed, nest, stress))
    walking_age(session)
    lively(session, seed, days)
    return session


def household_v2(index: int) -> tuple[dict[str, Any], str]:
    """Run A's household-v2 duck ``index`` (popsim_p4 ``v2_config``, charge 125): its v4
    Config and its family kind."""
    from duckmind.duckcore.routines import FAMILY_KINDS, make_profile

    draw = random.Random(700_000 + index)
    kinds = list(FAMILY_KINDS)
    kind = draw.choices(kinds, weights=[FAMILY_KINDS[k][0] for k in kinds])[0]
    seed = 10_000 + index
    profile = make_profile(seed ^ 0xFA11, kind)
    config = {
        "seed": seed,
        "name": "小鴨",
        "tof": profile.tof,
        "ear": profile.ear,
        "name_ear": profile.name_ear,
        "conservative": True,
        "battery_help": False,
        "nest": profile.nest,
        "camera": profile.camera,
        "nest_charge_per_min": 125,
        "dock_ppm": 850_000 if profile.camera else 700_000,
        "household": kind,
    }
    return config, kind


ROUTINE_MIX = {
    "daily": 0.30,
    "three_per_week": 0.15,
    "weekends": 0.15,
    "floor": 0.20,
    "docked": 0.10,
    "neglect": 0.10,
}


def household_legacy(index: int) -> tuple[dict[str, Any], str, str]:
    """Run A's legacy control duck ``index`` (popsim_p4 ``household_legacy``): its v4 Config
    and its two routines."""
    draw = random.Random(900_000 + index)
    routines = list(ROUTINE_MIX)
    first = draw.choices(routines, weights=list(ROUTINE_MIX.values()))[0]
    second = first
    if draw.random() < 0.20:
        second = draw.choice([r for r in routines if r != first])
    config = {
        "seed": 10_000 + index,
        "name": "小鴨",
        "tof": draw.random() < 0.30,
        "ear": draw.random() < 0.90,
        "name_ear": draw.random() < 0.80,
        "conservative": draw.random() < 0.70,
    }
    return config, first, second


def live_v2(session: Any, kind: str, days: int) -> None:
    send(session, "live", {"family": kind, "days": days})


def live_legacy(session: Any, first: str, second: str, days: int) -> None:
    half = days if second == first else days // 2
    send(session, "live", {"routine": first, "days": half})
    if half < days:
        send(session, "live", {"routine": second, "days": days - half})


# ---------------------------------------------------------------------------- projection

WALK_KEYS = (
    "verb",
    "vx",
    "vyaw",
    "seconds",
    "meters",
    "cm",
    "speed",
    "yaw_rate",
    "stage",
    "conservative",
    "battery",
    "floor",
    "purpose",
    "excursion",
    "family",
    "phase",
    "nested",
    "docked",
    "unplaced",
    "avoided",
    "moment",
    "homing",
)
# kind -> the payload keys that are behaviour (None: the whole payload)
BEHAVIOUR: dict[str, tuple[str, ...] | None] = {
    "duck.decision": None,
    "duck.expression": None,
    "duck.walk": WALK_KEYS,
    "duck.day_stats": None,
    "duck.reaction": None,
    "duck.greeting": None,
    "nest.docked": None,
    "nest.undocked": None,
    "nest.stranded": None,
    "nest.trouble": None,
    "nav.home_start": None,
    "duck.excursion": None,
    "power.on": (),
    "power.off": ("reason", "battery"),
    "duck.nap_start": None,
    "duck.nap_end": None,
    "duck.self_wake": None,
    "duck.bedtime": None,
    "duck.bid": None,
    "duck.bid_answered": None,
    "duck.bid_unanswered": None,
    "duck.moment": None,
    "duck.lockout": None,
    "env.family": None,
    "growth.gd": None,
    "growth.stage": None,
    "growth.milestone": None,
    "growth.unlocked": None,
    "learning.committed": None,
    "voice.babble": None,
    "voice.liked": None,
    "voice.promoted": None,
}
# CoreState fields whose every change is kept, with the second it happened
WATCHED = ("posture", "pos_x", "pos_y", "heading", "awake", "nested", "gd")
_EVENT_ID = re.compile(r"E\d{6,}")  # an event id, alone or inside an id (``U-E000317-…``)


def _plain(value: Any) -> Any:
    """A frozen payload as plain JSON-ish values, event ids blanked."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return _EVENT_ID.sub("E?", value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class Watch:
    """Keeps every change of :data:`WATCHED` on a Duck Core's state as it is made (the state
    object is given a subclass whose ``__setattr__`` notes the change; nothing else differs)."""

    def __init__(self, core: Any) -> None:
        self.rows: list[tuple[str, str, Any]] = []
        state = core.s
        rows, watched = self.rows, frozenset(WATCHED)
        base = type(state)

        def __setattr__(obj: Any, name: str, value: Any) -> None:
            if name in watched:
                old = getattr(obj, name, _MISSING)
                if old is _MISSING or old != value:
                    now = getattr(obj, "now", None)
                    rows.append((now.isoformat() if now is not None else "", name, _plain(value)))
            base.__setattr__(obj, name, value)

        watched_cls = type(
            f"Watched{base.__name__}", (base,), {"__setattr__": __setattr__, "__slots__": ()}
        )
        state.__class__ = watched_cls


_MISSING = object()


@dataclass
class Projection:
    rows: list[Any]
    counters: dict[str, int]
    kinds: dict[str, int]

    def digest(self) -> str:
        import hashlib
        import json

        data = json.dumps([self.rows, self.counters], ensure_ascii=False, default=str)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _counters(session: Any) -> dict[str, int]:
    counters = {k: int(v) for k, v in dict(session.core.counters).items() if int(v)}
    world = world_of(session)
    for purpose, count in dict(getattr(world, "counters", {}) or {}).items():
        if int(count):
            assert purpose not in counters, f"{purpose} is counted by Duck Core and the world"
            counters[purpose] = int(count)
    return dict(sorted(counters.items()))


def projection(session: Any, watch: Watch) -> Projection:
    rows: list[Any] = []
    kinds: dict[str, int] = {}
    for event in session.core.log.events:
        keys = BEHAVIOUR.get(event.kind, "-")
        if keys == "-":
            continue
        kinds[event.kind] = kinds.get(event.kind, 0) + 1
        payload = event.payload
        body = payload if keys is None else {k: payload.get(k) for k in keys if k in payload}
        rows.append(("event", event.occurred_at.isoformat(), event.kind, _plain(body)))
    merged = sorted(
        rows + [("state", at, name, value) for at, name, value in watch.rows],
        key=lambda row: (row[1], 0 if row[0] == "state" else 1),
    )
    return Projection(merged, _counters(session), dict(sorted(kinds.items())))


def first_difference(a: Projection, b: Projection, names: tuple[str, str] = ("v4", "v4_1")) -> str:
    """Where two projections part (empty when they do not)."""
    for index, (left, right) in enumerate(zip(a.rows, b.rows, strict=False)):
        if left != right:
            return (
                f"row {index} differs:\n  {names[0]}: {left}\n  {names[1]}: {right}\n"
                f"  (before it: {a.rows[index - 1] if index else None})"
            )
    if len(a.rows) != len(b.rows):
        index = min(len(a.rows), len(b.rows))
        extra = (a.rows if len(a.rows) > index else b.rows)[index]
        who = names[0] if len(a.rows) > index else names[1]
        return f"{who} has more rows from row {index}: {extra}"
    if a.counters != b.counters:
        return f"draw counters differ: {names[0]} {a.counters} vs {names[1]} {b.counters}"
    return ""


Runner = Callable[[Any], None]
