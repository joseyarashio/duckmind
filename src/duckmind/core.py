"""Deterministic functional self-model and transient-state reducer.

No LLM text may write this state. Durable learning enters only as an evidence-
checked `learning_committed` event from the learning engine.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Iterable, Mapping

from duckmind.provenance import Event

SCHEMA_VERSION = 1
PREFERENCE_PATH = re.compile(r"preferences\.[a-z0-9_-]+\.[a-z0-9_-]+")
TRUSTED_SOURCES = {
    "touch": "touch_sensor",
    "time_passed": "runtime_clock",
    "owner_play_enabled": "owner_control",
    "owner_stop": "owner_control",
    "robot_state": "robotd_state",
    "body_lost": "runtime_safety",
    "learning_committed": "learning_engine",
    "behavior_decision": "system1",
}


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _finite_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _snapshot_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _snapshot_fractions(
    value: object,
    name: str,
    *,
    required_keys: frozenset[str] | None = None,
    signed_keys: frozenset[str] = frozenset(),
) -> dict[str, float]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with text keys")
    if required_keys is not None and value.keys() != required_keys:
        raise ValueError(f"{name} fields do not match snapshot schema")
    result: dict[str, float] = {}
    for key, item in value.items():
        if key in signed_keys:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or not -1 <= item <= 1
            ):
                raise ValueError(f"{name}.{key} must be finite and between -1 and 1")
            result[key] = float(item)
        else:
            result[key] = _finite_fraction(item, f"{name}.{key}")
    return result


@dataclass(frozen=True, slots=True)
class Affect:
    valence: float = 0.0  # -1..1
    arousal: float = 0.2  # 0..1
    fear: float = 0.0  # 0..1

    def __post_init__(self) -> None:
        if (
            isinstance(self.valence, bool)
            or not isinstance(self.valence, (int, float))
            or not math.isfinite(self.valence)
            or not -1 <= self.valence <= 1
        ):
            raise ValueError("valence must be finite and between -1 and 1")
        _finite_fraction(self.arousal, "arousal")
        _finite_fraction(self.fear, "fear")


@dataclass(frozen=True, slots=True)
class Drives:
    social_need: float = 0.2
    rest_need: float = 0.1

    def __post_init__(self) -> None:
        _finite_fraction(self.social_need, "social_need")
        _finite_fraction(self.rest_need, "rest_need")


@dataclass(frozen=True, slots=True)
class Personality:
    sociability: float = 0.5
    curiosity: float = 0.5
    resilience: float = 0.5

    def __post_init__(self) -> None:
        _finite_fraction(self.sociability, "sociability")
        _finite_fraction(self.curiosity, "curiosity")
        _finite_fraction(self.resilience, "resilience")


@dataclass(frozen=True, slots=True)
class DuckSelf:
    """State used in behavior selection, not a claim of subjective awareness."""

    identity: str
    born_at: datetime
    affect: Affect = field(default_factory=Affect)
    drives: Drives = field(default_factory=Drives)
    personality: Personality = field(default_factory=Personality)
    relationships: Mapping[str, float] = field(default_factory=dict)
    preferences: Mapping[str, float] = field(default_factory=dict)
    applied_update_ids: tuple[str, ...] = ()
    play_enabled: bool = False
    fallen: bool = False
    limp: bool = False
    battery_fraction: float = 1.0
    current_goal: str | None = None
    last_decision_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("identity must be non-empty text")
        if self.born_at.utcoffset() is None or (
            self.last_decision_at is not None and self.last_decision_at.utcoffset() is None
        ):
            raise ValueError("Duck Self timestamps must be timezone-aware")
        _finite_fraction(self.battery_fraction, "battery_fraction")
        if any(not isinstance(flag, bool) for flag in (self.play_enabled, self.fallen, self.limp)):
            raise ValueError("Duck Self safety flags must be boolean")
        if self.current_goal is not None and not isinstance(self.current_goal, str):
            raise ValueError("current goal must be text or null")
        for name, values in (
            ("relationships", self.relationships),
            ("preferences", self.preferences),
        ):
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{name} keys must be non-empty text")
                _finite_fraction(value, name)
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        if any(not PREFERENCE_PATH.fullmatch(key) for key in self.preferences):
            raise ValueError("unsupported preference path")
        if len(set(self.applied_update_ids)) != len(self.applied_update_ids):
            raise ValueError("applied update IDs must be unique")

    @classmethod
    def new(cls, *, identity: str, born_at: datetime) -> DuckSelf:
        if not identity.strip() or born_at.tzinfo is None:
            raise ValueError("identity and timezone-aware birth time are required")
        return cls(identity=identity, born_at=born_at)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "born_at": self.born_at.isoformat(),
            "affect": asdict(self.affect),
            "drives": asdict(self.drives),
            "personality": asdict(self.personality),
            "relationships": dict(self.relationships),
            "preferences": dict(self.preferences),
            "applied_update_ids": list(self.applied_update_ids),
            "play_enabled": self.play_enabled,
            "fallen": self.fallen,
            "limp": self.limp,
            "battery_fraction": self.battery_fraction,
            "current_goal": self.current_goal,
            "last_decision_at": self.last_decision_at.isoformat()
            if self.last_decision_at
            else None,
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, object]) -> DuckSelf:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Duck Self snapshot schema")
        identity = value.get("identity")
        if not isinstance(identity, str):
            raise ValueError("identity must be text")
        play_enabled = value.get("play_enabled")
        fallen = value.get("fallen")
        limp = value.get("limp")
        if (
            not isinstance(play_enabled, bool)
            or not isinstance(fallen, bool)
            or not isinstance(limp, bool)
        ):
            raise ValueError("snapshot safety flags must be boolean")
        current_goal = value.get("current_goal")
        if current_goal is not None and not isinstance(current_goal, str):
            raise ValueError("current goal must be text or null")
        update_ids = value.get("applied_update_ids")
        if not isinstance(update_ids, list) or any(
            not isinstance(item, str) for item in update_ids
        ):
            raise ValueError("applied update IDs must be a text list")
        last_decision = value.get("last_decision_at")
        return cls(
            identity=identity,
            born_at=_snapshot_time(value.get("born_at"), "born_at"),
            affect=Affect(
                **_snapshot_fractions(
                    value.get("affect"),
                    "affect",
                    required_keys=frozenset({"valence", "arousal", "fear"}),
                    signed_keys=frozenset({"valence"}),
                )
            ),
            drives=Drives(
                **_snapshot_fractions(
                    value.get("drives"),
                    "drives",
                    required_keys=frozenset({"social_need", "rest_need"}),
                )
            ),
            personality=Personality(
                **_snapshot_fractions(
                    value.get("personality"),
                    "personality",
                    required_keys=frozenset({"sociability", "curiosity", "resilience"}),
                )
            ),
            relationships=_snapshot_fractions(value.get("relationships"), "relationships"),
            preferences=_snapshot_fractions(value.get("preferences"), "preferences"),
            applied_update_ids=tuple(update_ids),
            play_enabled=play_enabled,
            fallen=fallen,
            limp=limp,
            battery_fraction=_finite_fraction(value.get("battery_fraction"), "battery_fraction"),
            current_goal=current_goal,
            last_decision_at=_snapshot_time(last_decision, "last_decision_at")
            if last_decision is not None
            else None,
        )


def reduce_event(state: DuckSelf, event: Event) -> DuckSelf:
    required_source = TRUSTED_SOURCES.get(event.kind)
    if required_source is None:
        return state  # recorded facts not used by this reducer remain in the log
    if event.source.kind != required_source or event.source.reliability < 0.5:
        raise ValueError(f"{event.kind} requires a trusted source")
    payload = event.payload
    if event.kind == "touch":
        actor = payload.get("actor")
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("touch requires an actor ID")
        if event.source.reliability < 0.7:
            return state  # ambiguous contact is retained, but not appraised
        weight = event.source.reliability
        return replace(
            state,
            affect=Affect(
                valence=_bounded(state.affect.valence + 0.10 * weight, -1, 1),
                arousal=_bounded(state.affect.arousal + 0.04 * weight),
                fear=state.affect.fear,
            ),
            drives=replace(
                state.drives,
                social_need=_bounded(state.drives.social_need - 0.05 * weight),
            ),
        )
    if event.kind == "time_passed":
        elapsed = payload.get("elapsed_s")
        asleep = payload.get("asleep")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or not 0 <= elapsed <= 86400
        ):
            raise ValueError("elapsed_s must be finite and within one day")
        if not isinstance(asleep, bool):
            raise ValueError("asleep must be boolean")
        return replace(
            state,
            affect=Affect(
                valence=state.affect.valence * math.exp(-elapsed / 1800),
                arousal=0.2 + (state.affect.arousal - 0.2) * math.exp(-elapsed / 1200),
                fear=state.affect.fear * math.exp(-elapsed / 900),
            ),
            drives=Drives(
                # No unbounded "loneliness" or negative relationship update
                # merely because a family member was absent.
                social_need=min(
                    0.65, state.drives.social_need + (0 if asleep else elapsed / 86400)
                ),
                rest_need=_bounded(
                    state.drives.rest_need + (-elapsed / 10800 if asleep else elapsed / 28800)
                ),
            ),
        )
    if event.kind == "owner_stop":
        return replace(state, play_enabled=False, current_goal=None, last_decision_at=None)
    if event.kind == "body_lost":
        return replace(state, play_enabled=False, current_goal=None, last_decision_at=None)
    if event.kind == "owner_play_enabled":
        if state.fallen or state.limp or state.battery_fraction < 0.3:
            return state
        return replace(state, play_enabled=True)
    if event.kind == "robot_state":
        battery = _finite_fraction(payload.get("battery_fraction"), "battery_fraction")
        fallen = payload.get("fallen")
        limp = payload.get("limp")
        if not isinstance(fallen, bool) or not isinstance(limp, bool):
            raise ValueError("robot_state safety flags must be boolean")
        unsafe = fallen or limp or battery <= 0.2
        return replace(
            state,
            battery_fraction=battery,
            fallen=fallen,
            limp=limp,
            play_enabled=False if unsafe else state.play_enabled,
            current_goal=None if unsafe else state.current_goal,
        )
    if event.kind == "learning_committed":
        update_id = payload.get("update_id")
        parameter = payload.get("parameter")
        evidence = payload.get("evidence_event_ids")
        if not isinstance(update_id, str) or not update_id:
            raise ValueError("learning event needs update_id")
        if update_id in state.applied_update_ids:
            return state
        if not isinstance(parameter, str) or not PREFERENCE_PATH.fullmatch(parameter):
            raise ValueError("learning event has unsupported parameter")
        if not isinstance(evidence, (list, tuple)) or not evidence:
            raise ValueError("learning event needs evidence IDs")
        before = _finite_fraction(payload.get("before"), "before")
        after = _finite_fraction(payload.get("after"), "after")
        current = state.preferences.get(parameter, 0.5)
        if not math.isclose(current, before, abs_tol=1e-9) or abs(after - before) > 0.005 + 1e-12:
            raise ValueError("learning event conflicts with state or plasticity bound")
        preferences = dict(state.preferences)
        preferences[parameter] = after
        return replace(
            state,
            preferences=preferences,
            applied_update_ids=(*state.applied_update_ids, update_id),
        )
    if event.kind == "behavior_decision":
        goal = payload.get("goal")
        invitation_id = payload.get("invitation_event_id")
        if goal is not None and (not isinstance(goal, str) or not goal.startswith("play:")):
            raise ValueError("behavior decision has invalid goal")
        if not isinstance(invitation_id, str) or not invitation_id:
            raise ValueError("behavior decision requires invitation evidence")
        if (
            payload.get("choice") == "continue" and payload.get("reason") == "same_goal_cooldown"
        ) or (payload.get("choice") == "observe" and payload.get("reason") == "switch_cooldown"):
            # A suppressed invitation neither ends the admitted engagement nor
            # restarts its cooldown. The timestamp stays at that engagement.
            return state
        return replace(state, current_goal=goal, last_decision_at=event.occurred_at)
    raise AssertionError("trusted event kind was not reduced")


def replay_events(initial: DuckSelf, events: Iterable[Event]) -> DuckSelf:
    state = initial
    for event in events:
        state = reduce_event(state, event)
    return state
