"""T14 (phase 4, decision A14): ``EventLog.append``'s fast path gives the v4 digests.

The reference below is the v4 append, verbatim (``Event(...)`` freezes the payload, then
``canonical`` thaws and dumps it). Every event a real duck writes, and a set of edge-case
payloads, must give the same Event, the same canonical bytes and the same chain digest --
and every malformed call must raise the same error without appending anything.
"""

from __future__ import annotations

import enum
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from duckmind.duckcore import memory
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.duckcore.memory import CORE_VERSION, EventLog, canonical
from duckmind.provenance import Event, SourceRef


class ReferenceLog:
    """The v4 EventLog.append (git 79cc25d), kept here as the definition."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.digests: list[str] = []

    def append(
        self,
        kind: str,
        at: datetime,
        source: tuple[str, str, float],
        payload: Mapping[str, object],
        trace_id: str,
        parents: Sequence[str] = (),
    ) -> Event:
        seq = len(self.events) + 1
        source_kind, source_id, reliability = source
        event = Event(
            event_id=f"E{seq:06d}",
            kind=kind,
            occurred_at=at,
            observed_at=at,
            source=SourceRef(source_kind, source_id, reliability, CORE_VERSION),
            trace_id=trace_id,
            payload=payload,
            parent_event_ids=tuple(dict.fromkeys(parents)),
        )
        previous = self.digests[-1] if self.digests else "genesis"
        digest = hashlib.sha256((previous + canonical(event)).encode()).hexdigest()
        self.events.append(event)
        self.digests.append(digest)
        return event


def same_frozen(a: object, b: object) -> bool:
    """Equal and of the same types all the way down (MappingProxyType, tuple, scalars)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, MappingProxyType):
        assert isinstance(b, MappingProxyType)
        return list(a) == list(b) and all(same_frozen(a[key], b[key]) for key in a)
    if isinstance(a, tuple):
        assert isinstance(b, tuple)
        return len(a) == len(b) and all(same_frozen(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return repr(a) == repr(b)
    return a == b


def same_event(a: Event, b: Event) -> bool:
    return (
        a == b
        and type(a.source.reliability) is type(b.source.reliability)
        and a.occurred_at.isoformat() == b.occurred_at.isoformat()
        and same_frozen(a.payload, b.payload)
    )


def copy_plain(value: object) -> object:
    """A copy that keeps every container type (what the caller passed, at call time)."""
    kind = type(value)
    if kind is dict:
        return {key: copy_plain(item) for key, item in value.items()}  # type: ignore[attr-defined]
    if kind is list:
        return [copy_plain(item) for item in value]  # type: ignore[attr-defined]
    if kind is tuple:
        return tuple(copy_plain(item) for item in value)  # type: ignore[attr-defined]
    if kind is MappingProxyType:
        return MappingProxyType({key: copy_plain(item) for key, item in value.items()})  # type: ignore[attr-defined]
    return value


# --------------------------------------------------------------------- a real duck


def test_every_event_of_a_real_duck_matches_the_v4_append(monkeypatch: pytest.MonkeyPatch) -> None:
    """A family duck with a nest for 12 days (> 10,000 events): each append is checked at
    the moment it happens, against the reference fed a copy of the caller's arguments."""
    reference = ReferenceLog()
    fast_append = EventLog.append
    exact_calls = 0
    exact_append = EventLog._append_exact

    def checked(
        self: EventLog,
        kind: str,
        at: datetime,
        source: tuple[str, str, float],
        payload: Mapping[str, object],
        trace_id: str,
        parents: Sequence[str] = (),
    ) -> Event:
        copied = copy_plain(payload)
        assert isinstance(copied, Mapping)
        event = fast_append(self, kind, at, source, payload, trace_id, parents)
        want = reference.append(kind, at, source, copied, trace_id, tuple(parents))
        assert same_event(event, want), kind
        assert self.digests[-1] == reference.digests[-1], kind
        return event

    def counted(self: EventLog, *args: object, **kwargs: object) -> Event:
        nonlocal exact_calls
        exact_calls += 1
        return exact_append(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(EventLog, "append", checked)
    monkeypatch.setattr(EventLog, "_append_exact", counted)
    session = Session(
        Config(seed=10003, nest=True, camera=True, household="dual_income", battery_help=False)
    )
    dispatch(session, "live", {"family": "dual_income", "days": 12})
    log = session.core.log
    assert len(log.events) >= 10_000
    assert log.head == reference.digests[-1]
    assert log.verify()
    # the engine's own payloads are plain: none of them needs the exact path
    assert exact_calls == 0


# --------------------------------------------------------------------- edge cases

AT = datetime(2026, 9, 27, 21, 0, tzinfo=UTC)
TAIPEI = timezone(timedelta(hours=8))
SELF = ("self", "duckcore", 1.0)


class Colour(enum.StrEnum):
    RED = "red"


class Mixed(str, enum.Enum):
    BLUE = "blue"


class Level(enum.IntEnum):
    HIGH = 3


class LaterDay(datetime):
    pass


def calls() -> Iterator[tuple[str, tuple[object, ...]]]:
    """(label, (kind, at, source, payload, trace_id, parents)) -- all valid calls."""
    yield "empty", ("duck.walk", AT, SELF, {}, "life", ())
    yield (
        "plain nested",
        (
            "duck.walk",
            AT,
            SELF,
            {
                "verb": "wander",
                "vx": 0.12,
                "neg_zero": -0.0,
                "big": 10**30,
                "tiny": 1e-300,
                "huge": 1.7976931348623157e308,
                "ok": True,
                "none": None,
                "from": [1, 2.5, [3, {"z": 1, "a": ()}]],
                "to": (4, 5),
                "nested": {"b": {"c": [None, False, "文字"]}},
            },
            "life",
            ("E000001",),
        ),
    )
    yield (
        "text that needs escaping",
        (
            'kind"with\\quote',
            AT,
            ("imu", 'i"d\n', 0.25),
            {"文": "小鴨 \u0000\ttab \U0001f986", 'k"\\': "\x7f", "": ""},
            "trace é\n",
            ['p"1', "pé"],
        ),
    )
    yield "unsorted keys", ("a", AT, SELF, {"b": 1, "a": 2, "A": 3, "_": 4, "é": 5}, "t", ())
    yield (
        "proxies from another event",
        (
            "k",
            AT,
            SELF,
            {"spot": MappingProxyType({"y": 2, "x": (1, MappingProxyType({"q": 1}))})},
            "t",
            (),
        ),
    )
    yield "duplicate parents", ("k", AT, SELF, {"a": 1}, "t", ("E1", "E2", "E1"))
    yield "parents as a list", ("k", AT, SELF, {"a": 1}, "t", ["E2", "E1"])
    yield "parents as a generator", ("k", AT, SELF, {"a": 1}, "t", (p for p in ("E3", "E3")))
    yield "int reliability", ("k", AT, ("owner", "app", 1), {"a": 1}, "t", ())
    yield "float reliability", ("k", AT, ("owner", "app", 1.0), {"a": 1}, "t", ())
    yield "source as a list", ("k", AT, ["owner", "app", 0.5], {"a": 1}, "t", ())
    yield "same instant, other zone", ("k", AT.astimezone(TAIPEI), SELF, {"a": 1}, "t", ())
    yield "microseconds", ("k", AT + timedelta(microseconds=7), SELF, {"a": 1}, "t", ())
    yield "datetime subclass", ("k", LaterDay(2026, 1, 1, tzinfo=UTC), SELF, {"a": 1}, "t", ())
    yield "StrEnum key and value", ("k", AT, SELF, {Colour.RED: Colour.RED}, "t", ())
    yield "str-mixin Enum value", ("k", AT, SELF, {"c": Mixed.BLUE}, "t", ())
    yield "IntEnum value", ("k", AT, SELF, {"level": Level.HIGH}, "t", ())
    yield "OrderedDict payload", ("k", AT, SELF, OrderedDict([("b", 1), ("a", 2)]), "t", ())
    yield "OrderedDict inside", ("k", AT, SELF, {"o": OrderedDict([("b", 1)])}, "t", ())
    yield "proxy payload", ("k", AT, SELF, MappingProxyType({"a": [1]}), "t", ())
    yield "StrEnum kind", (Colour.RED, AT, SELF, {"a": 1}, "t", ())
    yield "a parent that is not text", ("k", AT, SELF, {"a": 1}, "t", (5, "E1"))


def test_edge_case_payloads_give_the_v4_events_and_digests() -> None:
    fast, reference = EventLog(), ReferenceLog()
    for label, args in calls():
        kind, at, source, payload, trace, parents = args
        parent_list = list(parents)  # type: ignore[call-overload]
        event = fast.append(kind, at, source, payload, trace, tuple(parent_list))  # type: ignore[arg-type]
        want = reference.append(kind, at, source, copy_plain(payload), trace, tuple(parent_list))  # type: ignore[arg-type]
        assert same_event(event, want), label
        assert fast.digests == reference.digests, label
        # and once more with the same objects (the source and timestamp caches)
        again = fast.append(kind, at, source, payload, trace, parent_list)  # type: ignore[arg-type]
        want = reference.append(kind, at, source, copy_plain(payload), trace, parent_list)  # type: ignore[arg-type]
        assert same_event(again, want), label
        assert fast.digests == reference.digests, label
    assert fast.verify()


def test_generator_parents_are_consumed_once_like_v4() -> None:
    fast, reference = EventLog(), ReferenceLog()
    event = fast.append("k", AT, SELF, {}, "t", (p for p in ("E9", "E8", "E9")))  # type: ignore[arg-type]
    want = reference.append("k", AT, SELF, {}, "t", (p for p in ("E9", "E8", "E9")))  # type: ignore[arg-type]
    assert event.parent_event_ids == want.parent_event_ids == ("E9", "E8")
    assert fast.digests == reference.digests


def test_a_new_source_tuple_each_time_and_equal_numbers_of_other_types() -> None:
    """1 and 1.0 are equal keys but different JSON ("1" vs "1.0"): never mixed up."""
    fast, reference = EventLog(), ReferenceLog()
    for index in range(150):
        reliability: float = 1 if index % 2 else 1.0
        source = ("owner", f"app{index % 3}", reliability)
        fast.append("k", AT, source, {"i": index}, "t")
        reference.append("k", AT, source, {"i": index}, "t")
        assert type(fast.events[-1].source.reliability) is type(reliability)
    assert fast.digests == reference.digests


def test_a_later_change_to_the_callers_payload_changes_nothing() -> None:
    log = EventLog()
    payload: dict[str, object] = {"list": [1, 2], "inner": {"a": 1}}
    event = log.append("k", AT, SELF, payload, "t")
    payload["list"].append(3)  # type: ignore[attr-defined]
    payload["inner"]["a"] = 2  # type: ignore[index]
    payload["new"] = True
    assert event.payload["list"] == (1, 2)
    assert event.payload["inner"]["a"] == 1  # type: ignore[index]
    assert "new" not in event.payload
    assert log.verify()
    with pytest.raises(TypeError):
        event.payload["new"] = 1  # type: ignore[index]


def raises_like_v4(call: Callable[[EventLog | ReferenceLog], object]) -> None:
    fast, reference = EventLog(), ReferenceLog()
    fast.append("k", AT, SELF, {"warm": 1}, "t")  # warm the caches
    reference.append("k", AT, SELF, {"warm": 1}, "t")
    with pytest.raises(Exception) as want:
        call(reference)
    with pytest.raises(Exception) as got:
        call(fast)
    assert type(got.value) is type(want.value)
    assert str(got.value) == str(want.value)
    assert len(fast.events) == len(fast.digests) == len(reference.events) == 1
    assert fast.get("E000002") is None


BAD_CALLS: list[tuple[str, Callable[[EventLog | ReferenceLog], object]]] = [
    ("nan", lambda log: log.append("k", AT, SELF, {"x": float("nan")}, "t")),
    ("inf deep", lambda log: log.append("k", AT, SELF, {"x": [{"y": float("inf")}]}, "t")),
    ("non-text key", lambda log: log.append("k", AT, SELF, {1: "a"}, "t")),  # type: ignore[dict-item]
    ("set value", lambda log: log.append("k", AT, SELF, {"s": {1}}, "t")),
    ("datetime value", lambda log: log.append("k", AT, SELF, {"d": AT}, "t")),
    ("payload not a mapping", lambda log: log.append("k", AT, SELF, [1], "t")),  # type: ignore[arg-type]
    ("empty kind", lambda log: log.append("", AT, SELF, {}, "t")),
    ("blank kind", lambda log: log.append("  ", AT, SELF, {}, "t")),
    ("blank trace", lambda log: log.append("k", AT, SELF, {}, " \n")),
    ("kind not text", lambda log: log.append(5, AT, SELF, {}, "t")),  # type: ignore[arg-type]
    ("naive time", lambda log: log.append("k", datetime(2026, 1, 1), SELF, {}, "t")),
    ("time not a datetime", lambda log: log.append("k", "2026", SELF, {}, "t")),  # type: ignore[arg-type]
    ("own parent", lambda log: log.append("k", AT, SELF, {}, "t", ("E000002",))),
    ("unhashable parent", lambda log: log.append("k", AT, SELF, {}, "t", ([1],))),  # type: ignore[arg-type]
    ("bool reliability", lambda log: log.append("k", AT, ("a", "b", True), {}, "t")),
    ("reliability > 1", lambda log: log.append("k", AT, ("a", "b", 1.5), {}, "t")),
    ("empty source id", lambda log: log.append("k", AT, ("a", " ", 0.5), {}, "t")),
    ("short source", lambda log: log.append("k", AT, ("a", "b"), {}, "t")),  # type: ignore[arg-type]
    ("bad source and bad kind", lambda log: log.append("", AT, ("a", "b", 2.0), {}, "t")),
]


@pytest.mark.parametrize("label,call", BAD_CALLS, ids=[label for label, _ in BAD_CALLS])
def test_malformed_calls_raise_the_v4_error_and_append_nothing(
    label: str, call: Callable[[EventLog | ReferenceLog], object]
) -> None:
    raises_like_v4(call)


def test_the_exact_path_is_the_v4_append() -> None:
    """``_append_exact`` (the fallback) is the reference too, on the same edge cases."""
    exact, reference = EventLog(), ReferenceLog()
    for label, args in calls():
        kind, at, source, payload, trace, parents = args
        parent_list = list(parents)  # type: ignore[call-overload]
        event = exact._append_exact(kind, at, source, payload, trace, parent_list)  # type: ignore[arg-type]
        want = reference.append(kind, at, source, copy_plain(payload), trace, parent_list)  # type: ignore[arg-type]
        assert same_event(event, want), label
    assert exact.digests == reference.digests


def test_the_fast_freeze_equals_provenance_freeze() -> None:
    from duckmind.provenance import _freeze_json

    for _, args in calls():
        payload = args[3]
        try:
            fast = memory._freeze_plain(payload)
        except memory._NotPlain:
            continue
        assert same_frozen(fast, _freeze_json(payload))
