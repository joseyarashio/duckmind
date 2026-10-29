"""The phase-4 invariant (design §2.2, G1; test T2): with LegacyWorld attached, a v4_1 duck
behaves exactly as the frozen v4 duck — every decision, draw, expression, walk, posture and
position change, dock, outing, nap, bid, moment, day's numbers and growth record, at the same
second, and the same draws in the end (Duck Core's counters and the world's together). Chain
heads differ, by design: v4_1 writes commands, input records and its own records.

The CI set: run A's household-v2 ducks 0–3 and legacy ducks 0–1 (the population gate's first
ducks, with popsim's own draws) for a week, and two stress ducks (falls 600× v4, weak righting,
failed stand-ups; with and without a nest) through two busy scripted days. A difference names
the first row where the two part. ``DUCKMIND_P4_DAYS`` makes the lives longer.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from phase4_harness import (
    Watch,
    first_difference,
    household_legacy,
    household_v2,
    live_legacy,
    live_v2,
    lively,
    lively_config,
    open_v40,
    open_v41,
    projection,
    walking_age,
)

DAYS = int(os.environ.get("DUCKMIND_P4_DAYS", "7"))


@dataclass(frozen=True)
class Case:
    name: str
    config: dict[str, Any]
    live: Callable[[Any], None]


def _v2(index: int) -> Case:
    config, kind = household_v2(index)
    return Case(f"v2-{index}-{kind}", config, lambda s: live_v2(s, kind, DAYS))


def _legacy(index: int) -> Case:
    config, first, second = household_legacy(index)
    return Case(f"legacy-{index}-{first}", config, lambda s: live_legacy(s, first, second, DAYS))


def _stress(seed: int, nest: bool) -> Case:
    def live(session: Any) -> None:
        walking_age(session)
        lively(session, seed, days=max(2, DAYS // 3))

    return Case(
        f"stress-{seed}-{'nest' if nest else 'floor'}", lively_config(seed, nest, True), live
    )


CASES = [*(_v2(i) for i in range(4)), *(_legacy(i) for i in range(2)), _stress(20, True)]
CASES.append(_stress(21, False))


def _run(open_session: Callable[[dict[str, Any]], Any], case: Case) -> tuple[Any, Any]:
    session = open_session(case.config)
    watch = Watch(session.core)
    case.live(session)
    return session, projection(session, watch)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_v41_with_legacy_world_behaves_exactly_like_v4(case: Case) -> None:
    old, before = _run(open_v40, case)
    new, after = _run(open_v41, case)
    difference = first_difference(before, after)
    assert not difference, f"{case.name}: v4_1 + LegacyWorld parts from v4 at {difference}"
    assert before.kinds == after.kinds
    assert new.core.log.head != old.core.log.head  # (new records, by design)
    assert new.core.log.verify()


# ---------------------------------------------------------------------------- the harness


def test_the_projection_is_blind_to_the_watch_and_sees_a_different_duck() -> None:
    """The check can fail: the watch changes nothing, the same duck projects the same, and a
    duck whose stand-ups fail differently does not."""
    case = _stress(20, True)
    session, first = _run(open_v40, case)
    unwatched = open_v40(case.config)
    case.live(unwatched)
    assert unwatched.core.log.head == session.core.log.head
    _again, second = _run(open_v40, case)
    assert not first_difference(first, second) and first.digest() == second.digest()
    other = Case(case.name, {**case.config, "stand_ppm": 650_000}, case.live)
    _other, third = _run(open_v40, other)
    assert first_difference(first, third), "a changed duck projected the same"
    assert any(row[0] == "state" and row[2] == "posture" for row in first.rows)
    assert any(row[0] == "event" and row[2] == "duck.walk" for row in first.rows)
    assert first.counters.get("stand.try", 0) > 0 and first.counters.get("fall", 0) > 0
