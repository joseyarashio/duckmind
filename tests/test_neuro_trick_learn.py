"""Phase 2 「學會聽話」: what praise, a scold and time do to a trick's proficiency, how that chance reaches the
command, and that the proficiency survives a save (snapshot 1 → 2)."""

from __future__ import annotations

import json
import random
from datetime import datetime

import pytest

from duckmind.life import snapshot as snap
from duckmind.life.clock import TPE
from duckmind.neuro import tricks as T
from duckmind.neuro.drives import Drives
from duckmind.neuro.genome import Genome
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.learn import Learner

GD = 50


def test_a_new_book_has_five_tricks_all_at_nothing() -> None:
    book = T.TrickBook()
    assert set(book.skills) == {"lie_down", "spin", "bow", "shake", "flip"}
    assert all(s.proficiency == 0.0 and s.total == 0 for s in book.skills.values())


# -- praise ---------------------------------------------------------------------------------------


def test_each_praise_of_a_day_is_a_smaller_step_and_the_day_has_a_cap() -> None:
    book = T.TrickBook()
    steps = [book.praise("bow", GD).delta for _ in range(6)]
    assert steps == pytest.approx(list(T.STEPS)) or sum(steps) == pytest.approx(T.CAP_PER_GD)
    assert steps == sorted(steps, reverse=True)  # never a bigger step later in the day
    capped = book.praise("bow", GD)
    assert capped.capped and capped.delta == pytest.approx(0.0)
    assert book.proficiency("bow") == pytest.approx(T.CAP_PER_GD)  # the day's cap, not a step more
    assert book.skills["bow"].tries_today == 7  # the tries still count: it practised


def test_the_cap_is_per_trick_and_per_growth_day() -> None:
    book = T.TrickBook()
    for _ in range(8):
        book.praise("bow", GD)
    first_day = book.proficiency("bow")
    assert book.praise("spin", GD).delta == pytest.approx(T.STEPS[0])  # another trick: its own cap
    assert not book.praise("bow", GD + 1).capped  # the next growth day: steps from the top again
    assert book.proficiency("bow") > first_day - 0.05  # (a day's rest costs only a little of it)


def test_the_bar_cannot_pass_one() -> None:
    book = T.TrickBook()
    book.skills["bow"].proficiency = 0.99
    book.praise("bow", GD)
    assert book.proficiency("bow") == 1.0


def test_shaping_is_half_a_step_and_is_counted_as_shaped() -> None:
    plain, shaped = T.TrickBook(), T.TrickBook()
    a = plain.praise("shake", GD)
    b = shaped.praise("shake", GD, shaped=True)
    assert b.delta == pytest.approx(a.delta * T.SHAPE_SHARE) and b.shaped and not a.shaped
    assert shaped.skills["shake"].shaped == 1 and plain.skills["shake"].shaped == 0


# -- a scold --------------------------------------------------------------------------------------


def test_a_scold_takes_half_a_step_off_and_the_try_does_not_count() -> None:
    book = T.TrickBook()
    book.skills["spin"].proficiency = 0.5
    before = book.skills["spin"]
    tries, total = before.tries_today, before.total
    out = book.scold("spin", GD)
    assert out.delta == pytest.approx(-T.SCOLD_SHARE * T.STEPS[0])
    assert book.proficiency("spin") == pytest.approx(0.5 - T.SCOLD_SHARE * T.STEPS[0])
    assert (book.skills["spin"].tries_today, book.skills["spin"].total) == (tries, total)


def test_a_scold_never_takes_a_learned_trick_below_the_floor_and_never_below_zero() -> None:
    book = T.TrickBook()
    book.skills["bow"].proficiency = T.FLOOR + 0.001
    for _ in range(10):
        book.scold("bow", GD)
    assert book.proficiency("bow") == pytest.approx(T.FLOOR)
    book.scold("flip", GD)  # a trick it never had: nothing to take off
    assert book.proficiency("flip") == 0.0


# -- time -----------------------------------------------------------------------------------------


def test_a_practised_trick_fades_toward_the_floor_with_the_time_constant_and_stays_there() -> None:
    book = T.TrickBook()
    book.settle(GD)
    book.skills["bow"].proficiency, book.skills["bow"].total = 0.9, 12
    book.settle(GD + 12)  # one time constant
    above = (0.9 - T.FLOOR) * 0.3678794
    assert book.proficiency("bow") == pytest.approx(T.FLOOR + above, abs=1e-4)
    book.settle(GD + 400)
    assert book.proficiency("bow") == pytest.approx(T.FLOOR, abs=1e-4)  # rusty, never gone


def test_a_trick_never_practised_stays_at_nothing_and_one_below_the_floor_is_not_lifted() -> None:
    book = T.TrickBook()
    book.settle(GD)
    book.skills["spin"].proficiency = 0.05  # (a scold's leftovers: below the floor)
    book.skills["spin"].total = 3
    book.settle(GD + 100)
    assert book.proficiency("flip") == 0.0 and book.proficiency("spin") == pytest.approx(0.05)


def test_a_clock_that_goes_back_does_not_forgive_the_days_cap() -> None:
    book = T.TrickBook()
    for _ in range(8):
        book.praise("bow", GD)
    book.settle(GD - 3)
    assert book.praise("bow", GD).capped


# -- the trick book as the page sees it -----------------------------------------------------------


def test_the_book_view_says_where_each_trick_stands_and_suggests_the_next() -> None:
    book = T.TrickBook()
    book.skills["bow"].proficiency, book.skills["bow"].total = 0.9, 5
    book.skills["spin"].proficiency, book.skills["spin"].total = 0.3, 2
    rows = {r["id"]: r for r in book.view(GD, locked=("flip",))}
    assert rows["bow"]["state"] == "learned" and rows["spin"]["state"] == "half"
    assert rows["shake"]["state"] == "none" and rows["flip"]["state"] == "locked"
    assert rows["flip"]["unlocked"] is False and rows["bow"]["proficiency_pct"] == 90
    # the one to teach next is the open trick it knows least (not the locked one)
    assert [r["id"] for r in rows.values() if r["suggest"]] == ["lie_down"]


# -- obeying --------------------------------------------------------------------------------------


def _rate(p: float, n: int = 4000, **kw: float | str) -> float:
    rng = random.Random(7)
    chance = T.obey_probability(p, **kw)
    return sum(rng.random() < chance for _ in range(n)) / n


def test_an_untaught_trick_is_done_at_most_three_times_in_ten_and_a_learned_one_always() -> None:
    assert 0.10 <= T.obey_probability(0.0) <= T.OBEY_BASE  # a stranger's guess
    assert T.obey_probability(0.0, bond=1.0, primary="snuggler") <= 0.30  # the closest bond is still a guess
    assert _rate(0.0) <= 0.30
    assert T.obey_probability(0.8) >= 0.99 and T.obey_probability(1.0) == pytest.approx(1.0)
    assert [T.obey_probability(p / 10) for p in range(11)] == sorted(
        T.obey_probability(p / 10) for p in range(11)
    )  # more proficient never means less likely


def test_fear_and_tiredness_lower_the_chance_a_lot_and_calm_does_not() -> None:
    calm = T.obey_probability(0.9)
    scared = T.obey_probability(0.9, fear=0.5)
    sleepy = T.obey_probability(0.9, sleepiness=0.9)
    both = T.obey_probability(0.9, fear=0.5, sleepiness=0.9)
    assert calm >= 0.99
    assert scared <= 0.5 * calm and sleepy <= 0.5 * calm and both < min(scared, sleepy)
    assert T.obey_probability(0.9, fear=0.1, sleepiness=0.3) == pytest.approx(calm)


def test_closeness_and_temperament_only_nudge_the_odds_never_decide_them() -> None:
    mid = 0.45
    far = T.obey_probability(mid, bond=0.0, primary="explorer")
    near = T.obey_probability(mid, bond=1.0, primary="snuggler")
    plain = T.obey_probability(mid)
    assert near > plain > far  # the direction is right …
    assert near - far < 0.35  # … the size is a nudge
    for who in ("", "snuggler", "explorer", "dreamer"):  # a trick it knows is done by anyone
        assert T.obey_probability(0.9, bond=0.0, primary=who) >= 0.99
        assert T.obey_probability(0.0, bond=1.0, primary=who) <= 0.30  # … and none is learnt by closeness


# -- the file -------------------------------------------------------------------------------------


class _Brain:
    """The parts of a brain the snapshot touches."""

    def __init__(self) -> None:
        p = Genome.from_seed(4107, gd=0).p
        self.learn = Learner(p)
        self.hab = Habituation(p)
        self.drives = Drives()
        self._t0, self._t_prev = 0, 0


def _capture(brain: _Brain) -> dict:
    return snap.capture(
        brain, name="蛋黃", seed=4107, gd=GD, calendar=datetime(2026, 10, 25, 9, 0, tzinfo=TPE),
        context={"home": "h", "start": "s"}, why="test", wall=1000.0,
    )  # fmt: skip


def test_proficiency_survives_a_save_and_a_restart() -> None:
    brain = _Brain()
    for _ in range(8):  # more than a day allows: the cap is reached before the save
        brain.learn.tricks.praise("bow", GD)
    brain.learn.tricks.scold("spin", GD)
    brain.learn.tricks.praise("shake", GD, shaped=True)
    kept = json.loads(json.dumps(_capture(brain)))  # through the file
    assert kept["version"] == snap.SNAPSHOT_VERSION == 2
    fresh = _Brain()
    snap.apply(fresh, kept, seed=4107, context={"home": "h", "start": "s"}, wall=1000.0)
    for trick in T.TRICKS:
        assert fresh.learn.tricks.skills[trick] == brain.learn.tricks.skills[trick], trick
    assert fresh.learn.tricks.day == GD
    assert fresh.learn.tricks.praise("bow", GD).capped  # the day's cap came back with it: no second chance


def test_a_version_one_snapshot_loads_and_its_duck_knows_no_tricks() -> None:
    brain = _Brain()
    old = json.loads(json.dumps(_capture(brain)))
    old["version"] = 1
    old["learner"].pop("tricks")  # exactly what the last build wrote
    fresh = _Brain()
    fresh.learn.tricks.praise("bow", GD)  # (something in the way: the load replaces it)
    snap.apply(fresh, old, seed=4107, context={"home": "h", "start": "s"}, wall=1000.0)
    assert all(s.proficiency == 0.0 for s in fresh.learn.tricks.skills.values())
    assert snap.upgrade(old)["version"] == 2


def test_a_newer_snapshot_than_this_build_is_still_refused() -> None:
    newer = {**_capture(_Brain()), "version": snap.SNAPSHOT_VERSION + 1}
    with pytest.raises(snap.SnapshotUnusable, match="newer"):
        snap.upgrade(newer)
