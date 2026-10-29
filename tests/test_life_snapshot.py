"""The neural duck's slow state as a versioned snapshot (docs/design/GROWTH-G1-2026-10-10.md, part 3):
what is kept, how time that passed while it was off counts, and what is refused."""

from __future__ import annotations

import json
import math
import os
import stat

import pytest

from duckmind.life import snapshot as snap
from duckmind.neuro.drives import Drives
from duckmind.neuro.genome import Genome
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.learn import Learner

LOOK = [0.30, 0.40, 0.50, 0.20, 0.10, 0.60]


class FakeBrain:
    """The parts of a NeuroBrain the snapshot touches."""

    def __init__(self) -> None:
        p = Genome.from_seed(4107, gd=0).p
        self.learn = Learner(p)
        self.hab = Habituation(p)
        self.drives = Drives()
        self._t0, self._t_prev = None, None

    def run_to(self, t_s: float) -> None:
        self._t0, self._t_prev = 0, int(t_s * 1000)


def taught() -> FakeBrain:
    brain = FakeBrain()
    brain.run_to(400.0)
    learn, t = brain.learn, 400.0
    pid = learn.see_person(5, LOOK, t - 30)
    learn.bind_name(pid, "小明", "小明", t - 20)
    learn.people[pid].bond = 0.6
    learn.people[pid].exchanges = 4
    learn.pend_name(pid, "阿姨")
    learn.keep_note(t - 10, pid, "喜歡抱抱")
    for k in range(3):  # a scare spot: three hard falls in one cell
        learn.fell((1.2, 0.3, 0.0), t - 100 + k * 10)
        assert learn.righted(t - 100 + k * 10 + 6.0, helped=False)
    learn.mark((2.0, 2.0, 0.0), 1.0, family=2.0)
    learn.see_thing(3, "toy", (0.0, 0.0, 0.0), 1.0, 0.5, t - 50, 5.0, 0.0)
    learn.cue("sound:name", t - 5)
    learn.outcome(1.0, t - 4)
    learn.at_dock((0.1, -0.2, 0.0))
    for k in range(6):
        brain.hab.present("sound:clap", 0.9, t - 60 + k, reflex="startle")
    brain.hab.sensitise("sound:clap", 0.5, t - 10)
    brain.drives.fear = 0.4
    brain.drives.energy = 0.3
    return brain


CTX = {"home": "default_home,no_nest", "start": "mat"}


def capture(brain: FakeBrain, wall: float = 1000.0):
    from datetime import datetime

    from duckmind.life.clock import TPE

    return snap.capture(
        brain,
        name="蛋黃",
        seed=4107,
        gd=0,
        calendar=datetime(2026, 10, 10, 15, 0, tzinfo=TPE),
        context=CTX,
        why="test",
        wall=wall,
        talk=0.4,
    )


def test_a_snapshot_is_plain_json_and_brings_back_people_places_and_habits() -> None:
    brain = taught()
    data = json.loads(json.dumps(capture(brain), ensure_ascii=False))  # it survives the file
    assert data["format"] == snap.SNAPSHOT_FORMAT and data["version"] == snap.SNAPSHOT_VERSION
    fresh = FakeBrain()
    talk: list[float] = []
    report = snap.apply(
        fresh, data, seed=4107, context=CTX, wall=1000.0, set_talk=talk.append
    )  # no time passed
    assert report["names"] == ["小明"] and report["people"] == 1 and report["avoided"] == 1
    person = fresh.learn.person_named("小明")
    assert person is not None and person.pending == {"阿姨": 1} and person.exchanges == 4
    assert math.isclose(person.bond, 0.6) and person.mean == pytest.approx(LOOK)
    assert fresh.learn.avoided_near((1.2, 0.3, 0.0))  # the scare spot is still avoided
    assert fresh.learn.dock == (0.1, -0.2) and fresh.learn.cues == brain.learn.cues
    assert talk == [0.4]
    # what it got used to: the clap's habituation level is the same (clock 0 now: a fresh brain)
    assert fresh.hab.level("sound:clap", 0.0) == pytest.approx(brain.hab.level("sound:clap", 400.0))
    assert fresh.hab.level("sound:clap", 0.0) < 0.9
    assert fresh.drives.fear == pytest.approx(0.4) and fresh.drives.energy == pytest.approx(0.3)
    assert fresh.learn.notes[-1][1:] == (person.pid, "喜歡抱抱")
    # a new person afterwards is a new record, never the old id again
    assert fresh.learn.see_person(9, [0.9, 0.1, 0.1, 0.9, 0.9, 0.1], 5.0) != person.pid


def test_the_time_it_was_off_counts() -> None:
    brain = taught()
    data = capture(brain, wall=1000.0)
    five_hours = 5 * 3600.0
    later = FakeBrain()
    snap.apply(later, data, seed=4107, context=CTX, wall=1000.0 + five_hours)
    person = later.learn.person_named("小明")
    assert person is not None
    assert person.bond == pytest.approx(0.6 * math.exp(-five_hours / (4 * 3600)))  # bonds fade
    assert later.hab.level("sound:clap", 0.0) > 0.99  # habituation recovered by itself
    assert later.drives.fear == pytest.approx(0.0, abs=1e-6)  # a fright is over
    assert person.seen_t < -five_hours  # it has not seen them for five hours, as it is
    assert later.learn.person_named("小明") is not None  # but the name and the look are kept
    assert later.learn.avoided_near((1.2, 0.3, 0.0))  # and a scare spot is a scare spot still


def test_places_are_kept_only_when_the_body_starts_where_it_did() -> None:
    data = capture(taught())
    other = FakeBrain()
    report = snap.apply(
        other, data, seed=4107, context={"home": "apartment", "start": "mat"}, wall=1000.0
    )
    assert report["skipped"] == ["places", "things", "dock"] and not other.learn.places
    assert other.learn.person_named("小明") is not None  # people and habits do not depend on it
    assert other.hab.level("sound:clap", 0.0) < 0.9


def test_a_snapshot_of_another_duck_or_a_newer_build_is_never_applied() -> None:
    data = capture(taught())
    with pytest.raises(snap.SnapshotUnusable, match="seed"):
        snap.apply(FakeBrain(), data, seed=1728, context=CTX)
    newer = {**data, "version": snap.SNAPSHOT_VERSION + 1}
    with pytest.raises(snap.SnapshotUnusable, match="newer"):
        snap.apply(FakeBrain(), newer, seed=4107, context=CTX)
    with pytest.raises(snap.SnapshotUnusable, match="not a neural snapshot"):
        snap.apply(FakeBrain(), {"format": "other"}, seed=4107, context=CTX)


def test_the_file_is_private_atomic_and_keeps_the_one_before(tmp_path) -> None:
    path, prev = tmp_path / "duck" / "neural.json", tmp_path / "duck" / "neural.prev.json"
    brain = taught()
    snap.write(path, capture(brain), prev)
    first = snap.read(path)
    assert first is not None and not prev.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # names and looks: the owner only
    brain.drives.fear = 0.9
    snap.write(path, capture(brain, wall=2000.0), prev)
    assert snap.read(prev) == first and snap.read(path)["saved_wall"] == 2000.0
    assert not list(path.parent.glob("*.tmp"))
    assert snap.read(tmp_path / "nothing.json") is None


def test_a_damaged_file_is_set_aside_with_a_note_not_overwritten(tmp_path) -> None:
    path = tmp_path / "neural.json"
    path.write_text("{ not json")
    with pytest.raises(snap.SnapshotUnusable, match="damaged"):
        snap.read(path)
    moved = snap.set_aside(path, "damaged")
    assert not path.exists() and moved.read_text() == "{ not json"
    assert moved.with_name(moved.name + ".why.txt").read_text().strip() == "damaged"
