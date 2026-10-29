"""One duck, one life, one clock (docs/design/GROWTH-G1-2026-10-10.md; runs in a child process, see
tests/test_life_isolated.py): the duck record, Duck Core
living on to the calendar, the neural duck built from Duck Core's state, and a restart that does
not make it forget: the people it knows, the places it learnt to avoid, what it got used to."""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore.api import Config
from duckmind.life import snapshot as snap
from duckmind.life.clock import TPE
from duckmind.life.record import LifeError, LifeRecord
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.neuro_live import DayClock, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimState, lights_payload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_neuro_live import robot_state  # noqa: E402

BIRTH = datetime(2026, 10, 10, 15, 0, tzinfo=TPE)
LOOK = [0.30, 0.40, 0.50, 0.20, 0.10, 0.60]


class Wall:
    """A wall clock the test moves (epoch seconds)."""

    def __init__(self, at: datetime) -> None:
        self.t = at.timestamp()

    def __call__(self) -> float:
        return self.t


class Life:
    """One start of the server's life side: SimState, LifeHost, the neural driver (no body)."""

    def __init__(
        self,
        folder: Path,
        wall: Wall,
        *,
        every_s: float = 600.0,
        attach: bool = False,
        backdate: int = 0,
    ) -> None:
        if (folder / "life.json").is_file():
            self.record = LifeRecord.load(folder)
        else:
            self.record = LifeRecord.create(
                folder,
                name="蛋黃",
                seed=4107,
                birth=BIRTH,
                nest=False,
                tof=True,
                backdate_days=backdate,
            )
        self.wall = wall
        self.clock = DayClock(None, wall=wall)
        config = Config(
            seed=4107, name="蛋黃", tof=True, nest=False, start=self.record.birth.isoformat()
        )
        self.state = SimState(
            config,
            None,
            self.record.core_path,
            life=self.record,
            clock=self.clock,
            attach_neural=attach,
        )
        self.host = LifeHost(self.record, self.state, self.clock, every_s=every_s, wall=wall)
        self.state.life_host = self.host
        self.driver = NeuroDriver(self.host.spec(NeuroSpec(duck="4107:0", seed=4107)))
        self.driver.day_clock = self.clock
        self.host.attach(self.driver)
        self.report = self.host.restore()

    @property
    def brain(self) -> Any:
        return self.driver.lanes.brain

    @property
    def core(self) -> Any:
        return self.state.session.core

    def live(self, seconds: float) -> None:
        for k in range(int(seconds / 0.02)):
            self.driver.tick(robot_state(k * 0.02))

    def teach(self) -> None:
        brain, learn = self.brain, self.brain.learn
        t = snap.brain_time(brain) + 1.0
        pid = learn.see_person(5, LOOK, t)
        learn.bind_name(pid, "小明", "小明", t)
        learn.people[pid].bond = 0.5
        for k in range(3):  # a scare spot: three hard falls in one cell
            learn.fell((1.2, 0.3, 0.0), t + k * 10)
            assert learn.righted(t + k * 10 + 6.0, helped=False)
        for k in range(6):
            brain.hab.present("sound:clap", 0.9, t + k, reflex="startle")

    def remembered(self) -> dict[str, Any]:
        brain = self.brain
        t = snap.brain_time(brain)
        return {
            "names": sorted(brain.learn.person_names().values()),
            "avoided": bool(brain.learn.avoided_near((1.2, 0.3, 0.0))),
            "clap": round(brain.hab.level("sound:clap", t), 2),
        }


def test_a_new_duck_hatches_as_a_hatchling_with_its_record_on_disk(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH))
    folder = tmp_path / "duck"
    assert life.core.s.gd == 0 and life.core.stage().stage_id == "hatchling"
    assert life.core.s.hatched_at == BIRTH and life.core.s.seed == 4107
    genome = life.driver.lanes.genome
    assert (genome["seed"], genome["gd"], genome["stage"], genome["branch"]) == (
        4107,
        0,
        "hatchling",
        None,
    )
    record = json.loads((folder / "life.json").read_text())
    assert (record["name"], record["seed"], record["birth"]) == ("蛋黃", 4107, BIRTH.isoformat())
    assert record["hardware"] == {"nest": False, "tof": True}
    saved = json.loads((folder / "core.json").read_text())
    assert saved["config"]["start"] == BIRTH.isoformat() and saved["config"]["seed"] == 4107
    assert saved["config"]["nest"] is False and saved["config"]["tof"] is True
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(folder / "core.json").st_mode) == 0o600
    life.host.start()  # …and its mind's file exists from the first moment, empty
    try:
        first = json.loads((folder / "neural.json").read_text())
        assert first["why"] == "hatch" and first["learner"]["people"] == []
        assert (
            first["duck"]["gd"] == 0
            and stat.S_IMODE(os.stat(folder / "neural.json").st_mode) == 0o600
        )
    finally:
        life.host._stop.set()
    with pytest.raises(LifeError, match="already holds a duck"):
        LifeRecord.create(folder, name="x", seed=1)


def test_a_clean_restart_remembers_the_people_the_scare_spots_and_what_it_got_used_to(
    tmp_path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.live(3.0)
    first.teach()
    before = first.remembered()
    assert before == {"names": ["小明"], "avoided": True, "clap": before["clap"]}
    assert before["clap"] < 0.9  # it got used to claps
    head = first.core.log.head
    wall.t += 20.0  # twenty seconds go by, then the server is stopped cleanly
    first.host.stop()
    assert (folder / "neural.json").is_file() and first.host.last_why == "stop"

    wall.t += 30.0  # …and started again half a minute later
    second = Life(folder, wall)
    assert second.report is not None and second.report["names"] == ["小明"]
    assert second.report["people"] == 1 and second.report["avoided"] == 1
    after = second.remembered()
    assert after["names"] == ["小明"] and after["avoided"] is True
    assert before["clap"] - 0.02 <= after["clap"] <= 1.0  # recovering a little in 50 s, no more
    assert after["clap"] < 0.9
    second.live(3.0)  # and it carries on living with that mind
    assert second.remembered()["names"] == ["小明"]
    # Duck Core: the same duck, replayed (not hatched again), living on to the calendar
    assert second.core.s.hatched_at == BIRTH and second.core.s.seed == 4107
    assert second.core.log.events[0].event_id == first.core.log.events[0].event_id
    assert second.core.log.head != head  # it lived the 50 s on
    assert second.state.session.core.s.now >= BIRTH + timedelta(seconds=50)


def test_a_crash_loses_only_what_came_after_the_last_snapshot(tmp_path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall, every_s=1.0)
    first.live(2.0)
    first.teach()
    assert first.host.snapshot_now("periodic")  # the ten-minute one
    first.brain.learn.bind_name(  # a second person, taught after it
        first.brain.learn.see_person(8, [0.9, 0.9, 0.1, 0.9, 0.2, 0.2], 50.0), "阿姨", "阿姨", 50.0
    )
    wall.t += 60.0  # killed: no stop(), no last snapshot
    second = Life(folder, wall)
    assert second.report is not None and second.report["names"] == ["小明"]


def test_time_off_counts_for_growth_as_the_familys_absence_and_the_genome_follows_the_day(
    tmp_path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    for _ in range(60):  # an hour awake on the first day (a minute at a time), then it is stopped
        wall.t += 60.0
        with first.state.lock:
            first.state.grow()
    first.host.stop()
    wall.t += 5 * 86400.0  # five days off
    second = Life(folder, wall)
    credits = [e.payload["credit"] for e in second.core.log.events if e.kind == "growth.gd"]
    assert credits == ["full", "half", "half", "half", "half"]  # a day lived, then days away
    assert second.core.s.gd == 3  # not the calendar's 5: the days away are half days
    genome = second.driver.lanes.genome
    assert genome["gd"] == 3 and genome["stage"] == "hatchling"  # built at the day it has
    assert "wander" not in genome["unlocked"]  # …and it cannot walk until its day 5
    # a duck that is on 24 h a day earns whole days
    wall2 = Wall(BIRTH)
    fresh = Life(tmp_path / "other", wall2)
    for _ in range(6 * 24 * 15):  # six days, four minutes at a time
        wall2.t += 240.0
        with fresh.state.lock:
            fresh.state.grow()
    assert fresh.core.s.gd == 6
    assert {e.payload["credit"] for e in fresh.core.log.events if e.kind == "growth.gd"} == {"full"}


def test_the_demo_duck_never_overwrites_the_life_ducks_mind(tmp_path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.live(1.0)
    first.teach()
    saved_before = (
        (folder / "neural.json").read_bytes() if (folder / "neural.json").exists() else b""
    )
    first.driver.switch("1728:20")  # the page shows the snuggler preset
    assert not first.driver.spec.life and first.host.last_why == "switch"
    said = first.host.status()["snapshot"]  # …and it is said, not silent
    assert said["on_screen"] is False and "蛋黃" in said["problem"]
    first.brain.learn.bind_name(first.brain.learn.see_person(2, LOOK, 5.0), "路人", "路人", 5.0)
    assert first.host.snapshot_now("periodic") is False  # a demo duck's mind is not the life duck's
    assert "路人".encode() not in (folder / "neural.json").read_bytes()
    assert (folder / "neural.json").read_bytes() != saved_before  # (the switch saved the life duck)
    first.driver.switch(first.host.spec_string())  # back to 蛋黃, picked by its seed and day
    assert first.driver.spec.life and first.remembered()["names"] == ["小明"]
    assert first.host.status()["snapshot"]["problem"] is None and first.host.snapshot_now("x")
    assert first.driver.lanes.genome["stage"] == "hatchling"


def test_an_unusable_snapshot_is_set_aside_and_the_duck_starts_with_a_clean_mind(tmp_path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.live(1.0)
    first.teach()
    first.host.stop()
    data = json.loads((folder / "neural.json").read_text())
    data["version"] = 99  # written by a newer build
    (folder / "neural.json").write_text(json.dumps(data))
    second = Life(folder, wall)
    assert second.report is None and "newer" in (second.host.problem or "")
    assert second.remembered()["names"] == []
    assert list(folder.glob("neural.unused-*.json"))  # kept, not overwritten
    assert not (folder / "neural.json").exists()


def test_the_rooms_lights_follow_the_same_clock_as_the_neural_duck(tmp_path) -> None:
    wall = Wall(datetime(2026, 10, 10, 10, 0, tzinfo=TPE))  # the morning
    life = Life(tmp_path / "duck", wall)
    sent: list[Any] = []

    class FakeLive:
        neuro = life.driver
        clock = None

        def take_fresh(self) -> bool:
            return False

        def set_clock(self, local: datetime, rate: float) -> None:
            sent.append((local, rate))

        def truth_now(self) -> None:
            return None

    life.state.live = FakeLive()  # type: ignore[assignment]
    for hour, label in ((10, "白天"), (18, "晚上開燈"), (2, "月光")):
        life.clock.set(hour * 60)  # the page's time-of-day choice
        light = lights_payload(life.state.light_time())
        assert light["when"][11:13] == f"{hour:02d}" and light["label"] == label
        # the room, the world camera (sync_clock) and the duck's circadian minute agree
        life.state.sync_clock()
        assert sent[-1][0].hour == hour and abs(life.clock.minute() - hour * 60) < 1.0
    # Duck Core's own clock is the calendar's, not what the lights follow in this mode
    assert life.core.s.now.hour != 2 and life.state.light_time() == life.clock.local()


def test_what_would_hatch_it_again_or_skip_its_time_is_refused(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH))
    for method, params in (
        ("new", {}),
        ("advance", {"minutes": 60}),
        ("live", {"family": "dual_income", "days": 3}),
        ("party", {}),
        ("realtime", {"on": True}),
        ("command", {"text": "過 3 天"}),
    ):
        with pytest.raises(ValueError) as error:
            life.state.refuse_in_life(method, params)
        assert "蛋黃" in str(error.value)
    life.state.refuse_in_life("command", {"text": "摸摸頭"})  # an ordinary act goes through
    life.state.refuse_in_life("act", {"action": "touch"})


def test_the_life_status_has_counts_only_no_names(tmp_path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.live(1.0)
    first.teach()
    first.host.stop()
    second = Life(folder, wall)
    status = second.host.status()
    assert status["growth_day"] == 0 and status["restored"]["people"] == 1
    assert "小明" not in json.dumps(status, ensure_ascii=False)


def test_a_snapshot_is_taken_every_night_and_every_so_often(tmp_path) -> None:
    import time

    wall = Wall(datetime(2026, 10, 10, 23, 59, tzinfo=TPE))
    life = Life(tmp_path / "duck", wall, every_s=3600.0)
    life.host.watch_s = 0.02
    life.live(1.0)
    life.host.start()
    try:
        deadline = time.time() + 5
        wall.t += 120.0  # past midnight on the duck's calendar: a new day
        while life.host.last_why != "night" and time.time() < deadline:
            time.sleep(0.02)
        assert life.host.last_why == "night" and (tmp_path / "duck" / "neural.json").is_file()
        saves = life.host.saves
        wall.t += 3600.0 + 1.0  # and one more hour: the every-so-often one
        while life.host.last_why != "periodic" and time.time() < deadline + 5:
            time.sleep(0.02)
        assert life.host.last_why == "periodic" and life.host.saves > saves
    finally:
        life.host._stop.set()


def test_the_life_ducks_branch_stays_its_natures_while_duck_core_lives_a_shadow_life(
    tmp_path,
) -> None:
    wall = Wall(BIRTH)
    life = Life(tmp_path / "duck", wall)
    for _ in range(50 * 24 * 15):  # fifty days, four minutes at a time
        wall.t += 240.0
        with life.state.lock:
            life.state.grow()
    assert life.core.s.gd == 50
    assert life.core.s.branch == "dreamer"  # the shadow life's own naps (not the duck's experience)
    genome = life.host.genome_builder()()
    assert (genome.gd, genome.stage, genome.branch) == (50, "juvenile", "explorer")


def test_a_backdated_hatch_starts_at_growth_day_48_as_todays_explorer_and_grows_on(
    tmp_path,
) -> None:
    """Owner decision 2026-10-10: 蛋黃 is hatched 48 days ago (Duck Core lives them awake at creation)
    and is, parameter for parameter, today's explorer; from then on it grows in real time."""
    import dataclasses

    from duckmind.neuro.genome import preset

    folder, wall = tmp_path / "duck", Wall(BIRTH)
    record = LifeRecord.create(
        folder, name="蛋黃", seed=4107, birth=BIRTH, nest=False, tof=True, backdate_days=48
    )
    assert record.birth == BIRTH - timedelta(days=48) and record.backdated_days == 48
    first = Life(folder, wall)
    credits = [e.payload["credit"] for e in first.core.log.events if e.kind == "growth.gd"]
    assert credits == ["full"] * 48 and first.core.s.gd == 48
    assert first.core.stage().stage_id == "juvenile" and first.core.s.hatched_at == record.birth
    assert first.core.s.now >= BIRTH
    genome = first.host.genome_builder()()
    explorer = preset("explorer")  # the live duck until now: seed 4107 at growth day 60
    assert (genome.gd, genome.stage, genome.branch) == (48, "juvenile", "explorer")
    assert dataclasses.asdict(genome.p) == dataclasses.asdict(explorer.p)
    assert genome.gains == explorer.gains and genome.vocal_weights == explorer.vocal_weights
    assert (
        genome.unlocked == explorer.unlocked
        and {"flee", "wander", "return_dock"} <= genome.unlocked
    )
    assert (genome.axes_raw, genome.nature, genome.stage_days, genome.pace) == (
        explorer.axes_raw,
        explorer.nature,
        explorer.stage_days,
        explorer.pace,
    )
    assert first.driver.lanes.genome["stage"] == "juvenile"
    first.host.stop()
    # restarted a day later it is not aged again: the real days go on from day 48
    wall.t += 86400.0 + 60.0
    second = Life(folder, wall)
    assert second.core.s.gd == 49  # (day 48 was lived awake: the next midnight is a whole day)
    assert second.core.s.hatched_at == record.birth
    assert second.core.log.events[0].event_id == first.core.log.events[0].event_id
