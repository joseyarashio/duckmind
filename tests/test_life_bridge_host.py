"""The server's side of the bridge (docs/design/GROWTH-G2-2026-10-10.md): a life whose Duck Core is
attached to the neural duck, what the host gives it, and a restart that keeps it."""

from __future__ import annotations

import json
import stat
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api
from duckmind.duckcore.api import Config
from duckmind.life.clock import TPE, DuckClock
from duckmind.life.record import LifeRecord
from duckmind.neuro import contract as C
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.server import SimState

BIRTH = datetime(2026, 10, 10, 15, 0, tzinfo=TPE)


class Wall:
    def __init__(self, at: datetime | float) -> None:
        self.t = at if isinstance(at, float) else at.timestamp()

    def __call__(self) -> float:
        return self.t


class Life:
    """One start of a life's server side without a body: the state, the host, a stub driver."""

    def __init__(
        self,
        folder: Path,
        wall: Wall,
        *,
        attach: bool = True,
        backdate: int = 0,
        screen_life: bool = True,
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
        self.clock = DuckClock(wall=wall)
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
        self.host = LifeHost(self.record, self.state, self.clock, every_s=600.0, wall=wall)
        self.state.life_host = self.host
        self.host.driver = SimpleNamespace(
            spec=SimpleNamespace(life=screen_life), lanes=SimpleNamespace(brain=None)
        )

    @property
    def core(self) -> Any:
        return self.state.session.core

    def frame(self, t_ms: int, **kw: Any) -> C.SensorFrame:
        return C.SensorFrame(
            t_ms=t_ms,
            touch=C.TouchSense(
                site=kw.get("site"), pressure=kw.get("pressure", 0.0), stroke=kw.get("stroke", 0.0)
            ),
            sound=C.SoundSense(level_db=66.0 if kw.get("sound") else 30.0, kind=kw.get("sound")),
        )

    def watch(self, seconds: float, **kw: Any) -> None:
        """The neural duck's ticks for ``seconds`` (10 a second), the wall clock moving with them."""
        for _ in range(int(seconds * 10)):
            self.wall.t += 0.1
            self.host.on_tick(int(self.wall.t * 1000) % 10**9, self.frame(0, **kw), None, None)

    def tick(self, ms: int, **kw: Any) -> None:
        self.host.on_tick(ms, self.frame(ms, **kw), None, None)

    def second(self) -> None:
        """The server's clock thread: the neural duck's news first, then the clock moves on."""
        self.wall.t += 1.0
        with self.state.lock:
            self.host.deliver()
            self.state.grow()
            self.state.save()


def live_a_pat(life: Life, base_ms: int = 1_000) -> None:
    for k in range(10):
        life.wall.t += 0.1
        life.tick(base_ms + k * 100, site="head", pressure=0.3, stroke=0.6)
    for k in range(10, 16):
        life.wall.t += 0.1
        life.tick(base_ms + k * 100)


def test_a_new_life_is_attached_from_its_first_moment_and_a_backdated_hatch_is_whole_days(
    tmp_path,
) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=12)
    core = life.core
    assert core.s.bridged and core.s.gd == 12
    credits = [e.payload["credit"] for e in core.log.events if e.kind == "growth.gd"]
    assert credits == ["full"] * 12  # (the days lived awake at creation, not days away)
    seam = next(i for i, e in enumerate(core.log.events) if e.kind == "sim.bridged")
    shadow = {"duck.expression", "duck.walk", "duck.nap_start", "voice.babble", "duck.bid"}
    assert not {e.kind for e in core.log.events[seam:]} & shadow
    saved = json.loads((tmp_path / "duck" / "core.json").read_text())
    assert saved["bridge"] == {"api": "bridge-v1"}
    assert saved["inputs"][0]["method"] == "bridge"


def test_a_pat_the_neural_duck_felt_becomes_touch_care_and_survives_a_restart(tmp_path) -> None:
    wall = Wall(BIRTH)
    life = Life(tmp_path / "duck", wall, backdate=14)
    life.second()
    live_a_pat(life)
    life.second()
    assert life.core.s.touches == 1 and life.host.delivered == 1
    again = Life(tmp_path / "duck", Wall(wall.t))
    assert again.core.s.touches == 1 and again.core.s.bridged
    assert json.dumps(api.save_json(again.state.session)["inputs"]) == json.dumps(
        api.save_json(life.state.session)["inputs"]
    )
    assert again.core.log.head == life.core.log.head


def test_what_it_lives_in_the_seconds_before_a_stop_is_not_lost(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14)
    life.second()
    for k in range(10):
        life.wall.t += 0.1
        life.tick(1_000 + k * 100, site="head", pressure=0.3, stroke=0.6)  # (still being petted)
    life.host.stop()
    again = Life(tmp_path / "duck", Wall(life.wall.t))
    assert again.core.s.touches == 1


def test_a_computer_that_slept_does_not_refuse_the_news(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14)
    life.second()
    life.wall.t += 4 * 3600  # asleep four hours
    live_a_pat(life)
    life.second()
    assert life.host.refused == 0 and life.core.s.touches == 1
    assert life.core.s.now >= datetime.fromtimestamp(life.wall.t - 2, TPE)


def test_a_demo_duck_on_screen_lives_nothing_for_the_life_duck(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14, screen_life=False)
    life.second()
    live_a_pat(life)
    life.second()
    assert life.core.s.touches == 0 and life.host.delivered == 0


def test_hands_on_duck_cores_own_page_are_refused_but_the_phone_is_not(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14)
    for action in ("pickup", "touch", "call_name", "power_off", "pad_play", "family_return"):
        with pytest.raises(ValueError, match="蛋黃"):
            life.state.refuse_in_life("act", {"action": action, "params": {}})
    with pytest.raises(ValueError):
        life.state.refuse_in_life("command", {"text": "摸摸牠"})
    for action in ("owner_hello", "owner_good", "owner_not_that", "owner_enough"):
        life.state.refuse_in_life("act", {"action": action, "params": {}})  # (no error)


def test_an_older_record_is_kept_whole_and_attached_where_it_stopped(tmp_path) -> None:
    folder = tmp_path / "duck"
    wall = Wall(BIRTH)
    old = Life(folder, wall, attach=False, backdate=20)
    assert not old.core.s.bridged
    old.state.save()
    before = (folder / "core.json").read_text()
    old_ids = [e.event_id for e in old.core.log.events]
    new = Life(folder, Wall(wall.t), attach=True)
    assert new.core.s.bridged
    backups = sorted(p.name for p in folder.iterdir() if p.name.startswith("core.json.pre-g2"))
    assert backups == ["core.json.pre-g2"]
    assert (folder / "core.json.pre-g2").read_text() == before
    assert stat.S_IMODE((folder / "core.json.pre-g2").stat().st_mode) == 0o600
    assert [e.event_id for e in new.core.log.events][: len(old_ids)] == old_ids
    new.state.save()
    saved = json.loads((folder / "core.json").read_text())
    old_saved = json.loads(before)
    assert saved["inputs"][: len(old_saved["inputs"])] == old_saved["inputs"]
    assert saved["inputs"][len(old_saved["inputs"])]["method"] == "bridge"


def test_the_status_counts_what_it_lived_and_names_no_one(tmp_path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14)
    life.second()
    live_a_pat(life)
    life.tick(5_000, sound="name")
    life.wall.t += 0.2
    life.tick(5_200, sound="name")
    life.wall.t += 2
    life.tick(7_000)
    life.second()
    with life.state.lock:
        status = life.host.status()
    bridge = status["bridge"]
    assert bridge["attached"] and bridge["delivered"] >= 2 and bridge["refused"] == 0
    assert bridge["seen"].get("touch") == 1 and bridge["seen"].get("name") == 1
    assert "蛋黃" not in json.dumps(bridge)


def test_whose_branch_the_body_follows_is_the_records_and_the_planner_can_change_it(
    tmp_path,
) -> None:
    from duckmind.life import upgrade

    folder = tmp_path / "duck"
    life = Life(folder, Wall(BIRTH), backdate=3)
    assert life.record.branch_from_core is False
    assert "branch_from_core" not in json.loads((folder / "life.json").read_text())  # a G1 file
    done = upgrade.set_branch_from_core(folder, True)
    assert done["was"] is False and Path(done["backup"]).is_file()
    assert json.loads((folder / "life.json").read_text())["branch_from_core"] is True
    assert LifeRecord.load(folder).branch_from_core is True
    upgrade.set_branch_from_core(folder, False)
    assert "branch_from_core" not in json.loads((folder / "life.json").read_text())
    fresh = LifeRecord.create(tmp_path / "new", name="x", seed=1, branch_from_core=True)
    assert LifeRecord.load(tmp_path / "new").branch_from_core and fresh.branch_from_core


def test_the_upgrade_tool_proves_before_it_writes_and_keeps_the_old_record_whole(tmp_path) -> None:
    from duckmind.life import upgrade

    folder = tmp_path / "duck"
    old = Life(folder, Wall(BIRTH), attach=False, backdate=20)
    old.state.save()
    before = (folder / "core.json").read_text()
    dry = upgrade.upgrade(folder, dry_run=True)
    assert dry["dry_run"] and not dry["upgraded"] and all(dry["proof"].values())
    assert (folder / "core.json").read_text() == before  # (nothing written)
    done = upgrade.upgrade(folder, stamp="test")
    assert done["upgraded"] and done["new"]["inputs"] == done["old"]["inputs"] + 1
    # (the seam: a nap or a walk it was in the middle of ends, and one sim.bridged)
    assert done["new"]["events"] > done["old"]["events"]
    backup = folder / "core.json.pre-g2-test"
    assert backup.read_text() == before and stat.S_IMODE(backup.stat().st_mode) == 0o600
    saved = json.loads((folder / "core.json").read_text())
    assert saved["bridge"] == {"api": "bridge-v1"} and saved["inputs"][-1]["method"] == "bridge"
    assert upgrade.upgrade(folder)["upgraded"] is False  # (once)
    # the server opens it as it is: attached, no second seam, its own backup untouched
    again = Life(folder, Wall(BIRTH), attach=True)
    assert again.core.s.bridged and not (folder / "core.json.pre-g2").exists()
    methods = [m for m, _ in again.state.session.inputs]
    assert methods.count("bridge") == 1
