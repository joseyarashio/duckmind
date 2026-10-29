"""The neural duck's experiences as Duck Core's inputs, with a real integrated brain on a real
life (docs/design/GROWTH-G2-2026-10-10.md; runs in a child process, see tests/test_life_isolated.py):
what the duck lives is given to Duck Core, and a restart keeps both the mind and the growth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api
from duckmind.neuro import contract as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_restart import BIRTH, Life, Wall  # noqa: E402


def pat(life: Life, base_ms: int = 1_000_000) -> None:
    """The brain's own tick sees a stroked head for a second (through the host's hook)."""
    for k in range(16):
        life.wall.t += 0.1
        site = "head" if k < 10 else None
        frame = C.SensorFrame(
            t_ms=base_ms + k * 100,
            touch=C.TouchSense(
                site=site, pressure=0.3 if site else 0.0, stroke=0.6 if site else 0.0
            ),
        )
        life.host.on_tick(base_ms + k * 100, frame, None, None)


def second(life: Life) -> None:
    life.wall.t += 1.0
    with life.state.lock:
        life.host.deliver()
        life.state.grow()
        life.state.save()


def test_an_attached_life_gives_the_real_brains_ticks_to_duck_core_and_a_restart_keeps_all(
    tmp_path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall, attach=True, backdate=48)
    assert first.core.s.bridged and first.core.s.gd == 48
    first.live(2.0)  # the real brain ticks through the host's hook: nothing to report, no error
    assert first.driver.stats.errors == 0
    first.teach()  # a person, a scare spot, a habit (the neural duck's mind)
    pat(first)
    second(first)
    assert first.core.s.touches == 1 and first.host.delivered == 1
    before = first.remembered()
    first.host.stop()

    wall.t += 30.0
    again = Life(folder, wall, attach=True, backdate=48)
    assert again.core.s.touches == 1 and again.core.s.bridged  # the growth: replayed
    assert again.report is not None and again.report["names"] == ["小明"]  # the mind: restored
    assert again.remembered()["names"] == before["names"] and again.remembered()["avoided"]
    saved = json.loads((folder / "core.json").read_text())
    methods = [i["method"] for i in saved["inputs"]]
    assert methods[0] == "bridge" and "experience" in methods
    assert saved["bridge"] == {"api": "bridge-v1"}
    assert api.replay_check(again.state.session)["identical"]
    # the same duck, not a hatch again, and it goes on living
    assert again.core.log.events[0].event_id == first.core.log.events[0].event_id
    pat(again, base_ms=5_000_000)
    second(again)
    assert again.core.s.touches == 2


def test_a_crash_keeps_what_was_saved_and_loses_only_the_last_half_minute(tmp_path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall, attach=True, backdate=20)
    pat(first)
    second(first)  # (saved: the clock thread saves after delivering)
    assert first.core.s.touches == 1
    pat(first, base_ms=3_000_000)  # …and a second pat that never reached the file
    first.wall.t += 1.0
    with first.state.lock:
        first.host.deliver()  # delivered, not saved: the process dies here
    wall.t += 5.0
    again = Life(folder, wall, attach=True, backdate=20)
    assert again.core.s.touches == 1


def test_the_growth_after_a_backdated_hatch_is_the_experience_of_the_neural_duck_only(
    tmp_path,
) -> None:
    """The G1 pit: Duck Core's shadow life (naps, babbles) chose 蛋黃's branch at day 48. Attached from
    its first moment, the 48 days it lives at creation are the calendar's alone."""
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    life = Life(folder, wall, attach=True, backdate=48)
    s = life.core.s
    assert (
        s.gd == 48
        and s.self_window == {}
        and s.care_window
        == {
            "talk": 0,
            "play": 0,
            "touch": 0,
            "calm": 0,
        }
    )
    seam = next(i for i, e in enumerate(life.core.log.events) if e.kind == "sim.bridged")
    assert seam <= 5  # (the seam is at the hatch: only its own first boot before it)
    assert not any(
        e.kind in {"duck.expression", "duck.walk", "duck.nap_start", "voice.babble", "duck.bid"}
        for e in life.core.log.events[seam:]
    )
    assert {m for m, _ in life.state.session.inputs} == {"bridge", "advance"}


def test_the_genome_follows_duck_cores_branch_only_when_the_record_says_so(tmp_path) -> None:
    import dataclasses

    folder, wall = tmp_path / "duck", Wall(BIRTH)
    life = Life(folder, wall, attach=True, backdate=48)
    core_branch = life.core.s.branch
    assert core_branch is not None
    assert life.record.branch_from_core is False  # (a record made without the flag: the nature's)
    off = life.host.genome_builder()()
    assert off.branch == "explorer"
    life.host.record = dataclasses.replace(life.record, branch_from_core=True)
    on = life.host.genome_builder()()
    assert on.branch == core_branch
    # …and it is written to life.json only when True (a G1 record's file is unchanged)
    assert "branch_from_core" not in life.record.to_json()
    assert life.host.record.to_json()["branch_from_core"] is True
