"""(Runs in a child process, see tests/test_life_isolated.py.) The seam of the growth integration: the neural genome built from Duck Core's own state
(``Genome.from_core_state``), at the growth day the calendar has reached."""

from __future__ import annotations

import dataclasses

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.neuro import integration
from duckmind.neuro.genome import Genome

BIRTH = "2026-10-10T15:00:00+08:00"


def _core(seed: int = 4107) -> Session:
    return Session(Config(seed=seed, name="蛋黃", start=BIRTH))


def test_at_growth_day_0_the_life_duck_is_the_seeds_hatchling() -> None:
    integration.install_default_skills()
    session = _core()
    genome = Genome.from_core_state(session.core.s)
    assert (genome.seed, genome.gd, genome.stage, genome.branch) == (4107, 0, "hatchling", None)
    # the same duck Genome.from_seed draws for the seed at day 0, parameter for parameter
    reference = Genome.from_seed(4107, gd=0)
    assert genome.summary() == reference.summary()
    assert dataclasses.asdict(genome.p) == dataclasses.asdict(reference.p)
    assert genome.unlocked == reference.unlocked and genome.gains == reference.gains
    assert genome.primary == "explorer"  # seed 4107's nature: the explorer's character stays


def test_the_hatchling_cannot_yet_walk_and_the_explorer_of_day_60_can() -> None:
    integration.install_default_skills()
    day0 = Genome.from_core_state(_core().core.s)
    explorer = Genome.from_seed(4107, gd=60)
    # no locomotion before its first steps (growth day 5 at this duck's pace)
    assert not {"wander", "flee", "follow", "return_dock", "retreat"} & day0.unlocked
    assert {"wander", "flee", "follow", "return_dock", "retreat"} <= explorer.unlocked
    assert day0.p.amplitude_cap < explorer.p.amplitude_cap and day0.p.leash_m < explorer.p.leash_m


def test_the_genome_follows_the_growth_day_the_calendar_reaches() -> None:
    integration.install_default_skills()
    session = _core()
    seen = {}
    for day in (0, 3, 5, 12, 50):
        while session.core.s.gd < day:
            dispatch(session, "advance", {"seconds": 24 * 3600})
        genome = Genome.from_core_state(session.core.s)
        seen[day] = (genome.gd, genome.stage, "wander" in genome.unlocked, genome.branch)
    assert seen[0] == (0, "hatchling", False, None)
    assert seen[3][2] is False and seen[5] == (5, "hatchling", True, None)
    assert seen[12][1] == "duckling" and seen[50][1] == "juvenile"


def test_until_the_ducks_own_experience_reaches_duck_core_the_branch_is_the_natures() -> None:
    """Duck Core's shadow life (no family, naps of its own) would turn the explorer into a dreamer
    at growth day ~48; G1 keeps the branch of the seed's nature (docs/design/GROWTH-G1, section 7)."""
    from duckmind.simweb import life_host

    integration.install_default_skills()
    session = _core()
    while session.core.s.gd < 50:
        for _ in range(360):
            dispatch(session, "advance", {"seconds": 240})
    state = session.core.s
    assert (
        state.branch == "dreamer"
        and state.branch_scores["dreamer"] > state.branch_scores["explorer"]
    )
    faithful = Genome.from_core_state(state)
    assert faithful.branch == "dreamer" and faithful.stage == "juvenile"  # what the state says
    # (G2: a NEW record follows Duck Core's branch; a record made with a shadow life behind it, like
    # this state, says False and keeps its nature's)
    assert life_host.BRANCH_FROM_DUCK_CORE is True
    nature = Genome.from_core_state(state, branch=Genome.from_seed(4107, gd=state.gd).branch)
    assert (
        nature.branch == "explorer"
        and nature.summary() == Genome.from_seed(4107, gd=state.gd).summary()
    )


def test_the_comparison_fixtures_are_the_explorer_at_two_ages() -> None:
    from duckmind.verify import life_fixtures, temperaments

    integration.install_default_skills()
    day0 = life_fixtures.genome("danhuang_gd0", 3)
    day60 = life_fixtures.genome("explorer_gd60", 3)
    assert (day0.seed, day0.gd, day0.stage) == (3, 0, "hatchling")  # (3: the trial's brain seed)
    assert (day60.gd, day60.stage, day60.branch) == (60, "juvenile", "explorer")
    assert day0.axes_raw == day60.axes_raw  # the same duck
    life_fixtures.install()  # the frozen module answers for them too, and for its own as before
    assert temperaments.genome("danhuang_gd0", 3).gd == 0
    assert temperaments.genome("bold_explorer", 3).gd == temperaments.GROWTH_DAY
    assert set(temperaments.PRESET_NAMES) == {
        "bold_explorer",
        "shy_snuggler",
        "lazy_dreamer",
        "social_songbird",
    }
