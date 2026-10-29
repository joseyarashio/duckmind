"""Personality (Duck Core v4, part E, phase P1): temperament, fair care, branch-v2,
adult-v2 names and epithets, first-word-v2, and each duck's own pace of growing up.

Care clearly raises the odds of a branch but never decides it alone; all four branches are
equally likely by nature; all twelve adult names and all five first words happen; and two
ducks grow up at different speeds, with the growth calendar always in order.
"""

from __future__ import annotations

import collections
from datetime import timedelta
from functools import cache

from duckmind.duckcore import commands, content
from duckmind.duckcore.api import Config, Session, dispatch, snapshot
from duckmind.duckcore.engine import RULES_V1, DuckCore
from duckmind.duckcore.fixed import SCALE
from duckmind.provenance import Event

BRANCHES = ("songbird", "explorer", "snuggler", "dreamer")


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


@cache
def _hatchlings(count: int = 400) -> tuple[DuckCore, ...]:
    return tuple(DuckCore(seed=seed) for seed in range(1, count + 1))


def _decide(core: DuckCore, care: dict[str, int], own: dict[str, int] | None = None) -> str:
    """Its branch for a given window of care (and experience), decided as at growth day 45."""
    core.s.care_window = dict(care)
    core.s.self_window = dict(own or {})
    core.s.branch = None
    core._choose_branch()
    assert core.s.branch is not None
    return core.s.branch


def test_the_temperament_is_its_own_and_drawn_once_from_its_seed():
    a, b = DuckCore(seed=31), DuckCore(seed=31)
    assert a.s.temperament == b.s.temperament and a.s.nature == b.s.nature
    low, high = content.TEMPERAMENT["range"]
    primaries = collections.Counter()
    for core in _hatchlings():
        assert set(content.TEMPERAMENT["axes"]) <= set(core.s.temperament)
        assert all(low <= v <= high for k, v in core.s.temperament.items())
        assert sum(core.s.nature.values()) <= SCALE and set(core.s.nature) == set(BRANCHES)
        primary = core.s.branch_detail["primary"]
        assert primary == max(sorted(core.s.nature), key=lambda key: core.s.nature[key])
        primaries[primary] += 1
    # All four temperaments lead equally often by nature (400 ducks: each about 25%,
    # 17.5–32.5% leaves room for the draw).
    assert all(70 <= primaries[b] <= 130 for b in BRANCHES), primaries
    hatched = _kinds(DuckCore(seed=31), "sim.hatched")[0].payload
    assert hatched["rules"]["branch"] == "branch-v2" and hatched["temperament"]


def test_each_duck_grows_up_at_its_own_pace_and_its_calendar_stays_in_order():
    adults, juveniles = [], []
    for core in _hatchlings():
        days = core.s.stage_days
        assert 9 <= days["duckling"] <= 12
        assert 39 <= days["juvenile"] <= 52
        assert 157 <= days["adult"] <= 207
        adults.append(days["adult"])
        juveniles.append(days["juvenile"])
        order = [core.s.milestones[m.milestone_id]["gd"] for m in content.CALENDAR]
        assert order == sorted(order) and len(set(order)) == len(order), order
        assert core.s.milestones["branch"]["gd"] == days["juvenile"]
        assert core.s.milestones["adult"]["gd"] == days["adult"]
        assert core.s.milestones["duckling"]["gd"] == days["duckling"]
        assert core._stage_at(days["juvenile"] - 1).stage_id == "duckling"
        assert core._stage_at(days["juvenile"]).stage_id == "juvenile"
        assert core._stage_at(days["adult"]).stage_id == "adult"
    adults.sort()
    juveniles.sort()
    assert adults[360] - adults[40] >= 20  # p10–p90 at least 20 growth days apart
    assert juveniles[360] - juveniles[40] >= 8


def test_a_whole_year_uses_its_own_stage_days():
    session = Session(Config(seed=9, nest=True, conservative=False, household="elder_home"))
    dispatch(session, "live", {"family": "elder_home", "days": 60})
    core = session.core
    stages = {e.payload["stage"]: e.payload["gd"] for e in _kinds(core, "growth.stage")}
    assert stages["duckling"] == core.s.stage_days["duckling"]
    assert stages["juvenile"] == core.s.stage_days["juvenile"]
    branch = _kinds(core, "growth.branch")[0].payload
    assert branch["rule"] == "branch-v2" and core.s.gd >= core.s.stage_days["juvenile"]
    assert set(branch["probabilities"]) == set(BRANCHES)
    assert branch["branch"] in BRANCHES and 0 < sum(branch["probabilities"].values()) <= SCALE
    assert "每隻鴨長大的速度不太一樣" in snapshot(session)["growth"]["pace_note"]


def test_branch_v2_over_400_ducks_keeps_all_four_branches_common():
    average = dict(content.CARE_PRIOR)
    even = collections.Counter(
        _decide(core, {k: v * 100 // SCALE for k, v in average.items()}) for core in _hatchlings()
    )
    # An average family: every branch 15–35% (its nature decides, care adds nothing).
    assert all(60 <= even[b] <= 140 for b in BRANCHES), even
    playful = collections.Counter(
        _decide(core, {"talk": 20, "play": 80, "touch": 20, "calm": 20}) for core in _hatchlings()
    )
    # A family that plays much more than most: explorer clearly likelier, never certain,
    # and every other branch still happens.
    assert playful["explorer"] > even["explorer"] * 3 // 2, (playful, even)
    assert playful["explorer"] <= 0.60 * 400 and all(playful[b] >= 0.05 * 400 for b in BRANCHES)


def test_with_little_care_its_nature_and_its_own_experience_decide():
    counts = collections.Counter(
        _decide(core, {"talk": 5, "play": 2, "touch": 3, "calm": 1}, {"dreamer": 40})
        for core in _hatchlings()
    )
    assert counts["dreamer"] == max(counts.values())  # its own long naps count
    assert all(counts[b] > 0 for b in BRANCHES)
    event = _kinds(_hatchlings()[0], "growth.branch")[-1].payload
    assert event["nature_only"] and event["self"]["dreamer"] > event["self"]["songbird"]


def test_the_old_rules_still_decide_an_old_duck():
    core = DuckCore(seed=3, rules=RULES_V1)
    assert _decide(core, {"talk": 60, "play": 5, "touch": 5, "calm": 5}) in BRANCHES
    assert _kinds(core, "growth.branch")[-1].payload["rule"] == "branch-v1"


def test_all_twelve_adult_names_happen_and_none_takes_over():
    names = collections.Counter()
    for core in _hatchlings():
        core.s.branch = core.s.branch_detail["primary"]
        # an average family's care as a juvenile (its lift is even)
        core.s.care_adult = {k: v * 100 // SCALE for k, v in content.CARE_PRIOR_ADULT.items()}
        core.s.self_window = {}
        core.s.adult_name = None
        core._choose_adult()
        names[core.s.adult_name] += 1
        adult = _kinds(core, "growth.adult")[-1].payload
        assert adult["secondary"] != adult["primary"] and adult["rule"] == "adult-v2"
    assert len(names) == 12, names
    assert max(names.values()) <= 0.15 * 400, names


def test_the_epithet_comes_from_its_running_gag_or_its_most_frequent_moment():
    core = DuckCore(seed=5)
    core.s.branch = "songbird"
    core.s.moment_counts = {"zoomies": 3, "vacuum_dabble": 7, "sits_beside_nest": 20}
    core._choose_adult()
    assert core.s.epithet == "vacuum_dabble"  # (sits_beside_nest has no epithet)
    adult = _kinds(core, "growth.adult")[-1].payload
    assert adult["title"] == "愛撈水的" + core.s.adult_name
    core.s.gags = ["zoomies"]
    core.s.adult_name = None
    core._choose_adult()
    assert core.s.epithet == "zoomies"


def test_any_of_the_five_first_words_can_be_its_first():
    words = collections.Counter()
    for core in _hatchlings():
        word, weights = core._first_word()
        assert set(weights) == set(content.FIRST_WORDS) and all(w >= 1 for w in weights.values())
        words[word] += 1
    assert set(words) == set(content.FIRST_WORDS), words
    assert max(words.values()) <= 0.40 * 400, words


def test_the_sound_the_family_answered_early_on_becomes_likelier():
    core = DuckCore(seed=12)
    core.s.gd2 = 8  # growth day 4
    core.s.last_sounds = {"tags": ["coo"], "at": core.s.now}
    core.say("child", kind="voice")  # someone answers what it just said
    assert core.s.word_answers.get("coo") == 1
    core.s.word_answers = {"coo": 3}
    _, weights = core._first_word()
    assert weights["mm"] >= 10 and weights["mm"] == max(weights.values())


def test_the_family_can_still_pick_the_first_word():
    core = DuckCore(seed=4)
    core.choose("first_word", "mm")
    assert core.s.first_word == "mm"
    parsed = commands.parse("第一個字是咕", "owner")[0]
    assert parsed.params == {"kind": "first_word", "option": "mm"}
    assert commands.parse("第一個字選啾啾咻", "owner")[0].params["option"] == "yay"
    assert commands.parse("第一個字要嘎啾", "owner")[0].params["option"] == "hi"


def test_praise_counts_as_the_care_of_what_was_praised():
    expected = {"wander": "play", "snuggle": "touch", "stretch_yawn": "calm", "happy_sway": "talk"}
    for verb, care in expected.items():
        core = DuckCore(seed=2)
        core.s.last_flag = {"kind": "verb", "id": verb, "at": core.s.now, "event": "E000001"}
        core.pad_good("child")
        episodes = [e for e in _kinds(core, "care.episode") if e.payload.get("why") == "praise"]
        assert episodes and episodes[-1].payload["type"] == care, (verb, episodes)


def test_just_being_there_counts_too_as_play_while_it_explores_and_as_calm_company():
    session = Session(Config(seed=6, nest=True, conservative=False, household="elder_home"))
    dispatch(session, "live", {"family": "elder_home", "days": 20})
    whys = collections.Counter(
        (e.payload["type"], e.payload.get("why")) for e in _kinds(session.core, "care.episode")
    )
    assert whys[("play", "company_play")] > 0
    assert whys[("calm", "company_calm")] > 0
    company = [
        e.occurred_at
        for e in _kinds(session.core, "care.episode")
        if e.payload.get("why") == "company_play"
    ]
    assert all(b - a >= timedelta(minutes=10) for a, b in zip(company, company[1:], strict=False))
    own = collections.Counter(e.payload["branch"] for e in _kinds(session.core, "care.self"))
    assert own["explorer"] > 0  # new places of its own count as its own experience
