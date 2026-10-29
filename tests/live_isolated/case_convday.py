"""Live voice, LV4: the scripted conversation day (``duckmind.live.convday``) and the targets it measures
(``docs/design/live-voice/LV4-TARGETS.md``). The script, the world's people, the fake's proposals, one
real day end to end, and the seven targets as code: each target passes on numbers that meet it and fails,
alone, on numbers that do not. No network, no physics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any

import pytest

from duckmind.live import convday as D
from duckmind.live.drives import DriveParams
from duckmind.live.fake import FakeGemini
from duckmind.neuro.contract import MotorCommand

pytestmark = pytest.mark.usefixtures("no_network")


# ── the script ──────────────────────────────────────────────────────────────────────────


def test_the_day_is_the_one_in_the_targets_file() -> None:
    s = D.make_script(5)
    assert (s.kind, s.length_s, s.day_minute0, s.dark) == ("day", 590.0, 600.0, False)
    assert [(v.who, v.t_in, v.t_out) for v in s.visits] == [
        ("guest", 30.0, 100.0),
        ("mom", 125.0, 306.0),
        ("dad", 330.0, 450.0),
        ("mom", 470.0, 575.0),
    ]
    addressed = [ln for ln in s.lines if ln.addressed]
    groups = ("guest", "mom", "dad_pre", "dad_bang", "dad_post")
    by = {g: [ln for ln in addressed if ln.group == g] for g in groups}
    assert [len(v) for v in by.values()] == [4, 18, 3, 1, 3]
    assert [ln.ordinal for ln in by["mom"]] == list(range(1, 19))
    assert all(ln.kind == "hello" and ln.who == "guest" for ln in by["guest"])
    assert all(ln.kind == "name" for ln in addressed if ln.who != "guest")
    assert all(D.NAME in ln.text for ln in addressed if ln.who != "guest")
    intro = [ln for ln in s.lines if ln.group == "intro"]
    assert [ln.text for ln in intro] == ["這是媽媽。", "這是爸爸。"] and all(
        ln.who == "" for ln in intro
    )
    assert s.bang_t == 396.0 and by["dad_bang"][0].t == 399.0  # (the apology is 3 s after the bang)
    assert len(s.touches) == 2 and all(abs((b - a) - 4.0) < 1e-9 for a, b in s.touches)


def test_the_times_are_jittered_from_the_seed_within_a_second_and_a_half_and_the_order_is_kept() -> (
    None
):
    base = D.make_script(1)
    other = D.make_script(2)
    assert D.make_script(1) == base  # (same seed, same day)
    assert base != other
    ts = [ln.t for ln in base.lines]
    assert ts == sorted(ts)
    assert len(base.lines) == len(other.lines)
    for one, two in zip(
        base.lines, other.lines, strict=True
    ):  # (the same lines, in the same order)
        assert (one.group, one.ordinal, one.who) == (two.group, two.ordinal, two.who)
        if one.rigid:
            assert one.t == two.t
        else:
            assert abs(one.t - two.t) <= 2 * D.JITTER_S + 1e-9
    addressed = [ln.t for ln in base.lines if ln.addressed]
    gaps = [b - a for a, b in zip(addressed, addressed[1:], strict=False)]
    assert gaps and min(gaps) > 2.5  # (never two addressed lines on top of each other)


def test_the_night_run_is_dark_and_has_six_name_calls_fifteen_seconds_apart() -> None:
    s = D.make_script(3, "night")
    assert (s.kind, s.length_s, s.day_minute0, s.dark) == ("night", 150.0, 1380.0, True)
    assert [ln.t for ln in s.lines] == [pytest.approx(70.0 + 15.0 * k, abs=1.5) for k in range(6)]
    assert all(ln.addressed and ln.kind == "name" for ln in s.lines)


# ── the fake's proposals ────────────────────────────────────────────────────────────────


def test_the_fake_makes_a_proposal_to_each_addressed_line_after_its_own_delays() -> None:
    s = D.make_script(4)
    todo = D.proposals_for(s, 4)
    vocalizes = [(t, index) for t, _seq, index, c in todo if c.name == "vocalize"]
    assert len(vocalizes) == 29  # (every addressed line, nothing for chatter and intros)
    for t, index in vocalizes:
        assert t == pytest.approx(s.lines[index].end + 0.9)
    kinds = {c.name for _t, _s, _i, c in todo}
    assert kinds == {"vocalize", "feel", "attend", "recognize", "remember"}
    assert [t for t, *_ in todo] == sorted(t for t, *_ in todo)
    recognised = [c.args["name"] for _t, _s, _i, c in todo if c.name == "recognize"]
    assert recognised == ["媽媽", "爸爸"]
    assert all(c.problems == () and not c.off_bank for _t, _s, _i, c in todo)


def test_the_proposals_are_one_two_or_three_tags_about_as_often_as_the_stress_says() -> None:
    counts = {1: 0, 2: 0, 3: 0}
    for seed in range(1, 40):
        for _t, _s, _i, c in D.proposals_for(D.make_script(seed), seed):
            if c.name == "vocalize":
                counts[len(c.args["tags"])] += 1
    total = sum(counts.values())
    assert counts[1] / total == pytest.approx(0.5, abs=0.07)
    assert counts[2] / total == pytest.approx(0.3, abs=0.07)
    assert counts[3] / total == pytest.approx(0.2, abs=0.07)
    # (the same seed makes the same proposals)
    one = [(t, c.args) for t, _s, _i, c in D.proposals_for(D.make_script(9), 9)]
    assert one == [(t, c.args) for t, _s, _i, c in D.proposals_for(D.make_script(9), 9)]


def test_the_default_fake_is_still_one_tag_a_reply_and_the_chatty_one_is_the_same_with_one_weight() -> (
    None
):
    plain = FakeGemini(duck_name=D.NAME)
    chatty_one = FakeGemini(duck_name=D.NAME, tag_counts=(1.0, 0.0, 0.0), chat_seed=3)
    for text in ("蛋黃，早安！", "你好！"):
        a, b = plain.react(text), chatty_one.react(text)
        assert [c["args"] for c in a] == [c["args"] for c in b]
        assert len(a[0]["args"]["tags"]) == 1


# ── the world's people ──────────────────────────────────────────────────────────────────


def test_a_person_stands_in_front_of_the_duck_for_a_line_with_a_look_of_their_own() -> None:
    s = D.make_script(1)
    world = D.ConvWorld(s, 1)
    seen: dict[str, list[tuple[float, ...]]] = {}
    in_front = {}
    for k in range(int(s.length_s * 10)):
        t = k / 10.0
        t_ms = int(round(t * 1000))
        frame = world.frame(t_ms)
        people = [x for x in frame.camera.sightings if x.kind == "person"]
        for ln in s.lines:
            if ln.addressed and abs(t - ln.t) < 0.05:
                in_front[(ln.group, ln.ordinal)] = people
        for p in people:
            assert p.look is not None
            seen.setdefault(next(v.who for v in s.visits if v.t_in <= t < v.t_out), []).append(
                p.look
            )
        world.apply(MotorCommand(t_ms, "rest_sit", 0, 0), 0.1)
    assert set(seen) == {"guest", "mom", "dad"}
    for who, looks in seen.items():
        mean = [statistics.fmean(x[i] for x in looks) for i in range(6)]
        assert all(abs(mean[i] - D.LOOKS[who][i]) < 0.02 for i in range(6))
        scatter = statistics.pstdev(x[0] for x in looks)
        assert 0.001 < scatter < 0.02  # (a little noise, like the eye)
    standing = [p for ps in in_front.values() for p in ps]
    assert len(in_front) == 29 and sum(1 for ps in in_front.values() if ps) >= 27
    assert all(1.0 < p.range_m < 1.9 and abs(p.bearing) < 0.4 for p in standing)


def test_a_person_steps_round_in_front_of_a_duck_that_turns_away_for_a_line_an_introduction_and_the_bang() -> (
    None
):
    """The family keeps where the duck can see them: when it turns away they step in front of it again
    (within 3 s of not being seen), and for a line, an introduction (until the proposal it makes has
    come) and the bang they are there."""
    from duckmind.neuro.contract import MotorCommand

    for seed in (1, 2, 3):
        s = D.make_script(seed)
        world = D.ConvWorld(s, seed)
        spin = MotorCommand(0, "wander", 0, 0, vx=0.05, vyaw=0.25, posture="stand")
        seen: dict[int, bool] = {}
        for k in range(int(s.length_s * 10)):
            frame = world.frame(k * 100)
            seen[k] = any(x.kind == "person" for x in frame.camera.sightings)
            world.apply(spin, 0.1)
        wanted = [round(ln.t * 10) for ln in s.lines if ln.addressed]
        wanted += [round((ln.end + 0.9) * 10) for ln in s.lines if ln.group == "intro"]
        wanted += [round(s.bang_t * 10)]
        share = sum(seen[k] for k in wanted) / len(wanted)
        assert share >= 0.9, (seed, share)
        for ln in s.lines:
            if ln.group == "intro":
                assert seen[round((ln.end + 0.9) * 10)], (seed, ln.text)
        assert seen[round(s.bang_t * 10)], seed
        # and they are not seen only by luck: over a visit the duck sees them most of the time
        visit = [seen[k] for k in range(1250, 3060)]
        assert sum(visit) / len(visit) > 0.7, seed


def test_nobody_is_there_outside_the_visits_and_in_the_dark() -> None:
    s = D.make_script(2)
    world = D.ConvWorld(s, 2)
    for k in range(int(130 * 10)):
        t = k / 10.0
        frame = world.frame(int(round(t * 1000)))
        people = [x for x in frame.camera.sightings if x.kind == "person"]
        if t < 29.0 or 101.0 < t < 124.0:
            assert not people
    night = D.ConvWorld(D.make_script(2, "night"), 2)
    for k in range(1500):
        frame = night.frame(k * 100)
        assert not [x for x in frame.camera.sightings if x.kind == "person"]
        assert frame.camera.light in ("black", "dark")


# ── one day, end to end ─────────────────────────────────────────────────────────────────


def test_a_day_is_deterministic_and_every_line_is_proposed_judged_and_answered_or_not() -> None:
    first = D.run_day("shy_snuggler", 2, DriveParams.neutral())
    again = D.run_day("shy_snuggler", 2, DriveParams.neutral())
    assert first.to_dict() == again.to_dict()
    assert first.faults == 0 and first.ticks == 5900
    assert len(first.proposals) == 29 and len(first.lines) == 29
    assert first.bridge_counts["vocalize:accepted"] == 29
    assert first.bridge_counts["recognize:accepted"] == 2
    assert {p["group"] for p in first.proposals} == {
        "guest",
        "mom",
        "dad_pre",
        "dad_bang",
        "dad_post",
    }
    voiced = [p for p in first.proposals if p["voiced"]]
    assert voiced and all(p["latency"] is not None and 0.0 < p["latency"] < 8.0 for p in voiced)
    assert all(1 <= p["tags_voiced"] <= p["tags_proposed"] <= 3 for p in voiced)
    assert not first.violations  # (nothing voiced while frightened or asleep)
    bond = first.bond
    assert bond["startled"] and bond["dad_after"] < bond["dad_before"]
    assert bond["mom_end"] > bond["mom_before"] + 0.2
    names = {p["name"] for p in first.people}
    assert {"媽媽", "爸爸", None} == names  # (mom and dad named; the guest never)


def test_the_night_run_has_the_duck_asleep_and_it_does_not_answer() -> None:
    night = D.run_day(
        "lazy_dreamer", 1, DriveParams.neutral().with_values(init_gain=0.0), kind="night"
    )
    assert night.faults == 0 and night.ticks == 1500
    assert len(night.proposals) == 6
    assert sum(1 for p in night.proposals if not p["available"]) >= 4  # (it was asleep)
    assert not night.violations and not any(
        p["voiced"] and not p["available"] for p in night.proposals
    )


def test_without_the_bridge_the_same_day_has_no_proposals_at_all() -> None:
    bare = D.run_day("bold_explorer", 1, attach=False)
    assert bare.proposals == [] and bare.drive_counts == {} and bare.ticks == 5900
    assert sum(bare.behaviours.values()) == pytest.approx(590.0)


# ── the seven targets as code ───────────────────────────────────────────────────────────

Rule = Callable[[str, int], bool]  # (group, ordinal) -> is this line answered


def proposals(rule: Rule, *, latency: float = 2.0, tags: int = 1) -> list[dict[str, Any]]:
    plan = (
        [("guest", "guest", k) for k in range(1, 5)]
        + [("mom", "mom", k) for k in range(1, 19)]
        + [("dad", "dad_pre", k) for k in range(1, 4)]
        + [("dad", "dad_bang", 4)]
        + [("dad", "dad_post", k) for k in range(5, 8)]
    )
    out = []
    for who, group, ordinal in plan:
        answered = (
            rule(group, ordinal) and group != "dad_bang"
        )  # (the apology comes while it is afraid)
        out.append(
            {
                "group": group,
                "who": who,
                "ordinal": ordinal,
                "available": group != "dad_bang",
                "voiced": answered,
                "latency": latency if answered else None,
                "tags_voiced": tags if answered else 0,
                "tags_proposed": 3,
            }
        )
    return out


def day(
    preset: str,
    seed: int,
    rule: Rule,
    *,
    latency: float = 2.0,
    tags: int = 1,
    starts: int = 1,
    rise: float = 0.4,
    drop: tuple[float, float] | None = (0.5, 0.3),
) -> dict[str, Any]:
    bond: dict[str, Any] = {"mom_before": 0.1, "mom_end": 0.1 + rise, "startled": drop is not None}
    if drop is not None:
        bond.update(dad_before=drop[0], dad_after=drop[1])
    return {
        "preset": preset,
        "seed": seed,
        "kind": "day",
        "proposals": proposals(rule, latency=latency, tags=tags),
        "violations": [],
        "call_out_starts": starts,
        "bond": bond,
    }


def night(preset: str, seed: int, *, violations: int = 0) -> dict[str, Any]:
    asleep = {"who": "mom", "available": False, "voiced": False, "latency": None, "tags_voiced": 0}
    return {
        "preset": preset,
        "seed": seed,
        "kind": "night",
        "proposals": [
            {"group": "mom", "ordinal": k, "tags_proposed": 2, **asleep} for k in range(1, 7)
        ],
        "violations": [{"t": 1.0}] * violations,
        "call_out_starts": 0,
        "bond": {},
    }


# who answers whom, per preset, in a world that meets every target (and in worlds that do not)
def song_rule(seed: int) -> Rule:
    return lambda g, o: not (g == "dad_post" and o == 5 and seed % 4 == 0)


def snug_rule(seed: int) -> Rule:
    def rule(g: str, o: int) -> bool:
        if g == "guest":
            return o == 1 and seed % 4 == 0  # one of the guest's lines in sixteen
        return not (g == "mom" and o >= 6 and seed % 5 == 0 and o % 3 == 0)

    return rule


def dream_rule(seed: int) -> Rule:
    return lambda g, o: (o + seed) % 3 == 0


def expl_rule(seed: int) -> Rule:
    return lambda g, o: (o + 2 * seed) % 4 == 0 or (o == 1 and g in ("guest", "dad_pre"))


def good_world(seeds: range = range(13, 29)) -> list[dict[str, Any]]:
    """Numbers that meet every target."""
    out: list[dict[str, Any]] = []
    for seed in seeds:
        out += [
            day("social_songbird", seed, song_rule(seed), latency=1.5, tags=2, starts=3),
            day("shy_snuggler", seed, snug_rule(seed), latency=1.8, tags=2, starts=1),
            day("lazy_dreamer", seed, dream_rule(seed), latency=3.2, starts=0),
            day("bold_explorer", seed, expl_rule(seed), latency=1.9, starts=1),
        ]
        out += [night(p, seed) for p in D.PRESETS]
    return out


def verdicts(results: list[dict[str, Any]]) -> dict[str, bool]:
    report = D.targets_report(results)
    return {v["name"].split(" ")[0]: v["pass"] for v in report["verdicts"]}


def swap(
    results: list[dict[str, Any]], preset: str, new: Callable[[int], dict[str, Any]]
) -> list[dict[str, Any]]:
    return [new(r["seed"]) if r["preset"] == preset and r["kind"] == "day" else r for r in results]


def test_numbers_that_meet_every_target_pass_every_target() -> None:
    report = D.targets_report(good_world())
    got = {v["name"].split(" ")[0]: v["pass"] for v in report["verdicts"]}
    detail = {v["name"]: v["detail"] for v in report["verdicts"]}
    assert got == dict.fromkeys("1234567", True), (got, detail)


def test_the_loss_is_small_for_numbers_that_meet_the_targets_and_large_for_a_flat_duck() -> None:
    loss, terms = D.search_loss(good_world(range(1, 13)), margin=0.0)
    flat = [
        day(p, seed, lambda g, o: True, tags=3, starts=0, rise=0.0, drop=(0.5, 0.5))
        for p in D.PRESETS
        for seed in range(1, 13)
    ]
    flat_loss, flat_terms = D.search_loss(flat, margin=0.0)
    assert loss < 0.05 and flat_loss > 10 * max(loss, 0.01)
    assert {"song_rate", "snug_stranger", "dream_rate", "exp_rate", "exp_tags", "bond"} <= set(
        terms
    )
    assert flat_terms["snug_stranger"] > 0.1 and flat_terms["exp_tags"] > 0.1


FAILING: list[tuple[str, str, Callable[[int], dict[str, Any]]]] = [
    (
        "1",
        "social_songbird",
        lambda s: day("social_songbird", s, lambda g, o: o % 2 == 0, starts=3),
    ),
    ("1", "social_songbird", lambda s: day("social_songbird", s, song_rule(s), starts=1)),
    ("2", "shy_snuggler", lambda s: day("shy_snuggler", s, lambda g, o: True)),
    (
        "3",
        "shy_snuggler",
        lambda s: day("shy_snuggler", s, lambda g, o: g != "guest" and not (g == "mom" and o >= 6)),
    ),
    (
        "4",
        "lazy_dreamer",
        lambda s: day("lazy_dreamer", s, lambda g, o: o % 2 == 0 or o % 3 == 0, latency=3.2),
    ),
    ("4", "lazy_dreamer", lambda s: day("lazy_dreamer", s, dream_rule(s), latency=1.0)),
    ("5", "bold_explorer", lambda s: day("bold_explorer", s, expl_rule(s), tags=2)),
    (
        "5",
        "bold_explorer",
        lambda s: day("bold_explorer", s, (lambda g, o: False) if s == 13 else expl_rule(s)),
    ),
    (
        "5",
        "bold_explorer",
        lambda s: day("bold_explorer", s, lambda g, o: g != "guest" and expl_rule(s)(g, o)),
    ),
    ("7", "bold_explorer", lambda s: day("bold_explorer", s, expl_rule(s), rise=0.1)),
    ("7", "bold_explorer", lambda s: day("bold_explorer", s, expl_rule(s), drop=(0.5, 0.49))),
    ("7", "bold_explorer", lambda s: day("bold_explorer", s, expl_rule(s), drop=(0.5, 0.05))),
    ("7", "bold_explorer", lambda s: day("bold_explorer", s, expl_rule(s), drop=None)),
]


@pytest.mark.parametrize(("target", "preset", "make"), FAILING)
def test_each_target_fails_alone_on_numbers_that_do_not_meet_it(
    target: str, preset: str, make: Callable[[int], dict[str, Any]]
) -> None:
    got = verdicts(swap(good_world(), preset, make))
    assert got[target] is False, (target, got)
    others = {k: v for k, v in got.items() if k != target}
    if preset == "bold_explorer":  # (the explorer's numbers touch the explorer's targets only)
        assert all(others[k] for k in ("1", "2", "3", "4", "6")), others


def test_a_voiced_tag_while_frightened_or_asleep_fails_and_so_does_a_check_with_nothing_in_it() -> (
    None
):
    results = good_world()
    results[-1] = night(results[-1]["preset"], results[-1]["seed"], violations=1)
    assert verdicts(results)["6"] is False
    emptied = [
        r for r in good_world() if r["kind"] == "day"
    ]  # (no night runs: too few unavailable)
    for r in emptied:
        r["proposals"] = [p for p in r["proposals"] if p["group"] != "dad_bang"]
    assert verdicts(emptied)["6"] is False


def test_a_duck_that_the_bang_did_not_startle_cannot_show_the_drop() -> None:
    results = good_world()
    for r in results:
        if r["preset"] == "social_songbird" and r["kind"] == "day" and r["seed"] < 24:
            r["bond"].update(startled=False)
    assert verdicts(results)["7"] is False


def test_the_paired_interval_and_the_t_values() -> None:
    mean, lo, hi = D._paired_ci([3, 4, 5, 6], [1, 1, 1, 1])
    assert mean == 3.5 and lo < mean < hi
    se = statistics.stdev([2, 3, 4, 5]) / 2
    assert hi - lo == pytest.approx(2 * 3.182 * se, rel=1e-3)
    assert D._t975(1000) == 1.96 and D._t975(15) == 2.131 and D._t975(22) == 2.086
    m, a, b = D._paired_ci([1.0], [0.0])
    assert (m, a, b) == (1.0, 1.0, 1.0)
    assert math.isfinite(D._paired_ci([2, 2, 2], [1, 1, 1])[1])
