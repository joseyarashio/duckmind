"""Live voice, LV2: names as data (``duckmind.live.names``) and the learner's per-person records
(``duckmind.neuro.learn``): one form for one name, a misheard name is not a new person, a name is
bound only to a person the duck itself sees, and nobody is made up. No network."""

from __future__ import annotations

import math

import pytest
from live_bridge_rig import DAD_LOOK, KID_LOOK, MOM_LOOK

from duckmind.live import names as N
from duckmind.neuro.genome import Genome
from duckmind.neuro.learn import PERSON_GONE_S, Learner

pytestmark = pytest.mark.usefixtures("no_network")

DUCK = "蛋黃"


# ── one written form ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("妈妈", "媽媽"),
        ("媽媽", "媽媽"),
        ("  媽 媽 ", "媽媽"),
        ("「媽媽」。", "媽媽"),
        ("這是妈妈。", "媽媽"),
        ("这是妈妈", "媽媽"),
        ("我的爷爷", "爺爺"),
        ("蛋黄", "蛋黃"),
        ("郑华", "鄭華"),
        ("Ａnna  Li", "AnnaLi"),
        ("", ""),
        ("。。。", ""),
    ],
)
def test_a_name_has_one_written_form(raw: str, want: str) -> None:
    assert N.normalise_name(raw) == want


def test_the_conversion_table_is_one_pair_per_character_and_changes_each() -> None:
    assert all(len(pair) == 2 and pair[0] != pair[1] for pair in N._PAIRS.split())
    assert len(N.S2T) == len(N._PAIRS.split())  # (no character twice)


# ── the words the transcriber is told to listen for ──────────────────────────────────


def test_the_vocabulary_is_the_ducks_name_then_the_family_names_each_once() -> None:
    assert N.vocabulary(DUCK, ["媽媽", "爺爺"]) == (DUCK, "媽媽", "爺爺")
    # the same name written another way, the duck's own name again, blanks and sentences are left out
    assert N.vocabulary(
        DUCK, ["媽媽", "妈妈", " 媽 媽 ", DUCK, "", "這是一個很長很長很長很長很長的名字"]
    ) == (
        DUCK,
        "媽媽",
    )
    assert N.vocabulary("", ["媽媽"]) == ("媽媽",)
    many = N.vocabulary(DUCK, [f"人{chr(0x4E00 + i)}" for i in range(40)])
    assert len(many) == N.MAX_VOCABULARY and many[0] == DUCK


# ── a transcript in the family's own script ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("heard", "shown"),
    [
        (
            "赵华, 过来这里。",
            "趙華, 過來這裡。",
        ),  # (live 2026-10-04: all of it came back simplified)
        ("你看到什么?", "你看到什麼?"),
        ("好贵。真棒。", "好貴。真棒。"),
        ("小黄", "小黃"),
        ("这是妈妈。", "這是媽媽。"),
        ("晚安。该睡觉了。", "晚安。該睡覺了。"),
        ("小美, 吃饭了。", "小美, 吃飯了。"),
        (
            "头发 干净 面包 周末",
            "頭髮 乾淨 麵包 週末",
        ),  # (words that are not character by character)
        ("家里 手里 里面 这里 公里", "家裡 手裡 裡面 這裡 公里"),
        ("一只鸭子, 只有一个", "一隻鴨子, 只有一個"),
        ("皇后 太后", "皇后 太后"),
    ],
)
def test_a_simplified_transcript_is_shown_in_traditional_characters(heard: str, shown: str) -> None:
    assert N.to_traditional_text(heard) == shown


@pytest.mark.parametrize(
    "text",
    [
        "蛋黃，過來這裡。",  # already traditional
        "mama",
        "Hallo, wie geht's? 123",
        "你好。",  # the same in both
        "ありがとう、学校へ行く",  # Japanese keeps its own forms
        "안녕하세요 学校",
        "",
        "   ",
    ],
)
def test_traditional_english_and_other_scripts_are_left_as_they_came(text: str) -> None:
    assert N.to_traditional_text(text) == text


def test_the_transcript_table_is_one_to_one_and_idempotent() -> None:
    pairs = N._TEXT_PAIRS.split()
    assert all(len(pair) == 2 and pair[0] != pair[1] for pair in pairs)
    assert len({pair[0] for pair in pairs}) == len(pairs)  # (no character twice)
    # the names' table is inside it, except where a word is not a surname (钟: 鍾 / 鐘)
    assert {k: v for k, v in N.S2T.items() if k != "钟"}.items() <= N.TEXT_S2T.items()
    assert N.TEXT_S2T["钟"] == "鐘"
    # converting twice changes nothing: no traditional form is itself a key
    assert not set(N.TEXT_S2T.values()) & set(N.TEXT_S2T)
    sample = "".join(N.TEXT_S2T)
    assert N.to_traditional_text(N.to_traditional_text(sample)) == N.to_traditional_text(sample)


def test_the_fixture_lines_and_the_ducks_name_are_untouched() -> None:
    import json
    from pathlib import Path

    lines = json.loads(
        (Path(__file__).parent / "fixtures" / "voice" / "manifest.json").read_text(encoding="utf-8")
    )
    for line in lines["utterances"]:
        assert N.to_traditional_text(line["text"]) == line["text"], line["id"]
    assert N.to_traditional_text(DUCK) == DUCK


def test_the_same_name_in_both_scripts_is_the_same_name() -> None:
    assert N.similarity("妈妈", "媽媽") == 1.0
    assert N.key("Anna") == N.key("ANNA ")


def test_names_that_are_not_alike_are_not_alike() -> None:
    assert N.similarity("小美", DUCK) < 0.3
    assert N.similarity("媽媽", "爸爸") < N.SIM_KNOWN_NAME  # (mom and dad are two people)
    assert N.similarity("王小明", "媽媽") < 0.3


def test_a_mishearing_of_the_ducks_name_sounds_like_it() -> None:
    # LV1: the model heard 蛋黃 as 郑华 in all eight call_name recordings
    assert N.similarity("郑华", DUCK) >= N.SIM_OWN_NAME
    assert N.similarity("蛋黄", DUCK) == 1.0


# ── what is done with a proposed name ────────────────────────────────────────────────


def decide(raw: str, **kw: object) -> N.NameDecision:
    base: dict[str, object] = {
        "in_view_pid": 1,
        "in_view_name": None,
        "known": {},
        "duck_name": DUCK,
        "speech_age_s": 1.0,
    }
    base.update(kw)
    return N.decide_name(raw, **base)  # type: ignore[arg-type]


def test_a_new_name_for_the_person_in_view_is_bound_when_someone_just_spoke() -> None:
    d = decide("妈妈")
    assert (d.action, d.reason, d.name, d.pid) == (N.BIND, "new_name", "媽媽", 1)


def test_a_name_with_nobody_in_view_is_not_made_into_a_person() -> None:
    d = decide("媽媽", in_view_pid=None)
    assert (d.action, d.reason) == (N.REFUSE, "no_person_in_view")


@pytest.mark.parametrize("raw", ["蛋黃", "蛋黄", "郑华", "鄭華"])
def test_the_ducks_own_name_and_what_sounds_like_it_is_nobodys_new_name(raw: str) -> None:
    d = decide(raw)
    assert d.action == N.REFUSE and d.reason in ("duck_name", "sounds_like_duck_name")
    assert d.near == DUCK


@pytest.mark.parametrize(
    "raw", ["", "。", "人", "他", "someone", "說話的人", "12345", "媽媽媽媽媽媽媽媽"]
)
def test_things_that_are_not_names_are_refused(raw: str) -> None:
    assert decide(raw).action == N.REFUSE


def test_a_name_the_duck_already_has_for_the_person_in_view_is_confirmed_not_added() -> None:
    d = decide("妈妈", known={1: "媽媽"}, in_view_name="媽媽")
    assert (d.action, d.reason, d.name) == (N.BIND, "confirmed", "媽媽")


def test_the_name_of_someone_else_it_knows_is_not_taken_for_a_stranger_in_view() -> None:
    d = decide("媽媽", in_view_pid=2, known={1: "媽媽"})
    assert (d.action, d.reason) == (N.PENDING, "name_taken")


def test_a_second_name_for_a_named_person_waits_for_evidence() -> None:
    d = decide("小美", in_view_name="媽媽", known={1: "媽媽"})
    assert (d.action, d.reason) == (N.PENDING, "conflict")
    assert N.promotable(d.reason, 1) is False and N.promotable(d.reason, 2) is True


def test_a_name_with_no_one_having_just_spoken_is_pending() -> None:
    assert decide("媽媽", speech_age_s=None).reason == "no_recent_speech"
    assert decide("媽媽", speech_age_s=60.0).action == N.PENDING


def test_a_pending_doubt_about_whose_name_it_is_never_promotes() -> None:
    assert not N.promotable("name_taken", 99)
    assert not N.promotable("sounds_like_duck_name", 99)


def test_a_hint_names_a_person_among_the_family_it_knows() -> None:
    known = {3: "媽媽", 4: "爺爺"}
    assert N.match_known("妈妈", known) == 3
    assert N.match_known("我的爷爷", known) == 4
    assert N.match_known("小美", known) is None


# ── the learner's per-person records ─────────────────────────────────────────────────


def learner() -> Learner:
    return Learner(Genome.from_seed(4107, gd=60).p)


def test_a_person_seen_again_is_the_same_record_and_a_new_look_is_a_new_one() -> None:
    lr = learner()
    a = lr.see_person(1, MOM_LOOK, 1.0)
    assert lr.see_person(1, MOM_LOOK, 1.5) == a  # the same eye track
    assert lr.see_person(9, MOM_LOOK, 30.0) == a  # a new track with the same look
    b = lr.see_person(2, KID_LOOK, 31.0)
    assert b != a
    c = lr.see_person(3, DAD_LOOK, 32.0)
    assert len({a, b, c}) == 3 and len(lr.people) == 3
    assert lr.people[a].sightings == 2  # (the track 1 and the track 9)


def test_without_a_look_the_one_person_seen_a_moment_ago_is_who_it_is() -> None:
    lr = learner()
    a = lr.see_person(1, None, 1.0)
    assert lr.see_person(5, None, 6.0) == a  # (the eye re-issued the track)
    # long gone: someone new
    assert lr.see_person(6, None, 6.0 + PERSON_GONE_S + 5) != a


def test_two_people_in_view_without_looks_are_two_records() -> None:
    lr = learner()
    a = lr.see_person(1, None, 1.0)
    b = lr.see_person(2, None, 1.0)  # (a different live track at the same moment)
    assert a != b


def test_a_name_is_bound_to_a_record_and_kept_with_what_the_family_wrote() -> None:
    lr = learner()
    pid = lr.see_person(1, MOM_LOOK, 1.0)
    assert lr.bind_name(pid, "媽媽", "妈妈", 2.0)
    assert lr.person_names() == {pid: "媽媽"} and lr.person_named("媽媽").pid == pid  # type: ignore[union-attr]
    assert lr.people[pid].name_raw == "妈妈" and lr.people[pid].bound_t == 2.0
    assert not lr.bind_name(99, "x", "x", 3.0)  # (nobody is made up from a name)
    assert len(lr.people) == 1


def test_one_name_is_one_person_the_same_role_seen_again_is_merged() -> None:
    lr = learner()
    a = lr.see_person(1, MOM_LOOK, 1.0)
    assert lr.bind_name(a, "伴侶", "伴侶", 2.0)
    lr.people[a].greeted, lr.people[a].bond = 2, 0.4
    b = lr.see_person(2, None, 30.0)  # (the same person, the eye has no look and a new track)
    assert b != a and len(lr.people) == 2
    lr.people[b].greeted = 1
    assert lr.bind_name(b, "伴侶", "伴侶", 31.0)  # (the page introduces the role again)
    assert lr.person_names() == {b: "伴侶"}  # one 伴侶, not two
    assert len(lr.people) == 1 and a not in lr.people
    assert lr.people[b].greeted == 3 and lr.people[b].bond == 0.4  # what it knew is kept
    assert lr.see_person(1, MOM_LOOK, 40.0) == b  # (the old track now means the one person)


def test_two_different_names_stay_two_people() -> None:
    lr = learner()
    a = lr.see_person(1, MOM_LOOK, 1.0)
    b = lr.see_person(2, None, 1.0)
    lr.bind_name(a, "伴侶", "伴侶", 2.0)
    lr.bind_name(b, "小孩", "小孩", 2.0)
    assert sorted(lr.person_names().values()) == ["伴侶", "小孩"] and len(lr.people) == 2


def test_a_pending_name_counts_evidence_and_is_bounded() -> None:
    lr = learner()
    pid = lr.see_person(1, MOM_LOOK, 1.0)
    assert lr.pend_name(pid, "媽媽") == 1 and lr.pend_name(pid, "媽媽") == 2
    for k in range(20):
        lr.pend_name(pid, f"名{k}")
    assert len(lr.people[pid].pending) <= 6
    assert lr.pend_name(99, "x") == 0


def test_notes_are_kept_locally_and_bounded() -> None:
    lr = learner()
    for k in range(80):
        lr.keep_note(float(k), None, f"事{k}")
    assert len(lr.notes) == 50 and lr.notes[-1][2] == "事79"


def test_the_look_mean_is_a_running_mean_and_ignores_missing_parts() -> None:
    lr = learner()
    pid = lr.see_person(1, (0.3, 0.3, 0.3, math.nan, math.nan, math.nan), 1.0)
    lr.see_person(1, (0.31, 0.29, 0.31, 0.8, math.nan, math.nan), 1.2)
    rec = lr.people[pid]
    assert rec.mean[3] == pytest.approx(0.8) and rec.mean[0] == pytest.approx(0.305, abs=1e-3)
