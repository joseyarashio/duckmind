"""The voice lineage as the live voice library (docs/design/GROWTH-G5-2026-10-11.md): Duck Core's
phrases, praise, song and age become a plan (pure, deterministic, versioned), the plan becomes clips
of the duck's own voice bank, and the clips reach the page between two calls.

The bank here is a stand-in of tones the test synthesises itself (no recorded voice of any kind):
what is checked is what the lineage does to a bank, not what a bank sounds like."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import content
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.life import voice_lineage as VL
from duckmind.simweb import lineage_voice as LV
from duckmind.simweb import voice

VSEED = 936833334
DUCK_SEED = 4107
F0 = {"chirp": 600.0, "inquire": 400.0, "coo": 200.0, "greet": 420.0, "alarm": 500.0, "peck": 300.0}


def tone(f0: float, seconds: float, harmonics: int = 3) -> np.ndarray:
    t = np.arange(int(seconds * LV.SR)) / LV.SR
    wave = sum(np.sin(2 * np.pi * f0 * (k + 1) * t) / (k + 1) for k in range(harmonics))
    envelope = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.03))
    return 0.5 * wave * envelope / max(1.0, float(np.max(np.abs(wave))))


def make_bank(root: Path, vseed: int = VSEED) -> Path:
    """A voice bank in upstream's layout (``v5/seed<voice seed>/<tag>/<tag>_<letter>.wav`` and the
    ``.seed`` marker) made of tones, so the clips can be measured."""
    folder = root / f"v{voice.BANK_VERSION}" / f"seed{vseed}"
    for tag in voice.TAGS:
        (folder / tag).mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(voice.file_names(tag)):
            seconds = 0.15 + 0.03 * (index % 7) + (0.6 if tag == "coo" else 0.0)
            LV.write_wav(
                folder / tag / name, tone(F0.get(tag, 500.0) * (1 + 0.01 * index), seconds)
            )
    (folder / voice.MARKER).write_text(f"{vseed}:{voice.BANK_VERSION}")
    return folder


def letters() -> dict[str, list[str]]:
    return {tag: list(voice.letters(tag)) for tag in voice.TAGS}


def phrase(code: str, tags: tuple[str, ...], **kw: Any) -> VL.PhraseView:
    return VL.PhraseView(
        code=code,
        tags=tags,
        gen=kw.get("gen", 0),
        parents=tuple(kw.get("parents", ())),
        praise=kw.get("praise", 0),
        promoted=kw.get("promoted", False),
        liked_days=kw.get("liked", 0),
    )


def view(gd: int = 60, **kw: Any) -> VL.LineageView:
    return VL.LineageView(
        seed=DUCK_SEED,
        gd=gd,
        stage=content.stage_for(gd).stage_id,
        branch=kw.get("branch", "explorer"),
        favorite_tag=kw.get("fav", "inquire"),
        first_word=kw.get("first", "hi"),
        phrases=tuple(kw.get("phrases", ())),
        promotions=tuple(kw.get("promotions", ())),
        song=tuple(kw.get("song", ())),
        song_version=1 if kw.get("song") else 0,
        molt="",
    )


FAMILY = [
    phrase("CI", ("chirp", "inquire"), praise=3, promoted=True, liked=2),
    phrase("COG", ("chirp", "coo", "greet"), praise=1, liked=1),
    phrase("GC", ("greet", "chirp"), gen=1, parents=("CI",), praise=2, promoted=True, liked=2),
]


def grown(gd: int = 60) -> VL.LineageView:
    return view(gd, phrases=FAMILY, promotions=("CI", "GC"), song=("CI", "GC"))


# ----------------------------------------------------------------------------- the plan (pure)


def test_the_same_lineage_is_the_same_plan() -> None:
    a = VL.plan_for(grown(), VSEED, letters())
    b = VL.plan_for(grown(), VSEED, letters())
    assert a == b and a.key == b.key
    assert VL.Plan.from_dict(json.loads(json.dumps(a.to_dict()))).key == a.key
    assert VL.plan_for(grown(), VSEED + 1, letters()).key != a.key  # (another duck's voice)


def test_a_day_that_changes_no_clip_keeps_the_key() -> None:
    # age moves in steps of a quarter semitone and 0.02 of a length: most days change nothing
    keys = {VL.plan_for(grown(gd), VSEED, letters()).key for gd in (100, 101)}
    assert len(keys) == 1
    assert (
        VL.plan_for(grown(100), VSEED, letters()).key
        != VL.plan_for(grown(120), VSEED, letters()).key
    )


def test_a_young_duck_is_higher_and_shorter_than_an_older_one() -> None:
    assert VL.age_pitch(0) == 4.0 and VL.age_pitch(180) == 0.0 and VL.age_pitch(400) == 0.0
    assert VL.age_pitch(3) > VL.age_pitch(60) > VL.age_pitch(120) > VL.age_pitch(180)
    assert VL.age_length(0) == 0.8 and VL.age_length(180) == 1.0
    young = VL.plan_for(view(5), VSEED, letters())
    old = VL.plan_for(view(150), VSEED, letters())
    for tag in VL.LINEAGE_TAGS:
        y, o = young.entry(tag), old.entry(tag)
        assert y is not None and o is not None
        mean = lambda e, f: sum(f(v.parts[0]) for v in e.variants) / len(e.variants)  # noqa: E731
        assert mean(y, lambda p: p.st) > mean(o, lambda p: p.st) + 1.0
        assert mean(y, lambda p: p.length) < mean(o, lambda p: p.length)


def test_an_older_duck_owns_more_variants_and_keeps_every_one_a_younger_had() -> None:
    young = VL.plan_for(view(5), VSEED, letters())
    old = VL.plan_for(view(200), VSEED, letters())
    for tag in VL.LINEAGE_TAGS:
        y = {v.letter for v in young.entry(tag).variants}  # type: ignore[union-attr]
        o = {v.letter for v in old.entry(tag).variants}  # type: ignore[union-attr]
        assert y < o, tag
    # the favourite sound is the richest of a hatchling's repertoire
    hatchling = VL.plan_for(view(2, fav="coo"), VSEED, letters())
    sizes = {tag: len(hatchling.entry(tag).variants) for tag in VL.LINEAGE_TAGS}  # type: ignore[union-attr]
    assert sizes["coo"] == max(sizes.values()) and sizes["coo"] > sizes["chirp"]


def test_what_it_learned_is_a_clip_of_its_own_in_the_order_it_matters() -> None:
    plan = VL.plan_for(grown(), VSEED, letters())
    names = [e.name for e in plan.entries]
    assert names[:4] == list(VL.LINEAGE_TAGS)
    assert {"ph_CI", "ph_GC", "ph_COG", "song"} <= set(names)
    # the first word is a call too (hi = greet, chirp = GC, which is also promoted: one clip)
    assert names.count("ph_GC") == 1
    song = plan.entry("song")
    assert song is not None and list(song.syllables) == ["chirp", "inquire", "greet", "chirp"]
    kinds = {e.code: e.kind for e in VL.phrase_entries(plan)}
    assert kinds["CI"] == "song_phrase" and kinds["COG"] == "liked"
    # syllables are the phrase's tags, one part each
    cog = plan.entry("ph_COG")
    assert cog is not None and [p.tag for p in cog.variants[0].parts] == ["chirp", "coo", "greet"]
    # a duck that learned nothing has the first word and its tags only
    bare = VL.plan_for(view(60), VSEED, letters())
    assert [e.name for e in bare.entries] == [*VL.LINEAGE_TAGS, "ph_GC"]
    assert bare.entry("ph_GC").kind == "first_word"  # type: ignore[union-attr]
    assert VL.plan_for(view(60, first=None), VSEED, letters()).entry("song") is None


def test_the_learned_phrases_are_capped_and_a_praised_one_is_held_longer() -> None:
    many = [
        phrase(f"C{'O' * n}", ("chirp",) + ("coo",) * n, liked=1, praise=1) for n in range(1, 8)
    ]
    plan = VL.plan_for(view(60, phrases=many, first=None), VSEED, letters())
    assert len(VL.phrase_entries(plan)) == VL.LIKED_MAX
    quiet = phrase("OC", ("coo", "chirp"), promoted=True, liked=2, praise=0)
    loud = phrase("CO", ("coo", "chirp"), promoted=True, liked=2, praise=5)
    plan = VL.plan_for(
        view(60, phrases=[quiet, loud], promotions=("OC", "CO"), first=None), VSEED, letters()
    )
    held = lambda code: plan.entry("ph_" + code).variants[0].parts[0].length  # type: ignore[union-attr]  # noqa: E731
    assert held("CO") > held("OC")


def test_a_question_rises_and_a_child_drifts_from_its_parent() -> None:
    ask = phrase("CI", ("chirp", "inquire"), promoted=True, liked=2)
    flat = phrase("IC", ("inquire", "chirp"), promoted=True, liked=2)
    plan = VL.plan_for(
        view(60, phrases=[ask, flat], promotions=("CI", "IC"), first=None), VSEED, letters()
    )
    last = plan.entry("ph_CI").variants[0].parts[-1]  # type: ignore[union-attr]
    assert last.tag == "inquire"
    twin = VL.plan_for(
        view(
            60,
            phrases=[phrase("CI", ("chirp", "inquire"), promoted=True, liked=2, gen=0)],
            promotions=("CI",),
            first=None,
        ),
        VSEED,
        letters(),
    )
    assert twin.entry("ph_CI").variants[0].parts[-1].st == last.st  # type: ignore[union-attr]
    # a mutation of CI: same tags, a generation on, differs only by the drift of its ancestry
    child = phrase("CI", ("chirp", "inquire"), promoted=True, liked=2, gen=2, parents=("OC",))
    other = phrase("OC", ("coo", "chirp"), gen=1)
    drifted = VL.plan_for(
        view(60, phrases=[child, other], promotions=("CI",), first=None), VSEED, letters()
    )
    sts = [p.st for p in drifted.entry("ph_CI").variants[0].parts]  # type: ignore[union-attr]
    base = [p.st for p in twin.entry("ph_CI").variants[0].parts]  # type: ignore[union-attr]
    shift = {round(a - b, 3) for a, b in zip(sts, base, strict=True)}
    expected = VL._drift(view(60, phrases=[child, other], promotions=("CI",), first=None), "CI")
    assert shift == {expected} and expected != 0.0


def test_nothing_is_planned_for_the_calls_duck_core_has_no_say_over() -> None:
    plan = VL.plan_for(grown(), VSEED, letters())
    assert (
        plan.entry("alarm") is None and plan.entry("peck") is None and plan.entry("wheee") is None
    )
    assert VL.LINEAGE_TAGS == ("chirp", "inquire", "coo", "greet")


def test_version_guard_a_changed_rule_or_version_is_a_new_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = VL.plan_for(grown(), VSEED, letters()).key
    hash_before = VL.rules_hash()
    monkeypatch.setitem(VL.RULES, "pitch_young_st", 5.0)
    assert VL.rules_hash() != hash_before
    assert VL.plan_for(grown(), VSEED, letters()).key != before
    monkeypatch.undo()
    monkeypatch.setattr(VL, "LIBRARY_VERSION", VL.LIBRARY_VERSION + 1)
    assert VL.plan_for(grown(), VSEED, letters()).key != before
    monkeypatch.undo()
    monkeypatch.setattr(content, "table_hash", lambda: "another content")
    assert VL.plan_for(grown(), VSEED, letters()).key != before


# ----------------------------------------------------------------------------- the sound


def f0(x: np.ndarray) -> float:
    return LV.f0_of(x)


def test_the_dsp_moves_pitch_and_length_on_their_own() -> None:
    x = tone(440.0, 0.4)
    assert abs(f0(x) - 440.0) < 8
    up = LV.shape_clip(x, 12.0, 1.0)
    assert abs(f0(up) / 880.0 - 1.0) < 0.03 and abs(up.size / x.size - 1.0) < 0.03
    longer = LV.shape_clip(x, 0.0, 1.5)
    assert abs(longer.size / x.size - 1.5) < 0.03 and abs(f0(longer) / 440.0 - 1.0) < 0.03
    both = LV.shape_clip(x, 4.0, 0.8)
    assert (
        abs(both.size / x.size - 0.8) < 0.03
        and abs(f0(both) / (440.0 * 2 ** (4 / 12)) - 1.0) < 0.04
    )
    cut = LV.shape_clip(tone(440.0, 1.5), 0.0, 1.0, max_s=0.6)
    assert abs(cut.size / LV.SR - 0.6) < 0.01 and abs(float(cut[-1])) < 0.05
    assert float(np.max(np.abs(both))) == pytest.approx(float(np.max(np.abs(x))), rel=0.02)


def test_a_library_is_the_same_bytes_every_time_and_a_young_one_is_higher(tmp_path: Path) -> None:
    bank = make_bank(tmp_path / "voices")
    young = VL.plan_for(view(5), VSEED, letters())
    old = VL.plan_for(view(180), VSEED, letters())
    one, two = tmp_path / "one", tmp_path / "two"
    LV.render(young, bank, one)
    LV.render(young, bank, two)
    assert LV.small_hash(one) == LV.small_hash(two)
    out_old = tmp_path / "old"
    LV.render(old, bank, out_old)
    assert LV.small_hash(one) != LV.small_hash(out_old)
    for tag in ("chirp", "inquire"):
        y = np.mean([f0(LV.read_wav(p)) for p in (one / tag).glob("*.wav")])
        o = np.mean([f0(LV.read_wav(p)) for p in (out_old / tag).glob("*.wav")])
        assert y > o * 1.15, (
            tag,
            y,
            o,
        )  # (+4 semitones is 26 % up; the variants' jitter is under 6 %)
    y_len = np.mean([LV.read_wav(p).size for p in (one / "chirp").glob("*.wav")])
    o_len = np.mean([LV.read_wav(p).size for p in (out_old / "chirp").glob("*.wav")])
    assert y_len < o_len
    # a phrase is its syllables one after another
    grown_plan = VL.plan_for(grown(), VSEED, letters())
    lengths = LV.render(grown_plan, bank, tmp_path / "grown")
    cog = lengths["ph_COG"]["ph_COG_a.wav"]
    assert 0.5 < cog < 3 * VL.SYLLABLE_MAX_S + 0.5
    assert lengths["song"]["song_a.wav"] > lengths["ph_CI"]["ph_CI_a.wav"]


# ----------------------------------------------------------------------------- the manager


class Driver:
    """A library on a bank made of tones, and a lineage the test can change."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "voices"
        make_bank(self.root)
        self.bank = voice.VoiceBank(None, self.root)
        self.view = view(5)
        self.logs: list[str] = []
        self.lv = LV.LineageVoice(
            self.bank, DUCK_SEED, lambda: self.view, calendar=lambda: "cal", log=self.logs.append
        )
        self.bank.lineage = self.lv
        assert voice.voice_seed(DUCK_SEED) == VSEED

    def refresh(self, why: str = "test") -> str:
        return self.lv.refresh(why, wait=True)


def test_boot_builds_the_library_and_the_manifest_has_it(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    assert d.refresh("boot") == "building"
    assert d.lv.active is not None and d.lv.status()["state"] == "ready"
    manifest = d.bank.manifest(DUCK_SEED, wait_s=0)
    assert manifest["ready"] and manifest["lineage"]["state"] == "ready"
    key = manifest["lineage"]["key"]
    assert manifest["tags"]["chirp"]["lineage"] is True
    assert all(f"/lineage/{key}/" in url for url in manifest["tags"]["chirp"]["variants"].values())
    assert (
        "alarm" in manifest["tags"] and "lineage" not in manifest["tags"]["alarm"]
    )  # the bank's own
    assert "ph_GC" in manifest["tags"]  # the first word
    # served from the same route as the bank's own
    url = next(iter(manifest["tags"]["chirp"]["variants"].values()))
    reply = d.bank.handle(url, DUCK_SEED)
    assert reply is not None and reply.status == 200 and reply.body[:4] == b"RIFF"
    # another duck's page gets nothing of it
    other = d.bank.manifest(DUCK_SEED + 1, wait_s=0)
    assert "lineage" not in other


def test_only_the_names_the_plan_wrote_are_served(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.refresh()
    lib = d.lv.active
    assert lib is not None
    good = f"{voice.ROUTE}/lineage/{lib.key}/chirp/{sorted(lib.files['chirp'])[0]}"
    assert d.bank.handle(good, DUCK_SEED).status == 200  # type: ignore[union-attr]
    for bad in (
        f"{voice.ROUTE}/lineage/{lib.key}/chirp/chirp_z.wav",
        f"{voice.ROUTE}/lineage/{lib.key}/alarm/alarm_a.wav",
        f"{voice.ROUTE}/lineage/{lib.key}/chirp/..%2Fplan.json",
        f"{voice.ROUTE}/lineage/{lib.key}/../chirp/chirp_a.wav",
        f"{voice.ROUTE}/lineage/{lib.key}/plan.json",
        f"{voice.ROUTE}/lineage/0000/chirp/chirp_a.wav",
        f"{voice.ROUTE}/lineage/{lib.key}/chirp",
    ):
        reply = d.bank.handle(bad, DUCK_SEED)
        assert reply is not None and reply.status == 404, bad
    assert d.bank.handle(good, DUCK_SEED + 1).status == 404  # type: ignore[union-attr]
    assert d.bank.handle(good, None).status == 404  # type: ignore[union-attr]


def test_a_changed_lineage_builds_a_new_library_and_a_same_one_builds_nothing(
    tmp_path: Path,
) -> None:
    d = Driver(tmp_path)
    d.refresh("boot")
    first = d.lv.active
    assert first is not None
    d.refresh("night")  # nothing changed
    assert d.lv.active is first and d.lv.unchanged == 1 and d.lv.builds == 1
    d.view = grown(60)
    d.refresh("night")
    second = d.lv.active
    assert second is not None and second.key != first.key and d.lv.swaps == 2
    assert d.lv.previous is first
    desc = d.lv.status()["active"]
    assert {c["clip"] for c in desc["calls_learned"]} >= {"ph_CI", "ph_GC", "ph_COG"}
    assert desc["song"]["tags"] == ["chirp", "inquire", "greet", "chirp"]
    assert d.lv.status()["last_update"]["why"] == "night"
    assert d.lv.status()["lineage_size"] == 3
    # the library of the one before still serves what a page that has not heard yet asks for
    old = f"{voice.ROUTE}/lineage/{first.key}/chirp/{sorted(first.files['chirp'])[0]}"
    assert d.bank.handle(old, DUCK_SEED).status == 200  # type: ignore[union-attr]
    # a library that is on disk is reused, not made again
    again = LV.build_library(second.plan, d.bank.bank_dir(VSEED), d.lv.root)  # type: ignore[arg-type]
    assert again.folder == second.folder


def test_the_library_is_swapped_between_two_calls_never_in_the_middle_of_one(
    tmp_path: Path,
) -> None:
    d = Driver(tmp_path)
    d.refresh("boot")
    first = d.lv.active
    assert first is not None
    # a brain is driving; it says a long coo at t = 2000 ms
    said = d.lv.shape([{"tag": "coo", "level": 0.5}], "preen", 2000.0)
    assert said[0]["lib"] == first.key8
    busy_until = d.lv._busy_until_ms
    assert busy_until >= 2000.0 + first.longest_ms("coo")
    # the lineage changes while it is sounding: the new library is built, and waits
    d.view = grown(60)
    d.refresh("night")
    pending = d.lv.pending
    assert pending is not None and pending.key != first.key and d.lv.active is first
    again = d.lv.shape([{"tag": "chirp", "level": 0.5}], "wander", 2100.0)
    assert again[0]["lib"] == first.key8  # (still the library of the call in the air)
    assert not d.lv.tick(busy_until - 1.0) and d.lv.active is first
    # the quiet moment: the next tick after the call is over puts it in
    out = d.lv.shape([{"tag": "chirp", "level": 0.5}], "wander", busy_until + 1.0)
    assert d.lv.active is pending and d.lv.pending is None and out[0]["lib"] == pending.key8
    assert d.lv.swaps == 2 and any("兩次叫聲之間" in line for line in d.logs)
    # a page that is not driven by a brain (a headless check) gets it at once
    other = Driver(tmp_path / "headless")
    other.refresh("boot")
    assert other.lv.active is not None


def test_what_the_duck_says_comes_from_the_lineage(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.view = grown(60)
    d.refresh("boot")
    lv, lib = d.lv, d.lv.active
    assert lib is not None
    # an ordinary call stays the call the brain chose, in the library's reshaped clips
    out = lv.shape([{"tag": "coo", "level": 0.4}], "preen", 10_000.0)
    assert out == [{"tag": "coo", "level": 0.4, "lib": lib.key8}]
    # a little_song is the song
    out = lv.shape([{"tag": "chirp", "level": 0.5}], "little_song", 20_000.0)
    assert out[0]["tag"] == "song" and out[0]["parts"] == ["chirp", "inquire", "greet", "chirp"]
    assert out[0]["from"] == "chirp" and out[0]["lib"] == lib.key8
    # while it sings, another ordinary call does not chop it ...
    assert lv.shape([{"tag": "chirp", "level": 0.5}], "wander", 20_100.0) == []
    assert lv.dropped == 1
    # ... but a fall's alarm does, and is the bank's own
    alarm = lv.shape([{"tag": "alarm", "level": 0.9}], "wander", 20_200.0)
    assert alarm == [{"tag": "alarm", "level": 0.9}]
    assert lv.shape([{"tag": "chirp", "level": 0.5}], "wander", 20_300.0)[0]["tag"] == "chirp"
    # calling out says one of the phrases it learned
    said = {
        lv.shape([{"tag": "inquire", "level": 0.5}], "call_out", 30_000.0 + 10_000.0 * n)[0]["tag"]
        for n in range(12)
    }
    assert said <= {"ph_CI", "ph_GC", "ph_COG"} and len(said) >= 2
    status = lv.status()
    assert status["counts"]["songs_sung"] == 1 and status["counts"]["phrases_sung"] == 12
    assert status["recent"][-1]["folder"].startswith(str(lib.folder))


def test_the_same_lineage_sings_the_same_sequence(tmp_path: Path) -> None:
    def sing(sub: str) -> list[str]:
        d = Driver(tmp_path / sub)
        d.view = grown(60)
        d.refresh()
        return [
            d.lv.shape([{"tag": "chirp", "level": 0.5}], "call_out", 10_000.0 * (n + 1))[0]["tag"]
            for n in range(10)
        ]

    assert sing("a") == sing("b")


def test_a_duck_with_no_library_is_the_bank_and_a_failed_build_is_said(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    calls = [{"tag": "chirp", "level": 0.5}]
    assert d.lv.shape(calls, "wander", 1000.0) is calls  # (no library yet: untouched)
    broken = Driver(tmp_path / "x")
    broken.bank = voice.VoiceBank(None, tmp_path / "x" / "empty")
    broken.lv = LV.LineageVoice(broken.bank, DUCK_SEED, lambda: broken.view, log=broken.logs.append)
    broken.lv.refresh("boot", wait=True)
    status = broken.lv.status()
    assert status["state"] == "failed" and status["active"] is None and status["problem"]
    assert broken.logs and "做不出來" in broken.logs[-1]
    assert broken.lv.shape(calls, "wander", 1000.0) is calls


def test_refresh_is_safe_from_many_threads(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.view = grown(60)
    threads = [threading.Thread(target=d.lv.refresh, args=(f"t{n}",)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    d.refresh("last")
    assert d.lv.active is not None and d.lv.builds == 1  # (one library, however often it was asked)


# ----------------------------------------------------------------------------- Duck Core is only read


def bridged(seed: int = DUCK_SEED) -> Session:
    session = Session(
        Config(seed=seed, name="蛋黃", tof=True, nest=False, start="2026-03-02T07:00:00+08:00")
    )
    dispatch(session, "bridge", {"api": "bridge-v1"})
    return session


def test_reading_the_lineage_changes_nothing_in_duck_core_and_follows_its_praise() -> None:
    session = bridged()
    core = session.core

    def now() -> int:
        return int(core.s.now.timestamp())

    dispatch(session, "advance", {"seconds": 3600})
    dispatch(
        session,
        "experience",
        {"events": [{"k": "voiced", "t": now() + 1, "tags": ["chirp", "coo"]}]},
    )
    dispatch(session, "act", {"action": "owner_good", "params": {}})
    before = dataclasses.asdict(core.s)
    head = core.log.head
    seen = VL.view_of(core.s)
    assert dataclasses.asdict(core.s) == before and core.log.head == head
    assert seen.size == 1 and seen.phrases[0].code == "CO" and seen.phrases[0].liked_days == 1
    plan = VL.plan_for(seen, voice.voice_seed(DUCK_SEED), letters())
    assert plan.entry("ph_CO") is not None and plan.entry("ph_CO").kind == "liked"  # type: ignore[union-attr]
    assert VL.plan_for(VL.view_of(core.s), voice.voice_seed(DUCK_SEED), letters()).key == plan.key
    digest = hashlib.sha256(json.dumps(before, sort_keys=True, default=str).encode()).hexdigest()
    assert (
        digest
        == hashlib.sha256(
            json.dumps(dataclasses.asdict(core.s), sort_keys=True, default=str).encode()
        ).hexdigest()
    )


# ----------------------------------------------------------------------------- the life host


def test_the_life_host_keeps_the_voice_and_says_it_in_api_life(tmp_path: Path) -> None:
    from datetime import datetime

    from duckmind.i18n.boundary import attach
    from duckmind.life.clock import TPE, DuckClock
    from duckmind.life.record import LifeRecord
    from duckmind.simweb.life_host import LifeHost
    from duckmind.simweb.server import SimState

    birth = datetime(2026, 10, 10, 15, 0, tzinfo=TPE)
    record = LifeRecord.create(
        tmp_path / "life", name="蛋黃", seed=DUCK_SEED, birth=birth, nest=False, tof=True,
        backdate_days=0,
    )  # fmt: skip
    wall = [birth.timestamp()]
    clock = DuckClock(wall=lambda: wall[0])
    config = Config(
        seed=DUCK_SEED, name="蛋黃", tof=True, nest=False, start=record.birth.isoformat()
    )
    state = SimState(config, None, record.core_path, life=record, clock=clock, attach_neural=True)
    host = LifeHost(record, state, clock, every_s=600.0, wall=lambda: wall[0])
    state.life_host = host
    assert host.status()["voice"] is None and host.refresh_voice("night") == "unavailable"

    voices = tmp_path / "voices"
    make_bank(voices)
    bank = voice.VoiceBank(None, voices)
    lineage = host.attach_voice(bank, root=tmp_path / "lineage")
    assert lineage is not None and bank.lineage is lineage
    assert host.refresh_voice("boot", wait=True) == "building"
    first = host.status()["voice"]
    assert first["state"] == "ready" and first["lineage_size"] == 0
    assert first["active"]["calls_learned"] == []  # (a new-born has no first word yet)
    assert first["last_update"]["why"] == "boot" and first["counts"]["swaps"] == 1

    # the neural duck babbled a phrase and the family praised it: the lineage has one more call
    core = state.session.core
    dispatch(state.session, "advance", {"seconds": 3 * 86400})
    t = int(core.s.now.timestamp())
    dispatch(
        state.session,
        "experience",
        {"events": [{"k": "voiced", "t": t + 1, "tags": ["greet", "coo"]}]},
    )
    dispatch(state.session, "act", {"action": "owner_good", "params": {}})
    assert host.refresh_voice("regrow", wait=True) == "building"  # (what G3's nightly regrow calls)
    after = host.status()["voice"]
    assert after["lineage_size"] == 1 and after["counts"]["swaps"] == 2
    assert any(c["code"] == "GO" and c["how"] == "liked" for c in after["active"]["calls_learned"])
    assert after["last_update"]["why"] == "regrow" and after["last_update"]["growth_day"] >= 3
    assert after["active"]["key"] != first["active"]["key"]
    # nothing changed: nothing is built
    host.refresh_voice("night", wait=True)
    assert host.status()["voice"]["counts"]["unchanged"] == 1
    json.dumps(attach(dict(host.status())))  # (the API answer goes through the i18n boundary)
