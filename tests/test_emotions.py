"""The 14 emotions (Pollen Robotics' microduck_emotions, 2026-10-08): the data, the player, the body link's head and
bow, the picker (the brain's emotion layer) and the duck's own voice for them."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from test_neuro_bodylink import DT, state

from duckmind import paths
from duckmind.duckbody import protocol as P
from duckmind.neuro import emotions as EO
from duckmind.neuro.bodylink import EMOTION_HEAD, BodyLink, BodyRequest
from duckmind.simweb import emotion_voice as EV

DATA = paths.data_dir() / "emotions" / "microduck_emotions-431c73d"


def test_the_data_is_the_workshops_fourteen() -> None:
    cat = EO.catalog()
    assert tuple(cat) == EO.ORDER and len(cat) == 14
    for name, e in cat.items():
        assert e.rows[0][0] == 0.0 and abs(e.rows[1][0] - 0.1) < 1e-9
        assert 0.0 <= e.sound_at_s <= 0.35, (
            name
        )  # the sound waits for the beak's keyframes, not longer
        assert e.sound_tag == f"emo_{name}"
        assert all(0.0 <= row[6] <= 1.0 for row in e.rows)
        assert all(0.0 <= row[5] <= EO.BOW_MAX for row in e.rows), name  # the trained bow range
    assert {n for n, e in cat.items() if e.sits} == {"devastated", "play_dead"}
    assert (
        cat["devastated"].sit_from_s == 0.0
    )  # the file's 0.3 s lead-in is gone: the press is t = 0
    assert cat["devastated"].duration_s == 8.5 and cat["sad"].duration_s == 7.5
    assert {n for n, e in cat.items() if e.bows} == {
        "sad",
        "angry",
        "mock",
        "excited",
        "impatient",
        "laugh",
    }
    for name, e in cat.items():
        for row in e.rows:  # every target is inside what the body link lets an emotion send
            for value, (lo, hi) in zip(row[1:5], EMOTION_HEAD, strict=True):
                assert lo <= value <= hi, (name, row)
    assert tuple(EO.BY_ID) == tuple(f"emo_{n}" for n in EO.ORDER)


def test_every_emotion_has_a_sound_recipe_and_the_page_list_matches() -> None:
    assert set(EV.EMOTIONS) == set(EO.ORDER)
    assert EV.VARIANTS["no"] == ("a", "b", "c") and EV.VARIANTS["angry"] == ("a", "b")
    rows = EO.status()
    assert [r["id"] for r in rows] == [f"emo_{n}" for n in EO.ORDER]
    assert all(r["group"] == "emotion" and r["needs"] == [] for r in rows)
    assert {r["id"] for r in rows if r["brain"]} == {
        f"emo_{n}" for n in EO.BRAIN_USE
    }  # yes / yes_fast / no: only the button


def test_the_player_interpolates_voices_once_and_ends() -> None:
    e = EO.catalog()["yes"]
    p = EO.Player(e, 100.0)
    a = p.frame(100.0)
    assert a.head == (0.0, 0.0, 0.0, 0.0) and a.voice is None and not a.done and a.pitch is None
    mid = p.frame(100.15)  # between the keyframes at 0.1 s (0.067) and 0.2 s (0.458)
    assert 0.067 < mid.head[1] < 0.458
    tags = [p.frame(100.0 + k * DT).voice for k in range(80)]
    assert tags.count("emo_yes") == 1 and abs(tags.index("emo_yes") - e.sound_at_s / DT) <= 1
    assert not p.frame(101.4).done and p.frame(101.6).done
    after = p.frame(110.0)
    assert after.head == (0.0, 0.0, 0.0, 0.0)  # (past the last key it holds the last: home)


def test_a_bowing_emotion_sends_a_pose_and_a_sitting_one_sits_then_stands() -> None:
    angry = EO.Player(EO.catalog()["angry"], 0.0)
    pitches = [angry.frame(k * 0.02).pitch for k in range(140)]
    assert all(p is not None and 0.0 <= p <= EO.BOW_MAX for p in pitches) and max(pitches) > 0.1
    dev = EO.Player(EO.catalog()["devastated"], 0.0)
    assert (
        dev.frame(0.0).sit
        and dev.frame(8.4).sit
        and dev.frame(9.5).sit is False
        and not dev.frame(9.5).done
    )
    assert dev.frame(8.5 + EO.SIT_HOLD_S + 0.01).done
    assert (
        dev.frame(5.0).head[0] < -1.0 and dev.frame(5.0).head[1] > 0.9
    )  # the droop, as the workshop played it
    yes = EO.Player(EO.catalog()["yes"], 0.0)
    assert yes.frame(0.5).sit is False


def test_abort_ends_it_at_once() -> None:
    p = EO.Player(EO.catalog()["sad"], 0.0)
    assert not p.frame(1.0).done
    p.abort("reflex")
    f = p.frame(1.02)
    assert f.done and p.aborted == "reflex" and f.voice is None


def test_the_body_link_gives_an_emotion_its_wide_head_and_its_bow() -> None:
    link = BodyLink()
    want = (-1.2, 0.8, 0.5, 0.0)
    req = BodyRequest(head=want, posture="stand", emotion=True, pose=0.12)
    outs = [link.tick(state(k * DT), req) for k in range(60)]
    head = [i for o in outs for i in o.intents if i.method == "robot.head"][-1]
    assert head.params["neck_pitch"] == -1.2 and head.params["head_pitch"] == 0.8
    poses = [i for o in outs for i in o.intents if i.method == "robot.pose"]
    assert poses and all(i.params["active"] and i.params["pitch"] == 0.12 for i in poses)
    # without the flag the same head is clamped into the seated range and no pose is ever sent
    link2 = BodyLink()
    outs2 = [link2.tick(state(k * DT), BodyRequest(head=want, posture="stand")) for k in range(60)]
    head2 = [i for o in outs2 for i in o.intents if i.method == "robot.head"][-1]
    assert head2.params["neck_pitch"] > -0.6 and not [
        i for o in outs2 for i in o.intents if i.method == "robot.pose"
    ]
    # the emotion ends: the pose is released once
    off = link.tick(state(61 * DT), BodyRequest(posture="stand"))
    release = [i for i in off.intents if i.method == "robot.pose"]
    assert len(release) == 1 and release[0].params["active"] is False
    assert not [
        i
        for i in link.tick(state(62 * DT), BodyRequest(posture="stand")).intents
        if i.method == "robot.pose"
    ]


def test_a_fall_takes_the_bow_and_the_head_back() -> None:
    link = BodyLink()
    req = BodyRequest(head=(-1.2, 0.8, 0.5, 0.0), posture="stand", emotion=True, pose=0.12)
    for k in range(30):
        link.tick(state(k * DT), req)
    out = link.tick(state(30 * DT, fallen=True, tilt=80.0), req)
    assert [i for i in out.intents if i.method == "robot.pose" and not i.params["active"]]
    heads = [i for i in out.intents if i.method == "robot.head"]
    assert heads and heads[-1].params["neck_pitch"] == 0.0 and out.righting != "idle" or out.gate


def test_the_pose_intent_is_sent_but_the_other_forbidden_methods_stay_forbidden() -> None:
    from duckmind.duckbody.reflex import GuardContext, check_intent
    from duckmind.duckbody.robotd_client import ALLOWED_METHODS, FORBIDDEN_METHODS

    assert "robot.pose" in ALLOWED_METHODS and "robot.pose" not in FORBIDDEN_METHODS
    assert {"robot.torque", "robot.gains", "robot.servo", "robot.enable"} <= FORBIDDEN_METHODS
    assert check_intent(P.Intent.pose(0.1), GuardContext()) is None
    assert check_intent(P.Intent("robot.torque", {}), GuardContext()) is not None


# ── the brain's emotion layer ─────────────────────────────────────────────────


def ctx(t: float, **kw) -> EO.Context:
    base = dict(
        t_s=t, behaviour="look_around", new_bout=False, reflex=None, asleep=False, wants_walk=False,
        moving=False, standing=True, quiet=True, drives={"social": 0.3, "curiosity": 0.5}, stuck=0.0,
        blocked=False, strong_scare=0.0, cornered=False, person_in_view=False,
    )  # fmt: skip
    base.update(kw)
    return EO.Context(**base)


def play(picker: EO.Picker, contexts, t0: float = 0.0) -> list[tuple[float, str, str]]:
    out = []
    for c in contexts:
        got = picker.step(c)
        if got is not None:
            picker.played(got.name, c.t_s)
            picker.ended(c.t_s + 3.0)
            out.append((c.t_s, got.name, got.why))
    return out


def test_temperament_weights_follow_the_owners_words() -> None:
    bold = EO.weights(EO.Temperament(bold=0.95, shy=0.05, songbird=0.1, lazy=0.1))
    shy = EO.weights(EO.Temperament(bold=0.05, shy=0.95, songbird=0.1, lazy=0.1))
    song = EO.weights(EO.Temperament(bold=0.5, shy=0.5, songbird=0.9, lazy=0.1))
    lazy = EO.weights(EO.Temperament(bold=0.5, shy=0.5, songbird=0.1, lazy=0.95))
    plain = EO.weights(EO.Temperament(bold=0.5, shy=0.5, songbird=0.1, lazy=0.1))
    for name in ("defiant", "angry", "mock"):
        assert bold[name] > shy[name], name
    for name in ("sad", "play_dead", "mmh", "devastated"):
        assert shy[name] > bold[name], name
    assert song["laugh"] > plain["laugh"] and song["excited"] > plain["excited"]
    assert all(lazy[k] < plain[k] for k in plain)  # a lazy duck: fewer of everything


def test_a_cue_makes_an_emotion_sometimes_and_a_cooldown_keeps_it_from_repeating() -> None:
    picker = EO.Picker(EO.Temperament(bold=0.5, shy=0.5, songbird=0.9, lazy=0.0), seed=3)
    seq = []
    for n in range(200):
        t = 20.0 + n * 20.0
        seq.append(ctx(t, cues=("arrival",)))
        for k in range(1, 8):
            seq.append(ctx(t + k * 0.5))
    got = play(picker, seq)
    names = {g[1] for g in got}
    assert names <= {"excited", "laugh"} and got  # (random: not every arrival; but over 200 some)
    assert len(got) < 200
    times = [g[0] for g in got if g[1] == "laugh"]
    assert all(b - a >= 60.0 for a, b in zip(times, times[1:], strict=False))  # its cooldown


def test_it_never_starts_while_busy() -> None:
    picker = EO.Picker(EO.Temperament(bold=0.9, shy=0.1, songbird=0.9, lazy=0.0), seed=1)
    busy_cases = [
        dict(reflex="startle"), dict(asleep=True), dict(behaviour="flee"), dict(behaviour="nap"),
        dict(behaviour="dance"), dict(quiet=False), dict(moving=True), dict(standing=False),
        dict(behaviour="return_dock"),
    ]  # fmt: skip
    for case in busy_cases:
        picker.pending.clear()
        got = []
        for n in range(400):
            got.append(
                picker.step(ctx(100.0 + n * 0.5, cues=("arrival", "speech", "greeted"), **case))
            )
        assert not any(got), case


def test_blocked_again_and_again_makes_impatience_and_the_bold_are_defiant_and_angry_more() -> None:
    def count(temper: EO.Temperament) -> dict[str, int]:
        totals: dict[str, int] = {}
        for seed in range(60):
            picker = EO.Picker(temper, seed=seed)
            seq = []
            for n in range(40):
                t = 10.0 + n * 40.0
                seq.append(ctx(t, blocked=True, stuck=1.0))
                seq.extend(ctx(t + k * 0.5) for k in range(1, 6))
            for _, name, _ in play(picker, seq):
                totals[name] = totals.get(name, 0) + 1
        return totals

    bold = count(EO.Temperament(bold=0.95, shy=0.05, songbird=0.1, lazy=0.1))
    shy = count(EO.Temperament(bold=0.05, shy=0.95, songbird=0.1, lazy=0.1))
    assert bold.get("impatient", 0) > 0 and bold.get("defiant", 0) > shy.get("defiant", 0)
    assert bold.get("angry", 0) >= shy.get("angry", 0)
    assert bold.get("angry", 0) < bold.get("impatient", 0)  # angry is rare


def test_loneliness_builds_slowly_and_devastated_is_rarer_than_sad() -> None:
    sad = dev = 0
    for seed in range(40):
        picker = EO.Picker(EO.Temperament(bold=0.1, shy=0.9, songbird=0.1, lazy=0.1), seed=seed)
        seq = [ctx(k * 1.0, drives={"social": 0.97}, quiet=True) for k in range(1500)]
        for _, name, _ in play(picker, seq):
            sad += name == "sad"
            dev += name == "devastated"
    assert sad > 0 and dev <= sad
    # nobody is lonely with a person in view, or before a minute
    picker = EO.Picker(EO.Temperament(bold=0.1, shy=0.9, songbird=0.1, lazy=0.1), seed=5)
    assert not play(
        picker, [ctx(k * 1.0, drives={"social": 0.97}, person_in_view=True) for k in range(1500)]
    )
    assert not play(picker, [ctx(k * 1.0, drives={"social": 0.97}) for k in range(60)])


def test_play_dead_only_for_a_cornered_shy_duck_after_a_strong_scare_and_it_is_rare() -> None:
    def runs(temper: EO.Temperament, cornered: bool) -> int:
        n = 0
        for seed in range(80):
            picker = EO.Picker(temper, seed=seed)
            seq = [ctx(10.0, strong_scare=0.95, cornered=cornered)] + [
                ctx(10.5 + k * 0.5) for k in range(14)
            ]
            n += sum(1 for g in play(picker, seq) if g[1] == "play_dead")
        return n

    shy = EO.Temperament(bold=0.05, shy=0.95, songbird=0.1, lazy=0.1)
    bold = EO.Temperament(bold=0.95, shy=0.05, songbird=0.1, lazy=0.1)
    assert runs(shy, True) > runs(bold, True) and runs(shy, False) == 0 and runs(shy, True) < 80


def test_the_same_seed_makes_the_same_choices() -> None:
    def once() -> list:
        picker = EO.Picker(EO.Temperament(bold=0.6, shy=0.4, songbird=0.4, lazy=0.2), seed=11)
        seq = []
        for n in range(80):
            seq.append(ctx(20.0 + n * 15.0, cues=("speech",) if n % 2 else ("arrival",)))
            seq.extend(ctx(20.0 + n * 15.0 + k * 0.5) for k in range(1, 8))
        return play(picker, seq)

    assert once() == once()


def test_temperament_of_a_genome() -> None:
    from duckmind.neuro.genome import preset

    bold = EO.Temperament.of(preset("explorer"))
    shy = EO.Temperament.of(preset("snuggler"))
    assert bold.bold > 0.8 and shy.bold < 0.1 and shy.lazy > 0.7


# ── the duck's own voice ───────────────────────────────────────────────────────


def test_the_voice_seed_of_yolk_and_its_pitch_scale() -> None:
    assert EV.Personality(936833334).pitch_center_hz == pytest.approx(414.6014, abs=1e-3)
    assert EV.Personality(EV.ROBOT_SEED).pitch_center_hz == pytest.approx(
        EV.ROBOT_PITCH_HZ, abs=1e-3
    )
    assert EV.Voice(EV.Personality(936833334)).scale == pytest.approx(1.742, abs=1e-3)


def test_every_sound_renders_without_a_bank_and_is_peak_normalised() -> None:
    v = EV.Voice(
        EV.Personality(936833334), None
    )  # no bank: the shock, the chirps and the alarm are synthesised
    for recipe in EV.RECIPES:
        sig = EV.render_one(recipe, v)
        assert 0.3 < len(sig) / EV.SR < 8.0, recipe.tag
        assert abs(float(np.abs(sig).max()) - 10 ** (-3 / 20)) < 1e-3
        assert np.isfinite(sig).all()


@pytest.mark.skipif(
    not (DATA / "sounds").exists(), reason="the microduck_emotions checkout is not here"
)
def test_the_notes_of_every_sound_match_the_originals_in_time() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "emotion_voice_check",
        Path(__file__).resolve().parents[1] / "tools" / "emotion_voice_check.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["emotion_voice_check"] = mod
    spec.loader.exec_module(mod)
    voice = EV.Voice(EV.Personality(936833334), None)
    recipes = {(r.tag, r.letter): r for r in EV.RECIPES}
    bad = []
    for name, key in mod.NAMES.items():
        sr, original = mod.read(DATA / "sounds" / f"{name}.wav")
        mine = EV.render_one(recipes[key], voice)
        a, b = mod.notes(original, sr), mod.notes(mine, EV.SR)
        if len(a) != len(b) or any(abs(p[0] - q[0]) > 0.03 for p, q in zip(a, b, strict=False)):
            bad.append(name)
    assert not bad, bad


def test_the_emotion_bank_renders_once_and_serves_only_its_own_names(tmp_path: Path) -> None:
    bank = EV.EmotionBank(tmp_path)
    assert bank.prepare(936833334, None) in ("rendering", "ready")
    thread = bank._threads.get(936833334)
    if thread is not None:
        thread.join(120)
    assert bank.ready(936833334) and bank.renders == 1
    tags = bank.tags(936833334, "/api/voice/v5/936833334")
    assert set(tags) == {f"emo_{n}" for n in EO.ORDER}
    assert list(tags["emo_no"]["variants"]) == ["a", "b", "c"]
    wav = bank.read(936833334, "emo_yes", "emo_yes_a.wav")
    assert wav is not None and wav[:4] == b"RIFF"
    assert bank.read(936833334, "emo_yes", "../../etc/passwd") is None
    assert bank.read(936833334, "emo_yes", "emo_yes_z.wav") is None
    assert bank.read(111, "emo_yes", "emo_yes_a.wav") is None
    # a second bank on the same folder finds it done (nothing is rendered again)
    other = EV.EmotionBank(tmp_path)
    assert other.ready(936833334) and other.renders == 0


def test_the_pitch_of_the_ducks_voice_is_the_ducks_not_the_robots() -> None:
    robot = EV.Voice(EV.Personality(EV.ROBOT_SEED), None)
    duck = EV.Voice(EV.Personality(936833334), None)
    recipe = next(r for r in EV.RECIPES if r.tag == "yes")

    def rough_f0(sig: np.ndarray) -> float:
        loud = np.flatnonzero(np.abs(sig) > 0.2)
        seg = sig[loud[0] : loud[0] + 4800]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / EV.SR)
        return (
            float(freqs[50 : np.argmax(spec[50:1500]) + 50][-1])
            if False
            else float(freqs[np.argmax(spec[20:300]) + 20])
        )

    assert rough_f0(EV.render_one(recipe, duck)) > 1.3 * rough_f0(EV.render_one(recipe, robot))


def test_random_is_seeded() -> None:
    assert random.Random(1).random() == random.Random(1).random()
