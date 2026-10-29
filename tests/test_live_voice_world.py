"""Live voice, LV0: the world stimulus "a family member says <line> from <place>" — a sound event the
hearing model hears like any voice and the microphone stream plays as audio. The world part runs
here on a stand-in world (no MuJoCo): the stimulus, its place, its level and its refusals do not
need physics."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.simweb import neuro_mic as M
from duckmind.world import neuro_audio as A
from duckmind.world import neuro_senses as ns
from duckmind.world import neuro_sound_synth as S
from duckmind.world import neuro_world as nw
from duckmind.world.neuro_utterances import FIXTURE_DIR_ENV, LIVE_PREFIX, MANIFEST


def _fixtures(root: Path) -> Path:
    lines = [
        ("call_name", "蛋黃，過來這裡。", "call_name", "蛋黃", 1.6),
        ("hello", "你好！", "greet", None, 1.1),
        ("question", "你看到什麼？", "ask_see", None, 1.4),
    ]
    entries = []
    for ident, text, intent, name, seconds in lines:
        wave = A.normalise(S.render("voice", seconds=seconds, seed=5, ident=len(ident))[0])
        A.write_wav(root / f"{ident}.wav", A.to_pcm16(wave), A.SR)
        entries.append(
            {
                "id": ident,
                "text": text,
                "speaker": "mom",
                "persona": "partner",
                "intent": intent,
                "name": name,
                "file": f"{ident}.wav",
                "duration_s": len(wave) / A.SR,
                "samples": len(wave),
                "active_rms_dbfs": -26.0,
            }
        )
    (root / MANIFEST).write_text(
        json.dumps({"version": 1, "sample_rate": A.SR, "speakers": {}, "utterances": entries}),
        encoding="utf-8",
    )
    return root


class _World:
    """The world part on a stand-in world: a clock, a lock, a duck at (0.5, 0.5) m."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: float = 12.0) -> None:
        monkeypatch.setenv(FIXTURE_DIR_ENV, str(_fixtures(tmp_path)))
        self.part = nw.NeuroWorld(seed=7)
        self.data = SimpleNamespace(time=now, xpos=np.array([[0.5, 0.5, 0.075]]))
        self.world = SimpleNamespace(lock=threading.RLock(), data=self.data)
        self.part.world = self.world
        self.part.robot = SimpleNamespace(trunk=0)
        self.part._wall_between = lambda *a, **k: False  # type: ignore[method-assign]

    def say(self, **request: Any) -> dict[str, Any]:
        return self.part.stimulate({"stim": "say", **request})

    def sounds(self) -> list[dict[str, Any]]:
        """The section's sound list, with the levels at the head the world computes."""
        with self.world.lock:
            self.part._levels(self.data, float(self.data.time))
        return [e.view() for e in self.part.sounds]


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _World:
    return _World(tmp_path, monkeypatch)


# ── the stimulus ─────────────────────────────────────────────────────────────────────


def test_say_is_a_page_stimulus_the_world_can_make() -> None:
    assert "say" in nw.STIMULI and "say" in ns.SUPPORTED_STIMULI
    assert set(ns.SUPPORTED_STIMULI) == set(nw.STIMULI)
    assert nw.STIMULI["say"]


def test_a_family_member_says_a_line_from_a_place_as_a_world_sound(world: _World) -> None:
    answer = world.say(utterance="call_name", who="partner", where_mm=[1500, 500])
    sound = answer["sound"]
    assert answer["ok"] and answer["stim"] == "say"
    assert sound["kind"] == "name" and sound["name"] is True and sound["env"] == "speech"
    assert sound["utterance"] == "call_name" and sound["src"] == "person:partner"
    assert sound["t0"] == 12.0 and sound["t1"] == pytest.approx(12.0 + 1.6, abs=0.03)
    assert sound["db1m"] == nw.SOUND_DB_1M["name"]
    # where the person stands: 1.5 m, 0.5 m; their mouth at their own height (the partner: 1.62 m)
    x, y, z = sound["pos_mm"]
    assert (x, y) == (1500, 500) and z == pytest.approx(1450 * 1.62 / 1.72, abs=2)
    assert answer["utterance"]["text"] == "蛋黃，過來這裡。" and answer["where_mm"][:2] == [
        1500,
        500,
    ]
    assert "蛋黃，過來這裡。" in str(answer["note_zh"])
    # and it is in the world's sound list
    assert [s["utterance"] for s in world.sounds()] == ["call_name"]


def test_the_kind_follows_the_line_a_name_a_greeting_or_other_speech(world: _World) -> None:
    kinds = {
        line: world.say(utterance=line, where_mm=[1000, 0])["sound"]["kind"]
        for line in ("call_name", "hello", "question")
    }
    assert kinds == {"call_name": "name", "hello": "hello", "question": "voice"}
    assert world.sounds()[1]["db1m"] == nw.SOUND_DB_1M["hello"]


def test_a_name_is_the_ducks_only_when_it_is_the_ducks_name(world: _World) -> None:
    other = world.say(utterance="call_name", where_mm=[1000, 0], duck_name="阿勇")["sound"]
    assert other["kind"] == "voice" and other["name"] is False  # (someone else's name: data)
    mine = world.say(utterance="call_name", where_mm=[1000, 0], duck_name="蛋黃")["sound"]
    assert mine["kind"] == "name" and mine["name"] is True


def test_the_level_at_the_head_is_the_worlds_distance_model(world: _World) -> None:
    world.say(utterance="question", where_mm=[1500, 500])  # 1.0 m away along the floor
    world.say(utterance="question", where_mm=[2500, 500])  # 2.0 m
    near, far = world.sounds()

    def metres(sound: dict[str, Any]) -> float:  # (a mouth 1.45 m up, the head 7.5 cm up)
        x, y, z = (v / 1000.0 for v in sound["pos_mm"])
        return math.dist((x, y, z), (0.5, 0.5, 0.075))

    assert near["db"] == pytest.approx(nw.level_at(62.0, metres(near)), abs=0.1)
    assert far["db"] == pytest.approx(nw.level_at(62.0, metres(far)), abs=0.1)
    assert near["db"] > far["db"]


def test_a_later_start_and_a_louder_voice_are_the_requests(world: _World) -> None:
    sound = world.say(utterance="hello", where_mm=[0, 0], delay_s=0.5, db1m=75.0)["sound"]
    assert sound["t0"] == 12.5 and sound["db1m"] == 75.0
    late = world.say(utterance="hello", where_mm=[0, 0], delay_s=999)["sound"]
    assert late["t0"] == 22.0  # (at most 10 s ahead)


def test_loose_numbers_in_a_request_fall_back_to_their_defaults(world: _World) -> None:
    sound = world.say(
        utterance="hello", where_mm=[0, 0], delay_s="soon", db1m=float("nan"), seconds="x"
    )["sound"]
    assert sound["t0"] == 12.0 and sound["db1m"] == nw.SOUND_DB_1M["hello"]
    live = world.say(live=True, where_mm=[0, 0], seconds="long", db1m=None)["sound"]
    assert live["t1"] - live["t0"] == pytest.approx(1.0)


def test_with_no_place_a_voice_comes_from_the_room(world: _World) -> None:
    sound = world.say(utterance="hello")["sound"]
    x, y, _ = sound["pos_mm"]
    assert 1.4 <= math.hypot(x / 1000 - 0.5, y / 1000 - 0.5) <= 2.6 and sound["src"] == "offscreen"


def test_a_browser_microphone_is_a_live_voice_event_for_a_second(world: _World) -> None:
    answer = world.say(live=True, who="owner", where_mm=[2000, 0], seconds=1.0)
    sound = answer["sound"]
    assert sound["utterance"] == f"{LIVE_PREFIX}owner" and sound["kind"] == "voice"
    assert sound["t1"] - sound["t0"] == pytest.approx(1.0)
    assert "utterance" not in answer  # (no scripted line behind it)
    assert world.say(live=True, where_mm=[0, 0], seconds=99)["sound"]["t1"] == pytest.approx(
        12.0 + 3.0
    )
    assert world.say(live=True, where_mm=[0, 0], seconds=0)["sound"]["t1"] == pytest.approx(
        12.0 + 0.2
    )


def test_an_event_without_a_line_leaves_the_sound_views_as_they_were(world: _World) -> None:
    world.part.add_sound("clap", (1.0, 0.5, 1.0), t0=12.0, seconds=0.04, src="x")
    assert "utterance" not in world.sounds()[0]


# ── what it refuses, in plain words ──────────────────────────────────────────────────


def test_an_unknown_line_names_the_lines_there_are(world: _World) -> None:
    with pytest.raises(ValueError) as caught:
        world.say(utterance="no_such_line", where_mm=[0, 0])
    text = str(caught.value)
    assert "no_such_line" in text and "call_name" in text and "hello" in text


def test_no_lines_at_all_is_said_not_crashed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path / "nowhere"))
    part = nw.NeuroWorld(seed=7)
    part.world = SimpleNamespace(lock=threading.RLock(), data=SimpleNamespace(time=0.0))
    with pytest.raises(ValueError) as caught:
        part.stimulate({"stim": "say", "utterance": "hello", "where_mm": [0, 0]})
    assert "劇本台詞" in str(caught.value)
    # (a live voice needs no recorded line)
    assert part.stimulate({"stim": "say", "live": True, "where_mm": [0, 0]})["ok"]


@pytest.mark.parametrize(
    "where",
    [[1], "ab", [float("nan"), 0], [0, float("inf")], [60_000, 0], [0, -60_000], ["x", "y"]],
)
def test_a_place_that_is_not_two_numbers_inside_the_home_is_refused(
    world: _World, where: Any
) -> None:
    with pytest.raises(ValueError) as caught:
        world.say(utterance="hello", where_mm=where)
    assert "公釐" in str(caught.value)
    assert world.part.sounds == []


# ── the hearing model and the microphone are one event's two senses ─────────────────


def test_the_hearing_model_hears_the_line_start_and_the_microphone_plays_it(world: _World) -> None:
    world.say(utterance="call_name", where_mm=[1500, 500], duck_name="蛋黃")
    sounds = world.sounds()
    event = sounds[0]
    hearing = ns.HearingModel(seed=7)
    mic = M.make_mic(seed=7)  # (the lines of $DUCKMIND_VOICE_FIXTURES)
    assert mic.library is not None and "call_name" in mic.library.lines
    onset_at, kinds, played = None, [], []
    for k in range(250):
        s = 12.0 + k * 0.02
        frame = mic.tick(s, sounds)
        hearing.update(sounds, now_s=s)
        sense = hearing.step(s, 28.0)
        if onset_at is None and sense.onset > 0:
            onset_at = s
        if sense.kind:
            kinds.append(sense.kind)
        if event["t0"] <= s < event["t1"]:
            played.append(A.to_float(frame.pcm))
    assert onset_at is not None and 12.0 <= onset_at <= 12.15
    assert kinds and set(kinds) <= {"name", "voice"}, "classed as speech, a name it may be"
    # the stream is as loud as the world's dB at the head says (the hearing model's own level)
    heard = A.active_rms_dbfs(np.concatenate(played))
    assert heard + A.FULL_SCALE_DB == pytest.approx(event["db"], abs=1.5)
    assert hearing.background < 40.0
