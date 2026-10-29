"""Live voice, LV0: the scripted family lines (``tests/fixtures/voice``): what is stored matches what
was asked for and what the local recogniser heard, the files are well formed and small, and every
line plays through the world's ``say`` stimulus into the duck's microphone at the level the world
gives its place. numpy only; the lines were made once with the local TTS (``tools/voice_fixtures``)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.simweb import neuro_mic as M
from duckmind.world import neuro_audio as A
from duckmind.world import neuro_senses as ns
from duckmind.world import neuro_world as nw
from duckmind.world.neuro_utterances import FIXTURE_DIR_ENV, MANIFEST, UtteranceLibrary
from duckmind.world.people import PERSONAS

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "voice"
LINES = ROOT / "tools" / "voice_fixtures" / "lines.json"
MAX_TOTAL_BYTES = 3_000_000
CJK = re.compile(r"[㐀-鿿]")
DUCK_NAME = "蛋黃"
# what the brief asks the set to hold: calling the duck, hello, an introduction, "what do you
# see", "stop calling", a question
NEEDED_INTENTS = {"call_name", "greet", "introduce", "ask_see", "stop_calling", "question"}


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((FIXTURES / MANIFEST).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def asked() -> dict[str, Any]:
    return json.loads(LINES.read_text(encoding="utf-8"))


def test_the_set_holds_what_the_brief_asks_for_and_is_small(manifest: dict[str, Any]) -> None:
    intents = {u["intent"] for u in manifest["utterances"]}
    assert NEEDED_INTENTS <= intents
    texts = {u["text"] for u in manifest["utterances"]}
    for must in ("你好！", "這是媽媽。", "你看到什麼？", "不要叫了。"):
        assert must in texts, must
    total = sum(p.stat().st_size for p in FIXTURES.iterdir() if p.suffix in (".wav", ".json"))
    assert total <= MAX_TOTAL_BYTES, total
    assert len(manifest["utterances"]) >= 10
    assert (
        manifest["sample_rate"] == 16_000 and manifest["channels"] == 1 and manifest["bits"] == 16
    )


def test_the_manifest_is_what_was_asked_for(
    manifest: dict[str, Any], asked: dict[str, Any]
) -> None:
    got = {u["id"]: u for u in manifest["utterances"]}
    want = {line["id"]: line for line in asked["lines"]}
    assert list(got) == list(want), "every asked line, in order, and nothing else"
    for ident, line in want.items():
        item = got[ident]
        assert item["text"] == line["text"] and item["speaker"] == line["speaker"]
        assert item["intent"] == line["intent"] and item["name"] == line.get("name")
        assert item["persona"] == asked["speakers"][line["speaker"]]["persona"]
        assert item["file"] == f"{ident}.wav"
        assert item["seed"] == line["seed"] + 1000 * item["take"]
    assert manifest["speakers"] == asked["speakers"]
    assert {p.id for p in PERSONAS} >= {s["persona"] for s in asked["speakers"].values()}


def test_the_lines_are_traditional_chinese_and_the_duck_is_named_only_where_asked(
    manifest: dict[str, Any],
) -> None:
    for item in manifest["utterances"]:
        assert CJK.search(item["text"]), item["id"]
        if item["name"]:
            assert item["name"] in item["text"], item["id"]  # (the line really says the name)
        else:
            assert DUCK_NAME not in item["text"], item["id"]  # (and no other line does)
    named = [u["id"] for u in manifest["utterances"] if u["name"]]
    assert len(named) >= 2 and "other_name" not in named


def test_every_file_is_16k_mono_16bit_and_what_the_manifest_says(manifest: dict[str, Any]) -> None:
    for item in manifest["utterances"]:
        path = FIXTURES / item["file"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"], item["id"]
        with wave.open(str(path), "rb") as handle:
            assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (
                1,
                2,
                16_000,
            )
            assert handle.getnframes() == item["samples"]
        assert item["duration_s"] == pytest.approx(item["samples"] / 16_000, abs=0.001)
        assert 0.4 <= item["duration_s"] <= 4.0, item["id"]
        x = A.to_float(A.read_wav(path)[0])
        assert A.active_rms_dbfs(x) == pytest.approx(A.REF_DBFS, abs=0.1), item["id"]
        assert item["active_rms_dbfs"] == pytest.approx(A.REF_DBFS, abs=0.1)
        assert 20 * math.log10(float(np.abs(x).max())) <= -3.0, "no clipping"


def test_a_line_starts_and_ends_close_to_its_speech(manifest: dict[str, Any]) -> None:
    for item in manifest["utterances"]:
        x = A.to_float(A.read_wav(FIXTURES / item["file"])[0]).astype(np.float64)
        n = len(x) // 160
        power = (x[: n * 160].reshape(n, 160) ** 2).mean(axis=1)  # 10 ms
        live = np.flatnonzero(power >= power.max() * 10 ** (-3.5))
        lead, tail = live[0] * 0.01, (n - 1 - live[-1]) * 0.01
        assert lead <= 0.25 and tail <= 0.35, (item["id"], lead, tail)
        # no pause inside a line longer than 0.7 s (a comma is shorter)
        gaps = np.diff(live) * 0.01
        assert gaps.max(initial=0.0) <= 0.7, item["id"]


def test_the_local_recogniser_heard_every_stored_line_as_its_text(manifest: dict[str, Any]) -> None:
    for item in manifest["utterances"]:
        assert item["asr_match"] == 1.0, (item["id"], item["text"], item["asr_check"])
        heard = re.sub(r"[\s，。！？、]", "", item["asr_check"])
        assert heard == re.sub(r"[\s，。！？、]", "", item["text"]), item["id"]


def test_the_default_library_is_this_folder_and_an_other_folder_can_be_named(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(FIXTURE_DIR_ENV, raising=False)
    library = UtteranceLibrary.default()
    assert library is not None and library.ids() == [u["id"] for u in manifest["utterances"]]
    assert library.root.resolve() == FIXTURES.resolve()
    for item in manifest["utterances"]:
        assert len(library.pcm(item["id"])) == item["samples"]
        assert library.get(item["id"]).name == item["name"]
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    assert UtteranceLibrary.default() is None


# ── every line, through the world and the microphone ─────────────────────────────────


def _world(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(FIXTURE_DIR_ENV, raising=False)
    part = nw.NeuroWorld(seed=7)
    data = SimpleNamespace(time=3.0, xpos=np.array([[0.0, 0.0, 0.075]]))
    world = SimpleNamespace(lock=threading.RLock(), data=data)
    part.world, part.robot = world, SimpleNamespace(trunk=0)
    part._wall_between = lambda *a, **k: False  # type: ignore[method-assign]
    return part, world, data


def test_every_line_plays_through_say_into_the_microphone_at_the_worlds_level(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    for item in manifest["utterances"]:
        part, world, data = _world(monkeypatch)
        who = item["persona"]
        part.stimulate(
            {
                "stim": "say",
                "utterance": item["id"],
                "who": who,
                "where_mm": [1500, 0],
                "duck_name": DUCK_NAME,
            }
        )
        with world.lock:
            part._levels(data, 3.0)
        sounds = [e.view() for e in part.sounds]
        (event,) = sounds
        assert event["t1"] - event["t0"] == pytest.approx(item["duration_s"], abs=0.001)
        assert event["kind"] == (
            "name" if item["name"] else "hello" if item["intent"] == "greet" else "voice"
        )
        mic = M.make_mic(seed=7)
        frames = [
            mic.tick(3.0 + k * 0.02, sounds) for k in range(int((item["duration_s"] + 0.5) / 0.02))
        ]
        x = np.concatenate([A.to_float(f.pcm) for f in frames])
        speech = x[: int(item["duration_s"] * A.SR)]
        assert A.active_rms_dbfs(speech) + A.FULL_SCALE_DB == pytest.approx(event["db"], abs=1.2), (
            item["id"]
        )
        assert A.rms_dbfs(x[int((item["duration_s"] + 0.1) * A.SR) :]) < -55, (
            "silence after the line"
        )
        # the hearing model hears the same event start
        hearing = ns.HearingModel(seed=7)
        onset = None
        for k in range(int((item["duration_s"] + 1.0) / 0.02)):
            s = 3.0 + k * 0.02
            hearing.update(sounds, now_s=s)
            if hearing.step(s, 28.0).onset > 0 and onset is None:
                onset = s
        assert onset is not None and 3.0 <= onset <= 3.2, item["id"]


def test_another_ducks_name_is_not_a_name_call(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    part, world, _ = _world(monkeypatch)
    answer = part.stimulate(
        {"stim": "say", "utterance": "call_name", "where_mm": [1000, 0], "duck_name": "阿勇"}
    )
    assert answer["sound"]["name"] is False and answer["sound"]["kind"] == "voice"
