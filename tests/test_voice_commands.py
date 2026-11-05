"""Voice commands (「按住說話」; docs/design/neural-duck/VOICE-COMMANDS-2026-11-04.md).

* The words: :func:`duckmind.neuro.commands.understand` turns a sentence into a command, the owner's praise /
  scold, or nothing, in six languages — including what the local recogniser really wrote for the
  preregistered test sentences (``tests/data/voice_transcripts.json``, recorded by
  ``tools/voice_commands/measure.py``).
* The recogniser wrapper (:class:`duckmind.speech.asr.Recognizer`) over a stand-in engine: a press with
  nothing said never reaches the model, the model is primed with the language's command words and the
  duck's name, its echo of that prompt is dropped, a clip it does not take is refused.
* The route (``POST /api/neuro/voice/command``) on the real simulator server: a clip becomes the command the
  duck hears (``source: voice``) or the feedback the brain takes, and the answer says what was heard and
  understood.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.duckcore.api import Config
from duckmind.neuro.command_words import FEEDBACK_PHRASES, LANGS
from duckmind.neuro.commands import understand
from duckmind.simweb import neuro_api as N
from duckmind.simweb.server import SimServer, SimState, make_handler
from duckmind.simweb.voice_cmd_api import VoiceHub
from duckmind.speech import asr, listen

DATA = Path(__file__).with_name("data") / "voice_transcripts.json"


def _did(text: str, lang: str, name: str | None = None) -> str:
    got = understand(text, lang=lang, duck_name=name)
    return got.verb or got.feedback or "none"


# ── the words ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "lang", "expect"),
    [
        ("乖", "zh-Hant", "praise"),
        ("好棒喔", "zh-Hant", "praise"),
        ("蛋黃真棒", "zh-Hant", "praise"),
        ("不對", "zh-Hant", "scold"),
        ("不乖", "zh-Hant", "scold"),  # (the longer phrase: 不乖 holds 乖)
        ("不可以坐下", "zh-Hant", "scold"),  # (a negated command is the owner's 不可以)
        ("乖，坐下", "zh-Hant", "sit"),  # (a command in the sentence wins)
        ("今天天氣很好", "zh-Hant", "none"),  # (很好 counts only on its own)
        ("很好", "zh-Hant", "praise"),
        ("不好意思", "zh-Hant", "none"),
        ("好棒", "zh-Hans", "praise"),
        ("不对哦", "zh-Hans", "scold"),
        ("Good boy!", "en", "praise"),
        ("not bad", "en", "praise"),
        ("No, not like that.", "en", "scold"),
        ("No.", "en", "scold"),
        ("Good job, now sit.", "en", "sit"),
        ("where are my keys", "en", "none"),
        ("いい子だね", "ja", "praise"),
        ("ダメ！", "ja", "scold"),
        ("¡Muy bien!", "es", "praise"),
        ("No, eso está mal.", "es", "scold"),
        ("잘했어", "ko", "praise"),
        ("안 돼", "ko", "scold"),
    ],
)
def test_a_spoken_sentence_is_a_command_a_praise_a_scold_or_nothing(text, lang, expect) -> None:
    assert _did(text, lang, "蛋黃") == expect


def test_every_language_has_praise_and_scold_words() -> None:
    for kind in ("praise", "scold"):
        assert set(FEEDBACK_PHRASES[kind]) == set(LANGS)
        for lang in LANGS:
            first = FEEDBACK_PHRASES[kind][lang][0].lstrip("=")
            assert _did(first, lang) == kind, (kind, lang, first)


def test_what_the_recogniser_wrote_for_the_test_sentences_still_reads_right() -> None:
    """Every transcript the chosen recogniser made of the preregistered sentences (clean speech) that came to
    the right thing still does; and per language at least the measured share of them do."""
    record = json.loads(DATA.read_text(encoding="utf-8"))
    rows = record["rows"]
    assert {r["lang"] for r in rows} == set(LANGS)
    for lang in LANGS:
        mine = [r for r in rows if r["lang"] == lang]
        right = [
            r for r in mine if _did(r["heard"], lang, record["duck_name"].get(lang)) == r["cls"]
        ]
        assert len(right) >= record["right"][lang], (lang, len(right), record["right"][lang])
    for r in rows:
        if r["was_right"]:
            assert _did(r["heard"], r["lang"], record["duck_name"].get(r["lang"])) == r["cls"], r


# ── the recogniser wrapper ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("heard", "lang", "expect"),
    [
        ("Tómbate.", "es", "lie_down"),  # (túmbate, one letter off)
        ("按我", "zh-Hant", "look_at_me"),  # (看我)
        # what the recogniser wrote for development sentences that are not for the duck: a command word's
        # sound inside a longer sentence is not that command (「mala」 in 「Pájala」, 跟上 in 晚上)
        ("Pájala, téle.", "es", "none"),
        ("Mañana madrugo.", "es", "none"),
        ("我們晚上吃什麼?", "zh-Hant", "none"),
        ("車子停在樓下。", "zh-Hant", "none"),
    ],
)
def test_a_misheard_command_is_taken_by_its_sound_but_a_longer_sentence_is_not(
    heard, lang, expect
) -> None:
    spoken = listen.read_words(asr.Transcript(heard, lang, 1.0, 0.1), lang)
    assert spoken.meaning == expect, (heard, spoken.by_sound)


class ScriptedEngine:
    """A stand-in Whisper: says ``text`` and remembers what it was asked."""

    name = "fake"
    model = "scripted"

    def __init__(self, text: str = "坐下") -> None:
        self.text = text
        self.calls: list[tuple[int, str, str]] = []
        self.loaded = 0

    def load(self) -> None:
        self.loaded += 1

    def run(self, audio: np.ndarray, language: str, prompt: str) -> str:
        self.calls.append((audio.size, language, prompt))
        return self.text


def _speech(seconds: float = 1.0, level: float = 0.2) -> bytes:
    t = np.arange(int(seconds * asr.SR)) / asr.SR
    return (np.sin(2 * np.pi * 220 * t) * level * 32767).astype("<i2").tobytes()


def test_a_press_with_nothing_said_never_reaches_the_model() -> None:
    engine = ScriptedEngine()
    rec = asr.Recognizer(engine)
    short = rec.hear(_speech(0.1), "zh-Hant")
    quiet = rec.hear(_speech(1.0, level=0.0005), "zh-Hant")
    assert (short.text, short.empty) == ("", "too_short")
    assert (quiet.text, quiet.empty) == ("", "silent")
    assert engine.calls == []
    heard = rec.hear(_speech(1.0), "zh-Hant")
    assert heard.text == "坐下" and heard.empty is None and len(engine.calls) == 1


def test_the_model_is_primed_with_the_languages_command_words_and_the_ducks_name() -> None:
    engine = ScriptedEngine()
    rec = asr.Recognizer(engine)
    rec.hear(_speech(), "zh-Hant", "蛋黃")
    rec.hear(_speech(), "zh-Hans")
    rec.hear(_speech(), "es")
    (_, lang_t, prompt_t), (_, lang_s, prompt_s), (_, lang_e, prompt_e) = engine.calls
    assert lang_t == lang_s == "zh" and lang_e == "es"
    assert prompt_t.startswith("蛋黃、") and "站起來" in prompt_t and "乖" in prompt_t
    assert "站起来" in prompt_s and "站起來" not in prompt_s  # (the script the viewer reads)
    assert "siéntate" in prompt_e and "buen chico" in prompt_e
    unprimed = ScriptedEngine()
    asr.Recognizer(unprimed, prompt=False).hear(_speech(), "en")
    assert unprimed.calls[0][2] == ""


def test_the_models_echo_of_its_prompt_is_not_words() -> None:
    echo = asr.prompt_for("zh-Hant")
    heard = asr.Recognizer(ScriptedEngine(echo)).hear(_speech(), "zh-Hant")
    assert (heard.text, heard.empty) == ("", "no_speech")


@pytest.mark.parametrize(
    ("pcm", "lang", "reason"),
    [
        (_speech(11.0), "en", "too_long"),
        (b"\x00" * 3201, "en", "odd_bytes"),
        (_speech(), "fr", "bad_lang"),
    ],
)
def test_a_clip_the_recogniser_does_not_take_is_refused(pcm, lang, reason) -> None:
    with pytest.raises(asr.Refused) as caught:
        asr.Recognizer(ScriptedEngine()).hear(pcm, lang)
    assert caught.value.reason == reason


def test_without_an_engine_package_there_is_no_recogniser_and_a_reason(monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in ("mlx_whisper", "faster_whisper"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    rec, why = asr.make()
    assert rec is None and "faster_whisper" in why
    assert asr.make("off") == (None, "off")


# ── the route on the simulator server ───────────────────────────────────────────────────────


class FakeDriver:
    """The live driver's two doors the route uses: what the duck heard (``command``) and the owner's word
    (``feedback``)."""

    def __init__(self, *, trick: str | None = "spin") -> None:
        self.commands: list[Any] = []
        self.feedback_calls: list[tuple[str, str | None]] = []
        self.trick = trick

    def command(self, command: Any) -> dict[str, Any]:
        self.commands.append(command)
        return {"state": "queued", "verb": command.verb, "state_zh": "聽到了"}

    def feedback(self, kind: str, trick: str | None) -> dict[str, Any]:
        self.feedback_calls.append((kind, trick))
        if self.trick is None:
            return {"trick": None, "reason": "no_trick"}
        return {"trick": self.trick, "proficiency": 0.2, "step": 0.04, "capped": False,
                "shaped": False, "counted": kind != "scold", "reason": None}  # fmt: skip


@pytest.fixture()
def server():
    driver = FakeDriver()
    engine = ScriptedEngine()
    hub = N.NeuroHub(enabled=True)
    hub.attach(N.DriverSource(lambda: driver))
    voice = VoiceHub(hub, asr.Recognizer(engine))
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub, voice_commands=voice))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{httpd.server_address[1]}", driver=driver,
                          engine=engine, hub=hub, voice=voice)  # fmt: skip
    httpd.shutdown()
    httpd.server_close()


def _post_clip(url: str, pcm: bytes) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, pcm, {"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_a_spoken_command_reaches_the_duck_as_what_it_heard(server) -> None:
    server.engine.text = f"{server.hub.choice.name}，坐下。"
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=zh-Hant", _speech())
    assert status == 200 and data["ok"] is True
    assert data["heard"]["text"] == f"{server.hub.choice.name}，坐下。"
    assert data["understood"]["kind"] == "command" and data["understood"]["verb"] == "sit"
    (command,) = server.driver.commands
    assert (command.verb, command.source, command.lang, command.named) == (
        "sit",
        "voice",
        "zh-Hant",
        True,
    )
    assert data["answer"]["state"] == "queued" and data["answer"]["heard"]["source"] == "voice"
    assert server.engine.calls[0][2].startswith(
        server.hub.choice.name
    )  # (the duck's name primes it)


@pytest.mark.parametrize(("said", "kind"), [("好棒！", "praise"), ("不對喔", "scold")])
def test_a_spoken_praise_or_scold_reaches_the_brain(server, said, kind) -> None:
    server.engine.text = said
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=zh-Hant", _speech())
    assert status == 200 and data["understood"] == {"kind": "feedback", "feedback": kind,
                                                    "lang": "zh-Hant", "confidence": data["understood"]["confidence"]}  # fmt: skip
    assert server.driver.feedback_calls == [(kind, None)] and server.driver.commands == []
    assert data["answer"]["ok"] is True and data["answer"]["trick"] == "spin"


def test_a_praise_with_no_trick_lately_says_so_and_is_still_heard(server) -> None:
    server.driver.trick = None
    server.engine.text = "乖"
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=zh-Hant", _speech())
    assert status == 200 and data["understood"]["feedback"] == "praise"
    assert data["answer"]["ok"] is False and data["answer"]["reason"] == "no_trick"


def test_words_that_are_not_for_the_duck_do_nothing(server) -> None:
    server.engine.text = "今天天氣很好"
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=zh-Hant", _speech())
    assert status == 200 and data["understood"] == {"kind": "none", "reason": "not_understood"}
    assert (
        data["answer"] is None
        and server.driver.commands == []
        and server.driver.feedback_calls == []
    )


def test_a_silent_press_does_nothing_and_says_why(server) -> None:
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=en", _speech(1.0, 0.0001))
    assert status == 200 and data["heard"]["empty"] == "silent" and data["answer"] is None
    assert server.engine.calls == []


def test_the_status_says_when_the_button_may_be_on(server) -> None:
    with urllib.request.urlopen(f"{server.url}/api/neuro/voice/status", timeout=10) as response:
        st = json.load(response)
    assert st["enabled"] is True and st["engine"] == "fake" and st["langs"] == list(LANGS)
    server.hub.enabled = False
    with urllib.request.urlopen(f"{server.url}/api/neuro/voice/status", timeout=10) as response:
        assert json.load(response)["enabled"] is False
    status, data = _post_clip(f"{server.url}/api/neuro/voice/command?lang=en", _speech())
    assert (status, data["reason"]) == (503, "not_running")


@pytest.mark.parametrize(
    ("query", "pcm", "status", "reason"),
    [
        ("lang=fr", _speech(), 400, "bad_lang"),
        ("lang=en", b"", 400, "empty_body"),
        ("lang=en", b"\x00" * 3201, 400, "odd_bytes"),
        ("lang=en", b"\x00" * (asr.SR * 2 * 11), 413, "too_long"),
    ],
)
def test_a_clip_that_cannot_be_heard_is_refused_with_a_reason(
    server, query, pcm, status, reason
) -> None:
    got, data = _post_clip(f"{server.url}/api/neuro/voice/command?{query}", pcm)
    assert (got, data["ok"], data["reason"]) == (status, False, reason)
    assert server.driver.commands == []


def test_without_a_recogniser_the_route_says_why() -> None:
    hub = N.NeuroHub(enabled=True)
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{url}/api/neuro/voice/status", timeout=10) as response:
            st = json.load(response)
        assert st["enabled"] is False and st["engine"] is None and st["problem"] == "off"
        status, data = _post_clip(f"{url}/api/neuro/voice/command?lang=en", _speech())
        assert (status, data["reason"]) == (503, "no_recognizer")
    finally:
        httpd.shutdown()
        httpd.server_close()
