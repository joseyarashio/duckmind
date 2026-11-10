"""「按住說話」 on the simulator page (``static/voicecmd.js``; docs/design/neural-duck/VOICE-COMMANDS-2026-11-04.md).

Run in Node over a fake page (a fake clock, a fake ``fetch``, a fake microphone, a tiny fake DOM): holding the button
records, letting go sends the clip as 16 kHz PCM to this server only, the line says what was heard and what it was
understood as, the command's answer goes to the command line and a praise / scold's to the feedback line; a tap, a
refused microphone and a server without a recogniser each say why. The words: every key in all six catalogs. How
it looks in a browser is checked by hand (1440×900 and 390×844, in the measurement record).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from duckmind.i18n import CODES

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
JS = (STATIC / "voicecmd.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("voicecmd_page_node_harness.mjs")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
RATE = 16000


def _run(scenario: str, data: Any = None) -> Any:
    args = [str(NODE), str(HARNESS), str(STATIC), scenario]
    if data is not None:
        args.append(json.dumps(data, ensure_ascii=False))
    done = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _answer(text: str, understood: dict[str, Any], answer: Any) -> dict[str, Any]:
    heard = {"text": text, "lang": "zh-Hant", "audio_s": 1.2, "took_s": 0.4, "empty": None if text else "no_speech"}
    return {"status": 200, "data": {"ok": True, "heard": heard, "understood": understood, "answer": answer, "took_s": 0.41}}


# ── the words ─────────────────────────────────────────────────────────────────────────────────


def test_every_word_the_voice_button_names_is_in_all_six_languages() -> None:
    keys = set(re.findall(r'"(page\.(?:voice|cmd\.verb|trick\.fb)\.[\w.]+)"', JS))
    keys |= {"page.cmd.mic", "page.voice.off.title"}
    assert len(keys) >= 30, keys
    for lang in CODES:
        text = (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")
        for key in keys:
            assert f'"{key}":' in text, (lang, key)


# ── voicecmd.js in node ───────────────────────────────────────────────────────────────────────


@needs_node
def test_a_48k_microphone_becomes_16k_pcm_without_losing_its_level() -> None:
    out = _run("pcm")
    assert out["length"] == RATE  # one second
    assert 16000 <= out["peak"] <= 16400  # 0.5 of full scale
    assert out["same"] == [0, 32767, -32767, 32767]  # (16 kHz is copied; past full scale is clipped)


@needs_node
def test_the_push_to_talk_button_leaves_the_ducks_calls_and_beak_alone() -> None:
    # (in a browser the page's 3D duck asked window.duckVoice.mouth() every frame and threw: voicecmd.js had taken
    # the calls' global name)
    out = _run("globals")
    assert out == {"calls_kept": True, "push_to_talk": True}


@needs_node
def test_hold_speak_let_go_sends_the_clip_and_shows_what_was_heard_and_done() -> None:
    answer = {"ok": True, "verb": "sit", "state": "queued", "state_zh": "聽到了"}
    out = _run("flow", {"seconds": 1.0, "lang": "ja", "answers": [_answer("坐下", {"kind": "command", "verb": "sit"}, answer)]})
    assert out["start"]["disabled"] is False and out["start"]["title"] == "page.voice.title"
    held = out["held"]
    assert (held["phase"], held["pressed"], held["label"], held["line"]) == ("listening", "true", "page.voice.release", "page.voice.listening")
    assert out["thinking"]["phase"] == "thinking" and out["thinking"]["line"] == "page.voice.thinking_line"
    (post,) = out["posts"]  # (one release: the pointer's up and its lost capture are one press)
    assert post["url"] == "/api/neuro/voice/command?lang=ja" and post["method"] == "POST"
    assert post["type"] == "application/octet-stream" and post["bytes"] == 2 * RATE  # 1 s of 16 kHz 16-bit
    assert out["mic"] == {"opened": 1, "closed": 1}  # (the microphone is open only while the button is held)
    end = out["end"]
    assert end["phase"] == "idle" and end["tone"] == "heard"
    assert end["line"].startswith('page.voice.latency{"line":"page.voice.heard{\\"text\\":\\"坐下\\",\\"meaning\\":\\"page.cmd.verb.sit\\"}"')
    assert out["shown"]["commands"] == [{"answer": answer, "verb": "sit"}] and out["shown"]["feedback"] == []


@needs_node
@pytest.mark.parametrize(("kind", "tone"), [("praise", "up"), ("scold", "down")])
def test_a_spoken_praise_or_scold_goes_to_the_feedback_line(kind, tone) -> None:
    answer = {"ok": True, "kind": kind, "trick": "spin", "proficiency": 0.2}
    out = _run("feedback", {"answers": [_answer("乖", {"kind": "feedback", "feedback": kind}, answer)]})
    assert out["end"]["tone"] == tone and f"page.trick.fb.{kind}" in out["end"]["line"]
    assert out["shown"]["feedback"] == [{"kind": kind, "status": 200, "answer": answer}]
    assert out["shown"]["commands"] == []


@needs_node
def test_a_praise_with_no_trick_lately_is_shown_as_the_feedback_buttons_show_it() -> None:
    answer = {"ok": False, "reason": "no_trick", "error": "最近沒有做把戲"}
    out = _run("feedback", {"answers": [_answer("乖", {"kind": "feedback", "feedback": "praise"}, answer)]})
    assert out["shown"]["feedback"] == [{"kind": "praise", "status": 400, "answer": answer}]


@needs_node
def test_space_held_on_the_focused_button_is_push_to_talk_too() -> None:
    out = _run("keys", {"answers": [_answer("過來", {"kind": "command", "verb": "come"}, {"ok": True, "verb": "come", "state": "queued"})]})
    assert out["held"]["phase"] == "listening" and len(out["posts"]) == 1
    assert out["shown"]["commands"][0]["verb"] == "come"


@needs_node
def test_a_tap_or_a_release_before_the_microphone_opened_sends_nothing() -> None:
    for scenario in ("short", "tap"):
        out = _run(scenario)
        assert out["posts"] == [], scenario
        assert out["end"]["line"] == "page.voice.too_short" and out["end"]["phase"] == "idle", scenario
        assert out["mic"]["opened"] == out["mic"]["closed"] == 1, scenario


@needs_node
def test_a_long_hold_is_cut_at_the_limit_and_still_sent() -> None:
    out = _run("cap", {"seconds": 12, "answers": [_answer("坐下", {"kind": "command", "verb": "sit"}, {"ok": True, "verb": "sit", "state": "queued"})]})
    (post,) = out["posts"]
    assert post["bytes"] <= 2 * RATE * 10 and out["mic"]["closed"] == 1


@needs_node
@pytest.mark.parametrize(("fail", "key"), [("denied", "page.voice.err.denied"), ("insecure", "page.voice.err.insecure"), ("nodevice", "page.voice.err.nodevice"), ("failed", "page.voice.err.mic_failed")])
def test_a_microphone_the_browser_will_not_open_says_why(fail, key) -> None:
    out = _run("denied", {"mic": {"fail": fail}})
    assert out["end"]["line"] == key and out["end"]["tone"] == "error" and out["posts"] == []


@needs_node
@pytest.mark.parametrize(
    ("status", "title"),
    [
        ({"ok": True, "enabled": False, "running": True, "engine": None, "problem": "ImportError"}, "page.voice.missing.title"),
        ({"ok": True, "enabled": False, "running": False, "engine": "mlx", "problem": None}, "page.voice.off.title"),
    ],
)
def test_the_button_is_off_and_says_why_without_a_recogniser_or_a_brain(status, title) -> None:
    out = _run("off", {"status": status})
    assert out["start"]["disabled"] is True and out["start"]["title"] == title
    assert out["start"]["titleHook"] == f"title:{title}"  # (a language switch says the same reason)


@needs_node
def test_every_answer_is_said_in_plain_words() -> None:
    cases = [
        [200, _answer("", {"kind": "none"}, None)["data"], 900],
        [200, {**_answer("", {"kind": "none"}, None)["data"], "heard": {"text": "", "empty": "too_short"}}, 300],
        [200, _answer("今天天氣很好", {"kind": "none", "reason": "not_understood"}, None)["data"], 800],
        [200, _answer("轉圈", {"kind": "command", "verb": "spin"}, {"ok": False, "state": "refused"})["data"], 800],
        [503, {"ok": False, "reason": "not_running"}, 50],
        [503, {"ok": False, "reason": "no_recognizer"}, 50],
        [429, {"ok": False, "reason": "busy"}, 50],
        [0, None, 50],
    ]
    lines = _run("lines", {"cases": cases})["lines"]
    keys = [line["key"] for line in lines]
    assert keys == ["page.voice.nothing", "page.voice.too_short", "page.voice.heard", "page.voice.heard",
                    "page.voice.err.not_running", "page.voice.err.no_recognizer", "page.voice.err.busy",
                    "page.voice.err.failed"]  # fmt: skip
    assert lines[2]["meaningKey"] == "page.voice.meaning.none" and lines[3]["tone"] == "refused"
    assert lines[0]["secs"] == 0.9 and lines[4]["secs"] is None
