"""Live voice, LV5: the talk panel's status over the whole chain with a real integrated brain.

The recordings of LV1 (real Gemini Live sessions) are replayed through session → grounding gate → bridge →
a real brain (``duckmind.live.chain``), and the panel's server side reads the result the way the page
would: what the cloud heard, each proposal and what the duck made of it (with a reason in plain words in
six languages when it did not), the people it learnt, the usage. Run in a child process (the default
skill tables are permanent for a process): see ``tests/test_live_page_isolated.py``. No network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.i18n import CODES, boundary, render
from duckmind.live import chain as CH
from duckmind.live import scenario as S
from duckmind.live.bridge import BridgeConfig
from duckmind.simweb import live_voice as LV
from duckmind.simweb import voice as V

pytestmark = pytest.mark.usefixtures("no_network")

FILE = S.load_scenario_file()
MATRIX = FILE.matrix()


def _voice_over(result: CH.ChainResult) -> LV.LiveVoice:
    """The panel's server side reading the chain's session, bridge and brain."""
    mic = result.mic
    driver = SimpleNamespace(
        lanes=SimpleNamespace(brain=result.brain, sound=SimpleNamespace(mic=mic, engine=None)),
        inject=lambda *a: {},
        _truth=None,
    )
    voice = LV.LiveVoice(
        driver,
        name_of=lambda: FILE.duck_name,
        session_factory=lambda soul, meter, enabled: result.session,
        clock=result.session.clock,
    )
    voice.session, voice.bridge = result.session, result.bridge
    voice._lanes = driver.lanes
    return voice


def _ids(sid: str) -> list[tuple[str, str, int]]:
    return [m for m in MATRIX if m[0] == sid]


def _words_in_six_languages(call: dict[str, Any]) -> None:
    for field in ("tool_words", "outcome_words", "reason_words"):
        words = call.get(field)
        if words is None:
            continue
        wire = call["_i18n"][field]
        shown = {lang: render(wire, lang, strict=True) for lang in CODES}
        assert all(shown.values()), (field, shown)
        assert shown["en"].isascii() or "–" in shown["en"] or "’" in shown["en"], shown["en"]


def test_a_name_call_shows_the_transcript_and_what_the_duck_did() -> None:
    # (LV7: the person in the recordings' scene is one the duck has not been introduced to. LV4's reserve
    # toward such a person, which the default keeps (``reply_urge_stranger`` 0), leaves none of the eight
    # call_name replays voiced, and the bound below needs one: the page's status of a *voiced* reply is
    # what this case reads. So it runs with the stranger given the urge to answer, as before the merge)
    voiced_runs = proposed_runs = 0
    for matrix_id in _ids("call_name"):
        result = CH.run_chain(
            FILE, *matrix_id, config=BridgeConfig(duck_name=FILE.duck_name, reply_urge_stranger=1.0)
        )
        status = boundary.attach(_voice_over(result).status())
        assert status["heard"], "the cloud's transcript of the family's words"
        assert all(isinstance(h["text"], str) and h["text"] for h in status["heard"])
        assert len(status["calls"]) == len(result.bridge.seen), (
            matrix_id
        )  # (the model may make none)
        outcomes = {c["outcome"] for c in status["calls"]}
        assert outcomes <= set(LV.OUTCOMES)
        for call in status["calls"]:
            assert call["tool"] in LV.TOOLS and call["tool_words"] is not None
            if call["outcome"] == "refused":
                assert call["reason"] in LV.REASONS and call["reason_words"] is not None
        voiced = sum(1 for c in status["calls"] if c["outcome"] == "voiced")
        assert voiced == len({v["seq"] for v in result.bridge.voiced}), (
            "voiced means the bridge voiced it"
        )
        assert status["usage"]["totals"]["tool_calls"] == len(result.bridge.seen)
        proposed_runs += 1 if status["calls"] else 0
        voiced_runs += 1 if voiced else 0
    assert proposed_runs >= 5 and voiced_runs >= 1, (proposed_runs, voiced_runs)


@pytest.mark.parametrize("matrix_id", _ids("what_do_you_see_empty"), ids=lambda m: f"{m[1]}-{m[2]}")
def test_a_look_at_nothing_is_refused_in_words(matrix_id: tuple[str, str, int]) -> None:
    result = CH.run_chain(FILE, *matrix_id)
    status = boundary.attach(_voice_over(result).status())
    attends = [c for c in status["calls"] if c["tool"] == "attend"]
    assert attends and all(
        c["outcome"] == "refused" and c["reason"] == "nothing_matches" for c in attends
    )
    for call in attends:
        _words_in_six_languages(call)
        assert call["kind_words"] is not None and call["hint"], (
            "the cloud's own word for it is data"
        )
    assert status["look"]["attended"]["reason"] == "nothing_matches"


def _bank_ready(tid: str) -> bool:
    seed = V.voice_seed(FILE.temperaments[tid].genome_seed)
    return V.VoiceBank(None, V.DEFAULT_ROOT).ready(seed)


@pytest.mark.parametrize("matrix_id", _ids("echo_only"), ids=lambda m: f"{m[1]}-{m[2]}")
def test_the_ducks_own_echo_never_becomes_a_reply_and_the_panel_says_so(
    matrix_id: tuple[str, str, int],
) -> None:
    if not _bank_ready(matrix_id[1]):
        pytest.skip("this duck's voice bank is not rendered here (its own calls are the echo)")
    result = CH.run_chain(FILE, *matrix_id)
    status = boundary.attach(_voice_over(result).status())
    assert not any(c["outcome"] in ("voiced", "attended", "bound") for c in status["calls"])
    refused = [c for c in status["calls"] if c["outcome"] == "refused"]
    assert all(c["reason"] in ("echo_only", "echo_cause") for c in refused)
    for call in refused:
        assert call["echo_suspect"] or call["reason"] == "echo_cause"
        _words_in_six_languages(call)
    assert all(h["echo"] for h in status["heard"]), "what it heard was only the duck's own voice"


@pytest.mark.parametrize(
    "matrix_id", _ids("intro_then_hello_mom")[:4], ids=lambda m: f"{m[1]}-{m[2]}"
)
def test_an_introduction_makes_one_named_person(matrix_id: tuple[str, str, int]) -> None:
    result = CH.run_chain(FILE, *matrix_id)
    status = boundary.attach(_voice_over(result).status())
    named = status["people"]["named"]
    assert len(named) == 1 and named[0]["name"] == "媽媽", (
        "one person, whichever way the model wrote the name"
    )
    assert named[0]["sightings"] >= 1 and named[0]["pending"] == []
    bound = [c for c in status["calls"] if c["tool"] == "recognize"]
    assert bound and all(c["outcome"] == "bound" for c in bound)
    assert status["look"]["in_view"]["people"] >= 0


@pytest.mark.parametrize("matrix_id", _ids("stop_calling")[:4], ids=lambda m: f"{m[1]}-{m[2]}")
def test_a_person_speaking_over_the_duck_is_a_stop_not_a_refusal(
    matrix_id: tuple[str, str, int],
) -> None:
    if not _bank_ready(matrix_id[1]):
        pytest.skip("this duck's voice bank is not rendered here (its own call is what is stopped)")
    result = CH.run_chain(FILE, *matrix_id)
    status = boundary.attach(_voice_over(result).status())
    assert result.bridge.barges, "the duck stopped its own call"
    assert all(c["tool"] in LV.TOOLS for c in status["calls"])
    assert status["gate"]["accepted"] + status["gate"]["refused_calls"] == len(result.bridge.seen)
