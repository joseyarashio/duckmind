"""Live voice, LV1: the scenarios (``tools/live_voice/scenarios.json``, ``duckmind.live.scenario``):
the file, the microphone frames made from it, the cassette key's inputs, and every scenario run
through the fake Gemini. No network, no physics: the world is the stand-in of the module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.live import scenario as S
from duckmind.live.cassette import Cassette, CassetteMismatch, scenario_hash
from duckmind.live.fake import FakeGemini
from duckmind.live.tools import BANK_TAGS
from duckmind.simweb import voice as V

pytestmark = pytest.mark.usefixtures("no_network")

FILE = S.load_scenario_file()
LIB = S.ScenarioRun(FILE, "call_name", "explorer", 7).library()
TOY, EMPTY = "eye_toy", "eye_empty"


@pytest.fixture(scope="module")
def eye_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two tiny pictures standing in for the duck's eye (the real ones come from a physics run)."""
    from duckmind.simweb import jpeg

    folder = tmp_path_factory.mktemp("eye")
    for k, name in enumerate((TOY, EMPTY)):
        rgb = np.random.default_rng(k).integers(0, 255, size=(16, 24, 3), dtype=np.uint8)
        (folder / f"{name}.jpg").write_bytes(jpeg.encode_numpy(rgb, 80))
    return folder


def run_of(
    sid: str, tid: str = "explorer", seed: int = 7, eye: Path | None = None, **kw: Any
) -> S.ScenarioRun:
    return S.ScenarioRun(FILE, sid, tid, seed, eye_dir=eye, require_bank=False, **kw)


# ── the file ───────────────────────────────────────────────────────────────────────


def test_the_file_has_the_scenarios_the_plan_asks_for() -> None:
    ids = set(FILE.scenarios)
    assert {
        "call_name",
        "hello",
        "intro_then_hello_mom",
        "intro_then_hello_kid",
        "other_name",
        "what_do_you_see",
        "what_do_you_see_empty",
        "stop_calling",
        "echo_only",
        "silence",
    } <= ids
    assert set(FILE.temperaments) == {"explorer", "snuggler", "dreamer", "hatchling"}
    assert FILE.seeds == (7, 11)
    assert FILE.duck_name == "蛋黃"


def test_the_recording_matrix_is_the_documented_one() -> None:
    matrix = FILE.matrix()
    assert len(matrix) == len(set(matrix)) == 68
    per: dict[str, int] = {}
    for sid, _tid, _seed in matrix:
        per[sid] = per.get(sid, 0) + 1
    assert per == {
        "silence": 4,
        "call_name": 8,
        "hello": 8,
        "intro_then_hello_mom": 8,
        "intro_then_hello_kid": 8,
        "other_name": 4,
        "what_do_you_see": 8,
        "what_do_you_see_empty": 4,
        "stop_calling": 8,
        "echo_only": 8,
    }
    run_ids = {run_of(*m).run_id for m in matrix}
    assert len(run_ids) == 68 and "call_name__explorer__s7" in run_ids


def test_every_line_picture_and_call_in_the_file_exists_and_fits() -> None:
    for sc in FILE.scenarios.values():
        assert sc.length_s > 0 and sc.tail_s >= 3 and sc.description
        for say in sc.say:
            line = LIB.get(say.utterance)
            assert 0 <= say.t and say.t + line.duration_s < sc.length_s, (sc.id, say)
            assert say.who == line.speaker, (sc.id, "the line's own voice")
        for call in sc.duck_calls:
            assert call.tag in BANK_TAGS and 0 < call.level <= 1 and call.t < sc.length_s
        for span in sc.images:
            assert span.from_s < span.until_s and span.fixture in (TOY, EMPTY)
        assert bool(sc.images) == sc.needs_eye
        assert [e.t for e in sc.context] == sorted(e.t for e in sc.context)


def test_a_scenario_file_with_an_unknown_temperament_is_refused(tmp_path: Path) -> None:
    raw = json.loads(S.SCENARIOS_PATH.read_text(encoding="utf-8"))
    raw["scenarios"][0]["temperaments"] = ["nobody"]
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="nobody"):
        S.load_scenario_file(bad)
    raw = json.loads(S.SCENARIOS_PATH.read_text(encoding="utf-8"))
    raw["version"] = 9
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        S.load_scenario_file(bad)


# ── the key's inputs ───────────────────────────────────────────────────────────────


def test_the_inputs_are_by_reference_and_stable() -> None:
    inputs = run_of("call_name").inputs()
    assert inputs == run_of("call_name").inputs()
    [say] = inputs["say"]
    assert (
        say["utterance"] == "call_name" and len(say["sha256"]) == 64
    )  # (a fixture id and its hash)
    assert inputs["model"] == "gemini-3.8-live"
    assert inputs["tool_schema"]["version"].startswith("tools-")
    assert inputs["soul"]["version"].startswith("soul-") and len(inputs["soul"]["sha256"]) == 64
    assert inputs["duck"]["genome_seed"] == 4107 and inputs["duck"]["voice_seed"] == 936833334
    text = json.dumps(inputs, ensure_ascii=False)
    assert len(text) < 5000  # (no raw audio or video in there)


def test_the_hash_changes_with_the_temperament_the_seed_and_the_scenario() -> None:
    keys = {
        scenario_hash(run_of(sid, tid, seed).inputs())
        for sid in ("call_name", "hello")
        for tid in FILE.temperaments
        for seed in (7, 11)
    }
    assert len(keys) == 16


def test_a_moved_speaker_changes_the_hash(tmp_path: Path) -> None:
    raw = json.loads(S.SCENARIOS_PATH.read_text(encoding="utf-8"))
    base = scenario_hash(run_of("call_name").inputs())
    for scenario in raw["scenarios"]:
        if scenario["id"] == "call_name":
            scenario["say"][0]["at_m"] = [2.5, 0.5]
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(raw), encoding="utf-8")
    other = S.ScenarioRun(
        S.load_scenario_file(moved), "call_name", "explorer", 7, require_bank=False
    )
    assert scenario_hash(other.inputs()) != base


def test_each_ducks_soul_is_its_own() -> None:
    texts = {tid: run_of("call_name", tid).soul_text() for tid in FILE.temperaments}
    assert len(set(texts.values())) == 4
    assert all("蛋黃" in t for t in texts.values())


def test_a_voice_is_quieter_the_farther_it_is() -> None:
    run = run_of("call_name")
    near = run.say_geometry(S.SayEvent(0, "call_name", "mom", (1.0, 0.0)))
    far = run.say_geometry(S.SayEvent(0, "call_name", "mom", (4.0, 0.0)))
    assert near["kind"] == "name" and near["db"] > far["db"]
    assert near["distance_m"] < far["distance_m"]
    hello = run.say_geometry(S.SayEvent(0, "hello", "mom", (2.0, 0.0)))
    other = run.say_geometry(S.SayEvent(0, "other_name", "mom", (2.0, 0.0)))
    assert (
        hello["kind"] == "hello" and other["kind"] == "voice"
    )  # (another name is not a name call)


# ── the frames ─────────────────────────────────────────────────────────────────────


def digest(run: S.ScenarioRun) -> str:
    h = hashlib.sha256()
    for step in run.steps():
        h.update(step.frame.pcm)
    return h.hexdigest()


def test_the_microphone_frames_are_deterministic_and_follow_the_seed() -> None:
    a = digest(run_of("call_name", seed=7))
    assert a == digest(run_of("call_name", seed=7))
    assert a != digest(run_of("call_name", seed=11))


def test_the_frames_are_20_ms_of_16k_mono_for_the_whole_scenario() -> None:
    run = run_of("hello")
    steps = list(run.steps())
    assert len(steps) == int(round(run.scenario.total_s / 0.02)) == 650
    assert all(len(s.frame.pcm) == 640 for s in steps)
    assert [round(s.t, 2) for s in steps[:3]] == [0.0, 0.02, 0.04]


def test_speech_is_loud_where_it_is_said_and_the_room_is_quiet_elsewhere() -> None:
    steps = list(run_of("call_name").steps())
    quiet = [s.frame.level_dbfs for s in steps if s.t < 1.9]
    spoken = [s.frame.level_dbfs for s in steps if 2.1 <= s.t <= 3.8]
    assert max(quiet) < -55 and not any(s.frame.echo for s in steps)
    assert max(spoken) > -35 and sum(v > -50 for v in spoken) > 0.4 * len(spoken)


def test_the_ducks_own_calls_are_flagged_where_it_calls() -> None:
    run = run_of("echo_only")
    flagged = [s.t for s in run.steps() if s.frame.echo]
    assert flagged
    for call in run.scenario.duck_calls:  # (a window opens at each call)
        assert any(call.t <= t < call.t + 0.8 for t in flagged), call
    assert not any(t < 1.99 for t in flagged)


@pytest.mark.skipif(
    not V.VoiceBank(None).ready(V.voice_seed(4107)),
    reason="the explorer's voice bank is not rendered on this machine",
)
def test_with_the_voice_bank_the_echo_is_loud_in_the_microphone() -> None:
    run = S.ScenarioRun(FILE, "echo_only", "explorer", 7)  # (require_bank: it must be there)
    echo = [s.frame.level_dbfs for s in run.steps() if s.frame.echo]
    assert max(echo) > -40  # (the speaker is a few cm from the microphone)


def test_a_scenario_that_needs_the_bank_says_so_when_it_is_not_there() -> None:
    absent = SimpleNamespace(ready=lambda seed: False)
    run = S.ScenarioRun(FILE, "echo_only", "explorer", 7, bank=absent)
    with pytest.raises(S.FixtureMissing, match="voice bank"):
        list(run.steps())
    # (and without the requirement the windows still say when the duck voiced)
    lenient = S.ScenarioRun(FILE, "echo_only", "explorer", 7, bank=absent, require_bank=False)
    assert any(s.frame.echo for s in lenient.steps())


def test_the_pictures_come_once_a_second_from_the_fixture(eye_dir: Path) -> None:
    run = run_of("what_do_you_see", eye=eye_dir)
    steps = list(run.steps())
    pics = [(round(s.t, 2), name) for s in steps for name, _jpeg in s.images]
    assert [t for t, _ in pics] == [0.5 + k for k in range(15)]
    assert {n for _, n in pics} == {TOY}
    assert all(jpeg[:2] == b"\xff\xd8" for s in steps for _n, jpeg in s.images)
    digest_of_file = hashlib.sha256((eye_dir / f"{TOY}.jpg").read_bytes()).hexdigest()
    assert run.inputs()["images"][0]["sha256"] == digest_of_file


def test_a_missing_picture_is_a_clear_error(tmp_path: Path) -> None:
    run = run_of("what_do_you_see", eye=tmp_path)
    with pytest.raises(S.FixtureMissing, match="eye_toy"):
        list(run.steps())


def test_the_status_text_changes_come_at_their_times() -> None:
    run = run_of("intro_then_hello_mom")
    ctx = [(round(s.t, 2), s.context) for s in run.steps() if s.context is not None]
    assert [t for t, _ in ctx] == [0.0, 8.0]
    first, later = ctx[0][1], ctx[1][1]
    assert first is not None and later is not None and later.known_people == ("媽媽",)
    assert "媽媽" in run.snapshot(later).text() and "媽媽" not in run.snapshot(first).text()


# ── through the fake ───────────────────────────────────────────────────────────────


def through_fake(run: S.ScenarioRun, **kw: Any) -> S.ScenarioResult:
    labels = run.image_labels() if run.scenario.images else {}
    return S.run_scenario(run, FakeGemini(lines=run.spoken_lines(), image_labels=labels, **kw))


def names(result: S.ScenarioResult) -> list[str]:
    return [c.name for c in result.calls]


@pytest.mark.parametrize("tid", list(FILE.temperaments))
def test_the_expected_reactions(tid: str, eye_dir: Path) -> None:
    assert names(through_fake(run_of("call_name", tid))) == ["vocalize", "feel"]
    assert names(through_fake(run_of("hello", tid))) == ["vocalize", "feel"]  # (嗨，小鴨！)
    assert names(through_fake(run_of("silence", tid))) == []
    assert names(through_fake(run_of("other_name", tid))) == []  # (somebody else's name)
    assert names(through_fake(run_of("stop_calling", tid))) == ["feel"]
    assert names(through_fake(run_of("echo_only", tid))) == []
    mom = through_fake(run_of("intro_then_hello_mom", tid))
    assert names(mom) == ["recognize", "remember", "vocalize", "attend"]
    assert mom.calls[0].args == {"name": "媽媽"} and mom.calls[0].t < 6 < 12 < mom.calls[2].t
    kid = through_fake(run_of("intro_then_hello_kid", tid))
    assert names(kid) == [
        "recognize",
        "remember",
        "vocalize",
        "feel",
    ]  # (another voice, another greeting)
    toy = through_fake(run_of("what_do_you_see", tid, eye=eye_dir))
    assert [(c.name, c.args) for c in toy.calls] == [
        ("attend", {"target_kind": "thing", "hint": "玩具"})
    ]
    empty = through_fake(run_of("what_do_you_see_empty", tid, eye=eye_dir))
    assert [c.name for c in empty.calls] == ["feel"]  # (nothing in view: no invented thing)


@pytest.mark.skipif(
    not V.VoiceBank(None).ready(V.voice_seed(4107)),
    reason="the explorer's voice bank is not rendered on this machine (its calls are the echo)",
)
def test_the_ducks_own_calls_that_the_model_mistakes_for_speech_are_all_marked() -> None:
    result = through_fake(run_of("echo_only"), hallucinate=True)
    assert result.calls and all(c.echo_suspect for c in result.calls)
    assert all(e["echo_suspect"] for e in result.log("heard"))


def test_the_barge_in_transcript_overlaps_the_ducks_voice_and_is_marked() -> None:
    result = through_fake(run_of("stop_calling"))
    [feel] = result.calls
    assert not feel.echo_suspect  # (the call answers a person: the speech is recent)
    heard = result.log("heard")
    assert heard and all(e["echo_suspect"] for e in heard)  # (it overlaps the duck's own call)


def test_the_whole_matrix_runs_through_the_fake_without_a_fault(eye_dir: Path) -> None:
    for sid, tid, seed in FILE.matrix():
        result = through_fake(run_of(sid, tid, seed, eye=eye_dir))
        status = result.session.status()
        assert status["faults"] == 0 and status["state"] == "off", (sid, tid, seed)
        assert status["calls"]["total"] == len(result.calls)
        assert result.frames >= 550


def test_a_recording_through_the_fake_replays_to_the_same_calls(tmp_path: Path) -> None:
    run = run_of("intro_then_hello_mom")
    cassette, original = S.record_scenario(
        run, FakeGemini(lines=run.spoken_lines()), S.ManualClock(), meta={"recorded_with": "fake"}
    )
    path = S.cassette_path(run.run_id, tmp_path)
    assert path.name == "intro_then_hello_mom__explorer__s7.json"
    cassette.save(path)
    again = S.replay(run, Cassette.load(path))
    got = [(c.name, c.args, round(c.t, 3)) for c in again.calls]
    assert got == [(c.name, c.args, round(c.t, 3)) for c in original.calls] and got
    assert cassette.complete and cassette.meta["run_id"] == run.run_id


def test_replaying_a_cassette_of_another_scenario_fails_loudly() -> None:
    run = run_of("call_name")
    cassette, _ = S.record_scenario(run, FakeGemini(lines=run.spoken_lines()), S.ManualClock())
    with pytest.raises(CassetteMismatch):
        S.replay(run_of("call_name", seed=11), cassette)
    with pytest.raises(CassetteMismatch):
        S.replay(run_of("call_name", "dreamer"), cassette)


def test_the_realtime_clock_paces_by_the_wall(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    slept: list[float] = []
    clock = S.RealtimeClock(lambda: now[0])
    now[0] = 100.5
    assert abs(clock() - 0.5) < 1e-9
    monkeypatch.setattr(S.time, "sleep", slept.append)
    clock.wait_until(2.0)
    clock.wait_until(0.1)  # (already past: no wait)
    assert slept == [pytest.approx(1.5)]
