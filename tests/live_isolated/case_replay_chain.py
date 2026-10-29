"""Live voice, LV2: every real recording (``tests/fixtures/live``, 68 Gemini Live sessions) through
the whole chain on a fake clock: session → grounding gate → bridge → a real integrated brain, with
the duck's own voice detection on the scenario's microphone frames. No network, no physics.

What is pinned down here is the chain, not the model's choices: no call of a refused category ever
reaches the selector (also with refused calls injected into real recordings), the ``echo_only``
recordings produce no accepted act at all, an introduction binds the name to the person the duck
itself perceives, a misheard name makes nobody, and a person speaking over the duck's call stops it
within the bound. The scene the duck perceives is the scripted one of ``duckmind.live.chain``."""

from __future__ import annotations

from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from duckmind.live import chain as CH
from duckmind.live import gate as G
from duckmind.live import scenario as S
from duckmind.live.chain import Injected
from duckmind.live.gate import DEFAULT_ALLOWED
from duckmind.simweb import voice as V

pytestmark = pytest.mark.usefixtures("no_network")

FILE = S.load_scenario_file()
FOLDER = S.CASSETTE_DIR
CASSETTES = sorted(p for p in FOLDER.glob("*.json") if p.name != "RECORDING.json")
HAVE_BANK = all(
    V.VoiceBank(None, V.DEFAULT_ROOT).ready(V.voice_seed(t.genome_seed))
    for t in FILE.temperaments.values()
)
needs_bank = pytest.mark.skipif(
    not HAVE_BANK, reason="the ducks' voice banks are not rendered here"
)
BOUND_S = 0.3  # the barge-in target: the stop within 300 ms of the speech onset
ONSET_TOL_S = 0.04


def parts(path: Path) -> tuple[str, str, int]:
    sid, tid, seed = path.stem.rsplit("__", 2)
    return sid, tid, int(seed.lstrip("s"))


@cache
def chain(sid: str, tid: str, seed: int) -> CH.ChainResult:
    return CH.run_chain(FILE, sid, tid, seed)


def of(sid: str) -> list[CH.ChainResult]:
    return [chain(s, t, seed) for s, t, seed in FILE.matrix() if s == sid]


def brain_vocals(result: CH.ChainResult) -> list[float]:
    return [c.t_ms / 1000.0 for c in result.cmds if c.vocal]


# ── all 68 run through, cleanly ────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", CASSETTES, ids=[p.stem for p in CASSETTES])
def test_a_recording_runs_through_the_whole_chain_without_a_fault(path: Path) -> None:
    result = chain(*parts(path))
    assert result.bridge.faults == 0 and result.session.faults == 0
    assert result.ticks > 20 * result.run.scenario.total_s
    view = result.bridge.view()
    assert view["faults"] == 0 and view["gate"]["accepted"] + view["gate"]["refused_calls"] == len(
        result.bridge.seen
    )
    # every call the model made was judged, and only what the gate let through was acted on
    made = [e for e in result.session.log if e.get("kind") == "call"]
    assert len(made) == len(result.bridge.seen)


@pytest.mark.parametrize("path", CASSETTES, ids=[p.stem for p in CASSETTES])
def test_no_call_of_a_refused_category_reaches_the_selector(path: Path) -> None:
    result = chain(*parts(path))
    bridge = result.bridge
    verdicts = {e["seq"]: e for e in bridge.gate.log}
    calls = {c.seq: c for c in bridge.seen}
    entered_seqs = {e["seq"] for e in bridge.entered}
    for seq in entered_seqs:  # everything that was given to the selector was accepted at the gate
        assert verdicts[seq]["accepted"], seq
        assert verdicts[seq]["tool"] in ("vocalize", "attend")
        call = calls[seq]
        echoed = call.echo_suspect or call.mark.cause_echo
        assert not echoed or verdicts[seq]["rescued"]  # (an echo-marked call only by local proof)
        for tag in verdicts[seq]["tags"]:
            assert tag["tag"] in DEFAULT_ALLOWED
    for seq, verdict in verdicts.items():
        if not verdict["accepted"]:
            assert seq not in entered_seqs
    # what the selector itself was ever given from outside (a tap on its input) came only from
    # those candidates, and not before the first of them was accepted
    channels = {c for e in bridge.entered for c in e["channels"]}
    first = min((e["t"] for e in bridge.entered), default=None)
    for t, inputs in result.taps:
        assert first is not None and t >= first - 1e-6 and set(inputs) <= channels
    if not bridge.entered:
        assert result.taps == [] and bridge.voiced == []


# ── echo only ──────────────────────────────────────────────────────────────────────────


@needs_bank
def test_without_the_local_detection_the_gate_is_strict_with_it_a_person_is_believed() -> None:
    """The model's `feel` after 不要叫了 follows the duck's own voice as much as the person's: by the
    echo marks alone all eight calls of the recordings are refused. The duck's own voice detection
    heard the person, and all eight are believed; none came before the person began at 2.2 s (an
    earlier recording had one, the model's `feel` on hearing the duck's own call: refused, rightly)."""
    strict_accepted = strict_refused = rescued = early = 0
    for sid, tid, seed in FILE.matrix():
        if sid != "stop_calling":
            continue
        plain = CH.run_chain(FILE, sid, tid, seed, use_vad=False)
        strict_accepted += plain.bridge.gate.status()["accepted"]
        strict_refused += plain.bridge.gate.status()["refused_calls"]
        assert all(
            e["reason"] in (G.ECHO_ONLY, G.ECHO_CAUSE)
            for e in plain.bridge.gate.log
            if not e["accepted"]
        )
        with_ears = chain(sid, tid, seed)
        rescued += sum(1 for e in with_ears.bridge.gate.log if e["accepted"] and e["rescued"])
        said = with_ears.run.scenario.say[0]
        # refused even with the duck's ears: only an echo call made before the person's line began
        refused_with_ears = [e for e in with_ears.bridge.gate.log if not e["accepted"]]
        assert all(e["reason"] == G.ECHO_ONLY and e["now"] < said.t for e in refused_with_ears), (
            tid,
            seed,
        )
        early += len(refused_with_ears)
        assert with_ears.bridge.gate.status()["accepted"] >= 1, (tid, seed)  # (the person is heard)
    assert (strict_accepted, strict_refused, rescued, early) == (0, 8, 8, 0)


@needs_bank
def test_the_echo_only_recordings_produce_no_accepted_act() -> None:
    results = of("echo_only")
    assert len(results) == 8
    refused = made = 0
    for r in results:
        gate = r.bridge.gate
        assert gate.status()["accepted"] == 0, r.run_id
        assert r.bridge.entered == [] and r.bridge.voiced == [] and r.bridge.nudges == []
        assert r.brain.learn.people == {} and r.brain.learn.notes == []
        assert (
            r.bridge.barges == [] and r.bridge.vad is not None and r.bridge.vad.voiced_frames == 0
        )
        refused += gate.refusals[G.ECHO_ONLY]
        assert set(gate.refusals) <= {G.ECHO_ONLY, G.ECHO_CAUSE}
        made += len(r.bridge.seen)
    # soul-1's model made three `feel` calls on hearing the duck itself, soul-2's none; the recordings
    # of 2026-10-29 made two (a `feel` in dreamer s7; a `vocalize` in hatchling s11 after it heard
    # "喂。" in the duck's own call): all refused as the duck's own voice. More such proposals are
    # injected just below.
    assert (made, refused) == (2, 2)


@needs_bank
def test_a_proposal_to_call_after_the_duck_heard_only_itself_is_refused_every_time() -> None:
    """The real model never proposed a call there (no echo loop): put those proposals in."""
    asked = [
        Injected(t, "vocalize", {"tags": [{"tag": "greet", "level": 0.6, "at_ms": 0}]})
        for t in (4.6, 6.4, 8.0)
    ]
    for sid, tid, seed in FILE.matrix():
        if sid != "echo_only":
            continue
        r = CH.run_chain(FILE, sid, tid, seed, inject=asked)
        assert r.bridge.entered == [] and r.bridge.voiced == []
        reasons = Counter(
            x["reason"] for e in r.bridge.gate.log if not e["accepted"] for x in e["refused"]
        )
        assert reasons[G.ECHO_ONLY] >= 3, (r.run_id, reasons)


# ── calls outside the bank, and the rest of what the gate refuses, injected ───────────


HOSTILE = [
    ("vocalize", {"tags": [{"tag": "quack", "level": 0.5, "at_ms": 0}]}),
    ("vocalize", {"tags": [{"tag": "alarm", "level": 1.0, "at_ms": 0}]}),
    ("vocalize", {"tags": [{"tag": "wheee", "level": 1.0, "at_ms": 0}]}),
    ("vocalize", {"tags": [{"tag": "peck", "level": 1.0, "at_ms": 0}]}),
    ("vocalize", {"tags": [{"tag": "greet", "level": "very", "at_ms": 0}]}),
    ("vocalize", {"tags": []}),
    ("move_servo", {"angle": 90}),
]


@pytest.mark.parametrize("sid", ["call_name", "hello", "intro_then_hello_mom", "what_do_you_see"])
def test_calls_outside_the_bank_are_blocked_every_time_in_real_recordings(sid: str) -> None:
    asked = [Injected(5.0 + 0.1 * k, n, a, id=f"x{k}") for k, (n, a) in enumerate(HOSTILE)]
    for s, tid, seed in FILE.matrix():
        if s != sid:
            continue
        r = CH.run_chain(FILE, s, tid, seed, inject=asked)
        injected = [
            e
            for e in r.bridge.gate.log
            if e["seq"] in {c.seq for c in r.bridge.seen if c.id.startswith("x")}
        ]
        assert len(injected) == len(HOSTILE)
        assert not any(e["accepted"] for e in injected), r.run_id  # (100 % blocked)
        reasons = Counter(x["reason"] for e in injected for x in e["refused"])
        assert reasons[G.TAG_UNKNOWN] == 1 and reasons[G.TAG_RESERVED] == 2
        assert reasons[G.TAG_NOT_VOICEABLE] == 1 and reasons[G.MALFORMED] == 2
        assert reasons[G.UNKNOWN_TOOL] == 1
        # whatever the real model asked for, nothing off the allowed list was ever voiced
        assert all(v["tag"] in DEFAULT_ALLOWED for v in r.bridge.voiced)
        assert all(t[0] in DEFAULT_ALLOWED for e in r.bridge.entered for t in e.get("tags", []))


def test_a_burst_of_calls_cannot_make_the_duck_call_without_end() -> None:
    asked = [
        Injected(
            5.0 + 0.2 * k,
            "vocalize",
            {"tags": [{"tag": "chirp", "level": 0.5, "at_ms": 0}]},
            id=f"x{k}",
        )
        for k in range(12)
    ]
    r = CH.run_chain(FILE, "hello", "hatchling", 7, inject=asked)
    cfg = r.bridge.gate.config
    taken = [  # (when each accepted tag was accepted)
        e["now"]
        for e in r.bridge.gate.log
        if e["accepted"] and e["tool"] == "vocalize"
        for _ in e["tags"]
    ]
    assert taken and len(taken) <= len(asked) + 1
    for t0 in taken:  # in no window of ten seconds more than the cap
        assert sum(1 for t in taken if t0 <= t < t0 + cfg.rate_window_s) <= cfg.rate_max_tags
    assert (
        r.bridge.gate.refusals[G.RATE_CAP] + r.bridge.gate.refusals[G.UTTERANCE_CAP]
        >= 12 - len(taken) + 1
    )
    assert len(r.bridge.voiced) <= cfg.rate_max_tags


@pytest.mark.parametrize("sid", ["call_name", "intro_then_hello_mom"])
def test_with_the_cloud_switched_off_when_the_calls_arrive_none_is_acted_on(sid: str) -> None:
    judged = 0
    for s, tid, seed in FILE.matrix():
        if s != sid:
            continue
        r = CH.run_chain(FILE, s, tid, seed, cloud_off_at_first_call=True)
        assert (
            r.bridge.entered == [] and r.bridge.voiced == [] and r.brain.learn.person_names() == {}
        )
        gate = r.bridge.gate
        assert gate.status()["accepted"] == 0
        assert gate.refusals[G.CLOUD_OFF] == len(r.bridge.seen)  # (each one refused for that)
        judged += len(r.bridge.seen)
    assert judged >= 7  # (the real model made calls in nearly every one of the eight)


# ── introductions, and a name that was misheard ───────────────────────────────────────


@pytest.mark.parametrize("sid", ["intro_then_hello_mom", "intro_then_hello_kid"])
def test_an_introduction_binds_the_name_to_the_person_the_duck_perceives(sid: str) -> None:
    mom = CH.LOOKS["mom"]
    for r in of(sid):
        learner = r.brain.learn
        recognized = [
            e for e in r.bridge.log if e["tool"] == "recognize" and e["decision"] == "accepted"
        ]
        assert recognized, r.run_id  # (every real model made the call after "這是媽媽。")
        named = [p for p in learner.people.values() if p.name]
        assert [p.name for p in named] == ["媽媽"], r.run_id  # (妈妈 and 媽媽: one person)
        person = named[0]
        assert person.name_raw in ("媽媽", "妈妈")
        # it is the person in front of the duck, by the look its own eye gave
        assert all(abs(a - b) < 1e-6 for a, b in zip(person.mean[:4], mom[:4], strict=True))
        assert person.bound_t is not None and 3.0 < person.bound_t < 7.0
        assert len(learner.people) == (1 if sid.endswith("mom") else 2)
        if sid.endswith("kid"):  # (the child who comes later is somebody it does not know)
            other = next(p for p in learner.people.values() if p.name is None)
            assert other.pending == {} and other.mean[0] != person.mean[0]
        assert r.bridge.faults == 0


def test_the_greeting_for_someone_it_knows_is_not_the_one_for_a_stranger() -> None:
    def replies(sid: str) -> list[dict[str, Any]]:
        out = []
        for r in of(sid):
            later = [e for e in r.bridge.entered if e["tool"] == "vocalize" and e["t"] > 9.0]
            out += later
        return out

    mom, kid = replies("intro_then_hello_mom"), replies("intro_then_hello_kid")
    assert len(mom) == len(kid) == 8
    assert all(e["known"] and e["addressee"] is not None for e in mom)  # (媽媽: it knows her)
    assert not any(e["known"] for e in kid)  # (the child: a stranger)
    # and the stranger gets a smaller push into the competition: the same call, a little more reserve
    brain = of("intro_then_hello_mom")[0].brain
    bridge = of("intro_then_hello_mom")[0].bridge
    known = chain_intent(bridge, True)
    stranger = chain_intent(bridge, False)
    assert bridge._weight(known, brain) > bridge._weight(stranger, brain)


def chain_intent(bridge: Any, known: bool) -> Any:
    from duckmind.live.bridge import Intent

    return Intent(1, "reply", 1, 0.0, 5.0, ("greet",), level=0.5, addressee=1, known=known)


@pytest.mark.parametrize("who", ["郑华", "鄭華", "蛋黄", "蛋黃"])
def test_a_misheard_name_does_not_make_a_new_person(who: str) -> None:
    """The model heard 蛋黃 as 郑华 in all 8 call_name recordings. Here it also calls recognize on it
    (it never did): mom is in view and was just heard speaking, and nobody becomes 郑华."""
    asked = [Injected(4.2, "recognize", {"name": who}, id="x0")]
    for s, tid, seed in FILE.matrix():
        if s != "call_name":
            continue
        r = CH.run_chain(FILE, s, tid, seed, inject=asked)
        learner = r.brain.learn
        assert learner.person_names() == {}, r.run_id
        assert len(learner.people) == 1  # (only the person it sees: no second one from a name)
        person = next(iter(learner.people.values()))
        assert person.pending == {} and person.name is None
        entry = next(e for e in r.bridge.log if e["tool"] == "recognize")
        assert entry["decision"] == "refused" and entry["reason"] in (
            "duck_name",
            "sounds_like_duck_name",
        )


def test_a_name_for_nobody_in_view_is_not_taken_for_a_person() -> None:
    asked = [Injected(4.5, "recognize", {"name": "媽媽"}, id="x0")]
    r = CH.run_chain(FILE, "echo_only", "snuggler", 7, inject=asked)  # (nobody there at all)
    assert r.brain.learn.people == {} and r.bridge.entered == []
    assert r.bridge.gate.refusals[G.ECHO_ONLY] + r.bridge.reasons["no_person_in_view"] >= 1


# ── what it looks at ───────────────────────────────────────────────────────────────────


def test_attention_lands_only_on_what_the_duck_itself_perceives() -> None:
    seen = of("what_do_you_see")
    assert len(seen) == 8
    for r in seen:
        attends = [e for e in r.bridge.entered if e["tool"] == "attend"]
        assert attends and all(
            e["target"]["kind"] == "thing" and e["target"]["track"] == 7 for e in attends
        )
    empty = of("what_do_you_see_empty")
    assert len(empty) == 4
    for r in empty:  # (the model pointed at far things that are in the picture: the duck sees none)
        assert r.bridge.entered == []
        assert [e["reason"] for e in r.bridge.log if e["tool"] == "attend"] == ["nothing_matches"]
    kid = [r for r in of("intro_then_hello_kid")]
    reasons = Counter(
        e["reason"]
        for r in kid
        for e in r.bridge.log
        if e["tool"] == "attend" and e["decision"] == "refused"
    )
    assert set(reasons) <= {"not_in_view"}  # (asked to look at 媽媽 once she had gone)


# ── a person speaks over the duck's call ───────────────────────────────────────────────


@needs_bank
def test_a_person_who_speaks_over_the_call_stops_it_within_the_bound_in_every_recording() -> None:
    results = of("stop_calling")
    assert len(results) == 8
    totals: list[float] = []
    for r in results:
        said = r.run.scenario.say[0]
        assert len(r.bridge.barges) == 1, r.run_id
        b = r.bridge.barges[0]
        assert b.stopped and abs(b.onset_t - said.t) <= ONSET_TOL_S
        assert b.detect_s <= 0.1 and b.apply_s <= 0.1
        assert b.total_s <= BOUND_S
        totals.append(b.total_s)
        # the stop is in the command the body got: a new sequence number with no call
        stops = [c for c in r.cmds if c.vocal is None and c.vocal_seq > 0]
        assert stops, r.run_id
    assert max(totals) <= BOUND_S


@needs_bank
def test_speech_with_no_call_of_the_ducks_to_stop_is_not_a_barge_in() -> None:
    for sid in ("call_name", "hello", "other_name", "what_do_you_see", "silence"):
        for r in of(sid):
            vocals = brain_vocals(r)
            for b in r.bridge.barges:  # a barge-in only where the duck was calling itself
                assert any(0.0 <= b.onset_t - v <= 1.0 for v in vocals), (r.run_id, b.view())
    for r in of("silence") + of("echo_only"):
        assert r.bridge.barges == []


@needs_bank
def test_the_closed_loop_stop_cuts_the_ducks_own_voice_in_the_microphone() -> None:
    lat: list[float] = []
    for tid in FILE.temperaments:
        for offset in (0.1, 0.2, 0.3, 0.4):
            trial, bridge, mic = CH.closed_loop_barge(FILE, tid, speak_offset_s=offset)
            assert trial.stopped and trial.stop_at is not None and trial.latency_s is not None, (
                trial
            )
            assert trial.latency_s <= BOUND_S, trial
            assert trial.echo_end is not None and trial.call_end_natural is not None
            assert trial.echo_end <= trial.stop_at + 0.03  # (the echo ends at the stop)
            assert (
                trial.echo_end < trial.call_end_natural - 0.1
            )  # (well before the call would have)
            lat.append(trial.latency_s)
    assert len(lat) == 16 and max(lat) <= BOUND_S


@needs_bank
def test_the_ducks_own_call_alone_does_not_stop_itself_in_the_closed_loop() -> None:
    for tid in FILE.temperaments:
        trial, bridge, mic = CH.closed_loop_barge(FILE, tid, speak_offset_s=None)
        assert trial.call_at is not None and not trial.stopped and trial.stop_at is None
        assert bridge.barges == [] and bridge.vad is not None and bridge.vad.voiced_frames == 0
        window = next(w for w in mic.vocal_windows() if w.t0 >= trial.call_at)  # (the cloud's call)
        assert window.tag == "coo" and window.t1 is not None and window.t1 - window.t0 >= 0.4
