"""Live voice, LV2: the local voice detection (``duckmind.live.vad``) on the LV0 microphone's own
frames. A person who speaks over the duck's call is heard within 60 ms of the onset; the duck's own
echo alone, every one of the ``echo_only`` mixes, is never taken for a person; with no reference
for its own voice the detector is deaf while it calls rather than wrong. The same with the motors'
noise (up to 58 dB) and a microphone clock that wanders against the audio the way the live one does
(LV5 found the detector hearing the duck's own calls as a person: its position in a call came from
the frames' clock stamps, which wander, not from the samples, which do not). No network."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from duckmind.live import scenario as S
from duckmind.live import vad as VD
from duckmind.simweb import voice as V

pytestmark = pytest.mark.usefixtures("no_network")

FILE = S.load_scenario_file()
HAVE_BANK = all(
    V.VoiceBank(None, V.DEFAULT_ROOT).ready(V.voice_seed(t.genome_seed))
    for t in FILE.temperaments.values()
)
needs_bank = pytest.mark.skipif(
    not HAVE_BANK, reason="the ducks' voice banks are not rendered here"
)
TEMPERAMENTS = list(FILE.temperaments)
SEEDS = (7, 11)
ONSET_TOLERANCE_S = 0.04
DECIDE_S = 0.1  # the detector knows within this of the speech onset


def drive(
    sid: str,
    tid: str,
    seed: int,
    *,
    use_refs: bool = True,
    late_gain: float = 1.0,
    jitter: float = 0.0,
    **kw: Any,
) -> tuple[VD.VoiceDetector, list[VD.SpeechEvent], S.ScenarioRun, Any]:
    """The scenario's microphone frames through a detector (the duck's own windows as the stream
    lists them). ``late_gain`` / ``jitter``: an echo that stops matching what the detector locked
    (its gain ``late_gain`` times what it was 0.1 s into the call, and a random wobble of the gain
    of ±``jitter`` per frame): a model of the echo that is not exact (tests only)."""
    run = S.ScenarioRun(FILE, sid, tid, seed, require_bank=True)
    mics: list[Any] = []
    make = run.make_mic

    def keep() -> Any:
        mic = make()
        mics.append(mic)
        return mic

    run.make_mic = keep  # type: ignore[method-assign]
    refs = VD.BankReferences(run.bank(), run.voice_seed()) if use_refs else None
    det = VD.VoiceDetector(refs, **kw)
    rng = random.Random(seed)
    events: list[VD.SpeechEvent] = []
    for step in run.steps():
        mic = mics[0]
        pcm = step.frame.pcm
        if (late_gain != 1.0 or jitter) and step.frame.echo:
            # (the echo is not what the detector locked: scale the frame's echo part; the person's
            # voice is not in the frames of the echo-only scenarios)
            into = [
                step.frame.sim_s - w.t0 for w in mic.vocal_windows() if w.t0 <= step.frame.sim_s
            ]
            late = bool(into) and min(into) > 0.1
            gain = (late_gain if late else 1.0) * (1.0 + jitter * rng.uniform(-1, 1))
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) * gain
            pcm = np.clip(x, -32768, 32767).astype("<i2").tobytes()
        events += det.process(pcm, step.frame.sim_s, mic.vocal_windows(), echo=step.frame.echo)
    return det, events, run, mics[0]


# ── a person speaks over the duck's call ───────────────────────────────────────────────


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_a_person_who_speaks_over_the_ducks_call_is_heard_at_once(tid: str, seed: int) -> None:
    det, events, run, _mic = drive("stop_calling", tid, seed)
    said = run.scenario.say[0]
    starts = [e for e in events if e.kind == "speech_start"]
    assert len(starts) == 1, [e.view() for e in events]  # (the duck's own two calls are not speech)
    start = starts[0]
    assert abs(start.t - said.t) <= ONSET_TOLERANCE_S
    assert start.decided_t - start.t <= DECIDE_S
    assert start.over_duck  # (it began while the duck was calling: a barge-in)
    ends = [e for e in events if e.kind == "speech_end"]
    assert len(ends) == 1 and ends[0].t >= said.t + 0.5


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_ducks_own_echo_alone_is_never_a_person(tid: str, seed: int) -> None:
    det, events, run, mic = drive("echo_only", tid, seed)
    assert events == [] and det.voiced_frames == 0
    assert det.echo_frames > 100  # (it called five times: the frames were full of its own voice)
    assert len(mic.vocal_windows()) == 5


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "sid", ["call_name", "hello", "intro_then_hello_mom", "other_name", "what_do_you_see"]
)
def test_speech_with_nothing_over_it_is_found_at_its_onset_and_not_called_a_barge_in(
    sid: str, tid: str, seed: int
) -> None:
    det, events, run, _mic = drive(sid, tid, seed)
    starts = [e for e in events if e.kind == "speech_start"]
    assert starts and all(not e.over_duck for e in starts)
    for said in run.scenario.say:
        assert any(abs(e.t - said.t) <= 0.1 for e in starts), said
    assert all(e.decided_t - e.t <= DECIDE_S for e in starts)


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
def test_a_silent_room_is_silent(tid: str) -> None:
    det, events, _run, _mic = drive("silence", tid, 7)
    assert events == [] and det.voiced_frames == 0
    assert det.floor_dbfs == pytest.approx(-62.0, abs=2.5)


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize(("late_gain", "jitter"), [(1.0, 0.05), (1.10, 0.0), (0.90, 0.0)])
def test_an_echo_a_little_off_what_the_duck_expects_is_still_not_a_person(
    tid: str, late_gain: float, jitter: float
) -> None:
    # (the sim's speaker-to-microphone path is a gain; a real one is not exact: a gain that drifts a
    # tenth after the lock, a wobble of 5 % a frame, leave a residual about 20-26 dB under the echo,
    # and the margin keeps it from being heard as a person)
    _det, events, _run, _mic = drive("echo_only", tid, 7, late_gain=late_gain, jitter=jitter)
    assert events == []


@needs_bank
def test_the_margin_has_an_end_a_very_wrong_echo_is_taken_for_a_person() -> None:
    # (said plainly: the margin alone is 18 dB, so an error of about 13 % in the echo's gain is at
    # its edge (10 % and a 5 % wobble together already is) and with the echo half as loud again as
    # locked the residual is only 6 dB under it: the duck's own call is heard as a voice - with
    # the check that what is left is still the duck's own voice switched off)
    _det, events, _run, _mic = drive("echo_only", "explorer", 7, late_gain=1.5, own_leftover=None)
    assert any(e.kind == "speech_start" for e in events)


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
def test_what_is_left_of_a_very_wrong_echo_is_still_the_duck_and_never_a_person(tid: str) -> None:
    # (what is left of its own call, taken out wrongly, is explained by the very call it is calling:
    # a person's speech is not, r2 under 0.55 against 0.8 and more for the duck's own leftover)
    det, events, _run, _mic = drive("echo_only", tid, 7, late_gain=1.5)
    assert events == [] and det.voiced_frames == 0
    assert det.own_leftover_frames > 0  # (the check did it, not the margin)


@needs_bank
def test_with_no_reference_for_its_own_voice_the_detector_is_deaf_while_it_calls() -> None:
    det, events, run, mic = drive("stop_calling", "explorer", 7, use_refs=False)
    # (it cannot tell the person from itself while it calls: it says nothing then, never a wrong
    # thing; once the call and its echo are over it hears the rest of the person's words)
    over = max(w.t1 for w in mic.vocal_windows()) + VD.ECHO_TAIL_S
    assert all(e.t >= over - 1e-6 and not e.over_duck for e in events if e.kind == "speech_start")
    quiet = drive("hello", "explorer", 7, use_refs=False)[1]
    assert any(
        e.kind == "speech_start" for e in quiet
    )  # (a person with no call going is still heard)


# ── the motors' noise and a clock that wanders (the live finding of LV5) ──────────────────────

NOISE_DB = (None, 40.0, 46.0, 52.0, 58.0)  # the duck's own motors at its microphone
CALL_TAGS = ("alarm", "greet", "inquire", "peck", "chirp", "coo")


def wandering_clock(seed: int, n: int) -> np.ndarray[Any, Any]:
    """The physics seconds the live microphone is ticked with: 20 ms a tick with the ~4 ms jitter
    measured live and now and then a stall (the clock jumps 40-200 ms ahead), after which the
    stream's own clock, which follows it slowly, runs fast for a while against its samples."""
    rng = np.random.default_rng([seed, 99])
    phys = 0.02 * np.arange(n) + rng.normal(0.0, 0.004, n)
    for at in rng.integers(20, n - 40, size=max(1, n // 120)):
        phys[at:] += rng.uniform(0.04, 0.2)
    return phys


def drive_live(
    sid: str,
    tid: str,
    seed: int,
    motor_db: float | None,
    *,
    calls: Sequence[tuple[float, str]] = (),
    wander: bool = True,
) -> tuple[VD.VoiceDetector, list[VD.SpeechEvent], S.ScenarioRun, Any]:
    """A scenario's microphone as the live one is made: ``calls`` (time, tag) are the duck's own
    (replacing the scenario's), the motors' noise is ``motor_db``, the clock wanders."""
    run = S.ScenarioRun(FILE, sid, tid, seed, require_bank=True)
    mic = run.make_mic()
    lib = run.library()
    det = VD.VoiceDetector(VD.BankReferences(run.bank(), run.voice_seed()))
    sounds = []
    for number, say in enumerate(sorted(run.scenario.say, key=lambda e: e.t), start=1):
        geo = run.say_geometry(say)
        sounds.append(
            {
                "id": number,
                "kind": geo["kind"],
                "t0": say.t,
                "t1": say.t + lib.get(say.utterance).duration_s,
                "db": geo["db"],
                "src": f"person:{say.who or 'guest'}",
                "utterance": say.utterance,
            }
        )
    own = sorted(calls) if calls else [(c.t, c.tag) for c in run.scenario.duck_calls]
    n = int(round((max([t for t, _ in own] + [run.scenario.length_s]) + 1.5) / 0.02))
    phys = wandering_clock(seed, n) if wander else 0.02 * np.arange(n)
    events: list[VD.SpeechEvent] = []
    nxt = 0
    for i in range(n):
        t = i * 0.02
        due = []
        while nxt < len(own) and own[nxt][0] <= t + 1e-9:
            due.append({"tag": own[nxt][1], "level": 0.7})
            nxt += 1
        live = [e for e in sounds if e["t0"] <= t + 1e-9]
        frame = mic.tick(float(phys[i]), live, calls=due, motor_db=motor_db, wall=t)
        events += det.process(frame.pcm, frame.sim_s, mic.vocal_windows(), echo=frame.echo)
    return det, events, run, mic


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize("motor_db", NOISE_DB)
def test_the_ducks_own_calls_are_never_a_person_with_noisy_motors_and_a_wandering_clock(
    tid: str, motor_db: float | None
) -> None:
    for seed in SEEDS:
        calls = [(2.0 + 1.8 * k, tag) for k, tag in enumerate(CALL_TAGS)]
        det, events, _run, mic = drive_live("silence", tid, seed, motor_db, calls=calls)
        assert events == [] and det.voiced_frames == 0, (seed, [e.view() for e in events])
        assert len(mic.vocal_windows()) == len(CALL_TAGS)
        assert det.echo_frames > 150


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize("motor_db", NOISE_DB)
def test_a_person_over_the_call_is_still_heard_at_once_with_noisy_motors_and_a_wandering_clock(
    tid: str, motor_db: float | None
) -> None:
    for seed in SEEDS:
        det, events, run, _mic = drive_live("stop_calling", tid, seed, motor_db)
        said = run.scenario.say[0]
        starts = [e for e in events if e.kind == "speech_start"]
        assert len(starts) == 1, (seed, [e.view() for e in events])
        # (the clock wanders: the person's stamp is the stream's, a little off the world's)
        assert abs(starts[0].t - said.t) <= 0.3 and starts[0].over_duck
        assert starts[0].decided_t - starts[0].t <= DECIDE_S + 0.04


@needs_bank
@pytest.mark.parametrize("tid", TEMPERAMENTS)
@pytest.mark.parametrize(
    ("first", "second", "apart_s"),
    [
        ("greet", "chirp", 0.02),
        ("chirp", "greet", 0.02),
        ("greet", "greet", 0.04),
        ("alarm", "inquire", 0.02),
    ],
)
def test_two_calls_started_a_frame_or_two_apart_are_never_a_person(
    tid: str, first: str, second: str, apart_s: float
) -> None:
    # (live: the bridge's reply and the brain's own answer to the same hello came 20 ms apart)
    for seed in SEEDS:
        calls = [(2.0, first), (2.0 + apart_s, second)]
        det, events, _run, mic = drive_live("silence", tid, seed, 40.0, calls=calls)
        assert events == [], (seed, [e.view() for e in events])
        assert len(mic.vocal_windows()) == 2 and det.echo_frames > 20


class _Tones:
    """Two variants of a call that start alike and part later, and come up slowly (like a coo)."""

    QUIET_SAMPLES = 8 * VD.FRAME  # the start, 3 % of the level the call then has

    def __init__(self) -> None:
        rng = np.random.default_rng(4)
        n = 8000
        shared = rng.standard_normal(n) * 6000.0
        self.waves = [shared.copy(), shared.copy()]
        for wave in self.waves:
            wave[2000:] = rng.standard_normal(n - 2000) * 6000.0
            wave[: self.QUIET_SAMPLES] *= 0.03

    def variants(self, tag: str) -> list[np.ndarray[Any, Any]]:
        return self.waves


def test_a_call_that_starts_under_the_motors_noise_is_not_locked_on_the_noise() -> None:
    refs = _Tones()
    det = VD.VoiceDetector(refs)
    window = type("W", (), {"t0": 1.0, "t1": 1.5, "tag": "x", "kind": "call"})()
    rng = np.random.default_rng(9)
    noise = 1500.0  # (-27 dBFS: the motors; the first 8 frames of the call are far under it)
    t, pos = 1.0, 0
    for k in range(18):
        x = 0.8 * refs.waves[1][pos : pos + VD.FRAME] + rng.standard_normal(VD.FRAME) * noise
        det.process(x.astype("<i2").tobytes(), t, [window], echo=True)
        t, pos = t + VD.FRAME_S, pos + VD.FRAME
        (lock,) = det._locks.values()
        if k < 8:  # (the old rule: three loud frames, locked, on nothing but the noise)
            assert not lock.locked, k
    assert lock.locked and lock.best is not None and lock.best[0] == 1  # (once the call is up)
    assert lock.gain == pytest.approx(0.8, rel=0.1)


# ── the pieces ─────────────────────────────────────────────────────────────────────────


def test_flatness_tells_noise_from_a_voiced_sound() -> None:
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(VD.FRAME) * 300
    t = np.arange(VD.FRAME) / VD.SR
    voiced = 3000 * (
        np.sin(2 * np.pi * 220 * t)
        + 0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.4 * np.sin(2 * np.pi * 880 * t)
        + 0.3 * np.sin(2 * np.pi * 1320 * t)
    )
    assert VD.flatness(noise) > VD.FLATNESS_MAX > VD.flatness(voiced)


def test_a_frame_has_320_samples() -> None:
    with pytest.raises(ValueError, match="320"):
        VD.VoiceDetector().process(b"\x00" * 100, 0.0)


def test_speech_between_answers_for_what_the_duck_itself_heard() -> None:
    det = VD.VoiceDetector()
    t = np.arange(VD.FRAME) / VD.SR
    voiced = 4000 * (np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 660 * t))
    voiced_pcm = voiced.astype("<i2").tobytes()
    quiet = (np.random.default_rng(2).standard_normal(VD.FRAME) * 25).astype("<i2").tobytes()
    clock = 0.0
    for _ in range(25):
        det.process(quiet, clock)
        clock += VD.FRAME_S
    assert not det.speech_between(0.0, 0.5)
    for _ in range(20):  # 0.4 s of a voice
        det.process(voiced_pcm, clock)
        clock += VD.FRAME_S
    assert det.speaking and det.speech_between(clock - 0.1, clock)  # (still going counts)
    for _ in range(15):
        det.process(quiet, clock)
        clock += VD.FRAME_S
    assert not det.speaking
    assert det.speech_between(0.4, 0.9) and not det.speech_between(0.0, 0.3)
    assert [e.kind for e in det.events] == ["speech_start", "speech_end"]
    assert det.events[0].t == pytest.approx(0.5, abs=0.02)


def test_a_click_shorter_than_the_debounce_is_not_speech() -> None:
    det = VD.VoiceDetector()
    t = np.arange(VD.FRAME) / VD.SR
    voiced = (4000 * (np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 660 * t))).astype(
        "<i2"
    )
    quiet = (np.random.default_rng(3).standard_normal(VD.FRAME) * 25).astype("<i2").tobytes()
    clock = 0.0
    for k in range(60):
        pcm = voiced.tobytes() if k in (20, 21) else quiet  # two frames (40 ms) only
        det.process(pcm, clock)
        clock += VD.FRAME_S
    assert list(det.events) == [] and det.voiced_frames == 2
