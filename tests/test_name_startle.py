"""Calling the duck's name must not startle it (owner-approved fix, 2026-10-02).

Machine verification v1 found it on the physics layer only: a shy duck startled at a name call
4/4, a lazy one 4/4, a social one 3/4 (the brain-level synthetic world: 0/16). The cause is a
timing one. The microphone model reports a sound's ONSET 20-40 ms after it starts, but duck-ear
names it later — an impulse 80-160 ms later, a spoken word (its own name included) only once it
has been said, about a second later — so the reflex layer holds an onset with ``kind=None``, and
it used to read "no class yet" as "the class noise" (startle weight 0.9, every sound alike).
These tests run the real sensor model into the real reflex arcs: no class label is handed over."""

from __future__ import annotations

import pytest

from duckmind.neuro import reflex as R
from duckmind.neuro.contract import SensorFrame, SoundSense
from duckmind.world import neuro_senses as ns

STARTLES = frozenset({"startle", "freeze"})
# a shy duck: the lowest boldness (the lowest startle threshold, the highest gain) and the most
# social one — the "very clingy" duck of the review that fled from its own name
SHY = R.params_from({"boldness": 0.0, "sociability": 0.9, "wit": 0.5}, stage="juvenile", side=1)
MIDDLE = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)
# the levels the machine-verification traces recorded at the duck's head: a name called from
# ~1.5-2 m 63.5-65.1 dB (level_db_head of the 16 call_name trials), a clap 75.9-78.0 dB
NAME_DB = 65.0
CLAP_DB = 77.5
HELLO_DB = 61.0  # the world's greeting is 4 dB under the name at the same distance


def _sound(kind: str, db: float, seconds: float, env: str, **kw: object) -> dict[str, object]:
    return {"id": 1, "kind": kind, "t0": 1.0, "t1": 1.0 + seconds, "db": db, "env": env, **kw}


def _frames(sound: dict[str, object], *, seed: int = 1, seconds: float = 4.0) -> list[SensorFrame]:
    """What the microphone model reports each 20 ms tick for one world sound."""
    mic = ns.HearingModel(seed=seed)
    frames = []
    for tick in range(int(seconds / 0.02)):
        s = tick * 0.02
        mic.update([sound], now_s=s)
        frames.append(SensorFrame(t_ms=round(s * 1000), sound=mic.step(s, self_db=20.0)))
    return frames


def _reflexes(params: R.ReflexParams, frames: list[SensorFrame], seed: int) -> set[str]:
    state = R.initial(seed)
    seen: set[str] = set()
    for frame in frames:
        state, out = R.step(params, state, frame, hab={}, sens={})
        seen |= {e.reflex for e in out.events}
    return seen


def test_the_microphone_gives_the_onset_before_the_class_of_a_spoken_name() -> None:
    """The premise: at the onset the model says no class yet, and it names the word only after
    it was said (this is what duck-ear's keyword spotter honestly needs)."""
    frames = _frames(_sound("name", NAME_DB, 0.7, "speech", name=True))
    first_onset = next(f for f in frames if f.sound.onset >= 0.08)
    first_class = next(f for f in frames if f.sound.kind is not None)
    assert first_onset.sound.kind is None
    assert first_class.t_ms - first_onset.t_ms > 500


@pytest.mark.parametrize("seed", range(1, 25))
def test_a_name_call_through_the_microphone_does_not_startle_a_shy_duck(seed: int) -> None:
    frames = _frames(_sound("name", NAME_DB, 0.7, "speech", name=True), seed=seed)
    seen = _reflexes(SHY, frames, seed)
    assert not seen & STARTLES, seen
    assert "orient" in seen  # it does listen


@pytest.mark.parametrize("seed", range(1, 25))
def test_a_hello_through_the_microphone_does_not_startle_a_shy_duck(seed: int) -> None:
    frames = _frames(_sound("hello", HELLO_DB, 0.6, "speech"), seed=seed)
    assert not _reflexes(SHY, frames, seed) & STARTLES


@pytest.mark.parametrize("seed", range(1, 9))
def test_a_name_called_from_one_metre_does_not_startle_either(seed: int) -> None:
    """70 dB at 1 m (a raised voice calling it) is the loudest name the world makes."""
    frames = _frames(_sound("name", 70.0, 0.7, "speech", name=True), seed=seed)
    assert not _reflexes(SHY, frames, seed) & STARTLES


@pytest.mark.parametrize("seed", range(1, 9))
def test_a_clap_through_the_microphone_still_startles_a_shy_duck(seed: int) -> None:
    frames = _frames(_sound("clap", CLAP_DB, 0.04, "impulse"), seed=seed)
    assert "startle" in _reflexes(SHY, frames, seed)


def test_a_loud_bang_startles_at_the_onset_with_no_class_yet() -> None:
    """Loud enough that nothing needs recognising: the startle does not wait for duck-ear."""
    frame = SensorFrame(t_ms=0, sound=SoundSense(level_db=82.0, onset=1.0, kind=None))
    out = R.step(SHY, R.initial(3), frame, hab={}, sens={})[1]
    assert out.events and out.events[0].reflex == "startle"


def _onset_then_class(
    kind: str, *, db: float = 62.0, after_ms: int = 120, seed: int = 4
) -> tuple[list[str], list[str]]:
    """An unnamed onset at ``db``, then duck-ear's class ``after_ms`` later: the reflexes the
    onset tick fired, and the ones the class tick fired."""
    state = R.initial(seed)
    onset = SensorFrame(t_ms=0, sound=SoundSense(level_db=db, onset=1.0, kind=None))
    state, out = R.step(SHY, state, onset, hab={}, sens={})
    at_onset = [e.reflex for e in out.events]
    named = []
    for t in range(20, after_ms + 200, 20):
        kind_now = kind if t >= after_ms else None
        frame = SensorFrame(t_ms=t, sound=SoundSense(level_db=db, onset=0.0, kind=kind_now))
        state, out = R.step(SHY, state, frame, hab={}, sens={})
        named += [e.reflex for e in out.events]
    return at_onset, named


def test_a_quiet_unnamed_onset_gets_a_look_not_a_startle() -> None:
    at_onset, _ = _onset_then_class("name")
    assert at_onset == ["orient"]


@pytest.mark.parametrize("kind", ["name", "hello", "voice"])
def test_a_class_that_is_no_threat_never_startles_when_it_arrives(kind: str) -> None:
    _, named = _onset_then_class(kind)
    assert "startle" not in named


@pytest.mark.parametrize("kind", ["noise", "door", "bang"])
def test_a_class_that_is_a_threat_startles_when_it_arrives(kind: str) -> None:
    """The class decides: what the shy duck startled at before, it still startles at — the
    moment the ear has named it, not before."""
    at_onset, named = _onset_then_class(kind)
    assert "startle" not in at_onset
    assert named == ["startle"]


def test_a_class_that_comes_too_late_does_not_belong_to_the_onset() -> None:
    _, named = _onset_then_class("noise", after_ms=R.SOUND_NAME_WAIT_MS + 100)
    assert named == []


def test_an_onset_that_startled_is_not_startled_at_again_when_it_is_named() -> None:
    state = R.initial(5)
    loud = SensorFrame(t_ms=0, sound=SoundSense(level_db=84.0, onset=1.0, kind=None))
    state, out = R.step(SHY, state, loud, hab={}, sens={})
    assert [e.reflex for e in out.events] == ["startle"]
    events = []
    for t in range(20, 400, 20):
        kind = "bang" if t >= 120 else None
        frame = SensorFrame(t_ms=t, sound=SoundSense(level_db=80.0, onset=0.0, kind=kind))
        state, out = R.step(SHY, state, frame, hab={}, sens={})
        events += out.events
    assert [e.reflex for e in events if e.reflex == "startle"] == []


def test_a_sound_named_at_its_onset_is_judged_by_its_class_as_before() -> None:
    """The brain-level synthetic world names a sound the tick it begins: nothing changed there."""
    name = SensorFrame(t_ms=0, sound=SoundSense(level_db=82.0, onset=1.0, kind="name"))
    for seed in range(10):
        out = R.step(SHY, R.initial(seed), name, hab={}, sens={})[1]
        assert out.events and all(e.reflex == "orient" for e in out.events)
    bang = SensorFrame(t_ms=0, sound=SoundSense(level_db=88.0, onset=1.0, kind="bang"))
    assert R.step(MIDDLE, R.initial(2), bang, hab={}, sens={})[1].events[0].reflex == "startle"
