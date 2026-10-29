"""a:sound (2026-09-28) — 聽音樂跳舞: the beat oscillator finds the beat in the microphone's onsets
(late, missed, extra), the head nods DOWN on the beat at a rate the head can follow, an expected
onset is flagged (no orienting to every kick), and a songbird's chirps land on the beat."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from duckmind.neuro import skill_dance as SD


def _train(
    bpm: float,
    *,
    seconds: float = 40.0,
    seed: int = 1,
    miss: float = 0.1,
    extra: int = 0,
    lag: tuple[float, float] = (0.02, 0.04),
    start: float = 1.0,
):
    """Onset strengths per 20-ms tick for a beat at ``bpm`` heard through a microphone model:
    each beat late by ``lag`` s, ``miss`` of them not heard, ``extra`` stray onsets."""
    rng = random.Random(seed)
    period = 60.0 / bpm
    beats = [start + k * period for k in range(int(seconds / period) + 2)]
    ticks: dict[int, float] = {}
    for b in beats:
        if rng.random() < miss:
            continue
        ticks[int(round((b + rng.uniform(*lag)) * 50))] = 0.4 + 0.4 * rng.random()
    for _ in range(extra):
        ticks[int(rng.uniform(start, seconds) * 50)] = 0.2
    return beats, ticks


def _run(osc: SD.BeatOscillator, ticks: dict[int, float], seconds: float):
    out = []
    for tick in range(int(seconds * 50)):
        out.append(osc.step(tick / 50.0, ticks.get(tick, 0.0)))
    return out


@pytest.mark.parametrize("bpm", [60, 72, 84, 104, 124, 150, 170])
def test_it_finds_the_beat_and_its_tempo(bpm: float) -> None:
    beats, ticks = _train(bpm, seed=bpm)
    osc = SD.BeatOscillator()
    trace = _run(osc, ticks, 40.0)
    locked = [t for t in trace if t.active and t.lock > 0.6]
    assert locked, f"no lock at {bpm}"
    first = trace.index(locked[0]) / 50.0
    assert first < 12.0, f"locked only at {first:.1f} s"
    assert abs(osc.hz * 60.0 - bpm) < 0.02 * bpm  # the true tempo, not double or half
    # its beats fall on the heard beats (the microphone's ~30 ms late)
    errors = []
    for i, tick in enumerate(trace):
        if tick.beat_edge and i / 50.0 > 15.0:
            t = i / 50.0
            near = min(beats, key=lambda b: abs(t - (b + 0.03)))
            errors.append(t - (near + 0.03))
    assert statistics.median(abs(e) for e in errors) < 0.03


def test_stray_onsets_do_not_spoil_it() -> None:
    beats, ticks = _train(104, seconds=60.0, extra=40, seed=5)
    osc = SD.BeatOscillator()
    trace = _run(osc, ticks, 60.0)
    assert abs(osc.hz * 60.0 - 104) < 2.0
    assert trace[-1].lock > 0.6
    # most on-beat onsets are expected; stray ones are not
    heard = [t for t in trace[20 * 50 :] if t.onset]
    assert sum(t.expected for t in heard) / len(heard) > 0.6


def test_an_expected_onset_is_flagged_and_a_surprise_is_not() -> None:
    _, ticks = _train(104, miss=0.0, seed=2)
    osc = SD.BeatOscillator()
    _run(osc, ticks, 20.0)
    assert osc.active and osc.lock > 0.8
    # half a period after a beat: a surprise
    t = 20.0
    while not 0.45 <= osc.phase <= 0.55:
        t += 0.005
        osc.step(t, 0.0)
    tick = osc.step(t + 0.005, 0.6)
    assert tick.onset and not tick.expected


def test_the_beat_is_lost_when_the_music_stops() -> None:
    _, ticks = _train(104, seconds=20.0, seed=3)
    osc = SD.BeatOscillator()
    _run(osc, ticks, 20.0)
    assert osc.active
    for tick in range(int(20.0 * 50), int(30.0 * 50)):
        osc.step(tick / 50.0, 0.0)
    assert not osc.active and osc.lock == 0.0


def test_the_nod_rate_is_what_the_head_can_follow() -> None:
    assert SD.beats_per_bob(84 / 60, "sit") == 1  # seated: every beat up to 2.2 Hz
    assert SD.beats_per_bob(124 / 60, "sit") == 1
    assert SD.beats_per_bob(84 / 60, "stand") == 2  # standing: ≤ 1.25 Hz
    assert SD.beats_per_bob(60 / 60, "stand") == 1
    assert SD.beats_per_bob(170 / 60, "stand") == 4


def test_the_head_is_down_on_the_beat() -> None:
    beat = SD.BeatTick(True, 0.0, 1.0, 0.95, 4, True, False, False, None)
    pose = SD.bob_pose(beat, "sit", 1.0, lead=False)
    assert pose.hp == pytest.approx(-SD.BOB_HP["sit"] * 1.0, rel=0.01)  # at its lowest
    half = SD.BeatTick(True, 0.5, 1.0, 0.95, 4, False, False, False, None)
    assert SD.bob_pose(half, "sit", 1.0, lead=False).hp > 0.9 * SD.BOB_HP["sit"]  # up between
    # with the lead the command is ahead by the head's lag (and the hearing's): at phase 1 − lead
    lead = (SD.BOB_LEAD_S["sit"] + SD.HEARING_LAG_S) * 1.0
    early = SD.BeatTick(True, 1.0 - lead, 1.0, 0.95, 3, False, False, False, None)
    assert SD.bob_pose(early, "sit", 1.0).hp == pytest.approx(-SD.BOB_HP["sit"], rel=0.02)


def test_no_nod_without_a_beat_or_without_a_lock() -> None:
    idle = SD.BeatTick(False, 0.0, 0.0, 0.0, 0, False, False, False, None)
    assert SD.bob_pose(idle, "stand", 1.0).hp == 0.0
    unsure = SD.BeatTick(True, 0.0, 1.5, 0.2, 2, True, False, False, None)
    assert SD.bob_pose(unsure, "stand", 1.0, lead=False).hp == 0.0  # sways only


def _dance(skill: SD.DanceSkill, ticks: dict[int, float], seconds: float, *, music: bool = True):
    chirps = []
    for tick in range(int(seconds * 50)):
        t_ms = tick * 20
        d = skill.step(
            t_ms,
            ticks.get(tick, 0.0),
            "music" if music and tick > 100 else None,
            dancing=tick > 400,
        )
        if d.chirp is not None:
            chirps.append((t_ms / 1000.0, d.chirp))
    return chirps, d


def test_a_songbird_sings_along_on_the_beat() -> None:
    beats, ticks = _train(96, seconds=60.0, miss=0.05, seed=4)
    skill = SD.DanceSkill(songbird=1.0, seed=3)
    skill.start_bout(1, 96 / 60)
    chirps, last = _dance(skill, ticks, 60.0)
    assert len(chirps) >= 15
    # each call starts CHIRP_LEAD_MS before a true beat (± a tick)
    offsets = [min((b - t for b in beats), key=abs) for t, _ in chirps]
    assert statistics.median(offsets) == pytest.approx(SD.CHIRP_LEAD_MS / 1000.0, abs=0.03)
    assert last.feature > 0.6 and last.music


def test_other_ducks_dance_quietly() -> None:
    _, ticks = _train(96, seconds=40.0, seed=4)
    chirps, _ = _dance(SD.DanceSkill(songbird=0.1, seed=3), ticks, 40.0)
    assert chirps == []


def test_without_music_the_pull_is_weaker() -> None:
    _, ticks = _train(96, seconds=30.0, seed=6)
    _, with_music = _dance(SD.DanceSkill(), ticks, 30.0, music=True)
    _, without = _dance(SD.DanceSkill(), ticks, 30.0, music=False)
    assert without.feature < 0.5 * with_music.feature
    assert math.isclose(without.beat.hz, with_music.beat.hz, rel_tol=0.02)


def test_a_lively_duck_prefers_a_faster_song() -> None:
    lively, calm = SD.DanceSkill(activity=0.9), SD.DanceSkill(activity=0.1)
    assert lively.preferred_bpm > calm.preferred_bpm
    assert lively.tempo_fit(124) > lively.tempo_fit(84)
    assert calm.tempo_fit(84) > calm.tempo_fit(124)
    assert 0.1 < lively.tempo_fit(60) < 1.0  # far from it still pulls a little


def test_dances_it_kept_at_are_learned() -> None:
    skill = SD.DanceSkill()
    for _ in range(4):
        skill.learn(20.0)
    assert skill.liking > 0.5
    liked = skill.liking
    skill.learn(1.0)  # dropped at once
    assert skill.liking < liked
    _, ticks = _train(96, seconds=30.0, seed=6)
    _, fresh = _dance(SD.DanceSkill(), ticks, 30.0)
    _, fond = _dance(SD.DanceSkill(liking=0.8), ticks, 30.0)
    assert fond.feature > fresh.feature


def _closed_loop(
    lag: float, seconds: float = 60.0, bpm: float = 104.0, noise: float = 0.0, seed: int = 1
) -> tuple[SD.NodTimer, list[float]]:
    """Its nods through a neck of delay ``lag`` (s), felt through its joint sense (PROPRIO_LAG_S
    older still, plus ``noise`` rad of other motion): a steady beat heard HEARING_LAG_S after
    the true beats (true beat k at k/hz). Returns the timer and the physical nods' offsets from
    the true beats (s) over the last 20 s."""
    rng = random.Random(seed)
    timer = SD.NodTimer()
    hz = bpm / 60.0
    joint: list[float] = []  # the head pitch commanded, per tick (+ = nose down)
    back = int(round((lag + SD.PROPRIO_LAG_S) / 0.02))
    physical_back = int(round(lag / 0.02))
    for i in range(int(seconds * 50)):
        t = i * 0.02
        heard = max(0.0, (t - SD.HEARING_LAG_S) * hz)
        beats = int(heard)
        beat = SD.BeatTick(
            active=True,
            phase=heard - beats,
            hz=hz,
            lock=0.95,
            beats=beats,
            beat_edge=False,
            onset=False,
            expected=False,
            error=None,
        )
        felt = (joint[i - back] if i >= back else 0.0) + rng.gauss(0.0, noise)
        timer.step(t, felt, beat, "stand", dancing=True)
        pose = SD.bob_pose(beat, "stand", 1.0, adjust=timer.adjust)
        timer.sent_nod(-pose.hp)
        joint.append(-0.35 * pose.hp)
    # where the physical head was lowest, against the true beats it was meant for
    n = SD.beats_per_bob(hz, "stand")
    physical = [0.0] * physical_back + joint
    offsets = []
    for k in range(0, int(seconds * hz), n):
        b = k / hz
        if b < seconds - 20.0 or b > seconds - 1.0:
            continue
        lo, hi = int((b - 0.3 / hz) * 50), int((b + 0.3 / hz) * 50)
        i = max(range(lo, hi), key=lambda j: physical[j])
        offsets.append(i * 0.02 - b)
    return timer, offsets


@pytest.mark.parametrize("lag", [0.17, 0.23, 0.29])
def test_it_learns_its_necks_delay_by_feel(lag: float) -> None:
    # a neck quicker or slower than its innate guess (0.23 s standing): comparing the nods it
    # sent with the nods it felt, it learns the delay and nods on the beat
    timer, offsets = _closed_loop(lag, noise=0.01)
    assert timer.felt >= 20
    assert timer.adjust == pytest.approx(lag - SD.BOB_LEAD_S["stand"], abs=0.025)
    assert abs(statistics.median(offsets)) <= 0.03


def test_it_does_not_judge_its_timing_without_a_sure_beat_or_a_felt_nod() -> None:
    rng = random.Random(2)
    timer = SD.NodTimer()
    unsure = SD.BeatTick(True, 0.0, 1.7, 0.4, 0, False, False, False, None)
    for i in range(500):
        assert timer.step(i * 0.02, rng.gauss(0, 0.1), unsure, "stand", dancing=True) is None
        timer.sent_nod(math.sin(i * 0.2))
    sure = SD.BeatTick(True, 0.0, 1.7, 0.95, 0, False, False, False, None)
    for i in range(500):  # it nods, but what it feels is other motion only
        assert timer.step(i * 0.02, rng.gauss(0, 0.1), sure, "stand", dancing=True) is None
        timer.sent_nod(math.sin(i * 0.2))
    assert timer.felt == 0 and timer.adjust == 0.0


def test_a_songs_syncopation_surprises_it_once_not_every_time() -> None:
    # a steady beat and, every 8 beats, an extra hit half a beat after the beat
    bpm = 104.0
    period = 60.0 / bpm
    beats, ticks = _train(bpm, seconds=50.0, miss=0.0, seed=4)
    syncopes = [b + 0.5 * period + 0.03 for b in beats[16::8]]
    for t in syncopes:
        ticks[int(round(t * 50))] = 0.6
    skill = SD.DanceSkill()
    heard = []
    for tick in range(int(50.0 * 50)):
        d = skill.step(tick * 20, ticks.get(tick, 0.0), "music", dancing=True)
        t = tick / 50.0
        if d.beat.onset and any(abs(t - s) < 0.03 for s in syncopes):
            heard.append((t, d.beat.expected or d.recurring))
    assert len(heard) >= 4
    assert not heard[0][1]  # the first one is a surprise …
    assert all(expected for _, expected in heard[1:])  # … the pattern's next ones are not
    # an off-beat hit where the pattern has none is still a surprise
    skill2 = SD.DanceSkill()
    lone = beats[40] + 0.5 * period + 0.03
    ticks2 = dict(_train(bpm, seconds=50.0, miss=0.0, seed=4)[1])
    ticks2[int(round(lone * 50))] = 0.6
    surprise = None
    for tick in range(int(50.0 * 50)):
        d = skill2.step(tick * 20, ticks2.get(tick, 0.0), "music", dancing=True)
        if d.beat.onset and abs(tick / 50.0 - lone) < 0.03:
            surprise = not (d.beat.expected or d.recurring)
    assert surprise is True
