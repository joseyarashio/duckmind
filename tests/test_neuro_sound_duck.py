"""a:sound (2026-09-28) — the whole neural duck headless with the sound skills (the official MJCF,
robotd's control path, the world, the senses, the brain, the motor, the body link, robotd's
streamed voice): the behaviours emerge in the one competition and the body performs them.

Short versions of the evidence runs (measurement records, not in the repo): one duck,
one scenario each."""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Iterator
from typing import Any

import pytest

from duckmind import paths

pytest.importorskip("mujoco")
pytest.importorskip("numpy")

MJCF = str(paths.microduck_mjcf_dir())
if not os.path.isdir(MJCF):
    pytest.skip("the official MJCF folder is not here", allow_module_level=True)

from duckmind.neuro import skill_dance as SD  # noqa: E402
from duckmind.neuro import skill_sound as SS  # noqa: E402
from duckmind.simweb import neuro_voice_stream as V  # noqa: E402

# The theremin pitch and the wheee ride are sung by our duck-voice-stream helper
# (tools/voicestream, built with cargo); without it robotd stays silent.
needs_helper = pytest.mark.skipif(V.helper_path() is None, reason="duck-voice-stream not built")


@pytest.fixture(autouse=True)
def _skills() -> Iterator[None]:
    SS.install()
    yield
    SS.uninstall()


def _duck(spec: str, **kw: Any) -> Any:
    from duckmind.neuro.genome import Genome, preset
    from duckmind.simweb.neuro_sound_headless import SoundDuck

    if ":" in spec:
        seed, gd = spec.split(":")
        genome = Genome.from_seed(int(seed), gd=int(gd))
    else:
        genome = preset(spec)
    duck = SoundDuck(genome, genome.seed, engine=kw.pop("engine", True), voice_rng=1, **kw)
    assert duck.stood
    return duck


def test_a_duck_dances_to_the_music_with_its_head_down_on_the_beat() -> None:
    from duckmind.world.neuro_music import Song

    duck = _duck("4:60")  # a songbird: it dances and sings along
    duck.run(2.0)
    duck.world("music", bpm=104, seed=7)
    rows: list[tuple[float, str, float, bool]] = []
    duck.run(
        40.0,
        lambda s: (
            rows.append((s.duck.now(), s.cmd.behaviour, s.head_joints()[1], s.brain.last.dancing))
            and False
        ),
    )
    danced = [r for r in rows if r[1] == "dance"]
    assert len(danced) > 0.3 * len(rows)
    last = duck.brain.last
    assert last is not None and abs(last.beat.bpm - 104) < 3.0
    music = duck.sound_section["music"]
    period = 60.0 / music["bpm"]
    beats = [music["t0"] + b for b, _ in Song(music["bpm"], 7).beats(0.0, 60.0)]
    # The beats whose whole window is nodded: a tick of the dance act in which a reflex has the
    # head has no nod (SoundSkills.post), so a window with one says nothing about the nod's timing.
    dancing = {round(r[0], 3) for r in rows if r[3]}
    n = SD.beats_per_bob(music["bpm"] / 60.0, "stand")
    offsets = []
    for b in beats:
        window = [r for r in rows if abs(r[0] - b) <= n * period / 2]
        if len(window) > 5 and all(round(r[0], 3) in dancing for r in window):
            low = max(window, key=lambda r: r[2])[0]  # the nose at its lowest
            if abs(low - b) <= 0.5 * period:
                offsets.append(low - b)
    assert len(offsets) >= 10
    assert abs(statistics.median(offsets)) < 0.06
    chirps = [v for v in duck.vocals if v[2] == "dance" and v[1] == "chirp"]
    assert len(chirps) >= 5  # the songbird sings along …
    near = [min(abs(c[0] - b) for b in beats) for c in chirps]
    assert statistics.median(near) < 0.06  # … on the beat
    assert not duck.fallen()


@needs_helper
def test_a_tof_duck_plays_the_theremin_and_robotd_sings_the_hand() -> None:
    duck = _duck("dreamer", tof=True)
    duck.run(2.0)
    duck.world("hover", play=True)
    notes: list[tuple[str, float | None, int | None]] = []

    def each(s: Any) -> bool:
        t = s.tick_out
        notes.append(
            (s.cmd.behaviour, None if t is None else t.note_hz, None if t is None else t.hand_mm)
        )
        return False

    # The first 8 s let the hand approach naturally and give the duck time to choose the
    # instrument. Then the person slowly moves the same hand back to 45 cm and holds it
    # there for the remaining 8 s. This provides an owned far-range pitch sample without
    # weakening orient/startle reflex ownership while the hand is approaching.
    duck.run(8.0, each)
    duck.world("hover", distance_mm=450)
    duck.run(8.0, each)
    playing = [n for n in notes if n[0] == "theremin"]
    assert len(playing) > 50 * 3  # it took up the instrument for seconds
    sounding = [n for n in notes if n[1] is not None and n[2] is not None]
    assert len(sounding) > 50 * 3
    # closer is higher: robotd's pitch follows the hand's distance
    near = [n[1] for n in sounding if n[2] < 250]
    far = [n[1] for n in sounding if n[2] > 380]
    assert near and far and statistics.median(near) > statistics.median(far)
    chunk = duck.stream.chunks[-1]
    assert chunk.n == 960
    assert not duck.fallen()


@needs_helper
def test_a_bold_duck_rides_with_the_wheee_and_shakes_after() -> None:
    duck = _duck("explorer")
    duck.run(2.0)
    duck.world("swing", cycles=3)
    states: list[tuple[str, bool, str | None, str | None]] = []

    def each(s: Any) -> bool:
        states.append(
            (
                s.cmd.behaviour,
                s.brain.last.wheee_hold,
                None if s.tick_out is None else s.tick_out.ride,
                s.tel.moment,
            )
        )
        return False

    duck.run(9.0, each)
    assert any(b == "wheee_ride" for b, *_ in states)
    assert any(r == "riding" for *_, r, _m in states)  # robotd's ride: start → loop …
    duck.stim("set_down")
    after: list[tuple[str, bool, str | None, str | None]] = []
    duck.run(
        8.0,
        lambda s: (
            after.append((s.cmd.behaviour, s.brain.last.wheee_hold, s.tick_out.ride, s.tel.moment))
            and False
        ),
    )
    rides = [r for *_, r, _m in after]
    assert "landing" in rides or rides[-1] == "off"  # … and its end segment after the set-down
    assert rides[-1] == "off"
    assert any(m == "head_shake" for *_, m in after) or any(b == "body_shake" for b, *_ in after)
    assert not duck.fallen()
    assert math.isfinite(duck.brain.skills.ride.value) and duck.brain.skills.ride.value > 0.0
