import dataclasses
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from duckmind.namekws import synth  # noqa: E402

SPEC = synth.NameSpec.parse("蛋黃", "dan4 huang2")
RATE = 24_000  # what Qwen3-TTS returns


def _quiet(_message: str) -> None:
    pass


@pytest.fixture
def no_pinyin(monkeypatch):
    """Decide with the built-in homophone table only, as when pypinyin is not installed."""
    monkeypatch.setattr(synth, "_pinyin_backend", lambda: None)


def _positives() -> list[str]:
    return [text for texts in synth.text_bank(SPEC).positives.values() for text in texts]


def _silence(seconds: float, rate: int = RATE):
    return np.zeros(round(seconds * rate), dtype=np.float32)


def _burst(seconds: float, rate: int = RATE, level: float = 0.15, seed: int = 0):
    """Noise with 5 ms fades: loud from its first to its last 10 ms frame."""
    rng = np.random.default_rng(seed)
    size = round(seconds * rate)
    audio = level * rng.standard_normal(size)
    fade = min(size // 2, round(0.005 * rate))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        audio[:fade] *= ramp
        audio[size - fade :] *= ramp[::-1]
    return audio.astype(np.float32)


class FakeDesign:
    """Stands in for Qwen3TTSModel (VoiceDesign): a noise burst per text, at 24 kHz."""

    def __init__(self, silent=(), runaway=(), oom_above=None):
        self.calls = []
        self.silent = set(silent)
        self.runaway = set(runaway)
        self.oom_above = oom_above

    def generate_voice_design(self, text, instruct, language=None, **kwargs):
        if self.oom_above is not None and len(text) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.calls.append(
            {"text": list(text), "instruct": list(instruct), "language": list(language), **kwargs}
        )
        wavs = []
        for index, item in enumerate(text):
            seed = 1000 * len(self.calls) + index
            if item in self.silent:
                wavs.append(_silence(0.8))
            elif item in self.runaway:  # never stopped: exactly the frame budget
                wavs.append(_burst(kwargs["max_new_tokens"] / 12.5, seed=seed))
            else:
                speech = 0.1 + 0.2 * len(synth.hanzi(item))
                wavs.append(
                    np.concatenate([_silence(0.2), _burst(speech, seed=seed), _silence(0.3)])
                )
        return wavs, RATE


class FakeClone:
    """Stands in for Qwen3TTSModel (Base): x-vector prompts, then cloned speech."""

    def __init__(self):
        self.prompted = []
        self.calls = []

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        assert x_vector_only_mode is True
        self.prompted += [(audio.size, rate) for audio, rate in ref_audio]
        return [f"prompt-{len(self.prompted) - len(ref_audio) + i}" for i in range(len(ref_audio))]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        assert len(voice_clone_prompt) == len(text) == len(language)
        self.calls.append({"text": list(text), "prompts": list(voice_clone_prompt), **kwargs})
        return [
            np.concatenate([_silence(0.1), _burst(0.5, seed=i), _silence(0.1)])
            for i in range(len(text))
        ], RATE


def _plan(voices=3, per_voice=4, negatives=4, seed=7):
    bank = synth.text_bank(SPEC)
    return synth.plan_jobs(
        synth.design_voices(voices, seed),
        bank,
        per_voice=per_voice,
        negatives_per_voice=negatives,
        seed=seed,
    )


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]


# ── Import weight ────────────────────────────────────────────────────────────


def test_importing_the_module_loads_no_model_library():
    code = (
        "import sys, duckmind.namekws.synth\n"
        "heavy = ('torch', 'qwen_tts', 'transformers', 'scipy', 'pypinyin', 'soundfile')\n"
        "print(sorted(name for name in heavy if name in sys.modules))"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.stdout.strip() == "[]"


# ── Voices ───────────────────────────────────────────────────────────────────


def test_voice_designs_are_seeded_and_keep_their_ids_when_more_are_asked_for():
    first = synth.design_voices(60, seed=3)
    assert first == synth.design_voices(60, seed=3)
    assert synth.design_voices(25, seed=3) == first[:25]
    assert [v.instruct for v in synth.design_voices(60, seed=4)] != [v.instruct for v in first]
    assert [v.voice_id for v in first[:3]] == ["d0000", "d0001", "d0002"]
    assert len({v.instruct for v in first}) == 60
    assert all(v.engine == "qwen3-voicedesign" and v.instruct.endswith("。") for v in first)


@pytest.mark.parametrize("count", [50, 300, 1200])
def test_about_forty_percent_of_the_voices_are_children(count):
    voices = synth.design_voices(count, seed=7)
    children = [v for v in voices if v.age_group in {"child_3_5", "child_6_9"}]
    assert abs(len(children) / count - 0.40) <= (0.05 if count < 100 else 0.02)
    assert all(("小男孩" in v.instruct) or ("小女孩" in v.instruct) for v in children)
    assert {v.age_group for v in voices} == {age.key for age in synth.AGES}


def test_voices_vary_in_every_described_attribute():
    voices = synth.design_voices(300, seed=7)
    assert {v.mood for v in voices} == set(synth.MOOD)
    assert {v.accent for v in voices} == set(synth.ACCENT)
    assert {v.pace for v in voices} == set(synth.PACE)
    assert {v.gender for v in voices} == {"male", "female"}
    assert any("台語腔" in v.instruct for v in voices)
    assert any("隔壁房間" in v.instruct for v in voices)


def test_clone_voices_take_one_reference_per_speaker_whatever_the_file_order(tmp_path):
    lines = [
        {"path": "a/1.wav", "speaker": "s1", "age_group": "child_6_9"},
        {"path": "a/2.wav", "speaker": "s1"},
        {"path": "b/1.wav", "speaker": 2, "text": "嗨小問"},
        {"path": "/abs/c.wav"},
    ]
    refs_file = tmp_path / "refs.jsonl"
    refs_file.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    refs = synth.read_clone_refs(refs_file)
    voices = synth.clone_voices(refs, 10, seed=7)
    assert len(voices) == 3  # s1, 2, and the speaker-less reference
    assert voices == synth.clone_voices(list(reversed(refs)), 10, seed=7)
    assert all(v.engine == "qwen3-clone" and v.instruct is None for v in voices)
    assert all(v.voice_id.startswith("c") and len(v.voice_id) == 11 for v in voices)
    assert str(tmp_path / "b" / "1.wav") in {v.ref_path for v in voices}
    assert "/abs/c.wav" in {v.ref_path for v in voices}
    assert synth.clone_voices(refs, 2, seed=7) == voices[:2]


# ── Texts ────────────────────────────────────────────────────────────────────


def test_each_voice_says_a_seeded_rotating_slice_of_the_texts():
    bank = synth.text_bank(SPEC)
    voices = synth.design_voices(40, seed=7)
    jobs = synth.plan_jobs(voices, bank, per_voice=12, negatives_per_voice=12, seed=7)
    assert jobs == synth.plan_jobs(voices, bank, per_voice=12, negatives_per_voice=12, seed=7)
    assert len(jobs) == 40 * 24 and len({job.clip_id for job in jobs}) == len(jobs)
    first = [job for job in jobs if job.voice.voice_id == "d0000"]
    assert Counter(job.category for job in first if job.kind == "positive") == {
        "alone": 5,
        "end": 4,
        "start": 2,
        "repeat": 1,
    }
    assert Counter(job.category for job in first if job.kind == "hard_negative") == {
        "near": 5,
        "embedded": 3,
        "family": 3,
        "syllable": 1,
    }
    # Together the voices say every form, and neighbouring voices say different ones.
    assert {j.text for j in jobs if j.kind == "positive"} == set(_positives())
    every_negative = {text for texts in bank.negatives.values() for text in texts}
    assert {j.text for j in jobs if j.kind == "hard_negative"} == every_negative
    ends = [[j.text for j in jobs if j.voice is v and j.category == "end"] for v in voices[:2]]
    assert ends[0] != ends[1]
    other = synth.plan_jobs(voices, bank, per_voice=12, negatives_per_voice=12, seed=8)
    assert [job.text for job in other] != [job.text for job in jobs]
    # More clips per voice keep the first ones, so an earlier run's files stay valid.
    more = synth.plan_jobs(voices[:1], bank, per_voice=16, negatives_per_voice=16, seed=7)
    assert [job.text for job in more if job.kind == "positive"][:12] == [
        job.text for job in first if job.kind == "positive"
    ]


def test_every_positive_says_the_name_mostly_alone_or_last():
    assert all("蛋黃" in text for text in _positives())
    positives = [job for job in _plan(voices=20, per_voice=12) if job.kind == "positive"]
    assert all("蛋黃" in job.text for job in positives)
    ending = sum(synth.hanzi(job.text).endswith("蛋黃") for job in positives)
    assert ending / len(positives) >= 0.8


def test_the_name_needs_one_pinyin_syllable_per_character():
    assert SPEC.syllables == ("dan4", "huang2") and SPEC.toneless == ("dan", "huang")
    assert synth.NameSpec.parse("蛋黃", " Dan4, HUANG2 ").syllables == ("dan4", "huang2")
    with pytest.raises(ValueError):
        synth.NameSpec.parse("蛋黃", "dan4")
    with pytest.raises(ValueError):
        synth.NameSpec.parse("DenHuang", "dan4 huang2")
    with pytest.raises(ValueError):
        synth.NameSpec.parse("蛋黃", "dàn huáng")


# ── The hard-negative validator ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["蛋黃", "蛋黄", "淡黃", "單黃", "淡黃！", "但黃～", "丹凰？", "蛋黃蛋黃", "淡黃，淡黃"]
)
def test_the_bare_name_and_its_homophones_are_never_hard_negatives(no_pinyin, text):
    assert synth.negative_problems(text, SPEC)


def test_a_positive_with_a_homophone_swapped_in_is_refused(no_pinyin):
    positives = _positives()
    assert any("positive" in p for p in synth.negative_problems("淡黃你在哪", SPEC, positives))
    assert any("positive" in p for p in synth.negative_problems("過來，單黃！", SPEC, positives))


@pytest.mark.parametrize(
    "text", ["淡黃色的衣服", "蛋黃酥", "鹹蛋黃", "彈簧床好軟", "單簧管", "蛋", "黃？", "過來"]
)
def test_context_or_a_single_syllable_keeps_a_near_sound_safe(no_pinyin, text):
    assert synth.negative_problems(text, SPEC, _positives()) == []


def test_the_built_in_hard_negatives_are_safe_and_cover_the_requested_words(no_pinyin):
    bank = synth.text_bank(SPEC)
    assert bank.dropped == {}
    negatives = [text for texts in bank.negatives.values() for text in texts]
    for word in (
        "蛋黃酥 鹹蛋黃 蛋黃哥 把蛋黃打散 淡黃色的衣服 彈簧 蛋白 蛋花湯 蛋糕 黃色的球 但是 當然 "
        "荒唐 皇上 單簧管 今天吃蛋黃酥 這個淡黃色好看 吃飯了 過來 好乖 你在哪 晚安"
    ).split():
        assert any(word in text for text in negatives), word
    assert {"蛋", "黃"} <= set(negatives)
    assert "淡黃" not in negatives and "彈簧" not in negatives  # near-sounds come with context


def test_the_plan_refuses_a_bank_with_an_unsafe_text():
    bank = synth.text_bank(SPEC)
    voice = synth.design_voices(1, 7)
    unsafe = dataclasses.replace(bank, negatives={**bank.negatives, "near": ("淡黃",)})
    with pytest.raises(ValueError, match="homophone of the bare name"):
        synth.plan_jobs(voice, unsafe, per_voice=1, negatives_per_voice=1, seed=7)
    nameless = dataclasses.replace(bank, positives={**bank.positives, "end": ("過來！",)})
    with pytest.raises(ValueError, match="does not say"):
        synth.plan_jobs(voice, nameless, per_voice=1, negatives_per_voice=1, seed=7)


def test_with_pypinyin_any_spelling_of_the_names_sound_is_refused():
    pytest.importorskip("pypinyin")
    positives = _positives()
    assert "has the name's pinyin" in synth.negative_problems("眈潢", SPEC)  # not in the table
    assert any(
        "pinyin of the positive" in p
        for p in synth.negative_problems("眈潢你在哪", SPEC, positives)
    )
    assert synth.pinyin_of("這個彈簧壞掉了")[2:4] == ("tan2", "huang2")  # pypinyin's reading fixed
    assert synth.negative_problems("彈簧", SPEC) == []  # tán huáng is not dàn huáng
    assert synth.text_bank(SPEC).dropped == {}


# ── Post-processing ──────────────────────────────────────────────────────────


def test_a_take_becomes_16k_int16_trimmed_to_150_ms_and_peaked_at_minus_3_dbfs():
    pytest.importorskip("scipy")
    audio = np.concatenate([_silence(1.0), _burst(0.6), _silence(1.2)])
    take = synth.postprocess(audio, RATE, "蛋黃你在哪")
    assert take.reason is None and take.pcm.dtype == np.int16
    assert take.speech_s == pytest.approx(0.6, abs=0.02)
    assert take.duration_s == pytest.approx(take.pcm.size / 16_000)
    assert take.duration_s == pytest.approx(0.6 + 2 * synth.PAD_S, abs=0.03)
    assert np.abs(take.pcm).max() == pytest.approx(10 ** (-3 / 20) * 32767, rel=0.001)
    edge = round(0.12 * 16_000)  # the padding is quiet, the speech starts right after it
    assert np.abs(take.pcm[:edge]).max() < 0.02 * 32767
    assert np.abs(take.pcm[edge + 800 : edge + 1600]).max() > 0.2 * 32767


@pytest.mark.parametrize("rate", [16_000, 22_050, 24_000, 44_100])
def test_any_rate_and_layout_comes_out_as_16k_mono(rate):
    pytest.importorskip("scipy")
    mono = np.concatenate([_silence(0.4, rate), _burst(0.5, rate), _silence(0.4, rate)])
    for audio in (mono, np.stack([mono, mono]), np.stack([mono, mono], axis=1)):
        take = synth.postprocess(audio, rate, "蛋黃")
        assert take.reason is None
        assert take.pcm.ndim == 1
        assert take.pcm.size == pytest.approx((0.5 + 2 * synth.PAD_S) * 16_000, abs=400)


def test_silence_clipping_runaways_and_implausible_lengths_are_rejected():
    pytest.importorskip("scipy")

    def reason(audio, text="蛋黃", **options):
        return synth.postprocess(audio, RATE, text, **options).reason

    assert reason(_silence(1.0)) == "empty"
    assert reason(np.zeros(0)) == "empty"
    assert reason(_burst(0.5, level=1e-5)) == "empty"  # −100 dBFS
    assert reason(np.clip(_burst(0.6) * 20, -1.0, 1.0)) == "clipped"
    peaky = _burst(0.6)
    assert reason(peaky / np.abs(peaky).max()) is None  # full scale once is not clipping
    assert reason(_burst(9.0)) == "too_long"
    blip = np.concatenate([_silence(0.5), _burst(0.05), _silence(0.5)])
    assert reason(blip, "我回來了，蛋黃！") == "too_short"
    broken = _burst(0.6)
    broken[100] = np.nan
    assert reason(broken) == "not_finite"
    assert reason(_burst(0.6), runaway=True) == "runaway"


def test_plausible_lengths_grow_with_syllables_and_pauses():
    low, high = synth.plausible_speech_s("蛋黃")
    assert 0.2 < low < 0.3 and 2.0 < high < 3.0
    assert (
        synth.plausible_speech_s("我回來了，蛋黃！")[1]
        > synth.plausible_speech_s("我回來了蛋黃")[1]
    )
    assert synth.token_budget(["蛋黃"]) < synth.token_budget(["我回來了，蛋黃！"]) <= 100


# ── Generation, metadata and resuming ────────────────────────────────────────


def test_generation_writes_16k_wavs_and_one_metadata_row_each(tmp_path):
    pytest.importorskip("scipy")
    soundfile = pytest.importorskip("soundfile")
    jobs = _plan()
    model = FakeDesign()
    summary = synth.generate(jobs, tmp_path, design_model=model, batch=5, log=_quiet)
    assert summary == {"written": 24, "rejected": {}, "failed": 0, "remaining": 0, "stopped": False}
    assert all(len(call["text"]) <= 5 for call in model.calls)
    assert all(call["language"] == ["Chinese"] * len(call["text"]) for call in model.calls)
    assert all(0 < call["max_new_tokens"] <= 100 for call in model.calls)
    instruct_of = {job.text + job.voice.voice_id: job.voice.instruct for job in jobs}
    assert sum(len(call["text"]) for call in model.calls) == 24
    said = {(job.text, job.voice.instruct) for job in jobs}  # each text with its own voice
    assert all(pair in said for call in model.calls for pair in zip(call["text"], call["instruct"]))
    rows = _rows(tmp_path)
    assert len(rows) == 24 and {row["kind"] for row in rows} == {"positive", "hard_negative"}
    for row in rows:
        assert {"path", "kind", "text", "voice_id", "engine", "instruct", "age_group"} <= set(row)
        assert row["asr_text"] is None and row["asr_ok"] is None
        assert row["path"] == row["clip_id"] + ".wav"
        assert row["path"].split("/")[:2] == [row["kind"], row["voice_id"]]
        assert row["instruct"] == instruct_of[row["text"] + row["voice_id"]]
        info = soundfile.info(str(tmp_path / row["path"]))
        assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
        assert row["duration_s"] == pytest.approx(info.frames / 16_000, abs=0.001)
    # Everything present: running again synthesizes nothing.
    again = synth.generate(jobs, tmp_path, design_model=FakeDesign(), batch=5, log=_quiet)
    assert again["written"] == 0 and again["remaining"] == 0


def test_a_crash_mid_write_is_repaired_and_only_lost_clips_are_made_again(tmp_path):
    pytest.importorskip("scipy")
    jobs = _plan()
    synth.generate(jobs, tmp_path, design_model=FakeDesign(), batch=8, log=_quiet)
    metadata = tmp_path / "metadata.jsonl"
    lines = metadata.read_text().splitlines()
    cut = json.loads(lines[-1])
    metadata.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:30])  # died mid-row
    gone = json.loads(lines[0])
    (tmp_path / gone["path"]).unlink()  # and a WAV went missing
    model = FakeDesign()
    summary = synth.generate(jobs, tmp_path, design_model=model, batch=8, log=_quiet)
    assert summary["written"] == 2
    assert sorted(text for call in model.calls for text in call["text"]) == sorted(
        [cut["text"], gone["text"]]
    )
    rows = _rows(tmp_path)  # every line parses again
    assert sorted(row["clip_id"] for row in rows) == sorted(job.clip_id for job in jobs)


def test_bad_takes_are_logged_retried_and_kept_out_of_the_metadata(tmp_path):
    pytest.importorskip("scipy")
    jobs = _plan(voices=2, per_voice=6, negatives=6)
    silent = next(job.text for job in jobs if job.kind == "positive")
    runaway = next(job.text for job in jobs if job.kind == "hard_negative")
    model = FakeDesign(silent={silent}, runaway={runaway})
    summary = synth.generate(jobs, tmp_path, design_model=model, batch=4, attempts=2, log=_quiet)
    n_silent = sum(job.text == silent for job in jobs)
    n_runaway = sum(job.text == runaway for job in jobs)
    assert summary["rejected"] == {"empty": 2 * n_silent, "runaway": 2 * n_runaway}
    assert summary["failed"] == summary["remaining"] == n_silent + n_runaway
    attempts = [json.loads(line) for line in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert Counter(row["attempt"] for row in attempts) == {
        1: n_silent + n_runaway,
        2: n_silent + n_runaway,
    }
    rows = _rows(tmp_path)
    assert len(rows) == len(jobs) - n_silent - n_runaway
    assert not {row["text"] for row in rows} & {silent, runaway}
    assert all((tmp_path / row["path"]).is_file() for row in rows)


def test_out_of_memory_splits_the_batch_and_a_time_limit_stops_cleanly(tmp_path):
    pytest.importorskip("scipy")
    jobs = _plan()
    stopped = synth.generate(
        jobs, tmp_path, design_model=FakeDesign(), deadline=time.monotonic() - 1, log=_quiet
    )
    assert stopped["stopped"] and stopped["written"] == 0 and stopped["remaining"] == len(jobs)
    model = FakeDesign(oom_above=2)
    summary = synth.generate(jobs, tmp_path, design_model=model, batch=8, log=_quiet)
    assert summary["written"] == len(jobs)
    assert max(len(call["text"]) for call in model.calls) == 2
    with pytest.raises(ValueError, match="no model"):
        synth.generate(_plan(seed=9), tmp_path / "other", log=_quiet)


def test_cloned_voices_get_one_prompt_each_and_are_marked(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    refs_file = tmp_path / "refs.jsonl"
    for name in ("r1", "r2"):
        synth.write_wav(tmp_path / f"{name}.wav", (_burst(1.0, 16_000) * 20000).astype(np.int16))
    refs_file.write_text(
        '{"path": "r1.wav", "speaker": "a", "age_group": "elder"}\n{"path": "r2.wav"}\n'
    )
    voices = synth.clone_voices(synth.read_clone_refs(refs_file), 5, seed=7)
    jobs = synth.plan_jobs(
        voices, synth.text_bank(SPEC), per_voice=3, negatives_per_voice=3, seed=7
    )
    model = FakeClone()
    out = tmp_path / "out"
    summary = synth.generate(jobs, out, clone_model=model, batch=4, log=_quiet)
    assert summary["written"] == 12
    assert model.prompted == [(16_000, 16_000), (16_000, 16_000)]  # once per voice
    assert all(call["non_streaming_mode"] is True for call in model.calls)
    rows = _rows(out)
    assert {row["engine"] for row in rows} == {"qwen3-clone"}
    assert all(row["instruct"] is None and row["clone_ref"].endswith(".wav") for row in rows)
    assert {row["age_group"] for row in rows} == {"elder", "unknown"}


# ── ASR ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "text", "heard", "keep"),
    [
        ("positive", "蛋黃！", "蛋黃。", True),
        ("positive", "蛋黃", "淡黃", True),  # a bare call may be written as its homophone
        ("positive", "蛋黃～", "但黃", True),
        ("positive", "欸，蛋黃！", "欸，蛋黄！", True),
        ("positive", "過來，蛋黃！", "過來淡黃", False),  # in context the characters must be right
        ("positive", "蛋黃", "安黃", False),  # the first syllable was garbled
        ("positive", "蛋黃", "黃", False),
        ("positive", "蛋黃", "", False),
        ("hard_negative", "蛋黃酥", "蛋黃酥", True),
        ("hard_negative", "蛋黃酥", "蛋黃", False),  # collapsed into the bare name
        ("hard_negative", "鹹蛋黃", "淡黃", False),
        ("hard_negative", "蛋", "蛋黃", False),
        ("hard_negative", "蛋", "但", True),
        ("hard_negative", "這個彈簧壞掉了", "這個蛋黃壞掉了", False),  # the name appeared
        ("hard_negative", "這個彈簧壞掉了", "這個彈簧壞掉了", True),
        ("hard_negative", "淡黃色的衣服", "蛋黃色的衣服", True),
        ("hard_negative", "過來", "過來", True),
        ("hard_negative", "過來", "", True),
    ],
)
def test_the_asr_decision(no_pinyin, kind, text, heard, keep):
    assert synth.asr_accepts(kind, text, heard, SPEC) is keep


def test_with_pypinyin_the_asr_decision_compares_toneless_pinyin():
    pytest.importorskip("pypinyin")
    assert synth.asr_accepts("positive", "過來，蛋黃！", "過來淡黃", SPEC)
    assert synth.asr_accepts("positive", "蛋黃", "單黃", SPEC)
    assert not synth.asr_accepts("positive", "蛋黃", "安黃", SPEC)
    assert not synth.asr_accepts("hard_negative", "蛋黃酥", "眈潢", SPEC)
    assert not synth.asr_accepts("hard_negative", "我想喝蛋花湯", "我想喝眈潢湯", SPEC)


def test_the_asr_pass_marks_clips_and_never_deletes_them(tmp_path, no_pinyin):
    pytest.importorskip("scipy")
    synth.generate(_plan(), tmp_path, design_model=FakeDesign(), batch=8, log=_quiet)
    rows = _rows(tmp_path)
    victim = next(
        row for row in rows if row["kind"] == "hard_negative" and "蛋黃" not in row["text"]
    )
    heard = iter("蛋黃" if row is victim else row["text"] for row in rows)
    sizes = []

    def fake_asr(clips):
        sizes.extend(clip.size for clip in clips)
        return [next(heard) for _ in clips]

    counts = synth.transcribe_clips(tmp_path, SPEC, fake_asr, batch=5, log=_quiet)
    assert counts["positive"] == {"ok": 12, "rejected": 0, "pending": 0}
    assert counts["hard_negative"] == {"ok": 11, "rejected": 1, "pending": 0}
    marked = _rows(tmp_path)
    assert [row["clip_id"] for row in marked if row["asr_ok"] is False] == [victim["clip_id"]]
    assert all((tmp_path / row["path"]).is_file() for row in marked)  # marked, never deleted
    assert sizes == pytest.approx([row["duration_s"] * 16_000 for row in rows], abs=8)  # 16 kHz
    again = synth.transcribe_clips(tmp_path, SPEC, lambda clips: pytest.fail("again"), log=_quiet)
    assert again == counts
    samples = synth.listen_samples(tmp_path, 6)
    assert [kind for _, kind, _ in samples] == ["positive", "hard_negative"] * 3
    assert all(Path(path).is_file() for path, _, _ in samples)
    everything = synth.listen_samples(tmp_path, 100)
    assert len(everything) == 23  # all but the clip the ASR rejected
    assert str(tmp_path / victim["path"]) not in {path for path, _, _ in everything}


def test_the_trainer_reads_the_output_and_leaves_out_asr_rejects(tmp_path, no_pinyin):
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    data = pytest.importorskip("duckmind.namekws.data")
    if not hasattr(data, "load_synth"):
        pytest.skip("duckmind.namekws.data has no load_synth")
    synth.generate(_plan(), tmp_path, design_model=FakeDesign(), batch=8, log=_quiet)
    rows = _rows(tmp_path)
    synth.transcribe_clips(tmp_path, SPEC, lambda clips: ["蛋黃"] * len(clips), log=_quiet)
    kept = sum(row["asr_ok"] for row in _rows(tmp_path))
    bank = data.load_synth(tmp_path)
    assert len(bank) == kept == sum(row["kind"] == "positive" for row in rows)
    assert {clip.group for clip in bank.clips} == {row["voice_id"] for row in rows}


# ── Throughput and the command line ──────────────────────────────────────────


def test_throughput_is_logged_with_an_eta_after_the_first_batches():
    now = [0.0]
    lines: list[str] = []
    progress = synth.Progress(100, lines.append, planned=1000, clock=lambda: now[0], every_s=30)
    progress.begin()
    for _ in range(5):
        now[0] += 2.0
        progress.update(10)
    assert len(lines) == 3  # each of the first three batches, then every 30 s
    assert "10/100 clips" in lines[0] and "5.0 clips/s" in lines[0] and "ETA 18 s" in lines[0]
    assert "the whole plan (1,000 clips) takes ~3 min" in lines[2]
    now[0] += 10.0
    progress.update(50)
    assert "100/100 clips" in lines[-1]


def test_a_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "synth"
    synth.main(
        [
            "--name",
            "蛋黃",
            "--pinyin",
            "dan4 huang2",
            "--out",
            str(out),
            "--voices",
            "10",
            "--dry-run",
        ]
    )
    printed = capsys.readouterr().out
    assert "10 designed" in printed and "120 positives + 120 hard negatives" in printed
    assert "d0000" in printed and not out.exists()


def test_the_command_line_runs_end_to_end_with_stand_in_models(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    monkeypatch.setattr(
        synth, "load_tts", lambda model_id, device, attn="auto", seed=0: FakeDesign()
    )
    monkeypatch.setattr(
        synth, "load_transcriber", lambda model_id, device, batch: lambda c: ["蛋黃"] * len(c)
    )
    out = tmp_path / "synth"
    argv = ["--name", "蛋黃", "--pinyin", "dan4 huang2", "--out", str(out), "--voices", "4"]
    argv += ["--per-voice", "3", "--negatives-per-voice", "3", "--batch", "4"]
    synth.main([*argv, "--asr-filter"])
    manifest = json.loads((out / "synth.json").read_text())
    assert manifest["clips"] == {"positive": 12, "hard_negative": 12}
    assert manifest["asr"]["positive"]["ok"] == 12
    assert manifest["asr"]["hard_negative"]["rejected"] == 12
    assert len((out / "voices.jsonl").read_text().splitlines()) == 4
    printed = capsys.readouterr().out
    assert "clips/s" in printed and "ETA" in printed
    monkeypatch.setattr(synth, "load_tts", lambda *a, **k: pytest.fail("nothing left to make"))
    synth.main(argv)  # resumes: everything is there already
    assert json.loads((out / "synth.json").read_text())["clips"]["positive"] == 12
