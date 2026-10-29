"""Synthetic speech for the name detector: the duck's name called by many voices, and speech
that sounds close to it but is not the name (Colab A100, next to Qwen3-TTS).

    python -m duckmind.namekws.synth --name 蛋黃 --pinyin "dan4 huang2" \\
        --out /content/kws/synth --voices 1200 --per-voice 12 --negatives-per-voice 12 \\
        --device cuda:0 --batch 32 [--clone-refs refs.jsonl --clone-voices 300] [--asr-filter]

* **Voices**: seeded Qwen3-TTS VoiceDesign descriptions (age, gender, pitch, pace, mood,
  accent; about 40 % children of 3–9), plus optional x-vector clones of real speakers
  (``--clone-refs``: jsonl of ``{"path", "text"?, "speaker"?}``).
* **Positives**: the name alone or ending a short call (蛋黃！, 過來，蛋黃！), a few with the
  name first; every voice says its own rotating slice of the forms.
* **Hard negatives**, by the same voices: the name's syllables inside other words and
  near-sounds, always with context (蛋黃酥, 淡黃色的衣服, 彈簧床), the single syllables 蛋 and
  黃, and family phrases without the name. :func:`negative_problems` refuses any text that
  would sound like a positive.
* **Clips**: mono 16 kHz 16-bit WAV, trimmed to the speech plus 150 ms, peaked at −3 dBFS.
  Empty, clipped, runaway and implausibly short or long takes go to ``rejected.jsonl`` and
  are tried again (``--attempts``).
* **ASR** (``--asr-filter``): Breeze-ASR-25 transcribes every clip. Positives must be heard
  as the name; hard negatives must not turn into it. Files are never deleted: ``asr_ok``
  marks them, and :func:`duckmind.namekws.data.load_synth` leaves out ``false``.

Output: ``positive/<voice_id>/<k>.wav``, ``hard_negative/<voice_id>/<k>.wav`` and
``metadata.jsonl`` (one row per WAV, paths relative to the output folder), plus
``voices.jsonl`` and ``synth.json``. Running the same command again resumes.
torch, qwen_tts and transformers are imported only when a model is loaded.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import sys
import time
import wave
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

RATE = 16_000
LANGUAGE = "Chinese"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ASR_MODEL = "MediaTek-Research/Breeze-ASR-25"
DESIGN_ENGINE = "qwen3-voicedesign"
CLONE_ENGINE = "qwen3-clone"
CODEC_HZ = 12.5  # Qwen3-TTS 12Hz codec: one frame per 1920 samples at 24 kHz
PEAK_DBFS = -3.0
PAD_S = 0.15
REFERENCE_COUNT = 12  # clips per voice that the text rotation tiles exactly
FORMAT = 1

# ── The name and what sounds like it ─────────────────────────────────────────

# CJK ideographs: extension A, unified and compatibility blocks.
_HANZI = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SYLLABLE = re.compile(r"^[a-z]+[1-5]?$")
_PUNCTUATION = re.compile(r"[，,、；;：:！!？?～~…。.\s]+")

# Characters that sound like each syllable of 蛋黃 whatever the tone, so the checks below work
# without pypinyin. 彈 is left out on purpose: in 彈簧 and 彈性 it is tán.
_SOUNDS_LIKE = {
    "dan": "蛋淡但誕旦單丹擔膽氮啖憚澹萏疸耽鄲撣殫簞",
    "huang": "黃黄皇凰簧煌蝗惶璜磺徨遑湟隍篁晃謊慌荒恍幌",
}
# ...and with the same tone as well: what an ASR may write for a bare call (淡黃).
_SAME_TONE = {"dan4": "蛋淡但誕旦氮啖憚澹萏", "huang2": "黃黄皇凰簧煌蝗惶璜磺徨遑湟隍篁"}
_VARIANTS = {"黃": "黄", "黄": "黃"}  # traditional and simplified spellings of one character
# pypinyin reads Traditional 彈簧 as dàn huáng, which is the name itself; it is tán huáng.
_PINYIN_FIXES = {"彈簧": [["tán"], ["huáng"]]}


def hanzi(text: str) -> str:
    """Only the Chinese characters of ``text``: punctuation, spaces and latin removed."""
    return "".join(_HANZI.findall(text))


def _toneless(syllables: Iterable[str]) -> tuple[str, ...]:
    return tuple(syllable.rstrip("12345") for syllable in syllables)


@dataclass(frozen=True)
class NameSpec:
    """The name as written (蛋黃) and as said, one toned syllable per character."""

    text: str
    syllables: tuple[str, ...]

    @classmethod
    def parse(cls, name: str, pinyin: str) -> NameSpec:
        chars = hanzi(name)
        syllables = tuple(
            item.lower().replace("ü", "v").replace("u:", "v")
            for item in re.split(r"[\s,，]+", pinyin.strip())
            if item
        )
        if not chars:
            raise ValueError(f"the name {name!r} must be written in Chinese characters")
        if len(syllables) != len(chars) or not all(_SYLLABLE.match(item) for item in syllables):
            raise ValueError(f"--pinyin needs one syllable like 'dan4' per character of {chars}")
        return cls(chars, syllables)

    @property
    def toneless(self) -> tuple[str, ...]:
        return _toneless(self.syllables)


@lru_cache(maxsize=1)
def _pinyin_backend() -> Callable[[str], list[str]] | None:
    try:
        from pypinyin import Style, lazy_pinyin, load_phrases_dict
    except ImportError:
        return None
    load_phrases_dict(_PINYIN_FIXES)

    def convert(text: str) -> list[str]:
        return list(
            lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
        )

    return convert


def pinyin_of(text: str) -> tuple[str, ...] | None:
    """Toned syllables (``dan4``) of the Chinese in ``text``; None when pypinyin is missing."""
    convert = _pinyin_backend()
    return None if convert is None else tuple(convert(text))


def _charsets(spec: NameSpec, mode: str) -> list[frozenset[str]]:
    """Per character of the name, what counts as it: ``literal`` (蛋黃/蛋黄), ``sound`` (any
    tone, from the table) or ``tone`` (same syllable and tone)."""
    sets = []
    for char, syllable in zip(spec.text, spec.syllables, strict=True):
        chars = {char, *_VARIANTS.get(char, "")}
        if mode == "sound":
            chars.update(_SOUNDS_LIKE.get(syllable.rstrip("12345"), ""))
        elif mode == "tone":
            chars.update(_SAME_TONE.get(syllable, ""))
        sets.append(frozenset(chars))
    return sets


def _match_at(chars: str, at: int, sets: Sequence[frozenset[str]]) -> bool:
    return at + len(sets) <= len(chars) and all(
        chars[at + offset] in options for offset, options in enumerate(sets)
    )


def _has_name(chars: str, sets: Sequence[frozenset[str]]) -> bool:
    return any(_match_at(chars, at, sets) for at in range(len(chars)))


def _only_name(chars: str, sets: Sequence[frozenset[str]]) -> bool:
    """``chars`` is the name said once or more (蛋黃, 蛋黃蛋黃) and nothing else."""
    size = len(sets)
    return (
        bool(chars)
        and len(chars) % size == 0
        and all(_match_at(chars, at, sets) for at in range(0, len(chars), size))
    )


def _has_run(items: Sequence[str], run: Sequence[str]) -> bool:
    size = len(run)
    return any(tuple(items[at : at + size]) == tuple(run) for at in range(len(items) - size + 1))


def _only_runs(items: Sequence[str], run: Sequence[str]) -> bool:
    size = len(run)
    return (
        bool(items)
        and len(items) % size == 0
        and all(tuple(items[at : at + size]) == tuple(run) for at in range(0, len(items), size))
    )


def _fold_name(chars: str, spec: NameSpec) -> str:
    """Every sound-alike of the name replaced by the name (淡黃你在哪 → 蛋黃你在哪)."""
    sets = _charsets(spec, "sound")
    folded, at = [], 0
    while at < len(chars):
        if _match_at(chars, at, sets):
            folded.append(spec.text)
            at += len(sets)
        else:
            folded.append(chars[at])
            at += 1
    return "".join(folded)


def sounds_like_name(text: str, spec: NameSpec) -> bool:
    """``text`` is nothing but the name, once or repeated, by characters or by pinyin."""
    if _only_name(hanzi(text), _charsets(spec, "sound")):
        return True
    syllables = pinyin_of(text)
    return syllables is not None and _only_runs(_toneless(syllables), spec.toneless)


def mentions_name(text: str, spec: NameSpec) -> bool:
    """The name's sound occurs somewhere in ``text`` (蛋黃酥, 淡黃色 and 單簧管 all count)."""
    if _has_name(hanzi(text), _charsets(spec, "sound")):
        return True
    syllables = pinyin_of(text)
    return syllables is not None and _has_run(_toneless(syllables), spec.toneless)


def heard_name(transcript: str, spec: NameSpec, *, bare: bool) -> bool:
    """A transcript contains the name: its characters; for a bare call also an exact homophone
    (淡黃); with pypinyin, anything with the name's toneless pinyin."""
    chars = hanzi(transcript)
    if _has_name(chars, _charsets(spec, "literal")):
        return True
    if bare and _has_name(chars, _charsets(spec, "tone")):
        return True
    syllables = pinyin_of(transcript)
    return syllables is not None and _has_run(_toneless(syllables), spec.toneless)


def asr_accepts(kind: str, text: str, transcript: str, spec: NameSpec) -> bool:
    """Whether a clip of ``text`` stays in the data after the ASR heard ``transcript``.

    A positive must be heard as the name. A hard negative must not collapse into the bare
    name (蛋黃酥 heard as 蛋黃), and must not gain the name when its text had none of it (a
    misread 彈簧 heard as 蛋黃)."""
    if kind == "positive":
        return heard_name(transcript, spec, bare=sounds_like_name(text, spec))
    if sounds_like_name(transcript, spec):
        return False
    return mentions_name(text, spec) or not mentions_name(transcript, spec)


def negative_problems(text: str, spec: NameSpec, positives: Iterable[str] = ()) -> list[str]:
    """Why ``text`` must not be a hard negative; empty when it is safe.

    A hard negative may hold the name inside a longer word (蛋黃酥) but must never *be* it:
    not its characters, not a homophone of the bare name (淡黃, 單黃), not the name's pinyin
    with or without tones (checked when pypinyin is installed), and not a positive with the
    name swapped for a homophone (淡黃你在哪)."""
    chars = hanzi(text)
    if not chars:
        return ["has no Chinese characters"]
    problems = []
    if _only_name(chars, _charsets(spec, "literal")):
        problems.append("is the name")
    elif _only_name(chars, _charsets(spec, "sound")):
        problems.append("is a homophone of the bare name")
    syllables = pinyin_of(text)
    if syllables is not None and _only_runs(_toneless(syllables), spec.toneless):
        problems.append("has the name's pinyin")
    folded = _fold_name(chars, spec)
    for positive in positives:
        if folded == _fold_name(hanzi(positive), spec):
            problems.append(f"sounds like the positive {positive!r}")
        elif syllables is not None and _toneless(syllables) == _toneless(pinyin_of(positive) or ()):
            problems.append(f"has the pinyin of the positive {positive!r}")
    return problems


# ── Texts ────────────────────────────────────────────────────────────────────

POSITIVE_FORMS: dict[str, tuple[str, ...]] = {
    # The name alone; the punctuation changes the prosody the TTS gives it.
    "alone": ("{name}", "{name}！", "{name}～", "{name}？", "{name}！！"),
    # The name ending a short call.
    "end": tuple(
        "欸，{name}！ 過來，{name}！ 嘿，{name}？ 小{name}～ 喂，{name}！ 來喔，{name}～ "
        "寶貝{name}～ 我的{name}～ 早安，{name}！ 晚安，{name}～ 你看，{name}！ "
        "吃飯囉，{name}！ 好乖喔，{name}～ 我回來了，{name}！".split()
    ),
    # The name first, then a pause or a few words.
    "start": ("{name}你在哪", "{name}，過來", "{name}，吃飯囉", "{name}，來這裡", "{name}你看"),
    "repeat": ("{name}{name}", "{name}！{name}！", "{name}～{name}～"),
}
POSITIVE_SHARES = {"alone": 0.42, "end": 0.36, "start": 0.12, "repeat": 0.10}

# Confusable speech for a name, always with context: its syllables inside other words, and
# near-sounds. Other names get only the single syllables and the family phrases.
CONFUSABLE: dict[str, dict[str, tuple[str, ...]]] = {
    "蛋黃": {
        "embedded": tuple(
            "蛋黃酥 鹹蛋黃 蛋黃哥 把蛋黃打散 今天吃蛋黃酥 "
            "蛋黃酥好好吃 蛋黃哥好可愛 加一點蛋黃醬".split()
        ),
        "near": tuple(
            "淡黃色的衣服 這個淡黃色好看 牆壁是淡黃色的 這個彈簧壞掉了 彈簧床好軟 "
            "我只吃蛋白 蛋白比較好吃 我想喝蛋花湯 生日蛋糕 蛋糕在冰箱裡 把黃色的球給我 "
            "黃色的球滾走了 但是我還沒寫完 當然可以啊 這也太荒唐了 皇上駕到 她在學單簧管 "
            "早餐吃蛋餅 我要一顆滷蛋 不要擔心 黃色小鴨好可愛 黃燈要停下來 聖誕節快到了 "
            "味道有點淡 蛋捲很好吃".split()
        ),
    }
}
# What a family says to a pet without its name, including calls to someone else.
FAMILY = tuple(
    "吃飯了 過來 好乖 你在哪 晚安 早安 你好可愛 不可以喔 等一下 走吧 要睡覺了 我回來了 "
    "你看這裡 好棒喔 來這邊 怎麼了 媽媽在這裡 我們去散步 欸，過來！ 嘿，小鴨！ "
    "寶貝，過來～ 喂，你在哪？ 小鴨鴨～ 布丁，過來！ 豆花～ 湯圓！ "
    "妹妹，吃飯了！ 哥哥，快來！".split()
)
NEGATIVE_SHARES = {"embedded": 0.25, "near": 0.40, "syllable": 0.08, "family": 0.27}


@dataclass(frozen=True)
class TextBank:
    spec: NameSpec
    positives: dict[str, tuple[str, ...]]  # category → texts
    negatives: dict[str, tuple[str, ...]]
    dropped: dict[str, list[str]]  # candidate negative → why it would sound like the name


def text_bank(spec: NameSpec) -> TextBank:
    """Positive and hard-negative texts for a name; unsafe negatives are dropped, not used."""
    positives = {
        category: tuple(form.format(name=spec.text) for form in forms)
        for category, forms in POSITIVE_FORMS.items()
    }
    every_positive = [text for texts in positives.values() for text in texts]
    confusable = CONFUSABLE.get(spec.text, {})
    candidates = {
        "embedded": confusable.get("embedded", ()),
        "near": confusable.get("near", ()),
        "syllable": tuple(dict.fromkeys(char + mark for char in spec.text for mark in ("", "！"))),
        "family": FAMILY,
    }
    negatives: dict[str, tuple[str, ...]] = {}
    dropped: dict[str, list[str]] = {}
    for category, texts in candidates.items():
        kept = []
        for text in texts:
            problems = negative_problems(text, spec, every_positive)
            if problems:
                dropped[text] = problems
            else:
                kept.append(text)
        negatives[category] = tuple(kept)
    return TextBank(spec, positives, negatives, dropped)


def _interleave(shares: Sequence[float], count: int) -> list[int]:
    """Smooth weighted round-robin: every prefix holds each category close to its share."""
    total = sum(shares)
    credit = [0.0] * len(shares)
    order = []
    for _ in range(count):
        for index, share in enumerate(shares):
            credit[index] += share
        best = max(range(len(shares)), key=credit.__getitem__)
        credit[best] -= total
        order.append(best)
    return order


def pick_texts(
    bank: Mapping[str, Sequence[str]], shares: Mapping[str, float], count: int, rotation: int
) -> list[tuple[str, str]]:
    """``count`` (category, text) pairs for one voice. ``rotation`` gives each voice its own
    slice of every category, so the voices together cover all forms evenly."""
    categories = [category for category in shares if bank.get(category)]
    if not categories:
        return []
    weights = [shares[category] for category in categories]
    total = sum(weights)
    used = dict.fromkeys(categories, 0)
    picks = []
    for index in _interleave(weights, count):
        category = categories[index]
        texts = bank[category]
        step = max(1, round(REFERENCE_COUNT * weights[index] / total))
        picks.append((category, texts[(rotation * step + used[category]) % len(texts)]))
        used[category] += 1
    return picks


# ── Voices ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Voice:
    voice_id: str
    engine: str
    instruct: str | None  # the VoiceDesign description; None for a clone
    age_group: str
    gender: str
    timbre: str = ""
    pitch: str = ""
    pace: str = ""
    mood: str = ""
    accent: str = ""
    rotation: int = 0  # which slice of the texts this voice says
    ref_path: str | None = None  # a clone's reference recording


@dataclass(frozen=True)
class _Age:
    key: str
    share: float
    male: str
    female: str
    timbres: tuple[str, ...]
    pitches: tuple[str, ...]


PITCH = {"very_high": "音調很高", "high": "音調偏高", "mid": "音調中等", "low": "音調偏低"}
_CHILD_PITCH = ("very_high", "high")
_GROWN_PITCH = ("high", "mid", "low")
AGES = (
    _Age(
        "child_3_5",
        0.18,
        "一個三到五歲的小男孩",
        "一個三到五歲的小女孩",
        ("奶聲奶氣", "童聲稚嫩", "咬字還不太清楚"),
        _CHILD_PITCH,
    ),
    _Age(
        "child_6_9",
        0.22,
        "一個六到九歲的小男孩",
        "一個六到九歲的小女孩",
        ("童聲清脆", "聲音響亮有活力", "童聲稚嫩"),
        _CHILD_PITCH,
    ),
    _Age(
        "teen",
        0.10,
        "一個國中男生",
        "一個國中女生",
        ("聲音青澀", "聲音清亮", "聲音有點沙啞"),
        _GROWN_PITCH,
    ),
    _Age(
        "young_adult",
        0.20,
        "一位二十多歲的年輕男生",
        "一位二十多歲的年輕女生",
        ("聲音清亮", "聲音溫暖", "聲音柔和", "聲音有點低沉"),
        _GROWN_PITCH,
    ),
    _Age(
        "middle_aged",
        0.17,
        "一位四、五十歲的爸爸",
        "一位四、五十歲的媽媽",
        ("聲音溫暖", "聲音渾厚", "聲音有點沙啞", "聲音爽朗"),
        _GROWN_PITCH,
    ),
    _Age(
        "elder",
        0.13,
        "一位七十歲左右的阿公",
        "一位七十歲左右的阿嬤",
        ("聲音蒼老", "聲音略帶沙啞", "聲音慈祥"),
        _GROWN_PITCH,
    ),
)
PACE = {
    "slow": ("說話很慢", 0.15),
    "slowish": ("說話偏慢", 0.30),
    "normal": ("語速正常", 0.35),
    "quick": ("說話偏快", 0.20),
}
MOOD = {
    "calling": ("正在大聲呼喚", 0.16),
    "happy": ("語氣開心", 0.13),
    "gentle": ("語氣溫柔", 0.12),
    "coaxing": ("用撒嬌的語氣", 0.10),
    "anxious": ("語氣著急", 0.10),
    "tired": ("聽起來很疲倦", 0.08),
    "excited": ("語氣興奮", 0.10),
    "quiet": ("小聲地說", 0.09),
    "next_room": ("從隔壁房間大聲喊，聲音有點遠", 0.12),
}
ACCENT = {
    "taiwan": ("講台灣腔的國語", 0.45),
    "taigi": ("講國語時帶一點台語腔", 0.20),
    "standard": ("講標準國語", 0.20),
    "southern": ("帶台灣南部口音", 0.15),
}
_GOLDEN = 0.6180339887498949


def _weighted(rng: random.Random, table: Mapping[str, tuple[str, float]]) -> str:
    keys = list(table)
    return rng.choices(keys, weights=[table[key][1] for key in keys])[0]


def _draw_voice(rng: random.Random, age: _Age, index: int) -> Voice:
    gender = rng.choice(("male", "female"))
    timbre = rng.choice(age.timbres)
    pitch = rng.choice(age.pitches)
    pace = _weighted(rng, PACE)
    mood = _weighted(rng, MOOD)
    accent = _weighted(rng, ACCENT)
    who = age.male if gender == "male" else age.female
    clauses = (who, timbre, PITCH[pitch], PACE[pace][0], ACCENT[accent][0], MOOD[mood][0])
    return Voice(
        voice_id=f"d{index:04d}",
        engine=DESIGN_ENGINE,
        instruct="，".join(clauses) + "。",
        age_group=age.key,
        gender=gender,
        timbre=timbre,
        pitch=pitch,
        pace=pace,
        mood=mood,
        accent=accent,
        rotation=index,
    )


def design_voices(count: int, seed: int) -> list[Voice]:
    """``count`` VoiceDesign voices. Voice ``i`` depends only on the seed and the voices before
    it, so asking for more voices keeps the first ones. Age groups follow a low-discrepancy
    sequence, so every prefix is close to the shares (40 % children)."""
    phase = random.Random(f"duckmind.namekws.synth:{seed}:ages").random()
    bounds = list(itertools.accumulate(age.share for age in AGES))
    voices: list[Voice] = []
    seen: set[str | None] = set()
    for index in range(count):
        rng = random.Random(f"duckmind.namekws.synth:{seed}:voice:{index}")
        where = (phase + index * _GOLDEN) % 1.0 * bounds[-1]
        age = AGES[min(bisect.bisect_right(bounds, where), len(AGES) - 1)]
        voice = _draw_voice(rng, age, index)
        for _ in range(100):  # a new description for every voice while combinations last
            if voice.instruct not in seen:
                break
            voice = _draw_voice(rng, age, index)
        seen.add(voice.instruct)
        voices.append(voice)
    return voices


@dataclass(frozen=True)
class CloneRef:
    path: str  # as written in the refs file; the voice id derives from it
    resolved: Path
    speaker: str | None = None
    age_group: str = "unknown"
    gender: str = "unknown"


def read_clone_refs(path: Path) -> list[CloneRef]:
    """One ``{"path", "text"?, "speaker"?, "age_group"?, "gender"?}`` per line; relative paths
    are relative to the file. ``text`` is accepted but unused: x-vector prompts ignore it."""
    refs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        given = str(row["path"])
        resolved = Path(given) if Path(given).is_absolute() else path.parent / given
        speaker = row.get("speaker")
        refs.append(
            CloneRef(
                given,
                resolved,
                None if speaker is None else str(speaker),
                str(row.get("age_group") or "unknown"),
                str(row.get("gender") or "unknown"),
            )
        )
    return refs


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clone_voices(refs: Sequence[CloneRef], count: int, seed: int) -> list[Voice]:
    """Up to ``count`` cloned voices, one reference per speaker, chosen by the seed (not by the
    order of the refs file) and named after the reference path."""
    first: dict[str, CloneRef] = {}
    for ref in sorted(refs, key=lambda item: _digest(item.path)):
        first.setdefault(ref.speaker or ref.path, ref)
    chosen = sorted(first.values(), key=lambda item: _digest(f"{seed}:{item.path}"))[:count]
    return [
        Voice(
            voice_id="c" + _digest(ref.path)[:10],
            engine=CLONE_ENGINE,
            instruct=None,
            age_group=ref.age_group,
            gender=ref.gender,
            rotation=int(_digest(ref.path)[:8], 16),
            ref_path=str(ref.resolved),
        )
        for ref in chosen
    ]


# ── The plan ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipJob:
    kind: str  # positive | hard_negative
    voice: Voice
    k: int
    text: str
    category: str  # alone, end, start, repeat | embedded, near, syllable, family

    @property
    def clip_id(self) -> str:
        return f"{self.kind}/{self.voice.voice_id}/{self.k}"

    @property
    def path(self) -> str:
        return f"{self.clip_id}.wav"


def _shuffled(texts: Sequence[str], key: str) -> tuple[str, ...]:
    order = list(texts)
    random.Random(key).shuffle(order)
    return tuple(order)


def plan_jobs(
    voices: Sequence[Voice],
    bank: TextBank,
    *,
    per_voice: int,
    negatives_per_voice: int,
    seed: int,
) -> list[ClipJob]:
    """Every clip to make, voice by voice (positives, then hard negatives). Refuses a bank
    whose positive lacks the name or whose hard negative sounds like a positive."""
    positives = [text for texts in bank.positives.values() for text in texts]
    for text in positives:
        if bank.spec.text not in text:
            raise ValueError(f"the positive {text!r} does not say {bank.spec.text}")
    for texts in bank.negatives.values():
        for text in texts:
            problems = negative_problems(text, bank.spec, positives)
            if problems:
                raise ValueError(f"the hard negative {text!r} {'; '.join(problems)}")
    parts = {
        kind: {
            category: _shuffled(texts, f"duckmind.namekws.synth:{seed}:{kind}:{category}")
            for category, texts in texts_by_category.items()
        }
        for kind, texts_by_category in (
            ("positive", bank.positives),
            ("hard_negative", bank.negatives),
        )
    }
    plan = (
        ("positive", per_voice, POSITIVE_SHARES),
        ("hard_negative", negatives_per_voice, NEGATIVE_SHARES),
    )
    jobs = []
    for voice in voices:
        for kind, count, shares in plan:
            picks = pick_texts(parts[kind], shares, count, voice.rotation)
            for k, (category, text) in enumerate(picks):
                jobs.append(ClipJob(kind, voice, k, text, category))
    return jobs


# ── Audio ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Take:
    """One synthesized clip after post-processing."""

    pcm: np.ndarray | None  # int16 mono at 16 kHz; None when rejected
    reason: str | None  # why it was rejected
    duration_s: float  # of the WAV, or of the raw audio when rejected
    speech_s: float = 0.0  # first to last loud frame


def plausible_speech_s(text: str) -> tuple[float, float]:
    """Bounds on the speech in a take of ``text``: from quick to drawn-out, calm to shouted."""
    syllables = max(1, len(hanzi(text)))
    pauses = sum(1 for match in _PUNCTUATION.finditer(text) if hanzi(text[match.end() :]))
    return 0.12 + 0.06 * syllables, 1.0 + 0.8 * syllables + 0.5 * pauses


def token_budget(texts: Iterable[str], codec_hz: float = CODEC_HZ) -> int:
    """Codec frames for the longest plausible take plus a second, so one runaway item cannot
    hold a whole batch for the models' default of 8192 frames (11 minutes of audio)."""
    longest = max(plausible_speech_s(text)[1] for text in texts)
    return math.ceil((longest + 1.0) * codec_hz)


def is_clipped(audio: np.ndarray, *, level: float = 0.999, fraction: float = 0.001) -> bool:
    """Flat tops at full scale: the Qwen3 decoder clamps to ±1, so overload piles up there."""
    rail = np.abs(audio) >= level
    flat = int(np.count_nonzero(rail[1:] & rail[:-1]))
    return flat >= max(3.0, fraction * audio.size)


def resample(audio: np.ndarray, rate: int, target: int = RATE) -> np.ndarray:
    if rate == target:
        return np.asarray(audio, dtype=np.float64)
    from scipy.signal import resample_poly

    divisor = math.gcd(int(rate), int(target))
    return np.asarray(resample_poly(audio, target // divisor, int(rate) // divisor), np.float64)


def speech_span(
    audio: np.ndarray, *, rel_db: float = 40.0, floor_db: float = -65.0, frame_s: float = 0.01
) -> tuple[int, int] | None:
    """Samples from the first to the last 10 ms frame within ``rel_db`` of the loudest one
    (and above ``floor_db`` dBFS); None when fewer than three frames are that loud."""
    hop = max(1, round(frame_s * RATE))
    count = audio.size // hop
    if count == 0:
        return None
    frames = audio[: count * hop].reshape(count, hop)
    level = 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-20)
    active = np.flatnonzero(level >= max(float(level.max()) - rel_db, floor_db))
    if active.size < 3:
        return None
    return int(active[0]) * hop, min(audio.size, (int(active[-1]) + 1) * hop)


def postprocess(audio: np.ndarray, rate: int, text: str, *, runaway: bool = False) -> Take:
    """Mono, 16 kHz, silence trimmed to 150 ms, peak at −3 dBFS, int16 — or a rejection."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=int(np.argmin(signal.shape)))  # channels are the short axis
    if signal.ndim != 1 or signal.size == 0:
        return Take(None, "empty", 0.0)
    raw_s = signal.size / rate
    if not np.isfinite(signal).all():
        return Take(None, "not_finite", raw_s)
    if runaway:
        return Take(None, "runaway", raw_s)
    if is_clipped(signal):
        return Take(None, "clipped", raw_s)
    signal = resample(signal, rate)
    span = speech_span(signal)
    if span is None:
        return Take(None, "empty", raw_s)
    start, end = span
    speech_s = (end - start) / RATE
    low, high = plausible_speech_s(text)
    if not low <= speech_s <= high:
        return Take(None, "too_short" if speech_s < low else "too_long", raw_s, speech_s)
    pad = round(PAD_S * RATE)
    signal = signal[max(0, start - pad) : min(signal.size, end + pad)]
    signal = signal * (10 ** (PEAK_DBFS / 20) / float(np.abs(signal).max()))
    pcm = np.round(signal * 32767).astype(np.int16)
    return Take(pcm, None, pcm.size / RATE, speech_s)


def write_wav(path: Path, pcm: np.ndarray, rate: int = RATE) -> None:
    """Mono 16-bit PCM, written under a temporary name first so a crash leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with wave.open(str(partial), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    os.replace(partial, path)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV (what this module writes) as float32 mono in [-1, 1], and its rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        channels, rate = handle.getnchannels(), handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    audio = data.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


# ── Metadata ─────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of a jsonl file and how many lines were damaged (unparseable or unterminated)."""
    if not path.is_file():
        return [], 0
    lines = path.read_text(encoding="utf-8").split("\n")
    rows: list[dict[str, Any]] = []
    damaged = 0
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            damaged += 1
            continue
        if not isinstance(row, dict):
            damaged += 1
            continue
        rows.append(row)
        if number == len(lines) - 1:
            damaged += 1  # whole but unterminated: the next append would glue two rows together
    return rows, damaged


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(partial, path)


class Metadata:
    """``metadata.jsonl`` (one row per WAV on disk) and ``rejected.jsonl`` (takes without one).

    Opening it drops a line cut off by a crash and rows whose WAV is gone, and (with
    ``repair``) rewrites the file, so it always parses and always describes existing audio."""

    def __init__(self, out: Path, *, repair: bool = True) -> None:
        self.out = out
        self.path = out / "metadata.jsonl"
        self.rejected_path = out / "rejected.jsonl"
        rows, damaged = _read_jsonl(self.path)
        self.rows = [row for row in rows if (out / str(row.get("path", ""))).is_file()]
        self.repaired = damaged + len(rows) - len(self.rows)
        if repair and self.repaired:
            self.rewrite()
        self._ids = {str(row.get("clip_id") or row["path"][:-4]) for row in self.rows}

    def __contains__(self, clip_id: object) -> bool:
        return clip_id in self._ids

    def append(self, rows: list[dict[str, Any]]) -> None:
        self._append(self.path, rows)
        self.rows.extend(rows)
        self._ids.update(str(row["clip_id"]) for row in rows)

    def reject(self, rows: list[dict[str, Any]]) -> None:
        self._append(self.rejected_path, rows)

    def rewrite(self) -> None:
        _write_jsonl(self.path, self.rows)

    @staticmethod
    def _append(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            handle.flush()
            os.fsync(handle.fileno())


def _row(job: ClipJob, take: Take) -> dict[str, Any]:
    voice = job.voice
    row: dict[str, Any] = {
        "clip_id": job.clip_id,
        "path": job.path,
        "kind": job.kind,
        "category": job.category,
        "text": job.text,
        "voice_id": voice.voice_id,
        "engine": voice.engine,
        "instruct": voice.instruct,
        "age_group": voice.age_group,
        "gender": voice.gender,
        "duration_s": round(take.duration_s, 3),
        "speech_s": round(take.speech_s, 3),
        "asr_text": None,
        "asr_ok": None,
    }
    if voice.ref_path:
        row["clone_ref"] = voice.ref_path
    return row


# ── Throughput ───────────────────────────────────────────────────────────────


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


class Progress:
    """Clips per second over the last batches and an ETA: logged after each of the first
    batches (the third also sizes the whole plan), then every ``every_s`` seconds."""

    def __init__(
        self,
        total: int,
        log: Callable[[str], None] = print,
        *,
        planned: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        every_s: float = 30.0,
        first: int = 3,
    ) -> None:
        self.total = total
        self.planned = planned
        self.log = log
        self.clock = clock
        self.every_s = every_s
        self.first = first
        self.done = 0
        self.batches = 0
        self.started: float | None = None
        self.last_log = -math.inf
        self.window: deque[tuple[float, int]] = deque(maxlen=20)

    def begin(self) -> None:
        if self.started is None:
            self.started = self.clock()
            self.window.append((self.started, 0))

    def update(self, clips: int) -> None:
        self.begin()
        now = self.clock()
        self.done += clips
        self.batches += 1
        self.window.append((now, self.done))
        if (
            self.batches <= self.first
            or now - self.last_log >= self.every_s
            or self.done >= self.total
        ):
            self.last_log = now
            self.log(self.status())

    def rate(self) -> float:
        (then, before), (now, after) = self.window[0], self.window[-1]
        return (after - before) / (now - then) if now > then else 0.0

    def status(self) -> str:
        rate = self.rate()
        left = max(0, self.total - self.done)
        eta = _duration(left / rate) if rate > 0 else "?"
        share = 100.0 * self.done / max(1, self.total)
        text = f"{self.done:,}/{self.total:,} clips ({share:.1f}%), {rate:.1f} clips/s, ETA {eta}"
        if self.batches == self.first and self.planned and rate > 0:
            text += (
                f"; the whole plan ({self.planned:,} clips) takes ~{_duration(self.planned / rate)}"
            )
        return text


# ── Synthesis ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Engine:
    """How a batch of jobs becomes audio: ``call(jobs, max_new_tokens) → (wavs, rate)``."""

    name: str
    call: Callable[[list[ClipJob], int], tuple[list[np.ndarray], int]]
    codec_hz: float = CODEC_HZ


def _codec_hz(model: Any) -> float:
    try:
        tokenizer = model.model.speech_tokenizer
        rate = float(tokenizer.get_output_sample_rate())
        return rate / float(tokenizer.get_decode_upsample_rate())
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return CODEC_HZ


def design_engine(model: Any) -> Engine:
    """Qwen3TTSModel (VoiceDesign): one description per text, all in one call."""

    def call(jobs: list[ClipJob], max_new_tokens: int) -> tuple[list[np.ndarray], int]:
        wavs, rate = model.generate_voice_design(
            text=[job.text for job in jobs],
            instruct=[job.voice.instruct or "" for job in jobs],
            language=[LANGUAGE] * len(jobs),
            max_new_tokens=max_new_tokens,
        )
        return list(wavs), int(rate)

    return Engine(DESIGN_ENGINE, call, _codec_hz(model))


def _load_ref(path: str) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio.mean(axis=1)), int(rate)


def clone_engine(
    model: Any, load_ref: Callable[[str], tuple[np.ndarray, int]] = _load_ref
) -> Engine:
    """Qwen3TTSModel (Base): x-vector-only prompts, made once per voice, mixed in a batch."""
    prompts: dict[str, Any] = {}

    def call(jobs: list[ClipJob], max_new_tokens: int) -> tuple[list[np.ndarray], int]:
        new = list({job.voice.voice_id: job.voice for job in jobs}.values())
        new = [voice for voice in new if voice.voice_id not in prompts]
        if new:
            items = model.create_voice_clone_prompt(
                ref_audio=[load_ref(str(voice.ref_path)) for voice in new],
                x_vector_only_mode=True,
            )
            prompts.update(zip((voice.voice_id for voice in new), items, strict=True))
        wavs, rate = model.generate_voice_clone(
            text=[job.text for job in jobs],
            language=[LANGUAGE] * len(jobs),
            voice_clone_prompt=[prompts[job.voice.voice_id] for job in jobs],
            non_streaming_mode=True,  # the whole text up front, as VoiceDesign does
            max_new_tokens=max_new_tokens,
        )
        return list(wavs), int(rate)

    return Engine(CLONE_ENGINE, call, _codec_hz(model))


def _out_of_memory(error: BaseException) -> bool:
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def _free_cuda() -> None:
    torch = sys.modules.get("torch")  # only when a model already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synthesize(engine: Engine, jobs: list[ClipJob]) -> list[Take]:
    budget = token_budget((job.text for job in jobs), engine.codec_hz)
    try:
        wavs, rate = engine.call(jobs, budget)
    except Exception as error:
        if len(jobs) == 1 or not _out_of_memory(error):
            raise
        _free_cuda()
        half = len(jobs) // 2
        return _synthesize(engine, jobs[:half]) + _synthesize(engine, jobs[half:])
    if len(wavs) != len(jobs):
        raise RuntimeError(f"{engine.name} returned {len(wavs)} clips for {len(jobs)} texts")
    full = budget * rate / engine.codec_hz  # samples of a take that never stopped
    return [
        postprocess(wav, rate, job.text, runaway=np.asarray(wav).shape[-1] >= full - 1)
        for job, wav in zip(jobs, wavs, strict=True)
    ]


def _batches(jobs: Sequence[ClipJob], size: int) -> Iterator[list[ClipJob]]:
    """Plan order, sorted by text length within windows of eight batches (less padding)."""
    window = size * 8
    for start in range(0, len(jobs), window):
        group = sorted(jobs[start : start + window], key=lambda job: len(hanzi(job.text)))
        for offset in range(0, len(group), size):
            yield group[offset : offset + size]


def generate(
    jobs: Sequence[ClipJob],
    out: Path,
    *,
    design_model: Any = None,
    clone_model: Any = None,
    batch: int = 32,
    attempts: int = 2,
    deadline: float | None = None,
    progress: Progress | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Synthesize, post-process and write every job that is not in ``metadata.jsonl`` yet.

    A rejected take is tried again, up to ``attempts`` tries per clip. After ``deadline`` (a
    ``time.monotonic()`` value) no batch starts; running again continues from there."""
    meta = Metadata(out)
    if meta.repaired:
        log(f"metadata: dropped {meta.repaired} cut-off lines or rows without a WAV")
    pending = [job for job in jobs if job.clip_id not in meta]
    engines: dict[str, Engine] = {}
    for name, model, build in (
        (DESIGN_ENGINE, design_model, design_engine),
        (CLONE_ENGINE, clone_model, clone_engine),
    ):
        if any(job.voice.engine == name for job in pending):
            if model is None:
                raise ValueError(f"{name} clips are pending but no model was given for them")
            engines[name] = build(model)
    progress = progress or Progress(len(pending), log)
    rejected: Counter[str] = Counter()
    written = failed = 0
    stopped = False
    for name, engine in engines.items():
        todo = [job for job in pending if job.voice.engine == name]
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                log(f"{name}: trying {len(todo):,} rejected clips again ({attempt}/{attempts})")
            retry: list[ClipJob] = []
            for chunk in _batches(todo, batch):
                if deadline is not None and time.monotonic() >= deadline:
                    stopped = True
                    break
                progress.begin()
                rows, misses = [], []
                for job, take in zip(chunk, _synthesize(engine, chunk), strict=True):
                    if take.pcm is not None:
                        write_wav(out / job.path, take.pcm)
                        rows.append(_row(job, take))
                        continue
                    rejected[str(take.reason)] += 1
                    misses.append(
                        {
                            "clip_id": job.clip_id,
                            "text": job.text,
                            "engine": name,
                            "reason": take.reason,
                            "attempt": attempt,
                            "duration_s": round(take.duration_s, 3),
                            "speech_s": round(take.speech_s, 3),
                        }
                    )
                    retry.append(job)
                meta.append(rows)
                meta.reject(misses)
                written += len(rows)
                progress.update(len(rows) + (len(misses) if attempt == attempts else 0))
            if stopped:
                break
            if attempt == attempts:
                failed += len(retry)
            todo = retry
            if not todo:
                break
        if stopped:
            break
    remaining = sum(1 for job in jobs if job.clip_id not in meta)
    if stopped:
        log(f"stopped at the time limit with {remaining:,} clips to go: run again to continue")
    return {
        "written": written,
        "rejected": dict(rejected),
        "failed": failed,
        "remaining": remaining,
        "stopped": stopped,
    }


# ── ASR filter ───────────────────────────────────────────────────────────────


def _chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _load_clip(path: Path) -> np.ndarray:
    audio, rate = read_wav(path)
    return audio if rate == RATE else resample(audio, rate).astype(np.float32)


def _asr_counts(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        ok = row.get("asr_ok")
        verdict = "pending" if ok is None else "ok" if ok else "rejected"
        kind = counts.setdefault(str(row.get("kind")), {"ok": 0, "rejected": 0, "pending": 0})
        kind[verdict] += 1
    return counts


def transcribe_clips(
    out: Path,
    spec: NameSpec,
    transcribe: Callable[[list[np.ndarray]], list[str]],
    *,
    batch: int = 32,
    log: Callable[[str], None] = print,
    flush_every: int = 20,
) -> dict[str, dict[str, int]]:
    """Transcribe the clips that have no ``asr_text`` yet, then decide ``asr_ok`` for every
    transcribed clip with today's rules. ``metadata.jsonl`` is rewritten atomically every
    ``flush_every`` batches, so an interrupted pass keeps its work. No file is deleted."""
    meta = Metadata(out)
    todo = [row for row in meta.rows if row.get("asr_text") is None]
    progress = Progress(len(todo), log)
    for number, chunk in enumerate(_chunks(todo, batch), start=1):
        progress.begin()
        heard = transcribe([_load_clip(out / str(row["path"])) for row in chunk])
        if len(heard) != len(chunk):
            raise RuntimeError(f"the ASR returned {len(heard)} transcripts for {len(chunk)} clips")
        for row, text in zip(chunk, heard, strict=True):
            row["asr_text"] = str(text).strip()
            row["asr_ok"] = asr_accepts(str(row["kind"]), str(row["text"]), row["asr_text"], spec)
        progress.update(len(chunk))
        if number % flush_every == 0:
            meta.rewrite()
    for row in meta.rows:
        if row.get("asr_text") is not None:
            row["asr_ok"] = asr_accepts(str(row["kind"]), str(row["text"]), row["asr_text"], spec)
    meta.rewrite()
    return _asr_counts(meta.rows)


# ── Notebook helpers ─────────────────────────────────────────────────────────


def listen_samples(out: Path | str, k: int = 6, *, seed: int = 0) -> list[tuple[str, str, str]]:
    """``k`` (path, kind, text) to play in a notebook, alternating positives and hard
    negatives from different voices; clips the ASR rejected are left out."""
    root = Path(out)
    rows, _ = _read_jsonl(root / "metadata.jsonl")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for kind in ("positive", "hard_negative"):
        pool = [row for row in rows if row.get("kind") == kind and row.get("asr_ok") is not False]
        rng.shuffle(pool)
        pools[kind] = pool
    picks = []
    for index in range(k):
        order = ("positive", "hard_negative") if index % 2 == 0 else ("hard_negative", "positive")
        chosen = next((name for name in order if pools[name]), None)
        if chosen is None:
            break
        row = pools[chosen].pop()
        picks.append((str(root / str(row["path"])), chosen, str(row["text"])))
    return picks


# ── Models (imported only here) ──────────────────────────────────────────────


def load_tts(model_id: str, device: str, attn: str = "auto", seed: int = 0) -> Any:
    """A Qwen3-TTS model in bfloat16: VoiceDesign or Base, by ``model_id``."""
    import torch
    from qwen_tts import Qwen3TTSModel
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()  # else every batch logs "Setting pad_token_id…"
    torch.manual_seed(seed)
    options: dict[str, Any] = {"device_map": device, "dtype": torch.bfloat16}
    if attn == "auto":
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else ""
    if attn:
        options["attn_implementation"] = attn
    return Qwen3TTSModel.from_pretrained(model_id, **options)


def load_transcriber(
    model_id: str, device: str, batch: int
) -> Callable[[list[np.ndarray]], list[str]]:
    """Breeze-ASR-25 (Whisper large-v2 for Taiwanese Mandarin) through transformers' pipeline:
    float16 on the GPU, 16 kHz float arrays in, one transcript out per clip."""
    import torch
    from transformers import pipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device, dtype=dtype)
    options: dict[str, Any] = {"language": "zh", "task": "transcribe"}

    def transcribe(clips: list[np.ndarray]) -> list[str]:
        nonlocal options
        inputs = [np.asarray(clip, dtype=np.float32) for clip in clips]
        try:
            results = asr(inputs, batch_size=batch, generate_kwargs=options)
        except ValueError as error:
            if not options or "language" not in str(error):
                raise
            options = {}  # a generation config that cannot force the language: let it detect
            results = asr(inputs, batch_size=batch, generate_kwargs=options)
        return [str(result["text"]).strip() for result in results]

    return transcribe


def _release() -> None:
    gc.collect()
    _free_cuda()


# ── Command line ─────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()

    def log(message: str) -> None:
        print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)

    counts = (args.voices, args.per_voice, args.negatives_per_voice, args.clone_voices)
    if min(counts) < 0 or min(args.batch, args.attempts, args.asr_batch) < 1:
        raise SystemExit("counts cannot be negative; batch sizes and attempts must be ≥ 1")
    spec = NameSpec.parse(args.name, args.pinyin)
    out = Path(args.out)
    bank = text_bank(spec)
    for text, problems in bank.dropped.items():
        log(f"skipped an unsafe hard negative {text!r}: {'; '.join(problems)}")
    if spec.text not in CONFUSABLE:
        log(
            f"no confusable words are built in for {spec.text}: negatives are syllables and phrases"
        )
    has_pinyin = _pinyin_backend() is not None
    voices = design_voices(args.voices, args.seed)
    designed = len(voices)
    if args.clone_refs:
        voices += clone_voices(read_clone_refs(Path(args.clone_refs)), args.clone_voices, args.seed)
    jobs = plan_jobs(
        voices,
        bank,
        per_voice=args.per_voice,
        negatives_per_voice=args.negatives_per_voice,
        seed=args.seed,
    )
    existing = Metadata(out, repair=False)
    pending = [job for job in jobs if job.clip_id not in existing]
    children = sum(voice.age_group.startswith("child") for voice in voices[:designed])
    positives = sum(job.kind == "positive" for job in jobs)
    log(
        f"name {spec.text} ({' '.join(spec.syllables)}); pypinyin "
        + ("installed" if has_pinyin else "missing: the built-in homophone table only")
    )
    log(
        f"voices: {designed} designed ({100 * children / max(1, designed):.0f}% children "
        f"aged 3-9), {len(voices) - designed} cloned"
    )
    log(
        f"clips: {positives:,} positives + {len(jobs) - positives:,} hard negatives; "
        f"{len(jobs) - len(pending):,} done, {len(pending):,} to make"
    )
    if args.dry_run:
        for voice in voices[:3]:
            said = [job.text for job in jobs if job.voice.voice_id == voice.voice_id]
            log(f"{voice.voice_id} {voice.instruct or voice.ref_path}: {' | '.join(said)}")
        return {"voices": len(voices), "clips": len(jobs), "pending": len(pending)}

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "voices.jsonl", (asdict(voice) for voice in voices))
    deadline = started + 60 * args.max_minutes if args.max_minutes > 0 else None
    progress = Progress(len(pending), log, planned=len(jobs))
    generation: dict[str, Any] = {}
    for engine, model_id in ((DESIGN_ENGINE, args.design_model), (CLONE_ENGINE, args.clone_model)):
        group = [job for job in pending if job.voice.engine == engine]
        if not group or (deadline is not None and time.monotonic() >= deadline):
            continue
        log(f"loading {model_id} for {len(group):,} clips")
        model = load_tts(model_id, args.device, args.attn, seed=args.seed)
        generation[engine] = generate(
            group,
            out,
            design_model=model if engine == DESIGN_ENGINE else None,
            clone_model=model if engine == CLONE_ENGINE else None,
            batch=args.batch,
            attempts=args.attempts,
            deadline=deadline,
            progress=progress,
            log=log,
        )
        del model
        _release()
    asr = None
    if args.asr_filter:
        log(f"loading {args.asr_model}")
        transcriber = load_transcriber(args.asr_model, args.device, args.asr_batch)
        asr = transcribe_clips(out, spec, transcriber, batch=args.asr_batch, log=log)
        for kind, verdicts in asr.items():
            log(
                f"ASR {kind}: {verdicts['ok']:,} kept, {verdicts['rejected']:,} marked asr_ok=false"
            )
        del transcriber
        _release()
    rows = Metadata(out, repair=False).rows
    manifest = {
        "format": FORMAT,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "name": spec.text,
        "pinyin": " ".join(spec.syllables),
        "seed": args.seed,
        "voices": {"designed": designed, "cloned": len(voices) - designed, "children": children},
        "per_voice": args.per_voice,
        "negatives_per_voice": args.negatives_per_voice,
        "models": {
            "design": args.design_model,
            "clone": args.clone_model if len(voices) > designed else None,
            "asr": args.asr_model if args.asr_filter else None,
        },
        "pypinyin": has_pinyin,
        "planned": len(jobs),
        "clips": dict(Counter(str(row.get("kind")) for row in rows)),
        "generation": generation,
        "asr": asr,
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / "synth.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"{len(rows):,} clips in {out}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic calls of the duck's name and hard negatives, with Qwen3-TTS"
    )
    parser.add_argument("--name", required=True, help="in Chinese characters, e.g. 蛋黃")
    parser.add_argument("--pinyin", required=True, help='one syllable per character: "dan4 huang2"')
    parser.add_argument("--out", required=True)
    parser.add_argument("--voices", type=int, default=1200, help="VoiceDesign voices")
    parser.add_argument("--per-voice", type=int, default=12, help="positives per voice")
    parser.add_argument("--negatives-per-voice", type=int, default=12)
    parser.add_argument("--clone-refs", default="", help='jsonl of {"path", "text"?, "speaker"?}')
    parser.add_argument("--clone-voices", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=32, help="texts per TTS call")
    parser.add_argument("--attempts", type=int, default=2, help="tries per clip")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-minutes", type=float, default=0.0, help="start no TTS batch after this (0: no limit)"
    )
    parser.add_argument("--asr-filter", action="store_true", help="mark clips with Breeze-ASR-25")
    parser.add_argument("--asr-batch", type=int, default=32)
    parser.add_argument("--design-model", default=DESIGN_MODEL)
    parser.add_argument("--clone-model", default=CLONE_MODEL)
    parser.add_argument("--asr-model", default=ASR_MODEL)
    parser.add_argument(
        "--attn", default="auto", help="auto (flash_attention_2 when installed), sdpa or eager"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
