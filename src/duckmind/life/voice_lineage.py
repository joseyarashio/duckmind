"""Duck Core's voice lineage as a plan for the live voice library (docs/design/GROWTH-G5-2026-10-11.md).

Duck Core keeps what the duck has learned to say: its first word, the phrases it babbled, the ones
the family praised (liked once, promoted after two growth days), the song built from them, and the
age that sets how long a phrase is. This module turns that state, and only that state, into a
**plan**: which clips of the duck's own voice bank the live page plays for each call, how each is
shaped (pitch, length), and which learned phrases and which song are clips of their own.

Pure and deterministic: stdlib only, no clock, no random numbers. The same lineage state, the same
voice seed and the same base bank letters give the same plan and the same :attr:`Plan.key`. Nothing
here writes to Duck Core or changes its numbers, so Duck Core's replay is untouched. What the rules
below are (the numbers in :data:`RULES`) is part of the key through :func:`rules_hash`, together
with :data:`LIBRARY_VERSION` and Duck Core's own content hash: change a number and every duck's
library is built again under a new key, and the old ones are never mixed with the new.

**What the lineage defines, and how it sounds**

* *Age* (growth day): a duckling's call is higher and shorter than an adult's. The pitch falls from
  ``PITCH_YOUNG_ST`` semitones above the voice bank's own at day 0 to the bank's pitch at the adult
  day, and a call grows from ``LENGTH_YOUNG`` of its length to all of it, by the same slope.
* *Repertoire*: the more it has grown, the more variants of a call it owns: two at hatching, one
  more at each stage, and more for the sound it likes best (its favourite tag), for its branch's
  favourite tags and for each sound the phrases it learned are made of. Which variants these are is
  fixed by a hash of the voice seed, and an older duck keeps every variant a younger one had.
* *Learned phrases* (the first word, phrases the family liked, promoted ones, the song's): each is a
  clip of its own, ``ph_<code>``, its syllables the phrase's tags in order (so a phrase of three
  tags is three syllables). A phrase is sung in its own tune, from its code; a question (a last tag
  ``inquire``) rises at its end; a phrase the family praised is held longer; and every generation
  of a phrase's ancestry drifts it by a little, as a phrase that was mutated sounds a little
  different from its parent.
* *The song*: the phrases of Duck Core's song one after the other, a clip ``song``.

Calls Duck Core has no say over stay as they are: ``alarm`` (a fall), ``peck`` (the goodbye), the
``wheee`` ride and the emotions are the voice bank's own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.duckcore import content

API = "lineage-voice-v1"
# Bump when the plan's shape or the meaning of a number below changes (it is in every key).
LIBRARY_VERSION = 1

# The calls whose clips the lineage reshapes. (`wheee` is a ride streamed from the bank's own
# segments, `alarm` is the fall reflex and `peck` the goodbye: Duck Core has no say there.)
LINEAGE_TAGS: tuple[str, ...] = ("chirp", "inquire", "coo", "greet")
# What the learned phrases are made of may also hold a short `wheee` (start and end, no loop).
PHRASE_TAGS: tuple[str, ...] = (*LINEAGE_TAGS, "wheee")
PHRASE_PREFIX = "ph_"
SONG = "song"

ADULT_GD = 180  # content.STAGES: the adult stage
PITCH_YOUNG_ST = 4.0  # semitones above the bank's pitch on growth day 0
LENGTH_YOUNG = 0.80  # a call's length on growth day 0, as a share of the bank's clip
PITCH_STEP = 0.25  # semitones: the plan only changes when a step is crossed
LENGTH_STEP = 0.02
VARIANT_JITTER_ST = (
    0.5  # each variant of a call is this far (at most) above or below the age's pitch
)
REPERTOIRE_BASE = 2
REPERTOIRE_PER_STAGE = 1
REPERTOIRE_FAVOURITE = 2
REPERTOIRE_BRANCH = 1
REPERTOIRE_PER_PHRASE_TAG = 1
REPERTOIRE_PHRASE_CAP = 2
CONTOUR_ST = (
    -0.75,
    -0.25,
    0.25,
    0.75,
)  # a phrase's tune: each syllable one of these, from its code
QUESTION_RISE_ST = 1.0  # a phrase that ends in `inquire` rises at its end
PRAISE_HOLD = 0.04  # a praised phrase is held this much longer per praise ...
PRAISE_HOLD_CAP = 5  # ... up to this many praises
DRIFT_ST = 0.125  # per generation of ancestry: (hash % 9 - 4) steps of this many semitones
DRIFT_DEPTH = 5
SYLLABLE_MAX_S = (
    0.6  # a long coo or greet in a phrase is cut here (with a fade) so a phrase stays a phrase
)
SYLLABLE_GAP_MS = 70
SONG_GAP_MS = 160
LEARNED_MAX = 12  # learned phrases that get a clip, at most
LIKED_MAX = 4  # of the family's liked, not yet promoted, phrases
SONG_MAX_PHRASES = 3  # (Duck Core's own limit)
# how often a learned phrase is chosen when the duck sings or calls out, by how it was learned
WEIGHT = {"promoted": 3, "song": 3, "liked": 2, "first_word": 1}

RULES: dict[str, Any] = {
    "adult_gd": ADULT_GD,
    "pitch_young_st": PITCH_YOUNG_ST,
    "length_young": LENGTH_YOUNG,
    "pitch_step": PITCH_STEP,
    "length_step": LENGTH_STEP,
    "variant_jitter_st": VARIANT_JITTER_ST,
    "repertoire": [
        REPERTOIRE_BASE,
        REPERTOIRE_PER_STAGE,
        REPERTOIRE_FAVOURITE,
        REPERTOIRE_BRANCH,
        REPERTOIRE_PER_PHRASE_TAG,
        REPERTOIRE_PHRASE_CAP,
    ],
    "contour_st": list(CONTOUR_ST),
    "question_rise_st": QUESTION_RISE_ST,
    "praise_hold": [PRAISE_HOLD, PRAISE_HOLD_CAP],
    "drift": [DRIFT_ST, DRIFT_DEPTH],
    "syllable": [SYLLABLE_MAX_S, SYLLABLE_GAP_MS, SONG_GAP_MS],
    "learned": [LEARNED_MAX, LIKED_MAX, SONG_MAX_PHRASES],
    "weight": sorted(WEIGHT.items()),
    "lineage_tags": list(LINEAGE_TAGS),
    "phrase_tags": list(PHRASE_TAGS),
}


def rules_hash() -> str:
    """The numbers above, hashed: part of every plan's key."""
    return hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()[:16]


def _h(*parts: object) -> int:
    """A stable 32-bit number from words and numbers (never Python's randomised ``hash``)."""
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _q(value: float, step: float) -> float:
    return round(round(value / step) * step, 4)


# ---------------------------------------------------------------------------------------- the lineage


@dataclass(frozen=True)
class PhraseView:
    code: str
    tags: tuple[str, ...]
    gen: int
    parents: tuple[str, ...]
    praise: int
    promoted: bool
    liked_days: int


@dataclass(frozen=True)
class LineageView:
    """What of Duck Core's state the voice reads (copied out: nothing here points into it)."""

    seed: int
    gd: int
    stage: str
    branch: str | None
    favorite_tag: str
    first_word: str | None
    phrases: tuple[PhraseView, ...]
    promotions: tuple[str, ...]
    song: tuple[str, ...]
    song_version: int
    molt: str  # "", "open", "kept", "changed"

    @property
    def size(self) -> int:
        """The lineage's size: every phrase Duck Core remembers (babbles, liked, promoted)."""
        return len(self.phrases)

    def to_dict(self) -> dict[str, Any]:
        """Plain data (json), for the gates that carry a lineage between processes."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineageView:
        return cls(
            seed=int(data["seed"]),
            gd=int(data["gd"]),
            stage=str(data["stage"]),
            branch=data["branch"],
            favorite_tag=str(data["favorite_tag"]),
            first_word=data["first_word"],
            phrases=tuple(
                PhraseView(
                    code=str(p["code"]),
                    tags=tuple(p["tags"]),
                    gen=int(p["gen"]),
                    parents=tuple(p["parents"]),
                    praise=int(p["praise"]),
                    promoted=bool(p["promoted"]),
                    liked_days=int(p["liked_days"]),
                )
                for p in data["phrases"]
            ),
            promotions=tuple(data["promotions"]),
            song=tuple(data["song"]),
            song_version=int(data["song_version"]),
            molt=str(data["molt"]),
        )


def view_of(core_state: Any) -> LineageView:
    """Read the lineage out of a Duck Core state (``DuckCore.s``). Read only."""
    phrases: list[PhraseView] = []
    for code, item in sorted(dict(core_state.phrases).items()):
        tags = tuple(str(tag) for tag in item.get("tags", ()))
        if not tags or any(tag not in PHRASE_TAGS for tag in tags):
            continue
        phrases.append(
            PhraseView(
                code=str(code),
                tags=tags,
                gen=int(item.get("gen", 0)),
                parents=tuple(str(parent) for parent in item.get("parents", ())),
                praise=int(item.get("praise", 0)),
                promoted=bool(item.get("promoted", False)),
                liked_days=len(item.get("liked_gds", ())),
            )
        )
    song = core_state.song or {}
    molt = core_state.molt or {}
    gd = int(core_state.gd)
    return LineageView(
        seed=int(core_state.seed),
        gd=gd,
        stage=content.stage_for(gd).stage_id,
        branch=None if core_state.branch is None else str(core_state.branch),
        favorite_tag=str(dict(core_state.quirks).get("favorite_tag", "chirp")),
        first_word=None if core_state.first_word is None else str(core_state.first_word),
        phrases=tuple(phrases),
        promotions=tuple(str(code) for code in core_state.promotions),
        song=tuple(str(code) for code in song.get("phrases", ())),
        song_version=int(song.get("version", 0)),
        molt=str(molt.get("status", "")),
    )


def learned(view: LineageView) -> list[tuple[str, str]]:
    """The phrases that get a clip of their own, as (code, how it was learned), most important
    first: the song's, the promoted (in the order they were), the family's liked, the first word."""
    known = {phrase.code: phrase for phrase in view.phrases}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(code: str, kind: str) -> None:
        if code in known and code not in seen and len(out) < LEARNED_MAX:
            seen.add(code)
            out.append((code, kind))

    for code in view.song[:SONG_MAX_PHRASES]:
        add(code, "song")
    for code in view.promotions:
        add(code, "promoted")
    liked = sorted(
        (p for p in view.phrases if p.liked_days and not p.promoted and p.code not in seen),
        key=lambda p: (-p.praise, p.code),
    )
    for phrase in liked[:LIKED_MAX]:
        add(phrase.code, "liked")
    if view.first_word is not None and view.first_word in content.WORDS:
        tags = tuple(content.WORDS[view.first_word])
        code = content.phrase_code(tags)
        if tags and all(tag in PHRASE_TAGS for tag in tags) and code not in seen:
            # (a first word is a call even when Duck Core never listed it among the phrases)
            if len(out) < LEARNED_MAX:
                out.append((code, "first_word"))
    return out


def _phrase(view: LineageView, code: str) -> PhraseView:
    for phrase in view.phrases:
        if phrase.code == code:
            return phrase
    word = content.WORDS.get(view.first_word or "", ())
    return PhraseView(code, tuple(word), 0, (), 0, False, 0)


# ---------------------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class Part:
    """One syllable: a clip of the bank (``tag`` ``letter``), shaped: ``st`` semitones up,
    ``length`` times as long, cut at ``max_s`` seconds (0: whole), then ``gap_ms`` of quiet."""

    tag: str
    letter: str
    st: float
    length: float
    max_s: float = 0.0
    gap_ms: int = 0


@dataclass(frozen=True)
class Variant:
    letter: str  # the variant's name in the library: the file is <entry>_<letter>.wav
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class Entry:
    name: str  # the tag the page asks for: chirp ... | ph_<code> | song
    kind: str  # tag | first_word | liked | promoted | song_phrase | song
    code: str  # a phrase's code ("" for a tag and the song)
    syllables: tuple[str, ...]  # the tags it is made of, in order (the words on the bubble)
    gen: int
    praise: int
    weight: int
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Plan:
    version: int
    rules: str
    content: str
    voice_seed: int
    gd: int
    stage: str
    entries: tuple[Entry, ...]

    @property
    def key(self) -> str:
        """The library's name on disk and in URLs: what sounds, hashed. (The growth day and the
        stage are only labels: a day that changes no clip keeps the key, and the library.)"""
        data = self.to_dict()
        del data["gd"], data["stage"]
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    @property
    def key8(self) -> str:
        return self.key[:8]

    def entry(self, name: str) -> Entry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": API,
            "version": self.version,
            "rules": self.rules,
            "content": self.content,
            "voice_seed": self.voice_seed,
            "gd": self.gd,
            "stage": self.stage,
            "entries": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "code": e.code,
                    "syllables": list(e.syllables),
                    "gen": e.gen,
                    "praise": e.praise,
                    "weight": e.weight,
                    "variants": [
                        {
                            "letter": v.letter,
                            "parts": [
                                [p.tag, p.letter, p.st, p.length, p.max_s, p.gap_ms]
                                for p in v.parts
                            ],
                        }
                        for v in e.variants
                    ],
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        entries = tuple(
            Entry(
                name=str(e["name"]),
                kind=str(e["kind"]),
                code=str(e["code"]),
                syllables=tuple(e["syllables"]),
                gen=int(e["gen"]),
                praise=int(e["praise"]),
                weight=int(e["weight"]),
                variants=tuple(
                    Variant(
                        str(v["letter"]),
                        tuple(
                            Part(
                                str(p[0]),
                                str(p[1]),
                                float(p[2]),
                                float(p[3]),
                                float(p[4]),
                                int(p[5]),
                            )
                            for p in v["parts"]
                        ),
                    )
                    for v in e["variants"]
                ),
            )
            for e in data["entries"]
        )
        return cls(
            version=int(data["version"]),
            rules=str(data["rules"]),
            content=str(data["content"]),
            voice_seed=int(data["voice_seed"]),
            gd=int(data["gd"]),
            stage=str(data["stage"]),
            entries=entries,
        )


def age_pitch(gd: int) -> float:
    """Semitones above the bank's pitch for a duck of this growth day (the plan's steps)."""
    share = min(max(int(gd), 0), ADULT_GD) / ADULT_GD
    return _q(PITCH_YOUNG_ST * (1.0 - share), PITCH_STEP)


def age_length(gd: int) -> float:
    """A call's length as a share of the bank's clip for a duck of this growth day."""
    share = min(max(int(gd), 0), ADULT_GD) / ADULT_GD
    return _q(LENGTH_YOUNG + (1.0 - LENGTH_YOUNG) * share, LENGTH_STEP)


def stage_index(stage: str) -> int:
    return [s.stage_id for s in content.STAGES].index(stage)


def _drift(view: LineageView, code: str) -> float:
    """A phrase drifts a little from its parents' at each generation of its ancestry."""
    known = {p.code: p for p in view.phrases}
    total = 0
    frontier = [code]
    seen: set[str] = set()
    for _ in range(DRIFT_DEPTH):
        nxt: list[str] = []
        for c in frontier:
            if c in seen or c not in known:
                continue
            seen.add(c)
            if known[c].gen > 0 or known[c].parents:
                total += _h("drift", c) % 9 - 4
            nxt.extend(known[c].parents)
        frontier = nxt
        if not frontier:
            break
    return _q(total * DRIFT_ST, DRIFT_ST)


def _owned_letters(
    view: LineageView, vseed: int, tag: str, letters: Sequence[str], phrase_tags: Iterable[str]
) -> list[str]:
    """The variants of a call the duck owns: as many as its growth, its likes and its phrases give,
    in a fixed shuffled order of the bank's letters (so a grown duck keeps what a young one had)."""
    order = sorted(letters, key=lambda letter: _h("owns", vseed, tag, letter))
    n = REPERTOIRE_BASE + REPERTOIRE_PER_STAGE * stage_index(view.stage)
    if tag == view.favorite_tag:
        n += REPERTOIRE_FAVOURITE
    if view.branch is not None and tag in content.BRANCH_BY_ID[view.branch].favorite_tags:
        n += REPERTOIRE_BRANCH
    used = sum(1 for t in phrase_tags if t == tag)
    n += min(REPERTOIRE_PHRASE_CAP, REPERTOIRE_PER_PHRASE_TAG * used)
    return order[: max(1, min(len(order), n))]


def _syllables(
    view: LineageView,
    vseed: int,
    code: str,
    tags: Sequence[str],
    owned: Mapping[str, Sequence[str]],
    bank: Mapping[str, Sequence[str]],
    *,
    jitter: float,
    hold: float,
    last_gap_ms: int,
) -> tuple[Part, ...]:
    base = age_pitch(view.gd) + _drift(view, code) + jitter
    length = age_length(view.gd) * hold
    parts: list[Part] = []
    for index, tag in enumerate(tags):
        letters = owned.get(tag) or bank.get(tag) or ("a",)
        letter = letters[_h("sing", vseed, code, index) % len(letters)]
        st = base + CONTOUR_ST[_h("tune", code, index) % len(CONTOUR_ST)]
        if index == len(tags) - 1 and tag == "inquire":
            st += QUESTION_RISE_ST
        last = index == len(tags) - 1
        parts.append(
            Part(
                tag,
                letter,
                _q(st, PITCH_STEP / 2),
                _q(length, LENGTH_STEP / 2),
                SYLLABLE_MAX_S,
                last_gap_ms if last else SYLLABLE_GAP_MS,
            )
        )
    return tuple(parts)


def plan_for(
    view: LineageView,
    vseed: int,
    bank: Mapping[str, Sequence[str]],
) -> Plan:
    """The library for this lineage: ``bank`` is the voice bank's letters per tag (what upstream
    rendered: ``chirp`` a–l ...). Deterministic."""
    chosen = learned(view)
    phrase_tags = [tag for code, _ in chosen for tag in _phrase(view, code).tags]
    owned: dict[str, list[str]] = {}
    entries: list[Entry] = []
    pitch = age_pitch(view.gd)
    length = age_length(view.gd)
    for tag in LINEAGE_TAGS:
        letters = list(bank.get(tag, ()))
        if not letters:
            continue
        owned[tag] = _owned_letters(view, vseed, tag, letters, phrase_tags)
        variants = tuple(
            Variant(
                letter,
                (
                    Part(
                        tag,
                        letter,
                        _q(
                            pitch
                            + (_h("pitch", vseed, tag, letter) % 5 - 2) * VARIANT_JITTER_ST / 2,
                            PITCH_STEP,
                        ),
                        length,
                    ),
                ),
            )
            for letter in sorted(owned[tag])
        )
        entries.append(Entry(tag, "tag", "", (tag,), 0, 0, 1, variants))
    wheee_letters = tuple(bank.get("wheee", ()))
    bank_for_phrases = {**{t: list(v) for t, v in bank.items()}, "wheee": list(wheee_letters)}
    song_set = set(view.song[:SONG_MAX_PHRASES])
    song_parts: list[Part] = []
    song_syllables: list[str] = []
    for code, kind in chosen:
        phrase = _phrase(view, code)
        if any(tag == "wheee" for tag in phrase.tags) and not wheee_letters:
            continue
        hold = 1.0 + PRAISE_HOLD * min(phrase.praise, PRAISE_HOLD_CAP)
        variants = tuple(
            Variant(
                letter,
                _syllables(
                    view,
                    vseed,
                    code,
                    phrase.tags,
                    owned,
                    bank_for_phrases,
                    jitter=jitter,
                    hold=hold,
                    last_gap_ms=0,
                ),
            )
            for letter, jitter in (("a", 0.0), ("b", -0.5 if kind != "first_word" else 0.5))
        )
        entries.append(
            Entry(
                PHRASE_PREFIX + code,
                kind if kind != "song" else "song_phrase",
                code,
                tuple(phrase.tags),
                phrase.gen,
                phrase.praise,
                WEIGHT[kind],
                variants,
            )
        )
        if code in song_set:
            sung = _syllables(
                view,
                vseed,
                code,
                phrase.tags,
                owned,
                bank_for_phrases,
                jitter=0.0,
                hold=hold,
                last_gap_ms=SONG_GAP_MS,
            )
            song_parts.extend(sung)
            song_syllables.extend(phrase.tags)
    if song_parts:
        last = song_parts[-1]
        song_parts[-1] = Part(last.tag, last.letter, last.st, last.length, last.max_s, 0)
        entries.append(
            Entry(
                SONG, "song", "", tuple(song_syllables), 0, 0, 1, (Variant("a", tuple(song_parts)),)
            )
        )
    return Plan(
        version=LIBRARY_VERSION,
        rules=rules_hash(),
        content=content.table_hash(),
        voice_seed=int(vseed),
        gd=int(view.gd),
        stage=view.stage,
        entries=tuple(entries),
    )


def phrase_entries(plan: Plan) -> list[Entry]:
    """The learned phrases' clips (not the song), the order they were planned."""
    return [e for e in plan.entries if e.name.startswith(PHRASE_PREFIX)]
