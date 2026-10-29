"""The tools the duck's cloud brain may call, declared once.

The cloud model never moves the body and never speaks: it can only **propose**. Five tools, all
``NON_BLOCKING`` (the conversation goes on while the body decides), declared here as plain
dictionaries — the SDK takes them as they are — so a cassette records exactly these shapes and
the tests need no SDK:

* ``vocalize(tags: [{tag, level 0..1, at_ms}])`` — which of the duck's own calls to make, how
  strongly, and how many milliseconds from now (the duck's mouth is the official voice bank:
  the tag is a name of a bank call, never words);
* ``attend(target_kind: person|thing|sound, hint)`` — what to pay attention to;
* ``recognize(name)`` — bind a name the family gave to the person in view right now;
* ``remember(note)`` — something worth keeping;
* ``feel(valence -1..1, talk 0..1)`` — how the model reads the moment: mood and wish to talk.

LV1 only records and queues what the model calls (:class:`ParsedCall`); acting on it, and
refusing what the voice bank or the rate limits do not allow, is LV2. :func:`parse_call` never
refuses a call: it normalises the arguments (clamped, trimmed) and lists what was wrong
(``problems``) and which tags are not in the bank (``off_bank``), so the later grounding check
and the metrics have the raw material.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

TOOL_SCHEMA_VERSION = "tools-1"
# the official voice bank's calls (simweb/voice.py TAGS; tests/test_live_tools.py keeps them equal)
BANK_TAGS: tuple[str, ...] = ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee")
TARGET_KINDS: tuple[str, ...] = ("person", "thing", "sound")
MAX_TAGS = 4
MAX_AT_MS = 10_000
MAX_TEXT = 200

VOCALIZE = "vocalize"
ATTEND = "attend"
RECOGNIZE = "recognize"
REMEMBER = "remember"
FEEL = "feel"
TOOL_NAMES: tuple[str, ...] = (VOCALIZE, ATTEND, RECOGNIZE, REMEMBER, FEEL)

_TAG_HELP = (
    "alarm（警戒、嚇到）、greet（打招呼）、inquire（疑問、好奇）、peck（輕輕啄一下的聲音）、"
    "chirp（短短的啾）、coo（咕咕，安心）、wheee（興奮拖長，玩的時候才用）"
)

_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": VOCALIZE,
        "behavior": "NON_BLOCKING",
        "description": (
            "提議小鴨叫一聲或幾聲。小鴨的嘴只有鴨叫聲庫，所以只能選這些叫法：" + _TAG_HELP + "。"
            "這只是提議，小鴨自己的身體會決定叫不叫。不要編造聲音庫裡沒有的叫法。"
        ),
        "parameters": {
            "type": "OBJECT",
            "required": ["tags"],
            "properties": {
                "tags": {
                    "type": "ARRAY",
                    "description": "要叫的聲音，依序排好，最多 4 個。",
                    "items": {
                        "type": "OBJECT",
                        "required": ["tag", "level"],
                        "properties": {
                            "tag": {"type": "STRING", "description": "叫法的名字（見上面的清單）"},
                            "level": {"type": "NUMBER", "description": "強度 0 到 1，1 是最用力"},
                            "at_ms": {
                                "type": "INTEGER",
                                "description": "從現在起幾毫秒後叫，0 是馬上",
                            },
                        },
                    },
                }
            },
        },
    },
    {
        "name": ATTEND,
        "behavior": "NON_BLOCKING",
        "description": (
            "提議小鴨把注意力放在某個人、某個東西或某個聲音上（牠會自己決定要不要轉過去）。"
            "只提議你真的聽到或看到的。"
        ),
        "parameters": {
            "type": "OBJECT",
            "required": ["target_kind"],
            "properties": {
                "target_kind": {
                    "type": "STRING",
                    "description": "person（人）、thing（東西）或 sound（聲音）",
                },
                "hint": {
                    "type": "STRING",
                    "description": "是哪一個？例如「媽媽」「玩具」「敲門聲」",
                },
            },
        },
    },
    {
        "name": RECOGNIZE,
        "behavior": "NON_BLOCKING",
        "description": (
            "家人說「這是某某」介紹一個人時，把這個名字和小鴨眼睛畫面裡目前看到的這個人綁在一起。"
            "名字用繁體中文照家人說的寫，不要自己改名字。沒有人在畫面裡，或沒有人介紹，就不要呼叫。"
        ),
        "parameters": {
            "type": "OBJECT",
            "required": ["name"],
            "properties": {"name": {"type": "STRING", "description": "家人給的名字"}},
        },
    },
    {
        "name": REMEMBER,
        "behavior": "NON_BLOCKING",
        "description": "記下一件值得小鴨記住的事，一句話，繁體中文。只記家人真的說過或畫面真的有的事。",
        "parameters": {
            "type": "OBJECT",
            "required": ["note"],
            "properties": {"note": {"type": "STRING", "description": "要記的事，一句話"}},
        },
    },
    {
        "name": FEEL,
        "behavior": "NON_BLOCKING",
        "description": "告訴小鴨的身體你覺得現在的氣氛怎樣：心情好不好，以及牠有多想說話。",
        "parameters": {
            "type": "OBJECT",
            "required": ["valence", "talk"],
            "properties": {
                "valence": {"type": "NUMBER", "description": "心情，-1（很不舒服）到 1（很開心）"},
                "talk": {"type": "NUMBER", "description": "想說話的程度，0（安靜）到 1（很想叫）"},
            },
        },
    },
)


def declarations() -> list[dict[str, Any]]:
    """The tool declarations (function declarations), a fresh copy each call."""
    return copy.deepcopy(list(_DECLARATIONS))


def tools_config() -> list[dict[str, Any]]:
    """The ``tools`` field of a session setup."""
    return [{"function_declarations": declarations()}]


def schema_hash() -> str:
    """A fingerprint of the declarations (a changed tool changes it, whatever the version says)."""
    text = json.dumps(_DECLARATIONS, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ParsedCall:
    """One tool call as the model made it and as the body will read it."""

    name: str
    raw: dict[str, Any]  # the arguments exactly as received
    args: dict[str, Any]  # normalised: clamped, trimmed, missing parts left out
    problems: tuple[str, ...] = ()  # what was wrong (the call is still recorded)
    off_bank: tuple[str, ...] = ()  # vocalize tags the voice bank does not have
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.problems

    def view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "raw": self.raw,
            "problems": list(self.problems),
            "off_bank": list(self.off_bank),
        }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned[:MAX_TEXT] if cleaned else None


def parse_call(name: str, args: Mapping[str, Any] | None) -> ParsedCall:
    """Read one call. Never raises, never drops a call: a bad one carries its ``problems``."""
    raw = dict(args or {})
    problems: list[str] = []
    off_bank: list[str] = []
    out: dict[str, Any] = {}
    if name == VOCALIZE:
        tags_in = raw.get("tags")
        if not isinstance(tags_in, (list, tuple)) or not tags_in:
            problems.append("tags_missing")
        else:
            if len(tags_in) > MAX_TAGS:
                problems.append("too_many_tags")
            tags: list[dict[str, Any]] = []
            for item in list(tags_in)[:MAX_TAGS]:
                if not isinstance(item, Mapping):
                    problems.append("tag_not_object")
                    continue
                tag = _text(item.get("tag"))
                if tag is None:
                    problems.append("tag_missing")
                    continue
                level = _number(item.get("level"))
                if level is None:
                    problems.append("level_missing")
                    level = 0.5
                elif not 0.0 <= level <= 1.0:
                    problems.append("level_out_of_range")
                at_ms = _number(item.get("at_ms", 0))
                if at_ms is None:
                    problems.append("at_ms_bad")
                    at_ms = 0.0
                elif not 0 <= at_ms <= MAX_AT_MS:
                    problems.append("at_ms_out_of_range")
                if tag not in BANK_TAGS:
                    off_bank.append(tag)
                tags.append(
                    {
                        "tag": tag,
                        "level": round(_clamp(level, 0.0, 1.0), 3),
                        "at_ms": int(_clamp(at_ms, 0, MAX_AT_MS)),
                    }
                )
            if tags:
                out["tags"] = tags
            elif "tags_missing" not in problems:
                problems.append("tags_missing")
    elif name == ATTEND:
        kind = _text(raw.get("target_kind"))
        if kind is None:
            problems.append("target_kind_missing")
        elif kind not in TARGET_KINDS:
            problems.append("target_kind_unknown")
            out["target_kind"] = kind
        else:
            out["target_kind"] = kind
        hint = _text(raw.get("hint"))
        if hint is not None:
            out["hint"] = hint
    elif name == RECOGNIZE:
        who = _text(raw.get("name"))
        if who is None:
            problems.append("name_missing")
        else:
            out["name"] = who
    elif name == REMEMBER:
        note = _text(raw.get("note"))
        if note is None:
            problems.append("note_missing")
        else:
            out["note"] = note
    elif name == FEEL:
        for key, low in (("valence", -1.0), ("talk", 0.0)):
            number = _number(raw.get(key))
            if number is None:
                problems.append(f"{key}_missing")
                continue
            if not low <= number <= 1.0:
                problems.append(f"{key}_out_of_range")
            out[key] = round(_clamp(number, low, 1.0), 3)
    else:
        problems.append("unknown_tool")
    return ParsedCall(
        name=str(name),
        raw=raw,
        args=out,
        problems=tuple(dict.fromkeys(problems)),
        off_bank=tuple(off_bank),
    )


__all__ = [
    "ATTEND",
    "BANK_TAGS",
    "FEEL",
    "MAX_AT_MS",
    "MAX_TAGS",
    "RECOGNIZE",
    "REMEMBER",
    "TARGET_KINDS",
    "TOOL_NAMES",
    "TOOL_SCHEMA_VERSION",
    "VOCALIZE",
    "ParsedCall",
    "declarations",
    "parse_call",
    "schema_hash",
    "tools_config",
]
