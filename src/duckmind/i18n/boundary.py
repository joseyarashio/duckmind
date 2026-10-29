"""The API boundary: page words get their message next to them, the words stay as they are.

``annotate(answer, engine="v4_0")`` returns a copy of an API answer in which every object that
holds page words also holds an ``_i18n`` map: field name → the wire message (see
``message.py``) for that field (a list field gets a list, ``None`` where an item has none)::

    {"note": "先張嘴，再咕～一聲",
     "_i18n": {"note": {"k": "core.engine.DuckCore._touch_reaction.4cd0c95a", "p": {…}}}}

The words themselves are never touched, so 繁體中文 stays byte-identical and old page code
keeps working; the page shows the ``_i18n`` message in the other five languages
(``I18N.field(obj, "note")`` in static/i18n.js).

Where a message comes from:

* a :class:`~duckmind.i18n.message.Msg` value (code that already speaks in keys): its own key;
* plain Duck Core words: the engine's pack (``packs.py``) matches them back to the template
  that wrote them — the frozen v4_0 engine can only be translated this way.

Fields that are data are never keyed: the duck's name, what the family typed, raw event JSON,
the memory graph's self node. When identical words were written at several places of the
engine, the field decides which one (a banner tag vs a button label).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from . import packs
from .matcher import NUMBER, TIMEISH
from .message import Msg, to_wire

CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
I18N_FIELD = "_i18n"

# Fields that are data: never keyed (json path patterns; [n] indexes are written [])
DATA = [
    re.compile(p)
    for p in (
        r"(^|\.)identity\.name$",
        r"\.results\[\]\.clause$",
        r"(^|\.)question$",
        r"(^|\.)bonds\.people\[\]\.name$",
        r"\.terms\[\]$",
        r"(^|\.)payload(\.|$|\[)",
    )
]
# Which sites a field's words come from, for identical words written at several places.
PREFER: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(r"banner\.actions\[\]\.label$"),
        ("api._button", "api.banner", "api._nest_banner", "api._nest_off_banner", "MORNING_LABEL"),
    ),
    (
        re.compile(r"banner\.tags\[\]$"),
        ("PHASE_ZH", "FAMILY_ZH", "MOMENT_SHORT", "api.banner", "tags.append"),
    ),
    (
        re.compile(r"banner\.(title|text)$"),
        (
            "api.banner",
            "api._nest",
            "api._wandering",
            "engine.DuckCore.still_reason",
            "engine.DuckCore.walk_reason",
        ),
    ),
    (re.compile(r"body\.reason$"), ("engine.DuckCore.still_reason",)),
    (re.compile(r"events\[\]\.category_zh$"), ("narrate.CATEGORY",)),
    (re.compile(r"events\[\]\.text$"), ("narrate.", "engine.", "content.")),
    (re.compile(r"journal\[\]\.detail$"), ("engine.DuckCore.",)),
    (re.compile(r"journal\[\]\.persona_zh$"), ("content.PERSONAS", "engine.DuckCore._journal")),
    (re.compile(r"(^|\.)note$"), ("engine.DuckCore.",)),
]
SEP_KEYS = {
    "、": "fmt.sep.enum",
    " ／ ": "fmt.sep.slash",
    "；": "fmt.sep.semicolon",
    "，": "fmt.sep.comma",
    " / ": "fmt.sep.slash_ascii",
    ", ": "fmt.sep.comma_ascii",
    "·": "fmt.sep.babble",
    "・": "fmt.sep.dot",
}


def is_data(path: str) -> bool:
    return any(p.search(path) for p in DATA)


class Boundary:
    """Keys the words of one engine's answers."""

    def __init__(self, engine: str | None, *, names: Iterable[str] = ()) -> None:
        self.engine = engine
        self.pack = packs.get(engine) if engine else None
        if self.pack is not None:
            self.pack.matcher.set_names(tuple(names))
        self.misses: dict[str, int] = {}

    # ---------------------------------------------------------------- one string
    def wire_for(self, path: str, text: str, parent: dict[str, Any] | None = None) -> Any:
        """The wire message for the words ``text`` found at ``path`` (None: data, or no match)."""
        if self.pack is None or is_data(path):
            return None
        if parent is not None and parent.get("type") == "self" and path.endswith(".label"):
            return None  # the memory graph's self node is the duck's name
        matcher = self.pack.matcher
        match = matcher.match(text).json()
        if match["status"] != "full":
            if len(self.misses) < 2000:
                self.misses[text[:120]] = self.misses.get(text[:120], 0) + 1
            return None
        if match.get("how") == "exact":
            preferred = self._prefer(path, text)
            if preferred:
                match = {**match, "id": preferred}
        return self._wire(match)

    def _prefer(self, path: str, text: str) -> str | None:
        assert self.pack is not None
        ids = self.pack.matcher.exact.get(text)
        if not ids or len(ids) == 1:
            return None
        for pattern, wanted in PREFER:
            if pattern.search(path):
                for want in wanted:
                    for tid in ids:
                        if want in tid:
                            return tid
        return None

    # ---------------------------------------------------------------- matcher result → wire
    def _wire(self, match: dict[str, Any]) -> Any:
        assert self.pack is not None
        how = match.get("how")
        if how == "exact":
            return {"k": self.pack.key(match["id"])}
        if how == "sounds":
            return self._sounds(str(match["params"]["sounds"]["value"]))
        if how == "segments":
            return {"join": [self._wire(seg) for seg in match["segments"]]}
        if how == "list":
            return self._param(match["params"]["items"], None)
        tid = match.get("id")
        if not tid:
            return None
        rename = self.pack.rename.get(tid, {})
        specs = {p["ph"]: p.get("spec") or "" for p in self.pack.parts_of.get(tid, []) if "ph" in p}
        params = {
            rename.get(name, name): self._param(value, specs.get(name))
            for name, value in (match.get("params") or {}).items()
        }
        out: dict[str, Any] = {"k": self.pack.key(tid)}
        if params:
            out["p"] = params
        return out

    def _sounds(self, text: str) -> Any:
        assert self.pack is not None
        sep = "·" if "·" in text else ("・" if "・" in text else "·")
        tokens = [t for t in re.split(r"[·・]", text)]
        return {
            "list": [
                {"k": self.pack.sound_key[t]} if t in self.pack.sound_key else t for t in tokens
            ],
            "sep": SEP_KEYS[sep],
        }

    def _param(self, param: Any, spec: str | None) -> Any:
        if not isinstance(param, dict):
            return param
        kind = param.get("kind")
        if kind == "empty":
            return ""
        if kind == "number":
            return number_value(str(param.get("value", "")), spec or "")
        if kind in {"time", "name", "code", "text"}:
            return str(param.get("value", ""))
        if kind == "sounds":
            return self._sounds(str(param.get("value", "")))
        if kind == "alt":
            if param.get("id"):
                assert self.pack is not None
                return {"k": self.pack.key(param["id"])}
            return str(param.get("value", ""))
        if kind == "list":
            sep = str(param.get("sep") or "、")
            return {
                "list": [self._param(item, None) for item in param.get("items") or []],
                "sep": SEP_KEYS.get(sep, "fmt.sep.enum"),
            }
        if kind == "cut":
            assert self.pack is not None
            inner = (
                self._wire(param)
                if param.get("how")
                else {"k": self.pack.key(str(param.get("id")))}
            )
            return {"cut": inner, "text": str(param.get("value", ""))}
        if kind == "nested":
            return self._wire(param)
        return str(param.get("value", ""))

    # ---------------------------------------------------------------- whole answers
    def annotate(self, answer: Any) -> Any:
        """A copy of ``answer`` with ``_i18n`` maps next to the words (see the module doc)."""
        return self._walk(answer, "")

    def _walk(self, obj: Any, path: str) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            messages: dict[str, Any] = dict(obj.get(I18N_FIELD) or {})
            for key, value in obj.items():
                if key == I18N_FIELD:
                    continue
                sub = f"{path}.{key}" if path else str(key)
                if isinstance(value, str):
                    wire = self._string(sub, value, obj)
                    if wire is not None and key not in messages:
                        messages[key] = wire
                    out[key] = value
                elif (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, str) or v is None for v in value)
                ):
                    wires = [
                        self._string(f"{sub}[]", v, obj) if isinstance(v, str) else None
                        for v in value
                    ]
                    if any(w is not None for w in wires) and key not in messages:
                        messages[key] = wires
                    out[key] = list(value)
                else:
                    out[key] = self._walk(value, sub)
            if messages:
                out[I18N_FIELD] = messages
            return out
        if isinstance(obj, list):
            return [self._walk(item, f"{path}[]") for item in obj]
        if isinstance(obj, tuple):
            return [self._walk(item, f"{path}[]") for item in obj]
        return obj

    def _string(self, path: str, value: str, parent: dict[str, Any]) -> Any:
        if isinstance(value, Msg):
            return to_wire(value)
        if not CJK.search(value):
            return None
        return self.wire_for(path, value, parent)


def number_value(text: str, spec: str) -> Any:
    """A number the engine printed, as a JSON number when that prints back the same digits
    (so the page can format it per language); otherwise the text itself."""
    text = text.strip()
    if not NUMBER.fullmatch(text) or TIMEISH.fullmatch(text):
        return text
    try:
        value: int | float = int(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError:
            return text
    try:
        again = format(value, spec) if spec else str(value)
    except (ValueError, TypeError):
        return text
    return value if again == text else text


def engine_of(session: Any) -> str | None:
    """Which engine pack reads a Duck Core session's words: "v4_0" for the frozen engine
    (蛋黃), "v4_1" for the current Duck Core, None for anything else."""
    module = type(session).__module__
    if module.startswith("duckmind.engines.v4_0"):
        return "v4_0"
    if module.startswith("duckmind.duckcore"):
        return "v4_1"
    return None


def names_in(answer: Any) -> set[str]:
    """The duck's name(s) an answer carries (``identity.name``): data inside other words."""
    found: set[str] = set()

    def walk(obj: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(obj, dict):
            identity = obj.get("identity")
            if isinstance(identity, dict) and isinstance(identity.get("name"), str):
                found.add(identity["name"])
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
        elif isinstance(obj, list):
            for value in obj[:50]:
                walk(value, depth + 1)

    walk(answer, 0)
    return found


def annotate(answer: Any, engine: str | None = None, *, names: Iterable[str] = ()) -> Any:
    """``Boundary(engine, names=…).annotate(answer)`` — see the module docstring. The duck's
    name is taken from the answer itself (``identity.name``) plus ``names``."""
    return Boundary(engine, names=set(names) | names_in(answer)).annotate(answer)


HOME_FIELDS = ("zh", "label", "name_zh")


def annotate_home(layout: Any, home_id: str | None = None, *, catalogs: Any = None) -> Any:
    """A home layout (world/homes/*.json, /api/home) with ``_i18n`` for its names.

    The layout files are hashed into the home's fingerprint, so their 繁體中文 is never
    edited; each name's key is ``world.homes.<home id>.<path>`` (the path of ids and field
    names inside the file, as the i18n inventory keyed them): ``world.homes.default_home.
    things.ball_felt.zh`` for 「毛線球」. A piece that came from the object library (``lib``) uses the
    library's own key, ``page.lib.item.<lib>``. Only names the catalog has get a message."""
    from .catalog import default
    from .locales import SOURCE

    cats = catalogs or default()
    home = home_id or (layout.get("id") if isinstance(layout, dict) else None)
    if not home:
        return layout

    def walk(obj: Any, prefix: str) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            messages: dict[str, Any] = dict(obj.get(I18N_FIELD) or {})
            for key, value in obj.items():
                if key == I18N_FIELD:
                    continue
                if isinstance(value, str) and key in HOME_FIELDS and CJK.search(value):
                    catalog_key = f"world.homes.{home}.{prefix}{key}"
                    if key == "zh" and obj.get("lib"):
                        # a piece of the object library keeps the library's name in every home
                        catalog_key = f"page.lib.item.{obj['lib']}"
                    # (only a name that still is the catalog's: an edited layout keeps its home's id and a
                    # name of its own, and that name is data)
                    if cats.get(catalog_key, SOURCE) == value:
                        messages[key] = {"k": catalog_key}
                    out[key] = value
                elif isinstance(value, (dict, list)) and key != "doc":
                    out[key] = walk(value, f"{prefix}{key}.")
                else:
                    out[key] = value
            if messages:
                out[I18N_FIELD] = messages
            return out
        if isinstance(obj, list):
            return [
                walk(item, f"{prefix}{item.get('id', index) if isinstance(item, dict) else index}.")
                for index, item in enumerate(obj)
            ]
        return obj

    return walk(layout, "")


def attach(answer: Any) -> Any:
    """Only the ``Msg`` values (no engine matching): for answers made by code that already
    speaks in keys (the simulator server, the neural duck, the world)."""
    return Boundary(None).annotate(answer)
