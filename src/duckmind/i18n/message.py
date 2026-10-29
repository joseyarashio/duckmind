"""Messages: words for the page as a catalog key plus values, in any of the six languages.

``msg(key, **values)`` returns a :class:`Msg`. A Msg *is* a ``str`` — its text is the
繁體中文 source rendered from the catalog, byte for byte what the old f-string printed — so
every old caller (``+``, ``in``, ``==``, ``json.dumps``, the tests that pin the words) keeps
working. It also remembers its key and values, so the server can send them to the page next
to the text (``wire()``), and the page shows them in the viewer's language::

    m = msg("neuro.rx.lean_in.felt_at", site=msg("neuro.touch_site.head"), felt=msg(...))
    str(m)            # '頭頂被摸，靠過去'
    m.wire()          # {'k': 'neuro.rx.lean_in.felt_at', 'p': {'site': {'k': …}, 'felt': {…}}}
    render(m, "en")   # the English catalog's words

Values: numbers (formatted per language), plain ``str`` (data shown as is: names, what the
family typed, raw errors), another Msg (nested words), ``lst([...])`` (a list joined by the
language's own separator), ``join(a, b)`` (words glued together).

The wire form (JSON, the same in ``static/i18n.js``)::

    {"k": key, "p": {name: value}}     a message ("p" left out when empty)
    "text"                             data, shown as is
    1.5                                a number
    {"list": [value, …], "sep": key}   a list, joined by the separator message ``sep``
    {"join": [value, …]}               values glued together
    {"cut": value, "text": "…"}        a piece the source cut from longer words (the source
                                       shows "text"; other languages show the whole value)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import icu
from .catalog import Catalogs, default
from .locales import PSEUDO, SOURCE

DEFAULT_SEP = "fmt.sep.enum"  # 「、」 in the source


class MissingKey(KeyError):
    pass


class Msg(str):
    """Page words: a catalog key and its values. ``str(msg)`` is the 繁體中文 text."""

    key: str
    params: dict[str, Any]

    def __new__(
        cls, key: str, params: Mapping[str, Any] | None = None, text: str | None = None
    ) -> Msg:
        values = dict(params or {})
        if text is None:
            text = _source_text(key, values)
        self = super().__new__(cls, text)
        self.key = key
        self.params = values
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (self.key, self.params, str(self)))

    def __copy__(self) -> Msg:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Msg:
        return self

    def wire(self) -> Any:
        return to_wire(self)

    def render(self, lang: str) -> str:
        return render(self, lang)


def _rebuild(key: str, params: dict[str, Any], text: str) -> Msg:
    return Msg(key, params, text)


def msg(key: str, /, **params: Any) -> Msg:
    """Words from the catalog: ``msg("page.x.y", n=3, name="蛋黃")``."""
    return Msg(key, params)


def content_msg(key: str, words: str) -> str:
    """Duck Core's words ``words`` as the catalog message ``key`` when the catalog says exactly
    those words (the same 繁體中文 text, now translatable); else the plain words."""
    try:
        said = msg(key)
    except KeyError:
        return words
    return said if str(said) == words else words


def from_wire(value: Any) -> Any:
    """The words of a wire form (``to_wire``'s) as a Msg again: for words that crossed a process
    boundary with their ``_i18n`` message and are joined into other words on this side."""
    if isinstance(value, dict):
        if "k" in value:
            params = {n: from_wire(v) for n, v in (value.get("p") or {}).items()}
            return Msg(str(value["k"]), params)
        if "list" in value:
            return lst([from_wire(v) for v in value["list"]], str(value.get("sep") or DEFAULT_SEP))
        if "join" in value:
            return join(*[from_wire(v) for v in value["join"]])
        if "cut" in value:
            return from_wire(value["cut"])
    return value


def carried_words(answer: Mapping[str, Any], field: str) -> Any:
    """``answer[field]`` as a Msg when the answer carries its ``_i18n`` message and that message
    still says exactly these words; else the words as they are."""
    words = answer.get(field)
    wire = (answer.get("_i18n") or {}).get(field) if isinstance(answer.get("_i18n"), dict) else None
    if not isinstance(words, str) or not isinstance(wire, dict):
        return words
    try:
        said = from_wire(wire)
    except Exception:
        return words
    return said if isinstance(said, Msg) and str(said) == words else words


def join(*parts: Any) -> Msg:
    """Values glued together (the source text is their texts one after another)."""
    items = [p for p in parts if p is not None and p != ""]
    return Msg("@join", {"parts": items}, "".join(_source_of(p) for p in items))


def lst(items: Iterable[Any], sep: str = DEFAULT_SEP) -> Msg:
    """A list joined by the language's own separator message (``fmt.sep.enum`` = 「、」)."""
    values = [i for i in items if i is not None and i != ""]
    joiner = _source_message(sep, strict=False) or ""
    return Msg("@list", {"items": values, "sep": sep}, joiner.join(_source_of(v) for v in values))


def _source_of(value: Any) -> str:
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return icu.format_plain(value, SOURCE)
    if isinstance(value, Mapping):
        return render(value, SOURCE)
    if value is None:
        return ""
    return str(value)


_catalogs: Catalogs | None = None


def use_catalogs(catalogs: Catalogs | None) -> None:
    """Point the message layer at other catalogs (tests); None goes back to the package's."""
    global _catalogs
    _catalogs = catalogs


def _cats() -> Catalogs:
    return _catalogs or default()


def _source_message(key: str, *, strict: bool) -> str | None:
    text = _cats().get(key, SOURCE)
    if text is None and strict:
        raise MissingKey(key)
    return text


def _source_text(key: str, params: dict[str, Any]) -> str:
    """The 繁體中文 text of a new Msg. A key missing from the catalog does not crash the
    duck: its text is the key itself (the i18n tests fail on it)."""
    if key == "@join":
        return "".join(_source_of(p) for p in params.get("parts", []))
    if key == "@list":
        joiner = _source_message(str(params.get("sep", DEFAULT_SEP)), strict=False) or ""
        return joiner.join(_source_of(v) for v in params.get("items", []))
    message = _source_message(key, strict=False)
    if message is None:
        return key
    return icu.format_nodes(icu.parse(message), params, SOURCE, _render_value, strict=False)


def to_wire(value: Any) -> Any:
    """A value in the JSON form the page renders (see the module docstring)."""
    if isinstance(value, Msg):
        if value.key == "@join":
            return {"join": [to_wire(p) for p in value.params.get("parts", [])]}
        if value.key == "@list":
            return {
                "list": [to_wire(v) for v in value.params.get("items", [])],
                "sep": value.params.get("sep", DEFAULT_SEP),
            }
        out: dict[str, Any] = {"k": value.key}
        if value.params:
            out["p"] = {name: to_wire(v) for name, v in value.params.items()}
        return out
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if value != value or value in {float("inf"), float("-inf")}:
            return str(value)
        return value
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return dict(value)  # already a wire value (the boundary's matches)
    if isinstance(value, (list, tuple)):
        return {"list": [to_wire(v) for v in value], "sep": DEFAULT_SEP}
    raise TypeError(f"cannot send {type(value).__name__} to the page as words: {value!r}")


def render(value: Any, lang: str, *, strict: bool = False, catalogs: Catalogs | None = None) -> str:
    """A value's words in ``lang`` (a Msg, a wire value, data text or a number).

    A key missing in ``lang`` falls back to the 繁體中文 source (``strict``: raises
    :class:`MissingKey`). The pseudo-locale renders the source with every catalog text
    wrapped in ⟦ ⟧.
    """
    cats = catalogs or _cats()
    if isinstance(value, Msg):
        if lang == SOURCE and catalogs is None:
            return str(value)
        return render(to_wire(value), lang, strict=strict, catalogs=cats)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return icu.format_plain(value, lang)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = {"list": list(value), "sep": DEFAULT_SEP}
    if not isinstance(value, Mapping):
        return str(value)

    def sub(inner: Any, _lang: str) -> str:
        return render(inner, lang, strict=strict, catalogs=cats)

    if "k" in value:
        key = str(value["k"])
        message = cats.get(key, lang)
        if message is None and lang not in {SOURCE, PSEUDO}:
            if strict:
                raise MissingKey(f"{key} ({lang})")
            message = cats.get(key, SOURCE)
        if message is None:
            if strict:
                raise MissingKey(key)
            return key
        params = value.get("p") or {}
        return icu.format_nodes(icu.parse(message), dict(params), lang, sub, strict=strict)
    if "join" in value:
        return "".join(sub(part, lang) for part in value["join"])
    if "list" in value:
        sep_key = str(value.get("sep") or DEFAULT_SEP)
        joiner = render({"k": sep_key}, lang, strict=strict, catalogs=cats)
        return joiner.join(sub(item, lang) for item in value["list"])
    if "cut" in value:
        if lang == SOURCE and "text" in value:
            return str(value["text"])
        return sub(value["cut"], lang)
    raise ValueError(f"not a wire value: {value!r}")


def _render_value(value: Any, lang: str) -> str:
    return render(value, lang)


def render_key(
    key: str, params: Mapping[str, Any] | None = None, lang: str = SOURCE, *, strict: bool = False
) -> str:
    """``render(key, params, lang)``: a catalog message with its values in ``lang``."""
    return render({"k": key, "p": dict(params or {})}, lang, strict=strict)


def is_wire(value: Any) -> bool:
    return isinstance(value, Mapping) and bool({"k", "join", "list", "cut"} & set(value))
