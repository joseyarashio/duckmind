"""ICU MessageFormat (the subset the page uses): parse, check and format a message.

Syntax, the same in every catalog and in ``static/i18n.js``::

    plain words                 {name}                      a value (text, a number, a message)
    {n, number}                 {n, number, integer}        {n, number, ::.0}   (see numbers.py)
    {n, plural, one {# day} other {# days}}                 also =0 {…}, offset:1
    {n, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}
    {side, select, left {…} right {…} other {…}}

Quoting (ICU "double optional" apostrophes): ``''`` is one apostrophe; an apostrophe right
before ``{`` ``}`` (or ``#`` inside a plural) starts quoted text up to the next apostrophe, so
``'{'`` is a literal brace. Any other apostrophe is literal (English "don't" needs no care).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import plural as plural_rules
from .locales import PSEUDO, PSEUDO_CLOSE, PSEUDO_OPEN
from .numbers import DEFAULT, format_number, format_plain, parse_style

NAME_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
NAME_CHARS = NAME_START + "0123456789"
TYPES = ("number", "plural", "selectordinal", "select")


class MessageError(ValueError):
    """A message that does not parse, or a value it cannot take."""


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Arg:
    name: str


@dataclass(frozen=True)
class Number:
    name: str
    style: str | None


@dataclass(frozen=True)
class Pound:
    pass


@dataclass(frozen=True)
class Choice:
    """plural / selectordinal / select."""

    kind: str
    name: str
    options: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    offset: int = 0


Node = Text | Arg | Number | Pound | Choice


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, what: str) -> MessageError:
        return MessageError(f"{what} at {self.pos} in {self.text!r}")

    def ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def message(self, depth: int, in_plural: bool) -> tuple[Node, ...]:
        nodes: list[Node] = []
        buf: list[str] = []

        def flush() -> None:
            if buf:
                nodes.append(Text("".join(buf)))
                buf.clear()

        text = self.text
        while self.pos < len(text):
            ch = text[self.pos]
            if ch == "'":
                nxt = text[self.pos + 1] if self.pos + 1 < len(text) else ""
                if nxt == "'":
                    buf.append("'")
                    self.pos += 2
                    continue
                if nxt in "{}" or (in_plural and nxt == "#") or nxt == "|":
                    # quoted text up to the next single apostrophe ('' inside is one ')
                    self.pos += 1
                    while self.pos < len(text):
                        c = text[self.pos]
                        if c == "'":
                            if self.pos + 1 < len(text) and text[self.pos + 1] == "'":
                                buf.append("'")
                                self.pos += 2
                                continue
                            self.pos += 1
                            break
                        buf.append(c)
                        self.pos += 1
                    continue
                buf.append("'")
                self.pos += 1
                continue
            if ch == "{":
                flush()
                nodes.append(self.argument(depth, in_plural))
                continue
            if ch == "}":
                if depth == 0:
                    raise self.error("unmatched }")
                flush()
                return tuple(nodes)
            if ch == "#" and in_plural:
                flush()
                nodes.append(Pound())
                self.pos += 1
                continue
            buf.append(ch)
            self.pos += 1
        if depth > 0:
            raise self.error("unclosed {")
        flush()
        return tuple(nodes)

    def name(self) -> str:
        start = self.pos
        if self.pos >= len(self.text) or self.text[self.pos] not in NAME_START:
            raise self.error("expected an argument name")
        while self.pos < len(self.text) and self.text[self.pos] in NAME_CHARS:
            self.pos += 1
        return self.text[start : self.pos]

    def word(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] in "_-=."
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def expect(self, ch: str) -> None:
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise self.error(f"expected {ch!r}")
        self.pos += 1

    def argument(self, depth: int, in_plural: bool) -> Node:
        self.expect("{")
        self.ws()
        name = self.name()
        self.ws()
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
            return Arg(name)
        self.expect(",")
        self.ws()
        kind = self.word()
        if kind not in TYPES:
            raise self.error(f"unknown argument type {kind!r}")
        self.ws()
        if kind == "number":
            if self.pos < len(self.text) and self.text[self.pos] == "}":
                self.pos += 1
                return Number(name, None)
            self.expect(",")
            end = self.text.find("}", self.pos)
            if end < 0:
                raise self.error("unclosed number style")
            style = self.text[self.pos : end].strip()
            self.pos = end + 1
            try:
                parse_style(style)  # a style the page cannot format is an error
            except ValueError as error:
                raise self.error(str(error)) from None
            return Number(name, style)
        self.expect(",")
        self.ws()
        offset = 0
        options: dict[str, tuple[Node, ...]] = {}
        while True:
            self.ws()
            if self.pos >= len(self.text):
                raise self.error("unclosed choice")
            if self.text[self.pos] == "}":
                self.pos += 1
                break
            if kind == "plural" and self.text.startswith("offset:", self.pos):
                self.pos += len("offset:")
                self.ws()
                digits = self.word()
                if not digits.isdigit():
                    raise self.error("expected a whole number after offset:")
                offset = int(digits)
                continue
            selector = self.word()
            if not selector:
                raise self.error("expected a selector")
            self.ws()
            self.expect("{")
            if selector in options:
                raise self.error(f"selector {selector!r} twice")
            options[selector] = self.message(depth + 1, in_plural or kind != "select")
            self.expect("}")
        if "other" not in options:
            raise self.error(f"{kind} {name!r} needs an 'other' case")
        return Choice(kind, name, options, offset)


_CACHE: dict[str, tuple[Node, ...]] = {}


def parse(text: str) -> tuple[Node, ...]:
    """The nodes of a message (cached)."""
    nodes = _CACHE.get(text)
    if nodes is None:
        nodes = _Parser(text).message(0, False)
        if len(_CACHE) < 50_000:
            _CACHE[text] = nodes
    return nodes


def escape(text: str) -> str:
    """Plain words as a message (braces quoted, apostrophes doubled)."""
    out: list[str] = []
    run: list[str] = []  # consecutive braces are quoted together ('{{' not '{''{')
    for ch in text:
        if ch in "{}":
            run.append(ch)
            continue
        if run:
            out.append("'" + "".join(run) + "'")
            run.clear()
        out.append("''" if ch == "'" else ch)
    if run:
        out.append("'" + "".join(run) + "'")
    return "".join(out)


def arguments(nodes: tuple[Node, ...]) -> dict[str, str]:
    """Every argument a message reads and how: plain, number, plural, selectordinal, select."""
    found: dict[str, str] = {}

    def walk(items: tuple[Node, ...]) -> None:
        for node in items:
            if isinstance(node, Arg):
                found.setdefault(node.name, "plain")
            elif isinstance(node, Number):
                found[node.name] = (
                    "number" if found.get(node.name) in {None, "plain"} else found[node.name]
                )
            elif isinstance(node, Choice):
                found[node.name] = node.kind
                for branch in node.options.values():
                    walk(branch)

    walk(nodes)
    return found


def plain_text(nodes: tuple[Node, ...]) -> str:
    """The message's own words (arguments left out): for length checks and searches."""
    out: list[str] = []
    for node in nodes:
        if isinstance(node, Text):
            out.append(node.value)
        elif isinstance(node, Choice):
            branch = node.options.get("other") or next(iter(node.options.values()))
            out.append(plain_text(branch))
    return "".join(out)


def check(text: str, lang: str) -> list[str]:
    """Problems of one message in one language (empty when fine)."""
    try:
        nodes = parse(text)
    except MessageError as error:
        return [str(error)]
    problems: list[str] = []

    def walk(items: tuple[Node, ...]) -> None:
        for node in items:
            if isinstance(node, Number):
                try:
                    parse_style(node.style)
                except ValueError as error:
                    problems.append(str(error))
            elif isinstance(node, Choice):
                if node.kind in {"plural", "selectordinal"}:
                    allowed = plural_rules.categories(
                        lang, ordinal_rules=node.kind == "selectordinal"
                    )
                    for selector in node.options:
                        if selector.startswith("="):
                            try:
                                float(selector[1:])
                            except ValueError:
                                problems.append(f"{node.name}: bad exact selector {selector!r}")
                        elif selector not in plural_rules.ALL_CATEGORIES:
                            problems.append(f"{node.name}: {selector!r} is not a plural category")
                        elif selector not in allowed:
                            problems.append(
                                f"{node.name}: {lang} has no {selector!r} {node.kind} form (it has {', '.join(allowed)})"
                            )
                for branch in node.options.values():
                    walk(branch)

    walk(nodes)
    return problems


ValueRenderer = Callable[[Any, str], str]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: Any) -> float | int | None:
    if _is_number(value):
        return value  # type: ignore[no-any-return]
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def format_nodes(
    nodes: tuple[Node, ...],
    params: dict[str, Any],
    lang: str,
    render_value: ValueRenderer,
    *,
    strict: bool = True,
    pound: str | None = None,
) -> str:
    """Render parsed nodes with ``params`` in ``lang`` (``render_value`` renders a value that
    is not a plain number: text, a nested message, a list …)."""
    out: list[str] = []
    for node in nodes:
        if isinstance(node, Text):
            if lang == PSEUDO and node.value.strip():
                out.append(PSEUDO_OPEN + node.value + PSEUDO_CLOSE)
            else:
                out.append(node.value)
            continue
        if isinstance(node, Pound):
            out.append(pound if pound is not None else "#")
            continue
        if node.name not in params:
            if strict:
                raise MessageError(f"missing value for {{{node.name}}}")
            out.append("{" + node.name + "}")
            continue
        value = params[node.name]
        if isinstance(node, Arg):
            if _is_number(value):
                out.append(format_plain(value, lang))
            else:
                out.append(render_value(value, lang))
        elif isinstance(node, Number):
            number = _as_number(value)
            if number is None:
                out.append(render_value(value, lang))
            else:
                out.append(format_number(number, lang, parse_style(node.style)))
        elif isinstance(node, Choice):
            if node.kind == "select":
                chosen = node.options.get(str(value), node.options["other"])
                out.append(
                    format_nodes(chosen, params, lang, render_value, strict=strict, pound=pound)
                )
                continue
            number = _as_number(value)
            if number is None:
                if strict:
                    raise MessageError(f"{{{node.name}}} needs a number, got {value!r}")
                number = 0
            exact = f"={number:g}" if isinstance(number, float) else f"={number}"
            branch: tuple[Any, ...] | None = node.options.get(exact)
            shown = number - node.offset
            if branch is None:
                if node.kind == "selectordinal":
                    category = plural_rules.ordinal(lang, shown)
                else:
                    category = plural_rules.cardinal(lang, shown)
                branch = node.options.get(category, node.options["other"])
            text = format_number(shown, lang, DEFAULT) if _is_number(shown) else str(shown)
            out.append(format_nodes(branch, params, lang, render_value, strict=strict, pound=text))
    return "".join(out)
