"""Number formatting for messages: a small, ICU-compatible subset of number skeletons.

A message argument ``{x, number, ::.0}`` shows ``x`` with exactly one decimal. Supported
skeleton tokens (space separated after ``::``):

* ``.0`` ``.00`` ``.000`` … — exactly that many fraction digits (Python ``.1f`` ``.2f`` …)
* ``.#`` ``.##`` ``.0#`` … — at most that many (``0`` = at least)
* ``precision-integer`` — no fraction digits (Python ``.0f``)
* ``integer-width/*00`` (or ``+00``) — at least two integer digits, zero padded (``02d``)
* ``group-auto`` — group thousands (numbers never group otherwise: the duck's numbers are
  small, and today's Traditional Chinese text never groups them)

``{x, number}`` alone shows up to three decimals; ``{x, number, integer}`` rounds to a whole
number. In the source language (繁體中文) formatting equals Python's own ``format(x, spec)``,
so a message renders exactly what the f-string it replaced printed. The page formats the same
skeletons with ``Intl.NumberFormat`` (Spanish writes ``1,5``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .locales import SOURCE, lang_info

_FRACTION = re.compile(r"^\.(0*)(#*)$")
_WIDTH = re.compile(r"^integer-width/[*+](0+)$")


class SkeletonError(ValueError):
    pass


@dataclass(frozen=True)
class NumberStyle:
    min_frac: int = 0
    max_frac: int = 3
    min_int: int = 1
    grouping: bool = False


DEFAULT = NumberStyle()
INTEGER = NumberStyle(0, 0, 1, False)


def parse_style(style: str | None) -> NumberStyle:
    """The style of ``{x, number[, style]}``: none, ``integer`` or a ``::`` skeleton."""
    if style is None or style.strip() == "":
        return DEFAULT
    style = style.strip()
    if style == "integer":
        return INTEGER
    if not style.startswith("::"):
        raise SkeletonError(f"unsupported number style {style!r} (use integer or a :: skeleton)")
    min_frac, max_frac, min_int, grouping = 0, 3, 1, False
    fraction_seen = False
    for token in style[2:].split():
        match = _FRACTION.match(token)
        width = _WIDTH.match(token)
        if match and (match.group(1) or match.group(2)):
            if fraction_seen:
                raise SkeletonError(f"two precisions in {style!r}")
            fraction_seen = True
            min_frac = len(match.group(1))
            max_frac = min_frac + len(match.group(2))
        elif token == "precision-integer":
            if fraction_seen:
                raise SkeletonError(f"two precisions in {style!r}")
            fraction_seen = True
            min_frac = max_frac = 0
        elif width:
            min_int = len(width.group(1))
        elif token in {"group-auto", "group-on-aligned", "group-min2"}:
            grouping = True
        elif token == "group-off":
            grouping = False
        else:
            raise SkeletonError(f"unsupported skeleton token {token!r} in {style!r}")
    return NumberStyle(min_frac, max_frac, min_int, grouping)


def _group(digits: str, sep: str, min_grouping: int) -> str:
    if len(digits) < 3 + min_grouping:
        return digits
    out: list[str] = []
    while len(digits) > 3:
        out.insert(0, digits[-3:])
        digits = digits[:-3]
    out.insert(0, digits)
    return sep.join(out)


def format_number(value: float | int, lang: str, style: NumberStyle = DEFAULT) -> str:
    """``value`` in ``style`` for language ``lang`` (Python rounding: round-half-even)."""
    info = lang_info(lang)
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return str(value)
    # Python's own rounding and digits (what the f-string printed), then the style's rules
    text = format(value, f".{style.max_frac}f")
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]
    whole, _, frac = text.partition(".")
    # at most max_frac digits were printed; drop optional trailing zeros down to min_frac
    while len(frac) > style.min_frac and frac.endswith("0"):
        frac = frac[:-1]
    whole = whole.rjust(style.min_int, "0")
    if style.grouping:
        whole = _group(whole, info.group, info.min_grouping)
    return sign + whole + (info.decimal + frac if frac else "")


def format_plain(value: float | int, lang: str) -> str:
    """A number put into a plain ``{x}`` argument (no number type).

    The source language prints it as Python does (``str``: what an f-string without a spec
    printed); other languages print the same digits with their own decimal mark, and a whole
    float without its ``.0`` (as the page does: in JavaScript ``3.0 === 3``).
    """
    if isinstance(value, bool):
        value = int(value)
    if lang == SOURCE:
        return str(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    info = lang_info(lang)
    if isinstance(value, float) and "e" not in text:
        text = text.replace(".", info.decimal)
    return text


def python_spec(style: NumberStyle) -> str | None:
    """The Python format spec an f-string used for this style, if there is exactly one."""
    if style.grouping:
        return None
    if style.min_frac == style.max_frac:
        return f".{style.max_frac}f"
    return None


def style_for_spec(spec: str) -> str | None:
    """The skeleton for a Python format spec (``.1f`` → ``::.0``, ``02d`` → integer-width)."""
    spec = spec.strip()
    if not spec:
        return None
    fixed = re.fullmatch(r"\.(\d+)f", spec)
    if fixed:
        digits = int(fixed.group(1))
        return "::precision-integer" if digits == 0 else "::." + "0" * digits
    padded = re.fullmatch(r"0(\d+)d", spec)
    if padded:
        return "::integer-width/*" + "0" * int(padded.group(1))
    if spec == "d":
        return "integer"
    return None
