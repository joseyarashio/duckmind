"""CLDR plural rules for the six page languages (cardinal and ordinal).

The page uses ``Intl.PluralRules``; this is the same rule set for Python (tests, and any text
the server renders itself). CLDR 44 rules:

* en — cardinal: one (i = 1 and v = 0), other; ordinal: one / two / few / other.
* es — cardinal: one (n = 1), many (i ≠ 0, i % 1000000 = 0, v = 0), other; ordinal: other.
* ja, ko, zh-Hant, zh-Hans — other only (both kinds).

A number that is whole (``3`` or ``3.0``) has no visible fraction digits here, as in
JavaScript where ``3.0 === 3``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .locales import PSEUDO, SOURCE

CARDINAL: dict[str, tuple[str, ...]] = {
    "zh-Hant": ("other",),
    "en": ("one", "other"),
    "ja": ("other",),
    "es": ("one", "many", "other"),
    "zh-Hans": ("other",),
    "ko": ("other",),
}
ORDINAL: dict[str, tuple[str, ...]] = {
    "zh-Hant": ("other",),
    "en": ("one", "two", "few", "other"),
    "ja": ("other",),
    "es": ("other",),
    "zh-Hans": ("other",),
    "ko": ("other",),
}
ALL_CATEGORIES = ("zero", "one", "two", "few", "many", "other")


@dataclass(frozen=True)
class Operands:
    n: float  # absolute value
    i: int  # integer digits
    v: int  # visible fraction digits (with trailing zeros)
    f: int  # visible fraction digits as a number


def operands(value: float | int | str) -> Operands:
    """CLDR operands of a number (or of a decimal string such as ``"1.50"``)."""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return Operands(float(abs(value)), abs(value), 0, 0)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return Operands(0.0, 0, 0, 0)
        if value.is_integer():
            whole = abs(int(value))
            return Operands(float(whole), whole, 0, 0)
        text = repr(abs(value))
        if "e" in text or "E" in text:
            text = f"{abs(value):.15f}".rstrip("0")
    else:
        text = str(value).strip().lstrip("+-")
    whole_text, _, frac_text = text.partition(".")
    whole = int(whole_text or "0")
    return Operands(abs(float(text or "0")), whole, len(frac_text), int(frac_text or "0"))


def _lang(code: str) -> str:
    return SOURCE if code == PSEUDO else code


def cardinal(code: str, value: float | int | str) -> str:
    """The cardinal plural category of ``value`` in language ``code``."""
    code = _lang(code)
    op = operands(value)
    if code == "en":
        return "one" if op.i == 1 and op.v == 0 else "other"
    if code == "es":
        if op.n == 1:
            return "one"
        if op.i != 0 and op.i % 1_000_000 == 0 and op.v == 0:
            return "many"
        return "other"
    return "other"


def ordinal(code: str, value: float | int | str) -> str:
    """The ordinal plural category (1st, 2nd, 3rd …) of ``value`` in language ``code``."""
    code = _lang(code)
    if code == "en":
        op = operands(value)
        n = op.i if op.v == 0 else -1
        if n % 10 == 1 and n % 100 != 11:
            return "one"
        if n % 10 == 2 and n % 100 != 12:
            return "two"
        if n % 10 == 3 and n % 100 != 13:
            return "few"
    return "other"


def categories(code: str, *, ordinal_rules: bool = False) -> tuple[str, ...]:
    table = ORDINAL if ordinal_rules else CARDINAL
    return table[_lang(code)]
