"""Every message key the page's own scripts and markup name is in the catalog the browser loads.

The source language (繁體中文) loads only the ``page`` catalog (static/i18n.js): a key of another
domain (``neuro.*``, ``core.*``, ``web.*``) named by page code would show as the raw key there,
while the other five languages, which load every domain, show it fine. So page code may only
name ``page.*`` / ``fmt.*`` keys, and each must exist in the page catalog of every language.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from duckmind.i18n import checks
from duckmind.i18n.locales import CODES

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
KEY = r"(?:page|fmt|neuro|core|web|world|curiosity|bodyhost|livebody|eye)\.[A-Za-z][A-Za-z0-9_|<>-]*(?:\.[A-Za-z0-9_|<>-]+)*"
# a key written out in code: "page.x.y" (not followed by ${…}: a key built from parts)
QUOTED = re.compile(rf"([\"'])({KEY})\1")
MARKED = re.compile(r"data-i18n(?:-html)?=\"(" + KEY + r")\"")
MARKED_ATTR = re.compile(r"data-i18n-attr=\"([^\"]+)\"")


def _page_catalog(code: str) -> dict[str, str]:
    text = (STATIC / f"i18n.{code}.page.js").read_text(encoding="utf-8")
    start = text.index("{", text.index("DuckI18n.load("))
    return json.loads(text[start : text.rindex("}") + 1])


def _keys_named_by_page_code() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(STATIC.iterdir()):
        if not path.is_file() or checks.is_catalog_file(path.name):
            continue
        if path.suffix not in {".js", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        keys = {m.group(2) for m in QUOTED.finditer(text)}
        keys |= set(MARKED.findall(text))
        for attrs in MARKED_ATTR.findall(text):
            keys |= {pair.split(":", 1)[1].strip() for pair in attrs.split(";") if ":" in pair}
        if keys:
            found[path.name] = keys
    return found


def test_page_code_names_only_keys_the_source_language_loads() -> None:
    wrong = {
        name: sorted(k for k in keys if not k.startswith(("page.", "fmt.")))
        for name, keys in _keys_named_by_page_code().items()
    }
    assert {name: keys for name, keys in wrong.items() if keys} == {}


def test_every_key_page_code_names_is_in_the_page_catalog_of_every_language() -> None:
    named = _keys_named_by_page_code()
    assert named, "the scan found no keys: the pattern is wrong"
    for code in CODES:
        have = set(_page_catalog(code))
        missing = {name: sorted(k for k in keys if k not in have) for name, keys in named.items()}
        assert {n: k for n, k in missing.items() if k} == {}, code
