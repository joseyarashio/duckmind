"""The catalogs: one file per language and domain, served to the page as plain static files.

``src/duckmind/simweb/static/i18n.<lang>.<domain>.js`` holds a JSON object of
``key → ICU message`` wrapped in one call, so the page loads it as a script (the simulator's
static server only serves flat ``.js``/``.css``/``.html`` files, and an old server process
that predates this code still serves them)::

    /* duckmind i18n catalog … */
    DuckI18n.load("en", "page", {
    "page.html.title.text": "Duck Simulator",
    …
    });

Everything between the first ``{`` after the call and the last ``}`` is strict JSON; the tools
here read and rewrite it canonically (sorted keys, one message per line).

Domains split the words by who makes them, so a page loads only what it shows: ``page`` (the
page's own words, formats, language names), ``web`` (the simulator server), ``core`` (Duck
Core, the world, curiosity, the body, the eye) and ``neuro`` (the neural duck). The source
language needs only ``page`` in the browser: server words reach it in 繁體中文 already.

``context.json`` (next to this module) tells translators where each text shows, its tone,
its values and how much room it has; ``glossary.json`` fixes the recurring terms.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

from .locales import CODES, PSEUDO, SOURCE

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "simweb" / "static"
CONTEXT_FILE = HERE / "context.json"
GLOSSARY_FILE = HERE / "glossary.json"

DOMAINS: tuple[str, ...] = ("page", "web", "core", "neuro")
# first key segment → domain
PREFIX_DOMAIN: dict[str, str] = {
    "page": "page",
    "fmt": "page",
    "lang": "page",
    "web": "web",
    "core": "core",
    "world": "core",
    "curiosity": "core",
    "bodyhost": "core",
    "livebody": "core",
    "eye": "core",
    "neuro": "neuro",
}
KEY = re.compile(r"^[a-z][a-z0-9_]*(?:\.[A-Za-z0-9_|<>-]+)+$")
_CALL = re.compile(r'DuckI18n\.load\(\s*"([A-Za-z-]+)"\s*,\s*"([a-z]+)"\s*,\s*')


class CatalogError(ValueError):
    pass


def domain_of(key: str) -> str:
    head = key.split(".", 1)[0]
    try:
        return PREFIX_DOMAIN[head]
    except KeyError:
        raise CatalogError(
            f"key {key!r}: unknown namespace {head!r} (use one of {sorted(PREFIX_DOMAIN)})"
        ) from None


def valid_key(key: str) -> bool:
    return bool(KEY.match(key)) and key.split(".", 1)[0] in PREFIX_DOMAIN


def src_hash(text: str) -> str:
    """A short fingerprint of a source text: context.json keeps it, so a changed 繁體中文 text
    under an old key (its translations now describe other words) is caught by the tests."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def catalog_path(lang: str, domain: str, static: Path | None = None) -> Path:
    return (static or STATIC) / f"i18n.{lang}.{domain}.js"


def parse_file(text: str, path: str = "<catalog>") -> tuple[str, str, dict[str, str]]:
    """(lang, domain, messages) of one catalog file."""
    call = _CALL.search(text)
    if call is None:
        raise CatalogError(f'{path}: no DuckI18n.load("<lang>", "<domain>", {{…}}) call')
    start = text.find("{", call.end() - 1)
    end = text.rfind("}")
    if start < 0 or end < start:
        raise CatalogError(f"{path}: no JSON object")
    tail = text[end + 1 :].strip()
    if not re.fullmatch(r"\)\s*;?", tail):
        raise CatalogError(f"{path}: the file must end with '}});' after the JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise CatalogError(f"{path}: not strict JSON: {error}") from None
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise CatalogError(f"{path}: must map keys to message strings")
    return call.group(1), call.group(2), data


def dump_file(lang: str, domain: str, messages: dict[str, str]) -> str:
    """The canonical text of a catalog file (sorted keys, one message per line)."""
    lines = [
        f"/* duckmind i18n catalog: {lang} / {domain} ({len(messages)} messages). Generated and checked by"
        " `python -m duckmind.i18n` (docs/i18n.md): strict JSON between the first '{' and the last '}'. */",
        f'DuckI18n.load("{lang}", "{domain}", {{',
    ]
    items = sorted(messages.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append(f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}{comma}")
    lines.append("});")
    return "\n".join(lines) + "\n"


def write_catalog(
    lang: str, domain: str, messages: dict[str, str], static: Path | None = None
) -> Path:
    for key in messages:
        if domain_of(key) != domain:
            raise CatalogError(f"{key!r} belongs to domain {domain_of(key)!r}, not {domain!r}")
    path = catalog_path(lang, domain, static)
    path.write_text(dump_file(lang, domain, messages), encoding="utf-8")
    return path


def read_catalog(lang: str, domain: str, static: Path | None = None) -> dict[str, str] | None:
    """One file's messages, or None when that language has no file for the domain yet."""
    path = catalog_path(lang, domain, static)
    if not path.is_file():
        return None
    got_lang, got_domain, messages = parse_file(path.read_text(encoding="utf-8"), str(path))
    if (got_lang, got_domain) != (lang, domain):
        raise CatalogError(f"{path}: says {got_lang}/{got_domain}, expected {lang}/{domain}")
    return messages


MANIFEST = "i18n.manifest.js"


def manifest_data(static: Path | None = None) -> dict[str, dict[str, str]]:
    """Which catalog files exist, with a short hash of each (the page never asks for a file
    that is not there, and a changed file gets a new URL)."""
    out: dict[str, dict[str, str]] = {}
    for code in CODES:
        for domain in DOMAINS:
            path = catalog_path(code, domain, static)
            if path.is_file():
                out.setdefault(code, {})[domain] = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return out


def dump_manifest(data: dict[str, dict[str, str]]) -> str:
    return (
        "/* duckmind i18n: which catalog files exist (generated by `python -m duckmind.i18n manifest`;"
        " the i18n tests check it). */\n"
        f"DuckI18n.manifest({json.dumps(data, sort_keys=True)});\n"
    )


def write_manifest(static: Path | None = None) -> Path:
    path = (static or STATIC) / MANIFEST
    path.write_text(dump_manifest(manifest_data(static)), encoding="utf-8")
    return path


def read_manifest(static: Path | None = None) -> dict[str, dict[str, str]] | None:
    path = (static or STATIC) / MANIFEST
    if not path.is_file():
        return None
    match = re.search(r"DuckI18n\.manifest\((\{.*\})\);", path.read_text(encoding="utf-8"), re.S)
    if match is None:
        raise CatalogError(f"{path}: no DuckI18n.manifest({{…}}) call")
    data = json.loads(match.group(1))
    return data if isinstance(data, dict) else None


def present_languages(static: Path | None = None) -> list[str]:
    """The languages that have at least one catalog file."""
    return [code for code in CODES if any(catalog_path(code, d, static).is_file() for d in DOMAINS)]


class Catalogs:
    """All catalogs of one static directory, loaded on first use (thread safe)."""

    def __init__(self, static: Path | None = None) -> None:
        self.static = static or STATIC
        self._lock = threading.Lock()
        self._messages: dict[str, dict[str, str]] = {}

    def messages(self, lang: str) -> dict[str, str]:
        lang = SOURCE if lang == PSEUDO else lang
        cached = self._messages.get(lang)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._messages.get(lang)
            if cached is None:
                merged: dict[str, str] = {}
                for domain in DOMAINS:
                    merged.update(read_catalog(lang, domain, self.static) or {})
                self._messages[lang] = cached = merged
        return cached

    def get(self, key: str, lang: str) -> str | None:
        return self.messages(lang).get(key)

    def reload(self) -> None:
        with self._lock:
            self._messages.clear()


_DEFAULT = Catalogs()


def default() -> Catalogs:
    return _DEFAULT


def source(key: str) -> str | None:
    """The 繁體中文 source message of a key."""
    return _DEFAULT.get(key, SOURCE)


_context_cache: dict[str, Any] | None = None


def context_file(path: Path | None = None) -> dict[str, Any]:
    """context.json: ``{"@": legend, "entries": {key: notes}}`` (another file: read fresh)."""
    global _context_cache
    if path is not None and path != CONTEXT_FILE:
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"@": {}, "entries": {}}
        )
    if _context_cache is None:
        _context_cache = (
            json.loads(CONTEXT_FILE.read_text(encoding="utf-8")) if CONTEXT_FILE.is_file() else {}
        )
    return _context_cache


def context() -> dict[str, Any]:
    """Translator notes per key (where it shows, tone, values, room, origin, source hash)."""
    entries = context_file().get("entries", {})
    return entries if isinstance(entries, dict) else {}


def glossary() -> dict[str, Any]:
    return json.loads(GLOSSARY_FILE.read_text(encoding="utf-8")) if GLOSSARY_FILE.is_file() else {}
