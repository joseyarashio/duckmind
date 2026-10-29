"""duckmind.i18n — the simulator pages in six languages (owner, 2026-09-28).

繁體中文 (zh-Hant) is the default and the source; English, 日本語, Español, 简体中文 and
한국어 are translations. Page words live in catalogs (``simweb/static/i18n.<lang>.<domain>.js``)
under stable keys; the page renders them with ``static/i18n.js``. How to add a string in all
six languages: ``docs/i18n.md``.

* :func:`msg` / :class:`Msg` — words by key; a Msg *is* the 繁體中文 ``str`` it replaces.
* :func:`render` — any message or wire value in any of the six languages (or the pseudo-locale).
* :mod:`duckmind.i18n.boundary` — adds ``{key, params}`` next to the words of an API answer
  (``_i18n`` maps), with the frozen engine's text matched back to its templates.
"""

from __future__ import annotations

from .catalog import Catalogs, context, domain_of, glossary, src_hash, valid_key
from .icu import MessageError
from .locales import BY_CODE, CODES, LANGS, PSEUDO, SOURCE, Lang, negotiate, normalize
from .message import (
    MissingKey,
    Msg,
    carried_words,
    content_msg,
    from_wire,
    is_wire,
    join,
    lst,
    msg,
    render,
    render_key,
    to_wire,
    use_catalogs,
)

__all__ = [
    "BY_CODE",
    "CODES",
    "LANGS",
    "PSEUDO",
    "SOURCE",
    "Catalogs",
    "Lang",
    "MessageError",
    "MissingKey",
    "Msg",
    "carried_words",
    "content_msg",
    "context",
    "domain_of",
    "from_wire",
    "glossary",
    "is_wire",
    "join",
    "lst",
    "msg",
    "negotiate",
    "normalize",
    "render",
    "render_key",
    "src_hash",
    "to_wire",
    "use_catalogs",
    "valid_key",
]
