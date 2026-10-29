"""The page's words in the simulator server's answers (docs/i18n.md, owner 2026-09-28).

Every answer the page reads gets ``_i18n`` maps next to its words just before ``json.dumps``
(:mod:`duckmind.i18n.boundary`). The words themselves stay byte for byte what they were
(繁體中文); the maps tell the page which catalog message each field is, so it can show it in
the viewer's language (``I18N.field`` in static/i18n.js):

* Duck Core's words — the frozen v4_0 engine's (蛋黃) and the current engine's — are matched
  back to the engine's own templates (:func:`call_answer`);
* the simulator's and the physics world's own words are :class:`~duckmind.i18n.Msg` values;
* words that crossed the physics body's socket (the world runs in the body server's process)
  keep the maps the body process attached (``simhome``); a map that no longer fits its words
  is dropped, and plain words that are exactly one catalog text of the world, the body or the
  server get that text's key (:func:`body_answer`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from duckmind.i18n import SOURCE, Msg, boundary, render, to_wire
from duckmind.i18n.catalog import default

I18N_FIELD = boundary.I18N_FIELD
# Whose plain words may be keyed by their exact text (never Duck Core's: its words are matched
# against the engine's own templates, with the field deciding between identical words).
NAMESPACES = ("world.", "web.", "bodyhost.", "livebody.", "eye.", "curiosity.")

_index: dict[str, str] | None = None
_lock = threading.Lock()


def literal_index() -> dict[str, str]:
    """繁體中文 words → a catalog key, for the catalog texts of :data:`NAMESPACES` that have no
    values (built once)."""
    global _index
    with _lock:
        if _index is None:
            out: dict[str, str] = {}
            for key, text in sorted(default().messages(SOURCE).items()):
                if (
                    key.startswith(NAMESPACES)
                    and "{" not in text
                    and "'" not in text
                    and boundary.CJK.search(text)
                ):
                    out.setdefault(text, key)
            _index = out
        return _index


def _fits(wire: Any, text: str) -> bool:
    try:
        return render(wire, SOURCE) == text
    except Exception:
        return False


def _wire(value: str, carried: Any, index: dict[str, str]) -> Any:
    if isinstance(value, Msg):
        return to_wire(value)
    if carried is not None and not isinstance(carried, list) and _fits(carried, value):
        return carried
    key = index.get(value)
    return {"k": key} if key else None


def keyed(obj: Any, index: dict[str, str] | None = None) -> Any:
    """A copy of ``obj`` whose words all have their ``_i18n`` messages where they have one:
    Msg values, carried maps that still fit, exact catalog texts (see the module doc)."""
    index = literal_index() if index is None else index
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        carried = obj.get(I18N_FIELD)
        carried = carried if isinstance(carried, dict) else {}
        messages: dict[str, Any] = {}
        for key, value in obj.items():
            if key == I18N_FIELD:
                continue
            if isinstance(value, str):
                wire = (
                    _wire(value, carried.get(key), index)
                    if boundary.CJK.search(value) or isinstance(value, Msg)
                    else None
                )
                if wire is not None:
                    messages[key] = wire
                out[key] = value
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(v, str) or v is None for v in value)
            ):
                old = carried.get(key)
                old = old if isinstance(old, list) else []
                wires = [
                    _wire(v, old[i] if i < len(old) else None, index)
                    if isinstance(v, str)
                    else None
                    for i, v in enumerate(value)
                ]
                if any(w is not None for w in wires):
                    messages[key] = wires
                out[key] = list(value)
            else:
                out[key] = keyed(value, index)
        if messages:
            out[I18N_FIELD] = messages
        return out
    if isinstance(obj, (list, tuple)):
        return [keyed(item, index) for item in obj]
    return obj


def call_answer(payload: Any, session: Any) -> Any:
    """``/api/call``: Duck Core's words matched to its engine's templates, plus the Msg values."""
    return boundary.annotate(payload, boundary.engine_of(session))


def growth_answer(payload: Any, engine: str | None = "v4_1", names: Iterable[str] = ()) -> Any:
    """``/api/life/growth`` (docs/design/GROWTH-G4-2026-10-11.md): the growth view's Msg values and
    Duck Core's diary lines matched to the engine's templates; the people's names are data."""
    known = {
        str(person.get("name"))
        for person in _walk_people(payload)
        if isinstance(person.get("name"), str)
    } | set(names)
    return boundary.annotate(payload, engine, names=known)


def _walk_people(payload: Any) -> list[dict[str, Any]]:
    bonds = payload.get("bonds") if isinstance(payload, dict) else None
    if not isinstance(bonds, dict):
        return []
    return [p for p in bonds.get("people") or [] if isinstance(p, dict)]


def body_answer(payload: Any) -> Any:
    """``/api/body`` and the other answers made by the simulator and the physics world."""
    return keyed(payload)


def home_answer(payload: dict[str, Any]) -> dict[str, Any]:
    """``/api/home`` (and ``home_json`` inside ``/api/body``): the home, room and toy names by
    their keys (the layout files are hashed into the home's fingerprint: never edited)."""
    layout = payload.get("layout")
    if not isinstance(layout, dict):
        return keyed(payload)
    home_id = str(layout.get("id") or payload.get("name") or "")
    out = keyed({key: value for key, value in payload.items() if key != "layout"})
    out["layout"] = keyed(boundary.annotate_home(layout, home_id or None))
    return out
