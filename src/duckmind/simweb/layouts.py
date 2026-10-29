"""The layout editor's server side (owner 2026-10-10; docs/design/HOME-FORMAT.md, REAL-HOME-2026-10-xx.md).

A layout is a home file (``home-layout-v2``) whose ``placements`` say which library items stand where
(:mod:`duckmind.world.objlib`). The editor on the 資料 page draws the home that runs, lets the viewer drag
items from the library, rotate, move and delete them, check the result (:mod:`duckmind.world.layoutcheck`),
save it under a name, load one, go back to the default, and apply it to the running simulator.

Files live in one folder (``--layouts-dir``; with ``--life`` the life's own ``layouts`` folder):

* ``<name>.json``: the layouts the viewer saved (plain data; share them, put them in a pull request);
* ``applied/<fingerprint>.json``: a layout the simulator ran, kept so the physics process can read it by
  path (the world server takes a home by name or by path);
* ``active.json``: ``{"home": <path or name>}``, the home to come back to after a restart.

Nothing here touches the physics: :meth:`duckmind.simweb.server.SimState.apply_layout` does the reload.
Plain Python; the server calls it with its own lock.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from duckmind.world import homelayout, layoutcheck, objlib

NAME_MAX = 40
NAME_RE = re.compile(r"^[0-9A-Za-z_\-㐀-鿿぀-ヿ가-힯 ]{1,%d}$" % NAME_MAX)
MAX_BYTES = 400_000
MAX_PLACEMENTS = 400


class LayoutProblem(ValueError):
    """What the viewer asked cannot be done; ``code`` is a message key suffix, ``params`` its values."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code = code
        self.params = params


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class LayoutStore:
    """The saved layouts of one folder."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, name: str) -> Path:
        if self.root is None:
            raise LayoutProblem("no_store")
        if not NAME_RE.match(name) or name.strip() != name or name in ("active", "applied"):
            raise LayoutProblem("bad_name", max=NAME_MAX)
        return self.root / f"{name}.json"

    def list(self) -> list[dict[str, Any]]:
        if self.root is None or not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.glob("*.json")):
            if path.stem == "active":
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "name": path.stem,
                    "home": raw.get("id"),
                    "placements": len(raw.get("placements") or []),
                    "saved": int(path.stat().st_mtime),
                }
            )
        return out

    def read(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LayoutProblem("not_found", name=name) from None
        except (OSError, ValueError):
            raise LayoutProblem("unreadable", name=name) from None
        if not isinstance(raw, dict):
            raise LayoutProblem("unreadable", name=name)
        return raw

    def save(self, name: str, raw: dict[str, Any]) -> Path:
        path = self._path(name)
        text = json.dumps(raw, ensure_ascii=False, indent=1) + "\n"
        if len(text.encode()) > MAX_BYTES:
            raise LayoutProblem("too_big")
        _atomic_write(path, text)
        return path

    def delete(self, name: str) -> None:
        path = self._path(name)
        try:
            path.unlink()
        except FileNotFoundError:
            raise LayoutProblem("not_found", name=name) from None

    def applied_path(self, raw: dict[str, Any]) -> Path:
        """Write the layout where the physics process can read it (named by what it is)."""
        if self.root is None:
            raise LayoutProblem("no_store")
        layout = homelayout.load(dict(raw))
        path = self.root / "applied" / f"{layout.id}-{layout.fingerprint[:12]}.json"
        if not path.exists():
            _atomic_write(path, json.dumps(raw, ensure_ascii=False, indent=1) + "\n")
        return path

    def remember(self, home: str | None) -> None:
        """The home to come back to after a restart (None: the server's own default)."""
        if self.root is None:
            return
        active = self.root / "active.json"
        if home is None:
            try:
                active.unlink()
            except FileNotFoundError:
                pass
            return
        _atomic_write(active, json.dumps({"home": home, "at": int(time.time())}) + "\n")

    def remembered(self) -> str | None:
        if self.root is None:
            return None
        try:
            data = json.loads((self.root / "active.json").read_text(encoding="utf-8"))
            home = str(data.get("home") or "")
        except (OSError, ValueError):
            return None
        if not home:
            return None
        if home.endswith(".json") and not Path(home).is_file():
            return None
        return home


def editable(layout: homelayout.Layout) -> dict[str, Any]:
    """The home as the editor holds it: the file's form (placements, not the pieces they made)."""
    return objlib.unexpand(layout.to_json())


def validate_upload(raw: Any) -> dict[str, Any]:
    """A layout from the page: a JSON object of a sane size with a placements list the library knows."""
    if not isinstance(raw, dict):
        raise LayoutProblem("bad_layout")
    if len(json.dumps(raw)) > MAX_BYTES:
        raise LayoutProblem("too_big")
    placements = raw.get("placements") or []
    if not isinstance(placements, list) or len(placements) > MAX_PLACEMENTS:
        raise LayoutProblem("bad_layout")
    return raw


def state_payload(
    layout: homelayout.Layout, *, home: str, default_home: str, store: LayoutStore
) -> dict[str, Any]:
    """What the editor shows: the layout it edits, the fixed pieces to draw (everything that is not a
    placement), the checks of this layout and what is saved."""
    from duckmind.i18n import boundary

    raw = editable(layout)
    page = layout.page_json()
    # the home's own words (its name, its rooms') with the message of each beside it
    words = boundary.annotate_home({"zh": layout.zh, "rooms": page["rooms"]}, layout.id)
    fixed_elements = [e for e in page["elements"] if not e.get("place") and e["flags"]["page"]]
    fixed_things = [t for t in page["things"] if not t.get("lib")]
    return {
        "ok": True,
        "home": home,
        "default_home": default_home,
        "id": layout.id,
        "zh": layout.zh,
        **({"_i18n": words["_i18n"]} if "_i18n" in words else {}),
        "layout": raw,
        "bounds_mm": page["bounds_mm"],
        "rooms": words["rooms"],
        "doorways": page["doorways"],
        "nest": page["nest"],
        "mat": page["mat"],
        "spawn": page["spawn"],
        "fixed": {"elements": fixed_elements, "things": fixed_things},
        "check": layoutcheck.check(raw),
        "saved": store.list(),
        "can_save": store.enabled,
    }
