"""Transitional truth, only ever less of it (phase-4 design A10, A12, §6, §15 risk 7).

LegacyWorld keeps a v4_1 duck bit-for-bit v4 by telling it five things no real body or eye
could know (``tests/legacy_truth_fields.json``): the TV's false touch is marked
``legacy_phantom``, every self-righting try is reported, the nest look answers from the pose
the duck believes it has, the dock's chance follows the pose error it believes it has, and a
walk's odometry is its own prediction. Phase 5 removes them. Until then:

* the list is well formed and may only shrink (``LISTED_AT_MOST`` goes down with it);
* it and ``duckmind.world.legacy.LEGACY_TRUTH`` name the same items, and each item's probe
  still appears in its file (a gone item is reported so the list is cut in the same change);
* no other ``legacy_*`` truth field appears in Duck Core, in the world port's schemas or in
  percept-v1.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LIST = Path(__file__).with_name("legacy_truth_fields.json")
LISTED_AT_MOST = 5  # the list may only shrink: lower this with it, never raise it (5: the
# phase-4 review listed two that LegacyWorld already had, the dock chance and the odometry)


def _items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = json.loads(LIST.read_text(encoding="utf-8"))["items"]
    return items


def _spellings(path: Path) -> set[str]:
    """Every identifier and string constant in a file (docstrings and comments included only
    as strings: a probe is looked for in the code's names and literals)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def _field_names(path: Path) -> set[str]:
    """String constants that look like a payload field (``"legacy_phantom"``): how a percept's
    fields are named in schemas and read by Duck Core (``p.get("…")``)."""
    return {
        node.value
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"[a-z][a-z0-9_]*", node.value)
    }


def test_the_list_is_well_formed_and_only_shrinks() -> None:
    items = _items()
    assert len(items) == LISTED_AT_MOST, (
        "legacy_truth_fields.json may only shrink: when an item goes, lower LISTED_AT_MOST "
        f"with it ({len(items)} listed, LISTED_AT_MOST = {LISTED_AT_MOST})"
    )
    ids = [item["id"] for item in items]
    assert len(set(ids)) == len(ids), "an item is listed twice"
    for item in items:
        assert set(item) == {"id", "field", "where", "probe", "until", "why"}, item
        assert item["why"].strip() and "\n" not in item["why"], item
        assert item["until"].startswith("phase 5"), item
        assert (ROOT / item["where"]).is_file(), item


def test_the_world_declares_exactly_the_listed_truth() -> None:
    from duckmind.world import legacy

    declared = set(getattr(legacy, "LEGACY_TRUTH", ()))
    listed = {item["id"] for item in _items()}
    assert declared <= listed, (
        f"LegacyWorld declares truth that is not listed (the list may not grow): "
        f"{sorted(declared - listed)}"
    )
    gone = sorted(listed - declared)
    assert not gone, (
        f"{gone} left LegacyWorld — good: remove them from legacy_truth_fields.json and lower "
        "LISTED_AT_MOST in the same change"
    )
    for item in _items():
        assert item["probe"] in _spellings(ROOT / item["where"]), (
            f"{item['id']}: its probe {item['probe']!r} is gone from {item['where']}; if the "
            "item is gone, cut it from the list"
        )


def test_no_other_legacy_truth_field_reaches_duck_core() -> None:
    allowed = {item["field"] for item in _items() if item["field"]}
    src = ROOT / "src" / "duckmind"
    files = [
        *sorted((src / "duckcore").rglob("*.py")),
        src / "worldport.py",
        src / "percept.py",
        src / "world" / "legacy.py",
    ]
    found = {
        f"{path.relative_to(src).as_posix()}: {text}"
        for path in files
        if path.is_file()
        for text in _field_names(path)
        if text.startswith("legacy_") and text not in allowed
    }
    assert not found, (
        "a legacy_* truth field that is not in legacy_truth_fields.json (the list may not "
        f"grow): {sorted(found)}"
    )
