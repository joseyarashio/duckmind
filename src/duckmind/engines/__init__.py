"""Frozen Duck Core engines (spec §7.6; phase-4 design A6).

A duck keeps the engine it was born on. Its Config names it (``engine_rev``); a saved duck
without one is a v4 duck. The current engine is ``duckmind.duckcore`` (revision ``CURRENT``).
Every older revision is a frozen copy generated from git under ``duckmind.engines.<rev>``
(``tools/freeze_engine.py``; SHA-256 manifest checked by ``tests/test_engine_frozen.py``;
never edited by hand), so an old duck's history replays byte for byte forever.

The frozen copies live here, outside ``duckmind.duckcore``, because the Duck Core scanners
(``tests/test_no_self_sensing.py``, ``tests/test_forbidden_identifiers.py``) read everything
under ``duckcore/``: a frozen v4 inside it would double every known exception. Nothing here
imports the world (spec R1); ``tests/test_imports.py`` follows the imports below.

``api`` routes a session by its revision: a frozen one is handed whole to that revision's own
``api`` (``dispatch``, ``snapshot``, ``banner``, ``feed``, ``replay_check``, …).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

CURRENT = "v4_1"  # what a new duck is born on (duckmind.duckcore)
OLD_SAVE = "v4_0"  # what a saved Config without ``engine_rev`` is
FROZEN = ("v4_0",)  # revisions kept as frozen copies, oldest first
REVS = (*FROZEN, CURRENT)


def rev_of(config: Mapping[str, Any]) -> str:
    """The engine revision a saved Config (a dict) was born on."""
    rev = config.get("engine_rev") or OLD_SAVE
    if rev not in REVS:
        raise ValueError(f"unknown Duck Core engine revision {rev!r} (known: {', '.join(REVS)})")
    return str(rev)


def is_frozen(rev: str) -> bool:
    return rev in FROZEN


def frozen_api(rev: str) -> ModuleType:
    """The ``api`` module of a frozen revision (imported on first use)."""
    if rev == "v4_0":
        from duckmind.engines.v4_0.duckcore import api

        return api
    raise ValueError(f"{rev!r} is not a frozen Duck Core engine (frozen: {', '.join(FROZEN)})")
