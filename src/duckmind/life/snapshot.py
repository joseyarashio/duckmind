"""The neural duck's slow state, saved so a restart does not make it forget.

What is kept (all of it local, in the duck's own folder, never sent anywhere):

* the **learner**: the places it learnt (0.5 m cells: how good, how often it was there, falls,
  bumps, the spots it avoids), the things it knows, the cues it learnt (what a call of its name
  means), where its nest is, and **the people** it knows: the name the family gave each (data,
  never translated), what they look like, how close the duck feels to them, and its notes;
* the **habituation**: how used to each kind of sound or sight it has become, and how jumpy it is;
* the **drives**: how tired, curious, social … it was.

What is not kept: anything of the moment (the reflex in progress, the gaze, the gait, the body's
pose, the eye's track ids, a command on its way). Those start fresh, as for a woken duck.

The file is versioned. A snapshot from a newer build, another duck, or a damaged file is never
applied and never overwritten silently: the file is moved aside with a note, and the duck starts
with a clean mind (:class:`SnapshotUnusable` says why). The clock matters: the time the server was
off counts. Habituation recovers, bonds fade, drives return toward a fresh morning, and a thing not
seen for a day is a thing not seen for a day.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from duckmind.life.record import write_private

SNAPSHOT_FORMAT = "duckmind.neuro-snapshot"
SNAPSHOT_VERSION = 2


def _v1_to_v2(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 2 (phase 2 「學會聽話」): the learner also keeps the proficiency of each trick. A version-1
    snapshot has none: its duck was never taught one, so every trick starts at 0 (the learner reads a missing
    ``tricks`` block as exactly that; this puts the empty block in so the file says so)."""
    from duckmind.neuro.tricks import TrickBook

    out = dict(snapshot)
    learner = dict(out.get("learner") or {})
    learner.setdefault("tricks", TrickBook().to_state())
    out["learner"] = learner
    out["version"] = 2
    return out


# migrations: version → function(snapshot dict) -> snapshot dict of the next version
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {1: _v1_to_v2}


class SnapshotUnusable(ValueError):
    """The snapshot cannot be applied to this duck (said in plain words)."""


def brain_time(brain: Any) -> float:
    """The brain's own clock now (seconds since its first step; 0 before it has stepped)."""
    t0, t_prev = getattr(brain, "_t0", None), getattr(brain, "_t_prev", None)
    return 0.0 if t0 is None or t_prev is None else (t_prev - t0) / 1000.0


def capture(
    brain: Any,
    *,
    name: str,
    seed: int,
    gd: int,
    calendar: datetime,
    context: dict[str, Any],
    why: str,
    wall: float | None = None,
    talk: float | None = None,
    regrow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The snapshot of ``brain``'s slow state now (call it with the tick lock held). ``regrow``
    (growth G3): the basis of the genome the brain runs, so a restart builds the same one
    (additive: a snapshot without it is as valid as ever, version 1)."""
    from duckmind.duckcore import content

    t_s = brain_time(brain)
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "why": why,
        "saved_wall": time.time() if wall is None else float(wall),
        "saved_calendar": calendar.isoformat(),
        "brain_t_s": round(t_s, 3),
        "duck": {
            "name": name,
            "seed": int(seed),
            "gd": int(gd),
            "table_hash": content.table_hash(),
        },
        "context": dict(context),
        "learner": brain.learn.to_state(t_s),
        "habituation": brain.hab.to_state(t_s),
        "drives": brain.drives.to_state(),
        "talk": None if talk is None else float(talk),
        **({} if regrow is None else {"regrow": regrow}),
        # the lasting map (additive: a snapshot without it is as valid as ever; kept only when the duck keeps one and has learnt something)
        **(
            {"map": brain.lastmap.to_state()}
            if getattr(getattr(brain, "lastmap", None), "cfg", None) is not None
            and brain.lastmap.cfg.enabled
            and brain.lastmap.cells
            else {}
        ),
    }


def upgrade(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The snapshot at this build's version, or SnapshotUnusable."""
    if snapshot.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotUnusable(f"not a neural snapshot (format {snapshot.get('format')!r})")
    try:
        version = int(snapshot.get("version", 0))
    except (TypeError, ValueError):
        raise SnapshotUnusable("the snapshot has no version number") from None
    if version > SNAPSHOT_VERSION:
        raise SnapshotUnusable(
            f"the snapshot is version {version}, written by a newer build "
            f"(this build reads up to {SNAPSHOT_VERSION})"
        )
    while version < SNAPSHOT_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise SnapshotUnusable(f"no way to read snapshot version {version}")
        snapshot = step(snapshot)
        version = int(snapshot["version"])
    return snapshot


def apply(
    brain: Any,
    snapshot: dict[str, Any],
    *,
    seed: int,
    context: dict[str, Any],
    wall: float | None = None,
    set_talk: Callable[[float], None] | None = None,
    bond_tau_s: float = 4.0 * 3600.0,
) -> dict[str, Any]:
    """Put a snapshot into ``brain`` (a fresh brain that has not stepped yet, or whose clock
    ``brain_time`` gives). Returns what was brought back, in numbers, or raises SnapshotUnusable
    when the snapshot is not this duck's. Places, things and the nest are kept only when the body
    starts where it did (same home, same start): they are in the body's own odometry frame."""
    snapshot = upgrade(snapshot)
    duck = snapshot.get("duck") or {}
    if int(duck.get("seed", -1)) != int(seed):
        raise SnapshotUnusable(
            f"the snapshot is of seed {duck.get('seed')}, this duck's seed is {seed}"
        )
    now = time.time() if wall is None else float(wall)
    down = max(0.0, now - float(snapshot.get("saved_wall", now)))
    t_now = brain_time(brain)
    learner = dict(snapshot.get("learner") or {})
    saved_ctx = snapshot.get("context") or {}
    same_frame = all(saved_ctx.get(k) == v for k, v in context.items() if k in ("home", "start"))
    skipped: list[str] = []
    if not same_frame:
        for key in ("places", "things", "dock"):
            if learner.get(key):
                skipped.append(key)
        learner["places"], learner["things"], learner["dock"] = [], [], None
    brain.learn.load_state(learner, t_now, down, bond_tau_s)
    lmap = getattr(brain, "lastmap", None)
    saved_map = snapshot.get("map")
    if saved_map:
        if lmap is not None and lmap.cfg.enabled:
            # same home and start: the map is in this boot's odometry frame as it is. Else it is kept in the dock's frame and goes into
            # this boot's frame at the first dock fix (a map saved with no dock pose cannot be, and is not taken)
            lmap.load_state(saved_map, t_now, down, rebase=not same_frame)
            if not same_frame:
                skipped.append("map (waits for the dock)" if saved_map.get("dock") else "map")
    brain.hab.load_state(snapshot.get("habituation") or {}, t_now, down)
    brain.drives.load_state(snapshot.get("drives") or {}, down)
    talk = snapshot.get("talk")
    if talk is not None and set_talk is not None:
        set_talk(float(talk))
    return {
        "saved_calendar": snapshot.get("saved_calendar"),
        "down_s": round(down, 1),
        "places": len(brain.learn.places),
        "things": len(brain.learn.things),
        "people": len(brain.learn.people),
        "names": sorted(brain.learn.person_names().values()),
        "avoided": sum(1 for s in brain.learn.places.values() if s.hard_falls >= 3),
        "cues": len(brain.learn.cues),
        "habituated": len(brain.hab._h),
        "skipped": skipped,
        "map_cells": 0 if lmap is None else len(lmap.cells),
    }


# ── the file ──────────────────────────────────────────────────────────────────


def write(path: Path, snapshot: dict[str, Any], prev: Path | None = None) -> None:
    """Replace the snapshot file (atomically). The one it replaces becomes ``prev``."""
    if prev is not None and path.is_file():
        try:
            os.replace(path, prev)
        except OSError:
            pass
    write_private(path, json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))


def read(path: Path) -> dict[str, Any] | None:
    """The snapshot in ``path`` (None: there is none yet). SnapshotUnusable: it is damaged."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SnapshotUnusable(f"{path} cannot be read: {error}") from error
    try:
        data = json.loads(text)
    except ValueError as error:
        raise SnapshotUnusable(f"{path} is damaged ({error})") from error
    if not isinstance(data, dict):
        raise SnapshotUnusable(f"{path} is not a snapshot")
    return data


def set_aside(path: Path, reason: str) -> Path:
    """Move an unusable snapshot out of the way (kept, with the time in its name) so nothing
    overwrites it; returns its new path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.stem}.unused-{stamp}{path.suffix}")
    os.replace(path, target)
    note = target.with_name(target.name + ".why.txt")
    try:
        write_private(note, reason + "\n")
    except OSError:
        pass
    return target
