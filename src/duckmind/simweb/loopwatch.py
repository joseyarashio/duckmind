"""A loop detector for the live duck (round 5, 2026-10-10): it only watches and writes a file; it never touches the body, the brain or the link.

The sighting it exists for: a long-lived duck, a bang, a flee that ends against something, and then ``turn_around`` ↔ ``bump_stop`` four times in a
row. That could not be reproduced on fresh boots (0 loops in about 150 flights on two trees), so the live server keeps the last
:data:`KEEP_S` seconds of what the driver saw and, when the loop happens (at least :data:`LOOP_BUMPS` ``bump_stop`` onsets within
:data:`LOOP_WINDOW_S` seconds and :data:`LOOP_RADIUS_M` metres of each other), writes them to ``<folder>/<time>.jsonl.gz``: the neuro samples
(behaviour, reflex, wanted / applied twist, policy), the link's decisions (gates, refusals, the twist sent, its events: the robotd skill
starts and refusals, the proprio edges), the request the brain made (walk, panic, pivot, skill) and the truth pose. At most one file per
:data:`MIN_GAP_S`; one log line when it writes. The write is done by a daemon thread on a copy, so a tick never waits for the disk.
"""

from __future__ import annotations

import gzip
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from duckmind import paths

KEEP_S = 60.0  # how much of the past a capture holds
LOOP_BUMPS = 3  # bump_stop onsets …
LOOP_WINDOW_S = 10.0  # … within this long …
LOOP_RADIUS_M = 0.6  # … and this far of each other (the truth pose, metres)
MIN_GAP_S = 600.0  # at most one capture file per this long
BUMP = "bump_stop"


def default_folder() -> Path:
    return paths.data_dir() / "live" / "loops"


class LoopWatch:
    """Feed it one record per driver tick; it keeps :data:`KEEP_S` of them and watches the ``bump_stop`` onsets.

    ``clock`` (monotonic seconds) is for the gap between files, ``stamp`` for the file's name, ``log`` for the one line; all three are
    injectable for the tests, ``sync`` writes in the caller's thread (the tests) instead of a daemon thread."""

    def __init__(
        self,
        folder: Path | str | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        stamp: Callable[[], str] | None = None,
        log: Callable[[str], None] | None = None,
        sync: bool = False,
    ) -> None:
        self.folder = Path(folder) if folder is not None else default_folder()
        self.clock = clock
        self.stamp = stamp or (lambda: time.strftime("%Y%m%d-%H%M%S"))
        self.log = log or (lambda line: print(line, flush=True))
        self.sync = sync
        self.records: deque[dict[str, Any]] = deque()
        self.onsets: deque[tuple[float, float, float]] = deque()  # (t_s, x_m, y_m)
        self._bumping = False
        self.last_write: float | None = None
        self.hits = 0  # loops seen (written or not)
        self.written: list[Path] = []

    def feed(self, record: Mapping[str, Any]) -> Path | None:
        """One tick. ``record`` has ``t_ms``, ``reflex`` and ``pose`` (x, y, heading in metres / rad, or None) at least; the rest is kept as it is.
        Returns the path when this tick completed a loop and a file is being written."""
        rec = dict(record)
        t_s = float(rec["t_ms"]) / 1000.0
        self.records.append(rec)
        while self.records and t_s - float(self.records[0]["t_ms"]) / 1000.0 > KEEP_S:
            self.records.popleft()
        bumping = rec.get("reflex") == BUMP
        onset = bumping and not self._bumping
        self._bumping = bumping
        pose = rec.get("pose")
        if not onset or not pose:
            return None
        self.onsets.append((t_s, float(pose[0]), float(pose[1])))
        while self.onsets and t_s - self.onsets[0][0] > LOOP_WINDOW_S:
            self.onsets.popleft()
        near = [
            o
            for o in self.onsets
            if math.hypot(o[1] - self.onsets[-1][1], o[2] - self.onsets[-1][2]) <= LOOP_RADIUS_M
        ]
        if len(near) < LOOP_BUMPS:
            return None
        self.hits += 1
        now = self.clock()
        if self.last_write is not None and now - self.last_write < MIN_GAP_S:
            return None
        self.last_write = now
        path = self.folder / f"{self.stamp()}.jsonl.gz"
        head = {
            "loop": {
                "t_s": round(t_s, 3),
                "bumps": [[round(o[0], 3), round(o[1], 3), round(o[2], 3)] for o in near],
            },
            "keep_s": KEEP_S,
        }
        rows = [head, *(dict(r) for r in self.records)]
        if self.sync:
            self._write(path, rows, len(near))
        else:
            threading.Thread(
                target=self._write, args=(path, rows, len(near)), name="loopwatch", daemon=True
            ).start()
        return path

    def _write(self, path: Path, rows: list[dict[str, Any]], bumps: int) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            tmp.replace(path)
            self.written.append(path)
            self.log(
                f"迴圈偵測：{bumps} 次 bump_stop 在 {LOOP_WINDOW_S:.0f} 秒內、{LOOP_RADIUS_M} 公尺內，已存最近 {KEEP_S:.0f} 秒到 {path}"
            )
        except (
            Exception
        ) as error:  # a capture that cannot be written must never reach the body loop
            try:
                self.log(f"迴圈偵測寫檔失敗：{type(error).__name__}: {error}")
            except Exception:
                pass
