"""Single-writer Duck Mind runtime with replay-based crash recovery."""

from __future__ import annotations

import fcntl
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping

from duckmind.core import DuckSelf, reduce_event
from duckmind.provenance import Event
from duckmind.store import SQLiteEventStore


class DuckRuntime:
    def __init__(self, path: str | Path, *, identity: str, born_at: datetime) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.path.with_name(self.path.name + ".runtime.lock")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            raise RuntimeError("another Duck Mind runtime owns this state directory") from error
        self._lock_fd = fd
        try:
            self.store = SQLiteEventStore(self.path)
            snapshot = self.store.load_snapshot("self")
            if snapshot is None:
                self.state = DuckSelf.new(identity=identity, born_at=born_at)
                self._revision = 0
                self._last_seq = 0
                self.refresh()
                if self._revision == 0:
                    self._save_snapshot()
            else:
                self.state = DuckSelf.from_snapshot(snapshot.state)
                if self.state.identity != identity:
                    raise ValueError("Duck identity mismatch for existing state database")
                self._revision = snapshot.revision
                self._last_seq = snapshot.last_event_seq
                self.refresh()
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            os.close(self._lock_fd)
            raise

    def _save_snapshot(self) -> None:
        snapshot = self.store.save_snapshot(
            "self",
            self.state.snapshot(),
            expected_revision=self._revision,
            last_event_seq=self._last_seq,
        )
        self._revision = snapshot.revision

    def refresh(self) -> int:
        entries = self.store.list_events(after_seq=self._last_seq)
        if not entries:
            return 0
        state = self.state
        for entry in entries:
            state = reduce_event(state, entry.event)
        self.state = state
        self._last_seq = entries[-1].seq
        self._save_snapshot()
        return len(entries)

    def ingest(self, event: Event, *, memory_job: Mapping[str, object] | None = None) -> bool:
        # Validate before persisting so an invalid external event cannot make
        # every later boot fail during replay.
        reduce_event(self.state, event)
        inserted = self.store.append_event(event, memory_job=memory_job)
        if inserted:
            self.refresh()
        return inserted

    def health(self) -> dict[str, object]:
        integrity = self.store.integrity_check()
        if integrity == "ok":
            # quick_check verifies SQLite pages; our digest verifies the
            # append-only application records themselves.
            self.store.list_events()
        return {
            "database_integrity": integrity,
            "identity": self.state.identity,
            "last_event_seq": self._last_seq,
            "play_enabled": self.state.play_enabled,
        }

    def close(self) -> None:
        self.store.close()
        os.close(self._lock_fd)

    def __enter__(self) -> DuckRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
