"""Fail-closed first-board check for the policy actually loaded by robotd."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

# The pinned v5 simulator report records this exact model digest; upstream's
# manifest has no per-file hashes. `robot.policies` origin is explicitly a
# display label, not a security boundary (robotd/src/main.rs at the pinned
# microduck commit). A changed official set needs a newly reviewed release.
# https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs
OFFICIAL_V5_WALK_PATH = "/opt/robot/policies/current/velstand.onnx"
OFFICIAL_V5_WALK_SHA256 = "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45"
MAX_WALK_BYTES = 16 * 1024 * 1024


def official_v5_walk_file_matches(
    path: Path = Path(OFFICIAL_V5_WALK_PATH),
    *,
    expected_sha256: str = OFFICIAL_V5_WALK_SHA256,
) -> bool:
    """Check current on-disk bytes; this cannot attest robotd's loaded memory."""
    try:
        if not path.is_file():
            return False
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_WALK_BYTES:
                return False
            # Python 3.12 hashlib.file_digest reads the binary file directly.
            # https://docs.python.org/3.12/library/hashlib.html#file-hashing
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = os.fstat(source.fileno())
        current = path.stat()

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        return (
            identity(before) == identity(after) == identity(current) and digest == expected_sha256
        )
    except (OSError, ValueError):
        return False


def loaded_official_v5_walk(report: object) -> bool:
    """Require the loaded, homed official walk slot with no override or error."""

    if not isinstance(report, Mapping):
        return False
    if (
        report.get("mode") != "walk"
        or report.get("enabled") is not True
        or report.get("homed") is not True
        or "change_error" not in report
        or report.get("change_error") is not None
    ):
        return False
    slots = report.get("slots")
    if not isinstance(slots, list) or not all(isinstance(slot, Mapping) for slot in slots):
        return False
    walk_slots = [slot for slot in slots if slot.get("slot") == "walk"]
    if len(walk_slots) != 1:
        return False
    walk = walk_slots[0]
    return (
        walk.get("path") == OFFICIAL_V5_WALK_PATH
        and walk.get("origin") == "official"
        and walk.get("overridden") is False
        and "error" in walk
        and walk.get("error") is None
    )
