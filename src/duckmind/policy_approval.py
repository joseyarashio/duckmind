"""Fail-closed first-board check for the policy actually loaded by robotd.

The check is a capability, not one release's bytes: the official walk slot must hold velstand from
Pollen's policy repository with the contract duckmind's head profile and walk rules are built on —
the set's own ``manifest.json`` names the file as the ``walk`` slot, entered standing, with the
61 → 14 observation/action contract at 50 Hz (``model_api`` 1). Sets v5 to v8 all say so (v8 changed
the weights, not the contract: UPSTREAM-0185-2026-11-05); a set that changes the contract, or a file
that is not the official set's, fails. Upstream's manifest carries no per-file hashes and robotd does
not verify any, so pinning one set's sha256 here only refused every board on a newer official set
(0.16.2+ and the factory image ship v8). Which release a board may play is the operator's bench
approval (``service`` binds it to the release ID); this module answers "is the loaded walk the
official velstand contract", against robotd's report and the files on disk.
https://github.com/pollen-robotics/microduck/blob/daemon-v0.18.5/scripts/seed-policies.sh
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

OFFICIAL_WALK_PATH = "/opt/robot/policies/current/velstand.onnx"
OFFICIAL_POLICY_REPO = "pollen-robotics/microduck-policies"
MAX_WALK_BYTES = 16 * 1024 * 1024
MAX_SIDE_FILE_BYTES = 256 * 1024  # manifest.json and .source are a few kilobytes
# the contract the walk network must declare (robotd's model_api 1: 61 observations → 14 actions)
WALK_CONTRACT = {"model_api": 1, "obs_len": 61, "action_len": 14}
WALK_ROBOT = {"model": "microduck", "control_hz": 50}


def _identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def _small_text(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    if not 0 < path.stat().st_size <= MAX_SIDE_FILE_BYTES:
        return None
    return path.read_text(encoding="utf-8")


def _source_repo(text: str) -> str | None:
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "repo":
            return value.strip()
    return None


def walk_contract_ok(manifest: object, name: str) -> bool:
    """The set's manifest declares ``name`` as its velstand walk slot with the 61 → 14 contract."""
    if not isinstance(manifest, Mapping):
        return False
    if any(manifest.get(key) != value for key, value in WALK_CONTRACT.items()):
        return False
    robot = manifest.get("robot")
    if not isinstance(robot, Mapping) or any(robot.get(k) != v for k, v in WALK_ROBOT.items()):
        return False
    policies = manifest.get("policies")
    if not isinstance(policies, list):
        return False
    walks = [p for p in policies if isinstance(p, Mapping) and p.get("slot") == "walk"]
    return (
        len(walks) == 1
        and walks[0].get("file") == name
        and walks[0].get("kind") == "perpetual"
        and walks[0].get("entry_pose") == "standing"
    )


def official_walk_fingerprint(path: Path = Path(OFFICIAL_WALK_PATH)) -> str | None:
    """The sha256 of the official walk file when the files on disk pass the capability check, else
    None; this cannot attest robotd's loaded memory. The walk file is a regular, bounded file of the
    official set (``.source`` names Pollen's repository) whose manifest declares it with the velstand
    contract (:func:`walk_contract_ok`), and nothing changed under the check (a ``current`` link
    switched to another release, or the file rewritten while it was hashed, fails). The digest is
    not compared with any pinned value: the caller keeps the one it approved play with and stops
    when the file under a running play changes."""
    try:
        release = path.parent.resolve(strict=True)
        manifest_text = _small_text(release / "manifest.json")
        source_text = _small_text(release / ".source")
        if manifest_text is None or source_text is None:
            return None
        if _source_repo(source_text) != OFFICIAL_POLICY_REPO:
            return None
        if not walk_contract_ok(json.loads(manifest_text), path.name):
            return None
        if not path.is_file():
            return None
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_WALK_BYTES:
                return None
            # Python 3.12 hashlib.file_digest reads the binary file directly.
            # https://docs.python.org/3.12/library/hashlib.html#file-hashing
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = os.fstat(source.fileno())
        current = path.stat()
        same_release = path.parent.resolve(strict=True) == release
        if same_release and _identity(before) == _identity(after) == _identity(current):
            return digest
        return None
    except (OSError, ValueError, UnicodeError):
        return None


def loaded_official_walk(report: object) -> bool:
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
        walk.get("path") == OFFICIAL_WALK_PATH
        and walk.get("origin") == "official"
        and walk.get("overridden") is False
        and "error" in walk
        and walk.get("error") is None
    )


def _bounded_sha256(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        with path.open("rb") as source:
            item = os.fstat(source.fileno())
            if not stat.S_ISREG(item.st_mode) or not 0 < item.st_size <= MAX_WALK_BYTES:
                return None
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def official_policy_hashes(walk: Path = Path(OFFICIAL_WALK_PATH)) -> dict[str, str | None]:
    """Which official walk and sit-stand networks this board runs, by the sha256 the brain's
    measured tables are keyed by (``neuro/walk_response.json``, ``neuro/seat_load.json``), and the
    set's version (``.source``). ``walk_sha256`` only when the walk passes the capability check
    (:func:`official_walk_fingerprint`); ``sitstand_sha256`` when that same set's manifest names a
    ``sitstand`` network (v5 to v9 all do: ``alpha_sitstand.onnx``). A report, not an approval: a
    set the brain was not measured on still passes the start checks, the brain then drives it with
    the reference walk's numbers."""
    out: dict[str, str | None] = {"set": None, "walk_sha256": None, "sitstand_sha256": None}
    out["walk_sha256"] = official_walk_fingerprint(walk)
    if out["walk_sha256"] is None:
        return out
    try:
        release = walk.parent.resolve(strict=True)
        manifest = json.loads(_small_text(release / "manifest.json") or "null")
        source = _small_text(release / ".source") or ""
    except (OSError, ValueError, UnicodeError):
        return out
    for line in source.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "version":
            out["set"] = value.strip() or None
    policies = manifest.get("policies") if isinstance(manifest, Mapping) else None
    seats = [
        p
        for p in (policies if isinstance(policies, list) else [])
        if isinstance(p, Mapping) and p.get("name") == "sitstand" and isinstance(p.get("file"), str)
    ]
    if len(seats) == 1 and "/" not in seats[0]["file"]:
        out["sitstand_sha256"] = _bounded_sha256(release / seats[0]["file"])
    return out
