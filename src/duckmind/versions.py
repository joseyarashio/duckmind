"""Content identity shared by the installed runtime and release evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def fingerprint_files(files: Mapping[str, bytes]) -> str:
    records = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def package_fingerprint(directory: Path | None = None) -> str:
    """Hash Python source, excluding caches and installation-specific metadata."""

    root = directory or Path(__file__).parent
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise ValueError(f"package source must not be a symlink: {path}")
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError("package source is empty")
    return fingerprint_files(files)
