"""What duckmind downloads but does not ship, pinned in one place.

``upstream.env`` names the upstream checkouts (URL, tag, commit), the poseJoints patch and the Hugging
Face repos and revisions; ``policies-v9.sha256``, ``policies-v5.sha256`` and ``motions.sha256`` pin
every file fetched from them; ``motion-guard.json`` carries the guard caps of the trained-motion table.
``python -m duckmind.setup_cmd`` and the public play image (``deploy/public/image/Dockerfile``) both
read these files; :mod:`duckmind.pins.hub` fetches and verifies, :mod:`duckmind.pins.motions` writes
the motion table.
"""

from __future__ import annotations

from pathlib import Path

from duckmind.pins.hub import read_pins

HERE = Path(__file__).resolve().parent
UPSTREAM_FILE = HERE / "upstream.env"
POLICIES_V9_FILE = HERE / "policies-v9.sha256"
POLICIES_V5_FILE = HERE / "policies-v5.sha256"
MOTIONS_FILE = HERE / "motions.sha256"
MOTION_GUARD_FILE = HERE / "motion-guard.json"


def read_env(path: Path = UPSTREAM_FILE) -> dict[str, str]:
    """The ``KEY=VALUE`` lines of ``upstream.env`` (comments and blank lines skipped)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, sep, value = line.partition("=")
            if not sep or not key.isidentifier():
                raise ValueError(f"{path}: not KEY=VALUE: {line!r}")
            out[key] = value
    return out


UPSTREAM = read_env()

__all__ = [
    "MOTIONS_FILE",
    "MOTION_GUARD_FILE",
    "POLICIES_V5_FILE",
    "POLICIES_V9_FILE",
    "UPSTREAM",
    "UPSTREAM_FILE",
    "read_env",
    "read_pins",
]
