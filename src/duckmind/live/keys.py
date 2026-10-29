"""Where the Gemini API key comes from, and keeping it out of everything else.

The key lives in the macOS Keychain (service ``duckmind-gemini-api-key``); the environment
variable ``GEMINI_API_KEY`` overrides it when set. It is read **at the moment a connection is
opened**, held only in a local variable, and never printed, logged, put in a file, a test, a
cassette, an exception message or a commit.

* :func:`load_api_key` — the key or None (never raises, never says why in a way that holds the key);
* :func:`redact` — any text with every key-shaped thing (and the exact key, when known) masked:
  the one function every message that leaves this package goes through;
* :func:`find_keys` — where in a text something key-shaped sits (positions only, never the match):
  what the tests that scan cassettes and logs use.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

KEYCHAIN_SERVICE = "duckmind-gemini-api-key"
ENV_VAR = "GEMINI_API_KEY"
KEYCHAIN_TIMEOUT_S = 5.0
MASK = "<redacted>"

# the keys this project's account makes ("AQ." + a long tail), Google's older "AIza…" shape, and
# the places a key rides in a request: ``?key=…`` in a URL, an ``x-goog-api-key`` header
_SHAPES = re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{20,}")
_IN_URL = re.compile(r"(?i)\b(key|api_key|apikey|access_token)=[^&\s\"'<>]+")
_IN_HEADER = re.compile(r"(?i)\b(x-goog-api-key|authorization)(\W{1,4})(?:bearer\s+)?[^\s\"',;]+")
# what the scanning tests look for: the owner's rule is "AQ." followed by 40 or more characters
SCAN_PATTERN = re.compile(r"AQ\.[A-Za-z0-9_\-]{40,}")
MIN_SECRET_LEN = 8  # a shorter "secret" would mask ordinary words

Runner = Callable[..., Any]


def load_api_key(
    *,
    environ: Mapping[str, str] | None = None,
    run: Runner = subprocess.run,
    platform: str = sys.platform,
) -> str | None:
    """The API key: ``$GEMINI_API_KEY`` if set, else the Keychain's, else None. Never raises."""
    env = os.environ if environ is None else environ
    given = (env.get(ENV_VAR) or "").strip()
    if given:
        return given
    if platform != "darwin":
        return None
    try:
        done = run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=KEYCHAIN_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(done, "returncode", 1) != 0:
        return None
    found = str(getattr(done, "stdout", "") or "").strip()
    return found or None


def redact(text: object, secrets: Iterable[str | None] = ()) -> str:
    """``text`` with every key-shaped thing masked, and each of ``secrets`` (the exact key a
    caller holds) wherever it appears."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= MIN_SECRET_LEN:
            out = out.replace(secret, MASK)
    out = _IN_URL.sub(lambda m: f"{m.group(1)}={MASK}", out)
    out = _IN_HEADER.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", out)
    return _SHAPES.sub(MASK, out)


def find_keys(text: str) -> list[int]:
    """Where in ``text`` something shaped like the owner's key starts (the scan the tests run on
    cassettes and logs). Positions only: the match itself is never returned."""
    return [m.start() for m in SCAN_PATTERN.finditer(text)]


__all__ = [
    "ENV_VAR",
    "KEYCHAIN_SERVICE",
    "MASK",
    "SCAN_PATTERN",
    "find_keys",
    "load_api_key",
    "redact",
]
