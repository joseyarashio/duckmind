#!/usr/bin/env python3
"""Fetch pinned files of one Hugging Face model repo revision and verify each one twice.

    python hub.py REPO REVISION PINS DEST

PINS lists ``<sha256>  <path in the repo>`` lines (``policies-v9.sha256``, ``motions.sha256`` ...
beside this file). Every listed file is downloaded from
``https://huggingface.co/REPO/resolve/REVISION/<path>`` into ``DEST/<basename>`` and kept only if

* its sha256 is the pinned one, and
* it is what the Hub's own tree listing of REVISION says it is: the LFS oid (a sha256) for an LFS
  file, the git blob id (sha1 of ``blob <size>\\0<bytes>``) for a plain file.

A file already in DEST is kept when its sha256 is the pinned one and never replaced otherwise.
The public play image runs this file on its own (standard library only, no duckmind installed);
``python -m duckmind.setup_cmd`` imports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

HUB = "https://huggingface.co"
Get = Callable[[str], bytes]


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "duckmind-setup"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed https URLs)
        return response.read()


def read_pins(path: Path) -> dict[str, str]:
    """``path in the repo`` → sha256 from a pin file."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest.lower()
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree(repo: str, revision: str, fetch: Get = get) -> dict[str, dict]:
    """path → the Hub's entry (``oid``, ``size``, ``lfs``) for every file of the revision."""
    rev = urllib.parse.quote(revision, safe="")
    entries = json.loads(fetch(f"{HUB}/api/models/{repo}/tree/{rev}?recursive=true"))
    return {e["path"]: e for e in entries if e.get("type") == "file"}


def file_url(repo: str, revision: str, name: str) -> str:
    return f"{HUB}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{name}"


def problem(name: str, data: bytes, sha256: str, entry: dict | None) -> str | None:
    """Why ``data`` is not the pinned ``name`` (None when it is): its sha256, then the Hub's entry."""
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        return f"{name}: sha256 {got}, pinned {sha256}"
    if entry is None:
        return f"{name}: not in the Hub's listing"
    lfs = entry.get("lfs")
    if lfs:
        if lfs.get("oid") != got or int(lfs.get("size", -1)) != len(data):
            return f"{name}: sha256 {got} is not the Hub's lfs oid {lfs.get('oid')}"
        return None
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 (git's id)
    if entry.get("oid") != blob:
        return f"{name}: git blob {blob} is not the Hub's {entry.get('oid')}"
    return None


def fetch_all(
    repo: str, revision: str, wanted: dict[str, str], dest: Path, fetch: Get = get
) -> tuple[list[str], list[str]]:
    """Put every pinned file of ``wanted`` into ``dest``; returns (kept names, problems).

    A file already there with the pinned sha256 is kept as it is; one with another sha256 is a
    problem and is left unchanged. A download is written beside its place first and moved there
    only when it passed both checks."""
    dest.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    bad: list[str] = []
    listing: dict[str, dict] | None = None
    for name, sha256 in sorted(wanted.items()):
        target = dest / Path(name).name
        if target.exists():
            if sha256_of(target) == sha256:
                kept.append(name)
            else:
                bad.append(f"{target}: there already with another sha256; left unchanged")
            continue
        try:
            if listing is None:
                listing = tree(repo, revision, fetch)
            data = fetch(file_url(repo, revision, name))
        except (OSError, ValueError) as err:
            bad.append(f"{name}: download failed ({err})")
            continue
        why = problem(name, data, sha256, listing.get(name))
        if why:
            bad.append(f"{why}; not kept")
            continue
        partial = dest / f".{target.name}.partial-{os.getpid()}"
        partial.write_bytes(data)
        partial.replace(target)
        kept.append(name)
    return kept, bad


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    repo, revision, pin_file, dest = argv[0], argv[1], Path(argv[2]), Path(argv[3])
    kept, bad = fetch_all(repo, revision, read_pins(pin_file), dest)
    for name in kept:
        print(f"ok {repo}@{revision}/{name}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
