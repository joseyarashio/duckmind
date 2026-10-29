"""Read-only serving of reviewed simulator skill-evidence manifests and clips."""

from __future__ import annotations

import json
import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "outputs" / "simulator-handoff-20260929" / "evidence.json"
)
_CLIP_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_CLIP_ROUTE = re.compile(r"/skill-evidence/([A-Za-z0-9_-]{1,80})\.mp4\Z")


def manifest_path() -> Path:
    configured = os.environ.get("DUCKMIND_SKILL_EVIDENCE")
    return Path(configured).expanduser() if configured else DEFAULT_MANIFEST


def load_manifest(path: Path | None = None) -> dict[str, Any] | None:
    """Return a validated manifest, or None while evidence has not been published."""
    source = path or manifest_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("scope") != "simulation"
        or not isinstance(payload.get("generated_at"), str)
        or not isinstance(payload.get("skills"), list)
    ):
        return None
    seen_ids: set[str] = set()
    for skill in payload["skills"]:
        if not isinstance(skill, dict) or not isinstance(skill.get("clips", []), list):
            return None
        for clip in skill.get("clips", []):
            if not isinstance(clip, dict):
                return None
            clip_id = clip.get("id")
            if (
                not isinstance(clip_id, str)
                or not _CLIP_ID.fullmatch(clip_id)
                or clip_id in seen_ids
                or clip.get("src") != f"/skill-evidence/{clip_id}.mp4"
            ):
                return None
            seen_ids.add(clip_id)
    return payload


def json_response() -> tuple[int, bytes]:
    payload = load_manifest()
    if payload is None:
        return HTTPStatus.SERVICE_UNAVAILABLE, b'{"ok":false,"error":"skill evidence not ready"}'
    return HTTPStatus.OK, json.dumps(payload, ensure_ascii=False).encode("utf-8")


def clip_response(path: str, range_header: str | None) -> tuple[int, bytes, dict[str, str]]:
    """Serve only a manifest-listed clip by its fixed basename, with one byte range."""
    match = _CLIP_ROUTE.fullmatch(path)
    if not match:
        return HTTPStatus.NOT_FOUND, b"not found", {}
    clip_id = match.group(1)
    payload = load_manifest()
    if payload is None:
        return HTTPStatus.SERVICE_UNAVAILABLE, b"skill evidence not ready", {}
    allowed = {
        clip["id"]
        for skill in payload["skills"]
        for clip in skill.get("clips", [])
        if isinstance(clip, dict) and isinstance(clip.get("id"), str)
    }
    if clip_id not in allowed:
        return HTTPStatus.NOT_FOUND, b"not found", {}
    clip_path = manifest_path().parent / "clips" / f"{clip_id}.mp4"
    try:
        clip_path.resolve(strict=True).relative_to(
            (manifest_path().parent / "clips").resolve(strict=True)
        )
        data = clip_path.read_bytes()
    except (OSError, ValueError):
        return HTTPStatus.NOT_FOUND, b"not found", {}
    headers = {"Accept-Ranges": "bytes", "Content-Type": "video/mp4"}
    if not range_header:
        return HTTPStatus.OK, data, headers
    if len(range_header) > 128:
        return (
            HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            b"",
            {"Content-Range": f"bytes */{len(data)}"},
        )
    match_range = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match_range or not any(match_range.groups()) or not data:
        return (
            HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            b"",
            {"Content-Range": f"bytes */{len(data)}"},
        )
    first_text, last_text = match_range.groups()
    size = len(data)
    if first_text:
        start = int(first_text)
        end = int(last_text) if last_text else size - 1
        if start >= size or end < start:
            return (
                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                b"",
                {"Content-Range": f"bytes */{size}"},
            )
        end = min(end, size - 1)
    else:
        suffix = int(last_text)
        if suffix <= 0:
            return (
                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                b"",
                {"Content-Range": f"bytes */{size}"},
            )
        start, end = max(0, size - suffix), size - 1
    body = data[start : end + 1]
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return HTTPStatus.PARTIAL_CONTENT, body, headers
