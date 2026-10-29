from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from duckmind.duckcore.api import Config
from duckmind.simweb.server import SimServer, SimState, make_handler


def _get(url: str, range_header: str | None = None) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"Range": range_header} if range_header else {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers.items())


def _serve(monkeypatch, request, tmp_path: Path, manifest: dict | None = None) -> str:
    path = tmp_path / "evidence.json"
    if manifest is not None:
        path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("DUCKMIND_SKILL_EVIDENCE", str(path))
    server = SimServer(("127.0.0.1", 0), make_handler(SimState(Config(seed=7))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    request.addfinalizer(close)
    return f"http://127.0.0.1:{server.server_address[1]}"


def _manifest(*clips: str) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-09-29T00:00:00Z",
        "scope": "simulation",
        "skills": [
            {
                "id": "bow",
                "clips": [{"id": clip, "src": f"/skill-evidence/{clip}.mp4"} for clip in clips],
            }
        ],
    }


def test_manifest_not_ready_returns_503(monkeypatch, request, tmp_path: Path) -> None:
    base = _serve(monkeypatch, request, tmp_path)
    status, body, _ = _get(base + "/api/skill-evidence")
    assert status == 503
    assert json.loads(body)["error"] == "skill evidence not ready"


def test_manifest_lists_allowlisted_clip_and_supports_exact_ranges(
    monkeypatch, request, tmp_path: Path
) -> None:
    base = _serve(monkeypatch, request, tmp_path, _manifest("bow-pass"))
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "bow-pass.mp4").write_bytes(b"0123456789")

    status, body, _ = _get(base + "/api/skill-evidence")
    assert status == 200
    assert json.loads(body)["skills"][0]["clips"][0]["src"] == "/skill-evidence/bow-pass.mp4"

    status, body, headers = _get(base + "/skill-evidence/bow-pass.mp4", "bytes=2-5")
    assert status == 206 and body == b"2345"
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert headers["Accept-Ranges"] == "bytes"

    status, body, headers = _get(base + "/skill-evidence/bow-pass.mp4", "bytes=-3")
    assert status == 206 and body == b"789"
    assert headers["Content-Range"] == "bytes 7-9/10"

    status, _, _ = _get(base + "/skill-evidence/bow-pass.mp4", "bytes=" + "9" * 5000 + "-")
    assert status == 416


def test_unlisted_traversal_and_missing_clips_are_not_served(
    monkeypatch, request, tmp_path: Path
) -> None:
    base = _serve(monkeypatch, request, tmp_path, _manifest("bow-pass"))
    clips = tmp_path / "clips"
    clips.mkdir()
    (tmp_path / "secret.mp4").write_bytes(b"secret")
    assert _get(base + "/skill-evidence/unlisted.mp4")[0] == 404
    assert _get(base + "/skill-evidence/%2e%2e.mp4")[0] == 404
    assert _get(base + "/skill-evidence/bow-pass.mp4")[0] == 404


def test_bad_manifest_clip_url_fails_closed(monkeypatch, request, tmp_path: Path) -> None:
    manifest = _manifest("bow-pass")
    manifest["skills"][0]["clips"][0]["src"] = "/etc/passwd"
    base = _serve(monkeypatch, request, tmp_path, manifest)
    assert _get(base + "/api/skill-evidence")[0] == 503
    assert _get(base + "/skill-evidence/bow-pass.mp4")[0] == 503


def test_duplicate_clip_ids_across_skills_fail_closed(monkeypatch, request, tmp_path: Path) -> None:
    manifest = _manifest("bow-pass")
    manifest["skills"].append(
        {"id": "step", "clips": [{"id": "bow-pass", "src": "/skill-evidence/bow-pass.mp4"}]}
    )
    base = _serve(monkeypatch, request, tmp_path, manifest)
    assert _get(base + "/api/skill-evidence")[0] == 503
