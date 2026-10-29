"""Live voice, LV1: the API key stays out of everything (``duckmind.live.keys``).

The key is read from the environment or the Keychain only when a connection opens; it is
never printed, logged, stored in a file, a test, a cassette, an exception message or a commit.
This file tests the reading and the masking, and **scans** every text file of the repository and
the recorder's logs for anything shaped like the owner's key ("AQ." followed by 40 or more
characters). The tests never read the Keychain."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind import paths
from duckmind.live import keys as K

pytestmark = pytest.mark.usefixtures("no_network")

ROOT = Path(__file__).resolve().parents[1]
# built at run time: this file never holds a key-shaped string
FAKE_KEY = "AQ." + "Ab3_-" * 9  # 3 + 45 characters
FAKE_AIZA = "AIza" + "Xy9_" * 9


def test_the_fake_key_is_shaped_like_the_real_one() -> None:
    assert len(FAKE_KEY) - 3 >= 40
    assert K.find_keys(f"x {FAKE_KEY} y") == [2]


def test_the_environment_overrides_the_keychain() -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Keychain was asked although the environment had a key")

    assert K.load_api_key(environ={K.ENV_VAR: "  from-env  "}, run=boom) == "from-env"


def test_the_keychain_is_read_by_service_name_when_the_environment_has_none() -> None:
    seen: list[Any] = []

    def run(cmd: Any, **kwargs: Any) -> Any:
        seen.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="from-keychain\n")

    assert K.load_api_key(environ={}, run=run, platform="darwin") == "from-keychain"
    cmd, kwargs = seen[0]
    assert cmd == ["security", "find-generic-password", "-s", "duckmind-gemini-api-key", "-w"]
    assert kwargs["timeout"] > 0 and kwargs["capture_output"] is True


@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(returncode=44, stdout=""),
        SimpleNamespace(returncode=0, stdout="  \n"),
        OSError("no security binary"),
        subprocess.TimeoutExpired("security", 5),
    ],
)
def test_no_key_is_none_never_an_exception(outcome: Any) -> None:
    def run(cmd: Any, **kwargs: Any) -> Any:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    assert K.load_api_key(environ={}, run=run, platform="darwin") is None


def test_other_platforms_have_only_the_environment() -> None:
    assert K.load_api_key(environ={}, platform="linux") is None
    assert K.load_api_key(environ={K.ENV_VAR: "k" * 10}, platform="linux") == "k" * 10


@pytest.mark.parametrize(
    "text",
    [
        f"connect failed for {FAKE_KEY} at host",
        f"wss://example.test/ws?key={FAKE_KEY}&alt=json",
        "wss://example.test/ws?api_key=notshapedlikeakey123",
        f"x-goog-api-key: {FAKE_KEY}",
        f"headers={{'x-goog-api-key': '{FAKE_AIZA}'}}",
        f"Authorization: Bearer {FAKE_KEY}",
    ],
)
def test_redact_masks_every_shape_a_key_travels_in(text: str) -> None:
    out = K.redact(text)
    assert FAKE_KEY not in out and FAKE_AIZA not in out and "notshapedlikeakey123" not in out
    assert K.MASK in out
    assert not K.find_keys(out)


def test_redact_masks_the_exact_key_even_when_it_is_not_key_shaped() -> None:
    odd = "plainsecretvalue"
    assert odd not in K.redact(f"failed with {odd} inside", [odd])
    assert K.redact("ordinary words, nothing secret", [odd]) == "ordinary words, nothing secret"
    assert K.redact("short", ["abc"]) == "short"  # (a three-letter "secret" would eat words)


def test_redact_takes_any_object() -> None:
    assert K.redact(ValueError(f"bad {FAKE_KEY}")) == f"bad {K.MASK}"
    assert K.redact(None) == "None"


def test_find_keys_returns_positions_only() -> None:
    text = f"a {FAKE_KEY} b {FAKE_KEY}"
    assert K.find_keys(text) == [2, 2 + len(FAKE_KEY) + 3]
    assert K.find_keys("AQ." + "a" * 39) == []  # one short of 40
    assert K.find_keys("AQ." + "a" * 40) == [0]
    assert all(isinstance(i, int) for i in K.find_keys(text))


# ── the scan: nothing key-shaped anywhere in the repository or the recorder's logs ──────────

SKIP_SUFFIXES = {".pyc", ".wav", ".jpg", ".jpeg", ".png", ".gz", ".npz", ".onnx", ".ttf", ".woff2"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "target", ".pytest_cache",
             ".ruff_cache", ".mypy_cache"}  # fmt: skip


def _text_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for path in folder.iterdir():
        if path.name in SKIP_DIRS:  # (a worktree's .git is a file)
            continue
        if path.is_dir() and not path.is_symlink():
            out += _text_files(path)
        elif path.is_file() and path.suffix not in SKIP_SUFFIXES:
            out.append(path)
    return out


def scan_targets() -> list[Path]:
    out = sorted(_text_files(ROOT))
    # the recorder's own logs (outside the repository), when this machine has any
    return out + sorted(_text_files(paths.data_dir() / "live-voice"))


def test_the_scan_finds_a_planted_key(tmp_path: Path) -> None:
    planted = tmp_path / "log.jsonl"
    planted.write_text(f'{{"error": "url had {FAKE_KEY}"}}\n', encoding="utf-8")
    hits = [p for p in (planted,) if K.find_keys(p.read_text(encoding="utf-8"))]
    assert hits == [planted]


def test_the_scan_covers_the_whole_repository() -> None:
    targets = set(scan_targets())
    for rel in ("pyproject.toml", "src/duckmind/simweb/server.py", "tests/conftest.py"):
        assert ROOT / rel in targets, rel
    assert not any(".git" in p.relative_to(ROOT).parts for p in targets if ROOT in p.parents)


def test_no_cassette_log_or_source_file_holds_a_key() -> None:
    targets = scan_targets()
    assert targets, "the scan found nothing to scan (the repository should be there)"
    leaks = [
        str(p.relative_to(ROOT) if ROOT in p.parents else p)
        for p in targets
        if K.find_keys(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert leaks == [], f"key-shaped text in: {leaks}"
