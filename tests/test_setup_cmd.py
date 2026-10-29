"""``python -m duckmind.setup_cmd`` with git and the network replaced by fakes (no download)."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from duckmind import paths, setup_cmd


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DUCKMIND_PATHS_FILE", str(tmp_path / "no-paths.toml"))
    for name in (
        "DUCKMIND_UPSTREAM",
        "DUCKMIND_POLICY_ROOT",
        "DUCKMIND_POLICY_DIR",
        "DUCKMIND_V5_DIR",
        "DUCKMIND_MICRODUCK_MJCF",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


class FakeGit:
    """Records git calls; ``checkout`` leaves a file the way a real checkout would."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, args: Sequence[str]) -> None:
        self.calls.append(list(args))
        if self.fail_on and self.fail_on in args:
            raise subprocess.CalledProcessError(128, ["git", *args])
        if args[0] == "init":
            Path(args[-1]).mkdir(parents=True)
        elif "checkout" in args:
            folder = Path(args[1])
            (folder / "README.md").write_text("checkout\n")
            if folder.name.startswith(".microduck_rl"):
                mjcf = folder / "src/mjlab_microduck/robot/microduck"
                mjcf.mkdir(parents=True)

    def urls(self) -> list[str]:
        return [c[-1] for c in self.calls if "remote" in c]


def fake_set(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> dict[str, str]:
    """Pin a small policy set whose sha256 values are those of ``files``."""
    pins = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    monkeypatch.setattr(setup_cmd, "V5_FILES", pins)
    return pins


def opener_for(files: dict[str, bytes]):
    requested: list[str] = []

    def opener(url: str) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(files[url.rsplit("/", 1)[-1]])

    opener.requested = requested  # type: ignore[attr-defined]
    return opener


def no_network(url: str) -> io.BytesIO:
    raise AssertionError(f"unexpected download of {url}")


def test_declined_mesh_licence_never_downloads_microduck_rl(capsys: pytest.CaptureFixture[str]):
    git = FakeGit()
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "no"

    status = setup_cmd.setup(skip_policies=True, ask=ask, run=git, opener=no_network)

    out = capsys.readouterr().out
    assert "CC BY-NC-SA 4.0" in out and "creativecommons.org/licenses/by-nc-sa/4.0" in out
    assert len(asked) == 1
    assert git.urls() == [setup_cmd.MICRODUCK.url]
    assert (paths.microduck_dir() / "README.md").is_file()
    assert not paths.microduck_rl_dir().exists()
    assert status == 0


def test_without_a_terminal_and_without_the_flag_microduck_rl_is_skipped():
    git = FakeGit()
    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, opener=no_network)
    assert setup_cmd.MICRODUCK_RL.url not in git.urls()
    assert not paths.microduck_rl_dir().exists()
    assert status == 0


def test_yes_clones_both_upstreams_at_their_pinned_commits(capsys: pytest.CaptureFixture[str]):
    git = FakeGit()
    status = setup_cmd.setup(skip_policies=True, ask=lambda _: " YES ", run=git, opener=no_network)

    fetches = [c for c in git.calls if "fetch" in c]
    assert [c[-1] for c in fetches] == [setup_cmd.MICRODUCK.commit, setup_cmd.MICRODUCK_RL.commit]
    assert all("--depth" in c for c in fetches)
    assert (paths.microduck_rl_dir() / "README.md").is_file()
    assert not list(paths.upstream_root().glob(".*partial*"))
    assert str(paths.microduck_mjcf_dir()) in capsys.readouterr().out
    assert status == 0


def test_the_flag_accepts_without_asking():
    git = FakeGit()

    def ask(prompt: str) -> str:
        raise AssertionError("asked although --accept-mesh-licence was given")

    status = setup_cmd.setup(
        accept_mesh_licence=True, skip_policies=True, ask=ask, run=git, opener=no_network
    )
    assert setup_cmd.MICRODUCK_RL.url in git.urls()
    assert status == 0


def test_a_checkout_at_another_commit_is_left_alone(monkeypatch: pytest.MonkeyPatch):
    paths.microduck_dir().mkdir(parents=True)
    (paths.microduck_dir() / "mine.txt").write_text("keep\n")
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: "0" * 40)
    git = FakeGit()

    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, opener=no_network)

    assert git.calls == []
    assert (paths.microduck_dir() / "mine.txt").read_text() == "keep\n"
    assert status == 1


def test_a_checkout_at_the_pinned_commit_is_accepted(monkeypatch: pytest.MonkeyPatch):
    paths.microduck_dir().mkdir(parents=True)
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: setup_cmd.MICRODUCK.commit)
    git = FakeGit()
    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, opener=no_network) == 0
    assert git.calls == []


def test_a_failed_clone_leaves_nothing_behind():
    git = FakeGit(fail_on="fetch")
    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, opener=no_network)
    assert not paths.microduck_dir().exists()
    assert not list(paths.upstream_root().glob(".*partial*"))
    assert status == 1


def test_policies_are_kept_only_with_their_pinned_sha256(monkeypatch: pytest.MonkeyPatch):
    files = {"manifest.json": b"{}", "velstand.onnx": b"walk", "roulade.onnx": b"roll"}
    fake_set(monkeypatch, files)
    opener = opener_for(files)

    assert setup_cmd.fetch_policies(opener) is True

    v5 = paths.v5_policy_dir()
    assert {p.name for p in v5.iterdir()} == set(files)
    assert (v5 / "roulade.onnx").read_bytes() == b"roll"
    assert opener.requested[0] == setup_cmd.V5_HF_URL.format(name="manifest.json")
    assert "/resolve/v5/" in opener.requested[0]
    current = paths.policy_dir()
    assert current.is_symlink() and current.resolve() == v5.resolve()
    # a second run downloads nothing
    assert setup_cmd.fetch_policies(no_network) is True


def test_a_policy_with_the_wrong_sha256_is_not_kept(monkeypatch: pytest.MonkeyPatch):
    fake_set(monkeypatch, {"velstand.onnx": b"walk", "roulade.onnx": b"roll"})
    served = {"velstand.onnx": b"walk", "roulade.onnx": b"tampered"}

    assert setup_cmd.fetch_policies(opener_for(served)) is False

    v5 = paths.v5_policy_dir()
    assert sorted(p.name for p in v5.iterdir()) == ["velstand.onnx"]
    assert not paths.policy_dir().exists()


def test_an_existing_policy_file_with_another_sha256_is_left_alone(monkeypatch: pytest.MonkeyPatch):
    fake_set(monkeypatch, {"velstand.onnx": b"walk"})
    v5 = paths.v5_policy_dir()
    v5.mkdir(parents=True)
    (v5 / "velstand.onnx").write_bytes(b"my own")

    assert setup_cmd.fetch_policies(no_network) is False
    assert (v5 / "velstand.onnx").read_bytes() == b"my own"


def test_an_existing_current_set_is_left_alone(monkeypatch: pytest.MonkeyPatch):
    files = {"velstand.onnx": b"walk"}
    fake_set(monkeypatch, files)
    current = paths.policy_dir()
    current.mkdir(parents=True)

    assert setup_cmd.fetch_policies(opener_for(files)) is True
    assert current.is_dir() and not current.is_symlink()
    assert list(current.iterdir()) == []


def test_a_failed_download_is_reported_and_cleaned_up(monkeypatch: pytest.MonkeyPatch):
    fake_set(monkeypatch, {"velstand.onnx": b"walk"})

    def offline(url: str) -> io.BytesIO:
        raise OSError("network is unreachable")

    assert setup_cmd.fetch_policies(offline) is False
    assert list(paths.v5_policy_dir().iterdir()) == []


def test_the_pinned_set_is_the_whole_v5_release():
    assert len(setup_cmd.V5_FILES) == 11
    assert "manifest.json" in setup_cmd.V5_FILES
    assert all(len(sha) == 64 for sha in setup_cmd.V5_FILES.values())
    assert setup_cmd.V5_FILES["roulade.onnx"].startswith("3d60da08")


def test_main_parses_the_setup_command(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_setup(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(setup_cmd, "setup", fake_setup)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert setup_cmd.main(["--accept-mesh-licence", "--skip-policies"]) == 0
    assert seen == {"accept_mesh_licence": True, "skip_policies": True, "ask": None}
    with pytest.raises(SystemExit):
        setup_cmd.main(["setup"])


def test_it_runs_as_a_module_not_a_console_script():
    done = subprocess.run(
        [sys.executable, "-m", "duckmind.setup_cmd", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(setup_cmd.__file__).parents[1])},
    )
    assert done.returncode == 0
    assert "python -m duckmind.setup_cmd" in done.stdout
    assert "--accept-mesh-licence" in done.stdout
