"""duckmind.paths: environment first, then the machine's paths.toml, then folders under DUCKMIND_HOME."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from duckmind import paths

ENVS = (
    paths.HOME_ENV,
    paths.PATHS_FILE_ENV,
    paths.DATA_ENV,
    paths.POLICY_ROOT_ENV,
    paths.POLICY_ENV,
    paths.V5_ENV,
    paths.UPSTREAM_ENV,
    paths.MJCF_ENV,
)


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv(paths.HOME_ENV, str(home))
    return home


def test_without_a_file_every_folder_is_under_the_home(clean: Path) -> None:
    assert paths.duckmind_home() == clean
    assert paths.data_dir() == clean / "data"
    assert paths.policy_root() == clean / "policies"
    assert paths.policy_dir() == clean / "policies" / "current"
    assert paths.v5_policy_dir() == clean / "policies" / "releases" / "seed-v5"
    assert paths.upstream_root() == clean / "upstream"
    assert paths.microduck_dir() == clean / "upstream" / "microduck"
    assert paths.microduck_rl_dir() == clean / "upstream" / "microduck_rl"
    assert paths.microduck_mjcf_dir() == (
        clean / "upstream" / "microduck_rl" / "src" / "mjlab_microduck" / "robot" / "microduck"
    )


def test_the_home_defaults_to_dot_duckmind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.HOME_ENV, raising=False)
    assert paths.duckmind_home() == Path.home() / ".duckmind"


def test_the_file_moves_folders_and_expands_the_tilde(clean: Path, tmp_path: Path) -> None:
    clean.mkdir()
    (clean / "paths.toml").write_text(
        f'data = "~/records"\npolicies = "{tmp_path / "pol"}"\nupstream = "{tmp_path / "up"}"\n',
        encoding="utf-8",
    )
    assert paths.data_dir() == Path.home() / "records"
    assert paths.policy_dir() == tmp_path / "pol" / "current"
    assert paths.v5_policy_dir() == tmp_path / "pol" / "releases" / "seed-v5"
    assert paths.microduck_dir() == tmp_path / "up" / "microduck"


def test_environment_beats_the_file(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "elsewhere.toml"
    file.write_text('data = "/from/file"\npolicies = "/from/file"\nupstream = "/from/file"\n')
    monkeypatch.setenv(paths.PATHS_FILE_ENV, str(file))
    assert paths.data_dir() == Path("/from/file")
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "d"))
    monkeypatch.setenv(paths.POLICY_ROOT_ENV, str(tmp_path / "p"))
    monkeypatch.setenv(paths.UPSTREAM_ENV, str(tmp_path / "u"))
    assert paths.data_dir() == tmp_path / "d"
    assert paths.policy_dir() == tmp_path / "p" / "current"
    assert paths.upstream_root() == tmp_path / "u"
    monkeypatch.setenv(paths.POLICY_ENV, str(tmp_path / "shipped"))
    monkeypatch.setenv(paths.V5_ENV, str(tmp_path / "v5"))
    monkeypatch.setenv(paths.MJCF_ENV, str(tmp_path / "mjcf"))
    assert paths.policy_dir() == tmp_path / "shipped"
    assert paths.v5_policy_dir() == tmp_path / "v5"
    assert paths.microduck_mjcf_dir() == tmp_path / "mjcf"


@pytest.mark.parametrize(
    "text",
    ['cache = "/x"\n', "data = 3\n", 'data = ""\n', "[data]\nx = 1\n", "data = \n"],
)
def test_a_bad_file_is_an_error_not_a_silent_default(clean: Path, text: str) -> None:
    clean.mkdir()
    (clean / "paths.toml").write_text(text, encoding="utf-8")
    with pytest.raises(paths.PathsFileError):
        paths.data_dir()


# ── this machine: the folders paths.toml names hold what the duck has always used ──────────

MACHINE_FILE = Path(os.environ.get(paths.PATHS_FILE_ENV) or Path.home() / ".duckmind" / "paths.toml")


@pytest.mark.skipif(not MACHINE_FILE.is_file(), reason="no machine paths.toml (a fresh checkout)")
def test_this_machines_folders_are_the_ones_it_already_had(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(paths.PATHS_FILE_ENV, str(MACHINE_FILE))
    named = paths.configured()
    assert set(named) == set(paths.KEYS)
    assert paths.data_dir() == named["data"] and (named["data"] / "voices").is_dir()
    assert paths.policy_dir() == named["policies"] / "current"
    assert (paths.policy_dir() / "manifest.json").is_file()
    assert (paths.microduck_mjcf_dir() / "robot_allcollisions.xml").is_file()
    assert paths.microduck_dir().is_dir() and paths.microduck_rl_dir().is_dir()
