"""``python -m duckmind.setup_cmd`` with git, cargo and the network replaced by fakes (no download)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from duckmind import paths, pins, setup_cmd, training
from duckmind.pins import hub
from duckmind.skilltrain import motion_table as MT
from duckmind.skilltrain import rl_contract

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "deploy/public/image/Dockerfile"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DUCKMIND_PATHS_FILE", str(tmp_path / "no-paths.toml"))
    for name in (
        "DUCKMIND_DATA_DIR",
        "DUCKMIND_UPSTREAM",
        "DUCKMIND_POLICY_ROOT",
        "DUCKMIND_POLICY_DIR",
        "DUCKMIND_V5_DIR",
        "DUCKMIND_V9_DIR",
        "DUCKMIND_MICRODUCK_MJCF",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


class FakeGit:
    """Records git calls; ``checkout`` leaves files the way a real checkout would and ``apply``
    patches a checkout once (``--check`` / ``--reverse --check`` answer like git does)."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on
        self.patched: set[str] = set()
        self.dirty: set[str] = set()

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
                (folder / "src/mjlab_microduck/robot/microduck").mkdir(parents=True)
                (folder / "src/mjlab_microduck/sim").mkdir(parents=True)
                (folder / "src/mjlab_microduck/sim/body_server.py").write_text("# body\n")
        elif "apply" in args:
            folder = str(Path(args[1]).resolve())
            patched = folder in self.patched
            if "--reverse" in args:
                if not patched:
                    raise subprocess.CalledProcessError(1, ["git", *args])
            elif patched or folder in self.dirty:
                raise subprocess.CalledProcessError(1, ["git", *args])
            elif "--check" not in args:
                self.patched.add(folder)

    def clean(self, folder: Path) -> bool:
        key = str(Path(folder).resolve())
        return key not in self.patched and key not in self.dirty

    def urls(self) -> list[str]:
        return [c[-1] for c in self.calls if "remote" in c]

    def applies(self) -> list[list[str]]:
        return [c for c in self.calls if "apply" in c and "--check" not in c]


class FakeCargo:
    """Records cargo builds and leaves the binary where cargo's dev profile puts it."""

    def __init__(self) -> None:
        self.builds: list[list[str]] = []

    def __call__(self, args: Sequence[str], cwd: Path) -> None:
        self.builds.append(list(args))
        triple = args[args.index("--target") + 1] if "--target" in args else None
        out = Path(cwd) / "target" / (f"{triple}/debug" if triple else "debug")
        out.mkdir(parents=True, exist_ok=True)
        (out / args[args.index("--bin") + 1]).write_text("binary\n")

    def binaries(self) -> list[str]:
        return [b[b.index("--bin") + 1] for b in self.builds]


class FakeHub:
    """Serves the Hugging Face tree listing and files of each (repo, revision); records the URLs."""

    def __init__(self, sets: dict[tuple[str, str], dict[str, bytes]]) -> None:
        self.sets = sets
        self.requested: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.requested.append(url)
        for (repo, rev), files in self.sets.items():
            if url == f"{hub.HUB}/api/models/{repo}/tree/{rev}?recursive=true":
                listing = [
                    {"type": "file", "path": n, "lfs": {"oid": _sha(d), "size": len(d)}}
                    for n, d in files.items()
                ]
                return json.dumps(listing).encode()
            prefix = f"{hub.HUB}/{repo}/resolve/{rev}/"
            if url.startswith(prefix):
                return files[url[len(prefix) :]]
        raise OSError(f"not served: {url}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def no_network(url: str) -> bytes:
    raise AssertionError(f"unexpected download of {url}")


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> FakeGit:
    fake = FakeGit()
    monkeypatch.setattr(setup_cmd, "is_clean", fake.clean)
    return fake


@pytest.fixture
def cargo(monkeypatch: pytest.MonkeyPatch) -> FakeCargo:
    fake = FakeCargo()
    monkeypatch.setattr(setup_cmd, "find_cargo", lambda: "cargo")
    monkeypatch.setattr(setup_cmd, "run_cargo", fake)
    return fake


def pin_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attr: str, files: dict[str, bytes]
) -> setup_cmd.HubSet:
    """Pin a small set (``files``' own sha256) in place of ``setup_cmd.<attr>``."""
    real = getattr(setup_cmd, attr)
    pin_file = tmp_path / f"{attr}.sha256"
    pin_file.write_text("".join(f"{_sha(d)}  {n}\n" for n, d in files.items()))
    fake = setup_cmd.HubSet(real.label, real.repo, real.revision, pin_file)
    monkeypatch.setattr(setup_cmd, attr, fake)
    return fake


ROULADE = b"roll"
V9 = {"velstand.onnx": b"v9 walk", "alpha_sitstand.onnx": b"sit", "roulade.onnx": ROULADE}
V5 = {"velstand.onnx": b"v5 walk", "roulade.onnx": ROULADE}
MOTIONS = {f"policies/{name}": f"net {name}".encode() for name in sorted(set(MT.FILES.values()))}
MOTIONS.pop("policies/roulade.onnx")


@pytest.fixture
def hub_sets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHub:
    """Small pinned v9, v5 and motion sets, served by a fake Hub."""
    sets = {}
    for attr, files in (("POLICIES_V9", V9), ("POLICIES_V5", V5), ("MOTIONS", MOTIONS)):
        hubset = pin_set(monkeypatch, tmp_path, attr, files)
        sets[(hubset.repo, hubset.revision)] = dict(files)
    monkeypatch.setattr(MT, "ROULADE_SHA256", _sha(ROULADE))
    return FakeHub(sets)


# ── the pins: one set, read by setup and by the public play image ────────────────────────────


def test_the_pins_parse_and_name_what_the_live_duck_runs():
    up = pins.UPSTREAM
    assert up["MICRODUCK_TAG"] == "daemon-v0.18.5"
    assert re.fullmatch(r"[0-9a-f]{40}", up["MICRODUCK_COMMIT"])
    assert setup_cmd.MICRODUCK.commit == up["MICRODUCK_COMMIT"]
    assert setup_cmd.MICRODUCK_RL.commit == up["MICRODUCK_RL_COMMIT"] == training.UPSTREAM_COMMIT
    assert setup_cmd.PATCH.is_file() and setup_cmd.PATCH.name == "robotd-poseJoints-0.18.5.patch"
    assert (setup_cmd.POLICIES_V9.repo, setup_cmd.POLICIES_V9.revision) == (
        "pollen-robotics/microduck-policies",
        "v9",
    )
    assert setup_cmd.MOTIONS.repo == "joseyarashio/duckmind-motion-skills"
    v9 = hub.read_pins(pins.POLICIES_V9_FILE)
    v5 = hub.read_pins(pins.POLICIES_V5_FILE)
    motions = hub.read_pins(pins.MOTIONS_FILE)
    for pinned in (v9, v5, motions):
        assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in pinned.values())
    assert len(v9) == len(v5) == 11 and "manifest.json" in v9
    assert v9["roulade.onnx"] == v5["roulade.onnx"] == MT.ROULADE_SHA256
    assert {n: v5[n] for n in rl_contract.V5_SHA256} == rl_contract.V5_SHA256
    assert {Path(n).name for n in motions} == set(MT.FILES.values()) - {"roulade.onnx"}
    guard = json.loads(pins.MOTION_GUARD_FILE.read_text(encoding="utf-8"))["guard"]
    assert guard["rules"] and guard["global_per_hour"] > 0


def test_the_public_image_reads_the_same_pin_files_and_keeps_none_of_its_own():
    text = DOCKERFILE.read_text(encoding="utf-8")
    copied = set(re.findall(r"src/duckmind/pins/([\w.-]+)", text))
    assert {"upstream.env", "hub.py", "policies-v9.sha256", "motions.sha256"} <= copied
    assert all((pins.HERE / name).is_file() for name in copied)
    assert "python -m duckmind.pins.motions" in text
    # no commit or sha256 of its own: every pin comes from src/duckmind/pins
    assert not re.search(r"\b[0-9a-f]{40,64}\b", text)
    assert not (REPO / "deploy/public/image/policies-v9.sha256").exists()


def test_a_bad_upstream_line_is_an_error(tmp_path: Path):
    bad = tmp_path / "upstream.env"
    bad.write_text("# note\nMICRODUCK_TAG daemon\n")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        pins.read_env(bad)


# ── microduck: the pinned commit, the poseJoints patch once, robotd built ────────────────────


def test_microduck_is_checked_out_patched_once_and_built(git, cargo, capsys, monkeypatch):
    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network) == 0

    md = paths.microduck_dir()
    fetches = [c for c in git.calls if "fetch" in c]
    assert [c[-1] for c in fetches] == [setup_cmd.MICRODUCK.commit]
    assert git.applies() == [["-C", str(md), "apply", str(setup_cmd.PATCH)]]
    assert cargo.binaries() == ["robotd", "tofd", "sounds"]
    assert all(b[:5] == ["cargo", "build", "--locked", "--profile", "dev"] for b in cargo.builds)
    assert setup_cmd.robotd_bin(md).is_file()

    # a second run: the checkout is accepted as it is, the patch is not applied again
    capsys.readouterr()
    git.calls.clear()
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: setup_cmd.MICRODUCK.commit)
    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network) == 0
    assert git.applies() == []
    assert "already in" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("system", "machine", "triple", "folder"),
    [
        ("darwin", "arm64", "aarch64-apple-darwin", "target/aarch64-apple-darwin/debug"),
        ("darwin", "x86_64", "x86_64-apple-darwin", "target/x86_64-apple-darwin/debug"),
        ("linux", "aarch64", None, "target/debug"),
    ],
)
def test_robotd_is_built_for_this_pythons_architecture(
    git, cargo, monkeypatch, system, machine, triple, folder
):
    """On a Mac a Rust toolchain for the other architecture (Rosetta) would build a robotd that
    cannot load this Python's ONNX Runtime: the target is named."""
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(setup_cmd.platform, "machine", lambda: machine)
    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network) == 0
    targets = {b[b.index("--target") + 1] if "--target" in b else None for b in cargo.builds}
    assert targets == {triple}
    assert setup_cmd.robotd_bin(paths.microduck_dir()) == paths.microduck_dir() / folder / "robotd"
    assert setup_cmd.robotd_bin(paths.microduck_dir()).is_file()


def test_an_already_patched_checkout_is_accepted(git, cargo, monkeypatch):
    md = paths.microduck_dir()
    md.mkdir(parents=True)
    git.patched.add(str(md.resolve()))
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: setup_cmd.MICRODUCK.commit)

    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network) == 0
    assert git.applies() == []
    assert len(cargo.builds) == 3


def test_a_checkout_with_other_changes_is_not_patched(git, cargo, monkeypatch):
    md = paths.microduck_dir()
    md.mkdir(parents=True)
    git.dirty.add(str(md.resolve()))
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: setup_cmd.MICRODUCK.commit)

    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network) == 1
    assert git.applies() == []
    assert cargo.builds == []


def test_a_checkout_at_another_commit_is_left_alone(git, cargo, monkeypatch, capsys):
    paths.microduck_dir().mkdir(parents=True)
    (paths.microduck_dir() / "mine.txt").write_text("keep\n")
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: "0" * 40)

    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network)

    assert git.calls == []
    assert cargo.builds == []
    assert (paths.microduck_dir() / "mine.txt").read_text() == "keep\n"
    assert "move that folder aside" in capsys.readouterr().out
    assert status == 1


def test_without_cargo_setup_says_how_to_get_rust_and_does_the_rest(
    git, hub_sets, monkeypatch, capsys
):
    monkeypatch.setattr(setup_cmd, "find_cargo", lambda: None)

    def no_build(args: Sequence[str], cwd: Path) -> None:
        raise AssertionError("cargo ran although it was not found")

    status = setup_cmd.setup(
        accept_mesh_licence=True, ask=None, run=git, fetch=hub_sets, build=no_build
    )

    out = capsys.readouterr().out
    assert "rustup" in out and "https://rustup.rs" in out and "without robotd" in out
    assert not setup_cmd.robotd_bin(paths.microduck_dir()).exists()
    # the rest still ran: microduck_rl, both policy sets, the motion table
    assert (paths.microduck_rl_dir() / "README.md").is_file()
    assert (paths.v9_policy_dir() / "velstand.onnx").read_bytes() == V9["velstand.onnx"]
    assert (paths.motions_dir() / "skills.json").is_file()
    # and the page starts without live physics
    assert "--microduck-mjcf" in out and "--live" not in out
    assert status == 1


def test_find_cargo_looks_where_rustup_puts_it(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "cargo"))
    assert setup_cmd.find_cargo() is None
    (tmp_path / "cargo/bin").mkdir(parents=True)
    (tmp_path / "cargo/bin/cargo").write_text("#!/bin/sh\n")
    assert setup_cmd.find_cargo() == str(tmp_path / "cargo/bin/cargo")


def test_a_failed_build_is_reported(git, monkeypatch, capsys):
    monkeypatch.setattr(setup_cmd, "find_cargo", lambda: "cargo")

    def broken(args: Sequence[str], cwd: Path) -> None:
        raise subprocess.CalledProcessError(101, list(args))

    assert setup_cmd.setup(skip_policies=True, ask=None, run=git, build=broken) == 1
    assert "building robotd failed" in capsys.readouterr().out


# ── microduck_rl: the mesh licence ───────────────────────────────────────────────────────────


def test_declined_mesh_licence_never_downloads_microduck_rl(git, cargo, capsys):
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "no"

    status = setup_cmd.setup(skip_policies=True, ask=ask, run=git, fetch=no_network)

    out = capsys.readouterr().out
    assert "CC BY-NC-SA 4.0" in out and "creativecommons.org/licenses/by-nc-sa/4.0" in out
    assert len(asked) == 1
    assert git.urls() == [setup_cmd.MICRODUCK.url]
    assert (paths.microduck_dir() / "README.md").is_file()
    assert not paths.microduck_rl_dir().exists()
    # the page still starts (a 2D duck, no live physics), and setup says how
    assert "-m duckmind.simweb --port 8765\n" in out and "--live" not in out
    assert status == 0


def test_the_licence_is_asked_before_the_long_build(git, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(setup_cmd, "find_cargo", lambda: "cargo")

    def ask(prompt: str) -> str:
        order.append("ask")
        return "no"

    def build(args: Sequence[str], cwd: Path) -> None:
        order.append("build")
        FakeCargo()(args, cwd)

    setup_cmd.setup(skip_policies=True, ask=ask, run=git, build=build)
    assert order[0] == "ask" and "build" in order


def test_without_a_terminal_and_without_the_flag_microduck_rl_is_skipped(git, cargo):
    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network)
    assert setup_cmd.MICRODUCK_RL.url not in git.urls()
    assert not paths.microduck_rl_dir().exists()
    assert status == 0


def test_yes_clones_both_upstreams_at_their_pinned_commits(git, cargo):
    status = setup_cmd.setup(skip_policies=True, ask=lambda _: " YES ", run=git, fetch=no_network)

    fetches = [c for c in git.calls if "fetch" in c]
    assert [c[-1] for c in fetches] == [setup_cmd.MICRODUCK.commit, setup_cmd.MICRODUCK_RL.commit]
    assert all("--depth" in c for c in fetches)
    assert (paths.microduck_rl_dir() / "README.md").is_file()
    assert not list(paths.upstream_root().glob(".*partial*"))
    assert status == 0


def test_the_flag_accepts_without_asking(git, cargo):
    def ask(prompt: str) -> str:
        raise AssertionError("asked although --accept-mesh-licence was given")

    status = setup_cmd.setup(
        accept_mesh_licence=True, skip_policies=True, ask=ask, run=git, fetch=no_network
    )
    assert setup_cmd.MICRODUCK_RL.url in git.urls()
    assert status == 0


def test_a_failed_clone_leaves_nothing_behind(monkeypatch, cargo):
    git = FakeGit(fail_on="fetch")
    status = setup_cmd.setup(skip_policies=True, ask=None, run=git, fetch=no_network)
    assert not paths.microduck_dir().exists()
    assert not list(paths.upstream_root().glob(".*partial*"))
    assert cargo.builds == []
    assert status == 1


# ── policies v9 and v5, duckmind's motions, the start command ────────────────────────────────


def test_the_whole_setup_prints_the_live_command_on_v9_with_robotd(
    git, cargo, hub_sets, capsys, monkeypatch
):
    status = setup_cmd.setup(accept_mesh_licence=True, ask=None, run=git, fetch=hub_sets)
    out = capsys.readouterr().out
    assert status == 0, out

    v9, v5 = paths.v9_policy_dir(), paths.v5_policy_dir()
    assert {p.name for p in v9.iterdir()} == set(V9)
    assert (v5 / "velstand.onnx").read_bytes() == V5["velstand.onnx"]
    assert paths.policy_dir().resolve() == v5.resolve()  # the tests' and tools' set stays v5
    assert any("/resolve/v9/" in u for u in hub_sets.requested)
    assert any(f"/resolve/{setup_cmd.MOTIONS.revision}/policies/" in u for u in hub_sets.requested)

    table = paths.motions_dir() / "skills.json"
    doc = json.loads(table.read_text(encoding="utf-8"))
    assert doc["files"]["walk"]["path"] == str((v9 / "velstand.onnx").resolve())
    assert MT.check(table) == []
    assert not list(paths.motions_dir().parent.glob(".motions-fetched-*"))

    command, live = setup_cmd.start_command()
    assert live and command in out
    md = paths.microduck_dir()
    for part in (
        f"DUCKMIND_ROBOTD_SKILLS={table}",
        "--live",
        "--brain neuro",
        f"--microduck {md}",
        f"--microduck-rl {paths.microduck_rl_dir()}",
        f"--policies {v9}",
        f"--robotd-bin {setup_cmd.robotd_bin(md)}",
    ):
        assert part in command
    assert "--host" not in command and "--public" not in command

    # a second run downloads nothing and changes nothing
    git.calls.clear()
    before = table.read_bytes()
    commits = {
        str(md): setup_cmd.MICRODUCK.commit,
        str(paths.microduck_rl_dir()): setup_cmd.MICRODUCK_RL.commit,
    }
    monkeypatch.setattr(setup_cmd, "head_commit", lambda folder: commits[str(folder)])
    assert setup_cmd.setup(ask=None, run=git, fetch=no_network) == 0
    assert table.read_bytes() == before
    assert git.applies() == []


def test_a_policy_with_the_wrong_sha256_is_not_kept(git, cargo, hub_sets, capsys):
    v9_set = setup_cmd.POLICIES_V9
    hub_sets.sets[(v9_set.repo, v9_set.revision)]["roulade.onnx"] = b"tampered"

    assert setup_cmd.setup(ask=None, run=git, fetch=hub_sets) == 1

    assert sorted(p.name for p in paths.v9_policy_dir().iterdir()) == [
        "alpha_sitstand.onnx",
        "velstand.onnx",
    ]
    assert not paths.motions_dir().exists()  # no table on a v9 set that is not whole
    assert "not kept" in capsys.readouterr().out


def test_a_motion_with_the_wrong_sha256_is_not_kept(git, cargo, hub_sets):
    motions = setup_cmd.MOTIONS
    hub_sets.sets[(motions.repo, motions.revision)]["policies/panic.onnx"] = b"tampered"

    assert setup_cmd.setup(ask=None, run=git, fetch=hub_sets) == 1

    assert not paths.motions_dir().exists()
    assert not list(paths.motions_dir().parent.glob(".motions-fetched-*"))
    assert (paths.v9_policy_dir() / "velstand.onnx").is_file()


def test_a_file_the_hub_does_not_list_as_pinned_is_not_kept(tmp_path: Path):
    data = b"walk"
    listing = json.dumps([{"type": "file", "path": "velstand.onnx", "oid": "0" * 40}]).encode()

    def fetch(url: str) -> bytes:
        return listing if "/api/models/" in url else data

    kept, bad = hub.fetch_all("r", "v", {"velstand.onnx": _sha(data)}, tmp_path / "d", fetch)
    assert kept == [] and "git blob" in bad[0]
    assert list((tmp_path / "d").iterdir()) == []


def test_an_existing_file_with_another_sha256_is_left_alone(git, cargo, hub_sets):
    v9 = paths.v9_policy_dir()
    v9.mkdir(parents=True)
    (v9 / "velstand.onnx").write_bytes(b"my own")

    assert setup_cmd.setup(ask=None, run=git, fetch=hub_sets) == 1
    assert (v9 / "velstand.onnx").read_bytes() == b"my own"
    assert not paths.motions_dir().exists()


def test_an_existing_motion_folder_is_left_alone(git, cargo, hub_sets, capsys):
    out = paths.motions_dir()
    out.mkdir(parents=True)
    (out / "skills.json").write_text('{"files": {"walk": {"path": "/nowhere", "sha256": "0"}}}')

    assert setup_cmd.setup(ask=None, run=git, fetch=hub_sets) == 1
    assert json.loads((out / "skills.json").read_text())["files"]["walk"]["path"] == "/nowhere"
    assert "left unchanged" in capsys.readouterr().out


def test_an_existing_current_set_is_left_alone(git, cargo, hub_sets):
    current = paths.policy_dir()
    current.mkdir(parents=True)

    assert setup_cmd.setup(ask=None, run=git, fetch=hub_sets) == 0
    assert current.is_dir() and not current.is_symlink()
    assert list(current.iterdir()) == []


def test_a_failed_download_is_reported_and_cleaned_up(git, cargo, hub_sets):
    def offline(url: str) -> bytes:
        raise OSError("network is unreachable")

    assert setup_cmd.setup(ask=None, run=git, fetch=offline) == 1
    assert list(paths.v9_policy_dir().iterdir()) == []
    assert not paths.motions_dir().exists()


def test_the_started_page_keeps_its_files_under_the_duckmind_home(home, monkeypatch):
    """The printed command names no state file: the page keeps the duck and its layouts under
    $DUCKMIND_HOME like everything setup put there, never in another ~/.duckmind."""
    from duckmind.simweb import server

    class _Stop(Exception):
        pass

    seen: dict[str, object] = {}
    make_clock = server._make_clock

    def clock(args, **kw):
        seen["state_file"] = args.state_file
        return make_clock(args, **kw)

    def layout_store(folder):
        seen["layouts"] = folder
        raise _Stop

    monkeypatch.setattr(server, "_make_clock", clock)
    monkeypatch.setattr(server.layouts, "LayoutStore", layout_store)
    with pytest.raises(_Stop):
        server.main(["--port", "0"])
    assert seen == {"state_file": home / "simweb-duck.json", "layouts": home / "layouts"}


# ── the command line ─────────────────────────────────────────────────────────────────────────


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
