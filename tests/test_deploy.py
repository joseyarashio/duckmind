"""Exercise deployment transactions without root, systemd, or a physical board.

Only copied scripts are path-rebased into tmp_path. Production paths stay fixed.
All mutating OS commands are doubles that reject targets outside that root;
SQLite backup and JSON health validation still run real Python code.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def test_published_systemd_unit_runs_unprivileged_boot_guard_before_daemon():
    unit = (DEPLOY / "duckmind.service").read_text()
    preflight = "ExecStartPre=/opt/duckmind/current/board-boot-guard.sh"
    daemon = "ExecStart=/opt/duckmind/current/venv/bin/duckmindd"
    assert "User=duckmind\n" in unit
    assert f"{preflight}\n" in unit
    assert f"{daemon} " in unit
    assert unit.index(preflight) < unit.index(daemon)


COMMAND_DOUBLE = r"""
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(os.environ["DEPLOY_TEST_ROOT"])
name = Path(sys.argv[0]).name
args = sys.argv[1:]
failure = os.environ.get("DEPLOY_TEST_FAILURE", "")

def local(value):
    path = Path(value)
    # Check lexical containment AND resolved containment, including symlinks.
    if not path.is_absolute() or not path.is_relative_to(root):
        raise RuntimeError(f"test attempted host access: {value}")
    if not path.resolve().is_relative_to(root):
        raise RuntimeError(f"test attempted symlink escape: {value}")
    return path

with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps([name, *args]) + "\n")

if name == "id":
    print("0" if args == ["-u"] else "duckmind robot")
elif name == "getent":
    print(f"{args[-1]}:x:1000:")
elif name in {"flock", "sleep", "chown"}:
    if name == "chown":
        local(args[-1])
elif name in {"groupadd", "useradd", "usermod"}:
    raise RuntimeError("fixture accounts already exist")
elif name == "sha256sum":
    path = local(args[0])
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
elif name == "install":
    directory = False
    mode = 0o755
    paths = []
    while args:
        arg = args.pop(0)
        if arg == "-d":
            directory = True
        elif arg in {"-m", "-o", "-g"}:
            value = args.pop(0)
            if arg == "-m":
                mode = int(value, 8)
        elif arg != "--":
            paths.append(local(arg))
    if directory:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(mode)
    else:
        src, dst = paths
        if failure == "unit_publish" and ".service.next." in dst.name:
            sys.exit(1)
        shutil.copyfile(src, dst)
        dst.chmod(mode)
elif name == "mv":
    src, dst = [local(arg) for arg in args if arg not in {"-T", "-Tf", "--"}]
    src.replace(dst)
    if failure == "kill_after_unit_publish" and dst == root / "etc/systemd/system/duckmind.service":
        os.kill(os.getppid(), signal.SIGKILL)
    if failure == "kill_after_pointer_publish" and dst == root / "opt/duckmind/current" and src.name.startswith(".current-next."):
        os.kill(os.getppid(), signal.SIGKILL)
elif name == "cp":
    src, dst = [local(arg) for arg in args if arg not in {"-p", "--"}]
    shutil.copy2(src, dst)
elif name == "ln":
    assert args[0] == "-s"
    local(args[2]).symlink_to(args[1])
elif name == "rm":
    for arg in args:
        if arg not in {"-f", "--"}:
            local(arg).unlink(missing_ok=True)
elif name == "mktemp":
    directory = args[0] == "-d"
    template = args[-1]
    prefix, suffix = template.split("XXXXXX")
    parent = local(str(Path(prefix).parent))
    prefix = Path(prefix).name
    if directory:
        print(tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=parent))
    else:
        fd, filename = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
        os.close(fd)
        print(filename)
elif name == "python3":
    if args[:2] == ["-m", "venv"]:
        if failure == "venv":
            sys.exit(1)
        bin_dir = local(args[2]) / "bin"
        bin_dir.mkdir(parents=True)
        for executable in ("pip", "duckmindd", "duckmindctl"):
            # pip is executed directly; the others are inspected by sudo below.
            path = bin_dir / executable
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
    else:
        if len(args) > 2 and "src.backup" in args[1]:
            local(args[2])
            local(args[3])
            if failure == "backup":
                sys.exit(1)
            if failure == "signal_backup":
                os.kill(os.getppid(), signal.SIGTERM)
                sys.exit(1)
        sys.exit(subprocess.call([sys.executable, *args]))
elif name == "systemctl":
    state_path = root / "systemctl.json"
    state = json.loads(state_path.read_text())
    command = args[0]
    unit = root / "etc/systemd/system/duckmind.service"
    current = root / "opt/duckmind/current"
    target = current.resolve().name
    if command == "is-active":
        sys.exit(0 if state["active"] else 3)
    if command == "is-enabled":
        sys.exit(0 if state["enabled"] else 1)
    if command == "stop":
        if failure == "stop":
            sys.exit(1)
        if not unit.exists():
            sys.exit(5)  # systemd cannot stop a unit that has never been installed
        state["active"] = False
    elif command == "start":
        if failure == f"start_{target}":
            sys.exit(1)
        assert unit.exists() and current.exists()
        state["active"] = True
    elif command == "enable":
        assert unit.exists()
        state["enabled"] = True
    elif command == "disable":
        assert unit.exists()
        state["enabled"] = False
    elif command != "daemon-reload":
        raise RuntimeError(f"unimplemented systemctl command: {command}")
    state_path.write_text(json.dumps(state))
elif name == "sudo":
    assert args[:2] == ["-u", "duckmind"]
    command = local(args[2]).resolve()
    # Model the unprivileged account's traversal, skipping pytest's outer temp dir.
    for parent in command.parents:
        if parent == root:
            break
        if not parent.stat().st_mode & stat.S_IXOTH:
            print(f"permission denied: {parent}", file=sys.stderr)
            sys.exit(126)
    assert command.stat().st_mode & stat.S_IXOTH
    if args[3:] == ["--help"]:
        sys.exit(0)
    assert command.name == "duckmindctl" and args[3:] == ["status"]
    release = command.parent.parent.parent.name
    healthy = failure != f"health_{release}"
    print(json.dumps({"ok": True, "result": {
        "database_integrity": "ok", "play_enabled": not healthy,
    }}))
    if failure == f"cli_exit_{release}":
        sys.exit(1)
else:
    raise RuntimeError(f"unknown command double: {name}")
"""


class BoardSandbox:
    def __init__(self, root: Path):
        self.root = root
        self.app = root / "opt/duckmind"
        self.unit = root / "etc/systemd/system/duckmind.service"
        self.unit.parent.mkdir(parents=True)
        (root / "run").mkdir()
        self.db = root / "var/lib/duckmind/duckmind.sqlite3"
        self.db.parent.mkdir(parents=True)
        self.deploy = root / "input/deploy"
        self.deploy.mkdir(parents=True)
        (self.deploy.parent / "README.md").write_text("Candidate README\n")
        (self.deploy / "duckmind.service").write_text("candidate-unit\n")
        self.wheel = root / "input/duckmind-0.1.0-py3-none-any.whl"
        self.wheel.write_bytes(b"wheel installed by a command double")
        self.sha = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.state = root / "systemctl.json"
        self.state.write_text(json.dumps({"active": False, "enabled": False}))
        for script in ("install-board.sh", "rollback-board.sh", "board-boot-guard.sh"):
            source = (DEPLOY / script).read_text()
            for path in (
                "/opt/duckmind",
                "/etc/systemd/system/duckmind.service",
                "/run/duckmind-install.lock",
                "/var/lib/duckmind",
            ):
                source = source.replace(path, str(root / path.lstrip("/")))
            (self.deploy / script).write_text(source)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for command in (
            "id",
            "getent",
            "flock",
            "sleep",
            "chown",
            "groupadd",
            "useradd",
            "usermod",
            "sha256sum",
            "install",
            "mv",
            "cp",
            "ln",
            "rm",
            "mktemp",
            "python3",
            "systemctl",
            "sudo",
        ):
            executable = bin_dir / command
            executable.write_text(f"#!{sys.executable}\n" + COMMAND_DOUBLE)
            executable.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "DEPLOY_TEST_ROOT": str(root),
        }

    def add_release(self, release="old", *, unit="old-unit\n"):
        target = self.app / "releases" / release
        bin_dir = target / "venv/bin"
        bin_dir.mkdir(parents=True)
        target.chmod(0o755)
        for name in ("duckmindd", "duckmindctl"):
            executable = bin_dir / name
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
        (target / "duckmind.service").write_text(unit)
        (target / "RELEASE_ID").write_text(f"{release}\n")
        guard = target / "board-boot-guard.sh"
        guard.write_text((self.deploy / "board-boot-guard.sh").read_text())
        guard.chmod(0o755)
        return target

    def existing(self, *, active=True, enabled=True):
        self.add_release()
        (self.app / "current").symlink_to("releases/old")
        self.unit.write_text("old-unit\n")
        self.state.write_text(json.dumps({"active": active, "enabled": enabled}))
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE evidence (value TEXT)")
            db.execute("INSERT INTO evidence VALUES ('keep me')")

    def run(self, script="install-board.sh", *, failure="", release="new", optimize=False):
        args = [str(self.wheel), self.sha, release] if script == "install-board.sh" else [release]
        return subprocess.run(
            ["/bin/sh", str(self.deploy / script), *args],
            env={
                **self.env,
                "DEPLOY_TEST_FAILURE": failure,
                "PYTHONOPTIMIZE": "1" if optimize else "",
            },
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def current(self):
        return os.readlink(self.app / "current")

    def guard(self, release="new"):
        return subprocess.run(
            ["/bin/sh", str(self.app / "releases" / release / "board-boot-guard.sh")],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

    def assert_original(self, *, active=True, enabled=True):
        assert self.current() == "releases/old"
        assert self.unit.read_text() == "old-unit\n"
        assert json.loads(self.state.read_text()) == {"active": active, "enabled": enabled}
        with sqlite3.connect(self.db) as db:
            assert db.execute("SELECT value FROM evidence").fetchall() == [("keep me",)]


@pytest.fixture
def board(tmp_path):
    return BoardSandbox(tmp_path)


def test_first_install_is_traversable_passive_and_starts_once(board):
    result = board.run()
    assert result.returncode == 0, result.stderr
    assert board.current() == "releases/new"
    assert stat.S_IMODE((board.app / "releases/new").stat().st_mode) == 0o755
    assert board.unit.read_text() == "candidate-unit\n"
    assert (board.app / "releases/new/RELEASE_ID").read_text() == "new\n"
    assert stat.S_IMODE((board.app / "releases/new/RELEASE_ID").stat().st_mode) == 0o644
    assert stat.S_IMODE((board.app / "releases/new/board-boot-guard.sh").stat().st_mode) == 0o755
    assert board.guard().returncode == 0
    assert json.loads(board.state.read_text()) == {"active": True, "enabled": True}
    calls = [json.loads(line) for line in (board.root / "calls.jsonl").read_text().splitlines()]
    assert sum(call[:2] == ["systemctl", "start"] for call in calls) == 1


@pytest.mark.parametrize(
    "failure",
    ["venv", "backup", "signal_backup", "unit_publish", "start_new", "health_new", "cli_exit_new"],
)
def test_upgrade_failure_restores_application_unit_and_preserves_database(board, failure):
    board.existing()
    result = board.run(failure=failure)
    assert result.returncode != 0
    board.assert_original()


def test_upgrade_backup_contains_original_data(board):
    board.existing()
    result = board.run()
    assert result.returncode == 0, result.stderr
    backups = list((board.db.parent / "backups").glob("before-new.*.sqlite3"))
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("SELECT value FROM evidence").fetchall() == [("keep me",)]


def test_failed_upgrade_does_not_activate_previously_disabled_service(board):
    board.existing(active=False, enabled=False)
    result = board.run(failure="health_new")
    assert result.returncode != 0
    board.assert_original(active=False, enabled=False)


def test_failed_first_install_removes_pointer_unit_and_enablement(board):
    result = board.run(failure="health_new")
    assert result.returncode != 0
    assert not (board.app / "current").is_symlink()
    assert not board.unit.exists()
    assert json.loads(board.state.read_text()) == {"active": False, "enabled": False}
    assert (board.app / "releases/new").exists()  # retained for diagnosis


def test_health_gate_cannot_be_bypassed_with_python_optimization(board):
    board.existing()
    result = board.run(failure="health_new", optimize=True)
    assert result.returncode != 0
    board.assert_original()


def test_rollback_uses_matching_archived_unit(board):
    board.existing()
    board.add_release("older", unit="older-unit\n")
    result = board.run("rollback-board.sh", release="older")
    assert result.returncode == 0, result.stderr
    assert board.current() == "releases/older"
    assert board.unit.read_text() == "older-unit\n"


@pytest.mark.parametrize(
    "failure", ["unit_publish", "start_older", "health_older", "cli_exit_older"]
)
def test_failed_rollback_restores_application_and_unit(board, failure):
    board.existing()
    board.add_release("older", unit="older-unit\n")
    result = board.run("rollback-board.sh", release="older", failure=failure)
    assert result.returncode != 0
    board.assert_original()


def test_rejects_non_release_current_pointer_before_service_mutation(board):
    board.existing()
    (board.app / "current").unlink()
    (board.app / "current").symlink_to("../somewhere-else")
    result = board.run()
    assert result.returncode != 0
    assert "local Duck Mind release" in result.stderr
    assert json.loads(board.state.read_text())["active"] is True


def test_install_refuses_split_current_unit_before_staging_or_stopping(board):
    board.existing()
    board.unit.write_text("unmatched-unit\n")

    result = board.run()

    assert result.returncode != 0
    assert "boot guard" in result.stderr
    assert not (board.app / "releases/new").exists()
    assert board.current() == "releases/old"
    assert board.unit.read_text() == "unmatched-unit\n"
    assert json.loads(board.state.read_text())["active"] is True
    calls = [json.loads(line) for line in (board.root / "calls.jsonl").read_text().splitlines()]
    assert ["systemctl", "stop", "duckmind.service"] not in calls


def test_install_refuses_orphan_unit_without_current_pointer(board):
    board.unit.write_text("orphan-unit\n")

    result = board.run()

    assert result.returncode != 0
    assert "unit exists without" in result.stderr
    assert not (board.app / "releases/new").exists()
    assert board.unit.read_text() == "orphan-unit\n"


def test_install_refuses_missing_unit_with_current_pointer(board):
    board.existing()
    board.unit.unlink()

    result = board.run()

    assert result.returncode != 0
    assert "boot guard" in result.stderr
    assert not (board.app / "releases/new").exists()
    assert board.current() == "releases/old"
    assert json.loads(board.state.read_text())["active"] is True


def test_install_refuses_current_release_without_executable_boot_guard(board):
    board.existing()
    guard = board.app / "releases/old/board-boot-guard.sh"
    guard.chmod(0o644)

    result = board.run()

    assert result.returncode != 0
    assert "boot guard" in result.stderr
    assert not (board.app / "releases/new").exists()
    assert board.current() == "releases/old"
    assert json.loads(board.state.read_text())["active"] is True


def test_rollback_refuses_split_current_unit_before_stopping(board):
    board.existing()
    board.add_release("older", unit="older-unit\n")
    board.unit.write_text("unmatched-unit\n")

    result = board.run("rollback-board.sh", release="older")

    assert result.returncode != 0
    assert "boot guard" in result.stderr
    assert board.current() == "releases/old"
    assert board.unit.read_text() == "unmatched-unit\n"
    assert json.loads(board.state.read_text())["active"] is True
    calls = [json.loads(line) for line in (board.root / "calls.jsonl").read_text().splitlines()]
    assert ["systemctl", "stop", "duckmind.service"] not in calls


def test_rollback_refuses_missing_archived_unit_before_service_mutation(board):
    board.existing()
    target = board.add_release("older")
    (target / "duckmind.service").unlink()
    result = board.run("rollback-board.sh", release="older")
    assert result.returncode != 0
    board.assert_original()


def test_rollback_refuses_missing_boot_guard_before_service_mutation(board):
    board.existing()
    target = board.add_release("older")
    (target / "board-boot-guard.sh").unlink()
    result = board.run("rollback-board.sh", release="older")
    assert result.returncode != 0
    assert "boot guard" in result.stderr
    board.assert_original()


def test_rollback_refuses_symlinked_release_directory_before_service_mutation(board):
    board.existing()
    target = board.add_release("older")
    archive = board.app / "archived-older"
    target.rename(archive)
    target.symlink_to("../archived-older")

    result = board.run("rollback-board.sh", release="older")

    assert result.returncode != 0
    assert "symlinked release" in result.stderr
    board.assert_original()


@pytest.mark.parametrize("fault", ["missing", "mismatch", "symlink"])
def test_rollback_refuses_unbound_release_identity_before_service_mutation(board, fault):
    board.existing()
    target = board.add_release("older")
    record = target / "RELEASE_ID"
    if fault == "missing":
        record.unlink()
    elif fault == "mismatch":
        record.write_text("other\n")
    else:
        record.unlink()
        record.symlink_to(board.app / "releases/old/RELEASE_ID")
    result = board.run("rollback-board.sh", release="older")
    assert result.returncode != 0
    assert "release identity record" in result.stderr
    board.assert_original()


def test_stop_failure_keeps_current_configuration_and_reports_incomplete_recovery(board):
    board.existing()
    result = board.run(failure="stop")
    assert result.returncode != 0
    assert "automatic recovery incomplete: cannot stop" in result.stderr
    board.assert_original()


def test_install_accepts_uppercase_sha256(board):
    board.sha = board.sha.upper()
    result = board.run()
    assert result.returncode == 0, result.stderr


def test_hash_mismatch_never_stops_existing_service(board):
    board.existing()
    board.sha = "0" * 64
    result = board.run()
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    board.assert_original()


@pytest.mark.parametrize("fault", ["unit", "identity", "pointer", "missing_guard"])
def test_boot_guard_refuses_split_or_incomplete_release(board, fault):
    assert board.run().returncode == 0
    if fault == "unit":
        board.unit.write_text("other-unit\n")
    elif fault == "identity":
        (board.app / "releases/new/RELEASE_ID").write_text("other\n")
    elif fault == "pointer":
        (board.app / "current").unlink()
        (board.app / "current").symlink_to("../outside")
    else:
        (board.app / "releases/new/board-boot-guard.sh").unlink()
    assert board.guard().returncode != 0


def test_sigkill_between_unit_and_pointer_publication_blocks_boot(board):
    board.existing()
    result = board.run(failure="kill_after_unit_publish")
    assert result.returncode != 0
    assert board.current() == "releases/old"
    assert board.unit.read_text() == "candidate-unit\n"
    assert board.guard("old").returncode != 0
    retry = board.run(release="third")
    assert retry.returncode != 0
    assert "boot guard" in retry.stderr
    assert not (board.app / "releases/third").exists()
    assert board.current() == "releases/old"


def test_sigkill_after_paired_publication_allows_passive_boot_guard(board):
    board.existing()
    result = board.run(failure="kill_after_pointer_publish")
    assert result.returncode != 0
    assert board.current() == "releases/new"
    assert board.unit.read_text() == "candidate-unit\n"
    assert board.guard("new").returncode == 0
    follow_up = board.run(release="third")
    assert follow_up.returncode == 0, follow_up.stderr
    assert board.current() == "releases/third"
    assert board.guard("third").returncode == 0


@pytest.mark.parametrize("release", ["-new", "_new", ".new", "a" * 65])
def test_invalid_release_identity_never_starts_install(board, release):
    result = board.run(release=release)
    assert result.returncode != 0
    assert "invalid release ID" in result.stderr
    assert not (board.app / "current").exists()
