"""Pollen's LAN doors round the bridge (duckmind.remotebody.doors): a real duck's bridge does not
serve while mediad's or netpadd's door is open, and ``tools/duck/close-mediad.sh`` closes them.

The doors here are fake listeners on 127.0.0.1; ``nft`` and ``systemctl`` are stubs in a folder
that is the script's whole PATH, so no firewall of this machine can be reached.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.remotebody import __main__ as cli
from duckmind.remotebody import bridge as B
from duckmind.remotebody import doors

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "duck" / "close-mediad.sh"
TOKEN = b"0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture
def lan():
    """127.0.0.1 as "the duck's LAN address": one TCP door that answers, one TCP port nothing
    listens on, one UDP port something holds, one UDP port that is free."""
    door = socket.socket()
    door.bind(("127.0.0.1", 0))
    door.listen(8)
    shut = socket.socket()
    shut.bind(("127.0.0.1", 0))  # (bound, never listening: a closed door)
    pad = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pad.bind(("127.0.0.1", 0))
    free = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    free.bind(("127.0.0.1", 0))
    free_port = free.getsockname()[1]
    free.close()
    yield SimpleNamespace(
        door=door.getsockname()[1],
        shut=shut.getsockname()[1],
        pad=pad.getsockname()[1],
        free=free_port,
    )
    for sock in (door, shut, pad):
        sock.close()


def test_a_look_finds_the_door_that_answers_and_the_udp_port_that_is_held(lan: Any) -> None:
    found = doors.look("127.0.0.1", [lan.door, lan.shut], lan.pad, timeout_s=0.5)
    assert found.tcp == (lan.door,) and found.netpadd == lan.pad and found.open
    closed = doors.look("127.0.0.1", [lan.shut], lan.free, timeout_s=0.5)
    assert closed.tcp == () and closed.netpadd is None and not closed.open
    assert doors.look(None).open  # (no address to knock on: nothing is known to be closed)


def bridge_argv(tmp_path: Path, lan: Any, *more: str, tcp: int, udp: int) -> list[str]:
    return [
        "--backend",
        "real",
        "--listen",
        "127.0.0.1:0",
        "--token-file",
        str(tmp_path / "token"),
        "--lan-address",
        "127.0.0.1",
        "--mediad-ports",
        str(tcp),
        "--netpadd-port",
        str(udp),
        "--events",
        str(tmp_path / "events.jsonl"),
        "--robotd",
        str(tmp_path / "no-robotd.sock"),
        "--updaterd",
        str(tmp_path / "no-updaterd.sock"),
        *more,
    ]


def events_of(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_real_ducks_bridge_does_not_start_while_mediads_door_answers(
    tmp_path: Path, lan: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # (no token file exists: the refusal comes before anything else is read or opened)
    assert B.main(bridge_argv(tmp_path, lan, tcp=lan.door, udp=lan.free)) == 2
    err = capsys.readouterr().err
    assert f"mediad's LAN doors are OPEN: 127.0.0.1 TCP port {lan.door}" in err
    assert "sudo sh tools/duck/close-mediad.sh --brain" in err
    assert "--allow-open-mediad" in err
    events = events_of(tmp_path)
    assert [e["event"] for e in events] == ["doors_open"]
    assert events[0]["tcp_open"] == [lan.door] and events[0]["allowed"] is False


def test_a_real_ducks_bridge_does_not_start_while_netpadd_listens(
    tmp_path: Path, lan: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert B.main(bridge_argv(tmp_path, lan, tcp=lan.shut, udp=lan.pad)) == 2
    err = capsys.readouterr().err
    assert f"something listens on UDP {lan.pad}" in err and "[netpad] enabled = false" in err
    assert events_of(tmp_path)[0]["netpadd"] == lan.pad


def serve_until_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge's main loop sleeps for good once it serves: the first such sleep is Ctrl-C."""
    import time

    def sleep(seconds: float) -> None:
        if seconds >= 3600:
            raise KeyboardInterrupt
        time.sleep(seconds)

    fake = SimpleNamespace(**{name: getattr(time, name) for name in dir(time)})
    fake.sleep = sleep
    monkeypatch.setattr(B, "time", fake)


def write_token(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_bytes(TOKEN)
    os.chmod(token, 0o600)


def test_with_the_doors_closed_it_serves_and_tells_the_brain_what_it_found(
    tmp_path: Path, lan: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_token(tmp_path)
    serve_until_interrupted(monkeypatch)
    told: list[dict[str, Any]] = []
    made = B.Bridge

    def bridge(**kw: Any) -> B.Bridge:
        told.append(kw["body_info"])
        return made(**kw)

    monkeypatch.setattr(B, "Bridge", bridge)
    assert B.main(bridge_argv(tmp_path, lan, tcp=lan.shut, udp=lan.free)) == 0
    assert "OPEN" not in capsys.readouterr().err
    assert [
        e["event"] for e in events_of(tmp_path) if e["event"].startswith(("doors", "listen"))
    ] == [
        "doors_closed",
        "listening",
    ]
    assert told[0]["doors"] == {
        "host": "127.0.0.1",
        "tcp_open": [],
        "netpadd": None,
        "allowed": False,
    }


def test_allow_open_mediad_serves_and_says_what_is_not_covered(
    tmp_path: Path, lan: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_token(tmp_path)
    serve_until_interrupted(monkeypatch)
    argv = bridge_argv(tmp_path, lan, "--allow-open-mediad", tcp=lan.door, udp=lan.free)
    assert B.main(argv) == 0
    err = capsys.readouterr().err
    assert "are OPEN" in err and "serving anyway (--allow-open-mediad)" in err
    names = [e["event"] for e in events_of(tmp_path)]
    assert "doors_open" in names and "listening" in names


def test_a_simulated_bodys_bridge_knocks_on_nothing(
    tmp_path: Path, lan: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_token(tmp_path)
    serve_until_interrupted(monkeypatch)
    argv = bridge_argv(tmp_path, lan, tcp=lan.door, udp=lan.pad)
    argv[1] = "simulator"
    assert B.main(argv) == 0
    assert not [e for e in events_of(tmp_path) if e["event"].startswith("doors")]


def test_the_bridge_knocks_on_the_address_it_listens_on(
    lan: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--listen <duck-address>:8797`` is the address knocked on; a loopback or any-address
    listen asks the default route's address instead (a firewall never closes loopback)."""
    asked: list[str | None] = []

    def look(host: str | None, *rest: Any, **kw: Any) -> doors.Doors:
        asked.append(host)
        return doors.Doors(host)

    monkeypatch.setattr(doors, "look", look)
    monkeypatch.setattr(doors, "own_lan_address", lambda: "10.0.0.7")
    for listen in ("10.0.0.50:8797", "127.0.0.1:8797", "0.0.0.0:8797"):
        args = B.bridge_parser().parse_args(["--token-file", "t", "--listen", listen])
        B.doors_for(args, lambda event, **fields: None)
    assert asked == ["10.0.0.50", "10.0.0.7", "10.0.0.7"]


def test_checks_words_for_what_the_bridge_found() -> None:
    assert cli.describe_bridge_doors({"tcp_open": [], "netpadd": None, "allowed": False}) == []
    lines = cli.describe_bridge_doors({"tcp_open": [8443], "netpadd": 4210, "allowed": True})
    assert "UDP 4210" in lines[0] and "[netpad] enabled = false" in lines[1]
    assert "--allow-open-mediad" in lines[2]


# ── tools/duck/close-mediad.sh ───────────────────────────────────────────────

STUB = """#!/bin/sh
echo "$(basename "$0") $*" >>"$STUB_LOG"
if [ "$(basename "$0")" = nft ]; then
    case "$*" in
        "-f -") cat >>"$STUB_LOG" ;;
        "list table inet duckmind") [ -e "$STUB_TABLE" ] || exit 1 ;;
    esac
    [ -z "${STUB_NFT_FAIL:-}" ] || exit 1
fi
exit 0
"""


@pytest.fixture
def duck_shell(tmp_path: Path):
    """Runs the script with a PATH of its own: the few tools it needs, and stub ``nft`` and
    ``systemctl`` that only write down what they were asked."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "sed", "cat", "mkdir", "dirname", "basename", "rm"):
        found = shutil.which(tool)
        assert found is not None
        (bin_dir / tool).symlink_to(found)
    log = tmp_path / "asked.log"
    root = tmp_path / "root"

    def stub(name: str) -> None:
        path = bin_dir / name
        path.write_text(STUB)
        path.chmod(0o755)

    def run(*args: str, nft: bool = True, table: bool = False, fail: bool = False) -> Any:
        for name in ("nft", "systemctl"):
            (bin_dir / name).unlink(missing_ok=True)
        stub("systemctl")
        if nft:
            stub("nft")
        table_mark = tmp_path / "table"
        table_mark.touch() if table else table_mark.unlink(missing_ok=True)
        env = {
            "PATH": str(bin_dir),
            "STUB_LOG": str(log),
            "STUB_TABLE": str(table_mark),
            "DESTDIR": str(root),
        }
        if fail:
            env["STUB_NFT_FAIL"] = "1"
        done = subprocess.run(
            [str(bin_dir / "sh"), str(SCRIPT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        done.asked = log.read_text() if log.exists() else ""
        return done

    return SimpleNamespace(run=run, root=root, log=log)


def test_print_shows_the_table_and_asks_nft_nothing(duck_shell: Any) -> None:
    done = duck_shell.run("--brain", "10.0.0.20", "--print")
    assert done.returncode == 0, done.stderr
    assert done.asked == ""
    table = done.stdout
    assert "table inet duckmind {" in table
    assert "tcp dport { 8443, 8080 } drop" in table and "udp dport 4210 drop" in table
    assert "tcp dport 8797 ip saddr != 10.0.0.20 drop" in table
    # the duck's own processes keep loopback; its own LAN address is closed like a LAN peer's
    assert 'iif "lo" ip daddr 127.0.0.0/8 accept' in table
    assert 'iif "lo" accept' not in table


@pytest.mark.parametrize(
    "address",
    ["", "10.0.0", "10.0.0.256", "1.2.3.4; reboot", "$(id)", "1.2.3.4 }", "a.b.c.d"],
)
def test_an_address_that_is_not_one_changes_nothing(duck_shell: Any, address: str) -> None:
    done = duck_shell.run("--brain", address)
    assert done.returncode == 2 and "IPv4 address" in done.stderr
    assert done.asked == "" and not duck_shell.root.exists()


def test_closing_loads_the_table_in_one_transaction_and_keeps_nothing_on_disk(
    duck_shell: Any,
) -> None:
    done = duck_shell.run("--brain", "10.0.0.20", "--bridge-port", "8800")
    assert done.returncode == 0, done.stderr
    lines = done.asked.splitlines()
    assert lines[0] == "nft -f -"
    assert lines[1:3] == ["table inet duckmind", "delete table inet duckmind"]
    assert "tcp dport 8800 ip saddr != 10.0.0.20 drop" in done.asked
    assert "systemctl" not in done.asked and not duck_shell.root.exists()
    assert "in memory only" in done.stdout


def test_at_boot_installs_the_unit_and_undo_takes_everything_out(duck_shell: Any) -> None:
    done = duck_shell.run("--brain", "10.0.0.20", "--at-boot")
    assert done.returncode == 0, done.stderr
    rules = duck_shell.root / "etc/duckmind/close-mediad.nft"
    unit = duck_shell.root / "etc/systemd/system/duckmind-close-mediad.service"
    assert "ip saddr != 10.0.0.20 drop" in rules.read_text()
    assert "-f /etc/duckmind/close-mediad.nft" in unit.read_text()
    assert "systemctl enable duckmind-close-mediad.service" in done.asked
    duck_shell.log.unlink()
    undone = duck_shell.run("--undo", table=True)
    assert undone.returncode == 0, undone.stderr
    assert "nft delete table inet duckmind" in undone.asked
    assert "systemctl disable duckmind-close-mediad.service" in undone.asked
    assert not rules.exists() and not unit.exists()


def test_when_nft_refuses_or_is_missing_it_says_so_and_fails(duck_shell: Any) -> None:
    refused = duck_shell.run("--brain", "10.0.0.20", "--at-boot", fail=True)
    assert refused.returncode == 2 and "nothing was changed" in refused.stderr
    assert not duck_shell.root.exists()  # (no boot unit for a table that did not load)
    missing = duck_shell.run("--brain", "10.0.0.20", nft=False)
    assert missing.returncode == 2 and "nft is not on this duck" in missing.stderr


def test_status_and_undo_without_the_table_change_nothing(duck_shell: Any) -> None:
    status = duck_shell.run("--status")
    assert (
        status.returncode == 0 and "NOT in" in status.stdout and "not kept at boot" in status.stdout
    )
    undo = duck_shell.run("--undo")
    assert undo.returncode == 0 and "delete" not in undo.asked
