"""Pollen's usage data on a real duck (owner 2026-11-08: off). updaterd answers ``telemetry.status``
and ``telemetry.set`` on its own socket; the bridge on a real duck turns it off once when it starts
(unless ``--keep-pollen-telemetry``), logs it, and ``check`` prints what the duck does. A fake
updaterd on a unix socket stands in for the duck's."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from duckmind.remotebody import bridge
from duckmind.remotebody.__main__ import describe_body
from duckmind.remotebody.telemetry import describe, settle
from duckmind.tmpdir import short_tmp


class FakeUpdaterd:
    """updaterd's two telemetry calls as ``updater/src/ipc.rs`` answers them (``enabled`` stored,
    ``active`` = enabled and not blocked by the environment); ``knows_telemetry=False`` is an
    updaterd older than 0.18.3 (method not found). Every request is recorded."""

    def __init__(
        self, path: str, *, enabled: bool, env_blocked: bool = False, knows_telemetry: bool = True
    ) -> None:
        self.enabled, self.env_blocked, self.knows = enabled, env_blocked, knows_telemetry
        self.calls: list[dict[str, Any]] = []
        self.connections = 0
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen(4)
        threading.Thread(target=self._accept, daemon=True).start()

    def methods(self) -> list[str]:
        return [c["method"] for c in self.calls]

    def close(self) -> None:
        self._server.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                request = json.loads(line)
                self.calls.append(request)
                method, rid = request["method"], request["id"]
                if not self.knows or not method.startswith("telemetry."):
                    answer = {"error": {"code": -32601, "message": f"unknown method {method}"}}
                else:
                    if method == "telemetry.set":
                        self.enabled = bool(request["params"]["enabled"])
                    answer = {
                        "result": {
                            "enabled": self.enabled,
                            "active": self.enabled and not self.env_blocked,
                        }
                    }
                f.write(json.dumps({"jsonrpc": "2.0", "id": rid, **answer}).encode() + b"\n")
                f.flush()


@pytest.fixture
def sock_path() -> Any:
    with tempfile.TemporaryDirectory(dir=short_tmp()) as d:
        yield str(Path(d) / "u.sock")


def _start(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list]:
    """The bridge's start-up body info, as ``bridge.main`` builds it, with its logged events."""
    monkeypatch.setattr(bridge, "real_body_info", lambda: {"backend": "real"})
    events: list[tuple[str, dict[str, Any]]] = []
    args = bridge.bridge_parser().parse_args(["--token-file", "unused", *argv])
    info = bridge.body_info_for(args, lambda event, **fields: events.append((event, fields)))
    return info, events


def test_the_bridge_on_a_real_duck_turns_pollens_telemetry_off_once_and_logs_it(
    sock_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    updaterd = FakeUpdaterd(sock_path, enabled=True)
    info, events = _start(["--updaterd", sock_path], monkeypatch)
    assert updaterd.methods() == ["telemetry.status", "telemetry.set"]
    assert updaterd.calls[1]["params"] == {"enabled": False}
    assert updaterd.enabled is False
    report = info["telemetry"]
    assert report["enabled"] is False and report["active"] is False and report["turned_off"]
    assert [e for e, _ in events] == ["telemetry"] and events[0][1]["turned_off"] is True
    line = describe(report)
    assert line.startswith("usage data (Pollen telemetry): off") and "turned it off" in line
    assert line in describe_body(info)


def test_an_already_off_duck_is_only_asked(sock_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    updaterd = FakeUpdaterd(sock_path, enabled=False)
    info, _ = _start(["--updaterd", sock_path], monkeypatch)
    assert updaterd.methods() == ["telemetry.status"]
    assert not info["telemetry"]["turned_off"]
    assert "already off" in describe(info["telemetry"])


def test_keep_pollen_telemetry_leaves_the_switch_alone_and_check_says_it_is_on(
    sock_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    updaterd = FakeUpdaterd(sock_path, enabled=True)
    info, _ = _start(["--updaterd", sock_path, "--keep-pollen-telemetry"], monkeypatch)
    assert updaterd.methods() == ["telemetry.status"] and updaterd.enabled
    line = describe(info["telemetry"])
    assert "ON, the duck sends it" in line and "--keep-pollen-telemetry" in line


def test_an_environment_blocked_updaterd_is_still_switched_off(sock_path: str) -> None:
    updaterd = FakeUpdaterd(sock_path, enabled=True, env_blocked=True)
    report = settle(sock_path)
    assert updaterd.methods() == ["telemetry.status", "telemetry.set"]
    assert report["enabled"] is False and report["active"] is False


def test_a_simulated_body_never_dials_updaterd(
    sock_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    updaterd = FakeUpdaterd(sock_path, enabled=True)
    info, events = _start(["--updaterd", sock_path, "--backend", "simulator"], monkeypatch)
    assert info == {"backend": "simulator"} and events == []
    assert updaterd.connections == 0
    assert not any("usage data" in line for line in describe_body(info))


def test_an_old_or_absent_updaterd_never_stops_the_bridge(
    sock_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent, _ = _start(["--updaterd", sock_path], monkeypatch)
    assert absent["telemetry"]["error"].startswith("cannot reach updaterd")
    assert "UNKNOWN" in describe(absent["telemetry"])
    FakeUpdaterd(sock_path, enabled=True, knows_telemetry=False)
    old, _ = _start(["--updaterd", sock_path], monkeypatch)
    assert "telemetry.status" in old["telemetry"]["error"]
    assert old["telemetry"]["enabled"] is None
    assert "not reported" in describe(None)
