"""The remote-body link (duckmind.remotebody): token, signed frames, deadman, reconnect, LAN only.

A fake robotd on a unix socket stands in for the duck; the bridge and the brain's link run in this
process and talk over loopback TCP, through :class:`NetShim` where a cut or a delay is needed.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from duckmind.body import BodyIntent
from duckmind.remotebody import link as L
from duckmind.remotebody.bridge import Bridge, Channel, SafetyKeeper, check_request
from duckmind.remotebody.client import RemoteLink
from duckmind.remotebody.netshim import Impairment, NetShim
from duckmind.robotd import RobotdBodyPort, RobotdProtocolError
from duckmind.tmpdir import short_tmp

TOKEN = b"0123456789abcdef0123456789abcdef0123456789abcdef"


class FakeRobotd:
    """hello / subscribe / stop / do answered; robot.state at 50 Hz with a settable policy; every
    request and notification recorded with its arrival time."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.policy = "stand"
        self.battery: dict[str, Any] | None = None  # robot.health's battery (None: not read yet)
        self.received: list[tuple[float, dict[str, Any]]] = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen(16)
        self._conns: list[socket.socket] = []
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def methods(self, since: float = 0.0) -> list[str]:
        return [m.get("method", "") for t, m in list(self.received) if t >= since]

    def first(self, method: str, since: float = 0.0) -> float | None:
        return next(
            (t for t, m in list(self.received) if t >= since and m.get("method") == method), None
        )

    def drop_all(self) -> None:
        for conn in list(self._conns):
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        self._conns.clear()

    def close(self) -> None:
        self._stop.set()
        self._server.close()
        self.drop_all()

    def _accept(self) -> None:
        self._server.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                continue
            self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buffer = b""
        subscribed = False
        conn.settimeout(0.02)
        while not self._stop.is_set():
            try:
                data = conn.recv(65536)
                if not data:
                    return
                buffer += data
            except TimeoutError:
                pass
            except OSError:
                return
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                message = json.loads(line)
                self.received.append((time.monotonic(), message))
                if "id" in message:
                    result: dict[str, Any] = {"accepted": True}
                    if message["method"] == "robot.subscribe":
                        subscribed = True
                        result = {"accepted": True, "walk": "velstand.onnx", "skills": []}
                    if message["method"] == "hello":
                        result = {"api_version": 37}
                    if message["method"] == "robot.health":
                        result = {"healthy": True, "degraded": False}
                        if self.battery is not None:
                            result["battery"] = dict(self.battery)
                    try:
                        conn.sendall(
                            json.dumps(
                                {"jsonrpc": "2.0", "id": message["id"], "result": result}
                            ).encode()
                            + b"\n"
                        )
                    except OSError:
                        return
            if subscribed:
                frame = {
                    "jsonrpc": "2.0",
                    "method": "robot.state",
                    "params": {
                        "t": time.monotonic(),
                        "policy": self.policy,
                        "move": {"requested": [0, 0, 0], "applied": [0, 0, 0], "limited_by": []},
                        "safety": {"fallen": False, "limp": False},
                        "joints": [0.0] * 15,
                    },
                }
                try:
                    conn.sendall(json.dumps(frame).encode() + b"\n")
                except OSError:
                    return


@pytest.fixture
def rig():
    """A fake robotd, its bridge (with the real safety keeper) and a shim in front of it."""
    with tempfile.TemporaryDirectory(prefix="dm-rb-t-", dir=short_tmp()) as directory:
        robotd = FakeRobotd(os.path.join(directory, "r.sock"))
        events: list[tuple[float, str, dict[str, Any]]] = []

        def log(event: str, **fields: Any) -> None:
            events.append((time.monotonic(), event, fields))

        keeper = SafetyKeeper(
            lambda: RobotdBodyPort(robotd.path, backend="simulator", timeout_s=1.0, hz=10), log=log
        )
        bridge = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={"robotd": Channel("robotd", unix=robotd.path)},
            keeper=keeper,
            body_info={"backend": "simulator"},
            log=log,
        )
        bridge.start()
        shim = NetShim(("127.0.0.1", bridge.port), seed=7).start()
        links: list[RemoteLink] = []

        def connect(token: bytes = TOKEN, via_shim: bool = True, **kw: Any) -> RemoteLink:
            port = shim.port if via_shim else bridge.port
            link = RemoteLink(f"remote://127.0.0.1:{port}", token, **kw)
            links.append(link)
            return link

        yield {
            "robotd": robotd,
            "bridge": bridge,
            "shim": shim,
            "connect": connect,
            "events": events,
        }
        for link in links:
            link.close()
        shim.close()
        bridge.close()
        robotd.close()


def wait_for(check, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return False


def test_the_brain_drives_robotd_through_the_link(rig: dict[str, Any]) -> None:
    link = rig["connect"]()
    link.start(timeout_s=3)
    port = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    try:
        assert port.capabilities["walk"] == "velstand.onnx"
        assert port.observe().policy == "stand"
        port.dispatch_intent(BodyIntent("robot.move", {"vx": 0.12, "vy": 0.0, "vyaw": 0.0}))
        assert wait_for(lambda: "robot.move" in rig["robotd"].methods())
        assert port.stop().accepted
    finally:
        port.close()
    assert link.stats()["rtt_ms_p50"] is not None


def test_a_wrong_token_is_refused_and_robotd_hears_nothing(rig: dict[str, Any]) -> None:
    link = rig["connect"](token=b"f" * 48)
    with pytest.raises(ConnectionError, match="refused our token"):
        link.start(timeout_s=1.5)
    assert set(rig["robotd"].methods()) <= {"hello", "robot.subscribe"}  # (the keeper's own)
    assert not rig["bridge"].link_up
    assert any(event == "auth_failed" for _, event, _ in rig["events"])


def test_a_bridge_that_does_not_hold_the_token_is_not_trusted() -> None:
    """Mutual: a fake 'body' that accepts anyone (no proof) never gets the brain's streams."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def impostor() -> None:
        conn, _ = server.accept()
        L.send_json_line(conn, {"protocol": L.PROTOCOL, "nonce": "ab" * 16})
        L.read_json_line(conn)
        L.send_json_line(conn, {"ok": True, "proof": "00" * 32, "channels": ["robotd"]})
        time.sleep(0.5)
        conn.close()

    threading.Thread(target=impostor, daemon=True).start()
    link = RemoteLink(f"remote://127.0.0.1:{server.getsockname()[1]}", TOKEN)
    with pytest.raises(ConnectionError, match="does not hold our token"):
        link.start(timeout_s=1.0)
    server.close()


def test_link_silence_stops_then_sits_the_duck_within_the_deadman(rig: dict[str, Any]) -> None:
    robotd = rig["robotd"]
    link = rig["connect"]()
    link.start(timeout_s=3)
    port = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    port.dispatch_intent(BodyIntent("robot.move", {"vx": 0.15, "vy": 0.0, "vyaw": 0.0}))
    assert wait_for(lambda: "robot.move" in robotd.methods())
    cut = rig["shim"].cut()
    assert wait_for(lambda: robotd.first("robot.stop", since=cut) is not None, 2.0)
    stop_after = robotd.first("robot.stop", since=cut) - cut
    # the deadman counts from the last frame heard, which came up to one heartbeat before the cut
    assert L.DEADMAN_S - L.HEARTBEAT_S <= stop_after <= L.DEADMAN_S + 0.25
    assert wait_for(lambda: robotd.first("robot.do", since=cut) is not None, 2.0)
    sit = [m for t, m in robotd.received if t >= cut and m.get("method") == "robot.do"]
    assert sit[0]["params"] == {"skill": "sit_toggle"}
    robotd.policy = "sit"
    assert wait_for(lambda: any(e == "sat" for _, e, _ in rig["events"]), 2.0)
    port.close()


def test_a_sitting_or_fallen_duck_is_only_stopped(rig: dict[str, Any]) -> None:
    robotd = rig["robotd"]
    robotd.policy = "sit"
    link = rig["connect"]()
    link.start(timeout_s=3)
    time.sleep(0.3)
    cut = rig["shim"].cut()
    assert wait_for(lambda: robotd.first("robot.stop", since=cut) is not None, 2.0)
    time.sleep(0.6)
    assert robotd.first("robot.do", since=cut) is None  # a toggle would stand it up


def test_a_cut_link_never_delivers_its_backlog_late(rig: dict[str, Any]) -> None:
    """Commands sent into a stalled link must not reach robotd after the deadman fired — even
    though TCP (and the shim) would deliver them once the network is back."""
    robotd = rig["robotd"]
    link = rig["connect"]()
    link.start(timeout_s=3)
    port = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    rig["shim"].cut()
    for _ in range(10):  # (into the stalled link, before either side has noticed)
        port.dispatch_intent(BodyIntent("robot.move", {"vx": 0.333, "vy": 0.0, "vyaw": 0.0}))
    assert wait_for(lambda: not rig["bridge"].link_up, 2.0)
    rig["shim"].restore()
    assert wait_for(lambda: link.up and rig["bridge"].link_up, 4.0)
    time.sleep(0.3)
    stale = [m for _, m in robotd.received if (m.get("params") or {}).get("vx") == 0.333]
    assert stale == []


def test_the_link_comes_back_and_the_keeper_lets_go(rig: dict[str, Any]) -> None:
    robotd = rig["robotd"]
    link = rig["connect"]()
    link.start(timeout_s=3)
    first = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    cut = rig["shim"].cut()
    assert wait_for(lambda: not link.up, 2.0)
    # robotd 'went away' on the brain's side: the old connection is closed
    with pytest.raises((ConnectionError, RobotdProtocolError, TimeoutError, OSError)):
        for _ in range(100):
            first.observe()
    restored = rig["shim"].restore()
    assert wait_for(lambda: link.up, 4.0)
    assert time.monotonic() - restored < 3.0
    second = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    assert second.observe().policy in {"stand", "walk", "sit"}
    assert any(e == "released" for _, e, _ in rig["events"])
    toggles = len([m for t, m in robotd.received if t >= cut and m.get("method") == "robot.do"])
    time.sleep(SafetyKeeperRetry.wait_s)
    after = len([m for t, m in robotd.received if t >= cut and m.get("method") == "robot.do"])
    assert after == toggles  # no more sit attempts once a brain is back
    second.close()
    assert link.connects == 2 and link.drops == 1


class SafetyKeeperRetry:
    wait_s = 0.8


def test_the_bridge_refuses_what_a_brain_never_needs(rig: dict[str, Any]) -> None:
    ok, answer = check_request(
        "robotd", b'{"jsonrpc":"2.0","id":4,"method":"robot.loadPolicy","params":{}}\n'
    )
    assert not ok and json.loads(answer)["error"]["code"] == -32601
    assert check_request(
        "robotd", b'{"jsonrpc":"2.0","method":"robot.shutdown","params":{}}\n'
    ) == (False, None)
    assert check_request("robotd", b'{"jsonrpc":"2.0","method":"robot.move","params":{}}\n') == (
        True,
        None,
    )
    ok, answer = check_request("world", b'{"op":"write","targets":[0]}\n')
    assert not ok and "refused" in json.loads(answer)["error"]
    assert check_request("world", b'{"op":"x_world"}\n') == (True, None)
    # and end to end: the call comes back as an error, robotd never sees it
    link = rig["connect"](via_shim=False)
    link.start(timeout_s=3)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(link.robotd_socket))
    sock.settimeout(2)
    sock.sendall(b'{"jsonrpc":"2.0","id":9,"method":"robot.relax","params":{}}\n')
    reply = json.loads(sock.makefile("rb").readline())
    assert reply["id"] == 9 and "refused" in reply["error"]["message"]
    sock.close()
    assert "robot.relax" not in rig["robotd"].methods()


def test_a_tampered_frame_ends_the_link_and_stops_the_duck(rig: dict[str, Any]) -> None:
    robotd = rig["robotd"]
    sock = socket.create_connection(("127.0.0.1", rig["bridge"].port))
    hello = L.read_json_line(sock)
    brain_nonce = "cd" * 16
    L.send_json_line(
        sock,
        {
            "protocol": L.PROTOCOL,
            "nonce": brain_nonce,
            "proof": L.proof(TOKEN, b"brain", hello["nonce"], brain_nonce),
        },
    )
    assert L.read_json_line(sock)["ok"] is True
    codec = L.FrameCodec(L.session_key(TOKEN, hello["nonce"], brain_nonce), b"b", b"B")
    codec.send(sock, b"P", 0, b"1")
    assert wait_for(lambda: rig["bridge"].link_up)
    since = time.monotonic()
    sock.sendall(b"O 1 2 6 " + b"00" * L.MAC_BYTES + b"\nrobotd")  # a forged open
    assert wait_for(lambda: robotd.first("robot.stop", since=since) is not None, 1.0)
    assert not rig["bridge"].link_up
    assert any("signature" in f.get("reason", "") for _, e, f in rig["events"] if e == "link_lost")
    sock.close()


def test_frames_out_of_sequence_are_refused() -> None:
    a, b = socket.socketpair()
    key = b"k" * 32
    sender = L.FrameCodec(key, b"b", b"B")
    receiver = L.FrameCodec(key, b"B", b"b")
    sender.send(a, b"D", 1, b"one")
    sender._send_seq = 5  # a replayed / dropped frame
    sender.send(a, b"D", 1, b"two")
    reader = L.FrameReader(b)
    assert receiver.recv(reader) == (b"D", 1, b"one")
    with pytest.raises(L.LinkError, match="sequence"):
        receiver.recv(reader)
    a.close()
    b.close()


def test_only_the_home_lan(tmp_path: Path) -> None:
    for host in ("127.0.0.1", "10.20.30.40", "10.0.0.3", "172.16.4.4", "fe80::1", "::1"):
        assert L.lan_address(host), host
    for host in ("8.8.8.8", "1.1.1.1", "0.0.0.0", "2001:4860::8888", "duck.example"):
        assert not L.lan_address(host), host
    with pytest.raises(ValueError, match="outside the home LAN"):
        L.parse_url("remote://8.8.8.8:8797")
    with pytest.raises(ValueError, match="token is a file"):
        L.parse_url("remote://127.0.0.1:8797?token=abc")
    with pytest.raises(ValueError, match="LAN only"):
        Bridge(
            listen=("8.8.8.8", 0),
            token=TOKEN,
            channels={"robotd": Channel("robotd", unix="/x")},
            keeper=None,
        )


def test_the_token_file_must_be_long_and_private(tmp_path: Path) -> None:
    path = tmp_path / "token"
    token = L.new_token(path)
    assert L.read_token(path) == token.encode()
    with pytest.raises(FileExistsError):
        L.new_token(path)
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        L.read_token(path)
    short = tmp_path / "short"
    short.write_text("abc\n")
    os.chmod(short, 0o600)
    with pytest.raises(ValueError):
        L.read_token(short)


def test_the_shim_adds_its_delay_both_ways(rig: dict[str, Any]) -> None:
    rig["shim"].impairment = Impairment(delay_ms=40.0, jitter_ms=5.0)
    link = rig["connect"]()
    link.start(timeout_s=3)
    time.sleep(0.8)
    stats = link.stats()
    assert 70.0 <= stats["rtt_ms_p50"] <= 160.0  # 2 × (40 ± 5) ms plus the machine's own time


def test_a_cut_shim_waits_instead_of_spinning() -> None:
    """While cut, data whose time has come is held: the shim's writers sleep on it instead of looping
    at full speed (CI 2026-11-05: spinning writers starved the bridge's deadman and the test runner)."""
    echo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)
    shim = NetShim(echo.getsockname(), seed=1).start()
    client = socket.create_connection(("127.0.0.1", shim.port), timeout=2.0)
    try:
        server_side, _ = echo.accept()
        shim.cut()
        for _ in range(5):  # due at once (no delay), held by the cut
            client.sendall(b"held\n")
            server_side.sendall(b"held\n")
        time.sleep(0.2)
        cpu0, wall0 = time.process_time(), time.monotonic()
        time.sleep(0.8)
        busy = (time.process_time() - cpu0) / (time.monotonic() - wall0)
        assert busy < 0.3  # a spinning writer keeps a whole core busy (≈ 1.0)
        server_side.close()
    finally:
        client.close()
        shim.close()
        echo.close()


def _new_threads(before: set[threading.Thread], name: str) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t not in before and t.name.startswith(name)]


def test_closing_the_bridge_link_and_shim_ends_their_listening_threads(
    rig: dict[str, Any],
) -> None:
    """close() ends the threads blocked in accept() (Linux, the duck's own OS, does not wake an
    accept() when another thread closes the socket; the bridge runs there for weeks)."""
    before = set(threading.enumerate())
    bridge = Bridge(
        listen=("127.0.0.1", 0),
        token=TOKEN,
        channels={"robotd": Channel("robotd", unix=rig["robotd"].path)},
        keeper=None,
        body_info={"backend": "simulator"},
    )
    bridge.start()
    shim = NetShim(("127.0.0.1", bridge.port), seed=3).start()
    link = RemoteLink(f"remote://127.0.0.1:{shim.port}", TOKEN)
    link.start(timeout_s=3)
    listening = (
        _new_threads(before, "bridge-accept")
        + _new_threads(before, "shim-accept")
        + _new_threads(before, "remote-link-robotd")
    )
    assert len(listening) == 3
    link.close()
    shim.close()
    bridge.close()
    assert wait_for(lambda: not any(t.is_alive() for t in listening), 2.0), [
        t.name for t in listening if t.is_alive()
    ]


def test_the_brain_drops_a_silent_link_within_lost_s(rig: dict[str, Any]) -> None:
    link = rig["connect"]()
    link.start(timeout_s=3)
    cut = rig["shim"].cut()
    assert wait_for(lambda: not link.up, 2.0)
    assert time.monotonic() - cut <= L.LOST_S + 0.3
    # a brain client finds robotd 'gone' while the link is down
    with pytest.raises((ConnectionError, OSError, RobotdProtocolError)):
        RobotdBodyPort(link.robotd_socket, backend="simulator", timeout_s=0.5)


def test_a_remote_livebody_reconnects_its_robotd_connection(
    rig: dict[str, Any], tmp_path: Path
) -> None:
    """A local body whose robotd 'goes away' is reported failed for good; a remote body's link
    drops and comes back, so its state thread opens a fresh connection and goes on."""
    from duckmind.livebody import LiveBody, LivePaths, RemoteBodySpec

    link = rig["connect"]()
    link.start(timeout_s=3)
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    body = LiveBody(paths, remote=RemoteBodySpec(str(link.url), tmp_path / "token"))
    body._link = link
    body._port = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
    thread = threading.Thread(target=body._read_state, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: body.policy == "stand")
        rig["shim"].cut()
        assert wait_for(lambda: not link.up, 2.0)
        rig["robotd"].policy = "sit"
        rig["shim"].restore()
        assert wait_for(lambda: body.policy == "sit", 6.0), body.error
        assert body.error is None
        assert thread.is_alive()
    finally:
        body._stop.set()
        thread.join(timeout=2)
        body._port.close()


def test_the_neural_driver_keeps_dialling_a_remote_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local robotd that never comes up is an error after ATTACH_WAIT_S; a remote body's link
    can stay down longer (a Wi-Fi drop), and the brain must take the body back when it returns."""
    from types import SimpleNamespace

    from duckmind.simweb import neuro_live

    monkeypatch.setattr(neuro_live, "ATTACH_WAIT_S", 0.2)
    monkeypatch.setattr(neuro_live, "RECONNECT_S", 0.05)
    with tempfile.TemporaryDirectory(prefix="dm-rb-d-", dir=short_tmp()) as directory:
        path = os.path.join(directory, "robotd.sock")
        robotds: list[FakeRobotd] = []
        threading.Timer(0.6, lambda: robotds.append(FakeRobotd(path))).start()
        body = SimpleNamespace(remote=object(), robotd_socket=Path(path))
        driver = SimpleNamespace(stats=SimpleNamespace(last_error=None))
        client = neuro_live.NeuroDriver._connect(driver, body, threading.Event())
        try:
            assert client is not None
        finally:
            if client is not None:
                neuro_live._close_client(client)
            for robotd in robotds:
                robotd.close()


def test_a_daemon_slow_to_accept_never_stalls_the_heartbeat(rig: dict[str, Any]) -> None:
    """Opening a stream connects to the daemon's socket; a daemon slow to accept (robotd stalled
    with its physics) must not hold up the link's reader and trip the deadman."""
    robotd = rig["robotd"]

    class SlowChannel(Channel):
        def connect(self, timeout_s: float = 1.0) -> socket.socket:
            time.sleep(L.DEADMAN_S + 0.5)
            return super().connect(timeout_s)

    events: list[str] = []
    bridge = Bridge(
        listen=("127.0.0.1", 0),
        token=TOKEN,
        channels={"robotd": SlowChannel("robotd", unix=robotd.path)},
        keeper=None,
        log=lambda event, **fields: events.append(event),
    )
    bridge.start()
    link = RemoteLink(f"remote://127.0.0.1:{bridge.port}", TOKEN)
    try:
        link.start(timeout_s=3)
        port = RobotdBodyPort(link.robotd_socket, backend="simulator", timeout_s=4.0)
        assert port.observe().policy == "stand"  # (answered once the slow connect finished)
        port.close()
        assert "link_lost" not in events and link.drops == 0
    finally:
        link.close()
        bridge.close()


def test_a_simulated_remote_body_runs_the_neural_ducks_world(tmp_path: Path) -> None:
    """The neural duck's page starts its own body with the world's neural side (the truth behind its
    camera, hearing and touch models). A sim-body for a remote neural brain must start the same world,
    else the brain runs with no camera, sound or touch (seen: the page's remote walk waited ~4.5 s
    for a head that never settled)."""
    from duckmind.remotebody.simbody import neural_duck_body
    from duckmind.simweb.neuro_live import world_extensions

    env: dict[str, str] = {"DUCKMIND_ROBOTD_SKILLS": "none"}
    info = neural_duck_body(env)
    assert "duckmind.world.neuro_integration" in world_extensions()
    assert env["DUCKMIND_SIMHOME_EXT"] == world_extensions() == info["world_extensions"]
    assert env["DUCKMIND_BODY_COMMAND_DELAY_MS"] == "20"
    mine = {"DUCKMIND_SIMHOME_EXT": "duckmind.world.worldkit", "DUCKMIND_ROBOTD_SKILLS": "none"}
    assert neural_duck_body(mine)["world_extensions"] == "duckmind.world.worldkit"  # (given wins)


# ── the battery: robotd's robot.health over the link ─────────────────────────────────────────


def test_a_remote_bodys_battery_is_its_robotds_robot_health(
    rig: dict[str, Any], tmp_path: Path
) -> None:
    """On a remote body the brain's battery is what the duck's robotd says (robot.health over the
    link), never the simulated world's pack: volts and percent, and no word about charging (robotd
    has no charging flag). After a Wi-Fi drop it reads again."""
    from duckmind.livebody import LiveBody, LivePaths, RemoteBodySpec
    from duckmind.simweb.neuro_live import _battery, with_body_battery

    rig["robotd"].battery = {"volts": 7.62, "percent": 64.0}
    link = rig["connect"]()
    link.start(timeout_s=3)
    paths = LivePaths(*(tmp_path / n for n in ("m", "r", "p", "robotd", "ort")))
    body = LiveBody(paths, remote=RemoteBodySpec(str(link.url), tmp_path / "token"))
    body._link = link
    try:
        assert wait_for(lambda: body.battery() is not None)
        reading = body.battery()
        assert reading is not None
        assert (reading["percent"], reading["volts"], reading["charging"]) == (64.0, 7.62, None)
        assert reading["source"] == "robotd"
        world = {"battery": {"percent": 12, "soc_ppm": 120_000, "charging": True}, "hand": {}}
        seen = with_body_battery(body, world)
        assert seen is not None and seen["hand"] == {}
        assert _battery(seen) == (0.64, False)  # (the duck's 64 %, not the world's 12 %)
        rig["shim"].cut()
        assert wait_for(lambda: not link.up, 2.0)
        rig["robotd"].battery = {"volts": 7.0, "percent": 25.0}
        rig["shim"].restore()
        assert wait_for(lambda: (body.battery() or {}).get("percent") == 25.0, 8.0)
    finally:
        body.stop()


def test_a_remote_body_without_a_reading_has_no_battery_and_a_local_one_keeps_the_worlds() -> None:
    from types import SimpleNamespace

    from duckmind.simweb.neuro_live import _battery, with_body_battery

    world = {"battery": {"percent": 40, "charging": True}, "people": []}
    assert with_body_battery(SimpleNamespace(remote=None), world) is world
    unread = with_body_battery(SimpleNamespace(remote=object(), battery=lambda: None), world)
    assert unread == {"people": []} and _battery(unread) == (None, None)
    reading = {"percent": 30.0, "volts": 7.08, "charging": None, "source": "robotd"}
    real = with_body_battery(SimpleNamespace(remote=object(), battery=lambda: reading), None)
    assert real == {
        "battery": {"percent": 30.0, "volts": 7.08, "charging": False, "source": "robotd"}
    }


def test_robotd_without_a_battery_reading_is_unknown_not_empty() -> None:
    from duckmind.remotebody.battery import parse

    assert parse({"healthy": True}) is None
    assert parse({"healthy": True, "battery": {"volts": 7.4}}) is None
    assert parse({"healthy": True, "battery": {"volts": 8.4, "percent": 100.0}}) == {
        "percent": 100.0,
        "volts": 8.4,
        "charging": None,
        "source": "robotd",
    }


def test_the_simulated_bodys_robotd_reports_the_worlds_pack() -> None:
    """robotd --sim says the nominal 7.4 V (the world never feeds it its pack, owner E1), so the
    sim-body's bridge puts the world's pack into robot.health — what a fed robotd would say (the
    loaded volts and its percentage of them) and whether it charges — and passes everything else
    (robot.state) through untouched."""
    from types import SimpleNamespace

    from duckmind.remotebody.battery import parse, world_health_battery
    from duckmind.remotebody.simbody import channels_for
    from duckmind.robotd import RobotdSafetyClient

    world: dict[str, Any] = {
        "battery": {"percent": 61, "volts_mv": 7310, "robotd_percent_if_fed": 44, "charging": True}
    }
    with tempfile.TemporaryDirectory(prefix="dm-rb-b-", dir=short_tmp()) as directory:
        robotd = FakeRobotd(os.path.join(directory, "r.sock"))
        robotd.battery = {"volts": 7.4, "percent": 50.0}
        sim = SimpleNamespace(
            robotd_socket=robotd.path, body_port=1, home=None, world=lambda: world
        )
        channel = channels_for(sim)["robotd"]
        assert channel.answers is not None
        bridge = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={
                "robotd": Channel(
                    "robotd", unix=robotd.path, answers=world_health_battery(lambda: world)
                )
            },
            keeper=None,
        )
        bridge.start()
        link = RemoteLink(f"remote://127.0.0.1:{bridge.port}", TOKEN)
        try:
            link.start(timeout_s=3)
            with RobotdSafetyClient(link.robotd_socket, timeout_s=2.0) as client:
                health = client.health()
                assert health["battery"] == {"volts": 7.31, "percent": 44, "charging": True}
                assert (parse(health) or {}).get("charging") is True
                world.clear()  # (no pack in the world: robotd's own answer)
                assert client.health()["battery"] == {"volts": 7.4, "percent": 50.0}
            port = RobotdBodyPort(link.robotd_socket, backend="simulator", hz=25)
            try:
                assert port.observe().policy == "stand"
            finally:
                port.close()
        finally:
            link.close()
            bridge.close()
            robotd.close()
