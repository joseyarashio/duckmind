"""``python -m duckmind.remotebody check`` against a bridge with a fake robotd and a fake mediad,
through the network shim: the policy hashes and whether the brain was measured on them, the
battery the brain gets, the clock offset, the camera's frame rate over the link, mediad's LAN doors
and the 10-minute Wi-Fi quality mode (shortened here) with its false deadman alarms."""

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

from duckmind.remotebody import __main__ as cli
from duckmind.remotebody.bridge import Bridge, Channel
from duckmind.remotebody.netshim import NetShim
from duckmind.remotebody.quality import LinkQuality
from duckmind.tmpdir import short_tmp
from tests.test_remotebody import TOKEN, FakeRobotd

W, H = 640, 360  # a UYVY frame of 460 kB (many link frames)


class FakeMedia:
    """mediad's frame socket: ``hello``, then one ``media.frame`` per connection (a header line,
    then exactly its bytes), and mediad closes the connection right after the frame."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.frames = 0
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen(16)
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._server.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            buffer = b""
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    request = json.loads(line)
                    if request["method"] == "hello":
                        result: dict[str, Any] = {"api_version": 37}
                        tail = b""
                    else:
                        tail = bytes(W * H * 2)
                        result = {"width": W, "height": H, "format": "UYVY", "bytes": len(tail),
                                  "captured_at_unix_us": time.time_ns() // 1000, "rotate": 0}  # fmt: skip
                        self.frames += 1
                    answer = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                    conn.sendall(json.dumps(answer).encode() + b"\n" + tail)
                    if tail:
                        return  # (one frame per connection: closed at once, as mediad does)


@pytest.fixture
def duck(tmp_path: Path):
    """A sim-body-like bridge (robotd with a battery, mediad's frames, the policy hashes a
    sim-body reports) behind a shim, a listening 'mediad door' and the token file."""
    from duckmind.neuro import seatcal, walkcal

    walk = next(iter(walkcal.load().walks))
    seat = next(iter(seatcal.load().seats))
    with tempfile.TemporaryDirectory(prefix="dm-rc-", dir=short_tmp()) as directory:
        robotd = FakeRobotd(os.path.join(directory, "r.sock"))
        robotd.policy = "sit"
        robotd.battery = {"volts": 7.62, "percent": 64.0}
        media = FakeMedia(os.path.join(directory, "m.sock"))
        bridge = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={
                "robotd": Channel("robotd", unix=robotd.path),
                "media": Channel("media", unix=media.path),
            },
            keeper=None,
            body_info={
                "backend": "simulator",
                "policies": {"walk_sha256": walk, "sitstand_sha256": seat},
            },
        )
        bridge.start()
        shim = NetShim(("127.0.0.1", bridge.port), seed=3).start()
        door = socket.socket()
        door.bind(("127.0.0.1", 0))
        door.listen(8)
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))  # (bound, never listening: a closed door)
        token = tmp_path / "token"
        token.write_bytes(TOKEN)
        os.chmod(token, 0o600)
        argv = [
            f"remote://127.0.0.1:{shim.port}",
            "--token-file",
            str(token),
            "--seconds",
            "0.5",
            "--camera-seconds",
            "1",
            "--mediad-ports",
            f"{door.getsockname()[1]},{closed.getsockname()[1]}",
        ]
        yield {
            "argv": argv,
            "shim": shim,
            "walk": walk,
            "seat": seat,
            "media": media,
            "door": door.getsockname()[1],
            "closed": closed.getsockname()[1],
            "bridge": bridge,
        }
        door.close()
        closed.close()
        shim.close()
        bridge.close()
        media.close()
        robotd.close()


def test_check_says_what_the_brain_needs_to_know_before_the_first_walk(
    duck: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.check(duck["argv"]) == 1  # (a door answers: the link is not called ready)
    out = capsys.readouterr().out
    assert f"sha256 {duck['walk']} (measured in the simulator)" in out
    assert "sit-stand: " in out and f"sha256 {duck['seat']} (measured in the simulator)" in out
    assert "battery (what the brain gets): 64 % (7.62 V" in out
    assert "charging: not reported by robotd" in out
    assert "clock: the duck's wall clock is " in out and " ms from this computer's (± " in out
    (camera,) = [line for line in out.splitlines() if line.startswith("camera over the link: ")]
    assert f"{W}×{H} UYVY" in camera and duck["media"].frames > 0
    # (mediad closes each frame's connection at once: the link must still deliver all of it)
    assert camera.endswith(", 0 failed)"), camera
    assert float(camera.split(": ")[1].split(" frames/s")[0]) > 0
    (door,) = [line for line in out.splitlines() if line.startswith("mediad's LAN doors")]
    assert f"OPEN from this computer: 127.0.0.1 port {duck['door']} (" in door
    assert str(duck["closed"]) not in door
    assert "docs/deploy/board.md, 'Closing mediad's LAN doors'" in out
    assert "sudo sh tools/duck/close-mediad.sh --brain" in out
    assert out.splitlines()[-1] == "NOT READY: mediad's LAN doors answer from this computer"
    assert "Wi-Fi quality" not in out


def closed_doors(duck: dict[str, Any]) -> list[str]:
    return [*duck["argv"], "--camera-seconds", "0", "--mediad-ports", str(duck["closed"])]


def test_check_calls_the_link_ready_only_when_no_door_answers(
    duck: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.check(closed_doors(duck)) == 0
    out = capsys.readouterr().out
    assert "mediad's LAN doors: closed from this computer (knocked on 127.0.0.1)" in out
    assert out.splitlines()[-1].startswith("ready: the link held")


def test_check_is_not_ready_while_the_bridge_found_netpadd_listening(
    duck: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """A UDP door cannot be knocked on from the computer: the bridge's own look on the duck
    (started with --allow-open-mediad here) is what check has."""
    duck["bridge"].body_info["doors"] = {
        "host": "10.0.0.50",
        "tcp_open": [],
        "netpadd": 4210,
        "allowed": True,
    }
    assert cli.check(closed_doors(duck)) == 1
    out = capsys.readouterr().out
    assert "something listened on UDP 4210 on the duck" in out
    assert "started with --allow-open-mediad while a door was open" in out
    assert out.splitlines()[-1].startswith("NOT READY: netpadd listened on the duck")


def test_the_wifi_mode_counts_a_drop_as_a_false_deadman_alarm(
    duck: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The brain kept pinging; the Wi-Fi held its packets 1.6 s; the bridge's 1.0 s deadman
    dropped the link (on a duck: stop and sit for nothing). That is one false alarm."""
    timer = threading.Timer(3.0, duck["shim"].cut)
    restore = threading.Timer(4.6, duck["shim"].restore)
    timer.start()
    restore.start()
    try:
        argv = [*duck["argv"], "--camera-seconds", "0", "--mediad-ports", "", "--wifi-minutes"]
        assert cli.check([*argv, str(9.0 / 60)]) == 1
    finally:
        timer.cancel()
        restore.cancel()
        duck["shim"].restore()
    out = capsys.readouterr().out
    assert "Wi-Fi quality: holding the link for 0.1 min" in out
    (summary,) = [line for line in out.splitlines() if line.startswith("  false deadman alarms")]
    assert summary.startswith("  false deadman alarms: 1 (")
    assert "at the 1000 ms deadman" in summary
    (total,) = [line for line in out.splitlines() if line.startswith("Wi-Fi quality over ")]
    assert "drops 1" in total


def test_the_longest_silence_at_the_duck_is_read_off_the_ducks_clock() -> None:
    q = LinkQuality()
    ms = 1_000_000
    for body_ms in (0, 100, 200, 800, 900):
        q.pong(20 * ms, body_ms * ms)
    q.new_link()  # (a reconnect: the silence across it is the drop, not a gap)
    q.pong(30 * ms, 5000 * ms, wall_offset_ns=-12 * ms)
    s = q.summary(seconds=360.0, drops=1, deadman_ms=1000)
    assert (s["duck_silence_ms_max"], s["near_misses"], s["pings"]) == (600, 1, 6)
    assert (s["false_alarms_per_hour"], s["near_misses_per_hour"]) == (10.0, 10.0)
    assert (s["rtt_ms_p50"], s["rtt_ms_max"]) == (20.0, 30.0)
    assert (q.wall_offset_ms, q.wall_offset_err_ms) == (-12.0, 15.0)


def test_a_brains_link_keeps_a_bounded_number_of_pongs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The brain's link lives for days at 10 pings a second: it keeps the last KEEP, and a window
    still sees its own pongs."""
    from duckmind.remotebody import quality

    monkeypatch.setattr(quality, "KEEP", 5)
    q = quality.LinkQuality()
    for i in range(12):
        q.pong(1_000_000 * (i + 1), i * 100_000_000)
    assert len(q.rtt_ms) == 5 and len(q.body_gaps_ms) == 5
    since = q.mark()
    q.pong(50_000_000, 2_000_000_000)
    window = q.summary(seconds=60.0, drops=0, deadman_ms=1000, since=since)
    assert (window["pings"], window["rtt_ms_max"], window["duck_silence_ms_max"]) == (1, 50.0, 900)
    assert q.summary(seconds=60.0, drops=0, deadman_ms=1000)["pings"] == 5
