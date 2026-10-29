"""A transient stall of the body host must not leave the live duck blind (2026-10-19: the sampler's
one read timed out, the thread ended, the error stayed for 20 h): the oracle's reads survive a
timeout and never take a late answer for the next one, the three body links retry, and only a link
silent for LINK_FAIL_S is reported (and the report is taken back when it speaks again). A body that
is really gone is still reported at once."""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind import livebody
from duckmind.livebody import LinkWatch, LiveBody, LivePaths
from duckmind.robotd import RobotdProtocolError, _RpcConnection
from duckmind.sim_live import SimOracle


def _body(tmp_path: Any) -> LiveBody:
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    return LiveBody(paths)  # never started: no processes; the threads are run by the test


def _until(check: Any, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


def _run(body: LiveBody, target: Any) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _stop(body: LiveBody, thread: threading.Thread) -> None:
    body._stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive()


def _snapshot(t: float) -> dict[str, Any]:
    return {
        "positions": [0.0] * 15,
        "sim_time": t,
        "trunk_z": 0.12,
        "trunk": [0.0, 0.0, 0.12],
        "imu": {"quat": [1.0, 0.0, 0.0, 0.0]},
    }


class FakeOracle:
    """SimOracle's reads: silent (a timeout) while ``silent`` is set or for ``misses`` reads; a
    failed read drops the connection, so ``lost`` until ``reconnect``."""

    def __init__(self, *, misses: int = 0, error: Exception | None = None) -> None:
        self.misses = misses
        self.error = error
        self.silent = threading.Event()
        self.lost = False
        self.reconnects = 0
        self.reads = 0
        self.t = 0.0

    def reconnect(self) -> None:
        self.reconnects += 1
        if self.silent.is_set() and self.error is not None:
            raise self.error
        self.lost = False

    def snapshot(self) -> dict[str, Any]:
        assert not self.lost, "a read on a dropped connection"
        self.reads += 1
        if self.silent.is_set() or self.misses > 0:
            self.misses = max(0, self.misses - 1)
            self.lost = True
            raise self.error or TimeoutError("timed out")
        self.t += 0.01
        return _snapshot(self.t)


# ── the oracle itself ───────────────────────────────────────────────────────────


def test_a_late_answer_is_never_read_as_the_answer_to_the_next_request() -> None:
    """The body server answers one read too late (past the timeout); that answer must not be taken
    for the next read's: the failed read drops the connection and the next one opens a new one."""
    slow = {"left": 1}
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)

    def serve(conn: socket.socket) -> None:
        with conn:
            stream = conn.makefile("rwb")
            while True:
                line = stream.readline()
                if not line:
                    return
                op = json.loads(line)["op"]
                if op == "hello":
                    answer: dict[str, Any] = {"protocol": 1}
                elif slow["left"] > 0:
                    slow["left"] -= 1
                    time.sleep(0.6)  # past the oracle's timeout
                    answer = _snapshot(111.0)  # (the late answer)
                else:
                    answer = _snapshot(2.0)
                try:
                    stream.write((json.dumps(answer) + "\n").encode())
                    stream.flush()
                except OSError:
                    return

    def accept() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=serve, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    oracle = SimOracle(listener.getsockname()[1], timeout_s=0.25)
    try:
        with pytest.raises(TimeoutError):
            oracle.snapshot()
        assert oracle.lost
        with pytest.raises(ConnectionError):
            oracle.snapshot()  # (nothing is read from a connection that failed)
        time.sleep(0.5)  # the late answer is on its way to the old connection now
        oracle.reconnect()
        assert not oracle.lost
        assert oracle.snapshot()["sim_time"] == 2.0  # this read's own answer, not the late one
        assert oracle.snapshot()["sim_time"] == 2.0
    finally:
        oracle.close()
        listener.close()
    with pytest.raises(ConnectionError):
        oracle.reconnect()  # (a closed oracle is final: the sampler cannot revive it after stop)


# ── how long a link may be silent ───────────────────────────────────────────────


def test_a_link_is_only_reported_after_it_has_been_silent_for_the_failure_time() -> None:
    now = {"t": 100.0}
    link = LinkWatch("oracle", "body oracle", clock=lambda: now["t"], retry_s=(0.2, 0.5, 1.0))
    assert link.heard() is None  # (it never missed: nothing to take back)
    waits = []
    for _ in range(5):
        now["t"] += 4.0
        wait, report = link.missed()
        waits.append((wait, report))
    assert [w for w, _ in waits] == [0.2, 0.5, 1.0, 1.0, 1.0]  # the last wait repeats
    assert [r for _, r in waits] == [False] * 5  # (20 s: under LINK_FAIL_S)
    now["t"] += 12.0
    wait, report = link.missed()
    assert report  # 32 s since it last spoke
    assert not link.missed()[1]  # and said once
    tries, silent_s, reported = link.heard() or (0, 0.0, False)
    assert (tries, round(silent_s), reported) == (7, 32, True)
    assert link.missed()[1] is False  # a new silence counts from here


# ── the sampler ─────────────────────────────────────────────────────────────────


def test_the_sampler_survives_a_timeout_and_the_body_is_never_reported_failed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-19 incident: one read timed out and the body was dead for the rest of the day."""
    monkeypatch.setattr(livebody, "LINK_RETRY_S", (0.02,))
    body = _body(tmp_path)
    body._oracle = FakeOracle(misses=1)  # type: ignore[assignment]
    thread = _run(body, body._sample)
    try:
        assert _until(lambda: body.latest is not None), "the sampler never got a frame back"
        assert body.error is None
        assert body._oracle.reconnects == 1  # type: ignore[attr-defined]
        frames = len(body.frames_since(None))
        assert _until(lambda: len(body.frames_since(None)) > frames)  # and it goes on sampling
        assert body.status()["running"] is True
    finally:
        _stop(body, thread)


def test_a_sustained_silence_is_reported_and_taken_back_when_samples_flow_again(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(livebody, "LINK_RETRY_S", (0.02,))
    monkeypatch.setattr(livebody, "LINK_FAIL_S", 0.3)
    body = _body(tmp_path)
    oracle = FakeOracle()
    oracle.silent.set()
    body._oracle = oracle  # type: ignore[assignment]
    thread = _run(body, body._sample)
    try:
        assert _until(lambda: body.error is not None), "a sustained silence was never reported"
        assert body.error and body.error.startswith("body oracle:")
        assert body.status()["running"] is False
        oracle.silent.clear()  # the host answers again
        assert _until(lambda: body.error is None), "the report was not taken back"
        assert _until(lambda: body.latest is not None)
        assert body.status()["running"] is True
    finally:
        _stop(body, thread)


def test_a_body_that_is_really_gone_is_reported_at_once(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _body(tmp_path)
    body._body = SimpleNamespace(poll=lambda: 1)  # type: ignore[assignment]  # (the process exited)
    oracle = FakeOracle(error=ConnectionRefusedError("refused"))
    oracle.silent.set()
    body._oracle = oracle  # type: ignore[assignment]
    thread = _run(body, body._sample)
    assert _until(lambda: body.error is not None, 2.0)  # not after LINK_FAIL_S (30 s)
    assert body.error and body.error.startswith("body oracle:")
    thread.join(timeout=2)
    assert not thread.is_alive()  # (the sampler ended: nothing is left to come back)


def test_a_host_that_answers_with_an_error_is_not_a_stall(tmp_path: Any) -> None:
    body = _body(tmp_path)
    oracle = FakeOracle(error=RuntimeError("MuJoCo body oracle error: {'error': 'bad'}"))
    oracle.silent.set()
    body._oracle = oracle  # type: ignore[assignment]
    thread = _run(body, body._sample)
    assert _until(lambda: body.error is not None, 2.0)
    thread.join(timeout=2)
    assert not thread.is_alive()


# ── robotd's two links ──────────────────────────────────────────────────────────


class Observed:
    policy = "walk"
    safety = SimpleNamespace(fallen=False)


class FakePort:
    """RobotdBodyPort.observe: silent (TimeoutError) while ``silent`` is set or ``misses`` times."""

    def __init__(self, misses: int = 0) -> None:
        self.misses = misses
        self.silent = threading.Event()

    def observe(self) -> Observed:
        if self.silent.is_set() or self.misses > 0:
            self.misses = max(0, self.misses - 1)
            time.sleep(0.01)
            raise TimeoutError("robotd response/state timeout")
        time.sleep(0.01)
        return Observed()


def test_robotds_state_stream_survives_a_stall_and_reports_only_a_sustained_one(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(livebody, "LINK_RETRY_S", (0.02,))
    monkeypatch.setattr(livebody, "LINK_FAIL_S", 0.3)
    body = _body(tmp_path)
    port = FakePort(misses=3)
    body._port = port  # type: ignore[assignment]
    thread = _run(body, body._read_state)
    try:
        assert _until(lambda: body.policy == "walk"), "the state reader ended on a timeout"
        assert body.error is None
        port.silent.set()
        assert _until(lambda: body.error is not None)
        assert body.error and body.error.startswith("robotd state:")
        port.silent.clear()
        assert _until(lambda: body.error is None)
    finally:
        _stop(body, thread)


def test_robotd_going_away_is_still_reported_at_once(tmp_path: Any) -> None:
    body = _body(tmp_path)

    class Gone:
        def observe(self) -> Observed:
            raise ConnectionError("robotd closed the socket")

    body._port = Gone()  # type: ignore[assignment]
    thread = _run(body, body._read_state)
    assert _until(lambda: body.error is not None, 2.0)
    assert body.error == "robotd state: robotd closed the socket"
    thread.join(timeout=2)
    assert not thread.is_alive()


class Wire:
    """_JsonLineSocket's two calls: the frames the daemon has sent, then silence."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = list(frames)
        self.sent: list[dict[str, Any]] = []

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def read(self, timeout_s: float) -> dict[str, Any]:
        if not self.frames:
            raise TimeoutError("robotd response/state timeout")
        return self.frames.pop(0)


def _rpc(frames: list[dict[str, Any]], next_id: int) -> _RpcConnection:
    rpc = object.__new__(_RpcConnection)
    rpc.wire = Wire(frames)  # type: ignore[assignment]
    rpc.timeout_s = 1.0
    rpc._next_id = next_id
    rpc._lock = threading.Lock()
    return rpc


def test_robotd_commands_skip_the_answer_to_a_call_that_timed_out() -> None:
    """A call that timed out leaves its answer on the way; read as the next call's answer it would
    leave every later command one answer behind for good."""
    late = {"jsonrpc": "2.0", "id": 3, "result": {"accepted": False}}
    unparsed = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700}}
    mine = {"jsonrpc": "2.0", "id": 5, "result": {"accepted": True}}
    assert _rpc([late, unparsed, mine], 5).call("robot.do") == {"accepted": True}


def test_robotd_commands_still_refuse_an_answer_that_is_not_theirs_or_missing() -> None:
    ahead = {"jsonrpc": "2.0", "id": 9, "result": {}}
    with pytest.raises(RobotdProtocolError):
        _rpc([ahead], 5).call("robot.do")
    late = {"jsonrpc": "2.0", "id": 3, "result": {}}
    with pytest.raises(TimeoutError):
        _rpc([late], 5).call("robot.do")  # (only stale answers: the call itself timed out)
