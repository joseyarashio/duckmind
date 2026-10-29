"""duck-body's loop and process: acts with act_id/deadline, B4, falls, and body.sock end to end."""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import threading
import time
from typing import Any

import pytest

from duckmind import percept
from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.protocol import ActOutput
from duckmind.duckbody.robotd_client import RobotState


def state(
    t: float, *, policy: str = "walk", load: float = 18.0, tilt: float = 2.0, fallen: bool = False
) -> RobotState:
    r = math.radians(tilt)
    currents = [0.0] * 15
    currents[0] = currents[10] = load
    joints = [0.0] * 15
    joints[5], joints[6] = 0.3491, 0.3491  # the head at HOME
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": fallen, "limp": False, "gravity": [math.sin(r), 0, -math.cos(r)]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": currents,
            "imu": {"gyro": [0.01, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def ns(t: float) -> int:
    return 10**10 + int(round(t * 1e9))


def lines_of(ticks: list, kind: str) -> list[dict]:
    return [line for tick in ticks for line in tick.lines if line.get("type") == kind]


def test_posture_act_toggles_retries_and_reports() -> None:
    loop = BodyLoop()
    acks = loop.command(
        {
            "type": "cmd",
            "cmd": "duck.posture",
            "act_id": "P1",
            "want": "sit",
            "deadline_ms": ns(20) // 10**6,
        },
        ns(0),
    )
    assert acks[0]["ok"] is True
    ticks = []
    for k in range(400):
        t = k * 0.02
        policy = "sit" if t >= 3.5 else "walk"  # the first toggle is ignored, the retry works
        ticks.append(loop.tick(state(t, policy=policy, load=12.5 if t >= 3.5 else 18.0)))
    dos = [
        (round(i * 0.02, 2), x.params["skill"])
        for i, tk in enumerate(ticks)
        for x in tk.intents
        if x.method == "robot.do"
    ]
    assert dos[:2] == [(0.0, "sit_toggle"), (3.0, "sit_toggle")]
    posture = [
        ln["event"] for ln in lines_of(ticks, "percept") if ln["event"]["kind"] == "body.posture"
    ]
    assert posture and posture[0]["got"] == "sit" and posture[0]["act_id"] == "P1"
    assert percept.validate(posture[0]) == []
    done = lines_of(ticks, "done")
    assert done[0]["act_id"] == "P1" and done[0]["kind"] == "body.posture"


def test_command_refusals() -> None:
    loop = BodyLoop()
    loop.tick(state(0.0))
    past = loop.command(
        {"type": "cmd", "cmd": "duck.posture", "act_id": "P0", "want": "sit", "deadline_ms": 1},
        ns(1),
    )
    assert past[0]["reason"] == "past_deadline"
    walk = loop.command(
        {
            "type": "cmd",
            "cmd": "duck.walk",
            "act_id": "W1",
            "vx_mm": 400,
            "deadline_ms": ns(60) // 10**6,
        },
        ns(1),
    )
    assert walk[0]["reason"] == "unavailable"  # no walk executor configured
    bad = loop.command({"type": "cmd", "cmd": "duck.walk", "act_id": "W2"}, ns(1))
    assert bad[0]["ok"] is False and bad[0]["reason"] == "invalid"
    loop.command(
        {
            "type": "cmd",
            "cmd": "duck.posture",
            "act_id": "P1",
            "want": "sit",
            "deadline_ms": ns(60) // 10**6,
        },
        ns(1),
    )
    busy = loop.command(
        {
            "type": "cmd",
            "cmd": "duck.posture",
            "act_id": "P2",
            "want": "stand",
            "deadline_ms": ns(60) // 10**6,
        },
        ns(1),
    )
    assert busy[0]["reason"] == "busy"


class Walker:
    """A stand-in walk executor: walks until preempted, then reports."""

    def __init__(self, cmd: Any, t_ms: int) -> None:
        self.command, self.act_id, self.ended = cmd, cmd.act_id, None

    def step(self, ctx: Any) -> ActOutput:
        return ActOutput(twist=(400, 0))

    def preempt(self, why: str, ctx: Any) -> ActOutput:
        body = {
            "kind": "walk.done",
            "act_id": self.act_id,
            "cmd": {"vx": 400, "vyaw": 0, "max_ms": 3000},
            "odom": {"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 0},
            "steps": 0,
            "reason": why,
        }
        return ActOutput(percepts=(body,), done=True)


def walking_loop() -> BodyLoop:
    return BodyLoop(BodyConfig(executors={"duck.walk": lambda c, t, loop: Walker(c, t)}))


def start_walk(loop: BodyLoop, deadline_s: float = 60.0) -> None:
    ack = loop.command(
        {
            "type": "cmd",
            "cmd": "duck.walk",
            "act_id": "W1",
            "vx_mm": 400,
            "deadline_ms": ns(deadline_s) // 10**6,
        },
        ns(0.5),
    )
    assert ack[0]["ok"] is True


def test_b4_the_duck_sits_when_it_feels_the_lift() -> None:
    loop = walking_loop()
    ticks = [loop.tick(state(k * 0.02)) for k in range(50)]  # 1 s standing: head settled
    start_walk(loop)
    ticks += [loop.tick(state(1.0 + k * 0.02, load=20.0)) for k in range(25)]
    moves = [x for tk in ticks for x in tk.intents if x.method == "robot.move"]
    assert moves and moves[0].params["vx"] == pytest.approx(0.4)
    # lifted: the legs carry almost nothing (the walk stops first, then the claim)
    ticks = [loop.tick(state(1.5 + k * 0.02, load=4.5)) for k in range(60)]
    events = [ln["event"] for ln in lines_of(ticks, "percept")]
    kinds = [e["kind"] for e in events]
    assert "walk.done" in kinds and "sense.lifted" in kinds
    done = next(e for e in events if e["kind"] == "walk.done")
    assert done["reason"] in ("reflex_unloaded", "lifted")
    dos = [x.params["skill"] for tk in ticks for x in tk.intents if x.method == "robot.do"]
    assert dos == ["sit_toggle"]
    last_move = [x for tk in ticks for x in tk.intents if x.method == "robot.move"][-1]
    assert last_move.params["vx"] == 0.0
    refused = loop.command(
        {
            "type": "cmd",
            "cmd": "duck.walk",
            "act_id": "W2",
            "vx_mm": 400,
            "deadline_ms": ns(99) // 10**6,
        },
        ns(3.0),
    )
    assert refused[0]["reason"] == "lifted"


def test_a_fall_preempts_the_walk_and_starts_righting() -> None:
    loop = walking_loop()
    for k in range(50):
        loop.tick(state(k * 0.02, load=20.0))
    start_walk(loop)
    ticks = [loop.tick(state(1.0 + k * 0.02, load=20.0, tilt=70.0, fallen=True)) for k in range(20)]
    events = [ln["event"] for ln in lines_of(ticks, "percept")]
    assert [e["kind"] for e in events][:2] == ["body.fell", "walk.done"]
    assert events[1]["reason"] == "fell" and events[0]["during"] == "walk"
    assert (
        loop.command(
            {
                "type": "cmd",
                "cmd": "duck.posture",
                "act_id": "P",
                "want": "sit",
                "deadline_ms": ns(99) // 10**6,
            },
            ns(2),
        )[0]["reason"]
        == "fallen"
    )


def test_stop_and_deadline_end_acts() -> None:
    loop = walking_loop()
    for k in range(50):
        loop.tick(state(k * 0.02, load=20.0))
    start_walk(loop, deadline_s=1.5)
    ticks = [loop.tick(state(1.0 + k * 0.02, load=20.0)) for k in range(40)]
    done = [ln["event"] for ln in lines_of(ticks, "percept") if ln["event"]["kind"] == "walk.done"]
    assert done[0]["reason"] == "deadline"
    start_walk(loop)
    loop.command({"type": "cmd", "cmd": "duck.stop", "why": "petted"}, ns(2.0))
    ticks = [loop.tick(state(2.0 + k * 0.02, load=20.0)) for k in range(5)]
    done = [ln["event"] for ln in lines_of(ticks, "percept") if ln["event"]["kind"] == "walk.done"]
    assert done[0]["reason"] == "preempt"


# ── the process, end to end over unix sockets ─────────────────────────────────


class FakeRobotd:
    """Answers hello/subscribe/model/health/stop/do and streams robot.state at 50 Hz."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(8)
        self.received: list[dict] = []
        self.stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        self.sock.settimeout(0.2)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buffer = b""
        subscribed = False
        t = 0.0
        conn.settimeout(0.02)
        while not self.stop.is_set():
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
                msg = json.loads(line)
                self.received.append(msg)
                if "id" in msg:
                    result: dict = {"accepted": True}
                    if msg["method"] == "robot.health":
                        result = {"healthy": True, "battery": {"percent": 71.0}, "cpu_temp_c": 55.0}
                    if msg["method"] == "robot.subscribe":
                        subscribed = True
                    conn.sendall(
                        json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode()
                        + b"\n"
                    )
            if subscribed:
                t += 0.02
                frame = {
                    "jsonrpc": "2.0",
                    "method": "robot.state",
                    "params": {
                        "t_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                        "t": t,
                        "policy": "walk",
                        "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
                        "safety": {"fallen": False, "limp": False, "gravity": [0, 0, -1]},
                        "joints": [0.0] * 5 + [0.3491, 0.3491, 0.0, 0.0] + [0.0] * 6,
                        "velocities": [0.0] * 15,
                        "currents_ma": [3.6] * 15,
                        "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]},
                    },
                }
                try:
                    conn.sendall(json.dumps(frame).encode() + b"\n")
                except OSError:
                    return

    def close(self) -> None:
        self.stop.set()
        self.sock.close()


def test_the_process_serves_body_sock() -> None:
    from duckmind.duckbody.server import BodyServer, Paths

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        robotd_path, body_path = os.path.join(tmp, "r.sock"), os.path.join(tmp, "b.sock")
        robotd = FakeRobotd(robotd_path)
        server = BodyServer(Paths(robotd_path, body_path), BodyConfig(src="live"))
        server.start()
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            for _ in range(200):
                try:
                    client.connect(body_path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.02)
            client.settimeout(3.0)
            reader = client.makefile("r")
            hello = json.loads(reader.readline())
            assert hello["type"] == "hello" and hello["api"] == "body-v1"
            deadline = hello["t_ms"] + 10_000
            client.sendall(
                json.dumps(
                    {
                        "type": "cmd",
                        "cmd": "duck.posture",
                        "act_id": "P1",
                        "want": "sit",
                        "deadline_ms": deadline,
                    }
                ).encode()
                + b"\n"
            )
            ack = json.loads(reader.readline())
            assert ack["type"] == "ack" and ack["ok"] is True and ack["act_id"] == "P1"
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not any(
                m.get("method") == "robot.do" for m in list(robotd.received)
            ):
                time.sleep(0.05)
            methods = [m.get("method") for m in list(robotd.received)]
            assert "robot.subscribe" in methods and "robot.health" in methods
            assert any(
                m.get("method") == "robot.do" and m["params"]["skill"] == "sit_toggle"
                for m in list(robotd.received)
            )
            sub = next(m for m in robotd.received if m.get("method") == "robot.subscribe")
            assert sub["params"] == {}  # every tick
            assert server.ticks >= 1
            client.close()
        finally:
            server.stop()
            robotd.close()
