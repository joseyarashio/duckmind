"""The duck-body process (spec §6, §7.3): the only place in duck-body with sockets, threads and
clocks. Everything it decides is :class:`~duckmind.duckbody.bodyloop.BodyLoop`'s.

It owns, as Duck Mind's only robotd client, three ``robotd.sock`` connections (state / command /
safety, :class:`~duckmind.duckbody.robotd_client.RobotdClient`), optionally ``tof.sock``
(``tof.stream``), and for every look one fresh ``media.sock`` connection whose fd it hands to
duck-eye over ``eye.sock`` (SCM_RIGHTS, riding on the ``eye.look`` line). It serves
``body.sock``: JSON lines, ``hello`` first, then commands in, ``ack``/``percept``/``done`` out
(:mod:`duckmind.duckbody.protocol`).

Clock: ``CLOCK_MONOTONIC`` via :func:`time.clock_gettime_ns` — robotd's ``t_ns`` clock (on macOS
``time.monotonic_ns`` is a different clock). Robot ticks carry their own ``t_ns``; commands are
stamped when they arrive. If robotd goes quiet the loop stops streaming moves (robotd's deadman
then zeroes the twist within 500 ms) and reconnects.
"""

from __future__ import annotations

import json
import os
import queue
import select
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.robotd_client import API_VERSION, Cadence, RobotdClient, RobotdError

MAX_LINE = 2_000_000


def now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _log(text: str) -> None:
    """One line on stderr (the process's log; nothing here reaches body.sock)."""
    print(f"duck-body: {text}", file=sys.stderr, flush=True)


LOG_LOOKS = os.environ.get("DUCKMIND_BODY_LOG_LOOKS") == "1"  # every eye.result on stderr


def _look_summary(line: Mapping[str, Any]) -> str:
    """One eye.result in a line: status, why, light, the floor's learning, things seen."""

    def part(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    results = part(line.get("results"))
    light, floor, things = (
        part(results.get("light")),
        part(results.get("floor")),
        part(results.get("things")),
    )
    seen = [
        f"{t.get('colour')}@{t.get('r_mm')}mm/{t.get('b_mrad')}mrad"
        for t in (things.get("things") or [])
        if isinstance(t, Mapping)
    ]
    return (
        f"eye.result {line.get('look_id')}: {line.get('status')} {line.get('why') or ''} "
        f"light={light.get('level')} floor(learned={floor.get('learned')} "
        f"walks={floor.get('walks')} free={floor.get('free_mm')}) things={seen}"
    )


def _hello_answer(media: socket.socket, limit: int = 65_536) -> None:
    """Read mediad's answer to our ``hello`` (JSON-RPC id 1) off a fresh media connection, so
    the connection handed to duck-eye starts clean (``eye.frames.fetch_on`` expects that)."""
    data = b""
    while not data.endswith(b"\n"):
        chunk = media.recv(1)
        if not chunk:
            raise ConnectionError("media.sock closed before answering hello")
        data += chunk
        if len(data) > limit:
            raise ValueError("media.sock hello answer too long")
    answer = json.loads(data)
    if not isinstance(answer, dict) or answer.get("id") != 1 or "error" in answer:
        raise ValueError(f"media.sock refused hello: {answer}")


def unix_ms() -> int:
    return time.time_ns() // 1_000_000


class UnixWire:
    """One JSON-lines unix-socket connection (the :class:`~.robotd_client.Wire` the client uses)."""

    def __init__(self, path: str | Path, timeout_s: float = 2.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(str(path))
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def send(self, obj: Mapping[str, Any]) -> None:
        data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with self.lock:
            self.sock.sendall(data)

    def send_line_with_fd(self, obj: Mapping[str, Any], fd: int) -> None:
        data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with self.lock:
            socket.send_fds(self.sock, [data], [fd])

    def recv(self, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                line = bytes(self.buffer[:end])
                del self.buffer[: end + 1]
                if not line.strip():
                    continue
                value = json.loads(line)
                return value if isinstance(value, dict) else None
            if len(self.buffer) > MAX_LINE:
                raise RobotdError("line too long")
            left = deadline - time.monotonic()
            # A zero timeout is a non-blocking poll, not an instruction to skip the socket.
            # Callers use recv(0) to drain already queued robot.state notifications after an
            # RPC; checking the deadline first stranded those frames in the kernel FIFO.
            if not select.select([self.sock], [], [], max(0.0, left))[0]:
                return None
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("peer closed the socket")
            self.buffer.extend(data)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class Paths:
    robotd: str
    body: str
    tof: str | None = None
    eye: str | None = None
    media: str | None = None


class _Client:
    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn
        self.lock = threading.Lock()
        self.alive = True

    def send(self, message: Mapping[str, Any]) -> None:
        if not self.alive:
            return
        try:
            with self.lock:
                self.conn.sendall(P.encode_line(message))
        except OSError:
            self.alive = False


class BodyServer:
    def __init__(
        self, paths: Paths, config: BodyConfig | None = None, loop: BodyLoop | None = None
    ) -> None:
        self.paths = paths
        self.loop = loop or BodyLoop(config)
        self.lock = threading.Lock()
        self.clients: list[_Client] = []
        self.stopping = threading.Event()
        self.eye_results: queue.Queue[dict[str, Any]] = queue.Queue()
        self.depth: P.DepthView | None = None
        self.robot: RobotdClient | None = None
        self.eye: UnixWire | None = None
        self.threads: list[threading.Thread] = []
        self.listener: socket.socket | None = None
        self.health = Cadence()
        self.sent = 0
        self.ticks = 0
        self.tof_frames: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.tracker: Any = None
        self.walk_memory: Any = None  # walkexec's WalkMemory, when main() made the executors
        self._refusals: dict[str, int] = {}  # guard refusal reason → when last logged (t_ns)
        self.failed: BaseException | None = None  # the brain's own error that ended the process

    # ── body.sock ──

    def _listen(self) -> None:
        path = self.paths.body
        if os.path.exists(path):
            os.unlink(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(4)
        listener.settimeout(0.5)
        self.listener = listener
        while not self.stopping.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            client = _Client(conn)
            caps = {"tof": self.paths.tof is not None, "eye": self.eye is not None, "pg": False}
            client.send(
                P.hello(
                    pv=self.loop.cfg.pv,
                    src=self.loop.cfg.src,
                    t_ms=now_ns() // 1_000_000,
                    unix_ms=unix_ms(),
                    caps=caps,
                )
            )
            with self.lock:
                self.clients.append(client)
            thread = threading.Thread(target=self._read_client, args=(client,), daemon=True)
            thread.start()

    def _read_client(self, client: _Client) -> None:
        buffer = bytearray()
        conn = client.conn
        conn.settimeout(0.5)
        while client.alive and not self.stopping.is_set():
            try:
                data = conn.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            buffer.extend(data)
            while (end := buffer.find(b"\n")) >= 0:
                line = bytes(buffer[:end])
                del buffer[: end + 1]
                try:
                    obj = P.decode_line(line)
                except P.ProtocolError as problem:
                    client.send(P.error(str(problem)))
                    continue
                if obj.get("type") != "cmd":
                    client.send(P.error("expected a line with type cmd"))
                    continue
                with self.lock:
                    replies = self.loop.command(obj, now_ns())
                for reply in replies:
                    client.send(reply)
        client.alive = False
        try:
            conn.close()
        except OSError:
            pass

    def _broadcast(self, lines: list[dict[str, Any]]) -> None:
        with self.lock:
            clients = [c for c in self.clients if c.alive]
            self.clients = clients
        for line in lines:
            for client in clients:
                client.send(line)

    # ── robotd ──

    def _connect_robot(self) -> RobotdClient:
        client = RobotdClient(
            UnixWire(self.paths.robotd), UnixWire(self.paths.robotd), UnixWire(self.paths.robotd)
        )
        client.start()
        return client

    def _robot_loop(self) -> None:
        while not self.stopping.is_set():
            try:
                self.robot = self._connect_robot()
            except (OSError, RobotdError, ConnectionError, json.JSONDecodeError):
                time.sleep(0.5)
                continue
            try:
                self._drive(self.robot)
            except (OSError, RobotdError, ConnectionError, json.JSONDecodeError):
                self.loop.proprio.epoch += 1  # robotd restarted: the odometry frame broke
                time.sleep(0.2)
            except Exception as problem:  # noqa: BLE001 — a fault in the brain itself
                self._fail(problem)
                return

    def _fail(self, problem: BaseException) -> None:
        """A fault in duck-body's own brain (not an I/O error): the legs stop, the log and every
        body.sock client hear why, and the process ends non-zero so its supervisor restarts it —
        never a silent drive thread with the reflexes gone."""
        import traceback

        self.failed = problem
        _log("brain fault, stopping: " + "".join(traceback.format_exception(problem)).strip())
        if self.robot is not None:
            try:
                self.robot.send(P.Intent.move(0.0, 0.0))
            except (OSError, RobotdError):
                pass
        self._broadcast([P.error(f"duck-body stopped: {problem!r}")])
        self.stopping.set()

    def _drive(self, robot: RobotdClient) -> None:
        while not self.stopping.is_set():
            state = robot.read_state(0.2)
            if state is None:
                continue
            eye: list[dict[str, Any]] = []
            while not self.eye_results.empty():
                eye.append(self.eye_results.get_nowait())
            if self.paths.tof is not None:
                self._update_depth(
                    self.loop.tap.ring[-1] if self.loop.tap.ring else None,
                    self.loop._during() == "walk",
                )
            with self.lock:
                tick = self.loop.tick(state, depth=self.depth, eye=eye)
            self.ticks += 1
            for intent, why in tick.refused:  # the guard's refusals, once per reason per 5 s
                said = self._refusals.get(why)
                if said is None or state.t_ns - said >= 5_000_000_000:
                    self._refusals[why] = state.t_ns
                    _log(f"guard refused {intent.method} {dict(intent.params)}: {why}")
            for intent in tick.intents:
                robot.send(intent)
                self.sent += 1
            if tick.lines:
                self._broadcast(tick.lines)
            if self.eye is not None and tick.pose is not None and tick.eye_pose:
                try:
                    self.eye.send(tick.pose.to_wire())
                except OSError:
                    self.eye = None
            for look in tick.looks:
                self._look(look, tick.pose.seq if tick.pose is not None else 0)
            self._send_learn()
            t = now_ns()
            if self.health.due(t):
                try:
                    lines = self.loop.health(robot.health(), t)
                except RobotdError:
                    lines = []
                if lines:
                    self._broadcast(lines)

    # ── duck-eye ──

    def _send_learn(self) -> None:
        """Owner D1: each finished walk's walked-safe ground (walkexec's ``eye.learn``) goes to
        duck-eye, which learns the floor from it."""
        memory = self.walk_memory
        if memory is None or not getattr(memory, "learn", None):
            return
        lessons, memory.learn[:] = list(memory.learn), []
        if self.eye is None:
            return
        for lesson in lessons:
            try:
                self.eye.send(lesson.to_json())
            except OSError:
                self.eye = None
                return

    def _connect_eye(self) -> None:
        if self.paths.eye is None:
            return
        try:
            wire = UnixWire(self.paths.eye)
            wire.send({"type": "eye.hello", "api": "eye-v1", "role": "duck-body"})
            answer = wire.recv(2.0)
            if not answer or answer.get("type") != "eye.hello":
                _log(f"duck-eye did not say hello: {answer}")
                wire.close()
                return
            self.eye = wire
        except (OSError, json.JSONDecodeError) as error:
            _log(f"no duck-eye at {self.paths.eye}: {error}")
            self.eye = None
            return
        thread = threading.Thread(target=self._read_eye, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _read_eye(self) -> None:
        while not self.stopping.is_set() and self.eye is not None:
            try:
                line = self.eye.recv(0.5)
            except (OSError, ConnectionError, json.JSONDecodeError):
                self.eye = None
                break
            if line is not None and line.get("type") == "eye.result":
                if line.get("status") not in ("ok", "retake") or LOG_LOOKS:
                    _log(_look_summary(line))
                self.eye_results.put(line)

    def _look(self, look: P.LookRequest, pose_ref: int) -> None:
        """One look: a fresh media.sock connection, its fd to duck-eye with the eye.look line."""
        if self.eye is None or self.paths.media is None:
            self.eye_results.put(
                {"type": "eye.result", "look_id": look.look_id, "status": "none", "why": "no_eye"}
            )
            return
        media = None
        try:
            media = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            media.settimeout(2.0)
            media.connect(self.paths.media)
            hello = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {"api_version": API_VERSION},
            }
            media.sendall(json.dumps(hello).encode() + b"\n")
            _hello_answer(media)  # read it here: duck-eye's first line must be its own frame's
            line = {
                "type": "eye.look",
                "look_id": look.look_id,
                "attempt": 0,
                "kind": getattr(look, "kind", "end_look"),
                "tasks": list(look.tasks),
                "pose_tap_ref": pose_ref,
                "not_before_ns": look.not_before_ns,
                "deadline_ns": None,
                "media_id": 2,
                "mode": getattr(look, "mode", "awake"),
                "hints": dict(getattr(look, "hints", {}) or {}),
            }
            self.eye.send_line_with_fd(line, media.fileno())
        except (OSError, ValueError) as error:
            _log(f"look {look.look_id}: no media connection for duck-eye ({error})")
            self.eye_results.put(
                {"type": "eye.result", "look_id": look.look_id, "status": "none", "why": "no_fd"}
            )
        finally:
            if media is not None:
                media.close()  # duck-eye holds its own copy of the fd now

    # ── ToF (tofd's tof.stream) ──

    def _tof_loop(self) -> None:
        assert self.paths.tof is not None
        while not self.stopping.is_set():
            try:
                wire = UnixWire(self.paths.tof)
                rpc = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "hello",
                    "params": {"api_version": API_VERSION},
                }
                wire.send(rpc)
                wire.recv(2.0)
                wire.send({"jsonrpc": "2.0", "id": 2, "method": "tof.stream", "params": {}})
                while not self.stopping.is_set():
                    frame = wire.recv(0.5)
                    if frame is None or frame.get("method") != "tof.frame":
                        continue
                    params = frame.get("params")
                    if isinstance(params, dict):
                        if self.tof_frames.full():
                            self.tof_frames.get_nowait()
                        self.tof_frames.put(params)
            except (OSError, ConnectionError, json.JSONDecodeError):
                self.depth = None
                time.sleep(0.5)

    def _update_depth(self, pose: P.PoseSample | None, walking: bool) -> None:
        """The newest tof.frame through body-motion's DepthTracker → the reflexes' DepthView."""
        latest = None
        while not self.tof_frames.empty():
            latest = self.tof_frames.get_nowait()
        if latest is None or pose is None or pose.tof_pos is None or pose.tof_quat is None:
            return
        from duckmind.duckbody import depth as D
        from duckmind.duckbody import tof_reproject as T

        if self.tracker is None:
            self.tracker = D.DepthTracker(standing_height_m=0.12)
        frame = D.ToFFrame.of(
            int(latest.get("t_ns", 0) or pose.t_ns),
            latest.get("distance_mm", []),
            latest.get("status", []),
        )
        state = self.loop.state
        down = state.down if state is not None else None
        sensor = T.Pose(pose.tof_pos, T.Quat(*pose.tof_quat))
        reading, _ = self.tracker.step(
            frame,
            D.SensorPose(sensor, down or (0.0, 0.0, -1.0), pose.trunk_height_mm),
            walking=walking,
        )
        self.depth = D.body_view(reading)

    # ── lifecycle ──

    def start(self) -> None:
        Path(self.paths.body).parent.mkdir(parents=True, exist_ok=True)
        self._connect_eye()
        targets = [self._listen, self._robot_loop]
        if self.paths.tof is not None:
            targets.append(self._tof_loop)
        for target in targets:
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stopping.set()
        if self.robot is not None:
            try:
                self.robot.send(P.Intent.move(0.0, 0.0))
            except (OSError, RobotdError):
                pass
        if self.listener is not None:
            self.listener.close()
        if self.eye is not None:
            self.eye.close()
        for thread in self.threads:
            thread.join(timeout=2.0)
        if os.path.exists(self.paths.body):
            try:
                os.unlink(self.paths.body)
            except OSError:
                pass


def default_executors(
    memory: Any = None, dock_style: str | None = None, dock_side: str | None = None
) -> dict[str, Any]:
    """The act executors the process runs (``BodyConfig.executors``): walkexec (every walk shares
    one :class:`~duckmind.duckbody.walkexec.WalkMemory` — ``memory``, else a new one — with
    darkgate as its dark gate), lookplan's ``duck.look`` on the same grid, dockexec (the home's
    dock style, backin by default — owner 2026-09-28 「6. 可以」 — and its turning side), and
    curiosity's step runner — each only if its module is there."""
    factories: dict[str, Any] = {}
    mem: Any = memory
    try:
        from duckmind.duckbody import walkexec
        from duckmind.duckbody.darkgate import walk_gate

        mem = memory if memory is not None else walkexec.WalkMemory()
        factories["duck.walk"] = walkexec.make_factory(walkexec.DEFAULT, mem, walk_gate)
        from duckmind.duckbody import lookplan

        # duck.look (→ sense.scan): its views mark the same free-cell grid the walks use
        factories["duck.look"] = lookplan.make_look_factory(mem.grid)
    except ImportError:
        pass
    try:
        from duckmind.duckbody import dockexec

        # the dock's style and turning side as configured (the home's dock; the command's own
        # style and side win)
        config = dockexec.DockConfig()
        if dock_style:
            config = replace(config, style=dock_style)
        if dock_side:
            config = replace(config, turn_side=dock_side)
        # a docking that closes PG leaves the dock's frame in the walks' memory (the walk out)
        factories["duck.dock"] = dockexec.make_factory(config, memory=mem)
    except ImportError:
        pass
    try:
        from duckmind.curiosity import act_runner

        factories["duck.curio"] = act_runner.executor
    except ImportError:
        pass
    return factories


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m duckmind.duckbody", description="duck-body")
    parser.add_argument("--robotd", required=True, help="robotd.sock")
    parser.add_argument("--body", required=True, help="body.sock to serve")
    parser.add_argument("--tof", help="tof.sock (a duck with a ToF)")
    parser.add_argument("--eye", help="eye.sock (duck-eye)")
    parser.add_argument("--media", help="media.sock (a fresh connection per look)")
    parser.add_argument("--src", default="live", choices=("robot", "live"))
    parser.add_argument("--vx-max-mm", type=int, default=None, help="owner A3's measured cap")
    parser.add_argument("--mjcf", help="the robot MJCF: leg-FK trunk height, head FK, steps")
    parser.add_argument("--anchors", help="robotd's odometry anchors.rs (the soles, for steps)")
    parser.add_argument(
        "--dock-style",
        choices=("forward", "backin"),
        default=None,
        help="how this duck's charging dock is entered (the home's dock; default backin)",
    )
    parser.add_argument(
        "--dock-side",
        choices=("right", "left"),
        default=None,
        help="backin's turning side: right (default) or left where the room needs the mirror",
    )
    args = parser.parse_args(argv)
    from duckmind.duckbody.reflex import Limits

    limits = Limits() if args.vx_max_mm is None else Limits(vx_max_mm=args.vx_max_mm)
    tap = None
    feet = None
    if args.mjcf:
        from duckmind.duckbody.footfalls import Footfalls, load_sole_points
        from duckmind.duckbody.posture import PostureTap
        from duckmind.duckbody.tof_reproject import Tree

        tap = PostureTap.from_mjcf(args.mjcf)
        tree = Tree.from_mjcf(args.mjcf)
        feet = Footfalls(tree, load_sole_points(args.anchors)) if args.anchors else Footfalls(tree)
    memory: Any = None
    try:
        from duckmind.duckbody.walkexec import WalkMemory

        memory = WalkMemory()
    except ImportError:
        pass
    config = BodyConfig(
        src=args.src,
        has_tof=args.tof is not None,
        limits=limits,
        executors=default_executors(memory, args.dock_style, args.dock_side),
        footfalls=feet,
    )
    server = BodyServer(
        Paths(args.robotd, args.body, args.tof, args.eye, args.media),
        config,
        BodyLoop(config, tap),
    )
    server.walk_memory = memory

    def _terminate(*_: object) -> None:
        raise KeyboardInterrupt  # SIGTERM stops like ^C: a zero move, sockets closed

    import signal

    signal.signal(signal.SIGTERM, _terminate)
    server.start()
    try:
        while not server.stopping.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 1 if server.failed is not None else 0


__all__ = ["BodyServer", "Paths", "UnixWire", "default_executors", "main", "now_ns"]
