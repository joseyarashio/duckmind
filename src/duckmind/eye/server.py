"""duck-eye, the process (spec §4.3, §6; owner decision D7): ``eye.sock`` in, ``eye.result`` out.

Two layers, so the headless rig, the determinism tests (P2) and the process run the same code:

* :class:`EyeCore` — the eye's state and its step functions, with every time passed in (no
  socket, no thread, no clock): the pose tap, the last ``robot.health``, the governor's budget,
  the light tracker and the tasks (with their memories, e.g. the floor model).
  :meth:`EyeCore.admit` asks the governor; :meth:`EyeCore.finish` turns one fetched frame into an
  :class:`~duckmind.eye.protocol.EyeResult` — or ``None`` while the pose sample after the
  capture may still arrive; :meth:`EyeCore.run_frame` does a whole look from bytes in hand
  (the rig, the tests).
* :class:`EyeServer` — sockets, threads and clocks: a unix socket (duck-body is its one client),
  a reader thread that takes poses, health and ``eye.learn`` as they come and queues looks, and a
  worker that serves the looks one at a time: governor, ``not_before_ns``, the frame on the
  handed-over fd (``media.frame``, id ``media_id``) between two clock pairs, the pose pairing
  (waiting for the tap's next sample), the tasks, the answer. The frame's bytes are dropped with
  the look; nothing but integers leaves.

**Tasks.** ``light`` is the eye's own (:mod:`duckmind.eye.light`, every look). The others come from
their modules when present (:data:`TASK_MODULES`): a class with ``name`` equal to the task,
``run(view: LookView) -> dict`` (integers only) and optionally ``learn(event: EyeLearn)``,
``digest_items() -> list[str]`` (what goes into the perception hash: thresholds, weights) and
``from_config(options: dict)``. A task that raises or answers anything but integers is reported
as ``{"error": …}`` and the look goes on. Geometry tasks need the pose: on a ``moved`` look only
``light`` runs. At most one NPU task a look, and only when the governor allows (``duck``).

Run: ``python -m duckmind.eye.server --socket /run/duckmind/eye.sock`` (beside the simulator:
``<state>/eye.sock``). It holds no robot socket and opens no camera: the only frames it sees
come through fds duck-body hands it (spec §6: not in the ``robot`` group; ``InaccessiblePaths``
for robotd.sock and tofd on the board).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import queue
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind.eye import frames as eye_frames
from duckmind.eye import governor
from duckmind.eye import light as eye_light
from duckmind.eye import protocol as proto
from duckmind.eye.camera import EyeCamera
from duckmind.eye.frames import (
    ClockBridge,
    ClockPair,
    FrameError,
    LookView,
    MediaFrameHeader,
    PoseTap,
    pair_pose,
    sample_quarter,
)
from duckmind.eye.protocol import EyeHealth, EyeLearn, EyeLook, EyeResult, PoseSample

TASK_MODULES = {
    "floor": "duckmind.eye.floorline",
    "nest": "duckmind.eye.nesttag",
    "things": "duckmind.eye.things",
    "duck": "duckmind.eye.detect",
}
NPU_TASKS = ("duck",)
POSE_TASKS = ("floor", "nest", "things", "duck")  # need the paired pose (bearings, ranges)
FETCH_TIMEOUT_S = 1.0  # mediad's own capture timeout is 0.5 s
MEDIAD_NO_FRAME = -32603  # mediad: "no frame arrived within the capture timeout"
LOOK_QUEUE = 2  # looks waiting behind the one being served; more are refused ("busy")
EYE_FILES = ("protocol.py", "frames.py", "light.py", "camera.py", "governor.py", "server.py")


# ── the tasks ───────────────────────────────────────────────────────────────


def _task_class(module: Any, name: str) -> Any | None:
    for _attr, value in inspect.getmembers(module, inspect.isclass):
        if getattr(value, "name", None) == name and callable(getattr(value, "run", None)):
            if value.__module__ == module.__name__:
                return value
    return None


def load_tasks(
    names: Iterable[str] = tuple(TASK_MODULES),
    options: Mapping[str, Mapping[str, Any]] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The task objects whose modules are here (a missing module: the task is not offered)."""
    out: dict[str, Any] = {}
    for name in names:
        module_name = TASK_MODULES.get(name)
        if module_name is None:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            if log:
                log(f"task {name}: {module_name} not available ({error})")
            continue
        cls = _task_class(module, name)
        if cls is None:
            if log:
                log(f"task {name}: {module_name} has no task class")
            continue
        opts = dict((options or {}).get(name, {}))
        try:
            factory = getattr(cls, "from_config", None)
            out[name] = factory(opts) if callable(factory) else cls()
        except Exception as error:  # a task that cannot start is not offered
            if log:
                log(f"task {name}: {type(error).__name__}: {error}")
    bind_tasks(out, log)
    return out


def bind_tasks(tasks: Mapping[str, Any], log: Callable[[str], None] | None = None) -> None:
    """Let tasks that share work find each other: every task with a callable ``bind`` gets the
    dict of loaded tasks (``things`` shares ``floor``'s learned model and reuses its line)."""
    for name, task in tasks.items():
        bind = getattr(task, "bind", None)
        if callable(bind):
            try:
                bind(dict(tasks))
            except Exception as error:
                if log:
                    log(f"task {name}: bind: {type(error).__name__}: {error}")


def perception_hash(
    tasks: Mapping[str, Any],
    thresholds: eye_light.Thresholds,
    camera: EyeCamera | None = None,
) -> str:
    """``perception_hash`` (spec §7.6): the eye's code, the tasks' code and weights, the light
    thresholds, K and the lens. ``p:`` + 16 hex digits."""
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in EYE_FILES:
        path = here / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    for name in sorted(tasks):
        task = tasks[name]
        source = inspect.getsourcefile(type(task))
        if source:
            digest.update(name.encode() + b"\0" + Path(source).read_bytes())
        items = getattr(task, "digest_items", None)
        if callable(items):
            for item in items():
                digest.update(str(item).encode())
    for item in thresholds.digest_items():
        digest.update(item.encode())
    cam = camera or EyeCamera.for_frame(1280, 720)
    for item in cam.config_digest_items():
        digest.update(item.encode())
    return "p:" + digest.hexdigest()[:16]


# ── the core: the eye's state and step functions ─────────────────────────────


@dataclass
class EyeConfig:
    thresholds: eye_light.Thresholds = eye_light.THRESHOLDS
    hash_frames: bool = True  # sha16 of each frame in eye.result (audit only)
    fetch_timeout_s: float = FETCH_TIMEOUT_S
    tasks: tuple[str, ...] = tuple(TASK_MODULES)
    task_options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Fetched:
    """A frame as the server fetched it for a look (bytes for this look only)."""

    header: MediaFrameHeader
    data: bytes
    bridge: ClockBridge
    cost_us: dict[str, int]


class EyeCore:
    """duck-eye's state; every method takes its time from the caller."""

    def __init__(
        self,
        config: EyeConfig | None = None,
        tasks: Mapping[str, Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or EyeConfig()
        if tasks is not None:
            self.tasks: dict[str, Any] = dict(tasks)
            bind_tasks(self.tasks, log)
        else:
            self.tasks = load_tasks(self.config.tasks, self.config.task_options, log)
        self.tap = PoseTap()
        self.health: EyeHealth | None = None
        self.budget = governor.Budget()
        self.light = eye_light.LightTracker(self.config.thresholds)
        self.pv = perception_hash(self.tasks, self.config.thresholds)
        self.looks = 0
        self.log = log

    # ── what duck-body tells the eye between looks ──────────────────

    def hello(self) -> proto.EyeHello:
        return proto.EyeHello(
            perception_hash=self.pv, tasks=proto.task_names(["light", *self.tasks])
        )

    def on_pose(self, sample: PoseSample) -> None:
        self.tap.add(sample)

    def on_health(self, health: EyeHealth) -> None:
        self.health = health

    def on_learn(self, event: EyeLearn) -> dict[str, Any]:
        """Hand a walk's outcome to every task that learns (the floor model, D1)."""
        out: dict[str, Any] = {}
        for name, task in self.tasks.items():
            learn = getattr(task, "learn", None)
            if callable(learn):
                try:
                    out[name] = learn(event)
                except Exception as error:
                    out[name] = f"{type(error).__name__}: {error}"
        return out

    # ── one look ────────────────────────────────────────────────────

    def admit(self, look: EyeLook, now_ns: int) -> governor.Decision:
        """The governor's word on this look (a retake is the same look: not paced again)."""
        decision = governor.admit(
            look.kind, look.mode, now_ns, self.budget, self.health, retake=look.attempt > 0
        )
        self.budget = decision.budget
        return decision

    def refuse(
        self,
        look: EyeLook,
        why: str,
        *,
        retake_in_ms: int | None = None,
        cost_us: Mapping[str, int] | None = None,
        gov: Mapping[str, Any] | None = None,
    ) -> EyeResult:
        """A look that got no frame (status ``none``)."""
        return EyeResult(
            look_id=look.look_id,
            attempt=look.attempt,
            kind=look.kind,
            status="none",
            why=why,
            retake_in_ms=retake_in_ms,
            pv=self.pv,
            cost_us=dict(cost_us or {}),
            gov=dict(gov or {}),
        )

    def finish(
        self,
        look: EyeLook,
        fetched: Fetched,
        now_ns: int,
        decision: governor.Decision | None = None,
        *,
        force: bool = False,
    ) -> EyeResult | None:
        """The result of a look whose frame is in hand; ``None`` while the pose sample after
        the capture may still come (call again when the tap grows, or with ``force`` to settle
        for what is there)."""
        header = fetched.header
        cost = dict(fetched.cost_us)
        gov = decision.allow.to_json() if decision is not None else {}
        if decision is not None:
            gov["npu"] = decision.npu
        capture_ns = fetched.bridge.capture_ns(header)
        pairing: eye_frames.Pairing
        if capture_ns is None:
            pairing = eye_frames.Pairing("no_pose")
            why = "clock"
        else:
            wait_until = now_ns if not force else capture_ns + eye_frames.POSE_WAIT_NS
            pairing = pair_pose(self.tap.samples, capture_ns, max(now_ns, wait_until))
            if pairing.status == "pending":
                return None
            why = {"moved": "head_moved", "no_pose": "no_pose"}.get(pairing.status, "")
        at_ns = capture_ns if capture_ns is not None else fetched.bridge.after.monotonic_ns
        frame_info: dict[str, Any] = {
            "w": header.width,
            "h": header.height,
            "rotate": header.rotate,
        }
        t0 = time.perf_counter_ns()
        if self.config.hash_frames:
            frame_info["sha16"] = eye_frames.frame_sha16(fetched.data)
        planes = sample_quarter(fetched.data, header.width, header.height, header.rotate)
        cost["sample"] = (time.perf_counter_ns() - t0) // 1000
        common: dict[str, Any] = {
            "look_id": look.look_id,
            "attempt": look.attempt,
            "kind": look.kind,
            "pv": self.pv,
            "captured_at_ns": capture_ns,
            "captured_at_unix_us": header.captured_at_unix_us,
            "frame": frame_info,
            "gov": gov,
        }
        if pairing.status != "ok" and look.attempt < proto.MAX_RETAKES:
            # the frame is not counted anywhere: the retake will be
            return EyeResult(
                status="retake",
                why=why,
                retake_in_ms=proto.RETAKE_IN_MS,
                pose=pairing.pose.to_json() if pairing.pose is not None else None,
                cost_us=cost,
                **common,
            )
        t0 = time.perf_counter_ns()
        reading = eye_light.read_quarter(planes.y, self.config.thresholds)
        light = self.light.update(reading, at_ns)
        cost["light"] = (time.perf_counter_ns() - t0) // 1000
        results: dict[str, dict[str, Any]] = {"light": light}
        self.looks += 1
        if pairing.status != "ok":
            return EyeResult(
                status="moved",
                why=why,
                pose=pairing.pose.to_json() if pairing.pose is not None else None,
                results=results,
                cost_us=cost,
                **common,
            )
        assert pairing.pose is not None
        view = LookView(
            look=look,
            header=header,
            data=fetched.data,
            camera=EyeCamera.for_header(header),
            planes=planes,
            captured_at_ns=at_ns,
            pose=pairing.pose,
            light=light,
        )
        npu_used = False
        for name in look.runs:
            if name == "light":
                continue
            task = self.tasks.get(name)
            if task is None:
                continue
            if name in NPU_TASKS:
                if npu_used or (decision is not None and not decision.npu):
                    continue
                npu_used = True
            t0 = time.perf_counter_ns()
            try:
                out = task.run(view)
            except Exception as error:
                out = {"error": type(error).__name__}
                if self.log:
                    self.log(f"task {name} on {look.look_id}: {type(error).__name__}: {error}")
            cost[name] = (time.perf_counter_ns() - t0) // 1000
            if out is None:
                out = {}
            if not isinstance(out, Mapping):
                out = {"error": "not_a_record"}
            elif proto.integers_only(out, name):
                out = {"error": "not_integers"}
            results[name] = dict(out)
        view.data = b""  # the bytes go with the look
        return EyeResult(
            status="ok",
            pose=pairing.pose.to_json(),
            results=results,
            cost_us=cost,
            **common,
        )

    def run_frame(
        self,
        look: EyeLook,
        header: MediaFrameHeader,
        data: bytes,
        *,
        captured_at_ns: int,
        poses: Iterable[PoseSample] = (),
        now_ns: int | None = None,
        admit: bool = True,
    ) -> EyeResult:
        """One whole look from bytes in hand (the headless rig, the tests): the poses are added
        to the tap, the capture is at ``captured_at_ns`` on CLOCK_MONOTONIC (the header's wall
        clock is bridged to it exactly), and the pairing does not wait."""
        for sample in poses:
            self.on_pose(sample)
        now = captured_at_ns if now_ns is None else now_ns
        decision = self.admit(look, now) if admit else None
        if decision is not None and not decision.ok:
            return self.refuse(
                look, "governor", retake_in_ms=decision.retry_in_ms, gov=decision.allow.to_json()
            )
        offset = header.captured_at_unix_us * 1000 - captured_at_ns
        pair = ClockPair(captured_at_ns + offset, captured_at_ns)
        fetched = Fetched(header, data, ClockBridge(pair, pair), {})
        result = self.finish(look, fetched, now, decision, force=True)
        assert result is not None
        return result


# ── the process: sockets, threads, clocks ──────────────────────────────────────


def monotonic_ns() -> int:
    """CLOCK_MONOTONIC (robotd's clock; not ``time.monotonic_ns`` on macOS)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def clock_pair() -> ClockPair:
    """REALTIME read between two MONOTONIC reads, paired with their midpoint."""
    a = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    r = time.clock_gettime_ns(time.CLOCK_REALTIME)
    b = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return ClockPair(r, (a + b) // 2)


def _stderr(line: str) -> None:
    sys.stderr.write(f"duck-eye: {line}\n")
    sys.stderr.flush()


class EyeServer:
    """duck-eye on a unix socket. ``start()`` serves on threads; ``close()`` stops."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        core: EyeCore | None = None,
        *,
        log: Callable[[str], None] = _stderr,
    ) -> None:
        self.path = Path(path)
        self.log = log
        self.core = core or EyeCore(log=log)
        self.lock = threading.Lock()
        self.new_pose = threading.Condition(self.lock)
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._client: socket.socket | None = None
        self._send_lock = threading.Lock()
        self.stats = {"clients": 0, "looks": 0, "ok": 0, "retake": 0, "moved": 0, "none": 0}

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> EyeServer:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.path))
            except OSError:
                self.path.unlink()  # a stale socket
            else:
                probe.close()
                raise RuntimeError(f"another duck-eye serves {self.path}")
            finally:
                probe.close()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.path))
        os.chmod(self.path, 0o660)
        listener.listen(2)
        listener.settimeout(0.2)
        self._listener = listener
        thread = threading.Thread(target=self._accept, name="eye-accept", daemon=True)
        thread.start()
        self._threads.append(thread)
        return self

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        client = self._client
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        with self.lock:
            self.new_pose.notify_all()
        for thread in self._threads:
            thread.join(timeout=timeout_s)
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        try:
            self.path.unlink()
        except OSError:
            pass

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.close()

    # ── connections ─────────────────────────────────────────────────

    def _accept(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if self._client is not None:
                self._send(conn, proto.error("busy", "duck-eye already has its duck-body"))
                conn.close()
                continue
            self._client = conn
            self.stats["clients"] += 1
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _send(self, conn: socket.socket, message: Mapping[str, Any]) -> None:
        data = proto.encode(message)
        with self._send_lock:
            try:
                conn.sendall(data)
            except OSError:
                pass

    def _serve(self, conn: socket.socket) -> None:
        reader = eye_frames.FdLineReader(conn, max_line=proto.MAX_LINE_BYTES)
        looks: queue.Queue[tuple[EyeLook, int | None] | None] = queue.Queue()
        worker = threading.Thread(
            target=self._work, args=(conn, looks), name="eye-looks", daemon=True
        )
        worker.start()
        greeted = False
        try:
            while not self._stop.is_set():
                try:
                    got = reader.read_line()
                except ValueError:
                    self._send(conn, proto.error("too_long", "line too long"))
                    break
                except OSError:
                    break
                if got is None:
                    break
                line, offset = got
                message: dict[str, Any] = {}
                try:
                    message = proto.decode(line)
                    item = proto.parse_to_eye(message)
                except proto.ProtocolError as error:
                    if error.code == "incomplete_pose":  # 25 Hz: counted, never answered
                        self.stats["incomplete_poses"] = self.stats.get("incomplete_poses", 0) + 1
                        reader.drop_stale(offset + len(line))
                        continue
                    fd = reader.take_fd(offset) if message.get("type") == proto.LOOK else None
                    if fd is not None:
                        os.close(fd)
                    raw = message.get("look_id")
                    look_id = raw if isinstance(raw, str) else None
                    self._send(conn, proto.error(error.code, str(error), look_id))
                    reader.drop_stale(offset + len(line))
                    if error.code == "api":
                        break
                    continue
                if isinstance(item, proto.Hello):
                    greeted = True
                    self._send(conn, self.core.hello().to_json())
                elif not greeted:
                    self._send(conn, proto.error("hello", "say eye.hello first"))
                elif isinstance(item, PoseSample):
                    with self.lock:
                        self.core.on_pose(item)
                        self.new_pose.notify_all()
                elif isinstance(item, EyeHealth):
                    with self.lock:
                        self.core.on_health(item)
                elif isinstance(item, EyeLearn):
                    with self.lock:
                        self.core.on_learn(item)
                elif isinstance(item, EyeLook):
                    fd = reader.take_fd(offset)
                    if looks.qsize() >= LOOK_QUEUE:
                        if fd is not None:
                            os.close(fd)
                        with self.lock:
                            result = self.core.refuse(item, "busy", retake_in_ms=200)
                        self._answer(conn, result)
                    else:
                        looks.put((item, fd))
                reader.drop_stale(offset + len(line))
        finally:
            looks.put(None)
            worker.join(timeout=5.0)
            reader.close()
            try:
                conn.close()
            except OSError:
                pass
            self._client = None

    def _answer(self, conn: socket.socket, result: EyeResult) -> None:
        self.stats["looks"] += 1
        self.stats[result.status] = self.stats.get(result.status, 0) + 1
        problems = result.problems()
        if problems:  # never let a float out
            self.log(f"result for {result.look_id} not integers: {problems[:2]}")
            result = self.core.refuse(
                EyeLook(result.look_id, ("light",), 0, 0, kind=result.kind),
                "frame_error",
            )
        self._send(conn, result.to_json())

    def _work(
        self, conn: socket.socket, looks: queue.Queue[tuple[EyeLook, int | None] | None]
    ) -> None:
        while True:
            item = looks.get()
            if item is None:
                return
            look, fd = item
            try:
                result = self._look(look, fd)
            except Exception as error:  # the eye must answer every look
                self.log(f"look {look.look_id}: {type(error).__name__}: {error}")
                with self.lock:
                    result = self.core.refuse(look, "frame_error")
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            self._answer(conn, result)

    def _look(self, look: EyeLook, fd: int | None) -> EyeResult:
        if fd is None:
            with self.lock:
                return self.core.refuse(look, "no_fd")
        with self.lock:
            decision = self.core.admit(look, monotonic_ns())
        gov = decision.allow.to_json()
        if not decision.ok:
            with self.lock:
                return self.core.refuse(
                    look, "governor", retake_in_ms=decision.retry_in_ms, gov=gov
                )
        cost: dict[str, int] = {}
        t_wait = monotonic_ns()
        wait = eye_frames.wait_ns(look.not_before_ns, t_wait)
        if look.deadline_ns is not None and t_wait + wait > look.deadline_ns:
            with self.lock:
                return self.core.refuse(look, "deadline", gov=gov)
        if wait:
            time.sleep(wait / 1e9)
        cost["wait"] = (monotonic_ns() - t_wait) // 1000
        media: socket.socket | None = None
        before = clock_pair()
        try:
            copy = os.dup(fd)
            try:
                media = socket.socket(fileno=copy)
            except OSError:
                os.close(copy)
                raise
            header, data = eye_frames.fetch_on(
                media, media_id=look.media_id, timeout_s=self.core.config.fetch_timeout_s
            )
        except TimeoutError:
            with self.lock:
                return self.core.refuse(look, "timeout", gov=gov)
        except FrameError as error:
            # mediad's own capture timeout (-32603: no frame within 500 ms) is a timeout
            why = "timeout" if error.code == MEDIAD_NO_FRAME else "frame_error"
            with self.lock:
                return self.core.refuse(look, why, gov=gov)
        except (OSError, ValueError):
            with self.lock:
                return self.core.refuse(look, "frame_error", gov=gov)
        finally:
            if media is not None:
                media.close()
        after = clock_pair()
        cost["fetch"] = (after.monotonic_ns - before.monotonic_ns) // 1000
        fetched = Fetched(header, data, ClockBridge(before, after), cost)
        with self.lock:
            while True:
                now = monotonic_ns()
                if look.deadline_ns is not None and now > look.deadline_ns:
                    return self.core.refuse(look, "deadline", gov=gov)
                result = self.core.finish(look, fetched, now, decision)
                if result is not None:
                    return result
                if self._stop.is_set():
                    return self.core.finish(look, fetched, now, decision, force=True) or (
                        self.core.refuse(look, "timeout", gov=gov)
                    )
                self.new_pose.wait(timeout=0.02)


def _keep_floor(core: EyeCore, path: Path) -> None:
    """Write the floor task's model to ``path`` after every ``eye.learn`` that added samples
    (atomically: a temporary file, then a rename)."""
    floor = core.tasks.get("floor")
    model = getattr(floor, "model", None)
    if model is None or not hasattr(model, "to_json"):
        return
    learn = core.on_learn

    def on_learn(event: EyeLearn) -> dict[str, Any]:
        out = learn(event)
        added = out.get("floor")
        if isinstance(added, int) and added > 0:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                spare = path.with_suffix(path.suffix + ".tmp")
                spare.write_text(json.dumps(model.to_json(), separators=(",", ":")), "utf-8")
                os.replace(spare, path)
            except OSError as error:
                _stderr(f"floor model not written to {path}: {error}")
        return out

    core.on_learn = on_learn  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duckmind.eye.server", description=__doc__)
    parser.add_argument("--socket", default=proto.EYE_SOCKET, help="eye.sock path")
    parser.add_argument("--no-hash", action="store_true", help="no frame sha16 in results")
    parser.add_argument(
        "--tasks", default=",".join(TASK_MODULES), help="tasks to offer besides light"
    )
    parser.add_argument("--duck-model", default=None, help="duck-detect .onnx/.rknn (optional)")
    parser.add_argument(
        "--floor-model",
        default=None,
        help="the duck's learned floor (JSON): read at start when it exists, written after "
        "every eye.learn that added samples (owner D1: the floor it learnt by walking)",
    )
    args = parser.parse_args(argv)
    options: dict[str, dict[str, Any]] = {}
    if args.duck_model:
        options["duck"] = {"model": args.duck_model}
    floor_file = Path(args.floor_model) if args.floor_model else None
    if floor_file is not None and floor_file.is_file():
        try:
            options["floor"] = {"model": json.loads(floor_file.read_text(encoding="utf-8"))}
        except (OSError, ValueError) as error:
            _stderr(f"floor model {floor_file} not read: {error}")
    config = EyeConfig(
        hash_frames=not args.no_hash,
        tasks=tuple(t for t in args.tasks.split(",") if t),
        task_options=options,
    )
    core = EyeCore(config, log=_stderr)
    if floor_file is not None:
        _keep_floor(core, floor_file)
    server = EyeServer(args.socket, core)
    _stderr(f"serving {args.socket} ({core.pv}; tasks: {', '.join(core.hello().tasks)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
