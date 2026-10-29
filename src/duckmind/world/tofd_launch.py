"""The official ``tofd`` beside the simulated world (spec §11.4; owner decisions F2, F3).

``tofd`` is the board's ToF daemon (``microduck/tof``), built here unchanged with
``~/.cargo/bin/cargo build --offline -p tof --bin tofd --target aarch64-apple-darwin`` (it
writes only the checkout's ``target/``; the binary sits next to the ``robotd`` the simulator
runs). It publishes ``tof.stream`` on a unix socket exactly as on the duck:

* **a duck with the ToF module**: ``tofd --sim 127.0.0.1:<body port> --hz 15`` polls the world's
  ``tof`` op (:mod:`duckmind.world.tof_model`) and publishes its answers verbatim; it sleeps a
  whole period after each answer, so "15 Hz" is about 14 frames a second, as upstream;
* **a duck without it** (F2): the same daemon on its hardware path, as ``tofd.service`` runs on
  every board: there is no I²C bus to find a sensor on, so ``tof.stream`` answers ``accepted``
  with no ``sensor`` and the reason in ``unavailable``, and no frame ever comes.

:class:`TofdLaunch` starts it under a small guardian (``python -m
duckmind.world.tofd_launch --guard <parent pid> -- <tofd …>``) that stops ``tofd`` when the
process that started it goes away, even when that process is killed without a chance to clean
up. ``DUCK_RUNTIME_DIR`` points the daemon's identity file (``<dir>/tofd/identity.json``) at the
sensors' folder; the socket is ``<dir>/tofd/tof.sock``. :func:`stream_status` and
:func:`subscribe` are a ``tof.stream`` client (JSON-RPC 2.0 lines; the frames are
``tof.frame`` notifications).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

DEFAULT_HZ = 15
TOF_STREAM = "tof.stream"
TOF_FRAME = "tof.frame"
TARGET = "aarch64-apple-darwin"
GUARD_POLL_S = 0.5


def find_tofd(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The tofd binary: ``explicit`` (a path; "auto" or empty searches), else
    ``$DUCKMIND_TOFD``, else next to ``$DUCKMIND_ROBOTD``, else in ``$DUCKMIND_MICRODUCK``'s
    ``target/aarch64-apple-darwin/debug`` or ``target/debug``."""
    given = str(explicit) if explicit is not None else os.environ.get("DUCKMIND_TOFD", "")
    if given and given.strip().lower() not in ("auto", "1", "on", "yes", "true"):
        path = Path(given).expanduser()
        return path if path.is_file() else None
    candidates: list[Path] = []
    robotd = os.environ.get("DUCKMIND_ROBOTD")
    if robotd:
        candidates.append(Path(robotd).expanduser().parent / "tofd")
    microduck = os.environ.get("DUCKMIND_MICRODUCK")
    if microduck:
        root = Path(microduck).expanduser()
        candidates += [
            root / "target" / TARGET / "debug" / "tofd",
            root / "target" / "debug" / "tofd",
        ]
    return next((c for c in candidates if c.is_file()), None)


def tofd_command(
    binary: Path | str,
    *,
    socket_path: Path | str,
    sim: str | None,
    hz: int = DEFAULT_HZ,
) -> list[str]:
    """The daemon's arguments: ``--sim host:port`` for a fitted sensor, the hardware path (no
    sensor to be found here) for a duck without one."""
    command = [str(binary), "--socket", str(socket_path), "--hz", str(int(hz))]
    if sim:
        command += ["--sim", sim]
    else:
        command += ["--no-imu"]
    return command


class TofdLaunch:
    """One ``tofd`` for one world: :meth:`start`, :meth:`status`, :meth:`stop`."""

    def __init__(
        self,
        binary: Path | str,
        *,
        runtime_dir: Path | str,
        body: tuple[str, int] | None,
        hz: int = DEFAULT_HZ,
        log_level: str = "warn",
        guard: bool = True,
    ) -> None:
        self.binary = Path(binary)
        self.runtime_dir = Path(runtime_dir)
        self.body = body  # None: a duck without the ToF module (F2)
        self.hz = int(hz)
        self.log_level = log_level
        self.guard = guard
        self.socket_path = self.runtime_dir / "tofd" / "tof.sock"
        self.log_path = self.runtime_dir / "tofd.log"
        self.process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def fitted(self) -> bool:
        return self.body is not None

    def command(self) -> list[str]:
        sim = f"{self.body[0]}:{self.body[1]}" if self.body is not None else None
        tofd = tofd_command(self.binary, socket_path=self.socket_path, sim=sim, hz=self.hz)
        if not self.guard:
            return tofd
        return [
            sys.executable,
            "-m",
            "duckmind.world.tofd_launch",
            "--guard",
            str(os.getpid()),
            "--",
            *tofd,
        ]

    def start(self) -> TofdLaunch:
        (self.runtime_dir / "tofd").mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        env = {
            **os.environ,
            "DUCK_RUNTIME_DIR": str(self.runtime_dir),
            "RUST_LOG": self.log_level,
            "NO_COLOR": "1",
        }
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (package_root, os.environ.get("PYTHONPATH", "")) if p
        )
        self.process = subprocess.Popen(
            self.command(), env=env, stdout=self._log, stderr=subprocess.STDOUT
        )
        return self

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        """Until the socket answers (or the process died)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            if self.socket_path.exists():
                try:
                    stream_status(self.socket_path, timeout_s=1.0)
                    return True
                except OSError:
                    pass
            time.sleep(0.05)
        return False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, Any]:
        """What the daemon says about itself (``TofStreamResult``) and its process."""
        out: dict[str, Any] = {
            "binary": str(self.binary),
            "fitted": self.fitted,
            "hz": self.hz,
            "socket": str(self.socket_path),
            "pid": self.process.pid if self.process is not None else None,
            "running": self.running,
            "log": str(self.log_path),
        }
        if self.running:
            try:
                out["stream"] = stream_status(self.socket_path, timeout_s=1.0)
            except (OSError, ValueError) as error:
                out["stream_error"] = f"{type(error).__name__}: {error}"
        return out

    def stop(self, timeout_s: float = 3.0) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_s)
        if self._log is not None:
            self._log.close()
            self._log = None


# ── a tof.stream client ─────────────────────────────────────────────────────────


def _connect(path: Path | str, timeout_s: float) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect(str(path))
    except OSError:
        sock.close()
        raise
    return sock


def _request(method: str, request_id: int = 1) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method}).encode() + b"\n"


def stream_status(path: Path | str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    """The ``TofStreamResult`` a subscriber gets (the connection is then closed)."""
    with _connect(path, timeout_s) as sock:
        sock.sendall(_request(TOF_STREAM))
        reader = sock.makefile("rb")
        line = reader.readline(65536)
    if not line:
        raise OSError("tofd closed the connection")
    response = json.loads(line)
    if response.get("error"):
        raise ValueError(f"tof.stream refused: {response['error']}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("tof.stream answered without a result")
    return result


def subscribe(
    path: Path | str, *, timeout_s: float = 2.0
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """Subscribe: the ``TofStreamResult`` and an iterator of ``TofFrame`` dicts (it ends when
    the daemon closes the stream; a read timeout raises ``TimeoutError``)."""
    sock = _connect(path, timeout_s)
    sock.sendall(_request(TOF_STREAM))
    reader = sock.makefile("rb")
    line = reader.readline(65536)
    if not line:
        sock.close()
        raise OSError("tofd closed the connection")
    result = json.loads(line).get("result") or {}

    def frames() -> Iterator[dict[str, Any]]:
        try:
            while True:
                raw = reader.readline(1 << 20)
                if not raw:
                    return
                message = json.loads(raw)
                if message.get("method") == TOF_FRAME and isinstance(message.get("params"), dict):
                    yield message["params"]
        finally:
            reader.close()
            sock.close()

    return result, frames()


# ── the guardian ────────────────────────────────────────────────────────────────


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def guard(parent: int, command: list[str]) -> int:
    """Run ``command``; stop it (SIGTERM, then SIGKILL) when ``parent`` is gone or this process
    is asked to stop. Returns the child's exit status."""
    child = subprocess.Popen(command)
    stopping = False

    def stop(signum: int, frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while child.poll() is None:
            if stopping or not _alive(parent) or os.getppid() != parent:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(GUARD_POLL_S)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
    return int(child.returncode or 0)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "--guard" or args[2] != "--":
        sys.stderr.write("usage: python -m duckmind.world.tofd_launch --guard PID -- TOFD ARGS…\n")
        raise SystemExit(2)
    raise SystemExit(guard(int(args[1]), args[3:]))


if __name__ == "__main__":
    main()
