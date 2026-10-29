"""The child of tests/test_life_stop_signal.py: a life server's serving loop (the real
``serve_until_stopped``, the real SIGTERM handler, the real ``LifeHost.stop``) without the body.
Prints ``READY <pid> <port>`` and serves until it is stopped."""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_life_bridge_host import BIRTH, Life, Wall  # noqa: E402
from test_life_snapshot import taught  # noqa: E402

from duckmind.simweb.server import serve_until_stopped  # noqa: E402


def main() -> None:
    folder = Path(sys.argv[1])
    life = Life(folder, Wall(BIRTH), backdate=14)
    brain = taught()
    life.host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True),
        lanes=SimpleNamespace(brain=brain),
        _tick_lock=threading.RLock(),
        start_at="mat",
        live_voice=None,
    )
    life.host.every_s = 3600.0
    life.host.start()  # (the hatch snapshot is on disk from the first moment)
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    print(f"READY {__import__('os').getpid()} {server.server_address[1]}", flush=True)
    serve_until_stopped(server, life.state, life.record, life.host)
    server.server_close()
    print("STOPPED", flush=True)


if __name__ == "__main__":
    main()
