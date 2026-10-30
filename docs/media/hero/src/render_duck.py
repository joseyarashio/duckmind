"""Render duck.png, the duck in the hero banners: the official Microduck 3D model as the
simulator page draws it, with a transparent background.

It needs the upstream microduck_rl checkout that ``duckmind setup`` downloads (its meshes are
Pollen Robotics' CC BY-NC-SA 4.0 files; they are read, never copied into this repository) and
Google Chrome. Run from the repository root, in the project's environment:

    DUCKMIND_MICRODUCK_MJCF=<microduck_rl>/src/mjlab_microduck/robot/microduck \\
        uv run python docs/media/hero/src/render_duck.py

A small web server on 127.0.0.1 (``--port``, 8782 by default) serves render-duck.html, the
simulator's robot description (the page's /api/robot) as /model.json, the meshes as /mesh/<name>
and the page's vendored three.js. Chrome opens the page twice: once to measure the duck's
outline, once, in a window of that shape, to take the picture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VENDOR = REPO / "src" / "duckmind" / "simweb" / "static" / "vendor"
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def chrome_binary() -> str:
    for candidate in (
        os.environ.get("CHROME"),
        MAC_CHROME,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("Google Chrome not found: set CHROME to its binary")


def serve(port: int) -> ThreadingHTTPServer:
    from duckmind.simweb import robot_model

    model = robot_model.load_model()
    if not model.get("available"):
        sys.exit(f"the robot model is not available: {model.get('reason')}")
    model_json = json.dumps(model).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            path = unquote(urlparse(self.path).path)
            body: bytes | None = None
            kind = "application/octet-stream"
            if path == "/render-duck.html":
                body, kind = (HERE / "render-duck.html").read_bytes(), "text/html; charset=utf-8"
            elif path == "/model.json":
                body, kind = model_json, "application/json"
            elif path.startswith("/mesh/"):
                mesh = robot_model.mesh_path(path[len("/mesh/") :])
                body = mesh.read_bytes() if mesh else None
            elif path.startswith("/vendor/"):
                file = (VENDOR / path[len("/vendor/") :]).resolve()
                if file.is_relative_to(VENDOR) and file.is_file():
                    body, kind = file.read_bytes(), "text/javascript; charset=utf-8"
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def chrome(url: str, size: tuple[int, int], extra: list[str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as profile:
        return subprocess.run(
            [
                chrome_binary(),
                "--headless=new",
                f"--user-data-dir={profile}",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--hide-scrollbars",
                "--default-background-color=00000000",
                "--virtual-time-budget=30000",
                f"--window-size={size[0]},{size[1]}",
                *extra,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "duck.png")
    parser.add_argument("--height", type=int, default=1100, help="picture height in pixels")
    parser.add_argument("--port", type=int, default=8782)
    parser.add_argument("--query", default="", help="extra page parameters, e.g. az=40&hy=0.3")
    args = parser.parse_args(argv)

    server = serve(args.port)
    try:
        url = f"http://127.0.0.1:{args.port}/render-duck.html?{args.query}"
        measured = chrome(url + "&measure=1", (800, 800), ["--dump-dom"])
        found = re.search(r'<pre id="aspect">([0-9.]+)</pre>', measured.stdout)
        if not found:
            sys.exit(f"the page did not finish:\n{measured.stderr[-2000:]}")
        aspect = float(found.group(1))
        # (headless Chrome's windows are at least 500 px wide)
        height = max(args.height, round(520 / aspect))
        width = round(height * aspect)
        shot = chrome(
            f"{url}&w={width}&h={height}", (width, height), [f"--screenshot={args.out.resolve()}"]
        )
        if not args.out.is_file():
            sys.exit(f"no picture:\n{shot.stderr[-2000:]}")
        print(f"{args.out}: {width}×{height}")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
