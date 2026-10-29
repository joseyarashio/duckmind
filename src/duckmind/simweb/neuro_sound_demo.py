"""a:sound's demo server: the simulator with the neural duck AND its sound skills (build phase).

Until the integration writes the sound skills into the shared files, this runs the unchanged
simulator (:func:`duckmind.simweb.server.main`, the same flags) with a:sound composed in at
runtime (:mod:`duckmind.simweb.neuro_sound_live`): the skills in every duck's brain, robotd's
streamed voice, the world extension (music with a tempo, the hovering hand, the ride), the
routes ``/api/neuro/sound/…`` and the page's strip (``neuro_sound.js`` / ``.css`` added to the
page as it is served). Nothing on disk is edited.

    python -m duckmind.simweb.neuro_sound_demo --port 8962 [server.py's flags …]
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path
from typing import Any

INJECT = (
    '<link rel="stylesheet" href="/neuro_sound.css">\n'
    '<script type="module" src="/neuro_sound.js"></script>\n'
)


def _microduck(argv: list[str]) -> Path | None:
    for i, arg in enumerate(argv):
        if arg == "--microduck" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--microduck="):
            return Path(arg.split("=", 1)[1])
    return None


def page() -> bytes:
    """The simulator's page with a:sound's strip added (before ``</body>``)."""
    html = resources.files("duckmind.simweb").joinpath("static", "index.html").read_text()
    marker = "</body>"
    at = html.rfind(marker)
    if at < 0:
        return (html + INJECT).encode()
    return (html[:at] + INJECT + html[at:]).encode()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # SIGTERM stops it like Ctrl+C: the bridge then stops robotd and the world server with it
    import signal

    def _term(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _term)
    from duckmind.simweb import neuro_sound_live as NSL
    from duckmind.simweb import server as SV

    NSL.patch(microduck=_microduck(argv))
    base_make = SV.make_handler

    def make_handler(state: Any, *args: Any, **kwargs: Any) -> Any:
        Base = base_make(state, *args, **kwargs)  # (the hub, the voice bank: as server.py passes)
        hub = NSL.SoundHub(lambda: getattr(getattr(state, "live", None), "neuro", None))

        class Handler(Base):  # type: ignore[misc, valid-type]
            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                path, _, query = self.path.partition("?")
                if hub.handles(path):
                    hub.get(self, path, query)
                    return
                if path in ("/", "/index.html"):
                    self._send(200, page(), "text/html; charset=utf-8")
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                path = self.path.partition("?")[0]
                if hub.handles(path):
                    hub.post(self, path)
                    return
                super().do_POST()

        return Handler

    SV.make_handler = make_handler  # type: ignore[assignment]
    SV.main(argv)


if __name__ == "__main__":
    main()
