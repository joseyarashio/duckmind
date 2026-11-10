"""Start this bench on another source tree: ``PYTHONPATH=<tree>/src python <here>/_boot.py gothere run …``.

``duckmind`` is then whatever is first on ``PYTHONPATH`` (the tree being measured, which may be older than the bench and
have no ``duckmind.bench`` at all), and ``duckmind.bench`` is this folder. Worker processes import this file again
(``multiprocessing`` runs the parent's main script first), so they get the same pair.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def graft() -> None:
    """Make ``duckmind.bench`` this folder, whatever tree ``duckmind`` comes from."""
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == _HERE:
        del sys.path[0]  # the script's own folder is not a package root
    import duckmind

    have = sys.modules.get("duckmind.bench")
    if have is not None and Path(getattr(have, "__file__", "") or "").resolve().parent == _HERE:
        return
    spec = importlib.util.spec_from_file_location(
        "duckmind.bench", _HERE / "__init__.py", submodule_search_locations=[str(_HERE)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["duckmind.bench"] = module
    spec.loader.exec_module(module)
    duckmind.bench = module  # type: ignore[attr-defined]


graft()

if __name__ == "__main__":
    from duckmind.bench.cli import main

    raise SystemExit(main())
