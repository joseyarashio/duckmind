"""Public play edge (deploy/public/edge): its Node tests run inside the Python suite.

The edge is a TypeScript Worker; its logic (the 10-slot gate and its line, one visitor's session and
container lifecycle, the proxy's path filter, the Turnstile check, the six-language words) lives in
plain modules under deploy/public/edge/src/core and is tested with Node's own test runner
(deploy/public/edge/test/*.test.ts, no packages, no network). Node 22.18 or later runs TypeScript
directly; older or missing Node skips.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

EDGE = Path(__file__).resolve().parents[1] / "deploy" / "public" / "edge"
NODE = shutil.which("node")


def _node_runs_typescript() -> bool:
    if NODE is None:
        return False
    out = subprocess.run([NODE, "--version"], capture_output=True, text=True, check=False).stdout
    m = re.match(r"v(\d+)\.(\d+)", out.strip())
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (22, 18)


pytestmark = pytest.mark.skipif(
    not _node_runs_typescript(), reason="needs Node 22.18+ (TypeScript)"
)


def test_edge_node_tests_pass() -> None:
    files = sorted(str(p.relative_to(EDGE)) for p in (EDGE / "test").glob("*.test.ts"))
    assert files, "no edge tests found"
    run = subprocess.run(
        [NODE, "--test", *files], cwd=EDGE, capture_output=True, text=True, timeout=120, check=False
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert re.search(r"^ℹ fail 0$", run.stdout, re.M), run.stdout[-2000:]
