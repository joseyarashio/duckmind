"""Public play front (deploy/public/edge): its Node tests run inside the Python suite.

The front is TypeScript with two hosts: a Cloudflare Worker (src/index.ts) and the EC2 host
(src/host/, one Docker duck at a time). Their shared logic (the gate and its line, one visitor's
session and container lifecycle, the routes and pages including the guiding page, the proxy's path
filter, the Turnstile check, the six-language words) lives in plain modules under src/core; the host's
own parts (its gate/session wiring, the docker runner, the client-address rule) under src/host. All of
it is tested with Node's own test runner (deploy/public/edge/test/*.test.ts, fake docker, no packages,
no network). Node 22.18 or later runs TypeScript directly; older or missing Node skips.
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
    # The reporter is named: Node's default differs by version (22 writes TAP when the output is not a
    # terminal, 24 writes "spec"), and the summary lines below are TAP's.
    run = subprocess.run(
        [NODE, "--test", "--test-reporter=tap", *files],
        cwd=EDGE,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert re.search(r"^# fail 0$", run.stdout, re.M), run.stdout[-2000:]
    passed = re.search(r"^# pass (\d+)$", run.stdout, re.M)
    assert passed and int(passed.group(1)) > 0, run.stdout[-2000:]
