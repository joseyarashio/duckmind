"""The growth integration (docs/design/GROWTH-G1-2026-10-10.md): the cases that build a real integrated
brain run in a child process.

``duckmind.neuro.integration.install_default_skills`` is intentionally permanent for the process,
and the legacy reversible skill fixtures expect the pristine tables (see ``test_live_isolated.py``,
``test_neuro_integration.py``). The restart cases (a real neural duck on a real life: teach, stop,
start again, check it is remembered) and the genome cases all make one, so they run together in one
fresh Python and this test says whether they all passed.

Run them directly while working on them::

    PYTHONPATH=src python -m pytest tests/life_isolated/case_*.py

They are not named ``test_*`` so the suite's own collection does not load them in this process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "tests" / "life_isolated").glob("case_*.py"))


def test_the_cases_that_need_a_real_brain_pass_in_their_own_process() -> None:
    assert [p.name for p in CASES] == [
        "case_bridge.py",
        "case_gate.py",
        "case_genome.py",
        "case_regrow.py",
        "case_restart.py",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
            "--color=no",
            *[str(p) for p in CASES],
        ],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    tail = (result.stdout or "")[-6000:] + (result.stderr or "")[-2000:]
    assert result.returncode == 0, tail
    assert " passed" in (result.stdout or ""), tail  # (and something ran)
    assert " failed" not in (result.stdout or "") and " error" not in (result.stdout or ""), tail
