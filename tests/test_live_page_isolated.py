"""Live voice, LV5: the talk panel's cases that build a real integrated brain run in a child process.

Same arrangement as ``tests/test_live_isolated.py``: ``install_default_skills`` is permanent for the
process and the legacy skill tests expect the pristine tables, so the real-brain cases of
``tests/live_page_isolated`` run together in one fresh Python and this test says whether they passed.
Run them directly while working on them::

    PYTHONPATH=src python -m pytest tests/live_page_isolated/case_*.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "tests" / "live_page_isolated").glob("case_*.py"))


def test_the_page_cases_that_need_a_real_brain_pass_in_their_own_process() -> None:
    assert [p.name for p in CASES] == ["case_page_chain.py"]
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
    assert " passed" in (result.stdout or ""), tail
    assert " failed" not in (result.stdout or "") and " error" not in (result.stdout or ""), tail
