"""Live voice, LV2: the cases that build a real integrated brain run in a child process.

``duckmind.neuro.integration.install_default_skills`` is intentionally permanent for the process
(the default registry of channels, features and programs is never unloaded), and the legacy
reversible skill fixtures (``skill_sound.install`` / ``uninstall``, ``skill_play.play_installed``,
``skill_social_brain.registered``) expect the pristine tables: a test in this pytest process that
makes an ``IntegratedBrain`` would change what those mean (see the same arrangement in
``test_neuro_integration.py``). The bridge, wiring and replay cases of ``tests/live_isolated`` all
make one, so they run together in one fresh Python and this test says whether they all passed.

Run them directly while working on them::

    PYTHONPATH=src python -m pytest tests/live_isolated/case_*.py

They are not named ``test_*`` so the suite's own collection does not load them in this process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "tests" / "live_isolated").glob("case_*.py"))


def test_the_cases_that_need_a_real_brain_pass_in_their_own_process() -> None:
    assert [p.name for p in CASES] == [
        "case_bridge.py",
        "case_command_api.py",
        "case_command_brain.py",
        "case_command_seek.py",
        "case_convday.py",
        "case_drives.py",
        "case_family_speaker.py",
        "case_replay_chain.py",
        "case_wiring.py",
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
