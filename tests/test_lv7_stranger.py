"""Bug 4 (2026-10-09): LV7's target 2 (the shy snuggler answers a stranger at most 0.3 of the time) was 0.312
(seeds 13-20, n = 32) with STRANGER_RESERVE 0.85; with 0.6 it is 0.188 (and 0.167 on seeds 1-12, 0.219 on 21-28).
The family's replies are not touched by the reserve (it multiplies only a person the family has not named).

The conversation days run in a child process: building the integrated brain installs skills and leaves global state
behind that other test files (test_neuro_claim, test_neuro_finish, …) are order-sensitive to."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from duckmind.live import bridge as B
from tests.checkout_parts import needs_design

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import json
from pathlib import Path
from duckmind.live import convday as D
from duckmind.live.bridge import BridgeConfig
from duckmind.live.drives import DriveParams

params = DriveParams.from_dict(json.loads(Path("docs/design/live-voice/lv4/drive-params.json").read_text())["params"])
cfg = BridgeConfig(duck_name=D.NAME, drives=params)
guest = [0, 0]
family = [0, 0]
for seed in range(13, 21):  # the held-out seeds LV7's check was run on, 2026-10-05
    for kind in ("day", "night"):
        run = D.run_day("shy_snuggler", seed, params, kind=kind, bridge_config=cfg).to_dict()
        for p in run["proposals"]:
            if not p["available"]:
                continue
            t = guest if p["who"] == "guest" else family
            t[0] += 1
            t[1] += bool(p["voiced"])
print(json.dumps({"guest": guest, "family": family}))
"""


def test_the_stranger_reserve_is_the_one_the_frozen_target_needs() -> None:
    assert B.STRANGER_RESERVE == 0.6


@needs_design
def test_a_shy_duck_answers_a_stranger_at_most_three_times_in_ten() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    (g_n, g_v), (f_n, f_v) = (
        (out["guest"][0], out["guest"][1]),
        (out["family"][0], out["family"][1]),
    )
    assert g_n >= 24
    assert g_v / g_n <= 0.3, (g_v, g_n)  # LV4's target 2 (frozen)
    assert f_v / f_n >= 0.95, out["family"]  # (the family is still answered)
