"""The walk to a place somebody pointed at (GOTHERE-V8-2026-11-06), in the headless rig: real_home, robotd's quick turn
installed as on the live duck, the place on open floor well off the duck's heading. It turns on the spot to face the place and
walks at it (before: an arc that carried it away while it turned, the errand ending 'stuck' or still under way at 40 s).
Physics; skipped without MuJoCo, onnxruntime, the official duck model or the quick-turn table."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind import paths  # noqa: E402
from duckmind.world import neuro_play_rig as R  # noqa: E402

SKILLS = paths.data_dir() / "live" / "motions" / "skills.json"
if not os.path.isdir(R.mjcf_dir()):
    pytest.skip("the official duck model is not on this machine", allow_module_level=True)
if not SKILLS.is_file():
    pytest.skip("robotd's quick-turn table is not on this machine", allow_module_level=True)

# (truth pose x, y, yaw; the place) — fair-scene clicks from the live go-there rows (gb-V5/V8, 2026-11-06) with a clear straight way
CLICKS = [
    ((1.684, -1.026, 2.455), (2.79, -0.17)),  # 1.4 m, 1.75 rad to the right
    ((1.3, -1.223, 0.297), (1.39, -2.62)),  # 1.4 m, 1.79 rad to the right
]


# (camera-dock 2026-11-09) the clicks came from the live duck with the alpha IMX219. On the production GC2093 (the default)
# the second did not arrive within 25 s: the camera's five floor-line sectors are ~0.42 rad wide, and each end was laid
# across its whole sector at the nearest range — a wall 5 cm beside the way where the floor was clear (49 such points on
# this walk, the alpha's 4) — and the walk turned and backed about them. An end is laid where the look saw it now
# (FreeSector.end_bearing); the test runs on the simulator's default camera
@pytest.mark.parametrize(("pose", "place"), CLICKS)
def test_a_place_well_off_the_heading_is_faced_on_the_spot_and_reached(
    pose, place, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from duckmind.neuro import lastmap
    from duckmind.neuro.genome import Genome
    from duckmind.skilltrain import guard as SG
    from duckmind.skilltrain import table as TB

    monkeypatch.setattr(lastmap, "LASTMAP_MODE", "soft")  # (the live duck's setting)
    seed = 9201
    with R.PlayNeuro(
        Genome.from_seed(seed, gd=50),
        seed,
        x=pose[0],
        y=pose[1],
        yaw=pose[2],
        home="real_home",
        when=datetime(2026, 10, 24, 10, 0),
    ) as rig:
        rig.run(3.0)
        doc = json.loads(SKILLS.read_text())
        rig.duck.install_table(
            [TB.SkillDef.from_json(s) for s in doc["skills"] if "quick_turn" in str(s.get("name"))]
        )
        rig.link.link.skills = SG.SkillGuard(("quick_turn_left", "quick_turn_right"))
        rig.brain.can_pivot = True
        x, y, yaw = rig.pose()
        bearing = math.atan2(place[1] - y, place[0] - x) - yaw
        rig.brain.go_to(
            math.atan2(math.sin(bearing), math.cos(bearing)), math.hypot(place[0] - x, place[1] - y)
        )
        t = 0.0
        while t < 25.0 and rig.brain.errand.view().get("state") not in ("arrived", "gave_up"):
            rig.run(0.5)
            t += 0.5
            assert not rig.duck.safety.fallen
        view = rig.brain.errand.view()
        x, y, _ = rig.pose()
        assert view.get("state") == "arrived", (t, view)
        assert math.hypot(place[0] - x, place[1] - y) <= 0.25
