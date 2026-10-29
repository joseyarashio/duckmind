"""a:play — the headless rig of the whole neural duck with the play acts (physics; skipped
without MuJoCo, onnxruntime or the official duck model)."""

from __future__ import annotations

import math
import os
from datetime import datetime

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind.world import neuro_play_rig as R  # noqa: E402

if not os.path.isdir(R.mjcf_dir()):
    pytest.skip("the official duck model is not on this machine", allow_module_level=True)


@pytest.fixture(scope="module")
def rig():  # type: ignore[no-untyped-def]
    from duckmind.neuro.genome import preset
    from duckmind.neuro.skill_play import play_installed

    with play_installed():
        genome = preset("dreamer")
    x, y = 0.3, -0.6
    yaw = math.atan2(-2.828 - y, -x)
    with R.PlayNeuro(genome, 4670, x=x, y=y, yaw=yaw, when=datetime(2026, 9, 28, 8, 30)) as r:
        r.run(4.0)
        yield r


def test_it_stands_where_it_was_put_and_keeps_standing(rig) -> None:  # type: ignore[no-untyped-def]
    assert rig.stood
    x, y, yaw = rig.pose()
    assert math.hypot(x - 0.3, y + 0.6) < 0.25
    assert not rig.fallen()


def test_it_reads_its_dock_tag_and_knows_the_spot_beside_it(rig) -> None:  # type: ignore[no-untyped-def]
    # facing its dock 2.3 m away: its first looks read the tag
    assert rig.brain.play.home_spot is not None
    spot = rig.drop_spot_world()
    assert spot is not None
    # the brain's spot (its odometry) and the truth agree to within its tag reading's error
    ox, oy, oyaw = rig.frame.body.odom
    hx, hy = rig.brain.play.home_spot
    bearing_b = math.atan2(hy - oy, hx - ox) - oyaw
    x, y, yaw = rig.pose()
    bearing_t = math.atan2(spot[1] - y, spot[0] - x) - yaw
    assert abs(math.atan2(math.sin(bearing_b - bearing_t), math.cos(bearing_b - bearing_t))) < 0.15


def test_the_jaw_and_the_light_meter_reach_the_brain(rig) -> None:  # type: ignore[no-untyped-def]
    assert getattr(rig.frame.body, "mouth", None) is not None
    lux = getattr(rig.frame.camera, "lux", None)
    assert lux is not None and lux > 0.0
    view = rig.tel.learned["play"]
    assert view["mouth_deg"] is not None and -10.0 < view["mouth_deg"] < 45.0
    sample = rig.part.sample(rig.duck.data)
    assert "sun" in sample and "lux_meter" in sample["cam"]


def test_the_sun_patch_is_on_the_floor_in_the_morning(rig) -> None:  # type: ignore[no-untyped-def]
    field = rig.sunfield.view()
    assert field["patches"], "a sunny September morning: the east window casts a patch"
    p = field["patches"][0]
    assert p["lux"] > 10_000 and p["area_m2"] > 0.2
