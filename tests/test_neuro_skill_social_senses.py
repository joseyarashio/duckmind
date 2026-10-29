"""a:social — the three MODEL readings (world side): a person's look, a voice's signature, the roof."""

from __future__ import annotations

import math
import random

import pytest

from duckmind.neuro.skill_imprint import look_distance
from duckmind.world import neuro_social_senses as NSS


def test_looks_come_only_in_light_and_the_shirt_only_from_afar() -> None:
    rng = random.Random(1)
    assert NSS.observe_look("child", "dark", 1.0, rng) is None
    assert NSS.observe_look("child", "black", 1.0, rng) is None
    close = [NSS.observe_look("owner", "bright", 0.8, rng) for _ in range(50)]
    far = [NSS.observe_look("owner", "bright", 1.8, rng) for _ in range(50)]
    close_looks = [c for c in close if c is not None]
    far_looks = [f for f in far if f is not None]
    assert len(close_looks) >= 40 and len(far_looks) >= 40  # P_LOOK bright 0.95
    assert all(math.isnan(c[4]) for c in close_looks)
    assert all(not math.isnan(f[4]) for f in far_looks)


def test_each_family_member_looks_like_themself() -> None:
    rng = random.Random(2)
    for who in NSS.LOOKS:
        looks = [NSS.observe_look(who, "bright", 1.6, rng) for _ in range(200)]
        dists = sorted(look_distance(lk, NSS.LOOKS[who]) for lk in looks if lk is not None)
        assert dists[len(dists) // 2] < 1.0  # the same person: within the join distance


def test_the_child_and_the_owner_are_told_apart_even_up_close() -> None:
    rng = random.Random(3)
    child = [NSS.observe_look("child", "bright", 0.8, rng) for _ in range(100)]
    near_owner = [
        look_distance(c, NSS.LOOKS["owner"]) < look_distance(c, NSS.LOOKS["child"])
        for c in child
        if c is not None
    ]
    assert sum(near_owner) <= 2  # short legs in blue jeans: rarely taken for the owner


def test_a_voice_signature_scatters_more_when_quiet() -> None:
    rng = random.Random(4)
    loud = [NSS.observe_voice("elder", 30.0, rng) for _ in range(200)]
    quiet = [NSS.observe_voice("elder", 6.0, rng) for _ in range(200)]
    assert NSS.observe_voice("elder", 1.0, rng) is None
    true = NSS.VOICES["elder"]

    def spread(vs: list) -> float:
        return sum(abs(v[0] - true[0]) for v in vs if v is not None) / len(vs)

    assert spread(quiet) > 2.0 * spread(loud)


def test_the_roof_reading_limits() -> None:
    rng = random.Random(5)
    assert NSS.read_roof(0.2, "dark", False, True, rng) is None  # too dark
    assert NSS.read_roof(0.2, "bright", False, False, rng) is None  # not looking up
    assert NSS.read_roof(None, "bright", False, True, rng) == NSS.ROOF_OPEN
    reading = NSS.read_roof(0.2, "bright", False, True, rng)
    assert reading is not None and reading == pytest.approx(0.2, abs=0.12)
    walking = [NSS.read_roof(0.2, "bright", True, True, rng) for _ in range(400)]
    share = sum(r is not None for r in walking) / len(walking)
    assert share == pytest.approx(NSS.P_SHARP_STEPPING, abs=0.08)  # smeared frames


def test_the_roof_truth_under_a_box_on_the_physics_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="3 3 0.1"/>
      <geom name="top" type="box" pos="0 0 0.42" size="0.7 0.4 0.02"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # a lens 0.2 m up looking ahead (+x), level, its picture upright: the sensor is turned — its
    # "right" points down (−z) and its "down" to the picture's left (+y) (neuro_world.camera)
    axes = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    under, up = NSS.roof_truth(model, data, (0.0, 0.0, 0.2), axes, groups=(1, 1, 1, 0, 0, 0))
    assert up and under == pytest.approx(0.2, abs=0.03)  # the table's underside, 0.2 m above
    outside, up2 = NSS.roof_truth(model, data, (2.0, 0.0, 0.2), axes, groups=(1, 1, 1, 0, 0, 0))
    assert up2 and outside is None
