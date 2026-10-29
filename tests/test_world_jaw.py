"""The duck's provisional jaw (owner decision A4): a world extension that gives the simulated head
a hinged lower beak robotd's ``robot.mouth`` drives, without changing the head's mass."""

from __future__ import annotations

import math
import os

import pytest

from duckmind.world import jaw


def test_mouth_target_is_robotds() -> None:
    assert jaw.mouth_target(0.0) == pytest.approx(math.radians(-5.0))
    assert jaw.mouth_target(1.0) == pytest.approx(math.radians(30.0))
    assert jaw.mouth_target(7.0) == jaw.mouth_target(1.0)
    assert jaw.mouth_target(float("nan")) == jaw.mouth_target(0.0)


def test_in_beak_reads_the_held_gap() -> None:
    assert jaw.in_beak(math.radians(10.0))  # holding the fin: ~10° (live, phase 3)
    assert not jaw.in_beak(math.radians(-4.8))  # empty
    assert not jaw.in_beak(math.radians(10.0), commanded_open=1.0)


def _spec():  # type: ignore[no-untyped-def]
    mujoco = pytest.importorskip("mujoco")
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not os.path.isdir(folder):
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (the official MJCF folder) is not set")
    from duckmind.world import homelayout

    layout = homelayout.load("default_home", mjcf_dir=folder)
    return mujoco, homelayout.compose(layout, collision="allcollisions", mjcf_dir=folder)


def test_the_jaw_keeps_the_head_mass_and_adds_a_mouth_actuator() -> None:
    mujoco, spec = _spec()
    plain = spec.copy()
    for key in list(plain.keys):
        plain.delete(key)
    m0 = plain.compile()
    head0 = m0.body_mass[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, jaw.HEAD)]
    jaw.add_jaw(spec)
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    info = jaw.info(model)
    assert info["provisional"] is True
    assert info["head_g"] + info["mandible_g"] == pytest.approx(head0 * 1000.0, abs=0.01)
    assert info["actuator"] >= 0
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names.count("mouth") == 1 and model.nu == m0.nu + 1
    with pytest.raises(ValueError):
        jaw.add_jaw(spec)  # once only


def test_the_hinge_follows_the_mouth_target() -> None:
    mujoco, spec = _spec()
    jaw.add_jaw(spec)
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jaw.ACTUATOR)
    data.ctrl[act] = jaw.mouth_target(1.0)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert jaw.mouth_open_deg(model, data) > 30.0
    data.ctrl[act] = jaw.mouth_target(0.0)
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert jaw.mouth_open_deg(model, data) < 3.0
