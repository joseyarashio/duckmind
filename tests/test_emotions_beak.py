"""Bug 3 (2026-10-09): the lower beak touched the neck housing in defiant (37 ticks of 5 presses) and angry (61).

The head turned to the side while the beak was wide open. The opening is now capped by the yaw
(:func:`duckmind.neuro.emotions.mouth_cap`). The check here is kinematic on the composed duck model with the jaw
(``world/jaw.py``): every keyframe, interpolated at 50 Hz, is posed and the contacts are read the way
``bodyhost.ContactWatch`` reads them (the beak's two bodies against the head chain's other bodies). It needs MuJoCo
and the official MJCF; the live n = 5 press measurement is in docs/design/neural-duck/BUGS-2026-10-09.md."""

from __future__ import annotations

import os

import pytest

from duckmind.neuro import emotions as EO

HOME = (0.3491, 0.3491, 0.0, 0.0)  # robotd's v5 HOME for the head: robot.head takes offsets from it


def test_the_cap_shrinks_with_the_yaw_and_only_defiant_and_angry_are_reshaped() -> None:
    assert EO.mouth_cap(0.0) == 1.0 and EO.mouth_cap(0.03) == 1.0
    assert EO.mouth_cap(0.6) == EO.mouth_cap(-0.6) < EO.mouth_cap(0.3) < 1.0
    raw = EO.load()["emotions"]
    cat = EO.catalog()
    for name in EO.ORDER:
        mine = [row[6] for row in cat[name].rows]
        if name in EO.BEAK_CAPPED:
            assert all(r[6] <= EO.mouth_cap(r[3]) + 1e-9 for r in cat[name].rows)
        else:
            lead = float(raw[name]["sound_at_s"])
            theirs = [row[6] for row in raw[name]["keyframes"] if row[0] - lead >= -1e-9]
            assert mine == pytest.approx(theirs)  # (the other twelve are exactly the workshop's)


def test_the_capped_two_still_open_the_beak_wide_enough_to_read_as_a_bark() -> None:
    cat = EO.catalog()
    assert max(r[6] for r in cat["defiant"].rows) >= 0.6
    assert max(r[6] for r in cat["angry"].rows) >= 0.6
    # the straight-ahead bark at the end of angry keeps its full opening
    assert max(r[6] for r in cat["angry"].rows if abs(r[3]) < 0.1) == 1.0


def _model():  # type: ignore[no-untyped-def]
    mujoco = pytest.importorskip("mujoco")
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not os.path.isdir(folder):
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (the official MJCF folder) is not set")
    from duckmind.world import homelayout, jaw

    layout = homelayout.load("default_home", mjcf_dir=folder)
    spec = homelayout.compose(layout, collision="allcollisions", mjcf_dir=folder)
    jaw.add_jaw(spec)
    for key in list(spec.keys):
        spec.delete(key)
    return mujoco, spec.compile(), jaw


def _touching_ticks(rows) -> int:  # type: ignore[no-untyped-def]
    import numpy as np

    mujoco, model, jaw = _model()
    data = mujoco.MjData(model)

    def body(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    def adr(name: str) -> int:
        return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    head = {body(n) for n in ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "mandible")}
    beak = {body(n) for n in ("jaw_soft", "mandible")}
    joints = [adr(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
    mouth = adr("mouth")
    table = np.array(rows, dtype=float)
    touching = 0
    for t in np.arange(table[0, 0], table[-1, 0] + 1e-9, 0.02):
        v = [float(np.interp(t, table[:, 0], table[:, k])) for k in range(1, 7)]
        mujoco.mj_resetData(model, data)
        for j, home, off in zip(joints, HOME, v[:4], strict=True):
            data.qpos[j] = home + off
        data.qpos[mouth] = jaw.mouth_target(v[5])
        mujoco.mj_forward(model, data)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (a in beak and b in head and b not in beak) or (
                b in beak and a in head and a not in beak
            ):
                touching += 1
                break
    return touching


@pytest.mark.parametrize("name", sorted(EO.BEAK_CAPPED))
def test_the_beak_never_touches_the_neck_housing_in_defiant_and_angry(name: str) -> None:
    assert _touching_ticks(EO.catalog()[name].rows) == 0


@pytest.mark.parametrize("name", sorted(EO.BEAK_CAPPED))
def test_the_workshops_own_keyframes_did_touch_so_the_check_can_fail(name: str) -> None:
    raw = EO.load()["emotions"][name]["keyframes"]
    assert _touching_ticks(raw) > 5
