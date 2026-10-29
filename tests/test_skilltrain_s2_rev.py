"""Stage-2 reward revisions (s2:train, 2026-09-29): p1 stays the default; p2 (opt-in) widens the crouch
and standing-head Gaussians and doubles the stepping cost; the wider terms are ≥ the fine ones where
the policy stands far from the command and equal them on the target.

    PYTHONPATH=src .venv-official-cpu/bin/python -m pytest tests/test_skilltrain_s2_rev.py
"""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("mjlab_microduck")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from duckmind.skilltrain import s2_tasks as T  # noqa: E402


def test_p1_is_the_default_and_unchanged():
    assert T.REWARD_REV == "p1"
    assert T.terms_for("pose") == dict(T.R3_TERMS)
    gaze = dict(T.R3_TERMS)
    gaze.update(T.R4_TERMS)
    assert T.terms_for("gaze", "p1") == gaze


def test_p2_changes_only_what_it_says():
    p1, p2 = T.terms_for("pose", "p1"), T.terms_for("pose", "p2")
    assert set(p1) == set(p2)
    changed = {n for n in p1 if p1[n] != p2[n]}
    assert changed == {"pose_track", "head_fine", "steps_standing"}
    assert p2["pose_track"][3] == {"z_std_coarse": 0.025}
    assert p2["head_fine"][3] == {"std_coarse": 0.35}
    assert p2["steps_standing"][1] == -2.0 and p2["steps_standing"][2] == [(0, -2.0)]
    assert p2["pose_track"][:3] == p1["pose_track"][:3]
    with pytest.raises(ValueError):
        T.terms_for("pose", "p9")


def test_env_cfg_carries_the_revision():
    cfg1 = T.make_env_cfg("pose")
    cfg2 = T.make_env_cfg("pose", reward_rev="p2")
    assert cfg1.rewards["pose_track"].params == {}
    assert cfg2.rewards["pose_track"].params == {"z_std_coarse": 0.025}
    assert cfg2.rewards["head_fine"].params == {"std_coarse": 0.35}
    assert cfg2.rewards["steps_standing"].weight == -2.0
    assert cfg1.rewards["steps_standing"].weight == -1.0
    # R4 from an R3-p2 start keeps p2's R3 terms at their final weights
    g2 = T.make_env_cfg("gaze", reward_rev="p2")
    assert g2.rewards["steps_standing"].weight == -2.0 and "cam_smear" in g2.rewards


def test_coarse_gaussian_gives_a_gradient_where_the_fine_one_is_flat():
    # the arithmetic behind p2: an untaken −20 mm crouch, an untaken 0.3 rad pull-back
    fine_z, coarse_z = math.exp(-((0.020 / 0.010) ** 2)), math.exp(-((0.020 / 0.025) ** 2))
    assert fine_z < 0.02 and coarse_z > 0.5
    d_fine = 2 * 0.020 / 0.010**2 * fine_z  # |d/dz| of the Gaussian at the error
    d_mix = 0.5 * (d_fine + 2 * 0.020 / 0.025**2 * coarse_z)
    assert d_mix > 2.5 * d_fine  # 2.8× the gradient, and the reward itself is > 0.3 instead of 0.02
    fine_h, coarse_h = math.exp(-((0.3 / 0.15) ** 2)), math.exp(-((0.3 / 0.35) ** 2))
    assert fine_h < 0.02 and coarse_h > 0.45
