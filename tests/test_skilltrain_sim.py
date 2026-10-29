"""The Mac gate harness end to end (:mod:`duckmind.skilltrain.programs` on
:class:`~duckmind.skilltrain.skillduck.SkillDuck`): the v5 walk network run as a skill.

One standing trial per servo model: the dummy skill starts one tick after the request, holds
0.5 s and unwinds 0.8 s exactly as robotd would, hands back without a fall, and — v5 standing
cannot crouch — is *not* performed as a startle. A chain + unwind entry re-requested for 2 s
still ends at 1.3 s. Needs numpy, MuJoCo 3.10, onnxruntime, the official MJCF
(``DUCKMIND_MICRODUCK_MJCF``) and the shipped policies; the bam case also needs bam (git
62bd8ce). Skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind.skilltrain import programs as P  # noqa: E402
from duckmind.skilltrain import table as T  # noqa: E402
from duckmind.world import headless_duck as hd  # noqa: E402

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
if not MJCF_DIR or not (Path(MJCF_DIR) / "scene_allcollisions.xml").is_file():
    pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)", allow_module_level=True)
POLICIES = hd.policy_dir()
if not (POLICIES / "velstand.onnx").is_file():
    pytest.skip(f"needs the shipped policies in {POLICIES}", allow_module_level=True)

SERVOS = ["mjcf"]
try:  # the bam case runs where the training-branch bam is installed
    import bam.model  # noqa: F401

    SERVOS.append("bam")
except ImportError:
    pass


@pytest.fixture(scope="module")
def skills() -> list[T.SkillDef]:
    return [e.skill for e in T.dummy(str(POLICIES / "velstand.onnx"))]


@pytest.mark.parametrize("servo", SERVOS)
def test_the_dummy_skill_runs_on_robotds_skill_path(servo: str, skills) -> None:
    cond = P.Condition("skill/dummy_stand/stand", "skill", "dummy", "stand", skill="dummy_stand")
    row = P.run_trial(cond, servo=servo, rep=0, skills=skills, policies=str(POLICIES))
    assert row["excluded"] is None, row["excluded"]
    assert row["sequence"] == ["dummy_stand", "dummy_stand:unwind"]
    m = row["markers"]
    assert m["latency_from_request_s"] == pytest.approx(0.02, abs=1e-6)  # the next tick
    assert m["t_hold_end"] - m["t_start"] == pytest.approx(0.5, abs=0.021)
    assert m["t_end"] - m["t_start"] == pytest.approx(1.3, abs=0.021)
    assert not row["metrics"]["fell"]
    assert not row["visibility"]["performed"]  # v5 standing does not crouch
    assert row["metrics"]["dip_030_mm"] < 5.0


@pytest.mark.parametrize("servo", SERVOS[:1])
def test_a_chain_with_an_unwind_ignores_its_rerequests_in_the_simulator(servo: str, skills) -> None:
    cond = P.semantics_conditions(n=1)[0]
    assert cond.skill == "dummy_chain_unwind"
    row = P.run_trial(cond, servo=servo, rep=0, skills=skills, policies=str(POLICIES))
    assert row["excluded"] is None
    assert row["metrics"]["skill_s"] == pytest.approx(1.3, abs=0.021)
    assert row["refreshed"] >= 10
    assert all(e["kind"] != "chain_restart" for e in row["events"])
