"""b:feasibility (2026-09-28): the B-skill table and targets as robotd will take them."""

from __future__ import annotations

import math

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import table as T


def _defs() -> list[T.SkillDef]:
    return [T.SkillDef(**d) for d in B.skill_defs()]


def test_entries_are_robotd_valid() -> None:
    found = T.problems(_defs(), check_files=False)
    assert found == [], found


def test_held_entries_chain_without_unwind_and_one_shots_unwind() -> None:
    for e in B.ENTRIES:
        if e.behaviour in ("turn", "march", "lie", "release"):
            assert e.chain and e.unwind_s == 0.0, e.name
        else:
            assert not e.chain and e.unwind_s > 0.0, e.name


def test_codes_are_unique_within_each_network() -> None:
    for net in B.NETS:
        codes = [e.command for e in B.entries(net)]
        assert len(set(codes)) == len(codes), net


def test_pinned_scale_and_gain() -> None:
    for d in _defs():
        assert d.action_scale == B.ACTION_SCALE and d.gain_ratio == B.GAIN_RATIO


def test_the_b_table_fits_robotd_with_stage1b_and_hop() -> None:
    """robotd takes 30 skills: v5's 3 + Stage 1b's live 20 + B's minimum 6 = 29; the hop's plan (4) does
    not fit next to all of them — the budget the orchestrator must settle is recorded, not hidden."""
    stage1b = T.stage1_live("/a/startle.onnx", "/a/comfort.onnx", "/a/velstand.onnx")
    assert len(T.resolve(stage1b)) == 23
    b_min = [d for d in _defs() if d.name in B.TABLE_BUDGET["b_minimum"]]
    assert len(T.resolve([*stage1b, *b_min])) == 29
    assert B.TABLE_BUDGET["free"] == T.MAX_SKILLS - 23


def test_postures_are_inside_the_joint_ranges() -> None:
    rng = {
        2: (-1.571, 1.571),
        3: (-1.571, 1.571),
        4: (-1.571, 1.571),
        5: (-1.571, 1.047),
        6: (-1.571, 1.571),
    }
    for post in (B.BOW, B.LIE):
        q = post.pose()
        for i, (lo, hi) in rng.items():
            assert lo < q[i] < hi, (post, i, q[i])
        assert q[11] == -q[2] and q[12] == -q[3] and q[13] == -q[4]
    assert B.BOW.pitch_deg < B.LIE.pitch_deg and B.BOW.drop_m < B.LIE.drop_m


def test_episode_lengths_cover_the_windows() -> None:
    assert B.episode_s("bskstep") >= B.HOLD_RANGE_S["march"][1]
    assert B.episode_s("bskbow") >= B.HOLD_RANGE_S["lie"][1] + 0.8
    assert B.episode_s("bskmat") >= 2.5 * B.HOLD_JITTER[1] + 0.5


def test_wilson() -> None:
    lo, hi = B.wilson(0, 30)
    assert lo == 0.0 and math.isclose(hi, 0.1135, abs_tol=2e-3)
    lo, hi = B.wilson(30, 30)
    assert hi == 1.0 and math.isclose(lo, 0.8865, abs_tol=2e-3)


def test_entry_mixes_are_distributions() -> None:
    for beh in B.BEHAVIOURS:
        mix = B.entry_mix(beh)
        assert math.isclose(sum(mix.values()), 1.0)
        assert set(mix) <= set(B.ENTRY_KINDS)
