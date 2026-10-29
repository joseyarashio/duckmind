"""Stage-1 training contract (tr:tasks): the numbers the two skill networks are trained on, and
their agreement with robotd's skill table (``duckmind.skilltrain.table``, the gate lane's).

Pure: standard library + numpy (the evaluator's per-duck outcome logic runs on synthetic traces).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.skilltrain import rl_contract as C
from duckmind.skilltrain import table as T

STARTLE = (
    "startle_1",
    "startle_2",
    "startle_3",
    "startle_4",
    "freeze_3",
    "freeze_4",
    "hop_back",
    "startle_release",
)
COMFORT = ("shake_1", "shake_2", "stretch_tall", "stretch_sit")


def test_variants_are_the_tables_entries_for_our_two_files() -> None:
    assert tuple(v.name for v in C.variants("startle")) == STARTLE
    assert tuple(v.name for v in C.variants("comfort")) == COMFORT
    entries = {e.name: e for e in C.stage1_entries()}
    for net in C.NETS:
        for v in C.variants(net):
            s = entries[v.name].skill
            # s1b:retrain: the only training window that differs from the table is the declared one (the 1.5-s
            # shake-off, run through table.stage1_live(shake_s=1.5) once it gates)
            hold = C.TRAIN_HOLD_S.get(entries[v.name].family, s.duration)
            assert (v.command, v.unwind, v.hold_s, v.unwind_s, v.chain) == (
                tuple(s.command),
                tuple(s.unwind),
                hold,
                s.unwind_s,
                s.chain,
            )
            assert s.path == C.PLACEHOLDER[net]
    assert C.TRAIN_HOLD_S == {"shake": 1.5}
    assert {v.name: v.hold_s for v in C.variants("comfort")}["shake_2"] == 1.5
    # every entry of the table runs one of our files, v5's walk (the abort) — nothing else
    others = [e for e in entries.values() if e.skill.path not in (*C.PLACEHOLDER.values(),)]
    assert not others


def test_scale_and_gain_are_pinned_and_what_training_runs() -> None:
    for net in C.NETS:
        assert C.pinned(net) == (1.0, 1.0)
        assert all(v.action_scale == 1.0 and v.gain_ratio == 1.0 for v in C.variants(net))
    # robotd's own refusals and meaning traps: none for our entries
    ours = [
        e
        for e in C.stage1_entries()
        if e.skill.path in (C.PLACEHOLDER["startle"], C.PLACEHOLDER["comfort"])
    ]
    assert T.problems(ours, check_files=False) == []


def test_chain_is_only_where_robotd_honours_it() -> None:
    for net in C.NETS:
        for v in C.variants(net):
            assert not (v.chain and v.unwind_s > 0), v.name
    rel = C.release_variant()
    assert rel.chain and rel.unwind_s == 0 and rel.command == (0.0, 0.0, 0.0)


def test_behaviours_and_postures() -> None:
    beh = {v.name: v.behaviour for n in C.NETS for v in C.variants(n)}
    assert beh["hop_back"] == "hop" and beh["startle_release"] == "rest"
    assert beh["stretch_sit"] == "stretch_sit" and beh["shake_2"] == "shake"
    assert {beh[f"startle_{i}"] for i in range(1, 5)} == {"crouch"}
    assert C.entry_mix("stretch_sit", ("sit",)) == {"seated": 1.0}
    mix = C.entry_mix("crouch", ("stand", "walk"))
    assert math.isclose(sum(mix.values()), 1.0) and mix["walk"] == pytest.approx(0.35)
    assert "seated" not in mix
    assert set(C.entry_mix("shake", ("stand",))) == {"stand", "head", "risen"}
    early = C.entry_mix("crouch", ("stand", "walk"), early=True)
    assert early["walk"] < mix["walk"]


def test_variant_weights() -> None:
    for net in C.NETS:
        w = C.variant_weights(net)
        assert math.isclose(sum(w.values()), 1.0)
    w = C.variant_weights("startle")
    assert w["hop_back"] == pytest.approx(0.25)
    assert w["startle_release"] == pytest.approx(C.REST_ONLY_WEIGHT)
    assert C.max_episode_s("startle") > 3.0 * C.HOLD_JITTER[1]


def test_crouch_targets_clear_their_gates() -> None:
    prev = None
    for k in (0.25, 0.5, 0.75, 1.0):
        t = C.crouch_target(k)
        dip_gate = C.gate_interp(C.GATES["crouch"]["dip_mm"], k)
        retract_gate = C.gate_interp(C.GATES["crouch"]["retract_rad"], k)
        assert t.dip_m * 1000 >= dip_gate + 2.5, k  # a margin over the visibility gate
        assert t.neck_rad >= retract_gate - 1e-9, k
        assert t.head_rad == pytest.approx(-t.neck_rad * 2 / 3)
        assert 0.15 <= t.ramp_s <= 0.25
        if prev is not None:
            assert t.dip_m > prev.dip_m and t.neck_rad > prev.neck_rad
        prev = t
        pose = t.pose()
        # the brain's neck pull-back direction (neck +, head −) and the legs bent
        assert (
            pose[C.NECK_PITCH] > C.HOME[C.NECK_PITCH] and pose[C.HEAD_PITCH] < C.HOME[C.HEAD_PITCH]
        )
        assert pose[3] > C.HOME[3] and pose[12] < C.HOME[12]  # knees fold (mirrored signs)


def test_kinematic_tables_are_mirrored_and_monotone() -> None:
    for table in (C.HOME_DIP_TABLE, C.SIT_RISE_TABLE):
        xs = [row[0] for row in table]
        assert xs == sorted(xs)
    zero = C.home_dip_pose(0.0)
    assert zero == pytest.approx(C.HOME)
    q = C.home_dip_pose(0.02)
    for li, ri in zip(C.LEFT_PITCH_CHAIN, C.RIGHT_PITCH_CHAIN, strict=True):
        assert q[li] - C.HOME[li] == pytest.approx(-(q[ri] - C.HOME[ri]))
    knees = [row[1][1] for row in C.HOME_DIP_TABLE]
    assert knees == sorted(knees)  # deeper dip, more knee
    # flat foot: hip − knee + ankle ≈ 0 in the leg plane (the knee axis is flipped)
    for _, (h, k, a) in C.HOME_DIP_TABLE:
        assert abs(h - k + a) < 0.01
    assert C.sit_rise_pose(0.0) == pytest.approx(C.sit_pose())


def test_stretch_and_shake_targets() -> None:
    tall, sit = C.STRETCH_TALL, C.STRETCH_SIT
    assert tall.rise_m * 1000 >= C.GATES["stretch_tall"]["rise_mm"]
    assert tall.rise_m <= C.HOME_REACH_M + 1e-9  # the legs cannot lift the trunk higher
    assert sit.rise_m * 1000 >= C.GATES["stretch_sit"]["rise_mm"]
    for t in (tall, sit):
        # camera pitch at a level trunk = Δneck − Δhead (radians)
        assert math.degrees(t.neck_rad - t.head_rad) >= C.GATES["stretch_tall"]["camera_deg"] + 5
        assert t.camera_deg >= C.GATES["stretch_tall"]["camera_deg"] + 3
    for a in (0.5, 1.0):
        s = C.shake_target(a)
        assert s.roll_deg[0] >= C.GATES["shake"]["roll_deg"] and s.roll_deg[0] < s.roll_deg[1]
        assert s.head_yaw_rad[0] >= C.GATES["shake"]["head_yaw_rad"]
        assert s.roll_hz == C.GATES["shake"]["roll_hz"]
    assert C.shake_target(1.0).roll_deg[0] > C.shake_target(0.5).roll_deg[0]


def test_wilson_and_rates() -> None:
    lo, hi = C.wilson(0, 512)
    assert lo == 0.0 and 0.0070 < hi < 0.0076  # the A100 run's 0/512 → [0, 0.75 %]
    assert C.rate(3, 512)[:2] == [3, 512]
    lo, hi = C.wilson(10, 17407)
    assert 0.0003 < lo < 0.0004 and 0.0010 < hi < 0.0011
    assert C.wilson(0, 0) == (0.0, 1.0)


def test_toml_section_carries_the_pinned_params(tmp_path) -> None:
    onnx = tmp_path / "startle.onnx"
    onnx.write_bytes(b"not a real network")
    entries = [e for e in C.stage1_entries() if e.skill.path == C.PLACEHOLDER["startle"]]
    defs = [T.SkillDef(**{**e.skill.__dict__, "path": str(onnx)}) for e in entries]
    text = T.toml_section(defs)
    assert text.count("[[policy.skill]]") == len(STARTLE)
    assert text.count("params = { action_scale = 1.0, gain_ratio = 1.0 }") == len(STARTLE)
    assert 'name = "startle_release"' in text and "chain = true" in text


# ── the evaluator's per-duck outcome on synthetic traces (numpy only) ────────────────────────


def _trace(ticks: int) -> dict[str, np.ndarray]:
    from duckmind.skilltrain.rl_eval import TRACE

    tr = {k: np.zeros((ticks, 1)) for k in TRACE}
    tr["z"][:] = C.STAND_Z
    tr["neck"][:] = C.HOME[C.NECK_PITCH]
    tr["feet"][:] = 2
    tr["tau_leg95"][:] = 0.1
    tr["tau_head95"][:] = 0.05
    tr["jacc99"][:] = 50.0
    return tr


def test_eval_outcome_detects_a_crouch_and_its_absence() -> None:
    from duckmind.skilltrain.rl_eval import duck_outcome

    v = next(x for x in C.variants("startle") if x.name == "startle_4")
    hold, unwind = 25, 40
    ticks = hold + unwind
    tr = _trace(ticks)
    t = np.arange(ticks) * C.DT
    ramp = np.clip(t / 0.2, 0.0, 1.0)
    tr["z"][:, 0] = C.STAND_Z - 0.025 * ramp
    tr["neck"][:, 0] = C.HOME[C.NECK_PITCH] + 0.33 * ramp
    good = duck_outcome("startle", v, "stand", 1.0, 0, tr, hold, unwind, 0, 0.0)
    assert good["performed"] and good["first_ok"] and not good["fall_skill"]
    assert good["dip_mm"] >= 20 and good["retract_rad"] >= 0.25
    still = duck_outcome("startle", v, "stand", 1.0, 0, _trace(ticks), hold, unwind, 0, 0.0)
    assert not still["performed"] and still["first_motion_s"] is None
    fallen = _trace(ticks)
    fallen["tilt"][10:, 0] = 80.0
    assert duck_outcome("startle", v, "stand", 1.0, 0, fallen, hold, unwind, 0, 0.0)["fall_skill"]


def test_eval_outcome_shake_window() -> None:
    from duckmind.skilltrain.rl_eval import duck_outcome

    v = next(x for x in C.variants("comfort") if x.name == "shake_2")
    hold, unwind = 50, 25
    tr = _trace(hold + unwind)
    t = np.arange(hold + unwind) * C.DT
    f = 3.2
    tr["roll"][:, 0] = math.radians(7.0) * np.sin(2 * math.pi * f * t)
    tr["roll_rate"][:, 0] = math.radians(7.0) * 2 * math.pi * f * np.cos(2 * math.pi * f * t)
    tr["head_yaw"][:, 0] = 0.25 * np.sin(2 * math.pi * 3.5 * t)
    tr["head_yaw_rate"][:, 0] = 0.25 * 2 * math.pi * 3.5 * np.cos(2 * math.pi * 3.5 * t)
    out = duck_outcome("comfort", v, "stand", 1.0, 0, tr, hold, unwind, 0, 0.0)
    assert out["performed"], out
    assert 6.0 < out["roll_amp_deg"] < 8.0 and 2.5 <= out["roll_hz_at_max"] <= 4.0
    slow = _trace(hold + unwind)
    slow["roll"][:, 0] = math.radians(7.0) * np.sin(2 * math.pi * 0.8 * t)
    slow["roll_rate"][:, 0] = math.radians(7.0) * 2 * math.pi * 0.8 * np.cos(2 * math.pi * 0.8 * t)
    assert not duck_outcome("comfort", v, "stand", 1.0, 0, slow, hold, unwind, 0, 0.0)["performed"]


def test_eval_outcome_stretch_sit_needs_the_reseat() -> None:
    from duckmind.skilltrain.rl_eval import duck_outcome

    v = next(x for x in C.variants("comfort") if x.name == "stretch_sit")
    hold, unwind = 100, 40
    tr = _trace(hold + unwind + 100)
    tr["z"][:, 0] = C.SIT_Z
    tr["z"][10:hold, 0] = C.SIT_Z + 0.014
    tr["cam"][10:hold, 0] = 20.0
    ok = duck_outcome("comfort", v, "seated", 1.0, 0, tr, hold, unwind, 100, 0.0)
    assert ok["performed"] and ok["reseated"]
    stuck = {k: a.copy() for k, a in tr.items()}
    stuck["z"][hold:, 0] = C.SIT_Z + 0.02
    assert not duck_outcome("comfort", v, "seated", 1.0, 0, stuck, hold, unwind, 100, 0.0)[
        "performed"
    ]
