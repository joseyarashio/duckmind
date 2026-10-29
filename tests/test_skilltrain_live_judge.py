"""The page witness's judge for trained skills (:mod:`duckmind.skilltrain.live_judge`): a skill
window is scored by the skill's own detectors (trunk dip, neck pull-back, one crouch, the stop),
not by the head command robotd ignores while the skill runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from duckmind.skilltrain import live_judge as J


@dataclass(frozen=True)
class Tick:
    t_ms: int
    policy: str
    head: tuple[float, float, float, float]
    trunk: tuple[float, float, float, float] | None
    trunk_t: float | None
    fallen: bool = False
    tilt_deg: float | None = 2.0


def ramp(t: float, t0: float, dur: float, size: float) -> float:
    return size * min(max((t - t0) / dur, 0.0), 1.0)


def ticks(policy, z, neck, x=lambda t: 0.0, yaw=lambda t: 0.0, seconds: float = 6.0):
    out = []
    for i in range(int(seconds * 50)):
        t = i * 0.02
        out.append(
            Tick(
                t_ms=int(round(t * 1000)),
                policy=policy(t),
                head=(neck(t), 0.0, yaw(t), 0.0),
                trunk=(x(t), 0.0, z(t), 0.0),
                trunk_t=t,
            )
        )
    return out


def test_the_thresholds_are_the_gates_own() -> None:
    M = pytest.importorskip("duckmind.skilltrain.metrics")  # (the main tree; not the wiring copy)
    assert (J.RETRACT_LO, J.RETRACT_HI, J.DIP_PER_K_MM) == (
        M.RETRACT_LO, M.RETRACT_HI, M.DIP_PER_K_MM)  # fmt: skip
    assert (J.STOP_S, J.STOP_CM, J.STILL_MPS) == (M.STOP_S, M.STOP_CM, M.STILL_MPS)
    assert (J.HOP_CM, J.HOP_S, J.HOP_YAW_DEG, J.SHAKE_YAW_RAD) == (
        M.HOP_CM, M.HOP_S, M.HOP_YAW_DEG, M.SHAKE_YAW_RAD)  # fmt: skip
    assert J.STRETCH_MM == M.STRETCH_MM
    z = [0.116, 0.09, 0.116, 0.09, 0.116]
    assert J.bob_m(z) == pytest.approx(M.bob(z))


def test_a_trained_startle_reads_done_though_the_named_head_turn_was_not_made() -> None:
    # the brain's reflex at 1.0 s; robotd starts startle_4 at 1.24 s (0.5 s + 0.8 s unwind)
    def policy(t: float) -> str:
        if 1.24 <= t < 1.74:
            return "startle_4"
        if 1.74 <= t < 2.54:
            return "startle_4:unwind"
        return "walk"

    tr = ticks(
        policy,
        z=lambda t: 0.116 - ramp(t, 1.26, 0.2, 0.027) + ramp(t, 1.74, 0.8, 0.027),
        neck=lambda t: ramp(t, 1.26, 0.15, 0.44) - ramp(t, 1.74, 0.8, 0.44),
    )
    a = J.judge_window(tr, "startle", 1000, named=("neck", "yaw_right"))
    assert a is not None and a["verdict"] == "done", a
    assert a["skill"] == {"entry": "startle_4", "family": "startle", "level": 1.0}
    assert a["checks"] == {"neck_back": True, "trunk_down": True, "one_crouch": True}
    assert a["deepest_mm"] == pytest.approx(27.0, abs=0.6)
    assert 0.2 <= a["motion_after_reflex_s"] <= 0.35
    assert "蹲低 2.7 公分" in a["zh"] and "頭轉向右邊" in a["zh"]  # said, not graded
    assert a["zh"].startswith("身體：")


def test_the_old_crouch_rise_crouch_is_not_one_crouch() -> None:
    def policy(t: float) -> str:
        if 1.0 <= t < 1.5:
            return "startle_4"
        if 1.5 <= t < 2.3:
            return "startle_4:unwind"
        if 3.0 <= t < 5.0:
            return "freeze_4"
        return "walk"

    tr = ticks(
        policy,
        z=lambda t: 0.116 - ramp(t, 1.0, 0.2, 0.027) + ramp(t, 1.5, 0.8, 0.027)
        - ramp(t, 3.0, 0.2, 0.027) + ramp(t, 5.0, 0.8, 0.027),
        neck=lambda t: ramp(t, 1.0, 0.15, 0.44) - ramp(t, 1.5, 0.8, 0.44),
    )  # fmt: skip
    a = J.judge_window(tr, "startle", 900)
    # the run of skills ends at the 0.7 s gap: the freeze is a new crouch, judged on its own;
    # over a window that holds both (the witness's 2.5-s freeze window after the gap) …
    assert a is not None and a["verdict"] == "done"
    b = J.judge_window([k for k in tr if k.t_ms >= 900], "startle", 900)
    assert b is not None

    # … and a continuous span (the takeover never lets it rise) scores one crouch
    def cont(t: float) -> str:
        if 1.0 <= t < 1.3:
            return "startle_4"
        if 1.3 <= t < 4.0:
            return "freeze_4"
        if 4.0 <= t < 4.8:
            return "startle_release"
        return "walk"

    one = ticks(cont, z=lambda t: 0.116 - ramp(t, 1.0, 0.2, 0.027) + ramp(t, 4.0, 0.8, 0.027),
                neck=lambda t: ramp(t, 1.0, 0.15, 0.44) - ramp(t, 4.0, 0.8, 0.44))  # fmt: skip
    c = J.judge_window(one, "startle", 950)
    assert c is not None and c["checks"]["one_crouch"] and c["bob_mm"] < 1.0

    # a bob inside one run (a takeover during the unwind that let it rise 12 mm) is caught
    def late(t: float) -> str:
        if 1.0 <= t < 1.5:
            return "startle_4"
        if 1.5 <= t < 2.0:
            return "startle_4:unwind"
        if 2.0 <= t < 4.0:
            return "freeze_4"
        return "walk"

    bobbed = ticks(late, z=lambda t: 0.116 - ramp(t, 1.0, 0.2, 0.027) + ramp(t, 1.5, 0.5, 0.012)
                   - ramp(t, 2.0, 0.2, 0.012) + ramp(t, 4.0, 0.8, 0.027),
                   neck=lambda t: ramp(t, 1.0, 0.15, 0.44))  # fmt: skip
    d = J.judge_window(bobbed, "startle", 950)
    assert d is not None and not d["checks"]["one_crouch"] and d["verdict"] == "partial"
    assert "站起來又蹲下" in d["zh"]


def test_a_walking_startle_must_stop_and_flinch() -> None:
    def x(t: float) -> float:
        if t < 2.0:
            return 0.3 * t
        s = min(t - 2.0, 0.4)
        return 0.6 + 0.3 * s - 0.375 * s * s

    def policy(t: float) -> str:
        return "startle_2" if 2.0 <= t < 2.5 else "startle_2:unwind" if 2.5 <= t < 3.3 else "walk"

    tr = ticks(policy, z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.015) + ramp(t, 2.5, 0.8, 0.015),
               neck=lambda t: ramp(t, 2.0, 0.15, 0.27), x=x)  # fmt: skip
    a = J.judge_window(tr, "startle", 1900)
    assert a is not None and a["walking"] and a["checks"]["stopped"], a
    assert a["verdict"] == "done" and "停下來" in a["zh"]
    # the same stop without the flinch (today's body): not done
    plain = ticks(policy, z=lambda t: 0.116, neck=lambda t: 0.0, x=x)
    b = J.judge_window(plain, "startle", 1900)
    assert b is not None and b["verdict"] == "partial" and not b["checks"]["trunk_down"]


def test_no_trained_skill_no_opinion_and_the_now_words() -> None:
    tr = ticks(lambda t: "walk", z=lambda t: 0.116, neck=lambda t: 0.0)
    assert J.judge_window(tr, "startle", 1000) is None
    assert J.doing_zh("freeze_3") == "蹲低定住不動"
    assert J.doing_zh("startle_2b:unwind") is not None
    assert J.doing_zh("walk") is None and J.skill_of("roulade") is None
    hop = ticks(lambda t: "hop_back" if 1.0 <= t < 1.8 else "walk",
                z=lambda t: 0.116, neck=lambda t: 0.0,
                x=lambda t: -ramp(t, 1.0, 0.5, 0.09))  # fmt: skip
    h = J.judge_window(hop, "startle", 950)
    assert h is not None and h["verdict"] == "done" and h["back_cm"] == pytest.approx(9.0, abs=0.3)
    shake = ticks(lambda t: "shake_2" if 1.0 <= t < 2.0 else "walk", z=lambda t: 0.116,
                  neck=lambda t: 0.0,
                  yaw=lambda t: 0.2 * math.sin(2 * math.pi * 3.5 * t) if 1.0 <= t < 2.0 else 0.0)  # fmt: skip
    s = J.judge_window(shake, "body_shake", 950)
    assert s is not None and s["verdict"] == "done"


def test_the_one_crouch_threshold_is_the_gates_own() -> None:
    M = pytest.importorskip("duckmind.skilltrain.metrics")
    assert J.BOB_MM == M.BOB_MM


def test_a_look_during_a_skill_is_not_graded() -> None:
    def policy(t: float) -> str:
        return "freeze_4" if 1.0 <= t < 4.0 else "startle_release" if 4.0 <= t < 4.8 else "walk"

    tr = ticks(policy, z=lambda t: 0.116 - ramp(t, 1.0, 0.2, 0.027) + ramp(t, 4.0, 0.8, 0.027),
               neck=lambda t: ramp(t, 1.0, 0.15, 0.44))  # fmt: skip
    a = J.judge_window(tr, "orient", 900, named=("yaw_right",))
    assert a is not None and a["verdict"] is None and a["not_graded"]
    assert "不算分" in a["zh"] and "蹲低定住" in a["zh"]
