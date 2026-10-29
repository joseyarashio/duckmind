"""The Stage-1 gates (:mod:`duckmind.skilltrain.metrics`) on synthetic 50 Hz traces.

Each motion is drawn to pass its plan §6 gate (and a "today" trace to fail it), so the checks
measure what the plan says: a crouch dips and pulls the neck back within 0.25–0.3 s and holds,
a stop stills the trunk, a hop moves back, a shake oscillates the roll at 2.5–4 Hz with the head
yawing, a stretch rises and looks up for ≥ 0.8 s; the Wilson numbers are the textbook's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.skilltrain import metrics as M

HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                 0.0, 0.0873, 0.4579, 0.0049, -0.4530])  # fmt: skip


def quat_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def trace(seconds: float = 6.0, **fns) -> dict:
    """A standing duck; ``fns`` override per-time functions (t ↦ value)."""
    t = np.arange(int(seconds * 50)) * 0.02
    n = len(t)
    z = np.array([fns.get("z", lambda _t: 0.116)(x) for x in t])
    x = np.array([fns.get("x", lambda _t: 0.0)(x) for x in t])
    roll = np.array([fns.get("roll", lambda _t: 0.0)(x) for x in t])
    neck = np.array([fns.get("neck", lambda _t: 0.0)(x) for x in t])
    hyaw = np.array([fns.get("hyaw", lambda _t: 0.0)(x) for x in t])
    cam = np.array([fns.get("cam", lambda _t: -11.0)(x) for x in t])
    q = np.tile(HOME, (n, 1))
    q[:, 5] += neck
    q[:, 7] += hyaw
    pos = np.stack([x, np.zeros(n), z], axis=1)
    vel = np.zeros((n, 3))
    vel[1:, 0] = np.diff(x) / 0.02
    labels = np.array([fns.get("label", lambda _t: "walk")(x) for x in t], dtype=object)
    return {
        "t": t,
        "pos": pos,
        "quat": np.array([quat_rpy(r, 0.0, 0.0) for r in roll]),
        "q": q,
        "qd": np.zeros((n, 14)),
        "qacc": np.zeros((n, 14)),
        "tau_pk": np.full((n, 14), 0.1),
        "vel": vel,
        "td": np.zeros((n, 2)),
        "feet": np.ones((n, 2)),
        "beak": np.tile([0.08, 0.0, 0.1], (n, 1)),
        "cam_pitch": cam,
        "label": labels,
        "fallen": np.zeros(n),
        "tilt": np.abs(np.degrees(roll)),
        "f_servo": np.zeros(n),
        "f_head": np.zeros(n),
        "f_trunk": np.zeros(n),
    }


def ramp(t: float, t0: float, dur: float, size: float) -> float:
    return size * min(max((t - t0) / dur, 0.0), 1.0)


def test_a_startle_crouch_passes_and_standing_still_does_not() -> None:
    tr = trace(
        z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.022) + ramp(t, 3.0, 0.5, 0.022),
        neck=lambda t: ramp(t, 2.0, 0.15, 0.3) - ramp(t, 3.0, 0.5, 0.3),
    )
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=3.0, t_end=3.8)
    assert m["first_motion_s"] <= 0.04
    assert m["retract_025_rad"] == pytest.approx(0.3, abs=0.01)
    assert m["dip_030_mm"] == pytest.approx(22.0, abs=0.5)
    assert m["hold_dev_mm"] < 1.0
    v = M.visibility("startle", 1.0, "stand", m)
    assert v["performed"], v
    still = M.trial_metrics(trace(), t_start=2.0, t_hold_end=2.5, t_end=3.3)
    v0 = M.visibility("startle", 0.25, "stand", still)
    assert not v0["performed"] and still["first_motion_s"] is None and still["dip_030_mm"] == 0


def test_a_stop_from_walking_and_a_hop_back() -> None:
    def x_walk(t: float) -> float:  # 0.2 m/s, stops smoothly within 0.4 s of t = 2
        if t < 2.0:
            return 0.2 * t
        s = min(t - 2.0, 0.4)
        return 0.4 + 0.2 * s - 0.25 * s * s

    tr = trace(x=x_walk)
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=2.5, t_end=3.3, t_resume=3.3)
    assert m["still_s"] is not None and m["still_s"] <= 0.5
    assert m["travel_cm"] < 5.0
    # (s1b:fix) a plain stop is today's body: the walking startle gate must fail it …
    v = M.visibility("startle", 0.5, "walk", m)
    assert not v["performed"]
    assert v["checks"]["stops_in_0.7s"] and v["checks"]["travel_le_12cm"]
    # … and pass a stop with the flinch: the neck back and the trunk down while stopping
    flinch = trace(
        x=x_walk,
        z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.014),
        neck=lambda t: ramp(t, 2.0, 0.15, 0.25),
    )
    mf = M.trial_metrics(flinch, t_start=2.0, t_hold_end=2.5, t_end=3.3, t_resume=3.3)
    assert M.visibility("startle", 0.5, "walk", mf)["performed"]
    assert mf["dip_040_mm"] >= mf["dip_030_mm"] >= 10.0 and mf["retract_040_rad"] >= 0.2
    hop = trace(x=lambda t: -ramp(t, 2.0, 0.5, 0.08))
    mh = M.trial_metrics(hop, t_start=2.0, t_hold_end=2.8, t_end=3.4)
    assert mh["back_070_cm"] == pytest.approx(8.0, abs=0.2)
    assert M.visibility("hop_back", 1.0, "stand", mh)["performed"]
    assert not M.visibility(
        "hop_back", 1.0, "stand", M.trial_metrics(trace(), t_start=2.0, t_hold_end=2.8, t_end=3.4)
    )["performed"]


def test_a_shake_off_and_a_head_only_shake() -> None:
    def on(t: float) -> float:
        return 1.0 if 2.0 <= t < 3.0 else 0.0

    tr = trace(
        roll=lambda t: on(t) * math.radians(6) * math.sin(2 * math.pi * 3.2 * (t - 2)),
        hyaw=lambda t: on(t) * 0.25 * math.sin(2 * math.pi * 3.5 * (t - 2)),
    )
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=3.0, t_end=3.5)
    assert m["roll_osc_s"] >= 0.5 and 2.5 <= m["roll_hz"] <= 4.0
    assert m["yaw_amp_rad"] == pytest.approx(0.25, abs=0.02) and m["yaw_hz"] >= 3.0
    assert M.visibility("shake", 1.0, "stand", m)["performed"]
    head_only = trace(hyaw=lambda t: on(t) * 0.1 * math.sin(2 * math.pi * 3 * (t - 2)))
    mh = M.trial_metrics(head_only, t_start=2.0, t_hold_end=3.0, t_end=3.5)
    v = M.visibility("shake", 1.0, "stand", mh)
    assert not v["performed"] and not v["checks"]["roll_osc_ge_0.5s"]


def test_a_stretch_tall_and_a_seated_stretch_with_its_reseat() -> None:
    tr = trace(
        z=lambda t: 0.116 + ramp(t, 2.0, 0.6, 0.011) - ramp(t, 3.8, 0.6, 0.011),
        cam=lambda t: -11.0 + ramp(t, 2.0, 0.5, 30.0) - ramp(t, 3.8, 0.5, 30.0),
    )
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=4.0, t_end=4.8)
    assert m["stretch_tall_hold_s"] >= 0.8
    assert M.visibility("stretch_tall", 1.0, "stand", m)["performed"]
    seat = trace(
        z=lambda t: 0.060 + ramp(t, 2.0, 0.6, 0.014) - ramp(t, 4.0, 0.6, 0.014),
        cam=lambda t: -11.0 + ramp(t, 2.0, 0.5, 30.0) - ramp(t, 4.0, 0.5, 30.0),
        label=lambda t: "sit" if t < 2.0 or t >= 4.8 else "stretch_sit",
    )
    ms = M.trial_metrics(seat, t_start=2.0, t_hold_end=4.0, t_end=4.8)
    assert ms["stretch_sit_hold_s"] >= 0.8 and ms["reseat_s"] is not None
    assert ms["reseat_s"] <= 1.5
    assert M.visibility("stretch_sit", 1.0, "sit", ms)["performed"]


def test_falls_righting_contacts_and_strain_are_measured() -> None:
    tr = trace()
    tr["tilt"][150:190] = 80.0
    tr["fallen"][160:185] = 1.0
    tr["f_servo"][120] = 3.0  # a housing hit while upright
    tr["tau_pk"][110:120, 2] = 0.5
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=2.5, t_end=3.3, ceiling_nm=0.64)
    assert m["fell"] and m["t45"] and m["servo_hits_upright"] == 1
    assert m["righted_s"] is not None and m["righted_8s"]
    assert m["strain_ticks"] == 10 and m["tau_max"] == 0.5
    s = M.summarize([{"metrics": m, "visibility": {"performed": False, "checks": {}}}])
    assert s["falls"]["k"] == 1 and s["verdicts"]["servo_hits_upright"] == "FAIL"


def test_wilson_and_rate_verdicts() -> None:
    assert M.wilson(0, 30) == pytest.approx((0.0, 0.1135), abs=1e-4)
    assert M.wilson(15, 30) == pytest.approx((0.3315, 0.6685), abs=1e-4)
    assert M.wilson(30, 30)[0] == pytest.approx(0.8865, abs=1e-4)
    assert M.rate_verdict(0, 30, 0.002) == "UNRESOLVED"
    assert M.rate_verdict(0, 2000, 0.002) == "PASS"
    assert M.rate_verdict(5, 30, 0.002) == "FAIL"
    assert M.rate_verdict(30, 30, 0.85, at_most=False) == "PASS"
    n = M.n_for_zero(0.002)
    assert M.wilson(0, n)[1] <= 0.002 < M.wilson(0, n - 1)[1]


def test_bob_finds_a_crouch_rise_crouch_and_not_a_single_crouch() -> None:
    one = [0.116] * 10 + [0.090] * 30 + [0.116] * 20  # down, held, up once
    assert M.bob(one) == pytest.approx(0.0, abs=1e-9)
    twice = [0.116] * 10 + [0.089] * 20 + [0.116] * 15 + [0.089] * 20 + [0.116] * 10
    assert M.bob(twice) == pytest.approx(0.027, abs=1e-9)
    partial = [0.116] * 10 + [0.089] * 20 + [0.100] * 5 + [0.089] * 20 + [0.116] * 10
    assert M.bob(partial) == pytest.approx(0.011, abs=1e-9)
    tr = trace(z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.027) + ramp(t, 2.8, 0.5, 0.027)
               - ramp(t, 4.0, 0.2, 0.027) + ramp(t, 5.0, 0.4, 0.027))  # fmt: skip
    m = M.trial_metrics(tr, t_start=2.0, t_hold_end=4.8, t_end=5.8)
    assert m["bob_mm"] == pytest.approx(27.0, abs=0.5)


def test_span_markers_cover_a_gap_between_two_entries() -> None:
    def label(t: float) -> str:
        if 2.0 <= t < 3.3:
            return "startle_4" if t < 2.5 else "startle_4:unwind"
        if 4.0 <= t < 5.0:
            return "freeze_4"
        if 5.0 <= t < 5.8:
            return "startle_release"
        return "walk"

    tr = trace(label=label)
    names = ["startle_4", "freeze_4", "startle_release"]
    first = M.any_skill_markers(tr, names)
    span = M.span_markers(tr, names)
    assert first["t_end"] == pytest.approx(3.3)
    assert span["t_start"] == pytest.approx(2.0) and span["t_end"] == pytest.approx(5.8)
    assert span["t_hold_end"] == pytest.approx(2.5)
    assert span["sequence"] == ["startle_4", "startle_4:unwind", "walk", "freeze_4",
                                "startle_release"]  # fmt: skip


def test_a_continuation_is_one_crouch_and_the_late_freeze_is_not() -> None:
    one = trace(
        z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.016) - ramp(t, 2.3, 0.3, 0.006)
        + ramp(t, 5.3, 0.8, 0.022),
        neck=lambda t: ramp(t, 2.0, 0.15, 0.30) - ramp(t, 5.3, 0.8, 0.30),
    )  # fmt: skip
    m = M.trial_metrics(one, t_start=2.0, t_hold_end=5.3, t_end=6.1)
    v = M.visibility("continuation", 0.5, "stand", m)
    assert v["performed"], v
    assert not M.visibility("freeze", 0.75, "stand", m)["checks"]["hold_within_3mm"]
    late = trace(
        z=lambda t: 0.116 - ramp(t, 2.0, 0.2, 0.027) + ramp(t, 2.5, 0.8, 0.027)
        - ramp(t, 4.0, 0.2, 0.027) + ramp(t, 7.0, 0.8, 0.027),
        neck=lambda t: ramp(t, 2.0, 0.15, 0.44),
        seconds=9.0,
    )  # fmt: skip
    ml = M.trial_metrics(late, t_start=2.0, t_hold_end=2.5, t_end=7.8)
    vl = M.visibility("continuation", 1.0, "stand", ml)
    assert not vl["performed"] and not vl["checks"]["one_crouch_bob_lt_8mm"]
