"""hop:train (2026-09-28): the servo-delay modes and the evaluator's summary — pure (numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import hop_eval as E
from duckmind.skilltrain import rl_contract as C


def test_training_delay_is_piecewise_constant_like_stage_1b() -> None:
    f = H.delay_fields("train")
    assert (f["delay_min_lag"], f["delay_max_lag"]) == (0, 8)  # 0–40 ms at the 5 ms physics step
    assert f["delay_update_period"] == 100  # a fresh lag every 0.5 s (staggered), p 0.5
    assert f["delay_hold_prob"] == pytest.approx(0.5)
    assert H.SERVO_LAG_TRAIN == C.SERVO_LAG_TRAIN  # the same draw as Stage 1b's retrain
    j = H.delay_fields("jitter")  # the calibration's: re-drawn every physics step
    assert (j["delay_min_lag"], j["delay_max_lag"], j["delay_update_period"]) == (0, 8, 0)


def test_fixed_delays_parse_from_ms_or_steps() -> None:
    assert H.parse_delay("train") == "train"
    assert H.parse_delay("JITTER") == "jitter"
    assert H.parse_delay("20ms") == 4
    assert H.parse_delay("40ms") == 8
    assert H.parse_delay("0") == 0
    assert H.delay_label(4) == "d20ms" and H.delay_label("train") == "train"
    fixed = H.delay_fields(8)
    assert (fixed["delay_min_lag"], fixed["delay_max_lag"], fixed["delay_update_period"]) == (
        8,
        8,
        0,
    )
    with pytest.raises(ValueError):
        H.parse_delay("7ms")  # not a whole physics step
    with pytest.raises(ValueError):
        H.delay_fields(21)


def _traces(t: int, n: int) -> dict[str, np.ndarray]:
    tr = {k: np.zeros((t, n), dtype=np.float32) for k in E.TRACE}
    tr["tau_leg"][:] = 0.3
    tr["foot_fz"][:] = 0.5 * H.BODY_WEIGHT_N
    return tr


def test_summary_pools_heights_with_landing_and_housing() -> None:
    cells = [(0, 0, "walk"), (7, 1, "stand")]  # 5 mm sharp walk, 40 mm round stand
    n = 4
    t, hold, unwind = 30, 20, 5
    tr = _traces(t, len(cells) * n)
    info = {
        "row": np.array([0, 7] * n),
        "col": np.array([0, 1] * n),
        "kind": np.zeros(2 * n),
        "height": np.array([0.005, 0.040] * n),
        "start_d": np.full(2 * n, 0.1),
        "hold": np.full(2 * n, hold),
        "unwind": np.full(2 * n, unwind),
        "cls": np.array([0.01, 0.04] * n),
    }
    tr["crossed"][10:, 0] = 1.0  # duck 0 (5 mm) crosses
    tr["crossed"][12:, 1] = 1.0  # duck 1 (40 mm) crosses …
    tr["servo_f"][11, 1] = 5.0  # … hitting a housing on the way
    tr["foot_fz"][13, 1] = 4.0 * H.BODY_WEIGHT_N  # and landing hard
    tr["tilt"][5:, 3] = 80.0  # duck 3 (40 mm) falls
    res = E.summarise(cells, n, tr, info, handback_ticks=0)
    h5, h40 = res["by_height"]["h5mm"], res["by_height"]["h40mm"]
    assert h5["crossed"][:2] == [1, 4] and h40["crossed"][:2] == [1, 4]
    assert h40["fall_skill"][:2] == [1, 4] and h5["fall_skill"][:2] == [0, 4]
    assert h40["housing_impact"][:2] == [1, 4]
    assert h40["landing_over_target"][:2] == [1, 4]  # 4.0 × body weight > 2.5
    assert h5["landing_over_target"][:2] == [0, 4]
    duck1 = next(d for d in res["ducks"] if d["cell"] == 1 and d["housing_impact"])
    assert duck1["foot_peak_bw"] == pytest.approx(4.0, abs=1e-3)
    assert res["cells"]["h40mm/round/stand"]["foot_peak_bw_p90"] >= 0.5


class _Cfg:
    def __init__(self, weight: float, **params: float) -> None:
        self.weight = weight
        self.params = dict(params)


class _Rewards:
    def __init__(self, terms: dict[str, _Cfg]) -> None:
        self.terms = terms
        self.active_terms = list(terms)

    def get_term_cfg(self, name: str) -> _Cfg:
        return self.terms[name]


class _Env:
    def __init__(self, terms: dict[str, _Cfg]) -> None:
        self.reward_manager = _Rewards(terms)
        self.unwrapped = self


def test_mid_run_override_keeps_signs_and_only_listed_terms() -> None:
    from duckmind.skilltrain import hop_train as T

    terms = {
        "progress": _Cfg(20.0, max_mps=0.35),
        "housing_touch": _Cfg(0.0, threshold=2.0),
        "servo_impact": _Cfg(-0.5, threshold=2.0),
        "action_rate_l2": _Cfg(-0.02),
    }
    env = _Env(terms)
    costs = {"housing_touch", "servo_impact", "action_rate_l2"}
    spec = {
        "id": "t1",
        "weights": {
            "housing_touch": -2.0,
            "servo_impact": 1.0,
            "action_rate_l2": -0.5,
            "progress": 15,
        },
        "params": {"progress": {"max_mps": 0.3, "bogus": 1}, "servo_impact": {"threshold": 3}},
    }
    lines: list[str] = []
    res = T.apply_override(env, spec, costs, lines.append)
    assert terms["housing_touch"].weight == -2.0  # a cost switched on
    assert terms["servo_impact"].weight == -0.5  # a cost may not become a reward
    assert "servo_impact" in res["rejected"]
    assert terms["action_rate_l2"].weight == -0.02  # the curriculum owns it
    assert "action_rate_l2" in res["rejected"]
    assert terms["progress"].weight == 15.0 and terms["progress"].params["max_mps"] == 0.3
    assert "progress.bogus" in res["rejected"]
    assert terms["servo_impact"].params["threshold"] == 3.0
    assert lines and "t1" in lines[0]


def test_override_file_is_read_once_per_write(tmp_path) -> None:
    from duckmind.skilltrain import hop_train as T

    p = tmp_path / "OVERRIDE.json"
    state: dict = {"mtime": None}
    assert T.poll_override(p, state) is None  # no file yet
    p.write_text('{"id": "a", "weights": {"fallen": -5}}')
    assert T.poll_override(p, state) == {"id": "a", "weights": {"fallen": -5}}
    assert T.poll_override(p, state) is None  # unchanged: applied once
    p.write_text("not json")
    import os

    os.utime(p, (1.0, 12345.0))
    bad = T.poll_override(p, state)
    assert bad is not None and "_error" in bad
