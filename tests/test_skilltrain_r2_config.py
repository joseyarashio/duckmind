from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")

from duckmind.skilltrain import bsk_tasks as BSK
from duckmind.skilltrain import bsk_train as BT
from duckmind.skilltrain import hop_tasks as HOP


def test_bsk_r2_rewards_are_opt_in_and_narrow() -> None:
    assert BSK.REWARD_REV == "b3"
    b3 = BSK.rewards("bskbow", "b3")
    b4 = BSK.rewards("bskbow", "b4")
    assert b3["leg_torque"].weight == -5.0
    assert b4["leg_torque"].weight == -15.0
    for name in b3.keys() - {"leg_torque"}:
        assert b3[name].weight == b4[name].weight
    assert BSK.rewards("bskstep", "b3")["leg_torque"].weight == -5.0


def test_bsk_r2_entry_mix_is_training_only_and_nonzero() -> None:
    cfg = BSK.make_env_cfg("bskbow", reward_rev="b4")
    overrides = cfg.events["skill_episode"].params["entry_mix_overrides"]
    assert overrides == {
        "bow": {"stand": 0.35, "walk": 0.35, "coast": 0.20, "head": 0.05, "risen": 0.05},
        "lie": {"stand": 0.35, "walk": 0.35, "coast": 0.20, "head": 0.05, "risen": 0.05},
    }
    assert sum(overrides["bow"].values()) == pytest.approx(1.0)
    assert overrides["bow"]["walk"] > 0
    eval_cfg = BSK.make_env_cfg("bskbow", eval_cells=[("bow", "stand")], reward_rev="b4")
    assert eval_cfg.events["skill_episode"].params["entry_mix_overrides"] is None


def test_bsk_fixed_delay_parser_and_revision_defaults() -> None:
    assert BT.parse_delay("train") == "train"
    assert BT.parse_delay("8steps") == 8
    assert BT.parse_delay("40ms") == 8
    assert BT.parse_delay("4") == 4
    for value in ("jitter", "45ms", "9steps"):
        with pytest.raises(ValueError):
            BT.parse_delay(value)
    args = BT.parse(["--net", "bskbow", "--policies", "/tmp/policies", "--out", "/tmp/out"])
    assert args.reward_rev == "b3" and args.delay == "train"
    fixed_cfg = BSK.make_env_cfg("bskbow", delay_steps=(8, 8))
    actuators = fixed_cfg.scene.entities["robot"].articulation.actuators
    delayed = next(a for a in actuators if hasattr(a, "delay_max_lag"))
    assert (delayed.delay_min_lag, delayed.delay_max_lag) == (8, 8)


def test_hop_h4_changes_only_requested_weights_and_keeps_h3_runtime() -> None:
    h2, h3, h4 = (HOP.rewards(rev) for rev in ("h2", "h3", "h4"))
    assert HOP.REWARD_REV == "h2"
    assert "fall_event" not in h2
    assert h3["housing_touch"].weight == -4.0 and h3["fall_event"].weight == -300.0
    assert h4["housing_touch"].weight == -12.0 and h4["fall_event"].weight == -600.0
    for name in h3.keys() - {"housing_touch", "fall_event"}:
        assert h3[name].weight == h4[name].weight
    cfg = HOP.make_env_cfg(reward_rev="h4", handback_s=2.0, curriculum_scale=0.0)
    assert cfg.scene.terrain.terrain_generator is not None
    assert cfg.curriculum["strip_levels"].params["max_level"] == 3
    assert cfg.events["hop_episode"].params["handback_s"] == 2.0


def test_delay_metadata_uses_configured_actuator_fields() -> None:
    from duckmind.skilltrain import bsk_train as BT
    from duckmind.skilltrain import hop_train as HT

    actuator = SimpleNamespace(
        delay_min_lag=8, delay_max_lag=8, delay_update_period=100, delay_hold_prob=0.5
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            scene=SimpleNamespace(
                entities={
                    "robot": SimpleNamespace(articulation=SimpleNamespace(actuators=[actuator]))
                }
            )
        )
    )
    assert BT._delay_meta(env)["delay_max_lag"] == 8
    assert HT._delay_meta(env)["delay_min_lag"] == 8


def test_h4_handback_loads_real_v5_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    from duckmind.skilltrain import hop_train as HT

    seen: list[tuple[str, str]] = []
    fake = ModuleType("duckmind.skilltrain.rl_onnx")
    fake.torch_policy = lambda path, device: seen.append((str(path), device)) or "v5-policy"
    monkeypatch.setitem(sys.modules, "duckmind.skilltrain.rl_onnx", fake)
    assert HT.load_handback_policy("h4", "/models", "cpu") == "v5-policy"
    assert seen == [("/models/velstand.onnx", "cpu")]
    assert HT.load_handback_policy("h2", "/models", "cpu") is None
