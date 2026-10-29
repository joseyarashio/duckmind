"""The queue's training tasks on the official physics (qt_*): every item builds, steps, and its commands / rewards behave.

Needs the official stack (mjlab, microduck_rl, torch); skipped where it is absent (the main venv). On the Spark:

    PYTHONPATH=src <microduck_rl checkout>/.venv/bin/python -m pytest tests/test_skilltrain_qt_mjlab.py -o addopts=
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("mjlab_microduck")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from duckmind.skilltrain import qt_contract as C  # noqa: E402


def _env(item: str, n: int = 8):
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from duckmind.skilltrain import qt_tasks as T

    cfg = T.make_env_cfg(item, topple_scale=0.0, prone_scale=0.0)
    cfg.scene.num_envs = n
    return ManagerBasedRlEnv(cfg, device="cpu")


@pytest.mark.parametrize("item", C.ITEMS)
def test_every_item_builds_and_steps_with_finite_rewards(item: str) -> None:
    env = _env(item)
    env.reset()
    for _ in range(3):
        obs, rew, term, trunc, extras = env.step(torch.zeros(env.num_envs, 14))
        assert torch.isfinite(rew).all() and torch.isfinite(obs["actor"]).all()
    assert obs["actor"].shape[1] == 61
    env.close()


def test_pose_commands_follow_robotds_smoothing_and_stay_in_range() -> None:
    env = _env("ex")
    env.reset()
    head = env.command_manager.get_term("head_pose")
    body = env.command_manager.get_term("body_pose")
    seen_h = seen_b = 0.0
    for _ in range(200):
        env.step(torch.zeros(env.num_envs, 14))
        seen_h = max(seen_h, float(head.command.abs().max()))
        seen_b = max(seen_b, float(body.command[:, 4].abs().max()))
        assert (body.command[:, [0, 1, 5]] == 0).all()  # robotd's body block: x, y, yaw stay 0
    assert 0.0 < seen_h <= 1.7 and 0.0 < seen_b <= 0.21
    env.close()


def test_the_style_code_rides_in_the_carrier_slots() -> None:
    env = _env("style", 16)
    obs, _ = env.reset()
    term = env.command_manager.get_term("body_pose")
    codes = torch.tensor([C.STYLE_CODE[s] for s in C.STYLES])
    got = obs["actor"][:, 55:57]
    assert torch.allclose(got, codes[term.style_id], atol=1e-5)
    env.close()
