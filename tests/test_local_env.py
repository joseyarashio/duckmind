"""Optional real native-MuJoCo checks; set DUCKMIND_RL_UPSTREAM to reviewed checkout."""

import os
import socket
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
mujoco = pytest.importorskip("mujoco")

from duckmind.local_env import ACTION_LIMIT, LocalBalanceEnv  # noqa: E402


@pytest.fixture
def env():
    checkout = os.environ.get("DUCKMIND_RL_UPSTREAM")
    if not checkout:
        pytest.skip("set DUCKMIND_RL_UPSTREAM for optional native MuJoCo integration")
    environment = LocalBalanceEnv(Path(checkout), horizon_steps=250)
    yield environment
    environment.close()


def test_seeded_reset_restores_same_finite_raw_contract(env):
    first, first_info = env.reset(seed=24)
    env.step(np.full(14, 0.1, dtype=np.float32))
    second, second_info = env.reset(seed=24)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (61,) and first.dtype == np.float32
    assert env.observation_space.contains(first)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first[34:], np.zeros(27, dtype=np.float32))
    assert first_info["survival_steps"] == second_info["survival_steps"] == 0
    assert first_info["actuator_model"] == "native_mujoco_position_pd_not_bam"
    assert first_info["trunk_z"] == first_info["height"]


def test_observation_matches_official_raw_wire_values_and_omits_mouth(env):
    env.reset(seed=1)
    env.world.data.qvel[env.body.trunk_dof + 3 : env.body.trunk_dof + 6] = [2, 3, 4]
    env.world.data.qvel[env.body.qvel_adr] = np.arange(14) + 5
    state = env.body.sensors()
    observation = env.observation()
    indices = [i for i in range(15) if i != 9]
    np.testing.assert_array_equal(observation[:3], np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_allclose(observation[3:6], state["imu"]["gravity"])
    np.testing.assert_allclose(
        observation[6:20], np.asarray(state["positions"])[indices] - env.home, atol=1e-7
    )
    np.testing.assert_array_equal(observation[20:34], np.arange(14, dtype=np.float32) + 5)
    np.testing.assert_array_equal(observation[48:], np.zeros(13, dtype=np.float32))


def test_step_applies_bounded_action_and_feeds_exact_applied_action_back(env):
    env.reset(seed=2)
    action = np.linspace(-1, 1, 14, dtype=np.float32)
    expected = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
    before_time = env.world.data.time
    observation, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_array_equal(observation[34:48], expected)
    np.testing.assert_allclose(env.world.data.ctrl[env.body.actuator_slice], env.home + expected)
    assert env.world.data.time - before_time == pytest.approx(0.02)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["survival_steps"] == 1
    assert info["survival_s"] == pytest.approx(0.02)
    assert info["height"] > 0 and np.isfinite(info["tilt_rad"])


@pytest.mark.parametrize(
    "action", [np.zeros(13), np.zeros((2, 7)), np.full(14, np.nan), np.full(14, np.inf)]
)
def test_invalid_action_is_rejected_before_physics_advances(env, action):
    env.reset(seed=3)
    previous_time = env.world.data.time
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert env.world.data.time == previous_time


def test_horizon_is_truncation_not_fall(env):
    env.horizon_steps = 1
    env.reset(seed=4)
    _, _, terminated, truncated, info = env.step(np.zeros(14, dtype=np.float32))
    assert not terminated and truncated
    assert not info["fell"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(14, dtype=np.float32))


@pytest.mark.parametrize("fault", ["height", "tilt"])
def test_fall_termination_uses_ground_truth_and_is_not_time_truncation(env, monkeypatch, fault):
    env.reset(seed=5)
    if fault == "height":
        env.world.data.qpos[env.body.trunk + 2] = 0.05
    else:
        env.world.data.qpos[env.body.trunk + 3 : env.body.trunk + 7] = [
            np.cos(0.5),
            np.sin(0.5),
            0,
            0,
        ]
    mujoco.mj_forward(env.world.model, env.world.data)
    monkeypatch.setattr(env.world, "step", lambda _count: None)
    _, _, terminated, truncated, info = env.step(np.zeros(14, dtype=np.float32))
    assert terminated and not truncated and info["fell"]
    assert info["termination_reason"] == fault


def test_offboard_environment_never_opens_socket(env, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("native CPU environment must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    independent = LocalBalanceEnv(env.upstream)
    try:
        independent.reset(seed=6)
        independent.step(np.zeros(14, dtype=np.float32))
    finally:
        independent.close()
