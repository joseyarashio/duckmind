"""robotd's pose mode, ported (:mod:`duckmind.skilltrain.s2_port`), against ``main.rs`` / ``control.rs``.

No ONNX: a fake policy set records what each network saw. Checked: the body block glides in at
cmd_alpha and snaps to 0 on release; pose mode zeroes the twist the network sees while the twist EMA
keeps integrating; with no stand network the walk network drives in pose mode (no handover); the
body block sits at obs 57/58/59 with x/y/yaw zero; the head EMA uses head_alpha; the target hook.
"""

from __future__ import annotations

import numpy as np

from duckmind.skilltrain import robotd_port as R
from duckmind.skilltrain import s2_port as S


class FakeNet:
    def __init__(self, name: str, value: float) -> None:
        self.name, self.value = name, value
        self.seen: list[np.ndarray] = []

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(obs, float).copy())
        return np.full(14, self.value, dtype=np.float32)


class FakePolicy(R.PolicySet):
    def __init__(self, stand: bool = False) -> None:  # noqa: D107
        self.walk = FakeNet("walk", 0.01)
        self.stand = FakeNet("stand", 0.02) if stand else None
        self.sitstand = FakeNet("sitstand", 0.03)
        self.skills = []
        self.active = None


def sensors() -> R.Sensors:
    return R.Sensors(np.zeros(3), np.array([0.0, 0.0, -1.0]), np.array(R.HOME), np.zeros(14))


def make(stand: bool = False, head_alpha: float = 1.0) -> tuple[S.PoseRobotd, FakePolicy]:
    policy = FakePolicy(stand=stand)
    robotd = S.PoseRobotd(policy, [], head_alpha=head_alpha)
    robotd.boot(seated=False)
    return robotd, policy


def test_the_body_glides_in_at_cmd_alpha_and_snaps_back_on_release() -> None:
    robotd, policy = make()
    robotd.pose(-0.02, 0.2, -0.1, active=True)
    expect = np.zeros(3)
    for k in range(20):
        robotd.tick(sensors(), k * R.DT)
        expect += 0.2 * (np.array([-0.02, 0.2, -0.1]) - expect)
        assert np.allclose(robotd.body_ema, expect)
        obs = policy.walk.seen[-1]
        assert np.allclose(obs[57:60], expect) and np.allclose(obs[[55, 56, 60]], 0.0)
    robotd.pose(0.0, 0.0, 0.0, active=False)
    robotd.tick(sensors(), 20 * R.DT)
    assert np.allclose(robotd.body_ema, 0.0) and np.allclose(policy.walk.seen[-1][55:61], 0.0)


def test_pose_mode_zeroes_the_twist_the_network_sees_while_the_ema_integrates() -> None:
    robotd, policy = make()
    for k in range(10):
        robotd.move(0.4, 0.0, 0.3, k * R.DT)
        robotd.tick(sensors(), k * R.DT)
    assert policy.walk.seen[-1][48] > 0.3
    robotd.pose(0.0, 0.1, 0.0, active=True)
    for k in range(10, 20):
        robotd.move(0.4, 0.0, 0.3, k * R.DT)
        step = robotd.tick(sensors(), k * R.DT)
        assert np.allclose(policy.walk.seen[-1][48:51], 0.0)
        assert step.label == "walk"  # no stand network: the walk network drives (no handover)
    assert robotd.twist_ema[0] > 0.39  # kept integrating the client's twist
    robotd.pose(0.0, 0.0, 0.0, active=False)
    robotd.move(0.4, 0.0, 0.3, 20 * R.DT)
    robotd.tick(sensors(), 20 * R.DT)
    assert policy.walk.seen[-1][48] > 0.39  # walking resumes from the EMA at once


def test_with_a_stand_network_pose_mode_hands_over_as_robotd_does() -> None:
    robotd, policy = make(stand=True)
    robotd.pose(0.0, 0.1, 0.0, active=True)
    step = robotd.tick(sensors(), 0.0)
    assert step.label == "stand" and len(policy.stand.seen) == 1 and not policy.walk.seen


def test_the_head_ema_uses_head_alpha_and_the_hook_post_processes_targets() -> None:
    robotd, _ = make(head_alpha=0.2)
    robotd.head((0.0, 0.0, 1.0, 0.0))
    robotd.tick(sensors(), 0.0)
    assert np.isclose(robotd.head_ema[2], 0.2)
    robotd.target_hook = lambda t: np.where(np.arange(14) == 7, 0.5, t)
    step = robotd.tick(sensors(), R.DT)
    assert np.isclose(step.targets[7], 0.5)


def test_body_block_helper() -> None:
    assert np.allclose(S.body_block(-0.01, 0.1, -0.2), [0, 0, -0.01, 0.1, -0.2, 0])
