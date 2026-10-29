"""robotd's skill path, ported (:mod:`duckmind.skilltrain.robotd_port`), against control.rs.

No ONNX here: a fake policy set records what each network saw. The semantics checked are the
ones the Stage-1 gates and the skill table rely on — zeroed head/body, the constant command
then the unwind, pre-emption, the chain (and that a chain with an unwind never restarts), the
re-seat, pinned vs default scale and gain, the shared last action, the deadman and the EMAs
running on under a skill, recurrent memory resets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.skilltrain import robotd_port as R
from duckmind.skilltrain.table import SkillDef


class FakeNet:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value
        self.seen: list[np.ndarray] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(obs, float).copy())
        return np.full(14, self.value, dtype=np.float32)


class FakePolicy(R.PolicySet):
    def __init__(self, n_skills: int, stand: bool = False) -> None:  # noqa: D107
        self.walk = FakeNet("walk", 0.01)
        self.stand = FakeNet("stand", 0.02) if stand else None
        self.sitstand = FakeNet("sitstand", 0.03)
        self.skills = [FakeNet(f"skill{i}", 0.1 * (i + 1)) for i in range(n_skills)]
        self.active = None


def sensors() -> R.Sensors:
    return R.Sensors(np.zeros(3), np.array([0.0, 0.0, -1.0]), np.array(R.HOME), np.zeros(14))


def make(skills: list[SkillDef], stand: bool = False) -> tuple[R.Robotd, FakePolicy]:
    policy = FakePolicy(len(skills), stand=stand)
    robotd = R.Robotd(policy, skills, head_alpha=1.0)
    robotd.boot(seated=False)
    return robotd, policy


def run(robotd: R.Robotd, ticks: int, t0: float = 0.0, **moving: float) -> list[R.Step]:
    out = []
    for k in range(ticks):
        now = t0 + k * R.DT
        if moving:
            robotd.move(moving.get("vx", 0.0), 0.0, moving.get("vyaw", 0.0), now)
        out.append(robotd.tick(sensors(), now))
    return out


def sd(
    name: str,
    duration: float = 0.5,
    unwind_s: float = 0.0,
    chain: bool = False,
    command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    scale: float | None = 1.0,
    gain: float | None = 1.0,
) -> SkillDef:
    return SkillDef(
        name, "/x.onnx", duration, command, (0.0, 0.0, 0.0), unwind_s, chain, scale, gain
    )


def test_a_skill_sees_its_constant_command_with_head_and_body_zeroed_then_its_unwind() -> None:
    robotd, policy = make([sd("crouch", 0.5, 0.8, command=(0.75, 0.0, 0.0))])
    robotd.head((0.3, -0.2, 0.4, 0.1))
    run(robotd, 5, vx=0.4)
    assert robotd.request("crouch") == (True, "queued")
    steps = run(robotd, 80, t0=0.1, vx=0.4)
    labels = [s.label for s in steps]
    assert labels[:25] == ["crouch"] * 25  # 0.5 s at 50 Hz
    assert labels[25:65] == ["crouch:unwind"] * 40  # 0.8 s
    assert labels[65] == "walk"
    hold, unwind = steps[0].command, steps[30].command
    assert hold.twist == (0.75, 0.0, 0.0) and unwind.twist == (0.0, 0.0, 0.0)
    assert hold.head == (0.0, 0.0, 0.0, 0.0) and hold.body == (0.0, 0.0, 0.0)
    obs = policy.skills[0].seen[0]
    assert obs[48:51].tolist() == pytest.approx([0.75, 0.0, 0.0])
    assert obs[51:61].tolist() == [0.0] * 10  # head 4 + body 6 are zero under a skill
    # the walk's smoothed command kept integrating underneath: walking resumes from it
    assert steps[65].command.twist[0] == pytest.approx(0.4, abs=1e-6)
    assert steps[65].command.head == pytest.approx((0.3, -0.2, 0.4, 0.1))


def test_a_skill_pre_empts_walking_at_once_and_the_first_observation_carries_walks_action() -> None:
    robotd, policy = make([sd("s")])
    run(robotd, 10, vx=0.4)
    robotd.request("s")
    step = run(robotd, 1, t0=0.2, vx=0.4)[0]
    assert step.label == "s" and step.net == ("skill", 0)
    first = policy.skills[0].seen[0]
    assert first[34:48].tolist() == pytest.approx([0.01] * 14)  # walk's last raw action


def test_chain_with_an_unwind_never_restarts_even_when_rerequested() -> None:
    robotd, _ = make([sd("c", 0.5, 0.8, chain=True)])
    robotd.request("c")
    labels = []
    for k in range(100):
        if k % 5 == 0 and k < 90:
            robotd.request("c")  # re-requested every 0.1 s, as a held button
        labels.append(robotd.tick(sensors(), k * R.DT).label)
    # the first run is exactly 0.5 s + 0.8 s whatever the re-requests …
    assert labels[:25] == ["c"] * 25 and labels[25:65] == ["c:unwind"] * 40
    assert labels[65] == "walk"
    kinds = [e["kind"] for e in robotd.controller.events]
    assert "chain_restart" not in kinds
    assert kinds.count("refreshed") >= 10  # … which were refreshed and ignored
    # … and a request after it ends starts a fresh run: held, the duck bobs down again
    assert kinds.count("started") == 2
    assert labels[65:70] == ["walk"] * 5 and labels[70] == "c"  # the next re-request, k = 70


def test_chain_without_unwind_holds_while_rerequested_and_ends_after() -> None:
    robotd, policy = make([sd("c", 0.5, 0.0, chain=True)])
    robotd.request("c")
    labels = []
    for k in range(150):
        if k % 5 == 0 and k < 100:
            robotd.request("c")
        labels.append(robotd.tick(sensors(), k * R.DT).label)
    held = labels.count("c")
    assert 100 <= held <= 125, held  # 2 s of requests: held to the window after the last one
    assert labels[-1] == "walk"
    restarts = [e for e in robotd.controller.events if e["kind"] == "chain_restart"]
    assert len(restarts) >= 3
    assert policy.walk.resets >= 1  # a chained replay resets every network's memory


def test_a_non_chaining_request_is_refused_while_a_skill_runs_and_a_chaining_one_pre_empts() -> (
    None
):
    long = sd("long", 2.0, 0.8)
    other = sd("other", 0.5)
    abort = sd("abort", 0.02, 0.0, chain=True, command=(0.0, 0.0, 0.0), scale=0.9)
    robotd, _ = make([long, other, abort])
    robotd.request("long")
    run(robotd, 10)
    robotd.request("other")
    assert robotd.tick(sensors(), 0.2).label == "long"
    refused = [e for e in robotd.controller.events if e["kind"] == "refused"]
    assert refused and refused[-1]["skill"] == "other"
    robotd.request("abort")
    assert robotd.tick(sensors(), 0.22).label == "abort"
    assert robotd.tick(sensors(), 0.24).label == "walk"  # one tick, then velstand


def test_a_seated_duck_keeps_its_seat_under_a_skill_and_is_reseated_after() -> None:
    robotd, _ = make([sd("s", 0.5, 0.8)])
    robotd.request("sit_toggle")
    assert robotd.tick(sensors(), 0.0).label == "sit"
    robotd.request("s")
    labels = [robotd.tick(sensors(), 0.02 * (k + 1)).label for k in range(70)]
    assert labels[0] == "s" and labels[24] == "s" and labels[25] == "s:unwind"
    assert labels[64] == "s:unwind" and labels[65] == "sit"
    step = robotd.last
    assert step is not None and step.command.twist == (1.0, 0.0, 0.0)  # the posture flag


def test_scale_and_gain_pinned_else_walking_tuning_without_a_stand_network() -> None:
    robotd, _ = make([sd("pinned", scale=1.0, gain=0.8), sd("free", scale=None, gain=None)])
    robotd.request("pinned")
    s = run(robotd, 1)[0]
    assert s.scale == pytest.approx(1.0) and s.gain == 160
    run(robotd, 30, t0=0.02)
    robotd.request("free")
    s = run(robotd, 1, t0=0.7)[0]
    assert s.label == "free" and s.scale == pytest.approx(0.9) and s.gain == 200
    # with a stand network and a zero command, an unpinned skill runs at standing tuning
    robotd2, _ = make([sd("free", command=(0.0, 0.0, 0.0), scale=None, gain=None)], stand=True)
    robotd2.request("free")
    s2 = run(robotd2, 1)[0]
    assert s2.scale == pytest.approx(1.0) and s2.gain == 160


def test_voltage_adaptation_scales_the_action_and_targets_are_clamped() -> None:
    robotd, _ = make([sd("s", scale=1.0)])
    robotd.request("s")
    step = robotd.tick(sensors(), 0.0, battery_v=6.5)
    assert step.scale == pytest.approx(7.4 / 6.5)
    assert np.all(np.abs(step.targets) <= math.pi)


def test_the_deadman_zeroes_a_stale_twist_but_keeps_the_head() -> None:
    robotd, _ = make([])
    robotd.move(0.4, 0.0, 0.0, 0.0)
    robotd.head((0.1, 0.0, 0.0, 0.0))
    steps = [robotd.tick(sensors(), k * R.DT) for k in range(60)]
    assert steps[20].command.twist[0] > 0.39  # fresh: the EMA reached it
    assert steps[-1].command.twist[0] < 0.05  # 1.2 s after the last move: zeroed and decayed
    assert steps[-1].command.head[0] == pytest.approx(0.1)


def test_unknown_names_are_refused_by_the_dispatcher_and_nothing_runs_before_boot() -> None:
    policy = FakePolicy(1)
    robotd = R.Robotd(policy, [sd("s")])
    assert robotd.request("s")[0] is False  # not driving yet
    robotd.boot(seated=False)
    ok, why = robotd.request("roulade")
    assert not ok and "no skill named" in why
    with pytest.raises(ValueError):
        R.Robotd(policy, [sd("s")], head_alpha=0.0)


def test_a_seated_boot_rises_through_sitstand_for_one_second() -> None:
    policy = FakePolicy(0)
    robotd = R.Robotd(policy, [])
    robotd.boot(seated=True)
    labels = [robotd.tick(sensors(), k * R.DT).label for k in range(60)]
    assert labels[:50] == ["rise"] * 50 and labels[50] == "walk"
