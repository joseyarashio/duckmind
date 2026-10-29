"""Pure tests for the guarded live one-shot bridge; no server or physics is started."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from duckmind.duckbody.proprio import ProprioOut
from duckmind.duckbody.protocol import ProprioView
from duckmind.duckbody.robotd_client import V5_HOME_HEAD, RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyRequest, LinkConfig, LinkOut
from duckmind.simweb.neuro_runtime import OneShotBridge


def state(t_ms: int, *, policy: str = "sit", applied=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0, 0.0)):
    joints = [0.0] * 15
    for index, home in zip((5, 6, 7, 8), V5_HOME_HEAD, strict=True):
        joints[index] = home + head[index - 5]
    return RobotState.from_wire(
        {
            "t": t_ms / 1000,
            "t_ns": t_ms * 1_000_000,
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": list(applied), "limited_by": []},
            "safety": {"fallen": False, "limp": False, "gravity": [0, 0, -1]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [0.0] * 15,
            "odom": {"position": [0, 0, 0], "yaw": 0},
            "imu": {"gyro": [0, 0, 0], "quat": [1, 0, 0, 0]},
            "head": list(head),
        }
    )


def setup(
    *,
    skill="kick_left",
    seq=1,
    behaviour="play_ball",
    reflex=None,
    posture="sit",
    fine=True,
    gate=(),
    unloaded=False,
    lifted=False,
    fallen=False,
    battery=100,
    policy="sit",
    out_twist=(0.0, 0.0),
    applied=(0.0, 0.0, 0.0),
    head=(0.0, 0.0, 0.0, 0.0),
    righting=False,
):
    bridge = OneShotBridge()
    cmd = C.MotorCommand(t_ms=0, behaviour=behaviour, bout=1, variant=1, reflex=reflex)
    brain = SimpleNamespace(play_out=SimpleNamespace(skill=skill, skill_seq=seq))
    linkout = LinkOut(
        intents=(),
        prox=ProprioOut(view=ProprioView(unloaded=unloaded, lifted=lifted, fallen=fallen)),
        gate=tuple(gate),
        twist=out_twist,
    )
    bodylink = SimpleNamespace(cfg=LinkConfig(), righting=SimpleNamespace(active=righting))
    return bridge, cmd, brain, linkout, bodylink


def tick(
    bridge,
    cmd,
    brain,
    out,
    bodylink,
    t_ms,
    *,
    battery=100,
    policy="stand",
    applied=(0.0, 0.0, 0.0),
    head=(0.0, 0.0, 0.0, 0.0),
):
    return bridge.step(
        t_ms,
        state(t_ms, policy=policy, applied=applied, head=head),
        BodyRequest(),
        cmd,
        brain,
        bodylink,
        out,
        battery_pct=battery,
    )


def test_ball_and_carry_sequence_one_are_distinct_requests():
    bridge, cmd, brain, out, link = setup()
    bridge.still_since_ns = bridge.home_since_ns = 0
    tick(bridge, cmd, brain, out, link, 300)
    brain.play_out.skill = "ground_pick"
    cmd = replace(cmd, behaviour="carry_toy")
    tick(bridge, cmd, brain, out, link, 1)
    assert [(r.behaviour, r.skill, r.seq) for r in bridge.records] == [
        ("play_ball", "kick_left", 1),
        ("carry_toy", "ground_pick", 1),
    ]


def test_waiting_for_body_and_head_retains_request_until_settled_then_dispatches_once():
    bridge, cmd, brain, out, link = setup(applied=(0.2, 0, 0), head=(0.4, 0, 0, 0))
    tick(bridge, cmd, brain, out, link, 0, applied=(0.2, 0, 0), head=(0.4, 0, 0, 0))
    assert bridge.pending is not None
    assert bridge.records[0].status in {"waiting_safety", "waiting_policy"}
    tick(bridge, cmd, brain, out, link, 300)
    result = tick(bridge, cmd, brain, out, link, 600)
    assert bridge.active is bridge.records[0]
    assert bridge.active.dispatched_ms == 600
    assert bridge.active.reason is None
    assert bridge.active.reason_detail is None
    assert sum(i.method == "robot.do" for i in result.intents) == 1
    again = tick(bridge, cmd, brain, out, link, 620)
    assert not any(i.method == "robot.do" for i in again.intents)
    assert len(bridge.records) == 1


def test_started_timestamp_is_set_once_and_exit_is_observed_from_policy():
    bridge, cmd, brain, out, link = setup()
    bridge.still_since_ns = bridge.home_since_ns = 0
    tick(bridge, cmd, brain, out, link, 300)
    assert bridge.active
    tick(bridge, cmd, brain, out, link, 320, policy="robotd.kick_left")
    assert bridge.records[0].reason is None
    assert bridge.records[0].reason_detail is None
    started = bridge.records[0].started_ms
    tick(bridge, cmd, brain, out, link, 340, policy="robotd.kick_left")
    assert bridge.records[0].started_ms == started == 320
    tick(bridge, cmd, brain, out, link, 360, policy="robotd.kick_left:unwind")
    assert bridge.records[0].status == "policy_started"
    assert not bridge.records[0].policy_exit_observed
    tick(bridge, cmd, brain, out, link, 380, policy="walk")
    assert bridge.records[0].policy_exit_observed
    assert bridge.records[0].exited_ms == 380
    assert bridge.records[0].outcome is None  # policy exit is not object success


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unloaded": True},
        {"lifted": True},
        {"fallen": True},
        {"righting": True},
        {"battery": 10},
        {"reflex": "withdraw"},
        {"gate": ("fallen",)},
    ],
)
def test_safety_gates_never_dispatch_one_shot(kwargs):
    battery = kwargs.pop("battery", 100)
    bridge, cmd, brain, out, link = setup(**kwargs)
    tick(bridge, cmd, brain, out, link, 0, battery=battery)
    assert not bridge.active
    assert not any(i.method == "robot.do" for i in out.intents)


def test_busy_new_request_does_not_replace_active_record_or_hide_its_exit():
    bridge, cmd, brain, out, link = setup()
    bridge.still_since_ns = bridge.home_since_ns = 0
    tick(bridge, cmd, brain, out, link, 300)
    first = bridge.active
    tick(bridge, cmd, brain, out, link, 320, policy="kick_left")
    brain.play_out.skill, brain.play_out.skill_seq = "ground_pick", 2
    tick(bridge, cmd, brain, out, link, 340, policy="kick_left")
    assert bridge.active is first
    assert first.started_ms == 320
    assert bridge.pending is not None and bridge.pending.skill == "ground_pick"
    tick(bridge, cmd, brain, out, link, 360, policy="sit")
    assert first.policy_exit_observed and first.exited_ms == 360


def test_interruption_at_time_zero_is_recorded_and_waits_for_uncancelable_policy_exit():
    bridge, cmd, brain, out, link = setup()
    bridge.still_since_ns = bridge.home_since_ns = -300_000_000
    tick(bridge, cmd, brain, out, link, 0)
    tick(bridge, cmd, brain, out, link, 0, policy="kick_left")
    cmd = replace(cmd, reflex="withdraw")
    tick(bridge, cmd, brain, out, link, 0, policy="kick_left")
    assert bridge.active.interrupted_ms == 0
    assert bridge.active.status == "interrupted_waiting_for_exit"
    tick(bridge, cmd, brain, out, link, 340, policy="sit")
    assert bridge.records[0].status == "interrupted_policy_exited"


def test_motor_command_velocity_is_not_replaced_by_raw_play_velocity():
    from duckmind.simweb.neuro_runtime import IntegratedMotor

    class Base:
        def __call__(self, t_ms, cmd, frame, robot):
            return BodyRequest(vx=cmd.vx, vyaw=cmd.vyaw), ()

    brain = SimpleNamespace(
        play_out=SimpleNamespace(
            drive=True,
            far=False,
            mouth=None,
            posture=None,
            fine=False,
            skill=None,
            skill_seq=0,
            phase="",
            event=None,
        )
    )
    motor = IntegratedMotor(Base(), brain)
    cmd = C.MotorCommand(t_ms=0, behaviour="play_ball", bout=1, variant=1, vx=0.4, vyaw=0.3)
    frame = C.SensorFrame(t_ms=0)
    request, _ = motor(0, cmd, frame, state(0))
    assert (request.vx, request.vyaw) == pytest.approx((0.4, 0.3))
