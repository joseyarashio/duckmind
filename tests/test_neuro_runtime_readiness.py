"""Runtime preparation and pending-action lifecycle; no server or physics is started."""

from __future__ import annotations

from types import SimpleNamespace

from duckmind.duckbody.proprio import ProprioOut
from duckmind.duckbody.protocol import ProprioView
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyRequest, LinkConfig, LinkOut
from duckmind.simweb.neuro_runtime import IntegratedMotor, OneShotBridge


def robot_state(t_ms: int, *, policy: str = "sit", head=(0.2, 0.0, 0.0, 0.0)) -> RobotState:
    joints = [0.0] * 15
    for index, home in zip((5, 6, 7, 8), (0.3491, 0.3491, 0.0, 0.0), strict=True):
        joints[index] = home + head[index - 5]
    return RobotState.from_wire(
        {
            "t": t_ms / 1000,
            "t_ns": t_ms * 1_000_000,
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0], "limited_by": []},
            "safety": {"fallen": False, "limp": False, "gravity": [0, 0, -1]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [0.0] * 15,
            "odom": {"position": [0, 0, 0], "yaw": 0},
            "imu": {"gyro": [0, 0, 0], "quat": [1, 0, 0, 0]},
            "head": list(head),
        }
    )


def test_one_shot_preparation_requests_home_and_stops_only_when_safe():
    class Base:
        def __call__(self, t_ms, cmd, frame, state):
            return BodyRequest(vx=0.3, vyaw=0.4, head=(0.4, -0.3, 0.2, 0.0), mouth=0.8), []

    brain = SimpleNamespace(
        play_out=SimpleNamespace(
            skill="ground_pick",
            skill_seq=1,
            drive=True,
            far=False,
            mouth=0.7,
            posture=None,
            fine=False,
            phase="grab",
            event=None,
        )
    )
    motor = IntegratedMotor(Base(), brain)
    cmd = C.MotorCommand(t_ms=0, behaviour="carry_toy", bout=1, variant=1)
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    request, _ = motor(0, cmd, frame, robot_state(0))
    assert (request.vx, request.vyaw, request.head) == (0.0, 0.0, None)
    assert request.mouth == 0.0  # no jaw timing before robotd policy acknowledgement

    reflex = C.MotorCommand(t_ms=20, behaviour="carry_toy", bout=1, variant=1, reflex="withdraw")
    untouched, _ = motor(20, reflex, frame, robot_state(20))
    assert (untouched.vx, untouched.vyaw, untouched.head) == (0.3, 0.4, (0.4, -0.3, 0.2, 0.0))

    unsafe = C.SensorFrame(t_ms=40, body=C.BodySense(posture="unknown", fallen=True))
    untouched, _ = motor(40, cmd, unsafe, robot_state(40, policy="held"))
    assert (untouched.vx, untouched.vyaw, untouched.head) == (0.3, 0.4, (0.4, -0.3, 0.2, 0.0))


def test_withdrawn_pending_request_is_closed_but_active_policy_is_still_tracked():
    bridge = OneShotBridge()
    cmd = C.MotorCommand(t_ms=0, behaviour="carry_toy", bout=1, variant=1)
    brain = SimpleNamespace(play_out=SimpleNamespace(skill="ground_pick", skill_seq=1))
    link = SimpleNamespace(cfg=LinkConfig(), righting=SimpleNamespace(active=False))
    out = LinkOut(intents=(), prox=ProprioOut(view=ProprioView()), gate=(), twist=(0.0, 0.0))
    bridge.step(0, robot_state(0), BodyRequest(), cmd, brain, link, out, battery_pct=100)
    assert bridge.pending is not None
    brain.play_out = SimpleNamespace(skill=None, skill_seq=0)
    bridge.step(20, robot_state(20), BodyRequest(), cmd, brain, link, out, battery_pct=100)
    assert bridge.pending is None
    assert bridge.records[0].status == "selection_ended"

    # An already dispatched action remains observable even after the brain withdraws it.
    bridge = OneShotBridge()
    bridge.still_since_ns = bridge.home_since_ns = 0
    brain.play_out = SimpleNamespace(skill="ground_pick", skill_seq=1)
    bridge.step(
        300,
        robot_state(300, policy="stand", head=(0, 0, 0, 0)),
        BodyRequest(),
        cmd,
        brain,
        link,
        out,
        battery_pct=100,
    )
    assert bridge.active is not None
    brain.play_out = SimpleNamespace(skill=None, skill_seq=0)
    bridge.step(
        320,
        robot_state(320, policy="ground_pick", head=(0, 0, 0, 0)),
        BodyRequest(),
        cmd,
        brain,
        link,
        out,
        battery_pct=100,
    )
    assert bridge.active is not None and bridge.active.started_ms == 320


def test_measured_home_during_transition_policy_does_not_dispatch():
    bridge = OneShotBridge()
    cmd = C.MotorCommand(t_ms=0, behaviour="carry_toy", bout=1, variant=1)
    brain = SimpleNamespace(play_out=SimpleNamespace(skill="ground_pick", skill_seq=1))
    link = SimpleNamespace(cfg=LinkConfig(), righting=SimpleNamespace(active=False))
    out = LinkOut(intents=(), prox=ProprioOut(view=ProprioView()), gate=(), twist=(0.0, 0.0))
    bridge.still_since_ns = bridge.home_since_ns = 0
    result = bridge.step(
        300,
        robot_state(300, policy="homing", head=(0, 0, 0, 0)),
        BodyRequest(),
        cmd,
        brain,
        link,
        out,
        battery_pct=100,
    )
    assert not any(intent.method == "robot.do" for intent in result.intents)
    assert bridge.pending is not None and bridge.pending.status == "waiting_safety"
    assert bridge.pending.reason == "policy_unsettled"


def test_sit_to_stand_request_waits_for_measured_stand_even_at_home():
    bridge = OneShotBridge()
    cmd = C.MotorCommand(t_ms=0, behaviour="carry_toy", bout=1, variant=1)
    brain = SimpleNamespace(play_out=SimpleNamespace(skill="ground_pick", skill_seq=1))
    link = SimpleNamespace(cfg=LinkConfig(), righting=SimpleNamespace(active=False))
    out = LinkOut(intents=(), prox=ProprioOut(view=ProprioView()), gate=(), twist=(0.0, 0.0))
    bridge.still_since_ns = bridge.home_since_ns = 0
    result = bridge.step(
        300,
        robot_state(300, policy="sit", head=(0, 0, 0, 0)),
        BodyRequest(posture="stand"),
        cmd,
        brain,
        link,
        out,
        battery_pct=100,
    )
    assert not any(intent.method == "robot.do" for intent in result.intents)
    assert bridge.pending is not None and bridge.pending.reason == "policy_unsettled"
