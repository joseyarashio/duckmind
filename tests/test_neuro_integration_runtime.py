"""Integrated motor keeps compatible play legs during explicitly head-only reflexes."""

from types import SimpleNamespace

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyRequest
from duckmind.simweb.neuro_runtime import IntegratedMotor


def _state():
    joints = [0.0] * 15
    for index, home in zip((5, 6, 7, 8), (0.3491, 0.3491, 0.0, 0.0), strict=True):
        joints[index] = home
    return RobotState.from_wire(
        {
            "t": 0.0,
            "t_ns": 0,
            "policy": "walk",
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0], "limited_by": []},
            "safety": {"fallen": False, "limp": False, "gravity": [0, 0, -1]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [0.0] * 15,
            "odom": {"position": [0, 0, 0], "yaw": 0},
            "imu": {"gyro": [0, 0, 0], "quat": [1, 0, 0, 0]},
            "head": [0, 0, 0, 0],
        }
    )


@pytest.mark.parametrize("reflex", ["orient", "lean_in"])
def test_integrated_motor_keeps_legs_but_not_reflex_owned_outputs(reflex):
    class Base:
        def __call__(self, t_ms, cmd, frame, state):
            return BodyRequest(
                vx=0.0, vyaw=0.0, head=(0.1, 0.2, 0.3, 0.0), mouth=0.2, posture="stand"
            ), []

    play = SimpleNamespace(
        drive=True,
        far=False,
        mouth=0.9,
        posture="sit",
        fine=True,
        skill="kick_left",
        skill_seq=4,
        phase="chase",
        event=None,
    )
    brain = SimpleNamespace(play_out=play)
    motor = IntegratedMotor(Base(), brain)
    cmd = C.MotorCommand(
        t_ms=0,
        behaviour="play_ball",
        bout=1,
        variant=1,
        vx=0.4,
        vyaw=-0.15,
        approach=True,
        reflex=reflex,
    )
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    request, _ = motor(0, cmd, frame, _state())
    assert (request.vx, request.vyaw, request.approach) == (0.4, -0.15, True)
    assert (request.head, request.mouth, request.posture) == ((0.1, 0.2, 0.3, 0.0), 0.2, "stand")
    assert motor.last["fine"] is True
    assert (motor.last["skill"], motor.last["skill_seq"]) == (None, 0)


def test_integrated_motor_does_not_apply_play_during_other_reflex():
    class Base:
        def __call__(self, t_ms, cmd, frame, state):
            return BodyRequest(vx=0.0, vyaw=0.0), []

    brain = SimpleNamespace(
        play_out=SimpleNamespace(
            drive=True,
            far=False,
            mouth=None,
            posture=None,
            fine=True,
            skill="kick_left",
            skill_seq=4,
            phase="kick",
            event=None,
        )
    )
    motor = IntegratedMotor(Base(), brain)
    cmd = C.MotorCommand(t_ms=0, behaviour="play_ball", bout=1, variant=1, vx=0.4, reflex="veer")
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    request, _ = motor(0, cmd, frame, _state())
    assert (request.vx, request.vyaw) == (0.0, 0.0)
    assert motor.last["fine"] is False
    assert (motor.last["skill"], motor.last["skill_seq"]) == (None, 0)
