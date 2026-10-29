"""body.sock (duckmind.duckbody.protocol): commands, lines, the posture-tap wire form."""

from __future__ import annotations

import pytest

from duckmind.duckbody import protocol as P


def cmd(**fields: object) -> dict:
    return {"type": "cmd", "act_id": "A1", "deadline_ms": 10_000, **fields}


def test_walk_command_defaults_and_fields() -> None:
    walk = P.parse_command(cmd(cmd="duck.walk", vx_mm=400))
    assert isinstance(walk, P.WalkCmd)
    assert (walk.vyaw_mrad, walk.max_ms, walk.pre_look, walk.stop_ahead_mm) == (
        0,
        3000,
        "auto",
        300,
    )
    assert walk.head == "home" and walk.toward is None and not walk.approach
    toward = {"look_id": "L1", "b_mrad": -40, "r_mm": 600, "seen_t_ms": 9000, "thing": "ball"}
    walk = P.parse_command(
        cmd(cmd="duck.walk", vx_mm=400, purpose="curio", head="down", toward=toward, curio_id="C2")
    )
    assert isinstance(walk, P.WalkCmd) and walk.approach and walk.toward is not None
    assert walk.toward.thing == "ball" and walk.head == "down"
    wire = P.command_to_wire(walk)
    assert P.parse_command(wire) == walk


def test_walk_attention_is_the_callers_and_on_by_default() -> None:
    """Owner 2026-09-28 ~12:30 「呆萌感」: the caller says where its attention is; ``on`` (Duck
    Core v4's walks) keeps the wire line as it was."""
    plain = P.parse_command(cmd(cmd="duck.walk", vx_mm=400))
    assert isinstance(plain, P.WalkCmd) and plain.attention == "on"
    assert "attention" not in P.command_to_wire(plain)
    for attention in ("low", "elsewhere"):
        walk = P.parse_command(cmd(cmd="duck.walk", vx_mm=400, attention=attention))
        assert isinstance(walk, P.WalkCmd) and walk.attention == attention
        wire = P.command_to_wire(walk)
        assert wire["attention"] == attention and P.parse_command(wire) == walk
    with pytest.raises(P.ProtocolError):
        P.parse_command(cmd(cmd="duck.walk", vx_mm=400, attention="dreamy"))


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"cmd": "duck.fly"}, "invalid"),
        ({"cmd": "duck.walk"}, "invalid"),  # no vx_mm
        ({"cmd": "duck.walk", "vx_mm": 0.4}, "invalid"),  # floats are not wire integers
        ({"cmd": "duck.walk", "vx_mm": 400, "pre_look": "peek"}, "invalid"),
        ({"cmd": "duck.posture", "want": "lie"}, "invalid"),
        ({"cmd": "duck.look", "kind": "stare"}, "invalid"),
        ({"cmd": "duck.dock", "style": "sideways"}, "invalid"),
        ({"cmd": "duck.walk", "vx_mm": 400, "act_id": "bad id!"}, "invalid"),
    ],
)
def test_bad_commands_are_refused(fields: dict, reason: str) -> None:
    with pytest.raises(P.ProtocolError) as problem:
        P.parse_command(cmd(**fields))
    assert problem.value.reason == reason


def test_every_command_parses() -> None:
    assert isinstance(P.parse_command(cmd(cmd="duck.look", kind="scan", want="nest")), P.LookCmd)
    assert isinstance(P.parse_command(cmd(cmd="duck.posture", want="sit")), P.PostureCmd)
    dock = P.parse_command(cmd(cmd="duck.dock", style="backin"))
    assert isinstance(dock, P.DockCmd) and dock.style == "backin"
    curio = P.parse_command(cmd(cmd="duck.curio", phase="step", step=2, params={"standoff_mm": 60}))
    assert isinstance(curio, P.CurioCmd) and curio.params["standoff_mm"] == 60
    stop = P.parse_command({"type": "cmd", "cmd": "duck.stop", "why": "petted"})
    assert isinstance(stop, P.StopCmd) and stop.why == "petted"
    express = P.parse_command({"type": "cmd", "cmd": "duck.sound", "name": "quack"})
    assert isinstance(express, P.ExpressCmd)


def test_server_lines() -> None:
    hello = P.hello(pv="p:0123456789abcdef", src="live", t_ms=5, unix_ms=6, caps={"tof": True})
    assert hello["api"] == "body-v1" and hello["percept"] == "percept-v1"
    assert P.ack("A1", True, 5) == {
        "type": "ack",
        "act_id": "A1",
        "ok": True,
        "reason": None,
        "t_ms": 5,
    }
    with pytest.raises(P.ProtocolError):
        P.ack("A1", False, 5, "because")
    assert P.decode_line(P.encode_line(P.done("A1", "walk.done", 9)))["kind"] == "walk.done"
    with pytest.raises(P.ProtocolError):
        P.decode_line(b"[1, 2]")


def test_intents() -> None:
    move = P.Intent.move(0.4, -0.2)
    assert move.method == "robot.move" and move.twist == (0.4, -0.2)
    assert P.Intent.head((0.1, 0.0, 0.2, 0.0)).params["head_yaw"] == 0.2
    assert P.Intent.do("sit_toggle").params == {"skill": "sit_toggle"}


def sample() -> P.PoseSample:
    return P.PoseSample(
        seq=12,
        t_ns=123_456_789,
        trunk_height_mm=117,
        camera_pos=(0.05, 0.0, 0.12),
        camera_quat=(0.5, -0.5, 0.5, -0.5),
        tof_pos=(0.05, 0.02, 0.12),
        tof_quat=(1.0, 0.0, 0.0, 0.0),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=(0.0, 0.01, 0.0),
        head=(0.35, 0.35, 0.0, 0.0),
        head_vel=0.05,
        odom=(1.0, 2.0, 0.5),
        epoch=3,
        still=True,
        policy="walk",
    )


def test_pose_sample_round_trips() -> None:
    s = sample()
    wire = s.to_wire()
    assert wire["type"] == "eye.pose"
    assert wire["camera_in_trunk"] == {"pos": [0.05, 0.0, 0.12], "quat": [0.5, -0.5, 0.5, -0.5]}
    assert P.PoseSample.from_wire(wire) == s


def test_duck_eye_reads_the_pose_line() -> None:
    eye_protocol = pytest.importorskip("duckmind.eye.protocol")
    got = eye_protocol.PoseSample.from_json(sample().to_wire())
    assert got.seq == 12 and got.t_ns == 123_456_789 and got.trunk_height_mm == 117
    assert got.cam_pos_m == (0.05, 0.0, 0.12)
    assert got.head == (0.35, 0.35, 0.0, 0.0)
