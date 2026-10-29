"""eye.protocol: the eye-v1 lines duck-body and duck-eye say to each other (spec §4.3, §6)."""

from __future__ import annotations

import json

import pytest

from duckmind.eye import protocol as p


def _look(**kw):
    base = dict(look_id="L7710", tasks=("floor", "light"), pose_tap_ref=41, not_before_ns=5_000)
    base.update(kw)
    return p.EyeLook(**base)


def test_a_look_goes_over_the_wire_and_back_unchanged():
    look = _look(
        kind="glance_down",
        attempt=1,
        deadline_ns=9_000,
        mode="outing",
        hints={"nest_b_mrad": -140, "target_id": "ball"},
    )
    line = p.encode(look.to_json())
    assert line.endswith(b"\n") and b"\n" not in line[:-1] and line.isascii()
    back = p.parse_to_eye(p.decode(line))
    assert back == look
    assert look.runs == ("light", "floor")  # light always runs, first
    assert look.retake().attempt == 2 and look.retake().look_id == look.look_id


@pytest.mark.parametrize(
    "change,field",
    [
        ({"look_id": "has space"}, "look_id"),
        ({"look_id": ""}, "look_id"),
        ({"tasks": []}, "tasks"),
        ({"tasks": ["floor", "floor"]}, "repeats"),
        ({"tasks": ["teleport"]}, "tasks"),
        ({"kind": "stare"}, "kind"),
        ({"attempt": 3}, "attempt"),
        ({"pose_tap_ref": -1}, "pose_tap_ref"),
        ({"not_before_ns": 1.5}, "not_before_ns"),
        ({"not_before_ns": True}, "not_before_ns"),
        ({"mode": "party"}, "mode"),
        ({"hints": {"b": 0.5}}, "float"),
    ],
)
def test_a_look_that_is_not_eye_v1_names_its_field(change, field):
    message = _look().to_json()
    message.update(change)
    with pytest.raises(p.ProtocolError, match=field):
        p.parse_to_eye(p.decode(p.encode(message)))


def test_a_look_built_in_code_is_held_to_the_same_rules():
    with pytest.raises(p.ProtocolError, match="float"):
        _look(hints={"nest_b_mrad": -0.14})
    with pytest.raises(p.ProtocolError, match="kind"):
        _look(kind="stare")


def test_lines_are_json_objects_with_a_known_type_and_no_nan():
    with pytest.raises(p.ProtocolError) as caught:
        p.decode(b"[1,2]\n")
    assert caught.value.code == "parse"
    with pytest.raises(p.ProtocolError) as caught:
        p.decode(b'{"type":"eye.teleport"}\n')
    assert caught.value.code == "unknown_type"
    with pytest.raises(p.ProtocolError):
        p.decode(b'{"type":"eye.health","cpu_temp_c":NaN}\n')
    with pytest.raises(p.ProtocolError) as caught:
        p.decode(b"x" * (p.MAX_LINE_BYTES + 1))
    assert caught.value.code == "too_long"
    with pytest.raises(p.ProtocolError):
        p.encode({"type": p.LEARN, "floor": [[0, 0]] * 20000})


def test_the_pose_sample_reads_robotd_shapes_and_integer_shapes():
    robotd = {
        "type": "eye.pose",
        "seq": 7,
        "t_ns": 123_000_000,
        "trunk_height_mm": 212,
        "camera": {"pos": [0.05, 0.0, 0.1], "quat": [0.5, -0.5, 0.5, -0.5]},
        "imu": {"quat": [2.0, 0.0, 0.0, 0.0]},  # normalised on the way in
        "head_joints": [0.1, -0.2, 0.0, 0.0],
    }
    a = p.PoseSample.from_json(robotd)
    assert a.imu_quat == (1.0, 0.0, 0.0, 0.0) and a.head == (0.1, -0.2, 0.0, 0.0)
    ints = {
        "type": "eye.pose",
        "sample": {
            "seq": 7,
            "t_ns": 123_000_000,
            "trunk_height_m": 0.212,
            "cam_pos_mm": [50, 0, 100],
            "cam_quat_e6": [500000, -500000, 500000, -500000],
            "imu_quat_e6": [1000000, 0, 0, 0],
            "head_mrad": [100, -200, 0, 0],
        },
    }
    b = p.PoseSample.from_json(ints)
    assert b.trunk_height_mm == 212 and b.cam_pos_m == pytest.approx(a.cam_pos_m)
    assert b.cam_quat == pytest.approx(a.cam_quat) and b.head == pytest.approx(a.head)
    assert p.PoseSample.from_json(p.decode(p.encode(a.to_json()))) == a
    for incomplete in (
        {"t_ns": 1, "trunk_height_mm": 200},
        {"t_ns": 1, "trunk_height_mm": None, "camera_in_trunk": robotd["camera"]},
        {"t_ns": 1, "trunk_height_mm": 200, "camera_in_trunk": None},
    ):
        with pytest.raises(p.ProtocolError) as caught:
            p.PoseSample.from_json(incomplete)
        assert caught.value.code == "incomplete_pose"
    body_core = {
        **robotd,
        "camera_in_trunk": robotd["camera"],
        "tof_in_trunk": None,
        "gyro": [0.0, 0.0, 0.0],
        "odom": [0.0, 0.0, 0.0],
        "head_vel": None,
        "epoch": 0,
        "still": True,
        "policy": "stand",
    }
    del body_core["camera"]
    assert p.PoseSample.from_json(body_core).cam_pos_m == a.cam_pos_m  # duck-body's wire
    with pytest.raises(p.ProtocolError, match="rotation"):
        p.PoseSample.from_json({**robotd, "camera": {"pos": [0, 0, 0], "quat": [0, 0, 0, 0]}})


def test_health_learn_and_hello():
    health = p.EyeHealth.from_json(
        {
            "type": "eye.health",
            "t_ns": 5,
            "cpu_temp_c": 71.5,
            "cpu_throttle": {"level": 0, "max_level": 6, "khz": 1_800_000, "max_khz": 1_800_000},
        }
    )
    assert health.cpu_throttle is not None and not health.cpu_throttle.throttled
    assert p.CpuThrottle(level=0, max_level=6, khz=1_416_000, max_khz=1_800_000).throttled
    assert p.CpuThrottle(level=2, max_level=6).throttled and not p.CpuThrottle().throttled
    assert p.EyeHealth.from_json(p.decode(p.encode(health.to_json()))) == health
    off_linux = p.EyeHealth.from_json({"type": "eye.health", "t_ns": 5})
    assert off_linux.cpu_temp_c is None and off_linux.cpu_throttle is None
    learn = p.EyeLearn.from_json(
        {
            "type": "eye.learn",
            "walk_id": "W1",
            "look_id": "L1",
            "outcome": "clear",
            "floor": [[300, 0], [600, -20]],
            "width_mm": 140,
        }
    )
    assert learn.floor == ((300, 0), (600, -20)) and learn.contact is None
    assert p.EyeLearn.from_json(p.decode(p.encode(learn.to_json()))) == learn
    with pytest.raises(p.ProtocolError, match="outcome"):
        p.EyeLearn.from_json({**learn.to_json(), "outcome": "great"})
    with pytest.raises(p.ProtocolError) as caught:
        p.Hello.from_json({"type": "eye.hello", "api": "eye-v0"})
    assert caught.value.code == "api"
    answer = p.EyeHello(perception_hash="p:0123456789abcdef", tasks=("light", "floor"))
    assert p.EyeHello.from_json(p.decode(p.encode(answer.to_json()))) == answer


def test_a_result_is_integers_only_and_says_why():
    ok = p.EyeResult(
        look_id="L1",
        attempt=0,
        kind="end_look",
        status="ok",
        pv="p:0123abcd",
        captured_at_ns=10,
        captured_at_unix_us=20,
        pose={"t_ns": 10, "cam_pos_mm": [1, 2, 3], "interp": "lerp"},
        results={"light": {"level": "dim", "luma": 80, "noise_q": 5}},
        frame={"w": 1280, "h": 720, "rotate": 90, "sha16": "00ff"},
        cost_us={"fetch": 70000},
    )
    back = p.EyeResult.from_json(p.decode(p.encode(ok.to_json())))
    assert back == ok and back.light == {"level": "dim", "luma": 80, "noise_q": 5}
    assert ok.problems() == []
    floaty = p.EyeResult(
        look_id="L1",
        attempt=0,
        kind="end_look",
        status="ok",
        pv="p:0123abcd",
        results={"floor": {"free_mm": [1.5]}},
    )
    assert floaty.problems() == ["results.floor.free_mm[0]: float is not an integer feature"]
    with pytest.raises(p.ProtocolError, match="float"):
        p.EyeResult.from_json(json.loads(json.dumps(floaty.to_json())))
    with pytest.raises(p.ProtocolError, match="needs a why"):
        p.EyeResult(look_id="L1", attempt=0, kind="end_look", status="none", pv="p:0123abcd")
    with pytest.raises(p.ProtocolError, match="pv"):
        p.EyeResult(look_id="L1", attempt=0, kind="end_look", status="ok", pv="nope")


def test_integers_only_walks_everything():
    assert p.integers_only({"a": [1, "x", None, True, {"b": [2**63 - 1]}]}) == []
    assert p.integers_only({"a": 2**70}) == [
        "results.a: 1180591620717411303424 does not fit 64 bits"
    ]
    assert p.integers_only({1: 2}, "r") == ["r: key 1 is not a string"]
    assert p.integers_only({"x": b"raw"}, "r") == ["r.x: bytes is not an integer feature"]
    assert p.task_names(["things", "light", "zzz", "light"]) == ("light", "things")
