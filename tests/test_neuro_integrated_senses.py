"""The live composed sense adapter preserves typed, delayed, freshness-aware observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.world import neuro_integration as integrated
from duckmind.world import neuro_senses as NSenses


@dataclass
class Doors:
    t_ms: int
    state: Any = None
    world: dict[str, Any] | None = None


def robot_state(mouth_rad: float = 0.0) -> RobotState:
    joints = [0.0] * 15
    joints[5] = joints[6] = 0.3491
    joints[9] = mouth_rad
    return RobotState.from_wire(
        {
            "t": 0.0,
            "t_ns": 0,
            "policy": "sit",
            "move": {"requested": [0.0, 0.0, 0.0], "applied": [0.0, 0.0, 0.0], "limited_by": []},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [3.5] * 15,
            "odom": {"position": [0.1, 0.2, 0.0], "yaw": 0.0},
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "head": [0.0, 0.0, 0.0, 0.0],
        }
    )


def sample(t: float, *, lux: float = 150.0, meter_lux: float = 80.0, seen=None, roof=1.2):
    return {
        "t": round(t, 4),
        "cam": {"lux": lux, "lux_meter": meter_lux, "fwd": [1.0, 0.0, 0.0], "backlight": False},
        "duck": {"yaw": 0.0},
        "seen": list(seen or ()),
        "sun": [],
        "roof_truth": {"height_m": roof, "looked_up": True},
    }


def person() -> dict[str, Any]:
    return {
        "id": "person:owner",
        "kind": "person",
        "cls": "person",
        "n": 1,
        "fov": 1,
        "vis": 1,
        "b": 0.08,
        "e": 0.05,
        "size": 0.5,
        "dist": 1.2,
        "vr": 0.0,
        "om": 0.5,
        "lux": 150.0,
        "rng": 1.1,
    }


def sound(sid: int, *, t0: float = 0.2, t1: float = 2.8) -> dict[str, Any]:
    return {
        "id": sid,
        "kind": "voice",
        "t0": t0,
        "t1": t1,
        "db": 78.0,
        "env": "speech",
        "name": False,
        "src": "person:owner",
    }


def world(
    samples: list[dict[str, Any]], t: float, sounds=(), *, jaw_present: bool | None = True
) -> dict[str, Any]:
    neuro = {
        "v": 1,
        "sim_time": t,
        "samples": samples[-8:],
        "sounds": list(sounds),
        "touches": [],
    }
    if jaw_present is not None:
        neuro["capabilities"] = {"jaw_present": jaw_present}
    return {"neuro": neuro}


def test_live_senses_first_tick_has_typed_defaults_and_reads_the_real_jaw_joint():
    hub = integrated.live_senses(seed=7)
    # The first tick can arrive before the camera queue has delivered a capture.
    frame = hub(
        Doors(t_ms=0, state=robot_state(0.271), world=world([sample(0.0)], 0.0, jaw_present=True))
    )

    assert isinstance(frame, C.SensorFrame)
    assert frame.t_ms == 0
    assert frame.body.mouth == pytest.approx(0.271)
    assert frame.sources["body"] == "real"
    assert frame.sources["camera"] == "model"
    assert frame.camera.lux is None and frame.camera.roof is None
    assert frame.camera.roof_t_ms is None


def test_live_adapter_populates_typed_look_voice_roof_lux_and_mouth_without_world_truth():
    hub = integrated.live_senses(seed=1)
    samples: list[dict[str, Any]] = []
    frames: list[C.SensorFrame] = []
    sounds = [sound(1, t0=0.2, t1=1.5)]
    for step in range(151):
        t = step * 0.02
        if step % 5 == 0:
            samples.append(sample(t, seen=[person()], meter_lux=75.0, roof=1.4))
        frames.append(
            hub(
                Doors(
                    t_ms=round(t * 1000),
                    state=robot_state(0.33),
                    world=world(samples, t, sounds, jaw_present=True),
                )
            )
        )

    people = [s for frame in frames for s in frame.camera.sightings if s.kind == "person"]
    typed_looks = [s.look for s in people if s.look is not None]
    assert typed_looks and all(len(look) == 6 for look in typed_looks)
    speech_frames = [frame for frame in frames if frame.sound.kind in {"voice", "name", "hello"}]
    assert speech_frames
    assert any(
        frame.sound.voice is not None and len(frame.sound.voice) == 2 for frame in speech_frames
    )
    assert any(
        frame.camera.roof is not None and frame.camera.roof_t_ms is not None for frame in frames
    )
    assert any(frame.camera.lux is not None for frame in frames)
    assert all(frame.body.mouth == pytest.approx(0.33) for frame in frames)
    assert all(frame.sources["camera"] == "model" for frame in frames)
    assert all(
        frame.sound.voice is None
        for frame in frames
        if frame.sound.kind not in {"voice", "name", "hello"}
    )

    wire = json.dumps([frame.to_json() for frame in frames], ensure_ascii=False)
    for world_only in ('"roof_truth"', '"lux_meter"', "person:owner", '"sim_time"'):
        assert world_only not in wire
    round_trip = C.SensorFrame.from_json(speech_frames[-1].to_json())
    assert round_trip == speech_frames[-1]


def test_camera_queue_delivers_after_capture_keeps_multiple_capture_sidecars_and_expires_stale_values():
    hub = integrated.live_senses(seed=11)
    first_sample = sample(0.0, meter_lux=65.0, roof=0.8)
    frame = hub(Doors(t_ms=0, state=robot_state(), world=world([first_sample], 0.0)))
    assert not frame.camera.usable
    assert len(hub.pending) == 1
    due_ms = hub.pending[0].due_ms
    assert NSenses.DETECT_LATENCY_MS[0] <= due_ms <= NSenses.DETECT_LATENCY_MS[1]

    delivered = None
    for t_ms in range(20, NSenses.DETECT_LATENCY_MS[1] + 21, 20):
        delivered = hub(
            Doors(t_ms=t_ms, state=robot_state(), world=world([first_sample], t_ms / 1000.0))
        )
        if delivered.camera.usable:
            break
    assert delivered is not None and delivered.camera.usable
    assert delivered.camera.age_ms <= NSenses.CAMERA_STALE_MS
    assert delivered.camera.roof is not None and delivered.camera.roof_t_ms == 0

    # A delayed world batch may contain several admitted camera captures. Each queued result must
    # retain that capture's own sensor sidecar; after FIFO delivery, the newest capture is visible.
    captures = [
        sample(0.0, meter_lux=55.0, roof=0.7),
        sample(0.6, meter_lux=85.0, roof=1.1),
        sample(1.2, meter_lux=115.0, roof=1.5),
    ]
    batched = integrated.live_senses(seed=13, vision_hz=2.0)
    newest = batched(Doors(t_ms=1200, state=robot_state(0.2), world=world(captures, 1.2)))
    assert batched.vision.passes == 3
    assert batched.looks == 3
    # The first two delayed results have arrived; the last capture remains queued until its own
    # detect latency elapses. Its lux/roof values travel with that capture, not later world truth.
    assert len(batched.pending) == 1
    queued = batched.pending[0]
    assert queued.due_ms >= 1200 + NSenses.DETECT_LATENCY_MS[0]
    assert queued.due_ms <= 1200 + NSenses.DETECT_LATENCY_MS[1]
    assert queued.camera.roof_t_ms == 1200
    assert queued.camera.roof == pytest.approx(1.5, abs=0.6)
    assert queued.camera.lux == pytest.approx(batched.vision.camera_readings["lux"])

    newest = batched(Doors(t_ms=queued.due_ms, state=robot_state(0.2), world=world([], 1.4)))
    assert not batched.pending
    assert newest.camera.usable and newest.camera.age_ms == 0
    assert newest.camera.roof_t_ms == 1200
    assert newest.camera.lux == pytest.approx(queued.camera.lux)
    assert newest.camera.roof == pytest.approx(queued.camera.roof)

    stale_ms = int(queued.due_ms + NSenses.CAMERA_STALE_MS + 1)
    stale = batched(Doors(t_ms=stale_ms, state=robot_state(), world=world([], stale_ms / 1000.0)))
    assert not stale.camera.usable
    assert stale.camera.lux is None
    assert stale.camera.roof is None and stale.camera.roof_t_ms is None


@pytest.mark.parametrize("jaw_present", [False, None])
def test_live_senses_does_not_infer_a_jaw_from_the_wire_joint_count(jaw_present):
    hub = integrated.live_senses(seed=7)
    frame = hub(
        Doors(
            t_ms=0,
            state=robot_state(0.271),
            world=world([sample(0.0)], 0.0, jaw_present=jaw_present),
        )
    )
    assert frame.body.mouth is None


def test_live_senses_reads_joint_nine_only_when_compiled_world_declares_jaw():
    hub = integrated.live_senses(seed=7)
    frame = hub(
        Doors(t_ms=0, state=robot_state(0.271), world=world([sample(0.0)], 0.0, jaw_present=True))
    )
    assert frame.body.mouth == pytest.approx(0.271)
