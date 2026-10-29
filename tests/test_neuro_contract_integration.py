from __future__ import annotations

import json

from duckmind.neuro.contract import BodySense, CameraSense, SensorFrame, Sighting, SoundSense


def test_integrated_sensor_channels_round_trip_as_typed_values() -> None:
    frame = SensorFrame(
        t_ms=12_340,
        body=BodySense(mouth=-0.04, active_skill="ground_pick"),
        camera=CameraSense(
            sightings=(
                Sighting(
                    kind="person",
                    bearing=0.2,
                    range_m=1.1,
                    size=0.31,
                    axis=0.4,
                    look=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                    lux=860.0,
                ),
            ),
            age_ms=80,
            lux=520.0,
            roof=0.46,
            roof_t_ms=12_300,
        ),
        sound=SoundSense(
            kind="voice",
            voice=(0.8, 0.6),
            voice_t_ms=12_320,
        ),
        sources={"body": "real", "camera": "model", "sound": "model"},
    )

    restored = SensorFrame.from_json(json.loads(json.dumps(frame.to_json())))

    assert restored.body.mouth == -0.04
    assert restored.body.active_skill == "ground_pick"
    assert restored.camera.lux == 520.0
    assert restored.camera.roof == 0.46 and restored.camera.roof_t_ms == 12_300
    sighting = restored.camera.sightings[0]
    assert sighting.look == (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    assert sighting.axis == 0.4 and sighting.lux == 860.0
    assert restored.sound.voice == (0.8, 0.6) and restored.sound.voice_t_ms == 12_320
    assert restored.sources["camera"] == "model"


def test_unavailable_integration_channels_stay_unknown() -> None:
    restored = SensorFrame.from_json({"t_ms": 1})

    assert restored.body.mouth is None
    assert restored.body.active_skill is None
    assert restored.camera.lux is None
    assert restored.camera.roof is None and restored.camera.roof_t_ms is None
    assert restored.sound.voice is None and restored.sound.voice_t_ms is None
