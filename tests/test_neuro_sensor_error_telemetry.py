"""Actual integrated sensor faults must reach public driver skill status, then clear on recovery."""

from __future__ import annotations

from types import SimpleNamespace

from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.world.neuro_integration import live_senses


def test_camera_world_error_is_visible_in_driver_telemetry_and_clears_on_recovery():
    brain = SimpleNamespace(
        play=SimpleNamespace(
            ball=SimpleNamespace(phase="idle", last_event=None),
            carry=SimpleNamespace(phase="idle", last_event=None),
            bask_ctl=SimpleNamespace(phase="idle"),
        ),
        play_out=SimpleNamespace(skill=None, skill_seq=0),
        sound_out=SimpleNamespace(
            view=lambda: {
                "theremin": {"up": False},
                "ride": {"riding": False, "lifted": False, "after": 0, "outcome": "none"},
                "dance": {"on": False},
            }
        ),
        hide_run=SimpleNamespace(phase="idle"),
        circle=SimpleNamespace(phase="idle"),
        social_view=lambda: {},
    )
    senses = live_senses(seed=3)
    lanes = Lanes(
        brain=brain,
        senses=senses,
        names={"brain": "real", "senses": "real", "motor": "real"},
    )
    lanes.sound = SimpleNamespace(engine=SimpleNamespace(view=lambda: {"helper": "test-helper"}))
    driver = NeuroDriver(NeuroSpec(), lanes=lanes)

    # The card must use the same actual-driver ToF frame that /api/neuro/sound/status reports.
    # Missing frame means unavailable; a real-shaped frame restores only theremin availability.
    no_tof = driver.telemetry()["skills"]
    assert no_tof["theremin"]["available"] is False
    assert no_tof["theremin"]["reason"] == "no_sensor"
    driver._tof_frame = {"t_ns": 120}
    with_tof = driver.telemetry()["skills"]
    assert with_tof["theremin"]["available"] is True
    assert with_tof["theremin"]["reason"] is None
    assert with_tof["ride"]["available"] is True
    assert with_tof["dance"]["available"] is True
    driver._tof_frame = None

    def tick(t_ms: int, errors=()):
        return senses(
            SimpleNamespace(
                t_ms=t_ms,
                state=None,
                link=None,
                events=(),
                depth=None,
                tof=None,
                pg=None,
                day_minute=600.0,
                world={
                    "neuro": {
                        "v": 1,
                        "sim_time": t_ms / 1000.0,
                        "samples": [],
                        "sounds": [],
                        "touches": [],
                        "errors": list(errors),
                        "capabilities": {"jaw_present": True},
                    }
                },
            )
        )

    tick(100, ("sample: TypeError: camera model failed",))
    broken = driver.telemetry()["skills"]
    for skill in ("follow", "hide", "circle", "ball", "carry", "sun"):
        assert broken[skill]["status"] == "missing"
        assert broken[skill]["reason"] == "world_sensor_error"
    assert broken["theremin"]["reason"] != "world_sensor_error"

    recovered = tick(120, ())
    healthy = driver.telemetry()["skills"]
    assert recovered.t_ms == 120
    assert senses.world_errors == ()
    for skill in ("follow", "hide", "circle", "ball", "carry", "sun"):
        assert healthy[skill]["available"] is True
        assert healthy[skill]["reason"] is None
