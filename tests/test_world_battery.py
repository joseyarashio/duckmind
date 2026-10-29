"""The world's battery (duckmind.world.battery): owner decisions E1 (display only: robotd keeps
the nominal 7.4 V) and E2 (low-voltage weakness is a parameter, off).

The pack model is integer arithmetic and runs everywhere; the world part's integration from
the duck's torques and the nest's contact needs MuJoCo (the render check runs it)."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from duckmind.world import battery
from duckmind.world.battery import PPM, Pack

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


def test_the_open_circuit_voltage_rises_with_charge_through_upstreams_span():
    values = [battery.ocv_mv(s) for s in range(0, PPM + 1, 50_000)]
    assert values == sorted(values) and values[0] == 6600 and values[-1] == 8400
    full = Pack()
    # full, under a 2 A load: about upstream's "8.2 V off a full charge, under load"
    assert 8100 <= full.volts_mv(2000) <= 8250
    low = Pack(soc_ppm=30_000)
    assert 6600 <= low.volts_mv(2000) <= 6700  # the sag floor near empty
    assert battery.robotd_percent(8200) == 100 and battery.robotd_percent(6600) == 0
    assert battery.robotd_percent(7400) == 50 and battery.robotd_percent(-1) == 0


def test_sitting_standing_and_walking_drain_about_like_v4_says():
    # v4: 1.6 %/min seated (BATTERY_DRAIN_PER_MIN 160), standing about 1.3×, walking about 2×.
    # Mean servo torques of a policy-driven duck (phase-2 headless measurement, robotd's
    # control path, 3 × 5 s each): seated 20, standing 31, walking 75 mN·m per servo.
    sit, stand, walk = (
        battery.drain_ppm_per_min(battery.body_load_ma([t] * 14, torque_on=True))
        for t in (20, 31, 75)
    )
    assert 14_000 <= sit <= 17_500  # 1.4–1.75 %/min
    assert 1.1 <= stand / sit <= 1.35 and 1.8 <= walk / sit <= 2.3
    off = battery.body_load_ma([0] * 14, torque_on=True, board_on=False)
    assert off == battery.QUIESCENT_MA
    assert battery.servo_ma(500, torque_on=False) == battery.SERVO_OFF_MA


def test_integration_is_exact_integer_arithmetic_and_does_not_drift():
    a, b = Pack(), Pack()
    for _ in range(3000):  # one minute in 20 ms steps
        a.step(20, 2100)
    b.step(60_000, 2100)
    assert a.soc_ppm == b.soc_ppm
    expect = PPM - 2100 * 60_000 * PPM // (battery.CAPACITY_MAH * battery.MS_PER_HOUR)
    assert abs(a.soc_ppm - expect) <= 1


def test_it_charges_only_on_the_contact_at_v4s_rate_then_tapers():
    pack = Pack(soc_ppm=200_000)
    assert pack.charging_ma(False) == 0 and pack.step(60_000, 2000, pg=False) > 0
    before = pack.soc_ppm
    pack.step(60_000, 2500, pg=True)  # a power path: the body runs off the charger
    assert pack.soc_ppm - before >= 12_500  # ≥ 1.25 %/min net
    near_full = Pack(soc_ppm=950_000)
    assert 0 < near_full.charging_ma(True) < battery.CHARGE_MA
    for _ in range(3600):  # an hour of tapering tops it up
        near_full.step(1000, 2000, pg=True)
    assert near_full.soc_ppm == PPM and near_full.charging_ma(True) == 0
    assert near_full.volts_mv(0, pg=True) <= battery.CHARGE_MAX_MV


def test_swaps_ageing_and_weakness():
    pack = Pack(soc_ppm=10)
    pack.step(10_000, 3000)
    assert pack.empty
    pack.swap()
    assert pack.soc_ppm == PPM and not pack.empty
    old = Pack(health_ppm=800_000)
    assert old.capacity_mams < Pack().capacity_mams and old.resistance_mohm > battery.R_PACK_MOHM
    assert battery.motor_scale_ppm(6800) == PPM  # E2 off: full strength
    assert battery.motor_scale_ppm(6800, weakness=True) == 6800 * PPM // 7400


def test_e1_the_battery_never_replaces_robotds_slow_reading():
    from duckmind.world.simhome import Registry

    registry = Registry()
    battery.register(registry)
    assert "slow" not in registry.ops and "x_battery" in registry.ops
    with pytest.raises(ValueError):
        registry.op("slow", lambda body, request: {}, override=True)
    part = battery.WorldBattery()
    snap = part.snapshot(None)
    assert snap["fed_to_robotd"] is False and snap["weakness"] is False
    assert snap["motor_scale_ppm"] == PPM


def test_the_battery_op_swaps_sets_and_refuses_nonsense():
    part = battery.WorldBattery()
    body = types.SimpleNamespace(world=types.SimpleNamespace(lock=__import__("threading").Lock()))
    assert part.battery_op(body, {"do": "set", "soc_ppm": 400_000})["soc_ppm"] == 400_000
    assert part.battery_op(body, {"do": "swap"})["percent"] == 100
    assert part.battery_op(body, {"do": "load", "name": "camera", "ma": 150})["loads_ma"] == {
        "camera": 150
    }
    assert part.battery_op(body, {"do": "power", "on": False})["powered"] is False
    for bad in ({"do": "set", "soc_ppm": 2 * PPM}, {"do": "power"}, {"do": "melt"}):
        with pytest.raises(ValueError):
            part.battery_op(body, bad)


def test_in_the_world_it_drains_off_the_nest_and_charges_on_it():
    from duckmind.world import worldkit

    world = worldkit.Headless("default_home", mjcf_dir=_mjcf())
    pack = world.kit.battery
    pack.pack.swap(500_000)
    world.run(1.0)  # seated on the nest: the contact is closed
    assert world.kit.nest.closed and pack.pg
    snap = pack.snapshot(world.world)
    assert snap["charging"] and snap["current_ma"] < 0 and snap["soc_ppm"] > 500_000
    world.kit.hand.command(world.world, {"do": "place", "to": "floor", "x_mm": 1500, "y_mm": 0})
    while world.kit.hand.script is not None:
        world.step()
    start = pack.pack.soc_ppm
    world.run(2.0)
    snap = pack.snapshot(world.world)
    assert not snap["pg"] and snap["current_ma"] > battery.BOARD_MA
    assert pack.pack.soc_ppm < start and snap["volts_mv"] < snap["ocv_mv"]
