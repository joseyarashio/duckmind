"""The heat governor's vectors (spec §4.6; ``duck-ipc-proto`` ``CpuThrottle::throttled``,
``robotd/src/soc.rs`` fixtures).

* ``CpuThrottle.throttled`` — duck-eye's copy (:class:`duckmind.eye.protocol.CpuThrottle`)
  against the upstream definition ("either reading counts": the cooling level wound up, or the
  ceiling below the hardware maximum) on the readings ``soc.rs``'s fixtures produce: a cooked
  Radxa Zero 3 (6/6, 408 of 1800 MHz), a throttled GPU that is not the CPU, the other kernel's
  spelling, the hardest cluster, an unthrottled board, a skipped unreadable policy.
* ``robot.health`` off Linux has neither ``cpu_temp_c`` nor ``cpu_throttle`` (absent, never 0)
  and the governor reads that as the normal tier.
* The tier edges of spec §4.6's table (< 70 °C; 70–78; 78–85; ≥ 85 °C or the clock ceiling below
  its maximum) against :func:`duckmind.eye.governor.allowance`, and its ``body.hot`` fields as a
  valid percept.
"""

from __future__ import annotations

import pytest

protocol = pytest.importorskip("duckmind.eye.protocol")

# (what soc.rs's fixture makes throttle_in() return) → throttled
SOC_FIXTURES = {
    "a_cooked_board_reports_both_halves": ((6, 6, 408_000, 1_800_000), True),
    "a_devfreq_cooler_is_not_the_cpu": ((0, 0, 1_800_000, 1_800_000), False),
    "the_other_kernel_spelling_is_the_same_device": ((2, 6, 1_104_000, 1_800_000), True),
    "the_hardest_throttled_cluster_is_the_one_reported": ((0, 0, 1_200_000, 2_400_000), True),
    "an_unthrottled_board_says_so": ((0, 6, 1_800_000, 1_800_000), False),
    "an_unreadable_policy_is_skipped": ((0, 0, 1_416_000, 1_800_000), True),
}


def _throttle(values: tuple[int, int, int, int]):
    level, max_level, khz, max_khz = values
    return protocol.CpuThrottle(level=level, max_level=max_level, khz=khz, max_khz=max_khz)


@pytest.mark.parametrize("name", sorted(SOC_FIXTURES))
def test_cpu_throttle_throttled_on_the_soc_fixtures(name: str) -> None:
    values, expected = SOC_FIXTURES[name]
    t = _throttle(values)
    assert t.throttled is expected, name
    back = protocol.CpuThrottle.from_json(t.to_json())
    assert back == t and back.throttled is expected


def test_a_zero_ceiling_alone_is_not_throttled() -> None:
    """``max_khz`` 0 means no cpufreq reading: the ceiling half cannot say throttled."""
    assert not protocol.CpuThrottle(level=0, max_level=0, khz=0, max_khz=0).throttled
    assert protocol.CpuThrottle(level=1, max_level=6, khz=0, max_khz=0).throttled


def test_health_off_linux_is_absent_not_zero() -> None:
    health = protocol.EyeHealth.from_json({"t_ns": 5})
    assert health.cpu_temp_c is None and health.cpu_throttle is None
    wire = health.to_json()
    assert wire["cpu_temp_c"] is None and wire["cpu_throttle"] is None


def test_health_rejects_junk_readings() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.EyeHealth.from_json({"t_ns": 5, "cpu_throttle": "hot"})
    with pytest.raises(protocol.ProtocolError):
        protocol.CpuThrottle.from_json({"level": -1})


# spec §4.6: (SoC °C or None, the throttle reading or None) → tier
# 0 < 70 °C: ≤ 2 looks/s (bursts 4/s up to 5 s); 1: 70–78 °C, ≤ 1/s, idle looks without NPU;
# 2: 78–85 °C, ≤ 0.5/s, only the safe looks and going home; 3: ≥ 85 °C or the CPU ceiling below
# its maximum — safe looks only, body.hot, rest.
TIER_EDGES = [
    (None, None, 0),  # the Mac sim: no reading → normal
    (45.0, None, 0),
    (69.9, None, 0),
    (70.0, None, 1),
    (77.9, None, 1),
    (78.0, None, 2),
    (84.9, None, 2),
    (85.0, None, 3),
    (95.0, (6, 6, 408_000, 1_800_000), 3),
    (60.0, (0, 6, 1_800_000, 1_800_000), 0),  # a reading that says "not throttled"
    (60.0, (0, 0, 1_416_000, 1_800_000), 3),  # a ceiling below the maximum at 60 °C: tier 3
]


@pytest.mark.parametrize("temp,throttle,expected", TIER_EDGES)
def test_governor_tier_edges(temp, throttle, expected) -> None:
    governor = pytest.importorskip("duckmind.eye.governor")
    reading = None if throttle is None else _throttle(throttle)
    allow = governor.allowance(protocol.EyeHealth(t_ns=0, cpu_temp_c=temp, cpu_throttle=reading))
    assert allow.tier == expected, (temp, throttle)
    assert allow.hot is (expected == 3)
    assert allow.safe_only is (expected >= 2)
    assert allow.npu is (expected <= 1)
    assert allow.rate_milli == (2000, 1000, 500, 500)[expected]  # spec §4.6 looks/s


def test_governor_never_heard_is_normal() -> None:
    governor = pytest.importorskip("duckmind.eye.governor")
    assert governor.allowance(None).tier == 0


def test_governor_hot_event_is_a_valid_percept() -> None:
    governor = pytest.importorskip("duckmind.eye.governor")
    percept = pytest.importorskip("duckmind.percept")
    allow = governor.allowance(
        protocol.EyeHealth(
            t_ns=0, cpu_temp_c=91.4, cpu_throttle=_throttle((6, 6, 408_000, 1_800_000))
        )
    )
    event = {"kind": "body.hot", "src": "live", "t_ms": 1, "pv": "p:0123456789abcdef"}
    assert percept.validate({**event, **allow.hot_event()}) == []
