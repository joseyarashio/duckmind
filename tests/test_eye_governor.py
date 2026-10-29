"""eye.governor: the look budget and the heat tiers (spec §4.6)."""

from __future__ import annotations

import pytest

from duckmind.eye import governor as g
from duckmind.eye.protocol import CpuThrottle, EyeHealth

S = 1_000_000_000


def health(temp=None, throttle=None):
    return EyeHealth(t_ns=0, cpu_temp_c=temp, cpu_throttle=throttle)


@pytest.mark.parametrize(
    "temp,tier",
    [
        (None, 0),
        (45.0, 0),
        (69.9, 0),
        (70.0, 1),
        (77.9, 1),
        (78.0, 2),
        (84.9, 2),
        (85.0, 3),
        (96, 3),
    ],
)
def test_the_tiers_follow_the_soc_temperature(temp, tier):
    allow = g.allowance(health(temp))
    assert allow.tier == tier
    assert allow.rate_milli == (2000, 1000, 500, 500)[tier]
    assert allow.safe_only == (tier >= 2) and allow.hot == (tier == 3) and allow.npu == (tier <= 1)


def test_a_throttled_clock_is_the_hot_tier_whatever_the_temperature():
    cooled = CpuThrottle(level=0, max_level=6, khz=1_800_000, max_khz=1_800_000)
    assert g.allowance(health(60.0, cooled)).tier == 0
    pinned = CpuThrottle(level=0, max_level=6, khz=1_200_000, max_khz=1_800_000)
    wound = CpuThrottle(level=3, max_level=6, khz=1_800_000, max_khz=1_800_000)
    for throttle in (pinned, wound):
        allow = g.allowance(health(None, throttle))
        assert allow.tier == 3 and allow.hot
    event = g.allowance(health(88.4, pinned)).hot_event()
    assert event == {"soc_c": 88, "ceiling_mhz": 1200, "tier": 3}
    assert g.allowance(None).tier == 0  # never heard: the Mac beside the simulator


def test_two_looks_a_second_on_average_and_bursts_of_four_for_five_seconds():
    budget = g.Budget()
    t = 0
    admitted = []
    # ask every 250 ms (4/s) for 30 s: 4/s while the burst lasts, then 2/s
    while t < 30 * S:
        d = g.admit("scan", "outing", t, budget, None)
        budget = d.budget
        if d.ok:
            admitted.append(t)
        t += S // 4
    first5 = [x for x in admitted if x < 5 * S]
    assert len(first5) == 20  # 4 looks/s for the first 5 s
    late = [x for x in admitted if 20 * S <= x < 30 * S]
    assert len(late) == 20  # 2 looks/s once the burst is spent
    assert len(admitted) <= 2 * 30 + (4 - 2) * 5 + 2


def test_looks_are_spaced_at_least_250_ms():
    d = g.admit("scan", "outing", 0, g.Budget(), None)
    d2 = g.admit("scan", "outing", 100_000_000, d.budget, None)
    assert d.ok and not d2.ok and d2.why == "spacing" and d2.retry_in_ms == 150
    # the safe look is paced too, but never refused for the budget
    d3 = g.admit("glance_down", "outing", 100_000_000, d.budget, None)
    assert not d3.ok and d3.why == "spacing"


def test_the_safe_looks_are_never_refused_for_budget_or_heat():
    budget = g.Budget(tokens_milli=0, t_ns=0, last_look_ns=None)
    hot = health(90.0)
    for kind in ("end_look", "glance_down"):
        d = g.admit(kind, "outing", 1, budget, hot)
        assert d.ok and not d.npu
    refused = g.admit("scan", "outing", 1, budget, hot)
    assert not refused.ok and refused.why == "hot"
    empty = g.admit("scan", "outing", 1, budget, None)
    assert not empty.ok and empty.why == "budget" and empty.retry_in_ms > 0


def test_going_home_is_allowed_when_hot_but_not_when_overheated():
    warm = health(80.0)
    budget = g.Budget()
    assert g.admit("seek_nest", "docking", 0, budget, warm).ok
    assert not g.admit("seek_nest", "outing", 0, budget, warm).ok
    assert not g.admit("idle_glance", "awake", 0, budget, warm).ok
    assert not g.admit("seek_nest", "docking", 0, budget, health(86.0)).ok


def test_npu_rules():
    assert g.admit("scan", "outing", 0, g.Budget(), None).npu
    warmish = health(72.0)
    assert g.admit("scan", "outing", 0, g.Budget(), warmish).npu
    assert not g.admit("idle_glance", "awake", 0, g.Budget(), warmish).npu
    assert g.npu_allowed("scan", g.allowance(health(80.0))) is False


def test_a_retake_is_the_same_look_not_another():
    d = g.admit("scan", "outing", 0, g.Budget(tokens_milli=1000, t_ns=0), None)
    assert d.ok and d.budget.tokens_milli == 0
    retake = g.admit("scan", "outing", 50_000_000, d.budget, None, retake=True)
    assert retake.ok and retake.budget.tokens_milli == d.budget.tokens_milli + 100
    assert retake.budget.last_look_ns == d.budget.last_look_ns
    assert not g.admit("scan", "outing", 50_000_000, d.budget, health(90.0), retake=True).ok
