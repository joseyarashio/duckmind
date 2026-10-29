"""The look governor (spec §4.6): how often the eye may look, and what, given the heat.

A pure function of ``robot.health``'s thermal part (duck-body forwards it once a second as
``eye.health``), the recent looks and the look being asked for. duck-body's look planner,
duck-eye's server and HomeWorld call the same code, so a surrogate duck is held to the same
budget as a real one.

**Tiers** (spec §4.6; the SoC temperature is ``cpu_temp_c``, "throttled" is upstream
``CpuThrottle::throttled()``: the cooling level is up, or the clock ceiling is below the hardware
maximum):

====  ==================  ==============================================================
tier  when                what is allowed
====  ==================  ==============================================================
0     < 70 °C             ≤ 2 looks/s on average, bursts of 4 looks/s for up to 5 s; NPU
1     70–78 °C            ≤ 1 look/s; idle looks run no NPU model
2     78–85 °C            ≤ 0.5 look/s; only the safe looks (``end_look``, ``glance_down``)
                          and going home (``seek_nest``, ``fixate`` while ``docking``)
3     ≥ 85 °C, throttled  the safe looks only; ``body.hot`` (Duck Core lets the duck rest)
====  ==================  ==============================================================

Unknown temperature (``cpu_temp_c`` absent: off Linux, as on the Mac beside the simulator)
reads as tier 0 unless the throttle says otherwise. **The safe looks are never refused** — the
pre-departure glance is what keeps the duck off a low box — they are only paced by the burst
limit (4 looks/s).

The average is a token bucket: ``rate`` tokens a second up to ``rate`` + a burst allowance of
(4 − ``rate``) × 5 s, one token a look; the burst limit is a minimum spacing of 250 ms.
:class:`Budget` is that bucket as plain data (``admit`` returns the next one), so a caller keeps
it wherever it keeps its state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from duckmind.eye.protocol import NEST_LOOKS, SAFE_LOOKS, CpuThrottle, EyeHealth

NS = 1_000_000_000
TIER_C = (70.0, 78.0, 85.0)  # tier boundaries, °C (spec §4.6)
RATE_MILLI = (2000, 1000, 500, 500)  # average looks/s × 1000 per tier
BURST_PER_S = 4  # short bursts, any tier (spec: 4 looks/s)
BURST_S = 5  # … for at most 5 s at tier 0
MIN_SPACING_NS = NS // BURST_PER_S
IDLE_LOOKS = ("idle_glance", "look_up", "meter")
TIER_ZH = ("正常", "有點熱：少看一點", "很熱：只看安全的", "過熱：休息")


@dataclass(frozen=True)
class Allowance:
    """What the heat allows now."""

    tier: int
    rate_milli: int  # average looks/s × 1000
    npu: bool  # an NPU model may run on a look (not on idle looks at tier 1)
    safe_only: bool  # only SAFE_LOOKS (and NEST_LOOKS while docking at tier 2)
    hot: bool  # tier 3: duck-body sends body.hot
    soc_c: int | None  # the temperature it was judged on (°C, rounded), None when unknown
    ceiling_mhz: int | None  # the clock ceiling, when known

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "rate_milli": self.rate_milli,
            "npu": self.npu,
            "safe_only": self.safe_only,
            "hot": self.hot,
        }

    def hot_event(self) -> dict[str, Any]:
        """percept-v1 ``body.hot``'s fields (duck-body sends it when the tier changes)."""
        return {"soc_c": self.soc_c, "ceiling_mhz": self.ceiling_mhz, "tier": self.tier}


def allowance(health: EyeHealth | None) -> Allowance:
    """The tier for the last ``robot.health`` (``None``: never heard: the normal tier)."""
    temp = health.cpu_temp_c if health is not None else None
    throttle: CpuThrottle | None = health.cpu_throttle if health is not None else None
    tier = 0
    if temp is not None:
        tier = sum(1 for boundary in TIER_C if temp >= boundary)
    if throttle is not None and throttle.throttled:
        tier = 3
    return Allowance(
        tier=tier,
        rate_milli=RATE_MILLI[tier],
        npu=tier <= 1,
        safe_only=tier >= 2,
        hot=tier >= 3,
        soc_c=None if temp is None else round(temp),
        ceiling_mhz=None if throttle is None or throttle.khz == 0 else throttle.khz // 1000,
    )


def kind_allowed(kind: str, mode: str, allow: Allowance) -> bool:
    """May a look of this kind run at all at this tier (the budget aside)?"""
    if kind in SAFE_LOOKS:
        return True
    if allow.tier == 2 and mode == "docking" and kind in NEST_LOOKS:
        return True
    return not allow.safe_only


def npu_allowed(kind: str, allow: Allowance) -> bool:
    """May this look run an NPU model (at most one a look, spec §4.4)?"""
    if not allow.npu:
        return False
    return not (allow.tier == 1 and kind in IDLE_LOOKS)


@dataclass(frozen=True)
class Budget:
    """The look budget's token bucket: tokens × 1000 at ``t_ns``, and the last look's time."""

    tokens_milli: int = 0
    t_ns: int | None = None
    last_look_ns: int | None = None

    @staticmethod
    def capacity_milli(rate_milli: int) -> int:
        """One second's worth plus the burst allowance: (4 looks/s − rate) × 5 s."""
        return rate_milli + max(0, BURST_PER_S * 1000 - rate_milli) * BURST_S

    def refilled(self, now_ns: int, rate_milli: int) -> Budget:
        cap = self.capacity_milli(rate_milli)
        if self.t_ns is None:
            return replace(self, tokens_milli=cap, t_ns=now_ns)
        elapsed = max(0, now_ns - self.t_ns)
        tokens = min(cap, self.tokens_milli + elapsed * rate_milli // NS)
        return replace(self, tokens_milli=tokens, t_ns=now_ns)


@dataclass(frozen=True)
class Decision:
    ok: bool
    why: str  # "ok", "hot" (kind not allowed at this tier), "budget", "spacing"
    budget: Budget  # the budget after this decision (spent only when ok)
    allow: Allowance
    npu: bool  # an NPU model may run on this look
    retry_in_ms: int  # when asking again could succeed (0 when ok)


def admit(
    kind: str,
    mode: str,
    now_ns: int,
    budget: Budget,
    health: EyeHealth | None,
    *,
    retake: bool = False,
) -> Decision:
    """May a look of ``kind`` start now? Spends a token when it may.

    The safe looks are always admitted except inside the 250 ms burst spacing (they then say
    ``spacing`` with the wait); they spend a token when there is one and never go into debt.
    A ``retake`` (the same look again after the head moved during the first frame, spec §4.3
    step 5) is neither paced nor charged: only the tier's kind rule applies."""
    allow = allowance(health)
    budget = budget.refilled(now_ns, allow.rate_milli)
    safe = kind in SAFE_LOOKS
    if not kind_allowed(kind, mode, allow):
        return Decision(False, "hot", budget, allow, False, 1000)
    if retake:
        return Decision(True, "ok", budget, allow, npu_allowed(kind, allow), 0)
    if budget.last_look_ns is not None and now_ns - budget.last_look_ns < MIN_SPACING_NS:
        wait = MIN_SPACING_NS - (now_ns - budget.last_look_ns)
        return Decision(False, "spacing", budget, allow, False, -(-wait // 1_000_000))
    if budget.tokens_milli < 1000 and not safe:
        missing = 1000 - budget.tokens_milli
        wait_ms = -(-missing * 1000 // max(1, allow.rate_milli))
        return Decision(False, "budget", budget, allow, False, wait_ms)
    spent = replace(budget, tokens_milli=max(0, budget.tokens_milli - 1000), last_look_ns=now_ns)
    return Decision(True, "ok", spent, allow, npu_allowed(kind, allow), 0)
