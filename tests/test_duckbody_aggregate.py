"""Folding rules (duckmind.duckbody.aggregate; spec §4.3 step 10, §7.2, §7.7)."""

from __future__ import annotations

from duckmind import percept
from duckmind.duckbody.aggregate import Aggregator, depth_band, hot_tier
from duckmind.duckbody.protocol import DepthView

PV = "p:00112233aabbccdd"


def agg() -> Aggregator:
    return Aggregator("live", PV)


def test_every_event_is_valid_percept_v1() -> None:
    a = agg()
    event = a.emit("body.posture", 1234, want="sit", got="sit", ms=900, fell=False, act_id="P1")
    assert percept.validate(event) == [] and event["t_ms"] == 1234 and event["pv"] == PV
    body = {"kind": "body.fell", "t_ms": 999, "during": "walk", "ctx": {}, "trigger": "tilt40"}
    stamped = a.emit_body(body, 5000)
    assert stamped["t_ms"] == 999  # when it was sensed, not when it was sent
    assert a.counts() == {"body.posture": 1, "body.fell": 1}


def test_light_on_change_and_every_ten_minutes() -> None:
    a = agg()
    assert a.light(0, "bright", 90, 2)["why"] == "change"
    assert a.light(1000, "bright", 91, 2) is None
    assert a.light(2000, "dim", 80, 5)["level"] == "dim"
    assert a.light(2000 + 599_999, "dim", 80, 5) is None
    assert a.light(2000 + 600_000, "dim", 80, 5)["why"] == "periodic"


def test_sightings_are_edges_with_a_5_s_gap() -> None:
    a = agg()
    kw = dict(what="nest", thing_id="tag36h11:17", conf_ppm=900_000, look_id="L1", light="bright")
    first = a.sighting(0, b_mrad=0, r_mm=1000, **kw)
    assert first is not None and first["edge"] == "new"
    assert a.sighting(1000, b_mrad=10, r_mm=1010, **kw) is None  # the same place
    assert a.sighting(2000, b_mrad=0, r_mm=1400, **kw) is None  # moved, but within 5 s
    moved = a.sighting(6000, b_mrad=0, r_mm=1400, **kw)
    assert moved is not None and moved["edge"] == "moved"
    lost = a.lost(9000, what="nest", thing_id="tag36h11:17", look_id="L3", light="dim")
    assert lost is not None and lost["edge"] == "lost"
    assert a.lost(9500, what="nest", thing_id="tag36h11:17", look_id="L4", light="dim") is None
    cue = a.sighting(
        10_000,
        what="nest",
        thing_id=None,
        b_mrad=-300,
        r_mm=None,
        conf_ppm=500_000,
        look_id="L5",
        light="dark",
        r_src="cue",
    )
    assert cue is not None and cue["r_mm"] is None and percept.validate(cue) == []


def view(ahead: int | None, drop: int = 0, touch: bool = False) -> DepthView:
    return DepthView(0, ahead, 1200, 1300, drop, touch, -100)


def test_depth_only_while_walking_on_crossings() -> None:
    a = agg()
    assert a.depth(0, view(1500), walking=True) is None
    assert a.depth(10, view(780), walking=True)["ahead_mm"] == 780  # crossed 800
    assert a.depth(20, view(700), walking=True) is None
    assert a.depth(30, view(290), walking=True) is not None  # crossed 500 and 300
    assert a.depth(40, view(290, touch=True), walking=True)["touch"] is True
    assert a.depth(50, view(100), walking=False) is None  # not walking: no event
    assert depth_band(None) == 0 and depth_band(150) == 4


def test_contacts_fold_into_walks() -> None:
    a = agg()
    assert a.contact(0, kind="blocked", side="front", ev=["odom"], walking=True) is None
    event = a.contact(0, kind="brushed", side="left", ev=["heading"], walking=False)
    assert event is not None and event["contact"] == "brushed"


def test_battery_steps_and_thermal_tiers() -> None:
    a = agg()
    assert a.battery(0, 71, "draining")["pct"] == 71
    assert a.battery(10, 70, "draining") is None  # same 5 % step
    assert a.battery(20, 69, "draining") is not None
    assert a.hot(0, 60, None) is None  # a cool start is not news
    assert a.hot(10, 72, 1800)["tier"] == 1
    assert a.hot(20, 74, 1800) is None
    assert a.hot(30, 86, None)["tier"] == 3
    assert [hot_tier(c) for c in (69, 70, 78, 85)] == [0, 1, 2, 3]
    assert hot_tier(60, throttled=True) == 3
