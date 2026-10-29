"""The dark gate (spec §4.5 walking rules; owner C1, C5, 2026-09-28 ~12:30 「2. 放寬」 — C2 is
reversed, no light refuses a walk — and 「3. 一律放寬」: since the phase-4 landing the dark no
longer caps a segment at 3 s either; it was pacing only)."""

from __future__ import annotations

import pytest

from duckmind.duckbody import darkgate
from duckmind.duckbody.darkgate import allow_walk, walk_gate
from duckmind.duckbody.protocol import ActContext, DepthView, ProprioView, Toward, WalkCmd


@pytest.mark.parametrize("level", ["bright", "dim"])
def test_light_walks_as_usual(level: str) -> None:
    for tof in (False, True):
        v = allow_walk(level, has_tof=tof)
        assert v.allow and v.max_ms is None and v.rule == "light"


def test_the_dark_walks_uncapped_and_c1_dock_light_is_still_dark() -> None:
    # C1: the dock light alone in total darkness is "dark" — a light level, walked like dark
    for tof in (False, True):
        v = allow_walk("dark", has_tof=tof)
        assert v.allow and v.max_ms is None and v.rule == "dark" and v.reason is None


@pytest.mark.parametrize("level", ["dark", "black", "none", "bogus"])
def test_owner_relaxed_no_light_refuses_or_shortens_a_walk(level: str) -> None:
    """「2. 放寬」: black (or the eye down) no longer keeps a camera-only duck still, and
    「一律放寬」: no segment cap in the dark — it walks as asked and physics decides; a walk at a
    visible backlight keeps its C5 label (a direction, never a range)."""
    for tof in (False, True):
        v = allow_walk(level, has_tof=tof)
        assert v.allow and v.reason is None and v.max_ms is None
        assert v.rule == "dark"
    toward = allow_walk(level, has_tof=False, backlight_heading_mrad=90)  # C5: direction only
    assert toward.allow and toward.max_ms is None and toward.rule == "toward_backlight"
    away = allow_walk(level, has_tof=False, backlight_heading_mrad=400)
    assert away.allow and away.max_ms is None and away.rule == "dark"
    assert not hasattr(darkgate, "DARK_SEGMENT_MS")  # (the 3 s cap is gone, not just unused)


def ctx(light: str, tof: bool) -> ActContext:
    depth = DepthView(0, 900, None, None, 0, False, 0) if tof else None
    return ActContext(
        t_ns=0, state=None, pose=None, depth=depth, proprio=ProprioView(), light=light
    )


def test_walkexec_adapter() -> None:
    wander = WalkCmd(act_id="W1", deadline_ms=9, vx_mm=400)
    assert walk_gate(ctx("black", False), wander) == (True, None, "dark")
    assert walk_gate(ctx("black", True), wander) == (True, None, "dark")
    light = Toward(look_id="L1", b_mrad=60, r_mm=0, seen_t_ms=0)
    home = WalkCmd(act_id="W2", deadline_ms=9, vx_mm=400, purpose="dock", toward=light)
    assert walk_gate(ctx("black", False), home) == (True, None, "toward_backlight")
    assert walk_gate(ctx("bright", False), wander) == (True, None, "light")
