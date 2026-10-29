"""The brain's answer to a low edge in its way (obstacles, 2026-10-10): it backs off in a small arc before it touches it. The ToF
side of the edge is tests/test_depth_low_edge.py; the whole duck at the obstacle course is tools/real_home/course.py
(docs/design/neural-duck/OBSTACLES-2026-10-10.md)."""

from __future__ import annotations

import pytest

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome
from duckmind.neuro.unstick import Escape


def _brain() -> NeuroBrain:
    g = Genome.from_axes(
        dict(
            boldness=0.9,
            activity=0.9,
            sociability=0.3,
            exploration=0.9,
            wit=0.7,
            chronotype=0.3,
            tempo=0.6,
        ),
        primary="explorer",
        branch="explorer",
        gd=120,
    )
    return NeuroBrain(g, seed=3)


def _frame(
    t_ms: int,
    *,
    vx: float = 0.31,
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tof: C.TofSense | None = None,
    tilt: float = 0.02,
) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture="walk", vx=vx, odom=odom, tilt=tilt),
        tof=tof or C.TofSense(),
    )


def _low_edge_frame(t_ms: int, *, ahead=300, top=60, edge=60, vx=0.31) -> C.SensorFrame:
    tof = C.TofSense(
        present=True,
        ahead_mm=ahead,
        age_ms=10,
        ahead_top_mm=top,
        low_edge_mm=edge,
        virtual=True,
    )
    return _frame(t_ms, vx=vx, tof=tof)


def test_a_wide_low_edge_close_ahead_makes_the_duck_back_off_in_a_small_arc():
    brain = _brain()
    brain._req_vx = 0.35
    esc = brain._low_block_escape(1000, _low_edge_frame(1000), (0.0, 0.0, 0.0), True)
    assert isinstance(esc, Escape) and esc.why == "low"
    assert esc.back_ms == B.LOW_BACK_MS[0] and esc.count == 1
    # not again at once, and the count goes up at one place
    assert brain._low_block_escape(1500, _low_edge_frame(1500), (0.0, 0.0, 0.0), True) is None
    again = brain._low_block_escape(
        1000 + B.LOW_BACK_GAP_MS + 10, _low_edge_frame(6000), (0.05, 0.0, 0.0), True
    )
    assert again is not None and again.count == 2


@pytest.mark.parametrize(
    "kw",
    [
        dict(ahead=500),  # still far
        dict(top=400, edge=0),  # a wall, not a low edge
        dict(edge=0),  # a narrow low thing (a bowl, a ball): not an edge
        dict(top=20, edge=0),  # a sill: floor to the ToF
    ],
)
def test_other_things_ahead_do_not_make_it_back_off(kw):
    brain = _brain()
    brain._req_vx = 0.35
    assert brain._low_block_escape(1000, _low_edge_frame(1000, **kw), (0.0, 0.0, 0.0), True) is None


def test_it_does_not_back_off_when_it_is_not_walking_at_it():
    brain = _brain()
    brain._req_vx = 0.0
    assert (
        brain._low_block_escape(1000, _low_edge_frame(1000, vx=0.0), (0.0, 0.0, 0.0), True) is None
    )
    brain._req_vx = 0.35
    assert brain._low_block_escape(1000, _low_edge_frame(1000), (0.0, 0.0, 0.0), False) is None


def test_the_edge_numbers_reach_the_brain_through_the_senses():
    from duckmind.duckbody import protocol as P
    from duckmind.world import neuro_senses as ns

    view = P.DepthView(
        t_ns=1_000_000_000,
        ahead_mm=300,
        left_mm=None,
        right_mm=None,
        drop_zones=0,
        touch=False,
        head_pitch_mrad=0,
        ahead_top_mm=62,
        virtual=True,
        low_edge_mm=62,
    )
    sense = ns.tof_sense(view, now_ns=1_010_000_000)
    assert (sense.ahead_mm, sense.ahead_top_mm, sense.low_edge_mm, sense.virtual) == (
        300,
        62,
        62,
        True,
    )
    plain = ns.tof_sense(None)
    assert not plain.present and plain.low_edge_mm == 0 and not plain.virtual
