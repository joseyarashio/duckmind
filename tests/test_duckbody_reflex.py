"""Reflexes (duckmind.duckbody.reflex; spec §5.3) and owner A1's one relaxation."""

from __future__ import annotations

import pytest

from duckmind.duckbody.protocol import DepthView, ProprioView, Toward, WalkCmd
from duckmind.duckbody.reflex import CLEAR, relaxed_near, verdict


def depth(ahead: int | None = 1500, *, drop: int = 0, touch: bool = False, b: int = 0) -> DepthView:
    return DepthView(
        t_ns=0,
        ahead_mm=ahead,
        left_mm=None,
        right_mm=None,
        drop_zones=drop,
        touch=touch,
        head_pitch_mrad=0,
        ahead_b_mrad=b,
    )


def walk(purpose: str = "wander", toward: Toward | None = None, stop: int = 300) -> WalkCmd:
    return WalkCmd(
        act_id="W1",
        deadline_ms=99_999,
        vx_mm=400,
        purpose=purpose,
        toward=toward,
        stop_ahead_mm=stop,
    )


BALL = Toward(look_id="L9", b_mrad=40, r_mm=600, seen_t_ms=10_000, thing="ball")


def test_priorities() -> None:
    body = ProprioView(posture="stand", policy="walk")
    assert verdict(proprio=body, depth=depth(), cmd=walk(), now_ms=0) == CLEAR
    assert verdict(proprio=body, depth=depth(250), cmd=walk(), now_ms=0).reason == "reflex_near"
    assert (
        verdict(proprio=body, depth=depth(250, drop=4), cmd=walk(), now_ms=0).reason
        == "reflex_drop"
    )
    assert (
        verdict(proprio=body, depth=depth(250, touch=True), cmd=walk(), now_ms=0).reason
        == "reflex_touch"
    )
    stalled = ProprioView(posture="stand", policy="walk", blocked=True)
    assert verdict(proprio=stalled, depth=None, cmd=walk(), now_ms=0).reason == "stall"
    unsure = ProprioView(posture="stand", policy="walk", unloaded=True)
    assert (
        verdict(proprio=unsure, depth=depth(250), cmd=walk(), now_ms=0).reason == "reflex_unloaded"
    )
    lifted = ProprioView(posture="stand", policy="walk", lifted=True, unloaded=True)
    assert verdict(proprio=lifted, depth=None, cmd=None, now_ms=0).reason == "lifted"
    fallen = ProprioView(posture="stand", policy="walk", lifted=True, fallen=True)
    assert verdict(proprio=fallen, depth=None, cmd=None, now_ms=0).reason == "fell"
    assert (
        verdict(proprio=body, depth=None, cmd=walk(), now_ms=0, dark_over_cap=True).reason
        == "reflex_dark"
    )


def test_a_walk_cannot_lower_its_own_stop() -> None:
    body = ProprioView(posture="stand", policy="walk")
    got = verdict(proprio=body, depth=depth(250), cmd=walk(stop=100), now_ms=0)
    assert got.reason == "reflex_near"  # 300 mm unless A1 applies
    assert verdict(
        proprio=body, depth=depth(450), cmd=walk(stop=500), now_ms=0
    ).halt  # stricter is fine


def test_a1_relaxes_only_toward_the_just_seen_target() -> None:
    body = ProprioView(posture="stand", policy="walk")
    curio = walk("curio", BALL, stop=80)
    # the ball ahead where it should be, seen 1 s ago: walk on to 80 mm
    assert relaxed_near(curio, depth(250, b=60), 11_000, walked_mm=300) == 80
    assert not verdict(
        proprio=body, depth=depth(250, b=60), cmd=curio, now_ms=11_000, walked_mm=300
    ).halt
    stop = verdict(proprio=body, depth=depth(70, b=60), cmd=curio, now_ms=11_000, walked_mm=500)
    assert stop.halt and stop.reason == "reflex_near" and stop.relaxed_mm == 80


@pytest.mark.parametrize(
    "cmd,view,now",
    [
        (walk("wander", BALL, stop=80), depth(250, b=60), 11_000),  # not a curiosity walk
        (walk("curio", None, stop=80), depth(250, b=60), 11_000),  # no target
        (walk("curio", BALL, stop=80), depth(250, b=60), 14_000),  # seen 4 s ago: stale
        (walk("curio", BALL, stop=80), depth(250, b=600), 11_000),  # the return is off to the side
        (walk("curio", BALL, stop=80), depth(900, b=40), 11_000),  # farther than the ball was
        (walk("curio", BALL, stop=80), depth(100, b=40), 10_100),  # much nearer than the ball
    ],
)
def test_a1_does_not_relax_otherwise(cmd: WalkCmd, view: DepthView, now: int) -> None:
    assert relaxed_near(cmd, view, now, walked_mm=0) is None
    if view.ahead_mm is not None and view.ahead_mm <= 300:
        body = ProprioView(posture="stand", policy="walk")
        assert verdict(proprio=body, depth=view, cmd=cmd, now_ms=now).reason == "reflex_near"


def test_a1_leaves_every_other_reflex_alone() -> None:
    body = ProprioView(posture="stand", policy="walk")
    curio = walk("curio", BALL, stop=80)
    assert (
        verdict(
            proprio=body,
            depth=depth(250, b=60, touch=True),
            cmd=curio,
            now_ms=11_000,
            walked_mm=300,
        ).reason
        == "reflex_touch"
    )
    assert (
        verdict(
            proprio=body, depth=depth(250, b=60, drop=5), cmd=curio, now_ms=11_000, walked_mm=300
        ).reason
        == "reflex_drop"
    )
    unsure = ProprioView(posture="stand", policy="walk", unloaded=True)
    assert verdict(proprio=unsure, depth=depth(250, b=60), cmd=curio, now_ms=11_000).halt
