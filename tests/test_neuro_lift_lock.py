"""A lift claim that never ends (docs/design/neural-duck/DOCK-FALL-2026-10-11.md §7.6).

A seated duck resting partly on a ledge (the dock's guide, a door sill) reads legs as light as a hand's (0.42-0.53 of a quiet stand
against 0.38-0.40), so proprio claims ``lifted`` and the floor band that sets it down is never read. The body link re-tests a claim that
stands (the head goes up and the legs answer or not) and ends a refuted one through the ordinary ``set_down`` edge.

The first group drives the real :class:`~duckmind.neuro.bodylink.BodyLink` with a small plant (a leg load that follows the head the way
the headless duck's did: +25 % at the raised head on a ledge, 0 % in a hand). The second runs the same link on the headless MuJoCo duck
(skipped without MuJoCo and the official MJCF)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from duckmind.neuro.bodylink import BodyLink, BodyRequest
from tests.test_neuro_bodylink import DT, state

WEDGE_LEGS = 15.4  # 0.44 of a quiet stand (35): a seat on the dock's guide
HELD_LEGS = 13.5  # 0.387: a hand
FLOOR_LEGS = 24.4  # 0.70: a flat floor


def plant(
    link: BodyLink, legs: float, response: float, seconds: float, t0: float, head: list[float]
) -> tuple[float, list[tuple[float, str]]]:
    """Seated duck for ``seconds``: the leg load rises by ``response`` as the head goes up; the head follows what the link sent."""
    events: list[tuple[float, str]] = []
    t = t0
    for k in range(int(round(seconds / DT))):
        t = t0 + k * DT
        up = min(1.0, max(0.0, -head[1] / 0.35))
        out = link.tick(
            state(t, policy="sit", load=legs * (1.0 + response * up), head=tuple(head)),
            BodyRequest(posture="sit"),
        )
        for i in range(4):
            head[i] += (out.head[i] - head[i]) * 0.1
        events.extend((t, e["kind"]) for e in out.events if e["kind"].startswith("proprio."))
    return t + DT, events


def kinds(events: list[tuple[float, str]], kind: str) -> list[float]:
    return [t for t, k in events if k == kind]


def test_a_seat_on_a_ledge_is_not_held_for_ever() -> None:
    link = BodyLink()
    _, events = plant(link, WEDGE_LEGS, 0.25, 70.0, 0.0, [0.0] * 4)
    assert len(kinds(events, "proprio.lifted")) == 1  # the light legs did claim it
    down = kinds(events, "proprio.set_down")
    assert len(down) == 1 and 20.0 <= down[0] <= 60.0  # two re-tests, 20 s apart, ended it
    assert link.proprio.lift == "ground"
    assert not kinds(events, "proprio.unloaded")[
        1:
    ]  # and the same light legs did not claim it again


def test_a_duck_in_a_hand_stays_held_through_every_re_test() -> None:
    link = BodyLink()
    head = [0.0] * 4
    t, events, starts, seen = 0.0, [], [], 0
    for _ in range(12):  # 12 x 20 s
        t, got = plant(link, HELD_LEGS, 0.0, 20.0, t, head)
        events += got
        if link.floor_probes != seen:
            seen = link.floor_probes
            starts.append(t)
    assert link.floor_probes >= 8  # it was re-tested again and again
    assert len(kinds(events, "proprio.lifted")) == 1 and not kinds(events, "proprio.set_down")
    assert link.proprio.lift == "lifted" and link.lifts_refuted == 0


def test_the_re_test_is_rate_limited() -> None:
    link = BodyLink()
    head = [0.0] * 4
    t, starts, seen = 0.0, [], 0
    for _ in range(300):  # 1 s steps
        t, _ = plant(link, HELD_LEGS, 0.0, 1.0, t, head)
        if link.floor_probes != seen:
            seen = link.floor_probes
            starts.append(t)
    assert len(starts) >= 6
    assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= 19.0


def test_a_seat_on_the_floor_and_a_standing_duck_are_never_re_tested() -> None:
    link = BodyLink()
    _, events = plant(link, FLOOR_LEGS, 0.02, 60.0, 0.0, [0.0] * 4)
    assert not events and link.floor_probes == 0


def test_a_hand_that_lifts_a_ledge_seated_duck_is_found_after_the_claim_was_refuted() -> None:
    link = BodyLink()
    head = [0.0] * 4
    t, events = plant(link, WEDGE_LEGS, 0.25, 70.0, 0.0, head)
    assert link.proprio.lift == "ground" and link.lifts_refuted == 1
    t0 = t
    t, more = plant(link, HELD_LEGS, 0.0, 45.0, t, head)  # a hand takes it off the ledge
    lifted = kinds(more, "proprio.lifted")
    assert lifted and lifted[0] - t0 <= 30.0  # the next re-test found the legs carry nothing
    assert link.proprio.lift == "lifted"


def test_a_fall_in_the_middle_of_a_re_test_ends_it_unjudged() -> None:
    link = BodyLink()
    head = [0.0] * 4
    t, _ = plant(
        link, WEDGE_LEGS, 0.25, 12.0, 0.0, head
    )  # the claim has stood past retest_after_s: a re-test is running
    assert link.probe is not None or link.floor_probes == 1
    for k in range(60):
        link.tick(
            state(t + k * DT, policy="sit", load=WEDGE_LEGS, head=tuple(head), fallen=True),
            BodyRequest(posture="sit"),
        )
    assert link.probe is None and link.lifts_refuted == 0 and link.probe_votes == 0


_MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")


def _need_headless() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    if not (_MJCF and (Path(_MJCF) / "robot_allcollisions.xml").is_file()):
        pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)")


def _headless(x: float, y: float, yaw: float, seconds: float, pick: float | None = None):
    from dataclasses import replace

    from duckmind.duckbody.robotd_client import RobotState
    from duckmind.neuro import motor as M
    from duckmind.neuro.bodylink import config_for_motor
    from duckmind.world.headless_duck import HeadlessDuck

    duck = HeadlessDuck("real_home", mjcf_dir=_MJCF, keyframe="STAND", x=x, y=y, yaw=yaw)
    duck.stand(timeout=10.0, hold=1.0)
    duck.teleport(x, y, yaw)
    duck.run(0.6)
    link = BodyLink(replace(config_for_motor(M.COMMAND_MAX)))
    t0, picked, edges, wrong = duck.now(), False, [], False
    prev = False
    while duck.now() - t0 < seconds:
        into = duck.now() - t0
        if pick is not None and not picked and into >= pick:
            duck.hand({"do": "pickup"})
            picked = True
        duck.tick()
        out = link.tick(RobotState.from_wire(duck.state()), BodyRequest(posture="sit"))
        for i in out.intents:
            p = dict(i.params or {})
            if i.method == "robot.move":
                duck.move(
                    float(p.get("vx", 0.0)), float(p.get("vy", 0.0)), float(p.get("vyaw", 0.0))
                )
            elif i.method == "robot.head":
                duck.head(
                    [
                        float(p.get(n, 0.0))
                        for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
                    ]
                )
            elif i.method == "robot.mouth":
                duck.mouth(float(p.get("open", 0.0)))
            elif i.method == "robot.do":
                duck.do(str(p.get("skill", "")))
        edges += [
            (round(into, 1), e["kind"]) for e in out.events if e["kind"].startswith("proprio.")
        ]
        held = bool(duck.truth()["held"])  # truth: a label only, never read by the link
        wrong = wrong or (held and prev and not out.prox.view.lifted)
        prev = out.prox.view.lifted
    return link, edges, bool(duck.truth()["held"]), wrong


@pytest.mark.parametrize("spot", [(0.92, -2.51, 1.05), (0.98, -2.44, 1.0), (2.32, 0.60, 0.0)])
def test_headless_a_duck_seated_on_the_guide_or_the_sill_is_let_go(
    spot: tuple[float, float, float],
) -> None:
    _need_headless()
    link, edges, held, _ = _headless(*spot, 55.0)
    assert not held
    assert any(k == "proprio.lifted" for _, k in edges)
    assert any(k == "proprio.set_down" for _, k in edges) and link.proprio.lift == "ground"


def test_headless_a_duck_in_the_world_s_hand_is_never_let_go() -> None:
    _need_headless()
    link, edges, held, wrong = _headless(0.30, -0.60, 1.05, 90.0, pick=5.0)
    assert held and link.proprio.lift == "lifted"
    assert link.floor_probes >= 3 and link.lifts_refuted == 0 and not wrong
    assert not any(k == "proprio.set_down" for _, k in edges)
