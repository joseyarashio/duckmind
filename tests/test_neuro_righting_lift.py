"""A fallen duck whose lift claim stands is judged, and rights itself (docs/design/neural-duck/LIFT-LOCK-RIGHTING-2026-10-18.md).

The 2026-10-17 live round (seed 1024, 6585 s): a duck that a lift claim covered tipped onto its beak and lay at 62 degrees. Robotd said fallen, but proprio does not open a
fall episode on a claimed duck (a duck cradled in arms is no fall), the claim cannot set down above 35 degrees (its floor bands are off), and the floor re-test (the head
HOME, then up, the legs answer or not) never judged it: the trunk's weight pins the neck at +0.71 rad from HOME whatever is asked, outside the pose it waits for. Behaviour
``righting``, gate ``lifted``, the body link's righting idle, for ever.

The first group drives the real :class:`~duckmind.neuro.bodylink.BodyLink` with a small plant on a trunk that lies at 62.5 degrees with the head pinned (a leg load that
follows the head the link COMMANDS the way the headless duck's did: +15 to +30 % at the raised head on the floor, -1 to -8 % in a hand). The second is proprio alone; the
third the probe alone. The last runs the headless MuJoCo duck (skipped without MuJoCo and the official MJCF)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from duckmind.duckbody.proprio import Proprio
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.neuro.liftcheck import PROBE_RESPONSE, FloorProbe
from tests.test_neuro_bodylink import DT, state

LEGS = 15.4  # 0.44 of a quiet stand (35): the legs of a light seat
LYING_HEAD = (
    0.71,
    -0.36,
    -0.12,
    -0.08,
)  # headless: on its beak at 62.5 degrees the neck rests here whatever is asked
HANGING_HEAD = (
    -1.55,
    0.73,
    -0.05,
    0.13,
)  # headless: a duck tipped 62 degrees in a hand lets its head hang
TILT = 62.5


def commanded_up(out) -> float:
    """How far up the head the link asked it to go (0..1)."""
    return min(1.0, max(0.0, -out.head[1] / 0.35)) if out is not None else 0.0


def seat_and_claim(link: BodyLink, seconds: float = 3.0) -> float:
    """The duck sits with light legs: the claim stands (what a hand, or the lip of a ledge, makes)."""
    t = 0.0
    for k in range(int(seconds / DT)):
        t = k * DT
        link.tick(state(t, policy="sit", load=LEGS), BodyRequest(posture="sit"))
    assert link.proprio.lift == "lifted"
    return t + DT


def tipped(
    link: BodyLink,
    response: float,
    seconds: float,
    t0: float,
    *,
    head: tuple[float, float, float, float] = LYING_HEAD,
    tilt: float = TILT,
    fallen: bool = True,
    rock: float = 0.3,
) -> tuple[float, list[tuple[float, str]], list]:
    """The trunk at ``tilt`` degrees, robotd's fall flag up, the head where it rests; the leg load answers the head the link commands by ``response``, and the trunk rocks by ``rock`` degrees with it (headless, lying on its beak: 0.3; welded to a hand: 0.000 to 0.002, pass 0.0)."""
    events: list[tuple[float, str]] = []
    outs: list = []
    up = 0.0
    t = t0
    for k in range(int(round(seconds / DT))):
        t = t0 + k * DT
        out = link.tick(
            state(
                t,
                policy="sit",
                load=LEGS * (1.0 + response * up),
                head=head,
                tilt=tilt + rock * up,
                fallen=fallen,
            ),
            BodyRequest(posture="sit"),
        )
        up = commanded_up(out)
        events.extend(
            (t, e["kind"]) for e in out.events if e["kind"].startswith(("proprio.", "body."))
        )
        outs.append(out)
    return t + DT, events, outs


def kinds(events: list[tuple[float, str]], kind: str) -> list[float]:
    return [t for t, k in events if k == kind]


def test_a_fallen_duck_under_a_lift_claim_on_the_floor_is_set_down_and_righted() -> None:
    link = BodyLink()
    t = seat_and_claim(link)
    _, events, outs = tipped(link, 0.20, 80.0, t)
    assert link.lifts_refuted == 1
    down, fell = kinds(events, "proprio.set_down"), kinds(events, "proprio.fell")
    assert (
        len(down) == 1 and 10.0 <= down[0] - t <= 60.0
    )  # two re-tests, 20 s apart, found the floor under it
    assert (
        len(fell) == 1 and fell[0] - down[0] <= 3 * DT
    )  # the fall the claim had held back begins at once
    assert kinds(events, "body.fell")  # and the righting with it
    assert any(o.righting != "idle" for o in outs) and any("fallen" in o.gate for o in outs)
    assert link.proprio.lift == "ground"


def test_a_duck_propped_at_50_degrees_under_a_claim_is_set_down_and_righted_as_a_tilt_fall() -> (
    None
):
    """Robotd says upright below 60 degrees, so no fall flag ever rises: proprio's trigger B (40 degrees held 1 s) is the fall a propped duck gets, and the claim held it back too."""
    link = BodyLink()
    t = seat_and_claim(link)
    _, events, outs = tipped(link, 0.20, 80.0, t, tilt=50.0, fallen=False)
    assert link.lifts_refuted == 1
    down, fell = kinds(events, "proprio.set_down"), kinds(events, "proprio.fell")
    assert len(down) == 1 and len(fell) == 1 and 0.0 <= fell[0] - down[0] <= 1.5
    assert kinds(events, "body.fell") and any(o.righting != "idle" for o in outs)


def test_a_duck_held_tilted_in_a_hand_stays_held_through_every_re_test() -> None:
    # (legs that hang in the hand: -8 % to +50 % with the head; a trunk a hand holds does not rock)
    for response in (0.0, -0.02, -0.08, 0.13, 0.20, 0.48):
        link = BodyLink()
        t = seat_and_claim(link)
        _, events, outs = tipped(link, response, 150.0, t, head=HANGING_HEAD, rock=0.001)
        assert link.floor_probes >= 6 and link.lifts_refuted == 0, response
        assert not kinds(events, "proprio.set_down") and not kinds(events, "proprio.fell"), response
        assert not kinds(events, "body.fell") and link.proprio.lift == "lifted", response
        assert all(o.righting == "idle" and "lifted" in o.gate for o in outs), response


def test_a_duck_that_gets_a_hand_after_the_floor_was_found_is_still_found() -> None:
    link = BodyLink()
    t = seat_and_claim(link)
    t, events, _ = tipped(link, 0.20, 80.0, t)
    assert link.lifts_refuted == 1
    # up again (the ordinary righting, tilt 0), sitting on the floor with light legs; then a hand takes it and tips it
    for k in range(int(8.0 / DT)):
        link.tick(state(t + k * DT, policy="stand", load=35.0), BodyRequest(posture="stand"))
    t += 8.0
    t = seat_and_claim_again(link, t)
    _, more, _ = tipped(link, 0.0, 60.0, t, head=HANGING_HEAD, rock=0.001)
    assert link.proprio.lift == "lifted" and not kinds(more, "proprio.set_down")


def seat_and_claim_again(link: BodyLink, t0: float) -> float:
    t = t0
    for k in range(int(4.0 / DT)):
        t = t0 + k * DT
        link.tick(state(t, policy="sit", load=LEGS * 0.9), BodyRequest(posture="sit"))
    assert link.proprio.lift == "lifted"
    return t + DT


# ── proprio: the fall a claim held back ───────────────────────────────────────


def _ticks(p: Proprio, t0: float, seconds: float, **kw):
    edges: list[tuple[float, str, str | None]] = []
    t = t0
    for k in range(int(seconds / DT)):
        t = t0 + k * DT
        out = p.step(state(t, **kw), cmd=(0.0, 0.0))
        edges.extend((t, e, out.fell_trigger) for e in out.edges)
    return t + DT, edges


def test_the_fall_a_lift_claim_held_back_begins_when_the_claim_is_dropped() -> None:
    p = Proprio()
    t, _ = _ticks(p, 0.0, 3.0, policy="sit", load=LEGS)
    assert p.lift == "lifted"
    t, edges = _ticks(p, t, 20.0, policy="sit", load=LEGS, tilt=TILT, fallen=True)
    assert (
        not [e for _, e, _ in edges if e == "fell"] and p.lift == "lifted"
    )  # a duck cradled on its side is no fall
    p.drop_claim()  # the body link judged the claim false
    t, edges = _ticks(p, t, 0.1, policy="sit", load=LEGS, tilt=TILT, fallen=True)
    assert [(e, trig) for _, e, trig in edges if e == "fell"] == [("fell", "fallen")]
    assert p.fall_open


def test_a_claim_that_stands_never_lets_a_fallen_flag_open_a_fall() -> None:
    p = Proprio()
    t, _ = _ticks(p, 0.0, 3.0, policy="sit", load=LEGS)
    _, edges = _ticks(p, t, 300.0, policy="sit", load=LEGS, tilt=TILT, fallen=True)
    assert not [e for _, e, _ in edges if e in ("fell", "set_down")] and p.lift == "lifted"


# ── the probe: a head that cannot be asked for a pose ─────────────────────────


def drive(
    load_at,
    *,
    tilted: bool,
    head: tuple[float, float, float, float] = LYING_HEAD,
    drift: float = 0.0,
    gyro: float = 0.02,
    rock: float = 0.3,
) -> str | None:
    """The probe on a 50 Hz clock with the head where it rests (it never follows the probe's pose); ``load_at(up)`` with ``up`` the probe's asked pose (0 HOME, 1 raised)."""
    probe = FloorProbe(0, tilted=tilted)
    for k in range(1500):
        t = k * 20
        up = 0.0 if probe.phase == "home" else 1.0
        h = (head[0] + drift * math.sin(k / 7.0), head[1], head[2], head[3])
        r = math.radians(TILT + rock * up)
        verdict = probe.step(
            t, load_at(up), h, gyro, ended=False, down=(math.sin(r), 0.0, -math.cos(r))
        )
        if verdict is not None:
            return verdict
    return None


def test_a_pinned_head_is_read_at_rest_when_the_trunk_is_tilted() -> None:
    assert drive(lambda up: 17.8 * (1.0 + 0.15 * up), tilted=True) == "supported"
    assert drive(lambda up: 16.3 * (1.0 + 0.29 * up), tilted=True) == "supported"
    # without the tilt the probe waits for a pose the neck cannot take (the lock)
    assert drive(lambda up: 17.8 * (1.0 + 0.15 * up), tilted=False) == "unclear"


def test_a_tilted_duck_whose_legs_hang_in_a_hand_is_not_set_down_by_its_leg_load_alone() -> None:
    """Headless, a duck tipped on its side in the world's hand: the hanging legs' small load moves +10 to +50 % with the head, but the trunk does not move (0.000 to 0.002 degrees)."""
    for response in (0.13, 0.20, 0.48):
        assert (
            drive(lambda up, r=response: 7.0 * (1.0 + r * up), tilted=True, rock=0.001) == "held"
        ), response
        assert (
            drive(lambda up, r=response: 7.0 * (1.0 + r * up), tilted=True, rock=0.3) == "supported"
        ), response


def test_a_tilted_duck_in_a_hand_does_not_answer() -> None:
    for response in (0.0, -0.007, -0.04, -0.08):
        verdict = drive(
            lambda up, r=response: 18.0 * (1.0 + r * up), tilted=True, head=HANGING_HEAD
        )
        assert verdict == "held", response
    assert PROBE_RESPONSE > 0.0


def test_a_head_that_keeps_moving_is_not_judged() -> None:
    assert drive(lambda up: 17.8 * (1.0 + 0.2 * up), tilted=True, drift=0.4) == "unclear"
    assert (
        drive(lambda up: 17.8 * (1.0 + 0.2 * up), tilted=True, gyro=0.6) == "unclear"
    )  # a trunk that is carried


# ── the whole headless duck ───────────────────────────────────────────────────

_MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF", "")


def _need_headless() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    if not (_MJCF and (Path(_MJCF) / "robot_allcollisions.xml").is_file()):
        pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)")


def _tipped_headless(mode: str, axis: str, deg: float, seconds: float):
    """tools/righting_lift/repro_lock.py's scene, in process: pick the seated duck up, tip it in the hand, then let go over the floor (``lie``) or never (``held``)."""
    from dataclasses import replace

    import mujoco
    import numpy as np

    from duckmind.duckbody.robotd_client import RobotState
    from duckmind.neuro import motor as M
    from duckmind.neuro.bodylink import config_for_motor
    from duckmind.world.headless_duck import HeadlessDuck

    spot = (0.30, -0.60, 1.05)
    duck = HeadlessDuck(
        "real_home",
        mjcf_dir=_MJCF,
        keyframe="STAND",
        x=spot[0],
        y=spot[1],
        yaw=spot[2],
        command_delay_ms=20.0,
    )
    duck.stand(timeout=10.0, hold=1.0)
    duck.teleport(*spot)
    duck.run(0.6)
    link = BodyLink(replace(config_for_motor(M.COMMAND_MAX)))
    hand, data = duck.kit.hand, duck.data
    t0, phase, ref = duck.now(), "pick", None
    events: list[tuple[float, str]] = []
    wrong = False
    while True:
        now = duck.now() - t0
        if phase == "pick" and now >= 2.0:
            duck.hand({"do": "pickup"})
            phase = "hold"
        if (
            phase == "hold"
            and duck.truth()["held"]
            and link.proprio.lift == "lifted"
            and link.posture_want == "sit"
            and now > 8.0
        ):
            half = math.radians(deg) / 2.0
            dq = np.array(
                [math.cos(half), 0, math.sin(half), 0]
                if axis == "y"
                else [math.cos(half), math.sin(half), 0, 0]
            )
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, data.mocap_quat[hand.mocap].copy(), dq)
            data.mocap_quat[hand.mocap] = q
            phase, ref = "tipped", now
        if phase == "tipped" and mode == "lie" and now > ref + 1.0:
            phase = "lowering"
        if phase == "lowering":
            data.mocap_pos[hand.mocap][2] -= 0.0008
            if float(data.xpos[duck.trunk][2]) < 0.075:
                data.eq_active[int(hand.weld)] = 0
                hand.holding, hand.script = False, None
                phase, ref = "lying", now
        if ref is not None and now > ref + seconds:
            break
        duck.tick()
        out = link.tick(RobotState.from_wire(duck.state()), BodyRequest(posture="stand"))
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
            elif i.method == "robot.do":
                duck.do(str(p.get("skill", "")))
        events.extend(
            (now, e["kind"]) for e in out.events if e["kind"].startswith(("proprio.", "body."))
        )
        wrong = wrong or (
            phase in ("tipped", "hold")
            and duck.truth()["held"]
            and not out.prox.view.lifted
            and now > 5.0
        )
    return link, duck, events, wrong


def test_headless_a_duck_dropped_on_its_beak_is_set_down_and_gets_up() -> None:
    _need_headless()
    link, duck, events, _ = _tipped_headless("lie", "y", -62.0, 90.0)
    assert not duck.truth()["held"]
    assert (
        kinds(events, "proprio.set_down")
        and kinds(events, "body.fell")
        and kinds(events, "body.righting")
    )
    assert duck.tilt_deg() < 25.0 and link.proprio.lift == "ground" and link.lifts_refuted == 1


@pytest.mark.parametrize(
    "axis,deg",
    [
        ("y", -62.0),
        ("y", -40.0),
        ("x", 90.0),
        ("x", -90.0),
        ("y", -90.0),
        (
            "x",
            45.0,
        ),  # the hanging legs' load answers the head here (+10 to +50 %): the trunk does not
        ("x", 62.0),
        ("x", -62.0),
    ],
)
def test_headless_a_duck_tipped_in_the_worlds_hand_is_never_let_go(axis: str, deg: float) -> None:
    _need_headless()
    link, duck, events, wrong = _tipped_headless("held", axis, deg, 100.0)
    assert duck.truth()["held"] and link.proprio.lift == "lifted" and not wrong
    assert link.floor_probes >= 4 and link.lifts_refuted == 0
    assert not kinds(events, "proprio.set_down") and not kinds(events, "body.fell")
