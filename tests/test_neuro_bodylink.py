"""neuro:live's body link: the neural duck's motor output onto robotd under duck-body's safety
(duckmind.neuro.bodylink; spec §5.3, §5.4; owner A3, B1, B2, B4; 2026-09-27 pet feedback)."""

from __future__ import annotations

import math

import pytest

from duckmind.duckbody import protocol as P
from duckmind.duckbody.reflex import GuardContext, check_intent
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro.bodylink import (
    WALK_HEAD,
    BodyLink,
    BodyRequest,
    clamp_head,
    head_inside,
    head_offsets_of,
    shape_twist,
)

HOME_JOINTS = (0.3491, 0.3491, 0.0, 0.0)
DT = 0.02


def state(
    t: float,
    *,
    policy: str = "stand",
    fallen: bool = False,
    tilt: float = 2.0,
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    load: float = 45.0,
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
) -> RobotState:
    """One robot.state tick: ``head`` offsets from HOME, ``load`` the ten legs' sum."""
    joints = [0.0] * 15
    for i, (home, off) in enumerate(zip(HOME_JOINTS, head, strict=True)):
        joints[5 + i] = home + off
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = load / 10.0
    r = math.radians(tilt)
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {
                "fallen": fallen,
                "limp": False,
                "gravity": [math.sin(r), 0.0, -math.cos(r)],
            },
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": list(gyro), "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": yaw},
        }
    )


def moves(intents) -> list[tuple[float, float]]:
    return [i.twist for i in intents if i.method == "robot.move"]


def run(link: BodyLink, request: BodyRequest, t0: float, seconds: float, **kw):
    outs = []
    for k in range(int(round(seconds / DT))):
        outs.append(link.tick(state(t0 + k * DT, **kw), request))
    return outs


def test_the_twist_follows_the_walking_rules() -> None:
    assert shape_twist(0.0, 0.8) == (0.35, 0.8, ("arc",))  # turns are arcs, never in place
    assert shape_twist(0.2, 0.0) == (0.35, 0.0, ("min_speed",))  # velstand needs ≥ 0.30
    assert shape_twist(0.1, 0.0) == (0.0, 0.0, ())  # a pause
    assert shape_twist(0.7, 0.0) == (0.55, 0.0, ("cap",))  # owner A3 cap
    assert shape_twist(0.5, 0.0, approach=True) == (0.40, 0.0, ("cap",))  # approaching: 0.40
    assert shape_twist(-0.3, 0.0) == (0.0, 0.0, ("no_reverse",))
    assert shape_twist(-0.6, 0.2, docking=True) == (-0.40, 0.2, ())  # docking may reverse
    assert shape_twist(0.45, 1.7)[1] == 1.0


def test_a_walk_is_streamed_every_tick_and_stopped_with_one_zero() -> None:
    link = BodyLink()
    walk = BodyRequest(vx=0.45, vyaw=0.3, posture="stand")
    outs = run(link, walk, 0.0, 1.0)
    streamed = [moves(o.intents) for o in outs]
    assert all(m == [(0.45, 0.3)] for m in streamed[1:])  # the deadman is fed every 20 ms
    assert all(not o.refused for o in outs)
    stop = run(link, BodyRequest(posture="stand"), 1.0, 0.5)
    assert moves(stop[0].intents) == [(0.0, 0.0)]
    assert all(moves(o.intents) == [] for o in stop[1:])  # nothing else writes the twist


def test_every_intent_passes_the_guard_with_a_live_head_inside_the_walking_envelope() -> None:
    link = BodyLink()
    head = (0.05, -0.1, 0.25, 0.08)  # a looking-around head, inside WALK_HEAD
    for k in range(100):
        out = link.tick(state(k * DT, head=head), BodyRequest(vx=0.5, head=head, posture="stand"))
        guard = GuardContext(head_home_ms=10_000)
        assert all(check_intent(i, guard) is None for i in out.intents)
    assert out.twist == (0.5, 0.0)


def test_a_head_outside_the_walking_envelope_holds_the_walk_and_is_brought_in() -> None:
    link = BodyLink()
    up = (0.0, -0.5, 0.0, 0.0)  # looking up further than the measured walking envelope
    out = link.tick(state(0.0, head=up), BodyRequest(vx=0.5, head=up, posture="stand"))
    assert out.twist == (0.0, 0.0) and "head" in out.gate
    assert out.head == clamp_head(up, WALK_HEAD)  # the head is sent into the envelope
    assert head_inside(out.head, WALK_HEAD, 0.0)
    side = (0.0, 0.0, 0.9, 0.0)  # a big look to the side is fine while walking (pet:probe)
    got = BodyLink().tick(state(0.0, head=side), BodyRequest(vx=0.5, head=side, posture="stand"))
    assert got.twist == (0.5, 0.0) and "head" not in got.gate


def test_the_walking_envelope_is_the_measured_one_and_the_motor_commands_fit_inside() -> None:
    from duckmind.neuro.bodylink import config_for_motor

    assert WALK_HEAD == ((-0.50, 0.25), (-0.35, 0.70), (-1.0, 1.0), (-0.31, 0.31))
    cfg = config_for_motor((0.40, 0.55, 0.95, 0.31))
    assert cfg.walk_head == WALK_HEAD
    assert cfg.walk_cmd_head == ((-0.40, 0.25), (-0.35, 0.55), (-0.95, 0.95), (-0.31, 0.31))
    assert cfg.still_head == ((-0.40, 0.40), (-0.55, 0.55), (-0.95, 0.95), (-0.31, 0.31))


def test_a_seated_duck_stands_up_before_it_walks_and_retries_the_toggle() -> None:
    link = BodyLink()
    walk = BodyRequest(vx=0.45, posture="stand")
    outs = run(link, walk, 0.0, 4.0, policy="sit", load=25.0)
    toggles = [k for k, o in enumerate(outs) for i in o.intents if i.method == "robot.do"]
    # the head settles at HOME for 0.3 s first (rule 3), then sit_toggle, again after 3 s
    assert toggles[0] == 15 and toggles[1] == 165
    assert all(o.twist == (0.0, 0.0) and "posture" in o.gate for o in outs)
    assert all(
        i.params == {"skill": "sit_toggle"}
        for o in outs
        for i in o.intents
        if i.method == "robot.do"
    )


def test_a_fall_holds_the_body_still_head_home_and_rights_through_velstand() -> None:
    link = BodyLink()
    run(link, BodyRequest(vx=0.45, posture="stand"), 0.0, 1.0)
    outs = run(
        link,
        BodyRequest(vx=0.45, head=(0.0, 0.1, 0.2, 0.0), posture="stand"),
        1.0,
        1.0,
        fallen=True,
        tilt=75.0,
        policy="stand",
    )
    kinds = [e["kind"] for o in outs for e in o.events]
    assert "body.fell" in kinds and "proprio.fell" in kinds
    assert all(o.twist == (0.0, 0.0) for o in outs)
    assert all("fallen" in o.gate for o in outs)
    assert outs[-1].head == (0.0, 0.0, 0.0, 0.0)  # the head HOME while righting
    assert outs[-1].righting == "try1"
    up = run(link, BodyRequest(posture="stand"), 2.0, 1.5)
    assert any(e["kind"] == "body.righting" and e["ok"] for o in up for e in o.events)


def test_the_duck_sits_when_it_senses_it_is_lifted_owner_b4() -> None:
    link = BodyLink()
    run(link, BodyRequest(posture="stand"), 0.0, 2.0)  # standing on the floor, settled
    outs = run(link, BodyRequest(vx=0.5, posture="stand"), 2.0, 1.0, load=8.0)
    assert any("proprio.lifted" == e["kind"] for o in outs for e in o.events)
    unsure = next(
        k for k, o in enumerate(outs) for e in o.events if e["kind"] == "proprio.unloaded"
    )
    assert unsure * DT <= 0.2  # the legs went light: it stops within 0.2 s …
    assert all(o.twist == (0.0, 0.0) for o in outs[unsure:])  # … and walks no more
    dos = [i for o in outs for i in o.intents if i.method == "robot.do"]
    assert dos and dos[0].params == {"skill": "sit_toggle"}
    assert outs[-1].posture_want == "sit" and "lifted" in outs[-1].gate


def test_under_30_percent_battery_it_does_not_stand_or_walk() -> None:
    link = BodyLink()
    out = link.tick(state(0.0), BodyRequest(vx=0.5, posture="stand"), battery_pct=25)
    assert out.twist == (0.0, 0.0) and "battery" in out.gate and out.posture_want == "sit"
    seated = BodyLink()
    got = seated.tick(
        state(0.0, policy="sit", load=25.0), BodyRequest(posture="stand"), battery_pct=20
    )
    assert not [i for i in got.intents if i.method == "robot.do"]  # never stands up


def test_a_touch_stops_the_walk_then_only_arcs_turn_away() -> None:
    link = BodyLink()
    run(link, BodyRequest(vx=0.5, posture="stand"), 0.0, 0.5)
    touch = P.DepthView(0, 80, None, None, 0, True, 0)
    out = link.tick(state(0.5), BodyRequest(vx=0.5, posture="stand"), depth=touch)
    assert out.twist == (0.0, 0.0) and "reflex_touch" in out.gate
    held = run(link, BodyRequest(vx=0.5, vyaw=0.8, posture="stand"), 0.52, 0.4)
    assert all(o.twist == (0.0, 0.0) for o in held)  # the reflex hold (0.5 s)
    straight = link.tick(state(1.1), BodyRequest(vx=0.5, posture="stand"))
    assert straight.twist == (0.0, 0.0) and "veer" in straight.gate
    arc = link.tick(state(1.12), BodyRequest(vx=0.5, vyaw=0.8, posture="stand"))
    assert arc.twist == (0.5, 0.8)  # turning away is allowed
    # not a set time: straight on stays barred until the heading has turned away (odometry)
    still_facing = link.tick(state(3.2, yaw=0.2), BodyRequest(vx=0.5, posture="stand"))
    assert still_facing.twist == (0.0, 0.0) and "veer" in still_facing.gate
    turned = link.tick(state(3.4, yaw=0.6), BodyRequest(vx=0.5, posture="stand"))
    assert turned.twist == (0.5, 0.0)


def test_a_docking_reverse_passes_the_guard_and_nothing_else_reverses() -> None:
    link = BodyLink()
    out = run(link, BodyRequest(vx=-0.4, posture="stand", docking=True), 0.0, 0.2)
    assert out[-1].twist == (-0.4, 0.0) and not out[-1].refused
    other = BodyLink()
    got = run(other, BodyRequest(vx=-0.4, posture="stand"), 0.0, 0.2)
    assert got[-1].twist == (0.0, 0.0) and "no_reverse" in got[-1].gate


def test_the_beak_and_the_head_are_resent_on_change_and_every_half_second() -> None:
    link = BodyLink()
    outs = run(link, BodyRequest(head=(0.0, 0.1, 0.0, 0.0), mouth=0.3, posture="stand"), 0.0, 1.2)
    heads = [k for k, o in enumerate(outs) if any(i.method == "robot.head" for i in o.intents)]
    mouths = [k for k, o in enumerate(outs) if any(i.method == "robot.mouth" for i in o.intents)]
    assert heads == [0, 25, 50] and mouths == [0, 25, 50]
    wide = link.tick(state(1.2), BodyRequest(mouth=0.9, posture="stand"))
    assert wide.mouth == pytest.approx(0.5)  # the validated opening caps the beak


def test_clip_units_become_robot_head_offsets() -> None:
    down = head_offsets_of({"np": 0.5, "hp": -0.5})  # lookplan.HALF_DOWN
    assert down == pytest.approx((-0.125, 0.175, 0.0, 0.0))
    left = head_offsets_of({"hy": 1.0}, side=1)
    assert left[2] == pytest.approx(0.6)


def test_a_posture_change_waits_for_the_head_to_settle_at_home() -> None:
    link = BodyLink()
    looking = (0.0, 0.2, 0.3, 0.0)  # looking down to the side while sitting
    outs = run(
        link,
        BodyRequest(head=looking, posture="stand"),
        0.0,
        1.0,
        policy="sit",
        load=25.0,
        head=looking,
    )
    assert not [i for o in outs for i in o.intents if i.method == "robot.do"]
    assert outs[-1].head == (0.0, 0.0, 0.0, 0.0)  # it is asked home first
    home = run(link, BodyRequest(head=looking, posture="stand"), 1.0, 0.5, policy="sit", load=25.0)
    dos = [k for k, o in enumerate(home) for i in o.intents if i.method == "robot.do"]
    assert dos == [15]  # 0.3 s after the measured head reached HOME
    assert all(o.head == (0.0, 0.0, 0.0, 0.0) for o in home)


def test_standing_with_the_neck_held_forward_still_sits_down() -> None:
    """Live 2026-09-27: standing, the walking network holds the neck ~0.13 rad forward of HOME
    whatever robot.head asks; the posture change must not wait for a neck that never comes."""
    link = BodyLink()
    neck = (-0.14, 0.0, 0.0, 0.0)
    outs = run(link, BodyRequest(posture="sit"), 0.0, 1.0, head=neck)
    dos = [k for k, o in enumerate(outs) for i in o.intents if i.method == "robot.do"]
    assert dos == [15]


def test_a_posture_change_waits_at_most_three_seconds_for_the_head() -> None:
    link = BodyLink()
    stuck = (0.0, 0.3, 0.0, 0.0)  # a head that never reaches HOME
    outs = run(link, BodyRequest(posture="sit"), 0.0, 4.0, head=stuck)
    dos = [k for k, o in enumerate(outs) for i in o.intents if i.method == "robot.do"]
    assert dos == [150]
    assert "posture_head" in outs[10].gate


def test_after_a_hit_on_one_side_only_an_arc_away_from_it_may_go_on() -> None:
    from duckmind.neuro.bodylink import _away

    assert (_away("left"), _away("right"), _away("front"), _away("unknown")) == (-1, 1, 0, 0)
    link = BodyLink()
    run(link, BodyRequest(vx=0.5, posture="stand"), 0.0, 0.5)
    # (as after proprio's blocked edge with the contact on the left: hold, then veer right)
    t = 0.5
    link.hold_until_ns = state(t).t_ns + int(0.1e9)
    link.veer_until_ns = link.hold_until_ns + int(2e9)
    link.veer_sign = _away("left")
    held = link.tick(state(t), BodyRequest(vx=0.5, vyaw=-0.8, posture="stand"))
    assert held.twist == (0.0, 0.0)
    into = link.tick(state(t + 0.2), BodyRequest(vx=0.5, vyaw=0.8, posture="stand"))
    assert into.twist == (0.0, 0.0) and "veer" in into.gate  # turning into the wall: no
    away = link.tick(state(t + 0.22), BodyRequest(vx=0.5, vyaw=-0.8, posture="stand"))
    assert away.twist == (0.5, -0.8)


def test_a_push_into_a_wall_is_not_taken_for_a_pickup() -> None:
    """Live 2026-09-27: blocked at 39.1 s → a false 'lifted' at 39.9 s (pressing into a wall
    unloads the legs for a moment the way a pickup does). Right after the body pushed into
    something, a lift claim must hold 1.5 s before it is believed; a claim the floor ends
    before that is dropped (no 'lifted', no sit-down)."""
    link = BodyLink()
    run(link, BodyRequest(vx=0.5, posture="stand"), 0.0, 1.0)
    touch = P.DepthView(0, 60, None, None, 0, True, 0)  # the nose against the wall
    link.tick(state(1.0), BodyRequest(vx=0.5, posture="stand"), depth=touch)
    light = run(link, BodyRequest(vx=0.5, posture="stand"), 1.02, 0.8, load=8.0)
    back = run(link, BodyRequest(vx=0.5, posture="stand"), 1.82, 1.2, load=45.0)
    kinds = [e["kind"] for o in light + back for e in o.events]
    assert "proprio.lifted" not in kinds and "proprio.set_down" not in kinds
    assert not any(o.prox.view.lifted for o in light + back)
    assert link.lifts_dropped == 1
    assert not [i for o in light + back for i in o.intents if i.method == "robot.do"]  # no sit


def test_a_real_pickup_after_a_push_is_believed_once_it_lasts() -> None:
    link = BodyLink()
    run(link, BodyRequest(vx=0.5, posture="stand"), 0.0, 1.0)
    touch = P.DepthView(0, 60, None, None, 0, True, 0)
    link.tick(state(1.0), BodyRequest(vx=0.5, posture="stand"), depth=touch)
    held = run(link, BodyRequest(vx=0.5, posture="stand"), 1.02, 3.0, load=8.0)
    told = [k for k, o in enumerate(held) for e in o.events if e["kind"] == "proprio.lifted"]
    assert len(told) == 1 and told[0] * DT >= 1.5  # believed, once, after it lasted
    assert held[-1].prox.view.lifted and "lifted" in held[-1].gate
    # far from any push, a pickup is believed at once (owner B4, as before)
    calm = BodyLink()
    run(calm, BodyRequest(posture="stand"), 0.0, 5.0)
    outs = run(calm, BodyRequest(posture="stand"), 5.0, 1.0, load=8.0)
    first = next(k for k, o in enumerate(outs) for e in o.events if e["kind"] == "proprio.lifted")
    assert first * DT < 0.9


def test_a_stumble_stops_the_legs_only_when_the_trunk_really_tips() -> None:
    """Live 2026-09-28: proprio's 'stumbled' (a gyro spike with doubled leg load) fired 107
    times in 35 min for the slow, shy duck (max tilt 7°) and each stopped its walk: 0.9-s
    spurts. A trip tips the trunk; a footfall's jolt does not."""
    walk = BodyRequest(vx=0.4, vyaw=0.0, posture="stand")
    jolt = BodyLink()
    run(jolt, walk, 0.0, 2.0)  # walking, the leg load's baseline learned
    out = run(jolt, walk, 2.0, 0.06, gyro=(3.5, 0.0, 0.0), load=120.0)  # a jolt, trunk level
    after = run(jolt, walk, 2.06, 0.5)
    assert any("stumbled" in o.prox.edges for o in out), [o.prox.edges for o in out]
    assert all(o.twist == (0.4, 0.0) for o in after), [o.gate for o in after]
    assert jolt.stumbles_ignored >= 1
    trip = BodyLink()
    run(trip, walk, 0.0, 2.0)
    out = run(trip, walk, 2.0, 0.06, gyro=(3.5, 0.0, 0.0), load=120.0, tilt=14.0)
    after = run(trip, walk, 2.06, 0.3, tilt=12.0)
    assert any(o.twist == (0.0, 0.0) and "stumble" in o.gate for o in out + after)


def test_a_hop_back_does_not_read_its_own_lean_as_a_trip() -> None:
    """Startle-hop 2026-10-10: the back-steps lean the trunk 10-13 degrees with a gyro jolt by design. Read as a trip, the stumble
    reflex armed `veer` (arcs only, up to 5 s) and the flight that followed was held back."""
    walk = BodyRequest(vx=0.4, vyaw=0.0, posture="stand")
    for policy, trips in (("walk", True), ("hop_back", False), ("hop_back:unwind", False)):
        link = BodyLink()
        run(link, walk, 0.0, 2.0)  # the leg load's baseline learned
        out = run(link, walk, 2.0, 0.06, policy=policy, gyro=(3.5, 0.0, 0.0), load=120.0, tilt=14.0)
        after = run(link, walk, 2.06, 0.3, policy=policy, tilt=12.0)
        tripped = any("stumble" in o.gate for o in out + after)
        assert tripped is trips, (policy, [o.gate for o in out + after][:5])
        assert (link.veer_until_ns > 0) is trips
