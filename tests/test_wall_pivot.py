"""2026-10-11 the wall pivot (WALL-LOOPS-2026-10-11.md): held at a wall and asked to walk on, a duck with the quick-turn entries turns away on the spot
with the quick turn instead of sweeping a forward arc back into the wall; without the entries (or for an approach, a flight, a docking, a skill) nothing changes."""

from __future__ import annotations

from test_neuro_bodylink import DT, moves, state

from duckmind.duckbody import protocol as P
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G

NAMES = [*G.QUICK_TURN_ENTRIES, G.ABORT]
ARC = BodyRequest(vx=0.37, vyaw=-0.9, posture="stand")  # the brain's turn_around arc, to the right
STRAIGHT = BodyRequest(vx=0.37, vyaw=0.0, posture="stand")


def _link(names=NAMES) -> BodyLink:
    link = BodyLink()
    link.skills = G.SkillGuard(names)
    return link


def _depth(ahead=250, left=None, right=None, touch=False) -> P.DepthView:
    return P.DepthView(0, ahead, left, right, 0, touch, 0)


def _do(out) -> list[str]:
    return [i.params["skill"] for i in out.intents if i.method == "robot.do"]


def _settle(link: BodyLink, ticks: int = 120) -> float:
    t = 0.0
    for _ in range(ticks):
        t += DT
        link.tick(state(t), BodyRequest(posture="stand"))
    return t


def _pivot_run(
    link: BodyLink, t: float, req: BodyRequest, depth_of, rate: float = 2.2, ticks: int = 120
):
    """Tick the link as robotd would run it: the label follows the ask, the heading turns at ``rate`` rad/s while the quick turn runs."""
    yaw = 0.0
    label = "walk"
    outs = []
    for _ in range(ticks):
        t += DT
        out = link.tick(state(t, policy=label, yaw=yaw), req, depth=depth_of(yaw))
        outs.append(out)
        ask = _do(out)
        if ask and ask[0].startswith("quick_turn"):
            label = ask[0]
        elif ask and ask[0] == G.ABORT:
            label = "walk"
        if label.startswith("quick_turn"):
            yaw += rate * DT * (1.0 if label.endswith("left") else -1.0)
    return outs, yaw, t


def test_a_duck_held_at_a_wall_turns_away_on_the_spot_and_hands_back() -> None:
    link = _link()
    t = _settle(link)
    # the wall is straight ahead until the heading has turned 0.8 rad, then clear
    outs, yaw, _ = _pivot_run(
        link, t, ARC, lambda y: _depth(250, touch=True) if abs(y) < 0.8 else _depth(None)
    )
    asks = [a for o in outs for a in _do(o)]
    assert asks[0] == "quick_turn_right"  # the side of the brain's arc
    assert G.ABORT in asks  # handed back to v5 once it faces away
    assert 1.0 <= abs(yaw) <= 1.5  # at least the minimum turn, well short of the maximum
    assert link.wall_pivots == 1 and link.wall_pivots_refused == 0
    during = [o for o in outs if _do(o) and _do(o)[0].startswith("quick_turn")]
    assert during and all(
        moves(o.intents) in ([], [(0.0, 0.0)]) for o in during
    )  # no arc into the wall meanwhile


def test_the_side_is_away_from_what_the_tof_sees_close_on_the_arcs_side() -> None:
    link = _link()
    t = _settle(link)
    # the brain's arc goes right but the ToF sees the right shut (200 mm) and the left open
    outs, _, _ = _pivot_run(
        link, t, ARC, lambda y: _depth(250, left=900, right=200, touch=True), ticks=10
    )
    assert [a for o in outs for a in _do(o)][0] == "quick_turn_left"


def test_the_budget_ends_it_where_the_pivots_do_not_get_it_out() -> None:
    link = _link()
    t = _settle(link)
    wall = lambda y: _depth(250, touch=True)  # noqa: E731  (the wall never clears: the turn runs to its limits)
    for _ in range(6):
        _, _, t = _pivot_run(link, t, ARC, wall, ticks=200)
    assert link.wall_pivots == link.cfg.wall_pivot_budget


def test_nothing_changes_without_the_quick_turn_entries() -> None:
    link = _link(["skill_abort", "startle_2"])
    t = _settle(link)
    outs, _, _ = _pivot_run(link, t, ARC, lambda y: _depth(250))
    assert all(not _do(o) for o in outs) and link.wall_pivots == 0
    assert any(moves(o.intents) for o in outs)


def test_a_flight_and_a_skill_are_never_pivots() -> None:
    for req in (
        BodyRequest(vx=0.5, vyaw=-0.9, panic=True, posture="stand"),
        BodyRequest(vx=0.37, vyaw=-0.9, skill="startle_2", posture="stand"),
    ):
        link = _link()
        t = _settle(link)
        _pivot_run(link, t, req, lambda y: _depth(250, touch=True), ticks=300)
        assert link.wall_pivots == 0, req


def test_a_walk_toward_a_goal_pivots_only_after_two_seconds_held_at_the_wall() -> None:
    for req in (
        BodyRequest(vx=0.37, vyaw=-0.9, approach=True, posture="stand"),
        BodyRequest(vx=0.35, vyaw=-0.9, docking=True, posture="stand"),
        BodyRequest(vx=0.37, vyaw=-0.9, toward=(0, 250), posture="stand"),
    ):
        link = _link()
        t = _settle(link)
        _pivot_run(
            link, t, req, lambda y: _depth(250, touch=True), ticks=80
        )  # 1.6 s: held, not yet
        assert link.wall_pivots == 0, req
        link = _link()
        t = _settle(link)
        outs, _, _ = _pivot_run(
            link, t, req, lambda y: _depth(250, touch=True), ticks=200
        )  # 4 s: it goes
        assert link.wall_pivots >= 1 and link.wall_pivots_refused == 0, req
        assert any(a in G.QUICK_TURN_ENTRIES for o in outs for a in _do(o)), (
            req
        )  # the skill guard let it run (also on a docking leg)


def test_no_wall_no_pivot_and_a_walk_with_the_way_clear_is_untouched() -> None:
    link = _link()
    t = _settle(link)
    outs, _, _ = _pivot_run(link, t, ARC, lambda y: _depth(1500), ticks=60)
    assert link.wall_pivots == 0 and all(not _do(o) for o in outs)
    assert any(moves(o.intents) and moves(o.intents)[-1][0] > 0.0 for o in outs)  # it walks the arc


def test_a_pivot_the_skill_guard_refuses_leaves_the_old_stop_and_veer_alone() -> None:
    link = _link()
    t = _settle(link)
    # a trunk tipped 25 degrees: the skill guard says no ("tilt"); the link falls back and waits 2 s before asking again
    out = None
    for _ in range(10):
        t += DT
        out = link.tick(state(t, tilt=25.0), ARC, depth=_depth(250, touch=True))
    assert out is not None and link.wall_pivots_refused >= 1 and not _do(out)
    assert link.wp is None


def test_a_wall_that_clears_ends_it_early_and_a_lift_ends_it_at_once() -> None:
    link = _link()
    t = _settle(link)
    # the wall is gone after 0.1 rad: it still turns the minimum, not more
    outs, yaw, t = _pivot_run(
        link, t, ARC, lambda y: _depth(250, touch=True) if abs(y) < 0.1 else _depth(None), ticks=80
    )
    assert link.cfg.wall_pivot_min_rad - 0.1 <= abs(yaw) <= link.cfg.wall_pivot_min_rad + 0.4
    link = _link()
    t = _settle(link)
    yaw = 0.0
    for k in range(60):
        t += DT
        label = "quick_turn_left" if k > 3 else "walk"
        out = link.tick(state(t, policy=label, yaw=yaw), ARC, depth=_depth(250, touch=True))
    assert link.wp is not None
    t += DT
    out = link.tick(
        state(t, policy="quick_turn_left", tilt=70.0, fallen=True),
        ARC,
        depth=_depth(250, touch=True),
    )
    assert link.wp is None and G.ABORT in _do(out)


def test_at_the_tof_stop_a_straight_walk_pivots_but_an_arc_is_let_through() -> None:
    link = _link()
    t = _settle(link)
    outs, _, _ = _pivot_run(
        link, t, ARC, lambda y: _depth(250), ticks=40
    )  # 300 mm ahead, no touch: the arc walks away
    assert link.wall_pivots == 0 and any(moves(o.intents) for o in outs)
    link = _link()
    t = _settle(link)
    outs, _, _ = _pivot_run(
        link, t, STRAIGHT, lambda y: _depth(250) if abs(y) < 0.8 else _depth(None)
    )
    assert link.wall_pivots == 1 and [a for o in outs for a in _do(o)][0] in G.QUICK_TURN_ENTRIES
