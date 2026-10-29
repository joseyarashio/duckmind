"""The way out of clutter (neuro/clutter.py): noticing that it goes round in one place, picking one way out, keeping to it."""

from __future__ import annotations

import math

from duckmind.neuro import clutter as CL
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.clutter import Clutter, ClutterConfig
from duckmind.neuro.contract import wrap_angle
from duckmind.neuro.genome import preset

ROUTE = ClutterConfig(
    action="route"
)  # (the route out that takes the legs; the default answer is "loosen")


class FakeBound:
    """A wall memory with a wall of points: clear everywhere except where ``closed`` says so (headings in the odometry frame)."""

    def __init__(self, closed: list[tuple[float, float]] | None = None, trail=None) -> None:
        self.closed = closed or []  # (heading centre, half width) closed within 0.2 m
        self.trail = trail or []
        self.ahead_v = 1.5
        self.behind_v = 1.5

    def ahead(self, odom):
        return self.ahead_v

    def _is_closed(self, h_abs: float) -> bool:
        return any(abs(wrap_angle(h_abs - c)) <= w for c, w in self.closed)

    def clearances(self, odom, headings):
        return [0.1 if self._is_closed(wrap_angle(odom[2] + h)) else 1.5 for h in headings]

    def clearance(self, odom, h):
        if abs(abs(wrap_angle(h)) - math.pi) < 0.2:
            return self.behind_v
        return self.clearances(odom, [h])[0]

    def _known(self, odom):
        return [], []

    def snapshot(self, centre, radius=3.4):
        """Points on the closed headings (a wall 0.2 m out along each) and the trail as known ground."""
        pts = []
        for c, w in self.closed:
            for k in range(-int(w / 0.1), int(w / 0.1) + 1):
                a = c + 0.1 * k
                pts.append((centre[0] + 0.25 * math.cos(a), centre[1] + 0.25 * math.sin(a)))
        return pts, list(self.trail)

    def known_free(self, odom, h, known=None):
        return 1.0


def knot(c: Clutter, t0: int, x: float = 0.0, y: float = 0.0, n: int = 4) -> int:
    t = t0
    for _ in range(n):
        c.note(t, x, y, "bump")
        t += 2000
    return t


def drive(c: Clutter, t: int, odom, bound, ahead_m=None, behind_m=None, **kw):
    if ahead_m is not None:
        bound.ahead_v = ahead_m
    if behind_m is not None:
        bound.behind_v = behind_m
    args = dict(active=True, want=True, bound=bound)
    args.update(kw)
    return c.step(t, odom, **args)


def test_three_contacts_at_one_place_start_an_escape_and_it_keeps_one_heading():
    c = Clutter(ROUTE)
    bound = FakeBound(closed=[(0.0, 0.7)])  # closed ahead (heading 0 +- 40 deg)
    t = knot(c, 0)
    d = drive(c, t, (0.0, 0.0, 0.0), bound, ahead_m=0.1)
    assert c.active and d is not None
    h0 = c.heading
    assert abs(wrap_angle(h0)) > 0.7  # a way that is not the closed one
    x = y = 0.0
    yaw = 0.0
    for k in range(1, 12):  # it walks the heading (turning toward it first) and keeps it
        err = wrap_angle(h0 - yaw)
        yaw += max(-0.1, min(0.1, err))
        x += 0.02 * math.cos(yaw)
        y += 0.02 * math.sin(yaw)
        d = drive(c, t + 100 * k, (x, y, yaw), bound, ahead_m=1.5)
        if d is None:
            break
        assert abs(wrap_angle(c.heading - h0)) < 0.5
    assert c.started == 1


def test_contacts_far_apart_are_no_knot():
    c = Clutter(ROUTE)
    c.note(0, 0.0, 0.0, "bump")
    c.note(5000, 2.0, 0.0, "bump")
    c.note(10_000, 4.0, 0.0, "bump")
    assert drive(c, 10_500, (4.0, 0.0, 0.0), FakeBound()) is None
    assert not c.active


def test_contacts_far_apart_in_time_are_no_knot():
    c = Clutter(ROUTE)
    for k in range(3):
        c.note(k * 30_000, 0.0, 0.0, "bump")
    assert drive(c, 70_000, (0.0, 0.0, 0.0), FakeBound()) is None


def test_events_closer_than_the_gap_count_once():
    c = Clutter(ROUTE)
    for k in range(10):
        c.note(k * 100, 0.0, 0.0, "bump")
    assert drive(c, 1_000, (0.0, 0.0, 0.0), FakeBound()) is None


def test_confined_while_wanting_to_walk_is_a_knot():
    c = Clutter(ROUTE)
    bound = FakeBound()
    c.note(5_000, 0.0, 0.0, "bump")
    c.note(20_000, 0.1, 0.0, "back")
    got = None
    for k in range(70):
        got = drive(c, 500 * k, (0.1 * math.sin(k), 0.1 * math.cos(k), 0.0), bound)
        if got is not None:
            break
    assert got is not None and c.active


def test_it_goes_the_way_it_came_when_the_rest_is_unknown():
    c = Clutter(ROUTE)
    # the trail it walked in by lies behind it (heading pi), the rest is as clear: the trail is the way out
    trail = [(-0.15 * k, 0.0) for k in range(1, 12)]
    bound = FakeBound(trail=trail)
    t = knot(c, 0, x=-0.05)
    drive(c, t, (-0.05, 0.0, 0.0), bound)
    assert abs(abs(wrap_angle(c.heading)) - math.pi) < 0.5


def test_a_heading_that_closes_fails_and_the_next_best_is_taken():
    c = Clutter(ROUTE)
    bound = FakeBound()
    t = knot(c, 0)
    drive(c, t, (0.0, 0.0, 0.0), bound)
    h0 = c.heading
    # aligned with it, then the way along it closes for longer than block_ms
    bound.closed = [(h0, 0.2)]
    yaw = h0
    for k in range(1, 40):
        drive(c, t + 100 * k, (0.0, 0.0, yaw), bound, ahead_m=0.1)
    assert abs(wrap_angle(c.heading - h0)) > 0.25


def test_no_escape_while_not_active_and_the_knot_is_forgotten_by_a_fall():
    c = Clutter(ROUTE)
    t = knot(c, 0)
    assert drive(c, t, (0.0, 0.0, 0.0), FakeBound(), active=False) is None
    assert not c.active
    assert drive(c, t + 100, (0.0, 0.0, 0.0), FakeBound()) is None or c.active


def test_out_ends_it_and_the_cooldown_after_a_give_up_keeps_it_from_starting_again():
    c = Clutter(ROUTE)
    bound = FakeBound()
    t = knot(c, 0)
    drive(c, t, (0.0, 0.0, 0.0), bound)
    assert c.active
    # it has got 1.2 m away and the way ahead is open: out
    d = drive(c, t + 5000, (1.2, 0.0, 0.0), bound)
    assert d is None and not c.active and c.finished == 1
    # and a time-out is a give-up with a cool-down
    c2 = Clutter(ROUTE)
    t = knot(c2, 0)
    drive(c2, t, (0.0, 0.0, 0.0), bound, ahead_m=0.1)
    d = drive(c2, t + 40_000, (0.0, 0.0, 0.0), bound, ahead_m=0.1)
    assert d is None and c2.gave_up == 1
    t2 = knot(c2, t + 41_000)
    assert drive(c2, t2, (0.0, 0.0, 0.0), bound) is None


def test_disabled_never_drives():
    c = Clutter(ClutterConfig(enabled=False))
    t = knot(c, 0)
    assert drive(c, t, (0.0, 0.0, 0.0), FakeBound()) is None


def test_turn_legs_alternate_forward_and_reverse_arcs_when_the_way_ahead_is_shut():
    c = Clutter(ROUTE)
    bound = FakeBound()
    t = knot(c, 0)
    seen = set()
    for k in range(60):
        d = drive(c, t + 100 * k, (0.0, 0.0, 0.0), bound, ahead_m=0.2, behind_m=1.0)
        if d is not None:
            seen.add(d.phase)
    assert "back" in seen or "go" in seen


def test_loosen_lets_the_goal_go_for_a_while_and_takes_it_back_when_it_is_out():
    c = Clutter(ClutterConfig(action="loosen", loosen_s=14.0, loosen_out_m=1.0, loosen_min_s=3.0))
    bound = FakeBound()
    t = knot(c, 0)
    assert drive(c, t, (0.0, 0.0, 0.0), bound) is None  # it takes no legs
    assert c.loosened == 1 and c.loosening(t + 100, (0.0, 0.0, 0.0))
    assert c.loosening(t + 5000, (0.2, 0.0, 0.0))  # still in the knot: still let go
    assert not c.loosening(t + 5000, (1.3, 0.0, 0.0))  # out: the pull is back
    # a knot again at once is not answered again (the cool-down)
    t2 = knot(c, t + 6000)
    assert drive(c, t2, (0.0, 0.0, 0.0), bound) is None and c.loosened == 1


def test_loosen_ends_by_itself():
    c = Clutter(ClutterConfig(action="loosen", loosen_s=10.0))
    t = knot(c, 0)
    drive(c, t, (0.0, 0.0, 0.0), FakeBound())
    assert c.loosening(t + 9000, (0.0, 0.0, 0.0))
    assert not c.loosening(t + 10_500, (0.0, 0.0, 0.0))


def test_a_hold_while_it_wants_to_walk_counts_as_a_contact():
    c = Clutter(ROUTE)
    bound = FakeBound()
    t = 0
    for _ in range(5):  # the legs held (a ToF stop), released, held again ...
        drive(c, t, (0.0, 0.0, 0.0), bound, held=True)
        drive(c, t + 700, (0.0, 0.0, 0.0), bound, held=False)
        t += 2000
    assert c.active


def test_the_brain_has_it_only_when_asked():
    genome = preset("explorer")
    assert NeuroBrain(genome, 1, clutter=True).clutter.cfg.enabled is True
    assert NeuroBrain(genome, 1, clutter=False).clutter.cfg.enabled is False
    assert (
        NeuroBrain(genome, 1).clutter.cfg.enabled is CL.CLUTTER_ENABLED
    )  # (off unless DUCKMIND_CLUTTER=1 was in the environment)


def test_the_default_answer_is_to_let_the_goal_go_not_to_take_the_legs():
    c = Clutter()
    assert c.cfg.action == "loosen"
    t = knot(c, 0)
    assert (
        drive(c, t, (0.0, 0.0, 0.0), FakeBound()) is None
        and not c.active
        and c.loosening(t + 10, None)
    )


def test_disabled_keeps_nothing():
    c = Clutter(ClutterConfig(enabled=False))
    for k in range(1000):
        c.note(k * 2000, 0.0, 0.0, "bump")
        assert drive(c, k * 2000, (0.0, 0.0, 0.0), FakeBound()) is None
    assert len(c._events) == 0 and not c._hist and not c._wide
