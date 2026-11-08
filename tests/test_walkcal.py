"""The loaded walk policy's map (duckmind.neuro.walkcal, 2026-11-05): the brain was tuned on walk policy v5, which turns
about half of what it is told; v8 turns 1.1-1.3 times it. The body link sends the loaded walk the command that answers as
v5 answered the brain's twist (measured tables keyed by the walk file's sha256); v5 itself gets exactly today's commands."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import walkcal
from duckmind.neuro.bodylink import DEFAULT, BodyLink, BodyRequest
from duckmind.simweb import neuro_live

TABLE = walkcal.load()
V5 = TABLE.walks[TABLE.reference]
V8 = next(w for w in TABLE.walks.values() if w.name.startswith("v8"))
DT = 0.02

# what the brain asks for while walking: cruise, steering, arcs that turn away, the tightest arcs
REQUESTS = [
    (vx, vyaw)
    for vx in (0.35, 0.40, 0.45, 0.50, 0.55)
    for vyaw in (-1.0, -0.9, -0.5, -0.28, -0.1, 0.0, 0.1, 0.28, 0.5, 0.9, 1.0)
]


def _state(t: float, applied: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> RobotState:
    joints = [0.0] * 15
    for i, home in enumerate((0.3491, 0.3491, 0.0, 0.0)):
        joints[5 + i] = home
    currents = [0.0] * 15
    for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        currents[i] = 4.5
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": "walk",
            "move": {"requested": list(applied), "applied": list(applied)},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "currents_ma": currents,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
    )


def _sent(link: BodyLink, vx: float, vyaw: float, *, t0: float = 0.0) -> tuple[float, float]:
    """The twist the link streams for a steady walk request (after it settles)."""
    twist = None
    for k in range(10):
        out = link.tick(_state(t0 + k * DT), BodyRequest(vx=vx, vyaw=vyaw, posture="stand"))
        moves = [i.twist for i in out.intents if i.method == "robot.move"]
        twist = moves[-1] if moves else twist
    assert twist is not None
    return twist


def _link(walk: walkcal.WalkMap | None) -> BodyLink:
    link = BodyLink(replace(DEFAULT, command_delay_ms=20.0))
    link.walk = walk
    return link


def _v8_map() -> walkcal.WalkMap:
    walk = TABLE.map_for(V8.walk_sha256, V8.body, 20.0)
    assert walk is not None
    return walk


def test_the_table_is_measured_for_both_walks_on_their_own_bodies() -> None:
    assert V5.body == "allcollisions" and V8.body == "allcollisions_backlash"
    assert V5.walk_sha256 != V8.walk_sha256
    # what the tables say about the walks (the reason for the map): v5 turns about half of the command, v8 more than it
    assert V5.at(0.40, 1.0)[1] < 0.75 and V5.at(0.40, -1.0)[1] > -1.0
    assert V8.at(0.40, 1.0)[1] > 1.0 and V8.at(0.40, -1.0)[1] < -1.0


def test_v5_gets_exactly_the_commands_it_always_did(monkeypatch: pytest.MonkeyPatch) -> None:
    # the driver finds no map for the reference walk, on any body
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V5.walk_sha256)
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision="allcollisions")
    assert neuro_live._walk_map(body, 20.0) is None
    assert neuro_live._walk_map(replace_ns(body, collision="allcollisions_backlash"), 20.0) is None
    assert neuro_live._walk_map(None, 20.0) is None
    # … so a v5 link sends what a link without the map sends, and the brain reads the state as robotd sent it
    today, v5 = _link(None), _link(neuro_live._walk_map(body, 20.0))
    for k, (vx, vyaw) in enumerate(REQUESTS):
        assert _sent(v5, vx, vyaw, t0=k) == _sent(today, vx, vyaw, t0=k)
    state = _state(0.0, (0.31, 0.0, 0.42))
    assert v5.planned_state(state) is state


def replace_ns(ns: SimpleNamespace, **kw: object) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(ns), **kw})


def test_the_map_of_a_walk_onto_itself_changes_nothing() -> None:
    same = walkcal.WalkMap(V5, V5)
    for vx, vyaw in REQUESTS:
        assert same.command(vx, vyaw) == pytest.approx((vx, vyaw), abs=1e-3)


def test_a_v8_link_sends_the_turn_the_brain_wants_not_double() -> None:
    today, v8 = _link(None), _link(_v8_map())
    for k, (vx, vyaw) in enumerate(REQUESTS):
        designed = _sent(
            today, vx, vyaw, t0=k
        )  # what v5 is sent today (with the servo delay's compensation)
        sent = _sent(v8, vx, vyaw, t0=k)
        want_speed, want_rate = V5.at(*designed)  # what the brain's twist did on v5
        speed, rate = V8.at(*sent)  # what the sent command does on v8
        assert rate == pytest.approx(want_rate, abs=0.03), (vx, vyaw, designed, sent)
        if 0.30 < sent[0] < 0.55:  # (inside v8's reach the path speed matches too)
            assert speed == pytest.approx(want_speed, abs=0.006), (vx, vyaw, designed, sent)
        # proprio's own-turn model reads the twist the brain meant, not the one v8 was sent
        assert v8.walk_sent == (sent, designed)
    # the failure it fixes: today's command on v8 turns about twice as hard as on v5
    designed = _sent(today, 0.45, 0.5)
    assert V8.at(*designed)[1] > 1.8 * V5.at(*designed)[1]
    assert abs(V8.at(*_sent(v8, 0.45, 0.5, t0=99.0))[1]) < 1.15 * abs(V5.at(*designed)[1])


def test_the_brain_reads_the_body_twist_in_the_units_it_plans_in() -> None:
    walk = _v8_map()
    link = _link(walk)
    for vx, vyaw in REQUESTS:
        sent = walk.command(vx, vyaw)
        if not 0.30 < sent[0] < 0.55:
            continue  # (beyond v8's reach — v5 walks faster at the top — the body did less than the brain meant)
        state = _state(0.0, (sent[0], 0.0, sent[1]))
        seen = link.planned_state(state)
        assert (seen.applied[0], seen.applied[2]) == pytest.approx((vx, vyaw), abs=0.03)
        assert (seen.requested[0], seen.requested[2]) == pytest.approx((vx, vyaw), abs=0.03)
        assert seen.odom_yaw == state.odom_yaw and seen.policy == state.policy


def test_commands_outside_the_table_go_out_as_they_are() -> None:
    walk = _v8_map()
    for twist in ((0.0, 0.0), (0.2, 0.5), (-0.4, 1.0), (0.6, 0.3)):
        assert walk.command(*twist) == twist
    v8 = _link(walk)
    # a docking reverse and a pause are not mapped
    out = v8.tick(_state(0.0), BodyRequest(vx=-0.4, vyaw=0.3, posture="stand", docking=True))
    assert [i.twist for i in out.intents if i.method == "robot.move"] in ([], [(-0.4, 0.3)])


def test_the_driver_maps_only_a_measured_walk_on_its_own_body_and_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision=V8.body)
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V8.walk_sha256)
    walk = neuro_live._walk_map(body, 20.0)
    assert walk is not None and walk.loaded == V8 and walk.reference == V5
    assert neuro_live._walk_map(body, 0.0) is None  # (measured with the simulator's 20 ms delay)
    assert neuro_live._walk_map(replace_ns(body, collision="allcollisions"), 20.0) is None
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: "0" * 64)
    assert neuro_live._walk_map(body, 20.0) is None  # an unmeasured walk: as it is


def test_the_table_file_is_well_formed() -> None:
    for w in TABLE.walks.values():
        assert len(w.speed) == len(w.vxs) and all(len(r) == len(w.vyaws) for r in w.speed)
        assert all(math.isfinite(v) and v >= 0.0 for r in w.speed for v in r)
        assert all(math.isfinite(v) for r in w.yaw for v in r)
    for w in (V5, V8):
        b = w.back_arc
        assert b is not None and b.vxs == (-0.4, -0.3)
        assert all(math.isfinite(v) and v >= 0.0 for r in b.speed for v in r)
        assert all(math.isfinite(v) for r in b.yaw for v in r)


# curiosity's back arc (and a reflex's escape) as the brain asks for it: BACK_ARC_VYAW at REVERSE_VX for 1 s
BACK_ARCS = [(-0.40, 1.0), (-0.40, -1.0)]


def _back_state(
    t: float, x: float, y: float, yaw: float, applied: tuple[float, float]
) -> RobotState:
    state = _state(t, (applied[0], 0.0, applied[1]))
    return replace(state, odom_pos=(x, y, 0.0), odom_yaw=yaw)


def _back_arc(
    link: BodyLink, body: walkcal.WalkResponse, vyaw: float, seconds: float = 1.4
) -> tuple[list, list, float]:
    """``link`` backs an arc on a body that answers each streamed twist as ``body``'s measured 1 s back arc does:
    the twists sent, proprio's edges, the heading turned."""
    back = body.back_arc
    assert back is not None
    x = y = yaw = 0.0
    sent: tuple[float, float] = (0.0, 0.0)
    twists, edges = [], []
    for k in range(int(seconds / DT)):
        out = link.tick(
            _back_state(k * DT, x, y, yaw, sent),
            BodyRequest(vx=-0.40, vyaw=vyaw, posture="stand", back_arc=True),
        )
        moves = [i.twist for i in out.intents if i.method == "robot.move"]
        if moves:
            sent = moves[-1]
            twists.append(sent)
        edges += list(out.prox.edges)
        if sent[0] < 0.0:
            speed, rate = back.at(*sent)
            yaw += rate * DT
            x -= speed * DT * math.cos(yaw)
            y -= speed * DT * math.sin(yaw)
    return twists, edges, yaw


def test_v5_backs_its_arcs_exactly_as_it_always_did(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V5.walk_sha256)
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision="allcollisions")
    for vx, vyaw in BACK_ARCS:
        v5 = _link(neuro_live._walk_map(body, 20.0))
        twists, edges, _ = _back_arc(v5, V5, vyaw)
        assert set(twists) == {(vx, vyaw)}
        assert "brushed" not in edges and "blocked" not in edges


def test_a_v8_back_arc_turns_as_v5s_and_is_not_read_as_a_graze() -> None:
    for vx, vyaw in BACK_ARCS:
        v8 = _link(_v8_map())
        twists, edges, turned = _back_arc(v8, V8, vyaw)
        assert twists and all(t[0] < 0.0 for t in twists)
        sent = twists[-1]
        # live wall check, 2026-11-06: sent as it is, v8's back arc turned ≈ 57° in its first second (v5 ≈ 26°) and
        # backed off fast enough to count as progress — proprio's 'brushed' at the end of 15 of 25 arcs, at no wall
        assert "brushed" not in edges, (vyaw, sent)
        assert V8.back_arc.at(*sent)[1] == pytest.approx(V5.back_arc.at(vx, vyaw)[1], abs=0.03), (
            vyaw,
            sent,
        )
        assert v8.walk_sent == (
            sent,
            (vx, vyaw),
        )  # proprio's own-turn model reads the arc the brain asked for
        _, _, v5_turned = _back_arc(_link(None), V5, vyaw)
        assert turned == pytest.approx(v5_turned, abs=math.radians(4.0))


def test_the_brain_reads_a_v8_back_arc_as_the_arc_it_asked_for_and_a_docking_reverse_as_it_is() -> (
    None
):
    for vx, vyaw in BACK_ARCS:
        link = _link(_v8_map())
        sent, _, _ = _back_arc(link, V8, vyaw, seconds=0.3)
        seen = link.planned_state(_state(0.0, (sent[-1][0], 0.0, sent[-1][1])))
        assert (seen.applied[0], seen.applied[2]) == pytest.approx((vx, vyaw), abs=0.03)
    link.tick(_state(9.0), BodyRequest(vx=-0.4, vyaw=0.3, posture="stand", docking=True))
    state = _state(9.02, (-0.4, 0.0, 0.3))
    assert link.planned_state(state).applied == state.applied


def test_a_walk_that_turns_on_the_spot_pivots_by_itself() -> None:
    from duckmind.duckbody import protocol as P
    from duckmind.duckbody.reflex import GuardContext, Limits, check_intent

    assert V8.pivots and not V5.pivots  # (measured standing: v8 1.2 rad/s at 1.0 asked, v5 none)
    v8, today = _link(_v8_map()), _link(None)
    for side in (1.0, -1.0):
        ask = BodyRequest(vx=0.37, vyaw=0.9 * side, pivot=side, posture="stand")
        outs = [v8.tick(_state(10.0 + k * DT), ask) for k in range(10)]
        assert all(not o.refused for o in outs)
        assert [i.twist for i in outs[-1].intents if i.method == "robot.move"] == [(0.0, side)]
        assert v8.pivot_in_place
        # v5 does not turn standing: its pivot stays an arc (robotd's quick-turn skill when installed)
        sent = _sent(today, 0.37, 0.9 * side, t0=20.0)
        assert sent[0] > 0.3 and not today.pivot_in_place
    # the guard lets a turn in place through only for such a walk
    ctx = GuardContext(head_home_ms=1000, moving=True)
    turn = P.Intent("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 1.0})
    assert check_intent(turn, ctx, Limits()) is not None
    assert check_intent(turn, ctx, Limits(turn_in_place=True)) is None


# the brain re-steers every few hundred ms while it walks (go-there, avoidance, the drift trim)
STEERING = (-0.48, 0.30, -0.20, 0.50, -0.60, 0.10)
CONTACTS = ("brushed", "blocked")


def _driven_link(walk: walkcal.WalkResponse, monkeypatch: pytest.MonkeyPatch) -> BodyLink:
    """A link as the driver sets it up for a body running ``walk`` on its own body with the simulator's delay."""
    body = SimpleNamespace(paths=SimpleNamespace(policies="policies"), collision=walk.body)
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: walk.walk_sha256)
    link = _link(neuro_live._walk_map(body, 20.0))
    link.use_proprio(neuro_live._walk_proprio(body, link.proprio.cfg, 20.0))
    return link


def _steered(
    link: BodyLink, walk: walkcal.WalkResponse, graze_at: float | None = None, seconds: float = 8.0
) -> list[tuple[float, str, str]]:
    """``link`` walks at 0.40 while the brain re-steers every 0.6 s, on a body that answers each streamed twist as
    ``walk``'s table says, as late as that walk was measured to (its turn_lag). ``graze_at``: a wall on its right pushes
    it 25° further left over 0.3 s from then. proprio's contact edges (t, edge, side)."""
    assert walk.turn_lag is not None
    dead, lag = walk.turn_lag
    x = y = yaw = rate = 0.0
    sent: tuple[float, float] = (0.0, 0.0)
    streamed: list[tuple[float, tuple[float, float]]] = []
    edges = []
    for k in range(int(seconds / DT)):
        t = k * DT
        rocking = (0.6, 0.4, 0.3)  # (a trunk that steps rocks)
        out = link.tick(
            replace(_back_state(t, x, y, yaw, sent), gyro=rocking),
            BodyRequest(vx=0.40, vyaw=STEERING[int(t / 0.6) % len(STEERING)], posture="stand"),
        )
        moves = [i.twist for i in out.intents if i.method == "robot.move"]
        if moves:
            sent = moves[-1]
        streamed.append((t, sent))
        edges += [(round(t, 2), e, out.prox.contact_side) for e in out.prox.edges if e in CONTACTS]
        due = [s for at, s in streamed if at <= t - dead]
        speed, target = walk.at(*due[-1]) if due and due[-1][0] > 0.0 else (0.0, 0.0)
        rate += (1.0 - math.exp(-DT / lag)) * (target - rate)
        yaw += rate * DT
        if graze_at is not None and graze_at <= t < graze_at + 0.3:
            yaw += math.radians(25.0) / 0.3 * DT
        x += speed * DT * math.cos(yaw)
        y += speed * DT * math.sin(yaw)
    return edges


@pytest.mark.parametrize("name", ["v5", "v8", "v9"])
def test_a_re_steered_walk_on_either_walk_is_not_read_as_a_graze_but_a_graze_is(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (v9's turn lag fitted on its own roam ticks, 2 × 30 min: 0.20 + 0.05 s, v5's)
    walk = {"v5": V5, "v8": V8, "v9": TABLE.walks[V9_WALK]}[name]
    assert walk.turn_lag is not None
    # the latest command times the window: 'brushed' after the big turn-command changes, at no wall (live, both walks:
    # 12-14 forward brushes per walking minute)
    assert _steered(_driven_link(walk, monkeypatch), walk) == []
    for graze_at in (3.1, 3.5, 4.3):
        hits = _steered(_driven_link(walk, monkeypatch), walk, graze_at)
        assert [e[1:] for e in hits] == [("brushed", "right")], (graze_at, hits)
        assert graze_at < hits[0][0] <= graze_at + 1.0


# policy set v9 (microduck #550, 2026-11-06): v7's velstand byte for byte, on the backlash body it declares
V9_WALK = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"


def _v9_map(monkeypatch: pytest.MonkeyPatch) -> walkcal.WalkMap:
    """The map the driver gives a body whose policy folder holds v9 (found by the walk file's sha256)."""
    body = SimpleNamespace(
        paths=SimpleNamespace(policies="policies"), collision="allcollisions_backlash"
    )
    monkeypatch.setattr(walkcal, "walk_sha256", lambda _p: V9_WALK)
    walk = neuro_live._walk_map(body, 20.0)
    assert walk is not None, "v9's walk is measured: the driver maps it"
    return walk


def test_a_v9_link_sends_the_turn_the_brain_wants(monkeypatch: pytest.MonkeyPatch) -> None:
    walk = _v9_map(monkeypatch)
    v9w = walk.loaded
    assert (
        v9w.walk_sha256 == V9_WALK and v9w.body == "allcollisions_backlash" and walk.reference == V5
    )
    today, v9 = _link(None), _link(walk)
    for k, (vx, vyaw) in enumerate(REQUESTS):
        designed = _sent(today, vx, vyaw, t0=k)
        sent = _sent(v9, vx, vyaw, t0=k)
        want_speed, want_rate = V5.at(*designed)
        speed, rate = v9w.at(*sent)
        lo, hi = v9w.at(sent[0], -1.0)[1], v9w.at(sent[0], 1.0)[1]
        if lo + 0.02 < want_rate < hi - 0.02:  # v9 can turn as v5 did: it does
            assert rate == pytest.approx(want_rate, abs=0.03), (vx, vyaw, designed, sent)
        else:  # (v9 turns right at most about as hard as v5: then it turns as hard as it can at that speed)
            assert rate == pytest.approx(lo if want_rate < 0 else hi, abs=0.03), (
                vx,
                vyaw,
                designed,
                sent,
            )
        if 0.30 < sent[0] < 0.55 and lo + 0.02 < want_rate < hi - 0.02:
            assert speed == pytest.approx(want_speed, abs=0.006), (vx, vyaw, designed, sent)
        assert v9.walk_sent == (sent, designed)
    # what it fixes: today's command on v9 turns left a third harder than on v5 (measured 0.56 vs 0.42 rad/s at (0.45, 1.0))
    designed = _sent(today, 0.45, 1.0)
    assert v9w.at(*designed)[1] > 1.25 * V5.at(*designed)[1]
    assert v9w.at(*_sent(v9, 0.45, 1.0, t0=99.0))[1] == pytest.approx(V5.at(*designed)[1], abs=0.03)


def test_v9_does_not_turn_on_the_spot_so_its_pivots_stay_arcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walk = _v9_map(monkeypatch)
    # measured standing at (0, ±1.0): v9 0.66-0.69 rad/s (v8 1.19-1.20, v5 0.00), under PIVOT_SHARE_MIN
    assert not walk.loaded.pivots
    link = _link(walk)
    for side in (1.0, -1.0):
        sent = _sent(link, 0.37, 0.9 * side, t0=30.0 + side)
        assert sent[0] > 0.25 and not link.pivot_in_place


def test_a_v9_back_arc_turns_as_v5s(monkeypatch: pytest.MonkeyPatch) -> None:
    walk = _v9_map(monkeypatch)
    v9w = walk.loaded
    assert v9w.back_arc is not None and v9w.back_arc.vxs == (-0.4, -0.3)
    for vx, vyaw in BACK_ARCS:
        link = _link(walk)
        twists, edges, turned = _back_arc(link, v9w, vyaw)
        sent = twists[-1]
        assert "brushed" not in edges and "blocked" not in edges, (vyaw, sent)
        assert link.walk_sent == (sent, (vx, vyaw))
        _, _, v5_turned = _back_arc(_link(None), V5, vyaw)
        if vyaw > 0:  # v9 backs a left arc as v5 does
            assert turned == pytest.approx(v5_turned, abs=math.radians(4.0)), (vyaw, sent)
        else:  # v9 backs a right arc at most 28° in the first second (v5 38°): it is asked its hardest
            assert sent == (-0.4, -1.0) and v5_turned < turned < 0.0
