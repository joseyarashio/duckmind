"""a:social — 睡前轉一圈 (the loop, its size, its side), the acts' programs, and the registry."""

from __future__ import annotations

import math

import pytest

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import repertoire as P
from duckmind.neuro import select as S
from duckmind.neuro import skill_circle as K
from duckmind.neuro import skill_social_brain as SB
from duckmind.neuro.contract import MotorCommand, SensorFrame, Target
from duckmind.neuro.genome import preset
from duckmind.world import neuro_social_synth as W


def test_the_priors_are_the_bodys_measured_arcs() -> None:
    arcs = K.ArcModel()
    # a:social's physics run: vx 0.35 — right r 0.236 m, left 0.375 m
    assert arcs.radius(-1) == pytest.approx(0.236, abs=0.02)
    assert arcs.radius(1) == pytest.approx(0.375, abs=0.03)
    assert arcs.loop_s(-1) < arcs.loop_s(1)


def test_the_loops_footprint() -> None:
    need = K.loop_needs(-1, 0.24)
    assert need[-math.pi / 2] == pytest.approx(0.48 + K.ROOM_MARGIN_M)
    assert need[0.0] == pytest.approx(0.24 + K.ROOM_MARGIN_M)
    assert need[-math.pi / 4] == pytest.approx(2 * 0.24 * math.sin(math.pi / 4) + K.ROOM_MARGIN_M)


def test_the_arc_model_learns_its_own_body() -> None:
    arcs = K.ArcModel()
    x = y = yaw = 0.0
    rate, speed, t = -0.6, 0.15, 0.0
    for _ in range(3000):  # 60 s of full-rate right arcs at 50 Hz
        arcs.observe(t, (x, y, yaw), 0.35, -1.0, rate, True)
        yaw += rate * 0.02
        x += speed * math.cos(yaw) * 0.02
        y += speed * math.sin(yaw) * 0.02
        t += 0.02
    assert arcs.rate[-1] == pytest.approx(0.6, abs=0.05)
    assert arcs.speed[-1] == pytest.approx(0.15, abs=0.02)
    assert arcs.rate[1] == pytest.approx(math.radians(26.3))  # the other side untouched


def _walk_loop(run: K.CircleRun, rate: float, dt: float = 0.05, limit: float = 60.0) -> float:
    t, yaw = 0.0, 0.0
    run.start(t, (0.0, 0.0, yaw), K.ArcModel())
    while run.phase == "turning" and t < limit:
        yaw = math.atan2(math.sin(yaw + rate * dt), math.cos(yaw + rate * dt))
        t += dt
        run.step(t, (0.0, 0.0, yaw))
    return t


def test_the_loop_is_closed_on_its_own_heading() -> None:
    run = K.CircleRun(innate_side=-1)
    t = _walk_loop(run, rate=-0.7)
    assert run.phase == "done"
    assert math.degrees(run.turned) >= 340.0
    assert t == pytest.approx(K.TURN_TARGET / 0.7, abs=0.2)
    # a slower body turns just as far, it only takes longer
    slow = K.CircleRun(innate_side=-1)
    _walk_loop(slow, rate=-0.35)
    assert slow.phase == "done"


def test_a_stall_stops_the_loop() -> None:
    run = K.CircleRun(innate_side=1)
    _walk_loop(run, rate=0.0, limit=10.0)
    assert run.phase == "stopped"


def test_the_side_its_own_if_there_is_room_else_the_other() -> None:
    arcs = K.ArcModel()
    everywhere = K.CircleRun(innate_side=1)
    assert everywhere.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=lambda h: 2.0)
    assert everywhere.side == 1

    def wall_left(h: float) -> float:  # a wall 0.3 m to its left
        return 0.3 if h > 0.2 else 2.0

    boxed = K.CircleRun(innate_side=1)
    assert boxed.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=wall_left)
    assert boxed.side == -1
    nowhere = K.CircleRun(innate_side=1)
    assert not nowhere.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=lambda h: 0.3)
    assert nowhere.phase == "stopped"


def test_the_twist_is_the_bodys_tightest_arc() -> None:
    run = K.CircleRun(innate_side=-1, tempo=0.0)
    run.start(0.0, (0.0, 0.0, 0.0), K.ArcModel())
    vx, vyaw = run.command()
    assert vx == pytest.approx(0.35) and vyaw == pytest.approx(-1.0)
    assert C.legal_walk(vx, vyaw, approach=True) == (vx, vyaw)


def _cmd(behaviour: str, **kw: object) -> MotorCommand:
    return MotorCommand(t_ms=0, behaviour=behaviour, bout=1, variant=7, **kw)  # type: ignore[arg-type]


def test_the_programs_are_seen_and_cue_their_posture() -> None:
    style = P.style_from({"sociability": 0.9}, stage="duckling")
    frame = SensorFrame(t_ms=0)
    with SB.registered():
        target = Target("person", 0.3, 0.7)
        for behaviour, cmd in (
            ("settle_feet", _cmd("settle_feet", target=target, amplitude=0.7)),
            ("hide", _cmd("hide", amplitude=0.7)),
            ("follow_mom", _cmd("follow_mom", target=target, amplitude=0.7)),
            ("circle_settle", _cmd("circle_settle", vx=0.35, vyaw=-1.0, amplitude=0.7)),
            ("peek_out", _cmd("peek_out", side=1, intensity=0.8, amplitude=0.7)),
        ):
            state = P.start(cmd, style, frame, 0)
            assert state.behaviour == behaviour
            assert P.gesture_deg(state.score) >= P.HOLD_DEG
            cues = {c.tag for c in state.score.cues if c.kind == "posture"}
            if behaviour in ("settle_feet", "hide"):
                assert "sit" in cues
    assert "hide" not in P.BUILDERS


def test_the_registry_adds_and_restores_the_shared_tables_exactly() -> None:
    before = (
        dict(C.BEHAVIOURS),
        C.MOMENTS,
        S.FEATURES,
        dict(S.F),
        {k: dict(v) for k, v in S.W.items()},
        dict(S.SATIATE_S),
        dict(B.LOCO),
        B.HEAD_ACTS,
        dict(P.BUILDERS),
        P.REACTIVE,
    )
    with SB.registered():
        assert set(SB.SOCIAL_BEHAVIOURS) <= set(C.BEHAVIOURS)
        assert "peek_out" in C.MOMENTS and "peek_out" in S.MOMENTS
        assert all(f in S.F for f in SB.SOCIAL_FEATURES)
        assert S.W["nap"]["circled"] > 0.0
        assert len(S.empty_features()) == len(S.FEATURES)
    after = (
        dict(C.BEHAVIOURS),
        C.MOMENTS,
        S.FEATURES,
        dict(S.F),
        {k: dict(v) for k, v in S.W.items()},
        dict(S.SATIATE_S),
        dict(B.LOCO),
        B.HEAD_ACTS,
        dict(P.BUILDERS),
        P.REACTIVE,
    )
    assert before == after


def test_the_social_brain_needs_the_registry() -> None:
    with pytest.raises(RuntimeError):
        SB.SocialBrain(preset("hatchling"), seed=1)


def test_synthetic_a_sleepy_standing_duck_circles_once_then_sleeps() -> None:
    with SB.registered():
        done = 0
        for name, seed in (("snuggler", 2), ("explorer", 1), ("hatchling", 1)):
            brain = SB.SocialBrain(preset(name), seed=seed)
            world = W.SocialSynth(seed=seed, day_minute0=1380.0, x=3.0, y=2.5, yaw=0.5)
            world._dock = (-50.0, -50.0)  # no dock in this room: it sleeps where it is
            brain.drives.sleep_pressure = 0.9
            brain.drives.energy = 0.5
            order: list[str] = []

            def each(t: int, cmd: MotorCommand, tel: object, order: list[str] = order) -> None:
                if not order or order[-1] != cmd.behaviour:
                    order.append(cmd.behaviour)

            W.run_social(brain, world, 240, each=each)
            if brain.circle.phase == "done":
                done += 1
                after = order[order.index("circle_settle") + 1 :]
                assert any(b in ("nap", "doze", "sit_down") for b in after[:4])
                assert math.degrees(brain.circle.turned) >= 340.0
        assert done >= 2


def test_no_room_here_it_walks_to_open_floor_first_then_loops() -> None:
    arcs = K.ArcModel()
    run = K.CircleRun(innate_side=-1)

    def cramped(h: float) -> float:  # a nook: walls close all round except straight behind
        return 2.0 if abs(abs(h) - math.pi) < 0.2 else 0.3

    assert run.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=cramped)
    assert run.phase == "seeking"
    assert abs(abs(run.seek_heading) - math.pi) < 0.2  # toward the open floor behind it
    vx, vyaw = run.command((0.0, 0.0, 0.0))
    assert vx >= 0.35 and abs(vyaw) == pytest.approx(1.0)  # an arc round toward it
    # it has walked 0.6 m into the open: now there is room for the loop
    run.step(3.0, (-0.6, 0.0, math.pi), arcs=arcs, clearance=lambda h: 2.0)
    assert run.phase == "turning"


def test_after_two_tries_without_room_it_just_sits() -> None:
    arcs = K.ArcModel()
    run = K.CircleRun(innate_side=1)

    def cramped(h: float) -> float:
        return 1.2 if h == 0.0 else 0.3

    assert run.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=cramped)
    for k in range(K.SEEK_TRIES):
        run.step(10.0 * (k + 1), (0.7 * (k + 1), 0.0, 0.0), arcs=arcs, clearance=cramped)
    assert run.phase == "stopped"
    assert K.best_open(lambda h: 0.3) is None


def test_a_touch_as_the_loop_begins_sends_it_to_find_room_and_try_again() -> None:
    arcs = K.ArcModel()
    run = K.CircleRun(innate_side=1)
    assert run.start(0.0, (0.0, 0.0, 0.0), arcs, clearance=lambda h: 2.0)
    assert run.phase == "turning"
    # its side brushed the wall beside it 0.1 s in: it walks off toward open floor …
    assert run.bumped(0.1, (0.0, 0.0, 0.0), clearance=lambda h: 2.0 if h == 0.0 else 0.4) == (
        "seeking"
    )
    assert run.seeks == 1
    # … a reflex's hold (1.2 s) waits: the seek's clock waits with it
    until = run.seek_until
    run.pause(1.2)
    assert run.seek_until == pytest.approx(until + 1.2)
    # there, room: it loops
    run.step(3.0, (0.6, 0.0, 0.0), arcs=arcs, clearance=lambda h: 2.0)
    assert run.phase == "turning"
    # late in a loop (or out of tries) a touch ends it where it is
    run.turned = 2.0
    assert run.bumped(8.0, (0.6, 0.3, 2.0), clearance=lambda h: 2.0) == "stopped"


def test_a_hold_does_not_count_as_a_stall() -> None:
    run = K.CircleRun(innate_side=1)
    run.start(0.0, (0.0, 0.0, 0.0), K.ArcModel(), clearance=lambda h: 2.0)
    t = 0.0
    while t < 10.0:  # a 10-s hold at the start (a long look): no heading change, not a stall
        run.pause(0.1)
        t += 0.1
    run.step(t, (0.0, 0.0, 0.0))
    assert run.phase == "turning"
