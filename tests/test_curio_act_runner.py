"""The Act Runner (curiosity P1) against a kinematic stand-in world: every verb ends with one
valid body.curio_step, every output obeys the curiosity rules, and the gates hold."""

from __future__ import annotations

import math
import random

import pytest
from curio_fake import HELD_MOUTH, FakeWorld, Thing, command, run, step_event

from duckmind.curiosity import prims
from duckmind.curiosity.act_runner import CurioStep, curio_step_problems
from duckmind.curiosity.calib import CALIB
from duckmind.curiosity.perceive import Fix, Tracker
from duckmind.duckbody.protocol import CurioCmd
from duckmind.percept import validate


def _check_outputs(runner: CurioStep, outputs: list) -> None:
    assert runner.violations == []
    for out in outputs:
        if out.do is not None:
            assert out.do in prims.AUTONOMOUS_SKILLS
            assert out.twist in (None, (0, 0))
        if out.twist is not None:
            vx, wz = out.twist
            assert abs(vx) <= 400 and abs(wz) <= 1000
            assert not (vx == 0 and wz != 0)
    for event in outputs[-1].percepts:
        assert validate(event) == [], event


def test_look_sees_the_thing_and_reports_it() -> None:
    world = FakeWorld(Thing("ball", 700.0, 100.0))
    runner, out, outputs = run(world, command("look", b_mrad=140, r_mm=700))
    event = step_event(out)
    assert curio_step_problems(event) == []
    assert event["outcome"] == "seen" and event["ok"] is True
    assert event["thing_after"]["seen"] is True
    assert abs(event["thing_after"]["x_mm"] - 700) < 30
    assert event["sound"] == "unknown"  # owner A5
    kinds = [p["kind"] for p in out.percepts]
    assert kinds.count("body.curio_step") == 1 and "sense.thing" in kinds
    _check_outputs(runner, outputs)


def test_a_thing_out_of_sight_is_lost() -> None:
    world = FakeWorld(Thing("ball", 700.0, 0.0), see=False)
    runner, out, outputs = run(world, command("look"))
    event = step_event(out)
    assert event["outcome"] == "lost" and event["ok"] is False and event["lost"] is True
    assert event["thing_after"] == {"seen": False}
    _check_outputs(runner, outputs)


@pytest.mark.parametrize("start", [(900.0, 0.0), (700.0, 250.0), (1200.0, -300.0)])
def test_kick_approaches_confirms_and_kicks(start: tuple[float, float]) -> None:
    world = FakeWorld(Thing("ball", *start), seed=3)
    runner, out, outputs = run(world, command("kick", b_mrad=0, r_mm=900, cls="ball_slow"))
    event = step_event(out)
    assert curio_step_problems(event) == []
    assert event["verb"] == "kick" and event["foot"] in ("left", "right")
    assert event["outcome"] in ("rolled", "moved"), event
    assert world.kicked
    assert event["looks"] >= 3 and event["bursts"] == 1
    assert event["cone"]["ok"] is True
    _check_outputs(runner, outputs)
    # the kick came after two agreeing looks with the body still
    kick_at = next(i for i, o in enumerate(outputs) if o.do is not None)
    assert all(o.twist in (None, (0, 0)) for o in outputs[kick_at - 15 : kick_at + 1])


def test_kick_cone_blocks_a_kick_toward_a_keep_out() -> None:
    world = FakeWorld(Thing("ball", 700.0, 0.0), seed=1)
    keep = [{"x_mm": 1400, "y_mm": 0, "r_mm": 1500, "why": "fragile"}]
    runner, out, outputs = run(world, command("kick", keep_out=keep))
    event = step_event(out)
    assert event["ended_by"] == "cone_blocked" and event["ok"] is False
    assert not world.kicked
    assert event["cone"]["ok"] is False
    _check_outputs(runner, outputs)


def test_peck_nudges_a_ball_under_the_beak() -> None:
    world = FakeWorld(Thing("ball", 500.0, 40.0), seed=2)
    runner, out, outputs = run(world, command("peck", cls="ball_slow", h_mm=70, d_mm=70))
    event = step_event(out)
    assert event["verb"] == "peck" and event["bursts"] == 1
    assert event["outcome"] in ("squirted", "moved", "nothing"), event
    assert any(o.do == "ground_pick" for o in outputs)
    assert all(o.mouth in (None, 0.0) for o in outputs)  # a peck keeps the beak closed
    _check_outputs(runner, outputs)


def test_peck_at_a_flat_thing_is_unreachable() -> None:
    world = FakeWorld(Thing("sticker", 400.0, 0.0, cls="flat", h=5, d=60, w=60))
    runner, out, outputs = run(world, command("peck", thing="sticker", h_mm=5))
    event = step_event(out)
    assert event["outcome"] == "unreachable" and event["ok"] is False
    assert not any(o.do for o in outputs)


def test_grab_holds_the_fin_then_carry_and_put_down() -> None:
    fin = Thing("fin", 450.0, 0.0, cls="handle_toy", w=60, h=70, d=10)
    world = FakeWorld(fin, seed=4)
    runner, out, outputs = run(
        world, command("grab", thing="fin", cls="handle_toy", carry_ok=True, h_mm=70, d_mm=10)
    )
    event = step_event(out)
    assert event["outcome"] == "held" and event["beak"] == "held", event
    assert event["mouth_mrad"] > 0
    opened = [o.mouth for o in outputs if o.mouth and o.mouth > 0]
    assert opened, "the beak opened for the grab"
    _check_outputs(runner, outputs)
    # the relay leg, with the fin still held
    runner2, out2, outputs2 = run(
        world, command("carry", thing="fin", step=2, cls="handle_toy", carry_ok=True, relay_ms=1000)
    )
    event2 = step_event(out2)
    assert event2["outcome"] == "held" and event2["dropped_by"] == "chosen", event2
    assert event2["carried_mm"] > 0
    assert not fin.held  # put down at the end of the leg


def test_grab_refused_for_a_thing_that_may_not_be_carried() -> None:
    world = FakeWorld(Thing("ball", 500.0, 0.0))
    runner, out, outputs = run(world, command("grab", carry_ok=False))
    event = step_event(out)
    assert event["ended_by"] == "not_allowed" and not any(o.do for o in outputs)


def test_bump_pushes_the_thing() -> None:
    box = Thing("box", 500.0, 0.0, cls="box", w=100, h=120, d=100)
    world = FakeWorld(box, seed=5)
    runner, out, outputs = run(world, command("bump", thing="box", cls="box", d_mm=100, h_mm=120))
    event = step_event(out)
    assert event["deliberate"] is True
    assert event["outcome"] in ("moved", "nothing", "stuck", "missed"), event
    _check_outputs(runner, outputs)


@pytest.mark.parametrize("light", ["dark", "black", "none"])
def test_no_curiosity_act_in_the_dark_on_a_thing_it_does_not_have(light: str) -> None:
    """Owner 「一律放寬」 (the phase-4 landing): the dark no longer forbids a curiosity act, but
    an act on a thing the duck's perception does not have cannot start there — it simply cannot
    see it (duck-eye names no things in the dark), so it ends ``dark`` without a move."""
    world = FakeWorld(Thing("ball", 500.0, 0.0), light=light)
    runner, out, outputs = run(world, command("kick"))
    event = step_event(out)
    assert event["ended_by"] == "dark" and event["ok"] is False
    assert not any(o.do or o.twist for o in outputs)
    # a stale memory is no memory: seen 11 s before (a thing that stood still keeps 10 s)
    stale = Tracker("ball")
    stale.update(Fix(500, 0, world.t_ms - 11_000, "C0.L1", "ball"))
    _, out_stale, _ = run(FakeWorld(Thing("ball", 500.0, 0.0), light=light), command("kick"),
                          tracker=stale)  # fmt: skip
    assert step_event(out_stale)["ended_by"] == "dark"


@pytest.mark.parametrize("light", ["dark", "black", "none"])
def test_in_the_dark_an_act_starts_on_what_its_perception_still_has(light: str) -> None:
    """Owner 「一律放寬」: the duck saw the ball a moment ago (its tracker still keeps the sight)
    and the room is dark now — the kick starts and walks toward where it remembers the ball. Its
    looks see nothing in the dark, so it cannot confirm and ends ``lost`` once the memory ages
    out: no rule stops it, the camera's honesty and physics decide."""
    world = FakeWorld(Thing("ball", 600.0, 0.0), light=light, seed=11)
    memory = Tracker("ball")
    memory.update(Fix(600, 0, world.t_ms - 500, "C0.L1", "ball"))  # seen 0.5 s ago, lit
    runner, out, outputs = run(world, command("kick", cls="ball_slow"), tracker=memory)
    event = step_event(out)
    assert event["ended_by"] != "dark", event
    assert any(o.twist is not None and o.twist[0] > 0 for o in outputs)  # it went toward it
    assert world.x > 50  # the body really moved (the fake world's odometry)
    assert event["ended_by"] in ("lost", "gave_up", "done"), event
    assert not world.kicked or event["ended_by"] == "done"
    _check_outputs(runner, outputs)


def _drive(world: FakeWorld, runner: CurioStep, limit_s: float = 90.0) -> tuple:
    """Run one step to its end in ``world`` (curio_fake.run for a step built elsewhere)."""
    for _ in range(int(limit_s * 1000 / 20)):
        out = runner.step(world.ctx())
        world.apply(out, runner)
        world.close_grip()
        if out.done:
            return out, world.outputs
    raise AssertionError(f"the step did not end in {limit_s} s (phase {runner.phase})")


def test_the_next_step_on_the_same_thing_starts_in_the_dark_on_what_the_last_one_saw() -> None:
    """Owner 「一律放寬」 through duck-body's own factory: a ``look`` in the light sees the ball,
    the lights go out, and the next step on the same ball (the body loop keeps the last step's
    memory of its thing, :class:`ThingMemory`) starts in the dark on that memory. Another thing,
    a broken odometry frame or an aged memory gives nothing to start on (``dark``); in the light
    the memory is not used at all (the step looks for itself, as before)."""
    from duckmind.curiosity.act_runner import ThingMemory, executor

    class Loop:
        class cfg:  # noqa: N801 - a stand-in for BodyConfig
            src = "live"
            pv = "p:0123456789abcdef"

    loop = Loop()
    world = FakeWorld(Thing("ball", 600.0, 0.0), seed=11)
    look = executor(command("look", b_mrad=0, r_mm=600), world.t_ms, loop)
    out, _ = _drive(world, look)
    assert step_event(out)["outcome"] == "seen" and look.tracker.fresh(world.t_ms)
    world.light = "dark"  # the lights go out right after the look
    kick = executor(command("kick", step=2, cls="ball_slow"), world.t_ms, loop)
    assert isinstance(kick.memory, ThingMemory) and kick.memory.thing == "ball"
    out2, outputs = _drive(world, kick)
    event = step_event(out2)
    assert event["ended_by"] != "dark", event  # it started on what it still had
    assert any(o.twist is not None and o.twist[0] > 0 for o in outputs)
    _check_outputs(kick, outputs)
    # nothing to start on: another thing, another odometry frame, a memory that aged out
    memory = look.remembered()
    assert memory.recall("box", memory.epoch, look.tracker.last.t_ms + 1) is None
    assert memory.recall("ball", (memory.epoch or 0) + 1, look.tracker.last.t_ms + 1) is None
    assert memory.recall("ball", memory.epoch, look.tracker.last.t_ms + 10_001) is None
    stale_world = FakeWorld(Thing("ball", 600.0, 0.0), light="dark", t_ms=world.t_ms + 20_000)
    stale = CurioStep(command("kick", step=3), memory=memory)
    out3, outputs3 = _drive(stale_world, stale)
    assert step_event(out3)["ended_by"] == "dark" and not any(o.twist for o in outputs3)
    # in the light the memory is not used: the same step with and without it does the same
    lit_a, lit_b = (FakeWorld(Thing("ball", 600.0, 0.0), seed=3) for _ in range(2))
    with_memory = CurioStep(command("kick", step=4), memory=memory)
    without = CurioStep(command("kick", step=4))
    out_a, outs_a = _drive(lit_a, with_memory)
    out_b, outs_b = _drive(lit_b, without)
    assert step_event(out_a) == step_event(out_b)
    assert [(o.twist, o.head, o.do) for o in outs_a] == [(o.twist, o.head, o.do) for o in outs_b]


def test_a_thing_in_the_beak_goes_on_in_the_dark() -> None:
    """``carry`` / ``put_down`` go by the mouth joint, not the camera: in the dark the duck still
    carries the fin it holds (and a put_down with an empty beak says it missed, not "dark")."""
    fin = Thing("fin", 90.0, 0.0, cls="handle_toy", w=60, h=70, d=10, held=True)
    world = FakeWorld(fin, light="dark", seed=4)
    world.mouth = HELD_MOUTH
    world.mouth_cmd = 0.0
    params = dict(cls="handle_toy", carry_ok=True, relay_ms=1000)
    runner, out, outputs = run(world, command("carry", thing="fin", step=2, **params))
    event = step_event(out)
    assert event["ended_by"] != "dark" and event["outcome"] == "held", event
    assert event["carried_mm"] > 0
    empty = FakeWorld(Thing("fin", 400.0, 0.0, cls="handle_toy"), light="black")
    _, out2, _ = run(empty, command("put_down", thing="fin", step=3, cls="handle_toy"))
    event2 = step_event(out2)
    assert event2["ended_by"] != "dark" and event2["outcome"] == "missed", event2


def test_lifted_mid_step_ends_it_at_once() -> None:
    world = FakeWorld(Thing("ball", 900.0, 0.0), lifted_at_ms=3000)
    runner, out, outputs = run(world, command("kick"))
    event = step_event(out)
    assert event["ended_by"] == "lifted"
    assert out.twist is None and out.do is None


def test_fell_mid_step_reports_it() -> None:
    world = FakeWorld(Thing("ball", 900.0, 0.0), fallen_at_ms=4000)
    runner, out, outputs = run(world, command("kick"))
    event = step_event(out)
    assert event["ended_by"] == "fell" and event["fell"] is True


def test_toward_is_set_only_while_approaching_a_fresh_sight() -> None:
    world = FakeWorld(Thing("ball", 900.0, 0.0), seed=7)
    runner = CurioStep(command("kick"))
    seen_toward = []
    for _ in range(4000):
        ctx = world.ctx()
        out = runner.step(ctx)
        toward = runner.toward
        if toward is not None:
            seen_toward.append((runner.phase, toward))
            assert runner.phase in ("approach", "confirm", "act")
            assert ctx.t_ms - toward.seen_t_ms <= 3000
            assert toward.thing == "ball"
        world.apply(out, runner)
        if out.done:
            break
    assert seen_toward, "an approach names its just-seen target (owner A1)"
    assert runner.toward is None  # done: no relaxation any more


def test_hasty_and_careful_patience_reach_the_report() -> None:
    for tier in ("hasty", "careful"):
        world = FakeWorld(Thing("ball", 800.0, 50.0), seed=8)
        runner, out, outputs = run(world, command("kick", tier=tier))
        event = step_event(out)
        assert event["tier"] == tier
        assert event["tol_pct"] == {"hasty": 130, "careful": 80}[tier]


def test_seeded_random_steps_never_break_a_rule() -> None:
    """Property loop (hypothesis is not a dependency): random verbs, places and patience."""
    rng = random.Random(20260927)
    verbs = ["look", "kick", "peck", "bump", "nibble", "tap", "dribble"]
    for trial in range(24):
        verb = rng.choice(verbs)
        r = rng.uniform(300.0, 1400.0)
        b = rng.uniform(-0.9, 0.9)
        thing = Thing(
            "toy",
            r * __import__("math").cos(b),
            r * __import__("math").sin(b),
            cls=rng.choice(["ball_slow", "box", "cup"]),
            h=rng.choice([40, 70, 120]),
            d=rng.choice([40, 70, 100]),
        )
        world = FakeWorld(thing, seed=trial, noise_mm=rng.choice([1.0, 3.0, 6.0]))
        params = {
            "tier": rng.choice(["hasty", "normal", "careful"]),
            "cls": thing.cls,
            "h_mm": thing.h,
            "d_mm": thing.d,
        }
        runner, out, outputs = run(
            world, command(verb, thing="toy", step=trial + 1, **params), limit_s=120.0
        )
        event = step_event(out)
        assert curio_step_problems(event) == [], (trial, verb, event)
        _check_outputs(runner, outputs)
        assert event["violations"] == 0


def test_a_step_command_must_be_phase_step() -> None:
    with pytest.raises(ValueError):
        CurioStep(CurioCmd(act_id="C1", deadline_ms=1, phase="start", params={"verb": "kick"}))
    with pytest.raises(ValueError):
        CurioStep(command("juggle"))


def test_calibrated_windows_are_the_measured_ones() -> None:
    assert CALIB.grab.contains(52, 0) and CALIB.grab.contains(70, 20)
    assert not CALIB.grab.contains(75, 0)
    assert CALIB.kick_window.contains(77, 49) and not CALIB.kick_window.contains(125, 49)
    assert "robotd" in CALIB.source


def test_the_body_loops_factory_never_uses_the_oracle() -> None:
    from duckmind.curiosity.act_runner import executor

    class Loop:
        class cfg:  # noqa: N801 - a stand-in for BodyConfig
            src = "live"
            pv = "p:0123456789abcdef"

    step = executor(command("kick"), 0, Loop())
    assert step.perception == "camera" and step.pv == "p:0123456789abcdef"
    Loop.cfg.src = "robot"
    assert executor(command("kick"), 0, Loop()).perception == "robot"
    oracle = CurioStep(command("look"), perception="oracle_noise")
    world = FakeWorld(Thing("ball", 700.0, 0.0))
    for _ in range(3000):
        out = oracle.step(world.ctx())
        world.apply(out, oracle)
        if out.done:
            break
    event = step_event(out)
    assert any("for Duck Core" in p for p in curio_step_problems(event))


def _head_down_moves(outputs: list) -> list:
    return [o for o in outputs if o.twist not in (None, (0, 0)) and o.head is not None]


def test_the_final_approach_may_walk_head_down() -> None:
    """Owner A2: near the thing, the creeps keep the half-down look pose."""
    world = FakeWorld(Thing("ball", 700.0, 0.0), seed=12)
    runner, out, outputs = run(world, command("peck", h_mm=70, d_mm=70))
    moves = _head_down_moves(outputs)
    assert moves, "some creeps walked head down"
    assert all(o.head == (-0.125, 0.175, 0.0, 0.0) for o in moves)  # only the validated pose
    assert all(abs(o.twist[0]) <= 400 for o in moves)
    assert step_event(out)["head_fallback"] == 0
    _check_outputs(runner, outputs)


def test_no_progress_head_down_raises_the_head() -> None:
    """Owner A2's fallback: a head-down creep that barely moves raises the head for good."""
    world = FakeWorld(Thing("ball", 700.0, 0.0), seed=12, stall_head_down=True)
    runner, out, outputs = run(world, command("peck", h_mm=70, d_mm=70))
    event = step_event(out)
    assert event["head_fallback"] >= 1
    moves = _head_down_moves(outputs)
    last_down = max(i for i, o in enumerate(outputs) if o in moves)
    later = [o for o in outputs[last_down + 1 :] if o.twist not in (None, (0, 0))]
    assert later and all(o.head is None for o in later)  # head up (HOME) from then on
    assert event["outcome"] != "lost"


def test_a_ball_kicked_out_of_sight_is_moved_not_missed() -> None:
    """Gone from its spot and not found along the kick: it moved (how far is unknown)."""
    world = FakeWorld(Thing("ball", 800.0, 60.0), seed=5, kick_mm=4500.0)
    runner, out, outputs = run(world, command("kick"))
    event = step_event(out)
    assert world.kicked
    assert event["outcome"] == "moved" and event["lost"] is True and event["moved_mm"] is None
    assert event["thing_after"] == {"seen": False}


@pytest.mark.parametrize("verb", ["kick", "peck", "bump", "grab", "dribble", "tap"])
def test_every_output_passes_duck_bodys_own_guard(verb: str) -> None:
    """The body loop's forbidden-command guard (duckbody.reflex.check_intent) with the step's
    walk_view / head_down_ok (owner A1/A2) refuses nothing the Act Runner sends."""
    from duckmind.duckbody.protocol import Intent
    from duckmind.duckbody.reflex import GuardContext, Limits, check_intent

    thing = Thing(
        "toy",
        650.0,
        120.0,
        cls="handle_toy" if verb == "grab" else "ball_slow",
        w=60 if verb == "grab" else 70,
        h=70,
        d=10 if verb == "grab" else 70,
    )
    world = FakeWorld(thing, seed=21)
    runner = CurioStep(
        command(verb, thing="toy", cls=thing.cls, h_mm=thing.h, d_mm=thing.d, carry_ok=True)
    )
    home_since = None
    refused = []
    for _ in range(6000):
        ctx = world.ctx()
        out = runner.step(ctx)
        walk = runner.walk_view
        at_home = runner.head == (0.0, 0.0, 0.0, 0.0)
        home_since = (home_since if home_since is not None else ctx.t_ms) if at_home else None
        guard = GuardContext(
            head_home_ms=0 if home_since is None else ctx.t_ms - home_since,
            act="duck.curio",
            approach=walk is not None and walk.approach,
            head_down_ok=runner.head_down_ok,
            back_arc_ok=True,
        )
        intents = [Intent.head(out.head or (0.0, 0.0, 0.0, 0.0))]
        if out.do:
            intents.append(Intent.do(out.do))
        if out.twist:
            intents.append(Intent.move(out.twist[0] / 1000.0, out.twist[1] / 1000.0))
        for intent in intents:
            why = check_intent(intent, guard, Limits())
            if why:
                refused.append((runner.phase, intent.method, why))
        world.apply(out, runner)
        if out.done:
            break
    assert out.done
    assert refused == []


def test_walk_view_lets_the_tof_stop_relax_toward_the_seen_thing_only() -> None:
    """Owner A1 through duck-body's own reflex: the running move, seen as a curio walk toward
    the just-seen thing, relaxes stop_near for a return where the thing is — not elsewhere."""
    from duckmind.duckbody.protocol import DepthView
    from duckmind.duckbody.reflex import STOP_AHEAD_MM, relaxed_near

    world = FakeWorld(Thing("box", 700.0, 0.0, cls="box", w=100, h=120, d=100), seed=9)
    runner = CurioStep(command("bump", thing="box", cls="box", h_mm=120, d_mm=100))
    checked = 0
    for _ in range(4000):
        ctx = world.ctx()
        out = runner.step(ctx)
        walk = runner.walk_view
        if walk is not None and walk.toward is not None and checked < 50:
            assert walk.purpose == "curio" and walk.vx_mm <= 400
            assert 50 <= walk.stop_ahead_mm < STOP_AHEAD_MM
            here = DepthView(
                t_ns=ctx.t_ns,
                ahead_mm=walk.toward.r_mm - 40,
                left_mm=None,
                right_mm=None,
                drop_zones=0,
                touch=False,
                head_pitch_mrad=0,
                ahead_b_mrad=walk.toward.b_mrad,
            )
            assert relaxed_near(walk, here, ctx.t_ms) == walk.stop_ahead_mm
            elsewhere = DepthView(
                t_ns=ctx.t_ns,
                ahead_mm=walk.toward.r_mm - 40,
                left_mm=None,
                right_mm=None,
                drop_zones=0,
                touch=False,
                head_pitch_mrad=0,
                ahead_b_mrad=walk.toward.b_mrad + 600,
            )
            assert relaxed_near(walk, elsewhere, ctx.t_ms) is None
            checked += 1
        world.apply(out, runner)
        if out.done:
            break
    assert checked > 0


@pytest.mark.parametrize("start", [(700.0, 250.0), (900.0, -300.0), (500.0, 0.0), (1200.0, 200.0)])
def test_looks_aim_where_the_thing_is_after_each_move(start: tuple[float, float]) -> None:
    """Phase-3 live finding: a look queued before a leg was aimed with the pre-leg range (a HOME
    look at a ball 20 cm away). The pose is now chosen when the look comes up."""
    world = FakeWorld(Thing("ball", *start), seed=31)
    runner, out, outputs = run(world, command("kick"))
    before_act = world.look_log[: max(1, len(world.look_log) - 3)]  # the approach's looks
    assert sum(before_act) >= 0.8 * len(before_act), world.look_log


def _queued_move(ball: tuple[int, int], verb: str = "kick", foot: str = "left") -> str:
    """The primitive the approach picks for a ball at this trunk-frame spot (white box)."""
    from duckmind.curiosity.perceive import Fix, Odom

    world = FakeWorld(Thing("ball", 5000.0, 0.0))
    runner = CurioStep(command(verb, foot=foot))
    ctx = world.ctx()
    runner.odom = Odom(0, 0, 0)
    runner.now_ms = ctx.t_ms
    runner.tracker.update(Fix(ball[0], ball[1], ctx.t_ms, "C1.L1", "ball"))
    runner.phase = "approach"
    runner.align_started_ms = ctx.t_ms
    runner._decide_approach(ctx)
    moves = [op.prim.name for op in runner.queue if op.kind == "move" and op.prim is not None]
    return moves[0] if moves else "none"


def test_small_side_errors_near_the_ball_are_fixed_by_steered_creeps() -> None:
    """Phase-3 live finding: a 1.0 s reverse arc turns ±33° under robotd, far too much for a
    few centimetres to the side; near the ball those are fixed by steered creeps (or a straight
    step back when there is no room), the reverse arc only for big errors."""
    # the kick's blind pre-window centre is ~(147, 49): 3 cm beyond it and 2.4 cm too far left
    assert _queued_move((176, 73)) == "creep"
    # inside the pre-window: no move, confirm
    assert _queued_move((150, 40)) == "none"
    # too near to steer (no room ahead) and a little off: straight back, the next creep steers
    assert _queued_move((100, 10)) == "back"
    # beside it, far off: the reverse arc
    assert _queued_move((150, 190)) == "back_arc"
    # far away: a leg
    assert _queued_move((900, 60)) == "leg"


def test_a_thing_off_to_the_side_is_reached_without_losing_it() -> None:
    """Phase-3 live finding: legs steering at 2·bearing for 2 s overturned and left the thing
    beside the duck, out of the head's reach; legs now turn the error over their length, and a
    thing believed beside the duck gets a body turn before the next look."""
    for i, bearing in enumerate((-0.45, 0.45, -0.3)):
        world = FakeWorld(
            Thing("ball", 700.0 * math.cos(bearing), 700.0 * math.sin(bearing)), seed=40 + i
        )
        runner, out, outputs = run(world, command("kick", b_mrad=int(bearing * 1000), r_mm=700))
        event = step_event(out)
        assert event["ended_by"] != "lost", event
        legs = [o.twist for o in outputs if o.twist and abs(o.twist[0]) == 400]
        assert legs


def test_body_turns_when_the_thing_is_beside_it() -> None:
    from duckmind.curiosity.perceive import Fix, Odom

    world = FakeWorld(Thing("ball", 5000.0, 0.0))
    runner = CurioStep(command("kick"))
    ctx = world.ctx()
    runner.odom = Odom(0, 0, 0)
    runner.now_ms = ctx.t_ms
    runner.tracker.update(Fix(90, 420, ctx.t_ms, "C1.L1", "ball"))  # 78° to the left
    runner.phase = "find"
    runner.lost_streak = 2
    runner._decide(ctx)
    moves = [op.prim for op in runner.queue if op.kind == "move" and op.prim is not None]
    assert moves and moves[0].name == "back_arc" and moves[0].vyaw_mrad > 0


def test_any_thing_is_the_one_at_the_believed_place_whatever_the_eye_calls_it() -> None:
    # the eye names the ball "brown" (an orange ball in the room's light); the page's 「用眼睛做」
    # only knows it is in front of the duck: label "*" with that belief finds it by place
    world = FakeWorld(Thing("brown", 520.0, -40.0), seed=5)
    runner, out, outputs = run(
        world, command("look", thing="ball_felt", label="*", b_mrad=0, r_mm=500)
    )
    event = step_event(out)
    assert event["outcome"] == "seen" and event["thing"] == "ball_felt"
    _check_outputs(runner, outputs)
    # a named label still has to match what the eye says
    world = FakeWorld(Thing("brown", 520.0, -40.0), seed=5)
    _runner, out, _outputs = run(
        world, command("look", thing="ball_felt", label="orange", b_mrad=0, r_mm=500)
    )
    assert step_event(out)["outcome"] == "lost"
