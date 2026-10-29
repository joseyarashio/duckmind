"""The phase-3 review's and verify gates' findings on duck-body, each pinned by a test.

* A1 belongs to curiosity acts: a plain ``duck.walk`` that names purpose ``curio`` gets no
  relaxed ToF stop, and head-down walking on a plain walk is only a near, fresh final approach.
* The guard: no walking while a look waits for its frame; no skills while unsure; reversing never
  faster than the measured −0.40.
* The light duck-body walks by is never brighter than the frame, and goes stale after 60 s.
* The ToF view carries the nearest return's bearing (A1's "is this the target").
* A duck held tilted in the arms is not a fall (no sit/stand in the hands); ``brushed`` follows
  the gait's measured turn; B2's second try stops when a leg motor stays past its stall.
* A walk stopped on an unsure lift that the floor explains ends ``stall`` with a contact.
* A look that sees the floor end sooner takes back older free cells.
* A fault in the brain stops the process loudly (legs zeroed, an error line, exit non-zero).
* The curiosity guard percept keeps its kind (the field is ``guard``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

from duckmind import percept
from duckmind.duckbody import depth as D
from duckmind.duckbody import lookplan as L
from duckmind.duckbody import walkexec as W
from duckmind.duckbody.bodyloop import BodyConfig, BodyLoop
from duckmind.duckbody.proprio import Proprio
from duckmind.duckbody.protocol import ActOutput, DepthView, Intent, LookRequest
from duckmind.duckbody.reflex import GuardContext, check_intent, verdict
from duckmind.duckbody.righting import Righting
from duckmind.duckbody.robotd_client import RobotState

NS = 1_000_000_000
PV = "p:0123456789abcdef"


def state(
    t: float,
    *,
    policy: str = "walk",
    load: float = 35.0,
    tilt_deg: float = 0.0,
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gyro: tuple[float, float, float] = (0.01, 0.0, 0.0),
    vx: float = 0.0,
    vyaw: float = 0.0,
    leg_current: float | None = None,
    head: tuple[float, float, float, float] | None = None,
) -> RobotState:
    g = (math.sin(math.radians(tilt_deg)), 0.0, -math.cos(math.radians(tilt_deg)))
    per = load / 10.0 if leg_current is None else leg_current
    currents = [per if i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14) else 0.0 for i in range(15)]
    joints = [0.0] * 15
    joints[5], joints[6] = 0.3491, 0.3491  # the head at HOME
    if head is not None:
        joints[5:9] = list(head)  # measured head joints (rad)
    return RobotState.from_wire(
        {
            "t_ns": 10 * NS + int(round(t * NS)),
            "policy": policy,
            "move": {"requested": [vx, 0.0, vyaw], "applied": [vx, 0.0, vyaw]},
            "safety": {"fallen": False, "limp": False, "gravity": list(g)},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": currents,
            "odom": {"position": [odom[0], odom[1], 0.0], "yaw": odom[2]},
            "imu": {"gyro": list(gyro), "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def ms(t: float) -> int:
    return (10 * NS + int(round(t * NS))) // 1_000_000


class Recorder:
    """An executor that streams a twist, asks the looks it is given, and records its contexts."""

    def __init__(self, cmd: Any, twist: tuple[int, int] = (400, 0)) -> None:
        self.command = cmd
        self.act_id = cmd.act_id
        self.twist = twist
        self.ctxs: list[Any] = []
        self.ask: list[LookRequest] = []

    def step(self, ctx: Any) -> ActOutput:
        self.ctxs.append(ctx)
        looks, self.ask = tuple(self.ask), []
        return ActOutput(twist=self.twist, looks=looks)

    def preempt(self, why: str, ctx: Any) -> ActOutput:
        return ActOutput(done=True, percepts=())


def loop_with(executor: dict[str, Any]) -> tuple[BodyLoop, dict[str, Any]]:
    made: dict[str, Any] = {}

    def factory(cmd: Any, t_ms: int, loop: Any) -> Recorder:
        rec = Recorder(cmd)
        rec.start_ms = t_ms  # type: ignore[attr-defined]
        made["act"] = rec
        return rec

    return BodyLoop(BodyConfig(executors={"duck.walk": factory, **executor})), made


def walk_cmd(t: float, **kw: Any) -> dict[str, Any]:
    base = {"type": "cmd", "cmd": "duck.walk", "act_id": "W1", "vx_mm": 400}
    base["deadline_ms"] = ms(t) + 60_000
    base.update(kw)
    return base


# ── A1 / A2 belong to curiosity acts ──────────────────────────────────────────


def test_a_plain_walk_naming_curio_gets_no_relaxed_stop() -> None:
    loop, made = loop_with({})
    toward = {"look_id": "L1", "b_mrad": 0, "r_mm": 200, "seen_t_ms": ms(0.5)}
    acks = loop.command(
        walk_cmd(1.0, purpose="curio", toward=toward, stop_ahead_mm=50), ms(1.0) * 1_000_000
    )
    assert acks[0]["ok"] is True
    near = DepthView(
        t_ns=ms(1.0) * 1_000_000,
        ahead_mm=120,
        left_mm=None,
        right_mm=None,
        drop_zones=0,
        touch=False,
        head_pitch_mrad=0,
        ahead_b_mrad=0,
    )
    loop.tick(state(1.0), depth=near)
    assert made["act"].ctxs[-1].halted_by == "reflex_near"  # the 30 cm stop holds
    # the reflex itself would relax it for a curiosity act (owner A1) — and only then
    cmd = made["act"].command
    view = loop.last_prox.view if loop.last_prox is not None else None
    assert view is not None
    assert verdict(proprio=view, depth=near, cmd=cmd, now_ms=ms(1.0), relax=False).halt
    assert not verdict(proprio=view, depth=near, cmd=cmd, now_ms=ms(1.0), relax=True).halt


@pytest.mark.parametrize(
    "r_mm,age_ms,ok",
    [(500, 500, True), (2000, 500, False), (500, 5000, False)],
)
def test_head_down_on_a_plain_walk_is_only_a_near_fresh_final_approach(
    r_mm: int, age_ms: int, ok: bool
) -> None:
    loop, made = loop_with({})
    t = 2.0
    toward = {"look_id": "L1", "b_mrad": 0, "r_mm": r_mm, "seen_t_ms": ms(t) - age_ms}
    loop.command(walk_cmd(t, head="down", toward=toward), ms(t) * 1_000_000)
    loop.tick(state(t))
    assert loop.last_guard.head_down_ok is ok
    ex = W.WalkExec(made["act"].command, start_ms=ms(t), grid=L.VetoGrid(), config=W.WalkConfig())
    assert ex.head_mode == ("down" if ok else "home")


# ── the guard ────────────────────────────────────────────────────────────────


def test_no_walking_while_a_look_waits_for_its_frame() -> None:
    loop, made = loop_with({})
    loop.command(walk_cmd(0.0), ms(0.0) * 1_000_000)
    for k in range(25):  # the head settles at HOME
        loop.tick(state(k * 0.02))
    made["act"].ask = [LookRequest(look_id="W1.pre", tasks=("light",))]
    loop.tick(state(0.5))
    tick = loop.tick(state(0.52))
    assert any(why == "no walking while a frame is being captured" for _, why in tick.refused)
    answer = {"look_id": "W1.pre", "status": "ok", "results": {"light": {"level": "bright"}}}
    tick = loop.tick(state(0.54), eye=[answer])
    assert not tick.refused
    assert any(i.method == "robot.move" and i.params["vx"] > 0 for i in tick.intents)
    # a look never answered is given up after 1.5 s
    made["act"].ask = [LookRequest(look_id="W1.lost", tasks=("light",))]
    loop.tick(state(0.56))
    assert loop.last_guard.capturing is False  # (built before this tick's asks)
    loop.tick(state(0.58))
    assert loop.last_guard.capturing is True
    loop.tick(state(2.2))
    assert loop.last_guard.capturing is False


def test_rule_3_is_judged_at_departure_and_a_walking_head_may_swing() -> None:
    settled = GuardContext(head_home_ms=1000, act="duck.walk")
    unsettled = GuardContext(head_home_ms=120, act="duck.walk")
    assert check_intent(Intent.move(0.55, 0.0), settled) is None
    assert check_intent(Intent.move(0.55, 0.0), unsettled)  # rule 3: not before a walk
    # once the walk streams, the gait swings the head: up to 0.40 rad in the A3 traces
    swinging = dataclasses.replace(unsettled, head_home_ms=0, moving=True, head_off_mrad=396)
    assert check_intent(Intent.move(0.55, -1.0), swinging) is None
    # … but a head commanded away, or truly displaced, still stops the walk
    assert check_intent(Intent.move(0.55, 0.0), dataclasses.replace(swinging, head_cmd_home=False))
    assert check_intent(Intent.move(0.55, 0.0), dataclasses.replace(swinging, head_off_mrad=600))
    # owner A2's final approach walks head down at approach speed only
    a2 = dataclasses.replace(swinging, head_cmd_home=False, approach=True, head_down_ok=True)
    assert check_intent(Intent.move(0.4, 0.0), a2) is None


def _swing_trials() -> dict[str, Any]:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "phase3_body" / "a3_head_swing.json"
    return dict(json.loads(path.read_text())["trials"])


@pytest.mark.parametrize("name", sorted(_swing_trials()))
def test_a_measured_walking_head_swing_is_never_refused(name: str) -> None:
    """The review's finding: the guard judged "settled 0.3 s" on every 50 Hz refresh, so the A3
    head swing at 0.55 m/s zeroed the walk within a step. Replayed through the whole BodyLoop,
    the walk now streams to its end once it has started."""
    trial = _swing_trials()[name]
    vx_mm, vyaw_mrad = round(trial["vx"] * 1000), round(trial["vyaw"] * 1000)
    loop, made = loop_with({})
    loop.command(walk_cmd(0.0, vx_mm=vx_mm), ms(0.0) * 1_000_000)
    made["act"].twist = (vx_mm, vyaw_mrad)
    refused_after_start: list[str] = []
    streamed = 0
    t = 0.0
    for _ in range(25):  # standing at HOME: the guard's 0.3 s settle, then the walk starts
        tick = loop.tick(state(t, load=110.0))
        t += 0.02
    assert loop.streamer.last != (0.0, 0.0)
    for head, odom in zip(trial["head"], trial["odom"]):
        for _ in range(2):  # the 25 Hz trace at the loop's 50 Hz
            s = state(t, head=tuple(head), odom=tuple(odom), load=110.0, gyro=(0.3, 0.2, 0.0))
            tick = loop.tick(s)
            refused_after_start += [why for _, why in tick.refused]
            streamed += sum(1 for i in tick.intents if i.method == "robot.move")
            t += 0.02
    assert trial["max_head_off_rad"] >= 0.3  # the trace really swings past the old 0.16
    assert refused_after_start == []
    assert loop.streamer.last == (trial["vx"], trial["vyaw"])
    assert streamed >= len(trial["head"]) * 2 * 0.9  # ~50 Hz refreshes for 6 s


def test_no_skills_while_unsure_and_no_fast_reversing() -> None:
    curio = GuardContext(head_home_ms=1000, act="duck.curio", back_arc_ok=True)
    assert check_intent(Intent.do("ground_pick"), curio) is None
    unsure = dataclasses.replace(curio, unloaded=True)
    assert check_intent(Intent.do("ground_pick"), unsure)
    assert check_intent(Intent.do("sit_toggle"), unsure) is None  # sitting is always allowed
    assert check_intent(Intent.move(-0.4, 1.0), curio) is None
    assert check_intent(Intent.move(-0.55, 1.0), curio)  # only −0.40 was ever measured


# ── the light duck-body walks by ─────────────────────────────────────────────


def test_the_light_is_never_brighter_than_the_frame_and_goes_stale() -> None:
    loop, made = loop_with({})
    loop.command(walk_cmd(0.0, deadline_ms=ms(0.0) + 600_000), ms(0.0) * 1_000_000)
    lagged = {
        "look_id": "a",
        "status": "ok",
        "results": {"light": {"level": "bright", "raw": "black", "luma": 20, "noise_q": 2}},
    }
    loop.tick(state(0.0), eye=[lagged])
    assert made["act"].ctxs[-1].light == "black"
    assert L.light_of(lagged) == "black"
    fine = {"look_id": "b", "status": "ok", "results": {"light": {"level": "dim", "raw": "dim"}}}
    loop.tick(state(1.0), eye=[fine])
    assert made["act"].ctxs[-1].light == "dim"
    loop.tick(state(60.9))
    assert made["act"].ctxs[-1].light == "dim"
    loop.tick(state(61.1))
    assert made["act"].ctxs[-1].light == "none"  # over 60 s old: the next walk glances first


def test_an_unknown_light_makes_the_walk_glance_first() -> None:
    from test_duckbody_walkexec import FakeBody, FakeEye, cmd, run, seen_grid

    body = FakeBody(light="none")
    eye = FakeEye()
    ex = W.WalkExec(cmd(pre_look="auto"), start_ms=0, grid=seen_grid())
    run(ex, body, eye)
    assert [a.look_id for a in eye.asked][:1] == ["w1.pre"]


# ── the ToF view ─────────────────────────────────────────────────────────────


def test_the_body_view_carries_the_nearest_returns_bearing() -> None:
    reading = D.DepthReading(
        t_ns=5, available=True, ahead_mm=420, drop_zones=2, drop=True, ahead_b_mrad=-137
    )
    view = D.body_view(reading)
    assert view is not None and view.ahead_b_mrad == -137 and view.drop_zones == 4
    calm = dataclasses.replace(reading, drop=False, drop_zones=7, left_mm=D.FAR_MM)
    view = D.body_view(calm)
    assert view is not None and view.drop_zones == 3 and view.left_mm is None
    assert D.body_view(D.DepthReading(t_ns=5, available=False)) is None


# ── body sense ───────────────────────────────────────────────────────────────


def test_a_duck_held_tilted_in_the_arms_is_not_a_fall() -> None:
    pr, rt = Proprio(), Righting()
    t = 0.0
    edges: list[str] = []
    dos: list[str] = []

    def tick(**kw: Any) -> None:
        nonlocal t
        s = state(t, **kw)
        out = pr.step(s, cmd=(0.0, 0.0))
        r = rt.step(s, out)
        edges.extend(out.edges)
        if r.do:
            dos.append(r.do)
        t += 0.02

    for _ in range(150):
        tick(policy="stand", load=35.0)
    for _ in range(50):
        tick(policy="stand", load=5.0)
    assert pr.lift == "lifted"
    for _ in range(1000):  # cradled on its side for 20 s
        tick(policy="stand", load=5.0, tilt_deg=70.0)
    assert "fell" not in edges and pr.lift == "lifted" and not rt.active and dos == []
    # a duck on the floor still falls
    fresh = Proprio()
    got = []
    for k in range(150):
        got += fresh.step(state(k * 0.02, policy="stand"), cmd=(0.0, 0.0)).edges
    for k in range(150, 250):
        got += fresh.step(state(k * 0.02, policy="stand", tilt_deg=50.0), cmd=(0.0, 0.0)).edges
    assert "fell" in got


@pytest.mark.parametrize("vx,vyaw", [(0.40, 1.0), (0.40, 0.5), (0.40, 0.18), (0.55, 0.0)])
def test_brushed_follows_the_gaits_own_turn(vx: float, vyaw: float) -> None:
    pr = Proprio()
    speed = 0.067 + 0.32 * vx
    rate = -0.175 + 0.597 * vyaw  # the gait's measured turn (lookplan's motion model)
    x = y = h = 0.0
    brushed = 0
    for i in range(int(30 / 0.02)):
        ts = i * 0.02
        x += speed * 0.02 * math.cos(h)
        y += speed * 0.02 * math.sin(h)
        h += rate * 0.02
        sway = 0.05 * math.sin(ts * 2 * math.pi * 1.6)
        s = state(ts, odom=(x, y, h + sway), load=110.0, gyro=(0.5 * math.sin(ts * 12), 0.3, 0.0))
        brushed += "brushed" in pr.step(s, cmd=(vx, vyaw)).edges
    assert brushed == 0
    # an unexplained 25° swing while still walking is a graze
    pr = Proprio()
    x = y = h = 0.0
    graze = 0
    for i in range(int(6 / 0.02)):
        ts = i * 0.02
        x += speed * 0.02 * math.cos(h)
        y += speed * 0.02 * math.sin(h)
        h += (rate + (0.45 if 3.0 <= ts < 4.0 else 0.0)) * 0.02
        s = state(ts, odom=(x, y, h), load=110.0, gyro=(0.5 * math.sin(ts * 12), 0.3, 0.0))
        graze += "brushed" in pr.step(s, cmd=(vx, vyaw)).edges
    assert graze >= 1


def test_the_second_try_stops_when_a_leg_motor_stays_past_its_stall() -> None:
    pr, rt = Proprio(), Righting()
    events: list[dict[str, Any]] = []
    t = 0.0
    for k in range(int(40 / 0.02)):
        t = k * 0.02
        tilt = 50.0 if t >= 3.0 else 0.0  # stuck nose-down on a threshold
        strain = 80.0 if rt.phase in ("sit", "rise") else None  # past the 64 proxy
        s = state(t, policy="stand", tilt_deg=tilt, leg_current=strain)
        out = rt.step(s, pr.step(s, cmd=(0.0, 0.0)))
        events += list(out.events)
    result = [e for e in events if e["kind"] == "body.righting"]
    assert result and result[0]["ok"] is False and result[0]["overload"] is True
    assert rt.last["over_stall_s"] >= 1.0 and rt.phase == "waiting_help"
    ev = {"src": "live", "t_ms": 1000, "pv": PV, **result[0]}
    assert percept.validate(ev) == []


# ── walkexec: the unsure stop the floor explains ─────────────────────────────


def test_a_wall_press_walk_ends_stall_with_a_contact() -> None:
    from test_duckbody_walkexec import FakeBody, FakeEye, cmd, seen_grid

    body = FakeBody()
    ex = W.WalkExec(cmd(max_ms=6000), start_ms=0, grid=seen_grid())
    eye = FakeEye()
    unsure = (1500, 2600)  # the trunk held still against a wall; the floor explains it after
    now = 0
    done: dict[str, Any] | None = None
    while now < 20_000:
        ctx = body.ctx(now, eye.results(now))
        if unsure[0] <= now < unsure[1]:
            ctx = dataclasses.replace(
                ctx,
                proprio=dataclasses.replace(ctx.proprio, unloaded=True),
                halted_by="reflex_unloaded",
            )
        out = ex.step(ctx)
        body.apply(out, now)
        eye.take(out, now)
        if out.done:
            done = dict(out.percepts[0])
            break
        now += 20
    assert done is not None
    assert done["reason"] == "stall" and done["contact"] == {"kind": "blocked", "side": "front"}
    assert ex.learn is None or ex.learn.outcome == "contact"
    assert percept.validate({**done, "src": "live", "pv": PV}) == []


# ── the free-cell memory ─────────────────────────────────────────────────────


def test_a_newer_look_takes_back_what_it_sees_end_sooner() -> None:
    grid = L.VetoGrid()
    origin = L.Pose2(0, 0, 0)
    grid.mark_view(origin, [(0, 60, 1500)], 0, "bright")
    assert grid.fresh(1000, 0, 1000, "bright")
    grid.mark_view(origin, [(0, 60, 600)], 2000, "bright")  # a box put down at 0.6 m
    assert grid.fresh(500, 0, 2000, "bright")
    assert not grid.fresh(800, 0, 2000, "bright")  # just past where the floor now ends
    assert grid.fresh(1300, 0, 2000, "bright")  # beyond the box: not seen either way
    grid.mark_view(origin, [(0, 60, 400)], 3000, "dark")  # dark frames prove nothing
    assert grid.fresh(500, 0, 3000, "bright")


# ── the process ──────────────────────────────────────────────────────────────


def test_a_brain_fault_stops_the_process_loudly(tmp_path: Any) -> None:
    from duckmind.duckbody import server as S

    class Robot:
        sent: list[Intent] = []

        def read_state(self, timeout: float) -> RobotState:
            return state(0.0)

        def send(self, intent: Intent) -> None:
            self.sent.append(intent)

    class Broken(BodyLoop):
        def tick(self, *a: Any, **k: Any) -> Any:
            raise ValueError("a bug in the brain")

    srv = S.BodyServer(S.Paths(str(tmp_path / "r.sock"), str(tmp_path / "b.sock")), loop=Broken())
    robot = Robot()
    srv._connect_robot = lambda: robot  # type: ignore[method-assign]
    heard: list[dict[str, Any]] = []
    srv._broadcast = heard.extend  # type: ignore[method-assign]
    srv._robot_loop()
    assert isinstance(srv.failed, ValueError) and srv.stopping.is_set()
    assert robot.sent and robot.sent[-1].method == "robot.move" and robot.sent[-1].params["vx"] == 0
    assert heard and heard[0]["type"] == "error" and "a bug in the brain" in heard[0]["error"]


# ── curiosity's guard percept ─────────────────────────────────────────────────


def test_the_tethered_guard_keeps_its_kind() -> None:
    import inspect

    from duckmind.curiosity import act_runner

    guard = {"guard": "tethered"}  # what the runner records (act_runner: self.guards)
    event = {
        "kind": "body.guard",
        "src": "live",
        "t_ms": 1000,
        "pv": PV,
        **{**guard, "act_id": "C1", "verb": "carry"},
    }
    assert event["kind"] == "body.guard" and percept.validate(event) == []
    assert '{"guard": "tethered"}' in inspect.getsource(act_runner)
    assert percept.validate({**event, "guard": "sleepy"})  # the guard's own list holds


def test_a_tethered_carry_reports_its_guard_through_the_body_loop() -> None:
    """The review's critical 4, driven for real: a fin that will not come along (tethered) makes
    the carry leg stall, the runner lets go (F2) and reports ``body.guard``; the report must be a
    valid percept (before the fix its ``kind`` overwrote the event's and the step raised, which
    killed duck-body's drive thread) and the body loop must stamp it like any other."""
    from curio_fake import FakeWorld, Thing, command, run, step_event

    class Tethered(FakeWorld):
        def _step_body(self, out: ActOutput) -> None:
            if self.thing.held and out.twist is not None and out.twist[0] > 0:
                out = dataclasses.replace(out, twist=(0, 0))  # the fin holds the duck back
            super()._step_body(out)

    fin = Thing("fin", 450.0, 0.0, cls="handle_toy", w=60, h=70, d=10)
    world = Tethered(fin, seed=4)
    _, grabbed, _ = run(
        world, command("grab", thing="fin", cls="handle_toy", carry_ok=True, h_mm=70, d_mm=10)
    )
    assert step_event(grabbed)["outcome"] == "held"
    _, out, _ = run(
        world, command("carry", thing="fin", step=2, cls="handle_toy", carry_ok=True, relay_ms=1000)
    )
    guards = [p for p in out.percepts if p.get("kind") == "body.guard"]
    assert [g["guard"] for g in guards] == ["tethered"]
    assert step_event(out)["beak"] == "tethered"
    loop = BodyLoop(BodyConfig())
    for body in out.percepts:
        event = loop.agg.emit_body(dict(body), 5000)
        assert percept.validate(event) == []
