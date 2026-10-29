"""walkexec (one duck.walk → walk.done): driven tick by tick by a fake body and a fake eye.

The fake body is kinematic: it walks at the motion model's speed for the twist walkexec asks
for, stands up a second after a ``sit_toggle``, reports the head where it was told (one tick
late) and is "still" 200 ms after the last move or head change. The fake eye answers each look
100 ms later with a bright frame and free floor. Nothing here is physics: the physics is in the
gate harness; this checks the executor's logic and its percept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from duckmind import percept
from duckmind.duckbody import lookplan as L
from duckmind.duckbody import walkexec as W
from duckmind.duckbody.protocol import (
    ActContext,
    ActOutput,
    DepthView,
    PoseSample,
    ProprioView,
    Toward,
    WalkCmd,
)

HOME = (0.3490658503988659, 0.3490658503988659, 0.0, 0.0)
TICK_MS = 20


@dataclass
class FakeBody:
    posture: str = "stand"
    x: float = 0.0
    y: float = 0.0
    h: float = 0.0
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    moving_until: int = -10_000
    head_changed: int = -10_000
    toggled_at: int | None = None
    steps: int = 0
    stuck_head_down: bool = False  # A2: a duck that steps on the spot while the head is down
    halted_by: str | None = None
    lifted: bool = False
    unloaded: bool = False
    light: str = "bright"
    depth: DepthView | None = None
    epoch: int = 0
    cornered: bool = False  # what the body loop's cornered tracker would say
    twists: list[tuple[int, int]] = field(default_factory=list)
    back_twists: list[tuple[int, int]] = field(default_factory=list)
    heads: list[Any] = field(default_factory=list)

    def apply(self, out: ActOutput, now: int) -> None:
        if out.do == "sit_toggle":
            self.toggled_at = now
        if self.toggled_at is not None and now - self.toggled_at >= 1000:
            self.posture = "stand" if self.posture == "sit" else "sit"
            self.toggled_at = None
        head = out.head or (0.0, 0.0, 0.0, 0.0)
        if tuple(head) != self.head:
            self.head = tuple(head)  # type: ignore[assignment]
            self.head_changed = now
        self.heads.append(out.head)
        if out.twist is not None and out.twist[0] > 0:
            self.twists.append(out.twist)
            self.moving_until = now
            vx, vyaw = out.twist
            down = self.head[0] < -0.05
            v = 0.0 if (down and self.stuck_head_down) else L.speed_mm_s(vx) / 1000.0
            w = L.yaw_rate_mrad_s(vyaw) / 1000.0
            dt = TICK_MS / 1000.0
            self.x += v * dt * math.cos(self.h)
            self.y += v * dt * math.sin(self.h)
            self.h += w * dt
            self.steps += 1 if (now // 100) % 2 == 0 and v > 0 else 0
        elif out.twist is not None and out.twist[0] < 0:
            # reversing at −0.40 moves ~0.15 m/s and turns −4.5°/s plus 0.6 of the command
            self.back_twists.append(out.twist)
            self.moving_until = now
            dt = TICK_MS / 1000.0
            self.x -= 0.15 * dt * math.cos(self.h)
            self.y -= 0.15 * dt * math.sin(self.h)
            self.h += (-0.0785 + 0.6 * out.twist[1] / 1000.0) * dt

    def ctx(self, now: int, eye: tuple[Any, ...] = ()) -> ActContext:
        still = now - self.moving_until >= 200 and now - self.head_changed >= 200
        pose = PoseSample(
            seq=now // TICK_MS,
            t_ns=now * 1_000_000,
            trunk_height_mm=125,
            camera_pos=(0.08, 0.0, 0.12),
            camera_quat=(1.0, 0.0, 0.0, 0.0),
            tof_pos=None,
            tof_quat=None,
            imu_quat=(1.0, 0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            head=tuple(HOME[i] + self.head[i] for i in range(4)),  # type: ignore[arg-type]
            head_vel=0.0 if still else 0.5,
            odom=(self.x, self.y, self.h),
            epoch=self.epoch,
            still=still,
            policy="stand" if self.posture == "stand" else "sit",
        )
        return ActContext(
            t_ns=now * 1_000_000,
            state=None,
            pose=pose,
            depth=self.depth,
            proprio=ProprioView(
                posture=self.posture,
                policy="stand" if self.posture == "stand" else "sit",
                lifted=self.lifted,
                unloaded=self.unloaded,
                steps=self.steps,
                battery_pct=80,
            ),
            light=self.light,
            eye=eye,
            halted_by=self.halted_by,
            cornered=self.cornered,
        )


@dataclass
class FakeEye:
    pending: list[tuple[int, str]] = field(default_factory=list)
    asked: list[Any] = field(default_factory=list)
    free_mm: int = 2500

    def take(self, out: ActOutput, now: int) -> None:
        for look in out.looks:
            self.asked.append(look)
            self.pending.append((now + 100, look.look_id))

    def results(self, now: int) -> tuple[dict[str, Any], ...]:
        ready = [lid for t, lid in self.pending if t <= now]
        self.pending = [(t, lid) for t, lid in self.pending if t > now]
        return tuple(
            {
                "look_id": lid,
                "status": "ok",
                "results": {
                    "light": {"level": "bright", "luma": 90, "noise_q": 2},
                    "floor": {"free_mm": [self.free_mm] * 5, "ends": ["unknown"] * 5},
                },
            }
            for lid in ready
        )


def run(
    ex: W.WalkExec,
    body: FakeBody,
    eye: FakeEye | None = None,
    *,
    until_ms: int = 30_000,
    at: dict[int, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    eye = eye or FakeEye()
    now = ex.start_ms
    while now < ex.start_ms + until_ms:
        for t, action in (at or {}).items():
            if now == ex.start_ms + t:
                action(body)
        out = ex.step(body.ctx(now, eye.results(now)))
        body.apply(out, now)
        eye.take(out, now)
        if out.done:
            assert out.percepts, "a finished walk carries its walk.done"
            return dict(out.percepts[0]), now
        now += TICK_MS
    raise AssertionError(f"walk did not finish; phase {ex.phase}")


def stamped(event: dict[str, Any]) -> dict[str, Any]:
    return {**event, "src": "live", "pv": "p:0123456789abcdef"}


def cmd(**kw: Any) -> WalkCmd:
    base: dict[str, Any] = dict(
        act_id="w1", deadline_ms=10**9, vx_mm=400, max_ms=3000, pre_look="none"
    )
    base.update(kw)
    return WalkCmd(**base)


def seen_grid(now: int = 0) -> L.VetoGrid:
    grid = L.VetoGrid()
    for yaw in (-480, 0, 480):
        grid.mark_view(
            L.Pose2(0, 0, 0),
            [(yaw + b, 70, 3000) for b in (-260, -130, 0, 130, 260)],
            now,
            "bright",
        )
    return grid


# ── the speed policy (owner A3) ────────────────────────────────────────────


def test_speed_policy() -> None:
    cfg = W.WalkConfig(normal_vx_mm=550)
    assert W.speed_policy(cmd(vx_mm=600), cfg)[:2] == (550, 0)
    assert W.speed_policy(cmd(vx_mm=450), cfg)[0] == 450
    assert W.speed_policy(cmd(vx_mm=300), cfg)[0] == 350  # 0.30 steps on the spot (phase 2 E2)
    toward = Toward(look_id="l1", b_mrad=0, r_mm=600, seen_t_ms=0)
    assert W.speed_policy(cmd(vx_mm=550, toward=toward), cfg) == (400, 0, "approach")
    assert W.speed_policy(cmd(vx_mm=550, vyaw_mrad=-1000), cfg) == (400, -1000, "arc")
    assert W.speed_policy(cmd(vx_mm=550, vyaw_mrad=300), cfg)[0] == 550  # gentle steering
    assert W.speed_policy(cmd(vx_mm=400, vyaw_mrad=900, trim_mrad=300), cfg)[1] == 1000
    assert W.speed_policy(cmd(vx_mm=0), cfg)[0] == 0
    # owner 「2. 可以」: a negative vx is a back step at −0.30…−0.40 m/s
    assert W.speed_policy(cmd(vx_mm=-1000, vyaw_mrad=-1500), cfg) == (-400, -1000, "back")
    assert W.speed_policy(cmd(vx_mm=-100), cfg) == (-300, 0, "back")
    assert W.speed_policy(cmd(vx_mm=-350), cfg)[0] == -350


def test_reason_mapping() -> None:
    assert W.reason_of("stop_near") == "reflex_near"
    assert W.reason_of("no_drop") == "reflex_drop"
    assert W.reason_of("reflex_touch") == "reflex_touch"
    assert W.reason_of("unloaded") == "reflex_unloaded"
    assert W.reason_of("lifted") == "lifted"
    assert W.reason_of("dark") == "reflex_dark"


# ── whole walks ─────────────────────────────────────────────────────────────


def test_a_seated_duck_stands_glances_walks_and_reports() -> None:
    body = FakeBody(posture="sit")
    eye = FakeEye()
    ex = W.WalkExec(
        cmd(vx_mm=400, vyaw_mrad=293, pre_look="glance", max_ms=2000),
        start_ms=1000,
        grid=L.VetoGrid(),
    )
    done, t_end = run(ex, body, eye)
    assert percept.validate(stamped(done)) == []
    assert done["kind"] == "walk.done" and done["reason"] == "time" and done["act_id"] == "w1"
    assert done["cmd"] == {"vx": 400, "vyaw": 293, "max_ms": 2000}
    assert done["pre"]["look_id"] == "w1.pre" and done["pre"]["light"] == "bright"
    assert done["end"]["look_id"] == "w1.end"
    assert 330 <= done["odom"]["dx_mm"] <= 420  # 0.195 m/s × 2 s
    assert abs(done["odom"]["dy_mm"]) < 60
    assert done["odom"]["path_mm"] >= done["odom"]["dx_mm"]
    assert done["steps"] > 0 and done["battery_pct"] == 80 and done["head"] == "home"
    kinds = [look.kind for look in eye.asked]
    assert kinds == ["glance_down", "end_look"]
    # the glance happened standing still, never while the twist was non-zero
    assert all(t == (400, 293) for t in body.twists)
    # D1: the walked ground goes back to the eye for the floor model
    assert ex.learn is not None and ex.learn.outcome == "clear" and ex.learn.look_id == "w1.pre"
    assert len(ex.learn.floor) >= 3 and ex.learn.floor[-1][0] > 200
    # the grid now remembers the glance's view
    assert ex.grid is not None and ex.grid.cells


def test_an_unseen_route_walks_one_slow_step_and_a_partly_seen_one_is_shortened() -> None:
    """Owner 2026-09-28 「3. 一律放寬，即便跌倒才是真實模擬」: the look veto no longer cancels a
    walk in the light. Nothing seen → one step of UNSEEN_STEP_MS at the approach speed (the
    phase-3 fix found every walk of an unlearned duck ended veto_look); a partly seen route is
    still shortened to what was seen, at the walk's own speed."""
    body = FakeBody()
    ex = W.WalkExec(cmd(vx_mm=550), start_ms=0, grid=L.VetoGrid())
    done, _ = run(ex, body)
    assert done["reason"] == "time" and ex.veto_why == "unseen"
    assert done["cmd"] == {"vx": L.UNSEEN_VX_MM, "vyaw": 0, "max_ms": L.UNSEEN_STEP_MS}
    assert body.twists and all(t == (L.UNSEEN_VX_MM, 0) for t in body.twists)
    assert done["odom"]["path_mm"] >= 300  # 0.195 m/s × 2 s, and the tail
    assert percept.validate(stamped(done)) == []
    grid = L.VetoGrid()
    grid.mark_view(L.Pose2(0, 0, 0), [(b, 70, 700) for b in (-260, -130, 0, 130, 260)], 0, "bright")
    ex2 = W.WalkExec(cmd(vyaw_mrad=293, max_ms=6000), start_ms=0, grid=grid)
    done2, _ = run(ex2, FakeBody())
    assert done2["reason"] == "time" and done2["cmd"]["max_ms"] < 6000
    assert ex2.veto_why == "shortened" and done2["cmd"]["vx"] == 400


def test_a_wall_the_glance_sees_right_there_still_turns_the_walk_away() -> None:
    """Seen, not unseen: the glance sees a wall 0.25 m ahead — the walk is refused
    ``veto_look`` (Duck Core turns another way); an unlearned eye's glance (every end
    ``unknown``, 350 mm) never is."""

    class WallEye(FakeEye):
        ends = "wall"

        def results(self, now: int) -> tuple[dict[str, Any], ...]:
            out = super().results(now)
            for r in out:
                r["results"]["floor"] = {"free_mm": [250] * 5, "ends": [self.ends] * 5}
            return out

    done, _ = run(
        W.WalkExec(cmd(pre_look="glance"), start_ms=0, grid=L.VetoGrid()), FakeBody(), WallEye()
    )
    assert done["reason"] == "veto_look" and done["odom"]["path_mm"] == 0
    boot = WallEye()
    boot.ends = "unknown"
    body = FakeBody()
    done2, _ = run(W.WalkExec(cmd(pre_look="glance"), start_ms=0, grid=L.VetoGrid()), body, boot)
    assert done2["reason"] == "time" and done2["odom"]["path_mm"] > 200


def test_after_a_step_into_unseen_floor_the_next_walk_glances_first() -> None:
    memory = W.WalkMemory()
    build = W.make_factory(W.WalkConfig(), memory)
    first = build(cmd(act_id="u1", pre_look="auto"), 0, None)
    first.last_glance = (0, 0)  # a recent glance, same heading: no glance of its own …
    done, t_end = run(first, FakeBody())
    assert done["reason"] == "time" and memory.last_unseen
    second = build(cmd(act_id="u2", pre_look="auto"), t_end + 100, None)
    assert second.after_unseen  # … but after the unseen step the next walk looks first
    eye = FakeEye()
    run(second, FakeBody(), eye)
    assert [look.kind for look in eye.asked][:1] == ["glance_down"]


def test_the_dark_refuses_nothing_it_steps_slowly_into_what_it_has_not_seen() -> None:
    """Owner 2026-09-28 ~12:30 「2. 放寬」 (C2 reversed): in the dark a walk onto floor no look
    saw free is the slow unseen step, as in the light — physics decides what it meets."""
    tof = DepthView(0, None, None, None, 0, False, 0)
    for light, depth in (("dark", tof), ("black", None), ("none", None)):
        body = FakeBody(light=light, depth=depth)
        ex = W.WalkExec(cmd(vx_mm=550), start_ms=0, grid=L.VetoGrid())
        done, _ = run(ex, body)
        assert done["reason"] == "time" and ex.veto_why == "unseen", light
        assert done["cmd"] == {"vx": L.UNSEEN_VX_MM, "vyaw": 0, "max_ms": L.UNSEEN_STEP_MS}
        assert done["odom"]["path_mm"] >= 300 and percept.validate(stamped(done)) == []


def _attentive_eye_run(attention: str, **flags: Any) -> tuple[dict[str, Any], list[str], Any]:
    memory = W.WalkMemory()
    memory.last_glance = (0, 0)  # a recent glance, same heading
    for name, value in flags.items():
        setattr(memory, name, value)
    build = W.make_factory(W.WalkConfig(), memory)
    ex = build(cmd(act_id="a1", vx_mm=550, pre_look="auto", attention=attention), 1000, None)
    eye = FakeEye()
    done, _ = run(ex, FakeBody(), eye)
    return done, [look.kind for look in eye.asked], ex


def test_attention_decides_the_glance_and_the_end_look() -> None:
    """Owner 2026-09-28 ~12:30 「保留鴨子偶爾不注意撞到東西的呆萌感」: a duck whose attention is
    elsewhere does not glance before the walk nor look at its end, and — not knowing what it has
    not looked at — walks into that floor as asked (no slow unseen step), so it may bump into
    what is there. ``on`` is today's rule (after an unseen step the next walk glances; unseen
    floor is a slow short step); ``low`` glances only after a bump and walks on as asked."""
    done, kinds, ex = _attentive_eye_run("on", last_unseen=True)
    assert kinds == ["glance_down", "end_look"] and "attention" not in done
    # it saw the floor ahead: its own speed, as far as it saw
    assert ex.veto_why in ("clear", "shortened") and done["cmd"]["vx"] == 550
    done, kinds, ex = _attentive_eye_run("on")  # no glance due, nothing seen: the slow step
    assert kinds == ["end_look"] and ex.veto_why == "unseen"
    assert done["cmd"] == {"vx": L.UNSEEN_VX_MM, "vyaw": 0, "max_ms": L.UNSEEN_STEP_MS}
    done, kinds, ex = _attentive_eye_run("elsewhere", last_unseen=True, last_contact=True)
    assert kinds == [] and done["pre"] is None and done["end"] is None
    assert ex.veto_why == "unseen" and done["cmd"] == {"vx": 550, "vyaw": 0, "max_ms": 3000}
    assert done["attention"] == "elsewhere" and percept.validate(stamped(done)) == []
    done, kinds, ex = _attentive_eye_run("low", last_unseen=True)
    assert kinds == ["end_look"] and done["attention"] == "low"
    assert ex.veto_why == "unseen" and done["cmd"]["vx"] == 550
    done, kinds, _ = _attentive_eye_run("low", last_contact=True)
    assert kinds == ["glance_down", "end_look"]
    # an explicit glance is always taken, whatever the attention
    eye = FakeEye()
    glance = W.WalkExec(cmd(pre_look="glance", attention="elsewhere"), 0, grid=L.VetoGrid())
    run(glance, FakeBody(), eye)
    assert [look.kind for look in eye.asked] == ["glance_down"]


def test_what_it_did_look_at_still_stops_an_inattentive_walk() -> None:
    """The stop depends on what the duck really looked at: a wall a recent look saw right there
    turns the walk away whatever its attention; floor the look never covered does not."""
    wall = L.VetoGrid()
    wall.mark_view(L.Pose2(0, 0, 0), [(b, 70, 200, "wall") for b in (-260, 0, 260)], 0, "bright")
    done, _ = run(W.WalkExec(cmd(attention="elsewhere"), 100, grid=wall), FakeBody())
    assert done["reason"] == "veto_look" and done["odom"]["path_mm"] == 0
    # a look aimed 0.54 rad to the left (at a toy) saw a wall 0.6 m off there: straight ahead
    # it never looked, so that floor is unseen — the slow step, not a refusal
    aside = L.VetoGrid()
    aside.mark_view(
        L.Pose2(0, 0, 0), [(540 + b, 70, 600, "wall") for b in (-130, 0, 130)], 0, "bright"
    )
    ex = W.WalkExec(cmd(attention="elsewhere"), 100, grid=aside)
    done2, _ = run(ex, FakeBody())
    assert done2["reason"] == "time" and ex.veto_why == "unseen" and done2["odom"]["path_mm"] > 400
    assert done2["cmd"]["max_ms"] == 3000  # as asked: it does not know it has not looked
    # a wall it did see 0.8 m ahead: the inattentive walk still stops before it
    far = L.VetoGrid()
    far.mark_view(L.Pose2(0, 0, 0), [(b, 70, 800, "wall") for b in (-260, 0, 260)], 0, "bright")
    ex3 = W.WalkExec(cmd(attention="elsewhere", max_ms=6000), 100, grid=far)
    done3, _ = run(ex3, FakeBody())
    assert done3["reason"] == "time" and done3["cmd"]["max_ms"] < 3500
    assert 400 < done3["odom"]["dx_mm"] < 800 - L.BODY_HALF_MM


# ── owner 2026-09-28 「2. 可以」: the small back step when cornered ─────────────


def test_a_back_step_goes_only_when_cornered() -> None:
    body = FakeBody()  # nothing says the ways forward are shut
    ex = W.WalkExec(cmd(vx_mm=-400, max_ms=3000), start_ms=0, grid=seen_grid())
    done, _ = run(ex, body)
    assert done["reason"] == "veto_look" and not ex.back_granted
    assert not body.back_twists and not body.twists and done["odom"]["path_mm"] == 0
    assert percept.validate(stamped(done)) == []
    black = FakeBody(cornered=True, light="black")  # 「2. 放寬」: the dark refuses nothing
    done_dark, _ = run(W.WalkExec(cmd(vx_mm=-400), start_ms=0, grid=seen_grid()), black)
    assert done_dark["reason"] == "time" and black.back_twists


def test_a_cornered_back_step_backs_at_most_0_2_m() -> None:
    body = FakeBody(cornered=True, depth=DepthView(0, 250, 300, 280, 0, False, -150))
    eye = FakeEye()
    ex = W.WalkExec(cmd(vx_mm=-550, max_ms=5000, pre_look="glance"), start_ms=0, grid=seen_grid())
    done, _ = run(ex, body, eye)
    assert ex.back_granted and done["reason"] == "time"
    assert done["cmd"]["vx"] == -400  # −0.30…−0.40 only
    assert body.back_twists and all(-400 <= vx <= -300 for vx, _ in body.back_twists)
    assert not body.twists  # never a step forward
    assert "glance_down" not in [look.kind for look in eye.asked]  # it cannot see behind
    assert -200 <= done["odom"]["dx_mm"] <= -100
    cfg = W.DEFAULT
    assert done["odom"]["path_mm"] + cfg.back_coast_mm <= cfg.back_max_mm + 10
    assert percept.validate(stamped(done)) == []
    # a back arc (turning while backing) is a back step too
    arc = W.WalkExec(cmd(vx_mm=-400, vyaw_mrad=1000, max_ms=2000), start_ms=0, grid=seen_grid())
    body2 = FakeBody(cornered=True)
    run(arc, body2)
    assert arc.back_granted and body2.back_twists
    assert all(t == (-400, 1000) for t in body2.back_twists)


def test_what_is_ahead_never_ends_a_back_step_but_a_bump_behind_does() -> None:
    near = FakeBody(cornered=True, halted_by="reflex_near")  # the thing ahead is still there
    done, _ = run(W.WalkExec(cmd(vx_mm=-400), start_ms=0, grid=seen_grid()), near)
    assert done["reason"] == "time" and done["odom"]["dx_mm"] < -100

    def bump(b: FakeBody) -> None:
        b.halted_by = "stall"

    behind = FakeBody(cornered=True)
    done2, _ = run(
        W.WalkExec(cmd(vx_mm=-400), start_ms=0, grid=seen_grid()), behind, at={900: bump}
    )
    assert done2["reason"] == "stall" and done2["odom"]["path_mm"] < 135


def test_dark_rules() -> None:
    """Owner 「2. 放寬」 and 「一律放寬」 (the phase-4 landing): in the dark a walk goes exactly
    as it would in the light — no refusal and, since the landing, no 3 s segment cap either (it
    was pacing only); the eye seeing less there is look_veto's business, not a rule."""
    body = FakeBody(light="black")  # no ToF on this duck (depth None): it walks (「2. 放寬」)
    done, _ = run(W.WalkExec(cmd(max_ms=6000), start_ms=0, grid=seen_grid()), body)
    assert done["reason"] == "time" and done["cmd"]["max_ms"] == 6000  # (not cut to 3 s)
    lit_done, _ = run(W.WalkExec(cmd(max_ms=6000), start_ms=0, grid=seen_grid()), FakeBody())
    assert (done["cmd"], done["odom"]) == (lit_done["cmd"], lit_done["odom"])  # as in the light
    capped, _ = run(W.WalkExec(cmd(max_ms=3000), start_ms=0, grid=seen_grid()), FakeBody())
    assert done["odom"]["path_mm"] > capped["odom"]["path_mm"] * 3 // 2  # past the old 3 s

    def refuse(ctx: ActContext, walk: WalkCmd) -> tuple[bool, int | None, str]:
        return False, None, "veto_dark"  # a stricter gate than the default may still refuse

    strict = W.WalkExec(cmd(), start_ms=0, grid=seen_grid(), dark_gate=refuse)
    done_strict, _ = run(strict, FakeBody(light="black"))
    assert done_strict["reason"] == "veto_dark" and done_strict["odom"]["path_mm"] == 0
    body2 = FakeBody(light="dark", depth=DepthView(0, None, None, None, 0, False, 0))
    done2, _ = run(
        W.WalkExec(cmd(vyaw_mrad=293, max_ms=6000), start_ms=100, grid=seen_grid()), body2
    )
    lit = FakeBody(light="bright", depth=DepthView(0, None, None, None, 0, False, 0))
    done_lit, _ = run(
        W.WalkExec(cmd(vyaw_mrad=293, max_ms=6000), start_ms=100, grid=seen_grid()), lit
    )
    assert done2["reason"] == done_lit["reason"] == "time"
    assert done2["cmd"] == done_lit["cmd"] and done2["cmd"]["max_ms"] > 3000  # as in the light
    for light in ("dark", "black", "none"):
        assert W.default_dark_gate(FakeBody(light=light).ctx(0, ()), cmd()) == (True, None, "dark")


def test_a_reflex_ends_the_walk_with_its_reason_and_the_tof_range() -> None:
    body = FakeBody(depth=DepthView(0, 900, None, None, 0, False, -150))

    def near(b: FakeBody) -> None:
        b.halted_by = "stop_near"
        b.depth = DepthView(0, 290, None, None, 0, False, -150)

    done, _ = run(W.WalkExec(cmd(max_ms=5000), start_ms=0, grid=seen_grid()), body, at={1500: near})
    assert done["reason"] == "reflex_near" and done["stop_mm"] == 290
    assert done["end"] is not None  # the end look still happens after a reflex stop


def test_lifted_ends_at_once_without_an_end_look() -> None:
    body = FakeBody()
    eye = FakeEye()

    def lift(b: FakeBody) -> None:
        b.lifted = True

    done, _ = run(
        W.WalkExec(cmd(max_ms=5000), start_ms=0, grid=seen_grid()), body, eye, at={1000: lift}
    )
    assert done["reason"] == "lifted" and done["end"] is None
    assert [look.kind for look in eye.asked] == []


def test_preempt_and_deadline() -> None:
    body = FakeBody()
    ex = W.WalkExec(cmd(max_ms=5000), start_ms=0, grid=seen_grid())
    for now in range(0, 1000, TICK_MS):
        body.apply(ex.step(body.ctx(now)), now)
    out = ex.preempt("preempt", body.ctx(1000))
    assert out.done and out.percepts[0]["reason"] == "preempt"
    done, _ = run(
        W.WalkExec(cmd(max_ms=5000, deadline_ms=1200), start_ms=0, grid=seen_grid()), FakeBody()
    )
    assert done["reason"] == "deadline"


def test_a_walk_that_never_stands_gives_up() -> None:
    body = FakeBody(posture="sit")
    body.toggled_at = None

    class Stubborn(FakeBody):
        def apply(self, out: ActOutput, now: int) -> None:  # the toggle never takes
            super().apply(replace(out, do=None), now)

    stubborn = Stubborn(posture="sit")
    done, _ = run(W.WalkExec(cmd(), start_ms=0, grid=seen_grid()), stubborn)
    assert done["reason"] == "stall" and done["odom"]["path_mm"] == 0


# ── owner A2: the head-down approach and its fallback ─────────────────────


def _approach(**kw: Any) -> WalkCmd:
    toward = Toward(look_id="t1", b_mrad=0, r_mm=500, seen_t_ms=0, thing="ball")
    return cmd(vx_mm=400, max_ms=4000, head="down", toward=toward, curio_id="C1", **kw)


def test_head_down_approach_walks_head_down() -> None:
    body = FakeBody()
    done, _ = run(W.WalkExec(_approach(), start_ms=0, grid=seen_grid()), body)
    assert done["reason"] == "time" and done["head"] == "down"
    moving = [h for h, t in zip(body.heads, body.heads, strict=True) if h is not None]
    assert moving and all(h == L.to_radians(L.HEAD_DOWN_WALK) for h in moving[:5])


def test_head_down_without_progress_raises_the_head_and_goes_on() -> None:
    body = FakeBody(stuck_head_down=True)
    ex = W.WalkExec(_approach(), start_ms=0, grid=seen_grid())
    done, _ = run(ex, body)
    assert ex.watch is not None and ex.watch.fired_ms is not None
    assert done["head"] == "home" and done["reason"] == "time"
    assert done["odom"]["dx_mm"] > 150  # it walked once the head was up
    off = W.WalkConfig(head_down_fallback=False)
    done2, _ = run(
        W.WalkExec(_approach(), start_ms=0, grid=seen_grid(), config=off),
        FakeBody(stuck_head_down=True),
    )
    assert done2["reason"] == "veto_head"


def test_head_down_needs_a_target() -> None:
    ex = W.WalkExec(cmd(head="down"), start_ms=0, grid=seen_grid())
    assert ex.head_mode == "home"


@pytest.mark.parametrize("reason", ["time", "reflex_near", "lifted"])
def test_every_walk_done_is_valid_percept(reason: str) -> None:
    body = FakeBody(depth=DepthView(0, 900, 1200, None, 0, False, -150))
    at = {}
    if reason == "reflex_near":
        at = {800: lambda b: setattr(b, "halted_by", "stop_near")}
    elif reason == "lifted":
        at = {800: lambda b: setattr(b, "lifted", True)}
    done, _ = run(W.WalkExec(cmd(max_ms=2000), start_ms=0, grid=seen_grid()), body, at=at)
    assert done["reason"] == reason
    assert percept.validate(stamped(done)) == []


def test_the_factory_shares_memory_between_walks() -> None:
    memory = W.WalkMemory(grid=seen_grid())
    build = W.make_factory(W.WalkConfig(), memory)
    first = build(cmd(act_id="a1", vyaw_mrad=-1000, pre_look="glance", max_ms=1500), 0, None)
    done, t_end = run(first, FakeBody())
    assert done["reason"] == "time" and memory.walks == 1
    assert memory.last_was_arc and memory.last_glance is not None and memory.learn
    second = build(cmd(act_id="a2", pre_look="auto"), t_end + 100, None)
    assert second.after_arc and second.grid is memory.grid  # after an arc: glance again


def test_an_unknown_posture_is_never_toggled() -> None:
    body = FakeBody(posture="unknown")
    ex = W.WalkExec(cmd(), start_ms=0, grid=seen_grid())
    for now in range(0, 7000, TICK_MS):
        out = ex.step(body.ctx(now))
        assert out.do is None
        if out.done:
            break
    assert ex.reason == "stall"


def test_a_back_step_holds_for_an_unsure_body_and_goes_on() -> None:
    """A reverse start dips the leg load under the unsure mark (2026-09-28 rig probe: 4/48 open
    floor starts, 9/24 cornered back steps): owner B3 — stop, wait to know; the floor explains it,
    so the back step settles the head and goes on (its time budget kept)."""
    body = FakeBody(cornered=True)

    def unsure(b: FakeBody) -> None:
        b.unloaded = True

    def floor(b: FakeBody) -> None:
        b.unloaded = False

    ex = W.WalkExec(cmd(vx_mm=-400, max_ms=2500), start_ms=0, grid=seen_grid())
    done, _ = run(ex, body, at={700: unsure, 1300: floor})
    assert ex.back_holds == 1 and done["reason"] == "time", (done, ex.back_holds)
    assert -200 <= done["odom"]["dx_mm"] <= -100
    stays = FakeBody(cornered=True)
    ex2 = W.WalkExec(cmd(vx_mm=-400, max_ms=2500), start_ms=0, grid=seen_grid())
    done2, _ = run(ex2, stays, at={700: unsure})  # never explained: it ends as B3 says
    assert done2["reason"] in ("reflex_unloaded", "stall") and ex2.back_holds == 0


def test_a_stall_while_the_reverse_spins_up_is_not_a_bump() -> None:
    body = FakeBody(cornered=True)

    def stall(b: FakeBody) -> None:
        b.halted_by = "stall"

    def clear(b: FakeBody) -> None:
        b.halted_by = None

    ex = W.WalkExec(cmd(vx_mm=-400, max_ms=2500), start_ms=0, grid=seen_grid())
    done, _ = run(ex, body, at={340: stall, 360: clear})  # 40 ms into the reverse
    assert done["reason"] == "time" and done["odom"]["dx_mm"] < -100


def test_leaving_the_dock_steers_out_along_its_axis() -> None:
    """The dock recommendation's step 8: a leave_nest walk right after a docking walks out along
    the dock's axis (the dock's frame the docking left in the walk memory, carried by odometry):
    no look veto on the dock's own lane, the heading brought onto the axis, out to 1 m. A plain
    walk from a seat turned 17° would run into the V guides (2026-09-28 rig: 13 of 51 exits
    stumbled into a guide and fell)."""
    from duckmind.duckbody import dockexec as D

    frame = D.DockFrame(
        fix=D.DockPose(250.0, 40.0, 0.3), odom0=(0.0, 0.0, 0.0), epoch=0,
        geometry=D.GEOMETRY["backin"],
    )  # fmt: skip
    memory = W.WalkMemory(dock=frame)
    build = W.make_factory(W.WalkConfig(), memory)
    ex = build(cmd(act_id="x1", purpose="leave_nest", pre_look="glance", max_ms=6000), 0, None)
    body = FakeBody()
    done, _ = run(ex, body)
    assert ex.leaving and ex.veto_why == "dock_lane" and done["reason"] == "time"
    out = frame.pose((body.x * 1000.0, body.y * 1000.0, body.h))
    assert out.x >= 950 and abs(out.y) < 60 and abs(out.h) < 0.25, out
    assert body.twists[0][1] < 0  # it first turned right, back toward the axis
    assert memory.dock is frame  # (the leave keeps the frame until a walk goes elsewhere)
    other = build(cmd(act_id="w9", pre_look="none"), 10_000, None)
    run(other, FakeBody())
    assert memory.dock is None
    # a lift since the docking (a new odometry epoch): the frame no longer holds
    lifted = W.WalkExec(cmd(purpose="leave_nest"), start_ms=0, grid=seen_grid(), leave=frame)
    run(lifted, FakeBody(epoch=1))
    assert not lifted.leaving
