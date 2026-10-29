"""dockexec (duck.dock → body.dock): both dock styles, the PG-gated retries, tag fixes carried by
odometry, and the geometry defaults against ``world.dock``; backin is dock:measure's recommended
choreography (owner 2026-09-28 「6. 可以」): the J-hook on either side, the far U-turn, the straight
reverse start, the alignment gate, the face stop with its walk-out, "stand, sit again".

The fake here is a 2-D duck in the dock's frame: it walks at the motion model's speed with the
measured rightward drift, reverses at 0.15 m/s and coasts 65 mm, slides 72 mm back when it sits,
and its odometry lives in a frame rotated and shifted from the dock's (the tracker must carry
fixes across). Backing in, the face stops its rear (the trunk never gets nearer the wall than the
face contact); sitting with the rear against the face only half-sits (dock:measure's pilot: PG
open). The fake eye reads the tag from the true pose when the tag is in view. PG closes when the
seated trunk is within a window around the seat. Nothing here is physics — the dock workflow's
headless runs and the phase-3 gates are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from duckmind import percept
from duckmind.duckbody import dockexec as D
from duckmind.duckbody import lookplan as L
from duckmind.duckbody.protocol import ActContext, ActOutput, DockCmd, PoseSample, ProprioView
from tests.checkout_parts import needs_evidence

TICK = 20
ODOM_ROT = 0.7  # the odometry frame is turned and shifted from the dock's
ODOM_SHIFT = (1.3, -0.4)


@dataclass
class FakeDock:
    style: str
    x: float
    y: float
    h: float
    posture: str = "sit"
    toggle_at: int | None = None
    coast: float = 0.0  # mm still to slide after a reverse stops
    last_vx: int = 0
    pg_fail: int = 0  # the first N sits read PG open whatever
    sits: int = 0
    see_tag: bool = True
    reverse_ticks: int = 0
    lifted: bool = False
    twists: list[tuple[int, int]] = field(default_factory=list)
    head_yaw: float = 0.0
    moved_at: int = -10_000
    half_sit: bool = False  # sat with the rear against the face (backin): PG stays open
    reverse_bias: float = 0.0  # extra turn while reversing (rad/s): a duck that drifts off
    pushed_ticks: int = 0  # ticks reversing against the face
    sit_xs: list[float] = field(default_factory=list)
    # leg odometry: counts each true displacement × this (1.0: exact); it sees no motion while
    # the face holds the trunk (the feet do not slip)
    odom_scale: float = 1.0
    ox: float = math.nan
    oy: float = math.nan

    def __post_init__(self) -> None:
        self.ox, self.oy = self.x, self.y

    def geo(self) -> D.DockGeometry:
        return D.GEOMETRY[self.style]

    def _face_stop(self) -> None:
        """Backing in, the face stops the rear: the trunk never nearer than the contact."""
        if self.style == "backin" and self.x < self.geo().face_contact_mm:
            self.x = float(self.geo().face_contact_mm)
            self.pushed_ticks += 1

    def apply(self, out: ActOutput, now: int) -> None:
        x0, y0 = self.x, self.y
        if out.do == "sit_toggle" and self.toggle_at is None:
            self.toggle_at = now
        if self.toggle_at is not None and now - self.toggle_at >= 1000:
            self.toggle_at = None
            if self.posture == "stand":
                self.posture = "sit"
                self.sits += 1
                contact = self.geo().face_contact_mm
                self.half_sit = self.style == "backin" and self.x <= contact + 20
                self.x += 72 * math.cos(self.h + math.pi)  # the sit slides the trunk back
                self.y += 72 * math.sin(self.h + math.pi)
                self._face_stop()
                self.sit_xs.append(self.x)
            else:
                self.posture = "stand"
                self.half_sit = False
                self.x += 48 * math.cos(self.h)
                self.y += 48 * math.sin(self.h)
        self.head_yaw = out.head[2] if out.head else 0.0
        dt = TICK / 1000.0
        self._move(out, now, dt)
        self.ox += self.odom_scale * (self.x - x0)
        self.oy += self.odom_scale * (self.y - y0)

    def _move(self, out: ActOutput, now: int, dt: float) -> None:
        if out.twist is not None and self.posture == "stand":
            vx, vyaw = out.twist
            self.twists.append(out.twist)
            self.moved_at = now
            if 0 < vx <= 300 and abs(vyaw) >= 900:
                # dock:measure's tight arc (vx 0.30, vyaw ±1.0): r ≈ 0.22 m either way
                v = L.speed_mm_s(vx)
                w = math.copysign(0.74, vyaw)
            elif vx > 0:
                v = L.speed_mm_s(vx)
                w = L.yaw_rate_mrad_s(vyaw) / 1000.0
            else:
                v = -150.0
                w = -0.0785 + 0.6 * vyaw / 1000.0 + self.reverse_bias
                self.reverse_ticks += 1
                self.coast = 65.0
            self.last_vx = vx
            self.x += v * dt * math.cos(self.h)
            self.y += v * dt * math.sin(self.h)
            self.h += w * dt
            self._face_stop()
        elif self.coast > 0:
            step = min(self.coast, 150.0 * dt)
            self.coast -= step
            self.x -= step * math.cos(self.h)
            self.y -= step * math.sin(self.h)
            self._face_stop()

    def pg(self) -> bool:
        if self.posture != "sit" or self.half_sit:
            return False
        if self.sits <= self.pg_fail:
            return False
        geo = self.geo()
        if self.style == "backin":  # the rear's spring plates: from the face to ~50 mm out
            return geo.face_contact_mm <= self.x <= geo.seat_mm + 50 and abs(self.y) <= 45
        return abs(self.x - geo.seat_mm) <= 40 and abs(self.y) <= 45

    def odom(self) -> tuple[float, float, float]:
        c, s = math.cos(ODOM_ROT), math.sin(ODOM_ROT)
        x, y = self.ox / 1000.0, self.oy / 1000.0
        return (ODOM_SHIFT[0] + c * x - s * y, ODOM_SHIFT[1] + s * x + c * y, self.h + ODOM_ROT)

    def ctx(self, now: int, eye: tuple[Any, ...]) -> ActContext:
        still = now - self.moved_at >= 200 and self.coast <= 0
        pose = PoseSample(
            seq=now // TICK,
            t_ns=now * 1_000_000,
            trunk_height_mm=125,
            camera_pos=None,
            camera_quat=None,
            tof_pos=None,
            tof_quat=None,
            imu_quat=None,
            gyro=None,
            head=(0.35, 0.35, 0.0, 0.0),
            head_vel=0.0,
            odom=self.odom(),
            epoch=0,
            still=still,
            policy=self.posture,
        )
        ctx = ActContext(
            t_ns=now * 1_000_000,
            state=None,
            pose=pose,
            depth=None,
            proprio=ProprioView(posture=self.posture, policy=self.posture, lifted=self.lifted),
            light="bright",
            eye=eye,
        )
        return ctx

    def fix(self, look_id: str) -> dict[str, Any]:
        # the face tag reads from 0.45 m to 3 m (closer it fills the picture); the plate's edge
        # tag reads looking down when it is 0.2–0.6 m ahead of the trunk
        # duck-eye's nest result as eye.nesttag.NestTask gives it: the tag under ``anchor`` with
        # ``yaw_mrad`` the heading of the tag's face normal in the duck's level frame (π when
        # the duck faces it squarely)
        geo = self.geo()
        nest: dict[str, Any] = {"ok": 1, "level": "bright", "anchor": None, "tags": [], "cue": None}
        for tag, tag_x, lo, hi in ((17, geo.face_mm, 450, 3000), (18, geo.edge_tag_mm, 200, 600)):
            dx, dy = tag_x - self.x, -self.y
            bearing = D.wrap(math.atan2(dy, dx) - self.h)
            r = math.hypot(dx, dy)
            if self.see_tag and abs(bearing - self.head_yaw) <= 0.33 and lo <= r <= hi:
                anchor = {
                    "id": tag,
                    "x_mm": int(round(r * math.cos(bearing))),
                    "y_mm": int(round(r * math.sin(bearing))),
                    "z_mm": 112,
                    "r_mm": int(round(r)),
                    "b_mrad": int(round(bearing * 1000)),
                    "e_mrad": -60,
                    "yaw_mrad": int(round(D.wrap(-self.h) * 1000)),
                    "yaw_amb_mrad": 0,
                    "conf_ppm": 900_000,
                }
                nest = dict(nest, anchor=anchor, tags=[anchor])
                break
        return {
            "look_id": look_id,
            "status": "ok",
            "results": {"light": {"level": "bright", "raw": "bright"}, "nest": nest},
        }


def drive(ex: D.DockExec, duck: FakeDock, until_ms: int = 240_000) -> dict[str, Any]:
    pending: list[tuple[int, str]] = []
    now = ex.start_ms
    while now < ex.start_ms + until_ms:
        ready = tuple(duck.fix(lid) for t, lid in pending if t <= now)
        pending = [(t, lid) for t, lid in pending if t > now]
        out = ex.step(duck.ctx(now, ready))
        duck.apply(out, now)
        for look in out.looks:
            assert "nest" in look.tasks
            pending.append((now + 120, look.look_id))
        if out.done:
            assert out.percepts
            return dict(out.percepts[0])
        now += TICK
    raise AssertionError(f"dock did not finish: {ex.phase} {ex.log[-12:]}")


def cmd(style: str | None = None, deadline: int = 10**9) -> DockCmd:
    return DockCmd(act_id="d1", deadline_ms=deadline, style=style)


def valid(event: dict[str, Any]) -> list[str]:
    return percept.validate({**event, "src": "live", "pv": "p:0123456789abcdef"})


def test_pose_from_fix() -> None:
    geo = D.GEOMETRY["forward"]
    p = D.pose_from_fix({"r_mm": 580, "b_mrad": 0, "yaw_mrad": 0}, geo)
    assert p is not None and p.x == pytest.approx(700) and p.y == pytest.approx(0, abs=1e-9)
    b = math.atan2(100, 580)
    p = D.pose_from_fix(
        {"r_mm": round(math.hypot(580, 100)), "b_mrad": round(b * 1000), "yaw_mrad": 0}, geo
    )
    assert p is not None and p.x == pytest.approx(700, abs=1) and p.y == pytest.approx(100, abs=1)
    assert D.pose_from_fix({"b_mrad": 100}, geo) is None  # a direction only (owner C5)


def test_forward_docks_on_the_first_try() -> None:
    duck = FakeDock("forward", x=900, y=80, h=math.pi + 0.2)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert valid(done) == []
    assert (
        done["contact"] is True and done["via"] == "pg" and done["tries"] == 1 and done["seen_tag"]
    )
    assert done["style"] == "forward"
    assert all(vx > 0 for vx, _ in duck.twists)  # walked in, never backwards
    assert ex.log.count("fix") >= 2  # one step, one look
    assert abs(duck.y) <= 45


def test_forward_from_off_the_axis_keeps_the_tag_where_the_head_can_find_it() -> None:
    """Phase-3 fix (rig re-run): from 0.36 m off the axis a bare 250 mm carrot turned the duck
    55° away from the dock (full right lock, 154° → 123°) and every later look missed the tag;
    walking in, the heading now aims no more than ``keep_tag_mrad`` off the tag's direction, so
    each fix finds it again."""
    duck = FakeDock("forward", x=1400, y=-360, h=math.radians(154.0))
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is True and done["tries"] == 1 and ex.misses == 0, ex.log
    first = next(t for t in duck.twists if t[0] > 0)
    # the tag is 10° left of the heading, the lane's 250 mm carrot 29° right: aim 26° off the
    # tag at most (the PI's kp 2.5 then asks ≈ −0.53 rad/s, not the −1.0 of the bare carrot)
    assert -700 <= first[1] <= -300, first
    bare = FakeDock("forward", x=1400, y=-360, h=math.radians(154.0))
    no_keep = D.DockConfig(style="forward", keep_tag_mrad=0)
    old = D.DockExec(cmd("forward"), start_ms=0, config=no_keep, pg_source=bare.pg)
    drive(old, bare)
    assert next(t for t in bare.twists if t[0] > 0)[1] == -1000  # the turn away it used to make


def test_forward_retries_by_reversing_out_when_pg_stays_open() -> None:
    duck = FakeDock("forward", x=850, y=-60, h=math.pi - 0.15, pg_fail=1)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is True and done["tries"] == 2
    assert duck.reverse_ticks > 0 and "reverse_out" in ex.log


def test_forward_gives_up_after_three_tries() -> None:
    duck = FakeDock("forward", x=850, y=0, h=math.pi, pg_fail=99)
    done = drive(D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg), duck)
    assert done["contact"] is False and done["tries"] == 3 and done["via"] == "none"
    assert valid(done) == []


def test_no_tag_no_dock() -> None:
    duck = FakeDock("forward", x=900, y=0, h=math.pi, see_tag=False)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is False and not done["seen_tag"] and ex.reason == "no_tag"
    assert not duck.twists  # it never walked blind


# ── backin: dock:measure's recommended choreography (owner 2026-09-28 「6. 可以」) ───────────


def phases(ex: D.DockExec, *names: str) -> list[str]:
    return [p for p in ex.log if p in names]


def test_backin_is_the_default_style() -> None:
    assert D.DockConfig().style == "backin" == D.DEFAULT_STYLE
    ex = D.DockExec(cmd(), start_ms=0)
    assert ex.config.style == "backin" and ex.side == "right" and ex.lane_y() == -375
    assert ex.turn_sign() == -1  # the right U-turn


def test_backin_j_hook_face_stop_walk_out_and_sit() -> None:
    """From 1.0 m straight out (inside the turning point) the duck walks out to the lane's far
    point, arcs round to face the dock, follows the lane in to 1.15 m, U-turns right onto the
    axis, reverses in (straight first, never faster than −0.40) until the face stops it, walks
    out 0.7 s and sits: PG on the first sit."""
    duck = FakeDock("backin", x=1000, y=-150, h=math.pi + 0.1)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert valid(done) == [] and done["style"] == "backin"
    assert done["contact"] is True and done["tries"] == 1 and done["seen_tag"], ex.log
    order = phases(ex, "to_lane", "to_turn", "uturn", "reverse_in", "bump_out", "sit")
    assert order[:4] == ["to_lane", "to_turn", "uturn", "reverse_in"], ex.log
    assert order[-3:] == ["reverse_in", "bump_out", "sit"], ex.log
    assert "line_up" not in ex.log  # the far U-turn replaced the old line-up leg
    assert any(s.startswith("face_stop") for s in ex.log)
    assert duck.pushed_ticks > 0  # the rear met the face: that was the stop
    reverse = [t for t in duck.twists if t[0] < 0]
    assert reverse and all(t[0] == -400 for t in reverse)  # never faster than −0.40
    assert reverse[0] == (-400, 80)  # the straight start: feed-forward only
    uturn = [t for t in duck.twists if t[0] == 300 and abs(t[1]) == 1000]
    assert uturn and all(t[1] == -1000 for t in uturn[-20:])  # the right U-turn at vx 0.30
    assert all(abs(t[0]) >= 300 for t in duck.twists)  # never on the spot
    assert duck.sit_xs and 158 <= duck.sit_xs[-1] <= 222  # seated on the plates, off the face


def test_backin_mirrored_j_hook_for_a_room_that_needs_the_left() -> None:
    """The apartment's dock has the corridor wall on its right: the command (or the home's
    configuration) turns the whole J-hook round — the lane on the dock's left, a left U-turn."""
    duck = FakeDock("backin", x=1700, y=250, h=math.pi - 0.1)
    left = DockCmd(act_id="d2", deadline_ms=10**9, side="left")
    ex = D.DockExec(left, start_ms=0, pg_source=duck.pg)
    assert ex.side == "left" and ex.lane_y() == 375 and ex.turn_sign() == 1
    done = drive(ex, duck)
    assert done["contact"] is True, ex.log
    uturn = [t for t in duck.twists if t[0] == 300 and abs(t[1]) == 1000]
    assert uturn and uturn[-1][1] == 1000  # a left U-turn
    configured = D.DockExec(cmd(), start_ms=0, config=D.DockConfig(turn_side="left"))
    assert configured.side == "left"
    with pytest.raises(ValueError):
        D.DockExec(cmd(), start_ms=0, config=D.DockConfig(turn_side="up"))


def test_backin_on_the_lane_already_skips_the_walk_out() -> None:
    duck = FakeDock("backin", x=1900, y=-300, h=math.pi)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is True
    assert ex.lane_loops == 0 and not any(t[0] == 300 and t[1] > 0 for t in duck.twists)
    assert ex.log.count("fix") >= 2  # one look a second along the lane, and the last one


def test_backin_retry_is_stand_and_sit_again_without_walking() -> None:
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi, pg_fail=1)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is True and done["tries"] == 2
    after = ex.log[ex.log.index("retry") :]
    assert after == ["retry", "sit", "pg"]  # stood up, sat again: no leg, no reverse, no look
    assert duck.sits == 2


def test_backin_gives_up_after_three_sits() -> None:
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi, pg_fail=99)
    done = drive(D.DockExec(cmd(), start_ms=0, pg_source=duck.pg), duck)
    assert done["contact"] is False and done["tries"] == 3 and done["via"] == "none"
    assert duck.sits == 3 and valid(done) == []


def test_backin_alignment_gate_walks_out_and_reverses_again() -> None:
    """Inside 0.6 m the gait-averaged pose must be on the axis (here a gate so tight that every
    reverse fails it): walk out along the axis and reverse again — at most twice; the third
    time the whole approach again (walk out, U-turn back, look), then it gives up."""
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi)
    config = D.DockConfig(gate_mm=1, gate_mrad=1)
    ex = D.DockExec(cmd(), start_ms=0, config=config, pg_source=duck.pg)
    done = drive(ex, duck)
    assert done["contact"] is False and ex.reason == "gate", (ex.reason, ex.log)
    assert ex.approaches == 2 and ex.log.count("gate") == 6
    assert ex.log.count("realign") == 4 and ex.log.count("walk_out") == 1
    assert ex.log.count("uturn") == 3  # the approach's, the way back to the tag's, the second
    assert duck.sits == 0  # it never sat misaligned
    assert valid(done) == []


def test_backin_face_stop_by_odometry_progress() -> None:
    """Leg odometry that under-counts by 10 % still says where the face is: after 30 mm of
    backing, < 12 mm of progress in 1.5 s is the face (the pose there reads ~95 mm short)."""
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi, odom_scale=0.9)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    done = drive(ex, duck)
    assert "face_stop:blocked" in ex.log and duck.pushed_ticks > 0, ex.log
    assert done["contact"] is True


def test_backin_blocked_early_counts_as_the_face() -> None:
    """blocked = arrived (dock:measure): the reverse leg never fails on a stall, it sits."""
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    now = 0
    pending: list[tuple[int, str]] = []
    stalled = False
    while now < 240_000:
        ready = tuple(duck.fix(lid) for t, lid in pending if t <= now)
        pending = [(t, lid) for t, lid in pending if t > now]
        ctx = duck.ctx(now, ready)
        if ex.phase == "reverse_in" and duck.x < 600 and not stalled:
            ctx = replace(ctx, halted_by="stall")  # the body's own stall reflex
            stalled = True
        out = ex.step(ctx)
        duck.apply(out, now)
        pending += [(now + 120, look.look_id) for look in out.looks]
        if out.done:
            break
        now += TICK
    assert "face_stop:stall" in ex.log and "bump_out" in ex.log


def test_backin_pose_stop_fallback() -> None:
    """``stop_by`` ``pose``: dock:measure's fallback stops by the pose at 289 mm (not 309) and
    sits there — no walk-out."""
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi)
    config = D.DockConfig(stop_by="pose")
    ex = D.DockExec(cmd(), start_ms=0, config=config, pg_source=duck.pg)
    done = drive(ex, duck)
    assert "pose_stop" in ex.log and "bump_out" not in ex.log
    assert duck.pushed_ticks == 0 or done["contact"] is True
    assert done["contact"] is True


def test_reverse_legs_ignore_what_is_ahead() -> None:
    """A dock's reverse leg backs away from what the ToF sees ahead (the forward style's retry
    faces the dock face): a near stop is not a reason to end it."""
    duck = FakeDock("forward", x=850, y=-60, h=math.pi - 0.15, pg_fail=1)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    now = 0
    pending: list[tuple[int, str]] = []
    while now < 240_000:
        ready = tuple(duck.fix(lid) for t, lid in pending if t <= now)
        pending = [(t, lid) for t, lid in pending if t > now]
        ctx = duck.ctx(now, ready)
        if ex.phase == "reverse_out":
            ctx = replace(ctx, halted_by="reflex_near")
        out = ex.step(ctx)
        duck.apply(out, now)
        pending += [(now + 120, look.look_id) for look in out.looks]
        if out.done:
            break
        now += TICK
    assert ex.reason == "pg" and "reverse_out" in ex.log


def test_lifted_or_deadline_ends_it() -> None:
    duck = FakeDock("forward", x=900, y=0, h=math.pi)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    for now in range(0, 3000, TICK):
        duck.apply(ex.step(duck.ctx(now, ())), now)
    duck.lifted = True
    out = ex.step(duck.ctx(3000, ()))
    assert out.done and out.percepts[0]["contact"] is False and ex.reason == "lifted"
    duck2 = FakeDock("forward", x=900, y=0, h=math.pi)
    done = drive(D.DockExec(cmd("forward", deadline=2000), start_ms=0, pg_source=duck2.pg), duck2)
    assert done["contact"] is False


def test_target_for_the_reflexes() -> None:
    duck = FakeDock("forward", x=900, y=0, h=math.pi)
    ex = D.DockExec(cmd("forward"), start_ms=0, pg_source=duck.pg)
    assert ex.target(duck.ctx(0, ())) is None
    now = 0
    while ex.fix_ms is None and now < 20_000:
        pending = ()
        out = ex.step(duck.ctx(now, pending))
        duck.apply(out, now)
        if out.looks:
            fixes = tuple(duck.fix(look.look_id) for look in out.looks)
            out2 = ex.step(duck.ctx(now + TICK, fixes))
            duck.apply(out2, now + TICK)
            now += TICK
        now += TICK
    target = ex.target(duck.ctx(now, ()))
    assert target is not None and abs(target["b_mrad"]) < 60 and 700 <= target["r_mm"] <= 800


def test_geometry_matches_the_world_dock() -> None:
    """forward is ``world.dock.approach()``; backin keeps the world's face, seat, edge tag and
    rear, with dock:measure's calibrated stop (309 → 289), turn (800 → 1150) and lane
    (333 → 375) — ``dock_styles.json`` → ``recommendation.dockexec_choreography_backin``."""
    dock = pytest.importorskip("duckmind.world.dock")
    for style in D.STYLES:
        shape = dock.shape(style)
        a = dock.approach(style, shape)
        geo = D.GEOMETRY[style]
        assert geo.face_mm == shape.face_mm
        assert geo.seat_mm == a["seat_x_mm"]
        assert geo.rear_mm == dock.REAR_MM
        assert (
            geo.edge_tag_mm == shape.depth_mm + shape.plate_length_mm - shape.edge_tag_mm // 2 - 15
        )
        if style == "forward":
            assert geo.line_up_mm == a["line_up_x_mm"] and geo.stop_mm == a["stop_x_mm"]
    backin = D.GEOMETRY["backin"]
    assert dock.approach("backin", dock.shape("backin"))["stop_x_mm"] - 20 == backin.stop_mm == 289
    assert (backin.turn_mm, backin.uturn_offset_mm, backin.far_mm) == (1150, 375, 1500)
    assert backin.face_contact_mm == dock.shape("backin").face_mm + dock.REAR_MM
    assert D.DockConfig().uturn_turn_mrad == 3107  # 178°
    assert set(D.STYLES) == set(dock.STYLES)


@needs_evidence
def test_the_choreography_numbers_are_dock_measures() -> None:
    import json
    from pathlib import Path

    evidence = Path(__file__).resolve().parents[1] / "docs/validation/evidence/phase2"
    reco = json.loads((evidence / "dock_styles.json").read_text())["recommendation"]
    assert reco["style"] == "backin" == D.DEFAULT_STYLE
    text = " ".join(reco["dockexec_choreography_backin"] + reco["dockexec_draft_differences"])
    for number in ("289", "1150", "375", "3107", "0.7 s", "−0.40", "0.8 s", "1.2 s", "600 mm"):
        assert number in text, number
    cfg = D.DockConfig()
    assert (cfg.reverse_straight_ms, cfg.kick_after_ms, cfg.kick_ms) == (800, 2000, 1200)
    assert (cfg.gate_x_mm, cfg.gate_mm, cfg.gate_mrad, cfg.max_relines) == (600, 80, 349, 2)
    assert (cfg.bump_walk_vx_mm, cfg.bump_walk_ms, cfg.blocked_window_ms) == (350, 700, 1500)
    assert (cfg.blocked_min_mm, cfg.blocked_progress_mm, cfg.max_sits) == (30, 12, 3)
    assert (cfg.uturn_vx_mm, cfg.uturn_vyaw_mrad, cfg.reverse_vx_mm) == (300, 1000, -400)


def test_the_factory_takes_the_commanded_style_and_side() -> None:
    build = D.make_factory(D.DockConfig(style="forward"))
    assert build(cmd(), 0, None).config.style == "forward"
    assert build(cmd("backin"), 0, None).config.style == "backin"
    assert build(cmd("backin"), 0, None).config.geo() == D.GEOMETRY["backin"]
    assert D.make_factory()(cmd(), 0, None).config.style == "backin"
    left = DockCmd(act_id="d3", deadline_ms=10**9, side="left")
    assert D.make_factory()(left, 0, None).side == "left"


def test_the_process_takes_the_homes_dock_style_and_side() -> None:
    """2026-09-28: the page's homes dock backin and the page passes the layout's side
    (``--dock-style``/``--dock-side``): a duck.dock that names neither follows them — the
    apartment's mirrored J-hook is left — and the command's own side still wins."""
    from duckmind.duckbody import server

    plain = DockCmd(act_id="d4", deadline_ms=10**9)
    ex = server.default_executors(None, "backin", "left")["duck.dock"](plain, 0, None)
    assert (ex.config.style, ex.side) == ("backin", "left")
    ex = server.default_executors()["duck.dock"](plain, 0, None)
    assert (ex.config.style, ex.side) == ("backin", "right")
    ex = server.default_executors(None, "forward", None)["duck.dock"](plain, 0, None)
    assert ex.config.style == "forward"
    right = DockCmd(act_id="d5", deadline_ms=10**9, side="right")
    assert server.default_executors(None, "backin", "left")["duck.dock"](right, 0, None).side == (
        "right"
    )


def test_backin_reverse_spin_up_is_not_the_face() -> None:
    """velstand spins up backwards in ~1 s and the start dips the leg load under the unsure mark
    (2026-09-28 rig probe): a stall before 30 mm of backing is not the face, and an unsure hold
    (owner B3: stop, wait to know) starts the leg over — straight start, progress window and all."""
    duck = FakeDock("backin", x=1600, y=-350, h=math.pi)
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg)
    now = 0
    pending: list[tuple[int, str]] = []
    reverse_at: int | None = None
    while now < 240_000:
        ready = tuple(duck.fix(lid) for t, lid in pending if t <= now)
        pending = [(t, lid) for t, lid in pending if t > now]
        ctx = duck.ctx(now, ready)
        if ex.phase == "reverse_in" and reverse_at is None and ex.leg and ex.leg.moving_ms:
            reverse_at = now
        if reverse_at is not None and ex.log.count("reverse_in") == 1:
            since = now - reverse_at
            if since == 100:  # 15 mm into the reverse (the fake has no spin-up)
                ctx = replace(ctx, halted_by="stall")  # the spin-up read as a stall
            if 600 <= since < 1000:
                ctx = replace(ctx, proprio=replace(ctx.proprio, unloaded=True))
        out = ex.step(ctx)
        duck.apply(out, now)
        pending += [(now + 120, look.look_id) for look in out.looks]
        if out.done:
            break
        now += TICK
    assert ex.reason == "pg", ex.log
    assert "unsure_hold" in ex.log
    stops = [s for s in ex.log if s.startswith("face_stop")]
    assert stops and duck.pushed_ticks > 0  # it stopped at the face, not at the spin-up


def test_a_docking_that_closes_pg_leaves_the_dock_frame_for_the_walk_out() -> None:
    from duckmind.duckbody import walkexec as W

    duck = FakeDock("backin", x=1600, y=-350, h=math.pi)
    memory = W.WalkMemory()
    ex = D.DockExec(cmd(), start_ms=0, pg_source=duck.pg, memory=memory)
    assert drive(ex, duck)["contact"] is True
    frame = memory.dock
    assert isinstance(frame, D.DockFrame) and frame.epoch == 0
    ox, oy, oh = duck.odom()
    here = frame.pose((ox * 1000.0, oy * 1000.0, oh))
    assert abs(here.x - duck.x) < 5 and abs(here.y - duck.y) < 5
    failed = FakeDock("backin", x=1600, y=-350, h=math.pi, pg_fail=99)
    memory2 = W.WalkMemory(dock=frame)
    drive(D.DockExec(cmd(), start_ms=0, pg_source=failed.pg, memory=memory2), failed)
    assert memory2.dock is None  # not docked: nothing to leave along
