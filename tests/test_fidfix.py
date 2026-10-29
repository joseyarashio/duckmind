"""fid:fix (2026-09-28) — the review's findings: every reflex plans what it will visibly do from
where the MEASURED head is, names only those parts, and the page's witness judges each named part
on the body; holds, moments and body events are said from the body; a stepping duck picked up
stops kicking at once; the page's speed is the body's."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from duckmind.duckbody.robotd_client import JOINT_NAMES, RobotState
from duckmind.neuro import reflex as R
from duckmind.neuro import repertoire as P
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.neuro.contract import (
    REFLEX_PARTS,
    BodySense,
    CameraSense,
    HeadPose,
    MotorCommand,
    SensorFrame,
    SoundSense,
)
from duckmind.simweb import neuro_api as N
from duckmind.simweb.neuro_witness import Tick, Witness

PARAMS = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)
DEG = math.pi / 180.0
HOME = 0.3491
STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"


def _body(posture: str = "stand", head: tuple[float, float, float, float] = (0, 0, 0, 0)) -> Any:
    """A body whose MEASURED head is ``head`` (offsets from HOME, rad)."""
    joints = (HOME + head[0], HOME + head[1], head[2], head[3])
    return BodySense(posture=posture, head=joints)


def _startle(head: tuple[float, float, float, float], posture: str = "stand", seed: int = 3) -> Any:
    d = R._Draw(seed)
    return R._startle(1000, "sound:clap", 0.8, None, PARAMS, 0.5, d, head=head, posture=posture)


# ── the startle: planned from where the head IS ───────────────────────────────


def test_a_startle_from_a_resting_head_throws_it_back() -> None:
    a = _startle((0.0, 0.0, 0.0, 0.0))
    assert "up" in a.parts and a.dip is None
    assert a.zh.startswith("嚇一跳，頭猛往後仰")


def test_a_second_clap_with_the_head_still_raised_jerks_it_down_first() -> None:
    """The review: 4 claps 6.5 s apart, the head held raised at ~13° by the freeze: the second
    startle had 0.4–1.2° of look-up left and still said 「頭猛往後仰」. Now it plans a dip."""
    raised = (-0.13, -0.22, 0.0, 0.0)  # the walking network's forward neck; the head pitch up 13°
    a = _startle(raised)
    assert "dip" in a.parts and "up" not in a.parts
    assert a.dip is not None and a.dip_ms >= 150
    assert a.zh.startswith("嚇一跳，頭猛一沉")
    # the dip goes well below where the nose is now, then the snap comes back up
    assert a.dip.hp * R.PITCH_PER_CLIP < 0.22 - 0.3
    assert a.pose.hp > 0.6


def test_seated_with_the_neck_forward_the_neck_is_named_too() -> None:
    a = _startle((-0.12, 0.05, 0.0, 0.0), posture="sit")
    assert "up" in a.parts and "neck" in a.parts
    assert a.zh.startswith("嚇一跳，頭猛往後一縮")
    b = _startle((0.0, 0.0, 0.0, 0.0), posture="sit")
    assert "neck" not in b.parts  # from HOME the seated neck's pull-back is too near the line


def test_the_sideways_jerk_leans_to_its_own_side_and_is_never_the_same() -> None:
    sides, sizes = [], []
    for seed in range(300):
        a = _startle((0.0, 0.0, 0.0, 0.0), seed=seed)
        yaw = a.pose.hy * R.YAW_PER_CLIP
        sides.append(yaw > 0)
        sizes.append(abs(yaw))
    own = sum(sides) / len(sides)  # PARAMS.side = +1 (left), lean +1
    assert 0.55 <= own <= 0.75, own  # ~65/35, not always the same side (the review: 24/33)
    assert statistics.pstdev(sizes) / statistics.mean(sizes) >= 0.15


def test_the_jerk_goes_the_other_way_when_the_head_has_no_room_that_way() -> None:
    for seed in range(40):
        a = _startle((0.0, 0.0, 0.95, 0.0), seed=seed)  # already turned far left
        assert a.pose.hy * R.YAW_PER_CLIP < 0.95  # never past the range
        if "yaw_right" in a.parts:
            assert a.zh.endswith("甩向右邊")


def test_the_parts_have_words() -> None:
    assert set(REFLEX_PARTS) >= {"up", "dip", "neck", "yaw_left", "yaw_right", "down"}


# ── orienting: from where the head points ─────────────────────────────────────


def _orient(bearing: float | None, head_yaw: float, seed: int = 5) -> Any:
    d = R._Draw(seed)
    head = (0.0, 0.0, head_yaw, 0.0)
    return R._orient(1000, "visual:thing", 0.5, bearing, 0.0, PARAMS, 0.5, d, head=head)


def test_an_orient_the_head_is_already_on_is_a_refixation_no_row() -> None:
    assert _orient(-0.40, -0.38) is None


def test_an_orient_says_the_way_the_head_goes() -> None:
    a = _orient(-0.5, 0.0)
    assert a.zh == "頭轉向右邊" and a.parts == ("yaw_right",)
    back = _orient(-0.35, -0.9)  # turned past it: the head comes back LEFT to look right
    assert back.zh == "頭轉回來一點，看右邊" and back.parts == ("yaw_left",)
    front = _orient(0.0, 0.6)
    assert front.zh == "頭轉回正前方" and front.parts == ("yaw_right",)


def test_a_thing_straight_ahead_is_looked_at_with_one_eye() -> None:
    a = _orient(0.05, 0.0)
    assert a is not None and a.zh == "側過頭，用一隻眼睛看前面"
    assert abs(a.aim) >= R.ONE_EYE_RAD[0] and a.parts


def test_a_sound_with_no_direction_cocks_the_head_from_where_it_is() -> None:
    d = R._Draw(8)
    a = R._orient(1000, "sound:clap", 0.5, None, 0.1, PARAMS, 0.5, d, head=(0.0, 0.0, 0.3, 0.0))
    assert a.zh == "側過頭聽（聽不出方向）" and a.parts
    assert abs(a.pose.hy * R.YAW_PER_CLIP - 0.3) >= R.TURN_SEEN_RAD


def _frames_with_sounds(times: list[int], until: int, head_yaw: float = 0.0) -> list[SensorFrame]:
    body = _body(head=(0.0, 0.0, head_yaw, 0.0))
    frames = []
    for t in range(0, until, 20):
        snd = SoundSense(level_db=58.0, onset=0.5, kind="voice") if t in times else SoundSense()
        frames.append(SensorFrame(t_ms=t, body=body, sound=snd))
    return frames


def _step_all(frames: list[SensorFrame], **kw: Any) -> list[Any]:
    state = R.initial(2)
    outs = []
    for frame in frames:
        state, out = R.step(PARAMS, state, frame, **kw)
        outs.append(out)
    return outs


def test_the_same_look_again_within_three_seconds_gets_no_second_row() -> None:
    outs = _step_all(_frames_with_sounds([200, 1400, 5400], 7000))
    rows = [e for o in outs for e in o.events if e.reflex == "orient"]
    heard = [p for o in outs for p in o.presented if p.stimulus == "sound:voice"]
    assert len(heard) == 3 and len(rows) == 2  # the second voice merged into the first look
    assert all(e.parts for e in rows)


def test_a_held_act_makes_every_look_covert() -> None:
    outs = _step_all(_frames_with_sounds([200, 3600], 6000), still_act=True)
    assert not [e for o in outs for e in o.events if e.reflex == "orient"]
    assert len([p for o in outs for p in o.presented if p.reflex == "orient"]) == 2
    assert all(o.head is None or o.active != "orient" for o in outs)


# ── the brain's moments: only with room ───────────────────────────────────────


def test_no_look_down_moment_with_the_head_already_low() -> None:
    low = SensorFrame(t_ms=0, body=_body(head=(-0.13, 0.12, 0.0, 0.0)))
    up_room, down_room = R.nose_room(low)
    assert down_room < 0.21
    raised = SensorFrame(t_ms=0, body=_body(head=(-0.13, -0.2, 0.0, 0.0)))
    assert R.nose_room(raised)[0] < 0.21


def test_a_look_down_moment_goes_down_from_where_the_head_is() -> None:
    style = P.style_from({"activity": 0.5}, stage="juvenile", side=1)
    cmd = MotorCommand(t_ms=0, behaviour="peek_down", bout=1, variant=3, amplitude=0.6)
    frame = SensorFrame(t_ms=0, body=BodySense(posture="stand"))
    start = (0.3, -0.35, 0.0, 0.0, 0.0)  # a forage's head-down pose
    state = P.start(cmd, style, frame, 0, start)
    lowest = min(state.score.pose_at(t)[1] for t in range(0, int(state.score.length), 20))
    assert lowest < start[1] - 0.3  # it goes further down from there, not to a fixed pose


# ── the witness: each named part, in its direction ────────────────────────────


def _witness(
    parts: list[str], pitch_deg: float = 0.0, yaw_deg: float = 0.0, *, reflex: str = "startle"
) -> dict:
    w = Witness()
    audits: list[dict] = []
    for i in range(160):
        t = i * 0.02
        u = min(1.0, max(0.0, (t - 1.08) / 0.2))
        head = (0.0, -pitch_deg * DEG * u, yaw_deg * DEG * u, 0.0)
        tick = Tick(t_ms=int(round(t * 1000)), head=head, trunk=(0.0, 0.0, 0.117, 0.0), trunk_t=t)
        events = []
        if t >= 1.0:
            events = [{"t_ms": 1000, "reflex": reflex, "stimulus": "sound:clap", "parts": parts}]
        audits += w.feed(tick, events)
    return next(a for a in audits if a.get("kind") == "reflex")


def test_the_named_part_is_judged_not_any_motion() -> None:
    done = _witness(["up", "yaw_right"], pitch_deg=14, yaw_deg=-20)
    assert done["verdict"] == "done" and done["zh"].startswith("身體：頭往後仰 14°")
    # the review: the head went sideways 20°, the named look-up never came → not done in green
    only_yaw = _witness(["up", "yaw_right"], pitch_deg=1, yaw_deg=-20)
    assert only_yaw["verdict"] == "partial"
    assert "頭往後仰沒有做出來" in only_yaw["zh"] or "頭往後仰只有" in only_yaw["zh"]
    # a turn the other way is not the named turn
    wrong = _witness(["yaw_right"], yaw_deg=18, reflex="orient")
    assert wrong["verdict"] == "not_done"


def test_a_dip_is_judged_downward() -> None:
    assert _witness(["dip"], pitch_deg=-12)["verdict"] == "done"
    assert _witness(["dip"], pitch_deg=12)["verdict"] == "not_done"


def test_a_head_already_moving_does_not_earn_the_reflex_its_turn() -> None:
    w = Witness()
    audits: list[dict] = []
    for i in range(160):
        t = i * 0.02
        yaw = -60 * DEG * min(1.0, max(0.0, (t - 0.8) / 0.5))  # turning right since 0.8 s
        tick = Tick(t_ms=int(round(t * 1000)), head=(0.0, 0.0, yaw, 0.0))
        ev = [{"t_ms": 1000, "reflex": "orient", "stimulus": "x", "parts": ["yaw_right"]}]
        audits += w.feed(tick, ev if t >= 1.0 else [])
    audit = next(a for a in audits if a.get("kind") == "reflex")
    got = audit["named"][0]["deg"]
    assert got < 60 * 0.6 - 3  # the part already under way before the reflex is not counted


def test_a_stop_hovering_at_the_line_is_one_stop() -> None:
    from duckmind.simweb.neuro_witness import _stop

    points = []
    x = 0.0
    for i in range(200):
        t = i * 0.02
        # it stops at 1 s, then its trunk hovers at 3.1–4 cm/s (sway at the line): one stop
        v = 0.15 if t < 1.0 else 0.02 if t < 1.5 else (0.031 if (i % 3) else 0.04)
        x += v * 0.02
        points.append((t, x, 0.0, 0.117, 0.0))
    still_at, held = _stop(points, 0.9, 4.0)
    assert still_at is not None and held >= 2.0


def test_a_fall_in_the_window_is_said_and_the_head_not_graded() -> None:
    w = Witness()
    audits: list[dict] = []
    for i in range(160):
        t = i * 0.02
        tick = Tick(
            t_ms=int(round(t * 1000)),
            head=(0.0, 0.0, 1.0 * (t > 1.2), 0.0),
            fallen=t > 1.3,
            tilt_deg=80.0 if t > 1.3 else 2.0,
        )
        ev = [{"t_ms": 1000, "reflex": "startle", "stimulus": "x", "parts": ["up"]}]
        audits += w.feed(tick, ev if t >= 1.0 else [])
    audit = next(a for a in audits if a.get("kind") == "reflex")
    assert audit["verdict"] is None and audit["zh"].startswith("身體：跌倒了")


def test_a_hold_is_timed_from_the_body_and_ends_at_the_first_move() -> None:
    w = Witness()
    out: list[dict] = []
    for i in range(300):
        t = i * 0.02
        yaw = 0.0 if t < 3.0 else 20 * DEG  # still for 2 s, then the head moves
        beh = "freeze" if 1.0 <= t < 5.0 else "wander"
        out += w.feed(Tick(t_ms=int(round(t * 1000)), head=(0.0, 0.0, yaw, 0.0), behaviour=beh))
    holds = [a for a in out if a["kind"] == "hold"]
    assert len(holds) == 1 and holds[0]["zh"].startswith("愣住 2.0 秒")
    assert holds[0]["moved"] and "然後頭動了" in holds[0]["zh"]


def _hold_rows(pitch_of: Any, acts: Any, seconds: float = 8.0) -> list[dict]:
    w = Witness()
    out: list[dict] = []
    for i in range(int(seconds * 50)):
        t = i * 0.02
        tick = Tick(t_ms=int(round(t * 1000)), head=(0.0, pitch_of(t), 0.0, 0.0), behaviour=acts(t))
        out += w.feed(tick)
    return [a for a in out if a["kind"] == "hold"]


def test_a_startle_going_on_into_its_freeze_is_one_hold() -> None:
    """Live (怯怯): 「愣住 1.1 秒」 then 「愣住 5.9 秒」 for one stillness — the startle act, then
    the freeze it went on into. Now one row, timed from the startle."""

    def snap(t: float) -> float:  # up 15° in 0.3 s, then held
        return -15 * DEG * min(1.0, max(0.0, (t - 1.0) / 0.3))

    def acts(t: float) -> str:
        return "startle" if 1.0 <= t < 2.0 else "freeze" if 2.0 <= t < 6.0 else "wander"

    holds = _hold_rows(snap, acts)
    assert len(holds) == 1 and holds[0]["zh"] == "愣住 5.0 秒" and not holds[0]["moved"]


def test_a_hold_is_timed_from_when_the_head_settled_into_its_pose() -> None:
    """Live: 「愣住 0.5 秒，然後頭動了」 was the end of the startle's own dip-and-snap (over only
    ~0.6 s after the start), then the head held still for seconds."""

    def dip_and_back(t: float) -> float:
        s = t - 1.0
        if s < 0.0:
            return 0.0
        if s < 0.25:
            return 20 * DEG * s / 0.25
        return 20 * DEG * max(0.0, 1.0 - (s - 0.25) / 0.35)

    def acts(t: float) -> str:
        return "startle" if 1.0 <= t < 4.0 else "wander"

    holds = _hold_rows(dip_and_back, acts)
    assert len(holds) == 1 and holds[0]["zh"] == "愣住 3.0 秒"


def _freeze_audit(release_at: float | None = None) -> dict:
    """A lights-off freeze: the head rises 19° over 0.5 s, holds; robotd's command moves on (the
    release back to the act) at ``release_at`` and the head then swings 30° (the act's peck)."""
    w = Witness()
    audits: list[dict] = []
    for i in range(200):
        t = i * 0.02
        u = min(1.0, max(0.0, (t - 1.1) / 0.5))
        released = release_at is not None and t >= release_at
        swing = 30 * DEG * min(1.0, (t - release_at) / 0.3) if released and release_at else 0.0
        head = (0.0, -19 * DEG * u, swing, 0.0)
        cmd = (0.0, -19 * DEG, 30 * DEG if released else 0.0, 0.0)
        tick = Tick(t_ms=int(round(t * 1000)), head=head, head_cmd=cmd, holding="freeze")
        ev = [{"t_ms": 1000, "reflex": "freeze", "stimulus": "light:off", "parts": []}]
        audits += w.feed(tick, ev if t >= 1.0 else [])
    return next(x for x in audits if x.get("kind") == "reflex")


def test_a_freeze_claims_stillness_and_the_page_says_the_move_into_it() -> None:
    """Live (怯怯, lights off while preening): the head rose 19° into the freeze pose, then held
    within 1° for 12 s, under 「身體做不到：頭沒定住（晃了 14°）」 — the rise was judged as the
    stillness. Headless, a 0.9-s freeze was judged over the 2.5-s window: the peck after it."""
    low = (0.0, 0.2, -0.26, 0.0)  # preening: the nose down, the head turned right
    a = R._freeze(1000, "light:off", 0.5, None, PARAMS, R._Draw(4), head=low)
    assert a.parts == () and R._zh("freeze", "light:off") == "定住不動"
    held = _freeze_audit()
    assert held["verdict"] == "done" and held["freeze_ok"] == "done"
    assert held["zh"] == "身體：頭先抬高 19°、頭定住不動"
    # released at 3.0 s, then the act's peck swings the head: judged still over its own hold
    released = _freeze_audit(release_at=3.0)
    assert released["freeze_ok"] == "done" and 1.0 <= released["freeze_held_s"] < 1.6
    # a freeze that moves on ~0.4 s after it settled was not seen as 「定住」
    short = _freeze_audit(release_at=2.2)
    assert short["freeze_ok"] == "partial" and "頭只定住一下" in short["zh"]
    assert R.FREEZE_MIN_MS >= 1000.0  # the freeze's own hold is long enough to be seen


def test_a_fall_is_said_once_and_the_righting_rows_say_when_it_got_up() -> None:
    """Live (阿勇, one push): 「跌倒了」 three times (proprio, duck-body and the page's own list)
    and 「自己爬起來了」 twice, the righting rows' body line 「跌倒了」 even under 「自己爬起來了」."""
    w = Witness()
    out: list[dict] = []
    for i in range(600):  # (the righting rows' windows: 8 s)
        t = i * 0.02
        down = 1.0 <= t < 2.0
        tick = Tick(
            t_ms=int(round(t * 1000)),
            head=(0.0, 0.0, 0.0, 0.0),
            fallen=down,
            tilt_deg=80.0 if down else 4.0,
        )
        body = [{"kind": "proprio.fell"}, {"kind": "body.fell"}] if abs(t - 1.0) < 0.01 else []
        if abs(t - 1.2) < 0.01:
            body = [{"kind": "proprio.fell"}]
        ev = [{"t_ms": 1000, "reflex": "righting", "stimulus": "body:fall"}] if t >= 1.0 else []
        if t >= 2.0:
            ev.append({"t_ms": 2000, "reflex": "righting", "stimulus": "body:righted"})
        out += w.feed(tick, ev, body)
    falls = [a for a in out if a["kind"] == "body" and a["what"] == "fell"]
    assert len(falls) == 1
    rows = {a["t_ms"]: a for a in out if a["kind"] == "reflex"}
    assert rows[1000]["verdict"] == "done" and rows[1000]["zh"] == "身體：站起來了（1.0 秒）"
    assert rows[2000]["zh"] == "身體：站起來了"
    assert "proprio.fell" not in N.BODY_EVENT_ZH


def test_a_moment_is_said_only_when_the_head_did_it() -> None:
    def run(yaw_deg: float) -> list[dict]:
        w = Witness()
        out: list[dict] = []
        for i in range(200):
            t = i * 0.02
            yaw = yaw_deg * DEG * min(1.0, max(0.0, (t - 1.1) / 0.3))
            mo = "glance" if t >= 1.0 else None
            tick = Tick(
                t_ms=int(round(t * 1000)),
                head=(0.0, 0.0, yaw, 0.0),
                moment=mo,
                moment_seq=1 if mo else 0,
            )
            out += w.feed(tick)
        return [a for a in out if a["kind"] == "moment"]

    assert run(25)[0]["ok"] is True
    assert run(4)[0]["ok"] is False


def test_a_stumble_is_said_only_when_the_trunk_tips_outside_a_posture_change() -> None:
    def run(tilt_jump: float, policy_change_at: float | None) -> list[dict]:
        w = Witness()
        out: list[dict] = []
        policy = "walk"
        for i in range(200):
            t = i * 0.02
            if policy_change_at is not None and t >= policy_change_at:
                policy = "sit"
            tilt = 2.0 + (tilt_jump if 2.05 <= t <= 2.3 else 0.0)
            tick = Tick(t_ms=int(round(t * 1000)), head=(0, 0, 0, 0), policy=policy, tilt_deg=tilt)
            events = [{"kind": "proprio.stumbled"}] if abs(t - 2.0) < 0.01 else []
            out += w.feed(tick, (), events)
        return [a for a in out if a["kind"] == "body"]

    assert [a["zh"] for a in run(9.0, None)] == ["絆了一下"]
    assert run(2.5, None) == []  # 2.5° is the gait's own rocking
    assert run(9.0, 1.5) == []  # inside a sit-down


def test_a_lift_is_said_only_when_the_trunk_really_rose() -> None:
    def run(z_up: float) -> list[dict]:
        w = Witness()
        out: list[dict] = []
        for i in range(200):
            t = i * 0.02
            z = 0.061 + (z_up if t > 1.2 else 0.0)
            tick = Tick(t_ms=int(round(t * 1000)), head=(0, 0, 0, 0), trunk=(0, 0, z, 0), trunk_t=t)
            events = [{"kind": "proprio.lifted"}] if abs(t - 1.0) < 0.01 else []
            out += w.feed(tick, (), events)
        return [a for a in out if a["kind"] == "body"]

    assert [a["zh"] for a in run(0.3)] == ["被抱起來了"]
    assert run(0.006) == []  # a push on a seated duck: its legs went light, it was not lifted


# ── the body link: picked up while stepping ───────────────────────────────────


def _state(t: float, leg_speed: float, gyro: float = 0.0, policy: str = "walk") -> RobotState:
    legs = [i for i, n in enumerate(JOINT_NAMES) if any(k in n for k in ("hip", "knee", "ankle"))]
    vel = [0.0] * len(JOINT_NAMES)
    for i in legs:
        vel[i] = leg_speed / math.sqrt(len(legs))
    joints = [0.0] * len(JOINT_NAMES)
    joints[5] = joints[6] = HOME
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0.4, 0, 0], "applied": [0.4, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "velocities": vel,
            "currents_ma": [10.0] * len(JOINT_NAMES),
            "imu": {"gyro": [gyro, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
    )


def test_legs_kicking_in_the_air_stop_the_walk_hold_the_head_and_sit_at_once() -> None:
    link = BodyLink()
    walk = BodyRequest(vx=0.45, vyaw=0.0, head=(0.0, -0.2, 0.3, 0.0), posture="stand")
    out = None
    for k in range(20):
        out = link.tick(_state(k * 0.02, 6.0), walk)  # stepping on a floor
    assert out is not None and "airborne" not in out.gate
    for k in range(20, 26):
        out = link.tick(_state(k * 0.02, 14.0), walk)  # the legs swing free
    assert "airborne" in out.gate and out.twist == (0.0, 0.0)
    assert out.head == (0.0, 0.0, 0.0, 0.0) and out.posture_want == "sit"
    assert link.airs == 1


def test_a_trip_that_throws_the_trunk_about_is_not_taken_for_a_lift() -> None:
    link = BodyLink()
    walk = BodyRequest(vx=0.45, vyaw=0.0, posture="stand")
    for k in range(30):
        out = link.tick(_state(k * 0.02, 14.0, gyro=3.0), walk)
    assert "airborne" not in out.gate and link.airs == 0


# ── the page: the body's words ────────────────────────────────────────────────


def test_the_speed_words_are_the_brains_wish_and_the_bodys_measure() -> None:
    assert (
        N.locomotion_zh(0.37, 0.0, true_mps=0.18) == "腦想走 0.37（身體實際 0.18 公尺/秒）・直直走"
    )
    assert "身體實際" not in N.locomotion_zh(0.37, 0.0)


def test_the_page_has_the_stimuli_under_the_view() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    strip = page[
        page.index('id="neuroQuick"') : page.index("</div>", page.index('id="neuroQuick"'))
    ]
    assert (
        "data-scenarios"
        in page[page.index('id="neuroQuick"') - 200 : page.index('id="neuroQuick"') + 200]
    )
    for key in ("clap", "call_name", "pat_head", "poke", "push", "pick_up", "set_down", "lights"):
        assert f'data-stim="{key}"' in strip
    js = (STATIC / "neuro.js").read_text(encoding="utf-8")
    assert "#neuroQuick" in js and "[data-lights]" in js
    # the stage strip sits between the 3D view and the readout, above the fold
    assert page.index('id="neuroQuick"') < page.index('id="duck3dReadout"')


def test_the_act_names_claim_only_what_this_body_does() -> None:
    from duckmind.neuro.contract import BEHAVIOURS

    assert BEHAVIOURS["stretch_yawn"].zh == "抬頭打哈欠"
    assert BEHAVIOURS["happy_sway"].zh == "開心晃頭"
    assert (
        BEHAVIOURS["zoomies"].zh == "興奮地走走轉轉" and "繞" not in BEHAVIOURS["zoomies"].body_zh
    )
    assert BEHAVIOURS["flee"].zh == "躲開"
    # (dock:build 2026-09-28: it now backs into its dock by itself — the name says so)
    assert BEHAVIOURS["return_dock"].zh == "自己回去充電"
    assert R._zh("stabilise", "body:tilt") == "身體晃了一下，停下來站穩"


def test_a_startle_during_a_rise_names_nothing() -> None:
    a = _startle((0.0, 0.0, 0.0, 0.0), posture="transition")
    assert a.parts == () and "頭先不動" in a.zh


def test_a_look_through_the_camera_uses_the_measured_head() -> None:
    lit = CameraSense(usable=True, light="bright")
    body = _body(head=(0.0, 0.0, 0.0, 0.0))
    frame = SensorFrame(t_ms=0, body=body, camera=lit)
    assert R.head_now(frame) == (0.0, 0.0, 0.0, 0.0)
    moved = replace(frame, body=_body(head=(0.0, -0.1, 0.4, 0.0)))
    assert R.head_now(moved)[2] == 0.4 and abs(R.head_now(moved)[1] + 0.1) < 1e-9
    assert HeadPose().np == 0.0
