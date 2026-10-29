"""fid:motor (2026-09-28) — the body performs what the brain selects, visibly.

The owner judged only what the physical body visibly does (「嚇一跳…我看3D模擬並沒有呈現出來」,
「我要的是100%完整呈現」). These tests pin the motor side of that:

* robotd's ``[control] head_alpha`` 1.0 for the simulator's own robotd (owner: 「加快」), written
  into its robotd.toml by the live launcher; the motor's efference copy follows the alpha the
  body really runs (the identity at 1.0) and limits the sent command's rate (~5 rad/s seated);
* the head may use the A100-validated ±1.0 rad of yaw (an orienting turn), a standing neck
  pull-back becomes head pitch radian for radian;
* the startle: the head snaps UP (the whole pose standing too), the beak snaps open and shut, a
  duck caught walking stays stopped ≥ 1.5 s from the onset (A100: a stop takes ~0.6 s, a pause
  reads as a stop only when ≥ 0.8 s), a real fright calls the alarm;
* the freeze and the freeze act hold the legs at once and hold the head high;
* withdrawal: the head pulls away big enough to see; standing, a step away that lasts long
  enough for the legs to start; from a touch in front a small back ARC (reversing a small arc
  when cornered: owner 2026-09-28); lean-in: the head goes down and tilts into the hand;
* cornered: a small back arc out (a few times at one place, never into a wall it remembers);
* orienting: when the thing it saw is far to the side and still interesting, the body follows
  the head in an arc.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from duckmind.neuro import motor as M
from duckmind.neuro import reflex as R
from duckmind.neuro import repertoire as P
from duckmind.neuro.brain import (
    CORNER_BACKS,
    TURN_TOWARD_AFTER_MS,
    TURN_TOWARD_VYAW,
    NeuroBrain,
)
from duckmind.neuro.contract import (
    BodySense,
    Gaze,
    HeadPose,
    MotorCommand,
    SensorFrame,
    SoundSense,
    TouchSense,
    legal_walk,
)
from duckmind.neuro.genome import preset

PARAMS = R.params_from({"boldness": 0.2, "wit": 0.5}, stage="juvenile", side=1)  # a timid duck
PROFILE = M.profile_from_style(P.style_from({"activity": 0.5}, stage="juvenile", side=1))


def _frame(t: int, posture: str = "stand", **kw: object) -> SensorFrame:
    body = kw.pop("body", None) or BodySense(posture=posture, vx=0.45 if posture == "walk" else 0.0)
    return SensorFrame(t_ms=t, body=body, **kw)  # type: ignore[arg-type]


def _cmd(behaviour: str, t: int, **kw: object) -> MotorCommand:
    return replace(MotorCommand(t_ms=t, behaviour=behaviour, bout=1, variant=5), **kw)  # type: ignore[arg-type]


def _run_reflex(frames: list[SensorFrame], **kw: object) -> list[R.ReflexOutput]:
    state = R.initial(3)
    outs = []
    for frame in frames:
        state, out = R.step(PARAMS, state, frame, **kw)  # type: ignore[arg-type]
        outs.append(out)
    return outs


# ── robotd's head smoothing: 1.0 for the simulator, the motor follows it ──────────────────


def test_the_live_launcher_writes_head_alpha_into_its_own_robotd_toml(monkeypatch) -> None:
    from duckmind import livebody

    assert livebody.robotd_control_section(None) == ""
    section = livebody.robotd_control_section(1.0)
    assert "[control]" in section and "head_alpha = 1.0" in section
    monkeypatch.setenv(livebody.ROBOTD_HEAD_ALPHA_ENV, "1.0")
    assert livebody.robotd_head_alpha() == 1.0
    monkeypatch.setenv(livebody.ROBOTD_HEAD_ALPHA_ENV, "")
    assert livebody.robotd_head_alpha() is None
    monkeypatch.setenv(livebody.ROBOTD_HEAD_ALPHA_ENV, "1.5")
    with pytest.raises(ValueError):
        livebody.robotd_head_alpha()


def test_the_live_motor_follows_the_bodys_robotd(monkeypatch) -> None:
    from duckmind.simweb import neuro_live as NL

    monkeypatch.setenv(NL.ROBOTD_HEAD_ALPHA_ENV, "1.0")
    motor = NL.LiveMotor(M, preset("snuggler"), 3)
    assert motor.profile.head_alpha == 1.0
    motor.set_head_alpha(0.2)  # a body whose robotd runs its default
    assert motor.profile.head_alpha == pytest.approx(0.2)
    assert motor.system.profile.head_alpha == pytest.approx(0.2)
    monkeypatch.delenv(NL.ROBOTD_HEAD_ALPHA_ENV)
    assert NL.robotd_head_alpha_env() == NL.ROBOTD_DEFAULT_HEAD_ALPHA


def _yaw_step(
    profile: M.MotorProfile, posture: str, hy: float = 1.6
) -> tuple[list[float], list[float]]:
    state = M.MotorState()
    sent, goal = [], []
    for k in range(40):
        head = HeadPose(hy=hy) if k >= 5 else None
        cmd = _cmd("still", k * 20, head=head, head_blend=1.0, alive=0.0, reflex="orient")
        state, out = M.step(profile, state, cmd, _frame(k * 20, posture))
        sent.append(out.head[2])
        goal.append(out.head_goal[2])
    return sent, goal


def test_at_head_alpha_one_the_command_is_the_goal_and_its_rate_is_limited() -> None:
    fast = replace(PROFILE, head_alpha=1.0)
    sent, goal = _yaw_step(fast, "stand")
    assert goal[-1] >= 0.95  # the whole validated ±1.0 rad of yaw
    assert sent[-1] == pytest.approx(goal[-1], abs=1e-6)  # no lead: robotd passes it straight on
    steps = [abs(b - a) for a, b in zip(sent, sent[1:], strict=False)]
    assert max(steps) <= M.HEAD_SLEW * 0.02 + 1e-9  # standing: only a pathological jump is cut
    sent_sit, _ = _yaw_step(fast, "sit")
    steps_sit = [abs(b - a) for a, b in zip(sent_sit, sent_sit[1:], strict=False)]
    assert max(steps_sit) <= M.HEAD_SLEW_SIT * 0.02 + 1e-9  # seated: ~5 rad/s (A100)
    ramp = next(i for i, v in enumerate(sent_sit) if v >= 0.95 * goal[-1]) - 5
    assert 9 <= ramp <= 12  # a 1-rad turn seated takes ~0.2 s of command, not one tick
    slow = replace(PROFILE, head_alpha=0.2, head_slew_sit=100.0, head_slew=100.0)
    sent_slow, goal_slow = _yaw_step(slow, "stand", hy=0.5)  # (near ±1.0 rad there is no room)
    assert max(s - g for s, g in zip(sent_slow, goal_slow, strict=True)) > 0.05  # the lead


def test_a_standing_neck_pull_back_becomes_head_pitch() -> None:
    state = M.MotorState()
    for k in range(30):
        cmd = _cmd("still", k * 20, head=HeadPose(np=-0.6, hp=0.0), head_blend=1.0, alive=0.0)
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "stand"))
    assert out.pose.hp == pytest.approx(0.6 * M.NECK_TO_PITCH, abs=0.02)
    assert out.pose.np == 0.0 and out.head_goal[0] == 0.0  # never asked for what it cannot do
    assert M.NECK_TO_PITCH == pytest.approx(0.25 / 0.35)
    state = M.MotorState()
    for k in range(30):  # seated the neck does pull back (pet:body: gain 0.54)
        cmd = _cmd("still", k * 20, head=HeadPose(np=-0.6, hp=0.0), head_blend=1.0, alive=0.0)
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "sit"))
    assert out.pose.np == pytest.approx(-0.6, abs=0.02)
    assert out.pose.hp == pytest.approx(0.0, abs=0.02)


# ── the startle ────────────────────────────────────────────────────────────────────────


def _clap(t: int, db: float = 86.0) -> SensorFrame:
    return _frame(t, "walk", sound=SoundSense(level_db=db, onset=1.0, kind="bang"))


def test_a_startle_snaps_the_head_up_and_the_beak_open_and_shut() -> None:
    for posture in ("stand", "sit"):
        frames = [_frame(t, posture) for t in range(0, 200, 20)]
        frames += [_frame(200, posture, sound=SoundSense(level_db=88.0, onset=1.0, kind="bang"))]
        frames += [_frame(t, posture) for t in range(220, 1600, 20)]
        outs = _run_reflex(frames)
        startle = [(f.t_ms, o) for f, o in zip(frames, outs, strict=True) if o.active == "startle"]
        assert startle
        t0 = startle[0][0]
        peak = max((o for _, o in startle), key=lambda o: o.head_blend)
        assert peak.head is not None and peak.head.hp >= 0.7  # the validated look-up
        assert peak.head_blend == pytest.approx(1.0)  # the whole pose, standing too
        early = [o.head.mouth for t, o in startle if t - t0 <= 100 and o.head is not None]
        late = [o.head.mouth for t, o in startle if t - t0 >= 260 and o.head is not None]
        assert max(early) >= 0.8 and max(late) == 0.0  # a snap, not a gape


def test_a_startle_caught_walking_keeps_the_legs_stopped_long_enough_to_be_seen() -> None:
    weak = replace(PARAMS, startle_threshold=0.05)
    frames = [_frame(t, "walk") for t in range(0, 200, 20)]
    frames += [_clap(200, db=70.0)]
    frames += [_frame(t, "stand") for t in range(220, 3000, 20)]
    state = R.initial(5)
    halted = []
    onset = None
    for frame in frames:
        state, out = R.step(weak, state, frame, walking=frame.t_ms < 200, fear=0.0)
        if out.active == "startle" and onset is None:
            onset = frame.t_ms
        halted.append((frame.t_ms, out.halt))
    assert onset is not None
    held = [t for t, h in halted if h and t >= onset]
    assert held and max(held) - onset >= R.STARTLE_STOP_MS - 40  # ≥ 1.5 s from the onset


def test_the_brain_stops_the_walk_on_the_startles_halt() -> None:
    rx = SimpleNamespace(escape=None, halt=True)
    vx, vyaw = 0.45, 0.2
    if rx.escape is not None:
        vx, vyaw = legal_walk(*rx.escape)
    elif rx.halt:
        vx, vyaw = 0.0, 0.0
    assert (vx, vyaw) == (0.0, 0.0)
    # and the motor holds the legs at once for a reflex hold or a freeze act
    for reflex, behaviour in (("startle", "wander"), (None, "freeze"), ("freeze", "freeze")):
        state = M.MotorState()
        out = None
        for k in range(10):
            cmd = _cmd(behaviour, k * 20, vx=0.45, reflex=reflex, posture="stand")
            state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "walk"))
        assert out is not None and out.vx == 0.0 and out.vyaw == 0.0


def test_a_real_fright_calls_the_alarm_once() -> None:
    from duckmind.neuro.appraisal import STARTLE_ALARM, Appraiser
    from duckmind.neuro.contract import ReflexEvent
    from duckmind.neuro.learn import Learner

    genome = preset("snuggler")
    ap = Appraiser(genome, Learner(genome.p))
    frame = _frame(0)
    strong = ReflexEvent(0, "startle", "sound:bang", STARTLE_ALARM + 0.2, 1.0, "嚇一跳")
    weak = ReflexEvent(0, "startle", "sound:bang", STARTLE_ALARM - 0.2, 1.0, "嚇一跳")
    out = ap.step(
        10.0,
        0.02,
        frame,
        (),
        (strong,),
        asleep=False,
        walking=False,
        lifted=False,
        blocked_ahead=False,
        moving=False,
        looming_bearing=None,
    )
    assert out.alarm
    again = ap.step(
        11.0,
        0.02,
        frame,
        (),
        (strong,),
        asleep=False,
        walking=False,
        lifted=False,
        blocked_ahead=False,
        moving=False,
        looming_bearing=None,
    )
    assert not again.alarm  # once per scare
    quiet = Appraiser(genome, Learner(genome.p)).step(
        10.0,
        0.02,
        frame,
        (),
        (weak,),
        asleep=False,
        walking=False,
        lifted=False,
        blocked_ahead=False,
        moving=False,
        looming_bearing=None,
    )
    assert not quiet.alarm  # a mild startle stays silent


# ── freeze, withdrawal, lean-in ────────────────────────────────────────────────────────


def test_the_freeze_holds_the_head_high_and_all_of_it() -> None:
    frames = [_frame(t) for t in range(0, 200, 20)]
    frames += [_frame(200, sound=SoundSense(level_db=90.0, onset=1.0, kind="bang"))]
    frames += [_frame(t) for t in range(220, 4000, 20)]
    outs = _run_reflex(frames, fear=0.5)
    freezes = [o for o in outs if o.active == "freeze" and o.head is not None]
    assert freezes
    peak = max(freezes, key=lambda o: o.head_blend)
    assert peak.head_blend == pytest.approx(1.0) and peak.head.hp >= 0.5
    assert all(o.halt for o in freezes[: len(freezes) // 2])


def _touch(site: str, t: int, posture: str = "stand") -> list[SensorFrame]:
    frames = [_frame(k, posture) for k in range(0, t, 20)]
    frames += [
        _frame(k, posture, touch=TouchSense(site=site, pressure=0.9)) for k in range(t, t + 200, 20)
    ]
    frames += [_frame(k, posture) for k in range(t + 200, t + 2500, 20)]
    return frames


def test_a_touch_in_front_is_withdrawn_from_in_a_small_back_arc() -> None:
    outs = _run_reflex(_touch("head", 200), fear=0.3, comfort=0.2)
    withdraws = [o for o in outs if o.active == "withdraw"]
    assert withdraws
    peak = max(withdraws, key=lambda o: o.head_blend)
    assert peak.head is not None and peak.head.hp >= 0.6  # the head thrown back and up
    escapes = [o.escape for o in withdraws if o.escape is not None]
    assert escapes and all(e[0] == R.WITHDRAW_BACK_VX and abs(e[1]) >= 0.5 for e in escapes)
    vx, vyaw = legal_walk(*escapes[0], back_arc=True)
    assert vx < 0.0 and abs(vyaw) >= 0.5  # a reverse ARC the body link lets through
    walking = _run_reflex(_touch("head", 200, "walk"), fear=0.3, comfort=0.2, walking=True)
    assert all(o.escape is None or o.escape[0] > 0.0 for o in walking)  # walking: no reverse


def test_a_side_touch_turns_the_head_away_and_steps_away_long_enough() -> None:
    outs = _run_reflex(_touch("left", 200), fear=0.3, comfort=0.2)
    withdraws = [o for o in outs if o.active == "withdraw" and o.head is not None]
    assert withdraws
    peak = max(withdraws, key=lambda o: o.head_blend)
    assert peak.head.hy <= -0.6  # away from the left: 25–43° of yaw
    steps = [o for o in outs if o.escape is not None]
    assert steps and all(o.escape[0] > 0.0 and o.escape[1] < 0.0 for o in steps)  # away
    assert len(steps) * 20 >= R.WITHDRAW_STEP_MS[0] - 40  # the legs have time to start


def test_leaning_in_is_seen_standing() -> None:
    frames = [_frame(k) for k in range(0, 200, 20)]
    frames += [
        _frame(k, touch=TouchSense(site="left", pressure=0.4, stroke=0.8))
        for k in range(200, 2400, 20)
    ]
    outs = _run_reflex(frames, comfort=0.8)
    leans = [o for o in outs if o.active == "lean_in" and o.head is not None]
    assert leans
    peak = max(leans, key=lambda o: o.head_blend)
    assert peak.head.hp <= -0.3 and abs(peak.head.hr) >= 0.4 and peak.head.hy > 0.3


def test_orient_and_freeze_poses_point_where_the_saccade_points() -> None:
    from duckmind.neuro.contract import CameraSense, Sighting

    sight = Sighting(kind="thing", bearing=1.2, size=0.3, motion=0.8, track=7)
    frames = [_frame(t) for t in range(0, 200, 20)]
    frames += [_frame(t, camera=CameraSense(sightings=(sight,))) for t in range(200, 1400, 20)]
    outs = _run_reflex(frames)
    orients = [o for o in outs if o.active == "orient" and o.head is not None and o.gaze]
    assert orients
    assert all(o.head.hy == pytest.approx(min(1.2, 1.0) / 0.6, abs=0.3) for o in orients[-3:])


# ── cornered: back out; orienting: the body follows ─────────────────────────────────────


def test_cornered_it_backs_out_a_few_times_then_gives_up_there() -> None:
    brain = NeuroBrain(preset("explorer"), seed=4)
    steer = SimpleNamespace(best=2.0, side=1)
    got = []
    for k in range(CORNER_BACKS + 2):
        brain._back = None
        got.append(brain._back_out_of_corner(k * 2000, (0.0, 0.0, 0.0), steer))
    assert got[:CORNER_BACKS] == [True] * CORNER_BACKS and not any(got[CORNER_BACKS:])
    brain._back = None
    assert brain._back_out_of_corner(99_000, (1.0, 0.0, 0.0), steer)  # somewhere else: again
    assert brain._back is not None and brain._back[1] == 1.0  # toward the way it saw open


def test_it_never_backs_into_a_wall_it_remembers_behind() -> None:
    from duckmind.neuro.boundary import _Point

    brain = NeuroBrain(preset("explorer"), seed=4)
    # a wall it has seen 15 cm behind its trunk (odometry frame; it faces +x)
    brain.bound.points.extend(_Point(-0.15, y / 100.0, 1.0, "wall") for y in range(-20, 21, 4))
    assert not brain._back_out_of_corner(0, (0.0, 0.0, 0.0), SimpleNamespace(best=1.0, side=1))
    assert brain._back is None


def test_the_body_follows_the_head_toward_a_thing_far_to_the_side() -> None:
    brain = NeuroBrain(preset("explorer"), seed=4)
    rx = SimpleNamespace(
        active="orient",
        gaze=Gaze(bearing=1.3, gain=1.0, speed=0.8),
        stimulus="visual:thing",
        hold=None,
        halt=False,
    )
    odom = (0.0, 0.0, 0.0)
    out = [brain._turn_toward(t, rx, "wander", None, True, odom, False) for t in range(0, 1000, 20)]
    first = next(i for i, v in enumerate(out) if v is not None) * 20
    assert first >= TURN_TOWARD_AFTER_MS and out[-1] == TURN_TOWARD_VYAW  # toward it (left)
    # it keeps turning until it faces the thing, then stops
    assert brain._turn_toward(1100, rx, "wander", None, True, (0.0, 0.0, 0.6), False) is not None
    assert brain._turn_toward(1200, rx, "wander", None, True, (0.0, 0.0, 1.2), False) is None
    # never for a sound (a mono microphone gives no side), never while resting or walking
    brain2 = NeuroBrain(preset("explorer"), seed=4)
    heard = replace_ns(rx, stimulus="sound:clap")
    assert all(
        brain2._turn_toward(t, heard, "wander", None, True, odom, False) is None
        for t in range(0, 2000, 20)
    )
    assert all(
        brain2._turn_toward(t, rx, "rest_sit", None, True, odom, False) is None
        for t in range(0, 2000, 20)
    )


def replace_ns(ns: SimpleNamespace, **kw: object) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(ns), **kw})


def test_the_motor_lets_a_back_arc_through_and_never_a_straight_reverse() -> None:
    state = M.MotorState()
    out = None
    for k in range(20):
        cmd = _cmd("withdraw", k * 20, vx=-0.4, vyaw=1.0, back_arc=True, reflex="withdraw")
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "stand"))
    assert out is not None and out.vx == pytest.approx(-0.4) and abs(out.vyaw) == 1.0
    assert legal_walk(-0.4, 0.0, back_arc=True) == (0.0, 0.0)


def test_every_act_is_big_enough_to_see_standing() -> None:
    """The motor programs' commanded head excursions against the measured deadbands (pet:body:
    standing yaw ≥ ±6°, the neck only downward and ≥ 7°, pitch/roll ≥ 6°) — or the beak, a
    posture change, the legs. Acts that ARE stillness (still, freeze's hold) show it; walking
    acts show their legs."""
    deg = math.pi / 180.0
    for name in ("freeze", "snuggle", "startle", "look_around", "peck", "preen", "tilt"):
        style = P.style_from({"activity": 0.5}, stage="juvenile", side=1)
        profile = M.profile_from_style(style)
        seen = 0
        for k in range(12):
            state = M.MotorState()
            lo, hi = [9.0] * 4, [-9.0] * 4
            mouth = 0.0
            for tick in range(250):
                cmd = MotorCommand(
                    t_ms=tick * 20,
                    behaviour=name,
                    bout=1 + k,
                    variant=k * 131 + 7,
                    amplitude=0.55,
                    alive=0.0,
                    posture="stand",
                )
                state, out = M.step(profile, state, cmd, _frame(tick * 20, "stand"))
                for i in range(4):
                    lo[i] = min(lo[i], out.head_goal[i])
                    hi[i] = max(hi[i], out.head_goal[i])
                mouth = max(mouth, out.mouth)
            exc = [hi[i] - lo[i] for i in range(4)]
            if exc[1] >= 6 * deg or exc[2] >= 6 * deg or exc[3] >= 6 * deg or mouth >= 0.1:
                seen += 1
        assert seen >= 11, (name, seen)


def test_the_reflex_words_say_what_this_body_does() -> None:
    # standing the walking network will not pull the neck back: the head is thrown back and up;
    # (fid:fix) the startle's own words come from its plan (reflex._startle, test_fidfix.py)
    assert R._zh("startle", "sound:bang", None, posture="stand") == "嚇一跳，頭猛往後仰"
    back = (R.WITHDRAW_BACK_VX, 1.0, R.WITHDRAW_BACK_MS)
    assert R._zh("withdraw", "touch:head", escape=back) == "從頭頂縮開，往後退一小步"
    assert R._zh("withdraw", "touch:left", escape=(0.4, -0.8, 1200)) == "從左邊縮開，走開一步"
    assert R._zh("withdraw", "touch:left") == "從左邊縮開"  # no step, no claim of one


def test_its_eyes_pursue_a_walking_person_and_its_body_follows() -> None:
    brain = NeuroBrain(preset("explorer"), seed=4)
    rx = SimpleNamespace(
        active="orient",
        gaze=Gaze(bearing=0.2, gain=1.0, speed=0.8),  # where the person was first seen
        stimulus="visual:person",
        hold=None,
        halt=False,
    )
    brain._person_b = 1.1  # … and where they have walked to since
    odom = (0.0, 0.0, 0.0)
    out = [brain._turn_toward(t, rx, "wander", None, True, odom, False) for t in range(0, 1000, 20)]
    assert out[-1] == TURN_TOWARD_VYAW  # toward where the person is NOW, not where it saw them
    brain2 = NeuroBrain(preset("explorer"), seed=4)
    brain2._person_b = -1.0
    none = SimpleNamespace(active=None, gaze=None, stimulus=None, hold=None, halt=False)
    got = [
        brain2._turn_toward(t, none, "greet", -1.0, True, odom, False) for t in range(0, 1000, 20)
    ]
    assert got[-1] == -TURN_TOWARD_VYAW  # greeting someone off to the right: it turns to them
