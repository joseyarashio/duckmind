"""neuro:motor — the locomotor rhythm, the gait phase and the motor integration.

The duck's traits are Duck Core's own (the same seed gives the same temperament, nature and
quirks as the engine hatches); walking comes in bursts and pauses whose lengths vary, at speeds
that keep changing inside the owner's rules (0.35–0.55, ≤ 0.40 in arcs and near things, |vyaw| ≤
1, arcs not spins); two temperaments walk measurably differently; robotd's head smoothing is
pre-compensated; every intent is legal and clamped; the posture waits for the program's cue.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import pytest

from duckmind.duckcore.engine import DuckCore
from duckmind.neuro import motor as M
from duckmind.neuro import repertoire as P
from duckmind.neuro.contract import (
    BEHAVIOURS,
    BodySense,
    Gaze,
    HeadPose,
    MotorCommand,
    SensorFrame,
    Target,
    legal_walk,
)
from duckmind.neuro.genome import Genome

# ── Duck Core's diversity design ──────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_the_motor_profile_is_the_genomes_duck_core_duck(seed: int) -> None:
    """ONE diversity mapping: the motor side takes the brain's Genome (Duck Core's own draws)."""
    core = DuckCore(seed=seed)
    genome = Genome.from_seed(seed, gd=120)
    profile = M.profile_from_genome(genome)
    side = 1 if core.s.quirks["tilt_side"] == "左" else -1
    assert profile.style.side == side == profile.reflex.side
    assert profile.style.habit == core.s.quirks["habit"] == profile.reflex.habit
    assert profile.style.favourite_tag == core.s.quirks["favorite_tag"]
    assert profile.style.stage == genome.stage
    for axis in ("boldness", "activity", "sociability", "exploration", "wit", "tempo"):
        assert getattr(profile.style, axis) == pytest.approx(genome.axes[axis])
    assert profile.reflex.hab_rate == pytest.approx(genome.p.hab_rate)
    assert profile.seed == seed


def test_stage_follows_the_ducks_own_pace() -> None:
    young = M.profile_from_genome(Genome.from_seed(7, gd=2))
    assert young.style.stage == "hatchling" and young.style.young == 1.0
    grown = M.profile_from_genome(Genome.from_seed(7, gd=400, branch="explorer"))
    assert grown.style.stage == "adult" and grown.style.w("explorer") > 0.8
    assert young.loco.tire > grown.loco.tire  # a hatchling walks in shorter spurts


# ── the locomotor rhythm ──────────────────────────────────────────────────────


def _walk(
    params: M.LocoParams, seconds: float, *, go: float = 0.8, seed: int = 1, **kw: object
) -> list[M.LocoOut]:
    state = M.LocoState(rng=seed)
    outs = []
    for k in range(int(seconds * 50)):
        inp = M.LocoInput(go=go, **kw)  # type: ignore[arg-type]
        state, out = M.loco_step(params, state, inp, k * 20)
        outs.append(out)
    return outs


def _runs(outs: list[M.LocoOut]) -> tuple[list[float], list[float]]:
    bursts, pauses = [], []
    start, walking = 0, outs[0].walking
    for i, out in enumerate(outs[1:], 1):
        if out.walking != walking:
            (bursts if walking else pauses).append((i - start) * 0.02)
            start, walking = i, out.walking
    return bursts, pauses


def test_no_drive_no_walking() -> None:
    assert not any(out.walking for out in _walk(M.LocoParams(), 60.0, go=0.0))


def test_walking_comes_in_bursts_and_pauses_that_vary() -> None:
    outs = _walk(M.LocoParams(), 600.0, go=0.8)
    bursts, pauses = _runs(outs)
    assert len(bursts) >= 20 and len(pauses) >= 20
    assert min(bursts) >= 1.5 and min(pauses) >= 0.45  # the body's own minimums
    cv = statistics.pstdev(bursts) / statistics.mean(bursts)
    assert cv > 0.25  # not a timer
    share = sum(out.walking for out in outs) / len(outs)
    assert 0.5 < share < 0.95  # it walks a lot, and it stops


def test_speed_and_turn_keep_changing_inside_the_rules() -> None:
    outs = [o for o in _walk(M.LocoParams(), 300.0, go=0.9, yaw_rate=0.0) if o.walking]
    speeds = [o.vx for o in outs]
    assert len({round(v, 3) for v in speeds}) > 30  # never one constant speed
    assert statistics.pstdev(speeds) > 0.02
    for out in outs:
        assert 0.35 <= out.vx <= 0.55
        assert abs(out.vyaw) <= 1.0
        if abs(out.vyaw) > 0.3:
            assert out.vx <= 0.40 + 1e-9  # arcs at ≤ 0.40
        assert (out.vx, out.vyaw) == legal_walk(out.vx, out.vyaw, approach=out.vx <= 0.40)
    turns = [o.vyaw for o in outs]
    assert max(turns) > 0.2 and min(turns) < -0.2  # it meanders both ways


def test_halt_and_approach() -> None:
    outs = _walk(M.LocoParams(), 30.0, go=1.0, halt=True)
    assert all(o.vx == 0.0 and o.vyaw == 0.0 for o in outs)
    outs = _walk(M.LocoParams(), 60.0, go=1.0, approach=True, bearing=0.2, range_m=1.0)
    assert all(o.vx <= 0.40 for o in outs if o.walking)
    steer = [o.vyaw for o in outs if o.walking]
    assert statistics.mean(steer) > 0.15  # toward the goal on its left


def test_course_keeping_cancels_a_measured_drift() -> None:
    params = replace(
        M.LocoParams(), meander_amp=0.0, turn_sigma=0.0, side_bias=0.0, course_gain=0.6
    )
    outs = _walk(params, 120.0, go=1.0, yaw_rate=-0.17)  # drifting right ~10°/s
    late = [o.vyaw for o in outs[-1500:] if o.walking]
    assert late and statistics.mean(late) > 0.1


def test_two_temperaments_walk_differently_and_each_stays_itself() -> None:
    busy = M.loco_params_from(
        P.style_from({"activity": 1.0, "exploration": 1.0}, stage="adult", branch="explorer")
    )
    calm = M.loco_params_from(
        P.style_from({"activity": 0.0, "exploration": 0.0}, stage="adult", branch="snuggler")
    )
    shares = {}
    for name, params in (("busy", busy), ("calm", calm)):
        values = []
        for seed in range(6):
            outs = _walk(params, 240.0, go=0.6, seed=seed)
            values.append(sum(o.walking for o in outs) / len(outs))
        shares[name] = values
    assert min(shares["busy"]) > max(shares["calm"])
    assert busy.vx_base > calm.vx_base and busy.meander_amp > calm.meander_amp
    hatch = M.loco_params_from(P.style_from({"activity": 0.5}, stage="hatchling"))
    adult = M.loco_params_from(P.style_from({"activity": 0.5}, stage="adult"))
    assert hatch.tire > adult.tire and hatch.turn_sigma > adult.turn_sigma


# ── the gait phase ────────────────────────────────────────────────────────────


def test_gait_phase_follows_a_stepping_leg() -> None:
    g = M.GaitState()
    phases = []
    for k in range(250):
        t = k * 0.02
        joint = -0.45 + 0.12 * math.sin(2 * math.pi * 2.4 * t)
        g, phase = M.gait_step(g, joint, 0.02)
        phases.append(phase)
    valid = [p for p in phases[100:] if p is not None]
    assert len(valid) > 140
    advance = sum(((b - a) % 1.0) for a, b in zip(valid, valid[1:], strict=False))
    assert advance / (len(valid) * 0.02) == pytest.approx(2.4, rel=0.1)
    g = M.GaitState()
    for _ in range(100):
        g, phase = M.gait_step(g, -0.45, 0.02)
    assert phase is None


# ── the motor integration ─────────────────────────────────────────────────────

PROFILE = M.profile_from_style(
    P.style_from({"activity": 0.5, "exploration": 0.5}, stage="juvenile", side=1)
)


def _cmd(behaviour: str, t: int, **kw: object) -> MotorCommand:
    return replace(MotorCommand(t_ms=t, behaviour=behaviour, bout=1, variant=5), **kw)  # type: ignore[arg-type]


def _frame(t: int, posture: str = "stand", **kw: object) -> SensorFrame:
    return SensorFrame(t_ms=t, body=BodySense(posture=posture, **kw))  # type: ignore[arg-type]


def test_every_behaviour_gives_legal_intents() -> None:
    for name, spec in BEHAVIOURS.items():
        state = M.MotorState()
        walking = spec.locomotes
        for k in range(300):
            t = k * 20
            vx, vyaw = (0.5, 0.6) if walking else (0.0, 0.0)
            cmd = _cmd(
                name,
                t,
                vx=vx,
                vyaw=vyaw,
                posture="stand",
                bob=0.8,
                stabilise=0.7,
                gaze=Gaze(bearing=0.4, gain=0.6, speed=0.5),
                target=Target(kind="thing", bearing=0.3, range_m=0.4),
            )
            body = "walk" if walking else ("sit" if spec.posture == "sit" else "stand")
            state, out = M.step(
                PROFILE,
                state,
                cmd,
                _frame(t, body, vx=0.45 if walking else 0.0, step_phase=(t / 400.0) % 1.0),
            )
            for i, value in enumerate(out.head):
                assert abs(value) <= M.COMMAND_MAX[i] + 1e-9
            assert 0.0 <= out.mouth <= 0.5
            assert out.posture in ("sit", "stand")
            if out.vx or out.vyaw:
                assert out.vx != 0.0  # no spin on the spot
                assert 0.35 <= out.vx <= 0.55 and abs(out.vyaw) <= 1.0
                if abs(out.vyaw) > 0.3:
                    assert out.vx <= 0.40 + 1e-9
            if body == "sit":
                assert out.vx == 0.0 and out.vyaw == 0.0


def test_the_seated_body_never_walks_and_sleep_sits() -> None:
    state = M.MotorState()
    for k in range(50):
        state, out = M.step(
            PROFILE, state, _cmd("wander", k * 20, vx=0.5, posture="stand"), _frame(k * 20, "sit")
        )
        assert out.vx == 0.0
    state, out = M.step(
        PROFILE, M.MotorState(), _cmd("nap", 0, asleep=True, posture="stand"), _frame(0, "stand")
    )
    assert out.posture == "sit" and out.vx == 0.0


def _ema_follow(sent: list[tuple[float, ...]], joint: int) -> list[float]:
    y, out = 0.0, []
    for u in sent:
        y += 0.2 * (u[joint] - y)
        out.append(y)
    return out


def test_robotd_smoothing_is_pre_compensated() -> None:
    # (a robotd left at its default head smoothing, 0.2: the prototype's own runs 1.0 — then the
    # pre-compensation is the identity; the rate limit is taken out here to see the lead alone)
    reflex = HeadPose(np=0.0, hp=0.0, hy=0.9, hr=0.0)
    results = {}
    for lead in (0.0, 1.0):
        profile = replace(PROFILE, lead=lead, head_alpha=0.2, head_slew_sit=100.0)
        state = M.MotorState()
        sent, goals = [], []
        for k in range(40):
            cmd = _cmd(
                "still",
                k * 20,
                head=reflex if k >= 5 else None,
                head_blend=1.0,
                alive=0.0,
                reflex="orient",
            )
            state, out = M.step(profile, state, cmd, _frame(k * 20, "sit"))
            sent.append(out.head)
            goals.append(out.head_goal[2])
        smoothed = _ema_follow(sent, 2)
        target = goals[-1]
        reach = next(i for i, v in enumerate(smoothed) if v >= 0.9 * target) - 5
        results[lead] = reach
    assert results[1.0] < results[0.0]  # robotd's own smoothing reaches the goal sooner
    assert results[1.0] <= 6  # ≤ ~120 ms of robotd's filter for a reflex turn


def test_reflex_overlay_and_gaze() -> None:
    pose = HeadPose(np=-0.7, hp=0.5, hy=-0.3, hr=0.0)
    state = M.MotorState()
    for k in range(20):
        state, out = M.step(
            PROFILE,
            state,
            _cmd("look_around", k * 20, head=pose, head_blend=1.0, alive=0.0, reflex="startle"),
            _frame(k * 20, "sit"),
        )
    carriage = PROFILE.style.carriage
    assert out.pose.np == pytest.approx(pose.np, abs=0.02)
    assert out.pose.hp == pytest.approx(pose.hp, abs=0.02)
    assert out.pose.hy == pytest.approx(pose.hy, abs=0.02)
    del carriage
    state = M.MotorState()
    for k in range(25):
        state, out = M.step(
            PROFILE,
            state,
            _cmd("watch", k * 20, alive=0.0, gaze=Gaze(bearing=0.3, gain=1.0, speed=1.0)),
            _frame(k * 20, "sit"),
        )
    assert out.pose.hy == pytest.approx(0.3 / 0.6, abs=0.08)  # the head turned to the bearing


def test_walking_head_is_kept_in_the_walking_envelope() -> None:
    profile = replace(PROFILE, walk_env=0.6)
    state = M.MotorState()
    for k in range(100):
        cmd = _cmd(
            "wander", k * 20, vx=0.5, head=HeadPose(np=1.0, hp=-1.0, hy=2.0, hr=1.0), head_blend=1.0
        )
        state, out = M.step(profile, state, cmd, _frame(k * 20, "walk", vx=0.45))
        assert max(abs(out.pose.np), abs(out.pose.hp), abs(out.pose.hr)) <= 0.6
        # (fid:motor) the yaw may use the A100-validated ±1.0 rad while walking, kept a margin
        # inside it (the body link reads the MEASURED head against ±1.0 rad)
        assert abs(out.head_goal[2]) <= M.YAW_MAX_RAD * M.WALK_MARGIN + 1e-9
    assert "envelope" in out.notes


def test_the_walking_sway_is_slow_and_locked_to_the_waddle() -> None:
    """MotorCommand.bob (integration 2026-09-28): ducks waddle, they do not head-bob like
    pigeons (pet:etho), and a stride-rate (2.4 Hz) head motion is beyond the head's bandwidth
    (pet:body) — so the head sways side to side (roll, a little yaw) at ≤ 1 Hz, at a third of the
    measured stride rate (a clean oscillation: review 2026-09-28), and the neck does not pump."""

    def head(bob: float) -> list[tuple[float, float, float]]:
        state = M.MotorState()
        values = []
        for k in range(200):
            t = k * 20
            phase = (t / 1000.0 * 2.4) % 1.0
            cmd = _cmd("wander", t, vx=0.5, bob=bob, alive=0.0, stabilise=0.0)
            state, out = M.step(PROFILE, state, cmd, _frame(t, "walk", vx=0.45, step_phase=phase))
            values.append((out.head_goal[0], out.head_goal[2], out.head_goal[3]))
        return values

    with_bob = head(1.0)[50:]
    without = head(0.0)[50:]
    roll = [a[2] - b[2] for a, b in zip(with_bob, without, strict=True)]
    yaw = [a[1] - b[1] for a, b in zip(with_bob, without, strict=True)]
    neck = [a[0] - b[0] for a, b in zip(with_bob, without, strict=True)]
    assert max(roll) - min(roll) > 2 * math.radians(3.0)  # a visible sway: ±3° and more …
    assert max(yaw) - min(yaw) > 0.04  # … above the yaw gearbox deadband (±3°)
    assert max(abs(v) for v in neck) < 0.01  # no stride-rate neck pumping
    ups = sum(1 for a, b in zip(roll, roll[1:], strict=False) if a <= 0 < b)
    assert 1 <= ups <= 3  # ≤ 1 Hz over 3 s (a 2.4 Hz bob would cross ~7 times)


def test_posture_waits_for_the_programs_cue() -> None:
    state = M.MotorState(posture="sit")
    postures = []
    for k in range(100):
        state, out = M.step(
            PROFILE, state, _cmd("stand_up", k * 20, posture="stand"), _frame(k * 20, "sit")
        )
        postures.append(out.posture)
    first = postures.index("stand")
    assert 5 <= first <= 40  # after the head's preparation
    assert all(p == "stand" for p in postures[first:])


def test_calls_are_emitted_once_and_open_the_beak() -> None:
    state = M.MotorState()
    sounds, mouths = [], []
    for k in range(40):
        cmd = _cmd(
            "still",
            k * 20,
            vocal="greet" if k >= 5 else None,
            vocal_seq=1 if k >= 5 else 0,
            vocal_level=0.8,
            alive=0.0,
        )
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "sit"))
        sounds.append(out.sound)
        mouths.append(out.mouth)
    assert sounds.count("greet") == 1
    assert max(mouths) > 0.3 and mouths[-1] == 0.0


def test_a_new_call_number_with_no_call_stops_the_one_that_sounds() -> None:
    """(live voice, LV2) a person spoke over the duck's call: the brain gives the next call number
    and no call. The beak closes at once, where the call would have kept it open."""

    def run(stop_at: int | None) -> list[float]:
        state = M.MotorState()
        mouths = []
        for k in range(40):
            if k < 5:
                call = dict(vocal=None, vocal_seq=0)
            elif stop_at is not None and k >= stop_at:
                call = dict(vocal=None, vocal_seq=2)  # (the stop: a new number, no call)
            else:
                call = dict(vocal="greet", vocal_seq=1) if k == 5 else dict(vocal=None, vocal_seq=1)
            cmd = _cmd("still", k * 20, vocal_level=0.8, alive=0.0, **call)
            state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "sit"))
            mouths.append(out.mouth)
        return mouths

    plain, stopped = run(None), run(12)
    assert max(plain) > 0.3 and plain[12] > 0.3  # (it sounds, the beak is open)
    assert stopped[11] > 0.3  # (until the stop)
    assert stopped[12] == 0.0 and stopped[13] == 0.0  # (and then it is shut, at once)
    assert stopped[:12] == plain[:12]  # (nothing before it changed)
    # a bumped number with no call while nothing sounds changes nothing
    quiet = M.MotorState()
    state2, out2 = M.step(
        PROFILE, quiet, _cmd("still", 0, vocal=None, vocal_seq=3, alive=0.0), _frame(0, "sit")
    )
    assert out2.sound is None and out2.mouth == 0.0 and state2.vocal_seq == 3


def test_motor_is_deterministic_and_bouts_differ() -> None:
    def run(variant: int) -> list[tuple[float, ...]]:
        state = M.MotorState(rng=3)
        out_heads = []
        for k in range(150):
            cmd = replace(_cmd("preen", k * 20), variant=variant)
            state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "sit"))
            out_heads.append(out.head)
        return out_heads

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_motor_system_wrapper() -> None:
    system = M.MotorSystem(M.profile_from_genome(Genome.from_seed(7, gd=30)), seed=7)
    out = system.step(_cmd("look_around", 0), _frame(0, "sit"))
    assert set(out.head_params()) == {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
    assert out.move_params() == {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
    assert out.to_json()["program"] == "look_around"
    pose = HeadPose(np=0.4, hp=-0.2, hy=0.5, hr=-0.3)
    back = M.offsets_to_pose(M.pose_to_offsets(pose))
    assert (back.np, back.hp, back.hy, back.hr) == pytest.approx((0.4, -0.2, 0.5, -0.3))


def test_the_head_does_not_chase_the_stride_rate_wobble() -> None:
    """Review 2026-09-28: a stride-rate (2.4 Hz) head motion is beyond the head's bandwidth
    (pet:body), and the feed-forward stabiliser made the live simulator's (mjcf) camera shake
    worse (pet:probe) — so it is off by default: the gait's yaw wobble does not reach the head
    command, a steady turn is not fought. (The reflex stays as a knob: ``profile.vcr``.)"""

    def head_yaw(profile: M.MotorProfile, yaw_of: object, rate_of: object) -> list[float]:
        state = M.MotorState()
        out_yaw = []
        for k in range(300):
            t = k * 20
            ts = t / 1000.0
            yaw = yaw_of(ts)  # type: ignore[operator]
            rate = rate_of(ts)  # type: ignore[operator]
            frame = SensorFrame(
                t_ms=t,
                body=BodySense(
                    posture="walk", vx=0.45, odom=(0.0, 0.0, yaw), gyro=(0.0, 0.0, rate)
                ),
            )
            cmd = _cmd("still", t, vx=0.45, stabilise=1.0, alive=0.0, bob=0.0)
            state, out = M.step(replace(profile, look_into_turn=0.0), state, cmd, frame)
            out_yaw.append(out.head_goal[2])
        return out_yaw[100:]

    w = 2 * math.pi * 2.4

    def wobble_yaw(s: float) -> float:
        return 0.08 * math.sin(w * s)

    def wobble_rate(s: float) -> float:
        return 0.08 * w * math.cos(w * s)

    assert PROFILE.vcr == 0.0
    calm = head_yaw(PROFILE, wobble_yaw, wobble_rate)
    assert max(calm) - min(calm) < 0.02  # the head does not chase the stride
    chasing = head_yaw(replace(PROFILE, vcr=1.0), wobble_yaw, wobble_rate)
    assert max(chasing) - min(chasing) > 0.1  # (what the old reflex did)
    turn = head_yaw(replace(PROFILE, vcr=1.0), lambda s: 0.5 * s, lambda s: 0.5)
    assert abs(statistics.mean(turn)) < 0.05  # a steady turn is not fought either way


def test_the_motor_side_never_imports_the_world() -> None:
    """Honest senses: the motor lane's modules read frames, never the simulator's truth."""
    import subprocess
    import sys

    code = (
        "import sys, duckmind.neuro.motor, duckmind.neuro.reflex, duckmind.neuro.repertoire; "
        "bad = [m for m in sys.modules if m.startswith('duckmind.world')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bursts_and_pauses_are_long_enough_for_the_body() -> None:
    """A 0.2 s burst or a 0.1 s pause from the brain is not a walk or a pause for this body
    (pet:probe: bursts < 0.4 s barely move, pauses < 0.5 s are coasted through); the motor
    holds a burst ≥ 0.8 s and a pause ≥ 0.45 s — but a reflex stops the body at once."""

    def sent(pattern: list[tuple[int, float]], reflex_at: int | None = None) -> list[float]:
        state = M.MotorState()
        out_vx = []
        for k in range(150):
            t = k * 20
            vx = next(v for until, v in pattern if t < until)
            reflex = "startle" if reflex_at is not None and t >= reflex_at else None
            cmd = _cmd("wander", t, vx=vx, vyaw=0.1 if vx else 0.0, reflex=reflex)
            state, out = M.step(PROFILE, state, cmd, _frame(t, "stand"))
            out_vx.append(out.vx)
        return out_vx

    short_burst = sent([(200, 0.45), (3000, 0.0)])
    assert sum(1 for v in short_burst if v > 0) * 20 >= 780
    short_pause = sent([(1000, 0.45), (1100, 0.0), (3000, 0.45)])
    stops = [i for i, v in enumerate(short_pause) if v == 0.0 and 40 <= i < 80]
    assert len(stops) * 20 >= 440 or not stops  # a real pause, or none at all
    halted = sent([(3000, 0.45)], reflex_at=500)
    assert all(v == 0.0 for v in halted[26:])  # the startle stops the legs at once
    for v in short_burst + short_pause:
        assert v == 0.0 or 0.35 <= v <= 0.55


def test_wander_scale_shapes_how_much_the_path_wanders() -> None:
    params = replace(M.LocoParams(), side_bias=0.0)

    def spread(scale: float) -> float:
        outs = [o for o in _walk(params, 240.0, go=1.0, seed=3, wander_scale=scale) if o.walking]
        return statistics.pstdev([o.vyaw for o in outs])

    straight, normal, wiggly = spread(0.3), spread(1.0), spread(1.8)
    assert straight < 0.6 * normal and wiggly > 1.3 * normal


def test_the_motor_learns_a_course_trim_against_the_bodys_drift() -> None:
    """Every body drifts right when told to walk straight (pet:probe −10…−15 °/s): the motor
    learns a yaw trim from the gyro while the intent is straight, keeps it between walks, and
    never lets it make a straight walk an 'arc' (the arc rule reads the intent)."""
    state = M.MotorState()
    trims, vx_sent = [], []
    for k in range(1500):
        t = k * 20
        sent_yaw = state.trim  # the plant: yaw rate = 0.5 × (commanded + trim) − 0.2 drift
        rate = 0.5 * sent_yaw - 0.2
        frame = SensorFrame(
            t_ms=t,
            body=BodySense(posture="walk", vx=0.5, gyro=(0.0, 0.0, rate), odom=(0.0, 0.0, 0.0)),
        )
        cmd = _cmd("wander", t, vx=0.55, vyaw=0.0, alive=0.0)
        state, out = M.step(PROFILE, state, cmd, frame)
        trims.append(out.trim)
        vx_sent.append(out.vx)
    assert 0.3 < trims[-1] <= M.TRIM_MAX  # it learned to cancel the right drift
    assert abs(0.5 * trims[-1] - 0.2) < 0.05
    assert vx_sent[-1] == pytest.approx(0.55)  # still a fast straight walk, not an arc


def test_the_head_stays_near_home_while_the_body_rises_or_sits() -> None:
    """A head throw during a rise can tip the duck (live, neuro:brain): during a posture
    transition every head goal is held within ±0.25 of the seated envelope."""
    state = M.MotorState()
    for k in range(60):
        cmd = _cmd("startle", k * 20, head=HeadPose(np=-1.0, hp=1.0, hy=0.9), head_blend=1.0)
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "transition"))
        assert max(abs(out.pose.np), abs(out.pose.hp), abs(out.pose.hy), abs(out.pose.hr)) <= 0.25
    assert "transition" in out.notes


def test_a_strong_reflex_veer_makes_an_arc_never_straight() -> None:
    params = replace(M.LocoParams(), meander_amp=0.6, turn_sigma=0.5)
    outs = [o for o in _walk(params, 120.0, go=1.0, seed=5, vyaw_bias=-0.9) if o.walking]
    assert outs
    assert all(o.vyaw <= -M.ARC_BIAS + 1e-9 for o in outs)  # always turning right, hard
    assert all(0.35 <= o.vx <= 0.40 + 1e-9 for o in outs)  # at arc speed


def test_backing_out_of_the_nest_only_when_docking() -> None:
    """Contract 23:58: the brain backs the duck off its nest with vx < 0 and docking=True; any
    other reverse is refused (owner: reversing only at the nest)."""
    outs = {}
    for docking in (True, False):
        state = M.MotorState()
        for k in range(60):
            cmd = _cmd("return_dock", k * 20, vx=-0.34, vyaw=0.0, docking=docking, posture="stand")
            state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "stand"))
        outs[docking] = out
    assert outs[True].vx == pytest.approx(-0.34) and outs[False].vx == 0.0
