"""fid:integrate (2026-09-28) — the fidelity lanes brought together: every gesture a program plays
is commanded big enough to be seen, a standing look down is made with the head pitch, a walking
定格 is long enough to read as a stop, pecks strike with the head, a page stimulus reaches the
world reader at once, and the words claim only what this body does."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

import pytest

from duckmind.neuro import motor as M
from duckmind.neuro import repertoire as P
from duckmind.neuro.contract import BEHAVIOURS, BodySense, HeadPose, MotorCommand, SensorFrame
from duckmind.neuro.livestub import StubBrain
from duckmind.simweb import neuro_live as NL

PROFILE = M.profile_from_style(P.style_from({"activity": 0.5}, stage="juvenile", side=1))
HATCHLING = P.style_from({"activity": 0.5}, stage="hatchling", side=1)


def _frame(t: int, posture: str = "stand") -> SensorFrame:
    return SensorFrame(
        t_ms=t, body=BodySense(posture=posture, vx=0.45 if posture == "walk" else 0.0)
    )


def _cmd(behaviour: str, t: int, **kw: object) -> MotorCommand:
    base = MotorCommand(t_ms=t, behaviour=behaviour, bout=1, variant=5)
    return replace(base, **kw)  # type: ignore[arg-type]


# ── the standing neck: a look down is the head pitch's ─────────────────────────


def test_a_standing_look_down_is_made_with_the_head_pitch() -> None:
    state = M.MotorState()
    for k in range(30):
        cmd = _cmd("still", k * 20, head=HeadPose(np=0.6, hp=0.0), head_blend=1.0, alive=0.0)
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "stand"))
    assert out.pose.np == 0.0 and out.head_goal[0] == 0.0  # the neck is not asked
    assert out.pose.hp == pytest.approx(-0.6 * M.NECK_TO_PITCH, abs=0.02)  # the head looks down
    # walking, the forward neck stays (owner A2's head-down approach, validated with the neck)
    state = M.MotorState()
    for k in range(30):
        cmd = _cmd(
            "wander", k * 20, head=HeadPose(np=0.6, hp=0.0), head_blend=1.0, alive=0.0, vx=0.4
        )
        state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "walk"))
    assert out.pose.np > 0.3


# ── every gesture big enough to be seen ───────────────────────────────────────


def _score(*poses: tuple[float, float, float, float]) -> P.Score:
    keys = tuple(P.Key(100.0 * i, (*pose, 0.0)) for i, pose in enumerate(poses))
    return P.Score(keys, length=100.0 * len(poses))


def test_a_small_gesture_grows_and_a_big_one_or_a_hold_is_left_alone() -> None:
    small = _score((0.0, 0.0, 0.2, 0.0), (0.0, 0.0, 0.35, 0.0))  # a 5° glance from 7°
    grown = P._visible(small, "look_around")
    assert P.gesture_deg(small) == pytest.approx(5.16, abs=0.05)
    assert P.gesture_deg(grown) == pytest.approx(P.VISIBLE_DEG, abs=0.05)
    assert grown.keys[0].pose == small.keys[0].pose  # about where it starts
    # (fid:fix) a moment is named on the page: it grows further, toward MOMENT_DEG (×≤ 4 here)
    moment = P._visible(small, "glance")
    assert P.gesture_deg(moment) == pytest.approx(min(P.MOMENT_DEG, 4 * 5.16), abs=0.1)
    big = _score((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.6, 0.0))  # 20.6°: a bold look, as it is
    assert P._visible(big, "look_around") == big
    bigger = _score((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.7, 0.0))  # 24°: a moment above its floor
    assert P._visible(bigger, "glance") == bigger
    hold = _score((0.0, 0.2, 0.1, 0.0), (0.0, 0.21, 0.12, 0.0))  # a fixation's drift
    assert P._visible(hold, "glance") == hold
    quiet = _score((0.3, -0.2, 0.0, 0.0), (0.4, -0.3, 0.0, 0.0))  # 「頭低一點」
    assert P.gesture_deg(P._visible(quiet, "rest_sit")) == pytest.approx(P.QUIET_DEG, abs=0.05)
    edge = _score((0.0, 0.0, 0.9, 0.0), (0.0, 0.0, 0.98, 0.0))  # never past the envelope
    assert max(k.pose[2] for k in P._visible(edge, "glance").keys) <= 1.0 + 1e-9


def _touches_envelope(score: P.Score) -> bool:
    return any(abs(key.pose[i]) >= 0.999 for key in score.keys for i in range(4))


@pytest.mark.parametrize("behaviour", sorted(set(BEHAVIOURS) - P.NO_FLOOR))
def test_a_calm_hatchlings_programs_are_commanded_big_enough_to_see(behaviour: str) -> None:
    """The smallest duck at a calm amplitude (the brain's hatchling cap 0.6 × 0.6): every unit
    that moves the head at all is commanded at the visible size (or as far as the envelope or
    the floor's gain lets it grow)."""
    spec = BEHAVIOURS[behaviour]
    posture = "stand" if spec.posture == "stand" else "sit"
    want = P.QUIET_DEG if behaviour in P.QUIET_PROGRAMS else P.VISIBLE_DEG
    floor = min(want, P.FLOOR_GAIN_MAX * P.HOLD_DEG)
    for variant in range(6):
        state = None
        seen: set[int] = set()
        for k in range(400):
            t = k * 20
            cmd = _cmd(behaviour, t, variant=variant, amplitude=0.36, tempo=1.0, intensity=0.5)
            state, _ = P.step(state, cmd, HATCHLING, _frame(t, posture), t)
            if state.unit in seen:
                continue
            seen.add(state.unit)
            size = P.gesture_deg(state.score)
            if size >= P.HOLD_DEG:
                assert size >= floor - 0.01 or _touches_envelope(state.score), (
                    behaviour,
                    variant,
                    state.unit,
                    round(size, 1),
                )


def test_a_hatchling_still_moves_smaller_than_an_adult_where_both_are_seen() -> None:
    def mean_abs_yaw(style: P.Style) -> float:
        total, n = 0.0, 0
        for variant in range(25):
            state = None
            for k in range(200):
                t = k * 20
                cmd = _cmd("look_around", t, variant=variant, amplitude=0.9, intensity=0.6)
                state, out = P.step(state, cmd, style, _frame(t, "sit"), t)
                total += abs(out.pose.hy)
                n += 1
        return total / n

    adult = P.style_from({"activity": 0.5}, stage="adult", side=1)
    assert mean_abs_yaw(HATCHLING) < 0.85 * mean_abs_yaw(adult)


# ── 定格 while walking, pecks ─────────────────────────────────────────────────


def test_a_walking_still_moment_lasts_long_enough_for_the_legs_to_stop_and_be_seen() -> None:
    for variant in range(20):
        walking = P.start(
            _cmd("still", 0, variant=variant, vx=0.45), HATCHLING, _frame(0, "walk"), 0
        )
        standing = P.start(_cmd("still", 0, variant=variant), HATCHLING, _frame(0, "stand"), 0)
        assert walking.score.length >= 1600.0  # legs ~0.5 s to stop + a stop held ≥ 0.8 s
        assert standing.score.length <= 1500.0 * 1.2 + 1.0


def _half_swings(values: list[float], min_pp: float) -> int:
    swings, lo, hi, going = 0, values[0], values[0], 0
    for v in values[1:]:
        lo, hi = min(lo, v), max(hi, v)
        if going >= 0 and hi - v >= min_pp:
            swings, going, lo = swings + 1, -1, v
        elif going <= 0 and v - lo >= min_pp:
            swings, going, hi = swings + 1, 1, v
    return swings


def test_pecks_strike_with_the_head_pitch_standing() -> None:
    deg = math.pi / 180.0
    ok = 0
    for variant in range(12):
        state = M.MotorState()
        pitch = []
        for k in range(300):
            cmd = _cmd("peck", k * 20, variant=variant, amplitude=0.55, alive=0.0, intensity=0.5)
            state, out = M.step(PROFILE, state, cmd, _frame(k * 20, "stand"))
            pitch.append(out.head_goal[1] / deg)
            assert out.head_goal[0] == 0.0  # standing, the neck is not asked
        ok += int(_half_swings(pitch, 8.0) >= 2)
    assert ok >= 11


# ── a page stimulus reaches the world reader at once ─────────────────────────


class _Body:
    def __init__(self) -> None:
        self.reads: list[float] = []

    def world(self) -> dict:
        self.reads.append(time.monotonic())
        return {}

    def frame(self) -> dict:
        return {}


def test_a_page_stimulus_wakes_the_world_reader_at_once(monkeypatch) -> None:
    monkeypatch.setattr(NL, "AUX_HZ", 0.5)  # (a 2-s poll: the wake must not wait for it)
    lanes = NL.Lanes(brain=StubBrain(seed=3), names={"brain": "stub"})
    driver = NL.NeuroDriver(NL.NeuroSpec(seed=3), lanes=lanes)
    body = _Body()
    stop = threading.Event()
    thread = threading.Thread(target=driver._aux, args=(body, stop), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while not body.reads and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(body.reads) == 1
        pressed = time.monotonic()
        with pytest.raises(ValueError):
            driver.inject("no_such_stimulus")  # (even a refused press wakes it: harmless)
        deadline = time.monotonic() + 1.0
        while len(body.reads) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(body.reads) >= 2 and body.reads[1] - pressed < 0.5
        # … and then quickly for a moment (the world's copy may not carry the event at once)
        time.sleep(0.5)
        quick = [t for t in body.reads if pressed < t < pressed + 0.5]
        assert len(quick) >= 5
        assert NL.AUX_FAST_S <= 0.05 and NL.AUX_FAST_FOR_S >= 0.5
    finally:
        stop.set()
        driver._aux_wake.set()
        thread.join(timeout=3.0)


# ── the words ─────────────────────────────────────────────────────────────────


def test_the_words_claim_only_what_this_body_does() -> None:
    """fid:audit: the walking network cannot puff the chest, jolt the body, sit lower, stand
    taller or stretch the neck standing, and its fastest walk is its ordinary top speed.

    The acts the social, play and sound skills add (``SOCIAL_BEHAVIOURS``, ``PLAY_BEHAVIOURS``,
    ``BEHAVIOURS`` of ``skill_sound``) are not the walking network's: once any test has built an
    integrated brain they are in the shared table for good (``install_default_skills`` never unloads),
    and ``hide`` really sits down (「坐低」). This test looks at the table's own acts only, whatever ran
    before it; each skill's words are held to its own measurements in its own tests."""
    from duckmind.neuro import skill_play as SP
    from duckmind.neuro import skill_social_brain as SB
    from duckmind.neuro import skill_sound as SS

    skill_acts = set(SB.SOCIAL_BEHAVIOURS) | set(SP.PLAY_BEHAVIOURS) | set(SS.BEHAVIOURS)
    own = {name: spec for name, spec in BEHAVIOURS.items() if name not in skill_acts}
    assert {"startle", "flee", "head_shake", "zoomies"} <= set(own)
    for name, spec in own.items():
        for text in (spec.zh, spec.body_zh):
            for claim in ("挺胸", "挺起胸", "身體一震", "坐低", "挺直", "伸長脖子", "暴衝"):
                assert claim not in text, (name, text)
    for name in ("flee", "head_shake", "zoomies"):  # (the beak IS fast: the raw actuator)
        assert "快速" not in BEHAVIOURS[name].body_zh, name
    assert "往後仰" in BEHAVIOURS["startle"].body_zh


# ── the witness: a new duck's first moments get no body line ─────────────────


def test_a_reflex_in_a_new_ducks_first_moments_gets_no_body_line() -> None:
    from duckmind.simweb.neuro_witness import BORN_SETTLE_MS, Tick, Witness

    witness = Witness()
    closed: list[dict] = []
    events = [
        {"t_ms": 60, "reflex": "orient", "stimulus": "visual:thing", "zh": "看正前方"},
        {"t_ms": 2000, "reflex": "orient", "stimulus": "sound:clap", "zh": "抬頭聽"},
    ]
    for i in range(250):
        t_ms = i * 20
        yaw = 0.8 if t_ms < 600 else 0.0  # (the body swinging into the new duck's pose)
        if t_ms >= 2100:
            yaw = 0.3
        tick = Tick(t_ms=t_ms, head=(0.0, 0.0, yaw, 0.0), head_cmd=(0.0, 0.0, yaw, 0.0))
        closed += witness.feed(tick, [e for e in events if e["t_ms"] <= t_ms])
    assert BORN_SETTLE_MS >= 500
    assert [a["key"][0] for a in closed] == [2000]  # only the reflex after the settling


def test_a_withdrawal_that_steps_away_is_not_told_its_legs_failed_to_stop() -> None:
    from duckmind.simweb.neuro_witness import Tick, Witness

    def run(words: str) -> dict:
        witness = Witness()
        closed: list[dict] = []
        event = {"t_ms": 2000, "reflex": "withdraw", "stimulus": "touch:right", "zh": words}
        x = 0.0
        for i in range(250):
            t_ms = i * 20
            x += 0.004 if t_ms < 4000 else 0.0  # moving 0.2 m/s: before the touch and after it
            yaw = 0.6 if t_ms >= 2100 else 0.0  # the head turns away, well seen
            tick = Tick(
                t_ms=t_ms,
                head=(0.0, 0.0, yaw, 0.0),
                head_cmd=(0.0, 0.0, yaw, 0.0),
                twist_cmd=(0.4, 0.0),
                policy="walk",
                trunk=(x, 0.0, 0.12, 0.0),
                trunk_t=t_ms / 1000.0,
            )
            closed += witness.feed(tick, [event] if t_ms >= 2000 else [])
        return closed[0]

    away = run("從右邊縮開，走開一步")
    assert away["legs"].get("steps_away") and "ok" not in away["legs"]
    assert away["verdict"] == "done" and "腳沒停下來" not in away["zh"] and "走開" in away["zh"]
    kept = run("從右邊縮開")  # (no step announced: a walking withdrawal must stop the legs)
    assert kept["legs"].get("ok") == "not_done" and "腳沒停下來" in kept["zh"]
