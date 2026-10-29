"""Round 3 (2026-10-09): after a touch it felt while stepping, a free walk turns away from the thing for ~10 s, by the duck's own
sensing only (the touch site, else the yaw rate it was turning at). Before, a wander burst could start as a tight arc towards a thing
standing against its side and sweep it along, burst after burst (the tissue box: 7 of 48 headless free lives with >= 8 knocks, 0 at 0 ms)."""

from __future__ import annotations

from test_neuro_brain import _genome

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C


def _brain(preset: str = "bold_explorer", seed: int = 3) -> B.NeuroBrain:
    return B.NeuroBrain(_genome(preset), seed=seed)


def _frame(
    t_ms: int, *, site: str | None = None, vyaw: float = 0.0, vx: float = 0.0
) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture="walk" if vx else "stand", vx=vx, vyaw=vyaw),
        touch=C.TouchSense(site=site, pressure=0.2) if site else C.TouchSense(),
    )


def test_a_placed_touch_says_the_side_and_the_walk_turns_away_from_it() -> None:
    brain = _brain()
    brain._last_step_s = 5.0  # (it was stepping a moment ago)
    brain._touch_away_update(5.4, _frame(5400, site="right"))
    assert (
        brain._touch_away == +1 and brain._touch_away_until == 5.4 + B.TOUCH_AVOID_S
    )  # (right touched: turn left)
    brain._last_step_s = 7.8
    brain._touch_away_update(8.0, _frame(8000, site="left"))
    assert brain._touch_away == -1


def test_an_unplaced_touch_is_on_the_side_it_was_turning_to_and_none_without_a_turn() -> None:
    brain = _brain()
    brain._last_step_s = 5.0
    brain._touch_away_update(5.4, _frame(5400, site="body", vyaw=-0.7))
    assert brain._touch_away == +1  # turning right into it: away is left
    other = _brain()
    other._last_step_s = 5.0
    other._touch_away_update(5.4, _frame(5400, site="body", vyaw=0.1))
    assert other._touch_away == 0 and other._touch_away_until < 0  # (no turn: nothing to say)
    standing = _brain()
    standing._last_step_s = 1.0  # it had not stepped for 4 s: a hand's touch, not its own walk
    standing._touch_away_update(5.4, _frame(5400, site="right"))
    assert standing._touch_away == 0


def _mean_vyaw(away: int) -> float:
    brain = _brain()
    t_ms, out = 0, []
    while t_ms < 3000:
        brain.step(t_ms, _frame(t_ms))
        t_ms += 50
    brain._touch_away, brain._touch_away_until = (
        away,
        brain._t0 / 1000.0 + 1e9,
    )  # (a memory that does not expire here)
    while t_ms < 3000 + 90_000:
        cmd, _ = brain.step(t_ms, _frame(t_ms))
        if cmd.vx > 0.0 and cmd.behaviour == "wander":
            out.append(cmd.vyaw)
        t_ms += 50
    assert len(out) > 200
    return sum(out) / len(out)


def test_a_free_walk_is_pushed_away_from_the_touched_side_and_not_otherwise() -> None:
    plain, left, right = _mean_vyaw(0), _mean_vyaw(+1), _mean_vyaw(-1)
    assert right < plain < left
    assert left - right >= 0.2


def test_a_flight_turns_away_from_the_touched_side_too() -> None:
    def flee_mean(away: int) -> float:
        brain = _brain()
        t_ms = 0
        while t_ms < 3000:
            brain.step(t_ms, _frame(t_ms))
            t_ms += 50
        brain._touch_away, brain._touch_away_until = away, brain._t0 / 1000.0 + 1e9
        bang = C.SoundSense(level_db=94.0, onset=1.0, kind="bang")
        out = []
        while t_ms < 3000 + 9000:
            frame = C.SensorFrame(
                t_ms=t_ms,
                body=C.BodySense(posture="stand"),
                sound=bang if t_ms == 3000 else C.SoundSense(),
            )
            cmd, _ = brain.step(t_ms, frame)
            if cmd.behaviour == "flee" and cmd.vx > 0.0:
                out.append(cmd.vyaw)
            t_ms += 50
        assert out  # (it still flees)
        return sum(out) / len(out)

    assert flee_mean(-1) < flee_mean(+1)
