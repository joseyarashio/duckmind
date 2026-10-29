"""Helpers of the LV2 tests (not a test file): queued calls made by hand, a real integrated brain on
a quiet frame, a person in view, and the clock they all share."""

from __future__ import annotations

from typing import Any

from duckmind.live.echo import CallMark
from duckmind.live.scenario import ManualClock
from duckmind.live.session import QueuedCall
from duckmind.live.tools import parse_call
from duckmind.neuro import contract as C
from duckmind.neuro.genome import Genome

CLEAN = CallMark(False, 1.0, 0, False, False)  # a person spoke a second ago, no echo in the way


def mark(**kw: Any) -> CallMark:
    base: dict[str, Any] = {
        "echo_only": False,
        "since_human_s": 1.0,
        "echo_ms_since_human": 0.0,
        "answered": False,
        "no_audio": False,
        "cause_age_s": None,
        "cause_echo": False,
    }
    base.update(kw)
    return CallMark(**base)


def make_call(
    name: str,
    raw: dict[str, Any],
    *,
    t: float = 5.0,
    seq: int = 1,
    call_mark: CallMark | None = None,
) -> QueuedCall:
    """A queued call as the session makes it from the model's arguments."""
    parsed = parse_call(name, raw)
    cm = call_mark or CLEAN
    return QueuedCall(
        seq=seq,
        id=f"c{seq}",
        name=parsed.name,
        args=parsed.args,
        raw=parsed.raw,
        problems=parsed.problems,
        off_bank=parsed.off_bank,
        t=t,
        wall=0.0,
        connection=1,
        mark=cm,
        echo_suspect=cm.echo_only,
    )


def vocalize(tags: list[dict[str, Any]], **kw: Any) -> QueuedCall:
    return make_call("vocalize", {"tags": tags}, **kw)


def tag(name: str = "greet", level: float = 0.5, at_ms: int = 0) -> dict[str, Any]:
    return {"tag": name, "level": level, "at_ms": at_ms}


def person_sight(
    track: int = 1,
    *,
    bearing: float = 0.0,
    range_m: float = 1.5,
    look: tuple[float, ...] | None = None,
) -> C.Sighting:
    return C.Sighting(
        kind="person",
        bearing=bearing,
        range_m=range_m,
        size=0.3,
        track=track,
        source="model",
        look=look,
    )


def thing_sight(track: int = 7, *, kind: str = "toy", bearing: float = 0.0, range_m: float = 0.55):
    return C.Sighting(
        kind=kind, bearing=bearing, range_m=range_m, size=0.08, track=track, source="eye"
    )


def frame(
    t_ms: int,
    *sightings: C.Sighting,
    sound_kind: str | None = None,
    level_db: float = 30.0,
    battery: float = 0.8,
    day_minute: float = 600.0,
) -> C.SensorFrame:
    """A quiet room at a sitting duck's eye: only what it is given."""
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture="sit"),
        camera=C.CameraSense(usable=True, sightings=tuple(sightings)),
        sound=C.SoundSense(level_db=level_db, kind=sound_kind, onset=1.0 if sound_kind else 0.0),
        power=C.PowerSense(battery=battery),
        day_minute=day_minute,
    )


# four looks, as the eye gives them (legs r, g, luma, length, shirt r, g): well apart
MOM_LOOK = (0.30, 0.28, 0.35, 0.80, 0.55, 0.30)
KID_LOOK = (0.50, 0.30, 0.55, 0.40, 0.30, 0.45)
DAD_LOOK = (0.20, 0.30, 0.20, 0.90, 0.25, 0.30)


def make_brain(
    seed: int = 4107, gd: int = 60, *, clock: ManualClock | None = None
) -> tuple[Any, ManualClock]:
    """An integrated brain of the explorer's genome and a clock to share with the bridge."""
    from duckmind.neuro.integration import make_brain as build

    clock = clock or ManualClock()
    return build(Genome.from_seed(seed, gd=gd), seed=seed), clock


def run_brain(
    brain: Any,
    clock: ManualClock,
    seconds: float,
    *,
    frames: Any = None,
    hz: float = 20.0,
    start_s: float | None = None,
    on_tick: Any = None,
) -> list[Any]:
    """Step ``brain`` for ``seconds`` at ``hz`` on ``clock`` (``frames(t_s)`` makes each tick's
    frame; a quiet room by default). Returns the motor commands."""
    cmds: list[Any] = []
    step = 1.0 / hz
    t = clock.t if start_s is None else start_s
    end = t + seconds
    while t < end - 1e-9:
        t = round(t + step, 6)
        clock.set(t)
        t_ms = int(round(t * 1000))
        fr = frames(t) if frames is not None else frame(t_ms)
        cmd, _tel = brain.step(t_ms, fr)
        cmds.append(cmd)
        if on_tick is not None:
            on_tick(t, cmd)
    return cmds
