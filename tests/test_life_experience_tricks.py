"""A trick it was told to do, ended, as bridge-v2's ``teach_trick`` (duckmind.life.experience): read
from the brain's own command telemetry (``learned['command']['last']``, ``CommandBrain.view``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from duckmind.duckcore import bridge_v2
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C

T0 = 1_773_000_000


def frame(t_ms: int) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture="stand", odom=(0.0, 0.0, 0.0)),
        events=(),
        camera=C.CameraSense(usable=True, sightings=()),
        sound=C.SoundSense(level_db=30.0, kind=None),
        touch=C.TouchSense(site=None, pressure=0.0, stroke=0.0),
        power=C.PowerSense(battery=0.8, on_dock=False, charging=False),
    )


def tel(last: dict[str, Any] | None) -> SimpleNamespace:
    """Telemetry as the brain's integration hands it on: the command intake's view in ``learned``."""
    return SimpleNamespace(
        learned={"command": {"last": last, "active": None, "pending": False}},
        reflex=None,
        reflexes=(),
    )


def last(
    seq: int, verb: str, state: str, done_ms: int | None = None, heard: float = 10.0
) -> dict[str, Any]:
    return {"seq": seq, "verb": verb, "state": state, "t_heard": heard, "done_ms": done_ms}


def watch(exp: Experiences, lasts: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(lasts):
        out += exp.observe(T0 + i, frame(i * 1000), None, tel(item))  # type: ignore[arg-type]
    return out


def test_a_trick_it_did_when_told_is_one_lesson_with_its_time() -> None:
    exp = Experiences()
    out = watch(
        exp,
        [
            None,
            last(1, "bow", "heard"),
            last(1, "bow", "doing"),
            last(1, "bow", "done", 4200),
            last(1, "bow", "done", 4200),
            last(1, "bow", "done", 4200),
        ],
    )
    assert out == [{"k": "teach_trick", "t": T0 + 3, "trick": "bow", "ms": 4200, "outcome": "done"}]
    assert bridge_v2.check(out[0])["outcome"] == "done"


def test_refused_and_given_up_are_lessons_too_interrupted_and_builtin_verbs_are_not() -> None:
    exp = Experiences()
    out = watch(
        exp,
        [
            None,
            last(1, "flip", "refused", 0),
            last(2, "sit", "done", 1500, heard=12.0),
            last(3, "spin", "interrupted", 900, heard=13.0),
            last(4, "shake", "heard", heard=14.0),
            last(4, "shake", "gave_up", 12_000, heard=14.0),
        ],
    )
    assert [(e["trick"], e["outcome"], e["ms"]) for e in out] == [
        ("flip", "refused", 0),
        ("shake", "gave_up", 12_000),
    ]


def test_a_new_brain_counting_from_one_again_is_a_new_command() -> None:
    exp = Experiences()
    out = watch(
        exp,
        [None, last(1, "bow", "done", 3000, heard=10.0), last(1, "bow", "done", 2000, heard=99.0)],
    )
    assert [e["ms"] for e in out] == [3000, 2000]


def test_a_lesson_that_ended_before_the_watch_began_is_not_told_again() -> None:
    exp = Experiences()
    out = watch(exp, [last(1, "bow", "done", 3000), last(1, "bow", "done", 3000)])
    assert out == []
    out = watch(exp, [last(2, "spin", "done", 2500, heard=20.0)])
    assert [e["trick"] for e in out] == ["spin"]
