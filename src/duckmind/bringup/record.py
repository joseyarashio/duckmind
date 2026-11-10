"""The guided bench recording: ``robot.state`` (and ``tof.frame``) over the remote-body link while a person takes the
duck through the bring-up poses (docs/deploy/bring-up.md).

A :class:`Segment` is one pose held for some seconds: the person is told what to arrange (:class:`Operator`), the duck
is waited for (its policy must say the posture), and every tick is written to the recording with the segment's name.
Cues (``lift``, ``set_down``; ``sit``, ``stand`` in the sit–stand cycles) are said at fixed moments and marked in the recording, so the fit knows when the hand came.

By default the tool is **passive**: it reads only, and the person changes the posture and walks the duck with the
duck's own control. With ``drive`` it sends the two things a brain sends — ``robot.do sit_toggle`` and a bounded
``robot.move`` (:data:`WALK_VX` for a segment's few seconds) — and nothing else; it never enables, homes or poses the
robot (SAFETY.md), and whatever ends the recording ends with a zero twist and ``robot.stop``.

The recording is JSON lines (gzip): a header (:data:`SCHEMA`, the body the bridge described), then
``{"seg", "rep", "state": robot.state params}``, ``{"seg", "rep", "mark", "t_ns"}`` and, for the ToF,
``{"seg", "rep", "tof": tof.frame params, "state": …}``.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "duckmind.body-recording/1"
RECORDING = "recording.jsonl.gz"
WALK_VX = 0.30  # the slowest forward walk the networks make (velstand needs ≥ 0.30): the bench walk
SETTLE_S = 2.0  # after the posture is reached, before a segment's first sample
POSTURE_WAIT_S = 60.0
TOGGLE_RETRY_S = 4.0
STAND = ("stand", "walk")
SIT = ("sit",)
POSTURE_CUES = ("sit", "stand")  # a cue that changes the posture (driven: one sit_toggle)
HELD = ("held",)  # robotd's own pickup pause (a real board's [pickup])


@dataclass(frozen=True)
class Segment:
    """One pose. ``policies``: what robotd's policy must read while it is recorded; ``cues``: (seconds into the
    recording, name); ``walk``: (from, to) seconds a forward walk is commanded (driven) or expected (passive);
    ``tof_mm``: the board's distance for a ToF segment (the frames are recorded, not the ticks alone)."""

    name: str
    stage: int
    say: str
    policies: tuple[str, ...]
    seconds: float
    reps: int = 1
    cues: tuple[tuple[float, str], ...] = ()
    walk: tuple[float, float] | None = None
    tof_mm: int | None = None
    posture: str | None = None  # "stand" | "sit": what a driven recording toggles to first


def _lift(name: str, posture: str, policies: tuple[str, ...], floor: str) -> Segment:
    return Segment(
        name,
        2,
        f"Put the duck on the floor, {floor}. At LIFT pick it up about 20 cm and hold it still; at "
        "SET DOWN put it back on the same spot and let go.",
        policies + HELD,
        13.0,
        reps=3,
        cues=((3.0, "lift"), (8.0, "set_down")),
        posture=posture,
    )


# The bench plan (docs/deploy/bring-up.md): stage 1 in the hand, stage 2 on the floor, stage 3 the supervised walk.
PLAN: tuple[Segment, ...] = (
    Segment(
        "air_sit",
        1,
        "Hold the duck in the air by its trunk, SEATED (legs tucked), still.",
        SIT + HELD,
        8.0,
        posture="sit",
    ),
    Segment(
        "air_stand",
        1,
        "Hold the duck in the air by its trunk, STANDING (legs hanging free), still.",
        STAND + HELD,
        10.0,
        posture="stand",
    ),
    Segment(
        "air_walk",
        1,
        "Keep holding it in the air, standing. The legs will STEP IN THE AIR for 4 s, then stop. Keep "
        "fingers clear of the legs.",
        STAND + HELD,
        10.0,
        walk=(1.0, 5.0),
        posture="stand",
    ),
    Segment(
        "floor_sit",
        2,
        "Put the duck on the floor, SEATED, and let go. Stay next to it.",
        SIT,
        8.0,
        posture="sit",
    ),
    Segment(
        "floor_stand",
        2,
        "The duck STANDS on the floor, nothing touching it. Stay next to it.",
        STAND,
        12.0,
        posture="stand",
    ),
    Segment(
        "cycle",
        2,
        "The duck stands on the floor. It will SIT DOWN and STAND UP once. Stay next to it, hands off unless it tips.",
        STAND + SIT,
        12.0,
        reps=5,
        cues=((1.0, "sit"), (6.5, "stand")),
        posture="stand",
    ),
    _lift("lift_stand", "stand", STAND, "STANDING"),
    _lift("lift_sit", "sit", SIT, "SEATED"),
    Segment(
        "walk",
        3,
        "Clear 1.5 m of floor ahead of the duck. It will WALK FORWARD for 4 s (about 0.6 m) and stop. Keep "
        "a hand next to it; do not touch it unless it tips.",
        STAND,
        9.0,
        reps=3,
        walk=(1.0, 5.0),
        posture="stand",
    ),
)
TOF_DISTANCES_MM = (0, 30, 60, 100, 150, 200, 300)


def tof_plan(distances_mm: Sequence[int] = TOF_DISTANCES_MM) -> tuple[Segment, ...]:
    """The guided ToF measurement: a flat board square to the duck's face at each known distance."""
    return tuple(
        Segment(
            f"tof_{mm}",
            2,
            (
                "Hold a flat board (a book) FLAT AGAINST the duck's face, covering the sensor."
                if mm == 0
                else f"Hold the flat board square to the duck's face, {mm} mm in front of the sensor "
                "(use a ruler or a spacer)."
            ),
            STAND + SIT + HELD,
            3.0,
            tof_mm=mm,
        )
        for mm in distances_mm
    )


def plan_for(stages: Sequence[int], *, tof: bool) -> tuple[Segment, ...]:
    out = tuple(s for s in PLAN if s.stage in stages)
    return out + (tof_plan() if tof and 2 in stages else ())


class Body(Protocol):
    """The duck on the link. ``read_state`` / ``read_tof``: the next raw ``robot.state`` / ``tof.frame`` params (None:
    nothing within the timeout). The three commands are all a recording ever sends."""

    info: Mapping[str, Any]

    def read_state(self, timeout_s: float) -> dict[str, Any] | None: ...

    def read_tof(self, timeout_s: float) -> dict[str, Any] | None: ...

    def sit_toggle(self) -> None: ...

    def move(self, vx: float) -> None: ...

    def stop(self) -> None: ...


class Operator(Protocol):
    """The person at the bench (the console; a rehearsal's scripted hands)."""

    def ready(self, segment: Segment, rep: int) -> None:
        """Tell them what to arrange and return when they have (the console waits for Enter)."""

    def cue(self, segment: Segment, rep: int, name: str) -> None:
        """A timed cue during the recording (``lift``, ``set_down``)."""

    def note(self, text: str) -> None: ...


class SegmentFailed(RuntimeError):
    """A segment that could not be recorded (the posture never came, the policy changed under it, a fall)."""


@dataclass
class Sink:
    """The recording file being written."""

    path: Path
    lines: int = 0
    _file: Any = field(default=None, repr=False)

    def open(self, header: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        self._file = gzip.open(self.path, "at", encoding="utf-8")
        if fresh:
            self.write({"schema": SCHEMA, **header})

    def write(self, line: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.lines += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _policy(state: Mapping[str, Any]) -> str:
    return str(state.get("policy") or "").split(":", 1)[0]


def _fallen(state: Mapping[str, Any]) -> bool:
    safety = state.get("safety")
    return isinstance(safety, Mapping) and safety.get("fallen") is True


def _t_ns(state: Mapping[str, Any]) -> int:
    value = state.get("t_ns")
    return int(value) if isinstance(value, int) else 0


def _next_state(body: Body, timeout_s: float = 2.0) -> dict[str, Any]:
    state = body.read_state(timeout_s)
    if state is None:
        raise SegmentFailed("robot.state stopped coming over the link")
    if not isinstance(state.get("t_ns"), int):
        # (the recording's clock is the duck's own: without it nothing here could ever time out)
        raise SegmentFailed("robot.state carries no t_ns (which robotd is this?)")
    return state


def reach(body: Body, segment: Segment, *, drive: bool, operator: Operator) -> None:
    """Wait until robotd's policy reads the segment's posture (driven: toggle towards it), then let it settle."""
    waited_ns: int | None = None
    toggled_ns: int | None = None
    while True:
        state = _next_state(body)
        t, policy = _t_ns(state), _policy(state)
        if _fallen(state):
            raise SegmentFailed("the duck has fallen: stand it up by hand, then record again")
        if policy in segment.policies:
            break
        waited_ns = t if waited_ns is None else waited_ns
        if (t - waited_ns) / 1e9 > POSTURE_WAIT_S:
            raise SegmentFailed(
                f"the duck never came to {'/'.join(segment.policies)} (its policy reads {policy!r})"
            )
        other = policy in STAND + SIT and segment.posture is not None
        if drive and other and (toggled_ns is None or (t - toggled_ns) / 1e9 >= TOGGLE_RETRY_S):
            toggled_ns = t
            body.sit_toggle()
        elif not drive and toggled_ns is None:
            toggled_ns = t
            operator.note(
                f"waiting for the duck to {segment.posture or '/'.join(segment.policies)} "
                f"(policy {policy!r}): change it with the duck's own control"
            )
    start = _t_ns(_next_state(body))
    while (_t_ns(_next_state(body)) - start) / 1e9 < SETTLE_S:
        pass


def record_segment(
    body: Body, segment: Segment, rep: int, sink: Sink, *, drive: bool, operator: Operator
) -> int:
    """Record one repetition; returns the ticks written. Raises :class:`SegmentFailed` (nothing of it counts: the
    caller marks the repetition void) when the posture is lost or the duck falls."""
    reach(body, segment, drive=drive, operator=operator)
    first = _next_state(body)
    t0 = _t_ns(first)
    cues = list(segment.cues)
    ticks = wrong = frames = 0
    walking = False
    state = first
    try:
        while True:
            t = (_t_ns(state) - t0) / 1e9
            if t >= segment.seconds:
                break
            while cues and t >= cues[0][0]:
                _, name = cues.pop(0)
                sink.write({"seg": segment.name, "rep": rep, "mark": name, "t_ns": _t_ns(state)})
                operator.cue(segment, rep, name)
                if drive and name in POSTURE_CUES:
                    body.sit_toggle()
            if segment.walk is not None and drive:
                want = segment.walk[0] <= t < segment.walk[1]
                if want:
                    body.move(WALK_VX)  # (every tick: robotd zeroes a twist that is not refreshed)
                elif walking:
                    body.move(0.0)
                walking = want
            if _fallen(state):
                raise SegmentFailed("the duck fell during the recording")
            wrong += _policy(state) not in segment.policies
            sink.write({"seg": segment.name, "rep": rep, "state": state})
            ticks += 1
            if segment.tof_mm is not None:
                frame = body.read_tof(0.0)
                if frame is not None:
                    sink.write({"seg": segment.name, "rep": rep, "tof": frame, "state": state})
                    frames += 1
            state = _next_state(body)
    finally:
        if walking or (segment.walk is not None and drive):
            body.move(0.0)
    if ticks == 0 or wrong > 0.1 * ticks:
        raise SegmentFailed(
            f"the posture changed during the recording ({wrong} of {ticks} ticks were not "
            f"{'/'.join(segment.policies)})"
        )
    if segment.tof_mm is not None and frames < 5:
        raise SegmentFailed(f"only {frames} ToF frames came (is tofd on the link?)")
    return ticks


def record(
    body: Body,
    plan: Sequence[Segment],
    sink: Sink,
    operator: Operator,
    *,
    drive: bool = False,
    retries: int = 2,
) -> dict[str, int]:
    """Record every segment of ``plan`` (each repetition, each at most ``retries`` more times after a failure, a
    failed try marked ``void`` in the recording); returns the ticks per segment. Ends with a zero twist and
    ``robot.stop`` whatever happens."""
    done: dict[str, int] = {}
    try:
        for segment in plan:
            for rep in range(segment.reps):
                for attempt in range(retries + 1):
                    operator.ready(segment, rep)
                    try:
                        ticks = record_segment(
                            body, segment, rep, sink, drive=drive, operator=operator
                        )
                    except SegmentFailed as error:
                        sink.write({"seg": segment.name, "rep": rep, "void": str(error)})
                        operator.note(f"{segment.name} #{rep + 1} does not count: {error}")
                        if attempt == retries:
                            raise
                        continue
                    done[segment.name] = done.get(segment.name, 0) + ticks
                    break
    finally:
        body.move(0.0)
        body.stop()
    return done


def read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A recording's header and lines, the repetitions marked ``void`` taken out (their last try stays)."""
    lines: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        header = json.loads(file.readline())
        if header.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA} recording")
        for raw in file:
            line = json.loads(raw)
            if "void" in line:
                lines = [
                    kept
                    for kept in lines
                    if (kept.get("seg"), kept.get("rep")) != (line.get("seg"), line.get("rep"))
                ]
            elif "schema" not in line:
                lines.append(line)
    return header, lines


class Console:
    """The person at a terminal: what to do is printed, Enter says "ready"."""

    def __init__(
        self,
        ask: Callable[[str], str] | None = None,
        say: Callable[[str], None] | None = None,
        beep: bool = True,
    ) -> None:
        self.ask = input if ask is None else ask
        self.say = print if say is None else say
        self.beep = beep

    def ready(self, segment: Segment, rep: int) -> None:
        count = f" ({rep + 1} of {segment.reps})" if segment.reps > 1 else ""
        self.say(f"\n[stage {segment.stage}] {segment.name}{count}: {segment.say}")
        self.say(f"  records {segment.seconds:.0f} s. Ctrl-C stops everything.")
        self.ask("  press Enter when ready… ")

    def cue(self, segment: Segment, rep: int, name: str) -> None:
        words = {
            "lift": "LIFT — pick it up now",
            "set_down": "SET DOWN — put it back now",
            "sit": "SIT — it sits down now (passive: sit it down with its own control)",
            "stand": "STAND — it stands up now (passive: stand it up with its own control)",
        }
        self.say(("\a" if self.beep else "") + f"  >>> {words.get(name, name)}")

    def note(self, text: str) -> None:
        self.say(f"  {text}")


def segments(
    lines: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[str, int, list[Mapping[str, Any]]]]:
    """The recording's lines grouped by (segment, repetition), in the order they were recorded."""
    order: list[tuple[str, int]] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for line in lines:
        key = (str(line.get("seg")), int(line.get("rep") or 0))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(line)
    for name, rep in order:
        yield name, rep, groups[(name, rep)]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "PLAN",
    "RECORDING",
    "SCHEMA",
    "TOF_DISTANCES_MM",
    "WALK_VX",
    "Body",
    "Console",
    "Operator",
    "Segment",
    "SegmentFailed",
    "Sink",
    "now_iso",
    "plan_for",
    "read",
    "record",
    "record_segment",
    "segments",
    "tof_plan",
]
