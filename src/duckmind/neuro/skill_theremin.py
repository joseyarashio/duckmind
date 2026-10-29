"""手的特雷門琴 — the hand theremin, for a duck with the ToF module (a:sound 2026-09-28).

Owner, 2026-09-28 (「A B 全數」, behaviour 2): for ToF ducks, the official voice's theremin — a
hand's distance in front of the head sets the pitch (upstream ``sounds::Stream``, exactly as
robotd drives it) — emerging from the neural brain.

**Who does what.** On the robot the instrument is robotd's: ``robot.theremin`` picks it up and
puts it down; while it is up robotd reads tofd's raw 8×8 frames itself (15 a second), finds the
hand (``kinematics::hand``: the nearest band 10–70 cm, a low percentile, a 250-ms hold), turns
its closeness into the note (``sounds::stream::hz_at``, this duck's own register), the level
and the beak's opening — one number — and streams ``Stream::wheee(personality, 0)``. The brain
only decides WHETHER to play: this module. (The simulator's robotd has no sound card, so
:mod:`duckmind.simweb.neuro_voice_stream` does robotd's part on the same frames with the same
code: see there.)

**What the brain senses** (honest senses): its ToF summary (:class:`~duckmind.neuro.contract.
TofSense`: the nearest non-floor return in the left, centre and right thirds of the 45° view,
15 Hz) and the camera's ``hand`` label (MODEL). A hand hovering in front of the head is a return
10–70 cm away that stays, and moves (a hand is never still; a wall or a chair leg is — those are
soon not interesting: :class:`HandHover` habituates to a still return). That is the selection
feature ``hand_hover``.

**Why it plays** (:class:`ThereminSkill`): ``hand_hover`` pulls the behaviour ``theremin`` in the
basal-ganglia competition together with play, curiosity and the family's company (weights in
:mod:`duckmind.neuro.skill_sound`), against fear and sleepiness, times its temperament (a
sociable, bold duck plays sooner and longer — a timid one withdraws from a hand near its face:
the reflexes' hand_loom). Playing is its own reward when the hand answers: while it plays and
the hand keeps moving (the person plays along), the learned value of a hovering hand rises
(Rescorla–Wagner, :attr:`ThereminSkill.value`), so next time it takes up the instrument sooner; a
fright during it (a startle, a touch on the beak) lowers it. While it plays the head holds still,
facing the hand (so the beam stays on it) — the beak and the note are robotd's.

Standard library only; floats in the prototype.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from duckmind.neuro.contract import SensorFrame

THEREMIN_VERSION = "neuro-theremin-2"

NEAR_MM = 100  # kinematics::hand Config::default near_m 0.10 (cover-glass crosstalk below)
FAR_MM = 700  # … far_m 0.70
THIRD_RAD = math.radians(15.0)  # the centre of the left / right third of the 45° view
APPEAR_S = 0.4  # a return this long in the band is "there"
GONE_S = 1.2  # … and gone this long after it was last seen
MOTION_TAU_S = 0.6  # the range's speed, low-passed
# How far the thing moved is read from its range over the last MOTION_WINDOW_S (while the duck's
# own head and trunk were still), each reading the median of the last three (a stray cell), less
# what the ToF's own noise can do: σ = 3 mm + 0.5 % of the range (tof_model, the official
# sim/tof.py), NOISE_K σ of it. Measured 2026-09-28 (a:sound, 12 × 90 s of free roaming with no hand
# anywhere): the low-passed |Δrange/Δt| of v1 read the noise on a still wall or box 45 cm ahead
# as "moving" (0.04–0.05 m/s from ±5 mm jitter at 15 Hz) and a playful duck played the theremin
# to it (6 bouts in 18 min, one 27 s long).
MOTION_WINDOW_S = 1.0
NOISE_K = 4.0
NOISE_SIGMA_M = 0.003
NOISE_SHARE = 0.005
MOVE_SPAN_M = 0.03  # moved this much beyond the noise within the window: fully "moving"
# … and the window holds only readings taken with the head where it was when the window began:
# a slow drift of its own head (a preen's 0.1–0.2 rad/s, under SELF_HEAD_RAD_S) swings the beam
# across a slanted surface and the range jumps 5–6 cm (the same probe: 1 bout in 18 min left).
HEAD_STILL_RAD = 0.02
LIVELY_MEMORY_S = 3.0  # it moved this recently: it is still the hand
NEAR_LIVELY_M = 0.5  # a moving return this near is a hand (farther: only with the eye's label)
SELF_GYRO = 0.3  # rad/s: its trunk turning — the room seems to move
SELF_HEAD_RAD_S = 0.25  # its head joints moving this fast — the same
SELF_SETTLE_S = 0.25  # … and a moment after its own motion stops
STILL_HABITUATE_S = 4.0  # a return that never moves: not a hand (a wall, a leg), after this
DOWN_AFTER_S = 0.35  # the hand gone this long: the instrument goes down
LEARN_PER_S = 0.01  # answered play: the value moves this share toward +1 a second (RW)
FRIGHT_DROP = 0.15  # a fright while playing: this share toward −1


@dataclass(frozen=True, slots=True)
class HoverView:
    """What the ToF (and the camera) say about something hovering in front of the head."""

    present: bool  # a return in the band, for APPEAR_S (and not gone GONE_S)
    range_m: float | None  # the nearest in-band return now (None: none this frame)
    bearing: float  # rad, + = the duck's left (which third it is in)
    moving: float  # 0..1 how much it moves (a hand) — a still return is not interesting
    hand_seen: bool  # the camera's ``hand`` label (MODEL) on something near
    hover: float  # the selection feature ``hand_hover`` 0..1
    still_s: float  # how long it has not moved


@dataclass(slots=True)
class HandHover:
    """A hand hovering in front of the beak, from the ToF summary (see the module docstring).

    Two things tell a hand from the room: the eye's ``hand`` label (MODEL: a hand near the face,
    lit), or a near return (≤ :data:`NEAR_LIVELY_M`) whose range changes while the duck's own head
    and body are still — its own head's motion makes the room seem to move, and that is cancelled
    (an efference copy: range changes while it moves its head are not the thing's)."""

    seen_s: float = 0.0  # how long a return has been in the band
    last_seen: float = -1e9
    last_range: float | None = None
    last_t: float | None = None
    speed: float = 0.0  # span / window, m/s (for the logs)
    still_s: float = 0.0
    bearing: float = 0.0
    lively_t: float = -1e9  # when it last moved (or the eye called it a hand)
    last_head: tuple[float, float, float, float] | None = None
    self_moving_until: float = -1e9
    span: float = 0.0  # m the range moved beyond the noise in the window (while self-still)
    raw: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    window: deque[tuple[float, float]] = field(default_factory=deque)
    last_reading: int | None = None
    last_age: int = 0
    win_head: tuple[float, float, float, float] | None = None

    def step(self, t_s: float, frame: SensorFrame) -> HoverView:
        dt = 0.0 if self.last_t is None else min(0.25, max(0.0, t_s - self.last_t))
        self.last_t = t_s
        tof = frame.tof
        # its own motion (the efference copy): the head joints moving or the trunk turning
        head = frame.body.head
        g = frame.body.gyro
        turning = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) > SELF_GYRO
        moved = (
            self.last_head is not None
            and dt > 0.0
            and max(abs(a - b) for a, b in zip(head, self.last_head, strict=True)) / dt
            > SELF_HEAD_RAD_S
        )
        self.last_head = head
        if turning or moved or frame.body.posture == "walk":
            self.self_moving_until = t_s + SELF_SETTLE_S
        self_still = t_s >= self.self_moving_until
        best: tuple[int, float] | None = None
        if tof.present:
            for mm, bearing in (
                (tof.ahead_mm, 0.0),
                (tof.left_mm, THIRD_RAD),
                (tof.right_mm, -THIRD_RAD),
            ):
                if mm is not None and NEAR_MM <= mm <= FAR_MM and (best is None or mm < best[0]):
                    best = (int(mm), bearing)
        hand_seen = any(
            s.kind == "hand" and (s.range_m is None or s.range_m < 0.8)
            for s in frame.camera.sightings
        )
        range_m: float | None = None
        age_before, self.last_age = self.last_age, int(tof.age_ms)
        if (
            self.win_head is not None
            and max(abs(a - b) for a, b in zip(head, self.win_head, strict=True)) > HEAD_STILL_RAD
        ):
            self_still = False  # its head is no longer where the window began
        if not self_still or not self.window:
            self.window.clear()  # its own motion moved the room: start over when it is still
            self.win_head = head
        if best is not None and not tof.touch:
            range_m = best[0] / 1000.0
            self.bearing += (best[1] - self.bearing) * min(1.0, dt / 0.3 if dt else 1.0)
            # a new ToF frame (15 a second; the brain ticks 50): its age starts over, or its
            # reading changed
            fresh = (
                self.last_reading is None
                or best[0] != self.last_reading
                or tof.age_ms == 0
                or tof.age_ms < age_before
            )
            self.last_reading = best[0]
            if fresh:
                self.raw.append(range_m)
                if self_still:
                    self.window.append((t_s, sorted(self.raw)[len(self.raw) // 2]))
            while self.window and t_s - self.window[0][0] > MOTION_WINDOW_S:
                self.window.popleft()
            if len(self.window) >= 3:
                values = [r for _, r in self.window]
                mean = sum(values) / len(values)
                noise = NOISE_K * (NOISE_SIGMA_M + NOISE_SHARE * mean)
                self.span = max(0.0, max(values) - min(values) - noise)
            else:
                self.span *= math.exp(-dt / MOTION_TAU_S) if dt else 1.0
            self.speed = self.span / MOTION_WINDOW_S  # (for the logs: m/s over the window)
            self.last_range = range_m
            self.last_seen = t_s
            self.seen_s += dt
        else:
            self.span *= math.exp(-dt / MOTION_TAU_S) if dt else 1.0
            self.speed = self.span / MOTION_WINDOW_S
            if t_s - self.last_seen > GONE_S:
                self.seen_s = 0.0
                self.last_range = None
                self.last_reading = None
                self.still_s = 0.0
                self.raw.clear()
                self.window.clear()
        near = self.last_range is not None and self.last_range <= NEAR_LIVELY_M
        moving = min(1.0, self.span / MOVE_SPAN_M) if near else 0.0
        if moving >= 0.5 or hand_seen:
            self.still_s = 0.0
            self.lively_t = t_s
        elif self.last_range is not None:
            self.still_s += dt
        present = self.seen_s >= APPEAR_S and t_s - self.last_seen <= GONE_S
        hover = 0.0
        if present and t_s - self.lively_t <= LIVELY_MEMORY_S:
            # only something that moves near it (or that the eye calls a hand) is a hand: a
            # wall, a chair leg or a box in front never is, however near
            lively = max(moving, 1.0 if hand_seen else 0.0)
            hover = 0.45 + 0.55 * lively
            if self.still_s > STILL_HABITUATE_S:  # it stopped moving long ago: not a hand
                hover *= math.exp(-(self.still_s - STILL_HABITUATE_S) / 2.0)
        return HoverView(
            present=present,
            range_m=range_m,
            bearing=self.bearing,
            moving=moving,
            hand_seen=hand_seen,
            hover=round(hover, 4),
            still_s=self.still_s,
        )


@dataclass(frozen=True, slots=True)
class ThereminTick:
    view: HoverView
    feature: float  # ``hand_hover`` for the selection (0 without a ToF module)
    up: bool  # the instrument is (to be) up: robot.theremin
    gaze_bearing: float  # where the head holds still (the hand's third)
    playing_s: float  # how long this play has gone on
    value: float  # the learned value of a hovering hand (−1..1)


@dataclass(slots=True)
class ThereminSkill:
    """Whether to play, and what playing does to it (see the module docstring)."""

    hover: HandHover | None = None
    value: float = 0.0  # learned: a hovering hand is play (Rescorla–Wagner, −1..1)
    playing_s: float = 0.0
    answered_s: float = 0.0  # play time with the hand moving (the person playing along)
    plays: int = 0
    last_t: float | None = None
    was_playing: bool = False

    def __post_init__(self) -> None:
        if self.hover is None:
            self.hover = HandHover()

    def step(self, t_ms: int, frame: SensorFrame, *, playing: bool, fright: bool) -> ThereminTick:
        """``playing``: the brain's act is ``theremin`` now; ``fright``: a startle or a withdrawal
        fired this tick (a scare while playing)."""
        t_s = t_ms / 1000.0
        dt = 0.0 if self.last_t is None else min(0.25, max(0.0, t_s - self.last_t))
        self.last_t = t_s
        assert self.hover is not None
        view = self.hover.step(t_s, frame)
        # what it learned of hovering hands scales the pull (a hand that played along draws it
        # more, one that frightened it less)
        feature = view.hover * (0.75 + 0.5 * self.value) if frame.tof.present else 0.0
        feature = max(0.0, min(1.0, feature))
        # it plays while its act is the theremin and the hand is there: the moment its own ToF
        # summary says the hand is gone, the instrument goes down (robotd's raw beam would
        # otherwise sing the floor a standing duck's lowest zones see ~45 cm ahead)
        gone = t_s - self.hover.last_seen
        up = playing and frame.tof.present and (view.present or gone <= DOWN_AFTER_S)
        if up:
            if not self.was_playing:
                self.plays += 1
                self.playing_s = 0.0
            self.playing_s += dt
            if view.present and view.moving >= 0.5:
                self.answered_s += dt
                # the hand answers: a hovering hand is play (toward +1, per minute of it)
                self.value += LEARN_PER_S * dt * (1.0 - self.value)
            if fright:
                self.value += FRIGHT_DROP * (-1.0 - self.value)
        self.was_playing = up
        self.value = max(-1.0, min(1.0, self.value))
        return ThereminTick(
            view=view,
            feature=feature,
            up=up,
            gaze_bearing=view.bearing,
            playing_s=self.playing_s if up else 0.0,
            value=self.value,
        )


__all__ = [
    "FAR_MM",
    "HandHover",
    "HoverView",
    "NEAR_MM",
    "THEREMIN_VERSION",
    "ThereminSkill",
    "ThereminTick",
]
