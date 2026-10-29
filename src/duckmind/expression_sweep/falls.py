"""robotd's own fall verdict, velstand's self-righting, and the sit ↔ stand cycle in phases.

Pure Python: the GPU runner applies :func:`next_run` to tensors exactly as written here,
and anyone can recompute a summary from the raw per-episode records.

**The fall rule** is robotd's (duck-control ``Safety::observe`` with the robotd-params
defaults): projected-gravity z above −0.5, i.e. the trunk tipped more than 60°, accumulated
for 200 ms is *fallen*; one upright sample clears it. The v2 sweep's count (any instant
past 45°) stays as a secondary column: it also counts near-falls.

**Self-righting** follows spec §5.4. robotd's verdict is a report, not a rule (limp-fall is
off by default), so velstand keeps driving. duck-body zeroes ``robot.move``; if the sitstand
network is in charge (seated) it sends ``stand``, so robotd rises for 1 s with the posture
flag at 0 and then hands over to velstand at zero twist. The duck is up again once the
trunk stays under 25° (which also clears robotd's verdict) for 1 s while velstand drives.
A try is 8 s from the fall (the official ``fallen_too_long``); duck-body then waits 2 s and
gives it a second 8 s. Nothing in robotd pauses the network in between short of cutting
torque, which the spec does not ask for, so velstand keeps running through the pause and
the second try is the window 10–18 s after the fall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from duckmind.expression_sweep.stats import describe, proportion

CONTROL_HZ = 50
FALL_GRAVITY_Z = -0.5  # robotd-params safety.fall_gravity_z
FALL_TILT_DEG = 60.0  # the same threshold as a tilt: gravity z = −cos(tilt) > −0.5
FALL_DEBOUNCE_S = 0.2  # robotd-params safety.fall_debounce_ms
LEGACY_FALL_TILT_DEG = 45.0  # the v2 sweep's count
SETTLE_BAD_TILT_DEG = 30.0  # tipped this far before any action: the reset, not the cycle
UPRIGHT_TILT_DEG = 25.0
UPRIGHT_HOLD_S = 1.0
TRY_S = 8.0
PAUSE_S = 2.0
RIGHTING_WINDOW_S = TRY_S + PAUSE_S + TRY_S
OFFICIAL_UP_Z_M = 0.09  # the official training also asks the trunk to be above 9 cm
TRACE_HZ = 10  # tilt and trunk height kept for every episode with a fall verdict
PHASES = ("rise", "stand", "sit", "seated")
MODES = ("sit", "rise", "up")  # robotd's Sit machine: sitstand flag 1, flag 0, velstand

T = TypeVar("T")


def next_run(run: T, condition: Any) -> T:
    """Consecutive-sample counter: +1 while ``condition`` holds, 0 as soon as it does not.

    Works on ints and bools and, unchanged, on torch tensors: the runner uses this.
    """
    return (run + 1) * condition  # type: ignore[operator, no-any-return]


def debounce_ticks(hz: int = CONTROL_HZ) -> int:
    """Samples of 'down' after which robotd calls it fallen (200 ms at 50 Hz = 10)."""
    return round(FALL_DEBOUNCE_S * hz)


def hold_ticks(hz: int = CONTROL_HZ) -> int:
    return round(UPRIGHT_HOLD_S * hz)


def robotd_fallen(tilts_deg: list[float], hz: int = CONTROL_HZ) -> list[bool]:
    """robotd's verdict after each sample of a tilt trace (one sample per control tick)."""
    run, out = 0, []
    for tilt in tilts_deg:
        run = next_run(run, tilt > FALL_TILT_DEG)
        out.append(run >= debounce_ticks(hz))
    return out


def landing(gravity_b: list[float] | None) -> str | None:
    """How a duck lies, from gravity in the trunk frame (the face points +x):
    ``front`` (face down), ``back`` (face up) or ``side``."""
    if gravity_b is None:
        return None
    gx, gy = gravity_b[0], gravity_b[1]
    if abs(gx) >= abs(gy):
        return "front" if gx > 0 else "back"
    return "side"


@dataclass(frozen=True)
class PostureSchedule:
    """One sit → rise → stand → sit cycle in control ticks (states after n steps, 50 Hz).

    robotd's controller per step: sitstand with the posture flag at 1 until ``rise_at``;
    flag 0 for robotd's ``sitstand_rise_s`` (1 s); velstand at zero twist from
    ``stand_at``; sitstand with flag 1 again from ``sit_at`` — exactly the v2 cycle. Phases
    (spec §10.3): *rise* from the rise command until 1 s after velstand takes over, *stand*
    until the sit command, *sit* the 3 s after it, *seated* a further hold so a fall that
    begins late in the sit window still shows. ``end`` leaves 18 s of self-righting after
    the latest possible fall.
    """

    hz: int = CONTROL_HZ
    settle_s: float = 2.0
    rise_s: float = 1.0
    after_handover_s: float = 1.0
    stand_s: float = 8.0
    sit_window_s: float = 3.0
    seated_hold_s: float = 2.0
    righting_s: float = RIGHTING_WINDOW_S

    def _ticks(self, seconds: float) -> int:
        return round(seconds * self.hz)

    @property
    def rise_at(self) -> int:
        return self._ticks(self.settle_s)

    @property
    def stand_at(self) -> int:
        return self.rise_at + self._ticks(self.rise_s)

    @property
    def rise_end(self) -> int:
        return self.stand_at + self._ticks(self.after_handover_s)

    @property
    def sit_at(self) -> int:
        return self.stand_at + self._ticks(self.stand_s)

    @property
    def sit_end(self) -> int:
        return self.sit_at + self._ticks(self.sit_window_s)

    @property
    def cycle_end(self) -> int:
        return self.sit_end + self._ticks(self.seated_hold_s)

    @property
    def end(self) -> int:
        return self.cycle_end + self._ticks(self.righting_s)

    def phase(self, tick: int) -> str:
        """The phase the state after ``tick`` control steps belongs to."""
        if tick < self.rise_at:
            return "settle"
        if tick < self.rise_end:
            return "rise"
        if tick < self.sit_at:
            return "stand"
        if tick < self.sit_end:
            return "sit"
        if tick < self.cycle_end:
            return "seated"
        return "after"

    def phase_start(self, phase: str) -> int:
        return {
            "rise": self.rise_at,
            "stand": self.rise_end,
            "sit": self.sit_at,
            "seated": self.sit_end,
        }[phase]

    def scripted_mode(self, tick: int) -> str:
        """robotd's controller for control step ``tick`` (1-based) of the script."""
        if tick < self.rise_at:
            return "sit"
        if tick < self.stand_at:
            return "rise"
        if tick < self.sit_at:
            return "up"
        return "sit"

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {
            name: getattr(self, name)
            for name in ("rise_at", "stand_at", "rise_end", "sit_at", "sit_end", "cycle_end", "end")
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostureSchedule:
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in fields})


def settle_bad(row: dict[str, Any]) -> bool:
    """Tipped before any action (the reset's drop into the seat): counted apart, never a fall.

    The runner's own verdict when the row carries it (it decided on the unrounded tilt and
    never started self-righting for such a duck); otherwise from the recorded fields.
    """
    if "settle_bad" in row:
        return bool(row["settle_bad"])
    return bool(row["settle_fallen"]) or row["settle_tilt_deg"] > SETTLE_BAD_TILT_DEG


def _in_cycle(tick: int | None, schedule: PostureSchedule) -> bool:
    return tick is not None and schedule.rise_at <= tick < schedule.cycle_end


def phase_falls(rows: list[dict[str, Any]], schedule: PostureSchedule) -> dict[str, Any]:
    """Per phase and rule: falls among the ducks that entered the phase still on script.

    A duck leaves the script at robotd's verdict (it is righting itself from then on), so a
    phase's ducks at risk are those without an earlier verdict and, for the 45° count,
    without an earlier 45° moment. ``per_cycle_rate`` divides by every settled cycle.
    """
    n = len(rows)
    out: dict[str, Any] = {}
    for phase in PHASES:
        start = schedule.phase_start(phase)
        entry: dict[str, Any] = {}
        for rule, key in (("robotd", "fall_tick"), ("tilt45", "tilt45_tick")):
            at_risk = [
                row
                for row in rows
                if (row["fall_tick"] is None or row["fall_tick"] >= start)
                and (row[key] is None or row[key] >= start)
            ]
            count = sum(
                1
                for row in at_risk
                if _in_cycle(row[key], schedule) and schedule.phase(row[key]) == phase
            )
            entry[rule] = proportion(count, len(at_risk)) | {
                "per_cycle_rate": round(count / n, 5) if n else None
            }
        out[phase] = entry
    out["any"] = {
        "robotd": proportion(sum(1 for row in rows if _in_cycle(row["fall_tick"], schedule)), n),
        "tilt45": proportion(sum(1 for row in rows if _in_cycle(row["tilt45_tick"], schedule)), n),
    }
    return out


def righting(rows: list[dict[str, Any]], schedule: PostureSchedule) -> dict[str, Any]:
    """Whether and when velstand got each fallen duck up again, by try, phase and landing."""
    hz = schedule.hz
    fallen = [row for row in rows if _in_cycle(row["fall_tick"], schedule)]

    def confirmed_s(row: dict[str, Any]) -> float | None:
        if row["righted_tick"] is None:
            return None
        return (row["righted_tick"] - row["fall_tick"]) / hz

    def tally(group: list[dict[str, Any]]) -> dict[str, Any]:
        seconds = [confirmed_s(row) for row in group]
        first = sum(1 for s in seconds if s is not None and s <= TRY_S)
        by_second = sum(1 for s in seconds if s is not None and s <= RIGHTING_WINDOW_S)
        return {
            "falls": len(group),
            "try1": proportion(first, len(group)),
            "by_try2": proportion(by_second, len(group)),
            "try2_after_try1_failed": proportion(by_second - first, len(group) - first),
        }

    up = [row for row in fallen if (s := confirmed_s(row)) is not None and s <= RIGHTING_WINDOW_S]
    out = tally(fallen)
    out["upright_after_s"] = describe(
        [(row["upright_tick"] - row["fall_tick"]) / hz for row in up], digits=3
    )
    out["confirmed_after_s"] = describe([confirmed_s(row) or 0.0 for row in up], digits=3)
    out["trunk_above_9cm_when_up"] = proportion(
        sum(1 for row in up if (row.get("z_righted") or 0.0) > OFFICIAL_UP_Z_M), len(up)
    )
    out["fell_again_after_up"] = proportion(
        sum(1 for row in up if row.get("refell_tick") is not None), len(up)
    )
    out["by_phase"] = {
        phase: tally([row for row in fallen if schedule.phase(row["fall_tick"]) == phase])
        for phase in PHASES
    }
    out["by_landing"] = {
        side: tally([row for row in fallen if landing(row.get("landed_gravity")) == side])
        for side in ("front", "back", "side")
    }
    return out


def summarize_posture(rows: list[dict[str, Any]], schedule: PostureSchedule) -> dict[str, Any]:
    """Segmented falls and self-righting for one task's posture rows (spec §10.3, §5.4)."""
    valid = [row for row in rows if not row.get("reset")]
    unsettled = [row for row in valid if settle_bad(row)]
    settled = [row for row in valid if not settle_bad(row)]
    return {
        "episodes_total": len(rows),
        "resets": len(rows) - len(valid),
        "settle_tilt": proportion(len(unsettled), len(valid)),
        "settled": len(settled),
        "phases": phase_falls(settled, schedule),
        "righting": righting(settled, schedule),
    }
