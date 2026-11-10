"""Fit a body's leg-load bands from its bench recording, and check them by replaying the recording through proprio.

Every band sits between two **measured levels** of the leg load (``RobotState.leg_load``, the body's own units):

* in the hand — ``H``: the level the load stays under for a whole claim hold in 95 % of the in-hand windows
  (:func:`hand_ceiling`; the ``air_*`` segments and the held part of the lifts);
* on the floor — ``F``: the level no window of a claim's hold ever stays under (:func:`floor_guard`: the lowest rolling
  maximum over the floor segments, the stops after the walks included), and ``P05`` the floor's 5th percentile.

A claim is possible only when the floor is clearly above the hand: ``F ≥ MIN_RATIO × H``. Then (:data:`AT`):
``lift_load`` at 40 % of the way from H to F, ``unsure_load`` at 60 %, the floor band's low edge at 80 % of the way to
P05, the reverse spin-up's level at 30 %. The fractions were chosen so that the levels proprio's notes give for the
simulator (≈ 9 standing and ≈ 13 seated in the hand, ≈ 35 standing and ≈ 25 seated on the floor) land by the
simulator's hand-set bands (20 / 24 / 28 standing, 18.5 / 20.5 / 21.5 seated: tests/test_bringup_calibrate.py).
The flailing level and the band's high edge sit between the floor's highest half-second mean and the level the legs
work at in the air after a walk is stopped there — when that is no higher, flailing cannot be told from standing by
load and the claim is put out of reach. A seat that does not load the legs more than a hand does turns the seated claim
off (levels 0); the standing claim cannot be done without: no calibration is written.

The **replay** then runs :class:`~duckmind.duckbody.proprio.Proprio` with the fitted bands over every recorded
repetition: no pickup may be claimed on the floor or during the walk, every lift must be claimed between its two cues
and set down after the second. Only a fit that passes is :attr:`Fit.usable`.

Pure (the recording's lines in, numbers out).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.bringup.record import SIT, STAND, segments
from duckmind.duckbody.calibration import SIMULATOR, LoadBands
from duckmind.duckbody.proprio import Proprio, ProprioConfig
from duckmind.duckbody.robotd_client import RobotState

MIN_RATIO = (
    1.3  # the floor's guard level over the hand's ceiling: under this the two cannot be told apart
)
AT = {
    "reverse": 0.3,
    "lift": 0.4,
    "unsure": 0.6,
    "floor_low": 0.8,
    "flail": 0.65,
    "floor_high": 0.9,
}
HAND_AFTER_LIFT_S = (
    2.0  # after the lift cue the person is still lifting: not yet "in the hand, still"
)
SPINUP_S = 1.0  # a walk's first second is its spin-up
FLAIL_AFTER_S = 1.0
MEAN_S = 0.5
NEEDED = (
    "air_stand",
    "air_sit",
    "floor_stand",
    "floor_sit",
    "cycle",
    "lift_stand",
    "lift_sit",
    "walk",
)
WALK_REPS = 3
LIFT_REPS = 3
OFF_SCALE = 3.0  # an out-of-reach level: this many times the highest floor load


@dataclass(frozen=True)
class Tick:
    t: float  # seconds from the repetition's first tick
    state: RobotState

    @property
    def load(self) -> float | None:
        return self.state.leg_load

    @property
    def walking(self) -> bool:
        return abs(self.state.requested[0]) >= ProprioConfig.walk_vx

    @property
    def gyro(self) -> float | None:
        g = self.state.gyro
        return None if g is None else math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])


@dataclass(frozen=True)
class Rep:
    """One recorded repetition: its ticks and its cues (name → seconds)."""

    segment: str
    rep: int
    ticks: tuple[Tick, ...]
    marks: Mapping[str, float]


def reps_of(lines: Sequence[Mapping[str, Any]]) -> list[Rep]:
    out: list[Rep] = []
    for name, rep, group in segments(lines):
        states = [
            RobotState.from_wire(line["state"])
            for line in group
            if "state" in line and "tof" not in line
        ]
        if not states:
            continue
        t0 = states[0].t_ns
        marks = {
            str(line["mark"]): (int(line["t_ns"]) - t0) / 1e9 for line in group if "mark" in line
        }
        out.append(Rep(name, rep, tuple(Tick((s.t_ns - t0) / 1e9, s) for s in states), marks))
    return out


Run = list[tuple[float, float]]  # (t, load) of consecutive ticks


def runs(rep: Rep, keep: Any) -> list[Run]:
    """The stretches of consecutive ticks ``keep`` accepts (a window never spans a gap)."""
    out: list[Run] = []
    current: Run = []
    for tick in rep.ticks:
        if tick.load is not None and keep(tick):
            current.append((tick.t, tick.load))
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def rolling(run: Run, window_s: float, how: Any) -> list[float]:
    """``how`` (max, min, mean) over every full window of ``window_s`` in the run."""
    out: list[float] = []
    start = 0
    for end, (t, _) in enumerate(run):
        while run[start][0] < t - window_s:
            start += 1
        if t - run[0][0] >= window_s * 0.95:
            values = [v for _, v in run[start : end + 1]]
            out.append(how(values))
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    at = q * (len(ordered) - 1)
    lo, hi = int(math.floor(at)), int(math.ceil(at))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def hand_ceiling(many: Sequence[Run], hold_s: float) -> float:
    """The level a load in the hand is under for a whole hold in 95 % of the windows."""
    values = [v for run in many for v in rolling(run, hold_s, max)]
    return quantile(values, 0.95)


def floor_guard(many: Sequence[Run], hold_s: float) -> float:
    """The level no window of a hold on the floor ever stays under."""
    values = [v for run in many for v in rolling(run, hold_s, max)]
    return min(values) if values else math.nan


def describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        **{
            f"p{int(q * 100):02d}": round(quantile(values, q), 2)
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _flat(many: Sequence[Run]) -> list[float]:
    return [v for run in many for _, v in run]


@dataclass(frozen=True)
class Separation:
    """Two levels that must be apart for a claim: ``low`` (the hand's, the floor's) under ``high``."""

    name: str
    low_name: str
    low: float
    high_name: str
    high: float
    needed: bool

    @property
    def ratio(self) -> float:
        return self.high / self.low if self.low > 0 else math.inf

    @property
    def ok(self) -> bool:
        return math.isfinite(self.low) and math.isfinite(self.high) and self.ratio >= MIN_RATIO

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low_name,
            "low_level": round(self.low, 2),
            "high": self.high_name,
            "high_level": round(self.high, 2),
            "ratio": None if not math.isfinite(self.ratio) else round(self.ratio, 2),
            "separated": self.ok,
            "needed": self.needed,
        }


@dataclass
class Fit:
    bands: LoadBands | None = None
    levels: dict[str, dict[str, float | int]] = field(default_factory=dict)
    separations: list[Separation] = field(default_factory=list)
    cannot: list[str] = field(
        default_factory=list
    )  # what this body's load cannot tell apart (and what follows)
    missing: list[str] = field(default_factory=list)  # why no calibration can be written yet
    replay: dict[str, Any] = field(default_factory=dict)
    gyro: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.bands is not None and not self.missing

    def to_json(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "bands": None if self.bands is None else self.bands.to_json(),
            "simulator_bands": SIMULATOR.proprio.to_json(),
            "levels": self.levels,
            "separations": [s.to_json() for s in self.separations],
            "cannot_separate": self.cannot,
            "missing": self.missing,
            "replay": self.replay,
            "gyro": self.gyro,
            "rule": {"min_ratio": MIN_RATIO, "at": AT},
        }


def _between(low: float, high: float, fraction: float) -> float:
    return low + fraction * (high - low)


def _posture_fit(
    hand: Sequence[Run], floor: Sequence[Run], lift_hold_s: float, unsure_hold_s: float
) -> tuple[float, float, float, float]:
    """(H, F at the lift hold, F at the unsure hold, the floor's 5th percentile)."""
    return (
        hand_ceiling(hand, lift_hold_s),
        floor_guard(floor, lift_hold_s),
        floor_guard(floor, unsure_hold_s),
        quantile(_flat(floor), 0.05),
    )


def fit(lines: Sequence[Mapping[str, Any]], cfg: ProprioConfig | None = None) -> Fit:
    """The bands for the body the recording was taken on, with everything the report shows."""
    cfg = cfg or ProprioConfig()
    reps = reps_of(lines)
    out = Fit()
    by_name: dict[str, list[Rep]] = {}
    for rep in reps:
        by_name.setdefault(rep.segment, []).append(rep)
    for name in NEEDED:
        if name not in by_name:
            out.missing.append(f"the segment {name} was not recorded")
    if len(by_name.get("walk", [])) < WALK_REPS and "walk" in by_name:
        out.missing.append(f"fewer than {WALK_REPS} walks were recorded")

    def collect(names: Sequence[str], keep: Any) -> list[Run]:
        return [run for name in names for rep in by_name.get(name, []) for run in runs(rep, keep)]

    def standing(tick: Tick) -> bool:
        return tick.state.policy in STAND and not tick.walking

    def seated(tick: Tick) -> bool:
        return tick.state.policy in SIT

    def in_hand(rep: Rep) -> Any:
        lift, down = rep.marks.get("lift"), rep.marks.get("set_down")
        return lambda tick: (
            lift is not None and down is not None and lift + HAND_AFTER_LIFT_S <= tick.t < down
        )

    def before_lift(rep: Rep) -> Any:
        lift = rep.marks.get("lift")
        return lambda tick: lift is not None and tick.t < lift

    hand_stand = collect(["air_stand"], lambda t: not t.walking)
    hand_sit = collect(["air_sit"], lambda t: True)
    floor_stand = collect(["floor_stand", "walk"], standing)
    floor_sit = collect(["floor_sit"], seated)
    for rep in by_name.get("lift_stand", []):
        hand_stand += runs(rep, in_hand(rep))
        floor_stand += runs(rep, lambda t, ok=before_lift(rep): ok(t) and standing(t))
    for rep in by_name.get("lift_sit", []):
        hand_sit += runs(rep, in_hand(rep))
        floor_sit += runs(rep, lambda t, ok=before_lift(rep): ok(t) and seated(t))
    walk_floor = collect(["walk"], lambda t: t.walking)
    walk_air = collect(["air_walk"], lambda t: t.walking)
    flail: list[Run] = []
    for rep in by_name.get("air_walk", []):
        stops = [
            t.t
            for a, t in zip(rep.ticks, rep.ticks[1:], strict=False)
            if a.walking and not t.walking
        ]
        if stops:
            flail += runs(rep, lambda t, at=stops[-1]: t.t >= at + FLAIL_AFTER_S and standing(t))
    out.levels = {
        "in the hand, standing": describe(_flat(hand_stand)),
        "in the hand, seated": describe(_flat(hand_sit)),
        "on the floor, standing (stopped)": describe(_flat(floor_stand)),
        "on the floor, seated": describe(_flat(floor_sit)),
        "walking on the floor": describe(_flat(walk_floor)),
        "stepping in the air": describe(_flat(walk_air)),
        "in the air after the walk stopped": describe(_flat(flail)),
    }
    if not hand_stand or not floor_stand:
        if not out.missing:
            out.missing.append(
                "no leg load was recorded in the hand or on the floor (currents_ma absent?)"
            )
        return out

    h, f_lift, f_unsure, p05 = _posture_fit(
        hand_stand, floor_stand, cfg.lift_hold_s, cfg.unsure_hold_s
    )
    stand = Separation("standing pickup", "in the hand", h, "on the floor", f_lift, needed=True)
    out.separations.append(stand)
    if not stand.ok:
        out.missing.append(
            f"standing in the hand ({h:.1f}) cannot be told from standing on the floor ({f_lift:.1f}) by leg "
            f"load: the ratio is {stand.ratio:.2f}, under {MIN_RATIO}"
        )
        return out
    lift = _between(h, f_lift, AT["lift"])
    unsure = max(lift, _between(h, f_unsure, AT["unsure"]))
    low = max(unsure, _between(h, p05, AT["floor_low"]))
    reverse = min(unsure, _between(h, f_lift, AT["reverse"]))

    # the seat
    sit_lift = sit_unsure = sit_low = 0.0
    sit_high = OFF_SCALE * max(_flat(floor_sit), default=max(_flat(floor_stand)))
    if hand_sit and floor_sit:
        hs, fs_lift, fs_unsure, ps05 = _posture_fit(
            hand_sit, floor_sit, cfg.lift_hold_sit_s, cfg.unsure_hold_s
        )
        seat = Separation("seated pickup", "in the hand", hs, "on the floor", fs_lift, needed=False)
        out.separations.append(seat)
        if seat.ok:
            sit_lift = _between(hs, fs_lift, AT["lift"])
            sit_unsure = max(sit_lift, _between(hs, fs_unsure, AT["unsure"]))
            sit_low = max(sit_unsure, _between(hs, ps05, AT["floor_low"]))
        else:
            out.cannot.append(
                f"a seated duck in the hand ({hs:.1f}) from one seated on the floor ({fs_lift:.1f}): the seated "
                "light-legs claim is turned off (levels 0); a seated pickup is then seen only by robotd's own "
                "pickup pause, and a seated duck in the hand that is asked to stand will try to"
            )

    # flailing in the air after a stop, and the floor band's high edge
    floor_top = max(
        (v for run in floor_stand for v in rolling(run, MEAN_S, _mean)), default=math.nan
    )
    floor_max = max(_flat(floor_stand))
    air = quantile([v for run in flail for v in rolling(run, MEAN_S, _mean)], 0.10)
    flailing = Separation(
        "legs flailing in the air after a stop",
        "standing on the floor",
        floor_top,
        "in the air",
        air,
        needed=False,
    )
    if flail:
        out.separations.append(flailing)
    if flail and flailing.ok:
        flail_load = _between(floor_top, air, AT["flail"])
        high = max(_between(floor_top, air, AT["floor_high"]), 1.2 * floor_max)
    else:
        flail_load = high = OFF_SCALE * floor_max
        out.cannot.append(
            "legs flailing in the air after a stop from standing on the floor"
            + (
                f" ({air:.1f} against {floor_top:.1f})"
                if flail
                else " (no such stretch was recorded)"
            )
            + ": the flailing claim is put out of reach; a duck stepping in the air is then stopped by its quiet "
            "trunk and its leg speed, and claimed once the legs go light"
        )

    # what load never tells apart (owner B3's own finding in the simulator): stepping in the air from walking
    if walk_floor and walk_air:
        a, b = quantile(_flat(walk_air), 0.5), quantile(_flat(walk_floor), 0.5)
        ratio = max(a, b) / min(a, b) if min(a, b) > 0 else math.inf
        if ratio < MIN_RATIO:
            out.cannot.append(
                f"stepping in the air ({a:.1f}) from walking on the floor ({b:.1f}) by leg load (as in the "
                "simulator): that is the quiet-trunk test's job, see the gyro numbers"
            )
    out.gyro = _gyro(by_name, cfg)
    if out.gyro.get("warning"):
        out.cannot.append(str(out.gyro["warning"]))

    out.bands = LoadBands(
        unsure_load=round(unsure, 1),
        unsure_load_sit=round(sit_unsure, 1),
        unsure_load_reverse=round(reverse, 1),
        lift_load=round(lift, 1),
        lift_load_sit=round(sit_lift, 1),
        flail_load=round(flail_load, 1),
        floor_band=(round(low, 1), round(high, 1)),
        floor_band_sit=(round(sit_low, 1), round(max(sit_high, sit_low + 1.0), 1)),
    )
    out.replay = replay(reps, cfg.calibrated(out.bands))
    out.missing += out.replay["failures"]
    return out


def _gyro(by_name: Mapping[str, Sequence[Rep]], cfg: ProprioConfig) -> dict[str, Any]:
    """The quiet-trunk test's two levels (rad/s RMS over its window): a duck stepping in the air against one walking
    on the floor. Not fitted (the level is a physical rate, not a body's scale), only checked."""

    def rms(names: Sequence[str]) -> list[float]:
        out: list[float] = []
        for name in names:
            for rep in by_name.get(name, []):
                run = [
                    (t.t, t.gyro)
                    for t in rep.ticks
                    if t.gyro is not None and t.walking and t.t >= SPINUP_S + 1.0
                ]
                out += rolling(
                    run, cfg.quiet_window_s, lambda v: math.sqrt(sum(x * x for x in v) / len(v))
                )
        return out

    air, floor = rms(["air_walk"]), rms(["walk"])
    result: dict[str, Any] = {"quiet_gyro": cfg.quiet_gyro}
    if air:
        result["stepping_in_the_air_p95"] = round(quantile(air, 0.95), 3)
    if floor:
        result["walking_on_the_floor_p05"] = round(quantile(floor, 0.05), 3)
    if floor and quantile(floor, 0.05) <= cfg.quiet_gyro:
        result["warning"] = (
            f"this body's trunk rocks less than the quiet-trunk level while walking on the floor (RMS p05 "
            f"{quantile(floor, 0.05):.3f} rad/s against {cfg.quiet_gyro}): walks would be stopped as 'unloaded'"
        )
    elif air and quantile(air, 0.95) >= cfg.quiet_gyro:
        result["warning"] = (
            f"held in the air and stepping, this body's trunk is not quiet (RMS p95 {quantile(air, 0.95):.3f} "
            f"rad/s against {cfg.quiet_gyro}; a hand shakes more than the simulator's): the quiet-trunk test "
            "will not see a duck stepping in the hand"
        )
    return result


def replay(reps: Sequence[Rep], cfg: ProprioConfig) -> dict[str, Any]:
    """proprio with ``cfg`` over every repetition: what it would have claimed, against what the person did."""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    notes: list[str] = []
    lifts = {"lift_stand": [0, 0], "lift_sit": [0, 0]}
    delays: list[float] = []
    for rep in reps:
        proprio = Proprio(cfg)
        edges: list[tuple[float, str, str]] = []
        fell = False
        for tick in rep.ticks:
            got = proprio.step(tick.state)
            fell = fell or tick.state.fallen
            edges += [(tick.t, edge, proprio.why_unsure) for edge in got.edges]
        names = [e for _, e, _ in edges]
        row: dict[str, Any] = {
            "segment": rep.segment,
            "rep": rep.rep,
            "seconds": round(rep.ticks[-1].t, 1),
            "edges": [[round(t, 2), e] for t, e, _ in edges],
        }
        label = f"{rep.segment} #{rep.rep + 1}"
        if rep.segment in lifts:
            lift, down = rep.marks.get("lift", math.inf), rep.marks.get("set_down", math.inf)
            claimed = [t for t, e, _ in edges if e == "lifted"]
            by_robotd = any(e == "lifted" and why == "robotd" for _, e, why in edges)
            early = [t for t in claimed if t < lift]
            good = [t for t in claimed if lift <= t < down + 1.0]
            set_down = [t for t, e, _ in edges if e == "set_down" and t >= down]
            lifts[rep.segment][1] += 1
            row.update(claimed=bool(good), by_robotd=by_robotd, set_down=bool(set_down))
            if early:
                failures.append(
                    f"{label}: a pickup was claimed {lift - early[0]:.1f} s before the hand came"
                )
            if good:
                lifts[rep.segment][0] += 1
                delays.append(good[0] - lift)
                row["claimed_after_cue_s"] = round(good[0] - lift, 2)
                if not set_down:
                    failures.append(f"{label}: the set-down was never sensed")
            elif cfg.lift_load_sit > 0 or rep.segment == "lift_stand":
                failures.append(f"{label}: the pickup was not sensed")
        elif rep.segment.startswith("air_") or rep.segment.startswith("tof_"):
            row["note"] = "in the hand (no claim is judged here)"
        else:
            if "lifted" in names:
                failures.append(f"{label}: a pickup was claimed with the duck on the floor")
            if rep.segment == "walk" and "unloaded" in names:
                failures.append(f"{label}: the walk would have been stopped as 'unloaded'")
            if rep.segment == "cycle" and "unloaded" in names:
                # (a seat that lands in a bounce unloads the legs for a moment: no claim, the walk only waits)
                notes.append(
                    f"{label}: the legs went light for a moment in the sit-stand cycle ('unloaded')"
                )
            elif rep.segment != "walk" and "unloaded" in names:
                failures.append(f"{label}: 'unloaded' with the duck still on the floor")
            if fell or "fell" in names:
                failures.append(f"{label}: the duck fell")
        rows.append(row)
    return {
        "rows": rows,
        "failures": failures,
        "notes": notes,
        "cycles": sum(rep.segment == "cycle" for rep in reps),
        "lifts": {k: {"claimed": v[0], "of": v[1]} for k, v in lifts.items()},
        "claim_after_cue_s": None
        if not delays
        else {"median": round(quantile(delays, 0.5), 2), "max": round(max(delays), 2)},
    }


__all__ = [
    "AT",
    "MIN_RATIO",
    "NEEDED",
    "Fit",
    "Rep",
    "Separation",
    "Tick",
    "describe",
    "fit",
    "floor_guard",
    "hand_ceiling",
    "quantile",
    "replay",
    "reps_of",
    "rolling",
    "runs",
]
