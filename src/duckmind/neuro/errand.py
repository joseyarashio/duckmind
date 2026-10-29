"""An errand: somewhere a person pointed to, as the duck heard it (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md).

``go_there`` is a sense input, not a motor command. What the brain is given is a place in its OWN odometry
frame (``target`` x, y in m; see :func:`duckmind.simweb.neuro_live.NeuroDriver.go_there` for how the page's
floor point becomes one, and how wrong it can be). The brain keeps it as an errand and goes on deciding from
its own senses:

* it takes the walk (``wander``, with the errand as the walk's goal) when nothing the selector protects has
  the body: never from sleep, a fright (fear ≥ 0.5), a walk home to charge, a docking (the selector's
  claim rules, ``Selector.claim_next``);
* the walk is the ordinary one: the lasting map's route, the wall memory's bend, the reflexes, the
  clutter escape. It holds on like a walk to a thing (``walking_to``) until a call, a stall or a rival
  that is urgent breaks it off;
* it may get distracted: whatever takes the body from the walk is the duck's own business. The errand
  waits; once the duck has done something else for :data:`REASSERT_S` it takes the walk again;
* it ends as ``arrived`` (within :data:`ARRIVE_M` of the place by its own belief, for :data:`ARRIVE_HOLD_S`),
  or ``gave_up`` with a reason: ``stuck`` (no nearer in :data:`STALL_S`), ``timeout`` (the budget), a
  blocker it could not get past while waiting (``asleep``, ``afraid``, ``low_battery``, ``docking``,
  ``busy``), ``lifted`` (a pick-up moves the duck without its odometry knowing: the place is lost),
  ``remote`` (a person took the legs), ``stopped``, ``replaced``.

Pure and clock-injected (``t_s`` is the brain's own clock). No world, no clock, no dice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

ARRIVE_M = 0.18  # within this of the place (trunk centre, by its own belief): it is there
ARRIVE_HOLD_S = 0.3  # … for this long
REASSERT_S = 3.0  # something else has had the body this long: the errand takes the walk again
START_WAIT_S = 10.0  # it could not take the walk in this long: it gives up, saying what held it
STALL_S = 25.0  # walking this long and no nearer than its best by STALL_PROGRESS_M (detours included): stuck
DISTRACTED_MAX_S = 40.0  # somebody else's business has had the body this long since it last walked: it gives up
STALL_PROGRESS_M = 0.10
BUDGET_BASE_S = 45.0  # the longest the errand may last: this plus BUDGET_PER_M per metre of the way
BUDGET_PER_M = 15.0
MAX_RANGE_M = 12.0  # a place farther than this is not an errand (the home is a few metres across)
OUTCOMES_KEEP = 24

REASONS = (
    "stuck",
    "timeout",
    "asleep",
    "afraid",
    "low_battery",
    "docking",
    "busy",
    "lifted",
    "remote",
    "stopped",
    "replaced",
)

WALK = "wander"  # the act that carries the errand
ERRAND_CLAIM = 1.2  # the walk's claim on the selector's command channel (above any wish of its own)
ARRIVAL = "look_around"  # the act it takes on arriving: it stands and looks about


@dataclass(frozen=True, slots=True)
class ErrandOut:
    """What the brain does with the errand this tick."""

    goal: tuple[float, float] | None = None  # (bearing rad from the trunk, range m): steer the walk to it
    claim: str | None = None  # an act to claim for the next tick (Selector.claim_next)
    arrived: bool = False  # it has just arrived (the walk stops this tick)
    walking: bool = False  # the walk is the errand's now (a walk to a place, like a walk to a thing)


_QUIET = ErrandOut()


@dataclass(slots=True)
class _Run:
    id: int
    x: float
    y: float
    who: str | None
    t0: float
    start_range: float
    budget_s: float
    state: str = "waiting"  # waiting (not yet walking) | walking
    best: float = math.inf
    stall_s: float = 0.0  # walking time without getting nearer
    away_s: float = 0.0  # time the walk was somebody else's (total)
    gap_s: float = 0.0  # … and since it last walked
    walked_s: float = 0.0
    near_s: float = 0.0
    last_range: float = math.inf
    last_t: float = 0.0
    walk_seen: bool = False  # it has been on the walk at least once


class Errand:
    """The errand a brain keeps (at most one: a new one replaces the old)."""

    def __init__(self) -> None:
        self._run: _Run | None = None
        self._seq = 0
        self.outcomes: list[dict[str, Any]] = []

    # ── the page's side ──

    @property
    def active(self) -> bool:
        return self._run is not None

    def start(
        self, x: float, y: float, t_s: float, odom: tuple[float, float, float], who: str | None = None
    ) -> int:
        """A place to go to (odometry frame). Replaces an errand under way. Returns its id."""
        if self._run is not None:
            self._end("gave_up", "replaced", t_s)
        rng = math.hypot(float(x) - odom[0], float(y) - odom[1])
        self._seq += 1
        self._run = _Run(
            id=self._seq,
            x=float(x),
            y=float(y),
            who=who,
            t0=t_s,
            start_range=rng,
            budget_s=BUDGET_BASE_S + BUDGET_PER_M * rng,
            best=rng,
            last_range=rng,
            last_t=t_s,
        )
        return self._seq

    def cancel(self, reason: str, t_s: float) -> bool:
        """End the errand from outside (``stopped``, ``remote``, ``lifted``). False when there is none."""
        if self._run is None:
            return False
        self._end("gave_up", reason, t_s)
        return True

    # ── the brain's side: one call per tick ──

    def step(
        self,
        t_s: float,
        dt: float,
        odom: tuple[float, float, float],
        *,
        behaviour: str,
        asleep: bool,
        afraid: bool,
        homing: bool,
        docking: bool,
        lifted: bool,
        can_walk: bool,
    ) -> ErrandOut:
        run = self._run
        if run is None:
            return _QUIET
        if lifted:
            self._end("gave_up", "lifted", t_s)
            return _QUIET
        rng = math.hypot(run.x - odom[0], run.y - odom[1])
        run.last_range, run.last_t = rng, t_s
        elapsed = t_s - run.t0
        if elapsed > run.budget_s:
            self._end("gave_up", "timeout", t_s)
            return _QUIET
        on_walk = behaviour == WALK
        if on_walk:
            run.state = "walking"
            run.walk_seen = True
            run.gap_s = 0.0
            run.walked_s += dt
            if rng < run.best - STALL_PROGRESS_M:
                run.best, run.stall_s = rng, 0.0
            else:
                run.stall_s += dt  # only walking time counts: a distraction is not a stall
                if run.stall_s > STALL_S:
                    self._end("gave_up", "stuck", t_s)
                    return _QUIET
            # arrived: close by its own belief and it has stayed there a moment
            run.near_s = run.near_s + dt if rng <= ARRIVE_M else 0.0
            if run.near_s >= ARRIVE_HOLD_S:
                self._end("arrived", None, t_s)
                return ErrandOut(claim=ARRIVAL, arrived=True)
            bearing = math.atan2(run.y - odom[1], run.x - odom[0]) - odom[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            return ErrandOut(goal=(bearing, rng), walking=True)
        # the walk is somebody else's (waiting to begin, or distracted)
        run.near_s = 0.0
        run.gap_s += dt
        run.away_s += dt
        if not run.walk_seen:
            run.state = "waiting"
        why = _blocker(asleep, afraid, homing, docking, can_walk)
        if not run.walk_seen and elapsed > START_WAIT_S:
            self._end("gave_up", why, t_s)
            return _QUIET
        if run.walk_seen and run.gap_s > DISTRACTED_MAX_S:
            self._end("gave_up", why, t_s)
            return _QUIET
        # it takes the walk: at once when it has not begun, after REASSERT_S of other business when it
        # was walking and something took the body from it
        if not run.walk_seen or run.gap_s >= REASSERT_S:
            return ErrandOut(claim=WALK)
        return _QUIET

    # ── the end ──

    def _end(self, state: str, reason: str | None, t_s: float) -> None:
        run = self._run
        if run is None:
            return
        self._run = None
        self.outcomes.append(
            {
                "id": run.id,
                "state": state,
                "reason": reason,
                "elapsed_s": round(t_s - run.t0, 1),
                "walked_s": round(run.walked_s, 1),
                "away_s": round(run.away_s, 1),
                "start_range_m": round(run.start_range, 2),
                "final_range_m": round(run.last_range, 2),
                "target": {"x_m": round(run.x, 3), "y_m": round(run.y, 3)},
                "who": run.who,
            }
        )
        del self.outcomes[:-OUTCOMES_KEEP]

    # ── telemetry ──

    def view(self, t_s: float | None = None) -> dict[str, Any]:
        run = self._run
        last = self.outcomes[-1] if self.outcomes else None
        out: dict[str, Any] = {"state": "idle", "last": last, "n": self._seq}
        if run is not None:
            out.update(
                {
                    "state": run.state,
                    "id": run.id,
                    "target": {"x_m": round(run.x, 3), "y_m": round(run.y, 3)},
                    "range_m": round(run.last_range, 3),
                    "start_range_m": round(run.start_range, 2),
                    "budget_s": round(run.budget_s, 1),
                    "away_s": round(run.away_s, 1),
                }
            )
            if t_s is not None:
                out["elapsed_s"] = round(t_s - run.t0, 1)
        elif last is not None:
            out["state"] = last["state"]
            out["id"] = last["id"]
            out["reason"] = last["reason"]
        return out


def _blocker(asleep: bool, afraid: bool, homing: bool, docking: bool, can_walk: bool) -> str:
    """What holds the body now, in the order the selector's claim rules refuse it."""
    if asleep:
        return "asleep"
    if afraid:
        return "afraid"
    if docking:
        return "docking"
    if homing or not can_walk:
        return "low_battery"
    return "busy"


__all__ = [
    "ARRIVAL",
    "ARRIVE_M",
    "BUDGET_BASE_S",
    "BUDGET_PER_M",
    "Errand",
    "ERRAND_CLAIM",
    "ErrandOut",
    "MAX_RANGE_M",
    "REASONS",
    "WALK",
]
