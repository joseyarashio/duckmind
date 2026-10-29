"""After a fall: getting back up (spec §5.4; owner B1, B2; phase 1 ``righting.md``).

:class:`Righting` is stepped once per ``robot.state`` tick with :class:`~duckmind.duckbody.
proprio.Proprio`'s output. One fall episode (proprio's ``fell`` … ``up``; a flag that flickers is
one fall) runs:

1. **On the trigger** — A: ``safety.fallen`` rose; B (owner B1): tilt over 40° for 1 s while
   robotd still says upright — duck-body zeroes ``robot.move`` (the body loop holds still while
   :attr:`RightingOut.active`) and records ``body.fell {during, ctx, trigger}``. If the posture
   network holds the seat (policy ``sit``) it sends ``sit_toggle`` so velstand takes over at zero
   command (spec §5.4 step 2).
2. **Try 1** — wait up to 8 s for "up": tilt < 25°, not fallen, policy walk/stand, held 1 s.
3. **Try 2** — owner B2, ON by default: ``sit_toggle`` (sit), 2 s, ``sit_toggle`` (stand), then
   up to 9 s (1 s rise + 8 s) for "up". Phase 1 rescued 32 of 36 ducks that otherwise needed a
   hand this way (at the 18th second; here it runs at the 8th). With ``sitstand=False`` try 2 is
   the spec's "2 s pause + 8 s more". The largest leg load seen during try 2 is kept
   (:attr:`Righting.last`) so the motor load can be measured. **Motor guard**: in the
   simulator 97.5 % of these tries push some leg motor past the real XL330's stall (0.64 N·m;
   median 0.26 s, at most 1.32 s over it, ``body_righting.json``) — so try 2 gives up as soon as
   any leg motor has been past :attr:`RightingConfig.stall_current` for
   :attr:`RightingConfig.overload_max_s` in all (3 of 187 simulated tries): ``ok: false``,
   ``overload: true``, and Duck Core calls for help. The threshold is the simulator's proxy
   (``currents_ma`` = |force| × 100, as owner B3's lift thresholds) until the board is measured.
4. **Result** — ``body.righting {ok, ms, tries, via: self, how: wait|sitstand, hard}``; ``hard``
   is the owner's "辛苦" (spec D3): more than 4 s, 2 tries, or help. Not up after try 2 →
   ``{ok: false, ms, tries: 2}``; Duck Core calls for help. If the duck is up later (a hand or
   the page's button righted it) → ``{ok: true, via: help}``.
5. **Waiting for help, with patience** (2026-10-09): when the wait began without the motor guard
   tripping, the sit-stand try is repeated every :attr:`RightingConfig.retry_every_s` (30 s), at
   most :attr:`RightingConfig.retry_max` (3) times, then the duck waits for help for good. A retry
   that works reports ``{ok: true, via: self, tries: 2 + n, how: sitstand}``; a retry that fails
   reports nothing new (the failure was already told). After an overload trip it never retries
   (the motors were past their stall: trying again would only heat them).

Pure: the tick's ``t_ns`` is the clock; percept bodies come out without stamps
(:mod:`duckmind.duckbody.aggregate` stamps them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.proprio import ProprioOut
from duckmind.duckbody.robotd_client import LEG_JOINTS, RobotState

NS = 1_000_000_000
MS = 1_000_000


@dataclass(frozen=True)
class RightingConfig:
    try_s: float = 8.0  # one try (the official fallen_too_long)
    pause_s: float = 2.0  # spec: between the tries when sitstand is off
    sit_s: float = 2.0  # B2: seated this long before standing again
    watch_s: float = 9.0  # B2: 1 s rise + 8 s
    up_hold_s: float = 1.0  # proprio's up hold (for the reported ms)
    hard_ms: int = 4000
    sitstand: bool = True  # owner B2: ON by default
    overload_guard: bool = True  # stop try 2 when a leg motor stays past its stall too long
    stall_current: float = 64.0  # the XL330's 0.64 N·m stall in the sim proxy (board: recalibrate)
    overload_max_s: float = 1.0
    retry_every_s: float = 30.0  # waiting for help: the sit-stand try again this often ...
    retry_max: int = 3  # ... at most this many times, never after an overload trip (0: never)


@dataclass(frozen=True)
class RightingOut:
    active: bool = False  # an episode is open: the body holds still, no acts
    do: str | None = None  # a robot.do skill to send now
    events: tuple[dict[str, Any], ...] = ()


@dataclass
class Righting:
    cfg: RightingConfig = field(default_factory=RightingConfig)

    def __post_init__(self) -> None:
        self.phase = "idle"  # idle / try1 / sit / rise / pause / try2 / waiting_help
        self.t_fall = 0
        self.t_phase = 0
        self.trigger = ""
        self.tries = 0
        self.how = "wait"
        self.peak_load = 0.0
        self.over_s = 0.0  # try 2: time any leg motor spent past the stall
        self.t_wait = 0  # when waiting_help began
        self.overloaded = False  # the wait began with the motor guard's trip: no retries
        self.retries = 0  # sit-stand retries made from waiting_help in this episode
        self.retrying = False  # a retry is under way (its failure is not told again)
        self.last_t: int | None = None
        self.last: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    def _result(self, t: int, ok: bool, via: str | None, overload: bool = False) -> dict[str, Any]:
        up_at = t - int(self.cfg.up_hold_s * NS) if ok else t
        ms = max(0, (up_at - self.t_fall) // MS)
        event: dict[str, Any] = {
            "kind": "body.righting",
            "ok": ok,
            "ms": int(ms),
            "tries": self.tries,
        }
        if ok and via is not None:
            event["via"] = via
        event["how"] = self.how
        if ok:
            event["hard"] = bool(ms > self.cfg.hard_ms or self.tries >= 2 or via == "help")
        if overload:
            event["overload"] = True
        self.last = {
            "ok": ok,
            "ms": int(ms),
            "tries": self.tries,
            "via": via,
            "how": self.how,
            "trigger": self.trigger,
            "peak_load_try2": round(self.peak_load, 1),
            "over_stall_s": round(self.over_s, 2),
            "overload": overload,
        }
        return event

    def step(
        self,
        state: RobotState,
        prox: ProprioOut,
        *,
        during: str = "still",
        ctx: dict[str, Any] | None = None,
    ) -> RightingOut:
        c = self.cfg
        t = state.t_ns
        events: list[dict[str, Any]] = []
        do: str | None = None
        if "fell" in prox.edges and self.phase in ("idle", "waiting_help"):
            if self.phase == "waiting_help":
                events.append(self._result(t, False, None))  # the old episode ends unresolved
            self.phase, self.t_fall, self.t_phase = "try1", t, t
            self.trigger = prox.fell_trigger or "fallen"
            self.tries, self.how, self.peak_load, self.over_s = 1, "wait", 0.0, 0.0
            self.overloaded, self.retries, self.retrying = False, 0, False
            events.append(
                {
                    "kind": "body.fell",
                    "during": during,
                    "ctx": dict(ctx or {}),
                    "trigger": self.trigger,
                }
            )
            if state.policy == "sit":
                do = "sit_toggle"  # flag 0: velstand takes over at zero command
            return RightingOut(True, do, tuple(events))
        if self.phase == "idle":
            return RightingOut()
        if "up" in prox.edges:
            via = "help" if self.phase == "waiting_help" else "self"
            events.append(self._result(t, True, via))
            self.phase = "idle"
            return RightingOut(False, None, tuple(events))
        age = (t - self.t_phase) / NS
        dt = 0.0 if self.last_t is None else max(0.0, (t - self.last_t) / NS)
        self.last_t = t
        if self.phase in ("sit", "rise", "try2") and prox.load is not None:
            self.peak_load = max(self.peak_load, prox.load)
        if c.overload_guard and self.phase in ("sit", "rise") and state.currents_ma is not None:
            peak = max(state.currents_ma[i] for i in LEG_JOINTS)
            if peak > c.stall_current:
                self.over_s += min(dt, 0.1)  # a gap in the ticks counts one tick at most
            if self.over_s >= c.overload_max_s:
                events.append(self._result(t, False, None, overload=True))
                self._wait_for_help(t, overload=True)
                return RightingOut(True, None, tuple(events))
        if self.phase == "try1" and age >= c.try_s:
            self.tries = 2
            if c.sitstand:
                self.phase, self.t_phase, self.how = "sit", t, "sitstand"
                do = "sit_toggle"
            else:
                self.phase, self.t_phase = "pause", t
        elif self.phase == "sit" and age >= c.sit_s:
            self.phase, self.t_phase = "rise", t
            do = "sit_toggle"
        elif self.phase == "rise" and age >= c.watch_s:
            if not self.retrying:
                events.append(self._result(t, False, None))
            self._wait_for_help(t)
        elif (
            self.phase == "waiting_help"
            and not self.overloaded
            and c.sitstand
            and self.retries < c.retry_max
            and (t - self.t_wait) / NS >= c.retry_every_s
        ):
            self.retries += 1
            self.retrying = True
            self.tries, self.how = 2 + self.retries, "sitstand"
            self.over_s = 0.0
            self.phase, self.t_phase = "sit", t
            do = "sit_toggle"
        elif self.phase == "pause" and age >= c.pause_s:
            self.phase, self.t_phase = "try2", t
        elif self.phase == "try2" and age >= c.try_s:
            events.append(self._result(t, False, None))
            self._wait_for_help(t)
        return RightingOut(True, do, tuple(events))

    def _wait_for_help(self, t: int, *, overload: bool = False) -> None:
        self.phase, self.t_wait = "waiting_help", t
        self.overloaded = overload


__all__ = ["Righting", "RightingConfig", "RightingOut"]
