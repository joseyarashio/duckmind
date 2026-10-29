"""Getting unstuck: the duck's own sense that it is pressing into something (or wedged in a corner)
and the small manoeuvre that gets it out (owner bug 2026-10-04: 「頂著東西往前走」, 「走到角落就卡死」).

Measured first (``tools/machine_verify/measure_unstuck.py``, headless physics): the duck had no
sense of *failing to get anywhere while it wants to walk*. Proprio's ``blocked`` needs a walk
that has already run 1 s and then 0.6–2 s of no progress, and the neural duck's walks are short
bursts between stops to look, so it almost never fired against a loose box; a touch it caused
itself was read as a stroke ("靠過去"); a wall the ToF sees ahead and a reflex that arcs the other
way than the boundary memory did left it standing at the wall for a minute.

This module is pure and clock-injected. It reads only what the duck's own senses give:

* the brain's own request (``want_vx``: the forward speed it asks for this tick);
* its contact odometry (progress against what the gait makes for that request) — a request the
  body link holds back (after a stall, with the ToF close) is no progress either;
* the cues that something is there: a touch it felt, the ToF close ahead, a stall or a brush
  (proprio), a wall the boundary memory has close ahead.

and says when to back off (:class:`Escape`): a small reverse ARC (never a straight reverse,
owner rule), how big (bigger when it keeps getting stuck at the same place: a corner), and that
the place it was blocked at is to be remembered. Turning away afterwards is the brain's, the
reflex arcs' and the boundary memory's, now agreeing on one side (brain.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnstickConfig:
    want_vx: float = 0.25  # a forward request at least this is a request to walk
    want_gap_ms: int = 700  # a pause this short does not end a run of wanting to walk
    press_run_ms: int = 800  # it has asked to walk this long (the gait's spin-up is ~1 s) …
    held_run_ms: int = 400  # … or this long when the body stands still under the request (the
    # body link holds the walk: nothing to spin up, and its walk requests come in short pulses)
    press_window_ms: int = 600  # … and over the last this long it made …
    press_ratio: float = 0.35  # … less than this share of the path the gait makes, with
    cue_now_ms: int = 400  # … something there right now (a cue this recent), for
    press_ms: int = 1000  # … this long (summed over the last PRESS_WINDOW_MS) = pressing
    press_total_ms: int = 5000
    cue_ms: int = 1500  # a cue that something is there counts this long (for the rest)
    still_window_ms: int = 8000  # confined: inside a small circle this long …
    still_radius_m: float = 0.20
    still_want_ms: int = 2500  # … wanting to walk at least this long inside it = stuck in place
    event_gap_ms: int = 1000  # contacts closer together than this are one contact
    again_ms: int = 10_000  # a second contact within this long …
    again_m: float = 0.45  # … and this near the first is "again": it backs off
    area_m: float = 0.60  # escapes this near each other are at one place …
    area_ms: int = 45_000  # … within this long
    cooldown_ms: int = 2200  # no new escape this soon after one began (it is still getting out)
    # the reverse arc: vx −0.40, |vyaw| 1.0 (calibrated standing, phase 3: 1.0 s ≈ 8–12 cm back
    # and 32–36°; 1.5 s ≈ 15–19 cm and 52–59°, 0 falls) — never longer than what was measured
    back_ms: tuple[int, int, int] = (1100, 1400, 1500)
    turn_ms: tuple[int, int] = (1500, 2500)  # after the reverse it keeps turning away this long
    expected_base: float = 0.067  # path speed the gait makes at a request (proprio's model)
    expected_gain: float = 0.32


DEFAULT = UnstickConfig()


@dataclass(frozen=True, slots=True)
class Escape:
    """A back-off is due: reverse in a small arc for ``back_ms``, then keep turning away for
    ``turn_ms``; ``count`` is how many escapes there have been at this place (1 first, 3+ a
    corner); ``why`` what the duck sensed: ``again`` (a second contact at the same place),
    ``pressing`` (it asks to walk and gets nowhere with something there: pushing on it, or the
    body link holding the walk back), ``confined`` (it stays inside a small circle). ``at_obstacle``:
    it knows what direction the thing is in (the one it was pushing on): the place is remembered
    as blocked."""

    back_ms: int
    turn_ms: int
    count: int
    why: str
    at_obstacle: bool = True


class Unstick:
    """The duck's sense of being stuck (module doc). One :meth:`step` per brain tick."""

    __slots__ = (
        "cfg",
        "_pos",
        "_want_since",
        "_want_last",
        "_press",
        "_cue_t",
        "_wants",
        "_contacts",
        "_escapes",
        "_last_escape_t",
        "_t",
        "fired",
    )

    def __init__(self, cfg: UnstickConfig = DEFAULT) -> None:
        self.cfg = cfg
        self._pos: deque[tuple[int, float, float]] = deque()
        self._want_since: int | None = None
        self._want_last = -1_000_000
        self._press: deque[tuple[int, int]] = deque()  # (t_ms, ms of pressing at this tick)
        self._cue_t = -1_000_000
        self._wants: deque[tuple[int, int]] = deque()  # (t_ms, ms of wanting at this tick)
        self._contacts: deque[tuple[int, float, float]] = deque()
        self._escapes: deque[tuple[int, float, float]] = deque()
        self._last_escape_t = -1_000_000
        self._t: int | None = None
        self.fired = 0

    # ── the pieces ───────────────────────────────────────────────────────────

    def _expected(self, vx: float) -> float:
        return max(0.0, self.cfg.expected_base + self.cfg.expected_gain * abs(vx))

    def _progress(self, t_ms: int, window_ms: int, x: float, y: float) -> tuple[float, int] | None:
        """(the path made over the last ``window_ms``, the span covered) from the odometry."""
        oldest = None
        for item in self._pos:
            if t_ms - item[0] <= window_ms:
                oldest = item
                break
        if oldest is None or t_ms - oldest[0] < 0.8 * window_ms:
            return None
        return math.hypot(x - oldest[1], y - oldest[2]), t_ms - oldest[0]

    def _count_here(self, t_ms: int, x: float, y: float) -> int:
        cfg = self.cfg
        while self._escapes and t_ms - self._escapes[0][0] > cfg.area_ms:
            self._escapes.popleft()
        return sum(1 for _t, ex, ey in self._escapes if math.hypot(x - ex, y - ey) <= cfg.area_m)

    def _fire(self, t_ms: int, x: float, y: float, why: str) -> Escape:
        cfg = self.cfg
        count = self._count_here(t_ms, x, y) + 1
        self._escapes.append((t_ms, x, y))
        self._last_escape_t = t_ms
        self.fired += 1
        # the contacts so far are answered: a new run of contact starts the count again
        self._contacts.clear()
        self._press.clear()
        self._wants.clear()
        self._want_since = None
        self._pos.clear()  # (the reverse and the turn that follow are not "staying in a circle")
        back = cfg.back_ms[min(count, len(cfg.back_ms)) - 1]
        turn = cfg.turn_ms[0 if count < 2 else 1]
        return Escape(back, turn, count, why, why != "confined")

    # ── the step ─────────────────────────────────────────────────────────────

    def step(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        *,
        active: bool,
        want_vx: float,
        cue: bool,
        contact: bool,
        held: bool = False,
        stationary: bool = False,
    ) -> Escape | None:
        """One tick. ``active``: standing, may walk, nothing else has the legs (not docking, not
        lifted, not fallen); ``want_vx``: the forward speed the brain asks for now; ``cue``:
        something is there (see the module doc); ``contact``: the duck's own body met something
        this tick (a touch while it walked, a stall, a brush) — a contact it did not cause (a
        hand) is not one; ``held``: the body stands still (nothing applied) under the request;
        ``stationary``: the act on now turns or works on one spot on purpose (sunbathing, the
        settling circle, going round a toy): its steps inside a small circle are not "confined"
        (``BehaviourSpec.stationary``). Pressing into something is still pressing, and a contact
        is still a contact."""
        cfg = self.cfg
        x, y = odom[0], odom[1]
        dt_ms = 0 if self._t is None else max(0, min(500, t_ms - self._t))
        self._t = t_ms
        self._pos.append((t_ms, x, y))
        while self._pos and t_ms - self._pos[0][0] > cfg.still_window_ms + 500:
            self._pos.popleft()
        if not active:
            self._want_since = None
            self._press.clear()
            self._wants.clear()
            return None
        if cue:
            self._cue_t = t_ms
        cued = t_ms - self._cue_t <= cfg.cue_ms
        cue_now = t_ms - self._cue_t <= cfg.cue_now_ms
        wants = want_vx >= cfg.want_vx
        if wants:
            if self._want_since is None or t_ms - self._want_last > cfg.want_gap_ms:
                self._want_since = t_ms
            self._want_last = t_ms
        elif self._want_since is not None and t_ms - self._want_last > cfg.want_gap_ms:
            self._want_since = None
        # (an act that stays on its spot on purpose does not feed "confined"; pressing, below, reads
        # ``wants`` as it is)
        self._wants.append((t_ms, dt_ms if wants and not stationary else 0))
        while self._wants and t_ms - self._wants[0][0] > cfg.still_window_ms:
            self._wants.popleft()
        # pressing: it has asked to walk for a while, something is there, and it made little of
        # the path the gait makes (pushing on it, or the body link holding the walk back)
        pressing_now = False
        if wants and cue_now and self._want_since is not None:
            run = t_ms - self._want_since
            if held and run >= cfg.held_run_ms:
                pressing_now = True
            elif run >= cfg.press_run_ms:
                made = self._progress(t_ms, cfg.press_window_ms, x, y)
                if made is not None:
                    path, span = made
                    pressing_now = path < cfg.press_ratio * self._expected(want_vx) * span / 1000.0
        self._press.append((t_ms, dt_ms if pressing_now else 0))
        while self._press and t_ms - self._press[0][0] > cfg.press_total_ms:
            self._press.popleft()
        cooling = t_ms - self._last_escape_t < cfg.cooldown_ms
        # a contact (its own doing): the first is the reflexes' (stop, arc away); a second at the
        # same place within a few seconds is "again" — back off
        again = False
        if contact:
            if not self._contacts or t_ms - self._contacts[-1][0] >= cfg.event_gap_ms:
                self._contacts.append((t_ms, x, y))
            while self._contacts and t_ms - self._contacts[0][0] > cfg.again_ms:
                self._contacts.popleft()
            if len(self._contacts) >= 2 and (
                math.hypot(x - self._contacts[0][1], y - self._contacts[0][2]) <= cfg.again_m
            ):
                again = True
        if cooling:
            return None
        if again:
            return self._fire(t_ms, x, y, "again")
        if sum(ms for _t, ms in self._press) >= cfg.press_ms:
            return self._fire(t_ms, x, y, "pressing")
        # confined: in a small circle for a long time while it kept asking to walk
        if cued and sum(ms for _t, ms in self._wants) >= cfg.still_want_ms:
            first = self._pos[0]
            if t_ms - first[0] >= cfg.still_window_ms - 500 and all(
                math.hypot(px - x, py - y) <= cfg.still_radius_m for _t, px, py in self._pos
            ):
                return self._fire(t_ms, x, y, "confined")
        return None


__all__ = ["DEFAULT", "Escape", "Unstick", "UnstickConfig"]
