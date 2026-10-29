"""The family helps the neural duck with its battery (Duck Core's ``battery_help`` setting).

Duck Core swaps a pack for itself when the setting is on and its battery falls below
``MIN_MOTION_BATTERY`` (``engine._minute``: "家人會幫牠換電池"). An attached Duck Core
(``bridge_on``, the life of the neural duck) simulates no body and no battery, so that line never
runs for it, and the neural duck's own pack (the physics world's ``x_battery``) drained to nothing
with nobody to swap it. Here the setting works for the neural duck:

* the household watches the world's pack, as the page's battery bar does (the world's truth,
  ``x_world``: whole percent, charging and the hand; :func:`pack_of`);
* when ``battery_help`` is on, the life has no nest and the pack falls below the same threshold
  Duck Core uses, someone comes over, lifts the duck, swaps the pack and sets it back down: the
  existing family visit for ``swap_battery`` (:mod:`duckmind.simweb.family`), made *silent* — nobody
  pressed a button, so Duck Core is not given an act for it;
* once for each low crossing: the pack has to be seen at or above the threshold again before another
  swap is made. Not while the duck is held by a hand or is on its charger, and not while another
  visit is on (a family member's own button); a visit that ended without swapping (the hand timed
  out, the world refused) is tried again with a growing wait (:data:`RETRY_S`, doubling up to
  :data:`RETRY_MAX_S`) for as long as the pack stays low, and says why it failed, as a number
  (``last_failure``: the ``FAIL_*`` codes of :mod:`duckmind.simweb.family`; ``last_refusal``: when
  the world's hand would not lift the duck, its own reason, ``REFUSE_*`` of
  :mod:`duckmind.world.hand`).

What Duck Core hears is only what the neural duck lived: the pack jumping up while it is not
charging is the ``feed`` experience (:mod:`duckmind.life.experience`), which Duck Core records as one
``power.battery_swap`` by the family. Nothing here dispatches a second one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from duckmind.duckcore.engine import MIN_MOTION_BATTERY
from duckmind.duckcore.fixed import SCALE
from duckmind.simweb.family import FAIL_NO_VISIT, FAIL_UNSAID

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore
    from duckmind.simweb.family import FamilyVisits, Visit

LOW_PCT = MIN_MOTION_BATTERY * 100 // SCALE  # 30: Duck Core's "too low to walk" (whole percent)
RETRY_S = 30.0  # a visit that did not swap the pack: the next try comes after this
RETRY_MAX_S = 300.0  # ... and after each failure in a row the wait doubles, up to this
PERSONA = "owner"  # whose body comes over (a household member; the world draws them walking in)


def helping(core: DuckCore, *, on_screen: bool, world_ready: bool) -> bool:
    """Whether the family helps this life with its pack right now: the life duck is the one on
    screen, the world has a hand and people, Duck Core is attached to it (an unattached Duck Core
    swaps for itself), awake, off any cable, has no nest, and the setting is on."""
    s = core.s
    return bool(
        on_screen
        and world_ready
        and getattr(s, "bridged", False)
        and s.battery_help
        and s.awake
        and not s.docked
        and not core.hw.nest
    )


def pack_of(world: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The household's view of the pack from the world's truth (``x_world``): whole ``percent``,
    ``charging`` (on its charger's contact) and ``held`` (a hand has the duck). None: no pack."""
    battery = (world or {}).get("battery")
    if not isinstance(battery, Mapping) or isinstance(battery.get("percent"), bool):
        return None
    percent = battery.get("percent")
    if not isinstance(percent, (int, float)):
        return None
    hand = (world or {}).get("hand")
    return {
        "percent": int(percent),
        "charging": bool(battery.get("charging") or battery.get("pg")),
        "held": bool(isinstance(hand, Mapping) and (hand.get("held") or hand.get("holding"))),
    }


class BatteryHelp:
    """The family's help with the pack: one :meth:`step` for every look at the world (several a
    second, from the family thread)."""

    def __init__(
        self,
        *,
        persona: str = PERSONA,
        clock: Callable[[], float] = time.monotonic,
        retry_s: float = RETRY_S,
        retry_max_s: float = RETRY_MAX_S,
    ) -> None:
        self.persona = persona
        self.clock = clock
        self.retry_s = retry_s
        self.retry_max_s = retry_max_s
        self.swaps = 0  # packs swapped
        self.failed = 0  # visits that ended without a swap
        self.last_before_pct: int | None = None  # the pack's percent just before the last swap
        self.last_failure: int | None = None  # why the last visit failed (family.FAIL_*), a number
        self.last_refusal: int | None = None  # and, when the hand refused: world/hand.py REFUSE_*
        self.streak = 0  # failures in a row since the pack was last swapped or seen back up
        self.visit: Visit | None = None  # the swap visit under way
        self._handled = False  # this low crossing has had its visit
        self._retry_at = 0.0
        self._low_pct: int | None = None  # the last reading below the threshold (the "before")

    def step(
        self, family: FamilyVisits, *, helping: bool, pack: Mapping[str, Any] | None
    ) -> int | None:
        """Look at the pack once. Returns the percent the pack was at when a swap this visit
        began has just been done (once for each swap), else None. ``pack``: :func:`pack_of` (None:
        no fresh reading); ``helping``: :func:`helping`."""
        percent = None if pack is None else int(pack["percent"])
        if percent is not None:
            if percent >= LOW_PCT:
                self._handled = False  # (the next fall below it is a new crossing)
                self.streak = 0
            else:
                self._low_pct = percent
        done = self._follow(family)
        visit = self.visit
        if visit is not None and (
            not helping or (pack is not None and (percent >= LOW_PCT or pack["charging"]))
        ):
            # (it does not need the pack any more: someone else swapped it, it is on its charger,
            # the setting went off) the person who has not yet got to the duck goes back out;
            # a hand already at work is let finish, and counted when it has
            if family.cancel(visit):
                self.visit = None
                self._handled = False
        if (
            not helping
            or percent is None
            or pack is None
            or percent >= LOW_PCT
            or self.visit is not None
            or self._handled
            or pack["charging"]
            or pack["held"]
            or family.visits  # (a visit is on, ours or someone's own button: after they leave)
            or self.clock() < self._retry_at
        ):
            return done
        visit = family.start("swap_battery", self.persona, silent=True)
        if visit is None:  # (no world to walk in, no hand: try again later)
            self._failed(FAIL_NO_VISIT)
            return done
        self.visit = visit
        self._handled = True
        return done

    def _failed(self, code: int, refusal: int | None = None) -> None:
        """A try ended without a swap: say why (a number, one log line) and wait before the next,
        twice as long for each failure in a row, up to ``retry_max_s``. ``refusal``: when the
        world's hand refused the lift, its own reason (world/hand.py ``REFUSE_*``)."""
        self.streak += 1
        self.last_failure = code
        self.last_refusal = refusal
        wait = min(self.retry_s * 2 ** (self.streak - 1), self.retry_max_s)
        self._retry_at = self.clock() + wait
        why = "" if refusal is None else f" (hand refused: {refusal})"
        print(
            f"battery help: try {self.streak} failed, reason {code}{why}, pack {self._low_pct} %, "
            f"next try in {wait:.0f} s",
            flush=True,
        )

    def _follow(self, family: FamilyVisits) -> int | None:
        visit = self.visit
        if visit is None:
            return None
        if visit.swapped:
            self.visit = None
            self.swaps += 1
            self.streak = 0
            self.last_before_pct = self._low_pct
            return self._low_pct
        if visit not in family.visits:  # over, and the pack was not swapped
            self.visit = None
            self.failed += 1
            self._handled = False
            self._failed(visit.troubles[0] if visit.troubles else FAIL_UNSAID, visit.refusal)
        return None

    def view(self) -> dict[str, Any]:
        """Numbers for ``/api/life`` (no words). ``last_failure``: why the last try failed
        (:mod:`duckmind.simweb.family` ``FAIL_*``), None when none has; ``last_refusal``: when that
        was the world's hand refusing the lift, why (``REFUSE_*`` of :mod:`duckmind.world.hand`);
        ``streak``: failures in a row; ``retry_in_s``: seconds until the next try may start, None
        when none is waited for."""
        wait = self._retry_at - self.clock()
        return {
            "swaps": self.swaps,
            "failed": self.failed,
            "last_before_pct": self.last_before_pct,
            "last_failure": self.last_failure,
            "last_refusal": self.last_refusal,
            "streak": self.streak,
            "retry_in_s": round(wait, 1) if self.streak and wait > 0 else None,
            "visiting": self.visit is not None,
            "threshold_pct": LOW_PCT,
        }
