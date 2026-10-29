"""Family members come over before they touch the duck (owner decision D3; physical-design spec
§8.8, W02).

When the owner presses a button as a family member (主人, 伴侶, 小孩, 長輩, 訪客) and the
button needs a hand on the duck — touching its head or beak, holding its head, a hand coming
near, picking it up, putting it on its nest, setting it upright, swapping its battery,
pressing its power button — that person's body first walks over in the physics world
(:mod:`duckmind.world.people`, keeping 40 cm from the duck: D4), then the world's hand
(:mod:`duckmind.world.hand`) does it from their hand, and only then does Duck Core hear about it
(a few seconds later than the button). Voice, the gamepad, the TV and the other buttons are
not touches: they go to Duck Core at once. There is no routine-driven wandering yet (D3): people
appear only for these buttons and walk back out a few seconds after.

This module is the page side's orchestrator; it never moves anything itself. It talks to the
world server through any object with ``call(op, **params) -> dict``
(:class:`duckmind.world.simhome.WorldClient`), is driven by :meth:`FamilyVisits.tick` (a few
times a second), and hands back the Duck Core acts that are due (:class:`Due`)::

    visits = FamilyVisits(client, seed=config.seed)
    answer = visits.request("pickup", "child", {})   # None: not a touch, act now
    ...
    for due in visits.tick():                        # every 0.1-0.2 s
        dispatch(session, "act", {"action": due.action, "persona": due.persona, "params": due.params})

Each :class:`Due` says what the hand already did (``done_by_hand``), so the live bridge does
not move the body a second time when Duck Core then reports the same thing (``nest.docked`` by
a hand after ``place_on_nest``, a set-upright…).

A visit can also be the household's own (:meth:`FamilyVisits.start` with ``silent=True``): nobody
pressed a button, so Duck Core is not given an act for it. The hand does its part in the world
and ``Visit.swapped`` says whether the pack really was swapped; whoever started it (the neural
duck's ``battery_help``, :mod:`duckmind.simweb.battery_help`) watches the visit, and the neural
duck feels it through its own senses.

A swap visit (``swap_battery``) is bounded, whoever started it: its hand never lifts the duck blind
(a hand whose state cannot be read is waited for, never swapped under), and the visit does not end
while the duck is still in the hand. A set-down that is refused is tried again (``PLACE_TRIES``),
and a lift, swap and set-down that is not done in ``SWAP_LIFT_S`` is cut short: the hand is told to
let go (``cancel``: lower the duck and release it). Past ``SWAP_BOUND_S`` the visit gives up
whatever the hand says. What went wrong is a number (``FAIL_*``, ``Visit.troubles``, the first is
the cause) and one log line.

Names come from the household's profile, read-only (``routines.make_profile`` with the same
seed rule as ``api.family_profile``); the bodies' clothes are the world's own
(:data:`duckmind.world.people.PERSONAS`).
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from duckmind.world import people as world_people

WALK_SLACK_S = 12.0  # a person who has not arrived this long after their eta: the hand goes anyway
HAND_TIMEOUT_S = 12.0  # a hand that never reaches its moment: the act goes to Duck Core anyway
LINGER_S = 4.0  # the person stays this long after the hand is done, then walks out
PROFILE_SALT = 0xFA11  # api.family_profile: make_profile(seed ^ 0xFA11, kind)
# A swap visit's bounds, counted from when its hand began: the lift, the swap and the set-down are
# done by SWAP_LIFT_S, or the hand is told to let go; by SWAP_BOUND_S the duck is let go or given up.
SWAP_LIFT_S = 40.0
SWAP_BOUND_S = 70.0
TRY_GAP_S = 2.0  # between two tries of the same hand step (the set-down, the letting go)
PLACE_TRIES = 3  # a set-down the world refuses is tried this many times, then the duck is let go
SWAP_WAIT_S = 30.0  # a swap visit waits this long for a free, readable hand, then gives up
# What went wrong in a swap visit, as numbers (the first of ``Visit.troubles`` is the cause):
FAIL_HAND_UNKNOWN = 1  # the hand's state could not be read, so it was not sent
FAIL_HAND_REFUSED = 2  # the world refused the lift
FAIL_NO_GRAB = 3  # the hand did not have the duck in time
FAIL_SWAP_REFUSED = 4  # the world refused the pack swap
FAIL_PLACE = 5  # the duck was not set down in time or the world refused the set-down
FAIL_RELEASE = 6  # the duck was still in the hand when the visit's bound ran out
FAIL_HAND_BUSY = 7  # the hand stayed on another script for the whole wait
FAIL_NO_VISIT = 8  # no visit could be started (no world to walk in)
FAIL_UNSAID = 9  # the visit ended without a swap and without saying why


class WorldPort(Protocol):
    def call(self, op: str, **params: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HandPlan:
    """What the hand does for a button, and when Duck Core hears about it."""

    do: str  # x_hand do
    when: str  # arrive (the hand is at the duck) | grab (it has the duck) | release (let go)
    extra: dict[str, Any] = field(default_factory=dict)
    swap_battery: bool = False


def hand_plan(action: str, params: dict[str, Any]) -> HandPlan | None:
    """The hand's part in a page button, or None when the button is not a touch."""
    seconds = params.get("seconds")
    timing = {} if seconds is None else {"seconds": seconds}
    if action == "touch":
        where = "beak" if params.get("where") == "beak" else "head"
        return HandPlan("touch", "arrive", {"where": where, **timing})
    if action == "hold_head":
        return HandPlan("hold_head", "arrive", {"where": "head", "seconds": seconds or 4})
    if action == "hand_near":
        speed = "slow" if params.get("speed") == "slow" else "fast"
        return HandPlan("near", "arrive", {"speed": speed})
    if action == "pickup":
        return HandPlan("pickup", "grab")
    if action == "place_on_nest":
        return HandPlan("place", "release", {"to": "nest"})
    if action == "set_upright":
        return HandPlan("upright", "release")
    if action == "swap_battery":  # lift it, swap the pack, set it back down where it was
        return HandPlan("pickup", "grab", swap_battery=True)
    if action in ("power_on", "power_off", "wake"):
        return HandPlan("touch", "arrive", {"where": "back", "seconds": 1.0})
    return None


def roster(seed: int, kind: str | None = None) -> dict[str, dict[str, Any]]:
    """The family's people for the page: persona → name, colour, height, and whether the
    household profile has them (a guest is always possible)."""
    from duckmind.duckcore.routines import make_profile

    profile = make_profile(seed ^ PROFILE_SALT, kind)
    members = {m.persona for m in profile.members}
    out: dict[str, dict[str, Any]] = {}
    for p in world_people.PERSONAS:
        out[p.id] = {
            "persona": p.id,
            "zh": p.zh,
            "colour": "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in p.shirt),
            "height_mm": int(round(p.height_m * 1000)),
            "in_family": p.id in members or p.id == "guest",
        }
    return out


@dataclass
class Due:
    """A Duck Core act whose moment has come."""

    visit: int
    action: str
    persona: str
    params: dict[str, Any]
    done_by_hand: dict[str, Any] | None  # what the physics hand already did
    late_s: float  # seconds after the button


@dataclass
class Visit:
    id: int
    action: str
    persona: str
    params: dict[str, Any]
    plan: HandPlan
    asked_s: float
    state: str = "walking"  # walking | queued | hand | after | lingering | leaving | done
    walk_eta_s: float | None = None
    started_s: float = 0.0
    script: int | None = None
    fired: bool = False
    phase_s: float = 0.0
    why_zh: list[str] = field(default_factory=list)
    silent: bool = False  # the household's own visit: no button, so Duck Core is given no act
    swapped: bool = False  # the world's pack was swapped by this visit's hand
    # a swap visit: when its hand began (its bounds count from here), when its hand step (the
    # set-down, the letting go) was last tried, how many set-downs the world refused so far, and
    # whether its own hand script was started (so the duck in it is ours to let go)
    first_s: float | None = None
    try_s: float = -math.inf
    tries: int = 0
    ours: bool = False
    troubles: list[int] = field(default_factory=list)  # FAIL_* codes; the first is the cause
    # why the world's hand refused the lift (world/hand.py ``REFUSE_*``, a number), when it did
    refusal: int | None = None


class FamilyVisits:
    """Button-driven visits (see the module docstring)."""

    def __init__(
        self,
        world: WorldPort | None,
        *,
        seed: int = 7,
        kind: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.world = world
        self.seed = seed
        self.kind = kind
        self.clock = clock
        self.visits: list[Visit] = []
        self._ids = itertools.count(1)
        self.errors: list[str] = []

    def people(self) -> dict[str, dict[str, Any]]:
        return roster(self.seed, self.kind)

    # ── the button ────────────────────────────────────────────────────────

    def request(
        self, action: str, persona: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """A page button as a family member. None: not a touch (send it to Duck Core now).
        Otherwise the visit is started and the act will come out of :meth:`tick`."""
        visit = self.start(action, persona, params)
        if visit is None:
            return None
        return {
            "visit": visit.id,
            "persona": visit.persona,
            "state": visit.state,
            "eta_s": None
            if visit.walk_eta_s is None
            else round(visit.walk_eta_s - visit.asked_s, 2),
            "why_zh": list(visit.why_zh),
        }

    def start(
        self,
        action: str,
        persona: str,
        params: dict[str, Any] | None = None,
        *,
        silent: bool = False,
    ) -> Visit | None:
        """Begin the visit of a family member for ``action`` (None: not a touch, or no world to walk
        in). ``silent``: the household's own visit (nobody pressed a button): its hand does what
        the action says in the world and :meth:`tick` gives Duck Core no act for it."""
        params = dict(params or {})
        plan = hand_plan(action, params)
        if plan is None or self.world is None:
            return None
        who = persona if any(p.id == persona for p in world_people.PERSONAS) else "guest"
        now = self.clock()
        visit = Visit(next(self._ids), action, who, params, plan, asked_s=now, silent=silent)
        present = self._present(who)
        if present is not None and present.get("state") in ("standing", "waiting", "walking"):
            answer = self._call("x_people", do="come", who=who, to="duck")
        else:
            answer = self._call("x_people", do="come", who=who, to="duck", **{"from": "entry"})
        if answer is None:
            visit.state = "hand"  # no body could come: the hand acts from above
            visit.why_zh.append("家人的身體走不過來：手直接來")
        else:
            eta = answer.get("eta_in_s")
            visit.walk_eta_s = now + (float(eta) if isinstance(eta, (int, float)) else 0.0)
        # a visit by the same person that is only lingering or leaving is over
        for old in self.visits:
            if old.persona == who and old.state in ("lingering", "leaving"):
                old.state = "done"
        self.visits.append(visit)
        return visit

    def cancel(self, visit: Visit) -> bool:
        """Call off a visit whose hand has not started (the person is still walking over, or
        waiting for the hand): they turn round and walk back out. A hand already at work is
        never cut. True: the visit was called off."""
        if visit.state not in ("walking", "queued"):
            return False
        visit.state = "lingering"
        visit.phase_s = self.clock() - LINGER_S  # (so they leave at the next tick)
        return True

    # ── the loop ──────────────────────────────────────────────────────────

    def tick(self) -> list[Due]:
        """Move every visit on; the Duck Core acts that are due now."""
        now = self.clock()
        due: list[Due] = []
        hand = (
            self._status("x_hand")
            if any(v.state in ("hand", "queued", "after", "walking") for v in self.visits)
            else None
        )
        for visit in self.visits:
            swap = visit.action == "swap_battery"
            if (
                swap
                and visit.state in ("hand", "after")
                and not visit.fired
                and visit.first_s is not None
                and now - visit.first_s > SWAP_LIFT_S
            ):
                # (the lift, the swap and the set-down were not done in time: let the duck go)
                self._trouble(visit, FAIL_PLACE if visit.swapped else FAIL_NO_GRAB, now)
                visit.fired = True
                visit.state = "after"
            if visit.state == "walking":
                person = self._present(visit.persona)
                arrived = person is not None and person.get("state") == "standing"
                late = visit.walk_eta_s is not None and now > visit.walk_eta_s + WALK_SLACK_S
                if arrived or late or person is None:
                    if late and not arrived:
                        visit.why_zh.append("家人還沒走到：手先過去")
                    visit.state = "queued"
                    visit.phase_s = now
            if visit.state == "queued":
                if self._hand_busy(hand) or (swap and hand is None):
                    # (the hand is on another script, or its state is not known: a swap waits,
                    # it never lifts the duck blind; but not for ever)
                    if swap and now - visit.phase_s > SWAP_WAIT_S:
                        self._trouble(
                            visit, FAIL_HAND_BUSY if hand is not None else FAIL_HAND_UNKNOWN, now
                        )
                        if not visit.silent:
                            due.append(self._due(visit, now, None))
                        visit.fired = True
                        visit.state = "lingering"
                        visit.phase_s = now
                    continue
                refused_before = self._refusals(hand)
                answer = self._call(
                    "x_hand", do=visit.plan.do, who=visit.persona, **visit.plan.extra
                )
                visit.started_s = now
                if swap:
                    visit.first_s = now
                if (
                    answer is None
                ):  # (the hand was refused: no script to wait for, nothing to set down)
                    visit.state = "after"
                    visit.fired = visit.fired or visit.plan.swap_battery
                    if swap:
                        visit.refusal = self._refusal_since(refused_before)
                        self._trouble(visit, FAIL_HAND_REFUSED, now)
                    if not visit.silent:
                        due.append(self._due(visit, now, None))
                    continue
                visit.script = answer.get("script")
                visit.ours = True
                visit.state = "hand"
                hand = self._status("x_hand")
            if visit.state == "hand":
                moment = self._moment(visit, hand)
                if moment or now - visit.started_s > HAND_TIMEOUT_S:
                    lift = visit.plan.swap_battery
                    if lift and moment:
                        # the pack is swapped while the duck is held, then it is set back down
                        visit.swapped = self._call("x_battery", do="swap") is not None
                        if not visit.swapped:
                            self._trouble(visit, FAIL_SWAP_REFUSED, now)
                    else:
                        if swap and not moment:
                            self._trouble(visit, FAIL_NO_GRAB if lift else FAIL_PLACE, now)
                        if not visit.silent:
                            due.append(self._due(visit, now, self._done(visit)))
                        visit.fired = True
                    visit.state = "after"
                    visit.phase_s = now
            if visit.state == "after":
                if visit.plan.swap_battery and not visit.fired:
                    if self._script_done(visit, hand) and now - visit.try_s >= TRY_GAP_S:
                        visit.try_s = now
                        answer = self._call("x_hand", do="place", to="here", who=visit.persona)
                        if answer is not None:
                            visit.script = answer.get("script")
                            visit.plan = HandPlan(
                                "place", "release", {"to": "here"}, swap_battery=False
                            )
                            visit.state = "hand"
                            visit.started_s = now
                            visit.tries = 0
                        else:
                            visit.tries += 1
                            if visit.tries >= PLACE_TRIES:
                                self._trouble(visit, FAIL_PLACE, now)
                                visit.fired = True
                    continue
                if swap:
                    if not self._set_down(visit, hand, now):
                        continue
                    visit.state = "lingering"  # (its part is over, whatever the hand's script)
                    visit.phase_s = now
                elif self._script_done(visit, hand):
                    visit.state = "lingering"
                    visit.phase_s = now
            if visit.state == "lingering" and now - visit.phase_s >= LINGER_S:
                if not any(
                    v is not visit
                    and v.persona == visit.persona
                    and v.state not in ("done", "leaving")
                    for v in self.visits
                ):
                    self._call("x_people", do="leave", who=visit.persona)
                visit.state = "leaving"
                visit.phase_s = now
            if visit.state == "leaving":
                person = self._present(visit.persona)
                if person is None or now - visit.phase_s > 30.0:
                    visit.state = "done"
        self.visits = [v for v in self.visits if v.state != "done"]
        return due

    def view(self) -> list[dict[str, Any]]:
        """The visits in progress, for the page."""
        now = self.clock()
        return [
            {
                "visit": v.id,
                "action": v.action,
                "persona": v.persona,
                "state": v.state,
                "since_s": round(now - v.asked_s, 2),
                "why_zh": list(v.why_zh),
            }
            for v in self.visits
        ]

    # ── helpers ───────────────────────────────────────────────────────────

    def _due(self, visit: Visit, now: float, done: dict[str, Any] | None) -> Due:
        return Due(
            visit=visit.id,
            action=visit.action,
            persona=visit.persona,
            params=dict(visit.params),
            done_by_hand=done,
            late_s=round(now - visit.asked_s, 3),
        )

    @staticmethod
    def _done(visit: Visit) -> dict[str, Any] | None:
        if visit.action == "place_on_nest":
            return {"hand": "place", "to": "nest"}
        if visit.action == "set_upright":
            return {"hand": "upright"}
        if visit.action == "pickup":
            return {"hand": "pickup", "holding": True}
        if visit.action == "swap_battery":
            return {"hand": "pickup+place", "to": "here", "battery": "swapped"}
        return None

    def _call(self, op: str, **params: Any) -> dict[str, Any] | None:
        if self.world is None:
            return None
        try:
            return self.world.call(op, **params)
        except Exception as error:  # the world server is gone or refused: no body, act anyway
            self.errors.append(f"{op}: {error}")
            del self.errors[:-20]
            return None

    def _status(self, op: str) -> dict[str, Any] | None:
        return self._call(op, do="status")

    def _present(self, who: str) -> dict[str, Any] | None:
        status = self._status("x_people")
        if status is None:
            return None
        for person in status.get("people", []):
            if person.get("who") == who:
                return person
        return None

    @staticmethod
    def _hand_busy(hand: dict[str, Any] | None) -> bool:
        """The one hand is in the middle of another visit's script."""
        return hand is not None and hand.get("script") is not None

    def _moment(self, visit: Visit, hand: dict[str, Any] | None) -> bool:
        if hand is None:
            # (the hand's state is not known: a swap waits for it, it must not take the pack out of
            # a duck nobody is holding; the other acts go on as before)
            return not visit.plan.swap_battery
        ours = hand.get("script") == visit.script
        last = hand.get("last") or {}
        finished = last.get("script") == visit.script
        if visit.plan.when == "arrive":
            return (ours and hand.get("state") in ("touch", "near")) or finished
        if visit.plan.when == "grab":
            return (ours or finished) and bool(hand.get("holding"))
        release = (last.get("target") or {}).get("release") if finished else None
        if ours and (hand.get("target") or {}).get("release"):
            return True
        return bool(release) or (finished and not last.get("ok", True))

    @staticmethod
    def _script_done(visit: Visit, hand: dict[str, Any] | None) -> bool:
        if hand is None or visit.script is None:  # (no hand, or none was ever started)
            return True
        return hand.get("script") != visit.script

    @staticmethod
    def _refusals(hand: dict[str, Any] | None) -> int:
        """How many scripts the hand has refused so far (its own count; 0 when not known)."""
        refused = (hand or {}).get("refused") or {}
        count = refused.get("count")
        return int(count) if isinstance(count, int) and not isinstance(count, bool) else 0

    def _refusal_since(self, before: int) -> int | None:
        """Why the hand refused the script just asked for (world/hand.py ``REFUSE_*``): the hand
        keeps the number of its last refusal, and counts them, so a refusal that was not the hand's
        (the world server gone) is not given an old one. None: no reason known."""
        hand = self._status("x_hand")
        refused = (hand or {}).get("refused") or {}
        code = refused.get("code")
        if self._refusals(hand) > before and isinstance(code, int) and not isinstance(code, bool):
            return code
        return None

    def _trouble(self, visit: Visit, code: int, now: float) -> None:
        visit.troubles.append(code)
        why = (
            "" if visit.refusal is None or code != FAIL_HAND_REFUSED else f", hand {visit.refusal}"
        )
        print(
            f"family: visit {visit.id} ({visit.action}) trouble {code}{why} "
            f"{now - visit.asked_s:.1f} s after it was asked",
            flush=True,
        )

    def _set_down(self, visit: Visit, hand: dict[str, Any] | None, now: float) -> bool:
        """A swap visit does not end with the duck still in its hand. True: its part is over (the
        duck is not held, or the hand is on somebody else's script now, or the bound ran out);
        else False, and a hand that still holds the duck once its script is done (or once the
        lift's time is up) is told to let go (``cancel``: lower it, release it) every
        ``TRY_GAP_S``. A hand whose state cannot be read is waited for, never sent to."""
        if not visit.ours or visit.first_s is None:  # (our hand never took hold: nothing to let go)
            return True
        done = self._script_done(visit, hand)
        held = hand is None or bool(hand.get("holding"))
        if hand is not None:
            if hand.get("script") not in (None, visit.script):
                return True  # (another script has the hand: not ours to hold or let go)
            if not held and done:
                return True
        if now - visit.first_s > SWAP_BOUND_S:
            if held:
                self._trouble(visit, FAIL_RELEASE, now)
                self._call("x_hand", do="cancel", who=visit.persona)  # (one last try)
            return True
        late = now - visit.first_s > SWAP_LIFT_S
        if hand is not None and (held or not done) and (done or late):
            if now - visit.try_s >= TRY_GAP_S:
                visit.try_s = now
                answer = self._call("x_hand", do="cancel", who=visit.persona)
                if answer is not None:
                    visit.script = answer.get("script")
        return False
