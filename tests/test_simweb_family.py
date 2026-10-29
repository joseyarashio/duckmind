"""Family visits (duckmind.simweb.family), owner decision D3: a touch button brings that family
member's body over first, the hand acts from their hand, and Duck Core hears about it only
then; other buttons go at once. Driven against a small stand-in world (no MuJoCo)."""

from __future__ import annotations

from typing import Any

import pytest

from duckmind.simweb import family
from duckmind.simweb.family import FamilyVisits, hand_plan


class FakeWorld:
    """The world server's x_people / x_hand / x_battery / x_world, on a clock the test turns."""

    SCRIPTS = {  # do → (moment the act is due, script end, holding after)
        "touch": (1.0, 2.5, False),
        "hold_head": (1.0, 2.5, False),
        "near": (1.0, 2.0, False),
        "pickup": (1.2, 1.6, True),
        "place": (2.5, 3.2, False),
        "upright": (1.8, 2.6, False),
        "cancel": (0.5, 1.0, False),
    }

    def __init__(self, walk_s: float = 3.0, *, arrive: bool = True) -> None:
        self.t = 0.0
        self.walk_s = walk_s
        self.arrive = arrive
        self.people: dict[str, dict[str, Any]] = {}
        self.hand: dict[str, Any] | None = None
        self.holding = False
        self.last: dict[str, Any] | None = None
        self.count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.swaps = 0

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        self.calls.append((op, params))
        if op == "x_people":
            return self._people(params)
        if op == "x_hand":
            return self._hand(params)
        if op == "x_battery":
            self.swaps += params.get("do") == "swap"
            return {"soc_ppm": 1_000_000}
        if op == "x_world":
            return {"sim_time": self.t}
        raise RuntimeError(f"unknown op {op}")

    def _people(self, params: dict[str, Any]) -> dict[str, Any]:
        do = params.get("do")
        if do == "status":
            out = []
            for who, p in list(self.people.items()):
                if p["state"] == "walking" and self.arrive and self.t >= p["at"]:
                    p["state"] = "standing"
                if p["state"] == "leaving" and self.t >= p["at"]:
                    del self.people[who]
                    continue
                out.append({"who": who, "state": p["state"]})
            return {"people": out}
        who = params["who"]
        if do == "come":
            self.people[who] = {"state": "walking", "at": self.t + self.walk_s}
            return {"ok": True, "who": who, "state": "walking", "eta_in_s": self.walk_s}
        if do == "leave":
            self.people[who] = {"state": "leaving", "at": self.t + 3.0}
            return {"ok": True, "who": who, "state": "leaving"}
        raise ValueError(do)

    def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
        do = params.get("do")
        self._advance()
        if do == "status":
            state = "idle"
            target: dict[str, Any] = {}
            if self.hand is not None:
                moment = self.hand["t0"] + self.SCRIPTS[self.hand["do"]][0]
                if self.t >= moment:
                    state = {"touch": "touch", "hold_head": "touch", "near": "near"}.get(
                        self.hand["do"], "carry"
                    )
                    if self.hand["do"] in ("place", "upright"):
                        target = {"release": {"released_s": moment}}
                else:
                    state = "reach"
            return {
                "state": state,
                "script": None if self.hand is None else self.hand["id"],
                "holding": self.holding,
                "target": target,
                "last": self.last,
            }
        if do in ("touch", "hold_head", "near") and self.holding:
            raise RuntimeError("holding")
        self.count += 1
        self.hand = {"id": self.count, "do": do, "t0": self.t, "params": params}
        return {"ok": True, "script": self.count, "do": do, "eta_s": 2.0}

    def _advance(self) -> None:
        if self.hand is None:
            return
        moment, end, holds = self.SCRIPTS[self.hand["do"]]
        if self.hand["do"] == "pickup" and self.t >= self.hand["t0"] + moment:
            self.holding = True
        if self.t >= self.hand["t0"] + end:
            if self.hand["do"] == "cancel":
                self.holding = False  # (lowered the duck and let go)
            release = None
            if self.hand["do"] in ("place", "upright"):
                release = {"released_s": self.hand["t0"] + moment}
                self.holding = False
            self.last = {
                "script": self.hand["id"],
                "ok": True,
                "target": {"release": release} if release else {},
            }
            self.hand = None


def _run(visits: FamilyVisits, world: FakeWorld, until: float, step: float = 0.1) -> list[Any]:
    due = []
    while world.t < until:
        world.t = round(world.t + step, 3)
        due += visits.tick()
    return due


def _visits(world: FakeWorld) -> FamilyVisits:
    return FamilyVisits(world, seed=7, clock=lambda: world.t)


def test_only_touches_bring_a_body_over():
    for action in (
        "say",
        "call_name",
        "tv",
        "pad_good",
        "owner_hello",
        "goodnight",
        "family_leave",
    ):
        assert hand_plan(action, {}) is None
    assert hand_plan("touch", {"where": "beak"}) == family.HandPlan(
        "touch", "arrive", {"where": "beak"}
    )
    assert hand_plan("pickup", {}).when == "grab"
    assert hand_plan("place_on_nest", {}).extra == {"to": "nest"}
    assert hand_plan("swap_battery", {}).swap_battery
    world = FakeWorld()
    visits = _visits(world)
    assert visits.request("say", "owner", {"kind": "voice"}) is None and world.calls == []
    assert FamilyVisits(None).request("touch", "owner", {}) is None


def test_a_touch_waits_for_the_walk_then_for_the_hand_then_they_leave():
    world = FakeWorld(walk_s=3.0)
    visits = _visits(world)
    answer = visits.request("touch", "child", {"seconds": 4, "firmness": "light"})
    assert answer is not None and answer["persona"] == "child" and answer["eta_s"] == 3.0
    assert world.calls[-1] == (
        "x_people",
        {"do": "come", "who": "child", "to": "duck", "from": "entry"},
    )
    assert _run(visits, world, 2.9) == []  # still walking over
    due = _run(visits, world, 5.0)
    assert len(due) == 1
    act = due[0]
    assert (act.action, act.persona, act.params) == (
        "touch",
        "child",
        {"seconds": 4, "firmness": "light"},
    )
    assert act.done_by_hand is None and 4.0 <= act.late_s <= 4.3
    hand_calls = [p for op, p in world.calls if op == "x_hand" and p.get("do") != "status"]
    assert hand_calls == [{"do": "touch", "who": "child", "where": "head", "seconds": 4}]
    _run(visits, world, 12.0)
    assert ("x_people", {"do": "leave", "who": "child"}) in world.calls
    _run(visits, world, 16.0)
    assert visits.visits == []


def test_a_pickup_is_heard_when_the_hand_has_the_duck():
    world = FakeWorld(walk_s=2.0)
    visits = _visits(world)
    visits.request("pickup", "owner", {})
    due = _run(visits, world, 6.0)
    assert [d.action for d in due] == ["pickup"]
    assert due[0].done_by_hand == {"hand": "pickup", "holding": True}
    assert 3.1 <= due[0].late_s <= 3.4  # walk 2 s + the hand's 1.2 s


@pytest.mark.parametrize(
    "action, done",
    [("place_on_nest", {"hand": "place", "to": "nest"}), ("set_upright", {"hand": "upright"})],
)
def test_a_set_down_is_heard_when_the_hand_lets_go(action, done):
    world = FakeWorld(walk_s=1.0)
    visits = _visits(world)
    visits.request(action, "partner", {})
    due = _run(visits, world, 8.0)
    assert len(due) == 1 and due[0].done_by_hand == done
    assert (
        due[0].late_s
        >= 1.0 + FakeWorld.SCRIPTS["place" if action == "place_on_nest" else "upright"][0]
    )


def test_a_battery_swap_lifts_swaps_and_sets_it_back_down_before_duck_core_hears():
    world = FakeWorld(walk_s=1.0)
    visits = _visits(world)
    visits.request("swap_battery", "owner", {})
    due = _run(visits, world, 10.0)
    assert world.swaps == 1 and [d.action for d in due] == ["swap_battery"]
    dos = [p["do"] for op, p in world.calls if op == "x_hand" and p.get("do") != "status"]
    assert dos == ["pickup", "place"]
    assert due[0].done_by_hand == {"hand": "pickup+place", "to": "here", "battery": "swapped"}


def test_a_person_who_cannot_arrive_does_not_hold_the_touch_back_forever():
    world = FakeWorld(walk_s=2.0, arrive=False)
    visits = _visits(world)
    visits.request("hand_near", "elder", {"speed": "slow"})
    due = _run(visits, world, 2.0 + family.WALK_SLACK_S + 3.0)
    assert [d.action for d in due] == ["hand_near"]
    assert any(
        "手先過去" in line for v in [visits] for line in sum((x.why_zh for x in v.visits), [])
    )


def test_without_a_world_server_the_act_still_happens():
    class Broken:
        def call(self, op: str, **params: Any) -> dict[str, Any]:
            raise ConnectionError("gone")

    clock = [0.0]
    visits = FamilyVisits(Broken(), clock=lambda: clock[0])
    answer = visits.request("touch", "owner", {})
    assert answer is not None and answer["state"] == "hand"
    clock[0] = 0.2
    due = visits.tick()
    assert [d.action for d in due] == ["touch"] and visits.errors


def test_the_roster_names_the_households_people_with_the_worlds_clothes():
    people = family.roster(7)
    assert list(people) == ["owner", "partner", "child", "elder", "guest"]
    assert people["owner"]["zh"] == "主人" and people["guest"]["in_family"]
    assert people["owner"]["in_family"]
    assert all(p["colour"].startswith("#") and len(p["colour"]) == 7 for p in people.values())
    single = family.roster(7, "single_long")
    assert not single["child"]["in_family"] and single["owner"]["in_family"]


# ── a swap visit is bounded (battery help, 2026-10-20) ───────────────────────────


class Flaky(FakeWorld):
    """A world whose hand answers ``status`` with nothing (a stalled host) for ``blind`` seconds,
    and refuses ``place`` for ``refuse_place`` tries."""

    def __init__(self, *, blind: float = 0.0, refuse_place: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.blind = blind
        self.refuse_place = refuse_place

    def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("do") == "status" and self.t < self.blind:
            raise ConnectionError("world x_hand: timed out")
        if params.get("do") == "place" and self.refuse_place > 0:
            self.refuse_place -= 1
            raise RuntimeError("busy")
        return super()._hand(params)


def _hand_dos(world: FakeWorld) -> list[str]:
    return [p["do"] for op, p in world.calls if op == "x_hand" and p.get("do") != "status"]


def test_a_swap_never_lifts_the_duck_while_the_hands_state_is_unknown_it_waits() -> None:
    """A None hand state used to count as the hand's 'moment': the pack was swapped under a duck
    nobody had lifted. For a swap an unknown state means wait."""
    world = Flaky(blind=6.0, walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    _run(visits, world, 5.0)
    assert world.swaps == 0 and _hand_dos(world) == []  # (waited: no lift, no swap)
    assert not visit.swapped
    due = _run(visits, world, 20.0)
    assert not due  # (silent)
    assert world.swaps == 1 and visit.swapped and _hand_dos(world) == ["pickup", "place"]
    assert not world.holding and not visits.visits


def test_a_swap_that_never_sees_the_hand_gives_up_without_swapping_or_lifting() -> None:
    from duckmind.simweb.family import FAIL_HAND_UNKNOWN, SWAP_WAIT_S

    world = Flaky(blind=1e9, walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    _run(visits, world, SWAP_WAIT_S + 15.0)
    assert world.swaps == 0 and _hand_dos(world) == []
    assert visit.troubles == [FAIL_HAND_UNKNOWN] and visit.fired
    _run(visits, world, SWAP_WAIT_S + 60.0)
    assert not visits.visits  # and the visit ended: the person walked out


def test_a_refused_set_down_is_tried_again_and_the_duck_is_set_down() -> None:
    """The place after the swap used to be asked once: refused, the visit ended 'after' with the
    duck still in the hand."""
    world = Flaky(refuse_place=2, walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    _run(visits, world, 40.0)
    assert world.swaps == 1 and visit.swapped
    assert _hand_dos(world).count("place") == 3  # two refused, the third took
    assert not world.holding and not visits.visits and visit.troubles == []


def test_a_set_down_the_world_keeps_refusing_ends_with_the_duck_let_go() -> None:
    from duckmind.simweb.family import FAIL_PLACE, SWAP_BOUND_S

    world = Flaky(refuse_place=10**6, walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    _run(visits, world, SWAP_BOUND_S + 20.0)
    assert visit.swapped and visit.troubles == [FAIL_PLACE]
    assert _hand_dos(world)[-1] == "cancel"  # told to lower the duck and let go
    assert not world.holding  # always placed back or released
    assert not visits.visits


def test_a_hand_that_never_has_the_duck_is_cut_short_within_the_bound() -> None:
    from duckmind.simweb.family import FAIL_NO_GRAB, SWAP_BOUND_S

    class Slow(FakeWorld):
        def _advance(self) -> None:  # (never gets hold of the duck)
            if self.hand is not None and self.hand["do"] == "pickup":
                return
            super()._advance()

    world = Slow(walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    _run(visits, world, SWAP_BOUND_S + 30.0)
    assert world.swaps == 0 and not visit.swapped
    assert visit.troubles[0] == FAIL_NO_GRAB
    assert not visits.visits and not world.holding


def test_a_swap_visit_always_ends_within_its_bound_whatever_the_hand_says() -> None:
    from duckmind.simweb.family import FAIL_RELEASE, SWAP_BOUND_S

    class Stuck(FakeWorld):
        """The hand holds the duck and ignores every command."""

        def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
            if params.get("do") == "status":
                return {
                    "state": "hold",
                    "script": None,
                    "holding": True,
                    "target": {},
                    "last": None,
                }
            if params.get("do") == "cancel":
                raise RuntimeError("busy")
            self.holding = True
            return super()._hand(params)

    world = Stuck(walk_s=1.0)
    visits = _visits(world)
    visit = visits.start("swap_battery", "owner", silent=True)
    assert visit is not None
    visit.script = None
    _run(visits, world, SWAP_BOUND_S + 40.0)
    assert not visits.visits  # (it did not wait for ever)
    assert FAIL_RELEASE in visit.troubles


def test_a_button_swap_is_as_bounded_and_still_tells_duck_core_once() -> None:
    world = Flaky(refuse_place=2, walk_s=1.0)
    visits = _visits(world)
    visits.request("swap_battery", "owner", {})
    due = _run(visits, world, 40.0)
    assert [d.action for d in due] == ["swap_battery"] and world.swaps == 1
    assert not world.holding and not visits.visits
