"""The family helps the neural duck with its battery (duckmind.simweb.battery_help): Duck Core's
``battery_help`` setting, for a life whose Duck Core is attached to the neural duck (it simulates no
battery of its own, so its own "swap a pack below 30 %" never runs for it).

The rules, against a stand-in world (no MuJoCo): the swap comes once for each low crossing, not
while the duck is held or charging, never with the setting off, with a nest, or for a demo duck.
The whole path, against a real MuJoCo home (skips without it, runs in the render check): the pack
drains below 30 %, a family member walks over, lifts the duck, swaps the pack and sets it back
down, and Duck Core hears one ``power.battery_swap`` by the family, from the neural duck's own
experience, and not a second one."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from test_life_bridge_host import BIRTH, Life, Wall
from test_simweb_family import FakeWorld, Flaky

from duckmind.duckcore.api import dispatch
from duckmind.duckcore.fixed import SCALE
from duckmind.neuro import contract as C
from duckmind.simweb import battery_help as bh
from duckmind.simweb.battery_help import LOW_PCT, BatteryHelp, helping, pack_of
from duckmind.simweb.family import FAIL_HAND_REFUSED, FAIL_NO_VISIT, FamilyVisits
from duckmind.simweb.server import SimState
from duckmind.world.hand import REFUSE_NO_WAY_OUT

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


# ── the rules (a stand-in world) ────────────────────────────────────────────────


class PackWorld(FakeWorld):
    """The family test's world with a pack: a swap puts it back to 100 %."""

    def __init__(self, percent: int = 25, **kw: Any) -> None:
        super().__init__(**kw)
        self.percent = percent
        self.charging = False

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        if op == "x_battery" and params.get("do") == "swap":
            self.percent = 100
        return super().call(op, **params)

    def pack(self) -> dict[str, Any] | None:
        return pack_of(
            {
                "battery": {"percent": self.percent, "charging": self.charging},
                "hand": {"held": self.holding},
            }
        )


class Refusing(PackWorld):
    """A world whose hand is busy: every pickup and set-down is refused."""

    def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("do") in ("pickup", "place"):
            raise RuntimeError("busy")
        return super()._hand(params)


def _family(world: FakeWorld, **kw: Any) -> tuple[FamilyVisits, BatteryHelp]:
    clock = lambda: world.t  # noqa: E731
    return FamilyVisits(world, seed=7, clock=clock), BatteryHelp(clock=clock, **kw)


def _drive(
    world: PackWorld,
    visits: FamilyVisits,
    help_: BatteryHelp,
    seconds: float,
    *,
    on: bool = True,
    step: float = 0.1,
    acts: list[Any] | None = None,
) -> list[int]:
    """The family loop for ``seconds`` (the server's: the household looks, then the visits move
    on); the percents each swap was made at. ``acts``: collects the acts Duck Core is given (a
    household visit gives none: only a family member's own button does)."""
    told: list[int] = []
    until = world.t + seconds
    while world.t < until:
        world.t = round(world.t + step, 3)
        out = help_.step(visits, helping=on, pack=world.pack())
        if out is not None:
            told.append(out)
        due = visits.tick()
        if acts is not None:
            acts.extend(due)
        else:
            assert not due, "a household visit gives Duck Core no act of its own"
    return told


def _hands(world: FakeWorld) -> list[str]:
    return [p["do"] for op, p in world.calls if op == "x_hand" and p.get("do") != "status"]


def test_the_threshold_is_duck_cores_own() -> None:
    from duckmind.duckcore.engine import MIN_MOTION_BATTERY

    assert LOW_PCT == MIN_MOTION_BATTERY * 100 // SCALE == 30


def _core(**kw: Any) -> Any:
    s = SimpleNamespace(
        bridged=True, battery_help=True, awake=True, docked=False, **{k: v for k, v in kw.items()}
    )
    return SimpleNamespace(s=s, hw=SimpleNamespace(nest=bool(kw.get("nest", False))))


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"battery_help": False}, False),  # the setting is off: it runs flat
        ({"nest": True}, False),  # a nest: it goes home to charge
        ({"bridged": False}, False),  # an unattached Duck Core swaps for itself
        ({"docked": True}, False),  # on the cable
        ({"awake": False}, False),
    ],
)
def test_the_family_helps_only_when_the_setting_is_on_and_there_is_no_nest(change, expected):
    core = _core()
    for key, value in change.items():
        if key == "nest":
            core.hw.nest = value
        else:
            setattr(core.s, key, value)
    assert helping(core, on_screen=True, world_ready=True) is expected


def test_the_family_does_not_help_a_demo_duck_or_a_body_without_a_household():
    assert not helping(_core(), on_screen=False, world_ready=True)
    assert not helping(_core(), on_screen=True, world_ready=False)


def test_the_pack_is_read_from_the_worlds_truth():
    assert pack_of(None) is None and pack_of({}) is None
    assert pack_of({"battery": {"percent": True}}) is None
    pack = pack_of(
        {"battery": {"percent": 28.6, "pg": True}, "hand": {"holding": True, "held": False}}
    )
    assert pack == {"percent": 28, "charging": True, "held": True}


def test_a_low_pack_is_swapped_once_through_the_hand_and_duck_core_is_given_no_act():
    world = PackWorld(25, walk_s=1.0)
    visits, help_ = _family(world)
    told = _drive(world, visits, help_, 15.0)
    assert world.swaps == 1 and world.percent == 100
    assert told == [25] and help_.swaps == 1 and help_.failed == 0
    assert _hands(world) == ["pickup", "place"]  # lifted, swapped while held, set back down
    assert not visits.visits  # and the person has left
    assert help_.view()["swaps"] == 1 and help_.view()["threshold_pct"] == 30


def test_a_pack_at_the_threshold_or_above_is_left_alone():
    world = PackWorld(LOW_PCT, walk_s=1.0)
    visits, help_ = _family(world)
    assert _drive(world, visits, help_, 10.0) == []
    assert world.swaps == 0 and not world.calls


def test_it_is_once_for_each_low_crossing_not_once_for_each_reading_below_it():
    world = PackWorld(25, walk_s=1.0)
    visits, help_ = _family(world)
    # the world's pack reads low for a while after the swap (a stale reading): no second visit
    real = world.pack
    stale = {"on": True}

    def pack() -> dict[str, Any] | None:
        got = real()
        if got is not None and stale["on"] and world.swaps:
            got["percent"] = 24
        return got

    world.pack = pack  # type: ignore[method-assign]
    _drive(world, visits, help_, 20.0)
    assert world.swaps == 1
    stale["on"] = False
    _drive(world, visits, help_, 10.0)
    assert world.swaps == 1  # (100 %: nothing to do)
    world.percent = 22  # it drains below the line again: a new crossing
    _drive(world, visits, help_, 15.0)
    assert world.swaps == 2 and help_.swaps == 2


def test_nothing_is_swapped_with_the_setting_off_and_the_visit_starts_when_it_goes_on():
    world = PackWorld(10, walk_s=1.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 20.0, on=False)
    assert world.swaps == 0 and not world.calls and not visits.visits
    _drive(world, visits, help_, 15.0, on=True)
    assert world.swaps == 1


def test_it_waits_while_a_hand_holds_the_duck_or_it_is_on_its_charger():
    world = PackWorld(20, walk_s=1.0)
    visits, help_ = _family(world)
    world.holding = True  # somebody's hand has it
    _drive(world, visits, help_, 5.0)
    assert not world.calls and not visits.visits
    world.holding = False
    world.charging = True  # on the contacts of a charger: the pack is being charged anyway
    _drive(world, visits, help_, 5.0)
    assert not world.calls and not visits.visits
    world.charging = False
    _drive(world, visits, help_, 15.0)
    assert world.swaps == 1


def test_it_waits_for_another_visit_and_does_not_cut_in_on_it():
    world = PackWorld(20, walk_s=1.0)
    visits, help_ = _family(world)
    visits.request("touch", "partner", {})  # the owner pressed a family member's button
    for _ in range(5):
        world.t = round(world.t + 0.1, 3)
        help_.step(visits, helping=True, pack=world.pack())
        visits.tick()
    assert help_.visit is None and not any(v.silent for v in visits.visits)
    assert not [1 for op, p in world.calls if op == "x_battery"]
    acts: list[Any] = []
    _drive(world, visits, help_, 30.0, acts=acts)
    assert world.swaps == 1 and [a.action for a in acts] == ["touch"]  # (their own act, once)


def test_a_person_still_on_the_way_turns_back_when_the_pack_no_longer_needs_swapping():
    world = PackWorld(20, walk_s=8.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 1.0)
    assert help_.visit is not None and help_.visit.state == "walking"
    world.percent = 100  # (someone else swapped it)
    _drive(world, visits, help_, 20.0)
    assert world.swaps == 0 and help_.swaps == 0 and help_.failed == 0
    assert _hands(world) == [] and not visits.visits
    assert ("x_people", {"do": "leave", "who": "owner"}) in world.calls


def test_a_refused_hand_is_not_a_swap_and_the_visit_does_not_hang_the_family():
    world = Refusing(20, walk_s=1.0)
    visits, help_ = _family(world)
    assert _drive(world, visits, help_, 20.0) == []
    assert world.swaps == 0 and help_.swaps == 0 and help_.failed == 1
    assert not visits.visits and help_.view()["visiting"] is False
    started = len([1 for op, p in world.calls if op == "x_people" and p.get("do") == "come"])
    _drive(world, visits, help_, 10.0)  # the next try waits for the retry time, it does not hammer
    assert (
        len([1 for op, p in world.calls if op == "x_people" and p.get("do") == "come"]) == started
    )
    _drive(world, visits, help_, 40.0)
    assert help_.failed >= 2 and world.swaps == 0


def test_a_body_with_no_household_to_walk_in_tries_again_later_instead_of_swapping():
    visits = FamilyVisits(None, seed=7, clock=lambda: 0.0)
    help_ = BatteryHelp(clock=lambda: 0.0, retry_s=30.0)
    pack = {"percent": 10, "charging": False, "held": False}
    assert help_.step(visits, helping=True, pack=pack) is None
    assert help_.visit is None and not visits.visits


def test_the_module_says_what_it_needs_of_duck_core():
    assert bh.PERSONA == "owner" and 0 < bh.RETRY_S <= bh.RETRY_MAX_S


class RefusesOnce(PackWorld):
    """The first lift is refused (the hand was busy); every later one is taken."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.refused = False

    def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("do") == "pickup" and not self.refused:
            self.refused = True
            raise RuntimeError("busy")
        return super()._hand(params)


class ReasonedRefusal(PackWorld):
    """The real hand's way of refusing: the lift is refused with the world's words, and its status
    keeps the last refusal as a number (``refused``: ``code`` and how many so far)."""

    def __init__(self, *a: Any, refuse: int = 1_000, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.refuse = refuse  # how many lifts it refuses before it takes one
        self.refused: dict[str, Any] | None = None

    def _hand(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("do") == "pickup" and self.refuse > 0:
            self.refuse -= 1
            count = (self.refused or {}).get("count", 0) + 1
            self.refused = {"code": REFUSE_NO_WAY_OUT, "count": count, "sim_s": self.t}
            raise RuntimeError(
                "body host x_hand: {'error': '小鴨在茶几底下，手沒辦法把牠拉出來抬起'}"
            )
        answer = super()._hand(params)
        if params.get("do") == "status":
            answer["refused"] = self.refused
        return answer


class BlindPack(Flaky, PackWorld):
    """A stalled host: the hand's status is unreadable for the first ``blind`` seconds."""


def test_a_failed_swap_is_retried_and_the_retry_swaps_the_pack_with_one_event() -> None:
    world = RefusesOnce(20, walk_s=1.0)
    visits, help_ = _family(world)
    told = _drive(world, visits, help_, 20.0)
    assert told == [] and help_.failed == 1 and help_.swaps == 0 and world.swaps == 0
    assert help_.view()["last_failure"] == FAIL_HAND_REFUSED and help_.view()["streak"] == 1
    assert help_.view()["retry_in_s"] is not None  # (waiting for the next try)
    told = _drive(world, visits, help_, 40.0)
    assert told == [20] and world.swaps == 1 and help_.swaps == 1  # the one growth event
    assert help_.failed == 1 and help_.view()["streak"] == 0 and help_.view()["retry_in_s"] is None
    _drive(world, visits, help_, 60.0)
    assert world.swaps == 1  # (once for each crossing: no second swap)


def test_a_refused_lift_says_why_the_hand_refused_as_a_number_in_the_life_block(capsys) -> None:
    world = ReasonedRefusal(20, walk_s=1.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 20.0)
    view = help_.view()
    assert view["last_failure"] == FAIL_HAND_REFUSED and view["last_refusal"] == REFUSE_NO_WAY_OUT
    assert view["failed"] == 1 and world.swaps == 0
    log = capsys.readouterr().out
    assert f"trouble {FAIL_HAND_REFUSED}, hand {REFUSE_NO_WAY_OUT}" in log  # (the family's line)
    assert f"hand refused: {REFUSE_NO_WAY_OUT}" in log  # (and the battery help's)
    for value in view.values():
        assert value is None or isinstance(value, (int, float, bool))  # (still no words)


def test_the_refusal_reason_is_not_carried_over_to_a_failure_the_hand_did_not_make() -> None:
    world = ReasonedRefusal(20, walk_s=1.0, refuse=1)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 20.0)
    assert help_.view()["last_refusal"] == REFUSE_NO_WAY_OUT
    _drive(world, visits, help_, 60.0)  # (the retry is taken: the swap is made)
    assert help_.swaps == 1 and world.swaps == 1
    help_._failed(FAIL_NO_VISIT)  # (a later failure that is not the hand's: no old reason on it)
    assert help_.view()["last_failure"] == FAIL_NO_VISIT and help_.view()["last_refusal"] is None


def test_a_hand_that_says_nothing_of_its_refusals_leaves_no_reason() -> None:
    world = Refusing(20, walk_s=1.0)  # (a hand of the old kind: refused, no number in its status)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 20.0)
    assert help_.view()["last_failure"] == FAIL_HAND_REFUSED
    assert help_.view()["last_refusal"] is None


def test_the_wait_between_tries_doubles_up_to_its_bound_and_resets_when_the_pack_is_up() -> None:
    now = {"t": 0.0}
    help_ = BatteryHelp(clock=lambda: now["t"], retry_s=30.0, retry_max_s=100.0)
    waits = []
    for _ in range(5):
        help_._failed(FAIL_NO_VISIT)
        waits.append(help_._retry_at - now["t"])
    assert waits == [30.0, 60.0, 100.0, 100.0, 100.0]
    assert help_.view()["last_failure"] == FAIL_NO_VISIT and help_.view()["streak"] == 5
    visits = FamilyVisits(None, seed=7, clock=lambda: now["t"])
    help_.step(visits, helping=True, pack={"percent": 90, "charging": False, "held": False})
    assert help_.view()["streak"] == 0  # (a pack seen back up: the next crossing starts afresh)


def test_for_a_swap_an_unknown_hand_state_means_wait_not_swap() -> None:
    world = BlindPack(blind=8.0, percent=20, walk_s=1.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 7.0)
    assert world.swaps == 0 and _hands(world) == []  # (it did not lift or swap blind)
    _drive(world, visits, help_, 40.0)
    assert world.swaps == 1 and help_.swaps == 1 and help_.failed == 0
    assert _hands(world) == ["pickup", "place"] and not world.holding


def test_a_swap_visit_ends_with_the_duck_placed_back_even_when_the_set_down_is_refused() -> None:
    world = BlindPack(refuse_place=2, percent=20, walk_s=1.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 45.0)
    assert world.swaps == 1 and help_.swaps == 1 and not world.holding and not visits.visits


def test_the_life_block_has_numbers_only_and_nothing_is_tried_with_the_setting_off() -> None:
    world = RefusesOnce(20, walk_s=1.0)
    visits, help_ = _family(world)
    _drive(world, visits, help_, 60.0, on=False)
    assert not world.calls and help_.view()["failed"] == 0 and help_.view()["last_failure"] is None
    for value in help_.view().values():
        assert value is None or isinstance(value, (int, float, bool))  # (no words)


# ── the whole path (a real MuJoCo home) ─────────────────────────────────────────


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


class HeadlessPort:
    """The world server's ops for a headless home: what ``WorldPort.call`` is on a live body."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.ops: dict[str, Any] = {}
        for part in world.kit.parts:
            if hasattr(part, "ops"):
                self.ops.update(part.ops())
        self.swaps = 0

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        if op == "x_battery" and params.get("do") == "swap":
            self.swaps += 1
        return self.ops[op](self.world, dict(params))

    def truth(self) -> dict[str, Any]:
        with self.world.world.lock:
            return {
                "battery": self.world.kit.battery.snapshot(self.world.world),
                "hand": self.world.kit.hand.snapshot(self.world.world),
            }


class _Driver:
    def __init__(self) -> None:
        self.notes: list[tuple[str, str]] = []

    def household_note(self, name: str, words: Any) -> None:
        self.notes.append((name, str(words)))


class Home:
    """A bridged life (no nest) over a real home: the server's family loop, the neural duck's
    power sense read from the world's pack, and Duck Core hearing what the duck lived."""

    def __init__(self, tmp_path: Path, percent: int, *, battery_help: bool = True) -> None:
        from duckmind.world import worldkit

        self.world = worldkit.Headless("real_home,no_nest", mjcf_dir=_mjcf())
        self.world.kit.battery.command({"do": "set", "soc_ppm": percent * 10_000})
        self.port = HeadlessPort(self.world)
        self.life = Life(tmp_path / "duck", Wall(BIRTH), backdate=14)
        self.life.host.driver.lanes.brain = SimpleNamespace(learn=object(), hab=object())
        self.driver = _Driver()
        self.live = SimpleNamespace(
            world_ready=lambda: True,
            world_truth=self.port.truth,
            neuro=self.driver,
            after_call=lambda *a, **k: None,
        )
        clock = lambda: float(self.world.data.time)  # noqa: E731
        state = self.life.state
        # The server's own family pass (SimState.family_step) on this household. The state's own
        # threads stay idle (they only act on a state that has a body): nothing else steps the world.
        self.server = SimpleNamespace(
            family=FamilyVisits(self.port, seed=4107, clock=clock),
            battery_help=BatteryHelp(clock=clock),
            live=self.live,
            life_host=self.life.host,
            session=state.session,
            lock=state.lock,
            act_due=state.act_due,
        )
        for name in ("family_step", "_household_pack"):
            setattr(self.server, name, MethodType(getattr(SimState, name), self.server))
        if not battery_help:
            dispatch(state.session, "set_setting", {"key": "battery_help", "value": False})
        self.sec = 0

    @property
    def core(self) -> Any:
        return self.life.core

    def swaps(self) -> list[Any]:
        return [e for e in self.core.log.events if e.kind == "power.battery_swap"]

    def pack(self) -> dict[str, Any]:
        return self.world.kit.battery.snapshot(self.world.world)

    def run(self, seconds: int) -> None:
        for _ in range(seconds):
            for _ in range(4):  # the family looks a few times a second
                self.world.run(0.25)
                self.server.family_step()
            pack = self.pack()
            frame = C.SensorFrame(
                t_ms=self.sec * 1000,
                power=C.PowerSense(
                    battery=pack["soc_ppm"] / 1_000_000,
                    on_dock=bool(pack["pg"]),
                    charging=bool(pack["charging"]),
                ),
            )
            self.life.host.on_tick(self.sec * 1000, frame, None, None)
            self.life.second()
            self.sec += 1

    def until_low(self) -> None:
        while self.pack()["percent"] >= LOW_PCT:
            self.run(1)
            assert self.sec < 120, "the pack never fell below the line"


def test_a_no_nest_duck_with_battery_help_has_its_pack_swapped_once_and_duck_core_hears_it_once(
    tmp_path,
):
    home = Home(tmp_path, 31)
    core = home.core
    assert core.s.bridged and core.s.battery_help and not core.hw.nest
    home.until_low()
    low = home.pack()["percent"]
    assert low < LOW_PCT
    home.run(75)  # a person walks over, lifts it, swaps the pack, sets it down, leaves
    pack = home.pack()
    assert pack["percent"] >= 90 and not pack["empty"]  # (a fresh pack, drained a little since)
    assert home.port.swaps == 1
    assert len(home.swaps()) == 1 and home.swaps()[0].payload["by"] == "family"
    assert home.life.host.delivered == 1  # (the neural duck's one `feed` experience)
    assert core.s.battery == SCALE  # Duck Core's battery agrees with the pack it was given
    assert home.driver.notes == [("battery_help", home.driver.notes[0][1])]
    assert str(low) in home.driver.notes[0][1]
    hand = home.world.kit.hand.snapshot(home.world.world)
    assert not hand["holding"] and hand["script"] is None  # it was set back down
    home.run(60)  # and nothing more comes of it: no second swap, no second event
    assert home.port.swaps == 1 and len(home.swaps()) == 1
    assert home.server.battery_help.view()["swaps"] == 1


def test_the_swap_is_one_growth_event_the_same_one_a_swap_the_duck_lived_alone_gives(tmp_path):
    """The family's swap and the same pack jumping up with nobody's visit (the existing page
    stimulus) leave Duck Core's log the same: one swap, nothing the visit added of its own."""
    home = Home(tmp_path / "a", 31)
    home.until_low()
    seam = len(home.core.log.events)
    home.run(75)
    family_kinds = [e.kind for e in home.core.log.events[seam:]]
    alone = Home(tmp_path / "b", 31, battery_help=False)
    alone.until_low()
    seam2 = len(alone.core.log.events)
    alone.run(5)
    alone.world.kit.battery.command({"do": "swap"})  # (the page's 「換一顆充飽的電池」)
    alone.run(70)  # (as long as the family's run: a minute's end counts the same debuts in both)
    alone_kinds = [e.kind for e in alone.core.log.events[seam2:]]
    assert family_kinds.count("power.battery_swap") == 1
    assert alone_kinds.count("power.battery_swap") == 1
    assert sorted(set(family_kinds) - {"power.battery_swap"}) == sorted(
        set(alone_kinds) - {"power.battery_swap"}
    )
    assert alone.port.swaps == 0 and alone.driver.notes == []


def test_with_battery_help_off_the_duck_runs_flat_as_the_setting_says(tmp_path):
    home = Home(tmp_path, 31, battery_help=False)
    assert not home.core.s.battery_help
    home.until_low()
    low = home.pack()["percent"]
    home.run(75)
    assert home.port.swaps == 0 and home.swaps() == [] and home.driver.notes == []
    assert home.pack()["percent"] <= low  # it only went down
    assert not home.server.family.visits and home.world.kit.people.snapshot(home.world.world) == []


def test_with_a_nest_nobody_swaps_the_pack_for_it(tmp_path):
    home = Home(tmp_path, 31)
    home.core.hw = replace(home.core.hw, nest=True)  # (a nest duck goes home to charge)
    home.until_low()
    home.run(60)
    assert home.core.hw.nest
    assert home.port.swaps == 0 and home.swaps() == [] and home.driver.notes == []
    assert home.world.kit.people.snapshot(home.world.world) == []


def test_a_demo_duck_on_screen_is_not_helped(tmp_path):
    home = Home(tmp_path, 31)
    home.life.host.driver.lanes.brain = None  # the page shows another duck of its picker
    home.until_low()
    home.run(30)
    assert home.port.swaps == 0 and home.swaps() == []


def test_a_duck_sitting_under_the_coffee_table_still_has_its_pack_swapped(tmp_path):
    """蛋黃 sits under the coffee table (40 cm clear, on a thick rug) for hours. Its swap visit was
    refused 110 times against 79 swaps in ~50 h: the hand could not slide it out sideways over the
    rug's edge and round the table's legs. A person reaches under the table: the swap is made."""
    from duckmind.world import worldkit

    home = Home(tmp_path, 31)
    world = home.world
    qadr = world.robot.qadr
    world.data.qpos[qadr : qadr + 2] = [-0.8, -2.0]  # (under the table, on the rug)
    world.data.qpos[qadr + 2] = 0.09
    world.data.qpos[qadr + 3 : qadr + 7] = worldkit.yaw_quat(0.0)
    world.data.qvel[:] = 0
    world.run(1.0)
    top = world.layout.element("coffee_table_1_top")
    x0, y0, x1, y1 = top.footprint()
    assert x0 < world.trunk[0] * 1000 < x1 and y0 < world.trunk[1] * 1000 < y1
    home.until_low()
    home.run(90)  # a person walks over, takes it out from under the table, swaps, sets it down
    view = home.server.battery_help.view()
    assert view["swaps"] == 1 and view["failed"] == 0 and view["last_failure"] is None
    assert home.port.swaps == 1 and home.pack()["percent"] >= 90
    assert len(home.swaps()) == 1 and home.swaps()[0].payload["by"] == "family"
    hand = world.kit.hand.snapshot(world.world)
    assert not hand["holding"] and hand["script"] is None and hand["refused"] is None
