"""What the duck does with a command it heard (neuro/command_brain.py): obeys a built-in verb when it is awake,
not afraid, not held, not fallen or in flight and not docking, in the duck's own time and from its own
sighting of the speaker; refuses with a reason code otherwise; reacts to a trick it was never taught
(a tilt of the head and a chirp, never the trick); and a command is a claim the live bridge's per-tick
``set_extra`` cannot erase."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
from live_bridge_rig import frame, make_brain, person_sight, run_brain

from duckmind.live.bridge import LiveBridge
from duckmind.neuro import command_brain as CB
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro.commands import Command

pytestmark = pytest.mark.usefixtures("no_network")

SEEDS = (4107, 42, 9101, 1)  # an explorer, a songbird, a dreamer, a dreamer (tempo 0.49)


def _frame(
    t: float,
    *,
    posture: str = "sit",
    sightings: tuple[C.Sighting, ...] = (),
    fallen: bool = False,
    battery: float = 0.8,
    light: str | None = None,
    events: tuple[str, ...] = (),
    on_dock: bool = False,
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    floor: tuple[tuple[float, float, str], ...] = (),
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tof: C.TofSense | None = None,
) -> C.SensorFrame:
    """``floor``: a look's floor line (bearing, free m, end) per sector, seen at this frame; ``tof``: its ToF's summary."""
    f = frame(int(round(t * 1000)), *sightings, battery=battery)
    camera = f.camera if light is None else replace(f.camera, light=light)
    if floor:
        line = tuple(C.FreeSector(bearing=b, width=0.13, free_m=m, end=e) for b, m, e in floor)
        camera = replace(camera, floor=line, floor_t_ms=int(round(t * 1000)))
    power = replace(f.power, on_dock=on_dock)
    body = C.BodySense(posture=posture, fallen=fallen, odom=odom, gyro=gyro)
    return replace(f, body=body, camera=camera, power=power, events=events, tof=tof or f.tof)


def _person(range_m: float = 1.5) -> tuple[C.Sighting, ...]:
    return (person_sight(1, range_m=range_m),)


def _duck(seed: int = 4107, *, bridge: bool = True, settle: float = 6.0, **kw: Any):
    brain, clock = make_brain(seed=seed)
    if bridge:
        brain.live = LiveBridge(clock=clock)
    brain.commands.clock = clock
    run_brain(brain, clock, settle, frames=lambda t: _frame(t, **kw))
    return brain, clock


def _tell(brain: Any, clock: Any, verb: str, seconds: float = 4.0, **kw: Any) -> tuple[Any, list[Any]]:
    ticket = brain.commands.hear(Command(verb))
    cmds = run_brain(brain, clock, seconds, frames=lambda t: _frame(t, **kw))
    return ticket, cmds


def _last(brain: Any) -> dict[str, Any]:
    return brain.commands.last


# -- obeying --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_a_builtin_verb_is_obeyed_by_every_temperament(seed: int) -> None:
    for verb, posture, behaviour in (
        ("look_at_me", "sit", "watch"),
        ("stand", "sit", "stand_up"),
        ("sit", "stand", "sit_down"),
        ("stop", "stand", "freeze"),
        ("come", "stand", "follow"),
        ("follow", "stand", "follow"),
    ):
        brain, clock = _duck(seed, posture=posture, sightings=_person())
        ticket, cmds = _tell(brain, clock, verb, posture=posture, sightings=_person())
        assert ticket.answer["state"] == "heard", (verb, ticket.answer)
        assert behaviour in {c.behaviour for c in cmds}, (seed, verb)
        assert _last(brain)["state"] in ("doing", "done"), (seed, verb, _last(brain))


def test_the_answer_is_given_at_the_next_tick_and_says_nothing_of_the_motion() -> None:
    brain, clock = _duck(sightings=_person())
    ticket = brain.commands.hear(Command("look_at_me"))
    assert not ticket.event.is_set()  # (the web thread waits for the brain's tick)
    run_brain(brain, clock, 0.1, frames=lambda t: _frame(t, sightings=_person()))
    assert ticket.event.is_set()
    assert ticket.answer["state"] == "heard" and "motion_ms" not in ticket.answer


def test_a_command_already_as_asked_is_done_at_once() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    ticket, _ = _tell(brain, clock, "stand", 0.5, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "done" and ticket.answer.get("already") is True
    assert _last(brain)["state"] == "done"


def test_a_command_heard_during_a_short_performance_waits_for_it_and_is_obeyed() -> None:
    """Live :8796 (2026-10-23): 'sit' said while the duck played a short moment (or a bump stop held
    it a beat) was refused ``body_busy`` at once; a pet sits when the moment is over."""
    brain, clock = _duck(posture="stand", sightings=_person())
    brain._perform_until_ms = int(round(clock.t * 1000)) + 1000
    ticket, cmds = _tell(brain, clock, "sit", 4.0, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "heard", ticket.answer
    assert "sit_down" in {c.behaviour for c in cmds}
    assert _last(brain)["state"] in ("doing", "done"), _last(brain)


def test_a_duck_walking_home_on_a_low_pack_says_so_at_once() -> None:
    """Live :8796 (2026-10-24): 'sit' / 'stop' said while it walked home to charge ended 'gave_up busy'
    after 5 s; the claim never takes the body from a pack running low, so it says why at once."""
    brain, clock = _duck(posture="stand", sightings=_person())
    brain._feat[S.F["homing"]] = S.HOMING_RELEASE
    brain.behaviour = "return_dock"
    ticket, _ = _tell(brain, clock, "sit", 0.1, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "low_battery", ticket.answer



def test_sit_finishes_when_the_body_has_sat_down() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    brain.commands.hear(Command("sit"))
    state = {"posture": "stand"}

    def frames(t: float) -> C.SensorFrame:
        return _frame(t, posture=state["posture"], sightings=_person())

    for _ in range(80):
        run_brain(brain, clock, 0.05, frames=frames)
        if brain.behaviour == "sit_down":
            state["posture"] = "sit"  # the body has sat down
    last = _last(brain)
    assert last["state"] == "done" and last["motion_ms"] is not None
    assert last["motion_ms"] <= last["done_ms"]


def test_sit_times_the_motion_at_the_start_of_the_change_not_the_end() -> None:
    """Live :8783 (2026-10-24): a sit-down's ``motion_ms`` was taken when the posture class had
    changed — the END of the sit-down (1.9 s median, over the spec's 1.5 s). The body's own sense
    says the change has begun (``transition``) a second before that; the latency is from there."""
    brain, clock = _duck(posture="stand", sightings=_person())
    brain.commands.hear(Command("sit"))
    state = {"posture": "stand"}

    def frames(t: float) -> C.SensorFrame:
        return _frame(t, posture=state["posture"], sightings=_person())

    for _ in range(80):
        run_brain(brain, clock, 0.05, frames=frames)
        if brain.behaviour == "sit_down":
            break
    assert brain.behaviour == "sit_down"
    state["posture"] = "transition"  # the body has begun the sit-down
    run_brain(brain, clock, 0.5, frames=frames)
    state["posture"] = "sit"  # the body has sat down
    run_brain(brain, clock, 0.5, frames=frames)
    last = _last(brain)
    assert last["state"] == "done" and last["motion_ms"] is not None
    assert last["motion_ms"] <= last["done_ms"] - 400, last


def test_stand_times_the_motion_at_the_start_of_the_change_not_the_end() -> None:
    brain, clock = _duck(posture="sit", sightings=_person())
    brain.commands.hear(Command("stand"))
    state = {"posture": "sit"}

    def frames(t: float) -> C.SensorFrame:
        return _frame(t, posture=state["posture"], sightings=_person())

    for _ in range(80):
        run_brain(brain, clock, 0.05, frames=frames)
        if brain.behaviour == "stand_up":
            break
    assert brain.behaviour == "stand_up"
    state["posture"] = "transition"  # the body has begun the rise
    run_brain(brain, clock, 0.5, frames=frames)
    state["posture"] = "stand"  # the body is up
    run_brain(brain, clock, 0.5, frames=frames)
    last = _last(brain)
    assert last["state"] == "done" and last["motion_ms"] is not None
    assert last["motion_ms"] <= last["done_ms"] - 400, last


def test_the_beat_leaves_the_body_room_within_the_specs_latency() -> None:
    """Spec §4: a command to the start of motion ≤ 1.5 s (median). The body's own change (the head
    settling at HOME, robotd's rise) takes ~0.9 s live (:8783, 2026-10-24), so the personality's
    beat may take no more than the rest of that budget."""
    beats = {}
    for seed in SEEDS:
        brain, clock = _duck(seed, sightings=_person())
        ticket = brain.commands.hear(Command("look_at_me"))
        run_brain(brain, clock, 0.1, frames=lambda t: _frame(t, sightings=_person()))
        beats[seed] = ticket.answer["beat_s"]
    assert min(beats.values()) >= CB.BEAT_BASE_S
    assert max(beats.values()) <= 0.6, beats  # (1.5 s − the body's own ~0.9 s)
    assert len(set(beats.values())) > 1  # (a dreamer is slower than an explorer)
    assert beats[9101] > beats[4107]


def test_the_personality_changes_only_the_beat_never_the_answer() -> None:
    beats = {}
    for seed in SEEDS:
        brain, clock = _duck(seed, sightings=_person())
        ticket = brain.commands.hear(Command("look_at_me"))
        run_brain(brain, clock, 0.1, frames=lambda t: _frame(t, sightings=_person()))
        assert ticket.answer["state"] == "heard"
        beats[seed] = ticket.answer["beat_s"]
    assert max(beats.values()) <= 0.6 and min(beats.values()) >= CB.BEAT_BASE_S
    assert len(set(beats.values())) > 1  # (a dreamer is slower than an explorer)
    assert beats[9101] > beats[4107]


def test_the_songbird_chirps_before_it_acts() -> None:
    brain, clock = _duck(42, sightings=_person())
    brain.commands.hear(Command("look_at_me"))
    cmds = run_brain(brain, clock, 3.0, frames=lambda t: _frame(t, sightings=_person()))
    assert any(c.vocal == "chirp" for c in cmds)


# -- the speaker comes from the duck's own sight --------------------------------------------------


def test_come_and_look_at_me_look_for_the_person_then_refuse() -> None:
    for verb in ("come", "look_at_me", "follow"):
        brain, clock = _duck(posture="stand")  # nobody in view, nobody believed
        ticket, _ = _tell(brain, clock, verb, CB.LOOK_FOR_S + 1.0, posture="stand")
        assert ticket.answer["state"] == "heard"  # (it heard: it looks about for them first)
        assert _last(brain)["state"] == "refused" and _last(brain)["reason"] == "no_one_seen"


def test_a_person_seen_a_moment_ago_is_believed_where_they_were() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(2.0))
    run_brain(brain, clock, 1.0, frames=lambda t: _frame(t, posture="stand"))  # (out of view)
    ticket, _ = _tell(brain, clock, "come", 0.3, posture="stand")
    assert ticket.answer["state"] == "heard"


def test_come_is_done_when_the_person_is_close() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(1.5))
    brain.commands.hear(Command("come"))
    for step in range(120):
        rng = max(0.4, 1.5 - 0.03 * step)  # (it walks up to them)
        run_brain(brain, clock, 0.05, frames=lambda t: _frame(t, posture="walk", sightings=_person(rng)))
    assert _last(brain)["state"] == "done"


def test_a_person_who_leaves_ends_come() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(2.5))
    brain.commands.hear(Command("come"))
    run_brain(
        brain,
        clock,
        CB.PERSON_LOST_S + CB.LOOK_FOR_LOST_S + 2.0,
        frames=lambda t: _frame(t, posture="stand"),
    )
    last = _last(brain)
    assert last["state"] == "gave_up" and last["reason"] == "no_one_seen"


# -- refusing -------------------------------------------------------------------------------------


def _asleep(seed: int = 4107):
    """A drowsy duck in the small hours, dozing: it stays asleep through the tests."""
    brain, clock = make_brain(seed=seed)
    brain.commands.clock = clock
    brain.drives.sleep_pressure = 1.0
    brain.drives.energy = 0.1
    sel = brain.sel
    sel.current = sel.names.index("doze")
    brain.behaviour = "doze"
    return brain, clock


def _night(t: float, **kw: Any) -> C.SensorFrame:
    return replace(_frame(t, **kw), day_minute=180.0)


def test_an_asleep_duck_does_not_answer() -> None:
    brain, clock = _asleep()
    run_brain(brain, clock, 0.2, frames=_night)
    assert brain.behaviour == "doze"
    ticket = brain.commands.hear(Command("stand"))
    run_brain(brain, clock, 0.3, frames=_night)
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "asleep"


def test_a_frightened_duck_does_not_answer() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    brain.drives.fear = 0.8
    ticket, _ = _tell(brain, clock, "sit", 0.2, posture="stand", sightings=_person())
    assert ticket.answer["reason"] in ("afraid", "fleeing", "body_busy")
    assert ticket.answer["state"] == "refused"


def test_a_duck_that_is_held_does_not_answer() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    run_brain(brain, clock, 0.5, frames=lambda t: _frame(t, posture="stand", events=("lifted",)))
    ticket, _ = _tell(brain, clock, "sit", 0.2, posture="stand", sightings=_person(), events=("lifted",))
    assert ticket.answer["state"] == "refused"
    assert ticket.answer["reason"] in ("held", "body_busy")


def test_a_fallen_duck_does_not_answer() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    ticket, _ = _tell(
        brain, clock, "come", 0.2, posture="stand", sightings=_person(), fallen=True
    )
    assert ticket.answer["state"] == "refused"
    assert ticket.answer["reason"] in ("fallen", "body_busy")


def test_a_duck_in_flight_does_not_answer() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    brain._startled_s = brain._t_prev / 1000.0
    ticket, _ = _tell(brain, clock, "follow", 0.2, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "refused"
    assert ticket.answer["reason"] in ("fleeing", "afraid", "body_busy")


def test_a_duck_docking_does_not_answer() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    brain.dock.mode = "docking"
    ticket, _ = _tell(brain, clock, "come", 0.2, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "docking"
    brain.dock.mode = "idle"


def test_a_duck_that_walks_with_a_low_battery_stays() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(), battery=0.2)
    ticket, _ = _tell(brain, clock, "come", 0.2, posture="stand", sightings=_person(), battery=0.2)
    assert ticket.answer["state"] == "refused"
    assert ticket.answer["reason"] in ("low_battery", "charging")


def test_a_duck_in_the_dark_does_not_walk_but_still_looks() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(), light="black")
    ticket, _ = _tell(brain, clock, "come", 0.2, posture="stand", sightings=_person(), light="black")
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "too_dark"


def test_go_home_with_no_home_known_says_so() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    ticket, _ = _tell(brain, clock, "go_home", 0.2, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "no_home"


def test_go_home_is_done_when_it_is_home() -> None:
    brain, clock = _duck(posture="sit", sightings=_person())
    brain.dock.mode = "home"
    ticket, _ = _tell(brain, clock, "go_home", 0.3, sightings=_person(), on_dock=True)
    assert ticket.answer["state"] == "done" and ticket.answer.get("already") is True


def test_a_duck_leaving_its_dock_is_not_sent_back_in_the_same_breath() -> None:
    brain, clock = _duck(posture="sit", sightings=_person(), on_dock=True)
    brain.dock.mode = "leaving"
    ticket, _ = _tell(brain, clock, "go_home", 0.2, sightings=_person(), on_dock=True)
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "docking"


def test_every_refusal_has_its_words_in_the_catalog() -> None:
    from duckmind.i18n import render

    for code, words in CB.REASON_MSG.items():
        assert words.key == f"neuro.command.reason.{code}"
        for lang in ("en", "ja", "es", "zh-Hans", "ko"):
            assert render(words, lang).strip(), (code, lang)
    for state, words in CB.STATE_MSG.items():
        assert words.key == f"neuro.command.state.{state}"


# -- tricks it has not been taught ----------------------------------------------------------------


def _teach(brain: Any, verb: str, proficiency: float) -> None:
    skill = brain.learn.tricks.skills[verb]
    skill.proficiency, skill.total = proficiency, 5 if proficiency else 0


def _body_of(verb: str) -> str:
    return CB.BEHAVIOUR_OF[verb]


@pytest.mark.parametrize("verb", ["lie_down", "spin", "bow", "shake"])
def test_a_taught_trick_is_done_when_told(verb: str) -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    _teach(brain, verb, 0.95)
    ticket, cmds = _tell(brain, clock, verb, 5.0, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "heard"
    acting = [i for i, c in enumerate(cmds) if c.behaviour == _body_of(verb)]
    assert acting  # the act itself, on its own channel
    # no puzzled chirp from the command (the reaction comes on the very next tick; a call of its own later is not it)
    assert not any(c.vocal == "inquire" for c in cmds[: acting[0] + 1])
    assert _last(brain)["state"] in ("doing", "done") and _last(brain)["proficiency"] == 0.95


def test_a_told_spin_that_is_done_stops_turning_even_if_its_act_began_again_midway() -> None:
    """(real simulator 2026-10-26) robotd's quick-turn hand-back blipped the behaviour for one tick, the spin act began again
    from nought and the duck went on turning after "done": 550-580° in all. Done means the turn is made: it stops there."""
    brain, clock = _duck(posture="stand", sightings=_person())
    _teach(brain, "spin", 0.95)
    brain.commands.hear(Command("spin"))
    yaw = [0.0]
    began_again = [False]

    def frames(t: float) -> C.SensorFrame:
        if brain.behaviour == "spin":
            yaw[0] += 3.0 / 20.0  # (its odometry: the body turns 3 rad/s while it spins)
            if not began_again[0] and yaw[0] > 3.0:
                began_again[0] = True
                brain._spun = 0.0  # the act began again (a new bout after a one-tick blip)
        heading = math.atan2(math.sin(yaw[0]), math.cos(yaw[0]))
        return _frame(t, posture="stand", sightings=_person(), odom=(0.0, 0.0, heading))

    run_brain(brain, clock, 8.0, frames=frames)
    assert began_again[0] and _last(brain)["state"] == "done"
    assert math.radians(300) <= yaw[0] <= math.radians(360)


@pytest.mark.parametrize("verb", ["lie_down", "spin", "bow", "shake"])
def test_an_untaught_trick_mostly_gets_the_puzzled_tilt(verb: str) -> None:
    done = chirped = refused = 0
    for seed in range(40):
        brain, clock = _duck(seed=1000 + seed, posture="stand", sightings=_person())
        ticket, cmds = _tell(brain, clock, verb, 3.0, posture="stand", sightings=_person())
        if ticket.answer["state"] == "refused":
            refused += 1
            assert ticket.answer["reason"] == "not_learned" and ticket.answer["reaction"] == "tilt"
            assert _last(brain)["p_obey"] <= 0.3
            assert not any(c.behaviour == _body_of(verb) for c in cmds)
            chirped += any(c.vocal == "inquire" for c in cmds)
        else:
            done += 1
    assert done <= 12  # a lucky guess now and then (the base chance is 0.15), never most of the time
    assert chirped >= refused // 2  # it says it heard (a call already in its beak may take the place)


def test_a_taught_trick_is_never_done_by_a_duck_that_is_held_or_afraid() -> None:
    held, clock = _duck(posture="stand", sightings=_person())
    _teach(held, "bow", 1.0)
    run_brain(held, clock, 0.5, frames=lambda t: _frame(t, posture="stand", events=("lifted",)))
    ticket, cmds = _tell(held, clock, "bow", 1.0, posture="stand", sightings=_person(), events=("lifted",))
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] in ("held", "body_busy")
    assert not any(c.behaviour == "bow" for c in cmds)
    scared, clock = _duck(posture="stand", sightings=_person())
    _teach(scared, "bow", 1.0)
    scared.drives.fear = 0.8
    ticket, cmds = _tell(scared, clock, "bow", 1.0, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "refused"
    assert not any(c.behaviour == "bow" for c in cmds)


def test_a_trick_the_duck_is_too_young_for_is_refused_for_its_age() -> None:
    brain, clock = make_brain(seed=4107, gd=20)  # (a duckling: its own juvenile day is far off)
    brain.commands.clock = clock
    brain.can_roll = True
    _teach(brain, "flip", 1.0)
    run_brain(brain, clock, 4.0, frames=lambda t: _frame(t, posture="stand"))
    ticket = _asked(brain, clock)
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "too_young"
    assert not any(c.behaviour == "flip" for c in run_brain(brain, clock, 1.0, frames=lambda t: _frame(t, posture="stand")))


def _roller(gd: int = 80, **kw: Any):
    """A duck old enough to flip whose body can roll, taught to 1.0, standing calm on open floor for a moment."""
    brain, clock = make_brain(seed=4107, gd=gd)
    brain.commands.clock = clock
    brain.can_roll = True
    _teach(brain, "flip", 1.0)
    run_brain(brain, clock, 4.0, frames=lambda t: _frame(t, posture="stand", **kw))
    return brain, clock


def _asked(brain: Any, clock: Any, seconds: float = 1.0, **kw: Any) -> Any:
    ticket = brain.commands.hear(Command("flip"))
    run_brain(brain, clock, seconds, frames=lambda t: _frame(t, posture="stand", **kw))
    return ticket


def test_a_calm_standing_duck_on_open_floor_takes_the_flip_up() -> None:
    brain, clock = _roller()
    ticket = _asked(brain, clock)
    assert ticket.answer["state"] == "heard", ticket.answer


@pytest.mark.parametrize(
    ("why", "reason"),
    [
        ({"posture": "sit"}, "not_standing"),
        ({"battery": 0.45}, "battery_half"),
        ({"events": ("stumbled",)}, "not_steady"),
        ({"on_dock": True}, "not_floor"),
    ],
)
def test_the_flip_is_refused_with_its_reason_from_the_ducks_own_senses(
    why: dict[str, Any], reason: str
) -> None:
    brain, clock = _roller()
    kw = {k: v for k, v in why.items() if k != "posture"}
    ticket = brain.commands.hear(Command("flip"))
    run_brain(brain, clock, 1.0, frames=lambda t: _frame(t, posture=why.get("posture", "stand"), **kw))
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == reason, ticket.answer
    assert not any(c.behaviour == "flip" for c in run_brain(brain, clock, 1.0, frames=lambda t: _frame(t, posture="stand")))


def test_the_flip_is_refused_when_something_is_in_the_run_ahead_or_close_by() -> None:
    # (the real-simulator margin run, LEARN-TRICKS §4.4: it rolls over and gets up by itself with the first solid 0.6 m
    # ahead and the nearest 0.3 m away; closer ahead some rolls stopped against it part way)
    ahead = (C.Sighting(kind="thing", bearing=0.0, range_m=0.5, size=0.1, track=9),)
    near = (C.Sighting(kind="thing", bearing=1.5, range_m=0.25, size=0.1, track=10),)
    clear_ahead = (C.Sighting(kind="thing", bearing=0.0, range_m=0.8, size=0.1, track=11),)
    clear_beside = (C.Sighting(kind="thing", bearing=1.5, range_m=0.4, size=0.1, track=12),)
    for sights, reason in ((ahead, "no_room"), (near, "no_room"), (clear_ahead, None), (clear_beside, None)):
        brain, clock = _roller()  # (calm on open floor; the thing comes into view as it is asked)
        ticket = _asked(brain, clock, 0.3, sightings=sights)
        if reason is None:
            assert ticket.answer["state"] == "heard", ticket.answer
        else:
            assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == reason, ticket.answer


def test_a_wall_beside_the_roll_path_leaves_room_but_one_across_it_does_not() -> None:
    # (FlipRoom, LEARN-TRICKS §4.5: the roll goes straight along its heading; a wall its look saw 25-35° to the side at
    # 0.4-0.6 m is beside that path — the margin run measured the room along the straight path, the body's width either side)
    beside = ((-0.55, 0.42, "wall"), (-0.42, 0.48, "wall"), (-0.29, 0.62, "wall"), (-0.16, 0.9, "wall"), (-0.03, 1.5, "unknown"))
    across = tuple((b, 0.5, "wall") for b in (-0.26, -0.13, 0.0, 0.13, 0.26))
    for floor, reason in ((beside, None), (across, "no_room")):
        brain, clock = _roller()
        run_brain(brain, clock, 0.5, frames=lambda t: _frame(t, posture="stand", floor=floor))
        ticket = _asked(brain, clock, 0.5, floor=floor)
        if reason is None:
            assert ticket.answer["state"] == "heard", ticket.answer
        else:
            assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == reason, ticket.answer


def test_a_brush_guess_beside_it_leaves_room_but_a_bump_ahead_does_not() -> None:
    # (FlipRoom, LEARN-TRICKS §4.5: a brush is the wall memory's guess from a kick of the heading — the side it came from, a
    # fixed 0.13 m away; on the most open spots none of the 14 brushes had a true solid on that side closer than 0.33 m. A
    # bump is the trunk stopped against something: it stays in the way)
    for events, gyro, reason in ((("brushed",), (0.0, 0.0, -0.9), None), (("blocked",), (0.0, 0.0, 0.0), "no_room")):
        brain, clock = _roller()
        run_brain(brain, clock, 0.1, frames=lambda t: _frame(t, posture="walk", events=events, gyro=gyro))
        run_brain(brain, clock, 2.5, frames=lambda t: _frame(t, posture="stand"))
        ticket = _asked(brain, clock, 0.5)
        if reason is None:
            assert ticket.answer["state"] == "heard", ticket.answer
        else:
            assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == reason, ticket.answer


def test_a_tof_return_just_beside_its_body_corridor_close_ahead_is_in_the_way_but_not_one_farther() -> None:
    # (FlipRoom, LEARN-TRICKS §4.5: the ToF's left / right are the nearest hits beside its ±0.10 m body corridor, anywhere up
    # to 0.7 m aside — one 0.43 m away can be 0.11 m beside the roll's path: a ToF duck accepted with a sofa leg 0.42 m ahead.
    # Counted out to 0.8 m they also took things well beside it: 5 of 37 clear open asks refused at 0.57-0.8 m)
    for side_mm, reason in ((430, "no_room"), (700, None)):
        tof = C.TofSense(present=True, ahead_mm=1400, left_mm=side_mm, right_mm=1400)
        brain, clock = _roller(tof=tof)
        ticket = _asked(brain, clock, 0.5, tof=tof)
        if reason is None:
            assert ticket.answer["state"] == "heard", ticket.answer
        else:
            assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == reason, ticket.answer


def test_the_flip_is_refused_before_it_has_stood_calm_and_after_a_recent_roll() -> None:
    brain, clock = make_brain(seed=4107, gd=80)
    brain.commands.clock = clock
    brain.can_roll = True
    _teach(brain, "flip", 1.0)
    run_brain(brain, clock, 0.4, frames=lambda t: _frame(t, posture="stand"))
    ticket = _asked(brain, clock, 0.3)
    assert ticket.answer["reason"] == "not_steady"  # (it has stood only a moment)
    brain, clock = _roller()
    brain.commands._rolls.append(clock() - 30.0)
    assert _asked(brain, clock).answer["reason"] == "rest_first"  # (rolled 30 s ago: 120 s apart)


def test_a_body_that_cannot_roll_says_not_yet() -> None:
    brain, clock = _roller()
    brain.can_roll = False
    assert _asked(brain, clock).answer["reason"] == "not_yet"


def test_the_flip_is_done_when_it_has_gone_over_and_stood_up_by_itself() -> None:
    brain, clock = _roller()
    ticket = _asked(brain, clock, 0.5)
    assert ticket.answer["state"] == "heard"

    def rolling(t: float) -> C.SensorFrame:
        k = t - t0
        tilt = 0.0 if k < 0.3 else (2.8 if k < 1.0 else 0.05)  # over, then up again
        f = _frame(t, posture="stand")
        return replace(f, body=replace(f.body, tilt=tilt))

    t0 = clock()
    run_brain(brain, clock, 0.8, frames=rolling)
    assert _last(brain)["state"] != "done"  # (over, not up yet)
    run_brain(brain, clock, 2.5, frames=rolling)
    assert _last(brain)["state"] == "done" and _last(brain)["motion_ms"] is not None
    nobody = _asked(brain, clock, 1.0)  # (it has just rolled: it rests before the next one)
    assert nobody.answer["reason"] == "rest_first"


def test_a_flip_the_body_never_takes_up_is_given_up_as_not_steady() -> None:
    brain, clock = _roller()
    _asked(brain, clock, CB.FLIP_START_S + 2.0)  # (the frames never tip: the guard said wait)
    assert _last(brain)["state"] == "gave_up" and _last(brain)["reason"] == "not_steady"


def _wobbly(t: float) -> C.SensorFrame:
    """Standing, but the trunk is not quiet (its gyro over the flip's still limit): not steady by its own senses."""
    f = _frame(t, posture="stand")
    return replace(f, body=replace(f.body, gyro=(0.6, 0.0, 0.0)))


def test_a_duck_unsettled_after_its_yes_rolls_only_once_it_has_stood_calm_again() -> None:
    brain, clock = _roller()
    ticket = _asked(brain, clock, 0.1)
    assert ticket.answer["state"] == "heard", ticket.answer
    shaken = run_brain(brain, clock, 1.5, frames=_wobbly)
    assert not any(c.behaviour == "flip" for c in shaken)  # (it waits: the roll is asked only from a steady stand)
    calm = run_brain(brain, clock, 3.0, frames=lambda t: _frame(t, posture="stand"))
    first = next(i for i, c in enumerate(calm) if c.behaviour == "flip")
    assert (calm[first].t_ms - calm[0].t_ms) / 1000.0 >= CB.FLIP_CALM_S - 0.1  # (calm this long again first)
    assert _last(brain)["state"] not in ("gave_up", "interrupted"), _last(brain)


def test_while_the_flip_waits_for_a_steady_stand_the_duck_makes_no_step_or_turn() -> None:
    brain, clock = _roller()
    ticket = brain.commands.hear(Command("flip"))
    heard_at = clock()
    # (steady when it hears it, so it says yes; not steady after: the roll waits)
    run_brain(brain, clock, 0.3, frames=lambda t: _frame(t, posture="stand") if t < heard_at + 0.06 else _wobbly(t))
    assert ticket.answer["state"] == "heard", ticket.answer
    brain.sel.commit_next("wander")  # (a walk of its own comes up while it waits)
    cmds = run_brain(brain, clock, 3.2, frames=_wobbly)
    assert all(c.vx == 0.0 and c.vyaw == 0.0 for c in cmds), [(c.behaviour, c.vx, c.vyaw) for c in cmds if c.vx or c.vyaw]


def test_a_duck_kept_still_makes_no_step_of_its_own_and_walks_again_after() -> None:
    free, clock = _roller()
    free.sel.commit_next("wander")
    assert any(c.vx != 0.0 for c in run_brain(free, clock, 2.5, frames=lambda t: _frame(t, posture="stand")))
    kept, clock = _roller()  # (the same duck, the same walk)
    kept.sel.commit_next("wander")
    kept.keep_still(int(round((clock() + 2.6) * 1000)))  # (held a little past the 2.5 s it is watched)
    held = run_brain(kept, clock, 2.5, frames=lambda t: _frame(t, posture="stand"))
    assert all(c.vx == 0.0 and c.vyaw == 0.0 for c in held)
    assert any(c.vx != 0.0 for c in run_brain(kept, clock, 4.0, frames=lambda t: _frame(t, posture="stand")))


@pytest.mark.parametrize("seed", [4107, 42])
def test_when_the_flip_is_over_the_ducks_own_wants_choose_what_comes_next_at_once(seed: int) -> None:
    # (the real simulator: the roll's bout stayed on with no want of its own, and only a fear act could take it — one of
    # noise size did, and robotd ran a 1 s panic run 2-3 s after the roll)
    for ends in ("done", "gave_up"):
        brain, clock = make_brain(seed=seed, gd=80)
        brain.commands.clock = clock
        brain.can_roll = True
        _teach(brain, "flip", 1.0)
        run_brain(brain, clock, 4.0, frames=lambda t: _frame(t, posture="stand"))
        assert _asked(brain, clock, 0.5).answer["state"] == "heard"
        t0 = clock()

        def rolling(t: float) -> C.SensorFrame:
            k = t - t0
            tilt = 0.0 if ends == "gave_up" or k < 0.3 else (2.8 if k < 1.0 else 0.05)  # over, then up (or never)
            f = _frame(t, posture="stand")
            return replace(f, body=replace(f.body, tilt=tilt))

        ticks: list[tuple[str, Any]] = []
        run_brain(brain, clock, 9.0, frames=rolling, on_tick=lambda t, c: ticks.append((_last(brain)["state"], c)))
        over = [i for i, (state, _c) in enumerate(ticks) if state == ends]
        assert over, _last(brain)
        after = [c for _state, c in ticks[over[0] :]]
        nxt = next(c for c in after if c.behaviour != "flip")
        assert nxt.t_ms - after[0].t_ms <= 300, (ends, nxt.t_ms - after[0].t_ms)
        assert nxt.behaviour not in ("flee", "retreat") and not any(c.panic for c in after), ends


def test_praise_after_a_told_trick_teaches_it_and_after_its_own_act_teaches_half_as_much() -> None:
    told, clock = _duck(posture="stand", sightings=_person())
    _teach(told, "bow", 0.8)
    _tell(told, clock, "bow", 2.5, posture="stand", sightings=_person())
    full = told.trick_feedback("praise")
    assert full["trick"] == "bow" and full["counted"] and not full["shaped"] and full["step"] > 0
    assert told.trick_feedback("praise")["reason"] == "already_taken"  # one word per performance
    assert told.learn.tricks.proficiency("bow") == pytest.approx(0.8 + full["step"])

    nothing, _ = _duck(posture="stand", sightings=_person())
    assert nothing.trick_feedback("praise")["reason"] == "no_trick"  # it did no trick lately

    own, clock = _duck(posture="stand", sightings=_person())
    own.commands._acts["shake"] = CB._Act(t=clock(), taught=False)  # (it shook itself off just now)
    half = own.trick_feedback("treat")
    assert half["trick"] == "shake" and half["shaped"] and half["step"] == pytest.approx(0.03)
    late, clock = _duck(posture="stand", sightings=_person())
    late.commands._acts["shake"] = CB._Act(t=clock() - 4.0, taught=False)
    assert late.trick_feedback("praise")["reason"] == "no_trick"  # more than 3 s later it is too late


def test_a_told_trick_waiting_for_its_word_is_credited_before_a_shake_it_did_by_itself() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    _teach(brain, "bow", 0.95)
    ticket, _ = _tell(brain, clock, "bow", 2.5, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "heard"
    brain.commands._acts["shake"] = CB._Act(t=clock(), taught=False)  # (a shake-off right after the bow)
    first = brain.trick_feedback("praise")
    assert first["trick"] == "bow" and first["counted"] and not first["shaped"]
    second = brain.trick_feedback("praise")  # the bow has had its word: now the shake is what is left
    assert second["trick"] == "shake" and second["shaped"] and second["counted"]
    scolded, clock = _duck(posture="stand", sightings=_person())
    _teach(scolded, "bow", 0.95)
    ticket, _ = _tell(scolded, clock, "bow", 2.5, posture="stand", sightings=_person())
    assert ticket.answer["state"] == "heard"
    scolded.commands._acts["shake"] = CB._Act(t=clock(), taught=False)
    assert scolded.trick_feedback("scold")["trick"] == "bow"  # a scold finds the told one first too


def test_a_scold_lowers_the_trick_and_does_not_count_as_a_try() -> None:
    brain, clock = _duck(posture="stand", sightings=_person())
    _teach(brain, "bow", 0.8)
    _tell(brain, clock, "bow", 2.5, posture="stand", sightings=_person())
    before = brain.learn.tricks.skills["bow"].total
    answer = brain.trick_feedback("scold")
    assert answer["trick"] == "bow" and answer["step"] < 0 and not answer["counted"]
    assert brain.learn.tricks.skills["bow"].total == before


def test_a_sleeping_duck_does_not_react_to_a_trick_either() -> None:
    brain, clock = _asleep()
    run_brain(brain, clock, 0.2, frames=_night)
    ticket = brain.commands.hear(Command("spin"))
    cmds = run_brain(brain, clock, 0.5, frames=_night)
    assert ticket.answer["reason"] == "asleep"
    assert not any(c.vocal == "inquire" for c in cmds)


# -- the claim ------------------------------------------------------------------------------------


def test_the_live_bridge_cannot_erase_a_command() -> None:
    """The bridge calls ``set_extra`` on every tick; the command's claim is on a channel of its own."""
    brain, clock = _duck(sightings=_person())
    sel = brain.sel
    index = sel.names.index("watch")
    brain.commands.hear(Command("look_at_me"))
    run_brain(brain, clock, 0.6, frames=lambda t: _frame(t, sightings=_person()))
    assert brain.commands._plan is not None and sel._cmd_claim[index] > 0.0
    sel.set_extra({})  # (what the bridge does when nobody speaks)
    assert sel._cmd_claim[index] > 0.0 and sel._cmd_extra[index] > 0.0


def test_a_command_adds_to_what_the_bridge_asks_for() -> None:
    brain, clock = make_brain()
    sel = brain.sel
    sel.set_extra({"watch": 0.5}, claims={"watch": 0.5})
    sel.set_command({"watch": 1.2}, claims={"watch": 1.2})
    index = sel.names.index("watch")
    assert sel._extra[index] == 0.5 and sel._cmd_extra[index] == 1.2
    assert sel._claims()[index] == pytest.approx(1.7)
    sel.set_extra({})
    assert sel._cmd_extra[index] == 1.2 and sel._claims()[index] == pytest.approx(1.2)
    sel.set_command({})
    assert sel._claims()[index] == 0.0 and sel._cmd_extra[index] == 0.0


def test_a_name_that_is_not_a_channel_is_not_applied() -> None:
    brain, _ = make_brain()
    assert brain.sel.set_command({"no_such_act": 1.0}) == {"no_such_act": 1.0}


def test_a_new_command_ends_the_old_one() -> None:
    brain, clock = _duck(sightings=_person())
    first = brain.commands.hear(Command("look_at_me"))
    run_brain(brain, clock, 0.2, frames=lambda t: _frame(t, sightings=_person()))
    second = brain.commands.hear(Command("stand"))
    run_brain(brain, clock, 0.2, frames=lambda t: _frame(t, sightings=_person()))
    assert first.answer["state"] == "heard" and second.answer["state"] == "heard"
    assert brain.commands._plan.ticket is second
    assert brain.commands.counts["interrupted"] == 1


def test_the_claim_is_given_back_when_the_command_ends() -> None:
    brain, clock = _duck(sightings=_person())
    brain.commands.hear(Command("look_at_me"))
    run_brain(brain, clock, 12.0, frames=lambda t: _frame(t, sightings=_person()))
    assert brain.commands._plan is None
    assert not any(brain.sel._cmd_extra) and not any(brain.sel._cmd_claim)


def test_a_command_that_never_finishes_is_given_up_at_its_deadline() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(2.0))
    brain.commands.hear(Command("come"))
    run_brain(
        brain, clock, CB.DEADLINE_S["come"] + 3.0,
        frames=lambda t: _frame(t, posture="stand", sightings=_person(2.0)),
    )
    last = _last(brain)
    assert last["state"] in ("done", "gave_up")
    if last["state"] == "gave_up":
        assert last["reason"] in ("timeout", "busy")
    assert brain.commands._plan is None


# -- telemetry ------------------------------------------------------------------------------------


def test_the_last_command_and_its_latency_are_in_the_brains_telemetry() -> None:
    brain, clock = _duck(sightings=_person())
    brain.commands.hear(Command("look_at_me", who="主人", lang="en"))
    run_brain(brain, clock, 6.0, frames=lambda t: _frame(t, sightings=_person()))
    _, tel = brain.step(int(round((clock.t + 0.05) * 1000)), _frame(clock.t + 0.05, sightings=_person()))
    view = tel.learned["command"]
    last = view["last"]
    assert last["verb"] == "look_at_me" and last["who"] == "主人" and last["lang"] == "en"
    assert last["state"] in ("doing", "done") and last["decided_ms"] is not None
    assert last["motion_ms"] is not None and last["motion_ms"] >= last["decided_ms"] >= 0
    assert view["counts"]["total"] == 1


def test_a_refusal_is_in_the_telemetry_with_its_words() -> None:
    brain, clock = _asleep()
    run_brain(brain, clock, 0.2, frames=_night)
    brain.commands.hear(Command("sit"))
    run_brain(brain, clock, 0.3, frames=_night)
    last = brain.commands.view()["last"]
    assert last["state"] == "refused" and last["reason"] == "asleep"
    assert last["reason_zh"].key == "neuro.command.reason.asleep"
    assert last["state_zh"].key == "neuro.command.state.refused"
    assert brain.commands.view()["counts"]["refused"] == 1


def test_a_fault_in_the_intake_never_stops_the_brain() -> None:
    brain, clock = _duck(sightings=_person())

    def broken(*_: Any) -> None:
        raise RuntimeError("boom")

    brain.commands.pre = broken  # type: ignore[method-assign]
    cmds = run_brain(brain, clock, 0.5, frames=lambda t: _frame(t, sightings=_person()))
    assert len(cmds) == 10 and brain.commands.faults == 10
