"""What the neural duck lived, as bridge-v1 events (duckmind.life.experience): each detector reads
only what the duck's own senses and brain gave, and says it once."""

from __future__ import annotations

import dataclasses
from typing import Any

from duckmind.duckcore import bridge
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C

T0 = 1_773_000_000


def frame(
    t_ms: int,
    *,
    touch: tuple[str | None, float, float] = (None, 0.0, 0.0),
    sound: tuple[str | None, float] = (None, 30.0),
    events: tuple[str, ...] = (),
    sightings: tuple[C.Sighting, ...] = (),
    battery: float = 0.8,
    on_dock: bool = False,
    charging: bool = False,
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    posture: str = "sit",
) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture=posture, odom=odom),
        events=events,
        camera=C.CameraSense(usable=True, sightings=sightings),
        sound=C.SoundSense(level_db=sound[1], kind=sound[0]),
        touch=C.TouchSense(site=touch[0], pressure=touch[1], stroke=touch[2]),
        power=C.PowerSense(battery=battery, on_dock=on_dock, charging=charging),
    )


def command(
    t_ms: int,
    behaviour: str = "look_around",
    *,
    asleep: bool = False,
    vocal: str | None = None,
    seq: int = 0,
) -> C.MotorCommand:
    return C.MotorCommand(
        t_ms=t_ms, behaviour=behaviour, bout=1, variant=1, asleep=asleep, vocal=vocal, vocal_seq=seq
    )


def run(
    exp: Experiences, frames: list[tuple[C.SensorFrame, C.MotorCommand | None]], start_s: int = T0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f, c in frames:
        out += exp.observe(start_s + f.t_ms // 1000, f, c, None)
    return out


def ticks(a_ms: int, b_ms: int, step: int = 100, **kw: Any) -> list[tuple[C.SensorFrame, None]]:
    return [(frame(t, **kw), None) for t in range(a_ms, b_ms, step)]


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [e["k"] for e in events]


def test_every_event_it_makes_is_a_valid_bridge_event() -> None:
    exp = Experiences()
    out = run(
        exp,
        ticks(0, 1000, touch=("head", 0.3, 0.6)) + ticks(1000, 2000),
    )
    assert kinds(out) == ["touch"]
    assert bridge.check(out[0])["where"] == "head"
    assert out[0]["ms"] >= 800 and out[0]["asleep"] == 0


def test_a_caress_and_a_poke_are_told_apart_by_the_touch_sense() -> None:
    caress = run(Experiences(), ticks(0, 800, touch=("back", 0.3, 0.6)) + ticks(800, 1800))
    poke = run(Experiences(), ticks(0, 300, touch=("left", 0.8, 0.0)) + ticks(300, 1300))
    assert [(e["k"], e.get("where")) for e in caress] == [("touch", "back")]
    assert kinds(poke) == ["poke"]


def test_a_hard_short_touch_on_its_side_is_a_poke_even_when_it_slides() -> None:
    """What the live touch model gave a page poke (measured on :8996): 0.22 s, press 0.87-1.0, stroke
    0.67 on a side; and a caress is never on a side."""
    side = run(Experiences(), ticks(0, 300, touch=("right", 1.0, 0.67)) + ticks(300, 1300))
    head = run(Experiences(), ticks(0, 300, touch=("head", 1.0, 0.67)) + ticks(300, 1300))
    assert kinds(side) == ["poke"] and kinds(head) == ["touch"]


def test_a_flicker_is_one_touch_and_a_blip_is_none() -> None:
    flicker = ticks(0, 200, touch=("head", 0.3, 0.6)) + ticks(200, 400)
    flicker += ticks(400, 700, touch=("head", 0.3, 0.6)) + ticks(700, 1500)
    assert kinds(run(Experiences(), flicker)) == ["touch"]
    blip = ticks(0, 100, touch=("head", 0.3, 0.6)) + ticks(100, 1000)
    assert run(Experiences(), blip) == []


def test_a_touch_while_asleep_or_leaning_in_or_after_an_act_says_so() -> None:
    exp = Experiences()
    frames: list[tuple[C.SensorFrame, C.MotorCommand | None]] = [
        (frame(t), command(t, "look_around")) for t in range(0, 2000, 100)
    ]
    frames += [
        (frame(t, touch=("head", 0.3, 0.6)), command(t, "doze", asleep=True))
        for t in range(2000, 3000, 100)
    ]
    frames += [(frame(t), command(t, "doze", asleep=True)) for t in range(3000, 3600, 100)]
    out = run(exp, frames)
    assert len(out) == 1 and out[0]["asleep"] == 1
    # what it was doing in the three seconds before: a verb it chose, or nothing
    exp = Experiences()
    frames = [(frame(t), command(t, "look_around")) for t in range(0, 1000, 100)]
    frames += [
        (frame(t, touch=("head", 0.3, 0.6)), command(t, "look_around"))
        for t in range(1000, 2000, 100)
    ]
    frames += [(frame(t), command(t, "look_around")) for t in range(2000, 2600, 100)]
    assert run(exp, frames)[0]["doing"] == "look_around"
    exp = Experiences()
    frames = [
        (frame(t, touch=("head", 0.3, 0.6)), command(t, "investigate")) for t in range(0, 1000, 100)
    ]
    frames += [(frame(t), command(t, "investigate")) for t in range(1000, 1600, 100)]
    assert run(exp, frames)[0]["doing"] == ""  # (investigating is no verb of Duck Core's to praise)


def test_leaning_into_the_hand_is_settling_for_it() -> None:
    exp = Experiences()
    lean = C.ReflexEvent(t_ms=300, reflex="lean_in", stimulus="touch:head", strength=0.5, gain=1.0)
    frames = ticks(0, 1000, touch=("head", 0.3, 0.6)) + ticks(1000, 1600)
    out: list[dict[str, Any]] = []
    tel = C.BrainTelemetry(
        t_ms=0, behaviour="x", behaviour_zh="", reason_zh="", bout=1, bout_ms=0, satiation=0.0,
        drives={}, arousal=0.0, valence=0.0, reflex=None, reflexes=(lean,), competition=(),
        habituation={}, sensitisation={}, learned={}, locomotion={}, stage="duckling", counts={},
    )  # fmt: skip
    for f, _ in frames:
        out += exp.observe(T0 + f.t_ms // 1000, f, None, tel)
    assert out[0]["settled"] == 1


def test_the_hand_that_lifts_it_is_no_caress() -> None:
    held = ticks(0, 400, touch=("head", 1.0, 0.1)) + [(frame(400, events=("lifted",)), None)]
    held += [(frame(t, touch=("head", 1.0, 0.1)), None) for t in range(500, 4500, 100)]
    held += ticks(4500, 6000) + [(frame(6000, events=("set_down",)), None)] + ticks(6100, 7000)
    assert kinds(run(Experiences(), held)) == ["lift"]  # (one lift, and not a touch of the hold)
    later = held + ticks(20_000, 20_500, touch=("head", 0.3, 0.6)) + ticks(20_500, 21_500)
    assert kinds(run(Experiences(), later)) == ["lift", "touch"]


def test_a_lift_and_the_ways_a_fall_ends() -> None:
    lift = run(
        Experiences(),
        [
            (frame(0), None),
            (frame(100, events=("lifted",)), None),
            (frame(300, events=("lifted",)), None),
        ],
    )
    assert kinds(lift) == ["lift"]  # one lift, however the body says it twice

    def fall(*after: tuple[int, str]) -> list[dict[str, Any]]:
        frames = [
            (frame(0, odom=(1.2, 0.34, 0.0)), None),
            (frame(100, events=("fell",), odom=(1.2, 0.34, 0.0)), None),
        ]
        frames += [(frame(t, events=(name,)), None) for t, name in after]
        frames += [(frame(t), None) for t in range(30_000, 31_000, 1000)]
        return run(Experiences(), frames)

    easy = fall((3000, "righted"))
    hard = fall((9000, "righted"))
    stuck = fall((3000, "righting_failed"), (14000, "righted_help"))
    helped = fall((2000, "righted_help"))
    assert [(e["k"], e["outcome"]) for e in easy + hard + stuck + helped] == [
        ("fall", "easy"),
        ("fall", "hard"),
        ("fall", "stuck"),
        ("fall", "helped"),
    ]
    assert (easy[0]["x_mm"], easy[0]["y_mm"]) == (1200, 340)
    for event in easy + hard + stuck + helped:
        bridge.check(event)


def test_a_fall_nobody_ends_is_stuck_after_two_minutes() -> None:
    frames = [(frame(0, events=("fell",)), None)] + [(frame(t * 1000), None) for t in range(1, 200)]
    out = run(Experiences(), frames)
    assert [(e["k"], e["outcome"]) for e in out] == [("fall", "stuck")]


def hand(look: float, track: int = 7) -> C.Sighting:
    return C.Sighting("hand", 0.0, 0.1, 0.4, 0.2, 0.8, look, track, "model")


def test_a_hand_is_slow_or_fast_by_its_looming_once_per_hand() -> None:
    slow = run(
        Experiences(),
        [(frame(t, sightings=(hand(0.3),)), None) for t in range(0, 1500, 100)] + ticks(1500, 2500),
    )
    fast = run(
        Experiences(),
        [(frame(t, sightings=(hand(2.5),)), None) for t in range(0, 1000, 100)] + ticks(1000, 2000),
    )
    assert [(e["k"], e["speed"]) for e in slow + fast] == [("hand", "slow"), ("hand", "fast")]


def test_a_fast_looming_thing_is_a_fast_hand_only_with_a_person_there() -> None:
    loom = C.Sighting("looming", 0.0, 0.1, None, 0.2, 0.8, 2.5, 0, "model")
    person = C.Sighting("person", 0.1, 0.3, 1.2, 0.25, 0.0, 0.0, 1, "model")
    alone = [(frame(t, sightings=(loom,)), None) for t in range(0, 1000, 100)] + ticks(1000, 2000)
    assert run(Experiences(), alone) == []
    with_person = [(frame(t, sightings=(person, loom)), None) for t in range(0, 1000, 100)]
    out = run(Experiences(), with_person + ticks(1000, 2000))
    assert [(e["k"], e.get("speed")) for e in out if e["k"] != "company"] == [("hand", "fast")]
    # …or when something is at its face for the ToF
    near = C.TofSense(present=True, ahead_mm=120)
    face = [
        (dataclasses.replace(frame(t, sightings=(loom,)), tof=near), None)
        for t in range(0, 1000, 100)
    ]
    assert [e["speed"] for e in run(Experiences(), face + ticks(1000, 2000))] == ["fast"]


def test_the_sounds_it_heard_once_each() -> None:
    def sound(kind: str, ms: int = 700) -> list[tuple[C.SensorFrame, None]]:
        return [(frame(t, sound=(kind, 66.0)), None) for t in range(0, ms, 100)] + ticks(
            ms, ms + 3000
        )

    assert kinds(run(Experiences(), sound("name"))) == ["name"]
    assert kinds(run(Experiences(), sound("voice", 1500))) == ["voice"]
    assert kinds(run(Experiences(), sound("hello"))) == ["hello"]
    assert [e["what"] for e in run(Experiences(), sound("bang", 200))] == ["bang"]
    assert run(Experiences(), sound("footsteps")) == [] and run(Experiences(), sound("music")) == []
    # its name with the TV on in the last five seconds could be the TV
    tv = [(frame(t, sound=("tv", 60.0)), None) for t in range(0, 2000, 100)]
    tv += [(frame(t, sound=("name", 66.0)), None) for t in range(2000, 2700, 100)]
    assert run(Experiences(), tv)[0]["tv"] == 1


def test_company_by_the_minute_and_a_visit_after_a_long_time_alone() -> None:
    person = C.Sighting("person", 0.1, 0.3, 1.2, 0.25, 0.0, 0.0, 1, "model")
    exp = Experiences()
    out: list[dict[str, Any]] = []
    # 12 minutes alone, then a person sits 3 minutes, then alone again
    for s in range(0, 12 * 60 + 190 + 60):
        here = 12 * 60 <= s < 12 * 60 + 180
        f = frame(s * 1000, sightings=(person,) if here else ())
        out += exp.observe(T0 + s, f, command(s * 1000, "rest_sit"), None)
    assert kinds(out).count("visit") == 1
    assert [e["moving"] for e in out if e["k"] == "company"] == [0, 0, 0, 0]
    for event in out:
        bridge.check(event)


def test_sleep_play_dock_and_feeding() -> None:
    exp = Experiences()
    out: list[dict[str, Any]] = []
    for s in range(0, 1500):
        asleep = 100 <= s < 400
        playing = 600 <= s < 700
        docked = 800 <= s < 1000
        battery = 0.3 if s < 1200 else 0.9
        f = frame(s * 1000, on_dock=docked, battery=battery)
        c = command(
            s * 1000, "play_ball" if playing else "doze" if asleep else "look_around", asleep=asleep
        )
        out += exp.observe(T0 + 9 * 3600 + s, f, c, None)
    out += exp.flush(T0 + 9 * 3600 + 1500)
    by = {e["k"]: e for e in out}
    assert by["sleep"]["minutes"] == 5 and by["sleep"]["night"] == 0
    assert by["play"]["minutes"] == 1
    assert by["dock"]["minutes"] == 3
    assert "feed" in by
    for event in out:
        bridge.check(event)


def test_the_night_is_the_night_by_the_calendar_hour() -> None:
    exp = Experiences()
    out: list[dict[str, Any]] = []
    from datetime import datetime

    from duckmind.life.experience import TPE

    t0 = int(datetime(2026, 3, 3, 23, 0, tzinfo=TPE).timestamp())
    for s in range(0, 900):
        asleep = s < 600
        out += exp.observe(t0 + s, frame(s * 1000), command(s * 1000, "doze", asleep=asleep), None)
    assert [(e["k"], e["night"]) for e in out] == [("sleep", 1)]


def test_what_it_said_is_grouped_into_one_call_or_song_and_told_only_in_company() -> None:
    person = C.Sighting("person", 0.1, 0.3, 1.2, 0.25, 0.0, 0.0, 1, "model")

    def said(company: bool) -> list[dict[str, Any]]:
        exp = Experiences()
        out: list[dict[str, Any]] = []
        seq = 0
        for t in range(0, 6000, 100):
            if t in (500, 1000, 1500, 4000):
                seq += 1
            tag = {500: "chirp", 1000: "coo", 1500: "inquire", 4000: "greet"}.get(t)
            c = command(t, "call_out", vocal=tag if tag else "chirp", seq=seq)
            f = frame(t, sightings=(person,) if company else ())
            out += exp.observe(T0 + t // 1000, f, c, None)
        return out + exp.flush(T0 + 7)

    assert [e for e in said(False) if e["k"] == "voiced"] == []  # (nobody there to answer)
    out = said(True)
    voiced = [e["tags"] for e in out if e["k"] == "voiced"]
    assert voiced == [["chirp", "coo", "inquire"], ["greet"]] or voiced == [
        ["coo", "inquire"],
        ["greet"],
    ]
    for event in out:
        bridge.check(event)


def test_it_is_deterministic_and_the_same_frames_give_the_same_events() -> None:
    frames = (
        ticks(0, 800, touch=("head", 0.3, 0.6))
        + ticks(800, 2000, sound=("hello", 66.0))
        + ticks(2000, 6000)
    )
    assert run(Experiences(), frames) == run(Experiences(), frames)
