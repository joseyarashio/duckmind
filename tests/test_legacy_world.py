"""LegacyWorld (phase 4 WP5, design §6, test T4; contract docs/design/phase4-boundary.md): v4's
dice and timed outcomes moved out of Duck Core, drawn with Duck Core's own key and schedule.

Two kinds of test:

* the world alone — every purpose draws ``rng.draw(seed, table_hash, purpose, n)`` with its own
  counter; a fall at a walk's last second comes before its end; righting tries at +2 s, then
  every 4 s, at most 3, one percept each; a stop ends the walk at once, a stop naming no act
  also drops the righting to come; answers that take no time come back from ``command``; every
  percept is valid percept-v1 (phase 4's three kinds included, WP9); header, hash, tape replay;
* the world *through the frozen v4 engine* — :class:`ShimCore` is the frozen v4 Duck Core with
  every relocated draw taken from a real :class:`LegacyWorld` (the world draws first, through
  its commands, polls and detectors, and the core may only re-read the value the world drew)
  and every walk end, fall and righting try delivered by the world's schedule instead of v4's
  own. Each outcome the core computes is checked against the world's percept, and the chain
  must end on exactly the head stock v4 produces: the vendored golden sessions (page hands, TV,
  gamepad, falls, righting, docking, failed stand-ups) and household/legacy ducks at default
  and stressed rates. This proves the world against v4 independently of the v4_1 engine.
"""

from __future__ import annotations

import collections
import gzip
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from duckmind import percept as P
from duckmind import worldport
from duckmind.duckcore import rng
from duckmind.engines.v4_0.duckcore import api as v4api
from duckmind.engines.v4_0.duckcore import content as v4content
from duckmind.engines.v4_0.duckcore import engine as v4engine
from duckmind.world import legacy
from duckmind.world.legacy import (
    PURPOSES,
    LegacyWorld,
    LegacyWorldError,
    header_from_v4_config,
    make_header,
    normalize_command,
    replay,
    world_replay_check,
)

GOLDEN = Path(__file__).parent / "golden"
T0 = 1_790_000_000_000  # a whole second, in ms
TABLE = v4content.table_hash()


def world(**settings: Any) -> LegacyWorld:
    return LegacyWorld(make_header(seed=11, table_hash="t" * 16, **settings), validate_first=None)


def walk(act: str = "W1", seconds: int = 5, **extra: Any) -> dict[str, Any]:
    """A walk command as Duck Core v4_1 sends it (the contract's wire dict)."""
    return {
        "cmd": "duck.walk",
        "act_id": act,
        "deadline_ms": T0 + seconds * 1000 + 5000,
        "seconds": seconds,
        "max_ms": seconds * 1000,
        "vx": 0.2,
        "vyaw": 0.0,
        "vx_mm": 200,
        "vyaw_mrad": 0,
        "cm": seconds * 15,
        "purpose": "wander",
        "bel_from": {"x_mm": 0, "y_mm": 0, "yaw_mrad": 0},
        "expect": {"x_mm": seconds * 150, "y_mm": 0, "yaw_mrad": 0},
        **extra,
    }


def act(cmd: str, act_id: str, **fields: Any) -> dict[str, Any]:
    return {"cmd": cmd, "act_id": act_id, "deadline_ms": T0 + 60_000, **fields}


def drain(w: LegacyWorld, until_ms: int) -> list[dict[str, Any]]:
    """What Duck Core's pull loop takes in by ``until_ms``."""
    out: list[dict[str, Any]] = []
    while (t := w.next_at_ms()) is not None and t <= until_ms:
        out += w.due(t)
    return out


def draws(w: LegacyWorld, purpose: str) -> list[int]:
    return [rng.draw(w.seed, w.table, purpose, n) for n in range(1, w.counters[purpose] + 1)]


# ── the world alone ───────────────────────────────────────────────────────────


def test_header_hash_and_stamps() -> None:
    header = make_header(seed=5, table_hash="abc", fall_ppm_per_s=90)
    assert header["rates"]["fall_ppm_per_s"] == 90 and header["marker"] is True
    w = LegacyWorld(header)
    assert isinstance(w, worldport.WorldPort)
    assert w.header() == header and w.src == "legacy"
    assert w.pv == "l:" + worldport.calib_hash(header) and P.PV_PATTERN.match(w.pv)
    other = make_header(seed=5, table_hash="abc", fall_ppm_per_s=91)
    assert worldport.calib_hash(other) != w.calib_hash
    assert w.command([walk(seconds=3)], T0) == []
    (done,) = drain(w, T0 + 3000)
    assert (done["src"], done["pv"], done["t_ms"], done["reason"]) == (
        "legacy",
        w.pv,
        T0 + 3000,
        "time",
    )
    for bad in (
        {**header, "world": "home"},
        {**header, "rates": {"fall_ppm_per_s": 1}},
        {**header, "sensors": {"conservative": True}},
        {**header, "note": "x"},
    ):
        with pytest.raises(LegacyWorldError):
            LegacyWorld(bad)
    # the contract's shorter header (sensors: ear only) starts conservative
    short = {**header, "sensors": {"ear": True}}
    assert LegacyWorld(short).conservative is True


def test_header_from_a_v4_config() -> None:
    config = v4api.Config(seed=42, fall_ppm_per_s=300, camera=False, ear=False, dock_ppm=700_000)
    header = header_from_v4_config(config, table_hash=TABLE)
    assert header == header_from_v4_config(
        {"seed": 42, "fall_ppm_per_s": 300, "camera": False, "ear": False, "dock_ppm": 700_000},
        table_hash=TABLE,
    )
    assert header["rates"] == {
        "fall_ppm_per_s": 300,
        "fall_ppm_per_start": 0,
        "righting_ppm": 600_000,
        "stand_ppm": 1_000_000,
        "dock_ppm": 700_000,
    }
    assert (header["marker"], header["sensors"]) == (False, {"ear": False, "conservative": True})
    assert header_from_v4_config({}, table_hash=TABLE)["seed"] == 7  # an old save's defaults
    assert worldport.make_world(header).header() == header  # registered on import


def test_walk_ends_on_time_and_draws_its_fall_with_the_v4_key() -> None:
    w = world()
    assert w.command([walk("W1", 4)], T0) == []
    (replaced,) = w.command([walk("W2", 6)], T0)  # (a walk over a walk ends the first)
    assert (replaced["act_id"], replaced["reason"], replaced["t_ms"]) == ("W1", "preempt", T0)
    assert w.counters == {"fall": 2}
    (done,) = drain(w, T0 + 60_000)
    assert (done["act_id"], done["reason"], done["t_ms"]) == ("W2", "time", T0 + 6000)
    assert done["cmd"] == {"vx": 200, "vyaw": 0, "max_ms": 6000}
    assert done["odom"] == {"dx_mm": 900, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 900}
    assert done["steps"] == 27 and done["purpose"] == "wander"  # 4.6 footfalls a second
    assert w.next_at_ms() is None


def test_odometry_is_in_the_frame_the_walk_started_in() -> None:
    w = world()
    turn = walk(
        "W1",
        4,
        bel_from={"x_mm": 1000, "y_mm": 500, "yaw_mrad": 1571},  # facing +y
        expect={"x_mm": 1000, "y_mm": 1100, "yaw_mrad": 2000},
    )
    w.command([turn], T0)
    (done,) = drain(w, T0 + 4000)
    assert done["odom"] == {"dx_mm": 600, "dy_mm": 0, "dyaw_mrad": 429, "path_mm": 600}


@pytest.mark.parametrize("seconds", [1, 3, 8])
def test_a_fall_comes_first_even_at_the_last_second(seconds: int) -> None:
    w = world(fall_ppm_per_s=1_000_000)  # every walk falls
    w.command([walk("W1", seconds)], T0)
    fall_at = 1 + rng.pick_index(rng.draw(w.seed, w.table, "fall.at", 1), seconds)
    assert w.next_at_ms() == T0 + fall_at * 1000
    out = w.due(T0 + fall_at * 1000)
    assert [p["kind"] for p in out] == ["body.fell", "walk.done"]
    assert out[0]["ctx"] == {"act_id": "W1"} and out[0]["during"] == "walk"
    assert out[1]["reason"] == "fell" and out[1]["t_ms"] == T0 + fall_at * 1000
    assert w.counters == {"fall": 1, "fall.at": 1}


def test_the_fall_chance_is_v4s() -> None:
    for rate, start in ((0, 0), (40_000, 0), (40_000, 20_000), (300_000, 0)):
        w = world(fall_ppm_per_s=rate, fall_ppm_per_start=start)
        falls = 0
        for n in range(1, 41):
            now = T0 + n * 100_000
            w.command([walk(f"W{n}", 5)], now)
            chance = min(P.PPM, rate * 5 + start * 2)
            falls += (rng.draw(w.seed, w.table, "fall", n) * P.PPM) >> 64 < chance
            w.command([{"cmd": "duck.stop", "why": "held"}], now + 99_000)  # (any righting)
            drain(w, now + 99_000)
            assert w.counters["fall"] == n
        assert w.counters.get("fall.at", 0) == falls


def test_righting_tries_2s_then_every_4s_at_most_3() -> None:
    w = world(fall_ppm_per_s=1_000_000, righting_ppm=0)  # it never gets up by itself
    w.command([walk("W1", 1)], T0)
    fell_ms = T0 + 1000
    out = drain(w, T0 + 3_600_000)
    tries = [p for p in out if p["kind"] == "body.righting"]
    assert [(p["t_ms"] - fell_ms, p["tries"], p["ok"]) for p in tries] == [
        (2000, 1, False),
        (6000, 2, False),
        (10000, 3, False),
    ]
    assert all(p["ms"] == p["t_ms"] - fell_ms and "via" not in p for p in tries)
    assert w.counters["righting"] == 3 and w.next_at_ms() is None
    up = world(fall_ppm_per_s=1_000_000, righting_ppm=1_000_000)
    up.command([walk("W1", 1)], T0)
    (first,) = [p for p in drain(up, T0 + 60_000) if p["kind"] == "body.righting"]
    assert (first["ok"], first["tries"], first["via"], first["hard"]) == (True, 1, "self", False)


def test_the_righting_outcome_follows_the_draw() -> None:
    w = world(fall_ppm_per_s=1_000_000, righting_ppm=500_000)
    got = []
    for n in range(20):
        w.command([walk(f"W{n}", 1)], T0 + n * 60_000)
        out = drain(w, T0 + n * 60_000 + 59_000)
        got += [p["ok"] for p in out if p["kind"] == "body.righting"]
    assert got == [(raw * P.PPM) >> 64 < 500_000 for raw in draws(w, "righting")]


def test_a_stop_ends_the_walk_at_once_and_a_bare_stop_drops_the_righting() -> None:
    w = world()
    w.command([walk("W1", 10)], T0)
    (done,) = w.command([act("duck.stop", "W1", why="picked_up")], T0 + 4000)
    assert (done["reason"], done["t_ms"], done["act_id"]) == ("lifted", T0 + 4000, "W1")
    assert done["odom"]["dx_mm"] == 600 and done["steps"] == 18  # 4 of 10 s
    assert w.next_at_ms() is None
    w.command([walk("W2", 10)], T0 + 5000)
    assert w.command([act("duck.stop", "W1", why="held")], T0 + 6000) == []  # another act
    assert w.command([{"cmd": "duck.curio", "act_id": "C1", "phase": "step"}], T0 + 6000) == []
    (done,) = w.command([{"cmd": "duck.stop", "why": "held"}], T0 + 7000)  # a bare stop
    assert (done["act_id"], done["reason"]) == ("W2", "preempt")
    falls = world(fall_ppm_per_s=1_000_000, righting_ppm=0)
    falls.command([walk("W1", 1)], T0)
    drain(falls, T0 + 3000)  # the fall and the first try
    assert falls.command([act("duck.stop", "W1", why="set_upright")], T0 + 4000) == []
    assert falls.next_at_ms() == T0 + 7000  # a stop naming the fallen walk: tries go on
    assert falls.command([{"cmd": "duck.stop", "why": "set_upright"}], T0 + 4000) == []
    assert falls.next_at_ms() is None and falls.counters["righting"] == 1


def test_a_stand_is_answered_at_once_and_drawn_only_below_certain() -> None:
    sure = world()
    (answer,) = sure.command([act("duck.posture", "P1", want="stand")], T0)
    assert answer == {
        "kind": "body.posture",
        "src": "legacy",
        "t_ms": T0,
        "pv": sure.pv,
        "want": "stand",
        "got": "stand",
        "ms": 0,
        "fell": False,
        "act_id": "P1",
    }
    assert "stand.try" not in sure.counters  # v4 does not draw at 1,000,000 ppm
    shaky = world(stand_ppm=330_000)
    got = [
        shaky.command([act("duck.posture", f"P{n}", want="stand")], T0)[0]["got"] for n in range(30)
    ]
    assert got == [
        "stand" if (raw * P.PPM) >> 64 < 330_000 else "sit" for raw in draws(shaky, "stand.try")
    ]
    (sat,) = shaky.command([act("duck.posture", "P99", want="sit")], T0)
    assert sat["got"] == "sit" and shaky.counters["stand.try"] == 30  # sitting has no dice


@pytest.mark.parametrize("marker", [True, False])
def test_the_dock_chance_is_v4s(marker: bool) -> None:
    w = world(dock_ppm=850_000, marker=marker)
    errs = [0, 100, 400, 799, 800, 5000]
    got = []
    for n, err in enumerate(errs):
        (answer,) = w.command([act("duck.dock", f"D{n}", bel_err_mm=err)], T0)
        assert (answer["tries"], answer["seen_tag"], answer["via"]) == (1, marker, "pg")
        got.append(answer["contact"])
    want = []
    for err, raw in zip(errs, draws(w, "dock.try"), strict=True):
        e = 0 if marker else err
        chance = 850_000 * max(0, min(10_000, 10_000 - e * 10_000 // 800)) // 10_000
        want.append((raw * P.PPM) >> 64 < chance)
    assert got == want


def test_the_nest_look_sees_the_marker_from_the_believed_pose() -> None:
    w = world()
    look = {"kind": "seek_nest", "want": "nest"}
    pose = {"x_mm": 300, "y_mm": 400, "yaw_mrad": 0}
    (seen,) = w.command([act("duck.look", "L1", bel_pose=pose, **look)], T0)
    assert (seen["kind"], seen["what"], seen["r_mm"], seen["r_src"], seen["look_id"]) == (
        "sense.sighting",
        "nest",
        500,
        "tag",
        "L1",
    )
    assert seen["b_mrad"] == -(3141 - 927)  # facing +x, the nest behind on the right
    (again,) = w.command([act("duck.look", "L2", bel_pose=[0.3, 0.4, 0.0], **look)], T0)
    assert again["r_mm"] == 500  # (v4's pose list in metres)
    blind = world(marker=False)
    (scan,) = blind.command([act("duck.look", "L1", bel_pose=pose, **look)], T0)
    assert scan["kind"] == "sense.scan" and scan["sightings"] == [] and scan["act_id"] == "L1"
    assert not w.counters and not blind.counters  # a look draws nothing


def test_tv_minutes_and_gamepad_minutes() -> None:
    w = world()
    heard = sum(len(w.poll("tv_minute", T0 + n * 60_000)) for n in range(3000))
    assert w.counters["false_alarm.tv"] == 3000
    assert heard == sum(rng.below(raw, 33) for raw in draws(w, "false_alarm.tv")) > 0
    touch = next(p for p in w.sent if p["kind"] == "sense.touch")
    assert touch["legacy_phantom"] is True and touch["path"] == "ear"
    deaf = world(ear=False)
    assert deaf.poll("tv_minute", T0) == [] and not deaf.counters
    plays = [w.poll("play_minute", T0, minute=m)[0] for m in range(7)]
    assert [("skill" in p, p["minute"]) for p in plays] == [(m % 3 == 0, m) for m in range(7)]
    meters = [3 + rng.pick_index(raw, 6) for raw in draws(w, "play.meters")]
    assert [p["meters"] for p in plays] == meters
    assert w.counters["play.skill"] == 3
    assert w.poll("rain_minute", T0) == []  # a poll it does not know


def test_the_touch_and_hold_detectors_are_v4s() -> None:
    cases = [
        ("crown", "scratch", True, True),
        ("crown", "light", True, False),
        ("crown", "normal", False, True),
        ("beak", "normal", True, True),
    ]
    for where, firmness, ear, conservative in cases:
        w = world(ear=ear, conservative=conservative)
        got = [
            w.touch("owner", seconds=4, where=where, firmness=firmness, now_ms=T0)
            for _ in range(40)
        ]
        if where == "crown":
            chance = {"scratch": 9000, "normal": 8500, "light": 7000}[firmness] if ear else 5000
        else:
            chance = 9000
        chance -= 500 if conservative else 0
        detected = [rng.below(raw, chance) for raw in draws(w, "detect.touch")]
        assert [g is not None for g in got] == detected
        path = "ear" if where == "crown" and ear else "joint"
        assert {g["path"] for g in got if g} == {path}
    w = world()
    w.set_setting("conservative", False, now_ms=T0)
    w.set_setting("floor", True, now_ms=T0)  # (not the detectors' business)
    assert w.conservative is False
    held = [w.hold_head(persona, moving=False, now_ms=T0) for persona in ("child", "owner") * 20]
    raws = draws(w, "detect.hold")
    assert [h is not None for h in held] == [
        rng.below(raw, 9000 if n % 2 == 0 else 9500) for n, raw in enumerate(raws)
    ]
    assert {h["moving"] for h in held if h} == {False}


def test_every_percept_is_valid_and_the_tape_replays() -> None:
    w = world(fall_ppm_per_s=200_000, righting_ppm=400_000, stand_ppm=500_000)
    r = random.Random(4)
    now = T0
    for n in range(400):
        now += r.randrange(1, 20) * 1000
        drain(w, now)
        roll = r.random()
        if roll < 0.35:
            w.command([walk(f"W{n}", r.randrange(1, 9))], now)
        elif roll < 0.45:
            stop = {"cmd": "duck.stop", "why": r.choice(["held", "picked_up", "power_off"])}
            w.command([stop], now)
        elif roll < 0.55:
            w.command([act("duck.posture", f"P{n}", want="stand")], now)
        elif roll < 0.62:
            w.command([act("duck.dock", f"D{n}", bel_err_mm=r.randrange(900))], now)
        elif roll < 0.68:
            pose = {"x_mm": r.randrange(-900, 900), "y_mm": r.randrange(-900, 900), "yaw_mrad": 7}
            w.command(
                [act("duck.look", f"L{n}", kind="seek_nest", want="nest", bel_pose=pose)], now
            )
        elif roll < 0.78:
            w.poll("tv_minute", now)
        elif roll < 0.83:
            w.poll("play_minute", now, minute=n % 5)
        elif roll < 0.9:
            where = r.choice(["crown", "beak"])
            w.touch("child", seconds=3, where=where, firmness="light", now_ms=now)
        elif roll < 0.95:
            w.hold_head("owner", now_ms=now)
        else:
            w.set_setting("conservative", r.random() < 0.5, now_ms=now)
            w.tv(r.random() < 0.5, now_ms=now)
            w.hand("pickup", persona="owner", now_ms=now)
    drain(w, now + 60_000)
    kinds = collections.Counter(p["kind"] for p in w.sent)
    assert set(kinds) >= {
        "walk.done",
        "body.fell",
        "body.righting",
        "body.posture",
        "body.dock",
        "sense.sighting",
        "sense.touch",
        "sense.hold_head",
        "body.play",
    }
    for p in w.sent:
        worldport.check(p)
        assert P.due_second(p["t_ms"]) * 1000 == p["t_ms"]
    assert [p["t_ms"] for p in w.sent] == sorted(p["t_ms"] for p in w.sent)
    tape = json.loads(json.dumps(w.tape_json()))  # (JSON all the way)
    assert replay(w.header(), tape) == w.sent
    assert world_replay_check(w.header(), tape, w.sent) == {
        "identical": True,
        "percepts": len(w.sent),
        "first_difference": None,
    }
    first_walk = next(i for i, rec in enumerate(tape) if rec["call"] == "command")
    tape[first_walk]["t_ms"] -= 1000
    assert world_replay_check(w.header(), tape, w.sent)["identical"] is False


@pytest.mark.parametrize("seed", range(6))
def test_what_is_due_does_not_depend_on_how_often_it_is_asked(seed: int) -> None:
    """Asked at every scheduled moment, or only just before each command (as Duck Core asks,
    pulling everything due before it acts), the world sends the same percepts."""
    r = random.Random(seed)
    now, script = T0, []
    for n in range(150):
        now += r.randrange(1, 40) * 1000
        roll = r.random()
        if roll < 0.5:
            script.append((now, walk(f"W{n}", r.randrange(1, 12))))
        elif roll < 0.6:
            script.append((now, {"cmd": "duck.stop", "why": "held"}))
        elif roll < 0.7:
            script.append((now, act("duck.posture", f"P{n}", want="stand")))
        else:
            script.append((now, None))  # (a TV minute)
    streams = []
    for fine in (True, False):
        w = world(fall_ppm_per_s=150_000, righting_ppm=300_000, stand_ppm=500_000)
        out: list[dict[str, Any]] = []
        for t_ms, command in script:
            out += drain(w, t_ms) if fine else w.due(t_ms)
            out += w.command([command], t_ms) if command else w.poll("tv_minute", t_ms)
        out += drain(w, now + 3_600_000) if fine else w.due(now + 3_600_000)
        streams.append(out)
    assert streams[0] == streams[1]
    assert {p["kind"] for p in streams[0]} >= {"body.fell", "body.righting", "walk.done"}


@pytest.mark.population_defaults  # (conftest checks every percept in the other tests)
def test_percepts_are_checked_first_1000_by_default_and_the_tape_can_be_off() -> None:
    header = make_header(seed=1, table_hash="t" * 16)
    assert LegacyWorld.VALIDATE_FIRST == 1000 and LegacyWorld(header)._validate_left == 1000
    quiet = LegacyWorld(header, keep_tape=False, validate_first=2)
    for n in range(4):
        quiet.command([act("duck.posture", f"P{n}", want="stand")], T0)
    assert quiet._validate_left == 0 and quiet.tape == [] and quiet.sent == []
    bad = LegacyWorld(header, validate_first=None)
    bad.pv = "l:not-hex"
    with pytest.raises(P.PerceptError):
        bad.command([act("duck.posture", "P1", want="stand")], T0)


def test_commands_are_read_from_the_wire_and_from_events() -> None:
    class Event:
        kind = "duck.walk"
        payload = {
            "act_id": "W7",
            "seconds": 4,
            "vx": 0.2,
            "vyaw": -0.1,
            "from": [0.0, 0.0, 0.0],
            "to": [0.6, 0.0, 0.0],
            "cm": 60,
        }

    cmd = normalize_command(Event())
    assert cmd is not None
    assert (cmd["seconds"], cmd["vx_mm"], cmd["vyaw_mrad"], cmd["expect"], cmd["path_mm"]) == (
        4,
        200,
        -100,
        [600, 0, 0],
        600,
    )
    record = {"kind": "duck.posture", "payload": {"posture": "stand", "from": "sit"}}
    assert normalize_command(record) is None
    assert normalize_command({"kind": "duck.stop", "payload": {"until": "…"}}) is None  # v4's
    assert normalize_command({"kind": "duck.walk", "payload": {"verb": "wander"}}) is None
    wire = {"cmd": "duck.dock", "act_id": "D1", "bel_err_mm": 30, "deadline_ms": 1}
    assert normalize_command(wire) == {"cmd": "duck.dock", "act_id": "D1", "bel_err_mm": 30}
    with pytest.raises(LegacyWorldError):
        normalize_command("duck.walk")


def test_the_world_draws_only_its_own_purposes() -> None:
    assert len(PURPOSES) == 10
    w = world(fall_ppm_per_s=500_000, stand_ppm=100_000)
    w.command([walk("W1", 3), act("duck.posture", "P", want="stand"), act("duck.dock", "D")], T0)
    drain(w, T0 + 60_000)
    w.poll("tv_minute", T0)
    w.poll("play_minute", T0, minute=0)
    w.touch("owner", now_ms=T0)
    w.hold_head("owner", now_ms=T0)
    assert set(w.counters) <= set(PURPOSES) and legacy.SRC == "legacy"


# ── the world through the frozen v4 engine ────────────────────────────────────

WORLD_PURPOSES = frozenset(PURPOSES)


def _ms(moment: datetime) -> int:
    return worldport.epoch_ms(moment)


def _at(t_ms: int) -> datetime:
    return worldport.at_ms(t_ms, v4engine.TPE)


class ShimCore(v4engine.DuckCore):
    """The frozen v4 Duck Core whose relocated dice and timed outcomes come from a real
    :class:`LegacyWorld` (see the module docstring)."""

    def __init__(self, **kwargs: Any) -> None:
        hw = kwargs.get("hardware") or v4engine.HardwareConfig()
        rates = {name: getattr(hw, name) for name in legacy.V4_RATES}
        header = header_from_v4_config(
            {
                "seed": kwargs.get("seed", 7),
                "camera": hw.marker,
                "ear": hw.ear,
                "conservative": kwargs.get("conservative", True),
                **rates,
            },
            table_hash=TABLE,
        )
        self.world = LegacyWorld(header, validate_first=None)  # (every percept checked)
        self.seen: collections.Counter[str] = collections.Counter()
        self._acts = 0
        self._walk_act: str | None = None
        self._play_minute = 0
        self._dock: dict[str, Any] | None = None
        super().__init__(**kwargs)

    def _act_id(self, prefix: str) -> str:
        self._acts += 1
        return f"{prefix}{self._acts}"

    def _command(self, cmd: dict[str, Any]) -> list[dict[str, Any]]:
        return self.world.command([cmd], _ms(self.s.now))

    def _rand(self, purpose: str) -> int:
        if purpose not in WORLD_PURPOSES:
            return super()._rand(purpose)
        n = self.counters.get(purpose, 0) + 1
        if purpose == "dock.try" and self.world.counters.get(purpose, 0) < n:
            # nest.dock_try draws right where v4_1 sends its dock command
            dock = act("duck.dock", self._act_id("D"), bel_err_mm=self.s.pose_err_mm)
            (self._dock,) = self._command(dock)
        if purpose == "play.meters" and self.world.counters.get(purpose, 0) < n:
            self.world.poll("play_minute", _ms(self.s.now), minute=self._play_minute)
            self._play_minute += 1
            self.seen["play"] += 1
        drawn = self.world.counters.get(purpose, 0)
        assert drawn == n, f"{purpose}: the core wants draw #{n}, the world drew {drawn}"
        self.counters[purpose] = n
        return rng.draw(self.seed, self.table, purpose, n)

    # the walk: the world draws its fall and owns its schedule
    def _walk(self, plan: dict[str, Any], expression: Any, **kwargs: Any) -> Any:
        s = self.s
        walk_id = self._act_id("W")
        (fx, fy, fh), (tx, ty, th) = plan["from"], plan["to"]
        cmd = {
            "cmd": "duck.walk",
            "act_id": walk_id,
            "max_ms": int(plan["seconds"]) * 1000,
            "vx_mm": int(plan["vx"]),
            "vyaw_mrad": int(plan["vyaw"]),
            "bel_from": {"x_mm": fx, "y_mm": fy, "yaw_mrad": fh},
            "expect": {"x_mm": tx, "y_mm": ty, "yaw_mrad": th},
        }
        assert self._command(cmd) == []
        event = super()._walk(plan, expression, **kwargs)
        assert s.walk_plan is not None
        self._walk_act = walk_id
        self.seen["walk"] += 1
        return event

    def _set_now(self, target: datetime) -> None:
        s = self.s
        world = self.world
        limit = _ms(target)
        while (t := world.next_at_ms()) is not None and t <= limit:
            s.now = max(s.now, _at(t))
            for p in world.due(t):
                self._apply(p)
        # v4's own clock events, verbatim (engine.DuckCore._set_now after its walk part)
        while True:
            midnight = self._next_midnight()
            if s.stir_at is not None and s.stir_at <= target and s.stir_at < midnight:
                s.now = max(s.now, s.stir_at)
                self._sleep_stir("dream")
            elif target >= midnight:
                self._finalize_day()
            else:
                break
        if self._morning_due() < target:
            latest = datetime.combine(target.date(), v4engine.time(7), v4engine.TPE)
            if latest >= target:
                latest -= timedelta(days=1)
            s.last_morning = latest.date()
        s.now = target

    def _apply(self, p: dict[str, Any]) -> None:
        s = self.s
        kind = p["kind"]
        self.seen[kind] += 1
        if kind == "body.fell":
            plan = s.walk_plan
            assert plan is not None and p["ctx"]["act_id"] == self._walk_act
            assert plan["fall_at"] == s.now, "v4 would fall at another second"
            self._walk_act = None
            self._fall()
        elif kind == "walk.done" and p["reason"] == "fell":
            assert s.walk_plan is None
        elif kind == "walk.done":
            assert p["reason"] == "time" and p["act_id"] == self._walk_act
            assert s.walk_plan is not None and s.walk_plan["fall_at"] is None
            assert s.walk_until == s.now
            self._walk_act = None
            self._finish_walk()
        elif kind == "body.righting":
            state = s.righting
            assert state is not None and state["next_at"] == s.now
            assert p["tries"] == int(state["tries"]) + 1
            before = len(self.log.events)
            self._righting_try()
            (mine,) = [e for e in self.log.events[before:] if e.kind == "body.righting"]
            assert mine.payload["ok"] == p["ok"]
        else:
            raise AssertionError(f"unexpected {kind}")

    def _stand_fails(self, parents: tuple[str, ...]) -> Any:
        (answer,) = self._command(act("duck.posture", self._act_id("P"), want="stand"))
        failed = super()._stand_fails(parents)
        assert (failed is None) == (answer["got"] == "stand")
        self.seen["stand"] += 1
        return failed

    def _emit(self, kind: str, payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if kind == "nest.dock_try":
            assert self._dock is not None and payload["ok"] == self._dock["contact"]
            self._dock = None
            self.seen["dock"] += 1
        elif kind == "sense.nest_seen":  # v4's nest look: the world must see the same
            s = self.s
            pose = {"x_mm": s.pos_x, "y_mm": s.pos_y, "yaw_mrad": s.heading}
            look = act("duck.look", self._act_id("L"), kind="seek_nest", want="nest", bel_pose=pose)
            (seen,) = self._command(look)
            assert seen["kind"] == "sense.sighting"
            assert (seen["b_mrad"], seen["r_mm"]) == (payload["bearing_mrad"], payload["range_mm"])
            self.seen["look"] += 1
        return super()._emit(kind, payload, *args, **kwargs)

    def _set_posture(self, posture: str, **kwargs: Any) -> Any:
        s = self.s
        why = str(kwargs.get("reason"))
        if s.posture != posture and s.posture == "walk":
            (done,) = self._command({"cmd": "duck.stop", "why": why, "act_id": self._walk_act})
            assert done["act_id"] == self._walk_act and done["reason"] in ("lifted", "preempt")
            self._walk_act = None
            self.seen["stop"] += 1
        elif s.posture != posture and s.posture == "fallen":
            assert self._command({"cmd": "duck.stop", "why": why}) == []
            self.seen["stop"] += 1
        return super()._set_posture(posture, **kwargs)

    def _tv_minute(self) -> None:
        heard = self.world.poll("tv_minute", _ms(self.s.now))
        before = len(self.log.events)
        super()._tv_minute()
        phantom = [e for e in self.log.events[before:] if e.kind == "sense.touch"]
        assert len(phantom) == len(heard)
        self.seen["tv"] += 1

    def pad_play(self, persona: str, *, minutes: int = 10) -> dict[str, Any]:
        self._play_minute = 0
        return super().pad_play(persona, minutes=minutes)

    def touch(self, persona: str, **kwargs: Any) -> dict[str, Any]:
        on = self.s.awake
        felt = self.world.touch(persona, now_ms=_ms(self.s.now), **kwargs) if on else None
        out = super().touch(persona, **kwargs)
        if on:
            assert out["perceived"] == (felt is not None)
            self.seen["touch"] += 1
        return out

    def hold_head(self, persona: str) -> dict[str, Any]:
        on = self.s.awake
        felt = None
        if on:
            moving = not self._until(self.s.napping_until)
            felt = self.world.hold_head(persona, moving=moving, now_ms=_ms(self.s.now))
        out = super().hold_head(persona)
        if on:
            assert out["perceived"] == (felt is not None)
            self.seen["hold"] += 1
        return out

    def set_setting(self, key: str, value: bool) -> dict[str, Any]:
        out = super().set_setting(key, value)
        self.world.set_setting(key, value, now_ms=_ms(self.s.now))
        return out


@contextmanager
def shimmed() -> Iterator[None]:
    original = v4api.DuckCore
    v4api.DuckCore = ShimCore  # type: ignore[misc]
    try:
        yield
    finally:
        v4api.DuckCore = original  # type: ignore[misc]


GOLDEN_FILES = sorted(p.name for p in GOLDEN.glob("*.json.gz"))


def test_the_vendored_golden_sessions_are_here() -> None:
    assert len(GOLDEN_FILES) >= 5


@pytest.mark.parametrize("name", GOLDEN_FILES)
def test_golden_sessions_end_on_v4s_head_with_the_world_rolling_the_dice(name: str) -> None:
    with gzip.open(GOLDEN / name, "rb") as f:
        saved = json.loads(f.read())
    config = dict(saved["config"])
    caps = str(config.get("caps_file") or "")
    if caps:  # (relative to the golden folder; read_caps would quietly ignore a missing file)
        path = caps if Path(caps).is_absolute() else str((GOLDEN / caps).resolve())
        assert Path(path).is_file(), f"{name}: its caps file {caps} is not here"
        config["caps_file"] = path
    with shimmed():
        session = v4api.Session(v4api.Config(**config))
        for item in saved["inputs"]:
            v4api.dispatch(session, item["method"], item["params"])
    core = session.core
    assert isinstance(core, ShimCore)
    assert (core.log.head, len(core.log.events)) == (saved["head"], saved["events"])
    w = core.world
    assert w.counters == {p: n for p, n in core.counters.items() if p in WORLD_PURPOSES}
    assert world_replay_check(w.header(), w.tape_json(), w.sent)["identical"]


def _population(
    kind: str, index: int, days: int, rates: dict[str, int]
) -> tuple[v4api.Session, v4api.Session]:
    """(stock v4, v4 through the shim) for one popsim-like duck."""
    out = []
    for shim in (False, True):
        if kind == "v2":
            from duckmind.engines.v4_0.duckcore.routines import FAMILY_KINDS, make_profile

            r = random.Random(700_000 + index)
            kinds = list(FAMILY_KINDS)
            family = r.choices(kinds, weights=[FAMILY_KINDS[k][0] for k in kinds])[0]
            profile = make_profile((10_000 + index) ^ 0xFA11, family)
            config = v4api.Config(
                seed=10_000 + index,
                tof=profile.tof,
                ear=profile.ear,
                name_ear=profile.name_ear,
                battery_help=False,
                nest=profile.nest,
                camera=profile.camera,
                dock_ppm=850_000 if profile.camera else 700_000,
                household=family,
                **rates,
            )
            params: dict[str, Any] = {"family": family, "days": days}
        else:
            r = random.Random(900_000 + index)
            routine = r.choice(["daily", "three_per_week", "weekends", "floor", "docked"])
            config = v4api.Config(
                seed=10_000 + index, ear=r.random() < 0.9, conservative=r.random() < 0.7, **rates
            )
            params = {"routine": routine, "days": days}
        if shim:
            with shimmed():
                session = v4api.Session(config)
                v4api.dispatch(session, "live", params)
        else:
            session = v4api.Session(config)
            v4api.dispatch(session, "live", params)
        out.append(session)
    return out[0], out[1]


STRESS = {
    "fall_ppm_per_s": 40_000,
    "righting_ppm": 150_000,
    "stand_ppm": 600_000,
    "fall_ppm_per_start": 20_000,
}


@pytest.mark.parametrize(
    ("kind", "index", "days", "rates"),
    [
        ("v2", 0, 6, {}),
        ("v2", 3, 6, {}),
        ("legacy", 1, 8, {}),
        ("v2", 20, 12, STRESS),
        ("legacy", 21, 8, STRESS),
    ],
)
def test_population_ducks_are_bit_identical(
    kind: str, index: int, days: int, rates: dict[str, int]
) -> None:
    stock, shim = _population(kind, index, days, rates)
    assert (shim.core.log.head, len(shim.core.log.events)) == (
        stock.core.log.head,
        len(stock.core.log.events),
    )
    seen = shim.core.seen  # type: ignore[attr-defined]
    assert seen["touch"] > 0 and seen["stand"] > 0
    if kind == "v2":  # (a legacy-routine hatchling does not walk yet in a week: g41 walks)
        assert seen["walk"] > 0 and seen["walk.done"] > 0 and seen["dock"] > 0
    if kind == "v2" and rates:
        assert seen["body.fell"] > 0 and seen["body.righting"] > 0 and seen["stop"] > 0


# ── the migration seam (WP13): a fresh world takes over from a v4 duck mid-life ─────────


def _to_seam(config: dict[str, Any], until: str) -> Any:
    """A frozen-v4 duck grown to walking age, then stepped 7 s at a time until it is walking
    (``until="walk"``) or lying after a fall (``until="righting"``)."""
    session = v4api.Session(v4api.Config(**config))
    for method, params in (
        ("live", {"routine": "floor", "days": 6}),
        ("advance", {"until": "07:00"}),
        ("act", {"action": "power_on", "params": {}, "persona": "owner"}),
        ("set_setting", {"key": "floor", "value": True}),
    ):
        v4api.dispatch(session, method, params)
    s = session.core.s
    for _ in range(20_000):
        if (until == "walk" and s.walk_plan is not None) or (until == "righting" and s.righting):
            return session
        v4api.dispatch(session, "advance", {"seconds": 7})
    raise AssertionError(f"no {until} in sight")


def _adopt(core: Any) -> None:
    """Hand a stock v4 core's outcome dice and schedule to a fresh LegacyWorld (in place)."""
    s = core.s
    header = header_from_v4_config(
        {
            "seed": core.seed,
            "camera": core.hw.marker,
            "ear": core.hw.ear,
            **{name: getattr(core.hw, name) for name in legacy.V4_RATES},
        },
        table_hash=TABLE,
    )
    walk = righting = None
    if s.walk_plan is not None:
        plan = s.walk_plan
        (fx, fy, fh), (tx, ty, th) = plan["from"], plan["to"]
        walk = {
            "act_id": "W0",
            "seconds": int(plan["seconds"]),
            "vx_mm": int(plan["vx"]),
            "vyaw_mrad": int(plan["vyaw"]),
            "bel_from": [fx, fy, fh],
            "expect": [tx, ty, th],
            "start_ms": _ms(s.walk_start),
            "end_ms": _ms(s.walk_until),
            "fall_ms": None if plan["fall_at"] is None else _ms(plan["fall_at"]),
        }
    if s.righting is not None:
        fell = next(e for e in reversed(core.log.events) if e.event_id == s.righting["fell"])
        righting = {
            "fell_ms": _ms(fell.occurred_at),
            "tries": int(s.righting["tries"]),
            "next_ms": _ms(s.righting["next_at"]),
        }
    world = LegacyWorld(header, validate_first=None)
    world.adopt(
        now_ms=_ms(s.now),
        counters={p: n for p, n in core.counters.items() if p in WORLD_PURPOSES},
        walk=walk,
        righting=righting,
        conservative=s.conservative,
        tv_on=s.tv_on,
    )
    core.__class__ = ShimCore
    core.world, core.seen, core._acts, core._play_minute, core._dock = (
        world,
        collections.Counter(),
        0,
        0,
        None,
    )
    core._walk_act = "W0" if walk is not None else None


@pytest.mark.parametrize("seam", ["walk", "righting"])
def test_a_fresh_world_takes_over_a_v4_duck_mid_life(seam: str) -> None:
    config = {"seed": 77, "fall_ppm_per_s": 60_000, "righting_ppm": 250_000, "stand_ppm": 700_000}
    stock, moved = _to_seam(config, seam), _to_seam(config, seam)
    assert stock.core.log.head == moved.core.log.head
    _adopt(moved.core)
    later = [("advance", {"seconds": 5})] * 12 + [("advance", {"minutes": 90})]
    later += [("act", {"action": "touch", "params": {"seconds": 4}, "persona": "child"})]
    later += [("advance", {"hours": 5})]
    for method, params in later:
        v4api.dispatch(stock, method, params)
        with shimmed():
            v4api.dispatch(moved, method, params)
    assert moved.core.log.head == stock.core.log.head
    seen = moved.core.seen  # type: ignore[attr-defined]
    assert seen["walk.done"] + seen["body.fell"] > 0 or seen["body.righting"] > 0
    w = moved.core.world  # type: ignore[attr-defined]
    assert world_replay_check(w.header(), w.tape_json(), w.sent)["identical"]
    with pytest.raises(LegacyWorldError):
        w.adopt(now_ms=0, counters={})
