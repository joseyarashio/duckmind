"""``percept-v1`` format tests (physical-design spec §7.2, §7.3, §7.7, appendix A; §17 期3 "percept-v1
格式測試").

* The appendix-A examples are goldens: each validates, round-trips through the canonical line
  byte for byte, and breaks as soon as one integer becomes a float (the spec's "全部整數").
* Every kind in :data:`duckmind.percept.SCHEMAS` has a minimal valid example here; dropping any
  required field, adding an unknown one (closed kinds), or putting a float anywhere is refused.
* The phase-3 owner additions (B1 ``trigger``, B2 ``how``, B3 ``reflex_unloaded``, A2 ``head`` /
  ``veto_head``, C5 ``r_src: cue`` with no range) are accepted, and values outside the enums are
  not.
* A seeded property loop (no hypothesis dependency): random valid events always validate,
  random single mutations of them never do; quantisation never lets a float through.
* §7.3 rule 4: :func:`due_second` is ``ceil(t_ms / 1000)``.
* §7.7: once duck-body's ``aggregate`` exists, a scripted excursion's event count stays inside
  the budget (skipped until then).
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any

import pytest

percept = pytest.importorskip("duckmind.percept")

PV = "p:3f9a0c1d2e4b5a69"
CV = "c:81d0aa11bb22cc33"

# ── appendix A (the spec writes the hash shortened as "p:3f9a…"; a full hex here) ─────────

APPENDIX_A: list[dict[str, Any]] = [
    {
        "kind": "walk.done", "src": "live", "pv": PV, "t_ms": 5231840,
        "act_id": "W1842", "cmd": {"vx": 300, "vyaw": 0, "max_ms": 6000},
        "pre": {"look_id": "L7710", "light": "bright", "free_mm": [620, 1180, 1400, 900, 410],
                "ends": ["thing", "wall", "unknown", "wall", "thing"],
                "trip": {"mm": 520, "mrad": -90}, "drop": 0},
        "end": {"look_id": "L7711", "light": "bright", "free_mm": [0, 380, 760, 1500, 1500],
                "ends": ["unknown", "wall", "wall", "unknown", "unknown"]},
        "odom": {"dx_mm": 842, "dy_mm": -181, "dyaw_mrad": -196, "path_mm": 861}, "steps": 13,
        "reason": "reflex_near", "contact": None, "battery_pct": 71,
    },
    {
        "kind": "sense.sighting", "src": "robot", "pv": PV, "t_ms": 5240210,
        "what": "nest", "id": "tag36h11:17", "b_mrad": -140, "e_mrad": -95, "r_mm": 1210,
        "r_src": "tag", "yaw_mrad": 2890, "conf_ppm": 930000, "light": "dim", "look_id": "L7715",
        "edge": "new",
    },
    {"kind": "sense.light", "src": "surrogate", "pv": CV, "t_ms": 64800000, "level": "dark",
     "luma": 71, "noise_q": 14},
    {"kind": "body.fell", "src": "live", "pv": PV, "t_ms": 5302400, "during": "posture",
     "ctx": {"look_id": "L7730"}},
    {"kind": "body.righting", "src": "live", "pv": PV, "t_ms": 5305300, "ok": True, "ms": 2900,
     "tries": 1, "via": "self"},
]  # fmt: skip

# ── one minimal valid event per kind ────────────────────────────────────────────

MINIMAL: dict[str, dict[str, Any]] = {
    "walk.done": {
        "act_id": "W1", "cmd": {"vx": 400, "vyaw": 0, "max_ms": 3000},
        "odom": {"dx_mm": 10, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 10}, "steps": 2,
        "reason": "time",
    },
    "sense.scan": {"act_id": "S1", "head_mrad": [480, 0, -420], "light": "bright",
                   "sectors": [{"b_mrad": 0, "w_mrad": 300, "free_mm": 900, "end": "wall",
                                "conf_ppm": 800000, "src": "cam"}],
                   "sightings": []},
    "sense.sighting": {"what": "thing", "b_mrad": 10, "conf_ppm": 500000, "look_id": "L1",
                       "edge": "new"},
    "sense.light": {"level": "bright", "luma": 90, "noise_q": 2},
    "sense.depth": {"ahead_mm": 800, "left_mm": None, "right_mm": 1200, "drop_zones": 0,
                    "touch": False, "head_pitch_mrad": -60},
    "sense.contact": {"contact": "blocked", "side": "front", "ev": ["odom", "gyro"]},
    "sense.lifted": {"conf_ppm": 950000, "posture": "stand"},
    "sense.set_down": {"conf_ppm": 900000, "posture": "sit", "on_nest": False},
    "body.fell": {"during": "walk", "ctx": {}},
    "body.righting": {"ok": False, "ms": 18000, "tries": 2},
    "body.posture": {"want": "stand", "got": "stand", "ms": 1400, "fell": False},
    "body.dock": {"contact": True, "via": "pg", "tries": 1, "seen_tag": True},
    "body.battery": {"pct": 64, "trend": "draining"},
    "body.hot": {"soc_c": 79, "tier": 2},
    "body.curio_step": {"id": "C12", "step": 3},
    "sense.thing": {"things": []},
    "body.guard": {"guard": "cliff"},
    "env.cue": {"cue": "name"},
    # phase 4 (WP9): the hands and the gamepad as the world hands them to Duck Core
    "sense.touch": {"where": "crown", "seconds": 3, "firmness": "normal", "path": "ear",
                    "latency_ms": 0},
    "sense.hold_head": {"path": "joint", "moving": False},
    "body.play": {"policy": "walk", "meters": 2},
}  # fmt: skip


# spec §7.2 gave ``sense.contact`` a field named ``kind`` (percept.py calls it ``contact``) and the
# curiosity spec gives ``body.guard`` one (cliff/spit/…): the same name as the envelope's event
# ``kind``, so one JSON object cannot carry both — percept.py calls the guard's field ``guard``.
COLLIDING: set[str] = set()


def event(kind: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "src": "live", "t_ms": 1000, "pv": PV, **MINIMAL[kind], **extra}


@pytest.mark.parametrize("kind,value", [("body.guard", "cliff")])
def test_colliding_kinds_can_be_sent(kind: str, value: str) -> None:
    body = {k: v for k, v in MINIMAL[kind].items() if k != "kind"}
    ev = {"kind": kind, "src": "live", "t_ms": 1000, "pv": PV, **body}
    for name in ("contact", "what", "guard", "contact_kind", "guard_kind"):
        if percept.validate({**ev, name: value}) == []:
            return
    assert percept.validate(ev) == [], f"{kind}: no field left to say {value}"


def test_appendix_a_examples_are_valid_and_round_trip() -> None:
    for ev in APPENDIX_A:
        assert percept.validate(ev) == [], ev["kind"]
        line = percept.encode(ev)
        assert percept.decode(line) == ev
        assert percept.encode(percept.decode(line)) == line, "canonical line is stable"
        assert json.loads(line) == ev


def test_every_kind_has_a_minimal_example_here() -> None:
    assert set(MINIMAL) == set(percept.SCHEMAS), sorted(set(percept.SCHEMAS) ^ set(MINIMAL))
    for kind in set(MINIMAL) - COLLIDING:
        assert percept.validate(event(kind)) == [], kind


def _floats_in(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Paths of every integer (not bool) inside ``value``."""
    out: list[tuple[Any, ...]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            out += _floats_in(v, (*path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out += _floats_in(v, (*path, i))
    elif isinstance(value, int) and not isinstance(value, bool):
        out.append(path)
    return out


def _set(obj: Any, path: tuple[Any, ...], value: Any) -> None:
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def test_a_float_anywhere_is_refused() -> None:
    for ev in [*APPENDIX_A, *(event(k) for k in MINIMAL if k not in COLLIDING)]:
        for path in _floats_in(ev):
            bad = copy.deepcopy(ev)
            _set(bad, path, float(_get(ev, path)) + 0.5)
            assert percept.validate(bad), f"{ev['kind']}: a float at {path} went through"


def _get(obj: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        obj = obj[key]
    return obj


def test_missing_stamps_and_bad_stamps_are_refused() -> None:
    good = event("sense.light")
    for stamp in percept.STAMPS:
        bad = dict(good)
        bad.pop(stamp)
        assert percept.validate(bad), f"missing {stamp} went through"
    for field_name, value in (
        ("src", "world"),
        ("t_ms", -1),
        ("t_ms", True),
        ("pv", "p:xyz"),
        ("pv", "p:3f9a…"),
        ("kind", "sense.pickup"),
    ):
        bad = dict(good)
        bad[field_name] = value
        assert percept.validate(bad), f"{field_name}={value!r} went through"


def test_required_fields_are_required_and_closed_kinds_are_closed() -> None:
    open_kinds = {"body.curio_step", "sense.thing", "body.guard", "env.cue"}
    for kind, body in MINIMAL.items():
        if kind in COLLIDING:
            continue
        for name in body:
            bad = event(kind)
            bad.pop(name)
            required = percept.validate(bad)
            # optional fields may be dropped; the required ones may not
            if required:
                assert any(name in p for p in required), (kind, name, required)
        extra = event(kind, surprise=1)
        if kind in open_kinds:
            assert percept.validate(extra) == [], kind
        else:
            assert percept.validate(extra), f"{kind}: an unknown field went through"


def test_phase3_owner_additions() -> None:
    ok = [
        event("body.fell", trigger="tilt40"),  # B1
        event("body.fell", trigger="fallen"),
        event("body.righting", how="sitstand", via="self"),  # B2
        event("walk.done", reason="reflex_unloaded"),  # B3: stopped, not claimed "lifted"
        event("walk.done", reason="veto_head", head="down"),  # A2
        event("walk.done", reason="deadline", stop_mm=None, purpose="approach"),
        event("sense.sighting", r_src="cue", r_mm=None),  # C5: a direction only
        event("sense.light", level="none", luma=0, noise_q=0),  # the eye is down
    ]
    for ev in ok:
        assert percept.validate(ev) == [], ev
    bad = [
        event("body.fell", trigger="tilt30"),
        event("body.righting", how="kick"),
        event("walk.done", reason="lifted_maybe"),
        event("walk.done", head="up"),
        event("sense.sighting", r_src="guess"),
        event("body.hot", tier=4),
        event("sense.light", level="grey", luma=0, noise_q=0),
    ]
    for ev in bad:
        assert percept.validate(ev), ev


def test_phase4_kinds_are_percept_v1s_own() -> None:
    """WP9 (phase-4 landing): ``sense.touch``, ``sense.hold_head`` and ``body.play`` are percept-v1
    kinds with exactly the fields phase 4 validated provisionally; ``legacy_phantom`` (LegacyWorld's
    transitional mark on the TV's false touch, design A10) is the only optional touch field."""
    ok = [
        event("sense.touch", legacy_phantom=True, path="ear"),
        event("sense.touch", seconds=percept.TOUCH_MAX_S, path="joint", where="beak"),
        event("sense.hold_head", moving=True),
        event("body.play", skill="spin", minute=3),
    ]
    for ev in ok:
        assert percept.validate(ev) == [], ev
    bad = [
        event("sense.touch", seconds=percept.TOUCH_MAX_S + 1),
        event("sense.touch", seconds=-1),
        event("sense.touch", path="skin"),
        event("sense.touch", legacy_phantom=1),
        event("sense.touch", latency_ms=-5),
        event("sense.hold_head", moving="yes"),
        event("body.play", meters=-1),
        event("body.play", minute=1.5),
    ]
    for ev in bad:
        assert percept.validate(ev), ev
    assert {"sense.touch", "sense.hold_head", "body.play"} <= set(percept.KINDS)
    assert all(percept.known_kind(kind) for kind in ("sense.touch", "body.play"))
    assert not percept.known_kind("sense.pickup")  # (a Duck Core input record, not a percept)
    worldport = pytest.importorskip("duckmind.worldport")
    assert not hasattr(worldport, "PROVISIONAL_SCHEMAS")
    assert worldport.validate is percept.validate and worldport.check is percept.check
    assert worldport.known_kind is percept.known_kind


def test_due_second_is_the_ceiling() -> None:
    assert [percept.due_second(t) for t in (0, 1, 999, 1000, 1001, 5231840)] == [
        0, 1, 1, 1, 2, 5232
    ]  # fmt: skip


def test_quantisation_gives_integers_and_refuses_non_finite() -> None:
    assert percept.mm(0.8425) == 842 and percept.mm(-0.1815) == -182
    assert percept.mrad(-0.196) == -196
    assert percept.ppm(1.7) == percept.PPM and percept.ppm(-0.1) == 0
    assert percept.pct(0.71, is_fraction=True) == 71
    for fn in (percept.mm, percept.mrad, percept.ppm):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(percept.PerceptError):
                fn(bad)
    assert all(isinstance(percept.mm(x / 7.0), int) for x in range(-50, 50))


def test_perception_hash_is_order_free_and_prefixed() -> None:
    a = percept.perception_hash({"code": "abc", "k": b"\x01\x02", "thresholds": "45"})
    b = percept.perception_hash({"thresholds": "45", "code": "abc", "k": b"\x01\x02"})
    c = percept.perception_hash({"code": "abc", "k": b"\x01\x02", "thresholds": "55"})
    assert a == b != c
    assert percept.PV_PATTERN.match(a)


# ── a seeded property loop (hypothesis is not a dependency) ─────────────────────


def _random_walk_done(rng: random.Random) -> dict[str, Any]:
    def fold() -> dict[str, Any]:
        return {
            "look_id": f"L{rng.randrange(10**6)}",
            "light": rng.choice(percept.LIGHT_LEVELS),
            "free_mm": [rng.randrange(0, 4001) for _ in range(5)],
            "ends": [rng.choice(percept.FREE_ENDS) for _ in range(5)],
        }

    ev: dict[str, Any] = {
        "kind": "walk.done",
        "src": rng.choice(percept.SOURCES),
        "t_ms": rng.randrange(0, 10**10),
        "pv": PV,
        "act_id": f"W{rng.randrange(10**6)}",
        "cmd": {"vx": rng.randrange(-400, 551), "vyaw": rng.randrange(-1000, 1001),
                "max_ms": rng.randrange(1, 60001)},
        "odom": {"dx_mm": rng.randrange(-3000, 3001), "dy_mm": rng.randrange(-3000, 3001),
                 "dyaw_mrad": rng.randrange(-3142, 3143), "path_mm": rng.randrange(0, 5000)},
        "steps": rng.randrange(0, 200),
        "reason": rng.choice(percept.WALK_REASONS),
    }  # fmt: skip
    if rng.random() < 0.7:
        ev["pre"] = fold()
    if rng.random() < 0.7:
        ev["end"] = fold()
    if rng.random() < 0.3:
        ev["contact"] = {
            "kind": rng.choice(percept.CONTACT_KINDS),
            "side": rng.choice(percept.SIDES),
        }
    if rng.random() < 0.5:
        ev["battery_pct"] = rng.randrange(0, 101)
    return ev


def test_random_valid_events_validate_and_single_mutations_do_not() -> None:
    rng = random.Random(20260927)
    for _ in range(400):
        ev = _random_walk_done(rng)
        assert percept.validate(ev) == [], ev
        assert percept.decode(percept.encode(ev)) == ev
        mutated = copy.deepcopy(ev)
        how = rng.randrange(4)
        if how == 0:
            mutated["reason"] = "teleported"
        elif how == 1:
            mutated["odom"]["path_mm"] = -1
        elif how == 2:
            mutated["cmd"]["vx"] = mutated["cmd"]["vx"] + 0.25
        else:
            mutated.pop(rng.choice(["act_id", "cmd", "odom", "steps", "reason"]))
        assert percept.validate(mutated), mutated


# ── §7.7 the event budget (needs duck-body's aggregate) ──────────────────────


def test_event_budget_over_a_scripted_excursion() -> None:
    aggregate = pytest.importorskip("duckmind.duckbody.aggregate")
    budget = getattr(aggregate, "DAILY_MEDIAN_BUDGET", 600)
    counter = getattr(aggregate, "EventBudget", None)
    if counter is None:
        pytest.skip("duckbody.aggregate has no EventBudget counter yet")
    assert budget <= 600  # spec §7.7: new sensation events ≤ 600 a day (median)
