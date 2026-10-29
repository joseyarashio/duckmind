"""percept-v1 (physical-design spec §7.2, appendix A): the schema, the integer rule, the stamps."""

from __future__ import annotations

import json

import pytest

from duckmind import percept as P

PV = "p:3f9a0c11d2e4b5a6"


def walk_done(**over: object) -> dict:
    event = {
        "kind": "walk.done",
        "src": "live",
        "pv": PV,
        "t_ms": 5231840,
        "act_id": "W1842",
        "cmd": {"vx": 300, "vyaw": 0, "max_ms": 6000},
        "pre": {
            "look_id": "L7710",
            "light": "bright",
            "free_mm": [620, 1180, 1400, 900, 410],
            "ends": ["thing", "wall", "unknown", "wall", "thing"],
            "trip": {"mm": 520, "mrad": -90},
            "drop": 0,
        },
        "end": {
            "look_id": "L7711",
            "light": "bright",
            "free_mm": [0, 380, 760, 1500, 1500],
            "ends": ["unknown", "wall", "wall", "unknown", "unknown"],
        },
        "odom": {"dx_mm": 842, "dy_mm": -181, "dyaw_mrad": -196, "path_mm": 861},
        "steps": 13,
        "reason": "reflex_near",
        "contact": None,
        "battery_pct": 71,
    }
    event.update(over)
    return event


APPENDIX_A = [
    walk_done(),
    {
        "kind": "sense.sighting",
        "src": "robot",
        "pv": PV,
        "t_ms": 5240210,
        "what": "nest",
        "id": "tag36h11:17",
        "b_mrad": -140,
        "e_mrad": -95,
        "r_mm": 1210,
        "r_src": "tag",
        "yaw_mrad": 2890,
        "conf_ppm": 930000,
        "light": "dim",
        "look_id": "L7715",
        "edge": "new",
    },
    {
        "kind": "sense.light",
        "src": "surrogate",
        "pv": "c:81d0aa55",
        "t_ms": 64800000,
        "level": "dark",
        "luma": 71,
        "noise_q": 14,
    },
    {
        "kind": "body.fell",
        "src": "live",
        "pv": PV,
        "t_ms": 5302400,
        "during": "posture",
        "ctx": {"look_id": "L7730"},
        "trigger": "tilt40",
    },
    {
        "kind": "body.righting",
        "src": "live",
        "pv": PV,
        "t_ms": 5305300,
        "ok": True,
        "ms": 2900,
        "tries": 1,
        "via": "self",
        "how": "wait",
    },
]


@pytest.mark.parametrize("event", APPENDIX_A, ids=lambda e: e["kind"])
def test_the_appendix_examples_are_valid(event: dict) -> None:
    assert P.validate(event) == []
    assert P.decode(P.encode(event)) == event


def test_floats_are_refused_anywhere() -> None:
    bad = walk_done(odom={"dx_mm": 842.5, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 861})
    problems = P.validate(bad)
    assert any("floats are not allowed" in p for p in problems)
    with pytest.raises(P.PerceptError):
        P.check(bad)
    assert P.validate(walk_done(t_ms=1.0))


def test_stamps_are_checked() -> None:
    assert any("event.src" in p for p in P.validate(walk_done(src="oracle")))
    assert any("event.pv" in p for p in P.validate(walk_done(pv="p:3f9a…")))
    assert any("event.t_ms" in p for p in P.validate(walk_done(t_ms=-1)))
    assert any("unknown kind" in p for p in P.validate(walk_done(kind="duck.walk")))


def test_unknown_fields_and_enums_are_refused() -> None:
    assert any("unknown field" in p for p in P.validate(walk_done(x_world=[1, 2])))
    assert any("reason" in p for p in P.validate(walk_done(reason="bumped")))
    wrong = walk_done()
    wrong["pre"] = dict(wrong["pre"], free_mm=[1, 2, 3])
    assert any("expected 5 items" in p for p in P.validate(wrong))


def test_phase3_additions() -> None:
    for reason in ("reflex_unloaded", "deadline", "veto_head", "reflex_dark"):
        assert P.validate(walk_done(reason=reason)) == []
    assert P.validate(walk_done(purpose="curio", head="down", stop_mm=120)) == []
    # owner 2026-09-28 「呆萌感」: the caller's attention on a walk, when it was not on its way
    assert P.validate(walk_done(attention="elsewhere")) == []
    assert P.validate(walk_done(attention="low")) == []
    assert P.validate(walk_done(attention="asleep"))
    fell = dict(APPENDIX_A[3], trigger="fallen")
    assert P.validate(fell) == []
    assert P.validate(dict(fell, trigger="tilt45"))
    lifted = {
        "kind": "sense.lifted",
        "src": "live",
        "pv": PV,
        "t_ms": 5,
        "conf_ppm": 900000,
        "posture": "stand",
        "on_nest": None,
    }
    assert P.validate(lifted) == []
    cue = dict(APPENDIX_A[1], r_mm=None, r_src="cue", edge="moved")
    assert P.validate(cue) == []
    contact = {
        "kind": "sense.contact",
        "src": "live",
        "pv": PV,
        "t_ms": 1,
        "contact": "brushed",
        "side": "left",
        "ev": ["heading"],
    }
    assert P.validate(contact) == []


def test_curiosity_kinds_pin_ids_and_the_integer_rule() -> None:
    step = {
        "kind": "body.curio_step",
        "src": "live",
        "pv": PV,
        "t_ms": 9,
        "id": "C3",
        "step": 2,
        "verb": "approach",
        "ok": True,
        "moved_mm": 210,
        "thing_after": {"x_mm": 1, "y_mm": 2, "err_mm": 3, "seen": True},
    }
    assert P.validate(step) == []
    assert P.validate(dict(step, moved_mm=0.2))
    assert P.validate({k: v for k, v in step.items() if k != "id"})


def test_make_and_quantisers() -> None:
    event = P.make("sense.light", src="live", t_ms=1000, pv=PV, level="dim", luma=80, noise_q=5)
    assert event["kind"] == "sense.light"
    assert P.mm(0.3004) == 300 and P.mrad(-0.0196) == -20 and P.ppm(1.5) == 1_000_000
    assert P.pct(0.714, is_fraction=True) == 71
    with pytest.raises(P.PerceptError):
        P.mm(float("nan"))


def test_due_second_is_the_ceiling() -> None:
    assert [P.due_second(t) for t in (0, 1, 999, 1000, 1001)] == [0, 1, 1, 1, 2]


def test_perception_hash_is_order_free_and_sensitive() -> None:
    a = P.perception_hash({"code": "x", "K": b"\x01\x02"})
    assert a == P.perception_hash({"K": b"\x01\x02", "code": "x"})
    assert a != P.perception_hash({"code": "y", "K": b"\x01\x02"})
    assert a.startswith("p:") and len(a) == 18
    assert P.validate(walk_done(pv=a)) == []


def test_encode_is_canonical_json() -> None:
    line = P.encode(walk_done())
    assert "\n" not in line and ", " not in line
    assert json.loads(line) == walk_done()
