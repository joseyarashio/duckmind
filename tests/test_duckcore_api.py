import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld: a v4_1 duck lives in it)
from duckmind import worldport
from duckmind.duckcore import api, commands, content
from duckmind.duckcore.api import (
    ENGINE_REV,
    FROZEN_REV,
    Config,
    LegacyActs,
    Session,
    clips,
    dispatch,
    feed,
    open_session,
    restore,
    save_json,
    snapshot,
)

SCRIPT = [
    ("act", {"action": "power_on", "persona": "owner"}),
    ("act", {"action": "touch", "persona": "child", "params": {"seconds": 10}}),
    ("act", {"action": "owner_hello"}),
    ("advance", {"minutes": 30}),
    ("command", {"text": "小孩摸頭 10 秒，然後主人說你好，過 1 小時"}),
    ("live", {"days": 12, "routine": "weekends"}),
    ("party", {"guest_mode": True}),
    ("forget_by_query", {"query": "被摸"}),
]


def _run(seed: int) -> Session:
    session = Session(Config(seed=seed))
    for method, params in SCRIPT:
        dispatch(session, method, params)
    return session


def test_same_seed_and_inputs_rebuild_the_same_chain():
    first, second = _run(7), _run(7)
    assert first.core.log.head == second.core.log.head
    assert first.core.log.verify()
    assert dispatch(first, "replay_check", {})["identical"] is True


def test_a_different_seed_is_a_different_duck():
    assert _run(7).core.log.head != _run(8).core.log.head


def test_tampering_with_history_breaks_the_chain():
    session = _run(7)
    event = session.core.log.events[3]
    object.__setattr__(event, "payload", {**event.payload, "forged": True})
    assert not session.core.log.verify()


def test_everything_the_page_receives_is_plain_json():
    session = _run(7)
    for method in (
        "graph",
        "diary",
        "events",
        "query",
        "why",
        "export",
        "clips",
        "replay_check",
        "percept_replay_check",
        "world_replay_check",
    ):
        json.dumps(
            dispatch(session, method, {"q": "唱歌"} if method == "query" else {}),
            ensure_ascii=False,
        )
    json.dumps(snapshot(session), ensure_ascii=False)
    json.dumps(feed(session, 0, 0), ensure_ascii=False)


def test_read_only_calls_are_not_recorded_as_inputs():
    session = Session(Config(seed=7))
    for method in (
        "state",
        "graph",
        "diary",
        "query",
        "events",
        "replay_check",
        "percept_replay_check",
        "world_replay_check",
    ):
        dispatch(session, method, {})
    assert session.inputs == []
    with pytest.raises(ValueError):
        dispatch(session, "enable_motors", {})


def test_the_command_language_chains_clauses_and_names_who_did_it():
    parsed = commands.parse("小孩摸頭 10 秒，然後主人說你好，過 1 小時")
    assert [command.kind for command in parsed] == ["act", "act", "advance"]
    assert [command.persona for command in parsed[:2]] == ["child", "owner"]
    assert parsed[0].method == "touch" and parsed[0].params.get("seconds") == 10
    assert parsed[2].params["minutes"] == 60


def test_unknown_commands_get_suggestions_instead_of_a_guess():
    (command,) = commands.parse("幫我訂披薩")
    assert command.kind == "unknown"
    assert command.params["suggestions"]


def test_the_core_never_sends_raw_robot_commands():
    """Standing up and floor walks are intents duck-body validates; never robotd's own calls."""
    methods = {command.method for text in commands.SUGGESTIONS for command in commands.parse(text)}
    assert not methods & {"enable", "init", "relax", "do", "move", "loadPolicy", "theremin"}


def test_clip_frames_map_to_joint_targets_in_the_right_direction():
    home = content.HEAD_HOME
    gaze = content.head_targets(dict(content.CLIPS["moon_gaze"][1][1]), side=1)
    assert gaze["neck_pitch"] > home["neck_pitch"]  # leans back...
    assert gaze["head_pitch"] < home["head_pitch"]  # ...and looks up
    tuck = content.head_targets(dict(content.CLIPS["tuck_away"][1][1]), side=1)
    assert tuck["neck_pitch"] < home["neck_pitch"] and tuck["head_pitch"] > home["head_pitch"]
    left = content.head_targets({"hy": 1.0, "hr": 1.0}, side=1)
    right = content.head_targets({"hy": 1.0, "hr": 1.0}, side=-1)
    assert left["head_yaw"] > 0 > right["head_yaw"]
    assert left["head_roll"] > 0 > right["head_roll"]
    assert content.head_targets({"m": 3.0}, side=1)["mouth"] == content.SEATED_ENVELOPE["mouth"]


def test_every_clip_stays_inside_the_seated_envelope():
    for name, frames in content.CLIPS.items():
        for _, frame in frames:
            assert all(-1.0 <= value <= 1.0 for value in frame.values()), name
            assert set(frame) <= {"np", "hp", "hy", "hr", "m"}, name
    for verb in content.VERBS:
        assert verb.verb_id in content.CLIPS, verb.verb_id


def test_the_page_gets_the_same_sign_table_the_core_uses():
    payload = clips(Session(Config(seed=7)))
    assert payload["sign"] == content.JOINT_SIGN
    assert payload["mouth_travel"] == pytest.approx(0.6108652, abs=1e-6)


# ---------------------------------------------------------------------------- phase 4: engines, the world, replay

# What the duck does (phase-4 design §2.2, the gate's behaviour projection), read the same from
# both engines: v4_1 records a stand-up as its body.posture result (no duck.posture(stand) of
# its own) and a fall as its own duck.posture(fallen) (v4: the body.fell it wrote), so those two
# are compared as lists of moments, in order.
BEHAVIOUR: dict[str, tuple[str, ...] | None] = {
    "duck.decision": ("rule", "verb", "trigger", "draw", "counter"),
    "duck.expression": ("verb", "reason", "amplitude", "sounds"),
    "duck.walk": ("verb", "vx", "vyaw", "seconds", "purpose"),
    "duck.posture": ("posture", "from", "reason"),
    "duck.day_stats": None,
    "nest.docked": ("by", "tries", "battery"),
    "nest.undocked": ("by", "battery"),
    "duck.excursion": ("phase", "reason", "ended_by"),
    "power.on": (),
    "power.off": ("reason", "battery"),
    "duck.nap_start": ("reason", "minutes"),
    "duck.bid": ("kind",),
    "duck.moment": ("id", "kind"),
    "duck.reaction": ("to", "verb", "sounds"),
    "duck.stop": ("until", "posture"),
    "duck.lockout": ("until",),
    "env.family": ("from", "to"),
    "growth.gd": ("gd", "credit"),
}


def _projection(core: Any) -> dict[str, list[Any]]:
    rows: list[Any] = []
    stands: list[str] = []
    falls: list[Any] = []
    for event in core.log.events:
        p, at = event.payload, event.occurred_at.isoformat()
        if event.kind == "duck.posture" and (p.get("posture"), p.get("from")) == ("stand", "sit"):
            stands.append(at)
        elif event.kind == "body.posture" and p.get("got") == "stand":
            stands.append(at)
        elif (event.kind == "duck.posture" and p.get("posture") == "fallen") or (
            event.kind == "body.fell" and "at" in p
        ):
            falls.append([at, p.get("at")])
        elif event.kind in BEHAVIOUR:
            keys = BEHAVIOUR[event.kind]
            rows.append([at, event.kind, dict(p) if keys is None else {k: p.get(k) for k in keys}])
    return {"rows": rows, "stands": stands, "falls": falls}


def _journal(core: Any) -> list[tuple[Any, ...]]:
    return [(j["at"], j["persona"], j["action"], j["detail"], j["perceived"]) for j in core.journal]


HANDS = [
    ("advance", {"until": "10:00"}),
    ("act", {"action": "power_on"}),
    ("set_setting", {"key": "floor", "value": True}),
    ("live", {"days": 6, "routine": "floor"}),
    ("advance", {"until": "09:00"}),
    ("act", {"action": "power_on"}),
    ("set_setting", {"key": "floor", "value": True}),
    *[
        step
        for i in range(8)
        for step in (
            ("act", {"action": "hold_head", "persona": "child" if i % 2 else "owner"}),
            ("advance", {"minutes": 2}),
            (
                "act",
                {
                    "action": "touch",
                    "persona": "owner",
                    "params": {
                        "seconds": 6,
                        "where": "beak" if i % 3 == 0 else "crown",
                        "firmness": ("scratch", "light", "normal")[i % 3],
                    },
                },
            ),
            ("act", {"action": "tv", "params": {"on": i % 4 != 3}}),
            ("advance", {"minutes": 50}),
        )
    ],
    ("set_setting", {"key": "conservative", "value": False}),
    ("act", {"action": "touch", "persona": "owner", "params": {"firmness": "light"}}),
    ("act", {"action": "pad_play", "persona": "child", "params": {"minutes": 7}}),
    ("act", {"action": "pickup", "persona": "owner"}),
    ("act", {"action": "power_off", "persona": "owner"}),
    ("act", {"action": "touch", "persona": "owner"}),
]
CASES = {
    "api_script": ({"seed": 7}, SCRIPT),
    "hands_tv_settings": ({"seed": 7}, HANDS),
    "clumsy_family": (
        {
            "seed": 13,
            "nest": True,
            "household": "dual_income",
            "fall_ppm_per_s": 40_000,
            "righting_ppm": 150_000,
            "stand_ppm": 600_000,
            "fall_ppm_per_start": 20_000,
        },
        [("live", {"family": "dual_income", "days": 5}), *HANDS[1:]],
    ),
    "clumsy_floor_party": (
        {"seed": 21, "fall_ppm_per_s": 40_000, "righting_ppm": 150_000},
        [("live", {"days": 8, "routine": "floor"}), ("party", {"guest_mode": False})],
    ),
}


def _both(name: str) -> tuple[Any, Session]:
    config, script = CASES[name]
    old = open_session(dict(config))
    new = open_session({**config, "engine_rev": ENGINE_REV})
    assert isinstance(new, Session)
    for method, params in script:
        dispatch(old, method, params)
        dispatch(new, method, params)
    return old, new


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_v4_1_duck_in_legacy_world_does_exactly_what_the_v4_duck_did(name: str) -> None:
    """The phase-4 invariant on the api's own paths: the family's hands (touch, head hold,
    their misses), the TV, the settings the sensors follow, gamepad play, pickup, power, a
    party, the command box, the old routines and a household-v2 family, at v4's rates and at
    stressed ones. The frozen v4 engine and v4_1 + LegacyWorld do the same things at the same
    moments, and the family's journal (the page's truth lane) is the same line for line."""
    old, new = _both(name)
    assert _projection(new.core) == _projection(old.core)
    assert _journal(new.core) == _journal(old.core)
    kinds = {event.kind for event in new.core.log.events}
    assert "duck.act" in kinds and "walk.done" in kinds or name == "api_script"
    assert not {"detect.touch", "detect.hold", "false_alarm.tv"} & set(new.core.counters)
    for check in ("replay_check", "percept_replay_check", "world_replay_check"):
        assert dispatch(new, check, {})["identical"] is True, check


def test_the_hands_case_reaches_every_detector_path() -> None:
    """(What the case above covers: a missed and a detected touch, a missed and a detected
    head hold, the TV mistaken for a touch, gamepad minutes, a touch on a switched-off duck.)"""
    _, new = _both("hands_tv_settings")
    journal = [(j["action"], j["perceived"]) for j in new.core.journal]
    assert ("touch", True) in journal and ("touch", False) in journal
    assert ("hold_head", True) in journal and ("hold_head", False) in journal
    assert ("false_alarm", False) in journal
    records = [event for event in new.core.log.events if event.kind == "sense.touch"]
    assert any(event.payload.get("legacy_phantom") for event in records)
    assert all(event.payload["pv"].startswith("l:") for event in records)
    assert any(event.kind == "body.play" for event in new.core.log.events)
    state = new.port.state()  # type: ignore[attr-defined]
    assert state["counters"]["detect.touch"] >= 8 and state["conservative"] is False


def test_a_new_duck_is_v4_1_in_a_legacy_world_with_v4s_rates() -> None:
    session = Session(Config(seed=7))
    assert session.config.engine_rev == ENGINE_REV == "v4_1"
    header = session.world
    assert header is not None and header["world"] == "legacy" and header["seed"] == 7
    assert header["table_hash"] == content.table_hash() == session.core.table
    assert header["rates"] == api.v4_world_fields()  # v4's own defaults, from the frozen v4
    assert header["marker"] is True and header["sensors"] == {"ear": True, "conservative": True}
    # R1: the rates are the world's, never the duck's own Config
    assert set(api.v4_world_fields()) == {
        "fall_ppm_per_s",
        "fall_ppm_per_start",
        "righting_ppm",
        "stand_ppm",
        "dock_ppm",
    }
    assert not set(api.v4_world_fields()) & set(Config.__dataclass_fields__)
    assert session.core.port is session.port and isinstance(session.hands, LegacyActs)
    assert snapshot(session)["engine_rev"] == ENGINE_REV
    assert snapshot(session)["identity"]["world"]["calib_hash"] == worldport.calib_hash(header)


def test_a_saved_duck_without_engine_rev_opens_on_the_frozen_engine(tmp_path: Path) -> None:
    from duckmind.engines.v4_0.duckcore import api as frozen

    saved = {"seed": 11, "name": "蛋黃", "fall_ppm_per_s": 150_000, "nest": False}
    # (a second argument that is no world header is ignored for a v4 duck: replay_golden.py
    # passes its corpus folder to --factory …:open_session)
    session = open_session(saved, tmp_path)
    assert isinstance(session, frozen.Session) and session.config.fall_ppm_per_s == 150_000
    stock = frozen.Session(frozen.Config(**saved))
    for method, params in SCRIPT[:5]:
        dispatch(session, method, params)
        frozen.dispatch(stock, method, params)
    assert session.core.log.head == stock.core.log.head
    assert api.config_json(session) == {**frozen.config_json(stock), "engine_rev": FROZEN_REV}
    assert api.world_json(session) is None
    assert snapshot(session)["engine_rev"] == FROZEN_REV and feed(session, 0, 0)["events"]
    assert api.banner(session)["title"] and clips(session)["clips"]
    assert api.family_profile(session, "hybrid").kind == "hybrid"
    assert dispatch(session, "replay_check", {})["identical"] is True
    again = restore(json.loads(json.dumps(save_json(session))))
    assert isinstance(again, frozen.Session) and again.core.log.head == session.core.log.head
    for method in ("sense", "percept_replay_check", "world_replay_check"):
        with pytest.raises(ValueError):
            dispatch(session, method, {"events": []})
    with pytest.raises(ValueError, match="no Config field"):
        open_session({"seed": 1, "fall_ppm": 3})  # an unknown field is never dropped silently


def test_open_session_splits_an_old_style_config_between_duck_and_world() -> None:
    session = open_session(
        {"seed": 5, "engine_rev": ENGINE_REV, "fall_ppm_per_s": 150_000, "camera": False}
    )
    assert isinstance(session, Session) and session.world is not None
    assert session.world["rates"]["fall_ppm_per_s"] == 150_000
    assert session.world["rates"]["righting_ppm"] == api.v4_world_fields()["righting_ppm"]
    assert session.world["marker"] is False
    config = api.config_json(session)
    assert config["engine_rev"] == ENGINE_REV and "fall_ppm_per_s" not in config
    header = api.world_json(session)
    assert open_session(config, header).world == header
    with pytest.raises(ValueError, match="belong to the world header"):
        open_session({**config, "fall_ppm_per_s": 1}, header)
    with pytest.raises(ValueError, match="unknown engine"):
        open_session({"engine_rev": "v9_9"})
    with pytest.raises(ValueError, match="no Config field"):
        open_session({"engine_rev": ENGINE_REV, "fall_ppm": 1})
    with pytest.raises(TypeError):
        open_session({"engine_rev": ENGINE_REV}, Path("."))
    with pytest.raises(ValueError, match="open_session"):
        Session(Config(engine_rev=FROZEN_REV))


def test_a_v4_1_duck_saves_and_restores_with_its_world() -> None:
    session = open_session({"seed": 9, "engine_rev": ENGINE_REV, "righting_ppm": 0})
    for method, params in SCRIPT[:5]:
        dispatch(session, method, params)
    saved = json.loads(json.dumps(save_json(session), ensure_ascii=False))
    assert (
        set(saved) == {"config", "world", "inputs", "content"}
        and saved["world"]["rates"]["righting_ppm"] == 0
    )
    assert saved["content"] == api.content_version()  # (the content version rides with the save)
    exported = dispatch(session, "export", {})
    assert exported["world"] == saved["world"] and exported["config"]["engine_rev"] == ENGINE_REV
    again = restore(saved)
    assert again.core.log.head == session.core.log.head and again.inputs == session.inputs


def test_the_three_replays_catch_a_tape_that_lies() -> None:
    session = _run(7)
    pr = dispatch(session, "percept_replay_check", {})
    assert pr["identical"] is True and pr["percepts"] > 0 and pr["all_percepts_taken"] is True
    wr = dispatch(session, "world_replay_check", {})
    assert wr["identical"] is True and wr["percepts"] > 0
    tape = session.core.tape
    assert tape is not None and tape.header["engine_rev"] == ENGINE_REV
    first = next(i for i, e in enumerate(tape.entries) if e.get("method") == "on_percept")
    del tape.entries[first]  # a family touch the duck never felt
    assert dispatch(session, "percept_replay_check", {})["identical"] is False
    quiet = Session(Config(seed=7), keep_tape=False)
    for method, params in SCRIPT[:4]:
        dispatch(quiet, method, params)
    assert quiet.core.tape is None
    assert dispatch(quiet, "percept_replay_check", {})["identical"] is False
    assert dispatch(quiet, "world_replay_check", {})["identical"] is None
    assert dispatch(quiet, "replay_check", {})["identical"] is True


def _touch(t_ms: int, **fields: Any) -> dict[str, Any]:
    return {
        "kind": "sense.touch",
        "src": "robot",
        "t_ms": t_ms,
        "pv": "p:" + "ab" * 8,
        "where": "crown",
        "seconds": 2,
        "firmness": "normal",
        "path": "ear",
        "latency_ms": 1100,
        **fields,
    }


def test_sense_checks_every_percept_and_takes_each_in_at_its_second() -> None:
    session = Session(Config(seed=7))
    dispatch(session, "act", {"action": "power_on", "persona": "owner"})
    now = session.core.now_ms()
    head, inputs = session.core.log.head, len(session.inputs)
    with pytest.raises(ValueError):  # one bad percept (a float) refuses the whole batch
        dispatch(
            session, "sense", {"events": [_touch(now + 2500), _touch(now + 3000, seconds=2.5)]}
        )
    assert session.core.log.head == head and len(session.inputs) == inputs
    reply = dispatch(session, "sense", {"events": [_touch(now + 2500)]})
    assert session.core.now_ms() == now + 3000  # due at ceil(t_ms / 1000)
    record = session.core.log.get(reply["taken"][0]["event"])
    assert (
        record is not None and record.kind == "sense.touch" and record.payload["t_ms"] == now + 2500
    )
    assert session.inputs[-1][0] == "sense"
    stale = {
        "kind": "walk.done",
        "src": "robot",
        "t_ms": now + 1000,  # (behind the clock: taken in now)
        "pv": "p:" + "ab" * 8,
        "act_id": "A999",
        "cmd": {"vx": 100, "vyaw": 0, "max_ms": 2000},
        "odom": {"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 0},
        "steps": 0,
        "reason": "time",
    }
    late = dispatch(session, "sense", {"events": [stale]})["taken"][0]
    assert late["taken"] is False and session.core.now_ms() == now + 3000
    assert session.core.log.get(late["event"]).payload["stale"] is True  # type: ignore[union-attr]
    for check in ("replay_check", "percept_replay_check"):
        assert dispatch(session, check, {})["identical"] is True, check


def test_the_family_proxy_rolls_detection_in_the_world_and_reads_the_duck() -> None:
    session = Session(Config(seed=7))
    hands = session.hands
    assert (
        hands.s is session.core.s and hands.hw is session.core.hw and hands.log is session.core.log
    )
    assert hands.advance_seconds == session.core.advance_seconds  # (bound methods pass through)
    dispatch(session, "act", {"action": "power_on", "persona": "owner"})
    counters = session.port.counters  # type: ignore[attr-defined]
    before = counters.get("detect.touch", 0)
    for _ in range(4):
        dispatch(session, "act", {"action": "touch", "persona": "child", "params": {"seconds": 2}})
    assert counters["detect.touch"] == before + 4 and "detect.touch" not in session.core.counters
    dispatch(session, "set_setting", {"key": "conservative", "value": False})
    dispatch(session, "act", {"action": "tv", "params": {"on": True}})
    state = session.port.state()  # type: ignore[attr-defined]
    assert state["conservative"] is False and state["tv_on"] is True
    dispatch(session, "act", {"action": "power_off", "persona": "owner"})
    off = dispatch(session, "act", {"action": "touch", "persona": "owner"})
    assert off == {"perceived": False, "note": "關機中"} and counters["detect.touch"] == before + 4
    told = [entry["call"] for entry in session.port.tape_json()]  # type: ignore[attr-defined]
    assert {"touch", "set_setting", "tv", "hand"} <= set(told)
    session.core.s.awake = True  # (a world with no hands of its own)
    with pytest.raises(ValueError, match="no family hands"):
        LegacyActs(session.core, worldport.QueuePort()).touch("owner")


def _event(kind: str, **payload: Any) -> Any:
    return SimpleNamespace(kind=kind, payload=payload, event_id="E1")


def test_the_page_motion_reads_both_engines_chains() -> None:
    motion = api._motion
    v4 = motion(_event("duck.walk", **{"from": [0.1, -0.2, 1.571], "to": [0.3, 0.0, 0.0]}))
    v41 = motion(
        _event(
            "duck.walk",
            bel_from={"x_mm": 100, "y_mm": -200, "yaw_mrad": 1571},
            expect={"x_mm": 300, "y_mm": 0, "yaw_mrad": 0},
        )
    )
    assert v4 is not None and v41 is not None and v41["walk"] == v4["walk"]
    assert motion(_event("body.fell", during="walk", ctx={"act_id": "A1"})) == {"fall": True}
    cut = {"walk": "W", "after_s": 3, "pose": [0.1, 0.1, 0.0]}
    assert motion(_event("body.fell", at=[1, 2, 3], walk_cut=cut)) == {
        "fall": True,
        "at": [1, 2, 3],
        "cut": cut,
    }
    fallen = motion(_event("duck.posture", posture="fallen", reason="fell", walk_cut=cut))
    assert fallen == {"posture": "fallen", "stop": True, "cut": cut}
    # v4_1 puts where it lies on its duck.posture(fallen), not on body.fell (LANDING §3b): the
    # page lays the fallen body there, as it did with v4's body.fell
    lies = [1.2, -0.4, 0.5]
    fell_here = motion(_event("duck.posture", posture="fallen", walk_cut=cut, at=lies))
    assert fell_here == {"posture": "fallen", "stop": True, "cut": cut, "at": lies}
    still = motion(_event("duck.posture", posture="fallen", walk_cut=None, at=lies))
    assert still == {"posture": "fallen", "at": lies}
    assert motion(_event("duck.posture", posture="sit", at=lies)) == {"posture": "sit"}
    assert motion(_event("body.posture", want="stand", got="stand")) == {"posture": "stand"}
    assert motion(_event("body.posture", want="stand", got="sit")) is None
    assert motion(_event("body.righting", ok=True, ms=0, tries=2)) == {
        "righting": {"try": 2, "ok": True}
    }
    assert motion(_event("body.righting", **{"try": 1, "ok": False})) == {
        "righting": {"try": 1, "ok": False}
    }
    assert motion(_event("duck.righting_end", helped="hand", tries=1)) == {
        "righting": {"try": 1, "ok": False}
    }
    assert motion(_event("duck.struggle")) == motion(_event("body.struggle"))
    for kind in ("duck.act", "walk.done", "duck.act_timeout", "duck.heard"):
        assert motion(_event(kind)) is None, kind


def test_the_feed_shows_the_walks_and_falls_not_the_commands() -> None:
    _, new = _both("clumsy_floor_party")
    items = feed(new, 0, 0, limit=100_000)["events"]
    kinds = {item["kind"] for item in items}
    assert not kinds & {"duck.act", "walk.done", "duck.decision"}
    assert "duck.walk" in kinds and "body.fell" in kinds
    assert all(item["motion"] == {"fall": True} for item in items if item["kind"] == "body.fell")
    json.dumps(items, ensure_ascii=False)


def test_the_new_inputs_are_recorded() -> None:
    assert "sense" in api.MUTATING
    assert {"thing_mark", "home_zone", "family_play", "answer_ask"} <= api.ACT_METHODS
    config = Config()
    assert (config.curiosity, config.home_kit, config.carry_enabled, config.toddler) == (
        True,
        "",
        True,
        False,
    )
    assert config.ball_leash_ext_mm == 0


def test_the_migration_seam_records_where_the_duck_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    """The api's side of WP13 (engines/migrate.py moves the state; a stand-in here): the moved
    duck keeps the config it hatched with and a ``migrate`` input at the seam, so a restore and
    replay_check go through the old engine and the same seam."""

    def stand_in(old: Any, to: str) -> Session:
        moved = open_session({**api.config_json(old), "engine_rev": to})
        assert isinstance(moved, Session)
        dispatch(moved, "advance", {"minutes": len(old.inputs)})
        return moved

    monkeypatch.setattr(api, "_migrator", lambda: stand_in)
    old = open_session({"seed": 4, "fall_ppm_per_s": 90})
    for method, params in SCRIPT[:3]:
        dispatch(old, method, params)
    moved = api.migrate(old)
    assert moved.origin == api.config_json(old) and moved.world["rates"]["fall_ppm_per_s"] == 90  # type: ignore[index]
    assert moved.inputs[-1] == ("migrate", {"to": ENGINE_REV}) and len(moved.inputs) == 4
    dispatch(moved, "advance", {"minutes": 5})
    saved = json.loads(json.dumps(save_json(moved)))
    assert saved["config"]["engine_rev"] == FROZEN_REV
    again = restore(saved)
    assert isinstance(again, Session) and again.core.log.head == moved.core.log.head
    assert dispatch(moved, "replay_check", {})["identical"] is True
    with pytest.raises(ValueError, match="already runs"):
        api.migrate(moved)
    with pytest.raises(ValueError, match="new session"):
        dispatch(moved, "migrate", {"to": ENGINE_REV})
    with pytest.raises(ValueError, match="cannot move"):
        api.migrate(old, FROZEN_REV)
