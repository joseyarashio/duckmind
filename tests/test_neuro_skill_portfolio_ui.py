"""The simulator's nine-skill portfolio stays honest and fully translated. The page's side runs in
Node over the page's own markup (tests/page_tree_node_harness.py), on the reports the real
serializer makes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from page_tree_node_harness import NODE, run

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")
LOCALES = ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko")
SKILLS = {
    "follow": "follow_mom",
    "hide": "hide",
    "circle": "circle_settle",
    "theremin": "theremin",
    "ride": "wheee_ride",
    "dance": "dance",
    "ball": "play_ball",
    "carry": "carry_toy",
    "sun": "sunbathe",
}
# why the runtime can refuse a one-shot skill (it says the code; the page says it in words)
RUNTIME_REASONS = (
    "policy_unsettled",
    "selection_changed",
    "policy_start_timeout",
    "readiness_timeout",
    "wrong_act",
    "body_unsafe",
    "guard_refused",
)


def _catalog(locale: str) -> dict[str, str]:
    text = (STATIC / f"i18n.{locale}.page.js").read_text()
    begin = text.index("{", text.index("DuckI18n.load("))
    return json.loads(text[begin : text.index("\n});", begin) + 2])


def _wire(payload: Any) -> Any:
    """An answer as the server sends it (every word with its catalog message beside it)."""
    from duckmind.i18n import boundary

    return json.loads(json.dumps(boundary.attach(payload), ensure_ascii=False))


@pytest.fixture(scope="module")
def portfolio() -> dict[str, dict[str, Any]]:
    """The page's nine skill cards after each of these telemetry answers: the stub duck's own
    answer, with the behaviour chosen and the skill reports of the real serializer put in."""
    if NODE is None:
        pytest.skip("needs Node")
    from duckmind.simweb import neuro_api as N

    hub = N.NeuroHub(enabled=True)
    hub.attach(N.StubSource(hub.card()))
    base = hub.telemetry(0)

    def answer(behaviour: str, skills: Any) -> Any:
        said = {**base, "behaviour": {**base["behaviour"], "key": behaviour}}
        return _wire(said if skills is None else {**said, "skills": skills})

    kicking = _serializer_lanes(behaviour="play_ball", ball_phase="kick")
    kick = {"behaviour": "play_ball", "skill": "kick_left", "seq": 3}
    scared = {"riding": False, "lifted": True, "after": 0, "outcome": "fear"}
    steps = {
        f"chosen:{skill}": answer(b, _serializer_lanes(behaviour=b)) for skill, b in SKILLS.items()
    }
    steps["no report"] = answer("play_ball", None)
    steps["doing"] = answer("play_ball", kicking)
    steps["not chosen"] = answer("forage", kicking)
    steps["cannot"] = answer(
        "carry_toy", _serializer_lanes(jaw=False, helper=False, behaviour="carry_toy")
    )
    steps["ride"] = answer(
        "held", _serializer_lanes(behaviour="held", reflex="lifted", ride=scared)
    )
    for reason in (*RUNTIME_REASONS, "a_reason_nobody_named"):
        refused = _serializer_lanes(
            behaviour="play_ball",
            play_skill="kick_left",
            play_seq=3,
            actions=[{**kick, "status": "refused", "reason": reason}],
        )
        steps[f"refused:{reason}"] = answer("play_ball", refused)
    drawn = run(
        "neuro", {"neuro.js"}, {"status": _wire(hub.status()), "telemetry": [*steps.values()]}
    )
    return dict(zip(steps, drawn["skills"], strict=True))


@needs_node
def test_the_page_has_a_card_for_each_of_the_nine_skills_and_marks_the_one_chosen(
    portfolio: dict[str, dict[str, Any]],
) -> None:
    assert set(_serializer_lanes()) == set(SKILLS)
    for skill in SKILLS:  # (the behaviour of each skill, chosen in turn)
        cards = portfolio[f"chosen:{skill}"]
        assert set(cards) == set(SKILLS)
        assert [name for name, card in cards.items() if card["intent"]] == [skill]
        assert cards[skill]["result"] is None  # (chosen is not done)
        for card in cards.values():
            assert card["says"] == f"page.neuro.skills.status.{card['state']}"


@needs_node
def test_a_chosen_behaviour_is_intent_and_the_runtimes_report_says_what_is_done(
    portfolio: dict[str, dict[str, Any]],
) -> None:
    # chosen, and no report came: it is not called running, waiting or done
    silent = portfolio["no report"]["ball"]
    assert silent["intent"] and silent["state"] == "unknown" and silent["result"] is None
    doing = portfolio["doing"]["ball"]
    assert doing["intent"] and doing["state"] == "active" and doing["result"] is None
    assert doing["phase"] == "page.neuro.skills.label.phase · page.neuro.skills.phase.kick"
    # a report that says "active" for a behaviour that is not the one chosen now is not shown active
    stale = portfolio["not chosen"]["ball"]
    assert not stale["intent"] and stale["state"] == "waiting"
    # what the body cannot do is missing, with the reason, however much it is chosen
    cannot = portfolio["cannot"]
    assert cannot["carry"]["intent"] and cannot["carry"]["state"] == "missing"
    assert cannot["carry"]["reason"] == "page.neuro.skills.reason.no_jaw"
    for skill in ("theremin", "ride", "dance"):
        assert cannot[skill]["state"] == "missing", skill
        assert cannot[skill]["reason"] == "page.neuro.skills.reason.no_voice_helper", skill
    # how a ride felt is said as it was, not as a success
    ride = portfolio["ride"]["ride"]
    assert ride["result"] == "page.neuro.skills.label.result · page.neuro.skills.result.fear"
    assert ride["state"] not in ("done", "interrupted")


@needs_node
def test_a_refused_skill_says_why_in_words(portfolio: dict[str, dict[str, Any]]) -> None:
    for reason in RUNTIME_REASONS:
        card = portfolio[f"refused:{reason}"]["ball"]
        assert card["state"] == "refused" and card["result"] is None, reason
        assert card["reason"] == f"page.neuro.skills.reason.{reason}", reason
    unnamed = portfolio["refused:a_reason_nobody_named"]["ball"]
    assert unnamed["state"] == "refused"
    assert unnamed["reason"] == "page.neuro.skills.reason.unknown"  # (never a raw code)


@needs_node
def test_sound_world_inputs_are_world_events_and_playback_is_reported_separately() -> None:
    """The sound scenes' buttons change the world (music, a hand over the duck, a swing) through
    ``/api/neuro/sound/world``: each thing the page sends is one the server passes to the world,
    none is a command to the brain. Whether the page itself can play sound is its own report."""
    from duckmind.simweb.neuro_sound_live import SoundHub

    strip = run("sound", {"neuro_sound.js"}, {"status": SoundHub(lambda: None).status()})
    assert strip["built"] and strip["in_play_tab"]
    assert len(strip["buttons"]) >= 7 and len(strip["sent"]) == len(strip["buttons"])
    world: list[tuple[str, dict[str, Any]]] = []

    def world_call(op: str, **params: Any) -> dict[str, Any]:
        world.append((op, params))
        return {}

    driver = SimpleNamespace(_body=SimpleNamespace(world_call=world_call), _aux_wake=None)
    for post in strip["sent"]:
        assert post["url"] == "/api/neuro/sound/world", post
        raw = json.dumps(post["body"]).encode()
        handler = SimpleNamespace(
            headers={"Content-Length": str(len(raw))},
            rfile=io.BytesIO(raw),
            wfile=io.BytesIO(),
            send_response=lambda status: None,
            send_header=lambda *a: None,
            end_headers=lambda: None,
        )
        SoundHub(lambda: driver).post(handler, post["url"])  # type: ignore[arg-type]
        assert json.loads(handler.wfile.getvalue())["ok"] is True, post
    assert [op for op, _ in world] == ["x_neuro_sound"] * len(strip["sent"])
    assert {params["do"] for _, params in world} == {
        "music",
        "music_off",
        "hover",
        "hover_end",
        "swing",
    }
    assert {params.get("tempo") for _, params in world} >= {"slow", "mid", "fast"}
    # all it asks the server besides: the sound's own status (and its stream, once it runs)
    assert set(strip["asked"]) == {"GET /api/neuro/sound/status", "POST /api/neuro/sound/world"}
    # the page's own playback is told apart from what the server says of the sound
    told = strip["told"]
    assert told["running"] is False and told["unavailable"] is False
    assert told["pcm"] == {"frames": 0, "unlocked": False}  # (no sound came, none could play)


def test_all_portfolio_and_sound_strings_exist_in_every_locale():
    required = {
        "page.neuro.skills.title",
        "page.neuro.skills.intro",
        "page.neuro.skills.selected",
        "page.neuro.skills.label.phase",
        "page.neuro.skills.label.result",
        "page.neuro.skills.label.stream",
        "page.neuro.skills.status.active",
        "page.neuro.skills.status.waiting",
        "page.neuro.skills.status.done",
        "page.neuro.skills.status.interrupted",
        "page.neuro.skills.status.refused",
        "page.neuro.skills.status.missing",
        "page.neuro.skills.status.unknown",
        "page.neuro.skills.category.social",
        "page.neuro.skills.category.play",
        "page.neuro.skills.category.sound",
        "page.neuro.skills.result.fear",
        "page.neuro.skills.result.joy",
        "page.neuro.skills.reason.unknown",
        "page.neuro.skills.reason.no_sensor",
        "page.neuro.skills.stream.unavailable",
        "page.neuro.skills.stream.received",
        "page.neuro.skills.stream.playing",
        "page.neuro.sound.header",
        "page.neuro.sound.hand.aria",
        "page.neuro.sound.music.volume",
        "page.neuro.sound.note.click",
        "page.neuro.sound.world.sent",
        "page.neuro.sound.world.failed",
        "page.neuro.sound.status.dance.beat",
        "page.neuro.sound.status.theremin.playing",
        "page.neuro.sound.status.ride.fun",
        "page.neuro.sound.status.ride.scared",
    }
    for skill in SKILLS:
        required.update((f"page.neuro.skills.{skill}.name", f"page.neuro.skills.{skill}.trigger"))
    phases = (
        "following",
        "waiting",
        "idle",
        "go",
        "in",
        "out",
        "failed",
        "turning",
        "seeking",
        "done",
        "stopped",
        "listening",
        "playing",
        "lifted",
        "riding",
        "settling",
        "dancing",
        "chase",
        "search",
        "line_up",
        "kick",
        "beak",
        "watch",
        "to_toy",
        "survey",
        "around",
        "backoff",
        "grab",
        "carry",
        "drop",
        "regrip",
        "into",
        "check",
        "settle",
        "deeper",
        "bask",
        "shift",
        "waiting_policy",
        "waiting_safety",
        "dispatched",
        "policy_started",
        "interrupted_waiting_for_exit",
        "refused",
        "readiness_timeout",
        "policy_start_timeout",
        "interrupted_policy_exited",
        "selection_ended",
        "policy_exited",
    )
    results = (
        "calm",
        "gave_up",
        "complete",
        "partial",
        "fear",
        "joy",
        "rolled",
        "lost",
        "missed",
        "unseen",
        "dropped_home",
    )
    reasons = (
        "no_jaw",
        "jaw_unknown",
        "no_voice_helper",
        "skill_state_unknown",
        "missing_brain",
        "missing_senses",
        "missing_motor",
        "policy_busy",
        "reflex_active",
        "battery_unknown",
        "battery",
        "unloaded",
        "lifted",
        "fallen_or_righting",
        "body_gate",
        "body_not_still",
        "head_not_home",
        "hide_failed",
        "circle_stopped",
        "world_sensor_error",
        "forbidden_intent",
        "wrong_act",
        "body_unsafe",
        "guard_refused",
        "policy_unsettled",
        "selection_changed",
        "policy_start_timeout",
        "readiness_timeout",
    )
    required.update(f"page.neuro.skills.phase.{code}" for code in phases)
    required.update(f"page.neuro.skills.result.{code}" for code in results)
    required.update(f"page.neuro.skills.reason.{code}" for code in reasons)
    for locale in LOCALES:
        catalog = _catalog(locale)
        assert required <= catalog.keys(), (locale, sorted(required - catalog.keys()))
        assert all(catalog[key] for key in required)


def test_done_status_is_neutral_and_does_not_claim_success_in_any_locale():
    expected = {
        "zh-Hant": "本次行為已結束",
        "en": "Behavior ended",
        "ja": "今回の行動は終了",
        "es": "Comportamiento finalizado",
        "zh-Hans": "本次行为已结束",
        "ko": "이번 행동 종료",
    }
    for locale, text in expected.items():
        assert _catalog(locale)["page.neuro.skills.status.done"] == text


def _serializer_lanes(
    *,
    jaw=True,
    helper=True,
    tof_ready=True,
    behaviour=None,
    reflex=None,
    play_phase="idle",
    ball_phase="idle",
    carry_phase="idle",
    sun_phase="idle",
    hide_phase="idle",
    ride=None,
    actions=None,
    play_skill=None,
    play_seq=0,
    world_errors=None,
):
    from types import SimpleNamespace

    from duckmind.simweb.neuro_runtime import skill_status

    class Sound:
        def view(self):
            return {
                "theremin": {"up": False},
                "ride": ride or {"riding": False, "lifted": False, "after": 0, "outcome": None},
                "dance": {"on": False},
            }

    brain = SimpleNamespace(
        play=SimpleNamespace(
            carry=SimpleNamespace(last_event=None, phase=carry_phase),
            ball=SimpleNamespace(phase=ball_phase, last_event=None),
            bask_ctl=SimpleNamespace(phase=sun_phase),
        ),
        play_out=SimpleNamespace(phase=play_phase, skill=play_skill, skill_seq=play_seq),
        sound_out=Sound(),
        hide_run=SimpleNamespace(phase=hide_phase),
        circle=SimpleNamespace(phase="idle"),
        social_view=lambda: {},
    )
    lanes = SimpleNamespace(
        brain=brain,
        names={"brain": "real", "senses": "real", "motor": "real"},
        senses=SimpleNamespace(capabilities={"jaw_present": jaw}),
        sound=SimpleNamespace(engine=SimpleNamespace(view=lambda: {"helper": helper})),
    )
    return skill_status(
        lanes,
        behaviour,
        1234,
        reflex=reflex,
        actions=actions,
        world_errors=world_errors,
        tof_ready=tof_ready,
    )


def test_real_skill_status_serializer_contract_and_capability_refusals():
    reports = _serializer_lanes(jaw=False, helper=False)
    assert set(reports) == {
        "follow",
        "hide",
        "circle",
        "theremin",
        "ride",
        "dance",
        "ball",
        "carry",
        "sun",
    }
    for report in reports.values():
        assert set(report) == {"available", "reason", "phase", "status", "result", "updated_ms"}
        assert report["updated_ms"] == 1234
    assert reports["carry"]["available"] is False
    assert reports["carry"]["reason"] == "no_jaw"
    for skill in ("theremin", "ride", "dance"):
        assert reports[skill]["available"] is False
        assert reports[skill]["reason"] == "no_voice_helper"


def test_theremin_requires_an_actual_tof_frame_without_gating_other_sound_skills():
    missing_frame = _serializer_lanes(tof_ready=False)
    assert missing_frame["theremin"]["available"] is False
    assert missing_frame["theremin"]["reason"] == "no_sensor"
    assert missing_frame["theremin"]["status"] == "missing"
    assert missing_frame["ride"]["available"] is True
    assert missing_frame["dance"]["available"] is True

    unknown_frame = _serializer_lanes(tof_ready=None)
    assert unknown_frame["theremin"]["available"] is False
    assert unknown_frame["theremin"]["reason"] == "sensor_not_ready"

    ready_frame = _serializer_lanes(tof_ready=True)
    assert ready_frame["theremin"]["available"] is True


def test_selected_behaviour_and_one_shot_dispatch_phase_do_not_claim_success():
    reports = _serializer_lanes(behaviour="play_ball", ball_phase="idle")
    assert reports["ball"]["status"] == "waiting"
    assert reports["ball"]["result"] is None
    active = _serializer_lanes(behaviour="play_ball", ball_phase="kick")
    assert active["ball"]["status"] == "active"
    assert active["ball"]["result"] is None


def test_reflex_interrupts_selected_non_ride_skill_but_ride_owner_is_not_interrupted():
    reports = _serializer_lanes(behaviour="play_ball", ball_phase="kick", reflex="withdraw")
    assert reports["ball"]["status"] == "interrupted"
    ride = _serializer_lanes(
        behaviour="held",
        reflex="lifted",
        ride={"riding": False, "lifted": True, "after": 0, "outcome": "fear"},
    )
    assert ride["ride"]["status"] != "interrupted"
    assert ride["ride"]["result"] == "fear"


def test_driver_source_snapshot_forwards_live_skills_and_actions():
    from duckmind.simweb.neuro_api import DriverSource

    expected = _serializer_lanes(jaw=False, helper=False)

    class Driver:
        def telemetry(self, since=0):
            return {
                "seq": 1,
                "state": "running",
                "lanes": {"brain": "real", "senses": "real", "motor": "real"},
                "stats": {},
                "brain": {"behaviour": "play_ball", "t_ms": 1234},
                "command": {"behaviour": "play_ball"},
                "frame": {},
                "skills": expected,
                "actions": [{"skill": "kick_left", "status": "refused", "reason": "battery"}],
            }

    source = DriverSource(lambda: Driver(), clock=lambda: 10.0)
    snapshot = source.snapshot()
    assert snapshot["skills"] == expected
    assert snapshot["actions"][0]["status"] == "refused"


def test_one_shot_action_states_map_to_skill_without_claiming_command_success():
    base = {"behaviour": "play_ball", "skill": "kick_left", "seq": 3, "reason": None}
    waiting = _serializer_lanes(
        behaviour="play_ball",
        play_skill="kick_left",
        play_seq=3,
        actions=[{**base, "status": "dispatched"}],
    )
    assert waiting["ball"]["status"] == "waiting" and waiting["ball"]["result"] is None
    active = _serializer_lanes(
        behaviour="play_ball",
        play_skill="kick_left",
        play_seq=3,
        actions=[{**base, "status": "policy_started"}],
    )
    assert active["ball"]["status"] == "active" and active["ball"]["result"] is None
    refused = _serializer_lanes(
        behaviour="play_ball",
        play_skill="kick_left",
        play_seq=3,
        actions=[{**base, "status": "refused", "reason": "battery"}],
    )
    assert refused["ball"]["status"] == "refused" and refused["ball"]["reason"] == "battery"
    interrupted = _serializer_lanes(
        behaviour="carry_toy",
        play_skill="ground_pick",
        play_seq=2,
        jaw=True,
        actions=[
            {
                "behaviour": "carry_toy",
                "skill": "ground_pick",
                "seq": 2,
                "status": "interrupted_waiting_for_exit",
                "reason": "active_robotd_skill_not_cancelable_by_stop",
            }
        ],
    )
    assert (
        interrupted["carry"]["status"] == "interrupted" and interrupted["carry"]["result"] is None
    )


def test_real_serializer_marks_only_vision_dependent_skills_missing_on_world_camera_errors():
    reports = _serializer_lanes(
        jaw=True,
        helper=True,
        world_errors=["sample: ValueError"],
    )
    for skill in ("follow", "hide", "circle", "ball", "carry", "sun"):
        assert reports[skill]["status"] == "missing"
        assert reports[skill]["reason"] == "world_sensor_error"
    for skill in ("theremin", "ride", "dance"):
        assert reports[skill]["status"] != "missing"
        assert reports[skill]["reason"] != "world_sensor_error"
