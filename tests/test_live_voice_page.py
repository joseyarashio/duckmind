"""Live voice, LV5: the talk panel's server side (``duckmind.simweb.live_voice`` and ``live_voice_api``).

The cloud switch is off at start and a switch moves what leaves the machine at once; the status says
what the cloud heard, what it proposed and what the duck made of each proposal; the duck-eye pictures
go up at most once a second and only while the cloud is on; a duck switch makes a new connection for
the new duck and keeps the switch and the usage meter; the routes answer through the real HTTP
server; every word the panel can show is in all six languages. No network (a fake Gemini on a manual
clock), no physics: the duck is a stand-in with the few attributes the panel reads. The same chain
with a real integrated brain is ``tests/live_page_isolated/case_page_chain.py``."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.duckcore.api import Config
from duckmind.i18n import CODES, msg, render
from duckmind.live import session as S
from duckmind.live.bridge import LiveBridge
from duckmind.live.fake import FakeGemini, SpokenLine
from duckmind.live.meter import UsageCaps
from duckmind.live.scenario import ManualClock
from duckmind.live.session import LiveSession, QueuedCall
from duckmind.simweb import live_voice as LV
from duckmind.simweb.live_voice_api import LiveHub
from duckmind.simweb.neuro_mic import MicStream
from duckmind.simweb.server import SimServer, SimState, make_handler

pytestmark = pytest.mark.usefixtures("loopback_only")

JPEG = b"\xff\xd8" + b"duck-eye" * 40 + b"\xff\xd9"


def _loud(n: int = 320) -> bytes:
    import math
    from array import array

    amp = 32768.0 * 10.0 ** (-30.0 / 20.0) * math.sqrt(2.0)
    return array("h", [int(amp * math.sin(0.2 * i)) for i in range(n)]).tobytes()


class _Brain:
    """What the panel reads of the neural brain (no skills, no physics)."""

    def __init__(self) -> None:
        self.behaviour = "wander"
        self.drives = SimpleNamespace(
            valence=0.2,
            fear=0.1,
            sleepiness=0.0,
            social=0.4,
            as_dict=lambda: {"social": 0.4, "curiosity": 0.6, "fear": 0.1},
        )
        self.genome = SimpleNamespace(
            summary=lambda: {"stage_zh": "幼鴨", "primary": "snuggler", "axes": {"boldness": 0.2}},
            axes={"boldness": 0.2, "sociability": 0.8},  # (what LV4's drives read of a genome)
            p=SimpleNamespace(vocal_rate=0.5),
        )
        self.people: dict[int, Any] = {}
        self.learn = SimpleNamespace(person_names=self._names, people=self.people)
        self.live: Any = None

    def _names(self) -> dict[int, str]:
        return {pid: p.name for pid, p in self.people.items() if p.name}


class _Eye:
    """The physics bridge's ``eye_image``: a new picture on every call."""

    def __init__(self) -> None:
        self.calls = 0

    def eye_image(self) -> tuple[bytes | None, str, str]:
        self.calls += 1
        return JPEG[:-2] + bytes([self.calls % 250]) + JPEG[-2:], f'"p{self.calls}"', ""


class Rig:
    """A duck, its live voice and a fake Gemini, on a clock the test moves."""

    def __init__(
        self, *, fake: FakeGemini | None = None, eye: bool = True, caps: Any = None
    ) -> None:
        self.clock = ManualClock()
        self.fake = fake or FakeGemini(duck_name="蛋黃")
        self.mic = MicStream(seed=7, wall=lambda: 0.0)
        self.brain = _Brain()
        self.eye = _Eye() if eye else None
        self.injected: list[tuple[str, dict[str, Any]]] = []
        self.truth: dict[str, Any] | None = {"trunk": (1.0, 2.0, 0.1), "quat": (1.0, 0.0, 0.0, 0.0)}
        self.sessions: list[LiveSession] = []
        self.driver = SimpleNamespace(
            lanes=SimpleNamespace(
                brain=self.brain, sound=SimpleNamespace(mic=self.mic, engine=None)
            ),
            inject=self._inject,
            _truth=self.truth,
        )
        self.voice = LV.LiveVoice(
            self.driver,
            eye_source=self.eye,
            name_of=lambda: "蛋黃",
            session_factory=self._factory,
            clock=self.clock,
            wall=lambda: 1_000.0 + self.clock.t,
            caps=caps,
        )

    def _factory(self, soul: Any, *, meter: Any, enabled: bool) -> LiveSession:
        session = LiveSession(self.fake, soul, clock=self.clock, meter=meter, enabled=enabled)
        self.sessions.append(session)
        return session

    def _inject(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.injected.append((name, dict(params)))
        return {
            "where_mm": params.get("where_mm") or [10, 20],
            "note_zh": msg("neuro.world.say.default"),
        }

    def run(self, seconds: float, *, loud: bool = False) -> None:
        """The duck's microphone for ``seconds`` (20 ms frames); ``loud``: a person speaking."""
        end = self.clock.t + seconds
        while self.clock.t < end - 1e-9:
            self.clock.set(round(self.clock.t + 0.02, 6))
            if loud:
                self.voice.session.feed_pcm(_loud(), echo=False)  # type: ignore[union-attr]
            self.mic.tick(self.clock.t, [], wall=self.clock.t)
            bridge = self.voice.bridge
            if (
                bridge is not None
            ):  # (the brain's own step reaches the bridge: here, its push of the state)
                bridge._push_context(self.brain, bridge.clock())

    @property
    def sent(self) -> Any:
        return self.fake.sent


@pytest.fixture()
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Like ``no_network`` (nothing reaches outside this machine, no Keychain), but the test's own
    HTTP server on 127.0.0.1 may be talked to."""
    import socket
    import subprocess

    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"this live-voice test tried to reach {host!r}")
        return real_connect(self, address)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this live-voice test tried to reach the network")

    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"this live-voice test tried to look up {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    real_run = subprocess.run

    def guarded_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if cmd and "security" in str(cmd[0] if isinstance(cmd, (list, tuple)) else cmd):
            raise AssertionError("this live-voice test tried to read the Keychain")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


@pytest.fixture()
def rig() -> Rig:
    return Rig()


# ── the cloud switch ─────────────────────────────────────────────────────────────────


def test_the_cloud_is_off_at_start_and_nothing_is_sent(rig: Rig) -> None:
    status = rig.voice.status()
    assert status["ok"] and status["enabled"] and status["ready"]
    cloud = status["cloud"]
    assert cloud["on"] is False and cloud["state"] == "off" and cloud["connected"] is False
    assert cloud["sending"] == {"audio": False, "video": False, "text": False}
    assert "雲端關著" in cloud["words"]
    rig.run(3.0)
    assert (rig.sent.opens, rig.sent.audio_frames, rig.sent.images, rig.sent.texts) == (
        [],
        0,
        0,
        [],
    )
    assert rig.eye is not None and rig.eye.calls == 0, (
        "the eye is not even asked while the cloud is off"
    )
    assert rig.voice.status()["cloud"]["sending"]["video"] is False


def test_turning_the_cloud_on_connects_and_then_sound_and_picture_go_up(rig: Rig) -> None:
    answer = rig.voice.set_cloud(True)
    assert (
        answer["ok"] and answer["cloud"]["on"] is True and answer["cloud"]["state"] == "connecting"
    )
    assert answer["cloud"]["sending"]["audio"] is False, (
        "switched on, not connected: nothing has left yet"
    )
    rig.run(1.0)
    cloud = rig.voice.status()["cloud"]
    assert cloud["state"] == "online" and cloud["connected"] and cloud["understands_speech"]
    assert cloud["sending"] == {"audio": True, "video": True, "text": True}
    assert rig.sent.audio_frames > 20 and len(rig.sent.opens) == 1
    rig.run(3.0)
    assert rig.sent.images >= 2, "the duck's eye goes up beside the microphone"
    assert rig.sent.texts, "and the duck's status text"
    assert rig.voice.status()["usage"]["totals"]["frames"] == rig.sent.images


def test_the_eye_pictures_go_at_most_once_a_second_each_picture_once(rig: Rig) -> None:
    rig.voice.set_cloud(True)
    rig.run(1.0)
    before = rig.sent.images
    rig.run(10.0)
    sent = rig.sent.images - before
    assert 8 <= sent <= 10, sent  # (one a second, never faster)
    assert rig.voice.eye_frame() is not None and rig.voice.eye_frame().jpeg.startswith(b"\xff\xd8")


def test_turning_it_off_stops_everything_at_once(rig: Rig) -> None:
    rig.voice.set_cloud(True)
    rig.run(3.0)
    frozen = (rig.sent.audio_frames, rig.sent.images, len(rig.sent.texts))
    assert frozen[0] > 0
    answer = rig.voice.set_cloud(False)
    assert answer["cloud"]["on"] is False and answer["cloud"]["state"] == "off"
    assert answer["cloud"]["sending"] == {"audio": False, "video": False, "text": False}
    eye_calls = rig.eye.calls  # type: ignore[union-attr]
    rig.run(5.0, loud=True)
    assert (rig.sent.audio_frames, rig.sent.images, len(rig.sent.texts)) == frozen
    assert rig.eye.calls == eye_calls, "the eye is not asked for pictures any more"  # type: ignore[union-attr]
    assert rig.sent.closes >= 1 and rig.sent.audio_ends >= 1
    assert "雲端關著" in rig.voice.status()["cloud"]["words"]


def test_the_switch_is_remembered_across_a_new_session(rig: Rig) -> None:
    rig.voice.set_cloud(True)
    rig.run(2.0)
    first = rig.voice.session
    rig.voice.on_switch()  # the family picked another duck
    second = rig.voice.session
    assert second is not first and second.enabled, "the new duck's session has the cloud on as well"
    assert first.enabled is False and first.state == S.OFF, "the old connection is closed"
    assert second.meter is first.meter, "the usage is the account's, not the duck's"
    rig.voice.set_cloud(False)
    rig.voice.on_switch()
    assert rig.voice.session.enabled is False


def test_a_duck_without_a_microphone_says_so_and_refuses_the_switch() -> None:
    clock = ManualClock()
    driver = SimpleNamespace(
        lanes=SimpleNamespace(brain=_Brain(), sound=SimpleNamespace(mic=None, engine=None)),
        inject=lambda *a: {},
        _truth=None,
    )
    voice = LV.LiveVoice(
        driver,
        name_of=lambda: "蛋黃",
        clock=clock,
        session_factory=lambda soul, meter, enabled: LiveSession(
            FakeGemini(), soul, clock=clock, meter=meter, enabled=enabled
        ),
    )
    status = voice.status()
    assert status["ready"] is False and "麥克風" in status["error"]
    assert status["cloud"]["sending"]["audio"] is False
    with pytest.raises(ValueError, match="麥克風"):
        voice.set_cloud(True)


def test_a_usage_cap_turns_the_cloud_off_by_itself_and_the_status_says_so() -> None:
    rig = Rig(caps=UsageCaps(audio_s=2.0))
    rig.voice.set_cloud(True)
    rig.run(6.0)
    cloud = rig.voice.status()
    assert cloud["cloud"]["state"] == "capped" and cloud["usage"]["over_cap"] == "audio_s"
    assert cloud["cloud"]["sending"]["audio"] is False
    assert "上限" in cloud["cloud"]["words"]
    assert cloud["usage"]["caps"]["audio_s"] == 2.0
    stopped = rig.sent.audio_frames
    rig.run(2.0)
    assert rig.sent.audio_frames == stopped


@pytest.mark.parametrize("outcome,state", [("quota", "quota"), ("network", "reconnecting")])
def test_a_cloud_that_cannot_be_reached_is_said_and_the_duck_lives_on(
    outcome: str, state: str
) -> None:
    rig = Rig(fake=FakeGemini(duck_name="蛋黃", outcomes=(outcome,)))
    rig.voice.set_cloud(True)
    rig.run(1.0)
    cloud = rig.voice.status()["cloud"]
    assert cloud["state"] == state and cloud["connected"] is False
    assert cloud["sending"] == {"audio": False, "video": False, "text": False}
    assert cloud["error_kind"] == outcome
    assert cloud["retry_in_s"] is not None and cloud["retry_in_s"] > 0
    assert rig.voice.faults == 0 and rig.mic.view()["frames"] > 40, (
        "the microphone and the duck go on"
    )


# ── what the cloud heard and what it proposed ─────────────────────────────────────────


def test_what_the_cloud_heard_comes_as_data_newest_first_and_joined(rig: Rig) -> None:
    rig.fake.__dict__["_lines"] = []
    rig.fake.lines = (
        SpokenLine(2.0, 3.0, "蛋黃，過來這裡。", "mom"),
        SpokenLine(8.0, 8.8, "你好！", "mom"),
    )
    rig.fake.__post_init__()
    rig.voice.set_cloud(True)
    for _ in range(0, 700):
        rig.clock.set(round(rig.clock.t + 0.02, 6))
        loud = 1.9 <= rig.clock.t <= 3.1 or 7.9 <= rig.clock.t <= 8.9
        rig.voice.session.feed_pcm(_loud() if loud else b"\x00\x00" * 320, echo=False)  # type: ignore[union-attr]
        rig.mic.tick(rig.clock.t, [], wall=rig.clock.t)
    heard = rig.voice.status()["heard"]
    assert [h["text"] for h in heard] == ["你好！", "蛋黃，過來這裡。"], (
        "two sentences, newest first"
    )
    assert all(h["echo"] is False and h["age_s"] >= 0 for h in heard)
    assert heard[0]["age_s"] < heard[1]["age_s"]


def _call(
    seq: int, name: str, args: dict[str, Any], *, t: float = 5.0, echo: bool = False
) -> QueuedCall:
    from duckmind.live.echo import CallMark
    from duckmind.live.tools import parse_call

    parsed = parse_call(name, args)
    mark = CallMark(echo, 1.0, 0.0, False, False)
    return QueuedCall(
        seq=seq,
        id=f"c{seq}",
        name=parsed.name,
        args=parsed.args,
        raw=parsed.raw,
        problems=parsed.problems,
        off_bank=parsed.off_bank,
        t=t,
        wall=0.0,
        connection=1,
        mark=mark,
        echo_suspect=echo,
    )


def _book(bridge: LiveBridge, call: QueuedCall, *records: dict[str, Any]) -> None:
    bridge.seen.append(call)
    for record in records:
        bridge.log.append({"t": 5.0, "seq": call.seq, "tool": call.name, **record})


def test_every_call_shows_what_became_of_it_with_its_reason(rig: Rig) -> None:
    bridge = rig.voice.bridge
    assert isinstance(bridge, LiveBridge)
    rig.clock.set(10.0)
    # a call it made: accepted, then voiced
    _book(
        bridge,
        _call(1, "vocalize", {"tags": [{"tag": "greet", "level": 0.5}]}),
        {"decision": "accepted", "stage": "bridge", "reason": None, "intent": 1},
        {"decision": "done", "stage": "selector", "reason": "voiced", "intent": 1},
    )
    bridge.voiced.append({"t": 5.5, "seq": 1, "intent": 1, "tag": "greet"})
    # a call the gate refused (a made-up call), one the duck had no mind for, one that lapsed
    _book(
        bridge,
        _call(2, "vocalize", {"tags": [{"tag": "meow", "level": 0.9}]}),
        {"decision": "refused", "stage": "gate", "reason": "tag_unknown"},
    )
    _book(
        bridge,
        _call(3, "attend", {"target_kind": "thing", "hint": "紅色的球"}),
        {"decision": "accepted", "stage": "bridge", "reason": None, "intent": 2},
        {"decision": "expired", "stage": "selector", "reason": "lost_competition", "intent": 2},
    )
    _book(
        bridge,
        _call(4, "recognize", {"name": "媽媽"}),
        {"decision": "accepted", "stage": "bridge", "reason": None},
    )
    _book(
        bridge,
        _call(5, "recognize", {"name": "蛋黃"}),
        {"decision": "refused", "stage": "bridge", "reason": "duck_name"},
    )
    _book(
        bridge,
        _call(6, "feel", {"valence": 0.3, "talk": 0.4}),
        {"decision": "accepted", "stage": "bridge", "reason": None},
    )
    _book(
        bridge,
        _call(7, "vocalize", {"tags": [{"tag": "coo", "level": 0.4}]}),
        {"decision": "accepted", "stage": "bridge", "reason": None, "intent": 3},
    )
    _book(
        bridge,
        _call(8, "vocalize", {"tags": [{"tag": "greet", "level": 0.8}]}, echo=True),
        {"decision": "refused", "stage": "gate", "reason": "echo_only"},
    )
    calls = rig.voice.status()["calls"]
    by_seq = {c["seq"]: c for c in calls}
    assert [c["seq"] for c in calls] == [8, 7, 6, 5, 4, 3, 2, 1], "newest first"
    assert (by_seq[1]["outcome"], by_seq[1]["tags"][0]["voiced"]) == ("voiced", True)
    assert by_seq[1]["tags"][0]["tag"] == "greet" and "招呼" in by_seq[1]["tags"][0]["meaning"]
    assert (by_seq[2]["outcome"], by_seq[2]["reason"]) == ("refused", "tag_unknown")
    assert by_seq[2]["tags"][0]["meaning"] is None, "a call the bank does not have has no meaning"
    assert (by_seq[3]["outcome"], by_seq[3]["reason"]) == ("lapsed", "lost_competition")
    assert by_seq[3]["hint"] == "紅色的球"
    assert by_seq[4]["outcome"] == "bound" and by_seq[4]["name"] == "媽媽"
    assert (by_seq[5]["outcome"], by_seq[5]["reason"]) == ("refused", "duck_name")
    assert by_seq[6]["outcome"] == "felt" and by_seq[6]["valence"] == 0.3
    assert by_seq[7]["outcome"] == "waiting"
    assert (by_seq[8]["outcome"], by_seq[8]["reason"], by_seq[8]["echo_suspect"]) == (
        "refused",
        "echo_only",
        True,
    )
    assert all(c["age_s"] >= 0 for c in calls)
    attended = rig.voice.status()["look"]["attended"]
    assert attended is not None and attended["seq"] == 3, (
        "the latest attend call is what it was asked to look at"
    )


def test_the_people_it_knows_are_the_ducks_own_records(rig: Rig) -> None:
    from duckmind.neuro.learn import PersonRecord

    mom = PersonRecord(pid=1, name="媽媽", greeted=3, sightings=2, pending={"阿姨": 1})
    rig.brain.people[1] = mom
    rig.brain.people[2] = PersonRecord(pid=2)
    people = rig.voice.status()["people"]
    assert people == {
        "named": [{"pid": 1, "name": "媽媽", "greeted": 3, "sightings": 2, "pending": ["阿姨"]}],
        "nameless": 1,
    }


def test_the_status_text_the_model_gets_comes_from_the_ducks_own_state(rig: Rig) -> None:
    from duckmind.neuro.learn import PersonRecord

    rig.brain.people[1] = PersonRecord(pid=1, name="媽媽")
    rig.voice.set_cloud(True)
    rig.run(2.0)
    text = rig.sent.texts[0][1]
    assert "媽媽" in text and "心情" in text and "不需要回應" in text
    soul = rig.sent.opens[0]
    assert soul["instruction_chars"] > 300
    session = rig.voice.session
    assert session is not None and "蛋黃" in session._soul_text and "幼鴨" in session._soul_text  # type: ignore[operator]


def test_the_bridge_tells_the_cloud_the_ducks_state_every_five_seconds_with_lv4s_drives(
    rig: Rig,
) -> None:
    """LV7 gap: the page used to build a snapshot of its own and leave ``context_every_s`` unset, so
    the wish to talk and the closeness to the person it sees never went up. Now the bridge pushes its
    own (``LiveBridge.snapshot``) every 5 s, as ``attach_live`` does, in the family's words."""
    from duckmind.live.wiring import CONTEXT_EVERY_S
    from duckmind.neuro.learn import PersonRecord

    bridge = rig.voice.bridge
    assert bridge is not None and CONTEXT_EVERY_S == 5.0
    assert bridge.config.context_every_s == CONTEXT_EVERY_S
    rig.brain.people[1] = PersonRecord(pid=1, name="媽媽")
    rig.voice.set_cloud(True)
    rig.run(2.0)
    text = rig.sent.texts[0][1]
    assert "想說話" in text and "媽媽（親近 " in text, (
        text
    )  # (LV4's: the wish to talk, the closeness)
    from duckmind.neuro.contract import behaviour_zh

    assert behaviour_zh("wander") != "wander" and f"正在做：{behaviour_zh('wander')}" in text, text
    snap = bridge.snapshot(rig.brain)
    assert snap.bonds == {"媽媽": 0.0} and "talk" in snap.drives


def test_the_transcriber_is_told_the_ducks_name_and_the_names_the_family_gave(rig: Rig) -> None:
    """The duck's name (``--live-voice-name``'s, here ``name_of``) and the learner's known family
    names go to the cloud's transcriber as its ``custom_vocabulary``; a name given later is in the
    next connection."""
    from duckmind.neuro.learn import PersonRecord

    session = rig.voice.session
    assert session is not None and session._vocabulary == ("蛋黃",)
    rig.voice.set_cloud(True)
    rig.run(1.0)
    assert rig.sent.opens[0]["vocabulary"] == ["蛋黃"]
    rig.brain.people[1] = PersonRecord(pid=1, name="媽媽")
    rig.run(3.0)  # (re-read every 2 s)
    assert session._vocabulary == ("蛋黃", "媽媽")
    rig.voice.on_switch()  # a new connection (the family picked another duck): opens with both
    rig.run(1.0)
    assert rig.sent.opens[-1]["vocabulary"] == ["蛋黃", "媽媽"]


# ── the scripted lines and the places ─────────────────────────────────────────────────


def test_the_places_are_round_the_duck_where_it_stands_now(rig: Rig) -> None:
    places = {p["id"]: p for p in rig.voice.places()}
    assert set(places) == set(LV.PLACES) and places["auto"] == {"id": "auto"}
    assert (places["front"]["x_mm"], places["front"]["y_mm"]) == (2000, 2000), (
        "1 m ahead along its heading"
    )
    assert (places["behind"]["x_mm"], places["behind"]["y_mm"]) == (0, 2000)
    assert places["left"]["y_mm"] == 3000 and places["right"]["y_mm"] == 1000
    assert places["far"]["x_mm"] == 4000
    rig.driver._truth = None
    assert rig.voice.places() == [{"id": "auto"}], "no pose, no places to offer but the world's"


def test_a_scripted_line_is_the_worlds_say_stimulus_with_the_ducks_name(rig: Rig) -> None:
    answer = rig.voice.say({"utterance": "call_name", "position": "front"})
    assert (
        answer["ok"] and answer["line"]["text"] == "蛋黃，過來這裡。" and answer["who"] == "partner"
    )
    ((name, request),) = rig.injected
    assert name == "say"
    assert request == {
        "utterance": "call_name",
        "duck_name": "蛋黃",
        "who": "partner",
        "where_mm": [2000, 2000],
    }
    rig.voice.say({"utterance": "hello", "who": "child"})
    assert rig.injected[1][1]["who"] == "child" and "where_mm" not in rig.injected[1][1]


def test_a_line_can_be_said_softly_or_loudly_within_what_a_person_can(rig: Rig) -> None:
    rig.voice.say({"utterance": "hello_kid", "db1m": 54})
    assert rig.injected[0][1]["db1m"] == 54.0
    for bad in (10, 140, "loud", float("nan")):
        with pytest.raises(ValueError) as caught:
            rig.voice.say({"utterance": "hello_kid", "db1m": bad})
        assert "分貝" in str(caught.value.args[0])
    assert len(rig.injected) == 1


def test_the_status_says_what_the_duck_is_weighing(rig: Rig) -> None:
    decision = rig.voice.status()["decision"]
    assert (
        decision["behaviour"] == "wander" and decision["intents"] == [] and decision["extras"] == {}
    )
    assert decision["barges"] == [] and "events" in decision["vad"]
    assert (decision["fear"], decision["sleepiness"], decision["social"]) == (0.1, 0.0, 0.4)
    assert decision["ear"] == {"name_events": 0, "hints_sent": 0, "skipped": {}, "last": []}


def test_the_status_says_what_the_ducks_own_ear_told_the_cloud(rig: Rig) -> None:
    """The duck's ear reports its name: one short text goes up (not the status), the decision block
    counts it, and with the cloud off the same event tells nothing."""
    from duckmind.live.soul import EAR_NAME_HINT

    bridge = rig.voice.bridge
    assert bridge is not None
    ear = SimpleNamespace(sound=SimpleNamespace(kind="name", level_db=61.5))
    silent = SimpleNamespace(sound=SimpleNamespace(kind=None, level_db=30.0))
    bridge._ear(ear, rig.clock.t)  # (cloud off: nothing is sent, and it says why)
    bridge._ear(silent, rig.clock.t)
    assert rig.sent.texts == []
    off = rig.voice.status()["decision"]["ear"]
    assert (off["name_events"], off["hints_sent"], off["skipped"]) == (1, 0, {"cloud_off": 1})
    rig.voice.set_cloud(True)
    rig.run(2.0)
    texts_before = len(rig.sent.texts)  # (the status text)
    bridge._ear(ear, rig.clock.t)
    hints = [t for t in rig.sent.texts if t[1] == EAR_NAME_HINT]
    assert len(hints) == 1 and hints[0][2] is False and len(rig.sent.texts) == texts_before + 1
    on = rig.voice.status()["decision"]["ear"]
    assert (on["name_events"], on["hints_sent"]) == (2, 1)
    assert on["last"][-1] == {"t": rig.clock.t, "sent": True, "why": None, "level_db": 61.5}
    json.dumps(rig.voice.status(), ensure_ascii=False)


@pytest.mark.parametrize(
    "params,words",
    [
        ({"utterance": "no_such_line"}, "no_such_line"),
        ({"utterance": "hello", "who": "stranger"}, "家人"),
        ({"utterance": "hello", "position": "ceiling"}, "站的位置"),
    ],
)
def test_a_scripted_line_that_cannot_be_said_is_refused_in_words(
    rig: Rig, params: dict, words: str
) -> None:
    with pytest.raises(ValueError) as caught:
        rig.voice.say(params)
    assert words in str(caught.value.args[0]) and rig.injected == []


def test_a_place_that_is_not_known_yet_is_refused(rig: Rig) -> None:
    rig.driver._truth = None
    with pytest.raises(ValueError, match="小鴨在哪裡"):
        rig.voice.say({"utterance": "hello", "position": "front"})


# ── the routes, through the real HTTP server ──────────────────────────────────────────


@pytest.fixture()
def served(rig: Rig):
    hub = LiveHub()
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, None, None, hub))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(base=f"http://127.0.0.1:{httpd.server_address[1]}", hub=hub, rig=rig)
    httpd.shutdown()
    httpd.server_close()


def _get(base: str, path: str) -> tuple[int, dict[str, Any] | bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as response:
            body = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, json.loads(body) if "json" in kind else body
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def _post(
    base: str, path: str, body: Any, *, raw: bytes | None = None
) -> tuple[int, dict[str, Any]]:
    data = raw if raw is not None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def test_a_server_without_live_voice_says_so_and_the_page_stays_away(served) -> None:
    status, body = _get(served.base, "/api/live/status")
    assert status == 200 and body == {"ok": True, "enabled": False}
    status, body = _post(served.base, "/api/live/cloud", {"on": True})
    assert status == 200 and body["ok"] is False and "live-voice" in body["error"]
    status, body = _post(served.base, "/api/live/say", {"utterance": "hello"})
    assert status == 200 and body["ok"] is False
    assert _get(served.base, "/api/live/eye.jpg")[0] == 204


def test_the_status_route_carries_the_words_in_six_languages(served) -> None:
    served.hub.attach(served.rig.voice)
    status, body = _get(served.base, "/api/live/status")
    assert status == 200 and body["enabled"] and body["cloud"]["on"] is False
    wire = body["cloud"]["_i18n"]["words"]
    shown = {lang: render(wire, lang, strict=True) for lang in CODES}
    assert len(set(shown.values())) == len(CODES), shown  # (a whole sentence: six different texts)
    assert (
        "cloud" in shown["en"].lower() and "云端" in shown["zh-Hans"] and "クラウド" in shown["ja"]
    )
    assert body["personas"] == ["owner", "partner", "child", "elder", "guest"]
    assert any(line["id"] == "call_name" for line in body["lines"])
    assert body["places"][0] == {"id": "auto"}


def test_the_switch_over_http_and_what_leaves(served) -> None:
    served.hub.attach(served.rig.voice)
    rig = served.rig
    status, body = _post(served.base, "/api/live/cloud", {"on": True})
    assert (
        status == 200
        and body["ok"]
        and body["cloud"]["on"]
        and body["cloud"]["state"] == "connecting"
    )
    rig.run(2.0)
    _, body = _get(served.base, "/api/live/status")
    assert (
        body["cloud"]["connected"]
        and body["cloud"]["sending"]["audio"]
        and body["cloud"]["sending"]["video"]
    )
    assert body["look"]["frame"]["url"].startswith("/api/live/eye.jpg?v=")
    assert body["look"]["frames_sent"] >= 1
    status, jpeg = _get(served.base, "/api/live/eye.jpg")
    assert (
        status == 200
        and isinstance(jpeg, bytes)
        and jpeg[:2] == b"\xff\xd8"
        and jpeg[-2:] == b"\xff\xd9"
    )
    status, body = _post(served.base, "/api/live/cloud", {"on": False})
    assert body["cloud"]["on"] is False and body["cloud"]["sending"]["audio"] is False
    n = rig.sent.audio_frames
    rig.run(2.0)
    assert rig.sent.audio_frames == n, "nothing is sent after the switch is off"


def test_a_scripted_line_over_http_and_a_refusal_over_http(served) -> None:
    served.hub.attach(served.rig.voice)
    status, body = _post(served.base, "/api/live/say", {"utterance": "hello", "position": "left"})
    assert status == 200 and body["ok"] and body["line"]["text"] == "你好！"
    assert served.rig.injected[0][1]["where_mm"] == [1000, 3000]
    status, body = _post(served.base, "/api/live/say", {"utterance": "nope"})
    assert status == 200 and body["ok"] is False and "nope" in body["error"]
    assert body["_i18n"]["error"]["k"] == "neuro.world_refusal.say_unknown"


@pytest.mark.parametrize(
    "path,raw,expect",
    [
        ("/api/live/cloud", b"not json", 400),
        ("/api/live/cloud", b'["on"]', 400),
        ("/api/live/cloud", b'{"on": "yes"}', 400),
        ("/api/live/cloud", b"{}", 400),
        ("/api/live/nothing", b"{}", 404),
    ],
)
def test_a_malformed_request_is_refused_with_the_right_status(
    served, path: str, raw: bytes, expect: int
) -> None:
    served.hub.attach(served.rig.voice)
    status, body = _post(served.base, path, None, raw=raw)
    assert status == expect and body["ok"] is False
    assert served.rig.voice.session.enabled is False, "a bad request never moves the switch"  # type: ignore[union-attr]


def test_a_request_without_a_length_or_too_big_is_refused(served) -> None:
    import http.client

    served.hub.attach(served.rig.voice)
    host, port = served.base.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.putrequest("POST", "/api/live/cloud")
    conn.endheaders()
    assert conn.getresponse().status == 411
    conn.close()
    status, _ = _post(served.base, "/api/live/say", None, raw=b"x" * 5000)
    assert status == 413
    status, body = _get(served.base, "/api/live/unknown")
    assert status == 404


# ── the server's flag ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--live-voice"],
        ["--live-voice", "--brain", "neuro"],
        ["--brain", "neuro", "--live-voice-cap-minutes", "0"],
    ],
)
def test_the_flag_needs_the_neural_duck_on_the_live_body(argv: list[str], capsys: Any) -> None:
    from duckmind.simweb import server

    with pytest.raises(SystemExit) as caught:
        server.main(argv)
    assert caught.value.code == 2
    assert "--live" in capsys.readouterr().err


# ── the words ────────────────────────────────────────────────────────────────────────


def test_every_word_the_panel_can_say_is_in_all_six_languages() -> None:
    keys = (
        [f"neuro.live.tool.{x}" for x in LV.TOOLS]
        + [f"neuro.live.tag.{x}" for x in LV.TAGS]
        + [f"neuro.live.attend.{x}" for x in LV.ATTEND_KINDS]
        + [f"neuro.live.outcome.{x}" for x in LV.OUTCOMES]
        + [f"neuro.live.reason.{x}" for x in LV.REASONS]
        + [f"neuro.live.status.{x}" for x in S.STATES]
        + [
            f"neuro.live.web.{x}"
            for x in (
                "not_ready",
                "no_mic",
                "not_enabled",
                "bad_request",
                "no_length",
                "too_big",
                "bad_who",
                "bad_position",
                "no_position",
                "bad_loudness",
            )
        ]
    )
    for key in keys:
        shown = {lang: render(msg(key, low=40, high=90), lang, strict=True) for lang in CODES}
        # (the Chinese scripts and Japanese may write a short word the same way: 叫, 人, 心情)
        assert len({shown["en"], shown["es"], shown["ko"], shown["zh-Hant"]}) == 4, (key, shown)
        assert all(shown[lang] for lang in CODES) and shown["ja"] != shown["en"], (key, shown)
        for lang in ("en", "es", "ko"):
            assert not re.search(r"[一-鿿]", shown[lang]), (key, lang, shown[lang])


def test_every_refusal_the_gate_and_the_bridge_can_give_has_words() -> None:
    from duckmind.live import bridge as B
    from duckmind.live import gate as G

    codes = set(G.REASONS) | {
        B.NO_REPLY_CHANNEL,
        B.NO_PERSON,
        B.SEVERAL_IN_VIEW,
        B.BARGE_IN,
        B.LOST,
    }
    assert codes - set(LV.REASONS) == set(), "a refusal the panel could not put in words"


def test_the_panels_codes_are_the_tools_and_calls_the_cloud_has() -> None:
    from duckmind.live.tools import BANK_TAGS, TARGET_KINDS, TOOL_NAMES

    assert set(LV.TOOLS) == set(TOOL_NAMES) and set(LV.TAGS) == set(BANK_TAGS)
    assert set(LV.ATTEND_KINDS) == set(TARGET_KINDS)


# ── what the duck's own voice did to the transcript ──────────────────────────────────


def test_a_transcript_that_overlaps_the_ducks_own_call_is_marked_as_a_possible_echo() -> None:
    rig = Rig(fake=FakeGemini(duck_name="蛋黃", lines=(SpokenLine(2.0, 2.6, "Mhm", "duck"),)))
    rig.voice.set_cloud(True)
    for _ in range(300):
        rig.clock.set(round(rig.clock.t + 0.02, 6))
        calling = 1.9 <= rig.clock.t <= 2.7
        rig.voice.session.feed_pcm(_loud() if calling else b"\x00\x00" * 320, echo=calling)  # type: ignore[union-attr]
        rig.mic.tick(rig.clock.t, [], wall=rig.clock.t)
    heard = rig.voice.status()["heard"]
    assert [h["text"] for h in heard] == ["Mhm"] and heard[0]["echo"] is True


def test_a_transcript_over_the_ducks_call_is_not_labelled_an_echo_when_the_duck_heard_a_person() -> (
    None
):
    """Live 2026-10-04: the owner's real words, spoken while the duck called, were labelled 「可能是小
    鴨自己的叫聲傳回來的回音」. The duck's own detector (the bridge's spans: the very evidence the
    grounding gate rescues a call with) heard a person: the transcript is theirs."""

    def run(person: bool) -> list[dict[str, Any]]:
        rig = Rig(fake=FakeGemini(duck_name="蛋黃", lines=(SpokenLine(2.0, 2.6, "你好", "mom"),)))
        rig.voice.set_cloud(True)
        for _ in range(300):
            rig.clock.set(round(rig.clock.t + 0.02, 6))
            calling = 1.9 <= rig.clock.t <= 2.7
            rig.voice.session.feed_pcm(_loud() if calling else b"\x00\x00" * 320, echo=calling)  # type: ignore[union-attr]
            rig.mic.tick(rig.clock.t, [], wall=rig.clock.t)
            if person and abs(rig.clock.t - 2.2) < 1e-6:
                assert rig.voice.bridge is not None
                rig.voice.bridge._spans.append([rig.clock.t, None])  # (a person began to speak)
        return list(rig.voice.status()["heard"])

    alone, together = run(False), run(True)
    assert [h["text"] for h in alone] == ["你好"] and alone[0]["echo"] is True
    assert [h["text"] for h in together] == ["你好"] and together[0]["echo"] is False


def test_what_the_cloud_heard_is_shown_in_traditional_characters_and_logged_as_it_came() -> None:
    """Live 2026-10-04: the transcriber answered in simplified characters (过来这里). The language
    hint makes it traditional; a sentence that still comes back simplified is converted for the page
    only, the session's log keeps the raw text."""
    raw = "赵华, 过来这里。"
    rig = Rig(fake=FakeGemini(duck_name="蛋黃", lines=(SpokenLine(2.0, 3.0, raw, "mom"),)))
    rig.voice.set_cloud(True)
    for _ in range(400):
        rig.clock.set(round(rig.clock.t + 0.02, 6))
        loud = 1.9 <= rig.clock.t <= 3.1
        rig.voice.session.feed_pcm(_loud() if loud else b"\x00\x00" * 320, echo=False)  # type: ignore[union-attr]
        rig.mic.tick(rig.clock.t, [], wall=rig.clock.t)
    assert [h["text"] for h in rig.voice.status()["heard"]] == ["趙華, 過來這裡。"]
    session = rig.voice.session
    assert session is not None
    assert "".join(e["text"] for e in session.log if e["kind"] == "heard") == raw


def test_the_language_hint_is_a_server_setting_that_reaches_the_connection() -> None:
    seen: list[tuple[str, ...]] = []

    class Spy(FakeGemini):
        def open(self, setup: Any, now: float) -> None:
            seen.append(tuple(setup.transcription_languages))
            super().open(setup, now)

    def open_with(languages: Any) -> None:
        fake = Spy(duck_name="蛋黃")
        clock = ManualClock()
        voice = LV.LiveVoice(
            SimpleNamespace(
                lanes=SimpleNamespace(
                    brain=_Brain(), sound=SimpleNamespace(mic=MicStream(seed=7), engine=None)
                ),
                inject=lambda *a: {},
                _truth=None,
            ),
            name_of=lambda: "蛋黃",
            clock=clock,
            transcription_languages=languages,
        )
        assert voice.session is not None
        # (the real transport is replaced by the fake: only the setup it is opened with matters)
        voice.session.transport = fake
        voice.set_cloud(True)
        for _ in range(50):
            clock.set(round(clock.t + 0.02, 6))
            voice.session.pump()
        voice.close()

    open_with(None)
    open_with(["ja-JP", "zh-TW"])
    open_with([])
    assert seen == [("zh-TW", "en-US"), ("ja-JP", "zh-TW"), ()]


# ── a duck switch from the page ──────────────────────────────────────────────────────


def test_the_page_switching_ducks_makes_the_live_voice_connect_again() -> None:
    from duckmind.simweb import neuro_api

    class Source:
        kind = "driver"

        def switch(self, choice: Any, card: Any) -> dict[str, Any]:
            return {}

    hub = neuro_api.NeuroHub(enabled=True, source=Source())  # type: ignore[arg-type]
    seen: list[str] = []
    hub.switch_hooks.append(lambda: seen.append("a"))
    hub.switch_hooks.append(lambda: (_ for _ in ()).throw(RuntimeError("a bad listener")))
    hub.switch_hooks.append(lambda: seen.append("c"))
    answer = hub.switch({"preset": "dreamer"})
    assert answer["ok"] and seen == ["a", "c"], (
        "every listener runs, a fault in one does not break the switch"
    )
    assert answer["hook_faults"] == ["RuntimeError: a bad listener"]


# ── the page's files ─────────────────────────────────────────────────────────────────


def _static(name: str) -> str:
    from importlib import resources

    return resources.files("duckmind.simweb").joinpath("static", name).read_text(encoding="utf-8")


def test_the_page_script_finds_every_element_it_uses() -> None:
    script, html = _static("live_voice.js"), _static("index.html")
    used = set(re.findall(r'\$\("(live[A-Za-z]+)"\)', script))
    assert {
        "liveVoice",
        "liveCloudSwitch",
        "liveMic",
        "liveWho",
        "livePlace",
        "livePush",
        "liveLine",
        "liveSay",
        "liveHeard",
        "liveCalls",
        "liveEye",
        "liveUsage",
        "liveChipCloud",
    } <= used
    missing = sorted(i for i in used if f'id="{i}"' not in html)
    assert missing == []
    assert len(re.findall(r'id="(live[A-Za-z]+)"', html)) == len(
        set(re.findall(r'id="(live[A-Za-z]+)"', html))
    )


def test_the_panel_stays_hidden_unless_the_server_runs_live_voice_and_is_a_scenario_card() -> None:
    html = _static("index.html")
    card = html[html.index('id="liveVoice"') - 80 : html.index('id="liveVoice"') + 200]
    assert "hidden" in card and "data-scenarios" in card
    assert "data-neuro" not in card, "neuro.js would show every data-neuro element in neural mode"
    chip = html[html.index('id="liveChipCloud"') - 60 : html.index('id="liveChipCloud"') + 120]
    assert "hidden" in chip and 'data-i18n="page.live.cloud.label_off"' in chip


def test_the_panels_style_uses_only_the_standards_tokens() -> None:
    css = _static("live_voice.css")
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", body), "a colour that is not a token"
    assert not re.search(r"\brgba?\(|\bhsla?\(|gradient\(", body), (
        "only the standard's tokens and flat fills"
    )
    assert not re.search(r"border-radius\s*:\s*(?!0\b)", body), "square corners"
    assert not re.search(r"blur\(|filter\s*:", body)
    for shadow in re.findall(r"box-shadow\s*:\s*([^;}]+)", body):
        assert re.fullmatch(r"-?\d+px -?\d+px 0 var\(--[a-z-]+\)", shadow.strip()), (
            shadow
        )  # hard, offset, no blur
    for colour in re.findall(r"var\((--[a-z0-9-]+)\)", body):
        assert colour in _static("style.css"), colour


def test_the_server_serves_the_panels_files(served) -> None:
    for name, kind in (
        ("live_voice.js", "text/javascript"),
        ("live_voice_worklet.js", "text/javascript"),
        ("live_voice.css", "text/css"),
    ):
        with urllib.request.urlopen(served.base + "/" + name, timeout=10) as response:
            assert response.status == 200 and response.headers["Content-Type"].startswith(kind)
            assert response.read().decode("utf-8") == _static(name)


def test_the_page_scripts_are_valid_javascript() -> None:
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("no node here")
    for name in ("live_voice.js", "live_voice_worklet.js"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / name
            path.write_text(_static(name), encoding="utf-8")
            done = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True, check=False
            )
        assert done.returncode == 0, (name, done.stderr[-800:])
