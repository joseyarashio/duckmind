"""``POST /api/neuro/command``: the words become one command (neuro/commands.py) and reach the duck's brain as what it
heard; the answer says queued or refused and why; a request that cannot be a command is a 400 with the reason
in the six languages' catalog; the stub and a server with no neural duck say so; the page's telemetry carries
the last command and its outcome."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from live_bridge_rig import person_sight
from test_neuro_live import robot_state

from duckmind.duckcore.api import Config
from duckmind.i18n import render
from duckmind.neuro import command_brain as CB
from duckmind.neuro import contract as C
from duckmind.neuro.commands import Command, CommandError
from duckmind.neuro.genome import Genome
from duckmind.neuro.integration import make_brain
from duckmind.simweb import neuro_api as N
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimServer, SimState, make_handler


def _serve(hub: N.NeuroHub | None):
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(url: str, body: dict[str, Any]) -> tuple[int, Any]:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


class _Ticker:
    """A real NeuroDriver with the real integrated brain, ticked on its own thread like the body loop does,
    with a person in view for the camera lane."""

    def __init__(self, seed: int = 4107) -> None:
        genome = Genome.from_seed(seed, gd=60)
        self.brain = make_brain(genome, seed)
        lanes = Lanes(brain=self.brain, names={"brain": "real", "senses": "stub", "motor": "stub"})
        self.driver = NeuroDriver(NeuroSpec(seed=seed), lanes=lanes)
        self.driver.state = "running"
        self.stop = threading.Event()
        self.person = True
        self.t = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            self.driver.tick(robot_state(self.t))
            self.t += 0.02
            time.sleep(0.002)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(5)


def _hub_with(source: Any) -> N.NeuroHub:
    hub = N.NeuroHub(enabled=True)
    hub.attach(source)
    return hub


def test_a_server_with_no_neural_duck_says_so() -> None:
    httpd, base = _serve(None)
    try:
        status, answer = _post(base + "/api/neuro/command", {"verb": "sit"})
        assert status == 400 and "--brain neuro" in answer["error"]
    finally:
        httpd.shutdown()


def test_the_stub_refuses_every_command_in_words() -> None:
    hub = _hub_with(None)
    hub.attach(N.StubSource(hub.card()))
    httpd, base = _serve(hub)
    try:
        status, answer = _post(base + "/api/neuro/command", {"verb": "sit"})
        assert status == 200 and answer["ok"] is False and answer["refused"] is True
        assert answer["reason"] == "not_running" and answer["heard"]["verb"] == "sit"
        assert answer["queued"] is False
    finally:
        hub.stop()
        httpd.shutdown()


@pytest.mark.parametrize(
    "body, reason",
    [
        ({}, "empty"),
        ({"text": "   "}, "empty"),
        ({"text": "blah blah"}, "not_understood"),
        ({"text": "don't sit", "lang": "en"}, "negated"),
        ({"verb": "dance"}, "unknown_verb"),
        ({"verb": "sit", "lang": "xx"}, "bad_lang"),
        ({"verb": "sit", "source": "carrier pigeon"}, "bad_source"),
        ({"verb": "sit", "confidence": 3}, "bad_confidence"),
        ({"verb": "go_there"}, "no_target"),
        ({"verb": "sit", "target": {"x_m": 1, "y_m": 1}}, "target_only_for_go_there"),
    ],
)
def test_a_request_that_is_no_command_is_a_400_with_its_reason(
    body: dict[str, Any], reason: str
) -> None:
    hub = _hub_with(None)
    hub.attach(N.StubSource(hub.card()))
    httpd, base = _serve(hub)
    try:
        status, answer = _post(base + "/api/neuro/command", body)
        assert status == 400 and answer["ok"] is False and answer["reason"] == reason
        assert answer["error"] == CB.ERROR_MSG[reason]
        assert answer["_i18n"]["error"]["k"] == f"neuro.command.error.{reason}"
        for lang in ("en", "ja", "es", "zh-Hans", "ko"):
            assert render(CB.ERROR_MSG[reason], lang).strip()
    finally:
        hub.stop()
        httpd.shutdown()


def test_every_command_error_the_parser_can_raise_has_its_words() -> None:
    from duckmind.neuro import commands as K

    for reason in (
        "empty",
        "negated",
        "not_understood",
        "unknown_verb",
        "bad_source",
        "bad_lang",
        "bad_confidence",
        "no_target",
        "target_only_for_go_there",
    ):
        assert isinstance(CB.error_words(CommandError(reason)), str)
        assert CB.error_words(CommandError(reason)).key == f"neuro.command.error.{reason}"
    assert K.CommandError("empty").reason == "empty"


def test_the_words_are_read_in_the_language_they_were_said_in() -> None:
    ticker = _Ticker()
    hub = _hub_with(N.DriverSource(lambda: ticker.driver))
    httpd, base = _serve(hub)
    try:
        for text, lang, verb in (
            ("坐下", "zh-Hant", "sit"),
            ("sit down", "en", "sit"),
            ("おすわり", "ja", "sit"),
            ("siéntate", "es", "sit"),
            ("坐下", "zh-Hans", "sit"),
            ("앉아", "ko", "sit"),
        ):
            status, answer = _post(base + "/api/neuro/command", {"text": text, "lang": lang})
            assert status == 200, (text, answer)
            assert answer["verb"] == verb and answer["heard"]["lang"] == lang, (text, answer)
            assert answer["heard"]["source"] == "text" and answer["heard"]["text"] == text
    finally:
        hub.stop()
        httpd.shutdown()
        ticker.close()


def test_a_command_reaches_the_brain_and_the_answer_says_queued_or_refused() -> None:
    ticker = _Ticker()
    hub = _hub_with(N.DriverSource(lambda: ticker.driver))
    httpd, base = _serve(hub)
    try:
        # a duck that is not asleep and not afraid, with nobody in sight: look_at_me is refused (no_one_seen)
        time.sleep(0.3)
        status, answer = _post(base + "/api/neuro/command", {"verb": "look_at_me", "lang": "en"})
        assert status == 200
        assert answer["ok"] is False and answer["refused"] is True and answer["queued"] is False
        assert answer["reason"] == "no_one_seen" and answer["state"] == "refused"
        assert answer["reason_zh"] == CB.REASON_MSG["no_one_seen"]
        assert answer["_i18n"]["reason_zh"]["k"] == "neuro.command.reason.no_one_seen"
        # a trick whose body is not wired yet (flip, until the roulade slice): heard, refused, and the reason says so
        status, answer = _post(base + "/api/neuro/command", {"text": "翻跟斗", "lang": "zh-Hant"})
        assert status == 200 and answer["verb"] == "flip" and answer["refused"] is True
        assert answer["reason"] in ("too_young", "not_yet") and answer["queued"] is False
        # a built-in verb the duck can do: queued
        status, answer = _post(base + "/api/neuro/command", {"verb": "stand"})
        assert status == 200
        assert (answer["ok"], answer["queued"]) == (True, True) or answer.get("already")
        assert answer["heard"] == {
            "verb": "stand",
            "lang": "zh-Hant",
            "confidence": 1.0,
            "named": False,
            "source": "button",
            "who": None,
        }
    finally:
        hub.stop()
        httpd.shutdown()
        ticker.close()


def test_the_telemetry_carries_the_last_command_and_its_outcome() -> None:
    ticker = _Ticker()
    hub = _hub_with(N.DriverSource(lambda: ticker.driver))
    httpd, base = _serve(hub)
    try:
        time.sleep(0.3)
        _post(base + "/api/neuro/command", {"verb": "spin"})
        time.sleep(0.4)
        telemetry = _get(base + "/api/neuro/telemetry?since=0")
        command = telemetry["command"]
        assert command["last"]["verb"] == "spin" and command["last"]["state"] == "refused"
        assert command["last"]["reason"] == "not_learned"
        assert command["last"]["reason_zh"] == CB.REASON_MSG["not_learned"]
        assert command["counts"]["total"] >= 1 and command["counts"]["refused"] >= 1
        assert command["latency_ms"]["n"] == 0
    finally:
        hub.stop()
        httpd.shutdown()
        ticker.close()


def test_a_newer_command_replaces_one_the_brain_has_not_taken_in_yet() -> None:
    brain = make_brain(Genome.from_seed(4107, gd=60), 4107)
    first = brain.commands.hear(Command("sit"))
    second = brain.commands.hear(Command("stand"))
    assert first.event.is_set() and first.answer["state"] == "interrupted"
    assert not second.event.is_set() and brain.commands.view()["pending"] is True


def test_a_brain_that_never_ticks_answers_busy_instead_of_hanging() -> None:
    brain = make_brain(Genome.from_seed(4107, gd=60), 4107)
    lanes = Lanes(brain=brain, names={"brain": "real", "senses": "stub", "motor": "stub"})
    driver = NeuroDriver(NeuroSpec(seed=4107), lanes=lanes)
    driver.state = "running"  # (nothing ticks it)
    hub = _hub_with(N.DriverSource(lambda: driver))
    started = time.monotonic()
    answer = hub.command({"verb": "sit"})
    assert time.monotonic() - started < CB.DECISION_WAIT_S * 2 + 1.0
    assert answer["refused"] is True and answer["reason"] == "busy" and answer["queued"] is False
    assert brain.commands.view()["pending"] is False


def test_a_pointed_go_there_goes_through_the_pointing_route_with_its_point() -> None:
    hub = _hub_with(None)
    source = N.StubSource(hub.card())
    seen: list[tuple[Any, Any, Any]] = []
    source.go_there = lambda x, y, who=None: (
        seen.append((x, y, who)) or {"ok": True, "queued": True}
    )  # type: ignore[attr-defined]
    hub.attach(source)
    answer = hub.command({"verb": "go_there", "target": {"x_m": 1.0, "y_m": 0.5}})
    assert answer["ok"] is True and answer["queued"] is True and answer["verb"] == "go_there"
    assert seen == [(1.0, 0.5, None)]
    source.go_there = lambda x, y, who=None: {"ok": False, "refused": True, "reason": "too_far"}  # type: ignore[attr-defined]
    answer = hub.command({"verb": "go_there", "target": {"x_m": 9.0, "y_m": 9.0}})
    assert answer["ok"] is False and answer["refused"] is True and answer["reason"] == "too_far"


def test_a_sighting_made_in_the_test_is_a_person() -> None:
    assert person_sight(1).kind == "person" and isinstance(C.Sighting, type)
