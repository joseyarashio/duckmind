"""Live voice, LV0: the endpoint that takes a browser microphone's PCM and injects it as a person at
a place (``duckmind.simweb.neuro_mic_api``): the happy path through the real HTTP server, the world
sound it asks for, and every limit (size, rate, speakers, place, format) with its words in the six
languages. No page, no MuJoCo, no network beyond 127.0.0.1."""

from __future__ import annotations

import http.client
import json
import math
import re
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.duckcore.api import Config
from duckmind.i18n import CODES, render
from duckmind.simweb import neuro_mic as M
from duckmind.simweb import neuro_mic_api as API
from duckmind.simweb.server import SimServer, SimState, make_handler
from duckmind.world import neuro_audio as A
from duckmind.world.neuro_world import level_at

NOISE_DBFS = -62.0


def _pcm(seconds: float, amp: float = 0.1, freq: float = 400.0) -> bytes:
    t = np.arange(int(seconds * A.SR)) / A.SR
    return A.to_pcm16(amp * np.sin(2 * math.pi * freq * t))


class _Body:
    """The body's ``world_call``: records what the hub asks of the world, answers like it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: Exception | None = None

    def world_call(self, op: str, **params: Any) -> dict[str, Any]:
        self.calls.append((op, params))
        if self.fail is not None:
            raise self.fail
        return {"ok": True, "stim": params.get("stim"), "sound": {"id": len(self.calls)}}


def _driver(mic: M.MicStream | None = None, body: Any = None) -> Any:
    sound = SimpleNamespace(mic=mic if mic is not None else M.MicStream(seed=7))
    return SimpleNamespace(
        lanes=SimpleNamespace(sound=sound),
        _body=body if body is not None else _Body(),
        _aux_wake=threading.Event(),
    )


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _hub(driver: Any = None) -> tuple[API.MicHub, Any, _Clock]:
    driver = driver if driver is not None else _driver()
    clock = _Clock()
    return API.MicHub(lambda: driver, clock=clock), driver, clock


# ── the happy path ───────────────────────────────────────────────────────────────────


def test_a_chunk_makes_a_person_at_a_place_speak_through_the_world() -> None:
    hub, driver, clock = _hub()
    out = hub.accept("owner", (1500.0, 500.0), _pcm(0.2))
    assert out["ok"] and out["who"] == "owner" and out["voiced"] is True
    assert out["accepted_ms"] == 200 and out["queued_ms"] == 200 and out["dropped_ms"] == 0
    ((op, params),) = driver._body.calls
    assert op == "x_neuro_stim"
    assert params == {
        "stim": "say",
        "live": True,
        "who": "owner",
        "seconds": API.SEGMENT_S,
        "where_mm": [1500.0, 500.0],
    }
    assert out["sound_id"] == 1
    assert driver._aux_wake.is_set(), "the world's truth is read again at once"
    assert driver.lanes.sound.mic.view()["live"]["owner"]["queued_ms"] == 200


def test_the_world_is_asked_again_only_as_the_voice_runs_out_and_when_the_person_moves() -> None:
    hub, driver, clock = _hub()
    calls = driver._body.calls
    for k in range(15):  # three seconds of speech in pieces of 200 ms, in real time
        hub.accept("owner", (1000.0, 0.0), _pcm(0.2))
        clock.t += 0.2
    assert 3 <= len(calls) <= 5, "about one sound a second, not one for every piece"
    n = len(calls)
    hub.accept("owner", (1000.0, 0.0), _pcm(0.2))
    assert len(calls) == n  # (the voice still has time left)
    clock.t += 0.3
    hub.accept("owner", (3000.0, 0.0), _pcm(0.2))  # she walked to the other side of the room
    assert len(calls) == n + 1 and calls[-1][1]["where_mm"] == [3000.0, 0.0]


def test_a_place_is_optional_and_then_the_world_chooses_it() -> None:
    hub, driver, _ = _hub()
    hub.accept("child", None, _pcm(0.2))
    assert "where_mm" not in driver._body.calls[0][1]


def test_a_pause_is_audio_without_a_world_sound() -> None:
    hub, driver, clock = _hub()
    out = hub.accept("owner", (0.0, 0.0), A.to_pcm16(np.zeros(3200)))
    assert out["ok"] and out["voiced"] is False and driver._body.calls == []
    clock.t += 0.2
    hub.accept("owner", (0.0, 0.0), _pcm(0.2))
    assert len(driver._body.calls) == 1  # (the first word)


def test_the_audio_reaches_the_ducks_microphone_at_the_level_of_the_place() -> None:
    mic = M.MicStream(seed=7, wall=lambda: 0.0)
    hub, driver, _ = _hub(_driver(mic))
    hub.accept("owner", (2000.0, 0.0), _pcm(0.6, 0.05))
    # what the world would then list (its sound, its level at the head: 2 m away)
    event = {
        "id": 1,
        "kind": "voice",
        "t0": 0.0,
        "t1": 1.0,
        "db": level_at(62.0, 2.0),
        "env": "speech",
        "src": "person:owner",
        "utterance": "live:owner",
    }
    frames = [mic.tick(k * 0.02, [event]) for k in range(30)]
    x = np.concatenate([A.to_float(f.pcm) for f in frames])
    assert A.active_rms_dbfs(x[: int(0.55 * A.SR)]) == pytest.approx(
        level_at(62.0, 2.0) - A.FULL_SCALE_DB, abs=1.0
    )


# ── the limits ───────────────────────────────────────────────────────────────────────


def _refusal(hub: API.MicHub, *args: Any) -> API.MicRefusal:
    with pytest.raises(API.MicRefusal) as caught:
        hub.accept(*args)
    return caught.value


@pytest.mark.parametrize("who", ["", "Owner", "1abc", "a b", "x" * 17, "é", "../etc"])
def test_a_speaker_key_is_a_short_lower_case_word(who: str) -> None:
    hub, driver, _ = _hub()
    refusal = _refusal(hub, who, None, _pcm(0.1))
    assert refusal.status == 400 and "16" in str(refusal.words)
    assert driver._body.calls == []


@pytest.mark.parametrize(
    "place",
    [(float("nan"), 0.0), (0.0, float("inf")), (50_001.0, 0.0), (0.0, -50_001.0)],
)
def test_a_place_outside_the_home_is_refused(place: tuple[float, float]) -> None:
    hub, driver, _ = _hub()
    refusal = _refusal(hub, "owner", place, _pcm(0.1))
    assert refusal.status == 400 and "公釐" in str(refusal.words)
    assert driver._body.calls == []
    assert hub.accept("owner", (50_000.0, -50_000.0), _pcm(0.1))["ok"]  # (the edge is inside)


@pytest.mark.parametrize("pcm", [b"", b"\x01", b"\x01\x02\x03"])
def test_audio_must_be_whole_16_bit_samples(pcm: bytes) -> None:
    hub, _, _ = _hub()
    refusal = _refusal(hub, "owner", None, pcm)
    assert refusal.status == 400 and "16 kHz" in str(refusal.words)


def test_a_piece_is_at_most_one_second() -> None:
    hub, driver, _ = _hub()
    assert API.MAX_CHUNK_BYTES == 32_000
    assert hub.accept("owner", None, _pcm(1.0))["ok"]
    refusal = _refusal(hub, "child", None, _pcm(1.02))
    assert refusal.status == 413 and "1" in str(refusal.words)
    assert len(driver.lanes.sound.mic.view()["live"]) == 1  # (nothing of it was queued)


def test_a_speaker_cannot_send_faster_than_real_time_for_long() -> None:
    hub, driver, clock = _hub()
    sent = 0.0
    refusals = 0
    status = None
    for _ in range(30):  # a second of audio every 0.1 s: ten times real time
        try:
            hub.accept("owner", None, _pcm(1.0))
            sent += 1.0
        except API.MicRefusal as refusal:
            refusals += 1
            status = refusal.status
        clock.t += 0.1
    assert status == 429 and refusals > 20
    assert sent <= API.BURST_S + API.RATE_X * 3.0 + 1.0, sent  # the burst and what the rate allows
    clock.t += 3.0  # a while later it may speak again
    assert hub.accept("owner", None, _pcm(0.5))["ok"]
    assert hub.refused == refusals


def test_one_speakers_rate_does_not_hold_up_another() -> None:
    hub, _, _ = _hub()
    for _ in range(2):
        hub.accept("owner", None, _pcm(1.0))
    assert _refusal(hub, "owner", None, _pcm(1.0)).status == 429
    assert hub.accept("child", None, _pcm(1.0))["ok"]


def test_at_most_three_people_speak_at_once_and_an_idle_one_makes_room() -> None:
    hub, _, clock = _hub()
    for who in ("owner", "partner", "child"):
        assert hub.accept(who, None, _pcm(0.1))["ok"]
    refusal = _refusal(hub, "elder", None, _pcm(0.1))
    assert refusal.status == 429 and "3" in str(refusal.words)
    assert hub.accept("owner", None, _pcm(0.1))["ok"]  # (one already speaking is welcome)
    clock.t += API.SPEAKER_IDLE_S + 1.0
    assert hub.accept("elder", None, _pcm(0.1))["ok"]


def test_without_the_ducks_microphone_or_the_world_it_says_so() -> None:
    nothing = API.MicHub(lambda: None)
    refusal = _refusal(nothing, "owner", None, _pcm(0.1))
    assert refusal.status == 503 and "麥克風" in str(refusal.words)
    no_world = SimpleNamespace(
        lanes=SimpleNamespace(sound=SimpleNamespace(mic=M.MicStream(seed=7))), _body=None
    )
    refusal = _refusal(API.MicHub(lambda: no_world), "owner", None, _pcm(0.1))
    assert refusal.status == 503 and "物理世界" in str(refusal.words)


def test_a_world_that_says_no_is_said_in_its_own_words_and_nothing_is_queued() -> None:
    body = _Body()
    body.fail = ValueError("世界還在啟動")
    hub, driver, clock = _hub(_driver(body=body))
    refusal = _refusal(hub, "owner", (0.0, 0.0), _pcm(0.2))
    assert refusal.status == 200 and "世界還在啟動" in str(refusal.words)
    assert driver.lanes.sound.mic.view()["live"] == {}
    body.fail = None
    clock.t += 1.0
    assert hub.accept("owner", (0.0, 0.0), _pcm(0.2))["ok"]  # (the next piece asks again)


def test_every_word_the_hub_can_say_is_in_all_six_languages() -> None:
    from duckmind.i18n import msg

    words = [
        msg("neuro.web.neuro_mic_api.bad_who"),
        msg("neuro.world_refusal.say_where"),
        msg("neuro.web.neuro_mic_api.bad_audio"),
        msg("neuro.web.neuro_mic_api.too_big", seconds=1),
        msg("neuro.web.neuro_mic_api.no_mic"),
        msg("neuro.web.neuro_mic_api.no_world"),
        msg("neuro.web.neuro_mic_api.too_many", count=3),
        msg("neuro.web.neuro_mic_api.too_fast"),
        msg("neuro.web.neuro_mic_api.no_length"),
    ]
    for word in words:
        shown = {lang: render(word, lang, strict=True) for lang in CODES}
        assert len(set(shown.values())) == len(CODES), (str(word), shown)  # six different texts
        assert not re.search(r"\{\w+\}", " ".join(shown.values())), shown  # every value filled in
        for lang in ("en", "es", "ko"):
            assert not re.search(r"[一-鿿]", shown[lang]), shown[lang]


# ── through the real HTTP server ─────────────────────────────────────────────────────


class _Bridge:
    """The live bridge as far as the server's clock and the hub look at it."""

    def __init__(self, neuro: Any) -> None:
        self.neuro = neuro

    def take_fresh(self) -> bool:
        return False

    def set_clock(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.fixture()
def server():
    state = SimState(Config(seed=7, name="測試"))
    driver = _driver()
    state.live = _Bridge(driver)  # (the bridge's neuro driver, as the hub finds it)
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(
        base=f"http://127.0.0.1:{httpd.server_address[1]}",
        port=httpd.server_address[1],
        driver=driver,
        state=state,
    )
    httpd.shutdown()
    httpd.server_close()


def _post(base: str, query: str, body: bytes) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base}/api/neuro/mic/chunk?{query}",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_a_post_over_http_injects_the_person(server) -> None:
    status, data = _post(server.base, "who=owner&x_mm=1500&y_mm=500", _pcm(0.2))
    assert status == 200 and data["ok"] and data["accepted_ms"] == 200
    ((op, params),) = server.driver._body.calls
    assert (
        op == "x_neuro_stim" and params["where_mm"] == [1500.0, 500.0] and params["who"] == "owner"
    )
    with urllib.request.urlopen(server.base + "/api/neuro/mic/status", timeout=10) as response:
        status_body = json.loads(response.read())
    assert status_body["running"] and status_body["accepted_ms"] == 200
    assert status_body["speakers"]["owner"]["accepted_ms"] == 200
    assert status_body["limits"]["max_chunk_bytes"] == 32_000
    assert status_body["limits"]["format"] == "pcm_s16le;rate=16000;channels=1"
    assert status_body["mic"]["rate"] == 16_000


def test_http_refusals_have_their_status_and_their_words_in_every_language(server) -> None:
    status, data = _post(server.base, "who=Owner", _pcm(0.1))
    assert status == 400 and not data["ok"] and "16" in data["error"]
    assert data["_i18n"]["error"]["k"] == "neuro.web.neuro_mic_api.bad_who"
    status, data = _post(server.base, "who=owner&x_mm=abc&y_mm=1", _pcm(0.1))
    assert status == 400 and "公釐" in data["error"]
    status, data = _post(server.base, "who=owner&x_mm=5", _pcm(0.1))  # one of the two
    assert status == 400
    status, data = _post(server.base, "who=owner", b"\x01\x02\x03")
    assert status == 400 and "16 kHz" in data["error"]
    status, data = _post(server.base, "who=owner", _pcm(1.5))
    assert status == 413 and data["_i18n"]["error"]["k"] == "neuro.web.neuro_mic_api.too_big"
    assert server.driver._body.calls == []


def test_a_body_with_no_length_is_refused_before_it_is_read(server) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.putrequest("POST", "/api/neuro/mic/chunk?who=owner")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.putheader("Content-Type", "application/octet-stream")
    conn.endheaders()
    conn.send(b"4\r\nabcd\r\n0\r\n\r\n")
    response = conn.getresponse()
    assert response.status == 411
    assert json.loads(response.read())["_i18n"]["error"]["k"] == "neuro.web.neuro_mic_api.no_length"
    conn.close()


def test_a_body_shorter_than_it_says_is_refused(server) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.putrequest("POST", "/api/neuro/mic/chunk?who=owner")
    conn.putheader("Content-Length", "640")
    conn.endheaders()
    conn.send(bytes(100))
    conn.sock.shutdown(1)  # (the page went away mid-body)
    response = conn.getresponse()
    assert response.status == 400
    conn.close()


def test_other_paths_under_the_route_and_a_missing_microphone(server) -> None:
    try:
        urllib.request.urlopen(server.base + "/api/neuro/mic/nope", timeout=10)
        raise AssertionError("not found expected")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    request = urllib.request.Request(server.base + "/api/neuro/mic/other", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 404
    server.state.live = None  # no neural duck running
    status, data = _post(server.base, "who=owner", _pcm(0.1))
    assert status == 503 and "麥克風" in data["error"]
    with urllib.request.urlopen(server.base + "/api/neuro/mic/status", timeout=10) as response:
        assert json.loads(response.read())["running"] is False


def test_the_existing_sound_and_neuro_routes_still_answer(server) -> None:
    server.state.live = None  # (the sound hub reads a real driver's lanes: here, none running)
    with urllib.request.urlopen(server.base + "/api/neuro/sound/status", timeout=10) as response:
        assert response.status == 200 and json.loads(response.read())["running"] is False
    with urllib.request.urlopen(server.base + "/api/neuro/status", timeout=10) as response:
        hub = json.loads(response.read())  # (the neural-duck hub is off here: its own answer)
    assert response.status == 200 and "running" not in hub
