"""voice:neuro — the neural duck's calls in its own official Microduck voice (owner 2026-09-28:
「模擬器鴨子叫時要真的播放出聲音，希望能跟官方的叫聲一致」).

* ``simweb/voice.py`` is the main tree's module, copied verbatim (stdlib only; upstream's seed
  derivation and bank layout);
* each neural duck — the picker's four and any random one — has its own voice from its seed,
  for life (a duck at another age: the same voice), and every card says which;
* the server's ``/api/voice/…`` routes serve the neural duck on screen, follow a switch (the old
  duck's files are no longer the page's), render the duck on screen first and the picker's four
  behind it; without the prototype they serve Duck Core's duck; without a checkout they say why;
* the page plays the bank (no synthesized stand-in), tells voice.js which duck is on screen, and
  sounds each call with the beak its robot.mouth command opened (the voice's own envelope only
  where there is no body beak).

A small stand-in for upstream's ``sounds`` binary renders tiny banks here (as in the main tree's
tests); the one test that runs the real binary skips unless the official checkout is configured
(DUCKMIND_MICRODUCK, or found beside DUCKMIND_MICRODUCK_MJCF)."""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from duckmind.duckcore.api import Config
from duckmind.simweb import neuro_api as N
from duckmind.simweb import voice
from duckmind.simweb.server import SimServer, SimState, make_handler

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
DANHUANG_VOICE = 2_838_558_865  # 蛋黃 (Duck Core seed 7)
# upstream's seed_from_id("duckmind-duck:<seed>") of the picker's four
PRESET_VOICES = {
    "explorer": 936_833_334,  # 阿勇 4107
    "snuggler": 3_677_786_832,  # 怯怯 1728
    "dreamer": 2_191_483_871,  # 慢慢 4670
    "hatchling": 2_282_912_503,  # 啾啾 3341
}

# Writes what `sounds render-all <dir> --seed <u32>` writes (tiny 48 kHz mono wavs) and logs each
# run beside itself (the main tree's test_simweb_voice.py stand-in).
FAKE_SOUNDS = r"""
import struct
import sys
from pathlib import Path

me = Path(sys.argv[0])
with open(me.with_suffix(".log"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
assert sys.argv[1] == "render-all" and sys.argv[3] == "--seed", sys.argv
out, seed = Path(sys.argv[2]), int(sys.argv[4])
counts = {"alarm": 10, "greet": 12, "inquire": 10, "peck": 10, "chirp": 12, "coo": 10, "wheee": 6}

def wav(n):
    data = struct.pack("<h", seed % 30000) * n
    head = struct.pack("<IHHIIHH", 16, 1, 1, 48000, 96000, 2, 16)
    return b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + head + b"data" + struct.pack("<I", len(data)) + data

for tag, count in counts.items():
    (out / tag).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        letter = chr(97 + i)
        parts = ("start", "loop", "end") if tag == "wheee" else (None,)
        for part in parts:
            name = f"{tag}_{part}_{letter}.wav" if part else f"{tag}_{letter}.wav"
            (out / tag / name).write_bytes(wav(64))
print(f"wrote under {out}")
"""


def make_checkout(root: Path) -> Path:
    """A microduck checkout with a built (stand-in) `sounds` and upstream's lib.rs constant."""
    checkout = root / "microduck"
    binary = checkout / "target" / "debug" / "sounds"
    binary.parent.mkdir(parents=True)
    binary.write_text(f"#!{sys.executable}\n{FAKE_SOUNDS}")
    binary.chmod(0o755)
    lib = checkout / "sounds" / "src" / "lib.rs"
    lib.parent.mkdir(parents=True)
    lib.write_text(f"pub const BANK_VERSION: u32 = {voice.BANK_VERSION};\n")
    return checkout


def rendered(checkout: Path) -> list[int]:
    """The voice seeds the stand-in rendered, in order."""
    log = checkout / "target" / "debug" / "sounds.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return [int(line.split()[3]) for line in lines]


def raw_get(port: int, path: str) -> tuple[int, bytes, dict[str, str]]:
    """A GET with the path exactly as written (no client-side normalisation)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest("GET", path, skip_accept_encoding=True)
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body, dict(response.getheaders())


def post(port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request(
        "POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    answer = json.loads(response.read())
    connection.close()
    return response.status, answer


def manifest_of(port: int, bank: voice.VoiceBank, duck_seed: int) -> dict[str, Any]:
    status, body, headers = raw_get(port, "/api/voice/manifest?t=1")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    manifest = json.loads(body)
    if manifest["state"] == "rendering":  # (its render runs in the background)
        bank.wait(duck_seed, 30)
        manifest = json.loads(raw_get(port, "/api/voice/manifest")[1])
    return manifest


# ── which voice each neural duck has ─────────────────────────────────────────────────


def test_the_voice_module_is_the_main_trees_stdlib_only_module():
    source = Path(voice.__file__).read_text()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    # (stdlib only, apart from the six-language message layer and the folder settings)
    allowed = ("from duckmind.i18n import", "from duckmind import paths")
    own = [line for line in imports if "duckmind" in line and not line.startswith(allowed)]
    assert imports and not own, "stdlib only"
    # upstream's own derivation (sounds::seed_from_id, pinned upstream) and bank layout
    assert voice.seed_from_id("test-serial") == 0xC96F_1146
    assert voice.BANK_VERSION == 5 and voice.BANK_FILES == 82
    assert voice.TAGS == ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee")
    assert voice.voice_seed(7) == DANHUANG_VOICE


def test_each_neural_duck_has_its_own_voice_for_life():
    voices = {item.preset: voice.voice_seed(item.seed) for item in N.PRESETS}
    assert voices == PRESET_VOICES
    assert len(set(voices.values())) == 4 and DANHUANG_VOICE not in voices.values()
    for item in N.PRESETS:
        card = N.duck_card(item)
        assert card["voice_seed"] == PRESET_VOICES[item.preset]
        # the same duck at another age keeps its voice (the seed is the duck)
        older = N.duck_card(N.DuckChoice(item.seed, item.gd + 40, "x"))
        assert older["voice_seed"] == card["voice_seed"]
    picks = iter([123456, 0, 5])  # seed, stage index, offset in the stage
    duck = N.random_choice(lambda n: next(picks) % n)
    assert N.duck_card(duck)["voice_seed"] == voice.voice_seed(123456)
    assert N.card_of({"gd": 1})["voice_seed"] is None  # (no seed known: no voice claimed)


def test_the_hub_voices_the_duck_on_screen_first_then_the_pickers_four(tmp_path):
    checkout = make_checkout(tmp_path)
    bank = voice.VoiceBank.from_microduck(checkout, tmp_path / "voices")
    hub = N.NeuroHub(enabled=True, choice=N.PRESET_BY_ID["snuggler"], voice_bank=bank)
    hub.attach(N.StubSource(hub.card()))
    try:
        assert hub.voice_duck() == 1728
        thread = hub.prepare_voices()
        assert thread is not None
        thread.join(60)
        order = rendered(checkout)
        assert order[0] == PRESET_VOICES["snuggler"], "the duck on screen first"
        assert sorted(order) == sorted(PRESET_VOICES.values()), "then the other three, once each"
        assert all(bank.ready(value) for value in PRESET_VOICES.values())
        # asked again (a restarted server): the cache, no render
        again = N.NeuroHub(
            enabled=True, voice_bank=voice.VoiceBank.from_microduck(checkout, tmp_path / "voices")
        )
        again_thread = again.prepare_voices()
        assert again_thread is not None
        again_thread.join(60)
        assert len(rendered(checkout)) == 4
        # a switch renders the new duck's voice at once, while its brain changes
        hub.switch({"seed": 42, "gd": 70})
        assert hub.voice_duck() == 42
        bank.wait(42, 30)
        assert bank.ready(voice.voice_seed(42)) and rendered(checkout)[-1] == voice.voice_seed(42)
    finally:
        hub.stop()
    assert N.NeuroHub(enabled=True).prepare_voices() is None  # (no voice here: nothing to do)


# ── the routes ───────────────────────────────────────────────────────────────────────


@pytest.fixture()
def stub_server(tmp_path):
    """The prototype's page with made-up telemetry (--neuro-stub) and a stand-in `sounds`."""
    bank = voice.VoiceBank.from_microduck(make_checkout(tmp_path), tmp_path / "voices")
    hub = N.NeuroHub(enabled=True, voice_bank=bank)  # (阿勇, the default duck)
    hub.attach(N.StubSource(hub.card()))
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub, bank))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], bank
    hub.stop()
    httpd.shutdown()
    httpd.server_close()


def test_the_page_gets_the_neural_ducks_own_voice_and_a_switch_changes_it(stub_server):
    port, bank = stub_server
    manifest = manifest_of(port, bank, 4107)
    explorer = PRESET_VOICES["explorer"]
    assert (
        manifest["ready"] and manifest["duck_seed"] == 4107 and manifest["voice_seed"] == explorer
    )
    url = manifest["tags"]["greet"]["variants"]["c"]
    assert url == f"/api/voice/v5/{explorer}/greet/greet_c.wav"
    status, body, headers = raw_get(port, url)
    assert status == 200 and body[:4] == b"RIFF" and headers["Content-Type"] == "audio/wav"
    # the card the page shows names the same voice; so does the picker, for each of the four
    shown = json.loads(raw_get(port, "/api/neuro/status")[1])
    assert shown["duck"]["voice_seed"] == explorer
    assert {duck["preset"]: duck["voice_seed"] for duck in shown["ducks"]} == PRESET_VOICES
    # 換一隻小鴨 → 怯怯: its own voice; 阿勇's files are no longer the page's
    code, answer = post(port, "/api/neuro/duck", {"preset": "snuggler"})
    assert code == 200 and answer["duck"]["voice_seed"] == PRESET_VOICES["snuggler"]
    manifest = manifest_of(port, bank, 1728)
    assert manifest["duck_seed"] == 1728 and manifest["voice_seed"] == PRESET_VOICES["snuggler"]
    assert raw_get(port, url)[0] == 404
    assert raw_get(port, manifest["tags"]["coo"]["variants"]["a"])[0] == 200
    wheee = manifest["tags"]["wheee"]
    assert wheee["segmented"] and set(wheee["variants"]["f"]) == {"start", "loop", "end"}
    # a random duck: its own voice from its own seed
    code, answer = post(port, "/api/neuro/duck", {"random": True})
    assert code == 200
    seed = int(answer["duck"]["seed"])
    manifest = manifest_of(port, bank, seed)
    assert manifest["duck_seed"] == seed
    assert manifest["voice_seed"] == voice.voice_seed(seed) == answer["duck"]["voice_seed"]
    # nothing a path could reach
    for path in (
        f"/api/voice/v5/{manifest['voice_seed']}/chirp/../coo/coo_a.wav",
        f"/api/voice/v5/{manifest['voice_seed']}/chirp/%2e%2e%2f.seed",
        f"/api/voice/v5/{manifest['voice_seed']}/.seed",
        f"/api/voice/v5/{PRESET_VOICES['snuggler']}/coo/coo_a.wav",  # the duck before
        "/api/voice/../../pyproject.toml",
        "/api/voice/v5/../voice.js",
    ):
        assert raw_get(port, path)[0] == 404, path


def test_a_normal_server_voices_duck_cores_duck_and_says_why_when_it_cannot(tmp_path):
    def serve(bank: voice.VoiceBank | None) -> tuple[SimServer, int]:
        state = SimState(Config(seed=7, name="測試"))
        httpd = SimServer(("127.0.0.1", 0), make_handler(state, None, bank))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, httpd.server_address[1]

    checkout = make_checkout(tmp_path)
    for bank, want in (
        (voice.VoiceBank.from_microduck(checkout, tmp_path / "voices"), "ready"),
        (None, "unavailable"),  # (a server built without one: never a missing route)
        # no checkout and nothing in its cache (a bank already there would still be served)
        (voice.VoiceBank.from_microduck(None, tmp_path / "empty"), "unavailable"),
    ):
        httpd, port = serve(bank)
        try:
            manifest = manifest_of(port, bank or voice.VoiceBank.off(), 7)
            assert manifest["state"] == want and manifest["duck_seed"] == 7
            if want == "ready":
                assert manifest["voice_seed"] == DANHUANG_VOICE  # Duck Core's duck: seed 7
            else:
                assert "不會出聲" in manifest["note"]
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_the_driver_gives_each_call_its_moment_on_the_beaks_clock():
    class Driver:
        def telemetry(self, since: int = 0) -> dict[str, Any]:
            return {
                "state": "running",
                "seq": 3,
                "samples": [{"t_ms": 1200, "mouth": 0.0}, {"t_ms": 1300, "mouth": 0.6}],
                "vocals": [{"seq": 2, "t_ms": 1240, "wall_ms": 99, "tag": "chirp", "level": 0.5}],
            }

    source = N.DriverSource(lambda: Driver(), clock=iter(range(0, 10_000)).__next__)
    [call] = source.snapshot(0)["vocal"]
    assert call["tag"] == "chirp" and call["t_ms"] == 1240 and call["level"] == 0.5


# ── the page's side ──────────────────────────────────────────────────────────────────


def test_the_page_plays_the_official_voice_with_the_ducks_own_beak():
    script = (STATIC / "voice.js").read_text()
    assert '"/api/voice/manifest"' in script and 'KEY = "duckmind.voice"' in script
    assert (
        "window.duckVoice = { call, mouth, stop, setVolume, setMuted, refresh: loadManifest, "
        "duck: setDuck, switching, gain: gainOf, state };"
    ) in script
    assert "createOscillator" not in script, "the synthesized stand-in is gone"
    assert "Math.random() * letters.length" in script  # robotd: a random variant each time
    # a mute between calls is silent from the next call's first sample (a ramp that ends)
    assert "linearRampToValueAtTime(gainOf(), now + 0.03)" in script
    neuro = (STATIC / "neuro.js").read_text()
    assert "window.duckVoice.duck(duck.seed)" in neuro  # the duck on screen: its own voice
    assert "voice.switching()" in neuro  # (a switch: never the old duck's voice meanwhile)
    assert "window.duckVoice.call([spoken.tag])" in neuro  # (no 3D view: voiced here)
    assert "t_ms: spoken.t_ms" in neuro
    duck3d = (STATIC / "duck3d.js").read_text()
    assert (
        "window.duckVoice.call([String(call.tag)], { maxWaitMs: CALL_LATE_MS, lib: call.lib })"
        in duck3d
    )
    # the beak: its robot.mouth command where the body samples carry it, else the sound's own
    assert "view.mouth = Math.min(1, Math.max(0, view.live.drawnMouth))" in duck3d
    assert "0.6 * window.duckVoice.mouth()" in duck3d
    page = (STATIC / "index.html").read_text()
    for hook in (
        'id="callSound"',
        'id="callMute"',
        'type="range" id="callVolume"',
        'id="callNote"',
    ):
        assert hook in page, hook
    assert (
        page.index('src="/voice.js"')
        < page.index('src="/neuro.js"')
        < page.index('src="/duck3d.js"')
    ), "duckVoice before its users"


# ── the real binary ──────────────────────────────────────────────────────────────────


def upstream_microduck() -> Path | None:
    named = os.environ.get("DUCKMIND_MICRODUCK")
    if named:
        return Path(named).expanduser()
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if mjcf:
        for parent in Path(mjcf).expanduser().resolve().parents:
            if (parent / "microduck" / "sounds" / "Cargo.toml").is_file():
                return parent / "microduck"
    return None


@pytest.mark.skipif(
    voice.find_sounds_bin(upstream_microduck()) is None,
    reason="the official microduck checkout with a built target/debug/sounds is not configured",
)
def test_the_real_binary_renders_a_neural_ducks_official_bank(tmp_path):
    bank = voice.VoiceBank.from_microduck(upstream_microduck(), tmp_path / "voices")
    hub = N.NeuroHub(enabled=True, choice=N.PRESET_BY_ID["hatchling"], voice_bank=bank)
    bank.render(voice.voice_seed(hub.voice_duck()))
    folder = bank.bank_dir(PRESET_VOICES["hatchling"])
    assert folder is not None and bank.ready(PRESET_VOICES["hatchling"])
    for tag, names in voice.WHITELIST.items():
        for name in names:
            with wave.open(str(folder / tag / name)) as sound:
                assert (sound.getframerate(), sound.getnchannels(), sound.getsampwidth()) == (
                    48_000,
                    1,
                    2,
                ), name
                assert sound.getnframes() > 1000, name
