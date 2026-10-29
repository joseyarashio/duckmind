"""The duck's own voice on the simulator page (simweb/voice.py, owner 2026-09-28): which official
voice a duck has, its bank's cache and lock, the routes that serve it (only the current duck's
whitelisted files, nothing a path could reach) and the feed's calls.

A small stand-in for upstream's ``sounds`` binary renders tiny banks here; the one test that runs
the real binary skips unless the official checkout is built (DUCKMIND_MICRODUCK, or found beside
DUCKMIND_MICRODUCK_MJCF)."""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import wave
from pathlib import Path

import pytest

from duckmind.duckcore.api import Config, dispatch, feed
from duckmind.simweb import voice
from duckmind.simweb.server import SimServer, SimState, make_handler

DANHUANG_SEED = 7  # 蛋黃
DANHUANG_VOICE = 2_838_558_865

# Writes what `sounds render-all <dir> --seed <u32>` writes (tiny 48 kHz mono wavs), logs each
# run beside itself; a `.mode` file beside it makes it fail ("fail") or leave a file out ("short").
FAKE_SOUNDS = r"""
import struct
import sys
from pathlib import Path

me = Path(sys.argv[0])
with open(me.with_suffix(".log"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
mode = me.with_suffix(".mode").read_text().strip() if me.with_suffix(".mode").exists() else "ok"
if mode == "fail":
    print("error: the synth fell over", file=sys.stderr)
    sys.exit(3)
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
            if mode == "short" and name == "coo_j.wav":
                continue
            (out / tag / name).write_bytes(wav(64))
print(f"wrote under {out}")
"""


def make_checkout(root: Path, bank_version: int = 5) -> Path:
    """A microduck checkout with a built (stand-in) `sounds` and upstream's lib.rs constant."""
    checkout = root / "microduck"
    binary = checkout / "target" / "debug" / "sounds"
    binary.parent.mkdir(parents=True)
    binary.write_text(f"#!{sys.executable}\n{FAKE_SOUNDS}")
    binary.chmod(0o755)
    lib = checkout / "sounds" / "src" / "lib.rs"
    lib.parent.mkdir(parents=True)
    lib.write_text(f"pub const BANK_VERSION: u32 = {bank_version};\n")
    return checkout


def runs(checkout: Path) -> list[str]:
    log = checkout / "target" / "debug" / "sounds.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture()
def bank(tmp_path: Path) -> voice.VoiceBank:
    return voice.VoiceBank.from_microduck(make_checkout(tmp_path), tmp_path / "voices")


def ready_bank(bank: voice.VoiceBank, duck_seed: int) -> dict:
    manifest = bank.manifest(duck_seed, wait_s=30)
    assert manifest["state"] == "ready", manifest
    return manifest


# ---------------------------------------------------------------- which voice a duck has


def test_the_seed_derivation_is_upstreams():
    # sounds::seed_from_id, pinned upstream: printf 'test-serial' | sha256sum → c96f1146...
    assert voice.seed_from_id("test-serial") == 0xC96F_1146


def test_each_duck_keeps_one_voice_and_danhuang_has_its_own():
    assert voice.voice_seed(DANHUANG_SEED) == voice.voice_seed(DANHUANG_SEED) == DANHUANG_VOICE
    # its own: not upstream's bench seed 7 (voices/v5/seed7), nor the neural demo's ducks'
    assert DANHUANG_VOICE != DANHUANG_SEED
    assert DANHUANG_VOICE not in {voice.voice_seed(seed) for seed in (4107, 1728, 3341)}
    voices = [voice.voice_seed(seed) for seed in range(2000)]
    assert len(set(voices)) == len(voices), "two seeds, two voices"
    assert all(0 <= value < 2**32 for value in voices), "a u32, as sounds --seed takes"


def test_the_whitelist_is_the_layout_upstream_renders():
    # sounds::render_all: 6 one-shot tags × their variants + wheee (6 × start/loop/end)
    assert voice.BANK_FILES == 10 + 12 + 10 + 10 + 12 + 10 + 6 * 3 == 82
    assert voice.TAGS == ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee")
    assert (
        "chirp_l.wav" in voice.WHITELIST["chirp"] and "chirp_m.wav" not in voice.WHITELIST["chirp"]
    )
    assert {"wheee_start_f.wav", "wheee_loop_f.wav", "wheee_end_f.wav"} <= voice.WHITELIST["wheee"]
    assert "wheee_a.wav" not in voice.WHITELIST["wheee"]
    names = [name for names in voice.WHITELIST.values() for name in names]
    assert all("/" not in name and ".." not in name and name.endswith(".wav") for name in names)


def test_calls_are_the_sounds_duck_core_recorded_and_words_make_none():
    assert voice.calls_of("duck.expression", {"verb": "perk_tilt", "sounds": ("chirp",)}) == [
        "chirp"
    ]
    assert voice.calls_of("duck.bedtime", {"sounds": ["coo", "coo"], "text": "晚安咕～咕～"}) == [
        "coo",
        "coo",
    ]
    babble = {"tags": ("chirp", "wheee", "coo"), "text": "啾·咻～·咕～", "code": "CWO"}
    assert voice.calls_of("voice.babble", babble) == ["chirp", "wheee", "coo"]
    # words only: a note, a reaction without sounds, a song's record, an owner's act
    assert voice.calls_of("duck.moment", {"text": "機智的一刻"}) == []
    assert voice.calls_of("duck.reaction", {"to": "callback", "text": "想起上次"}) == []
    assert voice.calls_of("voice.song", {"phrases": ["CWO"], "text": "我們家的歌"}) == []
    assert voice.calls_of("owner.hello", {"absence_h": 3}) == []
    assert voice.calls_of("duck.bid", {"sounds": []}) == []
    assert voice.calls_of("duck.expression", {"sounds": ["chirp", "moo", 3]}) == ["chirp"]
    assert voice.calls_of("duck.expression", None) == []


# ---------------------------------------------------------------- the bank's cache


def test_a_bank_is_rendered_once_then_served_from_the_cache(tmp_path, bank):
    checkout = tmp_path / "microduck"
    manifest = ready_bank(bank, DANHUANG_SEED)
    assert manifest["voice_seed"] == DANHUANG_VOICE and manifest["duck_seed"] == DANHUANG_SEED
    folder = tmp_path / "voices" / "v5" / f"seed{DANHUANG_VOICE}"
    assert bank.bank_dir(DANHUANG_VOICE) == folder
    assert (folder / ".seed").read_text() == f"{DANHUANG_VOICE}:5", "upstream's ensure-bank marker"
    assert sum(1 for _ in folder.rglob("*.wav")) == 82
    [run] = runs(checkout)
    command, staging, flag, seed = run.split()
    # rendered beside the bank into a staging folder, then renamed into place
    assert (command, flag, seed) == ("render-all", "--seed", str(DANHUANG_VOICE))
    assert Path(staging).parent.resolve() == folder.parent.resolve()
    assert Path(staging).name.startswith(f".seed{DANHUANG_VOICE}.") and staging.endswith(".new")
    assert not [p for p in folder.parent.iterdir() if p.name.endswith((".new", ".old"))]
    # asked again (and by another simulator on this machine): the cache, no second render
    assert bank.prepare(DANHUANG_SEED) == "ready"
    other = voice.VoiceBank.from_microduck(checkout, tmp_path / "voices")
    assert other.manifest(DANHUANG_SEED)["state"] == "ready"
    assert len(runs(checkout)) == 1 and bank.renders == 1 and other.renders == 0
    # the manifest lists every tag, its variants and each file's URL
    tags = manifest["tags"]
    assert list(tags) == list(voice.TAGS)
    assert {tag: len(entry["variants"]) for tag, entry in tags.items()} == {
        "alarm": 10,
        "greet": 12,
        "inquire": 10,
        "peck": 10,
        "chirp": 12,
        "coo": 10,
        "wheee": 6,
    }
    assert tags["chirp"]["variants"]["a"] == f"/api/voice/v5/{DANHUANG_VOICE}/chirp/chirp_a.wav"
    assert tags["wheee"]["segmented"] is True
    assert tags["wheee"]["variants"]["f"] == {
        part: f"/api/voice/v5/{DANHUANG_VOICE}/wheee/wheee_{part}_f.wav"
        for part in ("start", "loop", "end")
    }


def test_an_incomplete_or_foreign_bank_is_rendered_again(tmp_path, bank):
    checkout = tmp_path / "microduck"
    folder = tmp_path / "voices" / "v5" / f"seed{DANHUANG_VOICE}"
    (folder / "chirp").mkdir(parents=True)
    (folder / "chirp" / "chirp_a.wav").write_bytes(b"RIFF....")
    (folder / ".seed").write_text(f"{DANHUANG_VOICE}:4")  # another bank version's marker
    assert not bank.ready(DANHUANG_VOICE)
    ready_bank(bank, DANHUANG_SEED)
    assert len(runs(checkout)) == 1
    assert (folder / ".seed").read_text() == f"{DANHUANG_VOICE}:5"
    # a file gone from a finished bank: rendered again
    (folder / "coo" / "coo_c.wav").unlink()
    fresh = voice.VoiceBank.from_microduck(checkout, tmp_path / "voices")
    assert not fresh.ready(DANHUANG_VOICE)
    ready_bank(fresh, DANHUANG_SEED)
    assert len(runs(checkout)) == 2 and (folder / "coo" / "coo_c.wav").is_file()


@pytest.mark.parametrize(("mode", "said"), [("fail", "出錯"), ("short", "少寫了 1 個聲音檔")])
def test_a_failed_render_leaves_nothing_and_says_why(tmp_path, bank, mode, said):
    checkout = tmp_path / "microduck"
    (checkout / "target" / "debug" / "sounds.mode").write_text(mode)
    manifest = bank.manifest(DANHUANG_SEED, wait_s=30)
    assert manifest["state"] == "failed" and not manifest["ready"] and "tags" not in manifest
    assert said in manifest["note"] and manifest["retry_ms"] > 0
    base = tmp_path / "voices" / "v5"
    assert sorted(p.name for p in base.iterdir()) == [".render.lock"], "no half bank, no staging"
    assert bank.prepare(DANHUANG_SEED) == "failed", "not retried at once"
    assert len(runs(checkout)) == 1
    assert bank.handle(f"/api/voice/v5/{DANHUANG_VOICE}/chirp/chirp_a.wav", DANHUANG_SEED) == (
        voice.NOT_FOUND
    )


def test_one_render_at_a_time_on_this_machine(tmp_path):
    checkout = make_checkout(tmp_path)
    # two simulators (two banks over one cache) asking for the same duck at once, twice each
    banks = [voice.VoiceBank.from_microduck(checkout, tmp_path / "voices") for _ in range(2)]
    errors: list[BaseException] = []

    def ask(which: voice.VoiceBank) -> None:
        try:
            which.render(DANHUANG_VOICE)
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=ask, args=(banks[i % 2],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors
    assert len(runs(checkout)) == 1, "the lock: one render, the others found it done"
    assert all(which.ready(DANHUANG_VOICE) for which in banks)


def test_without_the_binary_the_page_is_told_why_in_plain_words(tmp_path):
    none = voice.VoiceBank.from_microduck(None, tmp_path / "voices")
    manifest = none.manifest(DANHUANG_SEED)
    assert manifest["state"] == "unavailable" and not manifest["ready"]
    assert "--microduck" in manifest["note"] and "不會出聲" in manifest["note"]
    unbuilt = tmp_path / "unbuilt"
    (unbuilt / "sounds" / "src").mkdir(parents=True)
    bank = voice.VoiceBank.from_microduck(unbuilt, tmp_path / "voices")
    assert bank.sounds_bin is None and "target/debug/sounds" in bank.manifest(7)["note"]
    newer = voice.VoiceBank.from_microduck(make_checkout(tmp_path / "v6", 6), tmp_path / "voices")
    assert newer.manifest(7)["state"] == "unavailable" and "第 6 版" in newer.manifest(7)["note"]
    assert voice.VoiceBank.off().manifest(7)["note"] == voice.OFF_ZH
    # a bank already in the cache is still served without the binary
    ready_bank(
        voice.VoiceBank.from_microduck(make_checkout(tmp_path / "b"), tmp_path / "voices"), 7
    )
    assert voice.VoiceBank.from_microduck(None, tmp_path / "voices").manifest(7)["ready"]


# ---------------------------------------------------------------- the routes


def test_only_the_current_ducks_whitelisted_files_are_served(tmp_path, bank):
    manifest = ready_bank(bank, DANHUANG_SEED)
    ready_bank(bank, 9)  # another duck's bank, also in the cache
    folder = bank.bank_dir(DANHUANG_VOICE)
    assert folder is not None
    for tag, entry in manifest["tags"].items():
        for variant in entry["variants"].values():
            for url in variant.values() if isinstance(variant, dict) else [variant]:
                reply = bank.handle(url, DANHUANG_SEED)
                assert reply is not None and reply.status == 200, url
                assert reply.content_type == "audio/wav" and "max-age" in reply.cache
                assert reply.body == (folder / tag / url.rsplit("/", 1)[1]).read_bytes()
    base = f"/api/voice/v5/{DANHUANG_VOICE}"
    refused = [
        f"{base}/chirp/chirp_m.wav",  # not a variant upstream renders
        f"{base}/chirp/coo_a.wav",  # another tag's file
        f"{base}/wheee/wheee_a.wav",  # not a segment
        f"{base}/chirp/../coo/coo_a.wav",
        f"{base}/chirp/%2e%2e/coo/coo_a.wav",
        f"{base}/chirp/..%2fcoo%2fcoo_a.wav",
        f"{base}/chirp/%2e%2e%2f.seed",
        f"{base}/.seed",
        f"{base}/chirp/.seed",
        f"{base}/chirp",
        f"{base}/chirp/",
        f"{base}//chirp/chirp_a.wav",
        f"{base}/chirp/chirp_a.wav/",
        f"{base}/CHIRP/chirp_a.wav",
        f"{base}/chirp/chirp_a.WAV",
        f"/api/voice/v4/{DANHUANG_VOICE}/chirp/chirp_a.wav",  # another bank version
        f"/api/voice/v5/{voice.voice_seed(9)}/chirp/chirp_a.wav",  # another duck's bank
        "/api/voice/v5/7/chirp/chirp_a.wav",  # the duck's seed is not its voice seed
        "/api/voice/v5/seed7/chirp/chirp_a.wav",
        "/api/voice/../index.html",
        "/api/voice/v5",
    ]
    for path in refused:
        assert bank.handle(path, DANHUANG_SEED) == voice.NOT_FOUND, path
    assert bank.handle("/api/voicex", DANHUANG_SEED) is None  # not a voice route at all
    assert bank.handle("/api/robot", DANHUANG_SEED) is None
    assert bank.handle(f"{base}/chirp/chirp_a.wav", None) == voice.NOT_FOUND
    assert bank.handle("/api/voice/manifest", None).status == 503


@pytest.fixture()
def served(tmp_path):
    bank = voice.VoiceBank.from_microduck(make_checkout(tmp_path), tmp_path / "voices")
    state = SimState(Config(seed=DANHUANG_SEED, name="測試"), voice_bank=bank)
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], state
    httpd.shutdown()
    httpd.server_close()


def raw_get(port: int, path: str) -> tuple[int, bytes, dict[str, str]]:
    """A GET with the path exactly as written (no client-side normalisation)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest("GET", path, skip_accept_encoding=True)
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body, dict(response.getheaders())


def call(port: int, method: str, params: dict | None = None) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    body = json.dumps({"method": method, "params": params or {}}).encode()
    connection.request("POST", "/api/call", body, {"Content-Type": "application/json"})
    reply = json.loads(connection.getresponse().read())
    connection.close()
    return reply


def test_the_server_serves_the_current_ducks_voice_and_follows_a_new_duck(served):
    port, state = served
    status, body, headers = raw_get(port, "/api/voice/manifest?t=1")
    manifest = json.loads(body)
    if manifest["state"] == "rendering":  # (the server started its render in the background)
        state.voice.wait(DANHUANG_SEED, 30)
        manifest = json.loads(raw_get(port, "/api/voice/manifest")[1])
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert manifest["ready"] and manifest["voice_seed"] == DANHUANG_VOICE
    url = manifest["tags"]["coo"]["variants"]["c"]
    status, body, headers = raw_get(port, url)
    assert status == 200 and body[:4] == b"RIFF" and headers["Content-Type"] == "audio/wav"
    assert headers["X-Content-Type-Options"] == "nosniff"
    for path in (
        f"/api/voice/v5/{DANHUANG_VOICE}/chirp/../coo/coo_a.wav",
        f"/api/voice/v5/{DANHUANG_VOICE}/chirp/%2e%2e%2f%2e%2e%2f.seed",
        f"/api/voice/v5/{DANHUANG_VOICE}/.seed",
        "/api/voice/../../pyproject.toml",
        "/api/voice/v5/../app.js",
    ):
        assert raw_get(port, path)[0] == 404, path
    # a new duck: its own voice; the old duck's files are no longer the page's
    assert call(port, "new", {"seed": 9, "name": "Pip"})["ok"]
    state.voice.wait(9, 30)
    manifest = json.loads(raw_get(port, "/api/voice/manifest")[1])
    assert manifest["ready"] and manifest["duck_seed"] == 9
    assert manifest["voice_seed"] == voice.voice_seed(9) != DANHUANG_VOICE
    assert raw_get(port, url)[0] == 404
    assert raw_get(port, manifest["tags"]["coo"]["variants"]["c"])[0] == 200


def test_the_feed_carries_the_calls_and_words_make_none(served):
    port, _state = served
    reply = call(port, "act", {"action": "owner_hello", "persona": "owner"})
    assert reply["ok"]
    events = reply["feed"]["events"]
    voiced = {item["kind"]: item["calls"] for item in events if item.get("calls")}
    assert voiced.get("duck.greeting"), events  # 主人專屬招呼: its sounds
    for item in events:
        if item["kind"] in {"owner.hello", "learning.committed"}:
            assert "calls" not in item, item
    # goodnight: 晚安咕～咕～
    reply = call(port, "act", {"action": "power_off", "persona": "owner"})
    calls = [item["calls"] for item in reply["feed"]["events"] if item["kind"] == "duck.bedtime"]
    assert calls == [["coo", "coo"]]


def test_calls_match_the_events_in_both_engines(tmp_path):
    """The frozen v4_0 duck (蛋黃's engine) records its sounds the same way."""
    from duckmind.duckcore.api import open_session

    state = SimState(Config(seed=DANHUANG_SEED, name="測試"))
    for session in (state.session, open_session({"engine_rev": "v4_0", "seed": 7, "name": "v4"})):
        state.session = session
        before = len(session.core.log.events)
        dispatch(session, "act", {"action": "owner_hello", "persona": "owner", "params": {}})
        items = state.annotate(feed(session, before, 0))["events"]
        greeting = [item for item in items if item["kind"] == "duck.greeting"]
        assert greeting and greeting[0]["calls"], items
        payload = session.core.log.events[greeting[0]["seq"]].payload
        assert greeting[0]["calls"] == list(payload["sounds"])


# ---------------------------------------------------------------- the real binary


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
def test_the_real_binary_renders_the_official_bank(tmp_path):
    bank = voice.VoiceBank.from_microduck(upstream_microduck(), tmp_path / "voices")
    manifest = ready_bank(bank, DANHUANG_SEED)
    folder = bank.bank_dir(DANHUANG_VOICE)
    assert folder is not None and manifest["voice_seed"] == DANHUANG_VOICE
    for tag, names in voice.WHITELIST.items():
        for name in names:
            with wave.open(str(folder / tag / name)) as sound:
                assert (sound.getframerate(), sound.getnchannels(), sound.getsampwidth()) == (
                    48_000,
                    1,
                    2,
                ), name
                assert sound.getnframes() > 1000, name
    assert sorted(p.name for p in folder.iterdir()) == [".seed", *sorted(voice.TAGS)]


# ---------------------------------------------------------------- the page's side


def test_the_page_has_the_voice_controls_and_hooks(served):
    port, _state = served
    static = Path(voice.__file__).parent / "static"
    page = (static / "index.html").read_text()
    for hook in ('id="callSound"', 'id="callMute"', 'aria-pressed="false"', 'id="callNote"'):
        assert hook in page, hook
    assert 'type="range" id="callVolume"' in page
    assert page.index('src="/voice.js"') < page.index('src="/app.js"'), "duckVoice before its users"
    status, body, headers = raw_get(port, "/voice.js")
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")
    script = body.decode()
    assert "window.duckVoice = { call, mouth, stop" in script
    assert '"/api/voice/manifest"' in script and 'KEY = "duckmind.voice"' in script
    # the 3D beak follows the sound; a clip's call starts with the clip; the 2D duck's are app.js's
    duck3d = (static / "duck3d.js").read_text()
    assert "window.duckVoice.mouth()" in duck3d and "voiceCall(motion.calls, duration)" in duck3d
    assert "calls: item.calls" in duck3d
    assert "function voice2d(calls" in (static / "app.js").read_text()
    # design standard: tokens only, square corners, hard shadows (docs/simulator-style.md)
    css = (static / "style.css").read_text()
    rules = [line for line in css.splitlines() if "duck3d-voice" in line or ".call-" in line]
    assert rules and not [line for line in rules if "#" in line.split("{", 1)[-1]]
    assert all("border-radius: 0" in line for line in rules if "border-radius" in line)
    assert not [line for line in rules if "blur" in line or "rgba(0" in line]
