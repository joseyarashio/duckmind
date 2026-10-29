"""neuro:web — the page's demo mode for the neural-duck prototype (「神經鴨原型（示範）」).

* the picker's four ducks come from Duck Core's own diversity design (each title is checked
  against the genome the brain gets), and every card's words come from that genome;
* the /api/neuro/… routes answer "off" on a normal server, serve the page's schema with the stub
  (``--neuro-stub``), stream it, take stimuli and switch ducks;
* :class:`DriverSource` turns the real brain's telemetry (``NeuroBrain.step``, stepped here on
  synthetic sensor frames: a clap) into the page's bars and words, each reflex said once;
* the page's markup keeps Duck Core's hooks and puts the prototype's parts behind
  ``[data-neuro]``, its styles use the design tokens only.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from duckmind.duckcore.api import Config
from duckmind.neuro import contract as C
from duckmind.simweb import neuro_api as N
from duckmind.simweb.server import SimServer, SimState, make_handler

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"


# ── the picker: Duck Core's diversity design ─────────────────────────────────────


def _genome(choice: N.DuckChoice) -> dict[str, Any]:
    return N.genome_summary(choice.seed, choice.gd)


def test_the_four_ducks_are_what_their_titles_say():
    ducks = {item.preset: _genome(item) for item in N.PRESETS}
    brave = ducks["explorer"]
    assert brave["stage"] == "juvenile" and brave["branch"] == "explorer"
    assert brave["axes"]["boldness"] >= 0.7 and brave["axes"]["exploration"] >= 0.7
    shy = ducks["snuggler"]
    assert shy["stage"] == "duckling" and shy["primary"] == "snuggler" and shy["branch"] is None
    assert shy["axes"]["boldness"] <= 0.3 and shy["axes"]["sociability"] >= 0.7
    slow = ducks["dreamer"]
    assert slow["stage"] == "adult" and slow["branch"] == "dreamer"
    assert slow["axes"]["activity"] <= 0.3 and slow["axes"]["chronotype"] >= 0.7
    assert slow["quirk_tempo"] <= 0.92  # its moves play slowly
    chick = ducks["hatchling"]
    assert chick["stage"] == "hatchling" and chick["primary"] == "songbird"
    assert chick["axes"]["sociability"] >= 0.7
    # Four natures, four stages, both innate sides.
    assert {d["primary"] for d in ducks.values()} == {"explorer", "snuggler", "dreamer", "songbird"}
    assert len({d["stage"] for d in ducks.values()}) == 4
    assert {d["side"] for d in ducks.values()} == {"左", "右"}


def test_a_ducks_card_is_its_genome_in_plain_words():
    card = N.duck_card(N.PRESET_BY_ID["snuggler"])
    assert card["name"] == "怯怯" and card["seed"] == 1728 and card["spec"] == "1728:20"
    assert card["stage_zh"] == "幼鴨" and card["branch_zh"] == "" and card["nature_zh"] == "依偎型"
    assert card["line_zh"].startswith("很膽小") and "幼鴨（天性：依偎型）" in card["line_zh"]
    assert card["line_zh"].endswith("歪頭習慣往右邊")
    assert [axis["key"] for axis in card["axes"]] == list(N.content.TEMPERAMENT["axes"])
    # The same seed is the same duck (Duck Core draws it from the seed alone).
    assert N.duck_card(N.PRESET_BY_ID["snuggler"]) == card
    # A card made from the brain's own genome summary names a preset it matches.
    assert N.card_of(_genome(N.PRESET_BY_ID["dreamer"]))["name"] == "慢慢"


def test_a_random_duck_can_be_any_age():
    picks = iter([123456, 0, 5])  # seed, stage index (hatchling), offset in the stage
    choice = N.random_choice(lambda n: next(picks) % n)
    assert choice.seed == 123456 and 1 <= choice.gd < 12
    assert N.duck_card(choice)["stage"] == "hatchling"
    with pytest.raises(ValueError):
        N.choice_from({"preset": "nobody"})
    assert N.choice_from({"seed": 42, "gd": 70}).spec == "42:70"
    assert N.choice_of_spec("", 7) == N.DuckChoice(7, 60, "小鴨007")
    assert N.choice_of_spec("4107:60", 7).name == "阿勇"
    assert N.choice_of_spec("dreamer", 7).seed == 4670


# ── the routes ───────────────────────────────────────────────────────────────────


def _serve(hub: N.NeuroHub | None):
    state = SimState(Config(seed=7, name="測試"))
    httpd = SimServer(("127.0.0.1", 0), make_handler(state, hub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def _post(url: str, body: dict[str, Any]) -> tuple[int, Any]:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_a_normal_server_says_the_prototype_is_off():
    httpd, base = _serve(None)
    try:
        assert _get(base + "/api/neuro/status") == {"ok": True, "enabled": False, "schema": 1}
        assert _get(base + "/api/neuro/telemetry")["enabled"] is False
        status, answer = _post(base + "/api/neuro/stimulus", {"kind": "clap"})
        assert status == 400 and "--brain neuro" in answer["error"]
        # Duck Core's own routes are untouched.
        assert _get(base + "/api/ping")["ok"] is True
    finally:
        httpd.shutdown()


def test_the_stub_serves_the_page_its_stream_its_stimuli_and_other_ducks():
    hub = N.NeuroHub(enabled=True)
    hub.attach(N.StubSource(hub.card()))
    httpd, base = _serve(hub)
    try:
        status = _get(base + "/api/neuro/status")
        assert status["enabled"] and status["source"] == "stub" and status["label_zh"] == N.LABEL_ZH
        assert [s["key"] for s in status["stimuli"]] == [
            "clap", "call_name", "wave", "pat_head", "roll_ball", "novel_object", "lights",
            "family_walk",
        ]  # fmt: skip
        assert {d["preset"] for d in status["ducks"]} == set(N.PRESET_BY_ID)
        first = _post(base + "/api/neuro/stimulus", {"kind": "clap"})[1]
        again = _post(base + "/api/neuro/stimulus", {"kind": "clap"})[1]
        assert again["strength"] < first["strength"]  # (the stub habituates too)
        t = _get(base + "/api/neuro/telemetry?since=0")
        assert "假的" in t["note_zh"]  # made-up numbers say so, always
        assert [d["zh"] for d in t["drives"]][:7] == [
            "精力",
            "睏",
            "好奇",
            "害怕",
            "想家人",
            "舒服",
            "無聊",
        ]
        assert len(t["reflexes"]) == 2 and t["seq"] >= 2
        assert _get(base + f"/api/neuro/telemetry?since={t['seq']}")["reflexes"] == []
        assert _post(base + "/api/neuro/stimulus", {"kind": "nothing"})[0] == 400
        code, switched = _post(base + "/api/neuro/duck", {"preset": "hatchling"})
        assert code == 200 and switched["duck"]["name"] == "啾啾"
        assert _get(base + "/api/neuro/status")["duck"]["name"] == "啾啾"
        assert _post(base + "/api/neuro/duck", {"preset": "nobody"})[0] == 400
        with urllib.request.urlopen(base + "/api/neuro/stream?since=0", timeout=10) as stream:
            assert stream.headers["Content-Type"].startswith("text/event-stream")
            got = []
            for raw in stream:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    got.append(json.loads(line[6:]))
                if len(got) == 2:
                    break
        assert got[0]["enabled"] and got[0]["duck"]["name"] == "啾啾"
    finally:
        hub.stop()
        httpd.shutdown()


# ── the real brain through DriverSource ──────────────────────────────────────────


class _Driver:
    """What NeuroDriver.telemetry gives, from the real brain on synthetic frames."""

    def __init__(self, choice: N.DuckChoice) -> None:
        from duckmind.neuro.brain import NeuroBrain
        from duckmind.neuro.genome import Genome

        self.genome = Genome.from_seed(choice.seed, gd=choice.gd)
        self.brain = NeuroBrain(self.genome, choice.seed)
        self.seq = 0
        self.vocals: list[dict[str, Any]] = []
        self.latest: dict[str, Any] = {}
        self.injected: list[str] = []
        self._last_vocal = 0

    def run(self, t0_ms: int, t1_ms: int, clap_at_ms: int | None = None) -> None:
        for t_ms in range(t0_ms, t1_ms, 20):
            sound = C.SoundSense()
            if clap_at_ms is not None and clap_at_ms <= t_ms < clap_at_ms + 60:
                sound = C.SoundSense(level_db=78.0, onset=1.0, kind="clap")
            frame = C.SensorFrame(t_ms=t_ms, sound=sound)
            cmd, tel = self.brain.step(t_ms, frame)
            if cmd.vocal and cmd.vocal_seq != self._last_vocal:
                self._last_vocal = cmd.vocal_seq
                self.seq += 1
                self.vocals.append({"seq": self.seq, "t_ms": t_ms, "tag": cmd.vocal})
            self.latest = {
                "brain": tel.to_json(),
                "frame": frame.to_json(),
                "command": cmd.to_json(),
            }
        self.seq += 1

    def telemetry(self, since: int = 0) -> dict[str, Any]:
        return {
            "state": "running",
            "error": None,
            "lanes": {"brain": "real", "senses": "stub", "motor": "real"},
            "notes": [],
            "genome": self.genome.summary(),
            "stats": {"hz": 50.0},
            "realtime": 0.99,
            "seq": self.seq,
            "vocals": [v for v in self.vocals if v["seq"] > since],
            "events": [],
            **self.latest,
        }

    def inject(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.injected.append(name)
        return {"note_zh": f"世界：{name}"}


def test_the_real_brains_telemetry_becomes_the_page():
    driver = _Driver(N.PRESET_BY_ID["snuggler"])
    source = N.DriverSource(lambda: driver, clock=iter(range(0, 10_000)).__next__)
    driver.run(0, 3000)
    quiet = source.snapshot(0)
    assert quiet["source"] == "brain" and quiet["duck"]["name"] == "怯怯"
    assert "感覺還是替身" in quiet["note_zh"]  # a stand-in lane is said on the page
    driver.run(3000, 6000, clap_at_ms=3500)
    t = source.snapshot(0)
    assert t["behaviour"]["zh"] and t["behaviour"]["why_zh"]
    assert [d["zh"] for d in t["drives"] if d["main"]] == [zh for _, zh in N.DRIVES]
    assert all(0.0 <= d["value"] <= 1.0 for d in t["drives"])
    assert 0.0 <= t["arousal"]["value"] <= 1.0
    assert 1 <= len(t["candidates"]) <= 6 and all(0.0 <= c["pull"] <= 1.0 for c in t["candidates"])
    heard = [r for r in t["reflexes"] if r.get("stimulus", "").startswith("sound:")]
    assert heard, t["reflexes"]  # the clap reached a reflex, through the sound sense only
    clap = next(h for h in t["habituation"] if h["key"] == "sound:clap")
    assert clap["count"] >= 1
    sound = next(s for s in t["senses"] if s["key"] == "sound")
    assert sound["modelled"] and sound["source"] == "model"
    # Each reflex is said once: a later snapshot from the last number has none of them again.
    later = source.snapshot(t["seq"])
    assert later["reflexes"] == []
    # The buttons go to the driver's world-side injection, with the contract's names.
    assert source.stimulus("pat_head", {})["note_zh"] == "世界：pet_head"
    assert source.stimulus("lights", {})["lights_on"] is False
    assert source.stimulus("lights", {})["lights_on"] is True
    assert driver.injected == ["pet_head", "lights_off", "lights_on"]


def test_a_new_duck_is_a_new_set_of_lanes_on_the_same_driver():
    from duckmind.simweb.neuro_live import NeuroSpec, load_lanes

    class Holder:
        def __init__(self) -> None:
            self.spec = NeuroSpec(duck="7:60", seed=7)
            self.lanes = load_lanes(self.spec)

    holder = Holder()
    answer = N.switch_driver(holder, N.PRESET_BY_ID["dreamer"])
    assert holder.spec.duck == "4670:200"
    assert holder.lanes.names["brain"] == "real" and answer["lanes"]["brain"] == "real"
    assert holder.lanes.genome is not None and holder.lanes.genome["seed"] == 4670


# ── the page ─────────────────────────────────────────────────────────────────────


def test_the_page_has_the_prototypes_parts_and_keeps_duck_cores_hooks():
    page = (STATIC / "index.html").read_text()
    stim = page[
        page.index('id="neuroStim"') : page.index("</section>", page.index('id="neuroStim"'))
    ]
    assert re.findall(r'data-stim="(\w+)"', stim) == [item.key for item in N.STIMULI]
    for label in ("拍手", "叫名字", "揮手", "摸頭", "滾球過去", "放一個新東西", "家人走過"):
        assert f">{label}<" in stim
    at = page.index('id="neuroStim"')
    tag = page[page.rindex("<section", 0, at) : page.index(">", at)]
    assert "data-scenarios" in tag  # (scenario controls live in a [data-scenarios] container)
    for hook in (
        'id="neuroSide"',
        'id="neuroBrain"',
        'id="neuroReflexes"',
        'id="neuroCands"',
        'id="neuroDrives"',
        'id="neuroHabits"',
        'id="neuroPlaces"',
        'id="neuroPicker"',
        'id="neuroDucks"',
        "神經鴨原型（示範）",
        "小鴨的腦",
        "換一隻小鴨",
    ):
        assert hook in page, hook
    assert '<script src="/neuro.js"></script>' in page
    # Duck Core's cards step aside in the prototype's mode — and keep every hook.
    for hook in ('id="duckBanner"', 'id="scenarios"', 'id="lab"', 'id="btnPlay"', 'id="chipGd"'):
        assert hook in page, hook
    assert page.count("data-core") >= 15 and page.count("data-neuro") >= 10
    script = (STATIC / "neuro.js").read_text()
    assert '"/api/neuro/status"' in script and "/api/neuro/stream" in script
    assert "/api/neuro/telemetry" in script and '"duck:neuro"' in script


def test_the_prototypes_styles_use_only_the_design_tokens():
    css = (STATIC / "style.css").read_text()
    block = css[css.index("/* ---------- 神經鴨原型") : css.index("/* ---------- desktop touches")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert not re.search(r"\b(rgba?|hsla?)\(", block)
    assert "border-radius" not in block and "blur(" not in block
    assert "body.neuro-mode [data-core] { display: none !important; }" in block


def test_the_walking_strip_shows_bursts_and_pauses_a_second_a_bar():
    # 10 Hz samples: 3 s walking at 0.4 then 0.5 m/s (turning left in the last), 2 s still,
    # 1 s walking again.
    walk = []
    for i in range(60):
        t_ms = i * 100
        second = i // 10
        vx = {0: 0.4, 1: 0.45, 2: 0.5, 5: 0.35}.get(second, 0.0)
        vyaw = 0.6 if second == 2 else 0.0
        walk.append((t_ms, vx, vyaw))
    strip = N.rhythm_of(walk, window_s=6)
    assert strip["bins"] == [0.4, 0.45, 0.5, 0.0, 0.0, 0.35]
    assert strip["turns"][2] == 0.6 and strip["turns"][0] == 0.0
    assert strip["share"] == 0.667
    assert N.rhythm_of([])["bins"] == []


def test_the_command_line_can_name_the_pages_ducks():
    import argparse

    args = argparse.Namespace(brain="neuro", neuro_duck="dreamer", neuro_stub=False, seed=7)
    hub = N.setup(args, lambda: None)
    assert args.neuro_duck == "4670:200"  # (what the driver, made after this, is told)
    assert hub.enabled and hub.choice.name == "慢慢" and isinstance(hub.source, N.DriverSource)
    off = N.setup(argparse.Namespace(brain="duckcore", neuro_duck="", neuro_stub=False, seed=7))
    assert not off.enabled and off.source is None
    stub = N.setup(argparse.Namespace(brain="duckcore", neuro_duck="", neuro_stub=True, seed=7))
    assert stub.enabled and isinstance(stub.source, N.StubSource)


# ── the server's --name is the name of the duck it starts with (fix/duck-name) ──


def _names(cards: list[dict[str, Any]]) -> dict[str, str]:
    return {card["preset"]: card["name"] for card in cards}


def _args(**given: Any):
    import argparse

    base: dict[str, Any] = {"brain": "duckcore", "neuro_duck": "explorer", "seed": 4107}
    return argparse.Namespace(**{**base, "neuro_stub": True, **given})


def test_without_a_given_name_every_duck_keeps_its_own_name():
    hub = N.setup(_args())
    assert hub.card()["name"] == "阿勇" and hub.status()["duck"]["name"] == "阿勇"
    assert _names(hub.status()["ducks"]) == {
        "explorer": "阿勇",
        "snuggler": "怯怯",
        "dreamer": "慢慢",
        "hatchling": "啾啾",
    }
    assert hub.telemetry()["duck"]["name"] == "阿勇"
    # (a blank name is no name)
    assert N.setup(_args(name_given="  ")).card()["name"] == "阿勇"


def test_a_given_name_is_the_card_the_status_and_the_telemetry_of_the_servers_duck():
    hub = N.setup(_args(name_given="蛋黃"))
    assert hub.choice.name == "蛋黃" and hub.choice.preset == "explorer"
    assert (hub.choice.seed, hub.choice.gd) == (4107, 60)  # the chosen genome is unchanged
    assert hub.card()["name"] == "蛋黃"
    assert hub.status()["duck"]["name"] == "蛋黃"
    assert hub.telemetry()["duck"]["name"] == "蛋黃"
    card = hub.card()
    assert card["preset"] == "explorer" and card["title_zh"] == N.PRESET_BY_ID["explorer"].title_zh
    # the picker: the server's own duck under its name, the other three keep theirs
    assert _names(hub.status()["ducks"]) == {
        "explorer": "蛋黃",
        "snuggler": "怯怯",
        "dreamer": "慢慢",
        "hatchling": "啾啾",
    }
    assert _names(N.preset_cards())["explorer"] == "阿勇"  # (the shared table is not touched)
    assert N.PRESET_BY_ID["explorer"].name == "阿勇"


def test_the_drivers_own_card_says_the_given_name_too():
    choice = N.PRESET_BY_ID["explorer"]
    driver = _Driver(choice)
    named = N.NeuroHub(enabled=True, choice=choice, given_name="蛋黃")
    named.attach(N.DriverSource(lambda: driver, clock=iter(range(0, 10_000)).__next__))
    plain = N.NeuroHub(enabled=True, choice=choice)
    plain.attach(N.DriverSource(lambda: driver, clock=iter(range(0, 10_000)).__next__))
    driver.run(0, 3000)
    assert named.telemetry()["duck"]["name"] == "蛋黃"
    assert named.telemetry()["duck"]["seed"] == 4107  # (only the name differs)
    assert plain.telemetry()["duck"]["name"] == "阿勇"
    # another duck's genome is never given the name
    other = _Driver(N.PRESET_BY_ID["snuggler"])
    other.run(0, 3000)
    hub = N.NeuroHub(enabled=True, choice=choice, given_name="蛋黃")
    hub.attach(N.DriverSource(lambda: other, clock=iter(range(0, 10_000)).__next__))
    assert hub.telemetry()["duck"]["name"] == "怯怯"


def test_the_given_name_stays_with_the_servers_duck_when_the_family_switches():
    hub = N.setup(_args(name_given="蛋黃"))
    assert hub.switch({"preset": "snuggler"})["duck"]["name"] == "怯怯"
    assert hub.card()["name"] == "怯怯" and hub.telemetry()["duck"]["name"] == "怯怯"
    assert _names(hub.status()["ducks"])["explorer"] == "蛋黃"
    back = hub.switch({"preset": "explorer"})
    assert back["duck"]["name"] == "蛋黃" and hub.card()["name"] == "蛋黃"
    assert "蛋黃" in back["note_zh"]
    # a duck asked for by seed keeps the name asked, even at the servers duck's seed and day
    asked = hub.switch({"seed": 4107, "gd": 60, "name": "小黑"})
    assert asked["duck"]["name"] == "小黑"
    hub.switch({"seed": 4107, "gd": 60})
    assert hub.card()["name"] != "蛋黃"


def test_a_given_name_names_a_duck_picked_by_seed_too_and_is_cut_at_twelve():
    hub = N.setup(_args(neuro_duck="7:62", seed=7, name_given="蛋黃"))
    assert hub.card()["name"] == "蛋黃" and hub.card()["spec"] == "7:62"
    assert _names(hub.status()["ducks"])["explorer"] == "阿勇"
    long = N.setup(_args(name_given="一二三四五六七八九十甲乙丙丁"))
    assert long.card()["name"] == "一二三四五六七八九十甲乙"


def test_the_given_name_reaches_the_page_over_http():
    hub = N.setup(_args(name_given="蛋黃"))
    httpd, base = _serve(hub)
    try:
        status = _get(base + "/api/neuro/status")
        assert status["duck"]["name"] == "蛋黃"
        assert _names(status["ducks"])["explorer"] == "蛋黃"
        assert _get(base + "/api/neuro/telemetry?since=0")["duck"]["name"] == "蛋黃"
        with urllib.request.urlopen(base + "/api/neuro/stream?since=0", timeout=10) as stream:
            for raw in stream:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    assert json.loads(line[6:])["duck"]["name"] == "蛋黃"
                    break
    finally:
        hub.stop()
        httpd.shutdown()


def test_the_command_line_name_is_given_only_when_it_is_given(monkeypatch):
    from duckmind.simweb import server

    class _Stop(Exception):
        pass

    seen: list[Any] = []

    def stop_at_setup(args, *rest, **more):
        seen.append(args)
        raise _Stop

    monkeypatch.setattr(server.neuro_api, "setup", stop_at_setup)
    base = ["--port", "0", "--state-file", ""]
    for extra, given, config_name in (
        ([], "", "小鴨"),
        (["--name", "蛋黃"], "蛋黃", "蛋黃"),
        (["--name", "小鴨"], "小鴨", "小鴨"),  # (asking for the default name is asking)
    ):
        with pytest.raises(_Stop):
            server.main([*base, *extra])
        assert seen[-1].name_given == given and seen[-1].name == config_name


# ── integration 2026-09-28 ───────────────────────────────────────────────────────


class _Source:
    kind = "brain"

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.days: list[float] = []
        self.helped = 0

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        return {"enabled": True, "seq": 0}

    def stimulus(self, kind: str, params: Any) -> dict[str, Any]:
        self.asked.append(kind)
        return {"note_zh": kind}

    def switch(self, choice: Any, card: Any) -> dict[str, Any]:
        return {}

    def set_day(self, minute: float) -> dict[str, Any]:
        self.days.append(minute)
        return {"minute": minute, "zh": "上午 10:00"}

    def help_up(self) -> dict[str, Any]:
        self.helped += 1
        return {"note_zh": "一隻手過去把牠扶起來"}

    def stop(self) -> None:
        return


def test_the_page_names_and_the_contract_names_both_work():
    source = _Source()
    hub = N.NeuroHub(enabled=True, source=source)
    for kind in ("pet_head", "pat_head", "walk_by", "family_walk", "lights_off", "clap_train"):
        assert hub.stimulus({"kind": kind})["ok"]
    assert source.asked == [
        "pat_head",
        "pat_head",
        "family_walk",
        "family_walk",
        "lights_off",
        "clap_train",
    ]
    with pytest.raises(ValueError):
        hub.stimulus({"kind": "teleport"})
    more = {item["key"]: item["zh"] for item in hub.status()["more"]}
    assert {"tv", "hello", "knock", "push", "knock_train", "clap_train"} <= set(more)
    assert more["knock_train"].startswith("慢慢敲") and more["clap_train"].startswith("快快拍")


def test_the_duck_clock_and_the_helping_hand_have_routes():
    source = _Source()
    hub = N.NeuroHub(enabled=True, source=source)
    choices = hub.status()["day_choices"]
    assert [c["zh"] for c in choices] == ["早上", "上午", "下午", "傍晚", "晚上", "半夜"]
    assert hub.set_day({"choice": "上午"})["day"]["minute"] == 600.0
    assert hub.set_day({"minute": 900})["ok"] and source.days == [600.0, 900.0]
    with pytest.raises(ValueError):
        hub.set_day({"choice": "午夜場"})
    assert hub.help_up()["ok"] and source.helped == 1
    off = N.NeuroHub()
    with pytest.raises(ValueError):
        off.help_up()


def test_a_reflex_hold_is_said_with_its_duration_and_the_beak_is_forwarded():
    class Driver:
        def __init__(self) -> None:
            self.samples: list[dict[str, Any]] = []

        def telemetry(self, since: int = 0) -> dict[str, Any]:
            return {
                "state": "running",
                "lanes": {"brain": "real", "senses": "real", "motor": "real"},
                "seq": len(self.samples),
                "samples": [s for s in self.samples if s["seq"] > since],
                "brain": {"behaviour": "freeze", "t_ms": 0},
                "command": {},
                "frame": {},
                "day": {"minute": 600.0, "zh": "上午 10:00", "rate": 1.0},
                # (fid:fix) the hold is timed from the BODY by the driver's witness
                "witness": {"seq": len(self.holds), "audits": list(self.holds), "now": None},
            }

    driver = Driver()
    driver.holds = []  # type: ignore[attr-defined]
    source = N.DriverSource(lambda: driver, clock=iter(range(0, 10_000)).__next__)
    t = 0
    for behaviour, n in (("wander", 5), ("startle", 3), ("freeze", 6), ("wander", 3)):
        for _ in range(n):
            t += 100
            driver.samples.append(
                {
                    "seq": len(driver.samples) + 1,
                    "t_ms": t,
                    "behaviour": behaviour,
                    "mouth_peak": 0.3 if behaviour == "startle" else 0.0,
                    "righting": "waiting_help" if behaviour == "freeze" else "idle",
                    "tilt_deg": 50.0 if behaviour == "freeze" else 2.0,
                }
            )
        source.snapshot(0)
    said = [r["zh"] for r in source.snapshot(0)["reflexes"]]
    # (fid:fix) the act's label alone says nothing about a hold: the head may be moving in it
    assert not any(text.startswith("愣住 ") for text in said), said
    driver.holds.append(  # type: ignore[attr-defined]
        {
            "kind": "hold",
            "seq": 1,
            "act": "startle",
            "t_ms": 600,
            "still_s": 0.9,
            "zh": "愣住 0.9 秒",
        }
    )
    snap = source.snapshot(0)
    said = [r["zh"] for r in snap["reflexes"]]
    assert any(text.startswith("愣住 ") and text.endswith(" 秒") for text in said), said
    seconds = float(next(text for text in said if text.startswith("愣住 ")).split()[1])
    assert 0.8 <= seconds <= 1.1  # startle then freeze: one hold of ~0.9 s
    assert snap["day"]["zh"] == "上午 10:00"
    assert snap["mouth"]["t_ms"] and max(snap["mouth"]["open"]) in (0.0, 0.3)
    assert snap["body"]["down"] is False  # the last sample is up again
    assert N.body_state({"righting": "waiting_help", "tilt_deg": 55})["waiting_help"] is True


def test_the_page_has_the_clock_row_and_the_help_button():
    page = (STATIC / "index.html").read_text()
    stim = page[
        page.index('id="neuroStim"') : page.index("</section>", page.index('id="neuroStim"'))
    ]
    for hook in ('id="neuroHelpRow"', 'id="neuroHelp"', 'id="neuroDay"', 'id="neuroDayButtons"'):
        assert hook in stim  # inside the card (a [data-scenarios] container)
    assert ">扶牠起來<" in stim


def test_a_world_refusal_is_an_answer_in_words_not_a_failed_request() -> None:
    """The world's hand is busy (it just set the duck down): the button gets ok false and the
    world's words (the browser logs no failed request); an unknown stimulus is still a 400."""

    class Busy(_Source):
        def stimulus(self, kind: str, params: Any) -> dict[str, Any]:
            raise ValueError("家裡的手正在做別的事，等一下再試")

    hub = N.NeuroHub(enabled=True, source=Busy())
    answer = hub.stimulus({"kind": "push"})
    assert answer["ok"] is False and answer["refused"] and "等一下" in answer["error"]
    with pytest.raises(ValueError):
        hub.stimulus({"kind": "nothing"})
