"""The page's real-duck mode, page side (docs/real-microduck.md): with ``/api/body``'s ``real`` the page
holds every control the simulated world does off — the reason as its title and a visible line (touch has
no hover) — shows the link's quality and the duck's own battery, and the 3D label says it draws the
duck's own joints and odometry, never physics truth. With ``real`` null every control is back.

``static/realduck.js`` runs in Node on a small stand-in DOM; ``bodylink.js`` and ``truth.js`` too."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
# (the sound-scenario strip is built by neuro_sound.js, not written in the page)
SOUND_JS = (STATIC / "neuro_sound.js").read_text(encoding="utf-8")
JS = STATIC / "realduck.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")

REFUSES = [
    "battery_swap",
    "demo",
    "family",
    "gothere",
    "help_up",
    "layout",
    "reset",
    "stimuli",
    "truth_overlay",
    "world_buttons",
]
REAL = {
    "rehearsal": False,
    "reason": "這是真的小鴨：沒有模擬的世界，這件事做不到",
    "refuses": REFUSES,
    "battery": None,
}

# The stand-in page: each element with the gate selectors it matches (as the browser's matches() would).
HARNESS = r"""
const api = require(__JS__);
const t = (key) => `T:${key}`;
function el(id, sels, title, extra) {
  const attrs = {};
  if (title !== null && title !== undefined) attrs.title = title;
  Object.assign(attrs, extra || {});
  return {
    id, disabled: false, hidden: false, dataset: {}, _sels: new Set(sels), _attrs: attrs,
    getAttribute(n) { return n in this._attrs ? this._attrs[n] : null; },
    setAttribute(n, v) { this._attrs[n] = String(v); },
    removeAttribute(n) { delete this._attrs[n]; },
    matches(s) { return s.split(",").some((x) => this._sels.has(x.trim())); },
  };
}
const S = '#neuroStim [data-stim]';
const els = [
  el("pat_head", [S], "摸頭的說明"),
  el("family_walk", ['#neuroStim [data-stim="family_walk"]', S], null),
  el("swap", ['#neuroStim [data-stim="swap_battery"]', S], null),
  el("help", ["#neuroHelp"], null),
  el("carry", ["#neuroCarry"], null),
  el("world_show", ["#worldTools [data-world]"], ""),
  el("liveDemo", ["#liveDemo"], "示範的說明"),
  el("liveUpright", ["#liveUpright"], "扶起的說明", { "aria-describedby": "other" }),
  el("liveReset", ["#liveReset"], "重設的說明"),
  el("apply", ['#homeEditor [data-le="apply"]'], null),
  el("apply_yes", ['#homeEditor [data-le="apply-yes"]'], null),
  el("open", ['#homeEditor [data-le="open"]'], null),
  el("core_swap", ['#scenarios [data-act="swap_battery"]'], null),
  el("battery_help", ["#setBatteryHelp"], null),
  el("motion", [], "動作"),  // (a control the real duck can do: never touched)
  el("undo", [], null),
  el("sound_music", ["#neuroSound button[data-sound]"], "慢慢的歌"),  // (neuro_sound.js's strip)
  el("sound_hand", ["#soundHand"], null),
];
const notes = ["realNoteWorld", "realNoteLive", "realNoteLayout", "realNoteCore", "realNoteSound"].map((id) => {
  const n = el(id, [], null);
  n.hidden = true;
  return n;
});
const root = {
  querySelectorAll(s) {
    if (s === "[data-real-gate]") return els.filter((e) => e.dataset.realGate !== undefined);
    // (as a browser: an element's dataset.realNote IS a data-real-note attribute)
    if (s === "[data-real-note]") return [...notes, ...els.filter((e) => e.dataset.realNote !== undefined)];
    return els.filter((e) => e.matches(s));
  },
};
const snap = () => {
  const out = {};
  for (const e of [...els, ...notes]) {
    out[e.id] = { disabled: e.disabled, hidden: e.hidden, title: e.getAttribute("title"), described: e.getAttribute("aria-describedby"), gate: e.dataset.realGate || null };
  }
  return out;
};
const steps = {};
const real = api.realOf(__REAL__);
api.apply(root, real, t);
steps.real = snap();
steps.blocked = { pat_head: !!api.gateOf(real, els[0]), motion: !!api.gateOf(real, els[14]) };
api.apply(root, real, t);  // (twice: nothing changes)
steps.again = snap();
api.apply(root, api.realOf({ refuses: ["layout"] }), t);
steps.layoutOnly = snap();
api.apply(root, api.realOf(null), t);
steps.simulated = snap();
steps.undefinedKeeps = api.realOf(undefined) === undefined;
steps.links = [
  api.linkLine({ up: true, rtt_ms_p50: 64.6, drops: 1, max_gap_ms: 812.4 }),
  api.linkLine({ up: true, rtt_ms_p50: null, drops: 0, max_gap_ms: 0 }),
  api.linkLine({ up: false, drops: 3 }),
  api.linkLine(null),
];
steps.batteries = [
  api.batteryLine(null),
  api.batteryLine({ percent: null, charging: null }),
  api.batteryLine({ percent: 87.4, volts: 7.9, charging: null, source: "robotd", age_s: 1 }),
  api.batteryLine({ percent: 140, charging: true }),
];
console.log(JSON.stringify(steps));
"""


@pytest.fixture(scope="module")
def page() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("needs Node")
    script = HARNESS.replace("__JS__", json.dumps(str(JS))).replace("__REAL__", json.dumps(REAL))
    run = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


GATED = {
    "pat_head": ("stimuli", "neuroStimNote"),
    "family_walk": ("family", "neuroStimNote"),
    "swap": ("battery_swap", "neuroStimNote"),
    "help": ("help_up", "neuroHelpNote"),
    "carry": ("help_up", "neuroHelpNote"),
    "world_show": ("world_buttons", "realNoteWorld"),
    "liveDemo": ("demo", "realNoteLive"),
    "liveUpright": ("help_up", "realNoteLive"),
    "liveReset": ("reset", "realNoteLive"),
    "apply": ("layout", "realNoteLayout"),
    "apply_yes": ("layout", "realNoteLayout"),
    "open": ("layout", "realNoteLayout"),
    "core_swap": ("battery_swap", "realNoteCore"),
    "battery_help": ("battery_swap", "realNoteCore"),
    "sound_music": ("stimuli", "realNoteSound"),
    "sound_hand": ("stimuli", "realNoteSound"),
}


@needs_node
def test_a_real_duck_holds_every_world_control_off_with_the_reason_and_a_visible_line(
    page: dict[str, Any],
) -> None:
    real = page["real"]
    for name, (code, note) in GATED.items():
        got = real[name]
        assert got["disabled"] and got["gate"] == code, name
        # greyed out, never hidden (before: a held control matched the notes' [data-real-note] and hid)
        assert got["hidden"] is False, name
        assert got["title"] == "T:page.real.no_world", name
        assert note in (got["described"] or "").split(), name
    assert real["liveUpright"]["described"] == "other realNoteLive"  # (its own description stays)
    for note in (
        "realNoteWorld",
        "realNoteLive",
        "realNoteLayout",
        "realNoteCore",
        "realNoteSound",
    ):
        assert real[note]["hidden"] is False, note
    # what a real duck can do is never touched
    assert real["motion"] == {
        "disabled": False,
        "hidden": False,
        "title": "動作",
        "described": None,
        "gate": None,
    }
    assert real["undo"]["disabled"] is False
    assert page["blocked"] == {"pat_head": True, "motion": False}
    assert page["again"] == real


@needs_node
def test_only_the_refused_features_are_held_and_a_simulated_body_gets_everything_back(
    page: dict[str, Any],
) -> None:
    layout = page["layoutOnly"]
    for name, (code, _) in GATED.items():
        assert layout[name]["disabled"] is (code == "layout"), name
    assert layout["realNoteLayout"]["hidden"] is False and layout["realNoteWorld"]["hidden"] is True
    sim = page["simulated"]
    for name in GATED:
        assert sim[name]["disabled"] is False and sim[name]["gate"] is None, name
        assert sim[name]["hidden"] is False, name
    # the titles they had come back; the reason's description is gone
    assert sim["pat_head"]["title"] == "摸頭的說明" and sim["liveDemo"]["title"] == "示範的說明"
    assert sim["world_show"]["title"] is None and sim["apply"]["title"] is None
    assert sim["liveUpright"]["described"] == "other" and sim["pat_head"]["described"] is None
    for note in ("realNoteWorld", "realNoteLive", "realNoteLayout", "realNoteCore"):
        assert sim[note]["hidden"] is True, note
    assert page["undefinedKeeps"] is True  # (an answer without `real` changes nothing)


@needs_node
@pytest.mark.parametrize("order", ["real-first", "strip-first"])
def test_the_sound_scenarios_strip_is_greyed_out_whenever_it_is_built(order: str) -> None:
    """Before: neuro_sound.js builds its strip when it likes; built after real mode was known
    (real-first) its buttons stayed on, and held off later they were hidden instead of greyed out."""
    harness = Path(__file__).resolve().parent / "realduck_sound_node_harness.mjs"
    run = subprocess.run(
        [str(NODE), str(harness), str(STATIC), order, json.dumps(REAL)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    said = json.loads(run.stdout.strip().splitlines()[-1])
    world = [k for k in said if k not in ("soundMusicVol", "_note")]
    assert len(world) == 8  # four music buttons, the hand's two and its slider, the swing
    for control in world:
        assert said[control] == [True, False, "stimuli"], control
    assert said["soundMusicVol"] == [False, False, None]  # (the page's own volume: never held)
    assert said["_note"] == [False]


@needs_node
def test_the_link_and_the_battery_in_plain_words(page: dict[str, Any]) -> None:
    up, new, down, none = page["links"]
    assert up == {
        "key": "page.real.chip.link_up",
        "params": {"ms": 65, "drops": 1, "gap": 812},
        "state": "up",
    }
    assert new == {
        "key": "page.real.chip.link_new",
        "params": {"drops": 0, "gap": 0},
        "state": "up",
    }
    assert down == {"key": "page.real.chip.link_down", "params": {"drops": 3}, "state": "down"}
    assert none is None
    unknown, no_percent, known, charging = page["batteries"]
    assert (
        unknown
        == no_percent
        == {"key": "page.real.chip.battery_unknown", "params": {}, "pct": None}
    )
    # (a real robotd does not say whether it charges: nothing is claimed)
    assert known == {"key": "page.real.chip.battery", "params": {"pct": 87}, "pct": 87}
    assert charging == {
        "key": "page.real.chip.battery_charging",
        "params": {"pct": 100},
        "pct": 100,
    }


@needs_node
def test_the_real_duck_card_says_real_or_rehearsal_with_its_numbers() -> None:
    link = {
        "url": "remote://duck.local:8797",
        "up": True,
        "rtt_ms_p50": 40.2,
        "connects": 1,
        "drops": 0,
        "max_gap_ms": 90,
        "deadman_ms": 1000,
    }
    answers = {
        "real": {"enabled": True, "available": True, "remote": link, "real": REAL},
        "rehearsal": {
            "enabled": True,
            "available": True,
            "remote": link,
            "real": {**REAL, "rehearsal": True},
        },
        "real_new": {
            "enabled": True,
            "available": True,
            "remote": {**link, "rtt_ms_p50": None},
            "real": REAL,
        },
        "real_down": {
            "enabled": True,
            "available": True,
            "remote": {**link, "up": False},
            "real": REAL,
        },
        "simulated": {"enabled": True, "available": True, "remote": link, "real": None},
    }
    script = (
        f"const api = require({json.dumps(str(STATIC / 'bodylink.js'))});"
        f"const answers = {json.dumps(answers)};"
        "const out = {};"
        "for (const [name, answer] of Object.entries(answers)) out[name] = api.linkState(answer);"
        "console.log(JSON.stringify(out));"
    )
    run = subprocess.run([str(NODE), "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    said = json.loads(run.stdout)
    assert said["real"]["state"] == "real" and said["real"]["key"] == "page.body.state.real"
    assert said["real"]["params"] == {"url": "remote://duck.local:8797", "ms": 40}
    assert said["real"]["stats"]["key"] == "page.body.stats"
    assert said["rehearsal"]["key"] == "page.body.state.rehearsal"
    assert said["real_new"]["key"] == "page.body.state.real_new"
    assert said["real_down"]["state"] == "remote_down"  # (a broken link says so first)
    assert said["simulated"]["state"] == "remote_up"


@needs_node
def test_the_3d_label_says_the_ducks_own_joints_and_odometry_never_physics() -> None:
    truth = STATIC / "truth.js"
    source = (
        f'import * as T from "{truth.as_uri()}";\n'
        "const base = { enabled: true, on: true, available: true };\n"
        "console.log(JSON.stringify({\n"
        "  real: T.modeLabel({ ...base, truth: true, real: true }),\n"
        "  fallen: T.modeLabel({ ...base, truth: true, real: true, physicsFallen: true, physicalHome: true }),\n"
        "  waiting: T.modeLabel({ ...base, truth: false, real: true }),\n"
        "  simulated: T.modeLabel({ ...base, truth: true }),\n"
        "}));\n"
    )
    run = subprocess.run(
        [str(NODE), "--input-type=module", "-e", source], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr
    said = json.loads(run.stdout)
    assert said["real"] == {
        "mode": "real",
        "title": "page.truth.REAL_TITLE",
        "detail": "page.truth.modeLabel.real",
    }
    assert said["fallen"]["detail"] == "page.truth.modeLabel.real_fallen"
    assert said["waiting"]["title"] == "page.truth.REAL_WAIT_TITLE"
    assert said["simulated"]["title"] == "page.truth.TRUTH_TITLE"


@needs_node
def test_the_3d_line_says_robotds_pause_in_plain_words_never_its_own_word() -> None:
    """Before: lifted, the 3D line of a real duck showed robotd's own English word ``held``."""
    truth = STATIC / "truth.js"
    source = (
        f'import * as T from "{truth.as_uri()}";\n'
        "console.log(JSON.stringify({\n"
        '  held: T.policyKey("held", true),\n'
        '  walk: T.policyKey("walk", true),\n'
        '  starting: T.policyKey("starting", true),\n'
        '  simWalk: T.policyKey("walk", false),\n'
        '  simOther: T.policyKey("starting", false),\n'
        "}));\n"
    )
    run = subprocess.run(
        [str(NODE), "--input-type=module", "-e", source], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr
    said = json.loads(run.stdout)
    assert said["held"] == "page.real.live.held"
    assert said["walk"] == said["simWalk"] == "page.duck3d.liveText.e5dca5"
    assert said["starting"] is None  # (a word robotd has that the page has none for: not shown)
    assert said["simOther"] is None
    from duckmind.i18n import CODES
    from duckmind.i18n.catalog import read_catalog

    for code in CODES:
        words = (read_catalog(code, "page") or {}).get("page.real.live.held")
        assert words and "held" not in words.lower().split(), code


@needs_node
def test_a_buttons_own_timer_never_lets_a_refused_control_go_again() -> None:
    """Before: 示範, 扶正, 重設, 扶牠起來 and 抱到地墊 switched themselves back on a few seconds
    after a click, also when real mode had been detected meanwhile (the server still refused)."""
    script = (
        f"const api = require({json.dumps(str(JS))});\n"
        "const el = (sels) => ({ disabled: true, matches: (s) => s.split(',').some((x) => sels.includes(x.trim())) });\n"
        f"const real = api.realOf({json.dumps(REAL)});\n"
        "const out = {};\n"
        "for (const [name, sels] of Object.entries({ demo: ['#liveDemo'], upright: ['#liveUpright'],"
        " reset: ['#liveReset'], help: ['#neuroHelp'], carry: ['#neuroCarry'], other: [] })) {\n"
        "  const a = el(sels); api.release(real, a);\n"
        "  const b = el(sels); api.release(null, b);\n"
        "  out[name] = [a.disabled, b.disabled];\n"
        "}\n"
        "console.log(JSON.stringify(out));\n"
    )
    run = subprocess.run([str(NODE), "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    said = json.loads(run.stdout)
    for name in ("demo", "upright", "reset", "help", "carry"):
        assert said[name] == [True, False], name  # held off in real mode, back in simulated mode
    assert said["other"] == [False, False]


def test_every_gate_names_a_control_on_the_page_and_every_word_is_in_six_languages() -> None:
    js = JS.read_text(encoding="utf-8")
    for selector in re.findall(r'"(#[^"]+)"|\'(#[^\']+)\'', js):
        selector = selector[0] or selector[1]
        for part in selector.split(","):
            ids = re.findall(r"#([A-Za-z]\w*)", part)
            for element in ids:
                assert f'id="{element}"' in PAGE or f'id: "{element}"' in SOUND_JS, (
                    selector,
                    element,
                )
            for name, value in re.findall(r'\[data-([\w-]+)="([\w-]+)"\]', part):
                if name == "stim":
                    continue  # (「更多」's buttons are built from the server's list)
                assert f'data-{name}="{value}"' in PAGE, (selector, name, value)
    for note in re.findall(r'note: "(\w+)"', js):
        assert f'id="{note}"' in PAGE, note
    for note in ("realNoteWorld", "realNoteLive", "realNoteLayout", "realNoteCore"):
        tag = PAGE[PAGE.index(f'id="{note}"') - 60 : PAGE.index(f'id="{note}"') + 120]
        assert "data-real-note" in tag and "hidden" in tag, note
    for chip in ("chipReal", "chipLink", "chipBattery"):
        tag = PAGE[PAGE.index(f'id="{chip}"') - 40 : PAGE.index(f'id="{chip}"') + 40]
        assert "data-real" in tag and "hidden" in tag, chip
    assert PAGE.index('src="/realduck.js"') < PAGE.index('src="/app.js"')
    keys = set(re.findall(r'"(page\.[\w.]+)"', js))
    for name in ("bodylink.js", "truth.js", "duck3d.js", "neuro.js", "app.js"):
        keys |= {
            k
            for k in re.findall(
                r'"(page\.(?:real|body\.state\.re|truth\.REAL|truth\.modeLabel\.real)[\w.]*)"',
                (STATIC / name).read_text(encoding="utf-8"),
            )
        }
    assert {
        "page.real.no_world",
        "page.real.note.stimuli",
        "page.real.note.help",
        "page.real.note.gothere",
        "page.real.eye.age",
    } <= keys
    from duckmind.i18n import CODES
    from duckmind.i18n.catalog import read_catalog

    for code in CODES:
        catalog = read_catalog(code, "page") or {}
        assert not keys - set(catalog), (code, sorted(keys - set(catalog)))
