"""i18n: the API boundary on REAL engine output (the frozen v4_0 engine 蛋黃 runs on, and v4_1).

A fresh duck lives a day on each engine (headless, no world, no server); every page string of
every answer (state, feed, results, the memory tabs) must get a message whose 繁體中文 rendering
is byte-identical to the engine's own words, from keys the catalog has. Data (the duck's name,
typed words, raw event JSON) must never be keyed. Nothing in the engine is touched: the
boundary reads its source and adds ``_i18n`` maps to a copy of the answer.
"""

from __future__ import annotations

import copy
import importlib
import re
from typing import Any

import pytest

from duckmind.i18n import Msg, boundary, msg, packs, render

CJK = re.compile(r"[㐀-鿿]")
STEPS: list[tuple[str, dict[str, Any]]] = (
    [("advance", {"minutes": 30})] * 4
    # the page's interaction buttons (the i18n core inventory's harvest used the same)
    + [
        ("act", {"action": a, "persona": p, "params": params})
        for a, p, params in (
            ("touch", "owner", {}),
            ("say", "child", {"kind": "voice"}),
            ("call_name", "child", {}),
            ("owner_hello", "owner", {}),
            ("owner_good", "owner", {}),
            ("pickup", "owner", {}),
            ("hand_near", "partner", {"speed": "fast"}),
            ("tv", "owner", {"on": True}),
            ("tv", "owner", {"on": False}),
            ("owner_walk", "owner", {}),
            ("hold_head", "partner", {}),
            ("owner_enough", "owner", {}),
            ("end_quiet", "owner", {}),
            ("place_on_nest", "owner", {}),
            ("pad_play", "child", {"minutes": 5}),
        )
    ]
    + [("advance", {"minutes": 5})] * 150
    + [("advance", {"minutes": 120})] * 6
    + [("advance", {"until": "07:00"})]
)
TABS: list[tuple[str, dict[str, Any]]] = [
    ("query", {"question": "你記得主人嗎？"}),
    ("query", {"question": "你是什麼個性？"}),
    ("diary", {}),
    ("graph", {}),
    ("events", {}),
    ("why", {}),
    ("command", {"text": "主人說你好"}),
    ("command", {"text": "為什麼不動？"}),
]


def _api(engine: str) -> Any:
    """The engine's API: the frozen v4_0 copy, or the tree's current Duck Core ("v4_1"; the
    neural-duck worktree still has the phase-3 Duck Core there)."""
    if engine == "v4_0":
        if not (packs.PACKAGE_ROOT / "engines" / "v4_0").exists():
            pytest.skip("this tree has no frozen v4_0 engine")
        return importlib.import_module("duckmind.engines.v4_0.duckcore.api")
    try:
        importlib.import_module("duckmind.world.legacy")  # registers the world v4_1 ducks live in
    except ModuleNotFoundError:
        pass
    return importlib.import_module("duckmind.duckcore.api")


def live_a_day(engine: str) -> list[dict[str, Any]]:
    api = _api(engine)
    config = api.Config(name="測試鴨", seed=11, nest=True)
    session = api.open_session(config) if hasattr(api, "open_session") else api.Session(config)
    cursor = {"events": 0, "journal": 0}
    answers: list[dict[str, Any]] = []
    for method, params in STEPS:
        result = api.dispatch(session, method, params)
        feed = api.feed(session, cursor["events"], cursor["journal"])
        cursor = feed["cursor"]
        answers.append({"result": result, "state": api.snapshot(session), "feed": feed})
    for method, params in TABS:
        answers.append({"result": api.dispatch(session, method, params)})
    return answers


@pytest.fixture(scope="module", params=["v4_0", "v4_1"])
def engine(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def answers(engine: str) -> list[dict[str, Any]]:
    return live_a_day(engine)


def walk(original: Any, annotated: Any, path: str, out: list[tuple[str, str, Any]]) -> None:
    """(path, words, wire or None) for every page string of an answer."""
    if isinstance(original, dict):
        messages = annotated.get("_i18n", {})
        for key, value in original.items():
            sub = f"{path}.{key}"
            if isinstance(value, str):
                if (
                    CJK.search(value)
                    and not boundary.is_data(sub)
                    and not (original.get("type") == "self" and key == "label")
                ):
                    out.append((sub, value, messages.get(key)))
            elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                wires = messages.get(key) or [None] * len(value)
                for text, wire in zip(value, wires, strict=True):
                    if CJK.search(text) and not boundary.is_data(f"{sub}[]"):
                        out.append((f"{sub}[]", text, wire))
            else:
                walk(value, annotated.get(key), sub, out)
    elif isinstance(original, list):
        for a, b in zip(original, annotated, strict=True):
            walk(a, b, f"{path}[]", out)


def test_every_page_string_is_keyed_and_renders_back_byte_identical(
    engine: str, answers: list[dict[str, Any]]
) -> None:
    b = boundary.Boundary(engine, names={"測試鴨"})
    seen: list[tuple[str, str, Any]] = []
    for answer in answers:
        walk(answer, b.annotate(answer), "", seen)
    assert len(seen) > 5000, "the day produced too few page strings to judge"
    unkeyed = sorted({(p, t) for p, t, w in seen if w is None})[:20]
    assert unkeyed == [], f"{len(unkeyed)} page strings without a message"
    wrong = []
    for path, text, wire in seen:
        again = render(wire, "zh-Hant", strict=True)
        if again != text:
            wrong.append((path, text, again))
    assert wrong[:10] == []
    # and every one renders in the pseudo-locale with no word left outside ⟦ ⟧ except data
    leftovers = []
    for _path, _text, wire in seen[:3000]:
        shown = render(wire, "pseudo", strict=True)
        outside = re.sub(r"⟦[^⟧]*⟧", "", shown).replace("測試鴨", "")
        if CJK.search(outside):
            leftovers.append((shown, outside))
    assert leftovers[:10] == []


def test_the_engine_pack_needs_no_key_the_catalog_lacks(engine: str) -> None:
    _api(engine)
    pack = packs.get(engine)
    assert pack is not None
    assert pack.missing == [], (
        "engine templates the page can show without a catalog key (python -m duckmind.i18n pack)"
    )
    assert len(pack.sites) > 1000


def test_annotate_leaves_the_answer_untouched(engine: str, answers: list[dict[str, Any]]) -> None:
    answer = answers[20]
    before = copy.deepcopy(answer)
    annotated = boundary.annotate(answer, engine)
    assert answer == before
    assert "_i18n" not in str(before)
    stripped = _strip(annotated)
    assert stripped == before


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "_i18n"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_data_is_never_keyed(engine: str) -> None:
    _api(engine)
    b = boundary.Boundary(engine, names={"小鴨"})
    # 「小鴨」 is both the feed tag for duck.* events and the default name: the field decides
    assert b.wire_for("state.identity.name", "小鴨") is None
    assert b.wire_for("result.results[].clause", "主人說你好") is None
    assert b.wire_for("result.event.payload.text", "我做了「轉身看看」") is None
    assert b.wire_for("result.nodes[].label", "小鴨", {"type": "self", "label": "小鴨"}) is None
    assert b.wire_for("feed.events[].category_zh", "小鴨") is not None


def test_msg_values_pass_through_the_boundary() -> None:
    answer = {
        "note": Msg("page.i18n.picker.label", {}, "語言"),
        "tags": ["plain", Msg("fmt.sep.enum", {}, "、")],
        "n": 3,
    }
    out = boundary.attach(answer)
    assert out["note"] == "語言" and out["n"] == 3
    assert out["_i18n"] == {
        "note": {"k": "page.i18n.picker.label"},
        "tags": [None, {"k": "fmt.sep.enum"}],
    }
    real = msg("page.i18n.picker.label")
    assert real == "語言" and boundary.attach({"x": real})["_i18n"]["x"] == {
        "k": "page.i18n.picker.label"
    }


def test_numbers_become_json_numbers_only_when_they_print_back_the_same() -> None:
    assert boundary.number_value("3", "") == 3
    assert boundary.number_value("0.40", ".2f") == 0.4
    assert boundary.number_value("0.5", "") == 0.5
    assert boundary.number_value("1.0", "") == 1.0
    assert boundary.number_value("+3", "") == "+3"
    assert boundary.number_value("07:30", "") == "07:30"
    assert boundary.number_value("0.40", "") == "0.40"  # str(0.4) would lose the zero


def test_home_names_are_keyed_without_touching_the_layout_files() -> None:
    import json

    homes = sorted((packs.PACKAGE_ROOT / "world" / "homes").glob("*.json"))
    assert homes
    keyed = 0
    for path in homes:
        layout = json.loads(path.read_text(encoding="utf-8"))
        if "id" not in layout or "zh" not in layout:
            continue  # an overlay without a name of its own
        out = boundary.annotate_home(layout)
        assert _strip(out) == layout
        assert render(out["_i18n"]["zh"], "zh-Hant", strict=True) == layout["zh"], path.name
        for thing in out.get("things", []):
            if "zh" in thing:
                assert render(thing["_i18n"]["zh"], "zh-Hant", strict=True) == thing["zh"], (
                    path.name,
                    thing.get("id"),
                )
                keyed += 1
    assert keyed > 5


def test_engine_of_names_the_pack(engine: str) -> None:
    api = _api(engine)
    config = api.Config(name="測試鴨", seed=3)
    session = api.open_session(config) if hasattr(api, "open_session") else api.Session(config)
    assert boundary.engine_of(session) == engine
    assert boundary.engine_of(object()) is None
