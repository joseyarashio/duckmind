"""Live voice, LV1: the five tools the duck's cloud brain may call (``duckmind.live.tools``).

All ``NON_BLOCKING``, declared as plain dictionaries (the SDK takes them as they are); the
parser normalises a call and says what was wrong with it, and never drops one."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from duckmind.live import tools as T

pytestmark = pytest.mark.usefixtures("no_network")


def test_there_are_five_tools_all_non_blocking() -> None:
    decls = T.declarations()
    assert (
        [d["name"] for d in decls]
        == list(T.TOOL_NAMES)
        == [
            "vocalize",
            "attend",
            "recognize",
            "remember",
            "feel",
        ]
    )
    assert all(d["behavior"] == "NON_BLOCKING" for d in decls)
    assert all(d["description"] and d["parameters"]["type"] == "OBJECT" for d in decls)
    assert T.tools_config() == [{"function_declarations": decls}]


def test_the_declarations_are_plain_json_and_a_fresh_copy_each_time() -> None:
    decls = T.declarations()
    assert json.loads(json.dumps(decls)) == decls
    decls[0]["name"] = "changed"
    assert T.declarations()[0]["name"] == "vocalize"


def test_vocalize_schema_has_the_tag_level_and_time() -> None:
    vocalize = T.declarations()[0]
    items = vocalize["parameters"]["properties"]["tags"]["items"]
    assert set(items["properties"]) == {"tag", "level", "at_ms"}
    assert items["required"] == ["tag", "level"]
    # the description names every call of the bank (and does not pretend the tag is an enum:
    # the grounding check of LV2, not the schema, is what keeps a model inside the bank)
    for tag in T.BANK_TAGS:
        assert tag in vocalize["description"]
    assert "enum" not in items["properties"]["tag"]


def test_attend_feel_recognize_remember_schemas() -> None:
    by = {d["name"]: d["parameters"] for d in T.declarations()}
    assert by["attend"]["required"] == ["target_kind"]
    assert set(by["attend"]["properties"]) == {"target_kind", "hint"}
    assert by["recognize"]["required"] == ["name"]
    assert by["remember"]["required"] == ["note"]
    assert by["feel"]["required"] == ["valence", "talk"]


def test_the_bank_tags_are_the_voice_banks_tags() -> None:
    from duckmind.simweb import voice

    assert T.BANK_TAGS == voice.TAGS


def test_the_schema_hash_follows_the_declarations() -> None:
    first = T.schema_hash()
    assert first == T.schema_hash() and len(first) == 16
    assert T.TOOL_SCHEMA_VERSION.startswith("tools-")


def test_the_sdk_takes_the_declarations_as_they_are() -> None:
    types = pytest.importorskip("google.genai.types")
    config = types.LiveConnectConfig.model_validate(
        {"response_modalities": ["AUDIO"], "tools": T.tools_config()}
    )
    declared = config.tools[0].function_declarations
    assert [d.name for d in declared] == list(T.TOOL_NAMES)
    assert all(d.behavior == types.Behavior.NON_BLOCKING for d in declared)


# ── parsing ────────────────────────────────────────────────────────────────────────


def test_a_good_vocalize_is_normalised_and_valid() -> None:
    call = T.parse_call(
        "vocalize",
        {"tags": [{"tag": "greet", "level": 0.8, "at_ms": 250}, {"tag": " chirp ", "level": 1}]},
    )
    assert call.valid and call.problems == () and call.off_bank == ()
    assert call.args == {
        "tags": [
            {"tag": "greet", "level": 0.8, "at_ms": 250},
            {"tag": "chirp", "level": 1.0, "at_ms": 0},
        ]
    }


def test_a_tag_outside_the_bank_is_kept_and_flagged_for_the_grounding_check() -> None:
    call = T.parse_call(
        "vocalize", {"tags": [{"tag": "你好", "level": 0.5}, {"tag": "greet", "level": 0.5}]}
    )
    assert call.off_bank == ("你好",) and call.valid
    assert [t["tag"] for t in call.args["tags"]] == ["你好", "greet"]


def test_out_of_range_numbers_are_clamped_and_listed() -> None:
    call = T.parse_call("vocalize", {"tags": [{"tag": "greet", "level": 7, "at_ms": -5}]})
    assert call.args["tags"] == [{"tag": "greet", "level": 1.0, "at_ms": 0}]
    assert set(call.problems) == {"level_out_of_range", "at_ms_out_of_range"}
    far = T.parse_call("vocalize", {"tags": [{"tag": "greet", "level": 0.1, "at_ms": 10**9}]})
    assert far.args["tags"][0]["at_ms"] == T.MAX_AT_MS


@pytest.mark.parametrize(
    "args, problem",
    [
        ({}, "tags_missing"),
        ({"tags": []}, "tags_missing"),
        ({"tags": "greet"}, "tags_missing"),
        ({"tags": ["greet"]}, "tag_not_object"),
        ({"tags": [{"level": 0.5}]}, "tag_missing"),
        ({"tags": [{"tag": "greet"}]}, "level_missing"),
        ({"tags": [{"tag": "greet", "level": float("nan")}]}, "level_missing"),
        ({"tags": [{"tag": "greet", "level": True}]}, "level_missing"),
        ({"tags": [{"tag": "greet", "level": 0.5, "at_ms": "soon"}]}, "at_ms_bad"),
    ],
)
def test_a_bad_vocalize_says_what_is_wrong(args: dict[str, Any], problem: str) -> None:
    call = T.parse_call("vocalize", args)
    assert problem in call.problems and not call.valid
    assert call.raw == args  # (the arguments as received are kept)


def test_too_many_tags_are_cut_and_said() -> None:
    tags = [{"tag": "chirp", "level": 0.4}] * (T.MAX_TAGS + 3)
    call = T.parse_call("vocalize", {"tags": tags})
    assert len(call.args["tags"]) == T.MAX_TAGS and "too_many_tags" in call.problems


def test_attend() -> None:
    ok = T.parse_call("attend", {"target_kind": "thing", "hint": "  玩具 "})
    assert ok.valid and ok.args == {"target_kind": "thing", "hint": "玩具"}
    assert T.parse_call("attend", {"target_kind": "person"}).args == {"target_kind": "person"}
    odd = T.parse_call("attend", {"target_kind": "animal"})
    assert "target_kind_unknown" in odd.problems and odd.args["target_kind"] == "animal"
    assert "target_kind_missing" in T.parse_call("attend", {"hint": "x"}).problems


def test_recognize_remember_feel() -> None:
    assert T.parse_call("recognize", {"name": " 媽媽 "}).args == {"name": "媽媽"}
    assert "name_missing" in T.parse_call("recognize", {"name": "  "}).problems
    note = "很長" * 300
    assert len(T.parse_call("remember", {"note": note}).args["note"]) == T.MAX_TEXT
    assert "note_missing" in T.parse_call("remember", {}).problems
    feel = T.parse_call("feel", {"valence": -0.4, "talk": 0.25})
    assert feel.valid and feel.args == {"valence": -0.4, "talk": 0.25}
    clamped = T.parse_call("feel", {"valence": -3, "talk": 2})
    assert clamped.args == {"valence": -1.0, "talk": 1.0}
    assert {"valence_out_of_range", "talk_out_of_range"} <= set(clamped.problems)
    assert {"valence_missing", "talk_missing"} <= set(T.parse_call("feel", {}).problems)


@pytest.mark.parametrize("args", [None, {}, {"x": 1}, {"tags": None}])
def test_an_unknown_tool_is_recorded_not_raised(args: Any) -> None:
    call = T.parse_call("open_the_door", args)
    assert call.problems == ("unknown_tool",) and call.name == "open_the_door"


def test_the_numbers_stay_finite() -> None:
    for call in (
        T.parse_call("feel", {"valence": math.inf, "talk": 0.5}),
        T.parse_call("vocalize", {"tags": [{"tag": "greet", "level": math.inf}]}),
    ):
        assert not call.valid
