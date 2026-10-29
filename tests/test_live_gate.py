"""Live voice, LV2: the grounding gate (``duckmind.live.gate``). What the cloud model proposes is
judged by rules: only the voice bank's calls, levels and delays bounded, rate and per-utterance
caps, nothing while the cloud switch is off, nothing that is only the duck hearing itself; every
refusal logged with its reason. No network."""

from __future__ import annotations

import math
from typing import Any

import pytest
from live_bridge_rig import make_call, mark, tag, vocalize

from duckmind.live import gate as G
from duckmind.live.tools import BANK_TAGS
from duckmind.neuro.contract import VOCAL_TAGS

pytestmark = pytest.mark.usefixtures("no_network")


def gate(**kw: Any) -> G.GroundingGate:
    return G.GroundingGate(**kw)


def reasons(verdict: G.Verdict) -> list[str]:
    return [r.reason for r in verdict.refused]


# ── what gets through ──────────────────────────────────────────────────────────────────


def test_a_bank_call_is_accepted_with_its_level_and_delay() -> None:
    verdict = gate().judge(vocalize([tag("greet", 0.6, 200)]), 5.0)
    assert verdict.accepted and verdict.reason is None and not verdict.refused
    assert [(t.tag, t.level, t.at_ms) for t in verdict.tags] == [("greet", 0.6, 200)]


def test_only_the_social_calls_of_the_bank_are_the_clouds_to_ask_for() -> None:
    allowed = set(G.DEFAULT_ALLOWED)
    assert allowed <= set(VOCAL_TAGS) <= set(BANK_TAGS)
    for name in BANK_TAGS:
        verdict = gate().judge(vocalize([tag(name)]), 5.0)
        assert verdict.accepted == (name in allowed), name
        if name not in allowed:
            expected = G.TAG_NOT_VOICEABLE if name not in VOCAL_TAGS else G.TAG_RESERVED
            assert reasons(verdict) == [expected], name
    # the fright call is the duck's own (reflex, pain), the ride's wheee is a stream
    assert "alarm" not in allowed and "wheee" not in allowed and "peck" not in allowed


@pytest.mark.parametrize("name", ["quack", "Greet", "moo", "hello", "", "  ", "greet; drop table"])
def test_a_call_that_is_not_in_the_bank_is_refused_every_time(name: str) -> None:
    g = gate()
    verdict = g.judge(vocalize([tag(name)]), 5.0)
    assert not verdict.accepted and verdict.tags == ()
    assert verdict.reason in (G.TAG_UNKNOWN, G.MALFORMED)
    assert g.status()["accepted"] == 0 and sum(g.refusals.values()) >= 1


def test_an_off_bank_tag_does_not_spoil_the_bank_tag_beside_it() -> None:
    g = gate()
    verdict = g.judge(vocalize([tag("quack", 0.5, 0), tag("coo", 0.4, 100)]), 5.0)
    assert verdict.accepted and [t.tag for t in verdict.tags] == ["coo"]
    assert reasons(verdict) == [G.TAG_UNKNOWN]
    assert g.refusals[G.TAG_UNKNOWN] == 1


# ── levels and delays ──────────────────────────────────────────────────────────────────


def test_levels_are_clamped_to_zero_one_and_a_silent_call_is_no_call() -> None:
    g = gate()
    loud = g.judge(vocalize([tag("greet", 9.0)]), 5.0)
    assert loud.accepted and loud.tags[0].level == 1.0 and "level_clamped" in loud.notes
    quiet = g.judge(vocalize([tag("greet", -3.0)], seq=2, t=40.0), 40.0)
    assert not quiet.accepted and quiet.reason == G.LEVEL_SILENT
    assert gate().judge(vocalize([tag("greet", 0.0)]), 5.0).reason == G.LEVEL_SILENT


@pytest.mark.parametrize("bad", ["loud", None, math.nan, math.inf, [0.5], True])
def test_a_level_that_is_not_a_number_is_a_broken_call_not_a_default(bad: Any) -> None:
    verdict = gate().judge(vocalize([{"tag": "greet", "level": bad, "at_ms": 0}]), 5.0)
    assert not verdict.accepted and verdict.reason == G.MALFORMED


def test_the_delay_is_bounded_and_the_tags_of_one_call_are_spaced() -> None:
    cfg = G.GateConfig()
    verdict = gate().judge(
        vocalize([tag("chirp", 0.5, 0), tag("chirp", 0.5, 10), tag("coo", 0.5, 99_999)]), 5.0
    )
    assert verdict.accepted
    ats = [t.at_ms for t in verdict.tags]
    assert ats[0] == 0 and ats[1] >= cfg.tag_gap_ms
    assert all(a <= cfg.at_ms_max for a in ats)
    assert "tags_spaced" in verdict.notes
    negative = gate().judge(vocalize([tag("greet", 0.5, -500)], seq=2), 5.0)
    assert negative.tags[0].at_ms == 0


def test_a_call_with_too_many_tags_is_cut_and_the_rest_logged() -> None:
    g = gate()
    verdict = g.judge(vocalize([tag("chirp", 0.5, 0)] * 6), 5.0)
    assert len(verdict.tags) <= g.config.max_tags
    assert reasons(verdict).count(G.TOO_MANY_TAGS) == 6 - g.config.max_tags


@pytest.mark.parametrize(
    "raw",
    [{}, {"tags": []}, {"tags": "greet"}, {"tags": [None]}, {"tags": [3, "greet"]}, {"tags": {}}],
)
def test_hostile_arguments_are_refused_not_crashed_on(raw: dict[str, Any]) -> None:
    verdict = gate().judge(make_call("vocalize", raw), 5.0)
    assert not verdict.accepted and verdict.reason == G.MALFORMED


# ── caps ───────────────────────────────────────────────────────────────────────────────


def test_the_rate_cap_holds_per_window_and_lets_go_after_it() -> None:
    g = gate()
    cfg = g.config
    got = 0
    for i in range(10):  # one tag per call, each its own utterance (human sound far apart)
        verdict = g.judge(
            vocalize([tag("chirp")], seq=i + 1, t=1.0 + i * 0.5, call_mark=mark(since_human_s=0.1)),
            1.0 + i * 0.5,
        )
        got += verdict.accepted
    assert got <= cfg.rate_max_tags
    assert g.refusals[G.RATE_CAP] + g.refusals[G.UTTERANCE_CAP] >= 10 - cfg.rate_max_tags
    later = g.judge(vocalize([tag("chirp")], seq=99, t=40.0), 40.0)
    assert later.accepted  # (the window has passed)


def test_the_rate_cap_alone_is_the_number_of_tags_per_ten_seconds() -> None:
    cfg = G.GateConfig(rate_max_tags=3, utterance_max_tags=99, utterance_max_calls=99)
    g = gate(config=cfg)
    accepted = [
        g.judge(vocalize([tag("coo")], seq=i, t=2.0 + i), 2.0 + i).accepted for i in range(1, 7)
    ]
    assert accepted == [True, True, True, False, False, False]
    assert g.refusals[G.RATE_CAP] == 3
    assert g.judge(vocalize([tag("coo")], seq=20, t=13.5), 13.5).accepted  # the first has aged out


def test_one_thing_a_person_said_cannot_make_the_duck_call_without_end() -> None:
    cfg = G.GateConfig()
    g = gate()
    # the same human sound (last heard at t=4.0) answered again and again
    verdicts = [
        g.judge(
            vocalize(
                [tag("greet")],
                seq=i,
                t=5.0 + 0.3 * i,
                call_mark=mark(since_human_s=1.0 + 0.3 * i),
            ),
            5.0 + 0.3 * i,
        )
        for i in range(5)
    ]
    assert sum(v.accepted for v in verdicts) == cfg.utterance_max_calls
    assert g.refusals[G.UTTERANCE_CAP] == 5 - cfg.utterance_max_calls
    # a new utterance (the last human sound is far from that one) starts afresh
    fresh = g.judge(
        vocalize([tag("greet")], seq=9, t=30.0, call_mark=mark(since_human_s=1.0)), 30.0
    )
    assert fresh.accepted


def test_the_other_tools_have_their_own_caps() -> None:
    g = gate()
    results = [
        g.judge(make_call("feel", {"valence": 0.5, "talk": 0.5}, seq=i, t=5.0 + 0.1 * i), 6.0)
        for i in range(8)
    ]
    limit = g.config.call_caps["feel"][0]
    assert sum(r.accepted for r in results) == limit
    assert g.refusals[G.CALL_CAP] == 8 - limit


# ── the cloud switch ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("vocalize", {"tags": [tag("greet")]}),
        ("attend", {"target_kind": "person"}),
        ("recognize", {"name": "媽媽"}),
        ("remember", {"note": "x"}),
        ("feel", {"valence": 0.1, "talk": 0.1}),
    ],
)
def test_nothing_the_cloud_says_is_acted_on_while_the_switch_is_off(
    name: str, raw: dict[str, Any]
) -> None:
    on = {"v": True}
    g = gate(cloud_on=lambda: on["v"])
    assert g.judge(make_call(name, raw, seq=1), 5.0).accepted
    on["v"] = False  # (the call was queued before the switch went off, judged after)
    verdict = g.judge(make_call(name, raw, seq=2), 5.1)
    assert not verdict.accepted and verdict.reason == G.CLOUD_OFF
    assert g.refusals[G.CLOUD_OFF] == 1


# ── echo ───────────────────────────────────────────────────────────────────────────────


def test_a_call_that_is_only_the_duck_hearing_itself_is_refused() -> None:
    g = gate()
    only = g.judge(vocalize([tag()], call_mark=mark(echo_only=True, since_human_s=None)), 5.0)
    assert not only.accepted and only.reason == G.ECHO_ONLY
    caused = g.judge(
        vocalize([tag()], seq=2, call_mark=mark(cause_echo=True, cause_age_s=0.4)), 5.0
    )
    assert not caused.accepted and caused.reason == G.ECHO_CAUSE
    other = g.judge(make_call("remember", {"note": "x"}, call_mark=mark(echo_only=True)), 5.0)
    assert not other.accepted and other.reason == G.ECHO_ONLY
    assert g.refusals[G.ECHO_ONLY] == 2


def test_local_evidence_of_a_person_rescues_a_call_the_echo_marks_would_refuse() -> None:
    heard = {"yes": True}
    asked: list[tuple[float, float]] = []

    def evidence(t0: float, t1: float) -> bool:
        asked.append((t0, t1))
        return heard["yes"]

    g = gate(human_evidence=evidence)
    call = vocalize([tag()], t=5.0, call_mark=mark(echo_only=True, since_human_s=None))
    verdict = g.judge(call, 5.0)
    assert verdict.accepted and verdict.rescued and asked and asked[0][1] >= 5.0
    heard["yes"] = False
    assert not g.judge(vocalize([tag()], seq=2, t=5.0, call_mark=call.mark), 5.0).accepted

    def broken(t0: float, t1: float) -> bool:
        raise RuntimeError("detector fault")

    assert not gate(human_evidence=broken).judge(call, 5.0).accepted  # (a fault never lets one in)


# ── time, names of tools, the other tools' fields ────────────────────────────────────


def test_a_reply_that_comes_after_its_moment_has_gone_is_refused() -> None:
    g = gate()
    assert g.judge(vocalize([tag()], t=5.0), 5.0).accepted
    stale = g.judge(vocalize([tag()], seq=2, t=5.0), 5.0 + g.config.stale_s + 0.1)
    assert not stale.accepted and stale.reason == G.STALE


def test_a_tool_that_is_not_one_of_the_five_is_refused() -> None:
    verdict = gate().judge(make_call("move_servo", {"angle": 90}), 5.0)
    assert not verdict.accepted and verdict.reason == G.UNKNOWN_TOOL


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("attend", {"target_kind": "star"}),
        ("attend", {}),
        ("recognize", {"name": "  "}),
        ("recognize", {}),
        ("remember", {"note": ""}),
        ("feel", {"valence": "happy", "talk": 0.5}),
        ("feel", {"valence": math.nan, "talk": 0.5}),
        ("feel", {"valence": 0.5}),
    ],
)
def test_the_other_tools_need_their_fields(name: str, raw: dict[str, Any]) -> None:
    verdict = gate().judge(make_call(name, raw), 5.0)
    assert not verdict.accepted and verdict.reason == G.MALFORMED


def test_feel_is_clamped_and_text_is_bounded() -> None:
    g = gate()
    felt = g.judge(make_call("feel", {"valence": 7, "talk": -2}), 5.0)
    assert felt.args == {"valence": 1.0, "talk": 0.0} and "clamped" in felt.notes
    note = g.judge(make_call("remember", {"note": "x" * 5000}, seq=2), 5.0)
    assert len(note.args["note"]) <= g.config.note_max


# ── every refusal is logged, with its reason ──────────────────────────────────────────


def test_every_decision_is_logged_and_counted_with_its_reason() -> None:
    g = gate()
    g.judge(vocalize([tag("greet")], seq=1), 5.0)
    g.judge(vocalize([tag("quack")], seq=2), 5.0)
    g.judge(vocalize([tag("alarm")], seq=3), 5.0)
    g.judge(make_call("feel", {"valence": "x", "talk": 1}, seq=4), 5.0)
    assert len(g.log) == 4
    assert [e["accepted"] for e in g.log] == [True, False, False, False]
    assert [e["reason"] for e in list(g.log)[1:]] == [G.TAG_UNKNOWN, G.TAG_RESERVED, G.MALFORMED]
    assert all(e["refused"] and e["refused"][0]["reason"] for e in list(g.log)[1:])
    status = g.status()
    assert status["accepted"] == 1 and status["refused_calls"] == 3
    assert status["refusals"] == {G.MALFORMED: 1, G.TAG_RESERVED: 1, G.TAG_UNKNOWN: 1}
    assert set(G.REASONS) >= set(status["refusals"])


def test_the_gate_never_raises_on_what_the_model_sends() -> None:
    g = gate()
    weird: list[Any] = [None, 1, "x", [], {}, {"tags": [{"tag": {"a": 1}, "level": []}]}]
    for i, raw in enumerate(weird):
        call = make_call("vocalize", {"tags": raw} if not isinstance(raw, dict) else raw, seq=i)
        assert not g.judge(call, 5.0).accepted
