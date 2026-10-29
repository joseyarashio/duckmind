"""Live voice, LV1: the usage meter (``duckmind.live.meter``): per minute counts, tokens from
the server's turns, quota errors, and caps that stop the cloud by themselves."""

from __future__ import annotations

import pytest

from duckmind.live.meter import AUDIO_TOKENS_PER_S, UsageCaps, UsageMeter

pytestmark = pytest.mark.usefixtures("no_network")

USAGE = {
    "total": 978,
    "prompt": 953,
    "response": 25,
    "thoughts": 113,
    "by_modality": {"TEXT": 529, "AUDIO": 277},
}


def test_counts_land_in_the_minute_they_happened() -> None:
    meter = UsageMeter()
    meter.add(10.0, "audio_s", 1.0)
    meter.add(50.0, "frames")
    meter.add(70.0, "audio_s", 2.0)
    minutes = meter.per_minute(75.0, count=3)
    assert [m["minute"] for m in minutes] == [-1, 0, 1]
    assert minutes[0]["audio_s"] == 0 and minutes[1]["audio_s"] == 1.0 and minutes[1]["frames"] == 1
    assert minutes[2]["audio_s"] == 2.0
    assert meter.totals["audio_s"] == 3.0


def test_the_server_reports_tokens_once_a_turn() -> None:
    meter = UsageMeter()
    meter.usage(5.0, USAGE)
    meter.usage(20.0, {**USAGE, "total": 1000, "prompt": 970})
    status = meter.status(30.0)
    assert status["totals"]["turns"] == 2 and status["totals"]["tokens"] == 1978
    assert status["this_minute"]["tokens"] == 1978 and status["context_tokens"] == 970
    assert status["by_modality"] == {"AUDIO": 554, "TEXT": 1058}


def test_audio_is_estimated_at_the_published_rate() -> None:
    meter = UsageMeter()
    meter.add(0.0, "audio_s", 10.0)
    assert meter.tokens_estimated() == int(10 * AUDIO_TOKENS_PER_S)
    assert meter.status(1.0)["this_minute"]["audio_tokens_est"] == 250


def test_quota_errors_and_reconnects_are_counted() -> None:
    meter = UsageMeter()
    for _ in range(3):
        meter.add(1.0, "quota_errors")
    meter.add(2.0, "reconnects")
    status = meter.status(3.0)
    assert status["totals"]["quota_errors"] == 3 and status["totals"]["reconnects"] == 1
    assert status["this_minute"]["quota_errors"] == 3


def test_a_cap_is_passed_when_the_total_goes_over() -> None:
    meter = UsageMeter(UsageCaps(tokens=1000, audio_s=60.0, frames=5))
    assert meter.over_cap() is None
    meter.add(0.0, "frames", 6)
    assert meter.over_cap() == "frames"
    meter = UsageMeter(UsageCaps(audio_s=60.0))
    meter.add(0.0, "audio_s", 61.0)
    assert meter.over_cap() == "audio_s"
    meter = UsageMeter(UsageCaps(tokens=1000))
    meter.usage(0.0, USAGE)
    assert meter.over_cap() is None
    meter.usage(1.0, USAGE)
    assert meter.over_cap() == "tokens"
    assert meter.status(2.0)["over_cap"] == "tokens" and meter.status(2.0)["caps"]["tokens"] == 1000


def test_no_caps_never_stop_anything() -> None:
    meter = UsageMeter()
    meter.add(0.0, "audio_s", 10**6)
    assert meter.over_cap() is None


def test_an_unknown_counter_is_a_bug_not_a_silent_zero() -> None:
    with pytest.raises(KeyError):
        UsageMeter().add(0.0, "nonsense")


def test_old_minutes_are_forgotten_but_the_totals_stay() -> None:
    meter = UsageMeter()
    for minute in range(200):
        meter.add(minute * 60.0, "frames")
    assert meter.totals["frames"] == 200
    assert len(meter._minutes) == 120
