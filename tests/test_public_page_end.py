"""The public duck page at the end of a visit (static/public.js): it goes quiet and opens the front's page.

Runs public.js in Node over a fake page and network (tests/public_end_node_harness.js) with a stand-in for the
rest of the page polling every 50 ms and holding a stream, as app.js and neuro.js do. When the download time runs
out, or the duck answers 404 for the session, nothing but the front's status reaches the network any more, every
stream is closed, and once the front says the session is gone the page opens "/". A stray 404 while the session
is on changes nothing; a normal (non-public) page is untouched; without a front the page stays quiet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("public_end_node_harness.js")

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node")


def _run(scenario: str) -> dict[str, Any]:
    out = subprocess.run(
        [NODE, str(HARNESS), scenario], capture_output=True, text=True, timeout=60, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_when_the_download_time_ends_the_page_goes_quiet_then_opens_the_front() -> None:
    r = _run("time")
    assert r["afterQuiet"] == ["/_edge/status"], r  # no poll, no new stream after it went quiet
    assert r["streamClosed"] is True
    assert r["statusCalls"] >= 3  # it waited while the front still said "ended"
    assert r["navigatedTo"] == "/"


def test_a_duck_that_answers_404_for_its_session_sends_the_page_to_the_front() -> None:
    r = _run("crash")
    assert r["afterQuiet"] == ["/_edge/status"], r
    assert r["streamClosed"] is True
    assert r["navigatedTo"] == "/"


def test_a_stray_404_while_the_session_is_on_ends_nothing() -> None:
    r = _run("stray404")
    assert r["navigatedTo"] is None
    assert r["streamClosed"] is False
    assert "/api/body" in r["afterQuiet"]  # the page kept polling
    assert r["bodyPolls"] > 60


def test_the_normal_page_is_untouched() -> None:
    r = _run("normal")
    assert r == {
        "scenario": "normal",
        "navigatedTo": None,
        "streamClosed": False,
        "statusCalls": 0,
        "bodyPolls": r["bodyPolls"],
        "afterQuiet": [],
    }
    assert r["bodyPolls"] > 60


def test_without_a_front_the_page_stays_quiet_on_the_thanks() -> None:
    r = _run("alone")
    assert r["afterQuiet"] == ["/_edge/status"], r
    assert r["streamClosed"] is True
    assert r["navigatedTo"] is None
