"""The test tiers (tests/tiers.py, the hooks in tests/conftest.py): the markers land on the tests
they are for, ``heavy`` follows from them, the fast tier leaves the heavy ones out, and a name
that is no longer in its file is reported instead of quietly joining the fast tier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import conftest
import pytest

from tests import tiers

ROOT = Path(__file__).resolve().parents[1]


def _names(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def _test_of(item: pytest.Function) -> str:
    return item.originalname if item.cls is None else f"{item.cls.__name__}.{item.originalname}"


def test_every_collected_test_carries_the_markers_of_its_tier(request: pytest.FixtureRequest):
    """Over whatever this run collected (the whole suite, the fast tier, one file)."""
    for item in request.session.items:
        if not isinstance(item, pytest.Function):
            continue
        names, file = _names(item), item.path.name
        assert ("render" in names) == (file in conftest.RENDER_FILES), item.nodeid
        for kind in tiers.kinds(file, _test_of(item)):
            assert kind in names, (item.nodeid, kind)
        if tiers.listed(tiers.SLOW, file, _test_of(item)) or names & {"render", "physics"}:
            assert "heavy" in names, item.nodeid


@pytest.mark.physics
def test_a_test_its_own_file_marks_physics_is_heavy_too(request: pytest.FixtureRequest):
    """(what a new MuJoCo test file writes: ``pytestmark = [pytest.mark.physics]``)"""
    assert request.node.get_closest_marker("heavy") is not None


def test_the_fast_tier_leaves_out_render_physics_and_measured_slow_tests():
    """A child pytest collects this file, a render file and a file with measured-slow tests with
    ``-m "not heavy"``: only the tests that are none of those are left."""
    slow_file, slow_tests = next(
        (file, tests)
        for file, tests in sorted(tiers.SLOW.items())
        if tests is not tiers.WHOLE and (ROOT / "tests" / file).is_file()
    )
    render_file = sorted(conftest.RENDER_FILES)[0]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"]
        + ["-p", "no:cacheprovider", "-m", "not heavy"]
        + ["tests/test_tiers.py", f"tests/{render_file}", f"tests/{slow_file}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    left = {line.split("[")[0] for line in out.stdout.splitlines() if "::" in line}
    mine = "tests/test_tiers.py::"
    assert mine + "test_every_collected_test_carries_the_markers_of_its_tier" in left
    assert mine + "test_a_test_its_own_file_marks_physics_is_heavy_too" not in left
    assert not any(line.startswith(f"tests/{render_file}::") for line in left), left
    in_slow_file = {line.split("::", 1)[1] for line in left if f"/{slow_file}::" in line}
    assert in_slow_file, "the file's other tests stay in the fast tier"
    assert not in_slow_file & {test.replace(".", "::") for test in slow_tests}


def test_a_name_that_is_no_longer_in_its_file_is_reported(monkeypatch: pytest.MonkeyPatch):
    me = sys.modules[__name__]
    here = {"test_tiers.py": me}
    assert conftest._gone(here) == []
    listed = ("test_a_name_that_is_no_longer_in_its_file_is_reported", "test_renamed_away")
    monkeypatch.setitem(tiers.SLOW, "test_tiers.py", listed)
    monkeypatch.setitem(tiers.TIMING, "test_tiers.py", ("TestGone.test_inside",))
    assert conftest._gone(here) == [
        "test_tiers.py::TestGone.test_inside",
        "test_tiers.py::test_renamed_away",
    ]
    monkeypatch.setitem(tiers.SLOW, "test_tiers.py", tiers.WHOLE)  # (a whole file names no test)
    monkeypatch.delitem(tiers.TIMING, "test_tiers.py")
    assert conftest._gone(here) == []
    assert conftest._gone({}) == []  # (a file that was not collected is not judged)
