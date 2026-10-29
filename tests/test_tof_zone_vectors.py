"""The upstream ``tof`` crate's zone vectors (``microduck/tof/src/lib.rs`` tests
status_decides_what_a_distance_means, valid_count_counts_ranges_only,
a_ragged_frame_reads_as_no_target), replayed against duck-body's two readers of a ``tof.frame``:

* :func:`duckmind.duckbody.tof_reproject.ranges_from_frame` — the Reprojector port's reader
  (``tof::Frame::zone``: status 5 or 9 with a positive distance is a range, anything else is not);
* :func:`duckmind.duckbody.depth.accept_weak` — DS-v1's reader, which also takes the weak
  statuses 4 and 13 when the previous frame read the same zone within 3 cm (spec §5.1, a
  deliberate extension: with no previous frame it must agree with the crate exactly).

The crate's three zone classes — a range, "no target" (255: empty space) and "unusable" (any
other status) — collapse to "range or not" in both readers; the tests pin that both agree with
the crate on which zones carry a range, including ragged frames (fewer distances or statuses
than zones must read as no target, never raise).
"""

from __future__ import annotations

import re

import pytest

from duckmind import paths
from duckmind.duckbody import tof_reproject as tr

UPSTREAM = paths.microduck_dir() / "tof/src/lib.rs"
ZONES = 64


def crate_zone(distance_mm: list[int], status: list[int], index: int) -> tuple[str, float | None]:
    """``tof::Frame::zone`` as written upstream: (class, metres)."""
    code = status[index] if index < len(status) else 255
    mm = distance_mm[index] if index < len(distance_mm) else 0
    if code in (5, 9):
        return ("range", mm / 1000.0) if mm > 0 else ("unusable", None)
    if code == 255:
        return ("no_target", None)
    return ("unusable", None)


def frame(status: int, distance_mm: int) -> tuple[list[int], list[int]]:
    return [distance_mm] * ZONES, [status] * ZONES


def _depth():
    return pytest.importorskip("duckmind.duckbody.depth")


def _both(
    distance_mm: list[int], status: list[int]
) -> tuple[list[float | None], list[float | None]]:
    depth = _depth()
    reproj = tr.ranges_from_frame(distance_mm, status)
    reproj += [None] * (ZONES - len(reproj))
    ds, _keep, used = depth.accept_weak(depth.ToFFrame.of(0, distance_mm, status), None, 30)
    assert used == 0, "no previous frame: no weak reading can be used"
    return reproj, ds


def test_status_decides_what_a_distance_means() -> None:
    cases = [
        ((5, 1234), ("range", 1.234)),
        ((9, 500), ("range", 0.5)),
        ((255, 0), ("no_target", None)),
        ((4, 900), ("unusable", None)),
        ((5, -3), ("unusable", None)),  # a valid status with a negative range is not a range
    ]
    for (code, mm), expected in cases:
        d, s = frame(code, mm)
        assert crate_zone(d, s, 0) == expected
        reproj, ds = _both(d, s)
        want = expected[1]
        assert reproj[0] == want, (code, mm)
        assert ds[0] == want, (code, mm)


def test_valid_count_counts_ranges_only() -> None:
    for (code, mm), count in (((5, 1000), ZONES), ((255, 0), 0), ((4, 1000), 0)):
        d, s = frame(code, mm)
        reproj, ds = _both(d, s)
        assert sum(r is not None for r in reproj) == count
        assert sum(r is not None for r in ds) == count


def test_a_ragged_frame_reads_as_no_target() -> None:
    distance_mm, status = [1000] * 3, [5] * 2
    assert crate_zone(distance_mm, status, 0) == ("range", 1.0)
    assert crate_zone(distance_mm, status, 2) == ("no_target", None), "missing status"
    assert crate_zone(distance_mm, status, 99) == ("no_target", None), "past the end"
    reproj, ds = _both(distance_mm, status)
    assert reproj[0] == 1.0 and ds[0] == 1.0
    assert reproj[2] is None and ds[2] is None
    assert len(ds) == ZONES and all(r is None for r in ds[3:])


def test_a_ragged_frame_with_more_statuses_than_distances() -> None:
    """The other ragged way round: statuses beyond the distances have no distance (0) and are
    not ranges, whatever they say."""
    reproj, ds = _both([700], [5] * 5)
    assert reproj[0] == 0.7 and ds[0] == 0.7
    assert all(r is None for r in ds[1:])


def test_weak_status_only_with_a_close_previous_reading() -> None:
    """DS-v1's extension: a weak zone (4 or 13) counts only within 3 cm of the previous frame's
    reading of the same zone; anything else stays the crate's 'unusable'."""
    depth = _depth()
    prev = depth.ToFFrame.of(0, [900] * ZONES, [5] * ZONES)
    _, keep, _ = depth.accept_weak(prev, None, 30)
    near = depth.ToFFrame.of(1, [920] * ZONES, [4] * ZONES)
    far = depth.ToFFrame.of(1, [950] * ZONES, [13] * ZONES)
    got_near, _, used_near = depth.accept_weak(near, keep, 30)
    got_far, _, used_far = depth.accept_weak(far, keep, 30)
    assert used_near == ZONES and all(r == 0.92 for r in got_near)
    assert used_far == 0 and all(r is None for r in got_far)


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="needs the upstream checkout (read-only)")
def test_every_upstream_zone_test_is_replayed() -> None:
    upstream = set(re.findall(r"#\[test\]\s*\n\s*fn (\w+)", UPSTREAM.read_text(encoding="utf-8")))
    here = {name.removeprefix("test_") for name in globals() if name.startswith("test_")}
    assert upstream and upstream <= here, f"not replayed: {sorted(upstream - here)}"
