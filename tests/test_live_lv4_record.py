"""Live voice, LV4: what the fit left in the repository is what the code says it is.

``docs/design/live-voice/lv4/`` holds the parameters the search found, its history, and every run of the
held-out evaluation (seeds 13-28, days and nights). These tests do not run the simulation: they check
that the defaults of ``DriveParams`` are those parameters, that the recorded runs are what they claim to
be, and that the seven targets of ``LV4-TARGETS.md`` (as ``convday.targets_report``) say about them what
the notes say. A change to the targets' code or to the defaults that no longer matches the record fails
here, and is then either wrong or a new fit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from duckmind.live import convday as D
from duckmind.live.drives import SEARCH_SPACE, DriveParams
from tests.checkout_parts import needs_design

pytestmark = [pytest.mark.usefixtures("no_network"), needs_design]

LV4 = Path(__file__).resolve().parents[1] / "docs" / "design" / "live-voice" / "lv4"


def runs() -> list[dict]:
    with gzip.open(LV4 / "heldout-13-28-runs.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def test_the_defaults_are_the_parameters_the_search_found() -> None:
    found = json.loads((LV4 / "drive-params.json").read_text(encoding="utf-8"))["params"]
    defaults = DriveParams().as_dict()
    assert set(found) == set(defaults)
    for name, value in found.items():
        assert defaults[name] == pytest.approx(value, abs=1e-5), name
    # (only the new drives' parameters were searched; the rest were fixed)
    searched = {s.name for s in SEARCH_SPACE}
    assert {n for n in found if n not in searched} == {"talk_sat_tag", "approach_gain"}
    for spec in SEARCH_SPACE:
        assert spec.low - 1e-9 <= found[spec.name] <= spec.high + 1e-9, spec.name


def test_the_search_ran_the_generations_the_notes_say_on_the_seeds_the_targets_file_says() -> None:
    a = (LV4 / "stageA-history.jsonl").read_text(encoding="utf-8").splitlines()
    b = (LV4 / "stageB-history.jsonl").read_text(encoding="utf-8").splitlines()
    assert (len(a), len(b)) == (40, 11)
    meta_a = json.loads((LV4 / "stageA-meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((LV4 / "stageB-meta.json").read_text(encoding="utf-8"))
    assert meta_a["seeds"] == list(range(1, 7)) and meta_b["seeds"] == list(range(1, 13))
    assert max(meta_a["seeds"] + meta_b["seeds"]) < min(
        D.HELD_OUT_SEEDS
    )  # (nothing held out was fitted)
    assert D.TRAIN_SEEDS == tuple(range(1, 13)) and D.HELD_OUT_SEEDS == tuple(range(13, 29))
    assert meta_a["space"] == [s.name for s in SEARCH_SPACE] == meta_b["space"]


def test_the_ledger_has_the_targets_before_the_first_search_and_no_held_out_run_before_the_last() -> (
    None
):
    lines = [
        ln for ln in (LV4 / "ledger.txt").read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    stamps = [ln for ln in lines if ln[:4] == "2026" and ln[10] == "T"]
    assert stamps[0] < "2026-10-02T19:19:14Z"  # (the targets commit, 18:42:16Z, then stage A)
    assert any(ln.startswith("FIRST HELD-OUT RUN") for ln in lines)
    assert [ln for ln in lines if "stage A start" in ln] and [
        ln for ln in lines if "stage B end" in ln
    ]


def test_the_recorded_runs_are_the_four_presets_on_the_sixteen_held_out_seeds_by_day_and_night() -> (
    None
):
    rs = runs()
    assert len(rs) == 4 * 16 * 2
    assert {r["seed"] for r in rs} == set(D.HELD_OUT_SEEDS)
    assert {r["preset"] for r in rs} == set(D.PRESETS)
    assert (
        sum(1 for r in rs if r["kind"] == "day") == sum(1 for r in rs if r["kind"] == "night") == 64
    )
    assert all(r["faults"] == 0 for r in rs)
    assert all(len(r["proposals"]) == (29 if r["kind"] == "day" else 6) for r in rs)


def test_the_targets_say_of_the_recorded_runs_what_the_stored_report_says() -> None:
    stored = json.loads((LV4 / "heldout-13-28-report.json").read_text(encoding="utf-8"))["report"]
    report = D.targets_report(runs())
    assert [v["name"] for v in report["verdicts"]] == [v["name"] for v in stored["verdicts"]]
    assert [v["pass"] for v in report["verdicts"]] == [v["pass"] for v in stored["verdicts"]]
    for preset, numbers in report["numbers"].items():
        for key in ("rate_all", "rate_stranger", "rate_warm", "latency_median", "tags_per_reply"):
            assert numbers[key] == pytest.approx(stored["numbers"][preset][key])
    assert stored["params"] == json.loads((LV4 / "drive-params.json").read_text())["params"]


def test_the_notes_numbers_on_the_held_out_seeds() -> None:
    """The numbers LV4-drives.md prints, from the runs themselves (so the notes cannot drift)."""
    report = D.targets_report(runs())
    got = {v["name"].split(" ")[0]: v["pass"] for v in report["verdicts"]}
    assert got == {"1": True, "2": True, "3": True, "4": True, "5": True, "6": True, "7": False}
    n = report["numbers"]
    assert n["social_songbird"]["rate_all"] == pytest.approx(0.8795, abs=5e-4)
    assert n["shy_snuggler"]["rate_stranger"] == pytest.approx(0.0781, abs=5e-4)
    assert n["shy_snuggler"]["rate_warm"] == pytest.approx(0.7981, abs=5e-4)
    assert n["lazy_dreamer"]["rate_all"] == pytest.approx(0.3438, abs=5e-4)
    assert n["bold_explorer"]["rate_all"] == pytest.approx(0.3355, abs=5e-4)
    assert n["bold_explorer"]["tags_per_reply"] == pytest.approx(1.1974, abs=5e-4)
    assert all(v["violations"] == 0 for v in n.values())
    # target 7 fails on two runs whose introduction could not be bound (the duck was in a corner,
    # facing the wall: nobody to put the name on); everywhere else it holds
    days = [r for r in runs() if r["kind"] == "day"]
    unnamed = [
        (r["preset"], r["seed"])
        for r in days
        if not any(p["name"] == "媽媽" for p in r["people"])
        or not any(p["name"] == "爸爸" for p in r["people"])
    ]
    assert sorted(unnamed) == [("bold_explorer", 20), ("shy_snuggler", 27)]
