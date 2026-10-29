"""tr:gate wiring: a gated skill table carries its servo delay and guard rules (no physics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from duckmind.skilltrain import gt_wiring as W
from duckmind.skilltrain import guard as G
from duckmind.skilltrain.table import SKILLS_ENV


def _table(tmp_path: Path, **blocks: object) -> Path:
    doc = {"format": "duckmind-skill-table-1", "skills": [], **blocks}
    path = tmp_path / "skills.json"
    path.write_text(json.dumps(doc))
    return path


def test_no_table_changes_nothing() -> None:
    assert W.body_env({"A": "1"}, {}) == {"A": "1"}
    assert W.guard_rules({}) == G.DEFAULT_RULES


def test_the_table_delay_reaches_the_body_server(tmp_path: Path) -> None:
    path = _table(tmp_path, body={"command_delay_ms": 20})
    env = W.body_env({}, {SKILLS_ENV: str(path)})
    assert env[W.DELAY_ENV] == "20"
    # an explicit setting wins over the table
    env = W.body_env({}, {SKILLS_ENV: str(path), W.DELAY_ENV: "0"})
    assert env[W.DELAY_ENV] == "0"


def test_a_bad_delay_is_refused(tmp_path: Path) -> None:
    path = _table(tmp_path, body={"command_delay_ms": 500})
    with pytest.raises(ValueError):
        W.body_env({}, {SKILLS_ENV: str(path)})


def test_guard_rules_follow_the_table(tmp_path: Path) -> None:
    doc = {
        "guard": {
            "global_per_hour": 40,
            "rules": {"hop_back": {"per_hour": 2, "min_gap_s": 90, "postures": ["stand"]}},
        }
    }
    rules = W.guard_rules(doc)
    assert rules["hop_back"].per_hour == 2
    assert rules["hop_back"].postures == frozenset({"stand"})
    assert rules["startle_1"] == G.DEFAULT_RULES["startle_1"]
    path = _table(tmp_path, **doc)
    guard = W.skill_guard(["hop_back", "startle_1"], {SKILLS_ENV: str(path)})
    assert guard.global_per_hour == 40
    walking = G.BodyView("walk", twist=0.3)
    assert not guard.request("hop_back", walking, 0.0).ok  # walking is not allowed any more
    standing = G.BodyView("walk", twist=0.0)
    assert guard.request("hop_back", standing, 0.1).reason == "settle"  # it has just stopped
    for k in range(80):  # the body link's watch() every tick: standing still 1.6 s
        guard.watch(standing, 0.1 + 0.02 * k)
    assert guard.request("hop_back", standing, 1.8).ok


def test_the_picker_only_names_loaded_skills() -> None:
    assert W.pick_loaded("shake_1", {"startle_1"}) is None
    assert W.pick_loaded("startle_1", {"startle_1"}) == "startle_1"
    assert W.pick_loaded(None, {"startle_1"}) is None


def test_the_table_carries_speed_limits_settle_times_and_holds() -> None:
    doc = {
        "guard": {
            "rules": {
                "freeze_4": {"postures": ["stand", "walk"], "max_walk_mps": 0.40},
                "shake_1": {"settle_s": 2.0},
            }
        }
    }
    rules = W.guard_rules(doc)
    assert rules["freeze_4"].max_walk_mps == 0.40 and rules["freeze_4"].hold
    assert rules["freeze_4"].continues == frozenset({"startle", "freeze"})
    assert rules["shake_1"].settle_s == 2.0
    assert rules["startle_4"].max_walk_mps == G.DEFAULT_RULES["startle_4"].max_walk_mps


def test_every_network_is_pinned_and_a_changed_one_is_refused(tmp_path: Path) -> None:
    startle = tmp_path / "startle.onnx"
    walk = tmp_path / "velstand.onnx"
    startle.write_bytes(b"startle network")
    walk.write_bytes(b"walk network v5")
    skills = [
        {"name": "startle_1", "path": str(startle), "duration": 0.5, "family": "startle"},
        {"name": "skill_abort", "path": str(walk), "duration": 0.02, "chain": True,
         "action_scale": 0.9, "gain_ratio": 1.0, "family": "abort"},
    ]  # fmt: skip
    skills[0].update({"action_scale": 1.0, "gain_ratio": 1.0})
    only_skill: dict[str, Any] = {"startle": {"path": str(startle), "sha256": W.sha256_of(startle)}}
    doc: dict[str, Any] = {
        "format": "duckmind-skill-table-1",
        "skills": skills,
        "files": only_skill,
    }
    problems = W.check_pins(doc)
    assert len(problems) == 1 and "skill_abort" in problems[0]  # the abort's velstand: unpinned
    doc["files"] = {**only_skill, "walk": {"path": str(walk), "sha256": W.sha256_of(walk)}}
    assert W.check_pins(doc) == []
    path = tmp_path / "skills.json"
    path.write_text(json.dumps(doc))
    text = W.verified_section({SKILLS_ENV: str(path)})
    assert text.count("[[policy.skill]]") == 2
    walk.write_bytes(b"walk network v6 (an OTA update)")
    assert any("changed" in p for p in W.check_pins(doc))
    with pytest.raises(ValueError):
        W.verified_section({SKILLS_ENV: str(path)})
    assert W.verified_section({}) == ""
