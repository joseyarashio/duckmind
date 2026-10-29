"""The Stage-1 skill table (:mod:`duckmind.skilltrain.table`) against robotd-params' rules.

The TOML section must parse, carry exactly ``SkillDef``'s keys (robotd refuses unknown ones),
sit after our launcher's ``[policy]`` and ``[control]`` tables, and resolve in robotd's order
(v5 built-ins first, new names appended, ``"none"`` removing); :func:`problems` must flag what
robotd refuses and the chain + unwind trap.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from duckmind.skilltrain import table as T

SKILLDEF_KEYS = {"name", "path", "duration", "command", "unwind", "unwind_s", "chain", "params"}


@pytest.fixture()
def onnx(tmp_path: Path) -> dict[str, str]:
    files = {}
    for name in ("startle", "comfort", "velstand"):
        p = tmp_path / f"{name}.onnx"
        p.write_bytes(b"not a real network")
        files[name] = str(p)
    return files


def launcher_toml(section: str) -> str:
    """What the live launcher writes: [policy], then fid:motor's [control], then the skills."""
    return (
        '[policy]\nenabled = true\nwalk = "/p/velstand.onnx"\nstand = "none"\n'
        'sitstand = "/p/alpha_sitstand.onnx"\n'
        "\n[control]\nhead_alpha = 1.0\n" + section
    )


def test_the_stage1_table_is_valid_and_parses_after_the_launchers_tables(onnx) -> None:
    entries = T.stage1(onnx["startle"], onnx["comfort"], onnx["velstand"])
    assert T.problems(entries) == []
    doc = tomllib.loads(launcher_toml(T.toml_section(entries)))
    skills = doc["policy"]["skill"]
    assert [s["name"] for s in skills] == [e.name for e in entries]
    for s in skills:
        assert set(s) <= SKILLDEF_KEYS, set(s) - SKILLDEF_KEYS
        assert set(s["params"]) == {"action_scale", "gain_ratio"}  # pinned, always
    assert doc["control"]["head_alpha"] == 1.0
    by = {s["name"]: s for s in skills}
    assert by["startle_4"]["command"] == [1.0, 0.0, 0.0]
    assert by["hop_back"]["command"] == [1.0, 1.0, 0.0]
    assert by["stretch_sit"]["command"] == [0.0, -1.0, 1.0]
    assert by["skill_abort"]["path"] == onnx["velstand"] and by["skill_abort"]["chain"] is True
    assert not any(s["chain"] and s["unwind_s"] > 0 for s in skills)
    assert len(T.resolve(entries)) == 3 + len(entries) <= T.MAX_SKILLS


def test_resolution_follows_robotd_builtins_first_then_ours_none_removes() -> None:
    ours = [
        T.SkillDef("startle_1", "/a.onnx", 0.5, action_scale=1.0, gain_ratio=1.0),
        T.SkillDef("kick_left", "none", 0.5),
        T.SkillDef("roulade", "/r2.onnx", 1.5, chain=True, action_scale=1.0, gain_ratio=1.0),
    ]
    names = [d.name for d in T.resolve(ours, "/pol")]
    assert names == ["roulade", "kick_right", "startle_1"]
    assert T.resolve(ours, "/pol")[0].path == "/r2.onnx"  # replaced in place


def test_problems_catch_what_robotd_refuses_and_the_chain_unwind_trap(onnx) -> None:
    bad = [
        T.SkillDef("sit_toggle", onnx["startle"], 0.5, action_scale=1.0, gain_ratio=1.0),
        T.SkillDef(
            "held", onnx["startle"], 0.5, unwind_s=0.8, chain=True, action_scale=1.0, gain_ratio=1.0
        ),
        T.SkillDef("free", onnx["startle"], 0.5),
        T.SkillDef("rel", "relative.onnx", 0.5, action_scale=1.0, gain_ratio=1.0),
        T.SkillDef("zero", onnx["startle"], 0.0, action_scale=1.0, gain_ratio=1.0),
    ]
    text = " | ".join(T.problems(bad))
    assert "belongs to robotd itself" in text
    assert "never chains" in text
    assert "pin params.action_scale" in text
    assert "must be absolute" in text
    assert "positive number of seconds" in text
    with pytest.raises(ValueError):
        T.toml_section(bad)
    many = [
        T.SkillDef(f"s{i}", onnx["startle"], 0.5, action_scale=1.0, gain_ratio=1.0)
        for i in range(28)
    ]
    assert any("at most 30" in p for p in T.problems(many))


def test_save_load_round_trip_and_the_launcher_env_hook(onnx, tmp_path: Path) -> None:
    entries = T.stage1(onnx["startle"], onnx["comfort"], onnx["velstand"])
    path = T.save(tmp_path / "skills.json", entries, exported="test")
    back = T.load(path)
    assert [e.to_json() for e in back] == [e.to_json() for e in entries]
    assert T.robotd_skill_section_from_env({}) == ""
    section = T.robotd_skill_section_from_env({T.SKILLS_ENV: str(path)})
    assert section == T.toml_section(entries)
    assert T.skill_names_from_env({T.SKILLS_ENV: str(path)})[0] == "startle_1"


def test_the_dummy_table_uses_the_walk_network_in_the_stage1_window_shapes(onnx) -> None:
    entries = T.dummy(onnx["velstand"])
    by = {e.name: e.skill for e in entries}
    assert by["dummy_stand"].duration == 0.5 and by["dummy_stand"].unwind_s == 0.8
    assert by["dummy_chain"].chain and by["dummy_chain"].unwind_s == 0.0
    problems = T.problems(entries)
    assert len(problems) == 1 and "dummy_chain_unwind" in problems[0]
