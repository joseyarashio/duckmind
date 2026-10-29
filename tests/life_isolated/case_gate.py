"""The G2 gate's harness, small (docs/design/GROWTH-G2-2026-10-10.md; runs in a child process, see
tests/test_life_isolated.py): the same neural duck, the bridge and Duck Core, three care styles,
a few days each — the styles already differ, nothing but the neural duck's experiences and the
clock reaches Duck Core, and the save replays to the same state."""

from __future__ import annotations

import sys
from pathlib import Path

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)

GROWTH = Path(__file__).resolve().parents[2] / "tools" / "growth"
if not GROWTH.is_dir():
    import pytest

    pytest.skip("tools/growth is not in this checkout", allow_module_level=True)
sys.path.insert(0, str(GROWTH))
import care_harness as H  # noqa: E402


def life(style: str, **kw):
    return H.run_life(1, style, days=9, care_from=1, care_days=8, session_s=60, **kw)


def test_the_three_care_styles_make_three_different_ducks_from_the_same_seed() -> None:
    attentive, neglectful, rough = life("attentive"), life("neglectful"), life("rough")
    assert attentive["bond_owner"] > 2000 == neglectful["bond_owner"] == rough["bond_owner"]
    assert len(attentive["prefs"]) > len(neglectful["prefs"]) >= len(rough["prefs"]) == 0
    assert rough["avoid_spots"] >= 1 and attentive["avoid_spots"] == neglectful["avoid_spots"] == 0
    care = {
        r["style"]: r["log_kinds"].get("care.episode", 0) for r in (attentive, neglectful, rough)
    }
    assert (
        care["attentive"] > 5 * care["neglectful"] > 0
    )  # (care before the duckling stage is logged, not yet counted toward its branch)
    for result in (attentive, neglectful, rough):
        assert result["replay_identical"] is True and result["gd"] == 9


def test_with_nobody_there_nothing_but_the_neural_ducks_own_life_reaches_duck_core() -> None:
    idle = H.run_life(1, "none", days=6, care_from=10**6, session_s=60)
    assert set(idle["inputs"]) == {"advance", "bridge", "experience"}
    assert set(idle["bridge_sent"]) <= {"explore", "dock", "sleep", "voiced", "company", "visit"}
    assert not {"touch", "poke", "lift", "hello", "name", "voice", "fall"} & set(
        idle["bridge_sent"]
    )
    shadow = {"duck.expression", "duck.walk", "duck.decision", "duck.sleep_stir", "voice.babble"}
    assert not shadow & set(idle["log_kinds"]) - {"duck.reaction"}
