"""The simulator opens a pre-phase-4 save on the frozen v4 engine (phase-4 review, test gap; the
path 蛋黃 takes when :8765 restarts).

A save written before phase 4 has no ``engine_rev`` and keeps the outcome rates in its config.
``SimState`` restores it with today's expression caps file (the server's ``caps_file`` override):
it must come back on the frozen engine with exactly the chain the frozen engine gives for that
config, caps and inputs, and saving it again must keep it a v4_0 duck.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duckmind.duckcore.api import Config
from duckmind.simweb.server import SimState

CAPS = Path(__file__).with_name("golden") / "caps" / "expression_caps.json"
INPUTS = [
    {"method": "act", "params": {"action": "power_on", "params": {}, "persona": "owner"}},
    {"method": "set_setting", "params": {"key": "floor", "value": True}},
    {"method": "advance", "params": {"minutes": 90}},
    {"method": "act", "params": {"action": "touch", "params": {"seconds": 5}, "persona": "child"}},
    {"method": "advance", "params": {"seconds": 45}},
]


def _v4_save() -> dict[str, Any]:
    """A save as the simulator wrote it before phase 4 (蛋黃's shape: no nest, legacy)."""
    from duckmind.engines.v4_0.duckcore import api as v4

    config = v4.config_json(v4.Session(v4.Config(seed=10, name="蛋黃", fall_ppm_per_s=90)))
    assert "engine_rev" not in config and config["fall_ppm_per_s"] == 90
    return {"config": config, "inputs": INPUTS, "realtime": False}


def _frozen_head(config: dict[str, Any], caps: str) -> str:
    from duckmind.engines.v4_0.duckcore import api as v4

    session = v4.Session(v4.Config(**{**config, "caps_file": caps}))
    for item in INPUTS:
        v4.dispatch(session, item["method"], item["params"])
    return str(session.core.log.head)


def test_an_old_save_restores_on_the_frozen_engine_with_the_servers_caps(tmp_path: Path) -> None:
    saved = _v4_save()
    path = tmp_path / "simweb-duck.json"
    path.write_text(json.dumps(saved, ensure_ascii=False))
    state = SimState(Config(seed=1, name="別的", caps_file=str(CAPS)), state_file=path)
    session = state.session
    assert type(session.core).__module__ == "duckmind.engines.v4_0.duckcore.engine"
    assert session.core.log.head == _frozen_head(saved["config"], str(CAPS))
    assert session.core.log.head != _frozen_head(saved["config"], ""), "the caps must matter"
    state._saved_inputs = -1
    state.save()
    again = json.loads(path.read_text())
    assert again["config"]["engine_rev"] == "v4_0" and again["world"] is None
    assert again["config"]["fall_ppm_per_s"] == 90 and again["inputs"] == INPUTS
    reopened = SimState(Config(seed=1, caps_file=str(CAPS)), state_file=path)
    assert reopened.session.core.log.head == session.core.log.head


def test_the_new_duck_dialogs_clumsy_duck_falls_in_its_world(tmp_path: Path) -> None:
    """The page audit's one changed control (dlg-clumsy, phase-4 integration): the dialog's
    clumsy duck still falls more often; its rate is in its world header now, not its hardware."""
    from duckmind.duckcore import api

    state = SimState(Config(seed=1), state_file=tmp_path / "duck.json")
    state.reset({"seed": 12, "fall_ppm_per_s": 150_000, "nest": True})
    session = state.session
    assert isinstance(session, api.Session) and session.config.engine_rev == "v4_1"
    assert session.world is not None and session.world["rates"]["fall_ppm_per_s"] == 150_000
    assert "fall_ppm_per_s" not in api.config_json(session)
    state.reset({"seed": 12})
    assert state.session.world is not None
    assert state.session.world["rates"]["fall_ppm_per_s"] == 50
