"""The simulator's robotd.toml switches off robotd's own idle head (0.16.1+) and pickup pause (0.15.2+) on
a daemon that has them, and never names them for one that does not (robotd refuses unknown fields)."""

import tomllib
from pathlib import Path

import pytest

from duckmind import livebody as LB

PARAMS_0185 = """
pub struct Params {
    pub safety: SafetyParams,
    pub pickup: PickupParams,
    pub idle: IdleParams,
}
pub struct SafetyParams {
    pub limp_fall: bool,
}
"""
PARAMS_0150 = """
pub struct Params {
    pub safety: SafetyParams,
}
"""


def checkout(tmp_path: Path, text: str) -> Path:
    src = tmp_path / "robotd-params" / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(text)
    return tmp_path


def test_a_newer_robotd_gets_its_idle_head_and_pickup_pause_switched_off(tmp_path) -> None:
    doc = tomllib.loads(LB.robotd_upstream_section(checkout(tmp_path, PARAMS_0185), {}))
    assert doc["idle"] == {"enabled": False}
    assert doc["pickup"] == {"enabled": False}
    assert "safety" not in doc  # limp-fall: the board's own default


def test_the_015_robotd_never_sees_the_new_sections(tmp_path) -> None:
    assert LB.robotd_upstream_section(checkout(tmp_path, PARAMS_0150), {}) == ""
    assert LB.robotd_upstream_section(tmp_path / "missing", {}) == ""


def test_each_choice_can_be_overridden(tmp_path) -> None:
    root = checkout(tmp_path, PARAMS_0185)
    env = {"DUCKMIND_ROBOTD_IDLE": "board", "DUCKMIND_ROBOTD_PICKUP": "on",
           "DUCKMIND_ROBOTD_LIMP_FALL": "off"}  # fmt: skip
    doc = tomllib.loads(LB.robotd_upstream_section(root, env))
    assert "idle" not in doc
    assert doc["pickup"] == {"enabled": True, "model": str(root / LB.PICKUP_MODEL)}
    assert doc["safety"] == {"limp_fall": False}
    with pytest.raises(ValueError):
        LB.robotd_upstream_section(root, {"DUCKMIND_ROBOTD_IDLE": "maybe"})


def test_the_real_checkouts_when_present() -> None:
    """The two checkouts the simulator runs (the data folder's upstream-live and upstream-0185)."""
    from duckmind import paths

    for name, sections in (("upstream-live", set()), ("upstream-0185", {"idle", "pickup"})):
        root = paths.data_dir() / name / "microduck"
        if not root.is_dir():
            pytest.skip(f"no {name} checkout")
        assert set(tomllib.loads(LB.robotd_upstream_section(root, {}))) == sections
