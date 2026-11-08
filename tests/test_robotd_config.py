"""A real board's two-layer robotd.toml as Duck Mind reads it (real duck prep 2026-11-08)."""

from __future__ import annotations

from pathlib import Path

from duckmind.robotd_config import RobotdConfig, board_config, merge_layers

BASE = """
[board]
version = "beta"

[pickup]
pause_threshold = 0.8

[idle]
after_s = 2.0

[imu_head]
enabled = true

[[policy.skill]]
name = "a"
"""


def _write(folder: Path, base: str, user: str | None) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "base.toml").write_text(base)
    if user is not None:
        (folder / "user.toml").write_text(user)
    return folder / "base.toml", folder / "user.toml"


def test_an_untouched_board_runs_robotds_defaults_idle_and_pickup_on(tmp_path) -> None:
    config = board_config(*_write(tmp_path, BASE, None))
    assert config == RobotdConfig(
        board="beta", idle=True, pickup=True, limp_fall=True, base_found=True, user_found=False
    )
    assert any("[idle] enabled" in line for line in config.findings())


def test_the_owner_layer_overrides_key_by_key(tmp_path) -> None:
    user = "[idle]\nenabled = false\n\n[pickup]\nenabled = false\n\n[safety]\nlimp_fall = false\n"
    config = board_config(*_write(tmp_path, BASE, user))
    assert (config.board, config.idle, config.pickup, config.limp_fall) == (
        "beta",
        False,
        False,
        False,
    )
    assert config.user_found and config.findings() == []


def test_merge_is_robotds_tables_key_by_key_lists_whole_renamed_sections_first() -> None:
    base = {"idle": {"after_s": 2.0, "enabled": True}, "imu_head": {"enabled": True},
            "policy": {"skill": [{"name": "a"}, {"name": "b"}]}}  # fmt: skip
    user = {"idle": {"enabled": False}, "pad_imu_head_control": {"gain": 2},
            "policy": {"skill": [{"name": "c"}]}}  # fmt: skip
    merged = merge_layers(base, user)
    assert merged["idle"] == {"after_s": 2.0, "enabled": False}
    assert merged["pad_imu_head_control"] == {"enabled": True, "gain": 2}
    assert "imu_head" not in merged
    assert merged["policy"]["skill"] == [{"name": "c"}]
    assert base["idle"]["enabled"] is True  # the layers themselves are not changed


def test_a_broken_owner_layer_is_set_aside_and_no_base_is_no_board(tmp_path) -> None:
    config = board_config(*_write(tmp_path, BASE, "[idle\nenabled = "))
    assert config is not None and config.idle and not config.user_found
    assert board_config(tmp_path / "absent.toml", tmp_path / "user.toml") is None


def test_an_unset_board_is_zero3_and_a_none_pickup_model_is_off(tmp_path) -> None:
    config = board_config(*_write(tmp_path, '[pickup]\nmodel = "none"\n', None))
    assert config.board == "zero3" and config.pickup is False
