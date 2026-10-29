"""The no-push profile isolates a training diagnostic, never a qualification."""

from types import SimpleNamespace

import pytest

from duckmind.offline_eval import EVALUATION_PROFILES, configure_probe_events


def test_play_no_push_uses_official_play_configuration_but_is_diagnostic() -> None:
    profile = EVALUATION_PROFILES["play-no-push"]
    assert profile.play is True
    assert profile.diagnostic_only is True
    assert profile.push_enabled is False
    assert profile.push_velocity_zeroed is False


def test_play_no_push_removes_only_the_push_event() -> None:
    push = object()
    reset = object()
    config = SimpleNamespace(events={"push_robot": push, "reset_base": reset})

    configure_probe_events(config, EVALUATION_PROFILES["play-no-push"])

    assert config.events == {"reset_base": reset}


def test_formal_play_preserves_push_event() -> None:
    push = object()
    config = SimpleNamespace(events={"push_robot": push})

    configure_probe_events(config, EVALUATION_PROFILES["play"])

    assert config.events["push_robot"] is push
    assert EVALUATION_PROFILES["play"].diagnostic_only is False


def test_play_zero_push_retains_event_and_schedule_but_zeroes_velocity() -> None:
    push = SimpleNamespace(
        interval_range_s=(0.5, 1.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
    config = SimpleNamespace(events={"push_robot": push, "reset_base": object()})
    profile = EVALUATION_PROFILES["play-zero-push"]

    configure_probe_events(config, profile)

    assert profile.play and profile.diagnostic_only and profile.push_enabled
    assert profile.push_velocity_zeroed
    assert config.events["push_robot"] is push
    assert push.interval_range_s == (0.5, 1.0)
    assert push.params["velocity_range"] == {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    assert "reset_base" in config.events


def test_play_zero_push_requires_official_velocity_range() -> None:
    config = SimpleNamespace(events={"push_robot": SimpleNamespace(params={})})

    with pytest.raises(ValueError, match="velocity_range"):
        configure_probe_events(config, EVALUATION_PROFILES["play-zero-push"])


def test_no_push_profile_refuses_upstream_configuration_without_expected_event() -> None:
    config = SimpleNamespace(events={"reset_base": object()})

    with pytest.raises(ValueError, match="push_robot"):
        configure_probe_events(config, EVALUATION_PROFILES["play-no-push"])
