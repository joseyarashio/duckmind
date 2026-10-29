import hashlib
from copy import deepcopy

import pytest

from duckmind.policy_approval import loaded_official_v5_walk, official_v5_walk_file_matches


def official_policy_report():
    return {
        "mode": "walk",
        "enabled": True,
        "homed": True,
        "change_error": None,
        "slots": [
            {
                "slot": "walk",
                "path": "/opt/robot/policies/current/velstand.onnx",
                "origin": "official",
                "overridden": False,
                "error": None,
            }
        ],
    }


def test_first_board_accepts_loaded_official_v5_walk() -> None:
    assert loaded_official_v5_walk(official_policy_report())


def test_official_walk_file_must_match_exact_reviewed_bytes(tmp_path) -> None:
    model = tmp_path / "velstand.onnx"
    model.write_bytes(b"reviewed-v5-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert official_v5_walk_file_matches(model, expected_sha256=digest)

    model.write_bytes(b"different-model")
    assert not official_v5_walk_file_matches(model, expected_sha256=digest)


def test_official_walk_file_check_fails_closed_on_missing_or_nonregular_file(tmp_path) -> None:
    model = tmp_path / "velstand.onnx"
    digest = hashlib.sha256(b"reviewed-v5-model").hexdigest()
    assert not official_v5_walk_file_matches(model, expected_sha256=digest)
    model.mkdir()
    assert not official_v5_walk_file_matches(model, expected_sha256=digest)


def test_official_walk_file_check_rejects_oversized_input(tmp_path, monkeypatch) -> None:
    model = tmp_path / "velstand.onnx"
    model.write_bytes(b"reviewed-v5-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setattr("duckmind.policy_approval.MAX_WALK_BYTES", 8)
    assert not official_v5_walk_file_matches(model, expected_sha256=digest)


def test_official_walk_file_check_rejects_release_symlink_switch_during_hash(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "releases/first"
    second = tmp_path / "releases/second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "velstand.onnx").write_bytes(b"reviewed-v5-model")
    (second / "velstand.onnx").write_bytes(b"different-model")
    current = tmp_path / "current"
    current.symlink_to(first, target_is_directory=True)
    model = current / "velstand.onnx"
    digest = hashlib.sha256(b"reviewed-v5-model").hexdigest()
    real_file_digest = hashlib.file_digest

    def switch_release(source, algorithm):
        result = real_file_digest(source, algorithm)
        current.unlink()
        current.symlink_to(second, target_is_directory=True)
        return result

    monkeypatch.setattr("duckmind.policy_approval.hashlib.file_digest", switch_release)
    assert not official_v5_walk_file_matches(model, expected_sha256=digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "roller"),
        ("enabled", False),
        ("homed", False),
        ("homed", None),
        ("change_error", "failed reload"),
    ],
)
def test_first_board_rejects_unready_policy_report(field, value) -> None:
    report = official_policy_report()
    report[field] = value
    assert not loaded_official_v5_walk(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "/var/lib/robot/policies/velstand.onnx"),
        ("origin", "community"),
        ("overridden", True),
        ("error", "override failed"),
    ],
)
def test_first_board_rejects_same_basename_override_or_slot_error(field, value) -> None:
    report = official_policy_report()
    report["slots"][0][field] = value
    assert not loaded_official_v5_walk(report)


def test_first_board_rejects_missing_or_duplicate_walk_slot() -> None:
    report = official_policy_report()
    report["slots"] = []
    assert not loaded_official_v5_walk(report)
    report["slots"] = [
        deepcopy(official_policy_report()["slots"][0]),
        deepcopy(official_policy_report()["slots"][0]),
    ]
    assert not loaded_official_v5_walk(report)


@pytest.mark.parametrize("field", ["change_error", "slots"])
def test_first_board_rejects_missing_required_report_field(field) -> None:
    report = official_policy_report()
    report.pop(field)
    assert not loaded_official_v5_walk(report)


def test_first_board_rejects_missing_walk_error_field() -> None:
    report = official_policy_report()
    report["slots"][0].pop("error")
    assert not loaded_official_v5_walk(report)


@pytest.mark.parametrize("report", [None, {}, {"slots": "not a list"}])
def test_first_board_rejects_malformed_report(report) -> None:
    assert not loaded_official_v5_walk(report)
