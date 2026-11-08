import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from duckmind.policy_approval import loaded_official_walk, official_walk_fingerprint

V8_DESCRIPTION = (
    "Walking gait that also stands still at zero command; From v8 it gets up from a fall smoothly"
)


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


def official_manifest() -> dict:
    """The fields of pollen-robotics/microduck-policies' manifest.json (v5 … v8) the check reads."""
    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
        "policies": [
            {"file": "alpha_walking.onnx", "kind": "perpetual"},
            {"file": "velstand.onnx", "kind": "perpetual", "slot": "walk",
             "entry_pose": "standing", "description": V8_DESCRIPTION},
            {"file": "alpha_sitstand.onnx", "name": "sitstand", "kind": "scripted"},
        ],
    }  # fmt: skip


def official_set(folder: Path, model: bytes = b"v8-walk", version: str = "v8") -> Path:
    """A policy set as upstream's seed-policies.sh installs it; returns its walk file."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "velstand.onnx").write_bytes(model)
    (folder / "manifest.json").write_text(json.dumps(official_manifest()))
    (folder / ".source").write_text(
        f"repo=pollen-robotics/microduck-policies\nversion={version}\nfetched=2026-11-05T00:00:00Z\n"
    )
    return folder / "velstand.onnx"


def test_first_board_accepts_loaded_official_walk() -> None:
    assert loaded_official_walk(official_policy_report())


@pytest.mark.parametrize("version", ["v5", "v8"])
def test_any_official_set_with_the_velstand_contract_passes(tmp_path, version) -> None:
    # v8's walk file (sha256 2fc7f8ee…) is not the v5 one the old pin named: the contract decides
    model = official_set(tmp_path / f"seed-{version}", b"walk " + version.encode(), version)
    assert official_walk_fingerprint(model) == hashlib.sha256(model.read_bytes()).hexdigest()


def test_fingerprint_follows_the_bytes(tmp_path) -> None:
    model = official_set(tmp_path)
    first = official_walk_fingerprint(model)
    model.write_bytes(b"different-model")
    assert official_walk_fingerprint(model) not in (None, first)


@pytest.mark.parametrize(
    "spoil",
    [
        lambda m: m.update(obs_len=62),  # another observation layout
        lambda m: m.update(action_len=12),
        lambda m: m.update(model_api=2),
        lambda m: m["robot"].update(control_hz=100),
        lambda m: m["policies"][1].update(slot=None),  # velstand is not the walk slot
        lambda m: m["policies"][0].update(slot="walk"),  # two walk slots
        lambda m: m["policies"][1].update(entry_pose="seated"),
        lambda m: m.pop("policies"),
    ],
)
def test_a_set_that_changes_the_contract_fails(tmp_path, spoil) -> None:
    model = official_set(tmp_path)
    manifest = official_manifest()
    spoil(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert official_walk_fingerprint(model) is None


def test_a_file_that_is_not_the_official_sets_fails(tmp_path) -> None:
    model = official_set(tmp_path)
    (tmp_path / ".source").write_text("repo=someone/microduck-sprint\nversion=v1\n")
    assert official_walk_fingerprint(model) is None
    (tmp_path / ".source").unlink()
    assert official_walk_fingerprint(model) is None
    official_set(tmp_path)
    (tmp_path / "manifest.json").write_text("not json")
    assert official_walk_fingerprint(model) is None


def test_official_walk_file_check_fails_closed_on_missing_or_nonregular_file(tmp_path) -> None:
    model = official_set(tmp_path)
    model.unlink()
    assert official_walk_fingerprint(model) is None
    model.mkdir()
    assert official_walk_fingerprint(model) is None


def test_official_walk_file_check_rejects_oversized_input(tmp_path, monkeypatch) -> None:
    model = official_set(tmp_path, b"reviewed-v8-model")
    monkeypatch.setattr("duckmind.policy_approval.MAX_WALK_BYTES", 8)
    assert official_walk_fingerprint(model) is None


def test_official_walk_file_check_rejects_release_symlink_switch_during_hash(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "releases/first"
    second = tmp_path / "releases/second"
    official_set(first, b"reviewed-model")
    official_set(second, b"different-model")
    current = tmp_path / "current"
    current.symlink_to(first, target_is_directory=True)
    model = current / "velstand.onnx"
    real_file_digest = hashlib.file_digest

    def switch_release(source, algorithm):
        result = real_file_digest(source, algorithm)
        current.unlink()
        current.symlink_to(second, target_is_directory=True)
        return result

    monkeypatch.setattr("duckmind.policy_approval.hashlib.file_digest", switch_release)
    assert official_walk_fingerprint(model) is None


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
    assert not loaded_official_walk(report)


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
    assert not loaded_official_walk(report)


def test_first_board_rejects_missing_or_duplicate_walk_slot() -> None:
    report = official_policy_report()
    report["slots"] = []
    assert not loaded_official_walk(report)
    report["slots"] = [
        deepcopy(official_policy_report()["slots"][0]),
        deepcopy(official_policy_report()["slots"][0]),
    ]
    assert not loaded_official_walk(report)


@pytest.mark.parametrize("field", ["change_error", "slots"])
def test_first_board_rejects_missing_required_report_field(field) -> None:
    report = official_policy_report()
    report.pop(field)
    assert not loaded_official_walk(report)


def test_first_board_rejects_missing_walk_error_field() -> None:
    report = official_policy_report()
    report["slots"][0].pop("error")
    assert not loaded_official_walk(report)


@pytest.mark.parametrize("report", [None, {}, {"slots": "not a list"}])
def test_first_board_rejects_malformed_report(report) -> None:
    assert not loaded_official_walk(report)


def test_policy_hashes_name_the_walk_and_seat_the_brain_was_measured_on(tmp_path) -> None:
    # (real duck prep 2026-11-08) the bridge on the duck reports which walk and sit-stand networks it
    # runs, by the sha256 the brain's measured tables (walkcal, seatcal) are keyed by
    from duckmind.policy_approval import official_policy_hashes

    walk = official_set(tmp_path / "seed-v9", b"v7 velstand", "v9")
    (walk.parent / "alpha_sitstand.onnx").write_bytes(b"v6+ sitstand")
    assert official_policy_hashes(walk) == {
        "set": "v9",
        "walk_sha256": hashlib.sha256(b"v7 velstand").hexdigest(),
        "sitstand_sha256": hashlib.sha256(b"v6+ sitstand").hexdigest(),
    }


def test_policy_hashes_of_a_walk_that_fails_the_contract_say_nothing(tmp_path) -> None:
    from duckmind.policy_approval import official_policy_hashes

    walk = official_set(tmp_path / "seed", b"walk", "v8")
    (walk.parent / "alpha_sitstand.onnx").write_bytes(b"seat")
    (walk.parent / ".source").write_text("repo=someone/else\nversion=v8\n")
    assert official_policy_hashes(walk) == {
        "set": None,
        "walk_sha256": None,
        "sitstand_sha256": None,
    }
    missing = official_set(tmp_path / "noseat", b"walk", "v8")  # no sit-stand file on disk
    assert official_policy_hashes(missing)["sitstand_sha256"] is None
