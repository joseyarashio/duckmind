import hashlib
from pathlib import Path

import pytest

from duckmind.sim_live import POLICY_FILES, policy_file_hashes
from duckmind.sim_pet import _body_server_command, _body_trace_path, run_sim_pet


def test_policy_identity_tracks_every_selected_model_byte(tmp_path):
    for name in POLICY_FILES:
        (tmp_path / name).write_bytes(f"model:{name}".encode())
    before = policy_file_hashes(tmp_path)
    assert set(before) == set(POLICY_FILES)
    assert before["velstand.onnx"] == hashlib.sha256(b"model:velstand.onnx").hexdigest()
    (tmp_path / "velstand.onnx").write_bytes(b"different")
    after = policy_file_hashes(tmp_path)
    assert before["velstand.onnx"] != after["velstand.onnx"]
    assert all(before[name] == after[name] for name in POLICY_FILES if name != "velstand.onnx")


@pytest.mark.parametrize("keyframe", ["HOME", "SIT", "STAND", "FOLD"])
def test_body_server_uses_explicit_initial_keyframe(keyframe):
    command = _body_server_command(port=12345, keyframe=keyframe)
    assert command[-4:] == ["--keyframe", keyframe, "--port", "12345"]
    assert command[1:4] == ["-m", "mjlab_microduck.sim.body_server", "--headless"]


def test_pet_harness_rejects_unknown_initial_pose_before_launch(tmp_path):
    with pytest.raises(ValueError, match="unsupported simulator keyframe"):
        run_sim_pet(
            microduck=tmp_path,
            microduck_rl=tmp_path,
            policies=tmp_path,
            robotd_binary=Path("/missing/robotd"),
            ort_lib=Path("/missing/ort"),
            report_path=tmp_path / "report.json",
            keyframe="UNREVIEWED",
        )


@pytest.mark.parametrize("score", [0, 1])
def test_pet_harness_accepts_binary_feedback_before_launch(tmp_path, score):
    with pytest.raises(FileNotFoundError, match="robotd missing"):
        run_sim_pet(
            microduck=tmp_path,
            microduck_rl=tmp_path,
            policies=tmp_path,
            robotd_binary=Path("/missing/robotd"),
            ort_lib=Path("/missing/ort"),
            report_path=tmp_path / "report.json",
            feedback_score=score,
        )


@pytest.mark.parametrize("score", [-1, 2, True, 0.5])
def test_pet_harness_rejects_nonbinary_feedback_before_launch(tmp_path, score):
    with pytest.raises(ValueError, match="binary feedback score"):
        run_sim_pet(
            microduck=tmp_path,
            microduck_rl=tmp_path,
            policies=tmp_path,
            robotd_binary=Path("/missing/robotd"),
            ort_lib=Path("/missing/ort"),
            report_path=tmp_path / "report.json",
            feedback_score=score,
        )


def test_pet_body_trace_is_sibling_of_source_bound_report(tmp_path):
    assert _body_trace_path(tmp_path / "pet-sit.json") == tmp_path / "pet-sit.body.json"
