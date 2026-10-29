import base64
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from duckmind.release import (
    EVIDENCE,
    build_bundle,
    inspect_wheel,
    prepare_release,
    release_inputs_fingerprint,
    validate_evidence,
    verify_bundle,
)
from duckmind.replay import ValidationReport
from duckmind.sim_live import POLICY_FILES
from duckmind.versions import package_fingerprint
from tests.checkout_parts import EVIDENCE as EVIDENCE_DIR

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_source(tmp_path, monkeypatch):
    if not EVIDENCE_DIR.is_dir():
        pytest.skip("the measurement records (docs/validation/evidence) are not in this checkout")
    repo = tmp_path / "repo"
    package = repo / "src/duckmind"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (package / "service.py").write_text("def main(): pass\n")
    (package / "cli.py").write_text("def main(): pass\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="duckmind"\nversion="0.1.0"\nrequires-python=">=3.12"\ndependencies=[]\n'
        '[project.scripts]\nduckmindd="duckmind.service:main"\nduckmindctl="duckmind.cli:main"\n'
    )
    for name in (".gitignore", "README.md", "architecture.md", "uv.lock"):
        (repo / name).write_text("fixture\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_fixture.py").write_text("def test_fixture(): assert True\n")
    (repo / "deploy").mkdir()
    (repo / "deploy/install-board.sh").write_text("#!/bin/sh\nexit 0\n")
    fingerprint = package_fingerprint(package)
    for name in EVIDENCE:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".jsonl"):
            records = [json.loads(line) for line in (REPO / name).read_text().splitlines()]
            records[0]["duckmind_source_sha256"] = fingerprint
            destination.write_text("".join(json.dumps(record) + "\n" for record in records))
        else:
            record = json.loads((REPO / name).read_text())
            record["duckmind_source_sha256"] = fingerprint
            destination.write_text(json.dumps(record))
    for report_name, trace_name in (
        (
            "docs/validation/evidence/official-v5-pet-service.json",
            "docs/validation/evidence/official-v5-pet-service.body.json",
        ),
        (
            "docs/validation/evidence/official-v5-pet-service-sit.json",
            "docs/validation/evidence/official-v5-pet-service-sit.body.json",
        ),
        (
            "docs/validation/evidence/official-v5-pet-service-negative.json",
            "docs/validation/evidence/official-v5-pet-service-negative.body.json",
        ),
    ):
        report_path = repo / report_name
        report = json.loads(report_path.read_text())
        report["body_trace_sha256"] = hashlib.sha256((repo / trace_name).read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
    monkeypatch.setattr("duckmind.release._git_state", lambda _: {"head": "fixture", "dirty": True})
    wheel = tmp_path / "duckmind-0.1.0-py3-none-any.whl"
    write_wheel(wheel, package)
    return repo, wheel


def write_wheel(
    wheel, package, *, requirement=None, omit=None, wrong_entry=False, bad_record=False
):
    metadata = "Metadata-Version: 2.3\nName: duckmind\nVersion: 0.1.0\nRequires-Python: >=3.12\n"
    if requirement:
        metadata += f"Requires-Dist: {requirement}\n"
    files = {f"duckmind/{path.name}": path.read_bytes() for path in package.glob("*.py")}
    prefix = "duckmind-0.1.0.dist-info/"
    files[prefix + "METADATA"] = metadata.encode()
    files[prefix + "WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files[prefix + "entry_points.txt"] = (
        b"[console_scripts]\nduckmindd = duckmind.service:main\nduckmindctl = duckmind.cli:main\n"
        if not wrong_entry
        else b"[console_scripts]\nduckmindd = duckmind.service:missing\n"
    )
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", str(len(data))])
    writer.writerow([prefix + "RECORD", "", ""])
    files[prefix + "RECORD"] = record.getvalue().encode()
    if bad_record:
        files["duckmind/cli.py"] += b"# not in RECORD\n"
    if omit:
        del files[prefix + omit]
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def bundle_fixture(release_source, tmp_path):
    repo, wheel = release_source
    return build_bundle(
        repo,
        wheel,
        tmp_path / "bundle",
        release_id="0.1.0-rc2",
        checks=[{"command": ["test-fixture"], "returncode": 0}],
        checked_inputs_sha256=release_inputs_fingerprint(repo),
    )


def test_bundle_records_exact_wheel_and_keeps_hardware_gate_open(release_source, tmp_path):
    bundle = bundle_fixture(release_source, tmp_path)
    manifest = verify_bundle(bundle)
    assert manifest["physical_hardware_gate"] == "not_run"
    assert manifest["git"]["dirty"] is True
    assert manifest["wheel"]["source_sha256"] == package_fingerprint(
        release_source[0] / "src/duckmind"
    )
    sbom = json.loads((bundle / "sbom.cdx.json").read_text())
    assert sbom["metadata"]["component"]["hashes"][0]["content"] == manifest["wheel"]["sha256"]
    assert sbom["dependencies"][0]["dependsOn"] == []


@pytest.mark.parametrize("attack", ["payload", "nested_manifest", "symlink", "checksum"])
def test_bundle_integrity_rejects_tampering(release_source, tmp_path, attack):
    bundle = bundle_fixture(release_source, tmp_path)
    if attack == "payload":
        (bundle / "README.md").write_text("changed")
    elif attack == "nested_manifest":
        (bundle / "docs/manifest.json").write_text("unlisted")
    elif attack == "symlink":
        (bundle / "injected").symlink_to(release_source[0] / "README.md")
    else:
        (bundle / "SHA256SUMS").write_text("changed")
    with pytest.raises(ValueError):
        verify_bundle(bundle)


@pytest.mark.parametrize("stale", ["wheel", "evidence"])
def test_stale_wheel_or_simulator_evidence_blocks_release(release_source, tmp_path, stale):
    repo, wheel = release_source
    if stale == "wheel":
        (repo / "src/duckmind/__init__.py").write_text("# source changed\n")
    else:
        report = json.loads((repo / EVIDENCE[0]).read_text())
        report["duckmind_source_sha256"] = "old-source"
        (repo / EVIDENCE[0]).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="sources differ|evidence is stale"):
        bundle_fixture(release_source, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fault", ["wrong_pose", "different_robotd", "fake_hardware_pass"])
def test_pet_release_requires_same_official_stack_in_home_and_sit(release_source, fault):
    repo, _ = release_source
    sit = repo / "docs/validation/evidence/official-v5-pet-service-sit.json"
    report = json.loads(sit.read_text())
    if fault == "wrong_pose":
        report["initial_keyframe"] = "HOME"
    elif fault == "different_robotd":
        report["robotd_binary_sha256"] = "0" * 64
    else:
        report["physical_hardware_gate"] = "pass"
    sit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="pose|official stack|hardware gate"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


@pytest.mark.parametrize("fault", ["changed_walk", "missing_hashes"])
def test_pet_release_rejects_same_manifest_with_different_policy_files(release_source, fault):
    repo, _ = release_source
    sit = repo / "docs/validation/evidence/official-v5-pet-service-sit.json"
    report = json.loads(sit.read_text())
    if fault == "changed_walk":
        report.setdefault("policy_files_sha256", {})["velstand.onnx"] = "0" * 64
    else:
        report.pop("policy_files_sha256", None)
    sit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="policy files|official stack"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_named_skill_release_rejects_same_manifest_with_different_policy_files(release_source):
    repo, _ = release_source
    path = repo / "docs/validation/evidence/official-v5-kick-left.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0].setdefault("policy_files_sha256", {})["ball_kick_left.onnx"] = "0" * 64
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="named-skill policy files"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


@pytest.mark.parametrize("field", ["ort_lib_sha256", "scene_sha256", "body_server_sha256"])
def test_pet_release_rejects_same_commits_with_different_simulator_bytes(release_source, field):
    repo, _ = release_source
    policy_hashes = {name: "1" * 64 for name in POLICY_FILES}
    for name in (
        "official-v5-pet-service.json",
        "official-v5-pet-service-sit.json",
        "official-v5-pet-service-negative.json",
    ):
        path = repo / "docs/validation/evidence" / name
        payload = json.loads(path.read_text())
        payload["policy_files_sha256"] = policy_hashes
        path.write_text(json.dumps(payload))
    trace = repo / "docs/validation/evidence/official-v5-kick-left.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    records[0]["policy_files_sha256"] = policy_hashes
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    sit = repo / "docs/validation/evidence/official-v5-pet-service-sit.json"
    report = json.loads(sit.read_text())
    report[field] = "0" * 64
    sit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="official stack"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_year_long_learning_evidence_must_show_causal_ablation(release_source):
    repo, _ = release_source
    year = repo / "docs/validation/evidence/365day-twins.json"
    report = json.loads(year.read_text())
    report["ablation"]["final_preference"] = report["duck_a"]["final_preference"]
    year.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="365-day causal"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_year_long_learning_delta_must_match_twin_difference(release_source):
    repo, _ = release_source
    year = repo / "docs/validation/evidence/365day-twins.json"
    report = json.loads(year.read_text())
    report["causal_preference_delta"] = 0.01
    year.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="365-day causal"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_pet_report_cannot_claim_learning_without_persisted_update(release_source):
    repo, _ = release_source
    home = repo / "docs/validation/evidence/official-v5-pet-service.json"
    report = json.loads(home.read_text())
    report["after_restart"]["learning_updates"] = 0
    home.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="learning persistence"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_release_requires_negative_feedback_on_official_body_path():
    assert "docs/validation/evidence/official-v5-pet-service-negative.json" in EVIDENCE
    assert "docs/validation/evidence/official-v5-pet-service-negative.body.json" in EVIDENCE


@pytest.mark.parametrize(
    "fault",
    ["wrong_score", "wrong_outcome", "wrong_direction", "no_update", "different_robotd"],
)
def test_negative_feedback_evidence_must_replay_and_learn_in_correct_direction(
    release_source, fault
):
    repo, _ = release_source
    path = repo / "docs/validation/evidence/official-v5-pet-service-negative.json"
    report = json.loads(path.read_text())
    if fault == "wrong_score":
        report["feedback_score"] = 1
    elif fault == "wrong_outcome":
        report["feedback"]["outcome"] = "success"
    elif fault == "wrong_direction":
        report["after_restart"]["preferences"][f"preferences.{report['actor']}.look_game"] = 0.505
    elif fault == "no_update":
        report["after_restart"]["learning_updates"] = 0
    else:
        report["robotd_binary_sha256"] = "0" * 64
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="HOME-negative pet feedback|official stack"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_negative_feedback_body_trace_cannot_claim_replay_after_tampering(release_source):
    repo, _ = release_source
    report_path = repo / "docs/validation/evidence/official-v5-pet-service-negative.json"
    trace_path = repo / "docs/validation/evidence/official-v5-pet-service-negative.body.json"
    report = json.loads(report_path.read_text())
    trace = json.loads(trace_path.read_text())
    trace["steps"][0]["params"]["head_pitch"] = 0.9
    trace_path.write_text(json.dumps(trace))
    report["body_trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="HOME-negative pet body trace"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


@pytest.mark.parametrize("fault", ["stop_still_enabled", "restart_database_fault", "lost_on_stop"])
def test_pet_learning_report_requires_passive_durable_stop(release_source, fault):
    repo, _ = release_source
    path = repo / "docs/validation/evidence/official-v5-pet-service-negative.json"
    report = json.loads(path.read_text())
    if fault == "stop_still_enabled":
        report["after_stop"]["play_enabled"] = True
    elif fault == "restart_database_fault":
        report["after_restart"]["database_integrity"] = "failed"
    else:
        report["after_stop"]["preferences"] = {}
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="HOME-negative pet feedback"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_release_requires_bodyport_command_replay(release_source, monkeypatch):
    repo, _ = release_source
    monkeypatch.setattr(
        "duckmind.release.replay_named_skill_trace",
        lambda _: ValidationReport(False, ("intent_mismatch",)),
        raising=False,
    )
    with pytest.raises(ValueError, match="body command replay"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


@pytest.mark.parametrize("attack", ["head_target", "interaction_id", "source_hash"])
def test_release_requires_source_bound_first_pet_body_replay(release_source, attack):
    repo, _ = release_source
    report_path = repo / "docs/validation/evidence/official-v5-pet-service.json"
    trace_path = repo / "docs/validation/evidence/official-v5-pet-service.body.json"
    report = json.loads(report_path.read_text())
    trace = json.loads(trace_path.read_text())
    if attack == "head_target":
        trace["steps"][0]["params"]["head_pitch"] = 0.9
    elif attack == "interaction_id":
        trace["interaction_id"] = "other-interaction"
    else:
        trace["duckmind_source_sha256"] = "0" * 64
    trace_path.write_text(json.dumps(trace))
    report["body_trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="pet body trace"):
        validate_evidence(repo, package_fingerprint(repo / "src/duckmind"))


def test_unbundled_runtime_dependency_is_rejected(release_source):
    repo, wheel = release_source
    write_wheel(wheel, repo / "src/duckmind", requirement="requests>=2")
    with pytest.raises(ValueError, match="non-dev dependency"):
        inspect_wheel(wheel, repo)


def test_existing_bundle_is_not_overwritten(release_source, tmp_path):
    bundle = bundle_fixture(release_source, tmp_path)
    original = (bundle / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        bundle_fixture(release_source, tmp_path)
    assert (bundle / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("omit", ["WHEEL", "RECORD", "entry_points.txt"])
def test_missing_wheel_installation_metadata_is_rejected(release_source, omit):
    repo, wheel = release_source
    write_wheel(wheel, repo / "src/duckmind", omit=omit)
    with pytest.raises(ValueError, match="required installation metadata"):
        inspect_wheel(wheel, repo)


@pytest.mark.parametrize("fault", ["wrong_entry", "bad_record"])
def test_invalid_wheel_metadata_is_rejected(release_source, fault):
    repo, wheel = release_source
    write_wheel(wheel, repo / "src/duckmind", **{fault: True})
    with pytest.raises(ValueError, match="entry points|RECORD mismatch"):
        inspect_wheel(wheel, repo)


@pytest.mark.parametrize("name", ["deploy/install-board.sh", "tests/test_fixture.py", "uv.lock"])
def test_changed_non_code_input_invalidates_checks(release_source, tmp_path, monkeypatch, name):
    repo, _ = release_source

    def fake_run(*args, **kwargs):
        (repo / name).write_text("changed after check started\n")

    monkeypatch.setattr("duckmind.release.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="inputs changed during checks"):
        prepare_release(repo, tmp_path / "bundle", "rc2")


def test_output_inside_copied_tree_is_rejected(release_source):
    repo, wheel = release_source
    with pytest.raises(ValueError, match="recursive copy"):
        build_bundle(
            repo,
            wheel,
            repo / "docs/bundle",
            release_id="rc2",
            checks=[{"returncode": 0}],
            checked_inputs_sha256=release_inputs_fingerprint(repo),
        )


@pytest.mark.parametrize("fault", ["wheel", "evidence"])
def test_mutation_during_copy_is_rejected(release_source, tmp_path, monkeypatch, fault):
    original_copy = shutil.copy2
    original_tree = shutil.copytree

    def copy_file(source, destination, **kwargs):
        result = original_copy(source, destination, **kwargs)
        if str(source).endswith(".whl"):
            with zipfile.ZipFile(destination, "a") as archive:
                archive.comment = b"changed during copy"
        return result

    def copy_tree(source, destination, *args, **kwargs):
        result = original_tree(source, destination, *args, **kwargs)
        if Path(source).name == "docs":
            report = Path(destination) / "validation/evidence/official-v5-pet-service.json"
            report.write_text("{}\n")
        return result

    if fault == "wheel":
        monkeypatch.setattr("duckmind.release.shutil.copy2", copy_file)
    else:
        monkeypatch.setattr("duckmind.release.shutil.copytree", copy_tree)
    with pytest.raises(ValueError, match="changed while copying"):
        bundle_fixture(release_source, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fault", ["wheel_identity", "sbom"])
def test_verifier_rejects_semantic_mismatch_even_with_matching_hashes(
    release_source, tmp_path, fault
):
    bundle = bundle_fixture(release_source, tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if fault == "wheel_identity":
        manifest["wheel"]["sha256"] = "0" * 64
    else:
        bom = json.loads((bundle / "sbom.cdx.json").read_text())
        bom["metadata"]["component"]["hashes"][0]["content"] = "0" * 64
        (bundle / "sbom.cdx.json").write_text(json.dumps(bom))
        manifest["files"]["sbom.cdx.json"] = hashlib.sha256(
            (bundle / "sbom.cdx.json").read_bytes()
        ).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    sums = {
        **manifest["files"],
        "manifest.json": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
    }
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )
    with pytest.raises(ValueError, match="wheel identity|SBOM differs"):
        verify_bundle(bundle)


def _rewrite_bundle_with_historical_evidence(
    bundle: Path, *, old_seven_file_inventory: bool = False
) -> str:
    """Make a self-consistent pre-body-trace bundle for verifier compatibility tests."""

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    historical_evidence = [
        name
        for name in manifest["evidence"]
        if "-negative." not in name
        and (old_seven_file_inventory or not name.endswith(".body.json"))
    ]
    for name in manifest["evidence"]:
        if name not in historical_evidence:
            (bundle / name).unlink()
    checks_path = bundle / "checks.json"
    checks = json.loads(checks_path.read_text())
    checks["inputs_sha256"] = release_inputs_fingerprint(bundle)
    checks_path.write_text(json.dumps(checks))
    manifest["inputs_sha256"] = checks["inputs_sha256"]
    manifest["evidence"] = historical_evidence
    manifest["files"] = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and path.relative_to(bundle).as_posix() not in {"manifest.json", "SHA256SUMS"}
    }
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sums = {**manifest["files"], "manifest.json": digest}
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(sums.items()))
    )
    return digest


def test_known_historical_bundle_remains_verifiable_without_running_bundled_code(
    release_source, tmp_path, monkeypatch
):
    bundle = bundle_fixture(release_source, tmp_path)
    digest = _rewrite_bundle_with_historical_evidence(bundle)
    monkeypatch.setattr("duckmind.release.LEGACY_MANIFESTS", {digest: "0.1.0-rc2"}, raising=False)

    assert verify_bundle(bundle)["release_id"] == "0.1.0-rc2"


def test_pinned_nine_file_bundle_uses_original_gate_only_after_integrity_checks(
    release_source, tmp_path, monkeypatch
):
    bundle = bundle_fixture(release_source, tmp_path)
    report_path = bundle / "docs/validation/evidence/official-v5-pet-service.json"
    report = json.loads(report_path.read_text())
    report.pop("policy_files_sha256")
    report_path.write_text(json.dumps(report))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report_name = report_path.relative_to(bundle).as_posix()
    manifest["files"][report_name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    checks_path = bundle / "checks.json"
    checks = json.loads(checks_path.read_text())
    checks["inputs_sha256"] = release_inputs_fingerprint(bundle)
    checks_path.write_text(json.dumps(checks))
    manifest["inputs_sha256"] = checks["inputs_sha256"]
    manifest["files"]["checks.json"] = hashlib.sha256(checks_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sums = {**manifest["files"], "manifest.json": digest}
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(sums.items()))
    )

    with pytest.raises(ValueError, match="policy files identity"):
        verify_bundle(bundle)
    monkeypatch.setattr("duckmind.release.LEGACY_MANIFESTS", {digest: "0.1.0-rc2"})
    assert verify_bundle(bundle)["release_id"] == "0.1.0-rc2"

    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(ValueError, match="payload is missing, changed"):
        verify_bundle(bundle)


def test_unlisted_historical_bundle_cannot_downgrade_current_evidence(release_source, tmp_path):
    bundle = bundle_fixture(release_source, tmp_path)
    _rewrite_bundle_with_historical_evidence(bundle)

    with pytest.raises(ValueError, match="evidence inventory"):
        verify_bundle(bundle)


def test_unlisted_old_seven_file_bundle_cannot_downgrade_current_evidence(release_source, tmp_path):
    bundle = bundle_fixture(release_source, tmp_path)
    _rewrite_bundle_with_historical_evidence(bundle, old_seven_file_inventory=True)

    with pytest.raises(ValueError, match="evidence inventory"):
        verify_bundle(bundle)


def test_pinned_historical_bundle_still_rejects_payload_tampering(
    release_source, tmp_path, monkeypatch
):
    bundle = bundle_fixture(release_source, tmp_path)
    digest = _rewrite_bundle_with_historical_evidence(bundle)
    monkeypatch.setattr("duckmind.release.LEGACY_MANIFESTS", {digest: "0.1.0-rc2"}, raising=False)
    (bundle / "README.md").write_text("tampered after pinning\n")

    with pytest.raises(ValueError, match="payload is missing, changed"):
        verify_bundle(bundle)
