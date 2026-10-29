"""Build and verify a source-matched, offline-installable first-board bundle."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import email
import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from duckmind.body_replay import replay_named_skill_trace
from duckmind.pet_body_trace import replay_front_gaze_trace
from duckmind.sim_live import POLICY_FILES
from duckmind.versions import fingerprint_files, package_fingerprint

SCHEMA_VERSION = 1
PET_HOME_EVIDENCE = "docs/validation/evidence/official-v5-pet-service.json"
PET_HOME_BODY_EVIDENCE = "docs/validation/evidence/official-v5-pet-service.body.json"
PET_SIT_EVIDENCE = "docs/validation/evidence/official-v5-pet-service-sit.json"
PET_SIT_BODY_EVIDENCE = "docs/validation/evidence/official-v5-pet-service-sit.body.json"
PET_NEGATIVE_EVIDENCE = "docs/validation/evidence/official-v5-pet-service-negative.json"
PET_NEGATIVE_BODY_EVIDENCE = "docs/validation/evidence/official-v5-pet-service-negative.body.json"
SKILL_EVIDENCE = "docs/validation/evidence/official-v5-kick-left.jsonl"
TWIN_EVIDENCE = "docs/validation/evidence/30day-twins.json"
YEAR_TWIN_EVIDENCE = "docs/validation/evidence/365day-twins.json"
EVIDENCE = (
    PET_HOME_EVIDENCE,
    PET_HOME_BODY_EVIDENCE,
    PET_SIT_EVIDENCE,
    PET_SIT_BODY_EVIDENCE,
    PET_NEGATIVE_EVIDENCE,
    PET_NEGATIVE_BODY_EVIDENCE,
    SKILL_EVIDENCE,
    TWIN_EVIDENCE,
    YEAR_TWIN_EVIDENCE,
)
# Exact already-built candidates. Their immutable manifests pin every payload
# hash; accept their integrity for rollback inspection without executing an
# older bundle's Python or weakening new-build evidence gates.
LEGACY_MANIFESTS = {
    "7ec162dae2fc0ef9921ddf1826ee27800d96c2c180692122bf78e73b7df4d310": "0.1.0-rc15",
    "35d1b15efe0ccc9faab3adfd8acc6bd3dfb7a552113eceb46b9d5862f680c984": "0.1.0-rc16",
    "25997286f0edd2db22d4254cf6eac7809b090d959deb121fdc5ad58c55df29a6": "0.1.0-rc17",
    "57a1bb6a6d49ff5a07cc97989f9c687ef5d2dd36e7a6c3476169f9be81a95241": "0.1.0-rc18",
    "206f39ec8e87f8bbcdf1e4c749ffd4a728e7b2944412a3cea8b2b2ec36b86bb9": "0.1.0-rc19",
    "940f9016b0a9938b1a22fc636ac4993c6a5b1b6df980f3a224d2ed51a3ce201f": "0.1.0-rc20",
    "e7a0f22b288981ed497474da37f40fb0f0077abf9963f4fc0586d09e56cadae4": "0.1.0-rc21",
    "fc1f26f367ec7e7b6fa7197dd5c52f6dc4a0084327571b590248254bba6fbf58": "0.1.0-rc22",
}
INPUT_ROOT_FILES = (".gitignore", "README.md", "architecture.md", "pyproject.toml", "uv.lock")
INPUT_TREES = ("src", "tests", "deploy", "docs")


class _EntryPointsParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_policy_hashes(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(POLICY_FILES)
        and all(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for digest in value.values()
        )
    )


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git_state(repo: Path) -> dict[str, object]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    return {"head": head, "dirty": dirty}


def release_inputs_fingerprint(repo: Path) -> str:
    """Bind checks to code, tests, deployment, build settings and shipped docs."""

    files = {}
    for name in INPUT_ROOT_FILES:
        if (repo / name).is_symlink():
            raise ValueError(f"symlink in release input: {name}")
        files[name] = (repo / name).read_bytes()
    for name in INPUT_TREES:
        root = repo / name
        if root.is_symlink():
            raise ValueError(f"symlink in release input: {name}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink in release input: {path}")
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                files[path.relative_to(repo).as_posix()] = path.read_bytes()
    return fingerprint_files(files)


def _validate_wheel_metadata(
    archive: zipfile.ZipFile, dist_info: str, scripts: dict[str, str]
) -> None:
    names = set(archive.namelist())
    required = {f"{dist_info}/{name}" for name in ("WHEEL", "RECORD", "entry_points.txt")}
    if not required <= names:
        raise ValueError("wheel is missing required installation metadata")
    wheel_metadata = email.message_from_bytes(archive.read(f"{dist_info}/WHEEL"))
    if (
        wheel_metadata["Wheel-Version"] != "1.0"
        or wheel_metadata["Root-Is-Purelib"] != "true"
        or wheel_metadata.get_all("Tag") != ["py3-none-any"]
    ):
        raise ValueError("unsupported wheel format; first-board profile requires pure Python")
    entries = _EntryPointsParser(interpolation=None)
    entries.read_string(archive.read(f"{dist_info}/entry_points.txt").decode())
    if (
        entries.defaults()
        or entries.sections() != ["console_scripts"]
        or dict(entries["console_scripts"]) != scripts
        or set(scripts) != {"duckmindd", "duckmindctl"}
    ):
        raise ValueError("wheel console entry points differ from project")
    record_name = f"{dist_info}/RECORD"
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
    if any(len(row) != 3 for row in rows) or len(rows) != len(names):
        raise ValueError("wheel RECORD is incomplete")
    if {row[0] for row in rows} != names:
        raise ValueError("wheel RECORD must enumerate each member exactly once")
    for name, digest, size in rows:
        if name == record_name:
            if digest or size:
                raise ValueError("wheel RECORD must not hash itself")
            continue
        data = archive.read(name)
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if digest != f"sha256={expected}" or size != str(len(data)):
            raise ValueError(f"wheel RECORD mismatch: {name}")


def inspect_wheel(wheel: Path, repo: Path) -> dict[str, str]:
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    if project.get("dependencies"):
        raise ValueError("offline first-board bundle currently requires no runtime dependencies")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel members")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel requires one METADATA file")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if metadata["Name"] != project["name"] or metadata["Version"] != project["version"]:
            raise ValueError("wheel identity differs from project")
        if metadata["Requires-Python"] != project["requires-python"]:
            raise ValueError("wheel Python requirement differs from project")
        for requirement in metadata.get_all("Requires-Dist", []):
            if not re.fullmatch(r"[^;]+;\s*extra\s*==\s*(['\"])dev\1", requirement):
                raise ValueError(
                    "wheel has a non-dev dependency; resolve its offline dependency set"
                )
        dist_info = metadata_names[0].rsplit("/", 1)[0]
        expected_stem = f"{project['name'].replace('-', '_')}-{project['version']}"
        if dist_info != f"{expected_stem}.dist-info" or wheel.name != (
            f"{expected_stem}-py3-none-any.whl"
        ):
            raise ValueError("wheel filename or metadata directory differs from project")
        _validate_wheel_metadata(archive, dist_info, project.get("scripts", {}))
        allowed_metadata = {
            f"{dist_info}/{name}" for name in ("METADATA", "WHEEL", "entry_points.txt", "RECORD")
        }
        sources: dict[str, bytes] = {}
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or member.as_posix() != name:
                raise ValueError("unsafe wheel member path")
            if name.startswith("duckmind/") and name.endswith(".py"):
                sources[name.removeprefix("duckmind/")] = archive.read(name)
            elif name not in allowed_metadata:
                raise ValueError(f"unexpected wheel member: {name}")
    source_hash = package_fingerprint(repo / "src/duckmind")
    if fingerprint_files(sources) != source_hash:
        raise ValueError("wheel Python sources differ from current repository")
    return {
        "name": project["name"],
        "version": project["version"],
        "requires_python": project["requires-python"],
        "sha256": _sha(wheel),
        "source_sha256": source_hash,
    }


def validate_evidence(repo: Path, source_hash: str) -> None:
    scenarios = (
        ("HOME-positive", "HOME", 1, PET_HOME_EVIDENCE, PET_HOME_BODY_EVIDENCE),
        ("SIT-positive", "SIT", 1, PET_SIT_EVIDENCE, PET_SIT_BODY_EVIDENCE),
        ("HOME-negative", "HOME", 0, PET_NEGATIVE_EVIDENCE, PET_NEGATIVE_BODY_EVIDENCE),
    )
    pets = {
        label: json.loads((repo / report_path).read_text())
        for label, _, _, report_path, _ in scenarios
    }
    twins = {
        days: json.loads((repo / path).read_text())
        for days, path in ((30, TWIN_EVIDENCE), (365, YEAR_TWIN_EVIDENCE))
    }
    records = [json.loads(line) for line in (repo / SKILL_EVIDENCE).read_text().splitlines()]
    runs = [row for row in records if row.get("type") == "run"]
    if len(runs) != 1:
        raise ValueError("named-skill evidence needs exactly one run")
    if any(
        record.get("duckmind_source_sha256") != source_hash
        for record in (*pets.values(), *twins.values(), runs[0])
    ):
        raise ValueError("simulator evidence is stale for the current Python source")
    official_stack = (
        "microduck_commit",
        "microduck_rl_commit",
        "robotd_binary_sha256",
        "ort_lib_sha256",
        "scene_sha256",
        "body_server_sha256",
        "policy_manifest_sha256",
    )
    home = pets["HOME-positive"]
    if any(not home.get(key) for key in official_stack):
        raise ValueError("pet simulator official stack identity is incomplete")
    policy_hashes = home.get("policy_files_sha256")
    if not _valid_policy_hashes(policy_hashes):
        raise ValueError("pet simulator policy files identity is incomplete")
    for label, pose, score, _, trace_name in scenarios:
        pet = pets[label]
        if pet.get("initial_keyframe") != pose:
            raise ValueError(f"pet simulator initial pose differs from {pose}")
        if any(pet.get(key) != home.get(key) for key in official_stack):
            raise ValueError("pet simulator evidence differs in official stack")
        if pet.get("policy_files_sha256") != policy_hashes:
            raise ValueError(f"{label} pet simulator policy files differ")
        if pet.get("execution_gate") != "pass" or pet.get("learning_gate") != "pass":
            raise ValueError("pet simulator gates did not pass")
        if pet.get("physical_hardware_gate") != "not_run":
            raise ValueError("pet simulator physical hardware gate must remain not_run")
        if pet.get("after_restart", {}).get("play_enabled") is not False:
            raise ValueError("pet restart must be passive")
        interaction = pet.get("interaction", {})
        feedback = pet.get("feedback", {})
        trace_path = repo / trace_name
        trace = json.loads(trace_path.read_text())
        if (
            pet.get("body_replay_gate") != "pass"
            or pet.get("body_trace_file") != trace_path.name
            or pet.get("body_trace_sha256") != _sha(trace_path)
            or trace.get("duckmind_source_sha256") != source_hash
            or trace.get("interaction_id") != interaction.get("interaction_id")
            or trace.get("result", {}).get("phase") != "completed"
            or not replay_front_gaze_trace(trace_path).passed
        ):
            raise ValueError(f"{label} pet body trace is invalid or did not replay")
        actor = pet.get("actor")
        preferences = pet.get("after_restart", {}).get("preferences")
        preference = (
            preferences.get(f"preferences.{actor}.look_game")
            if isinstance(actor, str) and isinstance(preferences, dict)
            else None
        )
        bounded_direction = (
            isinstance(preference, (int, float))
            and not isinstance(preference, bool)
            and (0.5 < preference <= 0.505000001 if score else 0.494999999 <= preference < 0.5)
        )
        if (
            interaction.get("execution") != "completed"
            or not interaction.get("feedback_allowed")
            or pet.get("feedback_source") != "synthetic_simulation_harness"
            or pet.get("feedback_score") != score
            or type(pet.get("feedback_score")) is not int
            or feedback.get("interaction_id") != interaction.get("interaction_id")
            or feedback.get("outcome") != ("success" if score else "failure")
            or not feedback.get("learning_update_id")
            or pet.get("after_restart", {}).get("learning_updates") != 1
            or pet.get("after_restart", {}).get("database_integrity") != "ok"
            or pet.get("after_restart", {}).get("storage_fault") is not None
            or pet.get("after_stop", {}).get("play_enabled") is not False
            or pet.get("after_stop", {}).get("preferences") != preferences
            or not bounded_direction
            or pet.get("stop", {}).get("robotd_stop_accepted") is not True
        ):
            raise ValueError(f"{label} pet feedback, learning persistence or stop is incomplete")
    if runs[0].get("policy_files_sha256") != policy_hashes:
        raise ValueError("named-skill policy files differ from pet simulator")
    if any(runs[0].get(key) != home.get(key) for key in official_stack):
        raise ValueError("named-skill official stack differs from pet simulator")
    if not replay_named_skill_trace(repo / SKILL_EVIDENCE).passed:
        raise ValueError("named-skill body command replay failed")
    outcome = [row for row in records if row.get("type") == "outcome"]
    if len(outcome) != 1 or outcome[0].get("execution_gate") != "pass":
        raise ValueError("named-skill execution gate did not pass")
    for days, twin in twins.items():
        a = twin.get("duck_a", {})
        b = twin.get("duck_b", {})
        if (
            twin.get("days") != days
            or twin.get("causal_preference_delta", 0) <= 0
            or abs(
                twin.get("causal_preference_delta", 0)
                - (a.get("final_preference", 0) - b.get("final_preference", 0))
            )
            > 1e-12
            or twin.get("ablation", {}).get("final_preference") != b.get("final_preference")
            or not 0.5 < a.get("final_preference", 0) < 1.0
            or b.get("final_preference") != 0.5
            or a.get("personality_sociability") != 0.5
            or b.get("personality_sociability") != 0.5
            or a.get("learning_updates", days + 1) > days
        ):
            raise ValueError(f"{days}-day causal learning evidence is incomplete")


def make_sbom(identity: dict[str, str], timestamp: str) -> dict[str, object]:
    """CycloneDX 1.6 inventory of this wheel; host/robot packages are external."""

    ref = f"pkg:pypi/{identity['name']}@{identity['version']}"
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "component": {
                "type": "application",
                "bom-ref": ref,
                "name": identity["name"],
                "version": identity["version"],
                "purl": ref,
                "hashes": [{"alg": "SHA-256", "content": identity["sha256"]}],
                "properties": [
                    {
                        "name": "duckmind:inventory:scope",
                        "value": "Application wheel only; Python, OS, robotd and ONNX policies are external prerequisites. Dev extras are not installed.",
                    }
                ],
            },
        },
        "components": [],
        "dependencies": [{"ref": ref, "dependsOn": []}],
    }


def _payload_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in release payload: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in {"manifest.json", "SHA256SUMS"}:
            result[relative] = _sha(path)
    return result


def build_bundle(
    repo: Path,
    wheel: Path,
    output: Path,
    *,
    release_id: str,
    checks: list[dict[str, object]],
    checked_inputs_sha256: str,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", release_id):
        raise ValueError("invalid release ID")
    if output.exists() or output.is_symlink():
        raise ValueError("release output already exists; use a new directory")
    resolved_output, resolved_repo = output.resolve(), repo.resolve()
    if resolved_output.is_relative_to(resolved_repo) and not resolved_output.is_relative_to(
        resolved_repo / "dist"
    ):
        raise ValueError("in-repository release output must be under dist to avoid recursive copy")
    if not checks or any(check.get("returncode") != 0 for check in checks):
        raise ValueError("release checks must all pass")
    if release_inputs_fingerprint(repo) != checked_inputs_sha256:
        raise ValueError("release inputs changed after checks")
    identity = inspect_wheel(wheel, repo)
    validate_evidence(repo, identity["source_sha256"])
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".duckmind-release-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir()
        (stage / "dist").mkdir()
        shutil.copy2(wheel, stage / "dist" / wheel.name)
        for name in INPUT_ROOT_FILES:
            if (repo / name).is_symlink():
                raise ValueError(f"symlink in release source: {name}")
            shutil.copy2(repo / name, stage / name)
        for name in INPUT_TREES:
            # Refuse symlinks before copying; do not follow external files.
            if (repo / name).is_symlink() or any(
                path.is_symlink() for path in (repo / name).rglob("*")
            ):
                raise ValueError(f"symlink in {name} source tree")
            shutil.copytree(
                repo / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        if release_inputs_fingerprint(stage) != checked_inputs_sha256:
            raise ValueError("release inputs changed while copying")
        if inspect_wheel(stage / "dist" / wheel.name, stage) != identity:
            raise ValueError("wheel changed while copying")
        validate_evidence(stage, identity["source_sha256"])
        _json_write(stage / "sbom.cdx.json", make_sbom(identity, timestamp))
        _json_write(
            stage / "checks.json",
            {
                "source_sha256": identity["source_sha256"],
                "inputs_sha256": checked_inputs_sha256,
                "checks": checks,
            },
        )
        payload_files = _payload_files(stage)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_id": release_id,
            "built_at": timestamp,
            "git": _git_state(repo),
            "wheel": {**identity, "path": f"dist/{wheel.name}"},
            "inputs_sha256": checked_inputs_sha256,
            "evidence": list(EVIDENCE),
            "physical_hardware_gate": "not_run",
            "files": payload_files,
        }
        _json_write(stage / "manifest.json", manifest)
        sums = {**payload_files, "manifest.json": _sha(stage / "manifest.json")}
        (stage / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
        )
        verify_bundle(stage)
        if release_inputs_fingerprint(repo) != checked_inputs_sha256:
            raise ValueError("release inputs changed while bundling")
        stage.rename(output)
    return output


def verify_bundle(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("bundle root must not be a symlink")
    actual_files = _payload_files(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("unsupported release manifest")
    expected = manifest["files"]
    for name, digest in expected.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        ):
            raise ValueError("invalid manifest file entry")
    if actual_files != expected:
        raise ValueError("release payload is missing, changed, or contains extra files")
    sums = {**expected, "manifest.json": _sha(root / "manifest.json")}
    expected_sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    if (root / "SHA256SUMS").read_text() != expected_sums:
        raise ValueError("release checksum list differs from manifest")
    inputs_sha256 = release_inputs_fingerprint(root)
    if inputs_sha256 != manifest.get("inputs_sha256"):
        raise ValueError("release input identity differs from manifest")
    wheel_record = manifest.get("wheel", {})
    wheel_path = wheel_record.get("path")
    if wheel_path not in expected or PurePosixPath(wheel_path).parent != PurePosixPath("dist"):
        raise ValueError("invalid wheel path in manifest")
    identity = inspect_wheel(root / wheel_path, root)
    if wheel_record != {**identity, "path": wheel_path}:
        raise ValueError("wheel identity differs from manifest")
    pinned_legacy = LEGACY_MANIFESTS.get(_sha(root / "manifest.json")) == manifest.get("release_id")
    if manifest.get("evidence") == list(EVIDENCE) and not pinned_legacy:
        validate_evidence(root, identity["source_sha256"])
    elif manifest.get("evidence") != list(EVIDENCE) and not pinned_legacy:
        raise ValueError("release evidence inventory differs")
    checks = json.loads((root / "checks.json").read_text())
    if (
        checks.get("source_sha256") != identity["source_sha256"]
        or checks.get("inputs_sha256") != inputs_sha256
        or not checks.get("checks")
        or any(check.get("returncode") != 0 for check in checks["checks"])
    ):
        raise ValueError("release check identity or result differs")
    sbom = json.loads((root / "sbom.cdx.json").read_text())
    expected_sbom = make_sbom(identity, manifest["built_at"])
    expected_sbom["serialNumber"] = sbom.get("serialNumber")
    if sbom != expected_sbom:
        raise ValueError("SBOM differs from wheel identity")
    if manifest.get("physical_hardware_gate") != "not_run":
        raise ValueError("this release profile cannot attest physical hardware")
    return manifest


def prepare_release(repo: Path, output: Path, release_id: str) -> Path:
    before = release_inputs_fingerprint(repo)
    checks: list[dict[str, object]] = []
    for args in (
        ["pytest"],
        ["ruff", "check", "src", "tests"],
        ["ruff", "format", "--check", "src", "tests"],
        ["mypy"],
    ):
        command = ["uv", "run", "--locked", "--python", "3.12", "--extra", "dev", *args]
        subprocess.run(command, cwd=repo, check=True)
        checks.append({"command": command, "returncode": 0})
    if before != release_inputs_fingerprint(repo):
        raise ValueError("release inputs changed during checks")
    with tempfile.TemporaryDirectory(prefix="duckmind-wheel-") as directory:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", directory], cwd=repo, check=True)
        wheels = list(Path(directory).glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("expected one built wheel")
        if before != release_inputs_fingerprint(repo):
            raise ValueError("release inputs changed while building wheel")
        return build_bundle(
            repo,
            wheels[0],
            output,
            release_id=release_id,
            checks=checks,
            checked_inputs_sha256=before,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--release-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_release(args.repo.resolve(), args.output.resolve(), args.release_id)
        print(
            json.dumps({"bundle": str(result), "manifest_sha256": _sha(result / "manifest.json")})
        )
    else:
        manifest = verify_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "release_id": manifest["release_id"],
                    "integrity": "pass",
                    "physical_hardware_gate": manifest["physical_hardware_gate"],
                }
            )
        )


if __name__ == "__main__":
    main()
