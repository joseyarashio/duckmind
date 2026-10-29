"""Live voice, LV1: the cloud connection's SDK is a required, pinned dependency (owner rule: required
dependencies, not optional) and the package stays out of the mind and the board's side."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("no_network")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "duckmind"


def test_google_genai_is_a_regular_pinned_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins = [d for d in project["dependencies"] if d.startswith("google-genai")]
    assert len(pins) == 1 and re.fullmatch(r"google-genai==\d+\.\d+\.\d+", pins[0]), pins
    optional = [d for group in project.get("optional-dependencies", {}).values() for d in group]
    assert not any(d.startswith("google-genai") for d in optional)


def test_the_lock_file_has_the_pinned_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pinned = next(d for d in project["dependencies"] if d.startswith("google-genai")).split("==")[1]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = [p["version"] for p in lock["package"] if p["name"] == "google-genai"]
    assert versions == [pinned]


def test_the_installed_sdk_is_the_pinned_one_when_it_is_installed() -> None:
    metadata = pytest.importorskip("importlib.metadata")
    try:
        installed = metadata.version("google-genai")
    except metadata.PackageNotFoundError:
        pytest.skip("google-genai is not installed in this environment")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pinned = next(d for d in project["dependencies"] if d.startswith("google-genai")).split("==")[1]
    assert installed == pinned


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "side", ["duckcore", "duckbody", "eye", "curiosity", "percept", "worldport"]
)
def test_the_mind_and_the_boards_side_do_not_import_the_live_connection(side: str) -> None:
    folder = SRC / side
    files = [folder] if folder.suffix == ".py" else sorted(folder.rglob("*.py"))
    files = files or [SRC / f"{side}.py"]
    for path in files:
        if not path.is_file():
            continue
        bad = {n for n in _imports(path) if n == "duckmind.live" or n.startswith("duckmind.live.")}
        assert not bad, f"{path.relative_to(ROOT)} imports {sorted(bad)}"


def test_the_sdk_is_imported_only_where_the_real_connection_is() -> None:
    importers = [
        p.name
        for p in sorted((SRC / "live").glob("*.py"))
        if any(n == "google" or n.startswith("google.") for n in _imports(p))
    ]
    # (the SDK is imported inside a function of the transport, never at module level: the rest
    # of the package, and every test that does not need it, loads without it)
    assert importers == ["genai_transport.py"]
    tree = ast.parse((SRC / "live" / "genai_transport.py").read_text(encoding="utf-8"))
    top_level = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom)) and "google" in ast.dump(n)
    ]
    assert top_level == []
