"""The frozen v4 engine (spec §7.6, phase-4 design A6, test T13): ``src/duckmind/engines/v4_0``
is a generated copy of Duck Core at git 79cc25d that nobody edits.

* every file of the copy is in its ``MANIFEST.json`` with the SHA-256 it has on disk, and no
  file is missing or extra;
* the copy is closed: its ``duckmind`` imports stay inside it, so a change to today's
  ``duckcore``, ``provenance`` or ``expression_sweep`` can never reach an old duck;
* regenerating it from git (``tools/freeze_engine.py``) gives the same bytes, and each file
  differs from the original only in its import paths (when git has the commit);
* ``duckmind.engines`` routes a saved Config to its revision (none named → ``v4_0``).

The golden corpus (``tests/test_golden_corpus.py``) proves the copy still behaves as v4 did.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import duckmind.engines as engines

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "src" / "duckmind" / "engines" / "v4_0"
MANIFEST = FROZEN / "MANIFEST.json"
TOOL = ROOT / "tools" / "freeze_engine.py"
COMMIT = "79cc25d66ecc9c06442c8204f600204ef892ba18"
TABLE_V4 = "d56679edadd377a9"  # the v4 content table (golden corpus MANIFEST, engine.tables)
PACKAGE = "duckmind.engines.v4_0"


def _manifest() -> dict[str, object]:
    data: dict[str, object] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data


def _files() -> list[dict[str, object]]:
    files = _manifest()["files"]
    assert isinstance(files, list)
    return files


def _on_disk() -> dict[str, bytes]:
    return {
        path.relative_to(FROZEN).as_posix(): path.read_bytes()
        for path in sorted(FROZEN.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path != MANIFEST
    }


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("freeze_engine", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("freeze_engine", module)
    spec.loader.exec_module(module)
    return module


def _git_has_commit() -> bool:
    done = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{COMMIT}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def test_every_frozen_file_matches_its_manifest_hash() -> None:
    manifest = _manifest()
    assert manifest["format"] == "duckmind-frozen-engine-v1"
    assert manifest["rev"] == "v4_0" and manifest["commit"] == COMMIT
    assert manifest["package"] == PACKAGE
    listed = {str(entry["path"]): str(entry["sha256"]) for entry in _files()}
    disk = _on_disk()
    assert sorted(disk) == sorted(listed), (
        "the frozen engine's files and its MANIFEST differ: "
        f"missing {sorted(set(listed) - set(disk))}, unlisted {sorted(set(disk) - set(listed))}"
    )
    changed = [
        path for path, data in disk.items() if hashlib.sha256(data).hexdigest() != listed[path]
    ]
    assert not changed, (
        f"the frozen v4 engine was edited ({', '.join(changed)}). Never edit it: an old duck "
        "must replay byte for byte. Regenerate it with `python tools/freeze_engine.py`; a new "
        "behaviour is a new engine revision."
    )
    lines = "".join(f"{path} {digest}\n" for path, digest in sorted(listed.items()))
    assert hashlib.sha256(lines.encode()).hexdigest() == manifest["tree_sha256"]


def test_the_manifest_names_the_twenty_one_originals() -> None:
    sources = sorted(str(entry["from"]) for entry in _files() if entry["from"] is not None)
    duckcore = [s for s in sources if s.startswith("src/duckmind/duckcore/")]
    assert len(duckcore) == 16, duckcore
    assert "src/duckmind/provenance.py" in sources
    assert {
        "src/duckmind/expression_sweep/envelope.py",
        "src/duckmind/expression_sweep/clips.py",
        "src/duckmind/expression_sweep/looks.py",
    } <= set(sources)
    for entry in _files():
        if entry["from"] is None:
            assert entry["path"] == "__init__.py"
            continue
        assert len(str(entry["blob"])) == 40 and len(str(entry["source_sha256"])) == 64, entry


def _module(path: str) -> tuple[str, bool]:
    parts = path.removesuffix(".py").split("/")
    package = parts[-1] == "__init__"
    if package:
        parts.pop()
    return ".".join([PACKAGE, *parts]), package


def test_the_frozen_engine_imports_only_itself() -> None:
    disk = {path: data for path, data in _on_disk().items() if path.endswith(".py")}
    modules = {_module(path)[0] for path in disk}
    problems = []
    for path, data in disk.items():
        module, is_package = _module(path)
        package = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(data.decode("utf-8"), filename=path)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [base])])
                names = [base]
            elif isinstance(node, ast.Call) and node.args:
                first = node.args[0]
                func = node.func
                callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if callee in {"import_module", "__import__"} and isinstance(first, ast.Constant):
                    names = [str(first.value)]
            problems += [
                f"{path}:{getattr(node, 'lineno', '?')} imports {name}"
                for name in names
                if (name == "duckmind" or name.startswith("duckmind.")) and name not in modules
            ]
    assert not problems, "the frozen engine must not reach outside itself:\n" + "\n".join(problems)


def test_the_frozen_engine_is_a_separate_copy_of_v4() -> None:
    from duckmind.duckcore import engine as current
    from duckmind.engines.v4_0.duckcore import api, content, engine

    assert engine.DuckCore is not current.DuckCore
    assert api.DuckCore is engine.DuckCore
    assert content.table_hash() == TABLE_V4
    assert engines.frozen_api("v4_0") is api
    assert Path(api.__file__).resolve().is_relative_to(FROZEN)


@pytest.mark.skipif(
    not _git_has_commit(), reason=f"git does not have {COMMIT[:7]} (a shallow clone)"
)
def test_regenerating_from_git_gives_the_same_bytes() -> None:
    tool = _tool()
    files, manifest, tree = tool.build(tool.REVISIONS["v4_0"])
    assert tree == _manifest()["tree_sha256"]
    assert tool.manifest_bytes(manifest) == MANIFEST.read_bytes()
    assert files == _on_disk()


@pytest.mark.skipif(
    not _git_has_commit(), reason=f"git does not have {COMMIT[:7]} (a shallow clone)"
)
def test_each_copy_differs_from_git_only_in_import_paths() -> None:
    tool = _tool()
    revision = tool.REVISIONS["v4_0"]
    for entry in _files():
        if entry["from"] is None:
            continue
        original = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{COMMIT}:{entry['from']}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(original).hexdigest() == entry["source_sha256"], entry["path"]
        copy = (FROZEN / str(entry["path"])).read_text(encoding="utf-8")
        mapped = tool._unfrozen(ast.parse(copy), revision)
        assert ast.dump(mapped) == ast.dump(ast.parse(original)), entry["path"]
        changed = [
            (a, b)
            for a, b in zip(original.decode().splitlines(), copy.splitlines()[1:], strict=True)
            if a != b
        ]
        assert len(changed) == entry["import_rewrites"], entry["path"]
        assert all(b == a.replace("duckmind.", f"{PACKAGE}.", 1) for a, b in changed), changed


def test_a_saved_config_names_its_engine() -> None:
    assert engines.rev_of({}) == "v4_0"  # a saved duck from before phase 4
    assert engines.rev_of({"engine_rev": None}) == "v4_0"
    assert engines.rev_of({"engine_rev": "v4_0"}) == "v4_0"
    assert engines.rev_of({"engine_rev": "v4_1"}) == engines.CURRENT == "v4_1"
    with pytest.raises(ValueError, match="unknown"):
        engines.rev_of({"engine_rev": "v9_9"})
    assert engines.is_frozen("v4_0") and not engines.is_frozen(engines.CURRENT)
    assert engines.REVS == ("v4_0", "v4_1")
    with pytest.raises(ValueError, match="not a frozen"):
        engines.frozen_api(engines.CURRENT)
