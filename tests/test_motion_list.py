"""The public list of everything the duck can do (``docs/motions.md``, ``docs/motions.zh-Hant.md``).

The two files are written by ``tools/motion_list.py`` from the code's own registries. This test
checks that:

* the committed files are exactly what the tool writes now (so a new act, a changed cap or a new
  catalog word cannot leave the list behind);
* every act, skill, page motion, emotion, reflex and command appears in both files as an id in
  backticks, and in the section where it belongs;
* the files name no machine path.

The tool installs the default skill registry, which is permanent for the process it runs in (see
``tests/test_neuro_integration.py``), so every case that needs the registries runs the tool in a
child process and reads its output. This process only loads the tool's text and constants.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "motion_list.py"
DOCS = ROOT / "docs"
FILES = ("motions.md", "motions.zh-Hant.md")
HOW_TO_FIX = "Run `python tools/motion_list.py` and commit the two files it writes."

# What the child process prints: every id the lists must name, from the live registries.
IDS_SCRIPT = r"""
import json
import sys

sys.path.insert(0, "tools")
import motion_list

reg = motion_list.Reg()
g, cw = reg.g, reg.cw
print(json.dumps({
    "behaviours": list(reg.behaviours),
    "skills": [
        n for n, r in g.DEFAULT_RULES.items() if r.family not in motion_list.RELEASE_FAMILIES
    ],
    "motions": [m.id for m in reg.motions.MOTIONS],
    "emotions": list(reg.emo.ORDER),
    "reflexes": list(reg.c.REFLEXES),
    "verbs": [*cw.BUILTIN, *cw.TRICKS, *cw.POINTED],
    "rendered": motion_list.render_all(),
}))
"""

# The tool's --check against a folder (argv[1]): empty, hand-edited, then correct.
CHECK_SCRIPT = r"""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "tools")
import motion_list

docs = Path(sys.argv[1])
out = {}
err = io.StringIO()
with contextlib.redirect_stderr(err):
    out["missing"] = motion_list.main(["--check"], docs_dir=docs)
out["message"] = err.getvalue()
with contextlib.redirect_stdout(io.StringIO()):
    motion_list.main([], docs_dir=docs)
    out["fresh"] = motion_list.main(["--check"], docs_dir=docs)
(docs / "motions.md").write_text("edited by hand\n", encoding="utf-8")
with contextlib.redirect_stderr(io.StringIO()):
    out["edited_en"] = motion_list.main(["--check"], docs_dir=docs)
print(json.dumps(out))
"""


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("motion_list", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("motion_list", module)
    spec.loader.exec_module(module)
    return module


def _child(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


@lru_cache(maxsize=1)
def _live() -> dict[str, object]:
    """The registries' ids and the text the tool writes, from a fresh Python."""
    done = _child("-c", IDS_SCRIPT)
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    data: dict[str, object] = json.loads(done.stdout)
    return data


def _ids(key: str) -> list[str]:
    value = _live()[key]
    assert isinstance(value, list) and value, f"no {key} read from the registries"
    return [str(v) for v in value]


def _doc(name: str) -> str:
    path = DOCS / name
    assert path.is_file(), f"{path.name} is missing. {HOW_TO_FIX}"
    return path.read_text(encoding="utf-8")


def _section(text: str, number: int) -> str:
    """The body of the ``## <number>. …`` section (its ``###`` parts included)."""
    found = re.search(rf"(?ms)^## {number}\. .*?\n(.*?)(?=^## |\Z)", text)
    assert found is not None, f"no section {number}. {HOW_TO_FIX}"
    return found.group(1)


def _row_ids(body: str) -> list[str]:
    """The id in the first cell of every table row that starts with one (a ``†`` mark dropped)."""
    ids = []
    for line in body.splitlines():
        if line.startswith("| `"):
            ids.append(line.split("|")[1].strip().split()[0].strip("`"))
    return ids


def test_the_committed_lists_are_what_the_tool_writes() -> None:
    rendered = _live()["rendered"]
    assert isinstance(rendered, dict)
    assert sorted(rendered) == sorted(FILES)
    for name in FILES:
        assert _doc(name) == rendered[name], f"docs/{name} is out of date. {HOW_TO_FIX}"


def test_the_tool_loads_and_names_its_two_files() -> None:
    tool = _tool()
    assert tool.FILES == {"en": "motions.md", "zh-Hant": "motions.zh-Hant.md"}
    assert tuple(tool.FILES.values()) == FILES
    assert tool.EN.keys() == tool.ZH.keys()  # every word exists in both languages


def test_the_check_option_passes_on_the_committed_files() -> None:
    done = _child(str(TOOL), "--check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "up to date" in done.stdout


def test_the_check_option_fails_for_a_missing_or_hand_edited_file(tmp_path: Path) -> None:
    done = _child("-c", CHECK_SCRIPT, str(tmp_path))
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    got = json.loads(done.stdout)
    assert got["missing"] == 1
    assert "motions.md" in got["message"] and "python tools/motion_list.py" in got["message"]
    assert got["fresh"] == 0
    assert got["edited_en"] == 1


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize(
    "kind", ["behaviours", "skills", "motions", "emotions", "reflexes", "verbs"]
)
def test_every_id_is_in_the_list_in_backticks(name: str, kind: str) -> None:
    text = _doc(name)
    missing = [i for i in _ids(kind) if f"`{i}`" not in text]
    assert not missing, f"docs/{name} has no `id` for these {kind}: {missing}. {HOW_TO_FIX}"


@pytest.mark.parametrize("name", FILES)
def test_every_behaviour_and_reflex_is_one_row_in_sections_2_5_and_7(name: str) -> None:
    text = _doc(name)
    rows = [i for n in (2, 5, 7) for i in _row_ids(_section(text, n))]
    wanted = sorted({*_ids("behaviours"), *_ids("reflexes")})
    assert sorted(rows) == wanted, (
        f"docs/{name}: sections 2, 5 and 7 must hold each behaviour and reflex once: "
        f"missing {sorted(set(wanted) - set(rows))}, repeated "
        f"{sorted({i for i in rows if rows.count(i) > 1})}, extra {sorted(set(rows) - set(wanted))}"
    )


@pytest.mark.parametrize("name", FILES)
def test_skills_and_page_motions_are_rows_of_section_3(name: str) -> None:
    rows = _row_ids(_section(_doc(name), 3))
    skills, motions = _ids("skills"), _ids("motions")
    assert [i for i in rows if i in skills][: len(skills)] == skills
    assert sorted(i for i in rows if i not in skills) == sorted(
        i for i in motions if i not in skills
    )
    assert all(i in rows for i in motions)


@pytest.mark.parametrize("name", FILES)
def test_emotions_and_commands_are_the_rows_of_sections_4_and_6(name: str) -> None:
    text = _doc(name)
    assert _row_ids(_section(text, 4)) == _ids("emotions")
    assert _row_ids(_section(text, 6)) == _ids("verbs")


@pytest.mark.parametrize("name", FILES)
def test_the_list_names_no_machine_path(name: str) -> None:
    text = _doc(name)
    # an absolute home or temp path, or the private measurement folder
    pattern = re.compile(r"/(Users|home|tmp)/|duckmind[-_]data")
    found = pattern.search(text)
    assert found is None, f"docs/{name} names a machine path ({found.group(0)!r})"


@pytest.mark.parametrize("name", FILES)
def test_each_file_is_marked_generated_links_its_twin_and_says_simulator_only(name: str) -> None:
    text = _doc(name)
    lines = text.splitlines()
    english = name == "motions.md"
    assert lines[0] == (
        "**English** | [繁體中文](motions.zh-Hant.md)"
        if english
        else "[English](motions.md) | **繁體中文**"
    )
    assert lines[2].startswith("<!--") and "tools/motion_list.py" in lines[2]
    assert "Do not edit by hand" in lines[2]
    assert "MuJoCo" in text  # the status line: measured in the simulator only
