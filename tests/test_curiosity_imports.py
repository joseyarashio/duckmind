"""Curiosity P1 runs on the board (as part of duck-body): nothing of the simulator, the physics,
the page or the world's truth, and only Duck Core's public modules — the same rule
``tests/test_imports.py`` holds for ``duckmind.duckbody`` (the integration step adds
``duckmind.curiosity`` to its BOARD_SIDE; until then this test holds it)."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "duckmind" / "curiosity"
NOT_ON_THE_BOARD = (
    "duckmind.world",
    "duckmind.simweb",
    "duckmind.livebody",
    "duckmind.bodyhost",
    "duckmind.simulator",
    "duckmind.sim_live",
    "duckmind.sim_pet",
    "duckmind.local_env",
    "mujoco",
    "mjlab_microduck",
    "numpy",
)
DUCKCORE_PUBLIC = ("duckmind.duckcore.fixed", "duckmind.duckcore.rng", "duckmind.duckcore.percept")


def _imports(path: Path) -> list[tuple[str, int]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = "duckmind.curiosity" + (("." + base) if base else "")
            found += [(f"{base}.{alias.name}", node.lineno) for alias in node.names]
            found.append((base, node.lineno))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                found.append((node.args[0].value, node.lineno))
    return found


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_the_curiosity_package_loads_nothing_of_the_simulator() -> None:
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 7
    problems = []
    for path in files:
        for name, line in _imports(path):
            if any(_under(name, prefix) for prefix in NOT_ON_THE_BOARD):
                problems.append(f"{path.name}:{line} imports {name}")
            if _under(name, "duckmind.duckcore") and not any(
                _under(name, p) for p in DUCKCORE_PUBLIC
            ):
                problems.append(f"{path.name}:{line} imports Duck Core internals ({name})")
    assert not problems, "\n".join(problems)


def test_the_act_runner_reads_no_clock_and_no_socket() -> None:
    """Pure, clock-injected step functions (phase-3 architecture rule): time comes in ActContext."""
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "curio_calib.py":
            continue  # the records tool reads files; it never runs on the body loop
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "import time",
            "import socket",
            "import threading",
            "monotonic(",
            "time.time(",
            "open(",
        ):
            assert forbidden not in text, f"{path.name} uses {forbidden!r}"


def test_what_the_curiosity_package_loads_is_board_side_too() -> None:
    """The closure (every src module it loads, transitively), by test_imports' own reader."""
    from test_imports import DUCKCORE_PUBLIC as PUBLIC
    from test_imports import NOT_ON_THE_BOARD as NOT_BOARD
    from test_imports import _chain, _closure, _imports, _modules
    from test_imports import _under as under

    roots = [name for name in _modules() if under(name, "duckmind.curiosity")]
    assert "duckmind.curiosity.act_runner" in roots
    seen = _closure(roots)
    problems = [
        f"{_chain(seen, module)} imports {name} (line {line})"
        for module in sorted(seen)
        for name, line in _imports(module)
        if any(under(name, prefix) for prefix in NOT_BOARD)
        or (under(name, "duckmind.duckcore") and name not in PUBLIC)
    ]
    assert not problems, "\n".join(problems)
